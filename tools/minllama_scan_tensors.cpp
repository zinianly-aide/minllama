#include "minllama.h"
#include "minllama_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct TensorStats {
    std::string name;
    std::uint32_t type = 0;
    std::uint32_t n_dims = 0;
    std::vector<std::uint64_t> dims;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double absmax = 0.0;
    bool has_nan = false;
    bool has_inf = false;
    bool has_huge = false;
    long long first_anomaly_block_offset = -1;
};

std::string type_name(std::uint32_t t) {
    switch (t) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 8: return "Q8_0";
        default: return "TYPE_" + std::to_string(t);
    }
}

std::string shape_string(const TensorInfo &info) {
    std::string out = "[";
    for (std::uint32_t i = 0; i < info.n_dims; ++i) {
        if (i) out += ",";
        out += std::to_string(info.dims[i]);
    }
    out += "]";
    return out;
}

bool load_tensor_values(const std::string &model_path,
                        const TensorIndex &tensor_index,
                        const std::string &name,
                        std::vector<float> *values) {
    return minllama::load_tensor_as_f32(model_path.c_str(), tensor_index, name, values);
}

bool compute_stats(const std::vector<float> &values, TensorStats *stats) {
    if (values.empty() || !stats) return false;
    double sum = 0.0;
    stats->min = stats->max = values[0];
    stats->absmax = std::fabs(values[0]);
    for (float v : values) {
        if (std::isnan(v)) stats->has_nan = true;
        if (std::isinf(v)) stats->has_inf = true;
        if (std::fabs(v) > 1e6f) stats->has_huge = true;
        stats->min = std::min(stats->min, static_cast<double>(v));
        stats->max = std::max(stats->max, static_cast<double>(v));
        stats->absmax = std::max(stats->absmax, std::fabs(static_cast<double>(v)));
        sum += v;
    }
    stats->mean = sum / static_cast<double>(values.size());
    return true;
}

bool decode_block_and_check(const std::string &model_path,
                            const TensorInfo &info,
                            const TensorView &view,
                            long long *first_offset,
                            std::uint16_t *scale_bits_out,
                            float *scale_value_out,
                            std::vector<int> *q_values_out,
                            std::vector<float> *decoded_out) {
    if (info.gguf_type != 2 && info.gguf_type != 8) {
        return false;
    }
    std::ifstream file(model_path, std::ios::binary);
    if (!file) return false;
    const std::uint64_t block_bytes = info.gguf_type == 2 ? 18 : 34;
    const std::uint64_t block_elems = 32;
    const std::uint64_t n_blocks = view.byte_size / block_bytes;
    file.seekg(static_cast<std::streamoff>(view.data_begin), std::ios::beg);
    for (std::uint64_t b = 0; b < n_blocks; ++b) {
        std::uint16_t scale_bits = 0;
        file.read(reinterpret_cast<char*>(&scale_bits), 2);
        if (!file) return false;
        const float scale = ml_fp16_to_fp32(scale_bits);
        std::vector<float> decoded(block_elems, 0.0f);
        std::vector<int> qvals(block_elems, 0);
        if (info.gguf_type == 2) {
            unsigned char packed[16] = {};
            file.read(reinterpret_cast<char*>(packed), 16);
            if (!file) return false;
            if (!minllama::decode_q4_0_block_f32(scale_bits, packed, 16, decoded.data(), decoded.size())) {
                return false;
            }
            for (int i = 0; i < 16; ++i) {
                qvals[i] = static_cast<int>(packed[i] & 0x0f) - 8;
                qvals[i + 16] = static_cast<int>((packed[i] >> 4) & 0x0f) - 8;
            }
        } else {
            unsigned char qs[32] = {};
            file.read(reinterpret_cast<char*>(qs), 32);
            if (!file) return false;
            if (!minllama::decode_q8_0_block_f32(scale_bits, qs, 32, decoded.data(), decoded.size())) {
                return false;
            }
            for (int i = 0; i < 32; ++i) qvals[i] = static_cast<int>(static_cast<std::int8_t>(qs[i]));
        }

        bool bad = !std::isfinite(scale);
        for (float v : decoded) {
            if (std::isnan(v) || std::isinf(v) || std::fabs(v) > 1e6f) {
                bad = true;
                break;
            }
        }
        if (bad) {
            if (first_offset) *first_offset = static_cast<long long>(info.offset + b * block_bytes);
            if (scale_bits_out) *scale_bits_out = scale_bits;
            if (scale_value_out) *scale_value_out = scale;
            if (q_values_out) *q_values_out = qvals;
            if (decoded_out) *decoded_out = decoded;
            return true;
        }
    }
    return true;
}

void print_stats_row(const TensorStats &s, const std::string &shape) {
    std::cout << s.name << "\t" << type_name(s.type) << "\t" << shape
              << "\tmin=" << s.min
              << "\tmax=" << s.max
              << "\tmean=" << s.mean
              << "\tabsmax=" << s.absmax
              << "\tNaN=" << (s.has_nan ? "Y" : "N")
              << "\tInf=" << (s.has_inf ? "Y" : "N")
              << "\thuge=" << (s.has_huge ? "Y" : "N")
              << "\tfirst_bad_block=" << s.first_anomaly_block_offset
              << "\n";
}

} // namespace

int main(int argc, char **argv) {
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        }
    }
    if (model_path.empty()) {
        std::cerr << "Usage: minllama_scan_tensors --model <path>\n";
        return 1;
    }

    ml_model *ml = ml_model_load(model_path.c_str());
    if (!ml) {
        std::cerr << "failed to load model\n";
        return 1;
    }

    std::vector<TensorStats> rows;
    rows.reserve(ml->tensor_index.tensors.size());
    for (std::size_t i = 0; i < ml->tensor_index.tensors.size(); ++i) {
        const TensorInfo &info = ml->tensor_index.tensors[i];
        const TensorView &view = ml->tensor_index.views[i];
        TensorStats s;
        s.name = info.name;
        s.type = info.gguf_type;
        s.n_dims = info.n_dims;
        s.dims.assign(info.dims.begin(), info.dims.begin() + info.n_dims);

        std::vector<float> values;
        if (load_tensor_values(model_path, ml->tensor_index, info.name, &values)) {
            compute_stats(values, &s);
        }
        if ((info.gguf_type == 2 || info.gguf_type == 8)) {
            std::uint16_t scale_bits = 0;
            float scale = 0.0f;
            std::vector<int> qvals;
            std::vector<float> decoded;
            decode_block_and_check(model_path, info, view, &s.first_anomaly_block_offset,
                                   &scale_bits, &scale, &qvals, &decoded);
        }
        rows.push_back(s);
    }

    std::sort(rows.begin(), rows.end(), [](const TensorStats &a, const TensorStats &b) {
        const bool abad = a.has_nan || a.has_inf || a.has_huge;
        const bool bbad = b.has_nan || b.has_inf || b.has_huge;
        if (abad != bbad) return abad > bbad;
        return a.absmax > b.absmax;
    });

    std::cout << "Top 20 tensors by anomaly/absmax:\n";
    for (std::size_t i = 0; i < std::min<std::size_t>(20, rows.size()); ++i) {
        const auto &s = rows[i];
        std::string shape = "[";
        for (std::size_t j = 0; j < s.dims.size(); ++j) {
            if (j) shape += ",";
            shape += std::to_string(s.dims[j]);
        }
        shape += "]";
        print_stats_row(s, shape);
    }

    const char *focus[] = {
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.ffn_gate.weight",
        "output_norm.weight",
        "output.weight",
    };
    std::cout << "\nFocused tensors:\n";
    for (const char *name : focus) {
        auto it = std::find_if(rows.begin(), rows.end(), [&](const TensorStats &s){ return s.name == name; });
        if (it == rows.end()) continue;
        std::string shape = "[";
        for (std::size_t j = 0; j < it->dims.size(); ++j) {
            if (j) shape += ",";
            shape += std::to_string(it->dims[j]);
        }
        shape += "]";
        print_stats_row(*it, shape);
    }

    ml_model_free(ml);
    return 0;
}
