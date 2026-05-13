#include "minllama_internal.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

namespace minllama {

// ----------------------------------------------------------------
// GPT-2 byte <-> unicode tables
// ----------------------------------------------------------------

void bpe_build_byte_tables(BpeTokenizer &tok) {
    tok.byte_encoder.clear();
    tok.byte_decoder.clear();

    // Build the set of bytes that have natural unicode representations.
    // These are: printable ASCII (33-126), plus some latin-1 chars.
    unsigned char natural[256] = {};
    for (int b = 33; b <= 126; ++b) natural[b] = 1;     // '!' to '~'
    for (int b = 161; b <= 172; ++b) natural[b] = 1;    // '¡' to '¬'
    for (int b = 174; b <= 255; ++b) natural[b] = 1;    // '®' to 'ÿ'

    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (natural[b]) {
            // Map to its own unicode codepoint as UTF-8
            std::string ch;
            unsigned int cp = static_cast<unsigned int>(b);
            if (cp < 0x80) {
                ch = static_cast<char>(cp);
            } else {
                ch += static_cast<char>(0xC0 | (cp >> 6));
                ch += static_cast<char>(0x80 | (cp & 0x3F));
            }
            tok.byte_encoder[b] = ch;
            tok.byte_decoder[ch] = static_cast<unsigned char>(b);
        } else {
            // Map to unicode private use area starting at U+0100
            unsigned int cp = 256 + extra;
            std::string ch;
            if (cp < 0x800) {
                ch += static_cast<char>(0xC0 | (cp >> 6));
                ch += static_cast<char>(0x80 | (cp & 0x3F));
            }
            tok.byte_encoder[b] = ch;
            tok.byte_decoder[ch] = static_cast<unsigned char>(b);
            ++extra;
        }
    }
}

// ----------------------------------------------------------------
// GGUF file-level read helpers (replicated to avoid coupling)
// ----------------------------------------------------------------

namespace {

bool read_u32_le(std::ifstream &f, std::uint32_t *out) {
    unsigned char b[4];
    f.read(reinterpret_cast<char *>(b), 4);
    if (f.gcount() != 4) return false;
    *out = static_cast<std::uint32_t>(b[0]) |
           (static_cast<std::uint32_t>(b[1]) << 8) |
           (static_cast<std::uint32_t>(b[2]) << 16) |
           (static_cast<std::uint32_t>(b[3]) << 24);
    return true;
}

bool read_u64_le(std::ifstream &f, std::uint64_t *out) {
    unsigned char b[8];
    f.read(reinterpret_cast<char *>(b), 8);
    if (f.gcount() != 8) return false;
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[i]) << (8 * i);
    *out = v;
    return true;
}

bool read_i32_le(std::ifstream &f, std::int32_t *out) {
    std::uint32_t bits = 0;
    if (!read_u32_le(f, &bits)) return false;
    *out = static_cast<std::int32_t>(bits);
    return true;
}

bool read_string(std::ifstream &f, std::string *out) {
    std::uint64_t size = 0;
    if (!read_u64_le(f, &size)) return false;
    if (size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return false;
    out->resize(static_cast<std::size_t>(size));
    if (size > 0) {
        f.read(out->data(), static_cast<std::streamsize>(size));
        if (f.gcount() != static_cast<std::streamsize>(size)) return false;
    }
    return true;
}

bool skip_value(std::ifstream &f, std::uint32_t type) {
    constexpr std::uint32_t kU32 = 4, kI32 = 5, kF32 = 6;
    constexpr std::uint32_t kBool = 7, kStr = 8, kArr = 9;
    constexpr std::uint32_t kU64 = 10, kF64 = 12;

    switch (type) {
    case kU32: case kI32: case kF32: { std::uint32_t x; return read_u32_le(f, &x); }
    case kU64: case kF64: { std::uint64_t x; return read_u64_le(f, &x); }
    case kBool: { unsigned char x; return static_cast<bool>(f.read(reinterpret_cast<char*>(&x), 1)); }
    case kStr: { std::string x; return read_string(f, &x); }
    case kArr: {
        std::uint32_t et; std::uint64_t n;
        if (!read_u32_le(f, &et) || !read_u64_le(f, &n)) return false;
        for (std::uint64_t i = 0; i < n; ++i)
            if (!skip_value(f, et)) return false;
        return true;
    }
    default: return false;
    }
}

} // anonymous

// ----------------------------------------------------------------
// Load BPE tokenizer from GGUF
// ----------------------------------------------------------------

bool bpe_tokenizer_load(const std::string &gguf_path, BpeTokenizer &tok, std::string *error) {
    auto fail = [error](const std::string &msg) {
        if (error) *error = msg;
        return false;
    };

    std::ifstream f(gguf_path, std::ios::binary);
    if (!f) return fail("cannot open GGUF");

    // Magic
    char magic[4];
    f.read(magic, 4);
    if (f.gcount() != 4 || std::memcmp(magic, "GGUF", 4) != 0)
        return fail("bad GGUF magic");

    // Version
    std::uint32_t version;
    if (!read_u32_le(f, &version) || version != 3)
        return fail("unsupported GGUF version");

    // n_tensors, n_kv
    std::uint64_t n_tensors, n_kv;
    if (!read_u64_le(f, &n_tensors) || !read_u64_le(f, &n_kv))
        return fail("bad GGUF header");

    bool has_tokens = false, has_merges = false;

    for (std::uint64_t i = 0; i < n_kv; ++i) {
        std::string key;
        std::uint32_t type_raw;
        if (!read_string(f, &key) || !read_u32_le(f, &type_raw))
            return fail("KV read error");

        if (key == "tokenizer.ggml.tokens") {
            if (type_raw != 9) return fail("tokens not array");
            std::uint32_t et; std::uint64_t n;
            if (!read_u32_le(f, &et) || !read_u64_le(f, &n)) return fail("tokens header");
            if (et != 8) return fail("tokens elem not string");
            tok.vocab.reserve(static_cast<std::size_t>(n));
            tok.token_to_id.reserve(static_cast<std::size_t>(n));
            for (std::uint64_t j = 0; j < n; ++j) {
                std::string s;
                if (!read_string(f, &s)) return fail("token read fail");
                tok.token_to_id[s] = static_cast<int>(tok.vocab.size());
                tok.vocab.push_back(std::move(s));
            }
            has_tokens = true;
        } else if (key == "tokenizer.ggml.merges") {
            if (type_raw != 9) return fail("merges not array");
            std::uint32_t et; std::uint64_t n;
            if (!read_u32_le(f, &et) || !read_u64_le(f, &n)) return fail("merges header");
            if (et != 8) return fail("merges elem not string");
            tok.merges.reserve(static_cast<std::size_t>(n));
            tok.merge_rank.reserve(static_cast<std::size_t>(n));
            for (std::uint64_t j = 0; j < n; ++j) {
                std::string s;
                if (!read_string(f, &s)) return fail("merge read fail");
                tok.merges.push_back(s);
                tok.merge_rank[s] = static_cast<int>(j);
            }
            has_merges = true;
        } else if (key == "tokenizer.ggml.bos_token_id") {
            if (type_raw != 4) return fail("bos not u32");
            std::uint32_t v;
            if (!read_u32_le(f, &v)) return fail("bos read");
            tok.bos_token_id = static_cast<int>(v);
        } else if (key == "tokenizer.ggml.eos_token_id") {
            if (type_raw != 4) return fail("eos not u32");
            std::uint32_t v;
            if (!read_u32_le(f, &v)) return fail("eos read");
            tok.eos_token_id = static_cast<int>(v);
        } else if (key == "tokenizer.ggml.unknown_token_id") {
            if (type_raw != 4) return fail("unk not u32");
            std::uint32_t v;
            if (!read_u32_le(f, &v)) return fail("unk read");
            tok.unk_token_id = static_cast<int>(v);
        } else if (key == "tokenizer.ggml.add_bos_token") {
            if (type_raw != 7) return fail("add_bos not bool");
            unsigned char v;
            if (!f.read(reinterpret_cast<char*>(&v), 1)) return fail("add_bos read");
            tok.add_bos_token = (v != 0);
        } else if (key == "tokenizer.ggml.add_eos_token") {
            if (type_raw != 7) return fail("add_eos not bool");
            unsigned char v;
            if (!f.read(reinterpret_cast<char*>(&v), 1)) return fail("add_eos read");
            tok.add_eos_token = (v != 0);
        } else {
            if (!skip_value(f, type_raw)) return fail("skip: " + key);
        }
    }

    if (!has_tokens) return fail("no tokens found");
    if (!has_merges) return fail("no merges found");

    // Build byte encoder/decoder tables
    bpe_build_byte_tables(tok);

    return true;
}

// ----------------------------------------------------------------
// BPE Encode
// ----------------------------------------------------------------

// Get the rank of a merge pair, or a large number if not in merges.
static int get_merge_rank(const BpeTokenizer &tok,
                          const std::string &left,
                          const std::string &right) {
    auto it = tok.merge_rank.find(left + " " + right);
    if (it != tok.merge_rank.end()) return it->second;
    return std::numeric_limits<int>::max();
}

// Apply BPE merges to a list of tokens (in-place).
static void bpe_apply_merges(const BpeTokenizer &tok,
                             std::vector<std::string> &tokens) {
    if (tokens.size() <= 1) return;

    while (true) {
        int best_rank = std::numeric_limits<int>::max();
        std::size_t best_i = 0;

        // Find the highest-priority (lowest rank) merge position.
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
            int r = get_merge_rank(tok, tokens[i], tokens[i + 1]);
            if (r < best_rank) {
                best_rank = r;
                best_i = i;
            }
        }

        if (best_rank == std::numeric_limits<int>::max()) break;

        // Apply the merge.
        std::string merged = tokens[best_i] + tokens[best_i + 1];
        tokens[best_i] = std::move(merged);
        tokens.erase(tokens.begin() + static_cast<long>(best_i) + 1);
    }
}

bool bpe_encode(const BpeTokenizer &tok, const std::string &text,
                std::vector<int> &ids) {
    ids.clear();

    if (tok.add_bos_token && tok.bos_token_id >= 0)
        ids.push_back(tok.bos_token_id);

    if (text.empty()) return true;

    // Step 1: Convert text to bytes, then bytes to unicode chars.
    std::string unicode_text;
    for (unsigned char c : text) {
        auto it = tok.byte_encoder.find(c);
        if (it != tok.byte_encoder.end()) {
            unicode_text += it->second;
        } else {
            // Fallback: use raw byte as char (shouldn't happen)
            unicode_text += static_cast<char>(c);
        }
    }

    // Step 2: GPT-2 style pre-tokenization.
    // Key rules from the GPT-2 regex pattern:
    // - " ?\\p{L}+" etc. = optional SINGLE space absorbed into next word
    // - "\\s+(?!\\S)" = trailing whitespace at end
    // - "\\s+" = other whitespace sequences (multispace, tab, newline)
    //
    // Implementation: split into spans of non-whitespace characters,
    // noting how many leading spaces each span has.  Non-space whitespace
    // (tab, newline) creates a boundary where spaces are flushed and
    // the next word does NOT get a Ġ prefix (it's like a new line start).

    struct Span { int lead_spaces; std::string chars; };
    std::vector<Span> spans;
    bool segment_start = true;  // true at start of text, after tab/newline
    {
        std::string::size_type i = 0;
        while (i < unicode_text.size()) {
            int lead_spaces = 0;
            std::string ch;

            // Read whitespace, handling tab/newline specially.
            while (i < unicode_text.size()) {
                unsigned char lead = static_cast<unsigned char>(unicode_text[i]);
                std::size_t clen = 1;
                if ((lead & 0x80) == 0) clen = 1;
                else if ((lead & 0xE0) == 0xC0) clen = 2;
                else if ((lead & 0xF0) == 0xE0) clen = 3;
                else clen = 4;
                if (i + clen > unicode_text.size()) clen = unicode_text.size() - i;
                ch = unicode_text.substr(i, clen);

                unsigned char orig_byte = 0;
                bool is_ws = false;
                auto dit = tok.byte_decoder.find(ch);
                if (dit != tok.byte_decoder.end()) {
                    orig_byte = dit->second;
                    is_ws = (orig_byte == ' ' || orig_byte == '\t' ||
                             orig_byte == '\n' || orig_byte == '\r');
                }
                if (ch == " ") { is_ws = true; orig_byte = ' '; }

                if (!is_ws) break;

                if (orig_byte == ' ') {
                    lead_spaces++;
                    i += clen;
                } else {
                    // Tab, newline, or CR: flush preceding spaces, emit ws byte token,
                    // and start a new segment.
                    for (int s = 0; s < lead_spaces; ++s)
                        spans.push_back({0, "\xC4\xA0"}); // Ġ as separate span
                    spans.push_back({0, ch});
                    lead_spaces = 0;
                    segment_start = true;
                    i += clen;
                }
            }

            // Read non-whitespace characters.
            std::string word_chars;
            while (i < unicode_text.size()) {
                unsigned char lead = static_cast<unsigned char>(unicode_text[i]);
                std::size_t clen = 1;
                if ((lead & 0x80) == 0) clen = 1;
                else if ((lead & 0xE0) == 0xC0) clen = 2;
                else if ((lead & 0xF0) == 0xE0) clen = 3;
                else clen = 4;
                if (i + clen > unicode_text.size()) clen = unicode_text.size() - i;
                ch = unicode_text.substr(i, clen);

                unsigned char orig_byte = 0;
                bool is_ws = false;
                auto dit = tok.byte_decoder.find(ch);
                if (dit != tok.byte_decoder.end()) {
                    orig_byte = dit->second;
                    is_ws = (orig_byte == ' ' || orig_byte == '\t' ||
                             orig_byte == '\n' || orig_byte == '\r');
                }
                if (ch == " ") { is_ws = true; }

                if (is_ws) break;
                word_chars += ch;
                i += clen;
            }

            spans.push_back({lead_spaces, word_chars});
            segment_start = false;
        }
    }

    // Process spans into BPE words.
    std::vector<std::vector<std::string>> words;
    bool is_first_segment = true;
    for (auto &span : spans) {
        if (span.chars.empty() && span.lead_spaces > 0) {
            // Whitespace-only span: group all spaces into one word
            // so BPE can merge them (e.g., Ġ+Ġ → "ĠĠ" = token 256)
            std::vector<std::string> space_word;
            for (int s = 0; s < span.lead_spaces; ++s)
                space_word.push_back("\xC4\xA0");
            words.push_back(std::move(space_word));
            continue;
        }

        std::vector<std::string> current;

        // Handle leading spaces.
        // Rule: the first space before a word is absorbed as Ġ prefix.
        // Additional spaces become a separate word (for BPE merging).
        // For the first segment of text (or after tab/newline), the same rule
        // applies: leading spaces still absorb one into the word.
        if (span.lead_spaces > 0) {
            // Absorb the first space as Ġ prefix.
            // Emit remaining spaces as a separate word (grouped for BPE merging).
            if (span.lead_spaces > 1) {
                std::vector<std::string> extra_spaces;
                for (int s = 1; s < span.lead_spaces; ++s)
                    extra_spaces.push_back("\xC4\xA0");
                words.push_back(std::move(extra_spaces));
            }
            current.push_back("\xC4\xA0");
        } else if (!is_first_segment) {
            // No leading spaces for a non-first word in a segment.
            // This happens after non-space whitespace (tab/newline).
            // Don't add Ġ prefix — the word starts fresh.
        }

        // Add word characters.
        for (std::string::size_type i = 0; i < span.chars.size(); ) {
            unsigned char lead = static_cast<unsigned char>(span.chars[i]);
            std::size_t clen = 1;
            if ((lead & 0x80) == 0) clen = 1;
            else if ((lead & 0xE0) == 0xC0) clen = 2;
            else if ((lead & 0xF0) == 0xE0) clen = 3;
            else clen = 4;
            if (i + clen > span.chars.size()) clen = span.chars.size() - i;
            current.push_back(span.chars.substr(i, clen));
            i += clen;
        }

        words.push_back(std::move(current));
        is_first_segment = false;
    }

    // Step 3: For each word, apply BPE merges, then map to vocab.
    for (auto &word : words) {
        // Apply merges to this word's character list.
        bpe_apply_merges(tok, word);

        // Convert merged tokens to IDs, with byte-level fallback.
        for (const auto &token : word) {
            auto it = tok.token_to_id.find(token);
            if (it != tok.token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                // Byte fallback: encode each UTF-8 byte of the token
                // as an individual byte-level token.
                bool all_ok = true;
                for (unsigned char byte_c : token) {
                    auto bit = tok.byte_encoder.find(byte_c);
                    std::string byte_str;
                    if (bit != tok.byte_encoder.end()) {
                        byte_str = bit->second;
                    } else {
                        byte_str = static_cast<char>(byte_c);
                    }
                    auto bt = tok.token_to_id.find(byte_str);
                    if (bt != tok.token_to_id.end()) {
                        ids.push_back(bt->second);
                    } else {
                        ids.push_back(tok.unk_token_id >= 0 ? tok.unk_token_id : 0);
                        all_ok = false;
                    }
                }
                (void)all_ok;
            }
        }
    }

    return true;
}

// ----------------------------------------------------------------
// BPE Decode
// ----------------------------------------------------------------

bool bpe_decode(const BpeTokenizer &tok, const int *ids, int count,
                std::string &text) {
    text.clear();
    if (count <= 0 || !ids) return true;

    // Step 1: Convert token IDs to strings, join.
    std::string joined;
    for (int i = 0; i < count; ++i) {
        int id = ids[i];
        if (id < 0 || id >= static_cast<int>(tok.vocab.size())) return false;
        joined += tok.vocab[id];
    }

    // Step 2: Convert unicode chars back to bytes.
    std::vector<unsigned char> bytes;

    for (std::string::size_type i = 0; i < joined.size(); ) {
        unsigned char lead = static_cast<unsigned char>(joined[i]);
        std::size_t clen = 1;
        if ((lead & 0x80) == 0) clen = 1;
        else if ((lead & 0xE0) == 0xC0) clen = 2;
        else if ((lead & 0xF0) == 0xE0) clen = 3;
        else clen = 4;
        if (i + clen > joined.size()) clen = joined.size() - i;

        std::string ch = joined.substr(i, clen);

        // Step 2a: Replace Ġ (U+0120) with space.
        if (ch == "\xC4\xA0") {
            bytes.push_back(' ');
            i += clen;
            continue;
        }

        // Step 2b: Look up in byte decoder.
        auto it = tok.byte_decoder.find(ch);
        if (it != tok.byte_decoder.end()) {
            bytes.push_back(it->second);
        } else {
            // Unknown character — try to interpret as raw byte.
            for (std::size_t k = 0; k < clen; ++k)
                bytes.push_back(static_cast<unsigned char>(joined[i + k]));
        }
        i += clen;
    }

    // Step 3: Convert bytes to string.
    text.assign(reinterpret_cast<char*>(bytes.data()), bytes.size());
    return true;
}

} // namespace minllama
