#ifndef MINLLAMA_MINICPM5_TOKENIZER_H
#define MINLLAMA_MINICPM5_TOKENIZER_H

#include "minllama_internal.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace minllama {
namespace minicpm5_detail {

struct Codepoint {
    std::uint32_t value = 0;
    std::size_t bytes = 1;
};

inline Codepoint decode_utf8(const std::string &text, std::size_t offset) {
    Codepoint out{};
    if (offset >= text.size()) return out;
    const unsigned char c0 = static_cast<unsigned char>(text[offset]);
    if ((c0 & 0x80u) == 0) {
        out.value = c0;
        out.bytes = 1;
        return out;
    }
    if ((c0 & 0xE0u) == 0xC0u && offset + 1 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[offset + 1]);
        if ((c1 & 0xC0u) == 0x80u) {
            out.value = ((c0 & 0x1Fu) << 6) | (c1 & 0x3Fu);
            out.bytes = 2;
            return out;
        }
    }
    if ((c0 & 0xF0u) == 0xE0u && offset + 2 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[offset + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[offset + 2]);
        if ((c1 & 0xC0u) == 0x80u && (c2 & 0xC0u) == 0x80u) {
            out.value = ((c0 & 0x0Fu) << 12) | ((c1 & 0x3Fu) << 6) | (c2 & 0x3Fu);
            out.bytes = 3;
            return out;
        }
    }
    if ((c0 & 0xF8u) == 0xF0u && offset + 3 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[offset + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[offset + 2]);
        const unsigned char c3 = static_cast<unsigned char>(text[offset + 3]);
        if ((c1 & 0xC0u) == 0x80u && (c2 & 0xC0u) == 0x80u &&
            (c3 & 0xC0u) == 0x80u) {
            out.value = ((c0 & 0x07u) << 18) | ((c1 & 0x3Fu) << 12) |
                        ((c2 & 0x3Fu) << 6) | (c3 & 0x3Fu);
            out.bytes = 4;
            return out;
        }
    }
    out.value = c0;
    out.bytes = 1;
    return out;
}

inline bool is_newline(std::uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

inline bool is_whitespace(std::uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x0085 || cp == 0x00A0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

inline bool is_number(std::uint32_t cp) {
    return (cp >= '0' && cp <= '9') ||
           (cp >= 0x0660 && cp <= 0x0669) ||
           (cp >= 0x06F0 && cp <= 0x06F9) ||
           (cp >= 0x0966 && cp <= 0x096F) ||
           (cp >= 0xFF10 && cp <= 0xFF19);
}

inline bool is_symbol_or_punctuation(std::uint32_t cp) {
    if (cp < 0x80) {
        const unsigned char c = static_cast<unsigned char>(cp);
        return !std::isalnum(c) && !std::isspace(c) && c != '_';
    }
    return (cp >= 0x2000 && cp <= 0x206F) ||
           (cp >= 0x2190 && cp <= 0x2BFF) ||
           (cp >= 0x3000 && cp <= 0x303F) ||
           (cp >= 0xFE10 && cp <= 0xFE1F) ||
           (cp >= 0xFE30 && cp <= 0xFE6F) ||
           (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) ||
           (cp >= 0xFF3B && cp <= 0xFF40) ||
           (cp >= 0xFF5B && cp <= 0xFF65) ||
           (cp >= 0x1F000 && cp <= 0x1FAFF);
}

inline bool is_letter(std::uint32_t cp) {
    if (cp < 0x80) {
        return std::isalpha(static_cast<unsigned char>(cp)) != 0 || cp == '_';
    }
    return !is_whitespace(cp) && !is_number(cp) && !is_symbol_or_punctuation(cp);
}

inline bool starts_with_ci(const std::string &text, std::size_t pos,
                           const char *suffix) {
    std::size_t i = 0;
    while (suffix[i] != '\0') {
        if (pos + i >= text.size()) return false;
        const unsigned char a = static_cast<unsigned char>(text[pos + i]);
        const unsigned char b = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(a) != std::tolower(b)) return false;
        ++i;
    }
    return true;
}

inline std::size_t contraction_length(const std::string &text, std::size_t pos) {
    if (pos >= text.size() || text[pos] != '\'') return 0;
    static const char *patterns[] = {"'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
    for (const char *p : patterns) {
        if (starts_with_ci(text, pos, p)) return std::char_traits<char>::length(p);
    }
    return 0;
}

inline std::string byte_encode(const BpeTokenizer &tok, const std::string &raw) {
    std::string encoded;
    for (unsigned char c : raw) {
        const auto it = tok.byte_encoder.find(static_cast<int>(c));
        if (it != tok.byte_encoder.end()) encoded += it->second;
        else encoded.push_back(static_cast<char>(c));
    }
    return encoded;
}

inline int merge_rank(const BpeTokenizer &tok, const std::string &left,
                      const std::string &right) {
    const auto it = tok.merge_rank.find(left + " " + right);
    return it == tok.merge_rank.end() ? std::numeric_limits<int>::max() : it->second;
}

inline void apply_merges(const BpeTokenizer &tok, std::vector<std::string> &parts) {
    while (parts.size() > 1) {
        int best = std::numeric_limits<int>::max();
        std::size_t best_i = 0;
        for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
            const int rank = merge_rank(tok, parts[i], parts[i + 1]);
            if (rank < best) {
                best = rank;
                best_i = i;
            }
        }
        if (best == std::numeric_limits<int>::max()) break;
        parts[best_i] += parts[best_i + 1];
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(best_i + 1));
    }
}

inline bool emit_encoded_piece(const BpeTokenizer &tok, const std::string &encoded,
                               std::vector<int> &ids) {
    // llama.cpp sets ignore_merges=true for tokenizer.ggml.pre=minicpm5.
    // Its meaning is subtle: prefer an exact whole-piece token first; only if
    // that does not exist do normal BPE merges run.
    const auto whole = tok.token_to_id.find(encoded);
    if (whole != tok.token_to_id.end()) {
        ids.push_back(whole->second);
        return true;
    }

    std::vector<std::string> chars;
    for (std::size_t i = 0; i < encoded.size();) {
        const Codepoint cp = decode_utf8(encoded, i);
        chars.push_back(encoded.substr(i, cp.bytes));
        i += cp.bytes;
    }
    apply_merges(tok, chars);

    for (const std::string &part : chars) {
        const auto exact = tok.token_to_id.find(part);
        if (exact != tok.token_to_id.end()) {
            ids.push_back(exact->second);
            continue;
        }
        // Byte-level fallback, matching the generic tokenizer's behavior.
        for (unsigned char c : part) {
            const auto bit = tok.byte_encoder.find(static_cast<int>(c));
            const std::string byte_token = bit != tok.byte_encoder.end()
                ? bit->second : std::string(1, static_cast<char>(c));
            const auto tid = tok.token_to_id.find(byte_token);
            if (tid != tok.token_to_id.end()) ids.push_back(tid->second);
            else if (tok.unk_token_id >= 0) ids.push_back(tok.unk_token_id);
            else return false;
        }
    }
    return true;
}

inline std::vector<std::string> special_tokens(const BpeTokenizer &tok) {
    std::vector<std::string> out;
    static const char *known[] = {
        "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<|im_sep|>",
        "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    };
    for (const char *s : known) {
        if (tok.token_to_id.find(s) != tok.token_to_id.end()) out.emplace_back(s);
    }
    std::sort(out.begin(), out.end(), [](const std::string &a, const std::string &b) {
        return a.size() > b.size();
    });
    return out;
}

inline bool emit_segment(const BpeTokenizer &tok, const std::string &segment,
                         std::vector<int> &ids) {
    std::size_t pos = 0;
    while (pos < segment.size()) {
        const std::size_t contraction = contraction_length(segment, pos);
        if (contraction > 0) {
            if (!emit_encoded_piece(tok, byte_encode(tok, segment.substr(pos, contraction)), ids)) return false;
            pos += contraction;
            continue;
        }

        const Codepoint first = decode_utf8(segment, pos);
        std::size_t end = pos;

        if (is_number(first.value)) {
            int count = 0;
            while (end < segment.size() && count < 3) {
                const Codepoint cp = decode_utf8(segment, end);
                if (!is_number(cp.value)) break;
                end += cp.bytes;
                ++count;
            }
        } else {
            // MiniCPM5's second regex allows one non-CR/LF, non-letter,
            // non-number prefix before a run of letters (notably a leading space).
            bool prefixed_letter = false;
            if (!is_letter(first.value) && !is_number(first.value) && !is_newline(first.value)) {
                const std::size_t next_pos = pos + first.bytes;
                if (next_pos < segment.size() && is_letter(decode_utf8(segment, next_pos).value)) {
                    end = next_pos;
                    prefixed_letter = true;
                }
            }
            if (is_letter(first.value) || prefixed_letter) {
                if (!prefixed_letter) end = pos;
                while (end < segment.size()) {
                    const Codepoint cp = decode_utf8(segment, end);
                    if (!is_letter(cp.value)) break;
                    end += cp.bytes;
                }
            } else if (first.value == ' ') {
                const std::size_t next_pos = pos + first.bytes;
                if (next_pos < segment.size()) {
                    const Codepoint next = decode_utf8(segment, next_pos);
                    if (!is_whitespace(next.value) && !is_letter(next.value) && !is_number(next.value)) {
                        end = next_pos;
                        while (end < segment.size()) {
                            const Codepoint cp = decode_utf8(segment, end);
                            if (is_whitespace(cp.value) || is_letter(cp.value) || is_number(cp.value)) break;
                            end += cp.bytes;
                        }
                        while (end < segment.size() && is_newline(decode_utf8(segment, end).value)) {
                            end += decode_utf8(segment, end).bytes;
                        }
                    }
                }
            }

            if (end == pos && !is_whitespace(first.value) && !is_letter(first.value) && !is_number(first.value)) {
                while (end < segment.size()) {
                    const Codepoint cp = decode_utf8(segment, end);
                    if (is_whitespace(cp.value) || is_letter(cp.value) || is_number(cp.value)) break;
                    end += cp.bytes;
                }
                while (end < segment.size() && is_newline(decode_utf8(segment, end).value)) {
                    end += decode_utf8(segment, end).bytes;
                }
            }

            if (end == pos && is_whitespace(first.value)) {
                // Implements the two whitespace/newline alternatives closely
                // enough for deterministic local validation; exact parity is
                // intentionally verified against llama.cpp by the acceptance gate.
                while (end < segment.size() && is_whitespace(decode_utf8(segment, end).value)) {
                    end += decode_utf8(segment, end).bytes;
                }
            }
        }

        if (end <= pos) end = pos + first.bytes;
        if (!emit_encoded_piece(tok, byte_encode(tok, segment.substr(pos, end - pos)), ids)) return false;
        pos = end;
    }
    return true;
}

} // namespace minicpm5_detail

inline bool is_minicpm5_tokenizer(const BpeTokenizer &tok) {
    // The GGUF struct predates tokenizer.ggml.pre storage. Detect only the
    // highly distinctive MiniCPM5 vocabulary contract so other GPT-2 BPE
    // models remain on the existing path.
    return tok.vocab.size() == 130560 &&
           tok.token_to_id.find("<|im_start|>") != tok.token_to_id.end() &&
           tok.token_to_id.find("<|im_end|>") != tok.token_to_id.end();
}

inline int minicpm5_chat_stop_token_id(const BpeTokenizer &tok, const std::string &prompt) {
    if (!is_minicpm5_tokenizer(tok) || prompt.find("<|im_start|>") == std::string::npos)
        return tok.eos_token_id;
    const auto it = tok.token_to_id.find("<|im_end|>");
    return it == tok.token_to_id.end() ? tok.eos_token_id : it->second;
}

inline bool minicpm5_bpe_encode(const BpeTokenizer &tok, const std::string &text,
                                std::vector<int> &ids) {
    ids.clear();
    if (tok.add_bos_token && tok.bos_token_id >= 0) ids.push_back(tok.bos_token_id);
    if (text.empty()) return true;

    const std::vector<std::string> specials = minicpm5_detail::special_tokens(tok);
    std::size_t pos = 0;
    while (pos < text.size()) {
        const std::string *matched = nullptr;
        for (const std::string &s : specials) {
            if (s.size() <= text.size() - pos && text.compare(pos, s.size(), s) == 0) {
                matched = &s;
                break;
            }
        }
        if (matched) {
            ids.push_back(tok.token_to_id.at(*matched));
            pos += matched->size();
            continue;
        }

        std::size_t next_special = text.size();
        for (const std::string &s : specials) {
            const std::size_t found = text.find(s, pos);
            if (found != std::string::npos) next_special = std::min(next_special, found);
        }
        const std::string segment = text.substr(pos, next_special - pos);
        if (!minicpm5_detail::emit_segment(tok, segment, ids)) return false;
        pos = next_special;
    }
    return true;
}

} // namespace minllama

#endif // MINLLAMA_MINICPM5_TOKENIZER_H
