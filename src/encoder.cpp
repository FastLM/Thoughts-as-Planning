#include "tap/encoder.hpp"

#include <cstdint>

namespace tap {

void StateEncoder::init(const HyperParams& p, std::uint64_t seed) {
  rng = seed;
  d_model = p.d_latent;
  d_token = p.token_embed_dim;
  vocab_size = p.vocab_size;
  d_ff = p.d_ff;
  token_table.resize(vocab_size, d_token);
  std::uint64_t st = seed + 0x9E3779B97F4A7C15ull;
  for (int r = 0; r < vocab_size; ++r) {
    for (int c = 0; c < d_token; ++c) {
      st = st * 6364136223846793005ull + 1ull;
      token_table(r, c) = (static_cast<float>(st & 0xffffu) / 65536.0f) * 0.02f - 0.01f;
    }
  }
  rng = st;
  to_model.init(d_token, d_model, rng);
  blocks.clear();
  for (int b = 0; b < p.num_encoder_blocks; ++b) {
    ResidualFfn ff;
    ff.init(d_model, d_ff, rng);
    blocks.push_back(std::move(ff));
  }
}

// FNV-1a 32
static std::uint32_t hash_str(const char* s, int len) {
  std::uint32_t h = 2166136261u;
  for (int i = 0; i < len; ++i) {
    h ^= static_cast<std::uint8_t>(s[i]);
    h *= 16777619u;
  }
  return h;
}

void StateEncoder::tokenize_concat(const std::string& x, const std::string& c, std::vector<std::uint32_t>& ids) const {
  ids.clear();
  std::string t = x;
  t += " \n";
  t += c;
  const char* p = t.c_str();
  int n = static_cast<int>(t.size());
  int start = 0;
  for (int i = 0; i <= n; ++i) {
    if (i == n || p[i] == ' ' || p[i] == '\t' || p[i] == '\n' || p[i] == '\r') {
      if (i > start) {
        std::uint32_t h = hash_str(p + start, i - start);
        ids.push_back(h % static_cast<std::uint32_t>(vocab_size));
        if (static_cast<int>(ids.size()) >= max_seq_tokens) {
          return;
        }
      }
      start = i + 1;
    }
  }
  if (ids.empty()) {
    ids.push_back(0);
  }
}

void StateEncoder::forward(const State& s, Vector& z, std::vector<std::uint32_t>*) const {
  std::vector<std::uint32_t> ids;
  tokenize_concat(s.task_input, s.chain, ids);
  Vector pooled = Vector::Zero(d_token);
  for (const auto& id : ids) {
    pooled += token_table.row(static_cast<int>(id)).transpose();
  }
  float inv = 1.0f / static_cast<float>(ids.size());
  pooled *= inv;
  Vector u;
  to_model.forward(pooled, u);
  z = u;
  for (const auto& ff : blocks) {
    Vector t1, g, y;
    ff.forward(z, t1, g, y);
    z = y;
  }
}

int StateEncoder::num_params() const {
  int t = d_token * vocab_size;
  t += d_token * d_model + d_model;  // to_model
  for (const auto& ff : blocks) {
    t += ff.w1.in_dim * ff.w1.out_dim + ff.w1.out_dim;
    t += ff.w2.in_dim * ff.w2.out_dim + ff.w2.out_dim;
  }
  return t;
}

}  // namespace tap
