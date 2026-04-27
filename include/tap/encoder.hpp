#pragma once

#include "tap/config.hpp"
#include "tap/layers.hpp"
#include "tap/types.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <vector>

namespace tap {

// Implements the paper's state-to-latent mechanism by mapping a concatenation of task
// input and reasoning chain into a vector z in R^d. A full self-attention encoder can be
// used as well; this reference does it through token mean-pooling, a linear up-projection,
// and a stack of residual feed-forward blocks, without changing the world model or planner
// interface.
struct StateEncoder {
  int d_model{};
  int d_token{};
  int vocab_size{};
  int d_ff{};
  int max_seq_tokens{128};

  // Token embedding E_vocab * d_token
  Matrix token_table;  // (vocab_size, d_token) row per id
  Linear to_model;     // d_token -> d_model

  std::vector<ResidualFfn> blocks;  // num_encoder_blocks
  std::uint64_t rng{};

  void init(const HyperParams& p, std::uint64_t seed);
  // Encode a state s = (x, c) to latent z
  void forward(const State& s, Vector& z, std::vector<std::uint32_t>* out_ids) const;
  // Token ids (hashed) for a concatenated "x || c" text
  void tokenize_concat(const std::string& x, const std::string& c, std::vector<std::uint32_t>& ids) const;

  int num_params() const;
};

}  // namespace tap
