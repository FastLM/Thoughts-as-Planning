#pragma once

#include "tap/config.hpp"
#include "tap/layers.hpp"
#include "tap/types.hpp"

namespace tap {

// Implements the paper's latent world model: next latent is the previous latent plus
// a residual MLP of [z, embed(action)].
struct TransitionModel {
  int d_latent{};
  int d_action_embed{};
  int d_hidden{};
  int n_actions{};

  Matrix action_embed;  // (n_actions, d_action_embed)
  Linear lin1, lin2;   // in = d_latent + d_action_embed, hidden, out d_latent

  std::uint64_t rng{};
  void init(const HyperParams& p, int n_action_types, std::uint64_t seed);
  // z' = z + g(z, a)
  // z' = z + MLP(concat(z, e(a)))
  void forward(const Vector& z, int action_id, Vector& z_next) const;
  // H steps with the same action (model-based lookahead when scoring candidates in planning)
  void roll_same_action(const Vector& z0, int action_id, int H, Vector& zH) const;

  int num_params() const;
};

}  // namespace tap
