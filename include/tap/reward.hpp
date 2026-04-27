#pragma once

#include "tap/config.hpp"
#include "tap/layers.hpp"

namespace tap {

// Implements the paper's learned utility: scalar reward as an MLP on latent z
struct RewardModel {
  int d_latent{};
  int d_hidden{};
  Linear lin1, lin2;  // d -> d_hidden -> 1
  std::uint64_t rng{};

  void init(const HyperParams& p, std::uint64_t seed);
  float forward(const Vector& z, Vector& h, Vector& pre2) const;

  int num_params() const;
};

}  // namespace tap
