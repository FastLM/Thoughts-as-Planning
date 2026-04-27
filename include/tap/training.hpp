#pragma once

#include "tap/config.hpp"
#include "tap/encoder.hpp"
#include "tap/transition.hpp"
#include "tap/reward.hpp"
#include "tap/types.hpp"

#include <Eigen/Dense>
#include <vector>

namespace tap {

// Joint dynamics and reward training on a batch; combines the paper's transition and
// reward objectives (SGD here; Adam is an easy drop-in)
struct WorldModelSystem {
  StateEncoder enc;
  TransitionModel trans;
  RewardModel rew;
  HyperParams hp;
  int n_actions{};

  void init(const HyperParams& p, int n_action_types, std::uint64_t seed);

  // MSE on dynamics + MSE on reward. Returns (L_dyn, L_rew) before average
  std::pair<float, float> train_on_batch(
      const std::vector<Transition>& batch, float lr,
      int step_index);

  float predict_reward(const State& s) const;
  void save_latent(const State& s, Eigen::VectorXf& z) const;  // utility
};

}  // namespace tap
