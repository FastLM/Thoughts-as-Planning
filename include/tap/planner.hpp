#pragma once

#include "tap/config.hpp"
#include "tap/encoder.hpp"
#include "tap/transition.hpp"
#include "tap/reward.hpp"
#include "tap/types.hpp"

#include <string>
#include <vector>

namespace tap {

// Outcome of the paper's planning loop: candidate edits, latent rollouts, real chain updates
struct PlanResult {
  std::string final_chain;
  std::vector<std::string> trace_chains;  // c_0, c_1, ...
  std::vector<int> selected_action_ids;   // per step
  std::vector<double> pred_rewards;       // R_hat at lookahead state for chosen branch
};

// Score rollouts in latent space, apply the chosen edit in text, and advance z with T_hat
// (do not re-encode the new chain to get z_{t+1})
PlanResult run_planning(
    const State& s0,
    const StateEncoder& enc,
    const TransitionModel& trans,
    const RewardModel& rew,
    const HyperParams& hp,
    int n_actions,
    std::uint64_t& seed,
    bool use_softmax = false  // if true, temperature-scaled soft scores then pick best; else argmax
);

}  // namespace tap
