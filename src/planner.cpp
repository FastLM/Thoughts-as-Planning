#include "tap/planner.hpp"

#include "tap/actions.hpp"
#include "tap/layers.hpp"
#include "tap/math_ops.hpp"

#include <vector>

namespace tap {

PlanResult run_planning(
    const State& s0,
    const StateEncoder& enc,
    const TransitionModel& trans,
    const RewardModel& rew,
    const HyperParams& hp,
    int n_actions,
    std::uint64_t& seed,
    bool use_softmax) {
  PlanResult out;
  out.trace_chains.push_back(s0.chain);
  State st;
  st.task_input = s0.task_input;
  st.chain = s0.chain;
  Vector z;
  enc.forward(st, z, nullptr);
  const int K = hp.num_candidates;
  for (int t = 0; t < hp.plan_steps; ++t) {
    std::vector<Action> cands;
    sample_random_actions(K, n_actions, seed, cands);
    std::vector<float> scores;
    scores.resize(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
      Action& ak = cands[static_cast<std::size_t>(k)];
      if (n_actions > 0) {
        ak.id = action_to_id(ak, n_actions);
      }
      Vector zH;
      trans.roll_same_action(z, ak.id, hp.planning_horizon, zH);
      Vector h, p2;
      const float rhat = rew.forward(zH, h, p2);
      scores[static_cast<std::size_t>(k)] = rhat;
    }
    int best = 0;
    if (use_softmax) {
      Vector logits(K);
      for (int k = 0; k < K; ++k) {
        logits(k) = scores[static_cast<std::size_t>(k)];
      }
      Vector prob;
      const float pick = softmax_argmax(logits, hp.softmax_temp, prob);
      best = static_cast<int>(pick);
    } else {
      for (int k = 1; k < K; ++k) {
        if (scores[static_cast<std::size_t>(k)] > scores[static_cast<std::size_t>(best)]) {
          best = k;
        }
      }
    }
    const Action& a_star = cands[static_cast<std::size_t>(best)];
    st.chain = apply_edit(st.chain, a_star);
    {
      Vector zH;
      trans.roll_same_action(z, a_star.id, hp.planning_horizon, zH);
      Vector h, p2;
      out.pred_rewards.push_back(static_cast<double>(rew.forward(zH, h, p2)));
    }
    out.selected_action_ids.push_back(a_star.id);
    out.trace_chains.push_back(st.chain);
    Vector z_next;
    trans.forward(z, a_star.id, z_next);
    z = z_next;
  }
  out.final_chain = st.chain;
  return out;
}

}  // namespace tap
