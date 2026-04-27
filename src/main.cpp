// Minimal demo: train the encoder/transition/reward stack on toy data, then run the planning
// loop that the paper uses for test-time chain-of-thought search.
#include "tap/config.hpp"
#include "tap/planner.hpp"
#include "tap/training.hpp"
#include "tap/types.hpp"
#include "tap/actions.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
  tap::HyperParams hp;
  hp.num_encoder_blocks = 2;
  hp.plan_steps = 3;
  hp.d_latent = 64;
  hp.d_ff = 256;
  hp.token_embed_dim = 32;
  hp.vocab_size = 256;
  hp.num_candidates = 4;
  hp.planning_horizon = 2;

  const int n_actions = 64;
  tap::WorldModelSystem sys;
  sys.init(hp, n_actions, 0xC0FFEEull);

  std::vector<tap::Transition> batch;
  for (int i = 0; i < 16; ++i) {
    tap::Transition t;
    t.s0.task_input = "GSM8K#";
    t.s0.task_input += std::to_string(i % 4);
    t.s0.chain = "Let us decompose the problem into steps\nstep 1: x";
    t.s1 = t.s0;
    t.a.scale = (i % 2) ? tap::ActionScale::Token : tap::ActionScale::Step;
    t.a.tok = tap::TokenOp::Add;
    t.a.step = tap::StepOp::Merge;
    t.a.i0 = 0;
    t.a.i1 = 0;
    t.a.payload = "h";
    t.s1.chain = tap::apply_edit(t.s0.chain, t.a);
    t.a.id = tap::action_to_id(t.a, n_actions);
    t.reward = 0.3f + 0.01f * (static_cast<float>(i) / 8.0f);
    batch.push_back(t);
  }
  for (int e = 0; e < 5; ++e) {
    auto p = sys.train_on_batch(batch, 5e-4f, e);
    if (e == 0 || e == 4) {
      std::cout << "epoch " << e
                << " L_dyn=" << p.first << " L_rew=" << p.second << "\n";
    }
  }

  tap::State s0;
  s0.task_input = "GSM8K#0";
  s0.chain = "step 0: givens are listed";
  std::uint64_t plan_seed = 0x5EED5EEDull;
  const tap::PlanResult r =
      tap::run_planning(s0, sys.enc, sys.trans, sys.rew, hp, n_actions, plan_seed, false);
  std::cout << "final chain:\n" << r.final_chain << "\n";
  std::cout << "edits: " << r.selected_action_ids.size() << " trace lines: " << r.trace_chains.size()
            << "\n";
  return 0;
}
