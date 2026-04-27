#include "tap/training.hpp"

#include "tap/actions.hpp"
#include "tap/math_ops.hpp"

#include <vector>

namespace tap {

namespace {

void linear_sgd(Linear& L, const Vector& x, const Vector& d_y, float lr) {
  L.W -= lr * d_y * x.transpose();
  L.b -= lr * d_y;
}

float transition_train_step(TransitionModel& t, int aid, const Vector& z, const Vector& z1, float lr) {
  if (t.n_actions < 1) {
    return 0.0f;
  }
  if (aid < 0) {
    aid = 0;
  }
  if (aid >= t.n_actions) {
    aid = t.n_actions - 1;
  }
  Vector h(t.d_latent + t.d_action_embed);
  h << z, t.action_embed.row(aid).transpose();
  Vector t1, g, delta;
  t.lin1.forward(h, t1);
  gelu_forward(t1, g);
  t.lin2.forward(g, delta);
  const Vector z_pred = z + delta;
  const Vector err = z_pred - z1;
  const float loss = 0.5f * err.squaredNorm();
  const Vector d_g = t.lin2.W.transpose() * err;
  t.lin2.W -= lr * (err * g.transpose());
  t.lin2.b -= lr * err;
  Vector d_t1;
  gelu_backward(t1, d_g, d_t1);
  linear_sgd(t.lin1, h, d_t1, lr);
  const Vector d_h = t.lin1.W.transpose() * d_t1;
  for (int i = 0; i < t.d_action_embed; ++i) {
    t.action_embed(aid, i) -= lr * d_h(t.d_latent + i);
  }
  return loss;
}

void reward_train_step(RewardModel& r, const Vector& z, float target, float lr) {
  Vector t1, g, r1;
  r.lin1.forward(z, t1);
  gelu_forward(t1, g);
  r.lin2.forward(g, r1);
  if (r1.size() < 1) {
    return;
  }
  const float e = r1(0) - target;
  Vector d_r(1);
  d_r(0) = e;
  const Vector d_g = r.lin2.W.transpose() * d_r;
  r.lin2.W -= lr * (d_r * g.transpose());
  r.lin2.b -= lr * d_r;
  Vector d_t1;
  gelu_backward(t1, d_g, d_t1);
  linear_sgd(r.lin1, z, d_t1, lr);
}

}  // namespace

void WorldModelSystem::init(const HyperParams& p, int n_action_types, std::uint64_t seed) {
  hp = p;
  n_actions = n_action_types;
  enc.init(p, seed);
  trans.init(p, n_action_types, seed + 0x3c6ef372ull);
  rew.init(p, seed + 0xbeefull);
}

void WorldModelSystem::save_latent(const State& s, Eigen::VectorXf& z) const { enc.forward(s, z, nullptr); }

float WorldModelSystem::predict_reward(const State& s) const {
  Vector z;
  enc.forward(s, z, nullptr);
  Vector h, p2;
  return rew.forward(z, h, p2);
}

std::pair<float, float> WorldModelSystem::train_on_batch(
    const std::vector<Transition>& batch, float lr, int) {
  float s_dyn = 0.0f, s_rew = 0.0f;
  if (batch.empty()) {
    return {0.0f, 0.0f};
  }
  const int B = static_cast<int>(batch.size());
  const float step = lr;
  for (int bi = 0; bi < B; ++bi) {
    const auto& tr = batch[static_cast<std::size_t>(bi)];
    const int aid = action_to_id(tr.a, n_actions);
    if (n_actions < 1 || (trans.n_actions > 0 && aid >= trans.n_actions)) {
      continue;
    }
    State s0 = tr.s0;
    State s1 = tr.s1;
    s1.task_input = s0.task_input;
    Vector z0, z1;
    enc.forward(s0, z0, nullptr);
    enc.forward(s1, z1, nullptr);
    s_dyn += transition_train_step(trans, aid, z0, z1, step);
    Vector h2, p2;
    const float rp = rew.forward(z0, h2, p2);
    s_rew += 0.5f * (rp - tr.reward) * (rp - tr.reward);
    reward_train_step(rew, z0, tr.reward, step);
  }
  return {s_dyn / B, s_rew / B};
}

}  // namespace tap
