#include "tap/transition.hpp"
#include "tap/math_ops.hpp"

namespace tap {

void TransitionModel::init(const HyperParams& p, int n_action_types, std::uint64_t seed) {
  rng = seed;
  d_latent = p.d_latent;
  d_action_embed = 32;
  d_hidden = 1024;  // transition MLP width (per reference)
  n_actions = n_action_types;
  action_embed.setZero(n_actions, d_action_embed);
  for (int r = 0; r < n_actions; ++r) {
    for (int c = 0; c < d_action_embed; ++c) {
      rng = rng * 1664525ull + 1013904223ull;
      action_embed(r, c) = (static_cast<float>(rng & 0xFFFF) / 65535.0f) * 0.04f - 0.02f;
    }
  }
  int din = d_latent + d_action_embed;
  lin1.init(din, d_hidden, rng);
  lin2.init(d_hidden, d_latent, rng);
}

void TransitionModel::forward(const Vector& z, int action_id, Vector& z_next) const {
  if (action_id < 0) {
    action_id = 0;
  }
  if (action_id >= n_actions) {
    action_id = n_actions - 1;
  }
  Vector h(d_latent + d_action_embed);
  h << z, action_embed.row(action_id).transpose();
  Vector t1, g, delta;
  lin1.forward(h, t1);
  gelu_forward(t1, g);
  lin2.forward(g, delta);
  z_next = z + delta;
}

void TransitionModel::roll_same_action(const Vector& z0, int action_id, int H, Vector& zH) const {
  zH = z0;
  for (int t = 0; t < H; ++t) {
    forward(zH, action_id, zH);
  }
}

int TransitionModel::num_params() const {
  return static_cast<int>(n_actions) * d_action_embed + lin1.in_dim * lin1.out_dim + lin1.out_dim +
         lin2.in_dim * lin2.out_dim + lin2.out_dim;
}

}  // namespace tap
