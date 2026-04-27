#include "tap/reward.hpp"
#include "tap/math_ops.hpp"

namespace tap {

void RewardModel::init(const HyperParams& p, std::uint64_t seed) {
  rng = seed;
  d_latent = p.d_latent;
  d_hidden = 256;
  lin1.init(d_latent, d_hidden, rng);
  lin2.init(d_hidden, 1, rng);
}

float RewardModel::forward(const Vector& z, Vector& h, Vector& pre2) const {
  Vector h0;
  lin1.forward(z, h0);
  Vector g;
  gelu_forward(h0, g);
  h = g;  // hidden after GELU, before the second linear
  lin2.forward(h, pre2);
  if (pre2.size() < 1) {
    return 0.0f;
  }
  return pre2(0);
}

int RewardModel::num_params() const {
  return d_latent * d_hidden + d_hidden + d_hidden * 1 + 1;
}

}  // namespace tap
