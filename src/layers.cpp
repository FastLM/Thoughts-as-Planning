#include "tap/layers.hpp"
#include "tap/math_ops.hpp"

#include <cmath>
#include <cstdint>

namespace {

float urand01(std::uint64_t& s) {
  s ^= s >> 12;
  s ^= s << 25;
  s ^= s >> 27;
  return static_cast<float>(s * 2685821657736338717ULL) / 18446744073709551616.0f;
}

float xavier_range(int in_, int out_) {
  return std::sqrt(2.0f / (static_cast<float>(in_ + out_)));
}

}  // namespace

namespace tap {

void Linear::init(int in_, int out_, std::uint64_t& rng) {
  in_dim = in_;
  out_dim = out_;
  W.resize(out_dim, in_dim);
  b.resize(out_dim);
  float s = xavier_range(in_, out_dim);
  for (int r = 0; r < out_dim; ++r) {
    for (int c = 0; c < in_dim; ++c) {
      W(r, c) = (2.0f * urand01(rng) - 1.0f) * s;
    }
    b[r] = 0.0f;
  }
}

void Linear::forward(const Vector& x, Vector& y) const { y = W * x + b; }

void Linear::forward_batch(const Matrix& x_batch, Matrix& y_batch) const {
  y_batch = W * x_batch;
  y_batch.colwise() += b;
}

void ResidualFfn::init(int d_model_, int d_ff_, std::uint64_t& rng) {
  d_model = d_model_;
  d_ff = d_ff_;
  w1.init(d_model_, d_ff_, rng);
  w2.init(d_ff_, d_model_, rng);
}

void ResidualFfn::forward(const Vector& x, Vector& t1, Vector& g, Vector& y) const {
  w1.forward(x, t1);
  gelu_forward(t1, g);
  Vector tmp;
  w2.forward(g, tmp);
  y = x + tmp;
}

}  // namespace tap
