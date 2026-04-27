#pragma once

#include <Eigen/Dense>
#include <vector>

namespace tap {

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

// y = W * x + b, W: (out, in)
struct Linear {
  int in_dim{};
  int out_dim{};
  Matrix W;
  Vector b;
  void init(int in_, int out_, std::uint64_t& rng);
  void forward(const Vector& x, Vector& y) const;
  void forward_batch(const Matrix& x_batch, Matrix& y_batch) const;
};

// FFN sub-block: y = x + W2 * GELU( W1 * x + b1 ) + b2  (no LayerNorm, for clear backprop in C++)
struct ResidualFfn {
  int d_model{};
  int d_ff{};
  Linear w1, w2;
  void init(int d_model_, int d_ff_, std::uint64_t& rng);
  // t1 = W1*x, g = GELU(t1), y = x + W2*g
  void forward(const Vector& x, Vector& t1, Vector& g, Vector& y) const;
};

}  // namespace tap
