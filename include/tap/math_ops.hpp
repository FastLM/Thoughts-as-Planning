#pragma once

#include <Eigen/Dense>

namespace tap {

using Vector = Eigen::VectorXf;

// tanh-approx GELU (Hendrycks & Gimpel style used in many implementations)
void gelu_forward(const Vector& x, Vector& y);
void gelu_backward(const Vector& x, const Vector& d_y, Vector& d_x);

float softmax_argmax(const Eigen::VectorXf& logits, float temperature, Eigen::VectorXf& prob);

}  // namespace tap
