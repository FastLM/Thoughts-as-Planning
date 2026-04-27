#include "tap/math_ops.hpp"

#include <algorithm>
#include <cmath>

namespace tap {

// GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))  ~ 0.5*x*(1+tanh( sqrt(2/pi) * (x+0.044715*x^3) ))
void gelu_forward(const Vector& x, Vector& y) {
  const int n = static_cast<int>(x.size());
  y.resize(n);
  for (int i = 0; i < n; ++i) {
    float t = x[i];
    float c1 = 0.797885f;  // sqrt(2/pi)
    float c2 = 0.044715f;
    float tanh_in = c1 * (t + c2 * t * t * t);
    float th = std::tanh(tanh_in);
    y[i] = 0.5f * t * (1.0f + th);
  }
}

void gelu_backward(const Vector& x, const Vector& d_y, Vector& d_x) {
  const int n = static_cast<int>(x.size());
  d_x.resize(n);
  for (int i = 0; i < n; ++i) {
    float t = x[i];
    float c1 = 0.797885f, c2 = 0.044715f;
    float inner = c1 * (t + c2 * t * t * t);
    float th = std::tanh(inner);
    // derivative of 0.5 t (1+th) w.r.t. t: use finite approximation for the tanh' term
    // dG/dt = 0.5*(1+th) + 0.5*t*sech^2(inner) * c1 * (1 + 3*c2*t*t)
    float d_inner_dt = c1 * (1.0f + 3.0f * c2 * t * t);
    float sech2 = 1.0f - th * th;
    float d_gel = 0.5f * (1.0f + th) + 0.5f * t * sech2 * d_inner_dt;
    d_x[i] = d_gel * d_y[i];
  }
}

float softmax_argmax(const Eigen::VectorXf& logits, float temperature,
                      Eigen::VectorXf& prob) {
  const int n = static_cast<int>(logits.size());
  prob.resize(n);
  float t = std::max(1e-5f, temperature);
  float maxv = logits[0];
  for (int i = 1; i < n; ++i) {
    if (logits[i] > maxv) {
      maxv = logits[i];
    }
  }
  float s = 0.0f;
  for (int i = 0; i < n; ++i) {
    float v = std::exp((logits[i] - maxv) / t);
    prob[i] = v;
    s += v;
  }
  for (int i = 0; i < n; ++i) {
    prob[i] /= s;
  }
  int am = 0;
  float pm = prob[0];
  for (int i = 1; i < n; ++i) {
    if (prob[i] > pm) {
      pm = prob[i];
      am = i;
    }
  }
  return static_cast<float>(am);
}

}  // namespace tap
