#pragma once

// Defaults for planning, training, and model width (tuned to match the published setup)
namespace tap {

struct HyperParams {
  // Shared by encoder, transition, and reward heads
  int d_latent = 512;
  int d_ff = 2048;          // FFN width ~ 4x d
  int token_embed_dim = 128;
  int vocab_size = 8192;
  int num_encoder_blocks = 4;  // Encoder depth; the reference uses a pooled-token + FFN
                                 // stack in place of a full transformer
  // Planning
  int planning_horizon = 3;  // H
  int plan_steps = 8;        // number of edit iterations
  int num_candidates = 10;   // K
  float softmax_temp = 0.1f;
  // Training
  float lr = 1e-4f;
  int batch_size = 32;        // (demo may use a smaller value)
  float lambda_dyn = 1.0f;
  float lambda_rew = 1.0f;
  float gamma = 0.99f;        // MDP discount
};

}  // namespace tap
