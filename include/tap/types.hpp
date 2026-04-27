#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tap {

// Multi-scale edits in the paper: token / step / structure
enum class ActionScale : std::uint8_t { Token = 0, Step = 1, Structure = 2 };

// Discrete op labels for a toy string editor
enum class TokenOp : std::uint8_t { Add, Delete, Replace, NoOp };
enum class StepOp : std::uint8_t { Reorder, Split, Merge, NoOp };
enum class StructOp : std::uint8_t { AddExample, InstrEdit, FmtChange, NoOp };

// (scale, type, target) with payload string when needed
struct Action {
  ActionScale scale{ActionScale::Token};
  TokenOp tok{TokenOp::NoOp};
  StepOp step{StepOp::NoOp};
  StructOp str{StructOp::NoOp};
  // Targets and optional payload for structural edits
  int i0{0};
  int i1{0};
  std::string payload;  // token to add/replace, or snippet for structure ops
  // Integer id in [0, n_actions) for MLP transition training
  int id{0};
};

// State = task x plus current reasoning chain c
struct State {
  std::string task_input;   // x
  std::string chain;        // c (reasoning chain as text)
};

struct Transition {
  State s0;
  Action a;
  State s1;  // next state
  float reward{};  // R(x, c) at s0 or s1 per training script convention
};

}  // namespace tap
