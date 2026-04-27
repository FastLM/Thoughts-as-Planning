#pragma once

#include "tap/types.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace tap {

// Map structured action to discrete id in [0, n_actions) for transition net
int action_to_id(const Action& a, int n_actions);

// Apply edit a to reasoning chain c (toy string operators for reference)
// Returns updated chain (may be unchanged on invalid)
std::string apply_edit(const std::string& c, const Action& a);

// Sample K random valid actions (multi-scale mixture)
void sample_random_actions(int K, int n_actions, std::uint64_t& seed,
                          std::vector<Action>& out);

}  // namespace tap
