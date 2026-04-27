#include "tap/actions.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace tap {

int action_to_id(const Action& a, int n_actions) {
  if (n_actions < 1) {
    return 0;
  }
  std::uint32_t w = 0;
  w ^= static_cast<std::uint32_t>(a.scale) * 0x1f;
  w ^= static_cast<std::uint32_t>(a.tok) * 0x2c;
  w ^= static_cast<std::uint32_t>(a.step) * 0x3a;
  w ^= static_cast<std::uint32_t>(a.str) * 0x4b;
  w ^= (static_cast<std::uint32_t>(a.i0) & 0xffu) << 8u;
  w ^= (static_cast<std::uint32_t>(a.i1) & 0xffu) << 16u;
  w ^= a.payload.size() & 0x7fu;
  return static_cast<int>(w % static_cast<std::uint32_t>(n_actions));
}

// Toy text editors: operate on a single-line or paragraph chain string
static std::vector<std::string> steps_split(const std::string& c) {
  std::vector<std::string> out;
  if (c.empty()) {
    out.push_back("step1: ");
    return out;
  }
  std::stringstream ss(c);
  std::string line;
  while (std::getline(ss, line, '\n')) {
    if (!line.empty()) {
      out.push_back(line);
    }
  }
  if (out.empty()) {
    out.push_back(c);
  }
  return out;
}

static std::string steps_join(const std::vector<std::string>& s) {
  if (s.empty()) {
    return "";
  }
  std::string r = s[0];
  for (std::size_t k = 1; k < s.size(); ++k) {
    r += '\n';
    r += s[k];
  }
  return r;
}

std::string apply_edit(const std::string& c, const Action& a) {
  std::string t = c;
  switch (a.scale) {
    case ActionScale::Token: {
      if (a.tok == TokenOp::NoOp) {
        return t;
      }
      // interpret as word-level in first line
      if (a.tok == TokenOp::Add) {
        if (t.empty()) {
          return a.payload;
        }
        int pos = std::max(0, a.i0 % 64);
        if (pos > static_cast<int>(t.size())) {
          pos = static_cast<int>(t.size());
        }
        t.insert(pos, a.payload);
        return t;
      }
      if (a.tok == TokenOp::Delete) {
        if (t.empty() || t.size() < 1) {
          return t;
        }
        int p = std::max(0, a.i0) % static_cast<int>(t.size() + 1);
        t.erase(t.begin() + p);
        return t;
      }
      if (a.tok == TokenOp::Replace) {
        if (a.payload.empty()) {
          return t;
        }
        if (a.i0 < 0 || t.empty()) {
          t = a.payload;
          return t;
        }
        // replace 1 char at p with payload
        int p = std::abs(a.i0) % static_cast<int>(t.size());
        t.replace(p, 1, a.payload.substr(0, 16));
        return t;
      }
    } break;
    case ActionScale::Step: {
      auto s = steps_split(t);
      int n = static_cast<int>(s.size());
      if (n < 1) {
        s.push_back("s: (empty)");
        n = 1;
      }
      if (a.step == StepOp::Reorder) {
        int i = a.i0 % n;
        int j = a.i1 % n;
        if (i < 0) {
          i += n;
        }
        if (j < 0) {
          j += n;
        }
        std::swap(s[static_cast<std::size_t>(i)], s[static_cast<std::size_t>(j)]);
        return steps_join(s);
      }
      if (a.step == StepOp::Split) {
        int p = a.i0 % n;
        if (p < 0) {
          p += n;
        }
        if (p >= 0 && p < n) {
          std::string& line = s[static_cast<std::size_t>(p)];
          if (!line.empty()) {
            int m = a.i1 % static_cast<int>(line.size());
            if (m < 0) {
              m += static_cast<int>(line.size());
            }
            s.insert(s.begin() + p + 1, line.substr(static_cast<std::size_t>(m)));
            line = line.substr(0, static_cast<std::size_t>(m));
          }
        }
        return steps_join(s);
      }
      if (a.step == StepOp::Merge) {
        if (n >= 2) {
          int p = a.i0 % (n - 1);
          if (p < 0) {
            p += n - 1;
          }
          s[static_cast<std::size_t>(p)] += " | " + s[static_cast<std::size_t>(p) + 1u];
          s.erase(s.begin() + p + 1);
        }
        return steps_join(s);
      }
    } break;
    case ActionScale::Structure: {
      std::string head = t;
      switch (a.str) {
        case StructOp::AddExample: {
          if (!a.payload.empty()) {
            head = std::string("ex: ") + a.payload + "\n" + head;
          } else {
            head = "ex: e.g. \n" + head;
          }
        } break;
        case StructOp::InstrEdit: {
          if (!a.payload.empty()) {
            head = "instr: " + a.payload + " || " + head;
          } else {
            head = "instr: clarify. || " + head;
          }
        } break;
        case StructOp::FmtChange: {
          if (!a.payload.empty()) {
            head = "{fmt:" + a.payload + "}" + head;
          } else {
            head = "{step}\n" + head;
          }
        } break;
        case StructOp::NoOp:
        default: break;
      }
      return head;
    } break;
  }
  return t;
}

void sample_random_actions(int K, int n_actions, std::uint64_t& seed, std::vector<Action>& out) {
  out.clear();
  out.resize(static_cast<std::size_t>(K));
  for (int k = 0; k < K; ++k) {
    std::uint64_t s = seed;
    s ^= s << 7;
    s ^= s >> 9;
    s *= 0x9E3779B97F4A7C15ull;
    seed = s;
    int r0 = static_cast<int>(s & 3u);
    auto& a = out[static_cast<std::size_t>(k)];
    a.scale = static_cast<ActionScale>(r0);
    a.tok = static_cast<TokenOp>((s >> 3) & 3u);
    a.step = static_cast<StepOp>((s >> 5) & 3u);
    a.str = static_cast<StructOp>((s >> 7) & 3u);
    a.i0 = static_cast<int>((s >> 9) & 63u);
    a.i1 = static_cast<int>((s >> 15) & 63u);
    a.payload = "p";
    if ((s & 1u) != 0u) {
      a.payload = "x";
    }
    a.id = static_cast<int>((s & 0xffffu) % static_cast<std::uint32_t>(std::max(1, n_actions)));
  }
}

}  // namespace tap