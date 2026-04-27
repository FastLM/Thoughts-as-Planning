# make (no CMake required)
# If Eigen is missing: git clone --depth 1 -b 3.4.0 \
#   https://gitlab.com/libeigen/eigen.git third_party/eigen3-src
# Or: cmake -S . -B build && cmake --build build
CXX = c++
EIGEN3_INC ?= third_party/eigen3-src
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude -I$(EIGEN3_INC)
CPPS = \
  src/encoder.cpp \
  src/transition.cpp \
  src/reward.cpp \
  src/layers.cpp \
  src/actions.cpp \
  src/planner.cpp \
  src/training.cpp \
  src/math_ops.cpp

.PHONY: all clean
all: bin/tap_run
bin/tap_run: $(CPPS) src/main.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $(CPPS) src/main.cpp
clean:
	rm -f bin/tap_run
