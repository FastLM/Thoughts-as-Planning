# Thoughts as Planning

C++ reference for our paper **Thoughts-as-Planning** — a latent state encoder, transition model, reward head, and a planning loop over text edits (toy `src/main.cpp`).

## Requirements

- C++17 compiler  
- [Eigen3](https://eigen.tuxfamily.org/)

## Build

**Make**. Put Eigen under `third_party/eigen3-src` (see command below) or set `EIGEN3_INC` to your Eigen include root.

```bash
git clone --depth 1 -b 3.4.0 \
  https://gitlab.com/libeigen/eigen.git third_party/eigen3-src
make
```

Binary: `bin/tap_run`.

**CMake** (fetches Eigen):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Run

```bash
./bin/tap_run        # if built with make
# or
./build/tap_run      # if built with cmake (location may vary)
```

## Layout

- `include/tap/` — headers (encoder, transition, reward, planner, training)  
- `src/` — implementations and demo entrypoint

## Citation

D. Liu, Y. Yu, and Y. N. Wu, *Thoughts-as-Planning: Latent World Models for Chain-of-Thoughts Optimization via Reinforcement Planning*, 2026.

```bibtex
@article{liu2026thoughtsasplanning,
  title  = {Thoughts-as-Planning: Latent World Models for Chain-of-Thoughts Optimization via Reinforcement Planning},
  author = {Liu, Dong and Yu, Yanxuan and Wu, Ying Nian}
}
```
