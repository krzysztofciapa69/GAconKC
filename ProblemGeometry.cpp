#include "ProblemGeometry.hpp"
#include <algorithm>
#include <chrono>

using namespace LcVRPContest;

ProblemGeometry::ProblemGeometry(const ProblemData &data, int id) : id_(id) {
  rng_.seed(static_cast<unsigned int>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count() +
      id * 999));
}

void ProblemGeometry::Initialize(ThreadSafeEvaluator *evaluator) {
  PrecomputeNeighbors(evaluator);
}

void ProblemGeometry::PrecomputeNeighbors(ThreadSafeEvaluator *evaluator) {
  int n = evaluator->GetSolutionSize();
  neighbors_.assign(n, std::vector<int>());

  for (int i = 0; i < n; ++i) {
    int u_id = i + 2;
    if (u_id > evaluator->GetDimension())
      continue;

    int u_node_idx = u_id - 1;

    std::vector<std::pair<double, int>> dists;
    dists.reserve(n);

    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;

      int v_id = j + 2;
      int v_node_idx = v_id - 1;

      double d = evaluator->GetDist(u_node_idx, v_node_idx);
      dists.push_back({d, j});
    }

    int neighbor_limit = Config::NUM_NEIGHBORS;
    if (n > Config::HUGE_INSTANCE_THRESHOLD) {
      neighbor_limit = 12; // Reduce for huge instances to speed up VND
    }

    size_t keep = std::min((size_t)neighbor_limit, dists.size());

    std::nth_element(dists.begin(), dists.begin() + keep, dists.end());

    neighbors_[i].reserve(keep);
    for (size_t k = 0; k < keep; ++k) {
      neighbors_[i].push_back(dists[k].second);
    }
  }
}
