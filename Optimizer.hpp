#pragma once
#include "Island.hpp"
#include "Evaluator.hpp"
#include "LocalSearch.hpp"
#include "ProblemGeometry.hpp"
#include "RoutePool.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace LcVRPContest {
class Optimizer {
public:
  Optimizer(Evaluator &evaluator);
  ~Optimizer();

  void Initialize();
  void RunIteration();
  int GetGeneration();
  Individual GetBestIndividual() { return current_best_indiv_; }
  std::vector<int> *GetCurrentBest() { return &current_best_; }
  double GetCurrentBestFitness() const { return current_best_fitness_; }
  void PrintIslandStats();

private:
  void IslandWorkerLoop(int island_idx);
  void PerformRingMigration();
  void PerformDiversityPulseMigration();  // aggressive 60s migration with BPD filter
  void StopThreads();

  Evaluator &evaluator_;

  // 6 islands in a ring: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
  // Even (0, 2, 4) = Exploration | Odd (1, 3, 5) = Exploitation
  std::array<ThreadSafeEvaluator *, 6> evaluators_;
  std::array<Island *, 6> islands_;

  // 6 worker threads (1 per island)
  std::vector<std::thread> worker_threads_;
  std::atomic<bool> is_running_;

  // Per-island generation counters
  std::array<std::atomic<long long>, 6> island_generations_;
  std::chrono::steady_clock::time_point start_time_;

  std::vector<int> current_best_;
  double current_best_fitness_;
  Individual current_best_indiv_;
  std::mt19937 rng_;
  std::mutex global_mutex_;
};
} // namespace LcVRPContest

