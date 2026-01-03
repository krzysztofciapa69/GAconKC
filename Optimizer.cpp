#include "Optimizer.hpp"
#include "Constants.hpp"
#include "ProblemData.hpp"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace LcVRPContest;
using namespace std;

Optimizer::Optimizer(Evaluator &evaluator)
    : evaluator_(evaluator), rng_(random_device{}()),
      current_best_fitness_(numeric_limits<double>::max()), is_running_(false) {
  const ProblemData &data = evaluator_.GetProblemData();
  int num_groups = evaluator.GetNumGroups();

  // Initialize generation counters
  for (int i = 0; i < 6; ++i) {
    island_generations_[i] = 0;
  }

  // Create evaluators and islands (6 total in ring)
  for (int i = 0; i < 6; ++i) {
    evaluators_[i] = new ThreadSafeEvaluator(data, num_groups);
    
    // Population size based on island type
    int pop_size = (i % 2 == 0) ? Config::EXPLORATION_POPULATION_SIZE 
                                : Config::EXPLOITATION_POPULATION_SIZE;
    
    islands_[i] = new Island(evaluators_[i], data, pop_size, i);
  }
}

Optimizer::~Optimizer() {
  StopThreads();

  // Clean up islands and evaluators
  for (int i = 0; i < 6; ++i) {
    delete islands_[i];
    delete evaluators_[i];
  }
}

void Optimizer::StopThreads() {
  is_running_ = false;

  for (auto &t : worker_threads_) {
    if (t.joinable())
      t.join();
  }
  worker_threads_.clear();
}


void Optimizer::Initialize() {
  StopThreads();

  // Initialize all 6 islands
  for (int i = 0; i < 6; ++i) {
    // Alternate initialization strategies
    INITIALIZATION_TYPE strategy = (i % 2 == 0) ? INITIALIZATION_TYPE::RANDOM 
                                                : INITIALIZATION_TYPE::CHUNKED;
    islands_[i]->Initialize(strategy);

    if (islands_[i]->GetBestFitness() < current_best_fitness_) {
      current_best_ = islands_[i]->GetBestSolution();
      current_best_fitness_ = islands_[i]->GetBestFitness();
      current_best_indiv_ = islands_[i]->GetBestIndividual();
    }
  }

  is_running_ = true;
  start_time_ = std::chrono::steady_clock::now();

  // Set start time on all islands for Endgame Mode detection
  for (int i = 0; i < 6; ++i) {
    islands_[i]->SetStartTime(start_time_);
  }

  // Start 6 worker threads (1 per island)
  for (int i = 0; i < 6; ++i) {
    worker_threads_.emplace_back(&Optimizer::IslandWorkerLoop, this, i);
  }

  cout << "[OPT] Started 6 island workers in RING topology (0->1->2->3->4->5->0)\n";
  cout << "[OPT] Even (0,2,4) = EXPLORATION | Odd (1,3,5) = EXPLOITATION\n";
}

void Optimizer::IslandWorkerLoop(int island_idx) {
  Island *my_island = islands_[island_idx];

  long long local_gen = 0;
  auto last_log_time = std::chrono::steady_clock::now();
  long long gens_since_log = 0;

  while (is_running_) {
    my_island->RunGeneration();
    local_gen++;
    gens_since_log++;
    island_generations_[island_idx]++;

    // Check if island found a new global best
    if (my_island->GetBestFitness() < current_best_fitness_) {
      std::lock_guard<std::mutex> lock(global_mutex_);
      if (my_island->GetBestFitness() < current_best_fitness_) {
        current_best_ = my_island->GetBestSolution();
        current_best_fitness_ = my_island->GetBestFitness();
        current_best_indiv_ = my_island->GetBestIndividual();

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time_).count();

        const char* island_type = my_island->IsExploration() ? "EXPLORE" : "EXPLOIT";
        cout << "\033[92m [Island " << island_idx << " " << island_type
             << "] NEW BEST: " << fixed << setprecision(2)
             << current_best_fitness_
             << " (returns=" << current_best_indiv_.GetReturnCount() << ")"
             << " @ gen " << local_gen << " (t=" << setprecision(1) << elapsed
             << "s)\033[0m\n";


      }
    }

    // Periodic stats logging
    auto now = std::chrono::steady_clock::now();
    double since_log = std::chrono::duration<double>(now - last_log_time).count();
    if (since_log >= Config::LOG_INTERVAL_SECONDS) {
      double gen_per_sec = gens_since_log / since_log;
      const char* island_type = my_island->IsExploration() ? "EXPLORE" : "EXPLOIT";
      cout << " [I" << island_idx << " " << island_type
           << "] " << gens_since_log << " gens in " << setprecision(1)
           << since_log << "s = " << setprecision(1) << gen_per_sec
           << " gen/s, best=" << setprecision(2) << my_island->GetBestFitness()
           << "\n";



      last_log_time = now;
      gens_since_log = 0;
    }
  }
}

void Optimizer::PerformRingMigration() {
  // Ring migration: 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 0
  // Strategy: Dynamic ratio based on target island health (CV)
  int total_migrants = 0;
  int total_elite = 0;
  
  for (int i = 0; i < 6; ++i) {
    // Calculate number of migrants based on source island population (10%)
    int pop_size = (i % 2 == 0) ? Config::EXPLORATION_POPULATION_SIZE 
                                : Config::EXPLOITATION_POPULATION_SIZE;
    int num_migrants = std::max(2, (int)(pop_size * Config::RING_MIGRATION_RATE));
    
    int next = (i + 1) % 6;
    double target_cv = islands_[next]->GetCurrentCV();
    
    // Dynamic Ratio Logic:
    // - Low Diversity (< 0.1): Target is Stagnant -> Send mostly DIVERSE (20% Elite, 80% Diverse)
    // - High Diversity (> 0.4): Target is Chaotic -> Send mostly ELITE (70% Elite, 30% Diverse)
    // - Balanced: 50/50
    double elite_ratio = 0.5;
    if (target_cv < 0.1) {
        elite_ratio = 0.2; 
    } else if (target_cv > 0.4) {
        elite_ratio = 0.7;
    }
    
    int best_count = (int)(num_migrants * elite_ratio);
    best_count = std::max(1, best_count); // Always send at least one elite
    int diverse_count = num_migrants - best_count;
    
    // Get target's best for diversity calculation
    Individual target_best = islands_[next]->GetBestIndividual();
    Individual source_best = islands_[i]->GetBestIndividual();
    
    // Inject Elites
    for (int m = 0; m < best_count; ++m) {
      islands_[next]->InjectImmigrant(source_best);
      total_migrants++;
      total_elite++;
    }
    
    // Inject Diverse
    for (int m = 0; m < diverse_count; ++m) {
      Individual diverse_ind = islands_[i]->GetMostDiverseMigrantFor(target_best);
      islands_[next]->InjectImmigrant(diverse_ind);
      total_migrants++;
    }
  }
  
  // Diagnostic: log diversity and dynamic migration stats
  double avg_elite_pct = (total_migrants > 0) ? (100.0 * total_elite / total_migrants) : 0.0;
  cout << " [RING] Migration: " << total_migrants << " migrants (Avg " 
       << fixed << setprecision(1) << avg_elite_pct << "% Elite based on target health)\n";
  cout << "        Diversity: ";
  for (int i = 0; i < 6; ++i) {
    cout << "I" << i << "=" << fixed << setprecision(2) << islands_[i]->GetCurrentCV() << " ";
  }
  cout << "\n";
}



void Optimizer::PrintIslandStats() {
  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - start_time_).count();

  cout << "\n=== Ring Island Statistics (t=" << fixed << setprecision(1) << elapsed
       << "s) ===\n";
  long long total_gens = 0;
  long long totalpr = 0;
  for (int i = 0; i < 6; ++i) {
    long long gens = island_generations_[i];
    total_gens += gens;
    double rate = gens / std::max(0.1, elapsed);
    const char* type = (i % 2 == 0) ? "EXPLORE" : "EXPLOIT";
    cout << "  I" << i << " (" << type << "): " << gens << " gens ("
         << setprecision(1) << rate << " gen/s)\n";
  }
  cout << "  TOTAL: " << total_gens << " gens (" << setprecision(1)
       << (total_gens / std::max(0.1, elapsed)) << " gen/s)\n";

  // Cache stats
  long long total_hits = 0, total_misses = 0;
  long long route_hits = 0, route_misses = 0;
  for (int i = 0; i < 6; ++i) {
	totalpr += islands_[i]->getPRStats();
    total_hits += islands_[i]->GetCacheHits();
    total_misses += islands_[i]->GetCacheMisses();
    
    // Access Evaluator stats
    // ThreadSafeEvaluator* eval = islands_[i]->GetEvaluator(); // REMOVED: Invalid
    route_hits += evaluators_[i]->GetRouteCacheHits();
    route_misses += evaluators_[i]->GetRouteCacheMisses();
  }
  
  double hit_rate = (total_hits + total_misses > 0) 
                    ? (100.0 * total_hits / (total_hits + total_misses)) : 0.0;
  
  double route_hit_rate = (route_hits + route_misses > 0)
                          ? (100.0 * route_hits / (route_hits + route_misses)) : 0.0;

  cout << "  Solution Cache (L1): " << total_hits << " hits / " << total_misses
       << " misses (" << setprecision(1) << hit_rate << "% hit rate)\n";
  cout << "  Route Cache (L2):    " << route_hits << " hits / " << route_misses
       << " misses (" << setprecision(1) << route_hit_rate << "% hit rate)\n";
       
  cout << "  Global Best: " << setprecision(2) << current_best_fitness_ << "\n";
  cout << "  Path Relinking Successes: " << totalpr << "\n";
  cout << "===============================\n\n";
}

void Optimizer::RunIteration() {
  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - start_time_).count();

  // === GLOBAL STATS LOGGING ===
  static double last_global_log = 0.0;
  if (elapsed - last_global_log >= Config::LOG_INTERVAL_SECONDS) {
    PrintIslandStats();
    last_global_log = elapsed;
  }

  // === RING MIGRATION (time-based) ===
  static double last_ring_migration = 0.0;
  if (elapsed - last_ring_migration >= Config::RING_MIGRATION_INTERVAL_SECONDS) {
    PerformRingMigration();
    last_ring_migration = elapsed;
  }
}

int Optimizer::GetGeneration() {
  long long total = 0;
  for (int i = 0; i < 6; ++i) {
    total += island_generations_[i];
  }
  return static_cast<int>(total / 6); // Average per island
}
