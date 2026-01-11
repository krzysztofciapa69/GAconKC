#include "Optimizer.hpp"
#include "Constants.hpp"
#include "ProblemData.hpp"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

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
  int num_customers = data.GetNumCustomers();
  
  for (int i = 0; i < 6; ++i) {
    evaluators_[i] = new ThreadSafeEvaluator(data, num_groups);

    // Population size based on island type AND instance size
    int pop_size;
    if (num_customers > Config::HUGE_INSTANCE_THRESHOLD) {
      // n > 3000: larger populations for diversity (was 15/10)
      pop_size = (i % 2 == 0) ? 30 : 20;
    } else if (num_customers > Config::LARGE_INSTANCE_THRESHOLD) {
      // n > 1500: reduced populations
      pop_size = (i % 2 == 0) ? Config::EXPLORATION_POP_LARGE 
                              : Config::EXPLOITATION_POP_LARGE;
    } else {
      // default
      pop_size = (i % 2 == 0) ? Config::EXPLORATION_POPULATION_SIZE
                              : Config::EXPLOITATION_POPULATION_SIZE;
    }

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

  // Set up ring predecessors for asynchronous migration
  // Ring topology: 0->1->2->3->4->5->0, so predecessor of i is (i+5)%6
  for (int i = 0; i < 6; ++i) {
    int pred = (i + 5) % 6; // 0<-5, 1<-0, 2<-1, 3<-2, 4<-3, 5<-4
    islands_[i]->SetRingPredecessor(islands_[pred]);
  }

  // Set up exploit_siblings: EXPLOIT islands (1, 3, 5) broadcast best to each other
  // EXPLORE islands broadcast to DEDICATED EXPLOIT partner only (prevents homogenization)
  islands_[1]->SetExploitSiblings({islands_[3], islands_[5]});  // I1 → I3, I5
  islands_[3]->SetExploitSiblings({islands_[1], islands_[5]});  // I3 → I1, I5
  islands_[5]->SetExploitSiblings({islands_[1], islands_[3]});  // I5 → I1, I3
  // DEDICATED PAIRS: Each EXPLORE broadcasts only to its paired EXPLOIT
  islands_[0]->SetExploitSiblings({islands_[1]});  // I0 → I1 only
  islands_[2]->SetExploitSiblings({islands_[3]});  // I2 → I3 only
  islands_[4]->SetExploitSiblings({islands_[5]});  // I4 → I5 only

  // Start 6 worker threads (1 per island)
  for (int i = 0; i < 6; ++i) {
    worker_threads_.emplace_back(&Optimizer::IslandWorkerLoop, this, i);
  }

  cout << "[OPT] Started 6 island workers in RING topology "
          "(0->1->2->3->4->5->0)\n";
  cout << "[OPT] Even (0,2,4) = EXPLORATION | Odd (1,3,5) = EXPLOITATION\n";
  cout << "[OPT] EXPLOIT islands broadcast best to siblings (non-native)\n";
  cout << "[OPT] Specialization: I1=Ejection, I3=PathRelink, I5=DeepSwap\n";
  cout << "[OPT] Migration: ASYNC pull-based (islands pull when stuck)\n";
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

        // Calculate structural difference (Broken Pairs Distance)
        double diff_pct = 100.0;
        if (current_best_indiv_.GetGenotype().size() > 0) {
          const std::vector<int> &g1 = current_best_indiv_.GetGenotype();
          // FIX: Store by value to avoid dangling reference from temporary
          // GetBestIndividual()
          Individual new_best = my_island->GetBestIndividual();
          const std::vector<int> &g2 = new_best.GetGenotype();
          const std::vector<int> &perm =
              evaluator_.GetProblemData().GetPermutation();
          int num_groups = evaluator_.GetNumGroups();
          int size = (int)g1.size();

          if (size > 0 && g1.size() == g2.size()) {
            std::vector<int> p1(size, -2), p2(size, -2);
            std::vector<int> last1(num_groups, -1), last2(num_groups, -1);

            // Debug stats
            // int p1_valid_count = 0;
            // int p2_valid_count = 0;

            for (int customer_id : perm) {
              int idx = customer_id - 2;
              if (idx < 0 || idx >= size)
                continue;

              int gr1 = g1[idx];
              if (gr1 >= 0 && gr1 < num_groups) {
                p1[idx] = last1[gr1];
                last1[gr1] = idx;
                // p1_valid_count++;
              }

              int gr2 = g2[idx];
              if (gr2 >= 0 && gr2 < num_groups) {
                p2[idx] = last2[gr2];
                last2[gr2] = idx;
                // p2_valid_count++;
              }
            }

            int dist = 0;
            for (int k = 0; k < size; ++k) {
              if (p1[k] != p2[k])
                dist++;
            }
            diff_pct = (dist * 100.0) / size;

            // Debug removed
          }
        }

        current_best_ = my_island->GetBestSolution();
        current_best_fitness_ = my_island->GetBestFitness();
        current_best_indiv_ = my_island->GetBestIndividual();

        auto now = std::chrono::steady_clock::now();
        double elapsed =
            std::chrono::duration<double>(now - start_time_).count();

        const char *island_type =
            my_island->IsExploration() ? "EXPLORE" : "EXPLOIT";

        // log truck load percentages
        const ProblemData &pd = evaluator_.GetProblemData();
        const std::vector<int> &genotype = current_best_indiv_.GetGenotype();
        const std::vector<int> &demands = pd.GetDemands();
        int capacity = pd.GetCapacity();
        int num_groups = evaluator_.GetNumGroups();

        std::vector<int> group_load(num_groups, 0);
        for (size_t ci = 0; ci < genotype.size(); ++ci) {
          int group = genotype[ci];
          if (group >= 0 && group < num_groups) {
            int customer_id = static_cast<int>(ci) + 2;
            int demand_idx = customer_id - 1; // demands is 0-indexed, customer_id is 1-based
            if (demand_idx >= 0 && demand_idx < static_cast<int>(demands.size())) {
              group_load[group] += demands[demand_idx];
            }
          }
        }

        // calculate and sort percentages descending
        std::vector<double> pcts(num_groups);
        for (int g = 0; g < num_groups; ++g) {
          pcts[g] = (capacity > 0) ? (100.0 * group_load[g] / capacity) : 0.0;
        }
        std::sort(pcts.begin(), pcts.end(), std::greater<double>());

        cout << "\033[92m [Island " << island_idx << " " << island_type
             << "] NEW BEST: " << fixed << setprecision(2)
             << current_best_fitness_
             << " (ret=" << current_best_indiv_.GetReturnCount()
             << ", diff=" << setprecision(1) << diff_pct << "%)"
             << " @ gen " << local_gen << " (t=" << setprecision(1) << elapsed
             << "s)\033[0m [";
        for (int g = 0; g < num_groups; ++g) {
          if (g > 0) cout << ", ";
          if (pcts[g] > 100.0) {
            cout << "\033[91m" << setprecision(0) << pcts[g] << "%\033[0m";
          } else if (pcts[g] > 95.0) {
            cout << "\033[93m" << setprecision(0) << pcts[g] << "%\033[0m";
          } else {
            cout << setprecision(0) << pcts[g] << "%";
          }
        }
        cout << "]\n";

      }
    }

    // Periodic stats logging
    auto now = std::chrono::steady_clock::now();
    double since_log =
        std::chrono::duration<double>(now - last_log_time).count();
    if (since_log >= Config::LOG_INTERVAL_SECONDS) {
      double gen_per_sec = gens_since_log / since_log;
      const char *island_type =
          my_island->IsExploration() ? "EXPLORE" : "EXPLOIT";
      cout << " [I" << island_idx << " " << island_type << "] "
           << gens_since_log << " gens in " << setprecision(1) << since_log
           << "s = " << setprecision(1) << gen_per_sec
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

  // === GLOBAL HOMOGENIZATION CHECK ===
  // Count unique best solutions across all islands
  std::set<uint64_t> unique_best_hashes;
  for (int i = 0; i < 6; ++i) {
    Individual best = islands_[i]->GetBestIndividual();
    uint64_t h = 1469598103934665603ULL;
    for (int x : best.GetGenotype()) {
      h ^= x;
      h *= 1099511628211ULL;
    }
    unique_best_hashes.insert(h);
  }
  bool global_homogenized = (unique_best_hashes.size() <= 2);

  if (global_homogenized) {
    cout << "\033[93m [RING] GLOBAL HOMOGENIZATION DETECTED ("
         << unique_best_hashes.size()
         << "/6 unique) - Forcing 0% elite migration\033[0m\n";
  }

  for (int i = 0; i < 6; ++i) {
    int next = (i + 1) % 6;

    // Get bests to check for homogenization
    Individual source_best = islands_[i]->GetBestIndividual();
    Individual target_best = islands_[next]->GetBestIndividual();

    // Check if bests are IDENTICAL - if so, we're homogenized, send only
    // diverse!
    bool bests_identical =
        (source_best.GetGenotype() == target_best.GetGenotype());

    // DIVERSITY-FIRST MIGRATION: fixed 2 elite + 3 diverse (max BPD from global
    // best)
    int elite_count = Config::MIGRATION_ELITE_COUNT;     // 2 elite
    int diverse_count = Config::MIGRATION_DIVERSE_COUNT; // 3 diverse

    if (global_homogenized || bests_identical) {
      elite_count = 0; // Full diversity mode when homogenized
      diverse_count = 5;
    }

    // Inject Elites (or sabotage if homogenized)
    for (int m = 0; m < elite_count; ++m) {
      Individual elite = islands_[i]->GetRandomEliteIndividual();
      islands_[next]->InjectImmigrant(elite);
      total_migrants++;
      total_elite++;
    }

    // Inject Diverse (max BPD from target's best)
    for (int m = 0; m < diverse_count; ++m) {
      Individual diverse_ind =
          islands_[i]->GetMostDiverseMigrantFor(target_best);
      islands_[next]->InjectImmigrant(diverse_ind);
      total_migrants++;
    }
  }

  // Diagnostic: log diversity and dynamic migration stats
  double avg_elite_pct =
      (total_migrants > 0) ? (100.0 * total_elite / total_migrants) : 0.0;
  cout << " [RING] Migration: " << total_migrants << " migrants (Avg " << fixed
       << setprecision(1) << avg_elite_pct
       << "% Elite based on target health)\n";
  cout << "        Diversity: ";
  for (int i = 0; i < 6; ++i) {
    cout << "I" << i << "=" << fixed << setprecision(2)
         << islands_[i]->GetCurrentCV() << " ";
  }
  cout << "\n";
}

// DIVERSITY-PULSE MIGRATION: Every 60s, aggressive injection of structurally
// different solutions Only accepts migrants with BPD > 20% relative to target
// island's best
void Optimizer::PerformDiversityPulseMigration() {
  cout << "\033[95m [PULSE] Diversity-Pulse Migration triggered\033[0m\n";

  int total_injected = 0;

  for (int i = 0; i < 6; ++i) {
    int next = (i + 1) % 6;

    // Get target's best for BPD comparison
    Individual target_best = islands_[next]->GetBestIndividual();
    int genotype_size = static_cast<int>(target_best.GetGenotype().size());
    int bpd_threshold = static_cast<int>(
        genotype_size * Config::DIVERSITY_PULSE_BPD_THRESHOLD); // 20%

    // Get 15% of source population
    int pop_size = (i % 2 == 0) ? Config::EXPLORATION_POPULATION_SIZE
                                : Config::EXPLOITATION_POPULATION_SIZE;
    int num_candidates =
        std::max(5, static_cast<int>(pop_size * Config::DIVERSITY_PULSE_RATE));

    // Get diverse candidates from source island
    for (int m = 0; m < num_candidates; ++m) {
      Individual candidate = islands_[i]->GetMostDiverseMigrantFor(target_best);

      // BPD filter: only inject if structurally different enough
      int bpd = islands_[i]->CalculateBrokenPairsDistancePublic(candidate,
                                                                target_best);

      if (bpd >= bpd_threshold) {
        islands_[next]->InjectImmigrant(candidate);
        total_injected++;
      }
    }
  }

  cout << " [PULSE] Injected " << total_injected << " diverse migrants (BPD>"
       << (Config::DIVERSITY_PULSE_BPD_THRESHOLD * 100) << "%)\n";
}

void Optimizer::PrintIslandStats() {
  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - start_time_).count();

  cout << "\n=== Ring Island Statistics (t=" << fixed << setprecision(1)
       << elapsed << "s) ===\n";
  long long total_gens = 0;
  long long totalpr = 0;
  for (int i = 0; i < 6; ++i) {
    long long gens = island_generations_[i];
    total_gens += gens;
    double rate = gens / std::max(0.1, elapsed);
    const char *type = (i % 2 == 0) ? "EXPLORE" : "EXPLOIT";
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
    // ThreadSafeEvaluator* eval = islands_[i]->GetEvaluator(); // REMOVED:
    // Invalid
    route_hits += evaluators_[i]->GetRouteCacheHits();
    route_misses += evaluators_[i]->GetRouteCacheMisses();
  }

  double hit_rate = (total_hits + total_misses > 0)
                        ? (100.0 * total_hits / (total_hits + total_misses))
                        : 0.0;

  double route_hit_rate =
      (route_hits + route_misses > 0)
          ? (100.0 * route_hits / (route_hits + route_misses))
          : 0.0;

  cout << "  Solution Cache (L1): " << total_hits << " hits / " << total_misses
       << " misses (" << setprecision(1) << hit_rate << "% hit rate)\n";
  cout << "  Route Cache (L2):    " << route_hits << " hits / " << route_misses
       << " misses (" << setprecision(1) << route_hit_rate << "% hit rate)\n";

  cout << "  Global Best: " << setprecision(2) << current_best_fitness_ << "\n";
  cout << "  Path Relinking Successes: " << totalpr << "\n";

  // === INTER-ISLAND HOMOGENIZATION CHECK ===
  // Compare best solutions across all islands to detect convergence to same
  // local optimum
  std::vector<uint64_t> best_hashes(6);
  std::vector<double> best_fits(6);
  for (int i = 0; i < 6; ++i) {
    Individual best = islands_[i]->GetBestIndividual();
    best_fits[i] = best.GetFitness();
    uint64_t h = 1469598103934665603ULL;
    for (int x : best.GetGenotype()) {
      h ^= x;
      h *= 1099511628211ULL;
    }
    best_hashes[i] = h;
  }

  // Count unique bests
  std::set<uint64_t> unique_bests(best_hashes.begin(), best_hashes.end());
  int unique_best_count = (int)unique_bests.size();

  // Count similar fitness (within 1000 of each other)
  int similar_fit_count = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = i + 1; j < 6; ++j) {
      if (std::abs(best_fits[i] - best_fits[j]) < 1000)
        similar_fit_count++;
    }
  }

  // Problem diagnosis
  std::string global_issues = "";
  if (unique_best_count <= 2)
    global_issues +=
        "\033[31m[HOMOGENIZED: " + std::to_string(unique_best_count) +
        "/6 unique bests]\033[0m ";
  if (similar_fit_count >= 10)
    global_issues +=
        "\033[33m[CONVERGED: " + std::to_string(similar_fit_count) +
        "/15 pairs similar]\033[0m ";

  double explore_avg = (best_fits[0] + best_fits[2] + best_fits[4]) / 3.0;
  double exploit_avg = (best_fits[1] + best_fits[3] + best_fits[5]) / 3.0;
  if (explore_avg < exploit_avg - 1000)
    global_issues += "\033[35m[EXPLORE_BETTER: exploiters lagging]\033[0m ";

  if (!global_issues.empty()) {
    cout << "  ISSUES: " << global_issues << "\n";
  }

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

  // NOTE: Migration is now ASYNCHRONOUS - each island pulls from predecessor
  // via TryPullMigrant() in RunGeneration() when is_stuck_ is true.
  // PerformRingMigration and PerformDiversityPulseMigration are no longer
  // called.
}

int Optimizer::GetGeneration() {
  long long total = 0;
  for (int i = 0; i < 6; ++i) {
    total += island_generations_[i];
  }
  return static_cast<int>(total / 6); // Average per island
}
