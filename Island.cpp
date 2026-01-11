#define _USE_MATH_DEFINES
#include "Island.hpp"
#include "Constants.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

using namespace LcVRPContest;
using namespace std;

static inline uint64_t HashGenotype64(const std::vector<int> &g) {
  uint64_t h = 1469598103934665603ULL;
  for (int x : g) {
    uint32_t v = static_cast<uint32_t>(x);
    h ^= (v & 0xFFu);
    h *= 1099511628211ULL;
    h ^= ((v >> 8) & 0xFFu);
    h *= 1099511628211ULL;
    h ^= ((v >> 16) & 0xFFu);
    h *= 1099511628211ULL;
    h ^= ((v >> 24) & 0xFFu);
    h *= 1099511628211ULL;
  }
  h ^= static_cast<uint64_t>(g.size()) + 0x9e3779b97f4a7c15ULL + (h << 6) +
       (h >> 2);
  return h;
}

Island::Island(ThreadSafeEvaluator *evaluator, const ProblemData &data,
               int population_size, int id)
    : evaluator_(evaluator), demands_(data.GetDemands()),
      capacity_(data.GetCapacity()), geometry_(data, id),
      local_search_(evaluator, &geometry_, id),
      population_size_(population_size), id_(id),
      current_best_(evaluator->GetSolutionSize()),
      rng_(static_cast<unsigned int>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count() +
          id * 100000)),
      split_(evaluator) {
#ifdef RESEARCH
  InitStats();
#endif
  population_.reserve(population_size_);
  local_cache_.InitHistory(evaluator->GetSolutionSize());

  int n = evaluator_->GetSolutionSize();
  int g = evaluator_->GetNumGroups();

  last_in_group1.resize(g);
  last_in_group2.resize(g);
  pred1.resize(n);
  pred2.resize(n);

  mutator_.Initialize(evaluator_, &geometry_, &split_);

  int dim = data.GetDimension();
  customer_ranks_.resize(dim + 1, 0);
  const auto &perm = data.GetPermutation();
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] >= 0 && perm[i] < (int)customer_ranks_.size()) {
      customer_ranks_[perm[i]] = static_cast<int>(i);
    }
  }

  start_time_ = std::chrono::steady_clock::now();
  last_alns_print_time_ = std::chrono::steady_clock::now();
  last_greedy_assembly_time_ = std::chrono::steady_clock::now();
}

double Island::EvaluateWithHistoryPenalty(const std::vector<int> &genotype) {
  double base_cost = SafeEvaluate(genotype);
  if (base_cost >= 1e20)
    return base_cost;

  double history_penalty = 0.0;
  double lambda = Config::HISTORY_LAMBDA;

  size_t size = genotype.size();
  if (size < 2)
    return base_cost;

  for (size_t i = 0; i < size - 1; ++i) {
    if (genotype[i] == genotype[i + 1]) {
      int freq = local_cache_.GetFrequency(i);
      history_penalty += freq * lambda;
    }
  }
  return base_cost + history_penalty;
}

double Island::SafeEvaluate(const std::vector<int> &genotype) {
#ifdef RESEARCH
  total_evaluations++;
#endif
  double distance = 0.0;
  int returns = 0;

  if (local_cache_.TryGet(genotype, distance, returns)) {
    cache_hits_++;
    TrackCacheResult(true);  // rolling window tracking
    return distance;
  }
  cache_misses_++;
  TrackCacheResult(false);  // rolling window tracking
  EvaluationResult result = evaluator_->EvaluateWithStats(genotype);

  distance = result.fitness;
  returns = result.returns;

  if (distance >= 1e9 || distance < 0.0) {
    distance = std::numeric_limits<double>::max();
  }
  local_cache_.Insert(genotype, distance, returns);

  return distance;
}

double Island::SafeEvaluate(Individual &indiv) {
#ifdef RESEARCH
  total_evaluations++;
#endif
  double distance = 0.0;
  int returns = 0;
  if (local_cache_.TryGet(indiv.AccessGenotype(), distance, returns)) {
    cache_hits_++;
    TrackCacheResult(true);  // rolling window tracking
    indiv.SetReturnCount(returns);
    indiv.SetFitness(distance); // raw fitness
    return distance;
  }
  cache_misses_++;
  TrackCacheResult(false);  // rolling window tracking;

  EvaluationResult result = evaluator_->EvaluateWithStats(indiv.GetGenotype());

  distance = result.fitness;
  returns = result.returns;

  if (distance >= 1e15 || distance < 0.0) {
    distance = std::numeric_limits<double>::max();
  }

  indiv.SetReturnCount(returns);
  indiv.SetFitness(distance); // raw fitness

  local_cache_.Insert(indiv.AccessGenotype(), distance, returns);

  return distance;

}

void Island::InitIndividual(Individual &indiv, INITIALIZATION_TYPE strategy) {
  int num_groups = evaluator_->GetNumGroups();
  std::vector<int> &genes = indiv.AccessGenotype();
  int num_clients = static_cast<int>(genes.size());

  if (strategy == INITIALIZATION_TYPE::SMART_STICKY) {
    InitIndividualSmartSticky(indiv);
    return;
  }

  std::vector<int> assignment_pool;
  assignment_pool.reserve(num_clients);
  for (int g = 0; g < num_groups; ++g) {
    assignment_pool.push_back(g);
  }
  int remaining_slots = num_clients - num_groups;

  if (remaining_slots > 0) {
    if (strategy == INITIALIZATION_TYPE::RR) {
      for (int i = 0; i < remaining_slots; ++i) {
        assignment_pool.push_back(i % num_groups);
      }
    } else if (strategy == INITIALIZATION_TYPE::CHUNKED) {
      int chunk_size =
          (num_groups > 0) ? (num_clients / num_groups) : num_clients;
      int current_group = 0;
      for (int i = 0; i < num_clients; ++i) {
        genes[i] = current_group;
        if ((i + 1) % chunk_size == 0 && current_group < num_groups - 1) {
          current_group++;
        }
      }
      if (num_clients > 2) {
        std::uniform_int_distribution<int> d_swap(0, num_clients - 1);
        int a = d_swap(rng_);
        int b = d_swap(rng_);
        std::swap(genes[a], genes[b]);
      }
      return;
    } else {
      std::uniform_int_distribution<int> dist(0, num_groups - 1);
      for (int i = 0; i < remaining_slots; ++i) {
        assignment_pool.push_back(dist(rng_));
      }
    }
  }
  if ((int)assignment_pool.size() < num_clients) {
    assignment_pool.resize(num_clients, 0);
  }

  std::shuffle(assignment_pool.begin(), assignment_pool.end(), rng_);

  for (int i = 0; i < num_clients; ++i) {
    genes[i] = assignment_pool[i];
  }
}

void Island::InitIndividualSmartSticky(Individual &indiv) {
  int num_groups = evaluator_->GetNumGroups();
  int capacity = evaluator_->GetCapacity();
  std::vector<int> &genes = indiv.AccessGenotype();
  const std::vector<int> &perm = evaluator_->GetPermutation();
  int num_clients = static_cast<int>(genes.size());

  double total_dist = 0.0;
  for (size_t i = 0; i < perm.size() - 1; ++i) {
    int u_idx = perm[i] - 1;
    int v_idx = perm[i + 1] - 1;
    if (u_idx > 0 && v_idx > 0) {
      total_dist += evaluator_->GetDist(u_idx, v_idx);
    }
  }

  double avg_seq_dist = total_dist / (double)std::max(1, (int)perm.size() - 1);
  double random_factor = std::uniform_real_distribution<double>(0.5, 1.0)(rng_);
  double sticky_threshold = avg_seq_dist * random_factor;

  std::vector<int> group_loads(num_groups, 0);
  int current_group = rng_() % num_groups;

  for (size_t i = 0; i < perm.size(); ++i) {
    int customer_id = perm[i];
    if (customer_id <= 1)
      continue;

    int gene_idx = customer_id - 2;
    if (gene_idx < 0 || gene_idx >= num_clients)
      continue;

    int demand = evaluator_->GetDemand(customer_id);
    bool keep_same_group = false;

    if (i > 0) {
      int prev_id = perm[i - 1];
      if (prev_id > 1) {
        int u_idx = prev_id - 1;
        int v_idx = customer_id - 1;
        double dist = evaluator_->GetDist(u_idx, v_idx);
        if (dist < sticky_threshold) {
          keep_same_group = true;
        }
      }
    }

    if (keep_same_group) {
      if (group_loads[current_group] + demand > capacity) {
        std::vector<int> candidates(num_groups);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::shuffle(candidates.begin(), candidates.end(), rng_);

        bool found = false;
        for (int g : candidates) {
          if (group_loads[g] + demand <= capacity) {
            current_group = g;
            found = true;
            break;
          }
        }
        if (!found)
          current_group = candidates[0];
      }
    } else {
      current_group = rng_() % num_groups;
    }

    genes[gene_idx] = current_group;
    group_loads[current_group] += demand;
  }
}

void Island::Initialize(INITIALIZATION_TYPE strategy) {
  geometry_.Initialize(evaluator_);
  CalibrateDiversity();
  population_.clear();
  int sol_size = evaluator_->GetSolutionSize();
  Individual splitInd(sol_size);
  if (!Config::split) {
    InitIndividual(splitInd, INITIALIZATION_TYPE::RANDOM);
    population_.push_back(splitInd);
  } else {
    ApplySplitToIndividual(splitInd);
    double splitFit = SafeEvaluate(splitInd);
    splitInd.SetFitness(splitFit);
    population_.push_back(splitInd);
    {
      std::lock_guard<std::mutex> lock(best_mutex_);
      current_best_ = splitInd;
    }
  }

  // Initialize remaining population with random individuals
  for (int i = 0; i < population_size_ - 1; ++i) {
    Individual indiv(sol_size);
    InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);

    double fit = SafeEvaluate(indiv);
    indiv.SetFitness(fit);
    population_.push_back(indiv);

    if (fit < current_best_.GetFitness()) {
      std::lock_guard<std::mutex> lock(best_mutex_);
      current_best_ = indiv;
    }
  }

  UpdateBiasedFitness();

}

void Island::RunGeneration() {
  ProcessBroadcastBuffer(); // Process any pending broadcasts first

  current_generation_++;

  if (ShouldTrackDiversity()) {
    UpdateBiasedFitness();
    UpdateAdaptiveParameters();
  }
  stagnation_count_++;

  // === DIAGNOSTIC LOGGING (every 30s per island) ===
  auto now_diag = std::chrono::steady_clock::now();
  double since_last_diag =
      std::chrono::duration<double>(now_diag - last_diag_time_).count();
  if (since_last_diag >= 30.0 && current_generation_ > 10) {
    last_diag_time_ = now_diag;
    double elapsed =
        std::chrono::duration<double>(now_diag - start_time_).count();

    // Calculate unique genotypes in population
    std::unordered_set<uint64_t> unique_hashes;
    double best_pop_fit = 1e18, worst_pop_fit = 0;
    double sum_fit = 0;
    {
      std::lock_guard<std::mutex> lock(population_mutex_);
      for (const auto &ind : population_) {
        uint64_t h = HashGenotype64(ind.GetGenotype());
        unique_hashes.insert(h);
        double f = ind.GetFitness();
        sum_fit += f;
        if (f < best_pop_fit)
          best_pop_fit = f;
        if (f > worst_pop_fit)
          worst_pop_fit = f;
      }
    }
    int unique_count = (int)unique_hashes.size();
    double unique_pct = 100.0 * unique_count / population_size_;
    double fitness_spread = worst_pop_fit - best_pop_fit;
    double avg_fit = sum_fit / population_size_;

    // Stagnation metrics
    long long gens_since_improve = current_generation_ - last_improvement_gen_;
    double stagnation_rate =
        (elapsed > 0) ? (double)gens_since_improve / elapsed : 0;

    // Current best vs global best gap
    double global_best;
    {
      std::lock_guard<std::mutex> lock(best_mutex_);
      global_best = current_best_.GetFitness();
    }
    double gap_to_best = best_pop_fit - global_best;

    double vnd_success_rate =
        (diag_vnd_calls_ > 0)
            ? (100.0 * diag_vnd_improvements_ / diag_vnd_calls_)
            : 0.0;
    double strong_mut_pct =
        (diag_mutations_ > 0)
            ? (100.0 * diag_strong_mutations_ / diag_mutations_)
            : 0.0;
    double better_pct =
        (diag_offspring_total_ > 0)
            ? (100.0 * diag_offspring_better_ / diag_offspring_total_)
            : 0.0;

    // Diagnose problems
    std::string issues = "";
    if (unique_pct < 30.0)
      issues += "[CLONE_FLOOD] ";
    if (vnd_success_rate < 5.0 && diag_vnd_calls_ > 50)
      issues += "[VND_STUCK] ";
    if (gens_since_improve > 500 && IsExploitation())
      issues += "[STAGNANT] ";
    if (fitness_spread < 100 && IsExploration())
      issues += "[LOW_DIVERSITY] ";
    if (better_pct < 10.0 && diag_offspring_total_ > 50)
      issues += "[POOR_OFFSPRING] ";
    if (gap_to_best > 5000 && IsExploitation())
      issues += "[LAGGING] ";

    // Crossover success rates
    double srex_rate = (diag_srex_calls_ > 0) ? (100.0 * diag_srex_wins_ / diag_srex_calls_) : 0.0;
    double neighbor_rate = (diag_neighbor_calls_ > 0) ? (100.0 * diag_neighbor_wins_ / diag_neighbor_calls_) : 0.0;
    double pr_rate = (diag_pr_calls_ > 0) ? (100.0 * diag_pr_wins_ / diag_pr_calls_) : 0.0;

    // Construct diagnostic log line atomically
    std::ostringstream oss;
    oss << " [DIAG I" << id_ << " " << (IsExploration() ? "EXP" : "EXT")
        << "] "
        << "VND: " << diag_vnd_improvements_ << "/" << diag_vnd_calls_
        << " (" << std::fixed << std::setprecision(0) << vnd_success_rate
        << "%) | ";

    // Show relevant crossover stats per island type
    if (IsExploration()) {
      oss << "XO: SREX " << diag_srex_wins_ << "/" << diag_srex_calls_ << "(" << srex_rate << "%) "
          << "NBR " << diag_neighbor_wins_ << "/" << diag_neighbor_calls_ << "(" << neighbor_rate << "%) | ";
    } else {
      // EXPLOIT: show PR crossover + epsilon-greedy VND operator success rates
      oss << "XO:PR " << diag_pr_wins_ << "/" << diag_pr_calls_ << "(" << pr_rate << "%) "
          << "| EpsG: S2=" << std::setprecision(0) << (adapt_swap_.success_rate * 100) << "% "
          << "Ej=" << (adapt_ejection_.success_rate * 100) << "% "
          << "S3=" << (adapt_swap3_.success_rate * 100) << "% "
          << "S4=" << (adapt_swap4_.success_rate * 100) << "% | ";
    }

    oss << "Uniq: " << unique_count << "/" << population_size_ << " ("
        << std::setprecision(0) << unique_pct << "%) | "
        << "Div: " << std::fixed << std::setprecision(3) << current_structural_diversity_ << " | "
        << "Gap: " << std::setprecision(0) << gap_to_best << " | "
        << "Stag: " << gens_since_improve << "g ";

    if (!issues.empty()) {
      oss << "\033[31m" << issues << "\033[0m";
    }
    oss << "\n";

    // Atomic write to stdout
    std::cout << oss.str();

    // Reset counters
    diag_vnd_calls_ = diag_vnd_improvements_ = 0;
    diag_mutations_ = diag_strong_mutations_ = 0;
    diag_crossovers_ = diag_offspring_better_ = diag_offspring_total_ = 0;
    diag_srex_calls_ = diag_srex_wins_ = 0;
    diag_neighbor_calls_ = diag_neighbor_wins_ = 0;
    diag_pr_calls_ = diag_pr_wins_ = 0;
    
    // Update adaptive probabilities for next window (EXPLOIT only)
    if (IsExploitation()) {
      UpdateAdaptiveProbabilities();
    }
    
    // === CONVERGENCE DETECTION VIA CACHE HIT RATE ===
    // Check rolling window cache hit rate every diagnostic interval
    double recent_hit_rate = GetRecentCacheHitRate();
    if (cache_result_window_.size() >= CACHE_WINDOW_SIZE / 2) {  // Need at least half window
      if (recent_hit_rate > 0.95 && !convergence_alarm_active_) {
        OnConvergenceCritical();
      } else if (recent_hit_rate > 0.90 && !convergence_alarm_active_) {
        OnConvergenceAlarm();
      } else if (recent_hit_rate > 0.85 && convergence_mutation_boost_ < 2.0) {
        OnConvergenceWarning();
      } else if (recent_hit_rate < 0.70) {
        // Reset mutation boost when cache hit rate is healthy
        convergence_mutation_boost_ = 1.0;
        convergence_alarm_active_ = false;
      }
    }
  }

  // ASYNCHRONOUS MIGRATION: Try to pull migrant from predecessor when stuck
  // (immigration_queue_ no longer used - migrants are injected directly in
  // InjectImmigrant)
  TryPullMigrant();

  const int lambda = population_size_;
  std::vector<Individual> offspring_pool;
  offspring_pool.reserve(lambda);

  double fitness_threshold = std::numeric_limits<double>::max();
  if (!population_.empty()) {
    fitness_threshold = population_[population_.size() / 2].GetFitness();
  }

  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - start_time_).count();
  bool is_endgame =
      (elapsed > Config::MAX_TIME_SECONDS * Config::ENDGAME_THRESHOLD);

    // Unified Operator Selection (Roulette Wheel) for Exploitation
    // OR Standard Crossover/Mutation for Exploration
    
    // Probabilities (Exploitation vs Exploration)
    
    std::uniform_real_distribution<double> d(0.0, 1.0); // Restored 'd'
    std::uniform_real_distribution<double> dist_op(0.0, 1.0);

    for (int i = 0; i < lambda; ++i) {
        Individual child(evaluator_->GetSolutionSize());
        int p1 = -1, p2 = -1;
        double op_val = dist_op(rng_);

        bool mutated = false;
        bool strong_mutation = false;
        int op_type = 0; // 0=None, 1=SREX, 2=Neighbor, 3=PR, 4=Mutation(RR/Split)
        int crossover_type = 0;
        double parent1_fit = 0.0;
        double parent2_fit = 0.0;
        bool operator_selected = false;  // Flag to skip normal operator selection

    if (IsExploitation()) {
       // === EXPLOIT STAGNATION RESCUE: Heavy R&R injection ===
       // When stuck for 500+ gens, periodically inject heavily perturbed solution
       long long time_since_improve = current_generation_ - last_improvement_gen_;
       if (time_since_improve > Config::EXPLOIT_RR_STAGNATION_TRIGGER &&
           i == 0 && current_generation_ % Config::EXPLOIT_RR_INTERVAL == 0) {
         // Pick random individual and apply heavy R&R (40% destruction)
         int victim = rng_() % population_.size();
         child = population_[victim];
         mutator_.ApplyRuinRecreate(child, Config::EXPLOIT_HEAVY_RR_INTENSITY, true, rng_);
         mutated = true;
         strong_mutation = true;
         op_type = 4;  // R&R mutation
         operator_selected = true;  // Skip normal operator selection
       }
       
       if (!operator_selected) {
       // === EXPLOITATION: UNIFIED OPERATOR SELECTION (Roulette Wheel) ===
       double pr_prob = 0.60;
       double rr_prob = 0.20;
       double split_prob = 0.20;
       
       // Load config based on ID
       if (id_ == 1) {
           pr_prob = Config::EXPLOIT_I1_OP_PR_PROB;
           rr_prob = Config::EXPLOIT_I1_OP_RR_PROB;
           split_prob = Config::EXPLOIT_I1_OP_SPLIT_PROB;
       } else if (id_ == 3) {
           pr_prob = Config::EXPLOIT_I3_OP_PR_PROB;
           rr_prob = Config::EXPLOIT_I3_OP_RR_PROB;
           split_prob = Config::EXPLOIT_I3_OP_SPLIT_PROB;
       } else if (id_ == 5) {
           pr_prob = Config::EXPLOIT_I5_OP_PR_PROB;
           rr_prob = Config::EXPLOIT_I5_OP_RR_PROB;
       } 

       // HUGE instances (n > 3000): Very limited PR for quality boost
       // PR is O(N²) - keep it rare to avoid stalling
       if (evaluator_->GetSolutionSize() > Config::HUGE_INSTANCE_THRESHOLD) {
           pr_prob = 0.01;    // 1% PR only (was 15%)
           rr_prob = 0.89;    // R&R dominates for speed
           split_prob = 0.10; // Keep structural changes
       }

       // Normalize if needed, but assuming sum=1.0 from Constants
       
       if (op_val < pr_prob) {
           // === OPERATOR 1: PATH RELINKING (CROSSOVER) ===
           // Requires 2 parents
           int candidates[4];
           for (int c = 0; c < 4; ++c) candidates[c] = SelectParentIndex();
           
           int best_c1 = 0, best_c2 = 1;
           int max_dist = -1;
           int num_groups = evaluator_->GetNumGroups();

           for (int c1 = 0; c1 < 4; ++c1) {
             for (int c2 = c1 + 1; c2 < 4; ++c2) {
               int d = CalculateBrokenPairsDistance(population_[candidates[c1]], 
                                                    population_[candidates[c2]], 
                                                    evaluator_->GetProblemData().GetPermutation(), 
                                                    num_groups);
               if (d > max_dist) {
                 max_dist = d;
                 best_c1 = candidates[c1];
                 best_c2 = candidates[c2];
               }
             }
           }
           p1 = best_c1; 
           p2 = best_c2;
           parent1_fit = population_[p1].GetFitness();
           parent2_fit = population_[p2].GetFitness();
           
           if (parent1_fit < parent2_fit) {
              child = population_[p1];
              double cost = child.GetFitness();
              local_search_.TryPathRelinking(child.AccessGenotype(), cost, population_[p2].GetGenotype());
              child.SetFitness(cost);
           } else {
              child = population_[p2];
              double cost = child.GetFitness();
              local_search_.TryPathRelinking(child.AccessGenotype(), cost, population_[p1].GetGenotype());
              child.SetFitness(cost);
           }
           op_type = 3; // PR
           crossover_type = 3;
           
       } else if (op_val < pr_prob + rr_prob) {
           // === OPERATOR 2: RUIN & RECREATE (MUTATION) ===
           p1 = SelectParentIndex();
           child = population_[p1];
           // Intensity 0.0 -> uses Base Exploitation Pct (10-25%)
           mutator_.ApplyRuinRecreate(child, 0.0, true, rng_);
           mutated = true;
           op_type = 4;
       } else {
           // === OPERATOR 3: MICROSPLIT (MUTATION) ===
           p1 = SelectParentIndex();
           child = population_[p1];
           // Level 2 = Small windows
           mutator_.ApplyMicroSplitMutation(child, 0.0, 2, rng_);
           mutated = true;
           op_type = 4;
       }
       }  // end if (!operator_selected)
       
    } else {
      // === EXPLORATION: STANDARD CROSSOVER/MUTATION ===
      double p_crossover = 0.60;
      
      // Standard tournament selection
      p1 = SelectParentIndex();
      p2 = SelectParentIndex();

      if (p1 >= 0 && p2 >= 0 && op_val < p_crossover) {
        // === CROSSOVER BRANCH ===
        parent1_fit = population_[p1].GetFitness();
        parent2_fit = population_[p2].GetFitness();
        
        if (is_endgame) {
          child = CrossoverNeighborBased(population_[p1], population_[p2]);
          crossover_type = 2; // Neighbor
        } else {
           // EXPLORE: SREX or Neighbor
           double seq_prob = (id_ <= 1) ? 0.31 : 0.5;
           crossover_type = (dist_op(rng_) < seq_prob) ? 1 : 2; 
           child = Crossover(population_[p1], population_[p2]);
        }
      } else {
        // === MUTATION BRANCH ===
        if (p1 >= 0) child = population_[p1];
        else InitIndividual(child, INITIALIZATION_TYPE::RANDOM);
        
        int m_res = ApplyMutation(child, is_endgame);
        mutated = (m_res > 0);
        strong_mutation = (m_res == 2);
      }
    }

    // Diagnostic tracking (removed goto label)
    if (mutated)
      diag_mutations_++;
    if (strong_mutation)
      diag_strong_mutations_++;
    diag_crossovers_++;
    
    // Cache problem size for this iteration
    int problem_size = evaluator_->GetSolutionSize();
    
    child.Canonicalize();
    double fit = 0;
    int ret = 0;
    if (!local_cache_.TryGet(child.GetGenotype(), fit, ret)) {
      cache_misses_++;
      EvaluationResult res = evaluator_->EvaluateWithStats(child.GetGenotype());
      fit = res.fitness;
      ret = res.returns;
      local_cache_.Insert(child.GetGenotype(), fit, ret);
    } else {
      cache_hits_++;
    }
    child.SetFitness(fit);
    child.SetReturnCount(ret);

    // Track crossover success: child better than BOTH parents (before mutation/VND)
    if (crossover_type > 0 && parent1_fit > 0 && parent2_fit > 0) {
      if (fit < parent1_fit && fit < parent2_fit) {
        if (crossover_type == 1) diag_srex_wins_++;
        else if (crossover_type == 2) diag_neighbor_wins_++;
        else if (crossover_type == 3) diag_pr_wins_++;
      }
      if (crossover_type == 1) diag_srex_calls_++;
      else if (crossover_type == 2) diag_neighbor_calls_++;
      else if (crossover_type == 3) diag_pr_calls_++;
    }

    bool promising = (fit < fitness_threshold);
    
    // VND probability - drastically reduced for large instances
    double vnd_prob;
    if (problem_size > Config::LARGE_INSTANCE_THRESHOLD) {
      vnd_prob = IsExploration() ? Config::EXPLORATION_VND_PROB_LARGE
                                 : Config::EXPLOITATION_VND_PROB_LARGE;
    } else {
      vnd_prob = IsExploration() ? Config::EXPLORATION_VND_PROB
                                 : Config::EXPLOITATION_VND_PROB;
    }
    
    bool exploration_vnd =
        IsExploration() &&
        (promising || (d(rng_) < Config::EXPLORATION_VND_EXTRA_PROB));
    
    // Skip VND entirely for exploration on large instances
    // Exploration relies on mutation diversity, not local search refinement
    if (problem_size > Config::LARGE_INSTANCE_THRESHOLD && IsExploration()) {
      exploration_vnd = false;
    }
    
    bool should_run_vnd = exploration_vnd || strong_mutation ||
                          (IsExploitation() && promising) ||
                          (d(rng_) < vnd_prob) || is_endgame;

    if (should_run_vnd) {
      int vnd_iters = GetVndIterations();
      if (is_endgame) {
        vnd_iters = Config::EXPLOITATION_VND_MAX; // Max power in endgame
      } else if (current_structural_diversity_ > 0.6) {
        vnd_iters = (int)(vnd_iters * 1.5);
      }
      bool allow_swap = IsExploitation() && Config::ALLOW_SWAP;
      
      // === EPSILON-GREEDY ADAPTIVE OPERATOR SELECTION ===
      // Select exactly ONE operator: 0=Swap, 1=Ejection, 2=3-Swap, 3=4-Swap
      bool allow_3swap = false;
      bool allow_ejection = false;
      bool allow_4swap = false;
      int selected_op = -1;  // Track which operator was selected for credit
      
      if (IsExploitation() && problem_size < Config::LARGE_INSTANCE_THRESHOLD && !strong_mutation) {
        // Select ONE operator using epsilon-greedy
        selected_op = SelectAdaptiveOperator();  // 0=Swap, 1=Ejection, 2=3-Swap, 3=4-Swap
        
        // Reset allow_swap to false, will enable only if selected
        allow_swap = false;
        
        // === COMBO OPERATOR SELECTION ===
        // Instead of selecting ONE operator, we enable the selected one PLUS neighbors
        // This allows broader exploration while still using adaptive learning
        switch (selected_op) {
          case 0:  // Swap selected: enable Swap + Ejection
            allow_swap = true;
            allow_ejection = true;
            adapt_swap_.calls++;
            break;
          case 1:  // Ejection selected: enable Ejection + 3-Swap
            allow_ejection = true;
            allow_3swap = true;
            adapt_ejection_.calls++;
            break;
          case 2:  // 3-Swap selected: enable 3-Swap + Ejection + 4-Swap
            allow_ejection = true;
            allow_3swap = true;
            allow_4swap = true;
            adapt_swap3_.calls++;
            break;
          case 3:  // 4-Swap selected: enable 4-Swap + 3-Swap
            allow_3swap = true;
            allow_4swap = true;
            adapt_swap4_.calls++;
            break;
        }
        
        // No PR in VND - it's already used as crossover (90% in RunGeneration)
        local_search_.SetGuideSolution({});
      } else if (IsExploration()) {
        // Exploration: minimal VND operators
        allow_swap = false;
        allow_3swap = false;
        allow_ejection = false;
        local_search_.SetGuideSolution({});
      }

      diag_vnd_calls_++;
      double fit_before = child.GetFitness();
      
      // 10% chance for FULL VND (unlocks 3-opt, Ejection, etc.)
      bool force_full_vnd = (d(rng_) < 0.10);
      if (force_full_vnd) {
          allow_swap = true;
          allow_3swap = true;
          allow_ejection = true; // Use Ejection if enabled globally
      }

      // DECOMPOSED VND: Use sector-based decomposition for large instances
      // exploration_mode=true for exploration islands -> ultra-fast optimization
      bool vnd_improved = false;
      if (problem_size > Config::LARGE_INSTANCE_THRESHOLD && !force_full_vnd) {
        // Use decomposed VND with exploration_mode flag
        // Exploration: 32 sectors, 1 iter, no boundary = ~20x faster
        // Exploitation: 16 sectors, full iters, 2 boundary passes = quality
        vnd_improved = local_search_.RunDecomposedVND(child, vnd_iters, IsExploration());
      } else {
        // Standard VND for small instances - full quality
        vnd_improved = local_search_.RunVND(child, vnd_iters, allow_swap, allow_3swap,
                                            allow_ejection, allow_4swap);
      }
      
      if (vnd_improved) {
        child.Canonicalize();
        if (!local_cache_.TryGet(child.GetGenotype(), fit, ret)) {
          cache_misses_++;
          EvaluationResult res =
              evaluator_->EvaluateWithStats(child.GetGenotype());
          fit = res.fitness;
          ret = res.returns;
          local_cache_.Insert(child.GetGenotype(), fit, ret);
        } else {
          cache_hits_++;
        }
        child.SetFitness(fit);
        child.SetReturnCount(ret);
        if (fit < fit_before) {
          diag_vnd_improvements_++;
          
          // === ADAPTIVE SUCCESS TRACKING ===
          // Credit the ONE operator that was selected for this VND run
          if (IsExploitation() && problem_size < Config::LARGE_INSTANCE_THRESHOLD && selected_op >= 0) {
            switch (selected_op) {
              case 0: adapt_swap_.wins++; break;
              case 1: adapt_ejection_.wins++; break;
              case 2: adapt_swap3_.wins++; break;
              case 3: adapt_swap4_.wins++; break;
            }
          }
        }
      }
    }

    // Track offspring quality
    diag_offspring_total_++;
    if (fit < fitness_threshold)
      diag_offspring_better_++;

    offspring_pool.push_back(std::move(child));

    // GLS update removed from inner loop to prevent noise

    // Track if we need to broadcast (do it OUTSIDE the lock to prevent deadlock)
    bool should_broadcast = false;
    Individual best_to_broadcast;
    
    {
      std::lock_guard<std::mutex> lock(best_mutex_);
      if (fit < current_best_.GetFitness()) {
        // Mark as native to this island
        offspring_pool.back().SetNative(true);
        offspring_pool.back().SetHomeIsland(id_);
        
        current_best_ = offspring_pool.back();
        stagnation_count_ = 0;
        last_improvement_gen_ = current_generation_;
        fitness_threshold = fit * 1.05;
        
        // Prepare for broadcast (but don't do it while holding lock!)
        // BOTH EXPLORE and EXPLOIT broadcast new global bests to EXPLOIT siblings
        // (EXPLORE discoveries are refined by EXPLOIT "chase" mechanisms)
        if (!exploit_siblings_.empty()) {
          should_broadcast = true;
          best_to_broadcast = current_best_;  // Copy for broadcast
        }
      }
    }
    
    // === NON-NATIVE BROADCAST (OUTSIDE LOCK) ===
    // EXPLOIT islands share best with other EXPLOIT islands (after warmup, with filters)
    // EXPLORE islands IMMEDIATELY share best with ALL EXPLOIT islands (no warmup, no filters!)
    // This ensures good exploration discoveries are instantly available for exploitation
    auto now_broadcast = std::chrono::steady_clock::now();
    double elapsed_broadcast = std::chrono::duration<double>(now_broadcast - start_time_).count();
    
    if (should_broadcast) {
      if (IsExploration()) {
        // EXPLORE: IMMEDIATE unconditional broadcast to all EXPLOIT siblings
        // No warmup delay, no filters - EXPLOIT needs fresh exploration material ASAP
        // Set home_island to THIS island's ID so receiver knows source
        best_to_broadcast.SetHomeIsland(id_);
        for (Island* sibling : exploit_siblings_) {
          if (sibling != nullptr) {
            sibling->ReceiveBroadcastBest(best_to_broadcast);
          }
        }
        // Log only occasionally to avoid spam
        if (current_generation_ % 100 == 0) {
          std::cout << "\033[36m [I" << id_ << " EXPLORE] Immediate broadcast to EXPLOIT siblings\033[0m\n";
        }
      } else {
        // EXPLOIT: Original logic with warmup (to avoid early noise)
        bool broadcast_enabled = (elapsed_broadcast > Config::BROADCAST_WARMUP_SECONDS);
        if (broadcast_enabled) {
          best_to_broadcast.SetHomeIsland(id_);  // Set source island ID
          for (Island* sibling : exploit_siblings_) {
            if (sibling != nullptr) {
              sibling->ReceiveBroadcastBest(best_to_broadcast);
            }
          }
        }
      }
    }
  }

  // GLS: update edge penalty removed from here.
  // We now penalize effective Local Optima inside the loop.
  // if (stagnation_count_ > 0 && IsExploration()) {
  //     std::lock_guard<std::mutex> lock(best_mutex_);
  //     UpdateEdgePenalty(current_best_.GetGenotype());
  // }

  ApplySuccessionAdaptive(offspring_pool);

  {
    std::lock_guard<std::mutex> lock(population_mutex_);
    for (auto &ind : population_)
      ind.IncrementStagnation();
  }

  long long time_since = current_generation_ - last_improvement_gen_;
  long long time_since_cat = current_generation_ - last_catastrophy_gen_;

  double worst_fit = 0.0;
  {
    std::lock_guard<std::mutex> lock(population_mutex_);
    for (const auto &ind : population_)
      if (ind.GetFitness() > worst_fit)
        worst_fit = ind.GetFitness();
  }

  double best_fit_for_catastrophe;
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    best_fit_for_catastrophe = current_best_.GetFitness();
  }

  // === ISLAND-SPECIFIC CATASTROPHE THRESHOLDS ===
  // EXPLOIT: faster trigger (2000g, VND<20%) to escape local optima quickly
  // EXPLORE: slower trigger (5000g, VND<3%) to allow deep exploration
  int stag_threshold = IsExploitation() 
      ? Config::EXPLOIT_CATASTROPHE_STAGNATION_GENS   // 2000g
      : Config::CATASTROPHE_STAGNATION_GENS;          // 5000g
  bool stagnation_trigger = (time_since > stag_threshold);

  // VND-based trigger with island-specific thresholds
  double vnd_success_rate =
      (diag_vnd_calls_ > 0) ? (100.0 * diag_vnd_improvements_ / diag_vnd_calls_)
                            : 100.0;
  double vnd_exhausted_thresh = IsExploitation() 
      ? Config::EXPLOIT_VND_EXHAUSTED_THRESHOLD   // 20%
      : Config::VND_EXHAUSTED_THRESHOLD;          // 3%
  bool vnd_exhausted = (vnd_success_rate < vnd_exhausted_thresh &&
                        diag_vnd_calls_ > Config::VND_EXHAUSTED_MIN_CALLS);

  if ((stagnation_trigger || vnd_exhausted) &&
      time_since_cat > Config::CATASTROPHE_MIN_GAP_GENS) {
    const char *reason = vnd_exhausted ? "VND_EXHAUSTED" : "STAGNATION";
    std::cout << "\033[96m [CATASTROPHE I" << id_ << "] Trigger: " << reason
              << " (stag=" << time_since << "g, VND=" << std::fixed
              << std::setprecision(1) << vnd_success_rate
              << "%, Div=" << std::setprecision(2)
              << current_structural_diversity_ << ")\033[0m\n";
    Catastrophy();
    last_catastrophy_gen_ = current_generation_;
  }

  // Update is_stuck_ flag for asynchronous migration
  is_stuck_.store(time_since > STUCK_THRESHOLD || vnd_exhausted,
                  std::memory_order_relaxed);

  if (IsExploitation()) {
    {
      std::lock_guard<std::mutex> lock(best_mutex_);
      // Only add to RoutePool if it's a high-quality solution (e.g., current best)
      // Optimization: No need to re-add the same best every generation if it hasn't changed.
      // We rely on route_pool_'s internal deduplication, but checking hash/timestamp is faster.
      // Add routes periodically OR on improvement - enables Frankenstein during stagnation
      if (stagnation_count_ == 0 || current_generation_ % 100 == 0) {
        route_pool_.AddRoutesFromSolution(current_best_.GetGenotype(), *evaluator_);
      }
    }

    size_t current_updates = route_pool_.GetTotalRoutesAdded();
    bool pool_updated = (current_updates > last_routes_added_snapshot_);

    // Trigger Frankenstein ONLY when RoutePool has actually learned
    // === ENDGAME MODE CHECK ===
    auto now_gen = std::chrono::steady_clock::now();
    double elapsed_gen =
        std::chrono::duration<double>(now_gen - start_time_).count();
    bool is_endgame =
        (elapsed_gen > Config::MAX_TIME_SECONDS * Config::ENDGAME_THRESHOLD);

    // Dynamic Parameters
    int vnd_iters = GetVndIterations();
    if (is_endgame) {
      // Boost VND in endgame for everyone
      vnd_iters = Config::EXPLOITATION_VND_MAX;
    }

    // FRANKENSTEIN
    bool use_frankenstein = Config::ENABLE_FRANKENSTEIN;
    // RESTRICTION: Frankenstein only for Exploitation islands to maintain
    // diversity
    if (IsExploration())
      use_frankenstein = false;

    // HEURISTIC: Disable Frankenstein for huge instances (>2000 customers)
    // Beam Search is too expensive (O(N^2) or worse) here
    if (evaluator_->GetSolutionSize() > Config::FRANKENSTEIN_MAX_INSTANCE_SIZE)
      use_frankenstein = false;

    if (use_frankenstein &&
        route_pool_.HasNewRoutesSince(last_routes_added_snapshot_)) {
      last_routes_added_snapshot_ = current_updates;
      Individual frankenstein = route_pool_.SolveBeamSearch(
          evaluator_, split_, Config::FRANKENSTEIN_BEAM_WIDTH);
      if (frankenstein.IsEvaluated() && frankenstein.GetFitness() < 1e9) {
        int vnd_iters = Config::FRANKENSTEIN_VND_ITERS;
        if (elapsed > Config::MAX_TIME_SECONDS * 0.8)
          vnd_iters = Config::FRANKENSTEIN_VND_ITERS_LATE;

        bool improved = false;
        for (int pass = 0; pass < Config::FRANKENSTEIN_VND_PASSES; ++pass) {
          if (local_search_.RunVND(frankenstein, vnd_iters, true, true, true))
            improved = true;
          else
            break;
        }

        if (improved) {
          frankenstein.Canonicalize();
          frankenstein.SetFitness(SafeEvaluate(frankenstein));
          // std::cout << " [BEAM] Frankenstein improved by VND! Final Fit: " <<
          // frankenstein.GetFitness() << std::endl;
        }

        // [ANTI-CLONE] Check if this frankenstein is already in population
        if (!ContainsSolution(frankenstein)) {
          std::lock_guard<std::mutex> lock(population_mutex_);

          // Force Injection Logic (User Request: "siłowo wstrzykiwany")
          // 10% chance to force inject, displacing a random individual (but not
          // the absolute best)
          bool force_injected = false;
          std::uniform_real_distribution<double> d_force(0.0, 1.0);
          if (d_force(rng_) < Config::FRANKENSTEIN_FORCE_INJECT_PROB) {
            int victim_idx = rng_() % population_.size();
            // Protect the absolute best from forced replacement to ensure
            // monotonicity of best found
            if (population_[victim_idx].GetFitness() >
                current_best_.GetFitness() + 1e-6) {
              population_[victim_idx] = frankenstein;
              std::cout << "\033[35m [BEAM] [Island " << id_
                        << "] Frankenstein FORCIBLY injected (Fit: "
                        << frankenstein.GetFitness() << ")\033[0m" << std::endl;
              force_injected = true;
            }
          }

          if (!force_injected) {
            int worst = GetWorstBiasedIndex();
            if (worst >= 0) {
              if (frankenstein.GetFitness() < population_[worst].GetFitness()) {
                population_[worst] = frankenstein;
                std::cout << "\033[35m [BEAM] [Island " << id_
                          << "] Frankenstein injected into population (Fit: "
                          << frankenstein.GetFitness() << ")\033[0m"
                          << std::endl;
              }
            }
          }
        }
        {
          std::lock_guard<std::mutex> lock(best_mutex_);
          if (frankenstein.GetFitness() < current_best_.GetFitness()) {
            current_best_ = frankenstein;
            stagnation_count_ = 0;
          }
        }
      }
      last_greedy_assembly_time_ = now;
    }
  }
}

bool Island::ContainsSolution(const Individual &ind) const {
  uint64_t h = HashGenotype64(ind.GetGenotype());
  int num_groups = evaluator_->GetNumGroups();
  const auto &perm = evaluator_->GetPermutation();
  int genotype_size = static_cast<int>(ind.GetGenotype().size());

  // BPD threshold: if < 10% of pairs differ, treat as clone (STATIC - simpler)
  int bpd_clone_threshold =
      std::max(10, genotype_size * 10 / 100); // 10% different

  for (const auto &p : population_) {
    // fast hash check first
    if (HashGenotype64(p.GetGenotype()) == h)
      return true;

    // BPD-based structural similarity (not fitness!)
    // this prevents ping-pong: if migrant is structurally similar to anyone,
    // reject
    int bpd = const_cast<Island *>(this)->CalculateBrokenPairsDistance(
        ind, p, perm, num_groups);

    if (bpd < bpd_clone_threshold) {
      return true; // too similar structurally - treat as clone
    }
  }
  return false;
}

// RunDebugDiagnostics() - REMOVED: Dead code, never called

int Island::ApplyMicroSplitMutation(Individual &child) {
  double stagnation_factor = std::min(1.0, (double)stagnation_count_ / 2000.0);

  // Per-island split level: I0=small(2), I2=medium(1), I4=large(0)
  int intensity;
  if (IsExploration()) {
    switch (id_) {
    case 0:
      intensity = Config::EXPLORE_I0_SPLIT_LEVEL;
      break;
    case 2:
      intensity = Config::EXPLORE_I2_SPLIT_LEVEL;
      break;
    case 4:
      intensity = Config::EXPLORE_I4_SPLIT_LEVEL;
      break;
    default:
      intensity = 1;
      break;
    }
  } else {
    intensity = 0; // exploitation uses smallest windows
  }
  bool success = mutator_.ApplyMicroSplitMutation(child, stagnation_factor,
                                                  intensity, rng_);

#ifdef RESEARCH
  return success ? (int)OpType::MUT_SIMPLE : 0;
#else
  return 0;
#endif
}

int Island::ApplyMutation(Individual &child, bool is_endgame) {
  std::uniform_real_distribution<double> d(0.0, 1.0);
  int executed_op = -1;
  bool mutated = false;
  bool strong_mutation = false; // To return if significant change happened, allowing VND

  // === EXCLUSIVE OPERATOR SETS FOR EXPLORE ===
  // Each EXPLORE island has a DISTINCT set of operators to maximize diversity
  // and minimize redundant exploration. This should reduce cache hit rate.
  if (IsExploration()) {
    double rnd = d(rng_);
    switch (id_) {
      case 0: // DESTRUKTOR - only R&R and Aggressive
        // Generates chaotic but diverse genetic material
        if (rnd < 0.65) {
          // Heavy Ruin & Recreate (30-70% destruction)
          double intensity = 0.3 + d(rng_) * 0.4; // random intensity 0.3-0.7
          mutator_.ApplyRuinRecreate(child, intensity, false, rng_);
          strong_mutation = true;
        } else {
          // Aggressive random mutation
          mutator_.AggressiveMutate(child, rng_);
          strong_mutation = true;
        }
        mutated = true;
        break;
        
      case 2: // RESTRUKTURYZATOR - only MergeSplit and MicroSplit
        // Restructures route boundaries without random destruction
        if (rnd < 0.55) {
          // Merge two groups and redistribute
          if (mutator_.ApplyMergeSplit(child, rng_)) {
            strong_mutation = true;
          }
        } else {
          // MicroSplit with varying window sizes (levels 0,1,2)
          int level = std::uniform_int_distribution<int>(0, 2)(rng_);
          double stagnation_factor = std::min(1.0, (double)stagnation_count_ / 2000.0);
          mutator_.ApplyMicroSplitMutation(child, stagnation_factor, level, rng_);
          strong_mutation = true;
        }
        mutated = true;
        break;
        
      case 4: // LOKALNY EKSPLORATOR - SmartMove, Simple, LoadBalance
        // Fine-grained local improvements, no destructive operators
        if (rnd < 0.45) {
          // Smart spatial move to nearby group
          mutator_.ApplySmartSpatialMove(child, rng_);
        } else if (rnd < 0.75) {
          // Simple swap or random move
          mutator_.ApplySimpleMutation(child, rng_);
        } else {
          // Load balancing (chain or swap)
          if (!ApplyLoadBalancingChainMutation(child)) {
            ApplyLoadBalancingSwapMutation(child);
          }
        }
        mutated = true;
        // Local moves are not "strong" - no guaranteed VND trigger
        break;
        
      default:
        // Fallback (shouldn't happen for EXPLORE islands)
        mutator_.ApplySimpleMutation(child, rng_);
        mutated = true;
        break;
    }
    
    // Return early for EXPLORE - exclusive operator already applied
    if (strong_mutation) return 2;
    if (mutated) return 1;
    return 0;
  }

  // === EXPLOITATION ISLANDS - ORIGINAL LOGIC ===
  // (Unchanged for I1, I3, I5)
  
  // 1. Structural Mutations (MicroSplit)
  if (d(rng_) < p_microsplit_) {
    ApplyMicroSplitMutation(child);
    strong_mutation = true;
    mutated = true;
#ifdef RESEARCH
    executed_op = (int)OpType::MUT_SIMPLE;
#endif
  }

  // 2. Standard Mutations (Aggressive / Smart Spatial / Ruin)
  if (d(rng_) < p_mutation_) {
    double rnd = d(rng_);
    double rr_threshold = Config::MUT_SPATIAL_THRESHOLD;

    if (rnd < Config::MUT_AGGRESSIVE_THRESHOLD) {
      mutator_.AggressiveMutate(child, rng_);
#ifdef RESEARCH
      executed_op = (int)OpType::MUT_AGGRESSIVE;
#endif
    } else if (rnd < rr_threshold) {
      mutator_.ApplySmartSpatialMove(child, rng_);
    } else {
      mutator_.ApplyRuinRecreate(child, (1 - current_structural_diversity_),
                                 true, rng_); // is_exploitation = true
#ifdef RESEARCH
      executed_op = (int)OpType::MUT_SPATIAL;
#endif
    }
    mutated = true;
  }

  // 2.5 MergeRegret mutation (re-enabled as requested)
  if (d(rng_) < 0.15) {
    if (ApplyMergeRegret(child)) {
      strong_mutation = true;
      mutated = true;
    }
  }

  // 3. Split Overloaded Routes (replaces Load Balancing)
  // "Remove Returns": Split overloaded routes into fresh vehicles to eliminate internal returns
  // 3. Load Balancing (Configurable) - SKIP for large instances (O(N²) kills performance)
  int problem_size = evaluator_->GetSolutionSize();
  bool allow_lb = Config::ALLOW_LOAD_BALANCING && 
                  problem_size < Config::LARGE_INSTANCE_THRESHOLD;
  if (allow_lb && d(rng_) < p_loadbalance_) {
    if (!ApplyLoadBalancingChainMutation(child))
      ApplyLoadBalancingSwapMutation(child);
    mutated = true;
  }

  // 4. Return Minimizer
  if (d(rng_) < p_retminimizer_) {
    mutator_.ApplyReturnMinimizer(child, rng_);
    mutated = true;
  }

  // 5. Merge Split (Targeted restructuring)
  if (d(rng_) < p_mergesplit_) {
    if (mutator_.ApplyMergeSplit(child, rng_)) {
      mutated = true;
      strong_mutation = true;
    }
  }

  // 6. 3-Swap (Heavy local move)
  double swap_chance = is_endgame ? Config::ENDGAME_P_SWAP3 : p_swap3_;
  if (d(rng_) < swap_chance) {
    if (local_search_.Try3Swap(child.AccessGenotype())) {
      mutated = true;
      strong_mutation = true;
    }
  }

  // 7. 4-Swap (Very heavy)
  double current_p_swap4 = is_endgame ? Config::ENDGAME_P_SWAP4 : p_swap4_;
  if (d(rng_) < current_p_swap4) {
    if (local_search_.Try4Swap(child.AccessGenotype())) {
      mutated = true;
      strong_mutation = true;
    }
  }

  if (strong_mutation)
    return 2;
  if (mutated)
    return 1;
  return 0;
}

void Island::Catastrophy() {
#ifdef RESEARCH
  catastrophy_activations++;
  cout << "Catastrophe on island [" << id_ << "] Div: " << std::scientific
       << std::setprecision(2) << current_structural_diversity_
       << std::defaultfloat << "\n";
#endif

  std::vector<Individual> new_pop;
  new_pop.reserve(population_size_);
  
  // === KEEP CURRENT BEST REGARDLESS OF NATIVE STATUS ===
  // Previously we reset non-native to native, but this breaks "chase mode"
  // where EXPLOIT tracks the global best from EXPLORE. Keep the best solution
  // as a seed - it will be refined, not replaced by random initialization.
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    new_pop.push_back(current_best_);
  }
  
  // === RETENTION: Keep top 33% of population as perturbation base ===
  // Instead of generating 100% new random solutions, keep elite individuals
  // and apply strong mutation. This preserves search progress.
  int keep_elite = std::max(1, population_size_ / 3);  // 33% retention
  
  {
    std::lock_guard<std::mutex> pop_lock(population_mutex_);
    // Sort population by fitness
    std::sort(population_.begin(), population_.end());
    
    // Add elite individuals (starting from index 1 since best is already added)
    for (int i = 0; i < keep_elite - 1 && i < (int)population_.size(); ++i) {
      Individual elite_copy = population_[i];
      // Apply perturbation to elite: 30% R&R mutation
      mutator_.ApplyRuinRecreate(elite_copy, 0.30, IsExploitation(), rng_);
      elite_copy.SetNative(true);
      elite_copy.SetHomeIsland(id_);
      double fit = SafeEvaluate(elite_copy.GetGenotype());
      elite_copy.SetFitness(fit);
      new_pop.push_back(elite_copy);
    }
  }
  
  int sol_size = evaluator_->GetSolutionSize();

  // Generate fresh candidates for remaining slots (67% of population)
  int remaining = population_size_ - (int)new_pop.size();
  std::vector<Individual> candidates;
  int candidates_count = remaining * 5;  // was population_size_ * 10

  for (int i = 0; i < candidates_count; ++i) {
    Individual indiv(sol_size);
    if (i % 2 == 0)
      InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);
    else
      InitIndividual(indiv, INITIALIZATION_TYPE::CHUNKED);

    // Mark as native to this island
    indiv.SetNative(true);
    indiv.SetHomeIsland(id_);

    // GLS: use clean SafeEvaluate (no double penalty)
    double fit = SafeEvaluate(indiv.GetGenotype());
    indiv.SetFitness(fit);
    candidates.push_back(indiv);
  }

  std::sort(candidates.begin(), candidates.end());

  for (int i = 0; i < remaining && i < (int)candidates.size(); ++i) {
    Individual &selected = candidates[i];
    int vnd_threshold = std::max(3, remaining / 5);
    if (i < vnd_threshold) {
      local_search_.RunVND(selected, 30, true, true, true, true);
    }
    double clean_fit = SafeEvaluate(selected);
    selected.SetFitness(clean_fit);
    new_pop.push_back(selected);
  }

  {
    std::lock_guard<std::mutex> lock(population_mutex_);
    population_ = new_pop;
    UpdateBiasedFitness();
  }

  // Reset adaptive operator success rates (fresh learning)
  if (IsExploitation()) {
    adapt_swap_.success_rate = 0.5;
    adapt_ejection_.success_rate = 0.5;
    adapt_swap3_.success_rate = 0.5;
    adapt_swap4_.success_rate = 0.5;
  }

  // IMMUNITY: 15 seconds of no migration after catastrophe
  immune_until_time_ =
      std::chrono::steady_clock::now() + std::chrono::seconds(15);
}

void Island::UpdateBiasedFitness() {
  int pop_size = static_cast<int>(population_.size());
  if (pop_size == 0)
    return;

  // Get current best for BPD comparison (only vs best, not entire elite)
  Individual best;
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    best = current_best_;
  }

  const std::vector<int> &perm = evaluator_->GetPermutation();
  int num_groups = evaluator_->GetNumGroups();

  double total_population_bpd = 0.0;
  int measurements_count = 0;

  // Calculate BPD only against the best individual (O(N) instead of O(N*E))
  // For large instances, sample instead of checking all
  int num_clients = evaluator_->GetSolutionSize();
  int sample_step = 1;
  if (num_clients > Config::HUGE_INSTANCE_THRESHOLD) {
    sample_step = std::max(1, pop_size / 10); // sample ~10 individuals
  } else if (num_clients > Config::LARGE_INSTANCE_THRESHOLD) {
    sample_step = std::max(1, pop_size / 20); // sample ~20 individuals
  }
  
  for (int i = 0; i < pop_size; ++i) {
    if (i % sample_step == 0) {
      int bpd = CalculateBrokenPairsDistance(population_[i], best, perm, num_groups);
      population_[i].SetDiversityScore(static_cast<double>(bpd));
      total_population_bpd += bpd;
      measurements_count++;
    } else {
      // Interpolate from previous sampled value
      int prev_sampled = (i / sample_step) * sample_step;
      population_[i].SetDiversityScore(population_[prev_sampled].GetDiversityScore());
    }
  }

  if (measurements_count > 0) {
    double avg_raw_bpd = total_population_bpd / measurements_count;
    double raw_diversity = avg_raw_bpd / (double)evaluator_->GetSolutionSize();
    double range = max_diversity_baseline_ - min_diversity_baseline_;
    if (range > 0.001) {
      current_structural_diversity_ =
          (raw_diversity - min_diversity_baseline_) / range;
      current_structural_diversity_ =
          std::max(0.0, std::min(1.0, current_structural_diversity_));
    } else {
      current_structural_diversity_ = 0.5;
    }
  } else {
    current_structural_diversity_ = 0.0;
  }

  // Sort by diversity for ranking
  std::vector<int> indices(pop_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return population_[a].GetDiversityScore() >
           population_[b].GetDiversityScore();
  });

  std::vector<int> rank_diversity(pop_size);
  for (int r = 0; r < pop_size; ++r)
    rank_diversity[indices[r]] = r;

  std::vector<int> rank_fitness(pop_size);
  std::vector<int> fit_indices(pop_size);
  std::iota(fit_indices.begin(), fit_indices.end(), 0);
  std::sort(fit_indices.begin(), fit_indices.end(), [&](int a, int b) {
    return population_[a].GetFitness() < population_[b].GetFitness();
  });
  for (int r = 0; r < pop_size; ++r)
    rank_fitness[fit_indices[r]] = r;

  // Biased fitness with fixed elite ratio (using 1/pop_size since comparing to
  // single best)
  double elite_ratio = 1.0 / (double)pop_size;
  for (int i = 0; i < pop_size; ++i) {
    double biased = (double)rank_fitness[i] +
                    (1.0 - elite_ratio) * (double)rank_diversity[i];
    population_[i].SetBiasedFitness(biased);
  }
}

int Island::CalculateBrokenPairsDistance(const Individual &ind1,
                                         const Individual &ind2,
                                         const std::vector<int> &permutation,
                                         int num_groups) {
  const std::vector<int> &g1 = ind1.GetGenotype();
  const std::vector<int> &g2 = ind2.GetGenotype();
  int size = static_cast<int>(g1.size());
  if (size == 0)
    return 0;

  std::fill(last_in_group1.begin(), last_in_group1.end(), -1);
  std::fill(last_in_group2.begin(), last_in_group2.end(), -1);

  for (int customer_id : permutation) {
    int idx = customer_id - 2;
    if (idx < 0 || idx >= size)
      continue;

    int group1 = g1[idx];
    if (group1 >= 0 && group1 < num_groups) {
      pred1[idx] = last_in_group1[group1];
      last_in_group1[group1] = idx;
    } else {
      pred1[idx] = -2;
    }

    int group2 = g2[idx];
    if (group2 >= 0 && group2 < num_groups) {
      pred2[idx] = last_in_group2[group2];
      last_in_group2[group2] = idx;
    } else {
      pred2[idx] = -2;
    }
  }

  int distance = 0;
  for (int i = 0; i < size; ++i) {
    if (pred1[i] != pred2[i])
      distance++;
  }
  return distance;
}

// CalculatePopulationCV() - REMOVED: Dead code, current_structural_diversity_
// is now correctly calculated via BPD in UpdateBiasedFitness()

int Island::ApplyLoadBalancing(Individual &child) {
  std::uniform_real_distribution<double> d(0.0, 1.0);
  double rnd = d(rng_);
  int executed_op = -1;

  if (rnd < 0.1) {
    if (ApplyLoadBalancingChainMutation(child)) {
      // Success
    }
#ifdef RESEARCH
    executed_op = (int)OpType::LB_CHAIN;
#endif
  } else if (rnd < 0.45) {
    if (ApplyLoadBalancingSwapMutation(child)) {
      // Success
    }
#ifdef RESEARCH
    executed_op = (int)OpType::LB_SWAP;
#endif
  } else {
    if (ApplyLoadBalancingSimple(child)) {
      // Success
    }
#ifdef RESEARCH
    executed_op = (int)OpType::LB_SIMPLE;
#endif
  }
  return executed_op;
}

bool Island::ApplyLoadBalancingChainMutation(Individual &individual) {
  std::vector<int> &solution = individual.AccessGenotype();
  int num_groups = evaluator_->GetNumGroups();
  int capacity = evaluator_->GetCapacity();
  const std::vector<int> &demands = evaluator_->GetDemands();
  const std::vector<int> &perm = evaluator_->GetPermutation();

  std::vector<int> loads(num_groups, 0);
  std::vector<std::vector<int>> group_clients(num_groups);

  for (int customer_id : perm) {
    int sol_idx = customer_id - 2;
    if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
      int g = solution[sol_idx];
      if (g >= 0 && g < num_groups) {
        loads[g] += demands[customer_id - 1];
        group_clients[g].push_back(customer_id);
      }
    }
  }

  std::vector<int> overloaded_groups;
  for (int g = 0; g < num_groups; ++g) {
    if (loads[g] > capacity)
      overloaded_groups.push_back(g);
  }
  if (overloaded_groups.empty())
    return false;

  std::shuffle(overloaded_groups.begin(), overloaded_groups.end(), rng_);
  bool any_change = false;
  const int MAX_DEPTH = 10;

  for (int start_group : overloaded_groups) {
    if (loads[start_group] <= capacity)
      continue;

    std::vector<ChainMove> chain_history;
    std::vector<bool> group_visited(num_groups, false);
    int current_group = start_group;
    group_visited[current_group] = true;
    bool chain_success = false;

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
      auto move =
          FindNextChainMove(current_group, group_visited, loads, group_clients);
      int best_client = move.first;
      int best_target = move.second;

      if (best_target == -1)
        break;

      int demand = demands[best_client - 1];
      int sol_idx = best_client - 2;

      if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
        solution[sol_idx] = best_target;
        loads[current_group] -= demand;
        loads[best_target] += demand;

        auto it = std::find(group_clients[current_group].begin(),
                            group_clients[current_group].end(), best_client);
        if (it != group_clients[current_group].end()) {
          *it = group_clients[current_group].back();
          group_clients[current_group].pop_back();
          group_clients[best_target].push_back(best_client);
        }

        chain_history.push_back(
            {best_client, current_group, best_target, demand});
        group_visited[best_target] = true;
      }

      bool all_ok = true;
      for (const auto &m : chain_history) {
        if (loads[m.from_group] > capacity)
          all_ok = false;
      }
      if (all_ok && loads[current_group] <= capacity) {
        chain_success = true;
        break;
      }
      current_group = best_target;
    }

    if (!chain_success) {
      // Revert
      for (int i = (int)chain_history.size() - 1; i >= 0; --i) {
        const auto &m = chain_history[i];
        int sol_idx = m.customer_id - 2;
        if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
          solution[sol_idx] = m.from_group;
          loads[m.from_group] += m.demand;
          loads[m.to_group] -= m.demand;
        }
      }
    } else {
      any_change = true;
    }
  }
  return any_change;
}

std::pair<int, int>
Island::FindNextChainMove(int group_idx, const std::vector<bool> &visited,
                          const std::vector<int> &loads,
                          const std::vector<std::vector<int>> &group_clients) {
  if (group_clients[group_idx].empty())
    return {-1, -1};
  int num_groups = static_cast<int>(loads.size());
  int capacity = evaluator_->GetCapacity();
  const std::vector<int> &demands = evaluator_->GetDemands();

  // Strategy 1: Find valid move to non-visited group
  std::vector<int> candidates = group_clients[group_idx];
  std::shuffle(candidates.begin(), candidates.end(), rng_);
  int trials = std::min((int)candidates.size(), 10);

  for (int i = 0; i < trials; ++i) {
    int client_id = candidates[i];
    int demand = demands[client_id - 1];
    for (int g = 0; g < num_groups; ++g) {
      if (g == group_idx || visited[g])
        continue;
      if (loads[g] + demand <= capacity)
        return {client_id, g};
    }
  }

  // Strategy 2: Force move to explore
  for (int i = 0; i < trials; ++i) {
    int client_id = candidates[i];
    int target_candidate = rng_() % num_groups;
    for (int k = 0; k < 5; k++) {
      int g = (target_candidate + k) % num_groups;
      if (g != group_idx && !visited[g])
        return {client_id, g};
    }
  }
  return {-1, -1};
}

bool Island::ApplyLoadBalancingSwapMutation(Individual &individual) {
  std::vector<int> &solution = individual.AccessGenotype();
  int num_groups = evaluator_->GetNumGroups();
  int capacity = evaluator_->GetCapacity();
  const std::vector<int> &demands = evaluator_->GetDemands();
  const std::vector<int> &perm = evaluator_->GetPermutation();

  std::vector<int> loads(num_groups, 0);
  std::vector<std::vector<int>> group_clients(num_groups);
  for (int customer_id : perm) {
    int sol_idx = customer_id - 2;
    if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
      int g = solution[sol_idx];
      if (g >= 0 && g < num_groups) {
        loads[g] += demands[customer_id - 1];
        group_clients[g].push_back(customer_id);
      }
    }
  }

  std::vector<int> overloaded_groups;
  for (int g = 0; g < num_groups; ++g) {
    if (loads[g] > capacity)
      overloaded_groups.push_back(g);
  }
  if (overloaded_groups.empty())
    return false;
  std::shuffle(overloaded_groups.begin(), overloaded_groups.end(), rng_);

  bool changed = false;
  for (int source_group : overloaded_groups) {
    if (loads[source_group] <= capacity)
      continue;
    std::vector<int> target_groups(num_groups);
    std::iota(target_groups.begin(), target_groups.end(), 0);
    std::shuffle(target_groups.begin(), target_groups.end(), rng_);

    bool fixed = false;
    for (int target_group : target_groups) {
      if (target_group == source_group)
        continue;
      if (fixed)
        break;

      std::shuffle(group_clients[source_group].begin(),
                   group_clients[source_group].end(), rng_);
      std::shuffle(group_clients[target_group].begin(),
                   group_clients[target_group].end(), rng_);

      for (int client_a : group_clients[source_group]) {
        int demand_a = demands[client_a - 1];
        for (int client_b : group_clients[target_group]) {
          int demand_b = demands[client_b - 1];
          if (demand_a <= demand_b)
            continue; // Only swap if reducing load

          if (loads[target_group] - demand_b + demand_a <= capacity) {
            int idx_a = client_a - 2;
            int idx_b = client_b - 2;
            if (idx_a >= 0 && idx_b >= 0) {
              solution[idx_a] = target_group;
              solution[idx_b] = source_group;
              loads[source_group] = loads[source_group] - demand_a + demand_b;
              loads[target_group] = loads[target_group] - demand_b + demand_a;
              changed = true;
              if (loads[source_group] <= capacity)
                fixed = true;
              goto next_target;
            }
          }
        }
      }
    next_target:;
    }
  }
  return changed;
}

bool Island::ApplyLoadBalancingSimple(Individual &individual) {
  std::vector<int> &solution = individual.AccessGenotype();
  int num_groups = evaluator_->GetNumGroups();
  int capacity = evaluator_->GetCapacity();
  const std::vector<int> &demands = evaluator_->GetDemands();

  std::vector<int> loads(num_groups, 0);
  for (size_t i = 0; i < solution.size(); ++i) {
    int g = solution[i];
    int cust_id = i + 2; // Approximate ID mapping if aligned
    // Correct mapping via Permutation is better but costly here, assumes
    // simplistic index But let's use proper mapping if possible. Actually the
    // caller logic used exact permutation loop. For simplicity, let's use the
    // permutation-based loop for load calculation BUT `solution` is indexed by
    // (customer_id - 2).
  }
  // Re-calculating loads properly:
  const std::vector<int> &perm = evaluator_->GetPermutation();
  for (int customer_id : perm) {
    int sol_idx = customer_id - 2;
    if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
      int g = solution[sol_idx];
      if (g >= 0 && g < num_groups)
        loads[g] += demands[customer_id - 1];
    }
  }

  std::vector<int> overloaded;
  std::vector<int> underloaded;
  for (int g = 0; g < num_groups; ++g) {
    if (loads[g] > capacity)
      overloaded.push_back(g);
    else
      underloaded.push_back(g);
  }
  if (overloaded.empty() || underloaded.empty())
    return false;

  std::shuffle(overloaded.begin(), overloaded.end(), rng_);
  bool changed = false;

  for (int source : overloaded) {
    if (loads[source] <= capacity)
      continue;
    // Find clients in this group
    std::vector<int> clients;
    for (int cust_id : perm) {
      int idx = cust_id - 2;
      if (idx >= 0 && solution[idx] == source)
        clients.push_back(cust_id);
    }
    std::shuffle(clients.begin(), clients.end(), rng_);

    for (int cust : clients) {
      int demand = demands[cust - 1];
      // Search for target
      std::shuffle(underloaded.begin(), underloaded.end(), rng_);
      for (int target : underloaded) {
        if (loads[target] + demand <= capacity) {
          int idx = cust - 2;
          solution[idx] = target;
          loads[source] -= demand;
          loads[target] += demand;
          changed = true;
          break;
        }
      }
      if (loads[source] <= capacity)
        break;
    }
  }
  return changed;
}

Individual Island::CrossoverNeighborBased(const Individual &p1,
                                          const Individual &p2) {
  const std::vector<int> &g1 = p1.GetGenotype();
  const std::vector<int> &g2 = p2.GetGenotype();
  int size = static_cast<int>(g1.size());
  Individual child(size);
  std::vector<int> &child_genes = child.AccessGenotype();

  if (size == 0 || !geometry_.HasNeighbors()) {
    // Fallback to uniform crossover
    for (int i = 0; i < size; ++i) {
      child_genes[i] = (rng_() % 2 == 0) ? g1[i] : g2[i];
    }
    return child;
  }

  // Pick a random center client
  std::uniform_int_distribution<int> dist_idx(0, size - 1);
  int center_idx = dist_idx(rng_);

  // Get center's neighbors - these come from p1
  const auto &neighbors = geometry_.GetNeighbors(center_idx);
  std::unordered_set<int> neighbor_set(neighbors.begin(), neighbors.end());
  neighbor_set.insert(center_idx); // include center itself

  // Clients in neighbor set take genes from p1, others from p2
  for (int i = 0; i < size; ++i) {
    child_genes[i] = (neighbor_set.count(i) > 0) ? g1[i] : g2[i];
  }
  return child;
}

// CrossoverSequence() - REMOVED: Dead code, replaced by ApplySREX()

// CrossoverUniform() - REMOVED: Dead code, never called

Individual Island::Crossover(const Individual &p1, const Individual &p2) {
#ifdef RESEARCH
  crossovers++;
#endif
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r = dist(rng_);
  double sequence_prob;

  if (id_ <= 1)
    sequence_prob = 0.31;
  else if (id_ <= 3)
    sequence_prob = 0.4;
  else
    sequence_prob = 0.6;

  if (r < sequence_prob)
    return ApplySREX(p1, p2);
  else
    return CrossoverNeighborBased(p1, p2);
}

void Island::PrintIndividual(const Individual &individual,
                             int global_generation) const {
  int num_groups = evaluator_->GetNumGroups();
  vector<int> group_counts(num_groups, 0);
  const vector<int> &genes = individual.GetGenotype();
  for (int g : genes)
    if (g >= 0 && g < num_groups)
      group_counts[g]++;

  int extra_returns = evaluator_->GetTotalDepotReturns(genes);
  cout << "   [Island " << id_ << "] Gen: " << setw(6) << global_generation
       << " | Dist: " << fixed << setprecision(2) << individual.GetFitness()
       << " | Ret: " << extra_returns << " | Groups: [";
  for (size_t i = 0; i < group_counts.size(); ++i)
    cout << group_counts[i] << (i < group_counts.size() - 1 ? "," : "");
  cout << "] Div: " << std::scientific << std::setprecision(2)
       << current_structural_diversity_ << std::defaultfloat << "\n";
}

void Island::ApplySplitToIndividual(Individual &indiv) {
  const std::vector<int> &global_perm = evaluator_->GetPermutation();
  int fleet_limit = evaluator_->GetNumGroups();
  SplitResult result = split_.RunLinear(global_perm);

  if (result.feasible) {
    std::vector<int> &genes = indiv.AccessGenotype();
    if (result.group_assignment.size() != genes.size())
      return;

    int routes_count = static_cast<int>(result.optimized_routes.size());
    for (size_t i = 0; i < genes.size(); ++i) {
      int assigned_route_id = result.group_assignment[i];
      genes[i] = (assigned_route_id < fleet_limit)
                     ? assigned_route_id
                     : (assigned_route_id % fleet_limit);
    }

    int excess_vehicles =
        (routes_count > fleet_limit) ? (routes_count - fleet_limit) : 0;
    indiv.SetFitness(result.total_cost);
    indiv.SetReturnCount(excess_vehicles);
  } else {
    indiv.SetFitness(1.0e30);
  }
}

// UpdateAdaptiveParameters: Tuned based on Island Type and Calibrated
// Structural Diversity
void Island::UpdateAdaptiveParameters() {
  double relative_div = 0.0;
  // Use calibrate baselines to normalize diversity
  if (max_diversity_baseline_ > 1e-9) {
    relative_div = (current_structural_diversity_ - min_diversity_baseline_) /
                   (max_diversity_baseline_ - min_diversity_baseline_);
  }
  relative_div = std::max(0.0, std::min(1.0, relative_div));

  // Defines how "chaotic" the population is relative to our baselines.
  // 0.0 = fully converged state, 1.0 = fully random state
  double chaos = current_structural_diversity_;

  // Base mutation probability based on exploration/exploitation role
  double base_mut_prob = IsExploration() ? Config::EXPLORATION_MUTATION_PROB
                                         : Config::EXPLOITATION_MUTATION_PROB;

  // Dynamic mutation probability adjustment
  // If chaos is low (converged), increase mutation to escape
  // If chaos is high, decrease mutation to exploit
  double dynamic_mut_prob = base_mut_prob +
                            (Config::ADAPTIVE_CHAOS_BOOST * (1.0 - chaos)) -
                            (Config::ADAPTIVE_CHAOS_PENALTY * chaos);

  // Per-exploration-island mutation ranges for differentiation
  double mut_min, mut_max;
  if (IsExploration()) {
    switch (id_) {
    case 0:
      mut_min = Config::EXPLORE_I0_MUT_MIN;
      mut_max = Config::EXPLORE_I0_MUT_MAX;
      break; // 20-60%
    case 2:
      mut_min = Config::EXPLORE_I2_MUT_MIN;
      mut_max = Config::EXPLORE_I2_MUT_MAX;
      break; // 30-65%
    case 4:
      mut_min = Config::EXPLORE_I4_MUT_MIN;
      mut_max = Config::EXPLORE_I4_MUT_MAX;
      break; // 40-70%
    default:
      mut_min = Config::ADAPTIVE_MUT_MIN;
      mut_max = Config::ADAPTIVE_MUT_MAX;
      break;
    }
  } else {
    mut_min = Config::ADAPTIVE_MUT_MIN;
    mut_max = Config::ADAPTIVE_MUT_MAX;
  }
  dynamic_mut_prob = std::max(mut_min, std::min(mut_max, dynamic_mut_prob));

  // Set member probabilities based on role and state
  p_microsplit_ = dynamic_mut_prob;
  p_mutation_ = dynamic_mut_prob * 0.8;
  p_loadbalance_ = dynamic_mut_prob * 0.5;
  p_retminimizer_ = dynamic_mut_prob * 0.6;
  p_mergesplit_ = dynamic_mut_prob * 0.5;

  if (IsExploitation()) {
    p_swap3_ = Config::EXPLOITATION_P_SWAP3;
    p_swap4_ = Config::EXPLOITATION_P_SWAP4;
    p_microsplit_ =
        std::max(Config::EXPLOITATION_MIN_MICROSPLIT, p_microsplit_); // min 20%

    // Fine-tuning for exploitation
    if (chaos < 0.1) {       // Very converged
      p_loadbalance_ *= 1.2; // Increase repair attempts
      p_retminimizer_ *= 1.2;
    }
  } else {
    // Exploration
    p_swap3_ = Config::EXPLORATION_P_SWAP3;
    p_swap4_ = Config::EXPLORATION_P_SWAP4; // rarely use expensive swaps in
                                            // exploration
  }

  // Legacy variables update (if still used elsewhere, though mostly replaced
  // now)
  adaptive_mutation_rate_ = dynamic_mut_prob;
  adaptive_ruin_chance_ = dynamic_mut_prob; // Simplified mapping

  if (IsExploration()) {
    adaptive_vnd_prob_ = MapRange(relative_div, 0.0, 1.0, 0.10, 0.40);
  } else {
    adaptive_vnd_prob_ = MapRange(relative_div, 0.0, 1.0, 0.50, 0.95);
  }
}

// ApplySuccessionAdaptive: Determines 'Elite' ratio based on structure
void Island::ApplySuccessionAdaptive(std::vector<Individual> &offspring_pool) {
  std::lock_guard<std::mutex> lock(population_mutex_);

  if (!offspring_pool.empty()) {
    population_.reserve(population_.size() + offspring_pool.size());
    for (auto &child : offspring_pool) {
      population_.push_back(std::move(child));
    }
  }
  if (population_.empty())
    return;

  // Deduplication
  std::sort(population_.begin(), population_.end(),
            [](const Individual &a, const Individual &b) {
              return a.GetFitness() < b.GetFitness();
            });

  std::vector<Individual> unique_candidates;
  unique_candidates.reserve(population_.size());
  std::unordered_set<uint64_t> used_hashes;

  for (const auto &ind : population_) {
    uint64_t h = HashGenotype64(ind.GetGenotype());
    if (used_hashes.find(h) == used_hashes.end()) {
      used_hashes.insert(h);
      unique_candidates.push_back(
          ind); // Copy needed as we iterate const ref but push back
    }
  }
  population_ = std::move(unique_candidates);

  if ((int)population_.size() <= population_size_) {
    UpdateBiasedFitness();
    return;
  }

  // Calculate Dynamic Elite Ratio based on Structural Diversity
  double relative_div = 0.0;
  if (max_diversity_baseline_ > 1e-9) {
    relative_div = (current_structural_diversity_ - min_diversity_baseline_) /
                   (max_diversity_baseline_ - min_diversity_baseline_);
  }
  relative_div = std::max(0.0, std::min(1.0, relative_div));

  // Determine split based on island type:
  // Explorer: Always maintain high diversity (even when chaotic)
  // Exploiter: Focus on elitism, allow more convergence

  double elite_ratio;
  if (IsExploration()) {
    // Explorer: 10% elite at low div, 50% elite at high div
    elite_ratio =
        MapRange(relative_div, 0.0, 1.0, Config::ELITE_RATIO_EXPLORATION_LOW,
                 Config::ELITE_RATIO_EXPLORATION_HIGH);
  } else {
    // Exploiter: 30% elite at low div, 90% elite at high div
    elite_ratio =
        MapRange(relative_div, 0.0, 1.0, Config::ELITE_RATIO_EXPLOITATION_LOW,
                 Config::ELITE_RATIO_EXPLOITATION_HIGH);
  }

  // Apply selection
  int elite_count = (int)(population_size_ * elite_ratio);
  elite_count = std::max(2, elite_count); // Always keep at least top 2

  std::vector<Individual> next_pop;
  next_pop.reserve(population_size_);
  std::unordered_set<int> taken_indices;

  // 1. Take Elites (Pure Fitness)
  for (int i = 0; i < (int)population_.size(); ++i) {
    if ((int)next_pop.size() >= elite_count)
      break;
    next_pop.push_back(population_[i]);
    taken_indices.insert(i);
  }

  // 2. Take Diverse (Biased Fitness)
  if ((int)next_pop.size() < population_size_) {
    UpdateBiasedFitness(); // Recalculate diversity ranks on FULL pool

    std::vector<int> biased_indices(population_.size());
    std::iota(biased_indices.begin(), biased_indices.end(), 0);
    std::sort(biased_indices.begin(), biased_indices.end(), [&](int a, int b) {
      return population_[a].GetBiasedFitness() <
             population_[b].GetBiasedFitness();
    });

    for (int idx : biased_indices) {
      if ((int)next_pop.size() >= population_size_)
        break;
      if (taken_indices.find(idx) == taken_indices.end()) {
        next_pop.push_back(population_[idx]);
        taken_indices.insert(idx);
      }
    }
  }

  // 3. Fallback (if still needed)
  if ((int)next_pop.size() < population_size_) {
    for (int i = 0; i < (int)population_.size(); ++i) {
      if ((int)next_pop.size() >= population_size_)
        break;
      if (taken_indices.find(i) == taken_indices.end()) {
        next_pop.push_back(population_[i]);
      }
    }
  }

  population_ = std::move(next_pop);

  // Sort final population by fitness
  std::sort(population_.begin(), population_.end(),
            [](const Individual &a, const Individual &b) {
              return a.GetFitness() < b.GetFitness();
            });

  UpdateBiasedFitness(); // Final update for next gen stats
}

#ifdef RESEARCH
std::vector<int> Island::CanonicalizeGenotype(const std::vector<int> &genotype,
                                              int num_groups) const {
  if (genotype.empty())
    return {};
  std::vector<int> canonical = genotype;
  std::vector<int> mapping(num_groups + 1, -1);
  int next_new_id = 0;
  for (size_t i = 0; i < canonical.size(); ++i) {
    int old_id = canonical[i];
    if (old_id < 0 || old_id >= (int)mapping.size())
      continue;
    if (mapping[old_id] == -1)
      mapping[old_id] = next_new_id++;
    canonical[i] = mapping[old_id];
  }
  return canonical;
}

void Island::InitStats() {
  op_stats_.resize((int)OpType::COUNT);
  op_stats_[(int)OpType::CROSSOVER] = {"Cross", 0, 0};
  op_stats_[(int)OpType::MUT_AGGRESSIVE] = {"Mut_Aggro", 0, 0};
  op_stats_[(int)OpType::MUT_SPATIAL] = {"Mut_Spatial", 0, 0};
  op_stats_[(int)OpType::MUT_SIMPLE] = {"Mut_Simple", 0, 0};
  op_stats_[(int)OpType::LB_CHAIN] = {"LB_Chain", 0, 0};
  op_stats_[(int)OpType::LB_SWAP] = {"LB_Swap", 0, 0};
  op_stats_[(int)OpType::LB_SIMPLE] = {"LB_Simple", 0, 0};
  op_stats_[(int)OpType::VND] = {"VND", 0, 0};
}

void Island::ExportState(int generation, bool is_catastrophe) const {
  std::ofstream hist("history.csv", std::ios::app);
  if (hist.is_open()) {
    size_t unique_sols = local_cache_.GetSize();
    hist << generation << "," << current_best_.GetFitness() << ","
         << (is_catastrophe ? 1 : 0) << "," << current_structural_diversity_
         << "," << adaptive_mutation_rate_ << "," << adaptive_vnd_prob_ << ","
         << adaptive_ruin_chance_ << "," << total_evaluations << ","
         << local_cache_hits << "," << unique_sols << std::endl;
  }
}
#endif

int Island::SelectParentIndex() {
  if (population_.empty())
    return -1;
  int pop_size = static_cast<int>(population_.size());
  std::uniform_int_distribution<int> dist(0, pop_size - 1);
  double best_val = std::numeric_limits<double>::max();
  int best_idx = -1;

  int tournament_size = IsExploration() ? Config::EXPLORATION_TOURNAMENT_SIZE
                                        : Config::EXPLOITATION_TOURNAMENT_SIZE;
  int t_size = std::min(tournament_size, pop_size);

  for (int i = 0; i < t_size; ++i) {
    int idx = dist(rng_);
    if (population_[idx].GetBiasedFitness() < best_val) {
      best_val = population_[idx].GetBiasedFitness();
      best_idx = idx;
    }
  }
  return best_idx;
}

int Island::GetWorstBiasedIndex() const {
  if (population_.empty())
    return -1;

  std::vector<int> best_genotype_copy;
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    best_genotype_copy = current_best_.GetGenotype();
  }

  int pop_size = static_cast<int>(population_.size());
  int worst = -1;
  double max_val = -1.0;
  for (int i = 0; i < pop_size; ++i) {
    if (population_[i].GetGenotype() == best_genotype_copy)
      continue;
    if (population_[i].GetBiasedFitness() > max_val) {
      max_val = population_[i].GetBiasedFitness();
      worst = i;
    }
  }
  return (worst == -1) ? GetWorstIndex() : worst;
}

int Island::GetWorstIndex() const {
  if (population_.empty())
    return -1;
  int pop_size = static_cast<int>(population_.size());
  int idx = 0;
  double worst = -1.0;
  for (int i = 0; i < pop_size; ++i) {
    if (population_[i].GetFitness() > worst) {
      worst = population_[i].GetFitness();
      idx = i;
    }
  }
  return idx;
}

void Island::InjectImmigrant(Individual &imigrant) {
  auto now = std::chrono::steady_clock::now();

  // IMMUNITY CHECK: catastrophe recovery period
  if (now < immune_until_time_) {
    long long stagnation = current_generation_ - last_improvement_gen_;
    if (stagnation < 2000) {
      return; // island is immune after catastrophe
    }
  }

  // OPPORTUNITY CHECK: Check if migrant is significantly better (>1%) than current best
  double best_fit;
  {
      std::lock_guard<std::mutex> lock(best_mutex_);
      best_fit = current_best_.GetFitness();
  }
  
  // Evaluate if not already
  double fit;
  if (imigrant.IsEvaluated()) {
    fit = imigrant.GetFitness();
  } else {
    EvaluationResult res =
        evaluator_->EvaluateWithStats(imigrant.GetGenotype());
    fit = res.fitness;
    imigrant.SetFitness(fit);
    imigrant.SetReturnCount(res.returns);
  }

  if (fit == std::numeric_limits<double>::max())
    return;

  bool is_stuck = is_stuck_.load(std::memory_order_relaxed);
  bool is_opportunity = (fit < best_fit * 0.99); // 1% better

  // Default: Only accept migrants when island is stuck
  // NEW: Also accept if it's a huge opportunity
  if (!is_stuck && !is_opportunity) {
    return; // Island is making progress and this isn't a breakthrough, don't disturb!
  }

  std::lock_guard<std::mutex> lock(population_mutex_);

  // REPLACE MOST SIMILAR: Replace the individual most structurally similar
  // to the immigrant (smallest BPD), enabling better diversification
  int similar_idx = FindMostSimilarIndex(imigrant);
  if (similar_idx >= 0 && similar_idx < (int)population_.size()) {
    // Only replace if immigrant is better or similar has stagnated
    if (imigrant.GetFitness() < population_[similar_idx].GetFitness() ||
        population_[similar_idx].GetStagnation() > 100) {
      population_[similar_idx] = imigrant;

      // Check if new global best
      std::lock_guard<std::mutex> best_lock(best_mutex_);
      if (imigrant.GetFitness() < current_best_.GetFitness()) {
        current_best_ = imigrant;
        last_improvement_gen_ = current_generation_;
        last_improvement_time_ = now;
      }
    }
  }

  // === IMMEDIATE PATH RELINKING - DISABLED FOR PERFORMANCE ===
  // This ran a full VND+PR on every migrant, but PR is already used as crossover (60-80%)
  // Disabling saves ~10% compute while maintaining quality via regular PR crossover
  /*
  {
      Individual pr_runner;
      double best_fit;
      {
          std::lock_guard<std::mutex> best_lock(best_mutex_);
          pr_runner = current_best_;
          best_fit = current_best_.GetFitness();
      }

      // Only run PR if meaningful (don't run if solutions are identical)
      if (std::abs(pr_runner.GetFitness() - imigrant.GetFitness()) > 1e-4) {
          local_search_.SetGuideSolution(imigrant.GetGenotype());
          
          // Run VND on our copy. Because 'guide_solution_' is set, 
          // LocalSearch will attempt Path Relinking if VND stalls.
          // We use current_best_ as start point because it's locally optimal, 
          // so VND will likely trigger PR immediately.
          local_search_.RunVND(pr_runner);

          // If we found something better than BOTH, apply it
          std::lock_guard<std::mutex> best_lock(best_mutex_);
          if (pr_runner.GetFitness() < current_best_.GetFitness()) {
              std::cout << " [PR-MIG] Immediate Path Relinking SUCCESS! " 
                        << (int)current_best_.GetFitness() << " -> " 
                        << (int)pr_runner.GetFitness() << std::endl;
              current_best_ = pr_runner;
              last_improvement_gen_ = current_generation_;
              last_improvement_time_ = std::chrono::steady_clock::now();
          }
      }
  }
  */
}

void Island::MutateIndividual(Individual &indiv) {
  // Force a strong mutation regardless of current probability settings
  // This is used for migration sabotage/diversification
  std::uniform_real_distribution<double> d(0.0, 1.0);
  double r = d(rng_);

  if (r < 0.5) {
    mutator_.ApplyRuinRecreate(indiv, 0.3, false,
                               rng_); // 30% ruin (exploration)
  } else {
    mutator_.ApplySmartSpatialMove(indiv, rng_);
  }

  // Evaluate immediately to ensure consistency
  double fit = SafeEvaluate(indiv);
  indiv.SetFitness(fit);
}

Individual Island::GetRandomIndividual() {
  std::lock_guard<std::mutex> lock(population_mutex_);
  if (population_.empty())
    return current_best_;
  std::uniform_int_distribution<int> dist(0, (int)population_.size() - 1);
  return population_[dist(rng_)];
}

Individual Island::GetMostDiverseMigrantFor(const Individual &target_best) {
  std::lock_guard<std::mutex> lock(population_mutex_);
  if (population_.empty())
    return current_best_;

  const std::vector<int> &perm = evaluator_->GetPermutation();
  int num_groups = evaluator_->GetNumGroups();

  int best_idx = 0;
  int max_distance = -1;

  for (int i = 0; i < (int)population_.size(); ++i) {
    int distance = CalculateBrokenPairsDistance(population_[i], target_best,
                                                perm, num_groups);
    if (distance > max_distance) {
      max_distance = distance;
      best_idx = i;
    }
  }
  return population_[best_idx];
}

Individual Island::GetRandomEliteIndividual() {
  std::lock_guard<std::mutex> lock(population_mutex_);
  if (population_.empty())
    return current_best_;

  // Population should be sorted by fitness (best first)
  // Pick random from top 30%
  int elite_size = std::max(1, (int)(population_.size() * 0.30));
  std::uniform_int_distribution<int> dist(0, elite_size - 1);
  return population_[dist(rng_)];
}

int Island::FindMostSimilarIndex(const Individual &immigrant) const {
  // Find the individual most similar to immigrant (smallest BPD)
  // Called with population_mutex_ already held
  if (population_.empty())
    return -1;

  const std::vector<int> &perm = evaluator_->GetPermutation();
  int num_groups = evaluator_->GetNumGroups();

  int best_idx = -1;
  int min_bpd = INT_MAX;

  for (int i = 0; i < (int)population_.size(); ++i) {
    int bpd = const_cast<Island *>(this)->CalculateBrokenPairsDistance(
        population_[i], immigrant, perm, num_groups);
    if (bpd < min_bpd) {
      min_bpd = bpd;
      best_idx = i;
    }
  }
  return best_idx;
}

void Island::TryPullMigrant() {
  // Asynchronous migration: pull migrant from predecessor
  if (!ring_predecessor_)
    return;

  // Check migration interval
  if (current_generation_ - last_migration_gen_ < MIGRATION_INTERVAL)
    return;

  // === GENE MIGRATION (Route Segments) ===
  // Pull top 5 best routes from neighbor and inject into our pool
  // This enriches the gene pool for Frankenstein (Beam Search)
  if (IsExploitation()) { // Usually only exploitation islands use Frankenstein/RoutePool
      auto top_routes = ring_predecessor_->GetTopRoutes(5);
      if (!top_routes.empty()) {
          std::lock_guard<std::mutex> lock(best_mutex_);
          route_pool_.ImportRoutes(top_routes);
      }
  }

  // === TRICKLE MIGRATION (Forced Diversity) ===
  // 5% chance to pull a diverse migrant regardless of stuck status
  // This ensures constant "gene flow" to prevent isolation in local optima
  std::uniform_real_distribution<double> dist_trickle(0.0, 1.0);
  if (dist_trickle(rng_) < 0.05) { // 5% chance per generation
      Individual migrant = ring_predecessor_->GetMostDiverseMigrantFor(current_best_);
      if (migrant.GetFitness() < 1e14) { // Valid check
          // Inject directly (InjectImmigrant handles logic, but we want to force this as opportunity-like)
          // We can't force InjectImmigrant to accept, but diverse migrant usually helps.
          // Let's rely on standard InjectImmigrant logic for now, or bypass if needed.
          // Actually InjectImmigrant requires stuck OR opportunity.
          // We want this to be accepted. Let's make it look like an opportunity or modify InjectImmigrant?
          // Simpler: Just force inject into population if not present.
          
          bool accepted = false;
          {
             std::lock_guard<std::mutex> lock(population_mutex_);
             if (!ContainsSolution(migrant)) {
                 int victim = GetWorstBiasedIndex();
                 if (victim >= 0) {
                     population_[victim] = migrant;
                     accepted = true;
                 }
             }
          }
          if (accepted) {
             // std::cout << " [TRICKLE] Injected diverse migrant into I" << id_ << "\n";
          }
      }
  }


  // REMOVED early return: checks are now inside InjectImmigrant to support Opportunity Pulls
  // if (!is_stuck_.load(std::memory_order_relaxed)) return;

  // Get best for comparison
  Individual my_best;
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    my_best = current_best_;
  }

  // === HYBRID STRATEGY: CHASE vs DIVERSIFY ===
  // 1. Check neighbor's absolute BEST.
  Individual neighbor_best = ring_predecessor_->GetBestIndividual();
  
  Individual migrant;
  bool chase_mode = false;

  // If neighbor is significantly better (>1%), ignore diversity and CHASE the best solution.
  // This prevents the "Golden Solution Trap" where we ignore a breakthrough because it's "too similar".
  if (neighbor_best.GetFitness() < my_best.GetFitness() * 0.99) {
      migrant = neighbor_best;
      chase_mode = true;
      std::cout << "\033[32m [MIG I" << id_ << "] CHASE MODE! Pulling Golden Solution: " 
                << (int)neighbor_best.GetFitness() << " (My Best: " << (int)my_best.GetFitness() << ")\033[0m" << std::endl;
  } else {
      // 2. Otherwise, standard Diversity Pull to maintain population variety
      migrant = ring_predecessor_->GetMostDiverseMigrantFor(my_best);
  }

  // Check for opportunity logging (for debug)
  if (!chase_mode && migrant.GetFitness() < my_best.GetFitness() * 0.99) {
       // This implies diversity pull accidentally found something great
       std::cout << "\033[33m [MIG I" << id_ << "] Opportunity Pull (Diverse)! " 
                 << (int)migrant.GetFitness() << " < " << (int)my_best.GetFitness() << "\033[0m" << std::endl;
  }

  last_migration_gen_ = current_generation_;

  // Inject the migrant (InjectImmigrant handles logic: Stuck OR Opportunity)
  // AND performs Immediate Path Relinking
  InjectImmigrant(migrant);
}

void Island::CalibrateDiversity() {
  const int SAMPLE_SIZE = 100;
  int n = evaluator_->GetSolutionSize();
  int num_groups = evaluator_->GetNumGroups();
  const std::vector<int> &perm = evaluator_->GetPermutation();

  std::vector<Individual> random_samples;
  random_samples.reserve(SAMPLE_SIZE);

  for (int i = 0; i < SAMPLE_SIZE; ++i) {
    Individual rnd_ind(n);
    InitIndividual(rnd_ind, INITIALIZATION_TYPE::RANDOM);
    random_samples.push_back(std::move(rnd_ind));
  }

  double total_broken_pairs = 0.0;
  long long comparisons_count = 0;
  const int PROBES_PER_IND = 5;

  for (int i = 0; i < SAMPLE_SIZE; ++i) {
    for (int k = 0; k < PROBES_PER_IND; ++k) {
      int other_idx = rng_() % SAMPLE_SIZE;
      if (i == other_idx)
        continue;
      int dist = CalculateBrokenPairsDistance(
          random_samples[i], random_samples[other_idx], perm, num_groups);
      total_broken_pairs += dist;
      comparisons_count++;
    }
  }

  if (comparisons_count > 0) {
    double avg_dist = total_broken_pairs / (double)comparisons_count;
    max_diversity_baseline_ = avg_dist / (double)n;
  } else {
    max_diversity_baseline_ = 1.0;
  }
  CalibrateConvergence();
}

void Island::CalibrateConvergence() {
  const int SAMPLE_SIZE = 50;
  int n = evaluator_->GetSolutionSize();
  int num_groups = evaluator_->GetNumGroups();
  const std::vector<int> &perm = evaluator_->GetPermutation();

  Individual base_ind(n);
  InitIndividual(base_ind, INITIALIZATION_TYPE::CHUNKED);

  std::vector<Individual> converged_samples;
  converged_samples.reserve(SAMPLE_SIZE);
  converged_samples.push_back(base_ind);

  for (int i = 1; i < SAMPLE_SIZE; ++i) {
    Individual variant = base_ind;
    std::vector<int> &geno = variant.AccessGenotype();
    int num_mutations = 1 + (rng_() % 3);
    for (int m = 0; m < num_mutations; ++m) {
      int idx = rng_() % n;
      geno[idx] = rng_() % num_groups;
    }
    converged_samples.push_back(std::move(variant));
  }

  double total_broken_pairs = 0.0;
  long long comparisons_count = 0;
  const int PROBES_PER_IND = 5;

  for (int i = 0; i < SAMPLE_SIZE; ++i) {
    for (int k = 0; k < PROBES_PER_IND; ++k) {
      int other_idx = rng_() % SAMPLE_SIZE;
      if (i == other_idx)
        continue;
      int dist = CalculateBrokenPairsDistance(
          converged_samples[i], converged_samples[other_idx], perm, num_groups);
      total_broken_pairs += dist;
      comparisons_count++;
    }
  }

  if (comparisons_count > 0) {
    double avg_dist = total_broken_pairs / (double)comparisons_count;
    min_diversity_baseline_ = avg_dist / (double)n;
  } else {
    min_diversity_baseline_ = 0.0;
  }
}

double Island::MapRange(double value, double in_min, double in_max,
                        double out_min, double out_max) const {
  if (in_max - in_min < 1e-9)
    return out_min;
  double clamped = std::max(in_min, std::min(in_max, value));
  return out_min + (out_max - out_min) * (clamped - in_min) / (in_max - in_min);
}

int Island::GetVndIterations() const {
  int base_min = IsExploration() ? Config::EXPLORATION_VND_MIN
                                 : Config::EXPLOITATION_VND_MIN;
  int base_max = IsExploration() ? Config::EXPLORATION_VND_MAX
                                 : Config::EXPLOITATION_VND_MAX;

  // ADAPTIVE CAP: Scale VND iterations based on problem size
  int problem_size = evaluator_->GetSolutionSize();
  if (problem_size > Config::HUGE_INSTANCE_THRESHOLD) {
    // n > 3000: very aggressive reduction
    base_max = IsExploration() ? 1 : 8;
    base_min = 1;
  } else if (problem_size > Config::LARGE_INSTANCE_THRESHOLD) {
    // n > 1500: use large instance constants
    base_max = IsExploration() ? Config::EXPLORATION_VND_MAX_LARGE 
                               : Config::EXPLOITATION_VND_MAX_LARGE;
    base_min = std::min(base_min, 3);
  }

  double result =
      base_max - (current_structural_diversity_ * (base_max - base_min));
  return static_cast<int>(
      std::max((double)base_min, std::min((double)base_max, result)));
}

double Island::GetMutationRate() const {
  if (IsExploration()) {
    return MapRange(current_structural_diversity_, min_diversity_baseline_,
                    max_diversity_baseline_, 0.50, 0.20);
  } else {
    return MapRange(current_structural_diversity_, min_diversity_baseline_,
                    max_diversity_baseline_, 0.15, 0.05);
  }
}

double Island::GetRuinChance() const {
  if (IsExploration()) {
    return MapRange(current_structural_diversity_, min_diversity_baseline_,
                    max_diversity_baseline_, 0.50, 0.20);
  } else {
    return MapRange(current_structural_diversity_, min_diversity_baseline_,
                    max_diversity_baseline_, 0.15, 0.05);
  }
}

double Island::GetMicrosplitChance() const {
  if (IsExploration()) {
    return MapRange(current_structural_diversity_, min_diversity_baseline_,
                    max_diversity_baseline_, 0.65, 0.4);
  } else {
    return MapRange(current_structural_diversity_, min_diversity_baseline_,
                    max_diversity_baseline_, 0.3, 0.2);
  }
}

bool Island::TryGetStuckIndividual(Individual &out) {
  std::lock_guard<std::mutex> lock(population_mutex_);
  if (!stuck_queue_.empty()) {
    out = stuck_queue_.back();
    stuck_queue_.pop_back();
    return true;
  }
  for (auto &ind : population_) {
    if (ind.GetStagnation() > Config::STAGNATION_THRESHOLD) {
      out = ind;
      ind.ResetStagnation();
      return true;
    }
  }
  return false;
}

Individual Island::ApplySREX(const Individual &p1, const Individual &p2) {
  int num_clients = evaluator_->GetSolutionSize();
  const std::vector<int> &g1 = p1.GetGenotype();
  const std::vector<int> &g2 = p2.GetGenotype();

  if (g1.size() != num_clients || g2.size() != num_clients) {
    Individual rnd(num_clients);
    InitIndividual(rnd, INITIALIZATION_TYPE::RANDOM);
    return rnd;
  }

  std::vector<int> child_genotype(num_clients, -1);
  std::vector<bool> is_covered(num_clients, false);

  // Helper lambda to build route map
  auto build_routes = [&](const std::vector<int> &g) {
    int max_g = 0;
    for (int x : g)
      if (x > max_g)
        max_g = x;
    if (max_g > num_clients)
      max_g = num_clients;
    std::vector<std::vector<int>> routes(max_g + 1);
    for (int i = 0; i < num_clients; ++i) {
      if (g[i] >= 0 && g[i] <= max_g)
        routes[g[i]].push_back(i);
    }
    return routes;
  };

  auto routes1 = build_routes(g1);
  auto routes2 = build_routes(g2);

  std::vector<int> active1, active2;
  for (size_t i = 0; i < routes1.size(); ++i)
    if (!routes1[i].empty())
      active1.push_back((int)i);
  for (size_t i = 0; i < routes2.size(); ++i)
    if (!routes2[i].empty())
      active2.push_back((int)i);

  if (!active1.empty())
    std::shuffle(active1.begin(), active1.end(), rng_);
  if (!active2.empty())
    std::shuffle(active2.begin(), active2.end(), rng_);

  int current_child_group = 0;
  std::vector<int> child_group_loads;
  child_group_loads.reserve(num_clients);

  // inherit 50% from P1
  int take1 = std::max(1, (int)active1.size() / 2);
  for (int i = 0; i < take1 && i < (int)active1.size(); ++i) {
    int g_idx = active1[i];
    int load = 0;
    for (int client : routes1[g_idx]) {
      child_genotype[client] = current_child_group;
      is_covered[client] = true;
      load += evaluator_->GetDemand(client + 2); // ID mapping
    }
    child_group_loads.push_back(load);
    current_child_group++;
  }

  // inherit from P2 if no conflict
  for (int g_idx : active2) {
    bool conflict = false;
    for (int client : routes2[g_idx]) {
      if (is_covered[client]) {
        conflict = true;
        break;
      }
    }
    if (!conflict) {
      int load = 0;
      for (int client : routes2[g_idx]) {
        child_genotype[client] = current_child_group;
        is_covered[client] = true;
        load += evaluator_->GetDemand(client + 2);
      }
      child_group_loads.push_back(load);
      current_child_group++;
    }
  }

  // REGRET-3 REPAIR with EJECTION CHAIN: prioritize clients with fewer options
  // (biggest gap between best, 2nd, and 3rd-best insertion cost)
  std::vector<int> unassigned;
  for (int i = 0; i < num_clients; ++i)
    if (!is_covered[i])
      unassigned.push_back(i);

  // SAFETY: limit iterations to prevent infinite loops from ejection chain
  int max_iterations =
      num_clients * 3; // allow up to 3x client count iterations
  int iteration_count = 0;
  int ejection_count = 0;
  const int MAX_EJECTIONS = 10; // limit total ejections per SREX

  while (!unassigned.empty() && iteration_count < max_iterations) {
    iteration_count++;
    int best_client = -1;
    int best_group = -1;
    double max_regret = -1e30;

    for (int client : unassigned) {
      int demand = evaluator_->GetDemand(client + 2);

      // Find best, second-best, and third-best feasible groups
      double cost1 = 1e30, cost2 = 1e30,
             cost3 = 1e30; // best, 2nd, 3rd load fit
      int group1 = -1;

      for (size_t g = 0; g < child_group_loads.size(); ++g) {
        if (child_group_loads[g] + demand <= capacity_) {
          double load_ratio =
              (double)(child_group_loads[g] + demand) / capacity_;
          if (load_ratio < cost1) {
            cost3 = cost2;
            cost2 = cost1;
            cost1 = load_ratio;
            group1 = (int)g;
          } else if (load_ratio < cost2) {
            cost3 = cost2;
            cost2 = load_ratio;
          } else if (load_ratio < cost3) {
            cost3 = load_ratio;
          }
        }
      }

      // REGRET-3: sum of (2nd - 1st) + (3rd - 1st)
      // Higher regret = client has fewer good options -> insert first!
      double regret;
      if (group1 >= 0) {
        double r2 = (cost2 < 1e29) ? (cost2 - cost1) : 0.5;
        double r3 = (cost3 < 1e29) ? (cost3 - cost1) : 0.5;
        regret = r2 + r3; // Regret-3
        if (cost2 >= 1e29)
          regret = 1e10; // only ONE option - critical
      } else {
        regret = 1e20; // NO options - must create new group, highest priority
      }

      if (regret > max_regret) {
        max_regret = regret;
        best_client = client;
        best_group = group1;
      }
    }

    if (best_client < 0)
      break; // safety

    int demand = evaluator_->GetDemand(best_client + 2);

    if (best_group >= 0) {
      // Insert into best group
      child_genotype[best_client] = best_group;
      child_group_loads[best_group] += demand;
    } else {
      // NO FEASIBLE GROUP - simple fallback: create new group
      // (SREX Ejection Chain disabled - too complex, causes issues)
      child_genotype[best_client] = current_child_group;
      child_group_loads.push_back(demand);
      current_child_group++;
    }

    // Remove from unassigned
    unassigned.erase(
        std::remove(unassigned.begin(), unassigned.end(), best_client),
        unassigned.end());
  }

  Individual child(child_genotype);
  child.Canonicalize();
  return child;
}

// === MERGE-REGRET OPERATOR ===
// Dissolves 2 routes with most shared neighbors and repairs with Regret-3
// Designed for tight-capacity problems (Slack < 10)
bool Island::ApplyMergeRegret(Individual &ind) {
  std::vector<int> &genotype = ind.AccessGenotype();
  int num_clients = static_cast<int>(genotype.size());
  int num_groups = evaluator_->GetNumGroups();

  if (num_groups < 3)
    return false; // need at least 3 routes

  // Build route data: members and loads
  struct RouteInfo {
    int group_id;
    std::vector<int> clients;
    int load;
  };
  std::vector<RouteInfo> routes;
  routes.reserve(num_groups);

  for (int g = 0; g < num_groups; ++g) {
    RouteInfo ri;
    ri.group_id = g;
    ri.load = 0;

    for (int c = 0; c < num_clients; ++c) {
      if (genotype[c] == g) {
        ri.clients.push_back(c);
        int cid = c + 2; // customer ID (matrix index)
        ri.load += evaluator_->GetDemand(cid);
      }
    }

    if (!ri.clients.empty()) {
      routes.push_back(ri);
    }
  }

  if (routes.size() < 3)
    return false;

  // Find 2 routes with most shared neighbors (instead of centroid distance)
  int max_shared = -1;
  int best_i = 0, best_j = 1;

  for (size_t i = 0; i < routes.size(); ++i) {
    for (size_t j = i + 1; j < routes.size(); ++j) {
      int shared_count = 0;

      // Count how many clients in route i have neighbors in route j
      for (int client_i : routes[i].clients) {
        const auto &neighbors = geometry_.GetNeighbors(client_i);
        for (int neighbor : neighbors) {
          if (neighbor < num_clients &&
              genotype[neighbor] == routes[j].group_id) {
            shared_count++;
            break; // only count once per client
          }
        }
      }
      // Also count from route j to route i
      for (int client_j : routes[j].clients) {
        const auto &neighbors = geometry_.GetNeighbors(client_j);
        for (int neighbor : neighbors) {
          if (neighbor < num_clients &&
              genotype[neighbor] == routes[i].group_id) {
            shared_count++;
            break;
          }
        }
      }

      if (shared_count > max_shared) {
        max_shared = shared_count;
        best_i = i;
        best_j = j;
      }
    }
  }

  // Dissolve: collect clients from both routes
  std::vector<int> dissolved;
  for (int c : routes[best_i].clients) {
    dissolved.push_back(c);
    genotype[c] = -1; // mark unassigned
  }
  for (int c : routes[best_j].clients) {
    dissolved.push_back(c);
    genotype[c] = -1;
  }

  if (dissolved.empty())
    return false;

  // Build group loads for remaining routes
  std::vector<int> group_loads(num_groups, 0);
  for (int c = 0; c < num_clients; ++c) {
    if (genotype[c] >= 0 && genotype[c] < num_groups) {
      group_loads[genotype[c]] += evaluator_->GetDemand(c + 2);
    }
  }

  // Regret-3 repair (same logic as SREX)
  int max_iterations = num_clients * 2;
  int iteration = 0;

  while (!dissolved.empty() && iteration < max_iterations) {
    iteration++;

    int best_client = -1;
    int best_group = -1;
    double max_regret = -1e30;

    for (int client : dissolved) {
      int demand = evaluator_->GetDemand(client + 2);

      // Find best, 2nd, 3rd feasible groups
      double cost1 = 1e30, cost2 = 1e30, cost3 = 1e30;
      int group1 = -1;

      for (int g = 0; g < num_groups; ++g) {
        if (group_loads[g] + demand <= capacity_) {
          double ratio = (double)(group_loads[g] + demand) / capacity_;
          if (ratio < cost1) {
            cost3 = cost2;
            cost2 = cost1;
            cost1 = ratio;
            group1 = g;
          } else if (ratio < cost2) {
            cost3 = cost2;
            cost2 = ratio;
          } else if (ratio < cost3) {
            cost3 = ratio;
          }
        }
      }

      // Regret-3: sum of differences
      double regret;
      if (group1 >= 0) {
        double r2 = (cost2 < 1e29) ? (cost2 - cost1) : 0.5;
        double r3 = (cost3 < 1e29) ? (cost3 - cost1) : 0.5;
        regret = r2 + r3;
        if (cost2 >= 1e29)
          regret = 1e10; // only one option
      } else {
        regret = 1e20; // no feasible group
      }

      if (regret > max_regret) {
        max_regret = regret;
        best_client = client;
        best_group = group1;
      }
    }

    if (best_client < 0)
      break;

    int demand = evaluator_->GetDemand(best_client + 2);

    if (best_group >= 0) {
      genotype[best_client] = best_group;
      group_loads[best_group] += demand;
    } else {
      // Find group with lowest load (force insert)
      int min_load_g = 0;
      for (int g = 1; g < num_groups; ++g) {
        if (group_loads[g] < group_loads[min_load_g])
          min_load_g = g;
      }
      genotype[best_client] = min_load_g;
      group_loads[min_load_g] += demand;
    }

    // Remove from dissolved
    dissolved.erase(
        std::remove(dissolved.begin(), dissolved.end(), best_client),
        dissolved.end());
  }

  ind.Canonicalize();
  return true;
}


std::vector<CachedRoute> Island::GetTopRoutes(int n) const {
    std::lock_guard<std::mutex> lock(best_mutex_);
    // Note: RoutePool handles its own locking, but we hold best_mutex_ to access route_pool_ object safely if needed? 
    // Actually route_pool_ is a member, accessing it is safe if thread-safe.
    // But let's just delegate.
    return route_pool_.GetBestRoutes(n);
}


void Island::ProcessBroadcastBuffer() {
  std::vector<Individual> processing_queue;
  {
    std::lock_guard<std::mutex> lock(broadcast_mutex_);
    if (broadcast_buffer_.empty()) return;
    processing_queue = std::move(broadcast_buffer_);
    broadcast_buffer_.clear();
  }

  // Process in main thread - SAFE to use BPD buffers and InjectImmigrant
  double my_fitness = GetBestFitness();

  for (auto& candidate : processing_queue) {
    double cand_fit = candidate.GetFitness();
    
    // Skip if significantly worse (>5%)
    if (cand_fit > my_fitness * 1.05) continue;
    
    // Check if from EXPLORE
    int home = candidate.GetHomeIsland();
    bool from_explore = (home == 0 || home == 2 || home == 4);
    
    // === STRATEGY: EXPLOIT keeps its own trajectory, just injects for diversity ===
    if (from_explore && IsExploitation()) {
      // DON'T replace current_best - EXPLOIT keeps its own search trajectory!
      // Just inject broadcast into population for diversity, let natural selection work
      if (!ContainsSolution(candidate) && cand_fit < my_fitness * 1.05) {
        InjectImmigrant(candidate);
        // Reduce log spam - only log if significantly better
        if (cand_fit < my_fitness * 0.99) {
          std::cout << "\033[36m [I" << id_ << " EXPLOIT] Injected diversity from I" << home 
                    << " (fit=" << std::fixed << std::setprecision(0) << cand_fit << ")\033[0m\n";
        }
      }
    } else {
      // EXPLOIT-to-EXPLOIT: Original logic with filters
      if (cand_fit >= my_fitness) continue;
      
      double fitness_gap = (my_fitness - cand_fit) / my_fitness;
      bool significantly_better = (fitness_gap > 0.01);
      
      bool different_enough = false;
      if (!significantly_better) {
        int bpd = CalculateBrokenPairsDistancePublic(current_best_, candidate);
        int threshold = static_cast<int>(evaluator_->GetSolutionSize() * 0.10);
        if (bpd > threshold) {
          different_enough = true;
        }
      }

      if (significantly_better || different_enough) {
        if (significantly_better) {
          std::cout << "\033[95m [BROADCAST I" << id_ << "] Accepting SUPERIOR broadcast from I" 
                    << candidate.GetHomeIsland() << " (Gap: " << std::fixed << std::setprecision(2) 
                    << (fitness_gap*100.0) << "%)\033[0m\n";
        }
        InjectImmigrant(candidate);
      }
    }
  }
}

void Island::ReceiveBroadcastBest(const Individual& best) {
  // Thread-safe buffering of broadcast
  std::lock_guard<std::mutex> lock(broadcast_mutex_);
  
  // Create a copy and mark as non-native
  Individual imported = best;
  imported.SetNative(false);
  
  // Just buffer it - no logic here to keep mutex time minimal
  broadcast_buffer_.push_back(std::move(imported));
}

void Island::UpdateAdaptiveProbabilities() {
  // === EPSILON-GREEDY: Update success rates ===
  // Called every diagnostic interval
  
  auto update_rate = [this](AdaptiveOperator& op) {
    if (op.calls > 0) {
      double current_rate = static_cast<double>(op.wins) / op.calls;
      // EMA update of success rate
      op.success_rate = ADAPT_ALPHA * current_rate + (1.0 - ADAPT_ALPHA) * op.success_rate;
      // Reset counters for next window
      op.calls = 0;
      op.wins = 0;
    }
  };
  
  update_rate(adapt_swap_);
  update_rate(adapt_ejection_);
  update_rate(adapt_swap3_);
  update_rate(adapt_swap4_);
}

int Island::SelectAdaptiveOperator() {
  // EPSILON-GREEDY selection for 4 operators
  // 90% exploit: pick best operator (highest success_rate)
  // 10% explore: random operator
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r = dist(rng_);
  
  if (r < ADAPT_EPSILON) {
    // Exploration: random operator among 4
    std::uniform_int_distribution<int> op_dist(0, 3);
    return op_dist(rng_);
  } else {
    // Exploitation: pick best among 4
    double rates[4] = {adapt_swap_.success_rate, adapt_ejection_.success_rate,
                       adapt_swap3_.success_rate, adapt_swap4_.success_rate};
    int best = 0;
    for (int i = 1; i < 4; i++) {
      if (rates[i] > rates[best]) best = i;
    }
    return best;
  }
}

// === CACHE HIT RATE TRACKING (CONVERGENCE DETECTION) ===

void Island::TrackCacheResult(bool was_hit) {
  cache_result_window_.push_back(was_hit);
  if (was_hit) cache_hits_in_window_++;
  
  // Maintain rolling window size
  if (static_cast<int>(cache_result_window_.size()) > CACHE_WINDOW_SIZE) {
    if (cache_result_window_.front()) cache_hits_in_window_--;
    cache_result_window_.pop_front();
  }
}

double Island::GetRecentCacheHitRate() const {
  if (cache_result_window_.empty()) return 0.0;
  return static_cast<double>(cache_hits_in_window_) / cache_result_window_.size();
}

void Island::OnConvergenceWarning() {
  // 85-90% cache hit rate: Double mutation rate for EXPLORE islands
  if (IsExploration()) {
    convergence_mutation_boost_ = 2.0;
    std::cout << "\033[33m [CONV-WARN I" << id_ 
              << "] Cache hit >85% - Boosting mutation 2x\033[0m\n";
  }
}

void Island::OnConvergenceAlarm() {
  // 90-95% cache hit rate: Force broadcast of best to all siblings
  convergence_alarm_active_ = true;
  
  if (!exploit_siblings_.empty()) {
    Individual best_copy;
    {
      std::lock_guard<std::mutex> lock(best_mutex_);
      best_copy = current_best_;
    }
    
    for (Island* sibling : exploit_siblings_) {
      if (sibling != nullptr) {
        sibling->ReceiveBroadcastBest(best_copy);
      }
    }
    std::cout << "\033[35m [CONV-ALARM I" << id_ 
              << "] Cache hit >90% - Force broadcasted best to " 
              << exploit_siblings_.size() << " siblings\033[0m\n";
  }
  
  // Also increase mutation for exploration islands
  if (IsExploration()) {
    convergence_mutation_boost_ = 3.0;
  }
}

void Island::OnConvergenceCritical() {
  // >95% cache hit rate: Mini-catastrophe for EXPLOIT (50% population restart)
  std::cout << "\033[91m [CONV-CRITICAL I" << id_ 
            << "] Cache hit >95% - ";
  
  if (IsExploitation()) {
    // Restart 50% of population with random individuals
    std::lock_guard<std::mutex> lock(population_mutex_);
    int restart_count = population_.size() / 2;
    
    for (int i = 0; i < restart_count; ++i) {
      int victim_idx = rng_() % population_.size();
      // Don't restart the absolute best
      if (population_[victim_idx].GetFitness() > current_best_.GetFitness() + 1e-6) {
        Individual new_ind(evaluator_->GetSolutionSize());
        InitIndividual(new_ind, INITIALIZATION_TYPE::RANDOM);
        population_[victim_idx] = new_ind;
      }
    }
    std::cout << "Restarted " << restart_count << " individuals\033[0m\n";
  } else {
    // EXPLORE: Maximum mutation boost
    convergence_mutation_boost_ = 5.0;
    std::cout << "Boosting mutation 5x\033[0m\n";
  }
  
  // Clear the rolling window to reset detection
  cache_result_window_.clear();
  cache_hits_in_window_ = 0;
  convergence_alarm_active_ = false;
}
