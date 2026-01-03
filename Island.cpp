#include "Island.hpp"
#include "Constants.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>
#include <cstdint>

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
    return distance;
  }
  cache_misses_++;
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
    indiv.SetReturnCount(returns);
    indiv.SetFitness(distance);
    return distance;
  }
  cache_misses_++;

  EvaluationResult result = evaluator_->EvaluateWithStats(indiv.GetGenotype());

  distance = result.fitness;
  returns = result.returns;

  if (distance >= 1e15 || distance < 0.0) {
    distance = std::numeric_limits<double>::max();
  }

  indiv.SetReturnCount(returns);
  indiv.SetFitness(distance);

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
      int chunk_size = (num_groups > 0) ? (num_clients / num_groups) : num_clients;
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
    if (customer_id <= 1) continue;

    int gene_idx = customer_id - 2;
    if (gene_idx < 0 || gene_idx >= num_clients) continue;

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
        if (!found) current_group = candidates[0];
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
  current_generation_++;
  UpdateAdaptiveParameters();
  stagnation_count_++;



  {
    std::lock_guard<std::mutex> lock(population_mutex_);
    if (!immigration_queue_.empty()) {
      for (auto &immigrant : immigration_queue_) {
        // [ANTI-CLONE] Reject if duplicate exists
        if (ContainsSolution(immigrant)) continue;

        int worst = GetWorstBiasedIndex();
        if (worst >= 0 && worst < (int)population_.size()) {
          if (immigrant.GetFitness() < population_[worst].GetFitness()) {
            population_[worst] = immigrant;
            {
              std::lock_guard<std::mutex> best_lock(best_mutex_);
              if (immigrant.GetFitness() < current_best_.GetFitness()) {
                current_best_ = immigrant;
              }
            }
          }
        }
      }
      UpdateBiasedFitness();
      immigration_queue_.clear();
    }
  }

  const int lambda = population_size_;
  std::vector<Individual> offspring_pool;
  offspring_pool.reserve(lambda);

  double fitness_threshold = std::numeric_limits<double>::max();
  if (!population_.empty()) {
    fitness_threshold = population_[population_.size() / 2].GetFitness();
  }

  auto now = std::chrono::steady_clock::now();
  double elapsed = std::chrono::duration<double>(now - start_time_).count();
  bool is_endgame = (elapsed > Config::MAX_TIME_SECONDS * 0.9);
  
  std::uniform_real_distribution<double> d(0.0, 1.0);

  for (int i = 0; i < lambda; ++i) {
    Individual child(evaluator_->GetSolutionSize());
    int p1 = SelectParentIndex();
    int p2 = SelectParentIndex();

    if (p1 >= 0 && p2 >= 0) {
      child = is_endgame ? CrossoverSpatial(population_[p1], population_[p2]) 
                         : Crossover(population_[p1], population_[p2]);
    } else {
      InitIndividual(child, INITIALIZATION_TYPE::RANDOM);
    }

    // Unified Mutation Call
    int mutation_result = ApplyMutation(child, is_endgame);
    bool mutated = (mutation_result > 0);
    bool strong_mutation = (mutation_result == 2);

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

    bool promising = (fit < fitness_threshold);
    double vnd_prob = IsExploration() ? Config::EXPLORATION_VND_PROB 
                                      : Config::EXPLOITATION_VND_PROB;
    bool exploration_vnd = IsExploration() && (promising || (d(rng_) < 0.25));
    bool should_run_vnd = exploration_vnd || strong_mutation ||
                          (IsExploitation() && promising) || (d(rng_) < vnd_prob) || is_endgame;

    if (should_run_vnd) {
      int vnd_iters = GetVndIterations();
      if (current_structural_diversity_    > 0.6 || is_endgame) vnd_iters = (int)(vnd_iters * 1.5);
      bool allow_swap = IsExploitation() && Config::ALLOW_SWAP;
      bool allow_3swap = IsExploitation() && Config::ALLOW_3SWAP && !strong_mutation;
      bool allow_ejection = IsExploitation() && Config::ALLOW_EJECTION;

      // Set guide solution for Path Relinking (use current_best as target)
      {
        std::lock_guard<std::mutex> lock(best_mutex_);
        local_search_.SetGuideSolution(current_best_.GetGenotype());
      }

      if (local_search_.RunVND(child, vnd_iters, allow_swap, allow_3swap, allow_ejection)) {
        child.Canonicalize();
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
      }
    }

    offspring_pool.push_back(std::move(child));

    {
      std::lock_guard<std::mutex> lock(best_mutex_);
      if (fit < current_best_.GetFitness()) {
        current_best_ = offspring_pool.back();
        stagnation_count_ = 0;
        local_cache_.UpdateHistory(current_best_.GetGenotype());
        last_improvement_gen_ = current_generation_;
        fitness_threshold = fit * 1.05;
      }
    }
  }

  ApplySuccessionAdaptive(offspring_pool);

  {
    std::lock_guard<std::mutex> lock(population_mutex_);
    for (auto &ind : population_) ind.IncrementStagnation();
  }

  if (ShouldTrackDiversity()) UpdateBiasedFitness();

  long long time_since = current_generation_ - last_improvement_gen_;
  long long time_since_cat = current_generation_ - last_catastrophy_gen_;

  double worst_fit = 0.0;
  {
    std::lock_guard<std::mutex> lock(population_mutex_);
    for (const auto &ind : population_) if (ind.GetFitness() > worst_fit) worst_fit = ind.GetFitness();
  }

  double best_fit_for_catastrophe;
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    best_fit_for_catastrophe = current_best_.GetFitness();
  }
  
  bool fitness_collapsed = (worst_fit > best_fit_for_catastrophe * 1.5);
  bool stagnation_trigger = (time_since > BASE_STAGNATION_LIMIT && current_structural_diversity_ < 0.05);

  if ((stagnation_trigger || fitness_collapsed) && time_since_cat > 500) {
    Catastrophy();
    last_catastrophy_gen_ = current_generation_;
  }

  if (IsExploitation()) {
    {
      std::lock_guard<std::mutex> lock(best_mutex_);
      route_pool_.AddRoutesFromSolution(current_best_.GetGenotype(), *evaluator_);
    }

    double since_last_assembly = std::chrono::duration<double>(now - last_greedy_assembly_time_).count();
    
    if (since_last_assembly >= 3.0 && route_pool_.GetSize() >= 10) {
      Individual frankenstein = route_pool_.SolveBeamSearch(evaluator_, split_, 50);
      if (frankenstein.IsEvaluated() && frankenstein.GetFitness() < 1e9) {
        int vnd_iters = 40;
        if (elapsed > Config::MAX_TIME_SECONDS * 0.8) vnd_iters = 60;

        bool improved = false;
        for (int pass = 0; pass < 3; ++pass) {
             if (local_search_.RunVND(frankenstein, vnd_iters, true, true, true)) improved = true;
             else break;
        }
        
        if (improved) {
            frankenstein.Canonicalize();
            frankenstein.SetFitness(SafeEvaluate(frankenstein));
            // std::cout << " [BEAM] Frankenstein improved by VND! Final Fit: " << frankenstein.GetFitness() << std::endl;
        }

        // [ANTI-CLONE] Check if this frankenstein is already in population
        if (!ContainsSolution(frankenstein)) {
            std::lock_guard<std::mutex> lock(population_mutex_);
            
            // Force Injection Logic (User Request: "siłowo wstrzykiwany")
            // 10% chance to force inject, displacing a random individual (but not the absolute best)
            bool force_injected = false;
            std::uniform_real_distribution<double> d_force(0.0, 1.0);
            if (d_force(rng_) < 0.10) { 
                int victim_idx = rng_() % population_.size();
                // Protect the absolute best from forced replacement to ensure monotonicity of best found
                if (population_[victim_idx].GetFitness() > current_best_.GetFitness() + 1e-6) { 
                    population_[victim_idx] = frankenstein;
                    std::cout << "\033[35m [BEAM] [Island " << id_ << "] Frankenstein FORCIBLY injected (Fit: " << frankenstein.GetFitness() << ")\033[0m" << std::endl;
                    force_injected = true;
                }
            }

            if (!force_injected) {
                int worst = GetWorstBiasedIndex();
                if (worst >= 0) {
                    if (frankenstein.GetFitness() < population_[worst].GetFitness()) {
                        population_[worst] = frankenstein;
                        std::cout << "\033[35m [BEAM] [Island " << id_ << "] Frankenstein injected into population (Fit: " << frankenstein.GetFitness() << ")\033[0m" << std::endl;
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

bool Island::ContainsSolution(const Individual& ind) const {
  uint64_t h = HashGenotype64(ind.GetGenotype());
  double fit = ind.GetFitness();
  for (const auto& p : population_) {
      if (std::abs(p.GetFitness() - fit) < 1e-4) {
          if (HashGenotype64(p.GetGenotype()) == h) return true;
      }
  }
  return false;
}

void Island::RunDebugDiagnostics() {
  std::cout << "\n--- [DIAGNOSTICS GEN " << current_generation_ << "] ---" << std::endl;
  // Simplified diagnostics to avoid clutter
  // Checks basic consistency if needed, currently empty for performance/cleanliness
  // as per refactoring request.
  std::cout << "Best Fix: " << current_best_.GetFitness() 
            << " | Div: " << current_structural_diversity_ << std::endl;
}

int Island::ApplyMicroSplitMutation(Individual &child) {
  double stagnation_factor = std::min(1.0, (double)stagnation_count_ / 2000.0);
  int intensity = IsExploration() ? 2 : 0;
  bool success = mutator_.ApplyMicroSplitMutation(child, stagnation_factor, intensity, rng_);

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

  // 1. Structural Mutations (MicroSplit)
  if (d(rng_) < p_microsplit_) {
    ApplyMicroSplitMutation(child);
    strong_mutation = true;
    mutated = true;
#ifdef RESEARCH
    executed_op = (int)OpType::MUT_SIMPLE; // Mapping MicroSplit to Simple for stats or new enum
#endif
  }

  // 2. Standard Mutations (Aggressive / Smart Spatial / Ruin)
  // Re-using the logic that was previously in ApplyMutation but now controlled by p_mutation_
  if (d(rng_) < p_mutation_) {
    double rnd = d(rng_);
    if (rnd < 0.05) {
      mutator_.AggressiveMutate(child, rng_);
#ifdef RESEARCH
      executed_op = (int)OpType::MUT_AGGRESSIVE;
#endif
    } else if (rnd < 0.35) {
      mutator_.ApplySmartSpatialMove(child, rng_);
    } else {
      mutator_.ApplyRuinRecreate(child, (1 - current_structural_diversity_), rng_);
#ifdef RESEARCH
      executed_op = (int)OpType::MUT_SPATIAL;
#endif
    }
    mutated = true;
  }

  // 3. Load Balancing
  // Configurable switch
  if (Config::ALLOW_LOAD_BALANCING && d(rng_) < p_loadbalance_) {
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
  double swap_chance = is_endgame ? 0.50 : p_swap3_;
  if (d(rng_) < swap_chance) {
    if (local_search_.Try3Swap(child.AccessGenotype())) {
      mutated = true;
      strong_mutation = true;
    }
  }

  // 7. 4-Swap (Very heavy)
  double current_p_swap4 = is_endgame ? 0.40 : p_swap4_;
  if (d(rng_) < current_p_swap4) {
    if (local_search_.Try4Swap(child.AccessGenotype())) {
      mutated = true;
      strong_mutation = true;
    }
  }

  // Return logic: currently returns int op type, but caller in RunGeneration needs to know if mutation happened
  // We can return executed_op for stats, and checking > -1 or similar. 
  // However, multiple ops can run. Let's return the most significant one or just generic.
  // The original RunGeneration used flags. 
  // We will return 1 if meaningful mutation happened (strong), 2 if weak, 0 if none.
  if (strong_mutation) return 2;
  if (mutated) return 1;
  return 0;
}

void Island::Catastrophy() {
#ifdef RESEARCH
  catastrophy_activations++;
  cout << "Catastrophe on island [" << id_ << "] CV: " << std::scientific
       << std::setprecision(2) << current_structural_diversity_
       << std::defaultfloat << "\n";
#endif

  std::vector<Individual> new_pop;
  new_pop.reserve(population_size_);
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    new_pop.push_back(current_best_);
  }
  int sol_size = evaluator_->GetSolutionSize();

  std::vector<Individual> candidates;
  int candidates_count = population_size_ * 10;

  for (int i = 0; i < candidates_count; ++i) {
    Individual indiv(sol_size);
    if (i % 2 == 0)
      InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);
    else
      InitIndividual(indiv, INITIALIZATION_TYPE::CHUNKED);

    double penalized_fit = EvaluateWithHistoryPenalty(indiv.GetGenotype());
    indiv.SetFitness(penalized_fit);
    candidates.push_back(indiv);
  }

  std::sort(candidates.begin(), candidates.end());

  for (int i = 0; i < population_size_ - 1; ++i) {
    Individual &selected = candidates[i];
    int vnd_threshold = std::max(5, population_size_ / 5);
    if (i < vnd_threshold) {
      local_search_.RunVND(selected, 30, true, true, true);
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
  CalculatePopulationCV();
  local_cache_.ClearHistory();
}

void Island::UpdateBiasedFitness() {
  int pop_size = static_cast<int>(population_.size());
  if (pop_size == 0) return;

  std::vector<int> indices(pop_size);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return population_[a].GetFitness() < population_[b].GetFitness();
  });

  const std::vector<int> &perm = evaluator_->GetPermutation();
  int num_groups = evaluator_->GetNumGroups();

  int ref_size = std::min(pop_size, std::max(2, Config::ELITERATIO));
  double total_population_bpd = 0.0;
  int measurements_count = 0;

  for (int i = 0; i < pop_size; ++i) {
    double dist_sum = 0.0;
    int idx_i = indices[i];
    int comparisons = 0;

    for (int k = 0; k < ref_size; ++k) {
      int idx_best = indices[k];
      if (idx_i == idx_best) continue;

      int bpd = CalculateBrokenPairsDistance(
          population_[idx_i], population_[idx_best], perm, num_groups);
      dist_sum += bpd;
      comparisons++;
    }

    double avg_dist = (comparisons > 0) ? (dist_sum / comparisons) : 0.0;
    population_[idx_i].SetDiversityScore(avg_dist);
    total_population_bpd += avg_dist;
    measurements_count++;
  }

  if (measurements_count > 0) {
    double avg_raw_bpd = total_population_bpd / measurements_count;
    double raw_diversity = avg_raw_bpd / (double)evaluator_->GetSolutionSize();
    double range = max_diversity_baseline_ - min_diversity_baseline_;
    if (range > 0.001) {
      current_structural_diversity_ = (raw_diversity - min_diversity_baseline_) / range;
      current_structural_diversity_ = std::max(0.0, std::min(1.0, current_structural_diversity_));
    } else {
      current_structural_diversity_ = 0.5;
    }
  } else {
    current_structural_diversity_ = 0.0;
  }

  std::sort(indices.begin(), indices.end(), [&](int a, int b) {
    return population_[a].GetDiversityScore() > population_[b].GetDiversityScore();
  });

  std::vector<int> rank_diversity(pop_size);
  for (int r = 0; r < pop_size; ++r) rank_diversity[indices[r]] = r;

  std::vector<int> rank_fitness(pop_size);
  std::vector<int> fit_indices(pop_size);
  std::iota(fit_indices.begin(), fit_indices.end(), 0);
  std::sort(fit_indices.begin(), fit_indices.end(), [&](int a, int b) {
    return population_[a].GetFitness() < population_[b].GetFitness();
  });
  for (int r = 0; r < pop_size; ++r) rank_fitness[fit_indices[r]] = r;

  double elite_ratio = (double)ref_size / (double)pop_size;
  for (int i = 0; i < pop_size; ++i) {
    double biased = (double)rank_fitness[i] + (1.0 - elite_ratio) * (double)rank_diversity[i];
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
  if (size == 0) return 0;

  std::fill(last_in_group1.begin(), last_in_group1.end(), -1);
  std::fill(last_in_group2.begin(), last_in_group2.end(), -1);

  for (int customer_id : permutation) {
    int idx = customer_id - 2;
    if (idx < 0 || idx >= size) continue;

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
    if (pred1[i] != pred2[i]) distance++;
  }
  return distance;
}

void Island::CalculatePopulationCV() {
  if (population_.empty()) {
    current_structural_diversity_ = 0.0;
    return;
  }
  double mean = 0.0;
  double M2 = 0.0;
  int n = 0;

  for (const auto &ind : population_) {
    double x = ind.GetFitness();
    if (x > 1e14) continue;
    n++;
    double delta = x - mean;
    mean += delta / n;
    double delta2 = x - mean;
    M2 += delta * delta2;
  }

  if (n < 2) {
    current_structural_diversity_ = 1.0;
    return;
  }
  double variance = M2 / (n - 1);
  double std_dev = std::sqrt(variance);
  current_structural_diversity_ = (mean > 1e-6) ? (std_dev / mean) : 0.0;
}

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
    if (loads[g] > capacity) overloaded_groups.push_back(g);
  }
  if (overloaded_groups.empty()) return false;

  std::shuffle(overloaded_groups.begin(), overloaded_groups.end(), rng_);
  bool any_change = false;
  const int MAX_DEPTH = 10;

  for (int start_group : overloaded_groups) {
    if (loads[start_group] <= capacity) continue;

    std::vector<ChainMove> chain_history;
    std::vector<bool> group_visited(num_groups, false);
    int current_group = start_group;
    group_visited[current_group] = true;
    bool chain_success = false;

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
      auto move = FindNextChainMove(current_group, group_visited, loads, group_clients);
      int best_client = move.first;
      int best_target = move.second;

      if (best_target == -1) break;

      int demand = demands[best_client - 1];
      int sol_idx = best_client - 2;

      if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
        solution[sol_idx] = best_target;
        loads[current_group] -= demand;
        loads[best_target] += demand;

        auto it = std::find(group_clients[current_group].begin(), group_clients[current_group].end(), best_client);
        if (it != group_clients[current_group].end()) {
          *it = group_clients[current_group].back();
          group_clients[current_group].pop_back();
          group_clients[best_target].push_back(best_client);
        }

        chain_history.push_back({best_client, current_group, best_target, demand});
        group_visited[best_target] = true;
      }

      bool all_ok = true;
      for (const auto &m : chain_history) {
        if (loads[m.from_group] > capacity) all_ok = false;
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

std::pair<int, int> Island::FindNextChainMove(int group_idx, 
                                        const std::vector<bool> &visited,
                                        const std::vector<int>& loads,
                                        const std::vector<std::vector<int>>& group_clients) {
    if (group_clients[group_idx].empty()) return {-1, -1};
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
            if (g == group_idx || visited[g]) continue;
            if (loads[g] + demand <= capacity) return {client_id, g};
        }
    }

    // Strategy 2: Force move to explore
    for (int i = 0; i < trials; ++i) {
        int client_id = candidates[i];
        int target_candidate = rng_() % num_groups;
        for (int k = 0; k < 5; k++) {
            int g = (target_candidate + k) % num_groups;
            if (g != group_idx && !visited[g]) return {client_id, g};
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
    if (loads[g] > capacity) overloaded_groups.push_back(g);
  }
  if (overloaded_groups.empty()) return false;
  std::shuffle(overloaded_groups.begin(), overloaded_groups.end(), rng_);
  
  bool changed = false;
  for (int source_group : overloaded_groups) {
    if (loads[source_group] <= capacity) continue;
    std::vector<int> target_groups(num_groups);
    std::iota(target_groups.begin(), target_groups.end(), 0);
    std::shuffle(target_groups.begin(), target_groups.end(), rng_);
    
    bool fixed = false;
    for (int target_group : target_groups) {
      if (target_group == source_group) continue;
      if (fixed) break;
      
      std::shuffle(group_clients[source_group].begin(), group_clients[source_group].end(), rng_);
      std::shuffle(group_clients[target_group].begin(), group_clients[target_group].end(), rng_);
      
      for (int client_a : group_clients[source_group]) {
          int demand_a = demands[client_a - 1];
          for (int client_b : group_clients[target_group]) {
              int demand_b = demands[client_b - 1];
              if (demand_a <= demand_b) continue; // Only swap if reducing load
              
              if (loads[target_group] - demand_b + demand_a <= capacity) {
                   int idx_a = client_a - 2;
                   int idx_b = client_b - 2;
                   if (idx_a >= 0 && idx_b >= 0) {
                       solution[idx_a] = target_group;
                       solution[idx_b] = source_group;
                       loads[source_group] = loads[source_group] - demand_a + demand_b;
                       loads[target_group] = loads[target_group] - demand_b + demand_a;
                       changed = true;
                       if (loads[source_group] <= capacity) fixed = true;
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
      // Correct mapping via Permutation is better but costly here, assumes simplistic index
      // But let's use proper mapping if possible.
      // Actually the caller logic used exact permutation loop.
      // For simplicity, let's use the permutation-based loop for load calculation
      // BUT `solution` is indexed by (customer_id - 2).
  }
  // Re-calculating loads properly:
  const std::vector<int> &perm = evaluator_->GetPermutation();
  for (int customer_id : perm) {
      int sol_idx = customer_id - 2;
      if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
          int g = solution[sol_idx];
          if (g >= 0 && g < num_groups) loads[g] += demands[customer_id - 1];
      }
  }

  std::vector<int> overloaded;
  std::vector<int> underloaded;
  for (int g = 0; g < num_groups; ++g) {
      if (loads[g] > capacity) overloaded.push_back(g);
      else underloaded.push_back(g);
  }
  if (overloaded.empty() || underloaded.empty()) return false;
  
  std::shuffle(overloaded.begin(), overloaded.end(), rng_);
  bool changed = false;
  
  for (int source : overloaded) {
     if (loads[source] <= capacity) continue;
     // Find clients in this group
     std::vector<int> clients;
     for (int cust_id : perm) {
         int idx = cust_id - 2;
         if (idx >= 0 && solution[idx] == source) clients.push_back(cust_id);
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
         if (loads[source] <= capacity) break;
     }
  }
  return changed;
}

Individual Island::CrossoverSpatial(const Individual &p1, const Individual &p2) {
  const std::vector<int> &g1 = p1.GetGenotype();
  const std::vector<int> &g2 = p2.GetGenotype();
  int size = static_cast<int>(g1.size());
  Individual child(size);
  std::vector<int> &child_genes = child.AccessGenotype();

  if (size == 0 || !geometry_.HasCoordinates()) return child;

  std::uniform_int_distribution<int> dist_idx(0, size - 1);
  int center_idx = dist_idx(rng_);
  const auto &center_coord = geometry_.GetCoordinate(center_idx + 1);

  int radius_idx = dist_idx(rng_);
  const auto &radius_coord = geometry_.GetCoordinate(radius_idx + 1);

  double r_sq = std::pow(center_coord.x - radius_coord.x, 2) + 
                std::pow(center_coord.y - radius_coord.y, 2);

  for (int i = 0; i < size; ++i) {
    const auto &px = geometry_.GetCoordinate(i + 1);
    double dist_sq = std::pow(center_coord.x - px.x, 2) + 
                     std::pow(center_coord.y - px.y, 2);
    child_genes[i] = (dist_sq <= r_sq) ? g1[i] : g2[i];
  }
  return child;
}

Individual Island::CrossoverSequence(const Individual &p1, const Individual &p2) {
  const std::vector<int> &perm = evaluator_->GetPermutation();
  int perm_size = static_cast<int>(perm.size());
  Individual child = p1;
  std::vector<int> &child_genes = child.AccessGenotype();
  const std::vector<int> &p2_genes = p2.GetGenotype();

  if (child_genes.empty()) return child;
  int cut_point = rng_() % perm_size;

  for (int i = cut_point; i < perm_size; ++i) {
    int customer_id = perm[i];
    int gene_idx = customer_id - 2;
    if (gene_idx >= 0 && gene_idx < (int)child_genes.size()) {
      child_genes[gene_idx] = p2_genes[gene_idx];
    }
  }
  return child;
}

Individual Island::CrossoverUniform(const Individual &p1, const Individual &p2) {
  const std::vector<int> &g1 = p1.GetGenotype();
  const std::vector<int> &g2 = p2.GetGenotype();
  int size = static_cast<int>(g1.size());
  Individual child(size);
  std::vector<int> &child_genes = child.AccessGenotype();

  for (int i = 0; i < size; ++i) {
    child_genes[i] = (rng_() % 2 == 0) ? g1[i] : g2[i];
  }
  return child;
}

Individual Island::Crossover(const Individual &p1, const Individual &p2) {
#ifdef RESEARCH
  crossovers++;
#endif
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  double r = dist(rng_);
  double sequence_prob;

  if (id_ <= 1) sequence_prob = 0.31;
  else if (id_ <= 3) sequence_prob = 0.4;
  else sequence_prob = 0.6;

  if (r < sequence_prob) return ApplySREX(p1, p2);
  else return CrossoverSpatial(p1, p2);
}

void Island::PrintIndividual(const Individual &individual, int global_generation) const {
  int num_groups = evaluator_->GetNumGroups();
  vector<int> group_counts(num_groups, 0);
  const vector<int> &genes = individual.GetGenotype();
  for (int g : genes)
    if (g >= 0 && g < num_groups) group_counts[g]++;
    
  int extra_returns = evaluator_->GetTotalDepotReturns(genes);
  cout << "   [Island " << id_ << "] Gen: " << setw(6) << global_generation
       << " | Dist: " << fixed << setprecision(2) << individual.GetFitness()
       << " | Ret: " << extra_returns << " | Groups: [";
  for (size_t i = 0; i < group_counts.size(); ++i)
    cout << group_counts[i] << (i < group_counts.size() - 1 ? "," : "");
  cout << "] CV: " << std::scientific << std::setprecision(2)
       << current_structural_diversity_ << std::defaultfloat << "\n";
}

void Island::ApplySplitToIndividual(Individual &indiv) {
  const std::vector<int> &global_perm = evaluator_->GetPermutation();
  int fleet_limit = evaluator_->GetNumGroups();
  SplitResult result = split_.RunLinear(global_perm);

  if (result.feasible) {
    std::vector<int> &genes = indiv.AccessGenotype();
    if (result.group_assignment.size() != genes.size()) return;

    int routes_count = static_cast<int>(result.optimized_routes.size());
    for (size_t i = 0; i < genes.size(); ++i) {
      int assigned_route_id = result.group_assignment[i];
      genes[i] = (assigned_route_id < fleet_limit) ? assigned_route_id : (assigned_route_id % fleet_limit);
    }
    
    int excess_vehicles = (routes_count > fleet_limit) ? (routes_count - fleet_limit) : 0;
    indiv.SetFitness(result.total_cost);
    indiv.SetReturnCount(excess_vehicles);
  } else {
    indiv.SetFitness(1.0e30);
  }
}

// UpdateAdaptiveParameters: Tuned based on Island Type and Calibrated Structural Diversity
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
  double dynamic_mut_prob = base_mut_prob + (0.3 * (1.0 - chaos)) - (0.1 * chaos);
  dynamic_mut_prob = std::max(0.05, std::min(0.80, dynamic_mut_prob));

  // Set member probabilities based on role and state
  p_microsplit_ = dynamic_mut_prob;
  p_mutation_ = dynamic_mut_prob * 0.8;
  p_loadbalance_ = dynamic_mut_prob * 0.5;
  p_retminimizer_ = dynamic_mut_prob * 0.6;
  p_mergesplit_ = dynamic_mut_prob * 0.5;
  
  if (IsExploitation()) {
      p_swap3_ = 0.30;
      p_swap4_ = 0.05;
      
      // Fine-tuning for exploitation
      if (chaos < 0.1) { // Very converged
          p_loadbalance_ *= 1.2; // Increase repair attempts
          p_retminimizer_ *= 1.2;
      }
  } else {
      // Exploration
      p_swap3_ = 0.10;
      p_swap4_ = 0.0; // Rarely use expensive swaps in exploration
  }

  // Legacy variables update (if still used elsewhere, though mostly replaced now)
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
  if (population_.empty()) return;

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
          unique_candidates.push_back(ind); // Copy needed as we iterate const ref but push back
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

  // Determine split:
  // If pop is CHAOTIC (high div), we can afford to keep mostly Elites (best fitness) to converge.
  // If pop is CONVERGED (low div), we must Force Diversity (keep individuals based on biased fitness/difference).
  
  double elite_ratio = 0.5;
  if (relative_div > 0.8) {
      elite_ratio = 1.0; // Fully chaotic -> Convergence mode -> Keep all Best
  } else if (relative_div > 0.5) {
      elite_ratio = 0.8; 
  } else if (relative_div > 0.2) {
      elite_ratio = 0.5;
  } else {
      elite_ratio = 0.2; // Stagnated -> Emergency Diversity -> Keep only top 20% best, rest diverse
  }
  
  // Apply selection
  int elite_count = (int)(population_size_ * elite_ratio);
  elite_count = std::max(2, elite_count); // Always keep at least top 2
  
  std::vector<Individual> next_pop;
  next_pop.reserve(population_size_);
  std::unordered_set<int> taken_indices;
  
  // 1. Take Elites (Pure Fitness)
  for (int i = 0; i < (int)population_.size(); ++i) {
      if ((int)next_pop.size() >= elite_count) break;
      next_pop.push_back(population_[i]);
      taken_indices.insert(i);
  }
  
  // 2. Take Diverse (Biased Fitness)
  if ((int)next_pop.size() < population_size_) {
       UpdateBiasedFitness(); // Recalculate diversity ranks on FULL pool
       
       std::vector<int> biased_indices(population_.size());
       std::iota(biased_indices.begin(), biased_indices.end(), 0);
       std::sort(biased_indices.begin(), biased_indices.end(), [&](int a, int b){
           return population_[a].GetBiasedFitness() < population_[b].GetBiasedFitness();
       });
       
       for (int idx : biased_indices) {
           if ((int)next_pop.size() >= population_size_) break;
           if (taken_indices.find(idx) == taken_indices.end()) {
               next_pop.push_back(population_[idx]);
               taken_indices.insert(idx);
           }
       }
  }
  
  // 3. Fallback (if still needed)
  if ((int)next_pop.size() < population_size_) {
      for (int i = 0; i < (int)population_.size(); ++i) {
          if ((int)next_pop.size() >= population_size_) break;
          if (taken_indices.find(i) == taken_indices.end()) {
              next_pop.push_back(population_[i]);
          }
      }
  }
  
  population_ = std::move(next_pop);
  
  // Sort final population by fitness
  std::sort(population_.begin(), population_.end(), [](const Individual& a, const Individual& b){
      return a.GetFitness() < b.GetFitness();
  });
  
  UpdateBiasedFitness(); // Final update for next gen stats
}

#ifdef RESEARCH
std::vector<int> Island::CanonicalizeGenotype(const std::vector<int> &genotype, int num_groups) const {
  if (genotype.empty()) return {};
  std::vector<int> canonical = genotype;
  std::vector<int> mapping(num_groups + 1, -1);
  int next_new_id = 0;
  for (size_t i = 0; i < canonical.size(); ++i) {
    int old_id = canonical[i];
    if (old_id < 0 || old_id >= (int)mapping.size()) continue;
    if (mapping[old_id] == -1) mapping[old_id] = next_new_id++;
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
  if (population_.empty()) return -1;
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
  if (population_.empty()) return -1;
  
  std::vector<int> best_genotype_copy;
  {
    std::lock_guard<std::mutex> lock(best_mutex_);
    best_genotype_copy = current_best_.GetGenotype();
  }

  int pop_size = static_cast<int>(population_.size());
  int worst = -1;
  double max_val = -1.0;
  for (int i = 0; i < pop_size; ++i) {
    if (population_[i].GetGenotype() == best_genotype_copy) continue;
    if (population_[i].GetBiasedFitness() > max_val) {
      max_val = population_[i].GetBiasedFitness();
      worst = i;
    }
  }
  return (worst == -1) ? GetWorstIndex() : worst;
}

int Island::GetWorstIndex() const {
  if (population_.empty()) return -1;
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
  double fit;
  if (imigrant.IsEvaluated()) {
    fit = imigrant.GetFitness();
  } else {
    EvaluationResult res = evaluator_->EvaluateWithStats(imigrant.GetGenotype());
    fit = res.fitness;
    imigrant.SetFitness(fit);
    imigrant.SetReturnCount(res.returns);
  }

  if (fit != std::numeric_limits<double>::max()) {
    double best_fit;
    {
      std::lock_guard<std::mutex> best_lock(best_mutex_);
      best_fit = current_best_.GetFitness();
    }
    double dist_to_best = std::abs(fit - best_fit);
    if (dist_to_best < 500.0 && fit >= best_fit) return;

    std::lock_guard<std::mutex> lock(population_mutex_);
    immigration_queue_.push_back(imigrant);
  }
}

Individual Island::GetRandomIndividual() {
  std::lock_guard<std::mutex> lock(population_mutex_);
  if (population_.empty()) return current_best_;
  std::uniform_int_distribution<int> dist(0, (int)population_.size() - 1);
  return population_[dist(rng_)];
}

Individual Island::GetMostDiverseMigrantFor(const Individual& target_best) {
  std::lock_guard<std::mutex> lock(population_mutex_);
  if (population_.empty()) return current_best_;
  
  const std::vector<int>& perm = evaluator_->GetPermutation();
  int num_groups = evaluator_->GetNumGroups();
  
  int best_idx = 0;
  int max_distance = -1;
  
  for (int i = 0; i < (int)population_.size(); ++i) {
    int distance = CalculateBrokenPairsDistance(population_[i], target_best, perm, num_groups);
    if (distance > max_distance) {
      max_distance = distance;
      best_idx = i;
    }
  }
  return population_[best_idx];
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
      if (i == other_idx) continue;
      int dist = CalculateBrokenPairsDistance(random_samples[i], random_samples[other_idx], perm, num_groups);
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
      if (i == other_idx) continue;
      int dist = CalculateBrokenPairsDistance(converged_samples[i], converged_samples[other_idx], perm, num_groups);
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
  if (in_max - in_min < 1e-9) return out_min;
  double clamped = std::max(in_min, std::min(in_max, value));
  return out_min + (out_max - out_min) * (clamped - in_min) / (in_max - in_min);
}

int Island::GetVndIterations() const {
  int base_min = IsExploration() ? Config::EXPLORATION_VND_MIN : Config::EXPLOITATION_VND_MIN;
  int base_max = IsExploration() ? Config::EXPLORATION_VND_MAX : Config::EXPLOITATION_VND_MAX;
  double result = base_max - (current_structural_diversity_ * (base_max - base_min));
  return static_cast<int>(std::max((double)base_min, std::min((double)base_max, result)));
}

double Island::GetMutationRate() const {
  if (IsExploration()) {
    return MapRange(current_structural_diversity_, min_diversity_baseline_, max_diversity_baseline_, 0.50, 0.20);
  } else {
    return MapRange(current_structural_diversity_, min_diversity_baseline_, max_diversity_baseline_, 0.15, 0.05);
  }
}

double Island::GetRuinChance() const {
  if (IsExploration()) {
    return MapRange(current_structural_diversity_, min_diversity_baseline_, max_diversity_baseline_, 0.50, 0.20);
  } else {
    return MapRange(current_structural_diversity_, min_diversity_baseline_, max_diversity_baseline_, 0.15, 0.05);
  }
}

double Island::GetMicrosplitChance() const {
  if (IsExploration()) {
    return MapRange(current_structural_diversity_, min_diversity_baseline_, max_diversity_baseline_, 0.65, 0.4);
  } else {
    return MapRange(current_structural_diversity_, min_diversity_baseline_, max_diversity_baseline_, 0.3, 0.2);
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
        Individual rnd(num_clients); InitIndividual(rnd, INITIALIZATION_TYPE::RANDOM); return rnd;
    }

    std::vector<int> child_genotype(num_clients, -1);
    std::vector<bool> is_covered(num_clients, false);
    
    // Helper lambda to build route map
    auto build_routes = [&](const std::vector<int>& g) {
        int max_g = 0;
        for (int x : g) if (x > max_g) max_g = x;
        if (max_g > num_clients) max_g = num_clients;
        std::vector<std::vector<int>> routes(max_g + 1);
        for (int i = 0; i < num_clients; ++i) {
            if (g[i] >= 0 && g[i] <= max_g) routes[g[i]].push_back(i);
        }
        return routes;
    };

    auto routes1 = build_routes(g1);
    auto routes2 = build_routes(g2);

    std::vector<int> active1, active2;
    for (size_t i = 0; i < routes1.size(); ++i) if (!routes1[i].empty()) active1.push_back((int)i);
    for (size_t i = 0; i < routes2.size(); ++i) if (!routes2[i].empty()) active2.push_back((int)i);

    if (!active1.empty()) std::shuffle(active1.begin(), active1.end(), rng_);
    if (!active2.empty()) std::shuffle(active2.begin(), active2.end(), rng_);

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
            if (is_covered[client]) { conflict = true; break; }
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

    // Repair unassigned
    std::vector<int> unassigned;
    for (int i = 0; i < num_clients; ++i) if (!is_covered[i]) unassigned.push_back(i);
    
    if (!unassigned.empty()) {
        std::shuffle(unassigned.begin(), unassigned.end(), rng_);
        for (int client : unassigned) {
            int demand = evaluator_->GetDemand(client + 2);
            bool placed = false;
            
            // Try existing
            if (!child_group_loads.empty()) {
                size_t start = rng_() % child_group_loads.size();
                for (size_t k = 0; k < child_group_loads.size(); ++k) {
                    size_t idx = (start + k) % child_group_loads.size();
                    if (child_group_loads[idx] + demand <= capacity_) {
                        child_genotype[client] = (int)idx;
                        child_group_loads[idx] += demand;
                        placed = true;
                        break;
                    }
                }
            }
            if (!placed) {
                child_genotype[client] = current_child_group;
                child_group_loads.push_back(demand);
                current_child_group++;
            }
        }
    }

    Individual child(child_genotype);
    child.Canonicalize();
    return child;
}
