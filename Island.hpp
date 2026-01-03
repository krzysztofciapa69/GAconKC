#pragma once
#include "Constants.hpp"
#include "Evaluator.hpp"
#include "Individual.hpp"
#include "LocalCache.hpp"
#include "ProblemData.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"

// Operations
#include "LocalSearch.hpp"
#include "Mutator.hpp"
#include "ProblemGeometry.hpp"
#include "RoutePool.hpp"

#include <array>
#include <atomic>
#include <mutex>
#include <random>
#include <vector>

namespace LcVRPContest {

enum class INITIALIZATION_TYPE { RANDOM, CHUNKED, RR, SMART_STICKY };

class Island {
public:
  Island(ThreadSafeEvaluator *evaluator, const ProblemData &data,
         int population_size, int id);

  void Initialize(INITIALIZATION_TYPE strategy);
  void RunGeneration();
  void InjectImmigrant(Individual &imigrant);
  double EvaluateWithHistoryPenalty(const std::vector<int> &genotype);

  std::vector<int> GetBestSolution() const {
    std::lock_guard<std::mutex> lock(best_mutex_);
    return current_best_.GetGenotype();
  }
  
  Individual GetBestIndividual() const {
    std::lock_guard<std::mutex> lock(best_mutex_);
    return current_best_;
  }
  
  Individual GetRandomIndividual();
  Individual GetMostDiverseMigrantFor(const Individual& target_best);
  
  double GetBestFitness() const {
    std::lock_guard<std::mutex> lock(best_mutex_);
    return current_best_.GetFitness();
  }
  
  void PrintIndividual(const Individual &individual, int global_generation) const;

  int GetId() const { return id_; }
  bool IsExploration() const { return (id_ % 2) == 0; } // Even: 0, 2, 4
  bool IsExploitation() const { return (id_ % 2) == 1; } // Odd: 1, 3, 5
  
  // Returns structural diversity (0=converged, 1=chaotic)
  bool ContainsSolution(const Individual& ind) const;
  double GetCurrentCV() const {
    return current_structural_diversity_;
  }

  // Cache statistics
  long long GetCacheHits() const { return cache_hits_; }
  long long GetCacheMisses() const { return cache_misses_; }

  // Get stuck individuals for inter-archipelago migration
  bool TryGetStuckIndividual(Individual &out);

  // Endgame Mode support
  void SetStartTime(std::chrono::steady_clock::time_point start_time) {
    start_time_ = start_time;
  }

  long long total_comps = 0;
  long long passes = 0;

#ifdef RESEARCH
  void ExportState(int generation, bool is_catastrophe) const;
#endif

  long long getPRStats() const {
    return local_search_.prsucc;
  }

private:
  ThreadSafeEvaluator *evaluator_;

  Mutator mutator_;
  const std::vector<int> &demands_;
  int capacity_;

  ProblemGeometry geometry_;
  LocalSearch local_search_;
  
  std::vector<int> customer_ranks_;
  int population_size_;
  int id_;

  LocalCache local_cache_;
  std::vector<Individual> population_;
  std::vector<Individual> stuck_queue_;
  std::vector<Individual> immigration_queue_; // Thread-safe queue
  Individual current_best_;
  std::mt19937 rng_;
  mutable std::mutex best_mutex_;
  std::mutex population_mutex_;
  int stagnation_count_ = 0;

  // Cache statistics
  long long cache_hits_ = 0;
  long long cache_misses_ = 0;

  // Timers
  std::chrono::steady_clock::time_point start_time_;
  std::chrono::steady_clock::time_point last_alns_print_time_;
  std::chrono::steady_clock::time_point last_greedy_assembly_time_;

  RoutePool route_pool_;
  Split split_;
  
  double max_diversity_baseline_ = 1.0;
  double min_diversity_baseline_ = 0.0;

  void CalibrateDiversity();
  void CalibrateConvergence(); 
  
  double current_structural_diversity_ = 0.0; // 0=converged, 1=chaotic
  double adaptive_mutation_rate_ = 0.0;
  double adaptive_vnd_prob_ = 0.0;
  double adaptive_ruin_chance_ = 0.0;
  
  // Adaptive Probabilities
  double p_microsplit_ = 0.0;
  double p_mutation_ = 0.0;
  double p_loadbalance_ = 0.0;
  double p_retminimizer_ = 0.0;
  double p_mergesplit_ = 0.0;
  double p_swap3_ = 0.0;
  double p_swap4_ = 0.0;

  long long current_generation_ = 0;
  long long last_improvement_gen_ = 0;
  long long last_catastrophy_gen_ = 0;

  const int BASE_STAGNATION_LIMIT = 1000;

  // Buffers
  std::vector<int> pred1;
  std::vector<int> pred2;
  std::vector<int> last_in_group1;
  std::vector<int> last_in_group2;

  // Methods
  void CalculatePopulationCV();
  void UpdateAdaptiveParameters();
  double MapRange(double value, double in_min, double in_max, double out_min, double out_max) const;

  // Dynamic parameters
  int GetVndIterations() const;
  double GetMutationRate() const;
  double GetRuinChance() const;
  double GetMicrosplitChance() const;
  bool ShouldTrackDiversity() const { return true; } 

  int ApplyLoadBalancing(Individual &child);

  double SafeEvaluate(Individual &indiv);
  double SafeEvaluate(const std::vector<int> &genotype);

  void InitIndividual(Individual &indiv, INITIALIZATION_TYPE strategy);
  void InitIndividualSmartSticky(Individual &indiv); // Extracted helper

  void UpdateBiasedFitness();
  void Catastrophy();
  int CalculateBrokenPairsDistance(const Individual &ind1,
                                   const Individual &ind2,
                                   const std::vector<int> &permutation,
                                   int num_groups);

  int SelectParentIndex();
  int GetWorstBiasedIndex() const;
  int GetWorstIndex() const;

  Individual CrossoverUniform(const Individual &p1, const Individual &p2);
  Individual Crossover(const Individual &p1, const Individual &p2);
  Individual CrossoverSequence(const Individual &p1, const Individual &p2);
  Individual CrossoverSpatial(const Individual &p1, const Individual &p2);

  int ApplyMutation(Individual &child, bool is_endgame);
  int ApplyMicroSplitMutation(Individual &child);

  bool ApplyLoadBalancingSwapMutation(Individual &individual);
  bool ApplyLoadBalancingChainMutation(Individual &individual);
  bool ApplyLoadBalancingSimple(Individual &individual); // Renamed from InternalLoadBalancingMutation

  // Helper for LoadBalancingChain
  struct ChainMove {
    int customer_id;
    int from_group;
    int to_group;
    int demand;
  };
  std::pair<int, int> FindNextChainMove(int group_idx, 
                                        const std::vector<bool> &visited,
                                        const std::vector<int>& loads,
                                        const std::vector<std::vector<int>>& group_clients);

  void ApplySplitToIndividual(Individual &indiv);
  Individual ApplySREX(const Individual &p1, const Individual &p2);

  void RunDebugDiagnostics();
  void ApplySuccessionAdaptive(std::vector<Individual> &offspring_pool);
  
  
#ifdef RESEARCH
  std::vector<int> CanonicalizeGenotype(const std::vector<int> &genotype, int num_groups) const;
  
  enum class OpType {
    CROSSOVER = 0,
    MUT_AGGRESSIVE,
    MUT_SPATIAL,
    MUT_SIMPLE,
    LB_CHAIN,
    LB_SWAP,
    LB_SIMPLE,
    VND,
    COUNT
  };
  
  struct OpStat {
    std::string name;
    long long calls;
    long long wins;
  };
  std::vector<OpStat> op_stats_;
  void InitStats();
  
  // Stat counters
  long long total_evaluations = 0;
  long long catastrophy_activations = 0;
  long long crossovers = 0;
  long long load_balancing_activations = 0;
#endif
};

} // namespace LcVRPContest