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
#include <deque>
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
  void SetRingPredecessor(Island *pred) { ring_predecessor_ = pred; }
  void TryPullMigrant(); // Pull migrant from predecessor when stuck
  int FindMostSimilarIndex(const Individual &immigrant) const;

  // Non-native broadcast: EXPLOIT islands share best solutions
  void SetExploitSiblings(std::vector<Island*> siblings) { exploit_siblings_ = siblings; }
  void ReceiveBroadcastBest(const Individual& best);  // receive broadcasted best from sibling

  double EvaluateWithHistoryPenalty(const std::vector<int> &genotype);
  void MutateIndividual(Individual &indiv);

  std::vector<int> GetBestSolution() const {
    std::lock_guard<std::mutex> lock(best_mutex_);
    return current_best_.GetGenotype();
  }

  Individual GetBestIndividual() const {
    std::lock_guard<std::mutex> lock(best_mutex_);
    return current_best_;
  }

  // Methods for inter-island communication
  Individual GetRandomIndividual();
  Individual GetMostDiverseMigrantFor(const Individual &target_best);
  Individual GetRandomEliteIndividual(); // Get random individual from top 30%

  // New method for wrapper access to Route Pool
  std::vector<CachedRoute> GetTopRoutes(int n) const;

  double GetBestFitness() const {
    std::lock_guard<std::mutex> lock(best_mutex_);
    return current_best_.GetFitness();
  }

  void PrintIndividual(const Individual &individual,
                       int global_generation) const;

  int GetId() const { return id_; }
  bool IsExploration() const { return (id_ % 2) == 0; }  // Even: 0, 2, 4
  bool IsExploitation() const { return (id_ % 2) == 1; } // Odd: 1, 3, 5

  // Returns structural diversity (0=converged, 1=chaotic)
  bool ContainsSolution(const Individual &ind) const;
  double GetCurrentCV() const { return current_structural_diversity_; }

  // Cache statistics
  long long GetCacheHits() const { return cache_hits_; }
  long long GetCacheMisses() const { return cache_misses_; }
  
  // Rolling window cache hit rate (convergence indicator)
  double GetRecentCacheHitRate() const;
  
  // Anti-convergence triggers
  void OnConvergenceWarning();   // 85-90% hit rate
  void OnConvergenceAlarm();     // 90-95% hit rate
  void OnConvergenceCritical();  // >95% hit rate

  // Get stuck individuals for inter-archipelago migration
  bool TryGetStuckIndividual(Individual &out);

  // Endgame Mode support
  void SetStartTime(std::chrono::steady_clock::time_point start_time) {
    start_time_ = start_time;
  }

  long long total_comps = 0;
  long long passes = 0;

  // Public BPD wrapper for Optimizer's diversity-pulse migration
  int CalculateBrokenPairsDistancePublic(const Individual &ind1,
                                         const Individual &ind2) {
    return CalculateBrokenPairsDistance(
        ind1, ind2, evaluator_->GetPermutation(), evaluator_->GetNumGroups());
  }

#ifdef RESEARCH
  void ExportState(int generation, bool is_catastrophe) const;
#endif

  long long getPRStats() const { return local_search_.prsucc; }

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

  // Asynchronous migration
  std::atomic<bool> is_stuck_{false}; // Flag: is island stuck
  Island *ring_predecessor_{nullptr}; // Pointer to predecessor in ring
  long long last_migration_gen_{0};   // Last generation when migrant was pulled
  static constexpr int MIGRATION_INTERVAL =
      50; // How often to check for migrants
  static constexpr int STUCK_THRESHOLD =
      500; // Stagnation threshold for acceptance

  // Non-native broadcast between EXPLOIT islands
  std::vector<Island*> exploit_siblings_;  // pointers to other EXPLOIT islands (for I1: {I3, I5})
  std::mutex broadcast_mutex_; // protects broadcast_buffer_
  std::vector<Individual> broadcast_buffer_; // Thread-safe buffer for incoming broadcasts
  
  void ProcessBroadcastBuffer(); // Process buffered broadcasts in main thread

  // Cache statistics
  long long cache_hits_ = 0;
  long long cache_misses_ = 0;
  
  // Rolling window cache tracking (convergence detection)
  static constexpr int CACHE_WINDOW_SIZE = 1000;
  std::deque<bool> cache_result_window_;  // true=hit, false=miss
  int cache_hits_in_window_ = 0;
  bool convergence_alarm_active_ = false;
  double convergence_mutation_boost_ = 1.0;  // multiplier for mutation rate during alarm
  
  void TrackCacheResult(bool was_hit);  // update rolling window

  // Diagnostic counters (reset every diagnostic interval)
  long long diag_vnd_calls_ = 0;
  long long diag_vnd_improvements_ = 0;
  long long diag_mutations_ = 0;
  long long diag_strong_mutations_ = 0;
  long long diag_crossovers_ = 0;
  long long diag_offspring_better_ = 0; // offspring better than median
  long long diag_offspring_total_ = 0;
  
  // Crossover success tracking (child better than BOTH parents)
  long long diag_srex_calls_ = 0;
  long long diag_srex_wins_ = 0;
  long long diag_neighbor_calls_ = 0;
  long long diag_neighbor_wins_ = 0;
  long long diag_pr_calls_ = 0;    // Path Relinking as crossover
  long long diag_pr_wins_ = 0;
  
  // === ADAPTIVE OPERATOR SELECTION FOR EXPLOIT ISLANDS ===
  // Uses EPSILON-GREEDY: 90% best operator, 10% random exploration
  // One operator is ALWAYS selected each VND call
  struct AdaptiveOperator {
    double success_rate = 0.5;  // running success rate (EMA)
    long long calls = 0;        // calls since last update
    long long wins = 0;         // improvements found
  };
  AdaptiveOperator adapt_swap_;       // 2-Swap (neighbor exchange)
  AdaptiveOperator adapt_ejection_;   // Ejection Chains
  AdaptiveOperator adapt_swap3_;      // 3-Swap  
  AdaptiveOperator adapt_swap4_;      // 4-Swap
  // Note: PR removed - it's already used as crossover (90% in RunGeneration)
  static constexpr double ADAPT_ALPHA = 0.2;      // EMA smoothing (faster reaction)
  static constexpr double ADAPT_EPSILON = Config::ADAPT_EPSILON;  // 25% exploration (from Config)
  
  // Update success rates (call periodically)
  void UpdateAdaptiveProbabilities();
  // Select one operator using epsilon-greedy
  int SelectAdaptiveOperator();  // returns 0=Swap, 1=Ejection, 2=3-Swap, 3=4-Swap
  
  std::chrono::steady_clock::time_point last_diag_time_;

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
  std::chrono::steady_clock::time_point
      immune_until_time_; // no migration until this time
  std::chrono::steady_clock::time_point
      last_improvement_time_; // for dynamic immunity

  // Stagnation limit - reduced cap for Exploration to prevent 10k gen sleep
  const int BASE_STAGNATION_LIMIT = 1000;
  const int MAX_EXPLORATION_STAGNATION = 3000; // Hard cap for I0, I2, I4

  // Buffers
  std::vector<int> pred1;
  std::vector<int> pred2;
  std::vector<int> last_in_group1;
  std::vector<int> last_in_group2;

  // Methods
  // CalculatePopulationCV() - REMOVED: Dead code
  void UpdateAdaptiveParameters();
  double MapRange(double value, double in_min, double in_max, double out_min,
                  double out_max) const;

  // Dynamic parameters
  int GetVndIterations() const;
  double GetMutationRate() const;
  double GetRuinChance() const;
  double GetMicrosplitChance() const;
  bool ShouldTrackDiversity() const {
    return current_generation_ % 10 == 0 ? true : false;
  }

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

  // CrossoverUniform() - REMOVED: Dead code, never called
  // CrossoverSequence() - REMOVED: Dead code, replaced by ApplySREX()
  Individual Crossover(const Individual &p1, const Individual &p2);

  Individual CrossoverNeighborBased(const Individual &p1, const Individual &p2);

  int ApplyMutation(Individual &child, bool is_endgame);
  int ApplyMicroSplitMutation(Individual &child);
  bool ApplyMergeRegret(
      Individual &ind); // dissolve 2 routes and repair with Regret-3

  bool ApplyLoadBalancingSwapMutation(Individual &individual);
  bool ApplyLoadBalancingChainMutation(Individual &individual);
  bool ApplyLoadBalancingSimple(
      Individual &individual); // Renamed from InternalLoadBalancingMutation

  // Helper for LoadBalancingChain
  struct ChainMove {
    int customer_id;
    int from_group;
    int to_group;
    int demand;
  };
  std::pair<int, int>
  FindNextChainMove(int group_idx, const std::vector<bool> &visited,
                    const std::vector<int> &loads,
                    const std::vector<std::vector<int>> &group_clients);

  void ApplySplitToIndividual(Individual &indiv);
  Individual ApplySREX(const Individual &p1, const Individual &p2);

  // RunDebugDiagnostics() - REMOVED: Dead code, never called
  void ApplySuccessionAdaptive(std::vector<Individual> &offspring_pool);

#ifdef RESEARCH
  std::vector<int> CanonicalizeGenotype(const std::vector<int> &genotype,
                                        int num_groups) const;

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

  // Track RoutePool updates for Frankenstein trigger
  size_t last_routes_added_snapshot_ = 0;
};

} // namespace LcVRPContest