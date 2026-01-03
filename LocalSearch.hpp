#pragma once
#include "Constants.hpp"
#include "Individual.hpp"
#include "ProblemGeometry.hpp"
#include "ThreadSafeEvaluator.hpp"
#include <random>
#include <vector>

namespace LcVRPContest {

// Structure to track client position in route for O(1) delta calculation
struct RoutePosition {
  int prev_client; // Previous client in route (-1 = depot start)
  int next_client; // Next client in route (-1 = depot end)
  int route_id;    // Which route (group) this client belongs to
  int position;    // Position index in route (for fast lookup)
};

class LocalSearch {
public:
  LocalSearch(ThreadSafeEvaluator *evaluator, const ProblemGeometry *geometry,
              int id);

  // Main VND methods
  bool RunVND(Individual &ind, bool heavy_mode = false);
  bool RunVND(Individual &ind, int max_iter, bool allow_swap,
              bool allow_3swap = false, bool allow_ejection = false);

  // 3-SWAP as standalone mutation (moved from VND for L2 speed)
  bool Try3Swap(std::vector<int> &genotype);
  // 4-SWAP as standalone mutation
  bool Try4Swap(std::vector<int> &genotype);

  // Set elite solution as guide for Path Relinking
  void SetGuideSolution(const std::vector<int> &guide) { guide_solution_ = guide; }
  const std::vector<int>& GetGuideSolution() const { return guide_solution_; }

  long long prsucc = 0; // Path Relinking success count
private:
  ThreadSafeEvaluator *evaluator_;
  const ProblemGeometry *geometry_;
  int id_;
  std::mt19937 rng_;

  // Fast matrix access
  const double *fast_matrix_ = nullptr;
  int matrix_dim_ = 0;

  // Working data structures
  std::vector<std::vector<int>> vnd_routes_;
  std::vector<double> vnd_loads_;
  std::vector<double> route_costs_;      // Cached route costs
  std::vector<int> max_cumulative_load_; // Max cumulative load per route (for
                                         // safe move check)

  std::vector<int> customer_ranks_;
  std::vector<int> client_indices_;
  std::vector<int> candidate_groups_;

  // Position tracking for O(1) delta calculation
  std::vector<RoutePosition> positions_;

  // Initialization
  void InitializeRanks();
  void BuildPositions(); // Build positions_ from vnd_routes_
  void
  BuildCumulativeLoads(); // Build max_cumulative_load_ for safe move detection
  void UpdatePositionsAfterMove(int client_id, int old_route, int new_route);

  // === HYBRID DELTA OPTIMIZATION ===
  // Check if inserting client into route is "safe" (won't cause new depot
  // return)
  bool IsSafeMove(int target_route, int client_id) const;
  // Fast O(1) delta for safe moves
  double CalculateFastInsertionDelta(int client_id, int target_route,
                                     int insert_pos) const;
  // Full simulation without vector modification (for unsafe moves)
  double SimulateRouteCostWithInsert(int target_route, int client_id,
                                     int insert_pos) const;
  double SimulateRouteCostWithRemoval(int source_route, int client_id) const;

  // VND strategies
  bool RunFullVND(Individual &ind, bool allow_swap);
  bool RunDecomposedVND(Individual &ind, bool allow_swap);

  // Main optimization loop
  bool OptimizeActiveSet(Individual &ind, int max_iter, bool allow_swap,
                         bool allow_3swap, bool allow_ejection = false);
  
  // Ejection Chain operator (multi-hop client relocation)
  bool TryEjectionChain(std::vector<int> &genotype, int start_client_idx, 
                        int max_depth = 3);
  
  // Path Relinking: explore trajectory between current solution and guide
  // Returns true if found improvement, updates genotype and best_cost
  bool TryPathRelinking(std::vector<int> &genotype, double &current_cost,
                        const std::vector<int> &guide_solution);
  
  // Guide solution for Path Relinking
  std::vector<int> guide_solution_;
  
  std::vector<bool> dlb_;

  // === DELTA EVALUATION (O(1)) ===
  // Calculate cost change when removing client from its current position
  double CalculateRemovalDelta(int client_id) const;

  // Calculate cost change when inserting client at best position in target
  // route Returns the delta and sets best_insert_pos
  double CalculateInsertionDelta(int client_id, int target_route,
                                 int &best_insert_pos) const;

  // Full simulation only when capacity overflow possible
  double SimulateRouteCost(const std::vector<int> &route_nodes) const;

  // Check if move causes capacity overflow
  bool WouldOverflow(int target_route, int client_id) const;

  void ResetDLB();

  // Legacy helper methods (kept for compatibility)
  int FindInsertionIndexBinary(const std::vector<int> &route,
                               int target_rank) const;
  double CalculateRemovalDelta(const std::vector<int> &route,
                               int client_id) const;
  double CalculateInsertionDelta(const std::vector<int> &route,
                                 int client_id) const;
};
} // namespace LcVRPContest
