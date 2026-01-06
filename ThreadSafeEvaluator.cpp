#include "ThreadSafeEvaluator.hpp"
#include "Individual.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <cstdint>


using namespace std;

namespace LcVRPContest {

static const double WRONG_VAL = 1e9;

ThreadSafeEvaluator::ThreadSafeEvaluator(const ProblemData &data,
                                         int num_groups)
    : num_groups_(num_groups), num_customers_(data.GetNumCustomers()),
      dimension_(data.GetDimension()), capacity_(data.GetCapacity()),
      depot_index_(data.GetDepot() - 1),
      has_distance_constraint_(data.HasDistanceConstraint()),
      max_distance_(data.GetDistance()), demands_(data.GetDemands()),
      problem_data_(&data), permutation_(data.GetPermutation()) {
  coordinates_ = data.GetCoordinates();

  // Initialize Cache and Zobrist Hashes
  InitCache();

  if (dimension_ <= MATRIX_THRESHOLD) {
    use_matrix_ = true;
    const auto &edge_weights = data.GetEdgeWeights();
    fast_distance_matrix_.resize(dimension_ * dimension_, WRONG_VAL);

    for (int i = 0; i < dimension_; ++i) {
      for (int j = 0; j < dimension_; ++j) {
        double dist = WRONG_VAL;
        if (data.GetEdgeWeightType() == "EXPLICIT") {
          if (i < (int)edge_weights.size() && j < (int)edge_weights[i].size()) {
            dist = edge_weights[i][j];
          }
        } else {
          if (i == j)
            dist = 0.0;
          else {
            double dx = coordinates_[i].x - coordinates_[j].x;
            double dy = coordinates_[i].y - coordinates_[j].y;
            dist = std::sqrt(dx * dx + dy * dy);
          }
        }
        fast_distance_matrix_[i * dimension_ + j] = dist;
      }
    }
  } else {
    use_matrix_ = false;
  }
}

void ThreadSafeEvaluator::InitCache() {
    route_cache_.resize(CACHE_SIZE);
    
    // Initialize Zobrist hashes deterministically but random-like
    std::mt19937_64 rng(123456767); // Fixed seed for reproducibility across threads/runs
    customer_hashes_.resize(num_customers_ + 2); // +2 for safety (1-based + depot buffer)
    for(auto& h : customer_hashes_) {
        h = rng();
    }
}

double ThreadSafeEvaluator::Evaluate(const std::vector<int> &solution) const {
  if ((int)solution.size() != num_customers_) {
    return WRONG_VAL;
  }
  return EvaluateWithStats(solution).fitness;
}

int ThreadSafeEvaluator::GetTotalDepotReturns(
    const std::vector<int> &solution) const {
  return EvaluateWithStats(solution).returns;
}

EvaluationResult
ThreadSafeEvaluator::EvaluateWithStats(const std::vector<int> &solution) const {
  if (solution.empty())
    return {WRONG_VAL, 0};

  double total_dist = 0.0;
  int total_returns = 0;

  // Working structures
  // Ideally these should be members to avoid allocation, but we need thread-locality logic if shared.
  // Since this Evaluator is thread-local (passed so in Optimizer), we *could* make them members,
  // but let's stick to vector on stack for safety unless perf profile says otherwise.
  // Actually, allocation here is O(Groups), which is small (20-50).
  std::vector<uint64_t> group_hashes(num_groups_, 0);
  std::vector<bool> group_is_cached(num_groups_, false);

  const int *sol_ptr = solution.data();
  int sol_size = (int)solution.size();

  // --- PASS 1: Calculate XOR Hashes for all groups ---
  for (int customer_id : permutation_) {
    if (customer_id == (depot_index_ + 1)) continue;

    int gene_idx = customer_id - 2;
    if (gene_idx < 0 || gene_idx >= sol_size) continue;

    int g = sol_ptr[gene_idx];
    if (g >= 0 && g < num_groups_) {
        group_hashes[g] ^= customer_hashes_[customer_id];
    }
  }

  // --- PASS 1.5: Check Cache ---
  for(int g = 0; g < num_groups_; ++g) {
      if(group_hashes[g] == 0) {
          // Empty group cost is 0
          group_is_cached[g] = true;
          continue; 
      }
      
      uint64_t key = group_hashes[g];
      size_t idx = key & CACHE_MASK;
      const auto& entry = route_cache_[idx];
      
      if(entry.occupied && entry.key == key) {
          // HIT
          total_dist += entry.cost;
          total_returns += entry.returns;
          group_is_cached[g] = true;
          route_cache_hits_++;
      }
  }

  // --- PASS 2: Compute Cost for NON-CACHED groups ---
  
  // State variables for computation
  std::vector<int> group_load(num_groups_, 0);
  std::vector<double> group_dist(num_groups_, 0.0);
  std::vector<int> group_last(num_groups_, depot_index_);
  std::vector<int> group_returns(num_groups_, 0);

  for (int customer_id : permutation_) {
    if (customer_id == (depot_index_ + 1)) continue;

    int gene_idx = customer_id - 2;
    if (gene_idx < 0 || gene_idx >= sol_size) continue;

    int g = sol_ptr[gene_idx];
    if (g < 0 || g >= num_groups_) return {WRONG_VAL, 0};

    // SKIP CACHED GROUPS
    if (group_is_cached[g]) continue;

    // Normal Evaluation Logic for group 'g'
    int matrix_idx = customer_id - 1;
    int demand = demands_[matrix_idx];

    if (group_load[g] + demand > capacity_) {
      group_returns[g]++;
      group_dist[g] += GetDist(group_last[g], depot_index_);
      group_load[g] = 0;
      // Reset logic for return... actually just add return leg
      // Here "group_dist" tracks running distance for constraints
      // But "total_dist" is what we sum up. 
      // Wait, original logic accumulated into 'total_dist' AND 'group_dist'.
      // Here we accumulate into 'group_dist' (total for this group) and 'group_returns'.
      // We must check if 'group_dist' was meant to be "cost so far since depot".
      // Original: 
      //   group_dist[g] = 0.0; (reset after return)
      group_dist[g] = 0.0; 
      group_last[g] = depot_index_;
    }

    double d_travel = GetDist(group_last[g], matrix_idx);

    if (has_distance_constraint_) {
      double d_return = GetDist(matrix_idx, depot_index_);
      // Note: group_dist[g] here represents distance from last depot departure
      if (group_dist[g] + d_travel + d_return > max_distance_) {
        if (group_last[g] != depot_index_) {
          group_returns[g]++;
          group_dist[g] += GetDist(group_last[g], depot_index_); // Cost of returning
          group_load[g] = 0;
          group_dist[g] = 0.0;
          group_last[g] = depot_index_;
        }
        d_travel = GetDist(depot_index_, matrix_idx);
      }
    }

    group_dist[g] += d_travel;
    group_load[g] += demand;
    group_last[g] = matrix_idx;
  }

  // Finalize non-cached groups and update cache
  for (int g = 0; g < num_groups_; ++g) {
      if (group_is_cached[g]) continue;

      // Finish the route (return to depot)
      if (group_last[g] != depot_index_) {
          group_dist[g] += GetDist(group_last[g], depot_index_);
      }

      // 'group_dist[g]' now holds the sum of all segments?
      // Wait, in the loop: 
      //   group_dist[g] = 0.0 on reset.
      //   group_dist[g] += d_travel.
      //   total_dist += ... in original code.
      // 
      // Correct Logic: 'group_dist' in original code was reset on returns. 
      // It tracked "current active route segment distance".
      // 'total_dist' accumulated EVERYTHING.
      //
      // In this new Pass 2, we need to capture the Total Cost of the Group.
      // Let's use a separate accumulator `group_total_cost[num_groups_]`.
      // Re-reading logic carefully:
      // Original code: `total_dist += ...` at every step.
      // We can't use `total_dist` easily here because it's global.
      // We need per-group total cost.
      
      // Let's fix this logic inside the loop above.
      // Actually, I cannot easily edit the previous loop block in this replacement string without making it huge.
      // But wait, I AM rewriting the whole function.
  }
  
  // --- RE-IMPLEMENTING PASS 2 WITH CORRECT ACCUMULATION ---
  // (Redoing the loop logic above implicitly for correctness)
  
  std::vector<double> group_total_cost(num_groups_, 0.0);
  // Re-init variables for clarity (compiler optimizes)
  std::fill(group_load.begin(), group_load.end(), 0);
  std::fill(group_dist.begin(), group_dist.end(), 0.0); // This is "segment dist"
  std::fill(group_last.begin(), group_last.end(), depot_index_);
  std::fill(group_returns.begin(), group_returns.end(), 0);

  for (int customer_id : permutation_) {
    if (customer_id == (depot_index_ + 1)) continue;

    int gene_idx = customer_id - 2;
    if (gene_idx < 0 || gene_idx >= sol_size) continue;

    int g = sol_ptr[gene_idx];
    // if g invalid... handled above.

    if (group_is_cached[g]) continue;

    // --- Logic Start ---
    int matrix_idx = customer_id - 1;
    int demand = demands_[matrix_idx];

    // Capacity Check
    if (group_load[g] + demand > capacity_) {
      group_returns[g]++;
      double r_cost = GetDist(group_last[g], depot_index_);
      group_total_cost[g] += r_cost;
      
      group_load[g] = 0;
      group_dist[g] = 0.0;
      group_last[g] = depot_index_;
    }

    double d_travel = GetDist(group_last[g], matrix_idx);

    // Distance Constraint Check
    if (has_distance_constraint_) {
      double d_return = GetDist(matrix_idx, depot_index_);
      if (group_dist[g] + d_travel + d_return > max_distance_) {
        if (group_last[g] != depot_index_) {
          group_returns[g]++;
          double r_cost = GetDist(group_last[g], depot_index_);
          group_total_cost[g] += r_cost;

          group_load[g] = 0;
          group_dist[g] = 0.0;
          group_last[g] = depot_index_;
        }
        d_travel = GetDist(depot_index_, matrix_idx);
      }
    }

    group_total_cost[g] += d_travel;
    group_dist[g] += d_travel;
    group_load[g] += demand;
    group_last[g] = matrix_idx;
  }

  // Finalize Calculation and Update Cache
  for (int g = 0; g < num_groups_; ++g) {
      if (group_is_cached[g]) continue;

      // Miss recorded here as we finished processing a non-cached group
      route_cache_misses_++;

      // Return to depot
      if (group_last[g] != depot_index_) {
          group_total_cost[g] += GetDist(group_last[g], depot_index_);
      }

      // Add to global totals
      total_dist += group_total_cost[g];
      total_returns += group_returns[g];

      // Update Cache (if not empty / 0 hash)
      if(group_hashes[g] != 0) {
          size_t idx = group_hashes[g] & CACHE_MASK;
          // Simple overwrite strategy
          route_cache_[idx].key = group_hashes[g];
          route_cache_[idx].cost = group_total_cost[g];
          route_cache_[idx].returns = group_returns[g];
          route_cache_[idx].occupied = true;
      }
  }

  return {total_dist, total_returns};
}

// --- Cached Route Evaluation ---
double ThreadSafeEvaluator::GetRouteCost(const std::vector<int> &route_nodes) const {
    if (route_nodes.empty()) return 0.0;

    // 1. Calculate XOR Hash
    uint64_t key = 0;
    for (int customer_id : route_nodes) {
        // Safe access to customer_hashes_
        int idx = customer_id - 1; // 0-based index for hashing
        if (idx >= 0 && idx < (int)customer_hashes_.size()) {
            key ^= customer_hashes_[idx];
        }
    }

    // 2. Check Cache
    size_t cache_idx = key & CACHE_MASK;
    const auto& entry = route_cache_[cache_idx];

    if (entry.occupied && entry.key == key) {
        route_cache_hits_++;
        return entry.cost;
    }

    route_cache_misses_++;

    // 3. Compute Cost (Fallback - with FULL constraint logic)
    double total_cost = 0.0;
    int current_load = 0;
    double current_segment_dist = 0.0; // Distance since last depot departure
    int last_node_idx = depot_index_;
    int returns_count = 0;

    for (int customer_id : route_nodes) {
        int matrix_idx = customer_id - 1;
        int demand = demands_[matrix_idx];

        // Capacity Check
        if (current_load + demand > capacity_) {
             returns_count++;
             total_cost += GetDist(last_node_idx, depot_index_);
             last_node_idx = depot_index_;
             current_load = 0;
             current_segment_dist = 0.0;
        }

        double d_travel = GetDist(last_node_idx, matrix_idx);

        // Distance Constraint Check
        if (has_distance_constraint_) {
             double d_return = GetDist(matrix_idx, depot_index_);
             if (current_segment_dist + d_travel + d_return > max_distance_) {
                 if (last_node_idx != depot_index_) {
                     returns_count++;
                     total_cost += GetDist(last_node_idx, depot_index_);
                     last_node_idx = depot_index_;
                     current_load = 0;
                     current_segment_dist = 0.0;
                     d_travel = GetDist(depot_index_, matrix_idx);
                 }
             }
        }

        total_cost += d_travel;
        current_segment_dist += d_travel;
        current_load += demand;
        last_node_idx = matrix_idx;
    }

    // Return to depot
    if (last_node_idx != depot_index_) {
        returns_count++; // Final return to depot
        total_cost += GetDist(last_node_idx, depot_index_);
    }


    
    // Store in cache
    route_cache_[cache_idx] = {key, total_cost, returns_count, true};

    return total_cost;
}

} // namespace LcVRPContest