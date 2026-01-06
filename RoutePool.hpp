#pragma once

#include "Individual.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>


namespace LcVRPContest {

// Trie Node for exact route deduplication
struct TrieNode {
  bool is_end_of_route = false;
  // Using map for flexibility, or vector if IDs are small/dense.
  // Given IDs can be up to N, map is safer for sparse children.
  // For critical perf, a sorted vector of children pairs could be better,
  // but map is O(log C) which is fine for route construction.
  // actually, since we sort the route, we just walk down.
  std::vector<std::pair<int, std::unique_ptr<TrieNode>>> children;

  TrieNode *FindChild(int id) {
    for (auto &p : children) {
      if (p.first == id)
        return p.second.get();
    }
    return nullptr;
  }

  TrieNode *GetOrAddChild(int id) {
    for (auto &p : children) {
      if (p.first == id)
        return p.second.get();
    }
    children.emplace_back(id, std::make_unique<TrieNode>());
    return children.back().second.get();
  }
};

class RouteTrie {
public:
  RouteTrie() : root_(std::make_unique<TrieNode>()) {}

  // Returns true if route was new and inserted, false if already existed
  bool Insert(const std::vector<int> &sorted_route) {
    TrieNode *curr = root_.get();
    for (int id : sorted_route) {
      curr = curr->GetOrAddChild(id);
    }
    if (curr->is_end_of_route) {
      return false; // Already existed
    }
    curr->is_end_of_route = true;
    return true;
  }

  void Clear() { root_ = std::make_unique<TrieNode>(); }

private:
  std::unique_ptr<TrieNode> root_;
};

// Cached route structure for long-term memory
struct CachedRoute {
  std::vector<int>
      nodes; // Customer IDs (sorted for Trie, but original order kept if
             // needed? Actually RoutePool stores them as set for assembly,
             // order re-optimized by Split. So sorted is fine, but Split needs
             // demands. Wait, existing code stores them in `nodes`.
             // AddRoutesFromSolution builds them. Let's store sorted for
             // consistency or just keep them as is and sort only for Trie?
             // Existing `SolveGreedy` uses them for set packing. Order doesn't
             // matter for packing. Order matters for "efficiency" calc maybe?
             // No, CalcRouteCost does re-eval. Storing canonical (sorted) nodes
             // is safer for "Set" semantics.)

  double cost;       // Total route cost
  double efficiency; // Quality score: (cost * compactness) / (size^1.3 * load_ratio^0.5)
  // size_t hash;             // Removed: Trie handles deduplication

  std::vector<uint64_t> bitmask; // For fast overlap check

  bool operator<(const CachedRoute &other) const {
    return efficiency < other.efficiency;
  }
};

// Thread-safe Route Pool for long-term memory
// Now instantiated per-island instead of global singleton
class RoutePool {
public:
  RoutePool() = default;

  // Delete copy/move constructors for safety if needed, or allow them.
  // Given it holds mutex and unique_ptrs, it's non-copyable by default due to unique_ptr?
  // Actually unique_ptr is move-only. Mutex is non-copyable and non-movable.
  // So RoutePool is non-copyable and non-movable unless we define custom move.
  // For now, let's keep it non-copyable/non-movable to be safe.
  RoutePool(const RoutePool &) = delete;
  RoutePool &operator=(const RoutePool &) = delete;
  RoutePool(RoutePool &&) = delete;
  RoutePool &operator=(RoutePool &&) = delete;

  // Add routes extracted from a solution
  void AddRoutesFromSolution(const std::vector<int> &solution,
                             const ThreadSafeEvaluator &evaluator);

  // Greedy Set Packing: Assemble best non-overlapping routes into Individual
  Individual SolveGreedy(ThreadSafeEvaluator *evaluator, Split &split);

  // Beam Search: Improved assembly using multi-path exploration
  Individual SolveBeamSearch(ThreadSafeEvaluator *evaluator, Split &split, 
                             int beam_width = 50);

  // Clear the pool
  void Clear();

  // Get current pool size
  size_t GetSize() const;

  // Get number of routes added since last clear
  size_t GetTotalRoutesAdded() const { return total_routes_added_; }

  // Check if new routes were added since a snapshot
  bool HasNewRoutesSince(size_t snapshot) const { return total_routes_added_ > snapshot; }

private:
  // Calculate route cost using evaluator logic (single route)
  double CalculateRouteCost(const std::vector<int> &route,
                            const ThreadSafeEvaluator &evaluator) const;

  void EvictWorstRoutes();

  mutable std::mutex mutex_;
  std::vector<CachedRoute> routes_;
  RouteTrie trie_; 
  size_t total_routes_added_ = 0;
};

} // namespace LcVRPContest
