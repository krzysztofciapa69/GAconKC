#pragma once
#include "Individual.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace LcVRPContest {

// Cached route structure for efficient beam search
struct CachedRoute {
    std::vector<int> nodes;          // customer IDs in visit order
    double cost = 0.0;               // total route distance
    double efficiency = 0.0;         // efficiency score (lower = better)
    uint64_t hash = 0;               // hash for O(1) deduplication
    std::vector<uint64_t> bitmask;   // customer coverage bitmask

    // Comparison operator for sorting (by efficiency)
    bool operator<(const CachedRoute& other) const {
        return efficiency < other.efficiency;
    }
};

// RoutePool: stores high-quality route segments for Frankenstein (beam search) construction
class RoutePool {
public:
    // Add routes from a complete solution
    void AddRoutesFromSolution(const std::vector<int>& solution,
                               const ThreadSafeEvaluator& evaluator);

    // Build new individual using beam search from cached routes
    Individual SolveBeamSearch(ThreadSafeEvaluator* evaluator,
                               Split& split, int beam_width);

    // Import routes from another pool (for inter-island migration)
    void ImportRoutes(const std::vector<CachedRoute>& imported_routes);

    // Get best n routes (for migration export)
    std::vector<CachedRoute> GetBestRoutes(int n) const;

    // Pool management
    void Clear();
    size_t GetSize() const;

    // Check if new routes were added since last snapshot
    bool HasNewRoutesSince(size_t snapshot) const {
        return total_routes_added_ > snapshot;
    }

    // Get total routes added (for snapshot comparison)
    size_t GetTotalRoutesAdded() const {
        return total_routes_added_;
    }

private:
    // Calculate cost for a single route
    double CalculateRouteCost(const std::vector<int>& route,
                              const ThreadSafeEvaluator& evaluator) const;

    // Hash a sorted route for O(1) deduplication
    uint64_t HashRoute(const std::vector<int>& sorted_route) const;

    // Evict worst routes when pool is full
    void EvictWorstRoutes();

    std::vector<CachedRoute> routes_;
    std::unordered_set<uint64_t> route_hashes_;  // O(1) deduplication
    mutable std::mutex mutex_;
    size_t total_routes_added_ = 0;
};

} // namespace LcVRPContest
