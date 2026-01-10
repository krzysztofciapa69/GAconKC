#pragma once

#include "Individual.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

namespace LcVRPContest {

// trie node for exact route deduplication
struct TrieNode {
    bool is_end_of_route = false;
    std::vector<std::pair<int, std::unique_ptr<TrieNode>>> children;

    TrieNode* FindChild(int id) {
        for (auto& p : children) {
            if (p.first == id) return p.second.get();
        }
        return nullptr;
    }

    TrieNode* GetOrAddChild(int id) {
        for (auto& p : children) {
            if (p.first == id) return p.second.get();
        }
        children.emplace_back(id, std::make_unique<TrieNode>());
        return children.back().second.get();
    }
};

class RouteTrie {
public:
    RouteTrie() : root_(std::make_unique<TrieNode>()) {}

    // returns true if route was new and inserted, false if already existed
    bool Insert(const std::vector<int>& sorted_route) {
        TrieNode* curr = root_.get();
        for (int id : sorted_route) {
            curr = curr->GetOrAddChild(id);
        }
        if (curr->is_end_of_route) return false;
        curr->is_end_of_route = true;
        return true;
    }

    void Clear() { root_ = std::make_unique<TrieNode>(); }

private:
    std::unique_ptr<TrieNode> root_;
};

// cached route structure for long-term memory
struct CachedRoute {
    std::vector<int> nodes;            // customer IDs in route
    double cost;                        // total route cost
    double efficiency;                  // quality score for sorting
    std::vector<uint64_t> bitmask;     // for fast overlap check

    bool operator<(const CachedRoute& other) const {
        return efficiency < other.efficiency;
    }
};

// thread-safe route pool for long-term memory (per-island)
class RoutePool {
public:
    RoutePool() = default;

    // non-copyable and non-movable
    RoutePool(const RoutePool&) = delete;
    RoutePool& operator=(const RoutePool&) = delete;
    RoutePool(RoutePool&&) = delete;
    RoutePool& operator=(RoutePool&&) = delete;

    // add routes extracted from a solution
    void AddRoutesFromSolution(const std::vector<int>& solution,
                               const ThreadSafeEvaluator& evaluator);

    // beam search: assemble best non-overlapping routes into Individual
    Individual SolveBeamSearch(ThreadSafeEvaluator* evaluator, Split& split,
                               int beam_width = 50);

    // route migration between islands
    std::vector<CachedRoute> GetBestRoutes(int n) const;
    void ImportRoutes(const std::vector<CachedRoute>& imported_routes);

    // clear the pool
    void Clear();

    // get current pool size
    size_t GetSize() const;

    // get number of routes added since last clear
    size_t GetTotalRoutesAdded() const { return total_routes_added_; }

    // check if new routes were added since a snapshot
    bool HasNewRoutesSince(size_t snapshot) const { 
        return total_routes_added_ > snapshot; 
    }

private:
    double CalculateRouteCost(const std::vector<int>& route,
                              const ThreadSafeEvaluator& evaluator) const;
    void EvictWorstRoutes();

    mutable std::mutex mutex_;
    std::vector<CachedRoute> routes_;
    RouteTrie trie_;
    size_t total_routes_added_ = 0;
};

} // namespace LcVRPContest
