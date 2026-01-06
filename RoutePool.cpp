#include "RoutePool.hpp"
#include "Constants.hpp"
#include <cmath>
#include <cstdio>
#include <random>

using namespace std;

namespace LcVRPContest {

    // ComputeRouteHash removed (replaced by Trie)

    double
        RoutePool::CalculateRouteCost(const std::vector<int>& route,
            const ThreadSafeEvaluator& evaluator) const {
        if (route.empty())
            return 0.0;

        const int depot_index = 0; // Depot is always index 0 (customer_id 1)
        const int capacity = evaluator.GetCapacity();
        const double max_distance = evaluator.GetProblemData().HasDistanceConstraint()
            ? evaluator.GetProblemData().GetDistance()
            : -1.0;

        double total_cost = 0.0;
        int current_load = 0;
        double current_distance = 0.0;
        int last_pos = depot_index;

        for (int customer_id : route) {
            int matrix_idx = customer_id - 1; // Convert to 0-based index
            int demand = static_cast<int>(evaluator.GetDemand(customer_id));

            // Check capacity constraint
            if (current_load + demand > capacity) {
                // Return to depot
                total_cost += current_distance + evaluator.GetDist(last_pos, depot_index);
                current_load = 0;
                current_distance = 0.0;
                last_pos = depot_index;
            }

            double dist_to_customer = evaluator.GetDist(last_pos, matrix_idx);

            // Check distance constraint
            if (max_distance > 0.0) {
                double dist_back = evaluator.GetDist(matrix_idx, depot_index);
                if (current_distance + dist_to_customer + dist_back > max_distance) {
                    if (last_pos != depot_index) {
                        total_cost +=
                            current_distance + evaluator.GetDist(last_pos, depot_index);
                        current_load = 0;
                        current_distance = 0.0;
                        last_pos = depot_index;
                    }
                    dist_to_customer = evaluator.GetDist(depot_index, matrix_idx);
                }
            }

            current_distance += dist_to_customer;
            current_load += demand;
            last_pos = matrix_idx;
        }

        // Final return to depot
        total_cost += current_distance + evaluator.GetDist(last_pos, depot_index);

        return total_cost;
    }

    void RoutePool::AddRoutesFromSolution(const std::vector<int>& solution,
        const ThreadSafeEvaluator& evaluator) {
        const auto& permutation = evaluator.GetPermutation();
        int num_groups = evaluator.GetNumGroups();
        int num_clients = evaluator.GetSolutionSize();

        // Build routes per group
        std::vector<std::vector<int>> group_routes(num_groups);

        for (int customer_id : permutation) {
            int gene_idx = customer_id - 2; // customer_id 2 -> index 0
            if (gene_idx < 0 || gene_idx >= (int)solution.size())
                continue;

            int group = solution[gene_idx];
            if (group >= 0 && group < num_groups) {
                group_routes[group].push_back(customer_id);
            }
        }

        // Add each non-trivial route to pool
        std::lock_guard<std::mutex> lock(mutex_);

        // Bitmask settings
        // Calculate mask size: (num_clients + 63) / 64
        // Indices 0..num_clients-1 represent customer_id 2..N+1
        size_t mask_words = (num_clients + 63) / 64;

        const int capacity = evaluator.GetCapacity();

        int added = 0;
        for (const auto& route : group_routes) {
            if (route.size() < Config::MIN_ROUTE_SIZE_FOR_POOL)
                continue;

            // Create sorted copy for Trie insertion (Set semantics)
            std::vector<int> sorted_route = route;
            std::sort(sorted_route.begin(), sorted_route.end());

            // TRIE deduplication
            if (!trie_.Insert(sorted_route)) {
                continue; // Already exists
            }

            double cost = CalculateRouteCost(route, evaluator);

            // === NEW QUALITY SCORE CALCULATION ===
            // 1. Calculate centroid of the route (exclude depot)
            double centroid_x = 0.0;
            double centroid_y = 0.0;
            int route_load = 0;

            for (int customer_id : route) {
                int matrix_idx = customer_id - 1; // customer_id to 0-based matrix index
                const auto& coord = evaluator.GetCoordinate(matrix_idx);
                centroid_x += coord.x;
                centroid_y += coord.y;
                route_load += static_cast<int>(evaluator.GetDemand(customer_id));
            }

            double route_size = static_cast<double>(route.size());
            centroid_x /= route_size;
            centroid_y /= route_size;

            // 2. Calculate mean distance to centroid (spatial compactness)
            double sum_dist_to_centroid = 0.0;
            for (int customer_id : route) {
                int matrix_idx = customer_id - 1;
                const auto& coord = evaluator.GetCoordinate(matrix_idx);
                double dx = coord.x - centroid_x;
                double dy = coord.y - centroid_y;
                sum_dist_to_centroid += std::sqrt(dx * dx + dy * dy);
            }
            double mean_dist_to_centroid = sum_dist_to_centroid / route_size;
            // Avoid zero in numerator - minimum compactness penalty
            if (mean_dist_to_centroid < 1.0) {
                mean_dist_to_centroid = 1.0;
            }

            // 3. Calculate load factor (capacity utilization)
            double load_ratio = static_cast<double>(route_load) / static_cast<double>(capacity);
            // Clamp load_ratio to avoid division issues, ensure at least small value
            if (load_ratio < 0.1) {
                load_ratio = 0.1;
            }
            if (load_ratio > 1.0) {
                load_ratio = 1.0; // Cap at 100% utilization
            }

            // 4. Compute final efficiency score
            // Lower is better: good features (size, load_ratio) in denominator
            //                  bad features (cost, spread) in numerator
            double efficiency = (cost * mean_dist_to_centroid) /
                (std::pow(route_size, 1.3) * std::pow(load_ratio, 0.5));

            CachedRoute cached;
            cached.nodes = route;
            cached.cost = cost;
            cached.efficiency = efficiency;

            // Generate Bitmask
            cached.bitmask.assign(mask_words, 0);
            for (int cid : route) {
                if (cid < 2)
                    continue; // Depot?
                int idx = cid - 2;
                if (idx < num_clients) {
                    cached.bitmask[idx / 64] |= (1ULL << (idx % 64));
                }
            }

            routes_.push_back(std::move(cached));
            total_routes_added_++;
            added++;
        }

        // Evict if over capacity
        if (routes_.size() > Config::ROUTE_POOL_MAX_SIZE) {
            EvictWorstRoutes();
        }
    }

    void RoutePool::EvictWorstRoutes() {
        // Sort by efficiency (worst = highest efficiency at end)
        std::sort(routes_.begin(), routes_.end());

        // Keep only top ROUTE_POOL_MAX_SIZE * 0.9 routes (remove 10%)
        size_t target_size = static_cast<size_t>(Config::ROUTE_POOL_MAX_SIZE * 0.9);

        // We should ideally remove evicted routes from Trie, but removing from Trie
        // is complex (reference counting or re-building).
        // Given this is a cache, implementing "Clear + Rebuild Trie" is simpler/safer
        // or implementing logical delete.
        // STRATEGY: Clear Trie and re-insert remaining survivors.

        trie_.Clear();

        routes_.resize(target_size);

        // Rebuild Trie from survivors
        for (const auto& r : routes_) {
            // We need sorted version for Trie
            std::vector<int> sorted = r.nodes;
            std::sort(sorted.begin(), sorted.end());
            trie_.Insert(sorted);
        }
    }

    Individual RoutePool::SolveGreedy(ThreadSafeEvaluator* evaluator,
        Split& split) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (routes_.empty()) {
            return Individual();
        }

        int num_customers = evaluator->GetSolutionSize();
        int num_groups = evaluator->GetNumGroups();
        size_t mask_words = (num_customers + 63) / 64;
        const auto& perm = evaluator->GetPermutation();

        // === MULTI-START GREEDY ===
        const int NUM_STARTS = Config::GREEDY_NUM_STARTS;
        std::mt19937 rng(std::random_device{}());

        Individual best_frankenstein;
        double best_fitness = std::numeric_limits<double>::max();
        int best_routes_used = 0;
        int best_customers_covered = 0;
        int best_leftovers_count = 0;

        for (int start_iter = 0; start_iter < NUM_STARTS; ++start_iter) {
            // Create working copy of routes
            std::vector<CachedRoute> working_routes = routes_;

            // Sort by efficiency (best first)
            std::sort(working_routes.begin(), working_routes.end());

            // For iterations > 0, shuffle top 30% of routes to explore different combinations
            if (start_iter > 0 && working_routes.size() > 10) {
                size_t shuffle_range = std::max(size_t(10), working_routes.size() * 30 / 100);
                std::shuffle(working_routes.begin(),
                    working_routes.begin() + shuffle_range,
                    rng);
            }

            // Reset state for this iteration
            std::vector<int> result_genotype(num_customers, -1);
            std::vector<uint64_t> current_mask(mask_words, 0);
            std::vector<int> result_permutation;
            result_permutation.reserve(num_customers);

            int current_group = 0;
            int routes_used = 0;
            int customers_covered = 0;

            // Greedy Set Packing: select non-overlapping routes
            for (const auto& route : working_routes) {
                if (current_group >= num_groups - 1)
                    break; // Leave one group for leftovers

                // BITMASK OVERLAP CHECK
                bool has_overlap = false;
                if (route.bitmask.size() != mask_words) {
                    continue;
                }

                for (size_t i = 0; i < mask_words; ++i) {
                    if (current_mask[i] & route.bitmask[i]) {
                        has_overlap = true;
                        break;
                    }
                }

                if (has_overlap)
                    continue;

                // Add this route - Update Mask
                for (size_t i = 0; i < mask_words; ++i) {
                    current_mask[i] |= route.bitmask[i];
                }

                // Fill result vars
                for (int customer_id : route.nodes) {
                    if (customer_id < 2)
                        continue;

                    int gene_idx = customer_id - 2;
                    if (gene_idx >= 0 && gene_idx < num_customers) {
                        result_genotype[gene_idx] = current_group;
                        result_permutation.push_back(customer_id);
                        customers_covered++;
                    }
                }

                current_group++;
                routes_used++;
            }

            // Handle unassigned customers (leftovers) with Nearest Neighbor sorting
            std::vector<int> leftovers_vec;
            for (int customer_id : perm) {
                if (customer_id < 2)
                    continue;
                int gene_idx = customer_id - 2;
                bool assigned = false;
                if (gene_idx < num_customers) {
                    assigned = (current_mask[gene_idx / 64] & (1ULL << (gene_idx % 64)));
                }
                if (!assigned) {
                    leftovers_vec.push_back(customer_id);
                }
            }

            int leftovers_count = 0;

            if (!leftovers_vec.empty()) {
                // === IMPROVED: Distribute leftovers across existing groups ===
                // Instead of putting all in one group, find best group for each leftover

                // Build current group loads
                std::vector<int> group_loads(num_groups, 0);
                for (int i = 0; i < num_customers; ++i) {
                    if (result_genotype[i] >= 0 && result_genotype[i] < num_groups) {
                        int customer_id = i + 2;
                        group_loads[result_genotype[i]] += static_cast<int>(evaluator->GetDemand(customer_id));
                    }
                }
                int capacity = evaluator->GetCapacity();

                // Sort leftovers by demand (largest first - bin packing heuristic)
                std::sort(leftovers_vec.begin(), leftovers_vec.end(),
                    [&](int a, int b) {
                        return evaluator->GetDemand(a) > evaluator->GetDemand(b);
                    });

                // Assign each leftover to best fitting group
                for (int customer_id : leftovers_vec) {
                    int gene_idx = customer_id - 2;
                    if (gene_idx < 0 || gene_idx >= num_customers) continue;

                    int demand = static_cast<int>(evaluator->GetDemand(customer_id));
                    int best_group = -1;
                    int min_slack = std::numeric_limits<int>::max();

                    // Find group with smallest slack that can fit this customer
                    for (int g = 0; g < num_groups; ++g) {
                        int slack = capacity - group_loads[g];
                        if (slack >= demand && slack < min_slack) {
                            min_slack = slack;
                            best_group = g;
                        }
                    }

                    // Fallback: if no group can fit, use the one with most slack
                    if (best_group == -1) {
                        int max_slack = -1;
                        for (int g = 0; g < num_groups; ++g) {
                            int slack = capacity - group_loads[g];
                            if (slack > max_slack) {
                                max_slack = slack;
                                best_group = g;
                            }
                        }
                    }

                    // Final fallback
                    if (best_group == -1) {
                        best_group = current_group;
                    }

                    result_genotype[gene_idx] = best_group;
                    group_loads[best_group] += demand;
                    result_permutation.push_back(customer_id);
                    leftovers_count++;
                }
            }

            // Validate genotype
            for (int i = 0; i < num_customers; ++i) {
                if (result_genotype[i] < 0) {
                    result_genotype[i] = current_group;
                }
            }

            // Create individual and run Split to optimize
            Individual frankenstein(result_genotype);

            // Run Split to re-optimize the grouping
            SplitResult split_result = split.RunLinear(result_permutation);
            if (!split_result.group_assignment.empty()) {
                for (int& g : split_result.group_assignment) {
                    if (g >= num_groups) {
                        g = g % num_groups;
                    }
                }
                frankenstein = Individual(split_result.group_assignment);
            }

            // Evaluate
            double fitness = evaluator->Evaluate(frankenstein.GetGenotype());
            frankenstein.SetFitness(fitness);

            // Keep best
            if (fitness < best_fitness) {
                best_fitness = fitness;
                best_frankenstein = frankenstein;
                best_routes_used = routes_used;
                best_customers_covered = customers_covered;
                best_leftovers_count = leftovers_count;
            }
        }

        // Enhanced diagnostics
        printf(" [GREEDY] multi-start=%d, routes=%d, covered=%d, leftovers=%d, fit=%.2f, pool=%zu\n",
            NUM_STARTS, best_routes_used, best_customers_covered, best_leftovers_count,
            best_fitness, routes_.size());

        return best_frankenstein;
    }

    void RoutePool::Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        routes_.clear();
        trie_.Clear();
        total_routes_added_ = 0;
    }

    size_t RoutePool::GetSize() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return routes_.size();
    }

    // ============================================================================
    // BEAM SEARCH IMPLEMENTATION
    // ============================================================================

    // BeamNode structure for multi-path exploration
    struct BeamNode {
        double total_cost = 0.0;
        double efficiency_sum = 0.0;
        std::vector<int> route_indices;
        std::vector<uint64_t> mask;
        int last_route_idx = -1;
        int customers_covered = 0;

        bool operator<(const BeamNode& other) const {
            // Maximize coverage first, then minimize cost
            if (customers_covered != other.customers_covered)
                return customers_covered > other.customers_covered;
            return total_cost < other.total_cost;
        }
    };
    Individual RoutePool::SolveBeamSearch(ThreadSafeEvaluator* evaluator,
        Split& split, int beam_width) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (routes_.empty()) {
            return Individual();
        }

        int num_customers = evaluator->GetSolutionSize();
        int num_groups = evaluator->GetNumGroups();
        size_t mask_words = (num_customers + 63) / 64;
        const auto& perm = evaluator->GetPermutation();

        // 1. Prepare sorted routes (best efficiency first)
        std::vector<CachedRoute> sorted_routes = routes_;
        std::sort(sorted_routes.begin(), sorted_routes.end());

        // 2. Initialize Beam
        std::vector<BeamNode> current_beam;
        BeamNode root;
        root.mask.assign(mask_words, 0);
        current_beam.push_back(root);

        // USTAWIENIA - POPRAWIONE
        int max_depth = num_groups + 2;
        int expansion_limit = 500;

        // 3. Beam Search Loop
        for (int depth = 0; depth < max_depth; ++depth) {
            std::vector<BeamNode> next_beam;
            bool any_expansion = false;

            for (const auto& node : current_beam) {
                if (node.customers_covered >= num_customers) {
                    next_beam.push_back(node);
                    continue;
                }

                int checked_count = 0;
                int added_count = 0;

                // POPRAWKA 1: Jeœli jesteœmy na pocz¹tku (pusty wêze³), pozwalamy na szersze poszukiwania.
                // Pozwalamy sprawdziæ wiêcej tras startowych, ¿eby nie utkn¹æ w lokalnym minimum "top 20".
                int current_branching_factor = (node.route_indices.empty()) ? 100 : 10;
                int current_expansion_limit = (node.route_indices.empty()) ? 2000 : expansion_limit;

                for (int i = node.last_route_idx + 1; i < (int)sorted_routes.size(); ++i) {
                    if (checked_count++ > current_expansion_limit) break;

                    const auto& candidate_route = sorted_routes[i];

                    if (candidate_route.bitmask.size() != mask_words) continue;

                    // Fast bitmask overlap check
                    bool overlap = false;
                    for (size_t mw = 0; mw < mask_words; ++mw) {
                        if (node.mask[mw] & candidate_route.bitmask[mw]) {
                            overlap = true;
                            break;
                        }
                    }
                    if (overlap) continue;

                    BeamNode new_node = node;
                    new_node.last_route_idx = i;
                    new_node.route_indices.push_back(i);
                    new_node.total_cost += candidate_route.cost;
                    new_node.efficiency_sum += candidate_route.efficiency;

                    int route_customers = 0;
                    for (int cid : candidate_route.nodes) {
                        if (cid >= 2) route_customers++;
                    }
                    new_node.customers_covered += route_customers;

                    for (size_t mw = 0; mw < mask_words; ++mw) {
                        new_node.mask[mw] |= candidate_route.bitmask[mw];
                    }

                    next_beam.push_back(new_node);
                    added_count++;
                    any_expansion = true;

                    if (added_count >= current_branching_factor) break;
                }
            }

            if (next_beam.empty()) break;

            // POPRAWKA 2: Inteligentniejsze sortowanie kandydatów.
            // Sortujemy nie tylko po liczbie klientów, ale po "wartoœci" stanu.
            // Preferujemy stany, które maj¹ du¿o klientów ORAZ niski koszt.
            // Heurystyka: Minimize (Cost - Covered * Penalty)
            double avg_cost_per_node = 100.0; // Mo¿na estymowaæ dynamicznie, ale sta³a wystarczy dla sortowania

            std::sort(next_beam.begin(), next_beam.end(),
                [avg_cost_per_node](const BeamNode& a, const BeamNode& b) {
                    // Obliczamy "Score". Im ni¿szy tym lepiej.
                    // Odejmujemy du¿¹ wartoœæ za ka¿dego obs³u¿onego klienta, ¿eby promowaæ pokrycie,
                    // ale dodajemy koszt, ¿eby karaæ drogie trasy.
                    double score_a = a.total_cost - (a.customers_covered * avg_cost_per_node * 2.0);
                    double score_b = b.total_cost - (b.customers_covered * avg_cost_per_node * 2.0);

                    return score_a < score_b;
                });

            if ((int)next_beam.size() > beam_width) {
                next_beam.resize(beam_width);
            }

            current_beam = std::move(next_beam);
            if (!any_expansion) break;
        }

        // 4. Select best state 
        if (current_beam.empty()) return Individual();

        // Tu te¿ sortujemy wynik koñcowy t¹ sam¹ logik¹ co wy¿ej, 
        // albo po prostu bierzemy ten z najwiêkszym pokryciem (bo to koniec algorytmu).
        // Najbezpieczniej wzi¹æ ten z [0], bo jest posortowany nasz¹ ulepszon¹ metryk¹.
        const BeamNode& best_node = current_beam[0];

        // ... (reszta kodu bez zmian: 5. Construct result, 6. Leftovers, etc.)

        // --- WKLEJ RESZTÊ FUNKCJI OD PUNKTU 5 W DÓ£ BEZ ZMIAN ---
        // (Poni¿ej skrót dla przypomnienia co tam by³o)

        std::vector<int> result_genotype(num_customers, -1);
        std::vector<int> result_permutation;
        result_permutation.reserve(num_customers);
        int current_group = 0;

        for (int r_idx : best_node.route_indices) {
            const auto& route = sorted_routes[r_idx];
            for (int cid : route.nodes) {
                if (cid < 2) continue;
                int gene_idx = cid - 2;
                if (gene_idx >= 0 && gene_idx < num_customers) {
                    result_genotype[gene_idx] = current_group;
                    result_permutation.push_back(cid);
                }
            }
            current_group++;
        }

        // Handle Leftovers Logic (identyczna jak w Twoim kodzie)
        std::vector<int> leftovers_vec;
        for (int customer_id : perm) {
            if (customer_id < 2) continue;
            int gene_idx = customer_id - 2;
            bool assigned = false;
            if (gene_idx < num_customers) {
                assigned = (best_node.mask[gene_idx / 64] & (1ULL << (gene_idx % 64)));
            }
            if (!assigned) {
                leftovers_vec.push_back(customer_id);
            }
        }

        int leftovers_count = (int)leftovers_vec.size();

        if (!leftovers_vec.empty()) {
            // (Twoja logika leftovers...)
            std::vector<int> group_loads(num_groups, 0);
            for (int i = 0; i < num_customers; ++i) {
                if (result_genotype[i] >= 0 && result_genotype[i] < num_groups) {
                    int customer_id = i + 2;
                    group_loads[result_genotype[i]] += static_cast<int>(evaluator->GetDemand(customer_id));
                }
            }
            int capacity = evaluator->GetCapacity();

            std::sort(leftovers_vec.begin(), leftovers_vec.end(),
                [&](int a, int b) { return evaluator->GetDemand(a) > evaluator->GetDemand(b); });

            for (int customer_id : leftovers_vec) {
                int gene_idx = customer_id - 2;
                if (gene_idx < 0 || gene_idx >= num_customers) continue;
                int demand = static_cast<int>(evaluator->GetDemand(customer_id));
                int best_group = -1;
                int min_slack = std::numeric_limits<int>::max();
                for (int g = 0; g < num_groups; ++g) {
                    int slack = capacity - group_loads[g];
                    if (slack >= demand && slack < min_slack) {
                        min_slack = slack;
                        best_group = g;
                    }
                }
                if (best_group == -1) {
                    int max_slack = -1;
                    for (int g = 0; g < num_groups; ++g) {
                        int slack = capacity - group_loads[g];
                        if (slack > max_slack) { max_slack = slack; best_group = g; }
                    }
                }
                if (best_group == -1) best_group = (current_group < num_groups) ? current_group : 0;

                result_genotype[gene_idx] = best_group;
                group_loads[best_group] += demand;
                result_permutation.push_back(customer_id);
            }
        }

        // Validate
        for (int i = 0; i < num_customers; ++i) {
            if (result_genotype[i] < 0) result_genotype[i] = 0;
        }

        Individual frankenstein(result_genotype);
   /*     SplitResult split_result = split.RunLinear(result_permutation);
        if (!split_result.group_assignment.empty()) {
            for (int& g : split_result.group_assignment) {
                if (g >= num_groups) g = g % num_groups;
            }
            frankenstein = Individual(split_result.group_assignment);
        }
        */
        double fitness = evaluator->Evaluate(frankenstein.GetGenotype());
        frankenstein.SetFitness(fitness);

        // Diagnostics
       // printf(" [BEAM] width=%d, routes=%zu, covered=%d, leftovers=%d, fit=%.2f\n",
         //   beam_width, best_node.route_indices.size(), best_node.customers_covered,
          //  leftovers_count, fitness);

        return frankenstein;
    }
}