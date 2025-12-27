#include "LocalSearch.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <chrono>
#include <cmath>

using namespace LcVRPContest;

LocalSearch::LocalSearch(ThreadSafeEvaluator* evaluator, const ProblemGeometry* geometry, int id)
    : evaluator_(evaluator), geometry_(geometry), id_(id)
{
    rng_.seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count() + id * 777));

    int n = evaluator_->GetSolutionSize();
    int g = evaluator_->GetNumGroups();

    vnd_routes_.resize(g);
    for (auto& r : vnd_routes_) {
        r.reserve(n * 2 / g + 16);
    }

    group_states_.resize(g);

    client_indices_.resize(n);
    candidate_groups_.reserve(Config::NUM_NEIGHBORS + 5);

    InitializeRanks();
}

void LocalSearch::InitializeRanks() {
    int dim = evaluator_->GetDimension();
    customer_ranks_.resize(dim + 1, 0);
    const auto& perm = evaluator_->GetPermutation();
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= 0 && perm[i] < (int)customer_ranks_.size()) {
            customer_ranks_[perm[i]] = static_cast<int>(i);
        }
    }
}

void LocalSearch::UpdateGroupState(int group_idx) {
    if (group_idx < 0 || group_idx >= (int)vnd_routes_.size()) return;

    const auto& route = vnd_routes_[group_idx];
    auto& states = group_states_[group_idx];

    if (states.size() != route.size()) {
        states.resize(route.size());
    }

    double total_dist = 0.0;
    double current_load = 0.0;
    int prev_idx = 0;
    int capacity = evaluator_->GetCapacity();

    for (size_t i = 0; i < route.size(); ++i) {
        int client_id = route[i];
        int curr_idx = client_id - 1;
        double demand = evaluator_->GetDemand(client_id);

        if (current_load + demand > capacity) {
            total_dist += evaluator_->GetDist(prev_idx, 0);
            total_dist += evaluator_->GetDist(0, curr_idx);
            current_load = demand;
        }
        else {
            total_dist += evaluator_->GetDist(prev_idx, curr_idx);
            current_load += demand;
        }
        prev_idx = curr_idx;

        states[i] = { total_dist, current_load, prev_idx };
    }
}

bool LocalSearch::RunVND(Individual& ind) {
    std::vector<int>& genotype = ind.AccessGenotype();
    if (genotype.empty()) return false;

    bool allow_swap = false;
    if (Config::VNDSWAP > 0.001) {
        std::uniform_real_distribution<double> d(0.0, 1.0);
        if (d(rng_) < Config::VNDSWAP) {
            allow_swap = true;
        }
    }

    int num_clients = static_cast<int>(genotype.size());
    int num_groups = evaluator_->GetNumGroups();
    int max_safe_id = (int)customer_ranks_.size() - 1;

    for (auto& r : vnd_routes_) r.clear();

    for (int i = 0; i < num_clients; ++i) {
        int c_id = i + 2;
        int g = genotype[i];
        if (c_id > max_safe_id) continue;
        if (g >= 0 && g < num_groups) {
            vnd_routes_[g].push_back(c_id);
        }
    }

    for (int g = 0; g < num_groups; ++g) {
        auto& r = vnd_routes_[g];
        if (r.size() > 1) {
            std::sort(r.begin(), r.end(),
                [&](int a, int b) { return customer_ranks_[a] < customer_ranks_[b]; });
        }
        UpdateGroupState(g);
    }

    const int FULL_VND_THRESHOLD = 600;

    if (num_clients > FULL_VND_THRESHOLD) {
        std::uniform_real_distribution<double> d_strategy(0.0, 1.0);
        const double FULL_CLEANUP_CHANCE = 0.02;

        if (d_strategy(rng_) < FULL_CLEANUP_CHANCE) {
            return RunFullVND(ind, allow_swap);
        }
        return RunDecomposedVND(ind, allow_swap);
    }
    else {
        return RunFullVND(ind, allow_swap);
    }
}

bool LocalSearch::RunDecomposedVND(Individual& ind, bool allow_swap) {
    int num_clients = static_cast<int>(ind.GetGenotype().size());
    bool global_improvement = false;

    int passes = std::min(Config::DECOMPOSEDVNDTRIES, 3);

    static std::vector<bool> in_active_set_buffer;
    if (in_active_set_buffer.size() < (size_t)num_clients) {
        in_active_set_buffer.resize(num_clients, false);
    }

    for (int pass = 0; pass < passes; ++pass) {
        client_indices_.clear();
        std::fill(in_active_set_buffer.begin(), in_active_set_buffer.end(), false);

        int centers_count = 4;
        for (int k = 0; k < centers_count; ++k) {
            int center_idx = rng_() % num_clients;

            if (!in_active_set_buffer[center_idx]) {
                in_active_set_buffer[center_idx] = true;
                client_indices_.push_back(center_idx);
            }

            const auto& neighbors = geometry_->GetNeighbors(center_idx);
            int added_neighbors = 0;

            for (int n_idx : neighbors) {
                if (added_neighbors >= 15) break;
                if (n_idx < num_clients && !in_active_set_buffer[n_idx]) {
                    in_active_set_buffer[n_idx] = true;
                    client_indices_.push_back(n_idx);
                    added_neighbors++;
                }
            }
        }

        int safety_guard = 0;
        while (client_indices_.size() < 30 && safety_guard++ < 100) {
            int rnd = rng_() % num_clients;
            if (!in_active_set_buffer[rnd]) {
                in_active_set_buffer[rnd] = true;
                client_indices_.push_back(rnd);
            }
        }

        if (OptimizeActiveSet(ind, 3, allow_swap)) {
            global_improvement = true;
        }
    }

    return global_improvement;
}

bool LocalSearch::RunFullVND(Individual& ind, bool allow_swap) {
    int num_clients = static_cast<int>(ind.GetGenotype().size());

    client_indices_.clear();
    if (client_indices_.capacity() < (size_t)num_clients) {
        client_indices_.reserve(num_clients);
    }

    for (int i = 0; i < num_clients; ++i) {
        client_indices_.push_back(i);
    }

    return OptimizeActiveSet(ind, 20, allow_swap);
}

bool LocalSearch::OptimizeActiveSet(Individual& ind, int max_iter, bool allow_swap) {
    std::vector<int>& genotype = ind.AccessGenotype();
    int num_groups = evaluator_->GetNumGroups();
    int num_clients = static_cast<int>(genotype.size());
    const double EPSILON = 1e-4;

    bool improvement = true;
    bool any_change = false;
    int iter = 0;

    std::vector<double> current_group_costs(num_groups, -1.0);

    auto get_group_cost = [&](int g) {
        if (current_group_costs[g] < -0.5) {
            current_group_costs[g] = GetPotentialRouteCost(vnd_routes_[g], g, -1, -1);
        }
        return current_group_costs[g];
        };

    std::uniform_real_distribution<double> d_ties(0.0, 1.0);

    while (improvement && iter < max_iter) {
        improvement = false;
        iter++;

        std::fill(current_group_costs.begin(), current_group_costs.end(), -1.0);
        std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);

        for (int client_idx : client_indices_) {
            if (client_idx >= num_clients) continue;

            int u = client_idx + 2;
            int g_u = genotype[client_idx];

            if (g_u < 0 || g_u >= num_groups) continue;

            double cost_source_curr = get_group_cost(g_u);

            int best_move_type = 0;
            int best_target_g = -1;
            int best_swap_v = -1;
            double best_delta = -EPSILON;
            int ties_count = 0;

            candidate_groups_.clear();
            const auto& my_neighbors = geometry_->GetNeighbors(client_idx);
            int neighbors_checked = 0;
            for (int neighbor_idx : my_neighbors) {
                if (neighbors_checked++ > 15) break;
                if (neighbor_idx >= num_clients) continue;
                int g_neighbor = genotype[neighbor_idx];
                if (g_neighbor != g_u && g_neighbor >= 0) candidate_groups_.push_back(g_neighbor);
            }
            if (candidate_groups_.size() < 3) {
                for (int k = 0; k < 2; ++k) candidate_groups_.push_back(rng_() % num_groups);
            }
            std::sort(candidate_groups_.begin(), candidate_groups_.end());
            candidate_groups_.erase(std::unique(candidate_groups_.begin(), candidate_groups_.end()), candidate_groups_.end());

            double cost_source_after_rem = GetPotentialRouteCost(vnd_routes_[g_u], g_u, u, -1);
            double delta_source_rem = cost_source_after_rem - cost_source_curr;

            for (int target_g : candidate_groups_) {
                if (target_g == g_u) continue;

                double cost_target_curr = get_group_cost(target_g);

                double cost_target_after_add = GetPotentialRouteCost(vnd_routes_[target_g], target_g, -1, u);
                double delta_target_add = cost_target_after_add - cost_target_curr;
                double total_delta = delta_source_rem + delta_target_add;

                if (total_delta <= best_delta) {
                    if (total_delta < best_delta - EPSILON) {
                        best_delta = total_delta;
                        ties_count = 1;
                        best_move_type = 1;
                        best_target_g = target_g;
                    }
                    else {
                        ties_count++;
                        if (d_ties(rng_) < (1.0 / ties_count)) {
                            best_move_type = 1;
                            best_target_g = target_g;
                        }
                    }
                }

                if (allow_swap) {
                    int swaps_checked = 0;
                    for (int v : vnd_routes_[target_g]) {
                        if (++swaps_checked > 10) break;

                        double cost_src_swap = GetPotentialRouteCost(vnd_routes_[g_u], g_u, u, v);
                        double cost_dst_swap = GetPotentialRouteCost(vnd_routes_[target_g], target_g, v, u);
                        double delta = (cost_src_swap - cost_source_curr) + (cost_dst_swap - cost_target_curr);

                        if (delta <= best_delta) {
                            if (delta < best_delta - EPSILON) {
                                best_delta = delta;
                                ties_count = 1;
                                best_move_type = 2;
                                best_target_g = target_g;
                                best_swap_v = v;
                            }
                            else {
                                ties_count++;
                                if (d_ties(rng_) < (1.0 / ties_count)) {
                                    best_move_type = 2;
                                    best_target_g = target_g;
                                    best_swap_v = v;
                                }
                            }
                        }
                    }
                }
            }

            if (best_move_type == 1) {
                auto& r_src = vnd_routes_[g_u];
                auto it_rem = std::lower_bound(r_src.begin(), r_src.end(), customer_ranks_[u],
                    [&](int id, int r) { return customer_ranks_[id] < r; });
                if (it_rem != r_src.end() && *it_rem == u) r_src.erase(it_rem);

                auto& r_dst = vnd_routes_[best_target_g];
                auto it_ins = std::upper_bound(r_dst.begin(), r_dst.end(), customer_ranks_[u],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_dst.insert(it_ins, u);

                current_group_costs[g_u] = -1.0;
                current_group_costs[best_target_g] = -1.0;

                genotype[client_idx] = best_target_g;

                UpdateGroupState(g_u);
                UpdateGroupState(best_target_g);

                improvement = true;
                any_change = true;
            }
            else if (best_move_type == 2) {
                int v = best_swap_v;
                int v_idx = v - 2;

                auto& r_u_vec = vnd_routes_[g_u];
                auto& r_v_vec = vnd_routes_[best_target_g];

                auto it_rem_u = std::lower_bound(r_u_vec.begin(), r_u_vec.end(), customer_ranks_[u],
                    [&](int id, int r) { return customer_ranks_[id] < r; });
                if (it_rem_u != r_u_vec.end() && *it_rem_u == u) r_u_vec.erase(it_rem_u);

                auto it_ins_v = std::upper_bound(r_u_vec.begin(), r_u_vec.end(), customer_ranks_[v],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_u_vec.insert(it_ins_v, v);

                auto it_rem_v = std::lower_bound(r_v_vec.begin(), r_v_vec.end(), customer_ranks_[v],
                    [&](int id, int r) { return customer_ranks_[id] < r; });
                if (it_rem_v != r_v_vec.end() && *it_rem_v == v) r_v_vec.erase(it_rem_v);

                auto it_ins_u = std::upper_bound(r_v_vec.begin(), r_v_vec.end(), customer_ranks_[u],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_v_vec.insert(it_ins_u, u);

                current_group_costs[g_u] = -1.0;
                current_group_costs[best_target_g] = -1.0;

                genotype[client_idx] = best_target_g;
                if (v_idx >= 0 && v_idx < (int)genotype.size()) genotype[v_idx] = g_u;

                UpdateGroupState(g_u);
                UpdateGroupState(best_target_g);

                improvement = true;
                any_change = true;
            }
        }
    }

    return any_change;
}

int LocalSearch::FindInsertionIndexBinary(const std::vector<int>& route, int target_rank) const {
    int left = 0;
    int right = static_cast<int>(route.size());
    while (left < right) {
        int mid = left + (right - left) / 2;
        int customer_in_route = route[mid];
        int current_rank = customer_ranks_[customer_in_route];
        if (current_rank < target_rank) {
            left = mid + 1;
        }
        else {
            right = mid;
        }
    }
    return left;
}

double LocalSearch::GetPotentialRouteCost(const std::vector<int>& route, int group_idx, int remove_client_id, int insert_client_id) const {
    int start_index = 0;
    RouteState state = { 0.0, 0.0, 0 };

    int insert_rank = (insert_client_id != -1) ? customer_ranks_[insert_client_id] : -1;
    int remove_rank = (remove_client_id != -1) ? customer_ranks_[remove_client_id] : -1;

    int insert_pos = -1;
    if (insert_client_id != -1) {
        insert_pos = FindInsertionIndexBinary(route, insert_rank);
    }

    int remove_pos = -1;
    if (remove_client_id != -1) {
        remove_pos = FindInsertionIndexBinary(route, remove_rank);
    }

    int change_pos = 1e9;
    if (insert_pos != -1) change_pos = insert_pos;
    if (remove_pos != -1) change_pos = std::min(change_pos, remove_pos);

    if (group_idx != -1 && change_pos > 0 && change_pos < 1e8) {
        if (group_idx < (int)group_states_.size() &&
            (change_pos - 1) < (int)group_states_[group_idx].size()) {

            state = group_states_[group_idx][change_pos - 1];
            start_index = change_pos;
        }
    }

    double total_dist = state.current_cost;
    double current_load = state.current_load;
    int prev_idx = state.last_location_idx;
    int capacity = evaluator_->GetCapacity();

    size_t i = start_index;
    bool inserted = (insert_client_id == -1);

    while (true) {
        int curr_client = -1;
        bool take_insert = false;

        if (!inserted) {
            if (i >= route.size()) {
                take_insert = true;
            }
            else {
                int curr_route_client = route[i];
                if (curr_route_client != remove_client_id) {
                    if (insert_rank < customer_ranks_[curr_route_client]) {
                        take_insert = true;
                    }
                }
            }
        }

        if (take_insert) {
            curr_client = insert_client_id;
            inserted = true;
        }
        else {
            if (i >= route.size()) break;
            curr_client = route[i];
            i++;
            if (curr_client == remove_client_id) continue;
        }

        double demand = evaluator_->GetDemand(curr_client);
        int curr_idx = curr_client - 1;

        if (current_load + demand > capacity) {
            total_dist += evaluator_->GetDist(prev_idx, 0);
            total_dist += evaluator_->GetDist(0, curr_idx);
            current_load = demand;
        }
        else {
            total_dist += evaluator_->GetDist(prev_idx, curr_idx);
            current_load += demand;
        }
        prev_idx = curr_idx;
    }

    total_dist += evaluator_->GetDist(prev_idx, 0);
    return total_dist;
}