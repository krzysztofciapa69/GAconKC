#include "LocalSearch.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <chrono>

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
    vnd_loads_.resize(g);

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

bool LocalSearch::RunVND(Individual& ind) {
    std::vector<int>& genotype = ind.AccessGenotype();
    if (genotype.empty()) return false;

    // 1. Decyzja o Swap (raz na ca³e uruchomienie)
    bool allow_swap = false;
    if (Config::VNDSWAP > 0.001) {
        std::uniform_real_distribution<double> d(0.0, 1.0);
        if (d(rng_) < Config::VNDSWAP) {
            allow_swap = true;
        }
    }

    // 2. Wspólna Inicjalizacja Struktur (Obowi¹zkowa dla obu metod)
    int num_clients = static_cast<int>(genotype.size());
    int num_groups = evaluator_->GetNumGroups();
    int max_safe_id = (int)customer_ranks_.size() - 1;

    for (auto& r : vnd_routes_) r.clear();
    std::fill(vnd_loads_.begin(), vnd_loads_.end(), 0.0);

    for (int i = 0; i < num_clients; ++i) {
        int c_id = i + 2;
        int g = genotype[i];
        if (c_id > max_safe_id) continue;
        if (g >= 0 && g < num_groups) {
            vnd_routes_[g].push_back(c_id);
            vnd_loads_[g] += evaluator_->GetDemand(c_id);
        }
    }

    // Sortowanie tras (niezbêdne do binary search w metodach Delta)
    for (auto& r : vnd_routes_) {
        if (r.size() > 1) {
            std::sort(r.begin(), r.end(),
                [&](int a, int b) { return customer_ranks_[a] < customer_ranks_[b]; });
        }
    }

const int FULL_VND_THRESHOLD = 600; 
    
    if (num_clients > FULL_VND_THRESHOLD) {
        // Definiujemy rozk³ad prawdopodobieñstwa
        std::uniform_real_distribution<double> d_strategy(0.0, 1.0);

        // "Lucky Shot" - Global Cleanup
        // Dajemy np. 2.0% szansy na uruchomienie Pe³nego VND nawet dla du¿ej instancji.
        // Dziêki temu raz na ~50 wywo³añ nast¹pi gruntowna optymalizacja ca³oœci.
        const double FULL_CLEANUP_CHANCE = 0.02; 

        if (d_strategy(rng_) < FULL_CLEANUP_CHANCE) {
            return RunFullVND(ind, allow_swap);
        }

        // W pozosta³ych 98% przypadków u¿ywamy szybkiego Decomposed
        return RunDecomposedVND(ind, allow_swap);
    } 
    else {
        // Dla ma³ych problemów zawsze pe³na jakoœæ
        return RunFullVND(ind, allow_swap);
    }
}
bool LocalSearch::RunDecomposedVND(Individual& ind, bool allow_swap) {
    int num_clients = static_cast<int>(ind.GetGenotype().size());
    bool global_improvement = false;

    // Limit prób (passów)
    int passes = std::min(Config::DECOMPOSEDVNDTRIES, 3);

    // Statyczny bufor, aby unikn¹æ realokacji pamiêci przy ka¿dym wywo³aniu dla 4000+ klientów
    static std::vector<bool> in_active_set_buffer;
    if (in_active_set_buffer.size() < (size_t)num_clients) {
        in_active_set_buffer.resize(num_clients, false);
    }

    for (int pass = 0; pass < passes; ++pass) {
        client_indices_.clear();

        // Reset flag odwiedzenia (szybki fill)
        std::fill(in_active_set_buffer.begin(), in_active_set_buffer.end(), false);

        // Wybór centrów (Sta³a liczba = Szybkoœæ)
        int centers_count = 4;

        for (int k = 0; k < centers_count; ++k) {
            int center_idx = rng_() % num_clients;

            if (!in_active_set_buffer[center_idx]) {
                in_active_set_buffer[center_idx] = true;
                client_indices_.push_back(center_idx);
            }

            // Dodajemy max 15 najbli¿szych s¹siadów ka¿dego centrum
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

        // Zabezpieczenie: jeœli wylosowaliœmy pechowo (ma³o klientów), dobierz losowych do min. 30
        int safety_guard = 0;
        while (client_indices_.size() < 30 && safety_guard++ < 100) {
            int rnd = rng_() % num_clients;
            if (!in_active_set_buffer[rnd]) {
                in_active_set_buffer[rnd] = true;
                client_indices_.push_back(rnd);
            }
        }

        // Uruchom optymalizacjê tylko dla wybranych indeksów
        // Limit iteracji = 3 (szybkie poprawki, nie pe³na konwergencja)
        if (OptimizeActiveSet(ind, 3, allow_swap)) {
            global_improvement = true;
        }
    }

    return global_improvement;
}

bool LocalSearch::RunFullVND(Individual& ind, bool allow_swap) {
    int num_clients = static_cast<int>(ind.GetGenotype().size());

    // Wyczyœæ i dodaj wszystkich klientów do listy aktywnej
    client_indices_.clear();
    if (client_indices_.capacity() < (size_t)num_clients) {
        client_indices_.reserve(num_clients);
    }

    for (int i = 0; i < num_clients; ++i) {
        client_indices_.push_back(i);
    }

    // Uruchom optymalizacjê z du¿ym limitem iteracji (jakoœæ > czas)
    return OptimizeActiveSet(ind, 20, allow_swap);
}

bool LocalSearch::OptimizeActiveSet(Individual& ind, int max_iter, bool allow_swap) {
    std::vector<int>& genotype = ind.AccessGenotype();
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    int num_clients = static_cast<int>(genotype.size()); // Do sprawdzeñ granic
    const double EPSILON = 1e-6;

    bool improvement = true;
    bool any_change = false;
    int iter = 0;

    std::uniform_real_distribution<double> d_ties(0.0, 1.0);

    while (improvement && iter < max_iter) {
        improvement = false;
        iter++;

        // Mieszamy kolejnoœæ przetwarzania, aby unikn¹æ biasu
        std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);

        for (int client_idx : client_indices_) {
            if (client_idx >= num_clients) continue;

            int u = client_idx + 2; // ID klienta (1..N, Depot=1)
            int g_u = genotype[client_idx];

            if (g_u < 0 || g_u >= num_groups) continue;

            double dem_u = evaluator_->GetDemand(u);
            double rem_u = CalculateRemovalDelta(vnd_routes_[g_u], u);

            int best_move_type = 0; // 1: Relocate, 2: Swap
            int best_target_g = -1;
            int best_swap_v = -1;
            double best_gain = -EPSILON;
            int ties_count = 0;

            // --- 1. Generowanie kandydatów (S¹siedzi + Losowe) ---
            candidate_groups_.clear();
            const auto& my_neighbors = geometry_->GetNeighbors(client_idx);

            // Sprawdzamy top 15 s¹siadów geometrycznych
            int neighbors_checked = 0;
            for (int neighbor_idx : my_neighbors) {
                if (neighbors_checked++ > 15) break;
                if (neighbor_idx >= num_clients) continue;

                int g_neighbor = genotype[neighbor_idx];
                if (g_neighbor != g_u && g_neighbor >= 0) {
                    candidate_groups_.push_back(g_neighbor);
                }
            }

            // Dodajemy 2 losowe grupy dla dywersyfikacji
            if (candidate_groups_.size() < 5) {
                for (int k = 0; k < 2; ++k) candidate_groups_.push_back(rng_() % num_groups);
            }

            std::sort(candidate_groups_.begin(), candidate_groups_.end());
            auto last = std::unique(candidate_groups_.begin(), candidate_groups_.end());
            candidate_groups_.erase(last, candidate_groups_.end());

            // --- 2. Ocena Ruchów ---
            for (int target_g : candidate_groups_) {
                if (target_g == g_u) continue;

                // A. RELOCATE (Przeniesienie u do target_g)
                if (vnd_loads_[target_g] + dem_u <= capacity) {
                    double ins_u = CalculateInsertionDelta(vnd_routes_[target_g], u);
                    double gain = rem_u + ins_u;

                    if (gain <= best_gain) {
                        if (gain < best_gain - EPSILON) {
                            best_gain = gain;
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
                }

                // B. SWAP (Wymiana u <-> v)
                if (allow_swap) {
                    for (int v : vnd_routes_[target_g]) {
                        double dem_v = evaluator_->GetDemand(v);

                        if (vnd_loads_[g_u] - dem_u + dem_v > capacity) continue;
                        if (vnd_loads_[target_g] - dem_v + dem_u > capacity) continue;

                        double rem_v = CalculateRemovalDelta(vnd_routes_[target_g], v);
                        double ins_v_to_gu = CalculateInsertionDelta(vnd_routes_[g_u], v);
                        double ins_u_to_gv = CalculateInsertionDelta(vnd_routes_[target_g], u);

                        double swap_gain = (rem_u + ins_v_to_gu) + (rem_v + ins_u_to_gv);

                        if (swap_gain <= best_gain) {
                            if (swap_gain < best_gain - EPSILON) {
                                best_gain = swap_gain;
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

            // --- 3. Wykonanie Najlepszego Ruchu ---
            if (best_move_type == 1) { // RELOCATE

                auto& r_src = vnd_routes_[g_u];
                auto it_rem = std::lower_bound(r_src.begin(), r_src.end(), customer_ranks_[u],
                    [&](int id, int r) { return customer_ranks_[id] < r; });

                if (it_rem != r_src.end() && *it_rem == u) {
                    r_src.erase(it_rem);
                    vnd_loads_[g_u] -= dem_u;

                    auto& r_dst = vnd_routes_[best_target_g];
                    auto it_ins = std::upper_bound(r_dst.begin(), r_dst.end(), customer_ranks_[u],
                        [&](int r, int id) { return r < customer_ranks_[id]; });

                    r_dst.insert(it_ins, u);
                    vnd_loads_[best_target_g] += dem_u;

                    genotype[client_idx] = best_target_g;
                    improvement = true;
                    any_change = true;
                }
            }
            else if (best_move_type == 2) { // SWAP
                int v = best_swap_v;
                int v_idx = v - 2;
                double dem_v = evaluator_->GetDemand(v);

                auto& r_u_vec = vnd_routes_[g_u];
                auto it_rem_u = std::lower_bound(r_u_vec.begin(), r_u_vec.end(), customer_ranks_[u],
                    [&](int id, int r) { return customer_ranks_[id] < r; });
                if (it_rem_u != r_u_vec.end() && *it_rem_u == u) r_u_vec.erase(it_rem_u);

                auto& r_v_vec = vnd_routes_[best_target_g];
                auto it_rem_v = std::lower_bound(r_v_vec.begin(), r_v_vec.end(), customer_ranks_[v],
                    [&](int id, int r) { return customer_ranks_[id] < r; });
                if (it_rem_v != r_v_vec.end() && *it_rem_v == v) r_v_vec.erase(it_rem_v);

                auto it_ins_v = std::upper_bound(r_u_vec.begin(), r_u_vec.end(), customer_ranks_[v],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_u_vec.insert(it_ins_v, v);

                auto it_ins_u = std::upper_bound(r_v_vec.begin(), r_v_vec.end(), customer_ranks_[u],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_v_vec.insert(it_ins_u, u);

                vnd_loads_[g_u] = vnd_loads_[g_u] - dem_u + dem_v;
                vnd_loads_[best_target_g] = vnd_loads_[best_target_g] - dem_v + dem_u;

                genotype[client_idx] = best_target_g;
                if (v_idx >= 0 && v_idx < (int)genotype.size()) {
                    genotype[v_idx] = g_u;
                }
                improvement = true;
                any_change = true;
            }
        } // koniec pêtli po client_indices_
    } // koniec pêtli while(improvement)

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

double LocalSearch::CalculateRemovalDelta(const std::vector<int>& route, int client_id) const {
    if (client_id >= (int)customer_ranks_.size()) return 0.0;

    int rank = customer_ranks_[client_id];
    int idx = FindInsertionIndexBinary(route, rank);

    if (idx >= (int)route.size() || route[idx] != client_id) {
        return 0.0;
    }

    int prev_id = (idx > 0) ? route[idx - 1] : 1;

    int next_id = (idx < (int)route.size() - 1) ? route[idx + 1] : 1;
    int p_idx = (prev_id > 1) ? prev_id - 1 : 0;
    int c_idx = client_id - 1;
    int n_idx = (next_id > 1) ? next_id - 1 : 0;

    double dist_removed = evaluator_->GetDist(p_idx, c_idx) + evaluator_->GetDist(c_idx, n_idx);
    double dist_added = evaluator_->GetDist(p_idx, n_idx);

    return dist_added - dist_removed;
}

double LocalSearch::CalculateInsertionDelta(const std::vector<int>& route, int client_id) const {
    if (client_id >= (int)customer_ranks_.size()) return 1e9;

    int rank = customer_ranks_[client_id];
    int idx = FindInsertionIndexBinary(route, rank);

    int prev_id = (idx > 0) ? route[idx - 1] : 1;

    int next_id = (idx < (int)route.size()) ? route[idx] : 1;
    int p_idx = (prev_id > 1) ? prev_id - 1 : 0;
    int c_idx = client_id - 1;
    int n_idx = (next_id > 1) ? next_id - 1 : 0;

    double dist_removed = evaluator_->GetDist(p_idx, n_idx);
    double dist_added = evaluator_->GetDist(p_idx, c_idx) + evaluator_->GetDist(c_idx, n_idx);

    return dist_added - dist_removed;
}