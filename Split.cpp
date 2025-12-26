#include "Split.hpp"
#include "ProblemGeometry.hpp" // Niezbêdne do u¿ycia metod geometry
#include <limits>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <deque>

using namespace LcVRPContest;

Split::Split(const ThreadSafeEvaluator* evaluator)
    : evaluator_(evaluator) {
    capacity_ = evaluator_->GetCapacity();

    max_distance_ = std::numeric_limits<double>::max();
    has_distance_constraint_ = false;

    depot_idx_ = 0;
    num_customers_ = evaluator_->GetSolutionSize();

    int size = num_customers_ + 100;
    D_.resize(size);
    Q_.resize(size);
    V_.resize(size);
    pred_.resize(size);
}

void Split::PrecomputeStructures(const std::vector<int>& giant_tour) {
    int n = static_cast<int>(giant_tour.size());
    if ((int)D_.size() <= n) {
        int new_size = n + 100;
        D_.resize(new_size);
        Q_.resize(new_size);
        V_.resize(new_size);
        pred_.resize(new_size);
    }

    D_[0] = 0.0;
    Q_[0] = 0;

    for (int i = 1; i <= n; ++i) {
        int curr_id = giant_tour[i - 1];
        // int curr_idx = curr_id - 2; // Nieu¿ywane w tej pêtli w tej wersji

        int prev_id = (i == 1) ? 1 : giant_tour[i - 2];

        int c_idx = (curr_id > 1) ? curr_id - 1 : 0;
        int p_idx = (prev_id > 1) ? prev_id - 1 : 0;

        double d = evaluator_->GetDist(p_idx, c_idx);
        double demand = evaluator_->GetDemand(curr_id);

        D_[i] = D_[i - 1] + d;
        Q_[i] = Q_[i - 1] + static_cast<int>(demand);
    }
}

SplitResult Split::RunLinear(const std::vector<int>& giant_tour) {
    PrecomputeStructures(giant_tour);
    int n = static_cast<int>(giant_tour.size());
    V_[0] = 0.0;

    std::deque<int> dq;
    dq.push_back(0);

    for (int i = 1; i <= n; ++i) {
        int curr_id = giant_tour[i - 1];
        int curr_idx = (curr_id > 1) ? curr_id - 1 : 0;

        while (!dq.empty()) {
            int front = dq.front();

            if (Q_[i] - Q_[front] > capacity_) {
                dq.pop_front();
                continue;
            }
            // Logika distance constraint usuniêta lub uproszczona w tej wersji dla szybkoœci, 
            // jeœli nie jest wymagana przez instancjê.
            break;
        }

        if (dq.empty()) {
            V_[i] = std::numeric_limits<double>::max();
            continue;
        }

        int best_pred = dq.front();

        int start_node_id = giant_tour[best_pred];
        int start_idx = (start_node_id > 1) ? start_node_id - 1 : 0;

        double d_depot_start = evaluator_->GetDist(depot_idx_, start_idx);
        double d_end_depot = evaluator_->GetDist(curr_idx, depot_idx_);
        double d_internal = D_[i] - D_[best_pred + 1];

        V_[i] = V_[best_pred] + d_depot_start + d_internal + d_end_depot;
        pred_[i] = best_pred;

        if (i < n) {
            if (V_[i] >= std::numeric_limits<double>::max()) continue;

            int next_node_id = giant_tour[i];
            int next_idx = (next_node_id > 1) ? next_node_id - 1 : 0;

            double d_depot_next = evaluator_->GetDist(depot_idx_, next_idx);
            double val_i = V_[i] + d_depot_next - D_[i + 1];

            while (!dq.empty()) {
                int back = dq.back();
                int back_next_id = giant_tour[back];
                int back_next_idx = (back_next_id > 1) ? back_next_id - 1 : 0;

                double d_depot_back_next = evaluator_->GetDist(depot_idx_, back_next_idx);
                double val_back = V_[back] + d_depot_back_next - D_[back + 1];

                if (val_i <= val_back) {
                    dq.pop_back();
                }
                else {
                    break;
                }
            }
            dq.push_back(i);
        }
    }

    return ReconstructResult(giant_tour);
}

SplitResult Split::RunBellman(const std::vector<int>& giant_tour) {
    PrecomputeStructures(giant_tour);
    int n = static_cast<int>(giant_tour.size());

    for (int i = 0; i <= n; ++i) V_[i] = 1e30;
    V_[0] = 0.0;

    for (int i = 1; i <= n; ++i) {
        int curr_id = giant_tour[i - 1];
        int curr_idx = curr_id - 1;

        for (int j = i - 1; j >= 0; --j) {
            int load = Q_[i] - Q_[j];
            if (load > capacity_) break;

            int start_node_id = giant_tour[j];
            int start_node_idx = start_node_id - 1;

            double d_depot_start = evaluator_->GetDist(depot_idx_, start_node_idx);
            double d_end_depot = evaluator_->GetDist(curr_idx, depot_idx_);
            double d_internal = D_[i] - D_[j + 1];

            double route_cost = d_depot_start + d_internal + d_end_depot;

            if (V_[j] + route_cost < V_[i]) {
                V_[i] = V_[j] + route_cost;
                pred_[i] = j;
            }
        }
    }
    return ReconstructResult(giant_tour);
}

SplitResult Split::ReconstructResult(const std::vector<int>& giant_tour) {
    SplitResult res;
    int n = static_cast<int>(giant_tour.size());

    if (V_[n] >= std::numeric_limits<double>::max()) {
        res.feasible = false;
        res.total_cost = std::numeric_limits<double>::max();
        return res;
    }

    res.feasible = true;
    res.total_cost = V_[n];

    int curr = n;
    while (curr > 0) {
        int prev = pred_[curr];
        std::vector<int> route;
        route.reserve(curr - prev);
        for (int k = prev + 1; k <= curr; ++k) {
            route.push_back(giant_tour[k - 1]);
        }
        res.optimized_routes.push_back(route);
        curr = prev;
    }
    std::reverse(res.optimized_routes.begin(), res.optimized_routes.end());

    res.group_assignment.assign(num_customers_, 0);

    for (size_t r_idx = 0; r_idx < res.optimized_routes.size(); ++r_idx) {
        for (int cust_id : res.optimized_routes[r_idx]) {
            int gene_idx = cust_id - 2;
            if (gene_idx >= 0 && gene_idx < num_customers_) {
                res.group_assignment[gene_idx] = static_cast<int>(r_idx);
            }
        }
    }
    return res;
}

// W pliku Split.cpp

void Split::ApplyMicroSplit(Individual& indiv, int start_idx, int end_idx, const ProblemGeometry* geometry) {
    std::vector<int>& genes = indiv.AccessGenotype();
    const std::vector<int>& giant_tour = evaluator_->GetPermutation();
    int gt_size = static_cast<int>(giant_tour.size());
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity(); // Pobieramy pojemnoœæ

    if (start_idx < 0 || end_idx >= gt_size || start_idx > end_idx) return;

    // 1. Obliczamy aktualne obci¹¿enie grup (Load Profile)
    // Ale IGNORUJEMY klientów, którzy s¹ wewn¹trz naszego okna [start_idx, end_idx]
    // bo oni bêd¹ zaraz przetasowani.
    std::vector<int> group_loads(num_groups, 0);

    // Mapa pomocnicza: czy klient o danym ID (z permutacji) jest w oknie?
    // Dla szybkoœci, po prostu iterujemy po genes i sprawdzamy czy gene_idx nale¿y do zbioru zmienianych.
    // Szybsza metoda: Iterujemy po genes. Jeœli gene[i] nie jest w oknie (trudne do sprawdzenia bez O(N)).
    // Najproœciej:
    // a) Policz load wszystkich
    for (size_t i = 0; i < genes.size(); ++i) {
        if (genes[i] >= 0 && genes[i] < num_groups) {
            int customer_id = i + 2; // Zak³adamy 0-based genes -> customer_id
            group_loads[genes[i]] += evaluator_->GetDemand(customer_id);
        }
    }
    // b) Odejmij load tych z okna
    for (int k = start_idx; k <= end_idx; ++k) {
        int customer_id = giant_tour[k];
        int gene_idx = customer_id - 2;
        if (gene_idx >= 0 && gene_idx < (int)genes.size()) {
            int current_g = genes[gene_idx];
            if (current_g >= 0 && current_g < num_groups) {
                group_loads[current_g] -= evaluator_->GetDemand(customer_id);
            }
        }
    }

    // --- ALGORYTM SPLIT (Bez zmian, to jest matematyka) ---
    int count = end_idx - start_idx + 1;

    if ((int)D_.size() <= count) {
        int new_size = count + 100;
        D_.resize(new_size); Q_.resize(new_size); V_.resize(new_size); pred_.resize(new_size);
    }

    D_[0] = 0.0; Q_[0] = 0;
    for (int i = 1; i <= count; ++i) {
        int curr_node = giant_tour[start_idx + i - 1];
        int curr_idx = (curr_node > 1) ? curr_node - 1 : 0;
        double dist = 0.0;
        if (i > 1) {
            int prev_node = giant_tour[start_idx + i - 2];
            int prev_idx = (prev_node > 1) ? prev_node - 1 : 0;
            dist = evaluator_->GetDist(prev_idx, curr_idx);
        }
        D_[i] = D_[i - 1] + dist;
        Q_[i] = Q_[i - 1] + evaluator_->GetDemand(curr_node);
    }

    std::deque<int> dq;
    dq.push_back(0);
    V_[0] = 0.0;

    for (int i = 1; i <= count; ++i) {
        int curr_node = giant_tour[start_idx + i - 1];
        int curr_idx = (curr_node > 1) ? curr_node - 1 : 0;
        while (!dq.empty() && Q_[i] - Q_[dq.front()] > capacity_) dq.pop_front();
        if (dq.empty()) { V_[i] = 1e30; continue; }

        int best = dq.front();
        int start_node = giant_tour[start_idx + best];
        int s_idx = (start_node > 1) ? start_node - 1 : 0;
        double d_in = evaluator_->GetDist(depot_idx_, s_idx);
        double d_out = evaluator_->GetDist(curr_idx, depot_idx_);
        V_[i] = V_[best] + d_in + (D_[i] - D_[best + 1]) + d_out;
        pred_[i] = best;

        if (i < count) {
            int next_node = giant_tour[start_idx + i];
            int next_idx = (next_node > 1) ? next_node - 1 : 0;
            double next_in = evaluator_->GetDist(depot_idx_, next_idx);
            double val_i = V_[i] - D_[i + 1] + next_in;
            while (!dq.empty()) {
                int back = dq.back();
                int back_next = giant_tour[start_idx + back];
                int back_idx = (back_next > 1) ? back_next - 1 : 0;
                double back_in = evaluator_->GetDist(depot_idx_, back_idx);
                if (val_i <= V_[back] - D_[back + 1] + back_in) dq.pop_back();
                else break;
            }
            dq.push_back(i);
        }
    }
    if (V_[count] >= 1e29) return;

    // --- INTELIGENTNE PRZYPISYWANIE GRUP ---
    int curr = count;
    std::vector<int> group_votes(num_groups, 0);

    // Struktura pomocnicza do rankingu grup
    struct GroupCandidate { int id; int votes; int load; };
    std::vector<GroupCandidate> candidates;
    candidates.reserve(num_groups);

    while (curr > 0) {
        int prev = pred_[curr];
        int segment_load = Q_[curr] - Q_[prev]; // Ile towaru ma ten nowy odcinek?

        std::fill(group_votes.begin(), group_votes.end(), 0);

        // G³osowanie s¹siadów (Geometry)
        for (int k = prev + 1; k <= curr; ++k) {
            int global_idx = start_idx + k - 1;
            int customer_id = giant_tour[global_idx];
            int gene_idx = customer_id - 2;

            if (gene_idx >= 0 && geometry != nullptr) {
                const auto& my_neighbors = geometry->GetNeighbors(gene_idx);
                // G³osuj¹ tylko najbli¿si s¹siedzi (np. top 5), ¿eby nie robiæ szumu
                int votes_cast = 0;
                for (int neighbor_idx : my_neighbors) {
                    if (votes_cast++ > 5) break;
                    if (neighbor_idx >= (int)genes.size()) continue;

                    int neighbor_group = genes[neighbor_idx];
                    if (neighbor_group >= 0 && neighbor_group < num_groups) {
                        group_votes[neighbor_group]++;
                    }
                }
            }
        }

        // Budowanie listy kandydatów
        candidates.clear();
        for (int g = 0; g < num_groups; ++g) {
            candidates.push_back({ g, group_votes[g], group_loads[g] });
        }

        // Sortowanie kandydatów:
        // 1. Najpierw ci, którzy maj¹ miejsce (Load + segment <= Cap)
        // 2. Wœród nich ci, którzy maj¹ najwiêcej g³osów
        // 3. Jeœli ¿aden nie ma miejsca, wybierz najmniej przeci¹¿onego
        std::sort(candidates.begin(), candidates.end(),
            [&](const GroupCandidate& a, const GroupCandidate& b) {
                bool a_fits = (a.load + segment_load <= capacity);
                bool b_fits = (b.load + segment_load <= capacity);

                if (a_fits && !b_fits) return true;
                if (!a_fits && b_fits) return false;

                if (a_fits) { // Obydwa pasuj¹ -> decyduj¹ g³osy
                    return a.votes > b.votes;
                }
                else { // ¯aden nie pasuje -> decyduje mniejsze prze³adowanie (mniejszy load)
                    return a.load < b.load;
                }
            });

        int best_group = candidates[0].id; // Zwyciêzca

        // Aktualizujemy load zwyciêskiej grupy (bo mo¿e dostaæ kolejny segment w nastêpnej iteracji pêtli while)
        group_loads[best_group] += segment_load;

        // Przypisanie w genotypie
        for (int k = prev + 1; k <= curr; ++k) {
            int global_idx = start_idx + k - 1;
            int customer_id = giant_tour[global_idx];
            int gene_idx = customer_id - 2;
            if (gene_idx >= 0 && gene_idx < (int)genes.size()) {
                genes[gene_idx] = best_group;
            }
        }
        curr = prev;
    }
}