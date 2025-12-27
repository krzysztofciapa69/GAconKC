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

// --- POPRAWIONA IMPLEMENTACJA ApplyMicroSplit ---
void Split::ApplyMicroSplit(Individual& indiv, int start_idx, int end_idx, const ProblemGeometry* geometry) {
std::vector<int>& genes = indiv.AccessGenotype();
const std::vector<int>& giant_tour = evaluator_->GetPermutation();
int gt_size = static_cast<int>(giant_tour.size());
int num_groups = evaluator_->GetNumGroups();
int capacity = evaluator_->GetCapacity();

if (start_idx < 0 || end_idx >= gt_size || start_idx > end_idx) return;

int count = end_idx - start_idx + 1;

// --- 1. ALGORYTM SPLIT (Wyznaczanie optymalnych ciêæ) ---
// (Bez zmian w logice liczenia V_ i pred_)
if ((int)D_.size() <= count) {
    int new_size = count + 100;
    D_.resize(new_size);
    Q_.resize(new_size);
    V_.resize(new_size);
    pred_.resize(new_size);
}

D_[0] = 0.0;
Q_[0] = 0;

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

    while (!dq.empty() && Q_[i] - Q_[dq.front()] > capacity) {
        dq.pop_front();
    }

    if (dq.empty()) {
        V_[i] = std::numeric_limits<double>::max();
        continue;
    }

    int best = dq.front();
    int start_node = giant_tour[start_idx + best];
    int s_idx = (start_node > 1) ? start_node - 1 : 0;

    double d_in = evaluator_->GetDist(depot_idx_, s_idx);
    double d_out = evaluator_->GetDist(curr_idx, depot_idx_);
    double d_route = D_[i] - D_[best + 1];

    V_[i] = V_[best] + d_in + d_route + d_out;
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

if (V_[count] >= std::numeric_limits<double>::max()) return;

// --- 2. REKONSTRUKCJA SEGMENTÓW ---
// Zbieramy segmenty do wektora, aby przetwarzaæ je OD LEWEJ DO PRAWEJ.
// Dziêki temu mo¿emy aktualizowaæ stan 'Predecessor' dla grup na bie¿¹co.
struct Segment {
    int start_k; // relative index 1..count
    int end_k;
    double demand;
};
std::vector<Segment> segments;
int curr = count;
while (curr > 0) {
    int prev = pred_[curr];
    segments.push_back({ prev + 1, curr, (double)(Q_[curr] - Q_[prev]) });
    curr = prev;
}
std::reverse(segments.begin(), segments.end());

// --- 3. PRZYGOTOWANIE STANU T£A (Skanowanie Giant Tour) ---
// Musimy wiedzieæ, co ka¿da grupa ma "przed" oknem i "za" oknem.
std::vector<int> current_group_loads(num_groups, 0);

// Obliczamy Loady grup pomijaj¹c klientów z okna [start_idx, end_idx]
for (size_t i = 0; i < giant_tour.size(); ++i) {
    int c_id = giant_tour[i]; // 1-based or 2-based? Permutation has customer IDs starting from 2 usually.
    // Assuming giant_tour contains IDs 2..N+1 or similar. Check used in Evaluator.
    if (c_id <= 1) continue; // Skip Depot if present

    int gene_idx = c_id - 2;
    if (gene_idx >= 0 && gene_idx < (int)genes.size()) {
        int g = genes[gene_idx];
        if (g >= 0 && g < num_groups) {
            // Jeœli jesteœmy POZA oknem, dodaj do load.
            if ((int)i < start_idx || (int)i > end_idx) {
                current_group_loads[g] += evaluator_->GetDemand(c_id);
            }
        }
    }
}

// ZnajdŸ poprzedników (Last node visited by group before window)
std::vector<int> group_prev(num_groups, depot_idx_); // Default to depot index
std::vector<bool> g_found(num_groups, false);
int found_cnt = 0;

for (int k = start_idx - 1; k >= 0; --k) {
    int c_id = giant_tour[k];
    if (c_id <= 1) continue;
    int g = genes[c_id - 2];
    if (g >= 0 && g < num_groups && !g_found[g]) {
        group_prev[g] = c_id - 1; // matrix index
        g_found[g] = true;
        found_cnt++;
        if (found_cnt == num_groups) break;
    }
}

// ZnajdŸ nastêpników (First node visited by group after window)
std::vector<int> group_next(num_groups, depot_idx_); // Default to depot index
std::fill(g_found.begin(), g_found.end(), false);
found_cnt = 0;

for (int k = end_idx + 1; k < gt_size; ++k) {
    int c_id = giant_tour[k];
    if (c_id <= 1) continue;
    int g = genes[c_id - 2];
    if (g >= 0 && g < num_groups && !g_found[g]) {
        group_next[g] = c_id - 1; // matrix index
        g_found[g] = true;
        found_cnt++;
        if (found_cnt == num_groups) break;
    }
}

// --- 4. INTELIGENTNE PRZYPISYWANIE (Matching Cost + Capacity) ---
for (const auto& seg : segments) {
    // Indeksy globalne w giant_tour
    int seg_start_idx = start_idx + seg.start_k - 1;
    int seg_end_idx = start_idx + seg.end_k - 1;

    int node_start_id = giant_tour[seg_start_idx];
    int node_end_id = giant_tour[seg_end_idx];

    int idx_start = node_start_id - 1;
    int idx_end = node_end_id - 1;

    // Opcjonalne g³osy jako Tie-Breaker
    std::vector<int> votes(num_groups, 0);
    if (geometry) {
        for (int k = seg.start_k; k <= seg.end_k; ++k) {
            int c_id = giant_tour[start_idx + k - 1];
            int gene_idx = c_id - 2;
            if (gene_idx >= 0) {
                for (int neighbor : geometry->GetNeighbors(gene_idx)) {
                    if (neighbor < (int)genes.size()) {
                        int ng = genes[neighbor];
                        if (ng >= 0 && ng < num_groups) votes[ng]++;
                    }
                }
            }
        }
    }

    double best_cost_delta = std::numeric_limits<double>::max();
    int best_g = -1;

    // Szukamy najlepszej grupy
    for (int g = 0; g < num_groups; ++g) {
        double load_after = current_group_loads[g] + seg.demand;
        bool fits = (load_after <= capacity);

        // Jeœli nie mieœci siê w capacity, ignorujemy (chyba ¿e nikt siê nie mieœci, wtedy fallback)
        if (!fits) continue;

        // Kalkulacja Delty Kosztu
        // Stare po³¹czenie: Prev -> Next
        // Nowe po³¹czenie: Prev -> [Start ... End] -> Next
        int p_idx = group_prev[g];
        int n_idx = group_next[g];

        double d_old = evaluator_->GetDist(p_idx, n_idx);
        double d_new = evaluator_->GetDist(p_idx, idx_start) + evaluator_->GetDist(idx_end, n_idx);

        double delta = d_new - d_old;

        // Tie-breaker (bonus za g³osy)
        delta -= (votes[g] * 0.1);

        if (delta < best_cost_delta) {
            best_cost_delta = delta;
            best_g = g;
        }
    }

    // Fallback: Jeœli ¿adna grupa nie ma miejsca, wybierz tê z najmniejszym przeci¹¿eniem
    if (best_g == -1) {
        double min_overload = std::numeric_limits<double>::max();
        for (int g = 0; g < num_groups; ++g) {
            double overload = (current_group_loads[g] + seg.demand) - capacity;
            // Preferujemy te¿ mniejszy koszt przy podobnym overloadzie
            if (overload < min_overload) {
                min_overload = overload;
                best_g = g;
            }
        }
    }

    // Przypisz segment
    if (best_g != -1) {
        // Aktualizujemy stan grupy
        current_group_loads[best_g] += (int)seg.demand;

        // KLUCZOWE: Koniec tego segmentu staje siê nowym "Prev" dla tej grupy
        // (dla ewentualnych kolejnych segmentów w tym oknie)
        group_prev[best_g] = idx_end;

        // Zapis do genotypu
        for (int k = seg.start_k; k <= seg.end_k; ++k) {
            int c_id = giant_tour[start_idx + k - 1];
            int gene_idx = c_id - 2;
            if (gene_idx >= 0 && gene_idx < (int)genes.size()) {
                genes[gene_idx] = best_g;
            }
        }
    }
}
}