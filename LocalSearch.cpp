#include "LocalSearch.hpp"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <limits>
#include <chrono>
#include <cmath> // dla std::max

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

    CalibratePenalty();

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

    // 2. Wspólna Inicjalizacja Struktur
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

    // Sortowanie tras
    for (auto& r : vnd_routes_) {
        if (r.size() > 1) {
            std::sort(r.begin(), r.end(),
                [&](int a, int b) { return customer_ranks_[a] < customer_ranks_[b]; });
        }
    }

    const int FULL_VND_THRESHOLD = 600;

    if (num_clients > FULL_VND_THRESHOLD) {
        std::uniform_real_distribution<double> d_strategy(0.0, 1.0);
        // "Lucky Shot" - Global Cleanup
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
    const double EPSILON = 1e-4; // Lekka tolerancja

    bool improvement = true;
    bool any_change = false;
    int iter = 0;

    // Buforowanie kosztów grup, ¿eby nie liczyæ ich ci¹gle od nowa
    // Jeœli groups siê nie zmieniaj¹, ich koszt jest sta³y.
    std::vector<double> current_group_costs(num_groups, -1.0);

    // Helper do pobierania kosztu (z cache lub obliczania)
    auto get_group_cost = [&](int g) {
        if (current_group_costs[g] < -0.5) {
            // Obliczamy koszt grupy 'g' bez ¿adnych modyfikacji (-1, -1)
            current_group_costs[g] = GetPotentialRouteCost(vnd_routes_[g], -1, -1);
        }
        return current_group_costs[g];
        };

    std::uniform_real_distribution<double> d_ties(0.0, 1.0);

    while (improvement && iter < max_iter) {
        improvement = false;
        iter++;

        // Reset cache kosztów na now¹ iteracjê (bo trasy siê zmieni³y)
        std::fill(current_group_costs.begin(), current_group_costs.end(), -1.0);

        std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);

        for (int client_idx : client_indices_) {
            if (client_idx >= num_clients) continue;

            int u = client_idx + 2;
            int g_u = genotype[client_idx];

            if (g_u < 0 || g_u >= num_groups) continue;

            // Pobieramy obecny koszt grupy Ÿród³owej
            double cost_source_curr = get_group_cost(g_u);

            int best_move_type = 0;
            int best_target_g = -1;
            int best_swap_v = -1;

            // Tutaj 'gain' to zmiana kosztu ca³kowitego. Ujemna = lepiej.
            double best_delta = -EPSILON;
            int ties_count = 0;

            // --- Generowanie Kandydatów ---
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

            // --- Ocena Ruchów (EXACT EVALUATION) ---

            // Koszt Ÿród³a PO usuniêciu U (wspólny dla wszystkich Relocate)
            double cost_source_after_rem = GetPotentialRouteCost(vnd_routes_[g_u], u, -1);
            double delta_source_rem = cost_source_after_rem - cost_source_curr;

            for (int target_g : candidate_groups_) {
                if (target_g == g_u) continue;

                double cost_target_curr = get_group_cost(target_g);

                // --- RELOCATE (u -> target_g) ---
                // Obliczamy koszt celu PO dodaniu U
                double cost_target_after_add = GetPotentialRouteCost(vnd_routes_[target_g], -1, u);

                double delta_target_add = cost_target_after_add - cost_target_curr;

                // Ca³kowita zmiana kosztu = (zmiana w Ÿródle) + (zmiana w celu)
                // To uwzglêdnia powroty, brak powrotów, wszystko. To jest CZYSTA PRAWDA.
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

                // --- SWAP (u <-> v) ---
                // Swap jest dro¿szy obliczeniowo, bo trzeba symulowaæ obie grupy naraz.
                // Robimy go rzadziej lub tylko dla obiecuj¹cych.
                if (allow_swap) {
                    // Dla SWAP musimy przeliczyæ obie grupy "na krzy¿"
                    // ród³o: usuwamy U, dodajemy V
                    // Cel: usuwamy V, dodajemy U

                    // Optymalizacja: sprawdzamy tylko kilku klientów z celu, nie wszystkich
                    int swaps_checked = 0;
                    for (int v : vnd_routes_[target_g]) {
                        if (++swaps_checked > 10) break; // Limit dla wydajnoœci

                        // Koszt Ÿród³a (usuñ U, dodaj V)
                        double cost_src_swap = GetPotentialRouteCost(vnd_routes_[g_u], u, v);
                        // Koszt celu (usuñ V, dodaj U)
                        double cost_dst_swap = GetPotentialRouteCost(vnd_routes_[target_g], v, u);

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

            // --- Wykonanie Ruchu ---
            if (best_move_type == 1) { // RELOCATE
                // Aktualizujemy wektory fizycznie
                auto& r_src = vnd_routes_[g_u];
                auto it_rem = std::lower_bound(r_src.begin(), r_src.end(), customer_ranks_[u],
                    [&](int id, int r) { return customer_ranks_[id] < r; });
                if (it_rem != r_src.end() && *it_rem == u) r_src.erase(it_rem);

                auto& r_dst = vnd_routes_[best_target_g];
                auto it_ins = std::upper_bound(r_dst.begin(), r_dst.end(), customer_ranks_[u],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_dst.insert(it_ins, u);

                // Oznaczamy cache kosztów jako nieaktualny dla tych grup
                current_group_costs[g_u] = -1.0;
                current_group_costs[best_target_g] = -1.0;

                genotype[client_idx] = best_target_g;
                improvement = true;
                any_change = true;
            }
            else if (best_move_type == 2) { // SWAP
                int v = best_swap_v;
                int v_idx = v - 2;

                auto& r_u_vec = vnd_routes_[g_u];
                auto& r_v_vec = vnd_routes_[best_target_g];

                // U z src -> out, V do src -> in
                auto it_rem_u = std::lower_bound(r_u_vec.begin(), r_u_vec.end(), customer_ranks_[u],
                    [&](int id, int r) { return customer_ranks_[id] < r; });
                if (it_rem_u != r_u_vec.end() && *it_rem_u == u) r_u_vec.erase(it_rem_u);

                auto it_ins_v = std::upper_bound(r_u_vec.begin(), r_u_vec.end(), customer_ranks_[v],
                    [&](int r, int id) { return r < customer_ranks_[id]; });
                r_u_vec.insert(it_ins_v, v);

                // V z dst -> out, U do dst -> in
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


void LocalSearch::CalibratePenalty() {
    int n = evaluator_->GetSolutionSize();
    if (n < 2) {
        current_penalty_factor_ = 1000.0;
        return;
    }

    // 1. Liczymy œredni¹ odleg³oœæ KLIENTA DO DEPOTU (a nie klienta do klienta)
    // Bo prze³adowanie = karny powrót do depotu.
    double total_depot_dist = 0.0;
    int samples = 0;

    // Próbkujemy co 10-tego klienta dla szybkoœci
    for (int i = 2; i <= n; i += 10) {
        // ID klienta to 'i', ale w GetDist(u, v) uzywamy indeksow macierzy
        // Zak³adam ze Evaluator ma metode GetDist(idx1, idx2)
        // Klient i (ID) -> index w macierzy (i-1)
        int idx = i - 1;
        total_depot_dist += evaluator_->GetDist(0, idx); // 0 to depot idx
        samples++;
    }

    double avg_depot_dist = (samples > 0) ? (total_depot_dist / samples) : 10.0;
    double avg_demand = 15.0; // Strzelamy lub mo¿na wyliczyæ œrednie ¿¹danie

    // IDEA: Przepe³nienie o 1 œrednie ¿¹danie (jednego klienta) 
    // powinno kosztowaæ tyle co wycieczka do bazy i z powrotem (2 * dist).
    // Factor = (2 * AvgDistToDepot) / AvgDemand
    // To jest "Sprawiedliwa Rynkowa Cena" prze³adowania.

    current_penalty_factor_ = (2.0 * avg_depot_dist) / avg_demand;

    // Zabezpieczenie przed skrajnoœciami
    if (current_penalty_factor_ < 1.0) current_penalty_factor_ = 1.0;

    // std::cout << "Calibrated Penalty: " << current_penalty_factor_ << std::endl;
}



// Dodaj to do LocalSearch.cpp przed OptimizeActiveSet

double LocalSearch::GetPotentialRouteCost(const std::vector<int>& route, int remove_client_id, int insert_client_id) const {
    double total_dist = 0.0;
    double current_load = 0.0;
    int capacity = evaluator_->GetCapacity();

    // Evaluator zaczyna od depotu
    int prev_idx = 0; // 0 to indeks depotu w macierzy odleg³oœci

    // Musimy iterowaæ po klientach zgodnie z ich RANG¥ (kolejnoœæ w permutacji),
    // poniewa¿ insert_client_id te¿ musi trafiæ w swoje miejsce w szeregu.

    // Pomocniczy wskaŸnik do iteracji po istniej¹cej trasie
    size_t i = 0;
    bool inserted = (insert_client_id == -1); // Jeœli nie ma insertu, uznajemy za wstawiony

    int loop_limit = static_cast<int>(route.size()) + 1;

    while (true) {
        int curr_client = -1;

        // Sprawdzamy, czy czas na wstawienie nowego klienta (zgodnie z rang¹)
        bool take_insert = false;
        if (!inserted) {
            int insert_rank = customer_ranks_[insert_client_id];
            if (i >= route.size()) {
                take_insert = true; // Koniec trasy, wstawiamy na koñcu
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
            if (i >= route.size()) break; // Koniec
            curr_client = route[i];
            i++;
            if (curr_client == remove_client_id) continue; // Pomijamy usuwanego
        }

        // --- LOGIKA EVALUATORA (Subtour cutting) ---
        // ID klienta (curr_client) mapujemy na indeks macierzy (curr_client - 1)
        // Zak³adam, ¿e GetDist(0, x) to dystans Depot->X.

        double demand = evaluator_->GetDemand(curr_client);
        int curr_idx = curr_client - 1; // Index do macierzy odleg³oœci

        if (current_load + demand > capacity) {
            // Prze³adowanie -> Powrót do depotu i ponowny wyjazd
            // Koszt: (prev -> depot) + (depot -> curr)
            total_dist += evaluator_->GetDist(prev_idx, 0);
            total_dist += evaluator_->GetDist(0, curr_idx);
            current_load = demand; // Reset ³adunku (tylko ten klient w nowym sub-tourze)
        }
        else {
            // Mieœci siê -> Bezpoœredni przejazd
            total_dist += evaluator_->GetDist(prev_idx, curr_idx);
            current_load += demand;
        }
        prev_idx = curr_idx;
    }

    // Powrót do depotu na koniec trasy
    total_dist += evaluator_->GetDist(prev_idx, 0);

    return total_dist;
}
