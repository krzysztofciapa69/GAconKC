#include "LocalSearch.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>

using namespace LcVRPContest;

LocalSearch::LocalSearch(ThreadSafeEvaluator *evaluator,
                         const ProblemGeometry *geometry, int id)
    : evaluator_(evaluator), geometry_(geometry), id_(id) {
  // Seedowanie RNG
  rng_.seed(static_cast<unsigned int>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count() +
      id * 777));

  int n = evaluator_->GetSolutionSize();
  int g = evaluator_->GetNumGroups();

  // Rezerwacja pamięci dla struktur roboczych
  vnd_routes_.resize(g);
  for (auto &r : vnd_routes_) {
    r.reserve(n * 2 / g + 16); // Zapas pamięci
  }
  vnd_loads_.resize(g);

  client_indices_.resize(n);
  candidate_groups_.reserve(Config::NUM_NEIGHBORS + 5);

  // Inicjalizacja wektora DLB
  dlb_.resize(n, false);

  InitializeRanks();

  // Fast matrix init
  if (evaluator_->HasMatrix()) {
    fast_matrix_ = evaluator_->GetFastDistanceMatrix();
    matrix_dim_ = evaluator_->GetDimension();
  }
}

void LocalSearch::InitializeRanks() {
  int dim = evaluator_->GetDimension();
  customer_ranks_.resize(dim + 1, 0);
  const auto &perm = evaluator_->GetPermutation();
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] >= 0 && perm[i] < (int)customer_ranks_.size()) {
      customer_ranks_[perm[i]] = static_cast<int>(i);
    }
  }
}

// --- METODA SYMULACJI (SERCE OPTYMALIZACJI) ---
// Oblicza rzeczywisty koszt trasy uwzględniając powroty do bazy przy
// przepełnieniu
// --- METODA SYMULACJI (SERCE OPTYMALIZACJI) ---
// Oblicza rzeczywisty koszt trasy uwzględniając powroty do bazy przy
// przepełnieniu
double
LocalSearch::SimulateRouteCost(const std::vector<int> &route_nodes) const {
  if (route_nodes.empty())
    return 0.0;

  // HYBRID APPROACH: Use Route Cache for full route evaluation
  // This speeds up the check significantly for revisited routes (30-40% hit
  // rate)
  return evaluator_->GetRouteCost(route_nodes);
}

void LocalSearch::ResetDLB() { std::fill(dlb_.begin(), dlb_.end(), false); }

// --- GŁÓWNA PĘTLA OPTYMALIZACJI (ACTIVE SET) ---
bool LocalSearch::OptimizeActiveSet(Individual &ind, int max_iter,
                                    bool allow_swap, bool allow_3swap,
                                    bool allow_ejection, bool allow_4swap) {
  std::vector<int> &genotype = ind.AccessGenotype();
  int num_groups = evaluator_->GetNumGroups();
  int num_clients = static_cast<int>(genotype.size());
  const double EPSILON = 1e-4;

  // Obsługa rozmiaru DLB (jeśli zmieniła się instancja)
  if (dlb_.size() != genotype.size()) {
    dlb_.assign(num_clients, false);
  } else {
    // Resetujemy DLB na starcie nowej fazy, aby upewnić się, że sprawdzimy
    // wszystko
    ResetDLB();
  }

  bool improvement = true;
  bool any_change = false;
  int iter = 0;

  // Bufor na trasę tymczasową (unikanie alokacji w pętli)
  std::vector<int> temp_route_buffer;
  temp_route_buffer.reserve(num_clients / num_groups + 50);

  while (improvement && iter < max_iter) {
    improvement = false;
    iter++;

    // Iterujemy po liście klientów
    for (int client_idx : client_indices_) {
      if (client_idx >= num_clients)
        continue;

      // --- DON'T LOOK BIT CHECK ---
      if (dlb_[client_idx])
        continue;
      // ----------------------------

      int u = client_idx + 2;         // ID klienta
      int g_u = genotype[client_idx]; // Obecna grupa

      if (g_u < 0 || g_u >= num_groups)
        continue;

      bool move_made = false;

      // ================================================================
      // KROK 1: OCENA KOSZTU USUNIĘCIA (SOURCE) - NO-COPY SIMULATION
      // ================================================================
      double cost_source_before = route_costs_[g_u];

      // NO-COPY: Simulate route cost without physically removing the client
      double cost_source_after = SimulateRouteCostWithRemoval(g_u, u);
      double source_delta = cost_source_after - cost_source_before;

      // ================================================================
      // KROK 2: GENEROWANIE KANDYDATÓW
      // ================================================================
      candidate_groups_.clear();

      // FULL SCAN dla małej liczby grup (< 50) - eliminuje problem "ślepego"
      // MDS Dla K=21 to kluczowe!
      // OPTIMIZED: Always use neighbor-based candidates (skip full scan)
      const auto &my_neighbors = geometry_->GetNeighbors(client_idx);
      int checked_count = 0;
      int max_neighbors = Config::NUM_NEIGHBORS;

      // ADAPTIVE: For huge instances, reduce neighborhood search size drastically
      if (num_clients > 3000)
        max_neighbors = 8;
      else if (num_clients > 2000)
        max_neighbors = 12;
      else if (num_clients > 1000)
        max_neighbors = 18;

      for (int neighbor_idx : my_neighbors) {
        if (checked_count++ > max_neighbors)
          break;
        if (neighbor_idx >= num_clients)
          continue;
        int g_neighbor = genotype[neighbor_idx];
        if (g_neighbor != g_u && g_neighbor >= 0 && g_neighbor < num_groups)
          candidate_groups_.push_back(g_neighbor);
      }

      // Add random candidates to escape local optima
      for (int i = 0; i < 3; ++i)
        candidate_groups_.push_back(rng_() % num_groups);

      std::sort(candidate_groups_.begin(), candidate_groups_.end());
      auto last =
          std::unique(candidate_groups_.begin(), candidate_groups_.end());
      candidate_groups_.erase(last, candidate_groups_.end());

      // ================================================================
      // KROK 3: OCENA RUCHÓW (TARGET) - HYBRID DELTA OPTIMIZATION
      // ================================================================
      int best_target_g = -1;
      double best_total_delta = -EPSILON;
      int best_insert_pos = 0;

      for (int target_g : candidate_groups_) {
        if (target_g == g_u)
          continue;

        double cost_target_before = route_costs_[target_g];
        const auto &route_tgt = vnd_routes_[target_g];

        // Find insertion position (rank-based ordering)
        int rank_u = customer_ranks_[u];
        auto it_ins = std::upper_bound(
            route_tgt.begin(), route_tgt.end(), rank_u,
            [&](int r, int id) { return r < customer_ranks_[id]; });
        int ins_pos = (int)std::distance(route_tgt.begin(), it_ins);

        double target_delta;

        // PRUNING: Calculate fast delta (lower bound).
        // If lower bound is worse than best found, skip simulation.
        double fast_target_delta =
            CalculateFastInsertionDelta(u, target_g, ins_pos);
        if (source_delta + fast_target_delta >= best_total_delta)
          continue;

        // HYBRID: Use fast O(1) delta for safe moves, full simulation for
        // unsafe
        if (IsSafeMove(target_g, u)) {
          // O(1) delta calculation - no depot returns will change
          target_delta = fast_target_delta;
        } else {
          // Full NO-COPY simulation - handles depot returns correctly
          double cost_target_after =
              SimulateRouteCostWithInsert(target_g, u, ins_pos);
          target_delta = cost_target_after - cost_target_before;
        }

        double total_delta = source_delta + target_delta;

        // Standard acceptance: better delta wins
        bool accept_move = (total_delta < best_total_delta);

        // SLACK-AWARE: Accept slightly worse moves (up to 1%) if they relieve
        // tight routes
        if (!accept_move && Config::VND_SLACK_AWARE && total_delta < 0) {
          double relative_delta =
              total_delta /
              std::max(1.0, route_costs_[g_u] + route_costs_[target_g]);
          if (relative_delta >
              -Config::VND_SLACK_TOLERANCE) { // within 1% tolerance
            // Check if source route was "tight" (>95% capacity)
            double src_load_ratio =
                (double)vnd_loads_[g_u] / evaluator_->GetCapacity();
            if (src_load_ratio > Config::VND_TIGHT_ROUTE_THRESHOLD) {
              // Moving client OUT of tight route = good for slack
              accept_move = true;
            }
          }
        }

        if (accept_move) {
          best_total_delta = total_delta;
          best_target_g = target_g;
          best_insert_pos = ins_pos;
        }
      }

      // ================================================================
      // KROK 4: WYKONANIE RUCHU
      // ================================================================
      if (best_target_g != -1) {
        int old_route = g_u;
        int new_route = best_target_g;

        // COMMIT: Usuń ze źródła
        auto &r_src = vnd_routes_[old_route];
        auto it_rem = std::find(r_src.begin(), r_src.end(), u);
        if (it_rem != r_src.end())
          r_src.erase(it_rem);
        vnd_loads_[old_route] -= evaluator_->GetDemand(u);

        // COMMIT: Dodaj do celu
        auto &r_dst = vnd_routes_[new_route];
        auto it_ins = std::upper_bound(
            r_dst.begin(), r_dst.end(), customer_ranks_[u],
            [&](int r, int id) { return r < customer_ranks_[id]; });
        r_dst.insert(it_ins, u);
        vnd_loads_[new_route] += evaluator_->GetDemand(u);

        // Aktualizacja genotypu
        genotype[client_idx] = new_route;

        // Update positions for O(1) delta in next iterations
        UpdatePositionsAfterMove(u, old_route, new_route);

        improvement = true;
        any_change = true;
        move_made = true;

        // --- DLB UPDATE: Obudź sąsiadów ---
        dlb_[client_idx] = false;

        // Budzimy sąsiadów geometrycznych
        for (int n_idx : geometry_->GetNeighbors(client_idx)) {
          if (n_idx < (int)dlb_.size())
            dlb_[n_idx] = false;
        }
      }
      // Block Relocate logic removed.

      // Jeśli nie znaleźliśmy ruchu -> Uśpij klienta (Set DLB)
      if (!move_made) {
        // === ULTRA-FAST SWAP OPERATOR ===
        // O(1) delta calculation using positions_ and fast_matrix_
        if (allow_swap) {
          double best_swap_delta = -EPSILON;
          int best_swap_target_g = -1;
          int best_swap_v = -1;

          const auto &neighbors = geometry_->GetNeighbors(client_idx);

          // Get u's position info for O(1) delta
          const auto &pos_u = positions_[client_idx];
          int prev_u = (pos_u.prev_client > 0) ? (pos_u.prev_client - 1) : 0;
          int next_u = (pos_u.next_client > 0) ? (pos_u.next_client - 1) : 0;
          int u_mat = u - 1; // Matrix index for u

          for (int n_idx : neighbors) {
            if (n_idx >= num_clients)
              continue;
            int target_g = genotype[n_idx];
            if (target_g == g_u || target_g < 0 || target_g >= num_groups)
              continue;

            int v = n_idx + 2;

            // Get v's position info for O(1) delta
            const auto &pos_v = positions_[n_idx];
            int prev_v = (pos_v.prev_client > 0) ? (pos_v.prev_client - 1) : 0;
            int next_v = (pos_v.next_client > 0) ? (pos_v.next_client - 1) : 0;
            int v_mat = v - 1; // Matrix index for v

            // === O(1) FAST DELTA ===
            // Delta for removing u: -d(prev_u, u) - d(u, next_u) + d(prev_u,
            // next_u) Delta for inserting v at u's old position: +d(prev_u, v)
            // + d(v, next_u) - d(prev_u, next_u) Net delta for source =
            // d(prev_u, v) + d(v, next_u) - d(prev_u, u) - d(u, next_u)

            double delta_src, delta_tgt;

            if (fast_matrix_) {
              // Source route: replace u with v
              double old_src = fast_matrix_[prev_u * matrix_dim_ + u_mat] +
                               fast_matrix_[u_mat * matrix_dim_ + next_u];
              double new_src = fast_matrix_[prev_u * matrix_dim_ + v_mat] +
                               fast_matrix_[v_mat * matrix_dim_ + next_u];
              delta_src = new_src - old_src;

              // Target route: replace v with u
              double old_tgt = fast_matrix_[prev_v * matrix_dim_ + v_mat] +
                               fast_matrix_[v_mat * matrix_dim_ + next_v];
              double new_tgt = fast_matrix_[prev_v * matrix_dim_ + u_mat] +
                               fast_matrix_[u_mat * matrix_dim_ + next_v];
              delta_tgt = new_tgt - old_tgt;
            } else {
              double old_src = evaluator_->GetDist(prev_u, u_mat) +
                               evaluator_->GetDist(u_mat, next_u);
              double new_src = evaluator_->GetDist(prev_u, v_mat) +
                               evaluator_->GetDist(v_mat, next_u);
              delta_src = new_src - old_src;

              double old_tgt = evaluator_->GetDist(prev_v, v_mat) +
                               evaluator_->GetDist(v_mat, next_v);
              double new_tgt = evaluator_->GetDist(prev_v, u_mat) +
                               evaluator_->GetDist(u_mat, next_v);
              delta_tgt = new_tgt - old_tgt;
            }

            double swap_delta = delta_src + delta_tgt;

            if (swap_delta < best_swap_delta) {
              best_swap_delta = swap_delta;
              best_swap_target_g = target_g;
              best_swap_v = v;
            }
          }

          // Wykonaj najlepszy SWAP jeśli znaleziono
          if (best_swap_target_g != -1 && best_swap_v != -1) {
            int v = best_swap_v;
            int v_idx = v - 2;
            int target_g = best_swap_target_g;

            // Boundary checks
            if (v_idx < 0 || v_idx >= num_clients)
              continue;

            // Usuń u z source, dodaj v
            auto &r_src = vnd_routes_[g_u];
            auto it_u_rem = std::find(r_src.begin(), r_src.end(), u);
            if (it_u_rem == r_src.end())
              continue; // Safety check
            r_src.erase(it_u_rem);

            auto it_v_ins = std::upper_bound(
                r_src.begin(), r_src.end(), customer_ranks_[v],
                [&](int r, int id) { return r < customer_ranks_[id]; });
            r_src.insert(it_v_ins, v);

            // Usuń v z target, dodaj u
            auto &r_tgt = vnd_routes_[target_g];
            auto it_v_rem = std::find(r_tgt.begin(), r_tgt.end(), v);
            if (it_v_rem == r_tgt.end()) {
              // Restore r_src state if target find fails (should not happen)
              auto it_v_back = std::find(r_src.begin(), r_src.end(), v);
              if (it_v_back != r_src.end())
                r_src.erase(it_v_back);
              auto it_u_back = std::upper_bound(
                  r_src.begin(), r_src.end(), customer_ranks_[u],
                  [&](int r, int id) { return r < customer_ranks_[id]; });
              r_src.insert(it_u_back, u);
              continue;
            }
            r_tgt.erase(it_v_rem);

            auto it_u_ins = std::upper_bound(
                r_tgt.begin(), r_tgt.end(), customer_ranks_[u],
                [&](int r, int id) { return r < customer_ranks_[id]; });
            r_tgt.insert(it_u_ins, u);

            // Update loads
            int demand_u = evaluator_->GetDemand(u);
            int demand_v = evaluator_->GetDemand(v);
            vnd_loads_[g_u] += demand_v - demand_u;
            vnd_loads_[target_g] += demand_u - demand_v;

            // Update genotype
            genotype[client_idx] = target_g;
            genotype[v_idx] = g_u;

            improvement = true;
            any_change = true;
            move_made = true;

            // Wake neighbors
            dlb_[client_idx] = false;
            if (v_idx < (int)dlb_.size())
              dlb_[v_idx] = false;
            for (int n_idx : geometry_->GetNeighbors(client_idx)) {
              if (n_idx < (int)dlb_.size())
                dlb_[n_idx] = false;
            }
            for (int n_idx : geometry_->GetNeighbors(v_idx)) {
              if (n_idx < (int)dlb_.size())
                dlb_[n_idx] = false;
            }

            // FAST UPDATE: Use delta instead of full simulation
            // Update route costs using the delta we already calculated
            route_costs_[g_u] += best_swap_delta / 2; // Approximate split
            route_costs_[target_g] += best_swap_delta / 2;

            // Update positions only for affected clients (minimal rebuild)
            // Update u's position (now in target_g route)
            const auto &r_tgt_new = vnd_routes_[target_g];
            for (int pos = 0; pos < (int)r_tgt_new.size(); ++pos) {
              int cid = r_tgt_new[pos];
              int cidx = cid - 2;
              if (cidx >= 0 && cidx < num_clients) {
                positions_[cidx].route_id = target_g;
                positions_[cidx].position = pos;
                positions_[cidx].prev_client =
                    (pos > 0) ? r_tgt_new[pos - 1] : -1;
                positions_[cidx].next_client =
                    (pos < (int)r_tgt_new.size() - 1) ? r_tgt_new[pos + 1] : -1;
              }
            }

            // Update v's position (now in g_u route)
            const auto &r_src_new = vnd_routes_[g_u];
            for (int pos = 0; pos < (int)r_src_new.size(); ++pos) {
              int cid = r_src_new[pos];
              int cidx = cid - 2;
              if (cidx >= 0 && cidx < num_clients) {
                positions_[cidx].route_id = g_u;
                positions_[cidx].position = pos;
                positions_[cidx].prev_client =
                    (pos > 0) ? r_src_new[pos - 1] : -1;
                positions_[cidx].next_client =
                    (pos < (int)r_src_new.size() - 1) ? r_src_new[pos + 1] : -1;
              }
            }
          }
        }

        if (!move_made) {
          dlb_[client_idx] = true;
        }
      }
    }
  }

  // === 3-SWAP PHASE (only for L2 exploitation) ===
  if (allow_3swap && any_change) {
    // Try 3-swap to find deeper improvements
    if (Try3Swap(genotype)) {
      any_change = true;
      // CRITICAL FIX: Rebuild positions after 3-swap modifies routes
      // Without this, positions_ becomes stale causing access violation
      BuildPositions();
    }
  }

  // === 4-SWAP PHASE ===
  if (allow_4swap && any_change) {
    if (Try4Swap(genotype)) {
      any_change = true;
      BuildPositions();
    }
  }

  // === EJECTION CHAIN PHASE (probabilistic, ~20% of clients) ===
  if (allow_ejection) {
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    const double EJECTION_PROBABILITY = Config::EJECTION_PROBABILITY;

    for (int client_idx = 0; client_idx < num_clients; ++client_idx) {
      // Skip with 80% probability
      if (prob_dist(rng_) > EJECTION_PROBABILITY)
        continue;

      // Skip if DLB says this client is settled
      if (dlb_[client_idx])
        continue;

      if (TryEjectionChain(genotype, client_idx, 2)) {
        any_change = true;
        improvement = true;

        // Wake neighbors
        dlb_[client_idx] = false;
        for (int n_idx : geometry_->GetNeighbors(client_idx)) {
          if (n_idx < (int)dlb_.size())
            dlb_[n_idx] = false;
        }
      }
    }
  }

  // === PATH RELINKING PHASE (escape mechanism when stuck) ===
  // Run with probability even if changes were made - adds stochastic
  // exploration
  std::uniform_real_distribution<double> pr_prob(0.0, 1.0);
  bool should_try_pr =
      (!any_change) ||
      (pr_prob(rng_) <
       Config::PATH_RELINK_PROBABILITY); // always try when stuck

  if (should_try_pr && !guide_solution_.empty() &&
      guide_solution_.size() == genotype.size()) {
    // Calculate current total cost for PR
    double current_total = 0.0;
    for (int g = 0; g < num_groups; ++g) {
      current_total += route_costs_[g];
    }

    // ADAPTIVE PR: Calculate guide cost and check threshold
    // Only run expensive PR if we are within range of the elite solution
    double guide_cost = 0.0;
    std::vector<std::vector<int>> guide_routes(num_groups);
    for (int i = 0; i < num_clients; ++i) {
      int g = guide_solution_[i];
      if (g >= 0 && g < num_groups) {
        guide_routes[g].push_back(i + 2); // ID = index + 2
      }
    }
    for (const auto &r : guide_routes) {
      guide_cost += SimulateRouteCost(r);
    }

    // Threshold: Allow PR only if within 15% of the guide (Best Known)
    // If we are too far off, it's waste of CPU to relink
    if (current_total < guide_cost * 1.15) {
      // Try Path Relinking towards the guide solution
      if (TryPathRelinking(genotype, current_total, guide_solution_)) {
        any_change = true;
        // No need to call BuildPositions() - TryPathRelinking does it
        // internally
      }
    }
  }

  return any_change;
}

bool LocalSearch::RunVND(Individual &ind, bool heavy_mode) {
  // Legacy interface - use heavy_mode to determine parameters
  int max_iter = heavy_mode ? 50 : 10;
  return RunVND(ind, max_iter, heavy_mode, heavy_mode, heavy_mode, heavy_mode);
}

bool LocalSearch::RunVND(Individual &ind, int max_iter, bool allow_swap,
                         bool allow_3swap, bool allow_ejection, bool allow_4swap) {
  std::vector<int> &genotype = ind.AccessGenotype();
  if (genotype.empty())
    return false;

  // Odtwórz trasy z genotypu do struktur roboczych
  int num_clients = static_cast<int>(genotype.size());
  int num_groups = evaluator_->GetNumGroups();

  // CRITICAL FIX: Inicjalizacja active set - bez tego VND nie działa!
  client_indices_.resize(num_clients);
  std::iota(client_indices_.begin(), client_indices_.end(), 0);

  // Shuffle dla lepszej eksploracji (unikanie lokalnych minimów)
  std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);

  for (auto &r : vnd_routes_)
    r.clear();
  std::fill(vnd_loads_.begin(), vnd_loads_.end(), 0.0);

  for (int i = 0; i < num_clients; ++i) {
    int u = i + 2;
    int g = genotype[i];
    if (g >= 0 && g < num_groups) {
      vnd_routes_[g].push_back(u);
      vnd_loads_[g] += evaluator_->GetDemand(u);
    }
  }

  // Sortuj trasy wg rankingu (wymagane dla poprawnego działania insert/remove)
  for (auto &r : vnd_routes_) {
    std::sort(r.begin(), r.end(), [&](int a, int b) {
      return customer_ranks_[a] < customer_ranks_[b];
    });
  }

  // Build position tracking for O(1) delta evaluation
  BuildPositions();

  return OptimizeActiveSet(ind, max_iter, allow_swap, allow_3swap,
                           allow_ejection, allow_4swap);
}

// === SECTOR-BASED DECOMPOSITION FOR LARGE INSTANCES ===

void LocalSearch::PrepareRoutesFromGenotype(const std::vector<int> &genotype) {
  int num_clients = static_cast<int>(genotype.size());
  int num_groups = evaluator_->GetNumGroups();
  
  for (auto &r : vnd_routes_) r.clear();
  std::fill(vnd_loads_.begin(), vnd_loads_.end(), 0.0);
  
  for (int i = 0; i < num_clients; ++i) {
    int u = i + 2;
    int g = genotype[i];
    if (g >= 0 && g < num_groups) {
      vnd_routes_[g].push_back(u);
      vnd_loads_[g] += evaluator_->GetDemand(u);
    }
  }
  
  // sort routes by customer rank (required for correct insertion)
  for (auto &r : vnd_routes_) {
    std::sort(r.begin(), r.end(), [&](int a, int b) {
      return customer_ranks_[a] < customer_ranks_[b];
    });
  }
  
  BuildPositions();
}

void LocalSearch::PartitionBySeedExpansion(int num_sectors) {
  int num_clients = evaluator_->GetSolutionSize();
  
  // already partitioned with same sector count? skip
  if (sectors_initialized_ && (int)sectors_.size() == num_sectors) {
    return;
  }
  
  // step 1: pick K seed clients evenly distributed
  std::vector<int> seeds(num_sectors);
  for (int s = 0; s < num_sectors; ++s) {
    seeds[s] = (s * num_clients) / num_sectors;
  }
  
  // shuffle seeds slightly for diversity across runs
  for (int s = 0; s < num_sectors; ++s) {
    int offset = (int)(rng_() % 50) - 25;
    seeds[s] = std::max(0, std::min(num_clients - 1, seeds[s] + offset));
  }
  
  // step 2: BFS expansion from all seeds simultaneously
  client_sector_.assign(num_clients, -1);
  std::queue<int> bfs_queue;
  
  for (int s = 0; s < num_sectors; ++s) {
    client_sector_[seeds[s]] = s;
    bfs_queue.push(seeds[s]);
  }
  
  while (!bfs_queue.empty()) {
    int curr = bfs_queue.front();
    bfs_queue.pop();
    int my_sector = client_sector_[curr];
    
    const auto &neighbors = geometry_->GetNeighbors(curr);
    for (int n_idx : neighbors) {
      if (n_idx < 0 || n_idx >= num_clients) continue;
      if (client_sector_[n_idx] == -1) {
        client_sector_[n_idx] = my_sector;
        bfs_queue.push(n_idx);
      }
    }
  }
  
  // handle orphans (not reached by BFS - assign to nearest seed)
  for (int i = 0; i < num_clients; ++i) {
    if (client_sector_[i] == -1) {
      // assign to sector 0 as fallback
      client_sector_[i] = i % num_sectors;
    }
  }
  
  // step 3: collect clients per sector
  sectors_.assign(num_sectors, SectorInfo());
  for (int s = 0; s < num_sectors; ++s) {
    sectors_[s].seed_client = seeds[s];
    sectors_[s].client_indices.clear();
  }
  
  for (int i = 0; i < num_clients; ++i) {
    int s = client_sector_[i];
    if (s >= 0 && s < num_sectors) {
      sectors_[s].client_indices.push_back(i);
    }
  }
  
  sectors_initialized_ = true;
}

bool LocalSearch::OptimizeSector(Individual &ind, int sector_id, int max_iter) {
  if (sector_id < 0 || sector_id >= (int)sectors_.size()) return false;
  
  const auto &sector = sectors_[sector_id];
  if (sector.client_indices.empty()) return false;
  
  // set active clients to only this sector
  client_indices_ = sector.client_indices;
  std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);
  
  // run optimization on restricted set
  return OptimizeActiveSet(ind, max_iter, false, false, false);
}

bool LocalSearch::RefineBoundaries(Individual &ind, int max_iter) {
  int num_clients = evaluator_->GetSolutionSize();
  int num_sectors = (int)sectors_.size();
  
  if (num_sectors == 0) return false;
  
  // identify boundary clients (neighbors in different sectors)
  client_indices_.clear();
  client_indices_.reserve(num_clients / 4);  // estimate 25% are boundaries
  
  for (int i = 0; i < num_clients; ++i) {
    int my_sector = client_sector_[i];
    bool is_boundary = false;
    
    const auto &neighbors = geometry_->GetNeighbors(i);
    for (int n_idx : neighbors) {
      if (n_idx < 0 || n_idx >= num_clients) continue;
      if (client_sector_[n_idx] != my_sector) {
        is_boundary = true;
        break;
      }
    }
    
    if (is_boundary) {
      client_indices_.push_back(i);
    }
  }
  
  if (client_indices_.empty()) return false;
  
  std::shuffle(client_indices_.begin(), client_indices_.end(), rng_);
  
  // run optimization allowing cross-sector moves
  return OptimizeActiveSet(ind, max_iter, false, false, false);
}

bool LocalSearch::RunDecomposedVND(Individual &ind, int max_iter, bool exploration_mode) {
  std::vector<int> &genotype = ind.AccessGenotype();
  if (genotype.empty()) return false;
  
  int num_clients = static_cast<int>(genotype.size());
  
  // for small instances, fall back to standard VND
  if (num_clients < Config::LARGE_INSTANCE_THRESHOLD) {
    return RunVND(ind, max_iter, false, false, false);
  }
  
  // EXPLORATION MODE: ultra-aggressive speed optimization
  // - More sectors (smaller = faster per-sector)
  // - Fewer iterations per sector
  // - Single boundary pass (or none)
  int num_sectors;
  int sector_iter;
  int boundary_passes;
  int boundary_iter;
  
  if (exploration_mode) {
    // EXPLORATION: speed is everything, quality is secondary
    if (num_clients > Config::HUGE_INSTANCE_THRESHOLD) {
      num_sectors = 32;   // n=4000+ -> 125 clients/sector (very fast)
      sector_iter = 1;    // minimal iterations
      boundary_passes = 0; // skip boundary entirely for speed
      boundary_iter = 0;
    } else if (num_clients > 2500) {
      num_sectors = 20;   // n=2500-4000
      sector_iter = 1;
      boundary_passes = 1; // single quick pass
      boundary_iter = 2;
    } else {
      num_sectors = 12;   // n=1500-2500
      sector_iter = 2;
      boundary_passes = 1;
      boundary_iter = 3;
    }
  } else {
    // EXPLOITATION: quality matters, pay the time cost
    if (num_clients > Config::HUGE_INSTANCE_THRESHOLD) {
      num_sectors = 24;  // n=4000+ -> ~160 clients/sector (faster)
      sector_iter = std::max(2, max_iter / num_sectors);
      boundary_passes = 2;
      boundary_iter = std::max(3, max_iter / 3);
    } else if (num_clients > 2500) {
      num_sectors = 12;
      sector_iter = std::max(2, max_iter / num_sectors);
      boundary_passes = 2;
      boundary_iter = std::max(3, max_iter / 3);
    } else {
      num_sectors = 8;
      sector_iter = std::max(2, max_iter / num_sectors);
      boundary_passes = 2;
      boundary_iter = std::max(3, max_iter / 3);
    }
  }
  
  // phase 1: partition (lazy, cached)
  PartitionBySeedExpansion(num_sectors);
  
  // phase 2: prepare routes from genotype
  PrepareRoutesFromGenotype(genotype);
  
  // reset DLB
  if (dlb_.size() != genotype.size()) {
    dlb_.assign(num_clients, false);
  } else {
    ResetDLB();
  }
  
  // phase 3: optimize each sector independently
  bool any_change = false;
  
  for (int s = 0; s < num_sectors; ++s) {
    if (OptimizeSector(ind, s, sector_iter)) {
      any_change = true;
    }
  }
  
  // phase 4: boundary refinement (configurable passes)
  for (int pass = 0; pass < boundary_passes; ++pass) {
    if (RefineBoundaries(ind, boundary_iter)) {
      any_change = true;
    }
  }
  
  return any_change;
}

bool LocalSearch::RunFullVND(Individual &ind, bool allow_swap) {
  int num_clients = static_cast<int>(ind.GetGenotype().size());

  // Dodajemy wszystkich klientów do listy aktywnej
  client_indices_.resize(num_clients);
  std::iota(client_indices_.begin(), client_indices_.end(), 0);

  // Uruchamiamy główną pętlę
  return OptimizeActiveSet(ind, 20, allow_swap, false);
}

// === O(1) DELTA EVALUATION IMPLEMENTATION ===

void LocalSearch::BuildPositions() {
  int num_clients = evaluator_->GetSolutionSize();
  int num_groups = evaluator_->GetNumGroups();

  positions_.resize(num_clients);
  route_costs_.resize(num_groups);

  // Initialize all positions
  for (int i = 0; i < num_clients; ++i) {
    positions_[i].prev_client = -1;
    positions_[i].next_client = -1;
    positions_[i].route_id = -1;
    positions_[i].position = -1;
  }

  // Build positions from routes and calculate route costs
  for (int g = 0; g < num_groups; ++g) {
    const auto &route = vnd_routes_[g];
    int route_size = static_cast<int>(route.size());

    for (int pos = 0; pos < route_size; ++pos) {
      int client_id = route[pos];
      int client_idx = client_id - 2; // Convert to 0-based index

      if (client_idx >= 0 && client_idx < num_clients) {
        positions_[client_idx].route_id = g;
        positions_[client_idx].position = pos;
        positions_[client_idx].prev_client = (pos > 0) ? route[pos - 1] : -1;
        positions_[client_idx].next_client =
            (pos < route_size - 1) ? route[pos + 1] : -1;
      }
    }

    // Cache route cost
    route_costs_[g] = SimulateRouteCost(route);
  }

  // Also build cumulative loads for safe move detection
  BuildCumulativeLoads();
}

void LocalSearch::BuildCumulativeLoads() {
  int num_groups = evaluator_->GetNumGroups();
  int capacity = evaluator_->GetCapacity();

  max_cumulative_load_.resize(num_groups);

  for (int g = 0; g < num_groups; ++g) {
    const auto &route = vnd_routes_[g];
    int max_load = 0;
    int current_load = 0;

    for (int client_id : route) {
      int demand = evaluator_->GetDemand(client_id);

      // Check if we would return to depot (capacity exceeded)
      if (current_load + demand > capacity) {
        current_load = 0; // Reset after depot return
      }

      current_load += demand;
      if (current_load > max_load) {
        max_load = current_load;
      }
    }

    max_cumulative_load_[g] = max_load;
  }
}

bool LocalSearch::IsSafeMove(int target_route, int client_id) const {
  if (target_route < 0 || target_route >= (int)max_cumulative_load_.size())
    return false;

  int demand = evaluator_->GetDemand(client_id);
  int capacity = evaluator_->GetCapacity();

  // Safe if adding this client won't cause any new depot returns
  // This is a conservative check - if max_load + demand <= capacity, no new
  // returns
  return (max_cumulative_load_[target_route] + demand <= capacity);
}

double LocalSearch::CalculateFastInsertionDelta(int client_id, int target_route,
                                                int insert_pos) const {
  if (target_route < 0 || target_route >= (int)vnd_routes_.size())
    return 1e30;

  const auto &route = vnd_routes_[target_route];
  int route_size = static_cast<int>(route.size());
  int curr_idx = client_id - 1; // Matrix index

  // Get prev and next at insert_pos
  int prev_idx =
      (insert_pos > 0) ? (route[insert_pos - 1] - 1) : 0; // 0 = depot
  int next_idx =
      (insert_pos < route_size) ? (route[insert_pos] - 1) : 0; // 0 = depot

  // Delta = dist(prev, new) + dist(new, next) - dist(prev, next)
  double old_edge, new_edges;

  if (fast_matrix_) {
    old_edge = fast_matrix_[prev_idx * matrix_dim_ + next_idx];
    new_edges = fast_matrix_[prev_idx * matrix_dim_ + curr_idx] +
                fast_matrix_[curr_idx * matrix_dim_ + next_idx];
  } else {
    old_edge = evaluator_->GetDist(prev_idx, next_idx);
    new_edges = evaluator_->GetDist(prev_idx, curr_idx) +
                evaluator_->GetDist(curr_idx, next_idx);
  }

  return new_edges - old_edge;
}

double LocalSearch::SimulateRouteCostWithInsert(int target_route, int client_id,
                                                int insert_pos) const {
  if (target_route < 0 || target_route >= (int)vnd_routes_.size())
    return 1e30;

  const auto &route = vnd_routes_[target_route];
  int capacity = evaluator_->GetCapacity();
  int route_size = static_cast<int>(route.size());
  bool check_dist = evaluator_->HasDistanceConstraint();
  double max_dist = evaluator_->GetMaxDistance();

  double total_cost = 0.0;
  int current_load = 0;
  double current_segment_dist = 0.0;
  int last_idx = 0; // Depot

  // Unify loops or duplicate logic? Duplicate for performance (avoid check in
  // loop). Actually, the logic is complex enough that duplicating might be
  // error prone. Let's use a lambda or just duplicative logic for now as it's
  // critical hot path.

  if (fast_matrix_) {
    for (int pos = 0; pos <= route_size; ++pos) {
      // Logic to determine "next customer" in the virtual route
      int current_customer_id = -1;

      // Virtual insertion logic
      if (pos == insert_pos) {
        current_customer_id = client_id;
      } else {
        // If we passed the insertion point, we take route[pos-1]
        // If not, we take route[pos]
        // Wait, the original loop structure was:
        // Check if we need to insert at current 'pos'. If so, process inserted.
        // THEN process route[pos].
        // Let's stick to that structure.
      }

      // --- Iteration logic handling insert ---
      // We will perform 1 or 2 iterations of logic inside this loop step
      int iterations = (pos == insert_pos) ? 2 : 1;
      // If iterations=2: pass 0 is inserted item, pass 1 is route[pos]
      // If pos==route_size, we only process inserted item if
      // insert_pos==route_size

      // Let's stick to the original "double if" structure, it's safer.

      // 1. Check Insertion
      if (pos == insert_pos) {
        int demand = evaluator_->GetDemand(client_id);
        int customer_idx = client_id - 1;

        // Capacity Logic
        if (current_load + demand > capacity) {
          total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
          last_idx = 0;
          current_load = 0;
          current_segment_dist = 0.0;
        }

        double d_travel = fast_matrix_[last_idx * matrix_dim_ + customer_idx];

        // Distance Logic
        if (check_dist) {
          double d_return = fast_matrix_[customer_idx * matrix_dim_ + 0];
          if (current_segment_dist + d_travel + d_return > max_dist) {
            if (last_idx != 0) {
              total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
              last_idx = 0;
              current_load = 0;
              current_segment_dist = 0.0;
              d_travel = fast_matrix_[0 * matrix_dim_ + customer_idx];
            }
          }
        }

        total_cost += d_travel;
        current_segment_dist += d_travel;
        current_load += demand;
        last_idx = customer_idx;

        if (pos == route_size)
          continue; // Final insertion, done
      }

      // 2. Check Existing
      if (pos < route_size) {
        int orig_client_id = route[pos];
        int demand = evaluator_->GetDemand(orig_client_id);
        int customer_idx = orig_client_id - 1;

        if (current_load + demand > capacity) {
          total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
          last_idx = 0;
          current_load = 0;
          current_segment_dist = 0.0;
        }

        double d_travel = fast_matrix_[last_idx * matrix_dim_ + customer_idx];

        if (check_dist) {
          double d_return = fast_matrix_[customer_idx * matrix_dim_ + 0];
          if (current_segment_dist + d_travel + d_return > max_dist) {
            if (last_idx != 0) {
              total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
              last_idx = 0;
              current_load = 0;
              current_segment_dist = 0.0;
              d_travel = fast_matrix_[0 * matrix_dim_ + customer_idx];
            }
          }
        }

        total_cost += d_travel;
        current_segment_dist += d_travel;
        current_load += demand;
        last_idx = customer_idx;
      }
    }
    // Return to depot
    if (last_idx != 0) {
      total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
    }
  } else {
    // SLOW PATH (Duplicated logic using evaluator->GetDist)
    for (int pos = 0; pos <= route_size; ++pos) {
      if (pos == insert_pos) {
        int demand = evaluator_->GetDemand(client_id);
        int customer_idx = client_id - 1;

        if (current_load + demand > capacity) {
          total_cost += evaluator_->GetDist(last_idx, 0);
          last_idx = 0;
          current_load = 0;
          current_segment_dist = 0.0;
        }

        double d_travel = evaluator_->GetDist(last_idx, customer_idx);

        if (check_dist) {
          double d_return = evaluator_->GetDist(customer_idx, 0);
          if (current_segment_dist + d_travel + d_return > max_dist) {
            if (last_idx != 0) {
              total_cost += evaluator_->GetDist(last_idx, 0);
              last_idx = 0;
              current_load = 0;
              current_segment_dist = 0.0;
              d_travel = evaluator_->GetDist(0, customer_idx);
            }
          }
        }

        total_cost += d_travel;
        current_segment_dist += d_travel;
        current_load += demand;
        last_idx = customer_idx;

        if (pos == route_size)
          continue;
      }

      if (pos < route_size) {
        int orig_client_id = route[pos];
        int demand = evaluator_->GetDemand(orig_client_id);
        int customer_idx = orig_client_id - 1;

        if (current_load + demand > capacity) {
          total_cost += evaluator_->GetDist(last_idx, 0);
          last_idx = 0;
          current_load = 0;
          current_segment_dist = 0.0;
        }

        double d_travel = evaluator_->GetDist(last_idx, customer_idx);

        if (check_dist) {
          double d_return = evaluator_->GetDist(customer_idx, 0);
          if (current_segment_dist + d_travel + d_return > max_dist) {
            if (last_idx != 0) {
              total_cost += evaluator_->GetDist(last_idx, 0);
              last_idx = 0;
              current_load = 0;
              current_segment_dist = 0.0;
              d_travel = evaluator_->GetDist(0, customer_idx);
            }
          }
        }

        total_cost += d_travel;
        current_segment_dist += d_travel;
        current_load += demand;
        last_idx = customer_idx;
      }
    }
    if (last_idx != 0) {
      total_cost += evaluator_->GetDist(last_idx, 0);
    }
  }

  return total_cost;
}

double LocalSearch::SimulateRouteCostWithRemoval(int source_route,
                                                 int client_id) const {
  if (source_route < 0 || source_route >= (int)vnd_routes_.size())
    return 1e30;

  const auto &route = vnd_routes_[source_route];
  int capacity = evaluator_->GetCapacity();
  bool check_dist = evaluator_->HasDistanceConstraint();
  double max_dist = evaluator_->GetMaxDistance();

  double total_cost = 0.0;
  int current_load = 0;
  double current_segment_dist = 0.0;
  int last_idx = 0; // Depot

  if (fast_matrix_) {
    // OPTIMIZED
    for (int orig_client_id : route) {
      if (orig_client_id == client_id)
        continue;
      int demand = evaluator_->GetDemand(orig_client_id);
      int customer_idx = orig_client_id - 1;

      if (current_load + demand > capacity) {
        total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
        last_idx = 0;
        current_load = 0;
        current_segment_dist = 0.0;
      }

      double d_travel = fast_matrix_[last_idx * matrix_dim_ + customer_idx];

      if (check_dist) {
        double d_return = fast_matrix_[customer_idx * matrix_dim_ + 0];
        if (current_segment_dist + d_travel + d_return > max_dist) {
          if (last_idx != 0) {
            total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
            last_idx = 0;
            current_load = 0;
            current_segment_dist = 0.0;
            d_travel = fast_matrix_[0 * matrix_dim_ + customer_idx];
          }
        }
      }

      total_cost += d_travel;
      current_segment_dist += d_travel;
      current_load += demand;
      last_idx = customer_idx;
    }
    if (last_idx != 0) {
      total_cost += fast_matrix_[last_idx * matrix_dim_ + 0];
    }
  } else {
    // SLOW
    for (int orig_client_id : route) {
      if (orig_client_id == client_id)
        continue;
      int demand = evaluator_->GetDemand(orig_client_id);
      int customer_idx = orig_client_id - 1;

      if (current_load + demand > capacity) {
        total_cost += evaluator_->GetDist(last_idx, 0);
        last_idx = 0;
        current_load = 0;
        current_segment_dist = 0.0;
      }

      double d_travel = evaluator_->GetDist(last_idx, customer_idx);

      if (check_dist) {
        double d_return = evaluator_->GetDist(customer_idx, 0);
        if (current_segment_dist + d_travel + d_return > max_dist) {
          if (last_idx != 0) {
            total_cost += evaluator_->GetDist(last_idx, 0);
            last_idx = 0;
            current_load = 0;
            current_segment_dist = 0.0;
            d_travel = evaluator_->GetDist(0, customer_idx);
          }
        }
      }

      total_cost += d_travel;
      current_segment_dist += d_travel;
      current_load += demand;
      last_idx = customer_idx;
    }
    if (last_idx != 0) {
      total_cost += evaluator_->GetDist(last_idx, 0);
    }
  }

  return total_cost;
}

void LocalSearch::UpdatePositionsAfterMove(int client_id, int old_route,
                                           int new_route) {
  // This is called after a move is committed
  // For now, we rebuild positions for affected routes (can be optimized later)
  int num_groups = evaluator_->GetNumGroups();
  int num_clients = evaluator_->GetSolutionSize();

  // Update positions for old_route
  if (old_route >= 0 && old_route < num_groups) {
    const auto &route = vnd_routes_[old_route];
    int route_size = static_cast<int>(route.size());
    for (int pos = 0; pos < route_size; ++pos) {
      int cid = route[pos];
      int cidx = cid - 2;
      if (cidx >= 0 && cidx < num_clients) {
        positions_[cidx].route_id = old_route;
        positions_[cidx].position = pos;
        positions_[cidx].prev_client = (pos > 0) ? route[pos - 1] : -1;
        positions_[cidx].next_client =
            (pos < route_size - 1) ? route[pos + 1] : -1;
      }
    }
    route_costs_[old_route] = SimulateRouteCost(route);

    // Update cumulative load for old_route
    int max_load = 0;
    int current_load = 0;
    int capacity = evaluator_->GetCapacity();
    for (int cid : route) {
      int demand = evaluator_->GetDemand(cid);
      if (current_load + demand > capacity)
        current_load = 0;
      current_load += demand;
      if (current_load > max_load)
        max_load = current_load;
    }
    if (old_route < (int)max_cumulative_load_.size())
      max_cumulative_load_[old_route] = max_load;
  }

  // Update positions for new_route
  if (new_route >= 0 && new_route < num_groups) {
    const auto &route = vnd_routes_[new_route];
    int route_size = static_cast<int>(route.size());
    for (int pos = 0; pos < route_size; ++pos) {
      int cid = route[pos];
      int cidx = cid - 2;
      if (cidx >= 0 && cidx < num_clients) {
        positions_[cidx].route_id = new_route;
        positions_[cidx].position = pos;
        positions_[cidx].prev_client = (pos > 0) ? route[pos - 1] : -1;
        positions_[cidx].next_client =
            (pos < route_size - 1) ? route[pos + 1] : -1;
      }
    }
    route_costs_[new_route] = SimulateRouteCost(route);

    // Update cumulative load for new_route
    int max_load = 0;
    int current_load = 0;
    int capacity = evaluator_->GetCapacity();
    for (int cid : route) {
      int demand = evaluator_->GetDemand(cid);
      if (current_load + demand > capacity)
        current_load = 0;
      current_load += demand;
      if (current_load > max_load)
        max_load = current_load;
    }
    if (new_route < (int)max_cumulative_load_.size())
      max_cumulative_load_[new_route] = max_load;
  }
}

double LocalSearch::CalculateRemovalDelta(int client_id) const {
  int client_idx = client_id - 2;
  if (client_idx < 0 || client_idx >= (int)positions_.size())
    return 0.0;

  const auto &pos = positions_[client_idx];
  if (pos.route_id < 0)
    return 0.0;

  // Get prev and next indices for distance calculation
  int prev_idx = (pos.prev_client > 0) ? (pos.prev_client - 1) : 0; // 0 = depot
  int curr_idx = client_id - 1; // Matrix index
  int next_idx = (pos.next_client > 0) ? (pos.next_client - 1) : 0; // 0 = depot

  // Delta = -dist(prev, curr) - dist(curr, next) + dist(prev, next)
  double old_cost, new_cost;

  if (fast_matrix_) {
    old_cost = fast_matrix_[prev_idx * matrix_dim_ + curr_idx] +
               fast_matrix_[curr_idx * matrix_dim_ + next_idx];
    new_cost = fast_matrix_[prev_idx * matrix_dim_ + next_idx];
  } else {
    old_cost = evaluator_->GetDist(prev_idx, curr_idx) +
               evaluator_->GetDist(curr_idx, next_idx);
    new_cost = evaluator_->GetDist(prev_idx, next_idx);
  }

  return new_cost - old_cost; // Negative = improvement
}

double LocalSearch::CalculateInsertionDelta(int client_id, int target_route,
                                            int &best_insert_pos) const {
  if (target_route < 0 || target_route >= (int)vnd_routes_.size())
    return 1e30;

  const auto &route = vnd_routes_[target_route];
//  int curr_idx = client_id - 1; // Matrix index

  // Determine strictly valid position based on Permutation Ranks
  // Evaluator builds routes by iterating permutation, so relative order is fixed.
  int rank_u = customer_ranks_[client_id];
  int route_size = static_cast<int>(route.size());
  
  int pos = 0;
  while (pos < route_size) {
      if (customer_ranks_[route[pos]] > rank_u) break;
      pos++;
  }
  best_insert_pos = pos;

  // Calculate Delta for this SINGLE valid position
  int prev_idx = (pos > 0) ? (route[pos - 1] - 1) : 0;      // 0 = depot
  int next_idx = (pos < route_size) ? (route[pos] - 1) : 0; // 0 = depot
  int curr_idx = client_id - 1;

  double old_cost, new_cost;

  if (fast_matrix_) {
    old_cost = fast_matrix_[prev_idx * matrix_dim_ + next_idx];
    new_cost = fast_matrix_[prev_idx * matrix_dim_ + curr_idx] +
               fast_matrix_[curr_idx * matrix_dim_ + next_idx];
  } else {
    old_cost = evaluator_->GetDist(prev_idx, next_idx);
    new_cost = evaluator_->GetDist(prev_idx, curr_idx) +
               evaluator_->GetDist(curr_idx, next_idx);
  }

  return new_cost - old_cost;
}

bool LocalSearch::WouldOverflow(int target_route, int client_id) const {
  if (target_route < 0 || target_route >= (int)vnd_loads_.size())
    return true;

  int demand = evaluator_->GetDemand(client_id);
  int capacity = evaluator_->GetCapacity();

  return (vnd_loads_[target_route] + demand > capacity);
}

// Legacy methods kept for interface compatibility
double LocalSearch::CalculateRemovalDelta(const std::vector<int> &route,
                                          int client_id) const {
  return CalculateRemovalDelta(client_id);
}
double LocalSearch::CalculateInsertionDelta(const std::vector<int> &route,
                                            int client_id) const {
  int dummy;
  return CalculateInsertionDelta(client_id, -1, dummy);
}
int LocalSearch::FindInsertionIndexBinary(const std::vector<int> &route,
                                          int target_rank) const {
  return 0;
}
// === N-SWAP OPERATORS (Permutation-based) ===
// Can be called standalone as mutation - initializes routes from genotype

// === FAST N-SWAP: Pick N random clients, test all group permutations, apply
// best ===
bool LocalSearch::Try3Swap(std::vector<int> &genotype) {
  int num_clients = static_cast<int>(genotype.size());
  if (num_clients < 3)
    return false;

  int num_groups = evaluator_->GetNumGroups();
  const double EPSILON = 1e-4;

  // === SMART CLIENT SELECTION ===
  // Prioritize clients from "tight routes" (>90% capacity) - they benefit most from reassignment
  int capacity = evaluator_->GetCapacity();
  double tight_threshold = capacity * 0.90;
  
  // Compute route loads on-the-fly for this genotype
  std::vector<int> route_loads(num_groups, 0);
  for (int i = 0; i < num_clients; ++i) {
    int g = genotype[i];
    if (g >= 0 && g < num_groups) {
      int client_id = i + 2;  // client ID = genotype index + 2
      route_loads[g] += evaluator_->GetDemand(client_id);
    }
  }
  
  // Build pool of "tight route clients"
  std::vector<int> tight_clients;
  for (int i = 0; i < num_clients; ++i) {
    int g = genotype[i];
    if (g >= 0 && g < num_groups && route_loads[g] > tight_threshold) {
      tight_clients.push_back(i);
    }
  }
  
  // Selection strategy: 60% from tight routes (if available), 40% random
  auto select_client = [&]() -> int {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    if (!tight_clients.empty() && d(rng_) < 0.60) {
      return tight_clients[rng_() % tight_clients.size()];
    }
    return rng_() % num_clients;
  };
  
  // Pick 3 distinct clients using smart selection
  int idx1 = select_client();
  int idx2 = select_client();
  int idx3 = select_client();

  // Ensure distinct
  int attempts = 0;
  while ((idx1 == idx2 || idx1 == idx3 || idx2 == idx3) && attempts++ < 20) {
    idx2 = select_client();
    idx3 = select_client();
  }
  if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
    return false;

  int g1 = genotype[idx1];
  int g2 = genotype[idx2];
  int g3 = genotype[idx3];

  // Skip if all in same group (no useful swap)
  if (g1 == g2 && g2 == g3)
    return false;

  // Build routes for affected groups only
  std::set<int> affected_set = {g1, g2, g3};
  std::vector<int> affected_groups(affected_set.begin(), affected_set.end());

  // Build local route copies for affected groups
  std::map<int, std::vector<int>> local_routes;
  std::map<int, double> base_costs;

  const auto &permutation = evaluator_->GetPermutation();
  for (int g : affected_groups) {
    local_routes[g].clear();
  }

  for (int perm_id : permutation) {
    int client_idx = perm_id - 2;
    if (client_idx >= 0 && client_idx < num_clients) {
      int g = genotype[client_idx];
      if (affected_set.count(g)) {
        local_routes[g].push_back(perm_id);
      }
    }
  }

  double current_total = 0.0;
  for (int g : affected_groups) {
    base_costs[g] = SimulateRouteCost(local_routes[g]);
    current_total += base_costs[g];
  }

  // All 6 permutations of (g1, g2, g3) assigned to (idx1, idx2, idx3)
  int perms[6][3] = {{g1, g2, g3}, // original - skip
                     {g1, g3, g2}, {g2, g1, g3}, {g2, g3, g1},
                     {g3, g1, g2}, {g3, g2, g1}};

  int best_perm = -1;
  double best_delta = 0.0;

  // Evaluate all non-original permutations, find best
  for (int p = 1; p < 6; ++p) {
    int new_g1 = perms[p][0];
    int new_g2 = perms[p][1];
    int new_g3 = perms[p][2];

    // Build temp routes with swapped assignments
    std::map<int, std::vector<int>> temp_routes;
    for (int g : affected_groups) {
      temp_routes[g].clear();
    }

    for (int perm_id : permutation) {
      int client_idx = perm_id - 2;
      if (client_idx >= 0 && client_idx < num_clients) {
        int g = genotype[client_idx];

        // Apply swap logic
        if (client_idx == idx1)
          g = new_g1;
        else if (client_idx == idx2)
          g = new_g2;
        else if (client_idx == idx3)
          g = new_g3;

        if (affected_set.count(g)) {
          temp_routes[g].push_back(perm_id);
        }
      }
    }

    double new_total = 0.0;
    for (int g : affected_groups) {
      new_total += SimulateRouteCost(temp_routes[g]);
    }

    double delta = new_total - current_total;
    if (delta < best_delta - EPSILON) {
      best_delta = delta;
      best_perm = p;
    }
  }

  // Apply best permutation if improvement found
  if (best_perm > 0) {
    genotype[idx1] = perms[best_perm][0];
    genotype[idx2] = perms[best_perm][1];
    genotype[idx3] = perms[best_perm][2];
    return true;
  }

  return false;
}

// === FAST 4-SWAP: Pick 4 random clients, test permutations, apply best ===
bool LocalSearch::Try4Swap(std::vector<int> &genotype) {
  int num_clients = static_cast<int>(genotype.size());
  if (num_clients < 4)
    return false;

  int num_groups = evaluator_->GetNumGroups();
  const double EPSILON = 1e-4;

  // Pick 4 distinct random clients
  std::vector<int> idx(4);
  int attempts = 0;
  bool distinct = false;
  while (!distinct && attempts++ < 30) {
    for (int i = 0; i < 4; ++i)
      idx[i] = rng_() % num_clients;
    distinct = (idx[0] != idx[1] && idx[0] != idx[2] && idx[0] != idx[3] &&
                idx[1] != idx[2] && idx[1] != idx[3] && idx[2] != idx[3]);
  }
  if (!distinct)
    return false;

  std::vector<int> g(4);
  for (int i = 0; i < 4; ++i)
    g[i] = genotype[idx[i]];

  // Skip if all in same group
  bool all_same = (g[0] == g[1] && g[1] == g[2] && g[2] == g[3]);
  if (all_same)
    return false;

  // Affected groups
  std::set<int> affected_set(g.begin(), g.end());
  std::vector<int> affected_groups(affected_set.begin(), affected_set.end());

  // Build local route copies
  std::map<int, std::vector<int>> local_routes;
  const auto &permutation = evaluator_->GetPermutation();

  for (int grp : affected_groups) {
    local_routes[grp].clear();
  }

  for (int perm_id : permutation) {
    int client_idx = perm_id - 2;
    if (client_idx >= 0 && client_idx < num_clients) {
      int grp = genotype[client_idx];
      if (affected_set.count(grp)) {
        local_routes[grp].push_back(perm_id);
      }
    }
  }

  double current_total = 0.0;
  for (int grp : affected_groups) {
    current_total += SimulateRouteCost(local_routes[grp]);
  }

  // Test permutations (limit to ~12 random ones for speed)
  int perm_order[4] = {0, 1, 2, 3};
  int best_perm_order[4] = {0, 1, 2, 3};
  double best_delta = 0.0;
  bool found = false;

  int perm_count = 0;
  while (std::next_permutation(perm_order, perm_order + 4) &&
         perm_count++ < 12) {
    // Build temp assignment
    std::map<int, std::vector<int>> temp_routes;
    for (int grp : affected_groups) {
      temp_routes[grp].clear();
    }

    for (int perm_id : permutation) {
      int client_idx = perm_id - 2;
      if (client_idx >= 0 && client_idx < num_clients) {
        int grp = genotype[client_idx];

        // Apply permuted swap
        for (int i = 0; i < 4; ++i) {
          if (client_idx == idx[i]) {
            grp = g[perm_order[i]];
            break;
          }
        }

        if (affected_set.count(grp)) {
          temp_routes[grp].push_back(perm_id);
        }
      }
    }

    double new_total = 0.0;
    for (int grp : affected_groups) {
      new_total += SimulateRouteCost(temp_routes[grp]);
    }

    double delta = new_total - current_total;
    if (delta < best_delta - EPSILON) {
      best_delta = delta;
      std::copy(perm_order, perm_order + 4, best_perm_order);
      found = true;
    }
  }

  // Apply best if found
  if (found) {
    for (int i = 0; i < 4; ++i) {
      genotype[idx[i]] = g[best_perm_order[i]];
    }
    return true;
  }

  return false;
}

// === EJECTION CHAIN OPERATOR ===
// Multi-hop client relocation: A->X (ejects B), B->Y (ejects C), C->Z
// Uses probabilistic sampling for efficiency (not exhaustive BFS)
bool LocalSearch::TryEjectionChain(std::vector<int> &genotype,
                                   int start_client_idx, int max_depth) {
  int num_clients = static_cast<int>(genotype.size());
  int num_groups = evaluator_->GetNumGroups();
  const double EPSILON = 1e-4;

  if (start_client_idx < 0 || start_client_idx >= num_clients)
    return false;

  int start_client_id = start_client_idx + 2;
  int start_group = genotype[start_client_idx];

  if (start_group < 0 || start_group >= num_groups)
    return false;

  // Calculate current total cost of potentially affected routes
  double original_total = 0.0;
  for (int g = 0; g < num_groups; ++g) {
    original_total += route_costs_[g];
  }

  // Structure to track chain state
  struct ChainMove {
    int client_id;      // Client being moved
    int from_group;     // Source group
    int to_group;       // Target group
    int ejected_client; // Client ejected from target (-1 if none)
  };

  std::vector<ChainMove> best_chain;
  double best_delta = -EPSILON; // Must improve

  // === LEVEL 1: Try moving start client ===
  const auto &neighbors1 = geometry_->GetNeighbors(start_client_idx);

  // Sample up to 5 random target groups from neighbors
  std::vector<int> target_groups1;
  for (int n_idx : neighbors1) {
    if (n_idx >= num_clients)
      continue;
    int g = genotype[n_idx];
    if (g != start_group && g >= 0 && g < num_groups) {
      target_groups1.push_back(g);
    }
    if (target_groups1.size() >= 5)
      break;
  }
  // Add 2 random groups for diversity
  for (int i = 0; i < 2; ++i) {
    int rg = rng_() % num_groups;
    if (rg != start_group)
      target_groups1.push_back(rg);
  }

  for (int target_g1 : target_groups1) {
    if (target_g1 == start_group)
      continue;

    // Find client to eject from target_g1 (pick random from route)
    const auto &route1 = vnd_routes_[target_g1];
    if (route1.empty())
      continue;

    // Calculate cost of move without ejection first
    double cost_src_after =
        SimulateRouteCostWithRemoval(start_group, start_client_id);

    // Find best insert position
    int rank_u = customer_ranks_[start_client_id];
    auto it_ins = std::upper_bound(
        route1.begin(), route1.end(), rank_u,
        [&](int r, int id) { return r < customer_ranks_[id]; });
    int ins_pos = static_cast<int>(std::distance(route1.begin(), it_ins));

    double cost_tgt_after =
        SimulateRouteCostWithInsert(target_g1, start_client_id, ins_pos);

    double delta1 = (cost_src_after + cost_tgt_after) -
                    (route_costs_[start_group] + route_costs_[target_g1]);

    // Even if delta1 > 0, we might recover with ejection chain
    // But for efficiency, skip if way too bad
    if (delta1 > route_costs_[start_group] * 0.1)
      continue;

    // === LEVEL 2: Try ejecting a client from target_g1 ===
    if (max_depth >= 2 && !route1.empty()) {
      // Sample up to 3 clients to potentially eject
      std::vector<int> eject_candidates;
      if (route1.size() <= 3) {
        eject_candidates = route1;
      } else {
        for (int i = 0; i < 3; ++i) {
          eject_candidates.push_back(route1[rng_() % route1.size()]);
        }
      }

      for (int eject_id1 : eject_candidates) {
        if (eject_id1 == start_client_id)
          continue;
        int eject_idx1 = eject_id1 - 2;
        if (eject_idx1 < 0 || eject_idx1 >= num_clients)
          continue;

        // Find target for ejected client
        const auto &neighbors2 = geometry_->GetNeighbors(eject_idx1);
        std::vector<int> target_groups2;
        for (int n_idx : neighbors2) {
          if (n_idx >= num_clients)
            continue;
          int g = genotype[n_idx];
          if (g != target_g1 && g != start_group && g >= 0 && g < num_groups) {
            target_groups2.push_back(g);
          }
          if (target_groups2.size() >= 3)
            break;
        }
        // Can also go back to start_group!
        target_groups2.push_back(start_group);

        for (int target_g2 : target_groups2) {
          // Simulate full chain:
          // 1. Remove start_client from start_group
          // 2. Add start_client to target_g1, remove eject_id1
          // 3. Add eject_id1 to target_g2

          // Build temporary routes
          std::vector<int> temp_src = vnd_routes_[start_group];
          auto it_rem =
              std::find(temp_src.begin(), temp_src.end(), start_client_id);
          if (it_rem != temp_src.end())
            temp_src.erase(it_rem);

          std::vector<int> temp_tgt1 = vnd_routes_[target_g1];
          it_rem = std::find(temp_tgt1.begin(), temp_tgt1.end(), eject_id1);
          if (it_rem != temp_tgt1.end())
            temp_tgt1.erase(it_rem);
          // Insert start_client
          auto it_ins1 = std::upper_bound(
              temp_tgt1.begin(), temp_tgt1.end(),
              customer_ranks_[start_client_id],
              [&](int r, int id) { return r < customer_ranks_[id]; });
          temp_tgt1.insert(it_ins1, start_client_id);

          std::vector<int> temp_tgt2 = vnd_routes_[target_g2];
          if (target_g2 == start_group) {
            temp_tgt2 = temp_src; // Use already modified source
          }
          auto it_ins2 = std::upper_bound(
              temp_tgt2.begin(), temp_tgt2.end(), customer_ranks_[eject_id1],
              [&](int r, int id) { return r < customer_ranks_[id]; });
          temp_tgt2.insert(it_ins2, eject_id1);

          // Calculate new costs
          double new_cost_src = SimulateRouteCost(temp_src);
          double new_cost_tgt1 = SimulateRouteCost(temp_tgt1);
          double new_cost_tgt2 = (target_g2 == start_group)
                                     ? SimulateRouteCost(temp_tgt2)
                                     : SimulateRouteCost(temp_tgt2);

          // Calculate delta
          double old_cost = route_costs_[start_group] + route_costs_[target_g1];
          if (target_g2 != start_group && target_g2 != target_g1) {
            old_cost += route_costs_[target_g2];
          }

          double new_cost = new_cost_src + new_cost_tgt1;
          if (target_g2 != start_group) {
            new_cost += new_cost_tgt2;
          }

          double total_delta = new_cost - old_cost;

          if (total_delta < best_delta) {
            best_delta = total_delta;
            best_chain.clear();
            best_chain.push_back(
                {start_client_id, start_group, target_g1, eject_id1});
            best_chain.push_back({eject_id1, target_g1, target_g2, -1});
          }
        }
      }
    }

    // Also check simple move (no ejection) if it improves
    if (delta1 < best_delta) {
      best_delta = delta1;
      best_chain.clear();
      best_chain.push_back({start_client_id, start_group, target_g1, -1});
    }
  }

  // Apply best chain if found
  if (!best_chain.empty() && best_delta < -EPSILON) {
    for (const auto &move : best_chain) {
      int client_idx = move.client_id - 2;
      if (client_idx >= 0 && client_idx < num_clients) {
        genotype[client_idx] = move.to_group;

        // Update vnd_routes_ for consistency
        auto &r_from = vnd_routes_[move.from_group];
        auto it = std::find(r_from.begin(), r_from.end(), move.client_id);
        if (it != r_from.end())
          r_from.erase(it);

        auto &r_to = vnd_routes_[move.to_group];
        auto it_ins = std::upper_bound(
            r_to.begin(), r_to.end(), customer_ranks_[move.client_id],
            [&](int r, int id) { return r < customer_ranks_[id]; });
        r_to.insert(it_ins, move.client_id);
      }
    }

    // Rebuild route costs for affected routes
    std::set<int> affected;
    for (const auto &m : best_chain) {
      affected.insert(m.from_group);
      affected.insert(m.to_group);
    }
    for (int g : affected) {
      route_costs_[g] = SimulateRouteCost(vnd_routes_[g]);
    }

    return true;
  }

  return false;
}

// ============================================================================
// PATH RELINKING IMPLEMENTATION
// Explores intermediate solutions between current solution and guide solution.
// At each step, changes one gene to match the guide, evaluating all candidates.
// Key insight: The path may pass through different basins of attraction,
// potentially escaping local optima.
// ============================================================================
bool LocalSearch::TryPathRelinking(std::vector<int> &genotype,
                                   double &current_cost,
                                   const std::vector<int> &guide_solution) {
  if (guide_solution.empty() || guide_solution.size() != genotype.size()) {
    return false;
  }

  int num_clients = static_cast<int>(genotype.size());
  int num_groups = evaluator_->GetNumGroups();
  const double EPSILON = 1e-4;

  // Build difference set: indices where genotype differs from guide
  std::vector<int> diff_indices;
  diff_indices.reserve(num_clients);
  for (int i = 0; i < num_clients; ++i) {
    if (genotype[i] != guide_solution[i]) {
      diff_indices.push_back(i);
    }
  }

  // If solutions are identical, nothing to explore
  if (diff_indices.empty()) {
    return false;
  }

  // Track best solution found during path exploration
  std::vector<int> best_genotype = genotype;
  double best_cost = current_cost;
  bool found_improvement = false;

  // Working copy for path exploration
  std::vector<int> working = genotype;

  // Build initial routes for simulation
  std::vector<std::vector<int>> routes(num_groups);
  for (int i = 0; i < num_clients; ++i) {
    int g = working[i];
    if (g >= 0 && g < num_groups) {
      routes[g].push_back(i + 2); // customer_id = index + 2
    }
  }
  for (auto &r : routes) {
    std::sort(r.begin(), r.end(), [&](int a, int b) {
      return customer_ranks_[a] < customer_ranks_[b];
    });
  }

  // Precompute route costs
  std::vector<double> costs(num_groups);
  double working_cost = 0.0;
  for (int g = 0; g < num_groups; ++g) {
    costs[g] = SimulateRouteCost(routes[g]);
    working_cost += costs[g];
  }

  // Path Relinking loop - explore intermediate solutions
  // Strategy: At each step, find the move (changing one gene to match guide)
  // that yields the best intermediate solution
  int max_steps = std::min((int)diff_indices.size(), num_clients / 2);

  for (int step = 0; step < max_steps && !diff_indices.empty(); ++step) {
    int best_move_idx = -1;
    double best_move_cost = 1e30;
    double best_move_delta = 1e30;

    // Evaluate all possible moves (each diff_index can be changed to match
    // guide)
    for (int di = 0; di < (int)diff_indices.size(); ++di) {
      int client_idx = diff_indices[di];
      int client_id = client_idx + 2;

      int old_group = working[client_idx];
      int new_group = guide_solution[client_idx];

      if (old_group == new_group)
        continue; // Already matches
      if (old_group < 0 || old_group >= num_groups)
        continue;
      if (new_group < 0 || new_group >= num_groups)
        continue;

      // Calculate delta for this move
      // Cost of removing from old route
      double old_route_cost = costs[old_group];
      std::vector<int> temp_old = routes[old_group];
      auto it = std::find(temp_old.begin(), temp_old.end(), client_id);
      if (it != temp_old.end())
        temp_old.erase(it);
      double new_old_cost = SimulateRouteCost(temp_old);

      // Cost of inserting into new route
      double old_new_cost = costs[new_group];
      std::vector<int> temp_new = routes[new_group];
      auto ins_pos = std::upper_bound(
          temp_new.begin(), temp_new.end(), customer_ranks_[client_id],
          [&](int r, int id) { return r < customer_ranks_[id]; });
      temp_new.insert(ins_pos, client_id);
      double new_new_cost = SimulateRouteCost(temp_new);

      double delta =
          (new_old_cost - old_route_cost) + (new_new_cost - old_new_cost);
      double candidate_cost = working_cost + delta;

      // Track best move (prefer moves that improve or move towards guide with
      // minimal loss)
      if (candidate_cost < best_move_cost) {
        best_move_cost = candidate_cost;
        best_move_delta = delta;
        best_move_idx = di;
      }
    }

    // Apply best move
    if (best_move_idx < 0)
      break;

    int client_idx = diff_indices[best_move_idx];
    int client_id = client_idx + 2;
    int old_group = working[client_idx];
    int new_group = guide_solution[client_idx];

    // Update working solution
    working[client_idx] = new_group;
    working_cost += best_move_delta;

    // Update routes
    auto &r_old = routes[old_group];
    auto it = std::find(r_old.begin(), r_old.end(), client_id);
    if (it != r_old.end())
      r_old.erase(it);

    auto &r_new = routes[new_group];
    auto ins_pos = std::upper_bound(
        r_new.begin(), r_new.end(), customer_ranks_[client_id],
        [&](int r, int id) { return r < customer_ranks_[id]; });
    r_new.insert(ins_pos, client_id);

    // Update costs
    costs[old_group] = SimulateRouteCost(r_old);
    costs[new_group] = SimulateRouteCost(r_new);

    // Remove from diff_indices
    diff_indices.erase(diff_indices.begin() + best_move_idx);

    // Check if this intermediate solution is better than best found
    if (working_cost < best_cost - EPSILON) {
      best_cost = working_cost;
      best_genotype = working;
      found_improvement = true;
    }

    // Intensification: If we found an improvement, try local optimization
    // This is optional but can help find even better solutions
    if (found_improvement && step % 5 == 0) {
      // Quick single-pass local search on current intermediate
      // (We don't call full VND to avoid recursion, just check neighbors)
      for (int ci = 0; ci < num_clients; ++ci) {
        int u = ci + 2;
        int g_u = working[ci];
        if (g_u < 0 || g_u >= num_groups)
          continue;

        // Check if moving u to a neighbor group improves
        double remove_delta = SimulateRouteCost(routes[g_u]) - costs[g_u];
        // Simplified: skip local intensification for speed
      }
    }
  }

  // Apply best solution found if improvement
  if (found_improvement) {
    if (current_cost - best_cost > 1.0) {
      //  std::cout << " [PR] Success! Cost: " << current_cost << " -> " <<
      //  best_cost << " (Delta: " << (current_cost - best_cost) << ")" <<
      //  std::endl;
      prsucc++;
    }
    genotype = best_genotype;
    current_cost = best_cost;

    // Rebuild internal structures for consistency
    for (auto &r : vnd_routes_)
      r.clear();
    for (int i = 0; i < num_clients; ++i) {
      int g = genotype[i];
      if (g >= 0 && g < num_groups) {
        vnd_routes_[g].push_back(i + 2);
      }
    }
    for (auto &r : vnd_routes_) {
      std::sort(r.begin(), r.end(), [&](int a, int b) {
        return customer_ranks_[a] < customer_ranks_[b];
      });
    }
    BuildPositions();
  }

  return found_improvement;
}
