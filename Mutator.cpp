#include "Mutator.hpp"
#include "Constants.hpp"
#include <cmath>
#include <limits>

using namespace LcVRPContest;

Mutator::Mutator()
    : evaluator_(nullptr), geometry_(nullptr), split_ptr_(nullptr) {}

void Mutator::Initialize(ThreadSafeEvaluator *eval, const ProblemGeometry *geo,
                         Split *split) {
  evaluator_ = eval;
  geometry_ = geo;
  split_ptr_ = split;

  int num_clients = eval->GetSolutionSize(); // Zakładam, że to liczba genów
  int num_groups = eval->GetNumGroups();

  // Rezerwacja pamięci "raz na zawsze"
  removed_indices_buffer_.reserve(num_clients);
  is_removed_buffer_.resize(num_clients, false); // resize, bo używamy indeksów
  group_centroids_buffer_.resize(num_groups);
  candidates_buffer_.reserve(20);
}

bool Mutator::ApplyRuinRecreate(Individual &indiv, double intensity,
                                bool is_exploitation, std::mt19937 &rng) {
  if (!geometry_ || !geometry_->HasCoordinates())
    return false;

  std::vector<int> &genes = indiv.AccessGenotype();
  int num_clients = static_cast<int>(genes.size());
  int num_groups = evaluator_->GetNumGroups();

  // 1. wybor centrum
  int center_idx = rng() % num_clients;

  // adaptive ruin percentage based on island type
  int min_rem = Config::RUIN_MIN_REMOVED;
  double pct;
  if (is_exploitation) {
    // exploitation: smaller windows (10-25%)
    pct = Config::RUIN_BASE_PCT_EXPLOITATION + (Config::RUIN_INTENSITY_SCALE_EXPLOITATION * intensity);
  } else {
    // exploration: larger windows (30-70%)
    pct = Config::RUIN_BASE_PCT + (Config::RUIN_INTENSITY_SCALE * intensity);
  }

  int max_rem = std::max(min_rem + 5, (int)(num_clients * pct));
  if (max_rem >= num_clients)
    max_rem = num_clients - 1;

  // Pobieramy sąsiadów
  const auto &neighbors = geometry_->GetNeighbors(center_idx);
  int available_neighbors = static_cast<int>(neighbors.size());

  if (max_rem > available_neighbors)
    max_rem = available_neighbors;
  int range = max_rem - min_rem;
  if (range <= 0)
    range = 1;

  int num_removed = min_rem + (rng() % range);
  if (num_removed < 1)
    return false;

  // --- DALEJ BEZ ZMIAN (Twoja świetna logika na buforach) ---
  removed_indices_buffer_.clear();
  removed_indices_buffer_.push_back(center_idx);

  for (int i = 0; i < num_removed; ++i) {
    removed_indices_buffer_.push_back(neighbors[i]);
  }

  // Reset buforów
  for (auto &info : group_centroids_buffer_) {
    info.sum_x = 0;
    info.sum_y = 0;
    info.count = 0;
  }
  std::fill(is_removed_buffer_.begin(), is_removed_buffer_.end(), false);

  for (int idx : removed_indices_buffer_) {
    if (idx < num_clients)
      is_removed_buffer_[idx] = true;
  }

  // Centroidy dla ISTNIEJĄCYCH
  for (int i = 0; i < num_clients; ++i) {
    if (is_removed_buffer_[i])
      continue;
    int g = genes[i];
    if (g >= 0 && g < num_groups) {
      int coord_idx = i + 1;
      const auto &coord = geometry_->GetCoordinate(coord_idx);
      group_centroids_buffer_[g].sum_x += coord.x;
      group_centroids_buffer_[g].sum_y += coord.y;
      group_centroids_buffer_[g].count++;
    }
  }

  // Usuwanie
  for (int idx : removed_indices_buffer_)
    genes[idx] = -1;

  std::shuffle(removed_indices_buffer_.begin(), removed_indices_buffer_.end(),
               rng);

  // Wstawianie (Greedy Spatial)
  for (int client_idx : removed_indices_buffer_) {
    int coord_idx = client_idx + 1;
    const auto &client_coord = geometry_->GetCoordinate(coord_idx);

    int best_g = -1;
    double best_dist_sq = std::numeric_limits<double>::max();

    for (int g = 0; g < num_groups; ++g) {
      double dist_sq = 1e15;
      if (group_centroids_buffer_[g].count > 0) {
        double gx =
            group_centroids_buffer_[g].sum_x / group_centroids_buffer_[g].count;
        double gy =
            group_centroids_buffer_[g].sum_y / group_centroids_buffer_[g].count;
        double dx = client_coord.x - gx;
        double dy = client_coord.y - gy;
        dist_sq = dx * dx + dy * dy;
      } else {
        dist_sq = 1e9; // Puste grupy jako alternatywa
      }

      if (dist_sq < best_dist_sq) {
        best_dist_sq = dist_sq;
        best_g = g;
      }
    }
    if (best_g == -1)
      best_g = rng() % num_groups;

    genes[client_idx] = best_g;

    // Update centroidu "on the fly" (ważne!)
    group_centroids_buffer_[best_g].sum_x += client_coord.x;
    group_centroids_buffer_[best_g].sum_y += client_coord.y;
    group_centroids_buffer_[best_g].count++;
  }

  return true;
}
bool Mutator::ApplySmartSpatialMove(Individual &indiv, std::mt19937 &rng) {
  if (!geometry_)
    return false;

  std::vector<int> &genes = indiv.AccessGenotype();
  int size = static_cast<int>(genes.size());
  int num_groups = evaluator_->GetNumGroups();

  int client_idx = rng() % size;
  int current_group = genes[client_idx];

  // Pobieramy referencję do vectora sąsiadów
  const auto &neighbors = geometry_->GetNeighbors(client_idx);
  if (neighbors.empty())
    return false;

  int best_target_group = -1;

  // Patrzymy na 15 najbliższych sąsiadów
  int check_limit = std::min((int)neighbors.size(), 15);
  for (int k = 0; k < check_limit; ++k) {
    int neighbor_idx = neighbors[k];
    int neighbor_group = genes[neighbor_idx];

    if (neighbor_group != current_group && neighbor_group >= 0 &&
        neighbor_group < num_groups) {
      best_target_group = neighbor_group;
      // Tutaj można by dodać sprawdzenie Capacity, jeśli chcemy być dokładni
      // Ale w LcVRP Evaluator i tak doda karę/powrót, więc można przenieść "na
      // ślepo"
      break;
    }
  }

  if (best_target_group != -1) {
    genes[client_idx] = best_target_group;
    return true;
  }

  // Jeśli nie znaleźliśmy lepszej grupy u sąsiadów, losowy strzał
  int random_g = rng() % num_groups;
  if (random_g != current_group) {
    genes[client_idx] = random_g;
    return true;
  }

  return false;
}

bool Mutator::ApplyMicroSplitMutation(Individual &child,
                                      double stagnation_factor, int level,
                                      std::mt19937 &rng) {
  if (!split_ptr_)
    return false;

  int n_perm = (int)evaluator_->GetPermutation().size();
  if (n_perm < 10)
    return false;

  // Logika zakresu z Island przeniesiona tutaj
  // Level-based window sizes: L2 = small (local), L0 = large (exploration)
  int base_min, base_max;
  switch (level) {
  case 0:
    base_min = 20;
    base_max = std::max(40, n_perm / 8);
    break; // Large windows
  case 1:
    base_min = 15;
    base_max = std::max(30, n_perm / 10);
    break; // Medium
  default:
    base_min = 8;
    base_max = std::max(15, n_perm / 15);
    break; // L2: Small windows
  }

  // Rozszerzamy zakres w miarę stagnacji
  int current_max = base_max + (int)((n_perm * 0.3) * stagnation_factor);
  int current_min = base_min + (int)(20 * stagnation_factor);

  current_max = std::min(current_max, n_perm - 1);
  current_min = std::min(current_min, current_max - 1);

  if (current_min < 5)
    current_min = 5; // Safety

  std::uniform_int_distribution<int> dist_len(current_min, current_max);
  int window_len = dist_len(rng);

  if (window_len >= n_perm)
    window_len = n_perm - 1;

  std::uniform_int_distribution<int> dist_start(0, n_perm - window_len);
  int start_idx = dist_start(rng);
  int end_idx = start_idx + window_len - 1;

  // Wywołanie metody Split (zakładamy że Split ma metodę ApplyMicroSplit
  // publiczną)
  split_ptr_->ApplyMicroSplit(child, start_idx, end_idx, geometry_, rng);

  return true;
}

bool Mutator::AggressiveMutate(Individual &indiv, std::mt19937 &rng) {
  std::vector<int> &genes = indiv.AccessGenotype();
  if (genes.empty())
    return false;

  int min_g = evaluator_->GetLowerBound();
  int max_g = evaluator_->GetUpperBound();
  int size = static_cast<int>(genes.size());

  std::uniform_int_distribution<int> dist_idx(0, size - 1);
  std::uniform_int_distribution<int> dist_grp(min_g, max_g);

  int start_idx = dist_idx(rng);
  std::uniform_int_distribution<int> dist_end(start_idx, size - 1);
  int end_idx = dist_end(rng);

  if (dist_idx(rng) % 2 == 0) {
    // Shuffle fragmentu
    std::shuffle(genes.begin() + start_idx, genes.begin() + end_idx + 1, rng);
  } else {
    // Przypisanie losowej grupy (niszczące)
    for (int i = start_idx; i <= end_idx; i++) {
      genes[i] = dist_grp(rng);
    }
  }
  return true;
}

bool Mutator::ApplySimpleMutation(Individual &indiv, std::mt19937 &rng) {
  int min_g = evaluator_->GetLowerBound();
  int max_g = evaluator_->GetUpperBound();
  std::vector<int> &genes = indiv.AccessGenotype();
  if (genes.empty())
    return false;

  std::uniform_int_distribution<int> dist_idx(
      0, static_cast<int>(genes.size()) - 1);
  std::uniform_int_distribution<int> dist_grp(min_g, max_g);

  // Swap dwóch punktów lub zmiana grupy
  if (rng() % 2 == 0) {
    int idx1 = dist_idx(rng);
    int idx2 = dist_idx(rng);
    std::swap(genes[idx1], genes[idx2]);
  } else {
    int idx = dist_idx(rng);
    int grp = dist_grp(rng);
    while (grp == genes[idx] && min_g != max_g)
      grp = dist_grp(rng);
    genes[idx] = grp;
  }
  return true;
}

// ==================================================================================
// RETURN MINIMIZER - Operator targetujący klientów powodujących przepełnienie
// capacity Strategia: Znajdź grupy z wieloma powrotami, przenieś "winowajców"
// do innych grup
// ==================================================================================
bool Mutator::ApplyReturnMinimizer(Individual &indiv, std::mt19937 &rng) {
  if (!evaluator_)
    return false;

  std::vector<int> &genes = indiv.AccessGenotype();
  int num_clients = static_cast<int>(genes.size());
  int num_groups = evaluator_->GetNumGroups();
  int capacity = evaluator_->GetCapacity();
  const auto &permutation = evaluator_->GetPermutation();

  // Krok 1: Zbuduj trasy (w kolejności permutacji) i znajdź klientów
  // powodujących overflow
  std::vector<std::vector<int>> routes(num_groups);
  for (int perm_id : permutation) {
    int client_idx = perm_id - 2; // permutation zawiera customer_id (2-based)
    if (client_idx >= 0 && client_idx < num_clients) {
      int g = genes[client_idx];
      if (g >= 0 && g < num_groups) {
        routes[g].push_back(client_idx);
      }
    }
  }

  // Krok 2: Dla każdej grupy, symuluj i znajdź klientów powodujących return
  // POPRAWKA: group_loads = SUMA demandów (do wyboru grupy docelowej)
  std::vector<int>
      overflow_clients; // client_idx klientów powodujących overflow
  std::vector<int> group_loads(num_groups,
                               0); // CAŁKOWITE obciążenie każdej grupy

  for (int g = 0; g < num_groups; ++g) {
    int current_load = 0; // Tylko do wykrywania overflow w symulacji
    int total_demand = 0; // Suma wszystkich demandów w grupie

    for (int client_idx : routes[g]) {
      int customer_id = client_idx + 2;
      int demand = evaluator_->GetDemand(customer_id);
      total_demand += demand; // Sumuj zawsze

      if (current_load + demand > capacity) {
        // Ten klient powoduje overflow!
        overflow_clients.push_back(client_idx);
        current_load = demand; // Reset po powrocie do depot
      } else {
        current_load += demand;
      }
    }
    group_loads[g] = total_demand; // POPRAWKA: cały demand, nie ostatni segment
  }

  if (overflow_clients.empty()) {
    return false; // Brak klientów do optymalizacji
  }

  // Krok 3: Wybierz losowego "winowajcę" i spróbuj go przenieść
  // OPTIMIZATION: Smart Target Selection (Geometry Aware)
  std::shuffle(overflow_clients.begin(), overflow_clients.end(), rng);

  int attempts = std::min(5, (int)overflow_clients.size());
  bool any_change = false;

  for (int a = 0; a < attempts; ++a) {
    int victim_idx = overflow_clients[a];
    int victim_id = victim_idx + 2;
    int victim_demand = evaluator_->GetDemand(victim_id);
    int old_group = genes[victim_idx];

    // Szukamy najlepszej grupy docelowej
    int best_target = -1;

    // Strategia 1: Sprawdź sąsiadów geometrycznych
    // Jeśli sąsiad jest w grupie, która ma miejsce -> BINGO (Cluster effect!)
    if (geometry_) {
      const auto &neighbors = geometry_->GetNeighbors(victim_idx);
      for (int neighbor : neighbors) {
        if (neighbor >= num_clients)
          continue;
        int neighbor_group = genes[neighbor];
        if (neighbor_group == old_group)
          continue;
        if (neighbor_group < 0 || neighbor_group >= num_groups)
          continue;

        // Sprawdź capacity
        int slack = capacity - group_loads[neighbor_group];
        if (slack >= victim_demand) {
          best_target = neighbor_group;
          break; // Znaleźliśmy świetnego kandydata (blisko i ma miejsce)
        }
      }
    }

    // Strategia 2: Jeśli Strategy 1 zawiodła, szukaj grupy z największym
    // Slackiem (Best Fit / Worst Fit)
    if (best_target == -1) {
      int max_slack = -1;
      for (int g = 0; g < num_groups; ++g) {
        if (g == old_group)
          continue;

        int slack = capacity - group_loads[g];
        if (slack >= victim_demand && slack > max_slack) {
          max_slack = slack;
          best_target = g;
        }
      }
    }

    // Strategia 3: Jeśli nadal nic (brak miejsca w żadnej grupie), wybierz
    // grupę najmniej obciążoną (nawet jeśli overflow, to mniejszy niż u nas -
    // Load Balancing)
    if (best_target == -1) {
      int min_load = INT_MAX;
      for (int g = 0; g < num_groups; ++g) {
        if (g == old_group)
          continue;
        if (group_loads[g] < min_load) {
          min_load = group_loads[g];
          best_target = g;
        }
      }
    }

    // Wykonaj ruch
    if (best_target != -1 && best_target != old_group) {
      genes[victim_idx] = best_target;
      group_loads[old_group] -= victim_demand;
      group_loads[best_target] += victim_demand;
      any_change = true;
    }
  }

  return any_change;
}

// ==================================================================================
// MERGE-SPLIT - Combine two groups and redistribute customers
// Strategy: Select two groups, merge them, then redistribute to balance
// capacities Simpler implementation that doesn't rely on ApplyMicroSplit
// ==================================================================================
bool Mutator::ApplyMergeSplit(Individual &indiv, std::mt19937 &rng) {
  if (!evaluator_)
    return false;

  std::vector<int> &genes = indiv.AccessGenotype();
  int num_clients = static_cast<int>(genes.size());
  int num_groups = evaluator_->GetNumGroups();
  const auto &permutation = evaluator_->GetPermutation();
  int capacity = evaluator_->GetCapacity();

  if (num_groups < 2 || num_clients < 2)
    return false;

  // Step 1: Count clients per group
  std::vector<int> group_counts(num_groups, 0);
  for (int i = 0; i < num_clients; ++i) {
    int g = genes[i];
    if (g >= 0 && g < num_groups) {
      group_counts[g]++;
    }
  }

  // Step 2: Find two non-empty groups to merge
  std::vector<int> non_empty_groups;
  for (int g = 0; g < num_groups; ++g) {
    if (group_counts[g] > 0) {
      non_empty_groups.push_back(g);
    }
  }

  if (non_empty_groups.size() < 2)
    return false;

  // Pick two random groups
  int idx1 = rng() % non_empty_groups.size();
  int idx2;
  do {
    idx2 = rng() % non_empty_groups.size();
  } while (idx2 == idx1);

  int g1 = non_empty_groups[idx1];
  int g2 = non_empty_groups[idx2];

  // Step 3: Build merged route in permutation order
  std::vector<int> merged_clients;
  merged_clients.reserve(group_counts[g1] + group_counts[g2]);

  for (int perm_id : permutation) {
    // Skip depot (ID=1) and invalid IDs
    if (perm_id < 2)
      continue;
    int client_idx = perm_id - 2;
    if (client_idx < 0 || client_idx >= num_clients)
      continue;
    int g = genes[client_idx];
    if (g == g1 || g == g2) {
      merged_clients.push_back(client_idx);
    }
  }

  if (merged_clients.size() < 2)
    return false;

  // Step 4: Re-split by greedy packing to g1 and g2
  int current_load = 0;
  bool use_g1 = true;

  for (int client_idx : merged_clients) {
    int customer_id = client_idx + 2;
    int demand = evaluator_->GetDemand(customer_id);

    // If adding this client would exceed capacity, switch groups
    if (current_load + demand > capacity && current_load > 0) {
      use_g1 = !use_g1; // Flip to other group
      current_load = 0;
    }

    genes[client_idx] = use_g1 ? g1 : g2;
    current_load += demand;
  }

  return true;
}
