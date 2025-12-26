#include "Island.hpp"
#include "Constants.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <cmath>

using namespace LcVRPContest;
using namespace std;

Island::Island(ThreadSafeEvaluator* evaluator, const ProblemData& data, int population_size, int id)
    : evaluator_(evaluator),
    demands_(data.GetDemands()),
    capacity_(data.GetCapacity()),
    geometry_(data, id),
    local_search_(evaluator, &geometry_, id),
    population_size_(population_size),
    id_(id),
    current_best_(evaluator->GetSolutionSize()),
    rng_(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count() + id)),
    split_(evaluator)
{
#ifdef RESEARCH
    InitStats();
#endif
    population_.reserve(population_size_);
    local_cache_.InitHistory(evaluator->GetSolutionSize());

    int n = evaluator_->GetSolutionSize();
    int g = evaluator_->GetNumGroups();

    // Bufory Broken Pairs
    last_in_group1.resize(g);
    last_in_group2.resize(g);
    pred1.resize(n);
    pred2.resize(n);

    // Bufory Spatial Ruin
    group_centroids_buffer_.resize(g);
    is_removed_buffer_.resize(n, false);
    removed_indices_buffer_.reserve(n);

    // Init customer ranks
    int dim = data.GetDimension();
    customer_ranks_.resize(dim + 1, 0);
    const auto& perm = data.GetPermutation();
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= 0 && perm[i] < (int)customer_ranks_.size()) {
            customer_ranks_[perm[i]] = static_cast<int>(i);
        }
    }
}

double Island::EvaluateWithHistoryPenalty(const std::vector<int>& genotype) {
#ifdef RESEARCH
    // hist++; 
#endif
    double base_cost = SafeEvaluate(genotype);
    if (base_cost >= 1e20) return base_cost;

    double history_penalty = 0.0;
    double lambda = Config::HISTORY_LAMBDA;

    size_t size = genotype.size();
    if (size < 2) return base_cost;

    for (size_t i = 0; i < size - 1; ++i) {
        if (genotype[i] == genotype[i + 1]) {
            int freq = local_cache_.GetFrequency(i);
            history_penalty += freq * lambda;
        }
    }
    return base_cost + history_penalty;
}

double Island::SafeEvaluate(const std::vector<int>& genotype) {
#ifdef RESEARCH
    total_evaluations++;
#endif
    double distance = 0.0;
    int returns = 0;

    if (local_cache_.TryGet(genotype, distance, returns)) {
#ifdef RESEARCH
        local_cache_hits++;
#endif
        return distance;
    }
    EvaluationResult result = evaluator_->EvaluateWithStats(genotype);

    distance = result.fitness;
    returns = result.returns;

    if (distance >= 1e9 || distance < 0.0) {
        distance = std::numeric_limits<double>::max();
    }
    local_cache_.Insert(genotype, distance, returns);

    return distance;
}

double Island::SafeEvaluate(Individual& indiv) {
#ifdef RESEARCH
    total_evaluations++;
#endif
    double distance = 0.0;
    int returns = 0;
    if (local_cache_.TryGet(indiv.AccessGenotype(), distance, returns)) {
#ifdef RESEARCH
        local_cache_hits++;
#endif
        indiv.SetReturnCount(returns);
        indiv.SetFitness(distance);
        return distance;
    }

    EvaluationResult result = evaluator_->EvaluateWithStats(indiv.GetGenotype());

    distance = result.fitness;
    returns = result.returns;

    if (distance >= 1e15 || distance < 0.0) {
        distance = std::numeric_limits<double>::max();
    }

    indiv.SetReturnCount(returns);
    indiv.SetFitness(distance);

    local_cache_.Insert(indiv.AccessGenotype(), distance, returns);

    return distance;
}

void Island::InitIndividual(Individual& indiv, INITIALIZATION_TYPE strategy) {
    int num_groups = evaluator_->GetNumGroups();
    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());

    std::vector<int> assignment_pool;
    assignment_pool.reserve(num_clients);
    for (int g = 0; g < num_groups; ++g) {
        assignment_pool.push_back(g);
    }
    int remaining_slots = num_clients - num_groups;

    if (remaining_slots > 0) {
        if (strategy == INITIALIZATION_TYPE::RR) {
            for (int i = 0; i < remaining_slots; ++i) {
                assignment_pool.push_back(i % num_groups);
            }
        }
        else if (strategy == INITIALIZATION_TYPE::CHUNKED) {
            int chunk_size = (num_groups > 0) ? (num_clients / num_groups) : num_clients;
            int current_group = 0;
            for (int i = 0; i < num_clients; ++i) {
                genes[i] = current_group;
                if ((i + 1) % chunk_size == 0 && current_group < num_groups - 1) {
                    current_group++;
                }
            }
            if (num_clients > 2) {
                std::uniform_int_distribution<int> d_swap(0, num_clients - 1);
                int a = d_swap(rng_);
                int b = d_swap(rng_);
                std::swap(genes[a], genes[b]);
            }
            return;
        }
        else {
            std::uniform_int_distribution<int> dist(0, num_groups - 1);
            for (int i = 0; i < remaining_slots; ++i) {
                assignment_pool.push_back(dist(rng_));
            }
        }
    }
    if ((int)assignment_pool.size() < num_clients) {
        assignment_pool.resize(num_clients, 0);
    }

    std::shuffle(assignment_pool.begin(), assignment_pool.end(), rng_);

    for (int i = 0; i < num_clients; ++i) {
        genes[i] = assignment_pool[i];
    }
}

void Island::Initialize(INITIALIZATION_TYPE strategy) {
    geometry_.Initialize(evaluator_);

    population_.clear();
    int sol_size = evaluator_->GetSolutionSize();
    Individual splitInd(sol_size);
    if (!Config::split) {
        InitIndividual(splitInd, INITIALIZATION_TYPE::RANDOM);
        population_.push_back(splitInd);
    }
    else {
        ApplySplitToIndividual(splitInd);
        double splitFit = SafeEvaluate(splitInd);
        splitInd.SetFitness(splitFit);
        population_.push_back(splitInd);
        current_best_ = splitInd;
        std::cout << "Split fitness: " << splitFit << std::endl;
    }

    for (int i = 0; i < population_size_ - 1; ++i) {
        Individual indiv(sol_size);
        InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);

        double fit = SafeEvaluate(indiv);
        indiv.SetFitness(fit);
        population_.push_back(indiv);

        if (fit < current_best_.GetFitness()) {
            current_best_ = indiv;
        }
    }
    UpdateBiasedFitness();
}

void Island::RunGeneration() {
    current_generation_++;
    UpdateAdaptiveParameters();
    stagnation_count_++;

    const int lambda = Config::ISLAND_POPULATION_SIZE;
    std::vector<Individual> offspring_pool;
    offspring_pool.reserve(lambda);

    double fitness_threshold = std::numeric_limits<double>::max();
    if (!population_.empty()) {
        fitness_threshold = population_[population_.size() / 2].GetFitness();
    }

    std::uniform_real_distribution<double> d(0.0, 1.0);

    for (int i = 0; i < lambda; ++i) {
        Individual child(evaluator_->GetSolutionSize());

        int p1 = SelectParentIndex();
        int p2 = SelectParentIndex();

        if (p1 >= 0 && p2 >= 0) {
            child = Crossover(population_[p1], population_[p2]);
        }
        else {
            InitIndividual(child, INITIALIZATION_TYPE::RANDOM);
        }

        bool mutated = false;
        if (d(rng_) < Config::MICROSPLITCHANCE) {
            ApplyMicroSplitMutation(child);
            mutated = true;
        }
        if (!mutated || d(rng_) < adaptive_mutation_rate_) {
            ApplyMutation(child);
        }

        if (d(rng_) < Config::LOADBALANCECHANCE) {
            if (!ApplyLoadBalancingChainMutation(child)) {
                ApplyLoadBalancingSwapMutation(child);
            }
        }

        child.Canonicalize();

        double current_fit = 0.0;
        int returns = 0;

        if (!local_cache_.TryGet(child.GetGenotype(), current_fit, returns)) {
            EvaluationResult result = evaluator_->EvaluateWithStats(child.GetGenotype());
            current_fit = result.fitness;
            returns = result.returns;
            local_cache_.Insert(child.GetGenotype(), current_fit, returns);
        }
        child.SetFitness(current_fit);
        child.SetReturnCount(returns);

        bool is_promising = (current_fit < fitness_threshold);
        bool lucky_shot = (d(rng_) < adaptive_vnd_prob_);

        if (is_promising || lucky_shot) {
            // TU ZMIANA: u¿ycie local_search_
            if (local_search_.RunVND(child)) {
                child.Canonicalize();
                if (!local_cache_.TryGet(child.GetGenotype(), current_fit, returns)) {
                    EvaluationResult result = evaluator_->EvaluateWithStats(child.GetGenotype());
                    current_fit = result.fitness;
                    returns = result.returns;
                    local_cache_.Insert(child.GetGenotype(), current_fit, returns);
                }
                child.SetFitness(current_fit);
                child.SetReturnCount(returns);
            }
        }

        bool is_duplicate = false;
        const double EPSILON = 1e-6;

        for (const auto& existing : population_) {
            if (std::abs(existing.GetFitness() - child.GetFitness()) < EPSILON) {
                if (existing.GetGenotype() == child.GetGenotype()) {
                    is_duplicate = true;
                    break;
                }
            }
        }

        if (!is_duplicate) {
            for (const auto& existing : offspring_pool) {
                if (std::abs(existing.GetFitness() - child.GetFitness()) < EPSILON) {
                    if (existing.GetGenotype() == child.GetGenotype()) {
                        is_duplicate = true;
                        break;
                    }
                }
            }
        }

        if (!is_duplicate) {
            offspring_pool.push_back(std::move(child));

            if (current_fit < current_best_.GetFitness()) {
                current_best_ = offspring_pool.back();
                stagnation_count_ = 0;
                local_cache_.UpdateHistory(current_best_.GetGenotype());
                last_improvement_gen_ = current_generation_;
                fitness_threshold = current_fit * 1.05;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(population_mutex_);

        population_.insert(population_.end(),
            std::make_move_iterator(offspring_pool.begin()),
            std::make_move_iterator(offspring_pool.end()));

        std::sort(population_.begin(), population_.end(), [](const Individual& a, const Individual& b) {
            return a.GetFitness() < b.GetFitness();
            });

        if ((int)population_.size() > population_size_) {
            population_.resize(population_size_);
        }
        UpdateBiasedFitness();
    }

    long long time_since_imp = current_generation_ - last_improvement_gen_;
    if (time_since_imp > BASE_STAGNATION_LIMIT && current_cv_ < 0.005) {
        Catastrophy();
        last_catastrophy_gen_ = current_generation_;
    }
}

/*
long long total_improved = 0;
void Island::RunGeneration() {
    // ==================================================================================
    //                                  DEBUG CONFIGURATION
    // Ustaw na TRUE tylko jeden (lub wybraną kombinację) modułów, aby go przetestować.
    // ==================================================================================

    // 1. Crossover (Krzyżowanie)
    // Jeśli FALSE: Dziecko jest klonem pierwszego rodzica (P1).
    const bool ENABLE_CROSSOVER = false;

    // 2. Mutacje (Wybierz jedną lub żadną)
    const bool ENABLE_MICRO_SPLIT = true;
    const bool ENABLE_MUTATION_STD = true; // Standard / Aggressive / Spatial (zależne od logiki w ApplyMutation)
    const bool ENABLE_LOAD_BALANCE = false; // LB Swap

    // 3. Local Search (VND)
    const bool ENABLE_VND = true;

    // 4. Tryb FORCE (Wymuszenie)
    // Jeśli TRUE: Ignoruje prawdopodobieństwa (np. 0.1) i wykonuje operację ZAWSZE.
    // Przydatne, żeby zobaczyć wpływ operatora na KAŻDEGO osobnika.
    const bool FORCE_EXECUTION = true;

    // 5. Logging (Czy wypisywać delty fitnessu?)
    const bool VERBOSE_LOG = true;
    // ==================================================================================

    current_generation_++;
    stagnation_count_++;
    UpdateAdaptiveParameters();

    std::vector<Individual> offspring;
    const int lambda = Config::ISLAND_POPULATION_SIZE;
    offspring.reserve(lambda);

    std::uniform_real_distribution<double> d(0.0, 1.0);

    // Statystyki dla tego debug-runa
    double sum_delta = 0.0;
    int improved_count = 0;
    int degraded_count = 0;

    for (int i = 0; i < lambda; ++i) {
        Individual child(evaluator_->GetSolutionSize());
        double parent_fitness = 0.0;

        // --- A. SELEKCJA I BAZA ---
        int p1 = SelectParentIndex();

        if (ENABLE_CROSSOVER) {
            int p2 = SelectParentIndex();
            if (p1 >= 0 && p2 >= 0) {
                child = Crossover(population_[p1], population_[p2]);
                // Jako punkt odniesienia bierzemy lepszego rodzica
                parent_fitness = std::min(population_[p1].GetFitness(), population_[p2].GetFitness());
            }
            else {
                InitIndividual(child, INITIALIZATION_TYPE::RANDOM);
                parent_fitness = 1e9;
            }
        }
        else {
            // Brak crossovera -> Klonujemy rodzica P1
            if (p1 >= 0) {
                child = population_[p1];
                parent_fitness = population_[p1].GetFitness();
            }
            else {
                InitIndividual(child, INITIALIZATION_TYPE::RANDOM);
            }
        }

        double fit_before_mut = parent_fitness; // Do śledzenia wpływu mutacji

        // --- B. MUTACJE ---
        bool mutated = false;

        // MicroSplit
        if (ENABLE_MICRO_SPLIT) {
            if (FORCE_EXECUTION || d(rng_) < Config::MICROSPLITCHANCE) {
                ApplyMicroSplitMutation(child);
                mutated = true;
            }
        }

        // Standard Mutation (Spatial / Aggressive / Smart)
        if (ENABLE_MUTATION_STD) {
            if (FORCE_EXECUTION || d(rng_) < adaptive_mutation_rate_) {
                // Tu możesz też na sztywno wpisać np. ApplySmartSpatialMove(child)
                // jeśli chcesz testować konkretną metodę z ApplyMutation
                ApplyMutation(child);
                mutated = true;
            }
        }

        // Load Balancing
        if (ENABLE_LOAD_BALANCE) {
            if (FORCE_EXECUTION || d(rng_) < Config::LOADBALANCECHANCE) {
                ApplyLoadBalancingSwapMutation(child);
                mutated = true;
            }
        }

        // --- C. EWALUACJA PO MUTACJACH ---
        child.Canonicalize();
        double fit_after_mut = SafeEvaluate(child);
        child.SetFitness(fit_after_mut);

        // --- D. LOCAL SEARCH (VND) ---
        if (ENABLE_VND) {
            // W trybie debug zazwyczaj chcemy widzieć jak VND działa na wszystko
            // ale możesz odkomentować warunek "is_promising"
            bool run_vnd = FORCE_EXECUTION;
            if (!FORCE_EXECUTION) {
                double median_fit = population_.empty() ? 1e9 : population_[population_.size() / 2].GetFitness();
                run_vnd = (fit_after_mut < median_fit) || (d(rng_) < adaptive_vnd_prob_);
            }

            if (run_vnd) {
                if (local_search_.RunVND(child)) {
                    child.Canonicalize();
                    double fit_vnd = SafeEvaluate(child);
                    child.SetFitness(fit_vnd);
                }
            }
        }

        double final_fit = child.GetFitness();

        // --- LOGOWANIE WPŁYWU OPERATORÓW ---
        if (VERBOSE_LOG) {
            double delta = final_fit - parent_fitness;
            sum_delta += delta;
            if (delta < -1e-3) {
                total_improved++;
                improved_count++;
            }// Lepszy (mniejszy koszt)
            else if (delta > 1e-3) degraded_count++; // Gorszy
        }

        // --- E. ELITYZM ---
        if (final_fit < current_best_.GetFitness()) {
            current_best_ = child;
            stagnation_count_ = 0;
            local_cache_.UpdateHistory(current_best_.GetGenotype());
            last_improvement_gen_ = current_generation_;
            if (VERBOSE_LOG) std::cout << " [!!! NEW BEST !!!] " << final_fit << std::endl;
        }

        // --- F. DODAWANIE (BEZ SPRAWDZANIA DUPLIKATÓW DLA SZYBKOŚCI DEBUGU) ---
        offspring.push_back(std::move(child));
    }

    // --- PODSUMOWANIE GENERACJI DEBUGOWEJ ---
    if (VERBOSE_LOG && current_generation_ % 10 == 0) {
        std::cout << "[DEBUG Gen " << current_generation_ << "] "
            << "Avg Delta: " << (sum_delta / lambda)
            << " | Improved: " << improved_count
            << " | Degraded: " << degraded_count
            << " | Best: " << current_best_.GetFitness()
            << " | avg improved: "<<(double) total_improved/current_generation_<< std::endl;
    }

    // --- G. SUKCESJA (STANDARD) ---
    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        if (!offspring.empty()) {
            population_.reserve(population_.size() + offspring.size());
            std::move(offspring.begin(), offspring.end(), std::back_inserter(population_));
        }
        UpdateBiasedFitness();
        std::sort(population_.begin(), population_.end(), [](const Individual& a, const Individual& b) {
            return a.GetBiasedFitness() < b.GetBiasedFitness();
            });
        if ((int)population_.size() > population_size_) {
            population_.resize(population_size_);
        }
    }
}
// --- RESTORED METHODS START HERE ---
*/


void Island::Catastrophy() {
#ifdef RESEARCH
    catastrophy_activations++;
#endif
    cout << "Catastrophy on island [" << id_ << "] CV: " << std::scientific << std::setprecision(2) << current_cv_ << std::defaultfloat << endl;

    std::vector<Individual> new_pop;
    new_pop.reserve(population_size_);
    new_pop.push_back(current_best_);
    int sol_size = evaluator_->GetSolutionSize();

    std::vector<Individual> candidates;
    int candidates_count = population_size_ * 10;

    for (int i = 0; i < candidates_count; ++i) {
        Individual indiv(sol_size);
        if (i % 2 == 0) InitIndividual(indiv, INITIALIZATION_TYPE::RANDOM);
        else InitIndividual(indiv, INITIALIZATION_TYPE::CHUNKED);

        double penalized_fit = EvaluateWithHistoryPenalty(indiv.GetGenotype());
        indiv.SetFitness(penalized_fit);
        candidates.push_back(indiv);
    }

    std::sort(candidates.begin(), candidates.end());

    for (int i = 0; i < population_size_ - 1; ++i) {
        Individual& selected = candidates[i];

        if (i < 5) local_search_.RunVND(selected); 

        double clean_fit = SafeEvaluate(selected);
        selected.SetFitness(clean_fit);

        new_pop.push_back(selected);
    }

    {
        std::lock_guard<std::mutex> lock(population_mutex_);
        population_ = new_pop;
        UpdateBiasedFitness();
    }
    CalculatePopulationCV();
    local_cache_.ClearHistory();
}

void Island::UpdateBiasedFitness() {
    int pop_size = static_cast<int>(population_.size());
    if (pop_size == 0) return;

    std::vector<int> indices(pop_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return population_[a].GetFitness() < population_[b].GetFitness();
        });

    const std::vector<int>& perm = evaluator_->GetPermutation();
    int num_groups = evaluator_->GetNumGroups();

    int ref_size = std::min(pop_size, std::max(2, Config::ELITERATIO));

    double total_population_bpd = 0.0;
    int measurements_count = 0;

    for (int i = 0; i < pop_size; ++i) {
        double dist_sum = 0.0;
        int idx_i = indices[i];
        int comparisons = 0;

        for (int k = 0; k < ref_size; ++k) {
            int idx_best = indices[k];

            if (idx_i == idx_best) continue;

            int bpd = CalculateBrokenPairsDistance(population_[idx_i], population_[idx_best], perm, num_groups);

            dist_sum += bpd;
            comparisons++;
        }

        double avg_dist = (comparisons > 0) ? (dist_sum / comparisons) : 0.0;

        population_[idx_i].SetDiversityScore(avg_dist);

        total_population_bpd += avg_dist;
        measurements_count++;
    }
    if (measurements_count > 0) {
        double avg_raw_bpd = total_population_bpd / measurements_count;
        current_structural_diversity_ = avg_raw_bpd / (double)evaluator_->GetSolutionSize();
    }
    else {
        current_structural_diversity_ = 0.0;
    }

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return population_[a].GetDiversityScore() > population_[b].GetDiversityScore();
        });

    std::vector<int> rank_diversity(pop_size);
    for (int r = 0; r < pop_size; ++r) rank_diversity[indices[r]] = r;


    std::vector<int> rank_fitness(pop_size);
    std::vector<int> fit_indices(pop_size);
    std::iota(fit_indices.begin(), fit_indices.end(), 0);
    std::sort(fit_indices.begin(), fit_indices.end(), [&](int a, int b) {
        return population_[a].GetFitness() < population_[b].GetFitness();
        });
    for (int r = 0; r < pop_size; ++r) rank_fitness[fit_indices[r]] = r;

    double elite_ratio = (double)ref_size / (double)pop_size;
    for (int i = 0; i < pop_size; ++i) {
        double biased = (double)rank_fitness[i] + (1.0 - elite_ratio) * (double)rank_diversity[i];
        population_[i].SetBiasedFitness(biased);
    }

    current_cv_ = current_structural_diversity_;
}

int Island::CalculateBrokenPairsDistance(const Individual& ind1, const Individual& ind2, const std::vector<int>& permutation, int num_groups) {
    const std::vector<int>& g1 = ind1.GetGenotype();
    const std::vector<int>& g2 = ind2.GetGenotype();

    int size = static_cast<int>(g1.size());
    if (size == 0) return 0;

    std::fill(last_in_group1.begin(), last_in_group1.end(), -1);
    std::fill(last_in_group2.begin(), last_in_group2.end(), -1);

    for (int customer_id : permutation) {
        int idx = customer_id - 2;

        if (idx < 0 || idx >= size) continue;

        int group1 = g1[idx];
        if (group1 >= 0 && group1 < num_groups) {
            pred1[idx] = last_in_group1[group1];
            last_in_group1[group1] = idx;
        }
        else {
            pred1[idx] = -2;
        }

        int group2 = g2[idx];
        if (group2 >= 0 && group2 < num_groups) {
            pred2[idx] = last_in_group2[group2];
            last_in_group2[group2] = idx;
        }
        else {
            pred2[idx] = -2;
        }
    }

    int distance = 0;
    for (int i = 0; i < size; ++i) {
        if (pred1[i] != pred2[i]) {
            distance++;
        }
    }

    return distance;
}

void Island::CalculatePopulationCV() {
    if (population_.empty()) {
        current_cv_ = 0.0;
        return;
    }
    double mean = 0.0;
    double M2 = 0.0;
    int n = 0;

    for (const auto& ind : population_) {
        double x = ind.GetFitness();

        if (x > 1e14) continue;

        n++;
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    if (n < 2) {
        current_cv_ = 1.0;
        return;
    }

    double variance = M2 / (n - 1);
    double std_dev = std::sqrt(variance);

    if (mean > 1e-6) {
        current_cv_ = std_dev / mean;
    }
    else {
        current_cv_ = 0.0;
    }
}

int Island::ApplyMutation(Individual& child) {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    double rnd = d(rng_);
    int executed_op = -1;

    if (rnd < adaptive_ruin_chance_) {
        if (d(rng_) < 0.05) {
            AggresiveMutate(child);
#ifdef RESEARCH
            executed_op = (int)OpType::MUT_AGGRESSIVE;
#endif
        }

        else if (d(rng_) < 0.35) {
            ApplySmartSpatialMove(child);
        }

        else {
            SpatialRuinAndRecreate(child);
#ifdef RESEARCH
            executed_op = (int)OpType::MUT_SPATIAL;
#endif
        }
    }
    return executed_op;
}

bool Island::ApplySmartSpatialMove(Individual& indiv) {
    std::vector<int>& genes = indiv.AccessGenotype();
    int size = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();

    // Wybierz losowego klienta
    int client_idx = rng_() % size;
    int current_group = genes[client_idx];

    // Pobierz jego najbliższych sąsiadów geometrycznych (nie z permutacji!)
    const auto& neighbors = geometry_.GetNeighbors(client_idx);
    if (neighbors.empty()) return false;

    // Sprawdź gdzie są sąsiedzi
    int best_target_group = -1;

    // Patrzymy na 3 najbliższych sąsiadów
    for (int k = 0; k < std::min((int)neighbors.size(), 3); ++k) {
        int neighbor_idx = neighbors[k];
        int neighbor_group = genes[neighbor_idx];

        // Jeśli sąsiad jest w innej grupie, to jest kandydat
        if (neighbor_group != current_group && neighbor_group >= 0 && neighbor_group < num_groups) {
            best_target_group = neighbor_group;
            break; // Znaleziono najbliższą "inną" grupę
        }
    }


    if (best_target_group != -1) {
        // Przenieś klienta do grupy jego sąsiada
        genes[client_idx] = best_target_group;
        return true;
    }

    return false;
}




int Island::ApplyMicroSplitMutation(Individual& child) {
    int executed_op = -1;
    int n_perm = (int)evaluator_->GetPermutation().size();

    int min_len = 15;
    int max_len = std::min(150, std::max(20, n_perm / 5));

    std::uniform_int_distribution<int> dist_len(min_len, max_len);
    int window_len = dist_len(rng_);

    if (window_len >= n_perm) window_len = n_perm - 1;

    std::uniform_int_distribution<int> dist_start(0, n_perm - window_len);
    int start_idx = dist_start(rng_);
    int end_idx = start_idx + window_len - 1;

    split_.ApplyMicroSplit(child, start_idx, end_idx, &geometry_);

#ifdef RESEARCH
    executed_op = (int)OpType::MUT_SIMPLE;
#endif
    return executed_op;
}

int Island::ApplyLoadBalancing(Individual& child) {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    double rnd = d(rng_);
    int executed_op = -1;

    if (rnd < 0.1) {
        if (ApplyLoadBalancingChainMutation(child)) {}
#ifdef RESEARCH
        executed_op = (int)OpType::LB_CHAIN;
#endif
    }
    else if (rnd < 0.45) {
        if (ApplyLoadBalancingSwapMutation(child)) {}
#ifdef RESEARCH
        executed_op = (int)OpType::LB_SWAP;
#endif
    }
    else {
        if (ApplyLoadBalancingMutation(child)) {}
#ifdef RESEARCH
        executed_op = (int)OpType::LB_SIMPLE;
#endif
    }
    return executed_op;
}

bool Island::SpatialRuinAndRecreate(Individual& indiv) {
#ifdef RESEARCH
    spatial_activations++;
#endif
    std::vector<int>& genes = indiv.AccessGenotype();
    int num_clients = static_cast<int>(genes.size());
    int num_groups = evaluator_->GetNumGroups();

    if (!geometry_.HasCoordinates()) return false;

    int center_idx = rng_() % num_clients;

    int min_rem = 5;
    int target_max = (stagnation_count_ > 1000) ? (int)(num_clients * 0.4) : (int)(num_clients * 0.20);
    int max_rem = std::max(min_rem + 5, target_max);
    if (max_rem >= num_clients) max_rem = num_clients - 1;

    const auto& neighbors = geometry_.GetNeighbors(center_idx);
    int available_neighbors = static_cast<int>(neighbors.size());
    if (max_rem > available_neighbors) max_rem = available_neighbors;

    int range = max_rem - min_rem;
    if (range <= 0) range = 1;
    int num_removed = min_rem + (rng_() % range);
    if (num_removed > available_neighbors) num_removed = available_neighbors;
    if (num_removed < 1) return false;

    // Buffer Hoisting (u¿ycie buforów klasy zamiast lokalnych)
    removed_indices_buffer_.clear();
    removed_indices_buffer_.push_back(center_idx);

    for (int i = 0; i < num_removed; ++i) {
        removed_indices_buffer_.push_back(neighbors[i]);
    }

    std::fill(group_centroids_buffer_.begin(), group_centroids_buffer_.end(), GroupInfo{ 0,0,0 });
    std::fill(is_removed_buffer_.begin(), is_removed_buffer_.end(), false);

    for (int idx : removed_indices_buffer_) is_removed_buffer_[idx] = true;

    for (int i = 0; i < num_clients; ++i) {
        if (is_removed_buffer_[i]) continue;

        int g = genes[i];
        if (g >= 0 && g < num_groups) {
            int coord_idx = i + 1;
            const auto& coord = geometry_.GetCoordinate(coord_idx);
            group_centroids_buffer_[g].sum_x += coord.x;
            group_centroids_buffer_[g].sum_y += coord.y;
            group_centroids_buffer_[g].count++;
        }
    }

    for (int idx : removed_indices_buffer_) {
        genes[idx] = -1;
    }

    std::shuffle(removed_indices_buffer_.begin(), removed_indices_buffer_.end(), rng_);

    for (int client_idx : removed_indices_buffer_) {
        int coord_idx = client_idx + 1;
        const auto& client_coord = geometry_.GetCoordinate(coord_idx);

        int best_g = -1;
        double best_dist_sq = std::numeric_limits<double>::max();

        for (int g = 0; g < num_groups; ++g) {
            double dist_sq = 1e15;

            if (group_centroids_buffer_[g].count > 0) {
                double gx = group_centroids_buffer_[g].sum_x / group_centroids_buffer_[g].count;
                double gy = group_centroids_buffer_[g].sum_y / group_centroids_buffer_[g].count;
                double dx = client_coord.x - gx;
                double dy = client_coord.y - gy;
                dist_sq = dx * dx + dy * dy;
            }
            else {
                dist_sq = 1e9;
            }

            if (dist_sq < best_dist_sq) {
                best_dist_sq = dist_sq;
                best_g = g;
            }
        }
        if (best_g == -1) best_g = rng_() % num_groups;

        genes[client_idx] = best_g;
        group_centroids_buffer_[best_g].sum_x += client_coord.x;
        group_centroids_buffer_[best_g].sum_y += client_coord.y;
        group_centroids_buffer_[best_g].count++;
    }
    return true;
}

Individual Island::CrossoverSpatial(const Individual& p1, const Individual& p2) {
    const std::vector<int>& g1 = p1.GetGenotype();
    const std::vector<int>& g2 = p2.GetGenotype();
    int size = static_cast<int>(g1.size());

    Individual child(size);
    std::vector<int>& child_genes = child.AccessGenotype();

    if (size == 0 || !geometry_.HasCoordinates()) return child;

    std::uniform_int_distribution<int> dist_idx(0, size - 1);
    int center_idx = dist_idx(rng_);

    const auto& center_coord = geometry_.GetCoordinate(center_idx + 1);

    int radius_idx = dist_idx(rng_);
    const auto& radius_coord = geometry_.GetCoordinate(radius_idx + 1);

    double r_sq = (center_coord.x - radius_coord.x) * (center_coord.x - radius_coord.x) +
        (center_coord.y - radius_coord.y) * (center_coord.y - radius_coord.y);

    for (int i = 0; i < size; ++i) {
        const auto& px = geometry_.GetCoordinate(i + 1);
        double dist_sq = (center_coord.x - px.x) * (center_coord.x - px.x) +
            (center_coord.y - px.y) * (center_coord.y - px.y);

        if (dist_sq <= r_sq) {
            child_genes[i] = g1[i];
        }
        else {
            child_genes[i] = g2[i];
        }
    }
    return child;
}

Individual Island::CrossoverSequence(const Individual& p1, const Individual& p2) {
    const std::vector<int>& perm = evaluator_->GetPermutation();
    int perm_size = static_cast<int>(perm.size());
    Individual child = p1;
    std::vector<int>& child_genes = child.AccessGenotype();
    const std::vector<int>& p2_genes = p2.GetGenotype();
    if (child_genes.empty()) return child;
    int cut_point = rng_() % perm_size;
    for (int i = cut_point; i < perm_size; ++i) {
        int customer_id = perm[i];
        int gene_idx = customer_id - 2;
        if (gene_idx >= 0 && gene_idx < (int)child_genes.size()) {
            child_genes[gene_idx] = p2_genes[gene_idx];
        }
    }
    return child;
}

Individual Island::Crossover(const Individual& p1, const Individual& p2) {
#ifdef RESEARCH
    crossovers++;
#endif
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (dist(rng_) < Config::CROSSOVERSEQ_PROBABILITY) {
        return CrossoverSequence(p1, p2);
    }
    else {
        return CrossoverSpatial(p1, p2);
    }
}

bool Island::Mutate(Individual& indiv) {
    int min_g = evaluator_->GetLowerBound();
    int max_g = evaluator_->GetUpperBound();
    vector<int>& genes = indiv.AccessGenotype();
    if (genes.empty()) return false;
    uniform_int_distribution<int> dist_idx(0, static_cast<int>(genes.size()) - 1);
    uniform_int_distribution<int> dist_grp(min_g, max_g);
    for (int k = 0; k < 2; k++) {
        int idx = dist_idx(rng_);
        int grp = dist_grp(rng_);
        while (grp == genes[idx]) grp = dist_grp(rng_);
        genes[idx] = grp;
    }
    return true;
}

bool Island::ApplyLoadBalancingMutation(Individual& individual) {
#ifdef RESEARCH
    load_balancing_activations++;
#endif
    std::vector<int>& solution = individual.AccessGenotype();
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    const std::vector<int>& demands = evaluator_->GetDemands();
    const std::vector<int>& perm = evaluator_->GetPermutation();
    std::vector<int> loads(num_groups, 0);
    for (int customer_id : perm) {
        int sol_idx = customer_id - 2;
        if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
            int g = solution[sol_idx];
            if (g >= 0 && g < num_groups) {
                loads[g] += demands[customer_id - 1];
            }
        }
    }
    std::vector<int> overloaded_groups;
    std::vector<int> underloaded_groups;
    for (int g = 0; g < num_groups; ++g) {
        if (loads[g] > capacity) overloaded_groups.push_back(g);
        else underloaded_groups.push_back(g);
    }
    if (overloaded_groups.empty()) return false;
    std::shuffle(overloaded_groups.begin(), overloaded_groups.end(), rng_);
    bool changed = false;
    for (int source_group : overloaded_groups) {
        if (underloaded_groups.empty()) break;
        std::vector<int> group_clients;
        for (int customer_id : perm) {
            int sol_idx = customer_id - 2;
            if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
                if (solution[sol_idx] == source_group) {
                    group_clients.push_back(customer_id);
                }
            }
        }
        std::shuffle(group_clients.begin(), group_clients.end(), rng_);
        for (int customer_to_move : group_clients) {
            if (loads[source_group] <= capacity) break;
            int demand = demands[customer_to_move - 1];
            int best_target = -1;
            int checks = 0;
            std::shuffle(underloaded_groups.begin(), underloaded_groups.end(), rng_);
            for (int target_group : underloaded_groups) {
                if (checks++ > 5) break;
                if (loads[target_group] + demand <= capacity) {
                    best_target = target_group;
                    break;
                }
            }
            if (best_target != -1) {
                int sol_idx = customer_to_move - 2;
                if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
                    solution[sol_idx] = best_target;
                    loads[source_group] -= demand;
                    loads[best_target] += demand;
                    changed = true;
                }
            }
        }
    }
    return changed;
}

bool Island::ApplyLoadBalancingChainMutation(Individual& individual) {
#ifdef RESEARCH
    load_balancing_activations++;
#endif
    std::vector<int>& solution = individual.AccessGenotype();
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    const std::vector<int>& demands = evaluator_->GetDemands();
    const std::vector<int>& perm = evaluator_->GetPermutation();
    std::vector<int> loads(num_groups, 0);
    std::vector<std::vector<int>> group_clients(num_groups);
    for (int customer_id : perm) {
        int sol_idx = customer_id - 2;
        if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
            int g = solution[sol_idx];
            if (g >= 0 && g < num_groups) {
                loads[g] += demands[customer_id - 1];
                group_clients[g].push_back(customer_id);
            }
        }
    }
    std::vector<int> overloaded_groups;
    for (int g = 0; g < num_groups; ++g) {
        if (loads[g] > capacity) overloaded_groups.push_back(g);
    }
    if (overloaded_groups.empty()) return false;
    std::shuffle(overloaded_groups.begin(), overloaded_groups.end(), rng_);
    struct Move { int customer_id; int from_group; int to_group; int demand; };
    const int MAX_DEPTH = 10;
    bool any_change = false;
    auto find_next_move = [&](int group_idx, const std::vector<bool>& visited) -> std::pair<int, int> {
        if (group_clients[group_idx].empty()) return { -1, -1 };
        std::shuffle(group_clients[group_idx].begin(), group_clients[group_idx].end(), rng_);
        int trials = std::min((int)group_clients[group_idx].size(), 10);
        for (int i = 0; i < trials; ++i) {
            int client_id = group_clients[group_idx][i];
            int demand = demands[client_id - 1];
            for (int g = 0; g < num_groups; ++g) {
                if (g == group_idx || visited[g]) continue;
                if (loads[g] + demand <= capacity) {
                    return { client_id, g };
                }
            }
        }
        for (int i = 0; i < trials; ++i) {
            int client_id = group_clients[group_idx][i];
            int target_candidate = rng_() % num_groups;
            for (int k = 0; k < 5; k++) {
                int g = (target_candidate + k) % num_groups;
                if (g != group_idx && !visited[g]) {
                    return { client_id, g };
                }
            }
        }
        return { -1, -1 };
        };
    for (int start_group : overloaded_groups) {
        if (loads[start_group] <= capacity) continue;
        std::vector<Move> chain_history;
        std::vector<bool> group_visited(num_groups, false);
        int current_group = start_group;
        group_visited[current_group] = true;
        bool chain_success = false;
        bool can_continue = true;
        for (int depth = 0; depth < MAX_DEPTH && !chain_success && can_continue; ++depth) {
            std::pair<int, int> move_data = find_next_move(current_group, group_visited);
            int best_client = move_data.first;
            int best_target = move_data.second;
            if (best_target == -1) {
                can_continue = false;
            }
            else {
                int demand = demands[best_client - 1];
                int sol_idx = best_client - 2;
                if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
                    solution[sol_idx] = best_target;
                    loads[current_group] -= demand;
                    loads[best_target] += demand;
                    auto it = std::find(group_clients[current_group].begin(), group_clients[current_group].end(), best_client);
                    if (it != group_clients[current_group].end()) {
                        *it = group_clients[current_group].back();
                        group_clients[current_group].pop_back();
                        group_clients[best_target].push_back(best_client);
                    }
                    chain_history.push_back({ best_client, current_group, best_target, demand });
                    group_visited[best_target] = true;
                }
                if (loads[best_target] <= capacity && loads[start_group] <= capacity) {
                    bool all_ok = true;
                    for (const auto& m : chain_history) {
                        if (loads[m.from_group] > capacity) {
                            all_ok = false;
                        }
                    }
                    if (all_ok && loads[current_group] <= capacity) {
                        chain_success = true;
                    }
                }
                current_group = best_target;
            }
        }
        if (!chain_success) {
            for (int i = (int)chain_history.size() - 1; i >= 0; --i) {
                const auto& m = chain_history[i];
                int sol_idx = m.customer_id - 2;
                if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
                    solution[sol_idx] = m.from_group;
                    loads[m.from_group] += m.demand;
                    loads[m.to_group] -= m.demand;
                }
            }
        }
        else {
            any_change = true;
        }
    }
    return any_change;
}

bool Island::ApplyLoadBalancingSwapMutation(Individual& individual) {
#ifdef RESEARCH
    load_balancing_activations++;
#endif
    std::vector<int>& solution = individual.AccessGenotype();
    int num_groups = evaluator_->GetNumGroups();
    int capacity = evaluator_->GetCapacity();
    const std::vector<int>& demands = evaluator_->GetDemands();
    const std::vector<int>& perm = evaluator_->GetPermutation();
    std::vector<int> loads(num_groups, 0);
    std::vector<std::vector<int>> group_clients(num_groups);
    for (int customer_id : perm) {
        int sol_idx = customer_id - 2;
        if (sol_idx >= 0 && sol_idx < (int)solution.size()) {
            int g = solution[sol_idx];
            if (g >= 0 && g < num_groups) {
                loads[g] += demands[customer_id - 1];
                group_clients[g].push_back(customer_id);
            }
        }
    }
    std::vector<int> overloaded_groups;
    for (int g = 0; g < num_groups; ++g) {
        if (loads[g] > capacity) overloaded_groups.push_back(g);
    }
    if (overloaded_groups.empty()) return false;
    std::shuffle(overloaded_groups.begin(), overloaded_groups.end(), rng_);
    bool changed = false;
    for (int source_group : overloaded_groups) {
        if (loads[source_group] <= capacity) continue;
        std::vector<int> target_groups(num_groups);
        std::iota(target_groups.begin(), target_groups.end(), 0);
        std::shuffle(target_groups.begin(), target_groups.end(), rng_);
        bool fixed_group = false;
        for (int target_group : target_groups) {
            if (target_group == source_group) continue;
            if (fixed_group) break;
            std::shuffle(group_clients[source_group].begin(), group_clients[source_group].end(), rng_);
            std::shuffle(group_clients[target_group].begin(), group_clients[target_group].end(), rng_);
            for (int client_a : group_clients[source_group]) {
                int demand_a = demands[client_a - 1];
                for (int client_b : group_clients[target_group]) {
                    int demand_b = demands[client_b - 1];
                    if (demand_a <= demand_b) continue;
                    if (loads[target_group] - demand_b + demand_a > capacity) continue;
                    int idx_a = client_a - 2;
                    int idx_b = client_b - 2;
                    if (idx_a >= 0 && idx_a < (int)solution.size() &&
                        idx_b >= 0 && idx_b < (int)solution.size()) {
                        solution[idx_a] = target_group;
                        solution[idx_b] = source_group;
                        loads[source_group] = loads[source_group] - demand_a + demand_b;
                        loads[target_group] = loads[target_group] - demand_b + demand_a;
                        changed = true;
                        if (loads[source_group] <= capacity) {
                            fixed_group = true;
                        }
                        goto next_target_search;
                    }
                }
            }
        next_target_search:;
        }
    }
    return changed;
}

bool Island::AggresiveMutate(Individual& indiv) {
#ifdef RESEARCH
    aggresive_mutation_activations++;
#endif
    vector<int>& genes = indiv.AccessGenotype();
    if (genes.empty()) return false;
    int min_g = evaluator_->GetLowerBound();
    int max_g = evaluator_->GetUpperBound();
    int size = static_cast<int>(genes.size());
    uniform_int_distribution<int> dist_idx(0, size - 1);
    uniform_int_distribution<int> dist_grp(min_g, max_g);
    int start_idx = dist_idx(rng_);
    uniform_int_distribution<int> dist_end(start_idx, size - 1);
    int end_idx = dist_end(rng_);
    if (dist_idx(rng_) % 2 == 0) {
        std::shuffle(genes.begin() + start_idx, genes.begin() + end_idx + 1, rng_);
    }
    else {
        for (int i = start_idx; i <= end_idx; i++) genes[i] = dist_grp(rng_);
    }
    return true;
}

void Island::PrintIndividual(const Individual& individual, int global_generation) const {
    int num_groups = evaluator_->GetNumGroups();
    vector<int> group_counts(num_groups, 0);
    const vector<int>& genes = individual.GetGenotype();
    for (int g : genes) if (g >= 0 && g < num_groups) group_counts[g]++;
    int extra_returns = evaluator_->GetTotalDepotReturns(genes);
    cout << "   [Island " << id_ << "] Gen: " << setw(6) << global_generation
        << " | Dist: " << fixed << setprecision(2) << individual.GetFitness()
        << " | Ret: " << extra_returns
        << " | Groups: [";
    for (size_t i = 0; i < group_counts.size(); ++i) cout << group_counts[i] << (i < group_counts.size() - 1 ? "," : "");
    cout << "] CV: " << std::scientific << std::setprecision(2) << current_cv_ << std::defaultfloat << endl;;
}

SplitResult Island::RunSplit(const std::vector<int>& permutation) {
    return split_.RunLinear(permutation);
}

void Island::ApplySplitToIndividual(Individual& indiv) {
    const std::vector<int>& global_perm = evaluator_->GetPermutation();
    int fleet_limit = evaluator_->GetNumGroups();
    SplitResult result = split_.RunLinear(global_perm);
    if (result.feasible) {
        std::vector<int>& genes = indiv.AccessGenotype();
        if (result.group_assignment.size() != genes.size()) return;
        int routes_count = static_cast<int>(result.optimized_routes.size());
        for (size_t i = 0; i < genes.size(); ++i) {
            int assigned_route_id = result.group_assignment[i];
            if (assigned_route_id < fleet_limit) {
                genes[i] = assigned_route_id;
            }
            else {
                genes[i] = assigned_route_id % fleet_limit;
            }
        }
        if (routes_count <= fleet_limit) {
            indiv.SetFitness(result.total_cost);
            indiv.SetReturnCount(0);
        }
        else {
            int excess_vehicles = routes_count - fleet_limit;
            indiv.SetFitness(result.total_cost);
            indiv.SetReturnCount(excess_vehicles);
        }
    }
    else {
        indiv.SetFitness(1.0e30);
    }
}

double Island::MapRange(double value, double in_min, double in_max, double out_min, double out_max) {
    if (value < in_min) value = in_min;
    if (value > in_max) value = in_max;
    double p = (value - in_min) / (in_max - in_min);
    return out_min + p * (out_max - out_min);
}

void Island::UpdateAdaptiveParameters() {
    double diversity = current_structural_diversity_;
    const double LOW_DIV = 0.01;
    const double HIGH_DIV = 0.15;
    adaptive_mutation_rate_ = MapRange(diversity, LOW_DIV, HIGH_DIV, 0.8, 0.05);
    adaptive_vnd_prob_ = MapRange(diversity, LOW_DIV, HIGH_DIV, 0.1, 1.0);
    adaptive_ruin_chance_ = MapRange(diversity, LOW_DIV, HIGH_DIV, 0.9, 0.1);
}

#ifdef RESEARCH
std::vector<int> Island::CanonicalizeGenotype(const std::vector<int>& genotype, int num_groups) const {
    if (genotype.empty()) return {};
    std::vector<int> canonical = genotype;
    std::vector<int> mapping(num_groups + 1, -1);
    int next_new_id = 0;
    for (size_t i = 0; i < canonical.size(); ++i) {
        int old_id = canonical[i];
        if (old_id < 0 || old_id >= (int)mapping.size()) continue;
        if (mapping[old_id] == -1) {
            mapping[old_id] = next_new_id++;
        }
        canonical[i] = mapping[old_id];
    }
    return canonical;
}

void Island::InitStats() {
    op_stats_.resize((int)OpType::COUNT);
    op_stats_[(int)OpType::CROSSOVER] = { "Cross", 0, 0 };
    op_stats_[(int)OpType::MUT_AGGRESSIVE] = { "Mut_Aggro", 0, 0 };
    op_stats_[(int)OpType::MUT_SPATIAL] = { "Mut_Spatial", 0, 0 };
    op_stats_[(int)OpType::MUT_SIMPLE] = { "Mut_Simple", 0, 0 };
    op_stats_[(int)OpType::LB_CHAIN] = { "LB_Chain", 0, 0 };
    op_stats_[(int)OpType::LB_SWAP] = { "LB_Swap", 0, 0 };
    op_stats_[(int)OpType::LB_SIMPLE] = { "LB_Simple", 0, 0 };
    op_stats_[(int)OpType::VND] = { "VND", 0, 0 };
}

void Island::ExportState(int generation, bool is_catastrophe) const {
    std::ofstream hist("history.csv", std::ios::app);
    if (hist.is_open()) {
        size_t unique_sols = local_cache_.GetSize();
        hist << generation << ","
            << current_best_.GetFitness() << ","
            << (is_catastrophe ? 1 : 0) << ","
            << current_cv_ << ","
            << adaptive_mutation_rate_ << ","
            << adaptive_vnd_prob_ << ","
            << adaptive_ruin_chance_ << ","
            << total_evaluations << ","
            << local_cache_hits << ","
            << unique_sols << std::endl;
    }
    std::ofstream traj("trajectory.csv", std::ios::app);
    if (traj.is_open()) {
        traj << generation << "," << current_best_.GetFitness();
        const auto& raw_genes = current_best_.GetGenotype();
        auto canonical_genes = CanonicalizeGenotype(raw_genes, evaluator_->GetNumGroups());
        for (int gene : canonical_genes) traj << "," << gene;
        traj << std::endl;
    }
    std::ofstream stats_file("mutation_stats.txt", std::ios::trunc);
    if (stats_file.is_open()) {
        stats_file << "Type,Name,Calls,Wins,Rate\n";
        for (const auto& stat : op_stats_) {
            double rate = (stat.calls > 0) ? ((double)stat.wins / stat.calls * 100.0) : 0.0;
            stats_file << "Mut," << stat.name << ","
                << stat.calls << "," << stat.wins << ","
                << std::fixed << std::setprecision(2) << rate << "\n";
        }
    }
}
#endif

int Island::SelectParentIndex() {
    if (population_.empty()) return -1;
    std::uniform_int_distribution<int> dist(0, population_size_ - 1);
    int best_idx = -1;
    double best_val = std::numeric_limits<double>::max();
    int t_size = std::min(Config::TOURNAMENT_SIZE, (int)population_.size());
    for (int i = 0; i < t_size; ++i) {
        int idx = dist(rng_);
        if (population_[idx].GetBiasedFitness() < best_val) {
            best_val = population_[idx].GetBiasedFitness();
            best_idx = idx;
        }
    }
    return best_idx;
}

int Island::GetWorstBiasedIndex() const {
    if (population_.empty()) return -1;
    int worst = -1;
    double max_val = -1.0;
    for (int i = 0; i < population_size_; ++i) {
        if (population_[i].GetGenotype() == current_best_.GetGenotype()) continue;
        if (population_[i].GetBiasedFitness() > max_val) {
            max_val = population_[i].GetBiasedFitness();
            worst = i;
        }
    }
    return (worst == -1) ? GetWorstIndex() : worst;
}

int Island::GetWorstIndex() const {
    if (population_.empty()) return -1;
    int idx = 0;
    double worst = -1.0;
    for (int i = 0; i < population_size_; ++i) {
        if (population_[i].GetFitness() > worst) {
            worst = population_[i].GetFitness();
            idx = i;
        }
    }
    return idx;
}

void Island::InjectImmigrant(Individual& imigrant) {
    double fit = SafeEvaluate(imigrant);
    if (fit != std::numeric_limits<double>::max()) {
        imigrant.SetFitness(fit);
        std::lock_guard<std::mutex> lock(population_mutex_);
        int worst = GetWorstBiasedIndex();
        if (worst >= 0 && worst < (int)population_.size()) {
            if (fit < population_[worst].GetFitness()) {
                population_[worst] = imigrant;
                if (fit < current_best_.GetFitness()) {
                    current_best_ = imigrant;
                }
                UpdateBiasedFitness();
            }
        }
    }
}