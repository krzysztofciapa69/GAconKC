#pragma once
#include "Evaluator.hpp"
#include "Individual.hpp"
#include "ThreadSafeEvaluator.hpp"
#include "ProblemData.hpp"
#include "Split.hpp"
#include "LocalCache.hpp"
#include "Constants.hpp"

// Nowe klasy
#include "ProblemGeometry.hpp"
#include "LocalSearch.hpp"

#include <vector>
#include <random>
#include <mutex>
#include <atomic>

namespace LcVRPContest {

    enum class INITIALIZATION_TYPE { RANDOM, CHUNKED, RR };

    class Island {
    public:
        Island(ThreadSafeEvaluator* evaluator, const ProblemData& data, int population_size, int id);

        void Initialize(INITIALIZATION_TYPE strategy);
        void RunGeneration();
        void InjectImmigrant(Individual& imigrant);
        double EvaluateWithHistoryPenalty(const std::vector<int>& genotype);

        const std::vector<int>& GetBestSolution() const { return current_best_.GetGenotype(); }
        const Individual& GetBestIndividual() const { return current_best_; }
        double GetBestFitness() const { return current_best_.GetFitness(); }
        void PrintIndividual(const Individual& individual, int global_generation) const;

        int GetId() const { return id_; }
        double getCurrentCV() const { return current_cv_; }

        long long total_comps = 0;
        long long passes = 0;

#ifdef RESEARCH
        void ExportState(int generation, bool is_catastrophe) const;
        std::atomic<long long> vnd_activations_{ 0 };
        std::atomic<long long> catastrophy_activations{ 0 };
        std::atomic<long long> load_balancing_activations{ 0 };
        std::atomic<long long> aggresive_mutation_activations{ 0 };
        std::atomic<long long> crossovers{ 0 };
        std::atomic<long long> spatial_activations{ 0 };
        std::atomic<long long> total_evaluations{ 0 };
        std::atomic<long long> local_cache_hits{ 0 };
#endif

    private:
        ThreadSafeEvaluator* evaluator_;
        const std::vector<int>& demands_;
        int capacity_;

        // Kompozycja - nowe obiekty
        ProblemGeometry geometry_;
        LocalSearch local_search_;

#ifdef RESEARCH
        enum class OpType {
            CROSSOVER, MUT_AGGRESSIVE, MUT_SPATIAL, MUT_SIMPLE,
            LB_CHAIN, LB_SWAP, LB_SIMPLE, VND, COUNT
        };
        struct OpStat {
            std::string name;
            long long calls = 0;
            long long wins = 0;
        };
        std::vector<OpStat> op_stats_;
        void InitStats();
        std::vector<int> CanonicalizeGenotype(const std::vector<int>& genotype, int num_groups) const;
#endif


        std::vector<int> customer_ranks_;
        int population_size_;
        int id_;

        LocalCache local_cache_;
        std::vector<Individual> population_;
        Individual current_best_;
        std::mt19937 rng_;
        std::mutex population_mutex_;
        int stagnation_count_ = 0;

        Split split_;

        double current_cv_ = 0.0;
        double adaptive_mutation_rate_ = 0.0;
        double adaptive_vnd_prob_ = 0.0;
        double adaptive_ruin_chance_ = 0.0;

        long long current_generation_ = 0;
        long long last_improvement_gen_ = 0;
        long long last_catastrophy_gen_ = 0;

        const int BASE_STAGNATION_LIMIT = 5000;
        double current_structural_diversity_ = 0.0;

        std::vector<Individual> offspring_pool_;


        std::vector<int> pred1;
        std::vector<int> pred2;
        std::vector<int> last_in_group1;
        std::vector<int> last_in_group2;


        struct GroupInfo { double sum_x = 0; double sum_y = 0; int count = 0; };
        std::vector<GroupInfo> group_centroids_buffer_;
        std::vector<bool> is_removed_buffer_;
        std::vector<int> removed_indices_buffer_;


        void CalculatePopulationCV();
        void UpdateAdaptiveParameters();
        double MapRange(double value, double in_min, double in_max, double out_min, double out_max);

        int ApplyLoadBalancing(Individual& child);
        int ApplyMutation(Individual& child);
        int ApplyMicroSplitMutation(Individual& child);

        double SafeEvaluate(Individual& indiv);
        double SafeEvaluate(const std::vector<int>& genotype);

        void InitIndividual(Individual& indiv, INITIALIZATION_TYPE strategy);

        SplitResult RunSplit(const std::vector<int>& permutation);

        void UpdateBiasedFitness();
        void Catastrophy();
        int CalculateBrokenPairsDistance(const Individual& ind1, const Individual& ind2, const std::vector<int>& permutation, int num_groups);

        int SelectParentIndex();
        int GetWorstBiasedIndex() const;
        int GetWorstIndex() const;

        Individual Crossover(const Individual& p1, const Individual& p2);
        Individual CrossoverSequence(const Individual& p1, const Individual& p2);
        Individual CrossoverSpatial(const Individual& p1, const Individual& p2);

        bool Mutate(Individual& indiv);
        bool AggresiveMutate(Individual& indiv);
        bool SpatialRuinAndRecreate(Individual& indiv);
        bool ApplySmartSpatialMove(Individual& indiv);

        bool ApplyLoadBalancingSwapMutation(Individual& individual);
        bool ApplyLoadBalancingChainMutation(Individual& individual);
        bool ApplyLoadBalancingMutation(Individual& individual);

        void ApplySplitToIndividual(Individual& indiv);
    };
}