#pragma once
#include "ThreadSafeEvaluator.hpp"
#include "ProblemGeometry.hpp"
#include "Individual.hpp"
#include "Constants.hpp"
#include <vector>
#include <random>

namespace LcVRPContest {

    class LocalSearch {
    public:

        LocalSearch(ThreadSafeEvaluator* evaluator, const ProblemGeometry* geometry, int id);

        bool RunVND(Individual& ind);


    private:

        ThreadSafeEvaluator* evaluator_;
        const ProblemGeometry* geometry_;

        int id_;
        std::mt19937 rng_;


        std::vector<std::vector<int>> vnd_routes_;
        std::vector<double> vnd_loads_;

        std::vector<int> customer_ranks_;


        std::vector<int> client_indices_;
        std::vector<int> candidate_groups_;

        bool RunFullVND(Individual& ind, bool allow_swap);
        bool RunDecomposedVND(Individual& ind, bool allow_swap);

   
        bool OptimizeActiveSet(Individual& ind, int max_iter, bool allow_swap);


        void InitializeRanks();
        int FindInsertionIndexBinary(const std::vector<int>& route, int target_rank) const;
        double CalculateRemovalDelta(const std::vector<int>& route, int client_id) const;
        double CalculateInsertionDelta(const std::vector<int>& route, int client_id) const;
    };
}