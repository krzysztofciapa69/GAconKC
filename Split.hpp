#pragma once
#include <vector>
#include <deque>
#include "ThreadSafeEvaluator.hpp"
#include "Individual.hpp"


namespace LcVRPContest {
    class ProblemGeometry;
}

namespace LcVRPContest {

    struct SplitResult {
        double total_cost;
        std::vector<std::vector<int>> optimized_routes;
        std::vector<int> group_assignment;
        bool feasible;
    };

    class Split {
    public:
        explicit Split(const ThreadSafeEvaluator* evaluator);

        SplitResult RunLinear(const std::vector<int>& giant_tour);
        SplitResult RunBellman(const std::vector<int>& giant_tour);

        void ApplyMicroSplit(Individual& indiv, int start_idx, int end_idx, const ProblemGeometry* geometry);

    private:
        const ThreadSafeEvaluator* evaluator_;
        int capacity_;
        double max_distance_;
        bool has_distance_constraint_;
        int depot_idx_;
        int num_customers_;

        std::vector<double> D_; // cumulated distance
        std::vector<int> Q_;    // cumulated demand
        std::vector<double> V_; // cost of the shortest route 
        std::vector<int> pred_; //predecesors

        void PrecomputeStructures(const std::vector<int>& giant_tour);
        SplitResult ReconstructResult(const std::vector<int>& giant_tour);
    };
}