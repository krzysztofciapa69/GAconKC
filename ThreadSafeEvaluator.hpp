#pragma once
#include "ProblemData.hpp"
#include <vector>
#include <cmath>
#include <limits>

namespace LcVRPContest {

    struct EvaluationResult {
        double fitness;
        int returns;
    };

    class ThreadSafeEvaluator {
    public:

        static const int MATRIX_THRESHOLD = 5000;

        ThreadSafeEvaluator(const ProblemData& data, int num_groups);

        double Evaluate(const std::vector<int>& solution) const;
        EvaluationResult EvaluateWithStats(const std::vector<int>& solution) const;

        int GetSolutionSize() const { return num_customers_; }
        int GetLowerBound() const { return 0; }
        int GetUpperBound() const { return num_groups_ - 1; }
        const std::vector<int>& GetPermutation() const { return permutation_; }
        std::vector<int> GetDemands() const { return demands_; }

        int GetDimension() const { return dimension_; }
        int GetCapacity() const { return capacity_; }
        int GetNumGroups() const { return num_groups_; }

        inline double GetDist(int i, int j) const {
            if (i == j) return 0.0;

            if (use_matrix_) {
                return fast_distance_matrix_[i * dimension_ + j];
            }
            else {
                double dx = coordinates_[i].x - coordinates_[j].x;
                double dy = coordinates_[i].y - coordinates_[j].y;
                return std::sqrt(dx * dx + dy * dy);
            }
        }

        inline double GetDemand(const int c_id) const {
            if (c_id - 1 >= 0 && c_id - 1 < (int)demands_.size())
                return demands_[c_id - 1];
            return 0;
        }

        int GetTotalDepotReturns(const std::vector<int>& solution) const;
        const ProblemData& GetProblemData() const { return *problem_data_; }
    private:
        int num_groups_;
        int num_customers_;
        int dimension_;
        int capacity_;
        int depot_index_;
        bool has_distance_constraint_;
        double max_distance_;
        const ProblemData* problem_data_;

        bool use_matrix_;

        std::vector<int> demands_;
        std::vector<int> permutation_;

        std::vector<double> fast_distance_matrix_;
        std::vector<Coordinate> coordinates_;
    };
}