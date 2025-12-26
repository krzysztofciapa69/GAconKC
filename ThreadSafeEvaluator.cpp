#include "ThreadSafeEvaluator.hpp"
#include <iostream>
#include <algorithm>
#include <limits>
#include "Individual.hpp"

using namespace std;

namespace LcVRPContest {

    static const double WRONG_VAL = 1e9;

    ThreadSafeEvaluator::ThreadSafeEvaluator(const ProblemData& data, int num_groups)
        : num_groups_(num_groups),
        num_customers_(data.GetNumCustomers()),
        dimension_(data.GetDimension()),
        capacity_(data.GetCapacity()),
        depot_index_(data.GetDepot() - 1),
        has_distance_constraint_(data.HasDistanceConstraint()),
        max_distance_(data.GetDistance()),
        demands_(data.GetDemands()),
        problem_data_(&data),
        permutation_(data.GetPermutation())
    {
        coordinates_ = data.GetCoordinates();

        if (dimension_ <= MATRIX_THRESHOLD) {
            use_matrix_ = true;

            const auto& edge_weights = data.GetEdgeWeights();

            fast_distance_matrix_.resize(dimension_ * dimension_, WRONG_VAL);

            for (int i = 0; i < dimension_; ++i) {
                for (int j = 0; j < dimension_; ++j) {
                    double dist = WRONG_VAL;

                    if (data.GetEdgeWeightType() == "EXPLICIT") {
                        if (i < (int)edge_weights.size() && j < (int)edge_weights[i].size()) {
                            dist = edge_weights[i][j];
                        }
                    }
                    else {
                        if (i == j) dist = 0.0;
                        else {
                            double dx = coordinates_[i].x - coordinates_[j].x;
                            double dy = coordinates_[i].y - coordinates_[j].y;
                            dist = std::sqrt(dx * dx + dy * dy);
                        }
                    }
                    fast_distance_matrix_[i * dimension_ + j] = dist;
                }
            }
        }
        else {
            use_matrix_ = false;

        }
    }

    double ThreadSafeEvaluator::Evaluate(const std::vector<int>& solution) const {
        if ((int)solution.size() != num_customers_) {
            return WRONG_VAL;
        }
        return EvaluateWithStats(solution).fitness;
    }

    EvaluationResult ThreadSafeEvaluator::EvaluateWithStats(const std::vector<int>& solution) const {
        if (solution.empty()) return { WRONG_VAL, 0 };

        double total_dist = 0.0;
        int returns = 0;

        std::vector<int> group_load(num_groups_, 0);
        std::vector<double> group_dist(num_groups_, 0.0);
        std::vector<int> group_last(num_groups_, depot_index_);

        const int* sol_ptr = solution.data();
        int sol_size = (int)solution.size();

        for (int customer_id : permutation_) {
            if (customer_id == (depot_index_ + 1)) continue;

            int matrix_idx = customer_id - 1;
            if (matrix_idx < 0 || matrix_idx >= dimension_) continue;

            int gene_idx = customer_id - 2;
            if (gene_idx < 0 || gene_idx >= sol_size) continue;

            int g = sol_ptr[gene_idx];
            if (g < 0 || g >= num_groups_) continue;

            int demand = demands_[matrix_idx];


            if (group_load[g] + demand > capacity_) {
                returns++;
                total_dist += GetDist(group_last[g], depot_index_);
                group_load[g] = 0;
                group_dist[g] = 0.0;
                group_last[g] = depot_index_;
            }

            double d_travel = GetDist(group_last[g], matrix_idx);

            if (has_distance_constraint_) {
                double d_return = GetDist(matrix_idx, depot_index_);
                if (group_dist[g] + d_travel + d_return > max_distance_) {
                    if (group_last[g] != depot_index_) {
                        returns++;
                        total_dist += GetDist(group_last[g], depot_index_);
                        group_load[g] = 0;
                        group_dist[g] = 0.0;
                        group_last[g] = depot_index_;
                    }
                    d_travel = GetDist(depot_index_, matrix_idx);
                }
            }

            total_dist += d_travel;
            group_dist[g] += d_travel;
            group_load[g] += demand;
            group_last[g] = matrix_idx;
        }

        for (int g = 0; g < num_groups_; ++g) {
            if (group_last[g] != depot_index_) {
                total_dist += GetDist(group_last[g], depot_index_);
            }
        }

        return { total_dist, returns };
    }

    int ThreadSafeEvaluator::GetTotalDepotReturns(const std::vector<int>& solution) const {
        return EvaluateWithStats(solution).returns;
    }

}