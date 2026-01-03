#include "ProblemGeometry.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <limits>
#include <chrono>

using namespace LcVRPContest;

ProblemGeometry::ProblemGeometry(const ProblemData& data, int id)
    : id_(id),
    coordinates_(data.GetCoordinates())
{
    rng_.seed(static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count() + id * 999));
}

void ProblemGeometry::Initialize(ThreadSafeEvaluator* evaluator) {
    bool coords_invalid = coordinates_.empty() || !HasValidCoordinates(coordinates_);

    if (coords_invalid) {
        GenerateArtificialCoordinates(evaluator);
        if (!artificial_coordinates_.empty()) {
            coordinates_ = artificial_coordinates_;
        }
    }

    PrecomputeNeighbors(evaluator);
}

void ProblemGeometry::PrecomputeNeighbors(ThreadSafeEvaluator* evaluator) {
    int n = evaluator->GetSolutionSize();
    neighbors_.assign(n, std::vector<int>());

    for (int i = 0; i < n; ++i) {
        int u_id = i + 2;
        if (u_id > evaluator->GetDimension()) continue;

        int u_node_idx = u_id - 1;

        std::vector<std::pair<double, int>> dists;
        dists.reserve(n);

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;

            int v_id = j + 2;
            int v_node_idx = v_id - 1;

            double d = evaluator->GetDist(u_node_idx, v_node_idx);
            dists.push_back({ d, j });
        }

        size_t keep = std::min((size_t)Config::NUM_NEIGHBORS, dists.size());

        std::nth_element(dists.begin(), dists.begin() + keep, dists.end());

        neighbors_[i].reserve(keep);
        for (size_t k = 0; k < keep; ++k) {
            neighbors_[i].push_back(dists[k].second);
        }
    }
}

bool ProblemGeometry::HasValidCoordinates(const std::vector<Coordinate>& coords) const {
    if (coords.empty()) return false;
    for (size_t i = 1; i < coords.size(); ++i) {
        if (std::abs(coords[i].x) > 0.0001 || std::abs(coords[i].y) > 0.0001) {
            return true;
        }
    }
    return false;
}

double ProblemGeometry::CalculateStress(ThreadSafeEvaluator* evaluator, const std::vector<Coordinate>& test_coords) {
    double stress = 0.0;
    int dim = evaluator->GetDimension();
    int samples = std::min(dim * 50, 10000);

    int valid_samples = 0;

    for (int k = 0; k < samples; ++k) {
        int i = rng_() % dim;
        int j = rng_() % dim;
        if (i == j) continue;

        double target_dist = evaluator->GetDist(i, j);
        if (target_dist < 1.0) target_dist = 1.0;

        double dx = test_coords[i].x - test_coords[j].x;
        double dy = test_coords[i].y - test_coords[j].y;
        double current_dist = std::sqrt(dx * dx + dy * dy);

        double err = (current_dist - target_dist);

        double term = (err * err) / target_dist;
        if (term > 1e10) term = 1e10;

        stress += term;
        valid_samples++;
    }

    if (valid_samples == 0) return 1e9;
    return stress / valid_samples;
}

void ProblemGeometry::GenerateArtificialCoordinates(ThreadSafeEvaluator* evaluator) {
    if (HasValidCoordinates(coordinates_)) return;

    int dim = evaluator->GetDimension();
    if (dim <= 0) return;

    artificial_coordinates_.resize(dim);
    std::uniform_real_distribution<double> safety_dist(0.0, 100.0);
    for (int i = 0; i < dim; ++i) artificial_coordinates_[i] = Coordinate(safety_dist(rng_), safety_dist(rng_));

    const int RESTARTS = 2;
    const int ITERATIONS = Config::CORDSITERATIONSMULTIPLIER * dim;  // Fix: było coordinates_.size()
    double learning_rate = 0.5;

    double best_stress = std::numeric_limits<double>::max();
    std::vector<Coordinate> best_coords = artificial_coordinates_;

    for (int attempt = 0; attempt < RESTARTS; ++attempt) {

        std::vector<Coordinate> current_coords(dim);
        double area_scale = 50.0 * std::sqrt((double)dim);
        std::uniform_real_distribution<double> dist_init(0.0, area_scale);

        for (int i = 0; i < dim; ++i) {
            current_coords[i] = Coordinate(dist_init(rng_), dist_init(rng_));
        }

        long long pairs_per_iter = std::min((long long)dim * 50, 100000LL);

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            double current_lr = learning_rate * std::exp(-3.0 * iter / ITERATIONS);

            for (int k = 0; k < pairs_per_iter; ++k) {
                int i = rng_() % dim;
                int j = rng_() % dim;
                if (i == j) continue;

                double target_dist = evaluator->GetDist(i, j);
                if (target_dist < 1.0) target_dist = 1.0;

                double dx = current_coords[i].x - current_coords[j].x;
                double dy = current_coords[i].y - current_coords[j].y;
                double current_dist = std::sqrt(dx * dx + dy * dy);

                if (current_dist < 1e-5) {
                    current_dist = 1e-5;
                    dx = 0.01;
                }

                double error = current_dist - target_dist;
                double weight = 1.0 / target_dist;
                double move = current_lr * weight * error;

                if (move > 10.0) move = 10.0;
                if (move < -10.0) move = -10.0;

                double move_x = (dx / current_dist) * move;
                double move_y = (dy / current_dist) * move;

                if (!std::isnan(move_x) && !std::isnan(move_y)) {
                    current_coords[j].x += move_x;
                    current_coords[j].y += move_y;
                    current_coords[i].x -= move_x;
                    current_coords[i].y -= move_y;
                }
            }
        }

        double stress = CalculateStress(evaluator, current_coords);

        if (!std::isnan(stress) && stress < best_stress) {
            best_stress = stress;
            best_coords = current_coords;
        }
    }

    artificial_coordinates_ = best_coords;
    std::cout << "[Island " << id_ << "] MDS Final. Best Stress: " << best_stress << "\n";
}