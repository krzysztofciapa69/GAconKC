#pragma once
#include "ProblemData.hpp"
#include "ThreadSafeEvaluator.hpp"
#include "Constants.hpp"
#include <vector>
#include <random>

namespace LcVRPContest {

    class ProblemGeometry {
    public:
        ProblemGeometry(const ProblemData& data, int id);


        void Initialize(ThreadSafeEvaluator* evaluator);


        inline const Coordinate& GetCoordinate(int index) const {
            if (index < 0 || index >= (int)coordinates_.size()) return coordinates_[0];
            return coordinates_[index];
        }

        inline const std::vector<int>& GetNeighbors(int index) const {
            return neighbors_[index];
        }

        bool HasCoordinates() const { return !coordinates_.empty(); }

    private:
        int id_;
        std::mt19937 rng_; 

        std::vector<Coordinate> coordinates_;
        std::vector<Coordinate> artificial_coordinates_;
        std::vector<std::vector<int>> neighbors_;

        bool HasValidCoordinates(const std::vector<Coordinate>& coords) const;
        void GenerateArtificialCoordinates(ThreadSafeEvaluator* evaluator);
        double CalculateStress(ThreadSafeEvaluator* evaluator, const std::vector<Coordinate>& test_coords);
        void PrecomputeNeighbors(ThreadSafeEvaluator* evaluator);
    };
}