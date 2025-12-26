#pragma once
#include <vector>

namespace LcVRPContest {

    class Individual {
    public:
        Individual();
        Individual(int size);
        Individual(const std::vector<int>& genotype);

        const std::vector<int>& GetGenotype() const;
        double GetFitness() const;
        double GetDiversityScore() const;
        double GetBiasedFitness() const;
        bool IsEvaluated() const;

        std::vector<int>& AccessGenotype();
        void SetFitness(double fitness);
        void SetDiversityScore(double score);
        void SetBiasedFitness(double biased_fitness);

        bool operator<(const Individual& other) const;

        void SetReturnCount(int count);
        int GetReturnCount() const;

        void PrintGroups(const std::vector<int>& permutation) const;

        void Canonicalize();

    private:
        std::vector<int> genotype_;
        double fitness_;
        int return_count_ = 0;
        double diversity_score_;
        double biased_fitness_;
        bool is_evaluated_;
    };
}
