#pragma once
#include "Evaluator.hpp"
#include "Island.hpp"
#include "ThreadSafeEvaluator.hpp"

#include <vector>
#include <thread>
#include <mutex>
#include <atomic> 
#include <random>

namespace LcVRPContest {
    class Optimizer {
    public:
        Optimizer(Evaluator& evaluator);
        ~Optimizer();

        void Initialize();
        void RunIteration();
        int GetGeneration();
        Individual GetBestIndividual() { return current_best_indiv_; }
        std::vector<int>* GetCurrentBest() { return &current_best_; }
        double GetCurrentBestFitness() const { return current_best_fitness_; }
        void PrintIslandStats();

    private:
        void WorkerLoop(int island_idx);
        void StopThreads();

        Evaluator& evaluator_;


        std::vector<ThreadSafeEvaluator*> fast_evaluators_;
        std::vector<Island*> islands_;


        std::vector<std::thread> worker_threads_;
        std::atomic<bool> is_running_;
        std::atomic<long long> total_generations_;
        std::atomic<int> iterations_migration_ = 0;
        long long last_migration_gen_ = 0;

        std::vector<int> current_best_;
        double current_best_fitness_;
        Individual current_best_indiv_;
        int num_islands_;
        std::mt19937 rng_;
        std::mutex global_mutex_;
    };
}