#include "Optimizer.hpp"
#include "Constants.hpp"
#include "ProblemData.hpp"
#include <iostream>
#include <limits>
#include <iomanip>


#include <fstream>
#include <string>

using namespace LcVRPContest;
using namespace std;



Optimizer::Optimizer(Evaluator& evaluator)
    : evaluator_(evaluator),
    rng_(random_device{}()),
    current_best_fitness_(numeric_limits<double>::max()),
    is_running_(false),
    total_generations_(0)
{
    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 4;
    num_islands_ = (hw_threads > 2) ? hw_threads - 1 : hw_threads;
    if (num_islands_ > 16) num_islands_ = 16;
    num_islands_ = 6;

    const ProblemData& data = evaluator_.GetProblemData();

    fast_evaluators_.reserve(num_islands_);
    for (int i = 0; i < num_islands_; ++i) {
        fast_evaluators_.push_back(new ThreadSafeEvaluator(data, evaluator.GetNumGroups()));
    }


    islands_.reserve(num_islands_);
    for (int i = 0; i < num_islands_; ++i) {
        islands_.push_back(new Island(fast_evaluators_[i], data, Config::ISLAND_POPULATION_SIZE, i));
    }
}

Optimizer::~Optimizer() {
    StopThreads();
    for (auto* island : islands_) delete island;
    islands_.clear();
    for (auto* fe : fast_evaluators_) delete fe;
    fast_evaluators_.clear();
}

void Optimizer::StopThreads() {
    is_running_ = false;
    for (auto& t : worker_threads_) {
        if (t.joinable()) t.join();
    }
    worker_threads_.clear();
}

void Optimizer::Initialize() {


    StopThreads();



    for (int i = 0; i < num_islands_; ++i) {
        Island* my_island = islands_[i];

        switch (i % 6) {
        case 0: my_island->Initialize(INITIALIZATION_TYPE::RANDOM); break;
        case 1: my_island->Initialize(INITIALIZATION_TYPE::RR); break;
        case 2: my_island->Initialize(INITIALIZATION_TYPE::CHUNKED); break;
        case 3: my_island->Initialize(INITIALIZATION_TYPE::CHUNKED); break;
        case 4: my_island->Initialize(INITIALIZATION_TYPE::RR); break;
        default: my_island->Initialize(INITIALIZATION_TYPE::RANDOM); break;
        }

        if (my_island->GetBestFitness() < current_best_fitness_) {
            current_best_ = my_island->GetBestSolution();
            current_best_fitness_ = my_island->GetBestFitness();
        }
    }

    is_running_ = true;
    total_generations_ = 0;
    iterations_migration_ = 0;

    for (int i = 0; i < num_islands_; ++i) {
        worker_threads_.emplace_back(&Optimizer::WorkerLoop, this, i);
    }
}

void Optimizer::WorkerLoop(int island_idx) {
    Island* my_island = islands_[island_idx];
    int last_catastrophes = 0;

    while (is_running_) {
        my_island->RunGeneration();

#ifdef RESEARCH
        if (island_idx == 0) {
            bool is_catastrophe = (my_island->catastrophy_activations > last_catastrophes);
            last_catastrophes = my_island->catastrophy_activations;
            if (is_catastrophe || total_generations_ % 30 == 0) {
                my_island->ExportState(total_generations_, is_catastrophe);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
#endif


        if (my_island->GetBestFitness() < current_best_fitness_) {
            std::lock_guard<std::mutex> lock(global_mutex_);
            if (my_island->GetBestFitness() < current_best_fitness_) {
                current_best_ = my_island->GetBestSolution();
                current_best_fitness_ = my_island->GetBestFitness();
                current_best_indiv_ = my_island->GetBestIndividual();
                my_island->PrintIndividual(my_island->GetBestIndividual(), total_generations_);
            }
        }
        total_generations_++;
    }
}


void Optimizer::PrintIslandStats() {
#ifdef RESEARCH
    for (int i = 0; i < num_islands_; ++i) {
        Island* my_island = islands_[i];
        cout << "\n Island " << my_island->GetId() << ": " << "VNDs: " << my_island->vnd_activations_ << " | Catast: " << my_island->catastrophy_activations << " | crossovers: " << my_island->crossovers << " | LBmutations: " << my_island->load_balancing_activations
            << " | Spatialmutations: " << my_island->spatial_activations << " | Aggmutations: " << my_island->aggresive_mutation_activations << " | hist: " << my_island->hist << " | total comps "<< my_island->total_comps << " | passes: " << my_island->passes <<" | ratiuo: "<<(double)my_island->passes/my_island->total_comps << "\n\n";

    }
#endif
}

void Optimizer::RunIteration() {
    long long current_gens = total_generations_; // Kopia dla spójnoœci

    if (current_gens - last_migration_gen_ > Config::MIGRATION_INTERVAL) {

        last_migration_gen_ = current_gens; // Zapisz kiedy robiliœmy migracjê

        // Logika migracji
        for (int i = 0; i < num_islands_; ++i) {
            int from = i;
            int to = (i + 1) % num_islands_;

            // Pobieramy kopiê, ¿eby nie blokowaæ mutexów na d³ugo
            Individual immigrant = islands_[from]->GetBestIndividual();

            // Wstrzykujemy
            islands_[to]->InjectImmigrant(immigrant);
        }

        // Opcjonalnie: wyœwietl log raz na migracjê, a nie 6 razy
        // cout << "[Optimizer] Migration performed at gen: " << current_gens << endl;
    }
   //  std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

int Optimizer::GetGeneration() {
    return total_generations_;
}

