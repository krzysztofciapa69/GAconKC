#include "Evaluator.hpp"
#include "Optimizer.hpp"
#include "ProblemLoader.hpp"
#include "Constants.hpp"
#include <iostream>
#include <chrono>
#include <conio.h>
#include <fstream>

#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif


using namespace LcVRPContest;
using namespace std;
bool FileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void LaunchDashboard() {
#ifdef RESEARCH

    std::string script_name = "live_runner.py";


    if (!FileExists(script_name)) {
        return;
    }

    int result = 0;
#ifdef _WIN32
    result = system("start py live_runner.py --passive");
    if (result != 0) result = system("start python live_runner.py --passive");
#else
    result = system("python3 live_runner.py --passive &");
#endif
    for (int i = 0; i < 10; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        std::cout << (10 - i) << " " << std::flush;
    }

#endif
}


void StartOptimization(const string& folder_name, const string& instance_name, int max_iterations, bool use_random_permutation) {
	ProblemLoader problem_loader(folder_name, instance_name, use_random_permutation);
	ProblemData problem_data = problem_loader.LoadProblem();

	int num_groups = problem_data.GetNumGroups();
	Evaluator evaluator(problem_data, num_groups);
	Optimizer optimizer(evaluator);

	optimizer.Initialize();

	for (int i = 0; i < max_iterations; ++i) {
        if (_kbhit()) {             // 1. SprawdŸ, czy coœ wciœniêto (b³yskawiczne)
            char ch = _getch();     // 2. Pobierz ten znak
            if (ch == ' ') {        // 3. Jeœli to spacja...
                cout << "\n[!] Przerwano recznie spacja." << endl;
                break;              // Wyjdz z petli, kod dalej wykona zapis wynikow
            }
        }


		optimizer.RunIteration();
	}

	vector<int>* best_solution = optimizer.GetCurrentBest();
	double best_fitness = evaluator.Evaluate(*best_solution);
	cout << "final best fitness: " << best_fitness << endl;
}

void StartOptimization(const string& folder_name, const string& instance_name, bool use_random_permutation) {

#ifdef RESEARCH
    {
        std::ofstream meta("active_instance.txt");
        meta << instance_name;

        std::ofstream("history.csv", std::ios::trunc) << "Gen,Fit,Cat,Div,Loc,Comp,Unq,Tot,Hits\n";
        std::ofstream("live_view.txt", std::ios::trunc);
        std::ofstream("trajectory.csv", std::ios::trunc);

        std::ofstream("mutation_stats.txt", std::ios::trunc);
    }
#endif


    cout << "instance:   " << instance_name << endl;
    ProblemLoader problem_loader(folder_name, instance_name, use_random_permutation);
    ProblemData problem_data = problem_loader.LoadProblem();

    std::vector<int> current_perm = problem_data.GetPermutation();
    std::random_device rd;
    std::mt19937 g(rd());


    int num_groups = problem_data.GetNumGroups();
    Evaluator evaluator(problem_data, num_groups);

    Optimizer optimizer(evaluator);
    ThreadSafeEvaluator tsevaluator(problem_data, num_groups);
    //CompareSplits(tsevaluator, 100);
    LaunchDashboard();
    optimizer.Initialize();




    auto start_time = std::chrono::high_resolution_clock::now();
    auto duration_limit = std::chrono::seconds(Config::MAX_TIME_SECONDS);
    cout << problem_data.GetPermutation().size() << " customers" << endl;
    while (std::chrono::high_resolution_clock::now() - start_time < duration_limit) {
        if (_kbhit()) {             // 1. SprawdŸ, czy coœ wciœniêto (b³yskawiczne)
            char ch = _getch();     // 2. Pobierz ten znak
            if (ch == ' ') {        // 3. Jeœli to spacja...
                cout << "\n[!] Przerwano recznie spacja." << endl;
                break;              // Wyjdz z petli, kod dalej wykona zapis wynikow
            }
        }


        optimizer.RunIteration();

    }


    vector<int>* best_solution = optimizer.GetCurrentBest();
    double best_fitness = evaluator.Evaluate(*best_solution);

    Individual best_indiv = optimizer.GetBestIndividual();
    cout << "FINAL RESULT | Fitness: " << fixed << best_fitness << " | Generations: " << optimizer.GetGeneration() << "| test time [s]: " << Config::MAX_TIME_SECONDS << endl;
    best_indiv.PrintGroups(problem_data.GetPermutation());
#ifdef RESEARCH
    optimizer.PrintIslandStats();
#endif
    cout << "Solution: ";
    vector<int> genotype = best_indiv.AccessGenotype();
    for (int gene : genotype) {
        cout << gene << ",";



    }

} 
int main() {

	bool use_random_permutation = true;


	std::cout << "ul";
	StartOptimization("Vrp-Set-D", "ORTEC-n323-k21", use_random_permutation);
	
	return 0;
}