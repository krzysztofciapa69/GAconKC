#pragma once
#include "Individual.hpp"
#include "ProblemGeometry.hpp"
#include "Split.hpp"
#include "ThreadSafeEvaluator.hpp"
#include <algorithm>
#include <random>
#include <vector>


namespace LcVRPContest {

class Mutator {
public:
  Mutator();

  // Inicjalizacja (wołana w konstruktorze Island)
  void Initialize(ThreadSafeEvaluator *eval, const ProblemGeometry *geo,
                  Split *split);

  // --- GŁÓWNE METODY ---

  // Inteligentne przesunięcie do sąsiada (Smart Spatial Move)
  bool ApplySmartSpatialMove(Individual &indiv, std::mt19937 &rng);

  // Agresywna mutacja (Shuffle / Random assign)
  bool AggressiveMutate(Individual &indiv, std::mt19937 &rng);

  // MicroSplit jako mutacja (wymaga wskaźnika do Split)
  // stagnation_factor: 0.0 (brak stagnacji) do 1.0 (pełna stagnacja) - wpływa
  // na wielkość okna
  bool ApplyMicroSplitMutation(Individual &indiv, double stagnation_factor,
                               int level, std::mt19937 &rng);

  // Prosta mutacja (Swap / Random Move)
  bool ApplySimpleMutation(Individual &indiv, std::mt19937 &rng);

  // Mutator.hpp
  // intensity: 0.0 (lekka mutacja) do 1.0 (ciężka mutacja)
  // is_exploitation: true = smaller windows (10-25%), false = larger (30-70%)
  bool ApplyRuinRecreate(Individual &indiv, double intensity,
                         bool is_exploitation, std::mt19937 &rng);

  // Return Minimizer - finds clients causing capacity overflows and relocates
  // them
  bool ApplyReturnMinimizer(Individual &indiv, std::mt19937 &rng);

  // Merge-Split - combines two groups into one, then re-splits optimally
  // Powerful operator for escaping local optima by restructuring route
  // boundaries
  bool ApplyMergeSplit(Individual &indiv, std::mt19937 &rng);

private:
  // Helper buffers for MergeSplit
  std::vector<int> merge_route_buffer_;
  ThreadSafeEvaluator *evaluator_;
  const ProblemGeometry *geometry_;
  Split *split_ptr_;

  // --- BUFORY PAMIĘCI (Unikanie alokacji w pętli - Cache Friendly) ---
  struct GroupInfo {
    double sum_x = 0;
    double sum_y = 0;
    int count = 0;
  };

  std::vector<int> removed_indices_buffer_;
  std::vector<bool> is_removed_buffer_;
  std::vector<GroupInfo> group_centroids_buffer_;

  // Pomocnicze bufory dla SmartMove
  std::vector<int> candidates_buffer_;
};

} // namespace LcVRPContest
