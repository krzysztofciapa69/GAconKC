#pragma once
#include "ThreadSafeEvaluator.hpp"
#include "ProblemGeometry.hpp"
#include "Individual.hpp"
#include "Constants.hpp"
#include <vector>
#include <random>

namespace LcVRPContest {

    class LocalSearch {
    public:
        LocalSearch(ThreadSafeEvaluator* evaluator, const ProblemGeometry* geometry, int id);

        // G³ówna metoda uruchamiaj¹ca (decyduje czy Full czy Decomposed)
        bool RunVND(Individual& ind);

    private:
        ThreadSafeEvaluator* evaluator_;
        const ProblemGeometry* geometry_;
        int id_;
        std::mt19937 rng_;

        // --- Struktury danych (robocze) ---
        // Przechowuj¹ trasy w formacie wygodnym do szybkich zmian
        std::vector<std::vector<int>> vnd_routes_;
        std::vector<double> vnd_loads_;            // Obci¹¿enie ka¿dej grupy (Capacity constraint)

        std::vector<int> customer_ranks_;          // Cache pozycji w globalnej permutacji
        std::vector<int> client_indices_;          // Lista klientów do optymalizacji (Active Set)
        std::vector<int> candidate_groups_;        // Bufor dla s¹siednich grup

        // --- Metody inicjalizacyjne ---
        void InitializeRanks();

        // --- Strategie VND ---
        bool RunFullVND(Individual& ind, bool allow_swap);
        bool RunDecomposedVND(Individual& ind, bool allow_swap);

        // G³ówna pêtla optymalizacyjna dla zadanego zbioru klientów (client_indices_)
        bool OptimizeActiveSet(Individual& ind, int max_iter, bool allow_swap);

        // --- Metody pomocnicze (Helpers) ---

        // Znajduje indeks wstawienia binarnie (trasy s¹ zawsze posortowane wg rang)
        int FindInsertionIndexBinary(const std::vector<int>& route, int target_rank) const;

        // Oblicza zmianê kosztu (Delta) przy usuniêciu klienta z trasy
        double CalculateRemovalDelta(const std::vector<int>& route, int client_id) const;

        // Oblicza zmianê kosztu (Delta) przy wstawieniu klienta do trasy
        double CalculateInsertionDelta(const std::vector<int>& route, int client_id) const;
    };
}