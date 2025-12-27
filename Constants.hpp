#pragma once

namespace LcVRPContest {
    namespace Config {

        constexpr int ISLAND_POPULATION_SIZE = 50;
        constexpr int TOURNAMENT_SIZE = 2;
        constexpr int HISTORY_LAMBDA = 100;
        constexpr int NUM_NEIGHBORS = 20;
        constexpr long long MAX_TIME_SECONDS = 20*60;


        constexpr int MIGRATION_INTERVAL = 1000;
        const double LOADBALANCECHANCE = 0.02;
        const double MICROSPLITCHANCE = 0.35;

        const int  DECOMPOSEDVNDTRIES = 5;
      //  const int ELITERATIO = std::max(2, ISLAND_POPULATION_SIZE / 10);
        const int ELITERATIO = 2;
        const bool split = true;
        const double VNDSWAP = 0.4;
        const double CROSSOVERSEQ_PROBABILITY = 0.2;

        constexpr int CORDSITERATIONSMULTIPLIER = 20;
    }
}