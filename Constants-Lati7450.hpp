#pragma once

namespace LcVRPContest {
    namespace Config {

        constexpr int ISLAND_POPULATION_SIZE = 25;
        constexpr int TOURNAMENT_SIZE = 2;
        constexpr int HISTORY_LAMBDA = 50;
        constexpr int NUM_NEIGHBORS = 20;
        constexpr long long MAX_TIME_SECONDS = 5*60;


        constexpr int MIGRATION_INTERVAL = 2500;
        const double LOADBALANCECHANCE = 0.75;
        const double MICROSPLITCHANCE = 0.5;

        const int  DECOMPOSEDVNDTRIES = 5;
        const int ELITERATIO = ISLAND_POPULATION_SIZE / 4;

        const bool split = false;
        const double VNDSWAP = 0;
        const double CROSSOVERSEQ_PROBABILITY = 0.1;

        constexpr int CORDSITERATIONS = 1000000;
    }
}