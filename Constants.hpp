#pragma once

namespace LcVRPContest {
namespace Config {

// === RING ISLAND CONFIG ===
constexpr int NUM_ISLANDS = 6;
constexpr double RING_MIGRATION_INTERVAL_SECONDS = 5.0;
constexpr double RING_MIGRATION_RATE = 0.10;  // 10% of population migrates

// === EXPLORATION CONFIG (Islands 0, 2, 4 - even) ===
// Goal: Maintain diversity, break local optima
constexpr int EXPLORATION_POPULATION_SIZE = 100;
constexpr int EXPLORATION_TOURNAMENT_SIZE = 2;
constexpr double EXPLORATION_MUTATION_PROB = 0.50;     // Aggressive
constexpr int EXPLORATION_VND_MIN =1;                  // Weak/none
constexpr int EXPLORATION_VND_MAX = 4;
constexpr double EXPLORATION_VND_PROB = 0.10;          // Rarely

// === EXPLOITATION CONFIG (Islands 1, 3, 5 - odd) ===
// Goal: Deep local search, refine solutions
constexpr int EXPLOITATION_POPULATION_SIZE = 40;
constexpr int EXPLOITATION_TOURNAMENT_SIZE = 6;
constexpr double EXPLOITATION_MUTATION_PROB = 0.05;    // Minimal
constexpr int EXPLOITATION_VND_MIN = 20;               // Deep
constexpr int EXPLOITATION_VND_MAX = 50;
constexpr double EXPLOITATION_VND_PROB = 0.95;         // Almost always

// Logging
constexpr double LOG_INTERVAL_SECONDS = 30.0;

// Stagnation detection (for inter-island migration)
constexpr int STAGNATION_THRESHOLD = 300;

// === SHARED CONFIG ===
constexpr int ISLAND_POPULATION_SIZE = 50; // Default fallback
constexpr int TOURNAMENT_SIZE = 4;
constexpr int HISTORY_LAMBDA = 100;
constexpr int NUM_NEIGHBORS = 20;
constexpr long long MAX_TIME_SECONDS = 30 * 60;

constexpr int MIGRATION_INTERVAL = 500;
constexpr int CROSS_POLLINATION_INTERVAL = 300;
constexpr int INTRA_ARCHIPELAGO_INTERVAL = 100;
const double LOADBALANCECHANCE = 0.10;
const double MICROSPLITCHANCE = 0.35;

const int DECOMPOSEDVNDTRIES = 5;
const int ELITERATIO = 5;
const bool split = true;
const double VNDSWAP = 0.4;
const double CROSSOVERSEQ_PROBABILITY = 0.3;

constexpr int CORDSITERATIONSMULTIPLIER = 20;

// VND operators for exploitation islands
constexpr bool ALLOW_SWAP = false;  // Disabled for speed (was ALLOW_SWAP_L2 = false)
constexpr bool ALLOW_3SWAP = true;
constexpr bool ALLOW_EJECTION = true;
constexpr bool ALLOW_LOAD_BALANCING = true;

// === ROUTE POOL CONFIG ===
constexpr size_t ROUTE_POOL_MAX_SIZE = 10000;
constexpr double GREEDY_ASSEMBLER_INTERVAL_SECONDS = 8.0;
constexpr int GREEDY_NUM_STARTS = 10;
constexpr size_t MIN_ROUTE_SIZE_FOR_POOL = 2;

constexpr double SPLIT_ROUTE_PENALTY = 1000000.0;
} // namespace Config
} // namespace LcVRPContest

