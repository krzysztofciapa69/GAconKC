#pragma once

namespace LcVRPContest {
namespace Config {

// === TIMING ===
constexpr long long MAX_TIME_SECONDS = 45 * 60;
constexpr double LOG_INTERVAL_SECONDS = 30.0;
constexpr int STAGNATION_THRESHOLD = 300;

// === LARGE INSTANCE SCALING ===
constexpr int LARGE_INSTANCE_THRESHOLD = 1500;
constexpr int HUGE_INSTANCE_THRESHOLD = 3000;
constexpr int EXPLOITATION_VND_MAX_LARGE = 15;   // was 70
constexpr int EXPLORATION_VND_MAX_LARGE = 2;     // was 4
constexpr int EXPLOITATION_POP_LARGE = 25;       // was 40
constexpr int EXPLORATION_POP_LARGE = 50;        // was 100
constexpr double EXPLOITATION_VND_PROB_LARGE = 0.20;  // was 0.95 - critical!
constexpr double EXPLORATION_VND_PROB_LARGE = 0.05;   // was 0.10

// === RING ISLAND CONFIG ===
constexpr int NUM_ISLANDS = 6;
constexpr double RING_MIGRATION_INTERVAL_SECONDS = 10.0; // was 5.0
constexpr double RING_MIGRATION_RATE =
    0.05; // 5% of population migrates (was 10%)

// === NON-NATIVE BROADCAST CONFIG ===
constexpr double BROADCAST_WARMUP_SECONDS = 30.0;  // was 60.0 - faster transfer of good solutions


// === EXPLORATION CONFIG (Islands 0, 2, 4 - even) ===
constexpr int EXPLORATION_POPULATION_SIZE = 100;
constexpr int EXPLORATION_TOURNAMENT_SIZE = 2;
constexpr double EXPLORATION_MUTATION_PROB = 0.50;
constexpr int EXPLORATION_VND_MIN = 1;
constexpr int EXPLORATION_VND_MAX = 4;
constexpr double EXPLORATION_VND_PROB = 0.40;  // was 0.10 - increased for better solutions
constexpr double EXPLORATION_VND_EXTRA_PROB =
    0.25; // extra VND chance for promising

// === PER-EXPLORATION ISLAND PARAMS (differentiation) ===
// I0: RuinRecreate specialist (aggressive R&R, small splits)
constexpr double EXPLORE_I0_MUT_MIN = 0.20;
constexpr double EXPLORE_I0_MUT_MAX = 0.60;
constexpr double EXPLORE_I0_RR_WEIGHT = 0.60; // 60% R&R vs 40% other
constexpr double EXPLORE_I0_SPLIT_WEIGHT = 0.30;
constexpr int EXPLORE_I0_SPLIT_LEVEL = 2; // small windows

// I2: Balanced explorer (even mix of operators)
constexpr double EXPLORE_I2_MUT_MIN = 0.30;
constexpr double EXPLORE_I2_MUT_MAX = 0.65;
constexpr double EXPLORE_I2_RR_WEIGHT = 0.40; // balanced
constexpr double EXPLORE_I2_SPLIT_WEIGHT = 0.50;
constexpr int EXPLORE_I2_SPLIT_LEVEL = 1; // medium windows

// I4: Split specialist (big windows, less R&R)
constexpr double EXPLORE_I4_MUT_MIN = 0.40;
constexpr double EXPLORE_I4_MUT_MAX = 0.70;
constexpr double EXPLORE_I4_RR_WEIGHT = 0.25;    // less R&R
constexpr double EXPLORE_I4_SPLIT_WEIGHT = 0.70; // heavy splits
constexpr int EXPLORE_I4_SPLIT_LEVEL = 0;        // large windows

// === EXPLOITATION CONFIG (Islands 1, 3, 5 - odd) ===
constexpr int EXPLOITATION_POPULATION_SIZE = 20;
constexpr int EXPLOITATION_TOURNAMENT_SIZE = 4;
constexpr double EXPLOITATION_MUTATION_PROB = 0.05;
constexpr int EXPLOITATION_VND_MIN = 30;
constexpr int EXPLOITATION_VND_MAX = 70;
constexpr double EXPLOITATION_VND_PROB = 0.98; // maxed for deep refinement

// === PER-EXPLOITATION ISLAND SPECIALIZATION ===
// I1: Ejection Chains specialist
constexpr double EXPLOIT_I1_OP_PR_PROB = 0.60;
constexpr double EXPLOIT_I1_OP_RR_PROB = 0.20;
constexpr double EXPLOIT_I1_OP_SPLIT_PROB = 0.20;

constexpr double EXPLOIT_I1_EJECTION_PROB = 0.50;  // 50% ejection (vs default 20%)
constexpr int EXPLOIT_I1_EJECTION_DEPTH = 4;       // deeper chains
constexpr double EXPLOIT_I1_SWAP3_PROB = 0.15;     // light swap
constexpr double EXPLOIT_I1_SWAP4_PROB = 0.05;

// I3: Path Relinking specialist  
constexpr double EXPLOIT_I3_OP_PR_PROB = 0.80;
constexpr double EXPLOIT_I3_OP_RR_PROB = 0.10;
constexpr double EXPLOIT_I3_OP_SPLIT_PROB = 0.10;

constexpr double EXPLOIT_I3_EJECTION_PROB = 0.10;  // minimal ejection
constexpr int EXPLOIT_I3_EJECTION_DEPTH = 2;
constexpr double EXPLOIT_I3_SWAP3_PROB = 0.15;
constexpr double EXPLOIT_I3_SWAP4_PROB = 0.05;

// I5: Deep Swap specialist (3-SWAP + 4-SWAP)
constexpr double EXPLOIT_I5_OP_PR_PROB = 0.60;
constexpr double EXPLOIT_I5_OP_RR_PROB = 0.20;
constexpr double EXPLOIT_I5_OP_SPLIT_PROB = 0.20;

constexpr double EXPLOIT_I5_EJECTION_PROB = 0.10;
constexpr int EXPLOIT_I5_EJECTION_DEPTH = 2;
constexpr double EXPLOIT_I5_SWAP3_PROB = 0.50;     // 50% 3-swap
constexpr double EXPLOIT_I5_SWAP4_PROB = 0.30;     // 30% 4-swap

// === SWAP PROBABILITIES ===
constexpr double EXPLOITATION_P_SWAP3 = 0.30;
constexpr double EXPLOITATION_P_SWAP4 = 0.20;  // was 0.05 - 4-swap more effective than random
constexpr double EXPLORATION_P_SWAP3 = 0.10;
constexpr double EXPLORATION_P_SWAP4 = 0.0;

// === EPSILON-GREEDY ADAPTIVE OPERATORS ===
constexpr double ADAPT_EPSILON = 0.25;  // 25% exploration of operators (was 0.10 implied)

// === ENDGAME CONFIG ===
constexpr double ENDGAME_THRESHOLD = 0.95; // % of MAX_TIME
constexpr double ENDGAME_P_SWAP3 = 0.50;
constexpr double ENDGAME_P_SWAP4 = 0.40;

// === MUTATION WEIGHTS ===
constexpr double MUT_AGGRESSIVE_THRESHOLD = 0.05;
constexpr double MUT_SPATIAL_THRESHOLD = 0.35;

// === ADAPTIVE MUTATION PARAMS ===
constexpr double ADAPTIVE_MUT_MIN = 0.05;
constexpr double ADAPTIVE_MUT_MAX = 0.80;
constexpr double ADAPTIVE_CHAOS_BOOST = 0.30;
constexpr double ADAPTIVE_CHAOS_PENALTY = 0.10;

// === CATASTROPHE CONFIG ===
constexpr int CATASTROPHE_STAGNATION_GENS = 5000;  // EXPLORE: less catastrophes = more refinement time
constexpr int EXPLOIT_CATASTROPHE_STAGNATION_GENS = 2000;  // EXPLOIT: faster catastrophes to escape local optima
constexpr int CATASTROPHE_MIN_GAP_GENS = 500;
constexpr double VND_EXHAUSTED_THRESHOLD = 3.0; // % - for EXPLORE
constexpr double EXPLOIT_VND_EXHAUSTED_THRESHOLD = 20.0;  // % - for EXPLOIT (higher = easier trigger)
constexpr int VND_EXHAUSTED_MIN_CALLS = 200;
constexpr int CATASTROPHE_CANDIDATES_MULTIPLIER = 10;
constexpr int CATASTROPHE_VND_ITERS = 30;

// === EXPLOIT ANTI-STAGNATION ===
constexpr double EXPLOIT_HEAVY_RR_INTENSITY = 0.40;   // 40% R&R during stagnation
constexpr int EXPLOIT_RR_STAGNATION_TRIGGER = 500;     // trigger heavy R&R after 500g stagnation
constexpr int EXPLOIT_RR_INTERVAL = 100;               // apply every 100 gens when stagnant

// === FRANKENSTEIN / BEAM SEARCH ===
constexpr bool ENABLE_FRANKENSTEIN = true;
constexpr int FRANKENSTEIN_BEAM_WIDTH = 50;
constexpr int FRANKENSTEIN_VND_ITERS = 40;
constexpr int FRANKENSTEIN_VND_ITERS_LATE = 60;
constexpr int FRANKENSTEIN_VND_PASSES = 3;
constexpr double FRANKENSTEIN_FORCE_INJECT_PROB = 0.10;
constexpr int FRANKENSTEIN_MAX_INSTANCE_SIZE = 5000;

// === CLONE DETECTION ===
constexpr double CLONE_SIMILARITY_THRESHOLD = 0.99;
constexpr double CLONE_FITNESS_TOLERANCE = 100.0;

// === ELITE CONFIG ===
constexpr double ELITE_RATIO_EXPLORATION_LOW = 0.10;
constexpr double ELITE_RATIO_EXPLORATION_HIGH = 0.50;
constexpr double ELITE_RATIO_EXPLOITATION_LOW = 0.30;
constexpr double ELITE_RATIO_EXPLOITATION_HIGH = 0.90;
constexpr double ELITE_SELECTION_RATIO = 0.30;
constexpr int ELITERATIO = 5;

// === RUIN & RECREATE ===
constexpr double RUIN_BASE_PCT = 0.30;        // exploration base
constexpr double RUIN_INTENSITY_SCALE = 0.40; // exploration scale → 30-70%
constexpr int RUIN_MIN_REMOVED = 5;
constexpr double RUIN_BASE_PCT_EXPLOITATION = 0.10; // exploitation base
constexpr double RUIN_INTENSITY_SCALE_EXPLOITATION =
    0.15; // exploitation scale → 10-25%
constexpr double EXPLOITATION_MIN_MICROSPLIT = 0.20; // min 20% for exploitation

// === LOCAL SEARCH ===
constexpr double EJECTION_PROBABILITY = 0.20;
constexpr double PATH_RELINK_PROBABILITY = 0.15;

// === MIGRATION DIVERSITY ===
constexpr int MIGRATION_ELITE_COUNT = 2; // fixed 2 elite migrants
constexpr int MIGRATION_DIVERSE_COUNT =
    3; // 3 most diverse (max BPD from global best)

// === DYNAMIC IMMUNITY ===
constexpr double PROGRESS_IMMUNITY_SECONDS =
    10.0; // if found NEW BEST in last Xs, block migration

// === DIVERSITY-PULSE MIGRATION === (DISABLED - conflicts with Ring)
constexpr double DIVERSITY_PULSE_INTERVAL_SECONDS =
    9999.0;                                   // effectively disabled
constexpr double DIVERSITY_PULSE_RATE = 0.15; // 15% of population
constexpr double DIVERSITY_PULSE_BPD_THRESHOLD =
    0.20; // must have >20% BPD difference

// === VND OPERATORS ===
constexpr bool ALLOW_SWAP = true;
constexpr bool ALLOW_3SWAP = true;
constexpr bool ALLOW_EJECTION = true;
constexpr bool ALLOW_LOAD_BALANCING = false;  // Enable for imbalanced routes

// === VND SLACK-AWARE === (DISABLED - may cause issues)
constexpr bool VND_SLACK_AWARE = false;      // disabled for stability
constexpr double VND_SLACK_TOLERANCE = 0.01; // accept up to 1% worse moves
constexpr double VND_TIGHT_ROUTE_THRESHOLD =
    0.95; // routes > 95% capacity are "tight"

// === ROUTE POOL CONFIG ===
constexpr size_t ROUTE_POOL_MAX_SIZE = 10000;
constexpr double GREEDY_ASSEMBLER_INTERVAL_SECONDS = 8.0;
constexpr int GREEDY_NUM_STARTS = 10;
constexpr size_t MIN_ROUTE_SIZE_FOR_POOL = 2;

// === MIGRATION (Optimizer) ===
constexpr double MIGRATION_ELITE_RATIO_MIN = 0.10;
constexpr double MIGRATION_ELITE_RATIO_MAX = 0.90;

// === CROSSOVER ===
constexpr double CROSSOVERSEQ_PROBABILITY = 0.30;

// === MISC ===
constexpr int NUM_NEIGHBORS = 20;
constexpr int HISTORY_LAMBDA = 100;
constexpr bool split = false;
constexpr double SPLIT_ROUTE_PENALTY = 0.0;

} // namespace Config
} // namespace LcVRPContest
