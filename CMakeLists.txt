cmake_minimum_required(VERSION 3.10)
project(LcVRPContest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Source files
set(SOURCES
    LcVRPContest.cpp
    Island.cpp
    LocalSearch.cpp
    Optimizer.cpp
    ProblemData.cpp
    ProblemLoader.cpp
    ProblemGeometry.cpp
    RoutePool.cpp
    Split.cpp
    ThreadSafeEvaluator.cpp
    Mutator.cpp
    Constants.cpp
    Evaluator.cpp
    Individual.cpp
)

# Headers (optional to list, but good for IDEs)
set(HEADERS
    Island.hpp
    LocalSearch.hpp
    Optimizer.hpp
    ProblemData.hpp
    ProblemLoader.hpp
    ProblemGeometry.hpp
    RoutePool.hpp
    Split.hpp
    ThreadSafeEvaluator.hpp
    Mutator.hpp
    Constants.hpp
    Evaluator.hpp
    Individual.hpp
    LocalCache.hpp
    Constants-Lati7450.hpp
)

add_executable(LcVRPContest ${SOURCES} ${HEADERS})

# Optimization flags (MSVC)
if(MSVC)
    target_compile_options(LcVRPContest PRIVATE /O2 /Oi /Ot /Oy /GL /fp:fast)
    target_link_options(LcVRPContest PRIVATE /LTCG)
else()
    target_compile_options(LcVRPContest PRIVATE -O3 -march=native -ffast-math)
endif()
