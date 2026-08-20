cmake_minimum_required(VERSION 3.25)

# ---------------------------------------------------------------------------
# Compiler selection.
#
# This block has to run BEFORE project(), because project() is what probes and
# locks in the compiler. Setting CMAKE_CXX_COMPILER afterwards has no effect.
# An explicit -DCMAKE_CXX_COMPILER=... or a CXX environment variable always
# wins, so this only supplies a default.
# ---------------------------------------------------------------------------
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(CLANGXX_EXECUTABLE NAMES clang++)
    if(CLANGXX_EXECUTABLE)
        set(CMAKE_CXX_COMPILER "${CLANGXX_EXECUTABLE}" CACHE FILEPATH "C++ compiler")
        message(STATUS "clang++ found at ${CLANGXX_EXECUTABLE}, using it")
    else()
        message(STATUS "clang++ not found, falling back to the system default compiler")
    endif()
endif()

project(cpp LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language level. std::source_location and concepts both require C++20.
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Let the test binary find libutils.so out of the build tree without
# LD_LIBRARY_PATH juggling.
set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)

# ---------------------------------------------------------------------------
# Strict warning set, applied only to our own targets so that third party
# code (GoogleTest) is never compiled with -Werror.
# ---------------------------------------------------------------------------
if(MSVC)
    set(PROJECT_STRICT_WARNINGS
        /W4
        /WX
        /permissive-
    )
    set(PROJECT_OPTIMIZATION
        /O2
    )
else()
    set(PROJECT_STRICT_WARNINGS
        -Wall
        -Wextra
        -Wpedantic
        -Werror              # every warning below is fatal, unused variables included
        -Wunused             # -Wunused-variable, -Wunused-parameter, -Wunused-but-set-variable, ...
        -Wshadow
        -Wnon-virtual-dtor
        -Woverloaded-virtual
        -Wcast-qual
        -Wcast-align
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
        -Wextra-semi
    )
    set(PROJECT_OPTIMIZATION
        -O3
    )
    # Useful but noisy through GoogleTest's macros. Enable per target once
    # you are happy with the baseline:
    #   -Wold-style-cast -Wconversion -Wsign-conversion -Wuseless-cast
endif()

# ---------------------------------------------------------------------------
# The utils library.
#
# Globbing is convenient but CMake cannot see a new file appear without being
# re-run; CONFIGURE_DEPENDS makes the build system re-glob on each build. It
# is still worth listing sources explicitly once the set settles down.
#
# utils is header only today, and add_library(... SHARED) with an empty source
# list is a hard configure error, so fall back to an INTERFACE target until
# the first .cpp shows up. Consumers link against utils::utils either way.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE UTILS_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp"
)
list(FILTER UTILS_SOURCES EXCLUDE REGEX "/test/")

if(UTILS_SOURCES)
    add_library(utils SHARED ${UTILS_SOURCES})
    set(UTILS_SCOPE PUBLIC)
    set_target_properties(utils PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(utils PRIVATE ${PROJECT_STRICT_WARNINGS} ${PROJECT_OPTIMIZATION})
    message(STATUS "utils: shared library from ${UTILS_SOURCES}")
else()
    add_library(utils INTERFACE)
    set(UTILS_SCOPE INTERFACE)
    message(STATUS "utils: no .cpp sources found, building as a header only INTERFACE target")
endif()

add_library(utils::utils ALIAS utils)

# Headers are included as "utils/assertions.hpp", so the include root is this
# directory rather than the utils subdirectory.
target_include_directories(utils ${UTILS_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:include>"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/json/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen"
)
target_compile_features(utils ${UTILS_SCOPE} cxx_std_20)

# ---------------------------------------------------------------------------
# The math library.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE MATH_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/math/*.cpp"
)
list(FILTER MATH_SOURCES EXCLUDE REGEX "/test/")

if(MATH_SOURCES)
    add_library(math SHARED ${MATH_SOURCES})
    set(MATH_SCOPE PUBLIC)
    set_target_properties(math PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(math PRIVATE ${PROJECT_STRICT_WARNINGS} ${PROJECT_OPTIMIZATION})
    message(STATUS "math: shared library from ${MATH_SOURCES}")
else()
    add_library(math INTERFACE)
    set(MATH_SCOPE INTERFACE)
    message(STATUS "math: no .cpp sources found, building as a header only INTERFACE target")
endif()

add_library(math::math ALIAS math)

target_include_directories(math ${MATH_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:include>"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen"
)
target_compile_features(math ${MATH_SCOPE} cxx_std_20)
target_link_libraries(math ${MATH_SCOPE} utils::utils)

# ---------------------------------------------------------------------------
# The physics library.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE PHYSICS_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/physics/*.cpp"
)
list(FILTER PHYSICS_SOURCES EXCLUDE REGEX "/test/")

if(PHYSICS_SOURCES)
    add_library(physics SHARED ${PHYSICS_SOURCES})
    set(PHYSICS_SCOPE PUBLIC)
    set_target_properties(physics PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(physics PRIVATE ${PROJECT_STRICT_WARNINGS} ${PROJECT_OPTIMIZATION})
    message(STATUS "physics: shared library from ${PHYSICS_SOURCES}")
else()
    add_library(physics INTERFACE)
    set(PHYSICS_SCOPE INTERFACE)
    message(STATUS "physics: no .cpp sources found, building as a header only INTERFACE target")
endif()

add_library(physics::physics ALIAS physics)

target_include_directories(physics ${PHYSICS_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:include>"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen"
)
target_compile_features(physics ${PHYSICS_SCOPE} cxx_std_20)
target_link_libraries(physics ${PHYSICS_SCOPE} math::math utils::utils)

# ---------------------------------------------------------------------------
# The simulation library.
# ---------------------------------------------------------------------------
file(GLOB_RECURSE SIM_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/simulation/*.cpp"
)
list(FILTER SIM_SOURCES EXCLUDE REGEX "/test/")

if(SIM_SOURCES)
    add_library(simulation SHARED ${SIM_SOURCES})
    set(SIM_SCOPE PUBLIC)
    set_target_properties(simulation PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(simulation PRIVATE ${PROJECT_STRICT_WARNINGS} ${PROJECT_OPTIMIZATION})
    message(STATUS "simulation: shared library from ${SIM_SOURCES}")
else()
    add_library(simulation INTERFACE)
    set(SIM_SCOPE INTERFACE)
    message(STATUS "simulation: no .cpp sources found, building as a header only INTERFACE target")
endif()

add_library(simulation::simulation ALIAS simulation)

target_include_directories(simulation ${SIM_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:include>"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen"
)
target_compile_features(simulation ${SIM_SCOPE} cxx_std_20)
target_link_libraries(simulation ${SIM_SCOPE} utils::utils)

# ---------------------------------------------------------------------------
# GoogleTest. Prefer a system install, fetch a pinned release otherwise.
# SYSTEM keeps GoogleTest's headers out of our warning set.
# ---------------------------------------------------------------------------
option(BUILD_TESTING "Build the tests" ON)

if(BUILD_TESTING)
    enable_testing()

    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        message(STATUS "GTest not found locally, fetching it")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.15.2
            GIT_SHALLOW    TRUE
            SYSTEM
            EXCLUDE_FROM_ALL
        )
        # Keep GoogleTest from overriding the parent project's CRT on Windows.
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    include(GoogleTest)

    add_executable(test_assertions utils/test/test_assertions.cpp)
    target_link_libraries(test_assertions
        PRIVATE
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_assertions PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_assertions
        PROPERTIES
            LABELS utils
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_file_utils utils/test/test_file_utils.cpp)
    target_link_libraries(test_file_utils
        PRIVATE
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_file_utils PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_file_utils
        PROPERTIES
            LABELS utils
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_functions math/test/test_functions.cpp)
    target_link_libraries(test_functions
        PRIVATE
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_functions PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_functions
        PROPERTIES
            LABELS math
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_linalg math/test/test_linalg.cpp)
    target_link_libraries(test_linalg
        PRIVATE
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_linalg PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_linalg
        PROPERTIES
            LABELS math
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_indexing math/test/test_indexing.cpp)
    target_link_libraries(test_indexing
        PRIVATE
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_indexing PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_indexing
        PROPERTIES
            LABELS math
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_finite_difference math/test/test_finite_difference.cpp)
    target_link_libraries(test_finite_difference
        PRIVATE
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_finite_difference PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_finite_difference
        PROPERTIES
            LABELS math
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_forces physics/test/test_forces.cpp)
    target_link_libraries(test_forces
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_forces PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_forces
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_damping physics/test/test_damping.cpp)
    target_link_libraries(test_damping
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_damping PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_damping
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_constraints physics/test/test_constraints.cpp)
    target_link_libraries(test_constraints
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_constraints PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_constraints
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_joints physics/test/test_joints.cpp)
    target_link_libraries(test_joints
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_joints PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_joints
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_rods physics/test/test_rods.cpp)
    target_link_libraries(test_rods
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_rods PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_rods
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_rigid_body physics/test/test_rigid_body.cpp)
    target_link_libraries(test_rigid_body
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_rigid_body PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_rigid_body
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_body_rules physics/test/test_body_rules.cpp)
    target_link_libraries(test_body_rules
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_body_rules PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_body_rules
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_dynamics_kinematics physics/test/test_dynamics_kinematics.cpp)
    target_link_libraries(test_dynamics_kinematics
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_dynamics_kinematics PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_dynamics_kinematics
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_bodies physics/test/test_bodies.cpp)
    target_link_libraries(test_bodies
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_bodies PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_bodies
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_minimum_distance physics/test/test_minimum_distance.cpp)
    target_link_libraries(test_minimum_distance
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_minimum_distance PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_minimum_distance
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_contacts physics/test/test_contacts.cpp)
    target_link_libraries(test_contacts
        PRIVATE
            physics::physics
            math::math
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_contacts PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_contacts
        PROPERTIES
            LABELS physics
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_solver simulation/test/test_solver.cpp)
    target_link_libraries(test_solver
        PRIVATE
            simulation::simulation
            utils::utils
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_solver PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_solver
        PROPERTIES
            LABELS simulation
        DISCOVERY_TIMEOUT 30
    )

    add_executable(test_diagnostics simulation/test/test_diagnostics.cpp)
    target_link_libraries(test_diagnostics
        PRIVATE
            simulation::simulation
            utils::utils
            physics::physics
            GTest::gtest
            GTest::gtest_main
    )
    target_compile_options(test_diagnostics PRIVATE ${PROJECT_STRICT_WARNINGS})
    gtest_discover_tests(test_diagnostics
        PROPERTIES
            LABELS simulation
        DISCOVERY_TIMEOUT 30
    )
endif()
