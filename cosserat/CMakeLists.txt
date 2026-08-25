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

project(cosserat LANGUAGES CXX)
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
# Only ELF and AIX understand $ORIGIN; CMake ignores this elsewhere, but
# saying so explicitly keeps the two rpath decisions side by side.
if(NOT APPLE AND NOT WIN32)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
endif()

# The installed libraries depend on one another, and the installed examples on
# all of them. Without a run path they are found at link time and then not at
# load time, which shows up as a binary that builds and refuses to start.
#
# The run path has to be relative, so that moving or renaming the install
# prefix does not break it, and the spelling of "relative to me" differs by
# platform. ELF uses $ORIGIN. Mach-O has no such variable at all and uses
# @loader_path, resolved against whichever binary is doing the loading; giving
# dyld a literal $ORIGIN produces exactly the run time failure this is meant to
# prevent, since it goes looking for a directory of that name.
#
# Either way there are two entries: the first is a library in lib finding its
# siblings, the second an executable in bin finding that lib.
if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../lib")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# Mach-O records an install name in every shared library, and a consumer links
# against whatever that says. @rpath defers the decision to the run path above
# rather than baking in the build directory. CMake defaults this on, but the
# whole scheme depends on it, so it is stated rather than assumed.
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
endif()

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
    # Two of the warnings below are demoted for GCC, further down, because GCC
    # raises them from inside Eigen after inlining and no amount of SYSTEM
    # marking suppresses a diagnostic raised that late. Clang raises neither.
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
    set(PROJECT_PROFILING
        -fprofile-instr-generate
        -fcoverage-mapping
    )
    # Useful but noisy through GoogleTest's macros. Enable per target:
    #   -Wold-style-cast -Wconversion -Wsign-conversion -Wuseless-cast
endif()

# ---------------------------------------------------------------------------
# GCC only. -Wnull-dereference and -Wmaybe-uninitialized are middle end
# diagnostics, raised once the optimiser has inlined vendored code into ours.
# The mechanism that keeps system headers quiet runs in the front end and never
# sees them, so marking Eigen SYSTEM does not help: a GCC build fails inside
# Eigen's BinaryFunctors.h with -Werror on.
#
# Demoted rather than switched off, so they still appear as warnings and a real
# one in our own code is still visible. Clang raises neither, so nothing
# changes there.
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND PROJECT_STRICT_WARNINGS
        -Wno-error=null-dereference
        -Wno-error=maybe-uninitialized
    )
endif()

message(STATUS "install rpath: ${CMAKE_INSTALL_RPATH}")

option(ENABLE_OPTIMIZATION "Enables code optimization" ON)
option(ENABLE_PROFILING "Enables LLVM code profiling" OFF)

if (ENABLE_PROFILING)
    add_link_options(-fprofile-instr-generate)
endif()

# ---------------------------------------------------------------------------
# The utils library.
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
    target_compile_options(utils PRIVATE ${PROJECT_STRICT_WARNINGS})
    if (ENABLE_OPTIMIZATION)
        target_compile_options(utils PRIVATE ${PROJECT_OPTIMIZATION})
    endif()
    if (ENABLE_PROFILING)
        target_compile_options(utils PRIVATE ${PROJECT_PROFILING})
    endif()
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
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>"
    "$<INSTALL_INTERFACE:cosserat/include>"
)

# Vendored headers, marked SYSTEM so third party code is not held to this
# project's warning set. Without it every build fails inside Eigen.
target_include_directories(utils SYSTEM ${UTILS_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/json/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/eigen>"
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
    target_compile_options(math PRIVATE ${PROJECT_STRICT_WARNINGS})
    if (ENABLE_OPTIMIZATION)
        target_compile_options(math PRIVATE ${PROJECT_OPTIMIZATION})
    endif()
    if (ENABLE_PROFILING)
        target_compile_options(math PRIVATE ${PROJECT_PROFILING})
    endif()
    message(STATUS "math: shared library from ${MATH_SOURCES}")
else()
    add_library(math INTERFACE)
    set(MATH_SCOPE INTERFACE)
    message(STATUS "math: no .cpp sources found, building as a header only INTERFACE target")
endif()

add_library(math::math ALIAS math)

target_include_directories(math ${MATH_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>"
    "$<INSTALL_INTERFACE:cosserat/include>"
)

# Vendored headers, marked SYSTEM so third party code is not held to this
# project's warning set. Without it every build fails inside Eigen.
target_include_directories(math SYSTEM ${MATH_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/eigen>"
)


# TriangleMeshDistance backs math/triangle_mesh_field.cpp. PRIVATE because the
# header hides it behind a pointer to an incomplete type, so nothing that links
# math needs this path. SYSTEM because it is third party code and must not be
# held to the project's -Werror set.
if(MATH_SOURCES)
    target_include_directories(math SYSTEM PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/TriangleMeshDistance/TriangleMeshDistance/include>"
    )

    # SYSTEM is not enough for two of GCC's warnings. Both are late diagnostics
    # raised by the optimiser, after the vendored code has been inlined into
    # ours, and the suppression that keeps system headers quiet works on the
    # front end and does not reach them. Clang does not raise either, so this
    # only shows up on a GCC build.
    #
    # Demoted rather than disabled, and only on the one file that includes the
    # vendored header, so the same warnings are still errors everywhere else.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set_source_files_properties(
            "${CMAKE_CURRENT_SOURCE_DIR}/math/triangle_mesh_field.cpp"
            PROPERTIES COMPILE_OPTIONS
                "-Wno-error=maybe-uninitialized;-Wno-error=null-dereference"
        )
    endif()
endif()
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
    target_compile_options(physics PRIVATE ${PROJECT_STRICT_WARNINGS})
    if (ENABLE_OPTIMIZATION)
        target_compile_options(physics PRIVATE ${PROJECT_OPTIMIZATION})
    endif()
    if (ENABLE_PROFILING)
        target_compile_options(physics PRIVATE ${PROJECT_PROFILING})
    endif()
    message(STATUS "physics: shared library from ${PHYSICS_SOURCES}")
else()
    add_library(physics INTERFACE)
    set(PHYSICS_SCOPE INTERFACE)
    message(STATUS "physics: no .cpp sources found, building as a header only INTERFACE target")
endif()

add_library(physics::physics ALIAS physics)

target_include_directories(physics ${PHYSICS_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>"
    "$<INSTALL_INTERFACE:cosserat/include>"
)

# Vendored headers, marked SYSTEM so third party code is not held to this
# project's warning set. Without it every build fails inside Eigen.
target_include_directories(physics SYSTEM ${PHYSICS_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/eigen>"
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
    target_compile_options(simulation PRIVATE ${PROJECT_STRICT_WARNINGS})
    if (ENABLE_OPTIMIZATION)
        target_compile_options(simulation PRIVATE ${PROJECT_OPTIMIZATION})
    endif()
    if (ENABLE_PROFILING)
        target_compile_options(simulation PRIVATE ${PROJECT_PROFILING})
    endif()
    message(STATUS "simulation: shared library from ${SIM_SOURCES}")
else()
    add_library(simulation INTERFACE)
    set(SIM_SCOPE INTERFACE)
    message(STATUS "simulation: no .cpp sources found, building as a header only INTERFACE target")
endif()

add_library(simulation::simulation ALIAS simulation)

target_include_directories(simulation ${SIM_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>"
    "$<INSTALL_INTERFACE:cosserat/include>"
)

# Vendored headers, marked SYSTEM so third party code is not held to this
# project's warning set. Without it every build fails inside Eigen.
target_include_directories(simulation SYSTEM ${SIM_SCOPE}
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/eigen>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/ordered-map/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/spdlog/include>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/cxxopts/include>"
)

target_compile_features(simulation ${SIM_SCOPE} cxx_std_20)
target_link_libraries(simulation ${SIM_SCOPE} utils::utils math::math physics::physics)

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

    # -----------------------------------------------------------------------
    # Test executables, one per test source, found rather than listed.
    #
    # A test's dependencies follow from the directory it sits in, not from the
    # file, so they are declared once per directory here. Physics tests link
    # the simulation library as well, because a quantity defined over a whole
    # collection can only be exercised against a real graph.
    #
    # Targets are named test_<directory>_<file> rather than test_<file>, since
    # a bare file name is only unique within its directory and two directories
    # are free to test similarly named things.
    # -----------------------------------------------------------------------
    set(UTILS_TEST_LIBS      utils::utils)
    set(MATH_TEST_LIBS       math::math utils::utils)
    set(PHYSICS_TEST_LIBS    simulation::simulation physics::physics math::math
                             utils::utils)
    set(SIMULATION_TEST_LIBS simulation::simulation physics::physics math::math
                             utils::utils)

    foreach(TEST_DIRECTORY IN ITEMS utils math physics simulation)
        string(TOUPPER "${TEST_DIRECTORY}" TEST_DIRECTORY_UPPER)

        file(GLOB TEST_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/${TEST_DIRECTORY}/test/*.cpp"
        )

        foreach(TEST_SOURCE IN LISTS TEST_SOURCES)
            get_filename_component(TEST_STEM "${TEST_SOURCE}" NAME_WE)
            # test_rods.cpp in physics becomes test_physics_rods, so the
            # directory is named once rather than twice.
            string(REGEX REPLACE "^test_" "" TEST_STEM "${TEST_STEM}")
            set(TEST_TARGET "test_${TEST_DIRECTORY}_${TEST_STEM}")

            add_executable(${TEST_TARGET} "${TEST_SOURCE}")
            target_link_libraries(${TEST_TARGET}
                PRIVATE
                    ${${TEST_DIRECTORY_UPPER}_TEST_LIBS}
                    GTest::gtest
                    GTest::gtest_main
            )
            target_compile_options(${TEST_TARGET}
                PRIVATE ${PROJECT_STRICT_WARNINGS}
            )
            gtest_discover_tests(${TEST_TARGET}
                PROPERTIES
                    LABELS ${TEST_DIRECTORY}
                DISCOVERY_TIMEOUT 30
            )
            list(APPEND COSSERAT_TEST_TARGETS ${TEST_TARGET})
        endforeach()

        list(LENGTH TEST_SOURCES TEST_COUNT)
        message(STATUS "${TEST_DIRECTORY}: ${TEST_COUNT} test executable(s)")
    endforeach()
endif()

option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    # -----------------------------------------------------------------------
    # Example executables, found rather than listed.
    #
    # Every example links the whole stack, since an example exists to show the
    # library being used and there is nothing to be gained from working out
    # which parts of it a given one happens to touch.
    #
    # An example at examples/<group>/<name>.cpp becomes <group>_<name>, so two
    # groups may both have a "basic" without colliding.
    # -----------------------------------------------------------------------
    file(GLOB_RECURSE EXAMPLE_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
    )

    foreach(EXAMPLE_SOURCE IN LISTS EXAMPLE_SOURCES)
        get_filename_component(EXAMPLE_STEM "${EXAMPLE_SOURCE}" NAME_WE)
        get_filename_component(EXAMPLE_DIRECTORY "${EXAMPLE_SOURCE}" DIRECTORY)
        get_filename_component(EXAMPLE_GROUP "${EXAMPLE_DIRECTORY}" NAME)

        # An example sitting directly in examples/ has no group to prefix with.
        if(EXAMPLE_GROUP STREQUAL "examples")
            set(EXAMPLE_TARGET "${EXAMPLE_STEM}")
        else()
            set(EXAMPLE_TARGET "${EXAMPLE_GROUP}_${EXAMPLE_STEM}")
        endif()

        add_executable(${EXAMPLE_TARGET} "${EXAMPLE_SOURCE}")
        target_link_libraries(${EXAMPLE_TARGET}
            PRIVATE
                simulation::simulation
                physics::physics
                math::math
                utils::utils
        )
        target_compile_options(${EXAMPLE_TARGET}
            PRIVATE ${PROJECT_STRICT_WARNINGS}
        )
        if (ENABLE_OPTIMIZATION)
            target_compile_options(${EXAMPLE_TARGET} PRIVATE ${PROJECT_OPTIMIZATION})
        endif()
        if (ENABLE_PROFILING)
            target_compile_options(${EXAMPLE_TARGET} PRIVATE ${PROJECT_PROFILING})
        endif()
        list(APPEND COSSERAT_EXAMPLE_TARGETS ${EXAMPLE_TARGET})
    endforeach()

    list(LENGTH EXAMPLE_SOURCES EXAMPLE_COUNT)
    message(STATUS "examples: ${EXAMPLE_COUNT} executable(s)")
endif()

# ---------------------------------------------------------------------------
# Documentation.
#
# Built by default when Doxygen is available, and quietly skipped when it is
# not: a missing documentation tool is not a reason to fail a build of the
# library itself. Turn it off with -DBUILD_DOCS=OFF.
#
# doxygen_add_docs writes its own Doxyfile from the DOXYGEN_* variables set
# below, so there is no checked in Doxyfile to drift out of step with this.
# ---------------------------------------------------------------------------
option(BUILD_DOCS "Build the Doxygen documentation" ON)

if(BUILD_DOCS)
    find_package(Doxygen QUIET)

    if(DOXYGEN_FOUND)
        set(DOXYGEN_PROJECT_NAME "Cosserat")
        set(DOXYGEN_PROJECT_BRIEF "Cosserat rod simulation")
        set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")

        # The comments in this project are written against these: @section and
        # @ref are used across files, and the maths is written as LaTeX.
        set(DOXYGEN_JAVADOC_AUTOBRIEF YES)
        set(DOXYGEN_EXTRACT_ALL YES)
        set(DOXYGEN_EXTRACT_STATIC YES)
        set(DOXYGEN_USE_MATHJAX YES)
        set(DOXYGEN_GENERATE_TREEVIEW YES)
        set(DOXYGEN_RECURSIVE YES)
        set(DOXYGEN_QUIET YES)
        set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)

        # Vendored dependencies and build output are not ours to document, and
        # test files document their reasoning for a reader of the source rather
        # than for a generated reference.
        set(DOXYGEN_EXCLUDE_PATTERNS
            "*/external/*"
            "*/build*/*"
            "*/test/*"
            "*/.git/*"
        )

        doxygen_add_docs(docs
            "${CMAKE_CURRENT_SOURCE_DIR}/math"
            "${CMAKE_CURRENT_SOURCE_DIR}/physics"
            "${CMAKE_CURRENT_SOURCE_DIR}/simulation"
            "${CMAKE_CURRENT_SOURCE_DIR}/utils"
            "${CMAKE_CURRENT_SOURCE_DIR}/examples"
            ALL
            COMMENT "Generating documentation with Doxygen"
        )
        message(STATUS "docs: Doxygen ${DOXYGEN_VERSION}, building by default")
    else()
        message(STATUS
            "docs: Doxygen not found, skipping. The build is otherwise "
            "unaffected; install Doxygen or pass -DBUILD_DOCS=OFF to silence "
            "this."
        )
    endif()
endif()

# ---------------------------------------------------------------------------
# Installation.
#
# Everything lands under a cosserat directory inside the install prefix, so
# that installing to /usr/local gives:
#
#   /usr/local/cosserat/lib                  the shared libraries
#   /usr/local/cosserat/bin                  examples
#   /usr/local/cosserat/include/cosserat/... the headers, structure mirrored
#
# A consumer then puts /usr/local/cosserat/include on the include path and
# writes #include <cosserat/physics/rods.hpp>.
#
# One include directory is enough, because the library spells its own includes
# the same way a consumer does: <cosserat/physics/rods.hpp>, resolved from the
# directory holding cosserat/. That is what the source layout is for, and it is
# why an installed header can include another installed header without the
# consumer needing to know where the tree is rooted.
# ---------------------------------------------------------------------------
set(COSSERAT_INSTALL_ROOT     "cosserat")
set(COSSERAT_INSTALL_LIBDIR   "${COSSERAT_INSTALL_ROOT}/lib")
set(COSSERAT_INSTALL_BINDIR   "${COSSERAT_INSTALL_ROOT}/bin")
set(COSSERAT_INSTALL_INCDIR   "${COSSERAT_INSTALL_ROOT}/include")
set(COSSERAT_INSTALL_HEADERS  "${COSSERAT_INSTALL_INCDIR}/cosserat")

install(TARGETS utils math physics simulation
    EXPORT cosserat-targets
    LIBRARY   DESTINATION "${COSSERAT_INSTALL_LIBDIR}"
    ARCHIVE   DESTINATION "${COSSERAT_INSTALL_LIBDIR}"
    RUNTIME   DESTINATION "${COSSERAT_INSTALL_BINDIR}"
    INCLUDES  DESTINATION "${COSSERAT_INSTALL_INCDIR}"
)

# Headers, with the source tree's structure mirrored under cosserat/.
foreach(HEADER_DIRECTORY IN ITEMS math physics simulation utils)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${HEADER_DIRECTORY}"
        DESTINATION "${COSSERAT_INSTALL_HEADERS}"
        FILES_MATCHING
            PATTERN "*.hpp"
            PATTERN "*.h"
            PATTERN "test" EXCLUDE
    )
endforeach()

# The vendored header only dependencies our own headers include. Without these
# the promise that one include path is enough would be false: a consumer would
# get as far as <cosserat/physics/rods.hpp> and then fail on <Eigen/Core>.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen/Eigen"
    DESTINATION "${COSSERAT_INSTALL_INCDIR}"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen/unsupported"
    DESTINATION "${COSSERAT_INSTALL_INCDIR}"
    OPTIONAL
)
foreach(VENDORED IN ITEMS
    "external/json/include/nlohmann"
    "external/ordered_map/include/tsl"
    "external/spdlog/include/spdlog"
    "external/cxxopts/include"
)
    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${VENDORED}")
        get_filename_component(VENDORED_NAME "${VENDORED}" NAME)
        if(VENDORED_NAME STREQUAL "include")
            # cxxopts puts its header directly in include/, with no directory
            # of its own to copy.
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${VENDORED}/"
                DESTINATION "${COSSERAT_INSTALL_INCDIR}"
                FILES_MATCHING PATTERN "*.hpp"
            )
        else()
            install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${VENDORED}"
                DESTINATION "${COSSERAT_INSTALL_INCDIR}"
            )
        endif()
    endif()
endforeach()

if(BUILD_EXAMPLES AND COSSERAT_EXAMPLE_TARGETS)
    install(TARGETS ${COSSERAT_EXAMPLE_TARGETS}
        RUNTIME DESTINATION "${COSSERAT_INSTALL_BINDIR}"
    )
endif()

# Tests are installed only when asked for. They are useful for checking an
# install on the machine it landed on, and pure noise otherwise.
option(INSTALL_TESTS "Install the test executables alongside the examples" OFF)
if(INSTALL_TESTS AND BUILD_TESTING AND COSSERAT_TEST_TARGETS)
    install(TARGETS ${COSSERAT_TEST_TARGETS}
        RUNTIME DESTINATION "${COSSERAT_INSTALL_BINDIR}"
    )
endif()

# A config package, so a consumer can find_package(cosserat) rather than
# hand-writing include and library paths.
install(EXPORT cosserat-targets
    FILE cosserat-targets.cmake
    NAMESPACE cosserat::
    DESTINATION "${COSSERAT_INSTALL_LIBDIR}/cmake/cosserat"
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cosserat-config-version.cmake"
    VERSION 0.1.0
    COMPATIBILITY SameMajorVersion
)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cosserat-config.cmake"
"include(\"\${CMAKE_CURRENT_LIST_DIR}/cosserat-targets.cmake\")\n"
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cosserat-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cosserat-config-version.cmake"
    DESTINATION "${COSSERAT_INSTALL_LIBDIR}/cmake/cosserat"
)
