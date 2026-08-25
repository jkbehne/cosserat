#!/usr/bin/env bash
#
# Builds and installs the Cosserat C++ library.
#
#   ./install.sh /usr/local
#
# Everything lands under a cosserat directory inside the prefix given, so the
# example above produces:
#
#   /usr/local/cosserat/lib                  the shared libraries
#   /usr/local/cosserat/bin                  the example programs
#   /usr/local/cosserat/include              the headers, and the vendored
#                                            ones the headers depend on
#
# A consumer then needs one include path, /usr/local/cosserat/include, and
# writes #include <cosserat/physics/rods.hpp>.
#
# Nothing outside the prefix is touched, and the build happens in a scratch
# directory that is removed afterwards unless --keep-build is given.

# Stop at the first failure rather than carrying on and installing half a
# library; treat an unset variable as a failure; and let a failure anywhere in
# a pipeline fail the pipeline rather than being masked by the last command.
set -o errexit
set -o nounset
set -o pipefail

readonly SCRIPT_NAME="$(basename "${0}")"
readonly SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LIBRARY_DIR="${SOURCE_DIR}/cosserat"

usage() {
    cat <<USAGE
Usage: ${SCRIPT_NAME} PREFIX [options]

Builds and installs the Cosserat C++ library beneath PREFIX/cosserat.

Arguments:
  PREFIX                Directory to install into. Created if it does not
                        exist. Installing to a system location such as
                        /usr/local usually needs sudo.

Options:
  -j, --jobs N          Parallel build jobs. Defaults to every core.
  -t, --build-type TYPE Release, Debug, RelWithDebInfo or MinSizeRel.
                        Defaults to Release.
      --with-tests      Build and run the test suite before installing, and
                        install the test binaries alongside the examples.
      --skip-examples   Do not build or install the example programs.
      --no-docs         Do not build the documentation even if Doxygen is
                        present.
      --keep-build      Leave the scratch build directory in place.
  -h, --help            Show this message.

Examples:
  ${SCRIPT_NAME} ~/.local
  sudo ${SCRIPT_NAME} /usr/local
  ${SCRIPT_NAME} /opt/cosserat --with-tests --build-type Debug
USAGE
}

# Writes a message to stderr and exits.
fail() {
    printf '%s: error: %s\n' "${SCRIPT_NAME}" "${1}" >&2
    exit 1
}

# Writes a progress heading, so a long build says where it has got to.
step() {
    printf '\n== %s\n' "${1}"
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
PREFIX=""
JOBS=""
BUILD_TYPE="Release"
WITH_TESTS="OFF"
BUILD_EXAMPLES="ON"
BUILD_DOCS="ON"
KEEP_BUILD="no"

while [[ ${#} -gt 0 ]]; do
    case "${1}" in
        -h|--help)
            usage
            exit 0
            ;;
        -j|--jobs)
            [[ ${#} -ge 2 ]] || fail "${1} needs a value"
            JOBS="${2}"
            shift 2
            ;;
        -t|--build-type)
            [[ ${#} -ge 2 ]] || fail "${1} needs a value"
            BUILD_TYPE="${2}"
            shift 2
            ;;
        --with-tests)
            WITH_TESTS="ON"
            shift
            ;;
        --skip-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --no-docs)
            BUILD_DOCS="OFF"
            shift
            ;;
        --keep-build)
            KEEP_BUILD="yes"
            shift
            ;;
        -*)
            fail "unknown option ${1}. Run ${SCRIPT_NAME} --help."
            ;;
        *)
            [[ -z "${PREFIX}" ]] || fail "more than one prefix given: ${PREFIX} and ${1}"
            PREFIX="${1}"
            shift
            ;;
    esac
done

if [[ -z "${PREFIX}" ]]; then
    usage >&2
    fail "no install prefix given"
fi

case "${BUILD_TYPE}" in
    Release|Debug|RelWithDebInfo|MinSizeRel) ;;
    *) fail "unknown build type ${BUILD_TYPE}" ;;
esac

# ---------------------------------------------------------------------------
# Checks worth making before a long build rather than after one
# ---------------------------------------------------------------------------
command -v cmake >/dev/null 2>&1 || fail "cmake was not found on PATH"

[[ -f "${LIBRARY_DIR}/CMakeLists.txt" ]] \
    || fail "no CMakeLists.txt under ${LIBRARY_DIR}. Run this from the repository root."

# The vendored dependencies are checked into the tree, so a missing Eigen means
# an incomplete checkout rather than a forgotten submodule step.
[[ -f "${LIBRARY_DIR}/external/eigen/Eigen/Dense" ]] \
    || fail "Eigen is missing from ${LIBRARY_DIR}/external/eigen. The checkout looks incomplete."

if [[ -z "${JOBS}" ]]; then
    if command -v nproc >/dev/null 2>&1; then
        JOBS="$(nproc)"
    elif command -v sysctl >/dev/null 2>&1; then
        JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 1)"
    else
        JOBS=1
    fi
fi

# Resolve the prefix to an absolute path before handing it to CMake, which
# otherwise interprets a relative one against the build directory rather than
# against where the script was run from.
mkdir -p "${PREFIX}" 2>/dev/null \
    || fail "cannot create ${PREFIX}. A system prefix usually needs sudo."
PREFIX="$(cd "${PREFIX}" && pwd)"

[[ -w "${PREFIX}" ]] || fail "${PREFIX} is not writable. A system prefix usually needs sudo."

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cosserat-build-XXXXXX")"
cleanup() {
    if [[ "${KEEP_BUILD}" == "yes" ]]; then
        printf '\nBuild directory kept at %s\n' "${BUILD_DIR}"
    else
        rm -rf "${BUILD_DIR}"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Build and install
# ---------------------------------------------------------------------------
printf 'Cosserat installer\n'
printf '  source      %s\n' "${LIBRARY_DIR}"
printf '  prefix      %s\n' "${PREFIX}"
printf '  build type  %s\n' "${BUILD_TYPE}"
printf '  jobs        %s\n' "${JOBS}"
printf '  tests       %s\n' "${WITH_TESTS}"
printf '  examples    %s\n' "${BUILD_EXAMPLES}"

step "Configuring"
cmake \
    -S "${LIBRARY_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DBUILD_TESTING="${WITH_TESTS}" \
    -DINSTALL_TESTS="${WITH_TESTS}" \
    -DBUILD_EXAMPLES="${BUILD_EXAMPLES}" \
    -DBUILD_DOCS="${BUILD_DOCS}"

step "Building with ${JOBS} job(s)"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

if [[ "${WITH_TESTS}" == "ON" ]]; then
    step "Running the test suite"
    # Run from the build directory, which is where CTest expects to be told
    # about the tests.
    (cd "${BUILD_DIR}" && ctest --output-on-failure)
fi

step "Installing to ${PREFIX}/cosserat"
cmake --install "${BUILD_DIR}"

# ---------------------------------------------------------------------------
# What the user needs to know afterwards
# ---------------------------------------------------------------------------
cat <<SUMMARY

Installed.

  libraries   ${PREFIX}/cosserat/lib
  headers     ${PREFIX}/cosserat/include
  programs    ${PREFIX}/cosserat/bin

To build against it directly:

  g++ -std=c++20 my_program.cpp \\
      -I${PREFIX}/cosserat/include \\
      -L${PREFIX}/cosserat/lib \\
      -lsimulation -lphysics -lmath -lutils \\
      -Wl,-rpath,${PREFIX}/cosserat/lib

Or from CMake:

  find_package(cosserat REQUIRED)
  target_link_libraries(my_target PRIVATE
      cosserat::simulation cosserat::physics cosserat::math cosserat::utils)

  cmake -S . -B build \\
      -DCMAKE_PREFIX_PATH=${PREFIX}/cosserat/lib/cmake/cosserat

Either way the headers are spelled the same:

  #include <cosserat/physics/rods.hpp>
SUMMARY
