#!/bin/bash

DIR="build"
if [ -d "$DIR" ]; then
    echo "Directory $DIR exists. Removing it..."
    rm -rf "$DIR"
else
    echo "No build directory found."
fi

mkdir build
cd build
cmake ..
cmake --build . --parallel
ctest --parallel
