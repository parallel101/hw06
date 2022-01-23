#!/bin/sh
set -e
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=/Users/yangzhikai/farewell/cpp_review/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
build/main
