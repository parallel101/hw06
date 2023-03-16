#!/bin/bash 
if [[ $1 = "clean" && -d build ]];then
    rm -rf build 
fi
set -e
cmake -B build
cmake --build build -- -j 4
build/main
