#!/bin/bash

export MPICH_CXX=g++-8
export HH_PATH=$HOME/hedgehog/hedgehog/build/

set -x 

#mpicxx ./tutorial6.cc -std=c++17 -pthread -fopenmp -g -o dgemm -O3 -o 1_no_overlap
mpicxx ./tutorial6.cc -std=c++17 -pthread -fopenmp -g -o dgemm -O3 -DOVERLAP_COMM -o 2_overlap
#mpicxx -I$HH_PATH/include/ ./tutorial6.cc -std=c++17 -pthread -fopenmp -g -o dgemm -O0 -DUSE_HH -DUSE_OPENBLAS -o 3_hh_no_overlap
mpicxx -I$HH_PATH/include/ ./tutorial6.cc -std=c++17 -pthread -fopenmp -g -o dgemm -O3 -DUSE_HH -DUSE_OPENBLAS -DOVERLAP_COMM -o 4_hh_overlap
