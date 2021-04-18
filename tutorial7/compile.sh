#!/bin/bash

export MPICH_CXX=g++-8
export HH_PATH=$HOME/hedgehog/hedgehog/build
export OpenBLASS_PATH=/home/damodars/install/OpenBLAS/build

set -x 

mpicxx -I$HH_PATH/include/ -I$OpenBLASS_PATH/inlcude -L$OpenBLASS_PATH/lib -lopenblas ./tutorial7.cc -std=c++17 -pthread -fopenmp -g -o dgemm -O0 -DUSE_HH -DUSE_OPENBLAS -DOVERLAP_COMM -o matmult
