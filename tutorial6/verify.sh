#!/bin/bash

mkdir -p output
rm ./output/*

set -x 

export RANKS=$1
export N=$2

#export RANKS=4
#export N=400

mpirun -np $RANKS ./2_overlap $N
ls output
mpirun -np $RANKS ./4_hh_overlap $N



