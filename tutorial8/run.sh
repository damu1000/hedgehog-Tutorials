#!/bin/bash

#run on a single macine with multiple GPUs. set visible devices to rank and call the executable
#Parameters: n and b (matrix size and block size)
#run as: mpirun -np 4 ./run.sh 8192 1024

export CUDA_VISIBLE_DEVICES=$PMI_RANK

./tutorial8_distribute_dgemm -n $1 -b $2 -x 8 -a 2 



#use cuda events instead of 
