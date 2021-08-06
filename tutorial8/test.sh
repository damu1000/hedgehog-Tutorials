#!/usr/bin/bash

export LOCAL_RANK=$(( PMI_RANK % 4 ))
echo inside test.sh $LOCAL_RANK

#$OMPI_COMM_WORLD_LOCAL_RANK

wrk_directory="/home/drs15/hedgehog/tutorials/tutorial8"

export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

#env 

#$wrk_directory/tutorial8_distribute_dgemm -x 16 -a 8 -n 64 -b 4 -v 1

#$wrk_directory/tutorial8_distribute_dgemm -x 16 -a 8 -n 16384 -b 2048

$wrk_directory/tutorial8_distribute_dgemm -x 16 -a 8 -n 32768 -b 2048

#$wrk_directory/tutorial8_distribute_dgemm -x 16 -a 8 -n 32768 -b 4096

