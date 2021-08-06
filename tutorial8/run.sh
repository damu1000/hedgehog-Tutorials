#!/usr/bin/bash

#SBATCH --partition=debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --job-name=output_afinity
#SBATCH -o output.txt

module purge
module load gcc/7.3.1 cuda/10.2 openmpi/4.0.4/gcc-7.3.1 cmake


export UCX_WARN_UNUSED_ENV_VARS=0
export UCX_NET_DEVICES=mlx5_0:1

export wrk_directory="/home/drs15/hedgehog/tutorials/tutorial8"


#mpirun $MPI_ARGS -np 4 $wrk_directory/tutorial8_distribute_dgemm



#srun --mpi=pmi2 -n $SLURM_NTASKS -m plane=2 --cpu-bind=verbose,threads  $wrk_directory/test.sh

cd $wrk_directory

#make

srun --mpi=pmi2 -n 4 -m plane=2 --cpu-bind=verbose,threads ./test.sh


