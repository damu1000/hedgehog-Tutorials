## Tutorial 7 - Under development. 

This is a basic working version. Check To Do list.

Using hedgehog to run double-precision distributed "streaming" matrix multiplication (dgemm) using [Cannon's algorithm](https://en.wikipedia.org/wiki/Cannon%27s_algorithm).

Tutorial 3 is modified and two more hedgehog tasks are added for each matrix to carry out MPI communication. Each task runs using one thread. Additional dependencies and the state manager are created to execute the main loop in Cannon's algorithm.

Compile: Update hedgehog path in the file "compile.sh" and run it. Tested with g++\-8 and MPICH 3.2.

Run: mpirun -np 9 ./matmult -n 18 -v 1 -b 2

For simplicity num of ranks must be a perfect square, and matrix size should be divisible by sqrt(num of ranks). Matrix is assumed to be a square matrix. e.g., running the above mpirun configuration will spawn 9 ranks with the matrix size of 18x18. Each rank will get a patch of 6x6. Each rank will divide the its patch into smaller blocks of 2x2 i.e., 9 blocks per rank and the multiplication is carried out as a "streaming" of blocks.

### To DO:
- Counts of blocks and "q" passed to tasks are tricky to figure out. Add comments in the code before the logic evaporates from the mind.
- Test with FP numbers.
- Verify multiple threads for product and addition task.
- Currently assuming the square matrices, and square blocks. So using nblocks in many places. Fix it to correct mblocks / pblocks. 
- See if moving finalize comm task after product task works fine. If yes, comm and addition can be overlapped.
- See if the two for loops in the finalize comm task can be merged. Will give better streaming.
- See if MPI_Testall can be called for the previous blocks for every new block pushed into the finalize comm task. Will help comm progress.
- Test performance, figure out ideal thread configuration.
- Find out a way to call destroyComm. Currently it is not destroyed causing a memory leak.
- Add changes between Tutorial 3 and 7.
- Currently assuming the square matrices, and square blocks. Fix for uneven, non square sizes.

Caution: Getting rid of race conditions / hangs / deadlocks coupled with comm is very tricky. Take baby steps.

