## Tutorial 7 - Under development. 

Using hedgehog to run distributed double-precision matrix multiplication (dgemm) using [Cannon's algorithm](https://en.wikipedia.org/wiki/Cannon%27s_algorithm).

Two hedgehog tasks are created for MPI communication and one for the main matrix multiplication. Each task runs using one thread. The dependencies and the state manager are created to execute the main loop in Cannon's algorithm.

Compile: Update hedgehog path in the file "compile.sh" and run it. Tested with g++\-8 and MPICH 3.2.

Run: run verify.sh as: 

./verify.sh \<num of ranks> \<matrix size>

For simplicity num of ranks must be a perfect square, and matrix size should be divisible by sqrt(p). Matrix is assumed to be a square matrix. e.g., running "./verify.sh 9 300" will spawn 9 ranks with the matrix size of 300x300. Each rank will get a patch of 100x100. Running verify.sh will first execute the non-hedgehog version, which dumps the result in the "output" directory. Later the hedgehog version of dgemm is executed, and the output is compared against the output of the non-hedgehog version.



