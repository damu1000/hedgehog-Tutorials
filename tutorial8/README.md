## Tutorial 8: Distributed GEMM on GPUs

Under development: This is a basic working version. 

combination of Tutorials 4 and 7.

Steps to convert a single node GEMM HH graph to multinode MPI graph.
- Add "comm" directory with comm.h and comm_tasks.h. Ensure order is correct either row major or column major.
- Init MPI and perform related setup (such as domain decomposition) in main().
- Call createCartComm.
- Allocate and initate matrices.
- Setup routines for gemm with HH and without hh and verification.
- Changes within HH routine:
    - Call setupCommPackage for A and B.
    - Create new tasks commInit and commFinalize for the matrices A and B.
    - Create new states and state managers corresponding to matrices A and B.
    - Create appropriate edges for new comm tasks and state managers.
    - Update row and column traversal tasks to initiate comm for every block.
    - Reset ttls in input state once they reach 0, check comments for more details.
    - Add "q" as an extra parameter for statePartialComputation and stateOutput constructors. Check respecitve files for usage.
    

