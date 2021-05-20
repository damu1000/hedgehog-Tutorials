// NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the
// software in any medium, provided that you keep intact this entire notice. You may improve, modify and create
// derivative works of the software or any portion of the software, and you may copy and distribute such modifications
// or works. Modified works should carry a notice stating that you changed the software and should note the date and
// nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the
// source of the software. NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND,
// EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR
// WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE
// CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS
// THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE. You
// are solely responsible for determining the appropriateness of using and distributing the software and you assume
// all risks associated with its use, including but not limited to the risks and costs of program errors, compliance
// with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of 
// operation. This software is not intended to be used in any situation where a failure could cause risk of injury or
// damage to property. The software developed by NIST employees is not subject to copyright protection within the
// United States.


#include <unistd.h>
#include <hedgehog/hedgehog.h>
#include <random>
#include "comm/comm.h"
#include "comm/comm_tasks.h"

#include "../utils/tclap/CmdLine.h"

#include "data/matrix_data.h"
#include "data/matrix_block_data.h"

#include "task/addition_task.h"
#include "task/matrix_row_traversal_task.h"
#include "task/matrix_column_traversal_task.h"

#include "cuda_tasks/cuda_copy_in_gpu.h"
#include "cuda_tasks/cuda_copy_out_gpu.h"
#include "cuda_tasks/cuda_product_task.h"

#include "state/cannon_state.h"
#include "state/cannon_state_manager.h"
#include "state/output_state.h"
#include "state/cuda_input_block_state.h"
#include "state/partial_computation_state.h"
#include "state/partial_computation_state_manager.h"


using MatrixType = double;
constexpr Order Ord = Order::Column;


class SizeConstraint : public TCLAP::Constraint<size_t> {
public:
	[[nodiscard]] std::string description() const override {
		return "Positive non null";
	}
	[[nodiscard]] std::string shortID() const override {
		return "NonNullInteger";
	}
	[[nodiscard]] bool check(size_t const &value) const override {
		return value > 0;
	}
};


//------------------------------------------- initialize the matrix -----------------------------------------------

MatrixType * initMatrix(size_t n) // n should be rows*cols
{
	// Mersenne Twister Random Generator
	uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
	std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
	std::mt19937_64 rng(ss);
	std::uniform_real_distribution<MatrixType> unif(0, 10); //choose real or int
	//std::uniform_int_distribution<MatrixType> unif(0, 10);

	MatrixType *data = new MatrixType[n]();
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for (size_t i = 0; i < n; ++i){
		//data[i] = unif(rng);
		data[i] = rank*n + i;
		//		data[i] = rank + 1;
		//data[i] = 1;
	}

	return data;
}


//------------------------------------------- verify with hh and without hh -----------------------------------------------

#define RED   "\x1B[31m"
#define RESET "\x1B[0m"

void inline verifyProduct(double* correct, double *hh, int n, int rank)
{
	//	for(int i=0; i<n; i++){
	//		for(int j=0; j<n; j++){
	//			if(fabs(hh[i*n + j] - correct[i*n + j]) > 1e-9 ){
	//				printf("%d ########## Output does not match. Error in %s %d: at cell (%d, %d) -> correct: %f \t computed:%f \t diff: %f\n"
	//						,rank,  __FILE__, __LINE__, i, j, correct[i*n + j], hh[i*n + j], fabs(hh[i*n + j] - correct[i*n + j]));
	//				exit(1);
	//			}
	//		}
	//	}
	//	printf("output matched for rank %d!! C:\n", rank);

	sleep(rank);
	printf("output of rank: %d\n", rank);
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			const char *color = (fabs(hh[j*n + i] - correct[j*n + i]) < 1e-9 ) ? RESET : RED; //red on error
			printf("%.1f|%s%.1f\t%s", correct[j*n + i], color, hh[j*n + i], RESET);
		}
		printf("\n");
	}
}



//----------------------------- Cannon's Algorithm. without HH. Used for verification -----------------------------

void matMult(double *A, double *B, double *C, int n, int q)
{
	CommPackage<double, 'a'> cpackA; //create and init comm packages for matrices A and B.
	CommPackage<double, 'b'> cpackB;
	cpackA.setupCommPackage(A, n, n, 0, 0, 1); //patch is not divided into blocks. So the blockSize = leading dimension = n and rowId=colId =0
	cpackB.setupCommPackage(B, n, n, 0, 0, 1);

	for(int k=0; k<q; k++)	//main loop of Cannon's Algorithm. running it from 0 to q rather than 1 to q, otherwise we miss multiplication of 1 block.
	{
		cpackA.initializeComm(); //init comm for this iteration
		cpackB.initializeComm();

//#pragma omp parallel for
		for(int i = 0; i<n; i++)
			for(int j = 0; j<n; j++)
				for(int l = 0; l<n; l++)
					C[j*n+i] += A[l*n+i]*B[j*n+l]; //TODO: verify indexing based on the row order or column order

		cpackA.finalizeComm(); //finalize comm
		cpackB.finalizeComm();
	}

	cpackA.destroyComm();
	cpackB.destroyComm();
}




//----------------------------------------- Cannon's Algorithm. with HH. -----------------------------------------

void matMultHH(size_t n, int q, size_t blockSize, size_t numberThreadProduct, size_t numberThreadAddition, MatrixType *dataA, MatrixType *dataB, MatrixType * dataC)
{
	size_t m = n, p = n;

	//Exchange initial patches as per cannon's algo. setupCommPackage with align=1 for entire patch will do it. Destroy comm later. Now each block uses its own comm.
	//Doing it per block causes race condition because blocks with alignment completed jump to next product task.
	CommPackage<MatrixType, 'a'> commA;
	CommPackage<MatrixType, 'b'> commB;
	commA.setupCommPackage(dataA, n, n, 0, 0, 1);
	commB.setupCommPackage(dataB, n, n, 0, 0, 1);
	commA.destroyComm();
	commB.destroyComm();

	// Wrap them to convenient object representing the matrices
	auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(n, m, blockSize, dataA);
	auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(m, p, blockSize, dataB);
	auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(n, p, blockSize, dataC);

	size_t nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1);
	size_t mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1);
	size_t pBlocks = std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1);

	// Graph
	auto matrixMultiplicationGraph =
			hh::Graph<MatrixBlockData<MatrixType, 'c', Ord>,
			MatrixData<MatrixType, 'a', Ord>, MatrixData<MatrixType, 'b', Ord>, MatrixData<MatrixType, 'c', Ord>>
			("Matrix Multiplication Graph");


	// Host Tasks
	auto taskTraversalA = std::make_shared<MatrixColumnTraversalTask<MatrixType, 'a', Order::Column>>();
	auto taskTraversalB = std::make_shared<MatrixRowTraversalTask<MatrixType, 'b', Order::Column>>();
	auto taskTraversalC = std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Order::Column>>();
	auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

	// comm tasks
	auto taskCommInitA = std::make_shared<commInitTask<MatrixType, 'a', Ord>>();
	auto taskCommInitB = std::make_shared<commInitTask<MatrixType, 'b', Ord>>();
	auto taskCommFinalizeA = std::make_shared<commFinalizeTask<MatrixType, 'a', Ord>>(nBlocks, mBlocks, nBlocks * mBlocks * pBlocks);
	auto taskCommFinalizeB = std::make_shared<commFinalizeTask<MatrixType, 'b', Ord>>(mBlocks, pBlocks, nBlocks * mBlocks * pBlocks);

	// Cuda tasks
	// Tasks
	auto copyInATask = std::make_shared<CudaCopyInGpu<MatrixType, 'a'>>(pBlocks, blockSize, n);
	auto copyInBTask = std::make_shared<CudaCopyInGpu<MatrixType, 'b'>>(nBlocks, blockSize, m);
	auto productTask = std::make_shared<CudaProductTask<MatrixType>>(p, numberThreadProduct);
	auto copyOutTask = std::make_shared<CudaCopyOutGpu<MatrixType>>(blockSize);

	// MemoryManagers
	auto cudaMemoryManagerA = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'a'>, size_t>>(nBlocks + 4, blockSize);
	auto cudaMemoryManagerB = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'b'>, size_t>>(pBlocks + 4, blockSize);
	auto cudaMemoryManagerProduct = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'p'>, size_t>>(8, blockSize);

	// Connect the memory manager
	productTask->connectMemoryManager(cudaMemoryManagerProduct);
	copyInATask->connectMemoryManager(cudaMemoryManagerA);
	copyInBTask->connectMemoryManager(cudaMemoryManagerB);

	// State
	auto stateInputBlock = std::make_shared<CudaInputBlockState<MatrixType>>(nBlocks, mBlocks, pBlocks);
	auto statePartialComputation = std::make_shared<PartialComputationState<MatrixType, Ord>>(nBlocks, pBlocks, nBlocks * mBlocks * pBlocks, q);
	auto stateOutput = std::make_shared<OutputState<MatrixType, Ord>>(nBlocks, pBlocks, mBlocks, q);

	//comm states
	auto stateCannonA = std::make_shared<CannonState<MatrixType, 'a', Ord>>(nBlocks*mBlocks, q); //output state to carry out loops of Cannon's algorithm
	auto stateCannonB = std::make_shared<CannonState<MatrixType, 'b', Ord>>(mBlocks*pBlocks, q);

	// StateManager
	auto stateManagerInputBlock =
			std::make_shared<hh::StateManager<
			std::pair<
			std::shared_ptr<CudaMatrixBlockData<MatrixType, 'a'>>,
			std::shared_ptr<CudaMatrixBlockData<MatrixType, 'b'>>>,
			CudaMatrixBlockData<MatrixType, 'a'>, CudaMatrixBlockData<MatrixType, 'b'>>
			>("Input State Manager", stateInputBlock);

	auto stateManagerPartialComputation = std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(statePartialComputation);

	auto stateManagerOutputBlock =
			std::make_shared<hh::StateManager<
			MatrixBlockData<MatrixType, 'c', Ord>,
			MatrixBlockData<MatrixType, 'c', Ord>>>("Output State Manager", stateOutput);

	//output state manager to carry out loops of Cannon's algorithm
	auto stateManagerCannonA = std::make_shared<CannonStateManager<MatrixType, 'a', Ord>>(stateCannonA);
	auto stateManagerCannonB = std::make_shared<CannonStateManager<MatrixType, 'b', Ord>>(stateCannonB);

	// Build the graph
	matrixMultiplicationGraph.input(taskTraversalA);
	matrixMultiplicationGraph.input(taskTraversalB);
	matrixMultiplicationGraph.input(taskTraversalC);

	matrixMultiplicationGraph.addEdge(taskTraversalA, taskCommInitA);          //1. pass A / B to init. init will call MPI send-recv for these blocks
	matrixMultiplicationGraph.addEdge(taskTraversalB, taskCommInitB);

	// Copy the blocks to the device (NVIDIA GPU)
	matrixMultiplicationGraph.addEdge(taskCommInitA, copyInATask);             //2. pass on the updated blocks to gpu for compute
	matrixMultiplicationGraph.addEdge(taskCommInitB, copyInBTask);

	matrixMultiplicationGraph.addEdge(taskCommInitA, taskCommFinalizeA);       //3. Pass on A/B blocks to finalize comm. Will be updated after finalize
	matrixMultiplicationGraph.addEdge(taskCommInitB, taskCommFinalizeB);

	// Connect to the State manager to wait for compatible block of A and B
	matrixMultiplicationGraph.addEdge(copyInATask, stateManagerInputBlock);
	matrixMultiplicationGraph.addEdge(copyInBTask, stateManagerInputBlock);

	// Do the CUDA product task
	matrixMultiplicationGraph.addEdge(stateManagerInputBlock, productTask);

	// Copy out the temporary block to the CPU for accumulation after the product
	matrixMultiplicationGraph.addEdge(productTask, copyOutTask);
	matrixMultiplicationGraph.addEdge(copyOutTask, stateManagerPartialComputation);

	// Use the same graph for the accumulation
	matrixMultiplicationGraph.addEdge(taskTraversalC, stateManagerPartialComputation);
	matrixMultiplicationGraph.addEdge(stateManagerPartialComputation, additionTask);
	matrixMultiplicationGraph.addEdge(additionTask, stateManagerPartialComputation);
	matrixMultiplicationGraph.addEdge(additionTask, stateManagerOutputBlock);

	matrixMultiplicationGraph.addEdge(additionTask, taskCommFinalizeA);        //4. Pass on matmult result to finalize. Actual finalization happens only when ALL blocks are computed. Avoids premature update of A/B.
	matrixMultiplicationGraph.addEdge(additionTask, taskCommFinalizeB);

	matrixMultiplicationGraph.addEdge(taskCommFinalizeA, stateManagerCannonA); //5. Cannon state checks if q iterations are over.
	matrixMultiplicationGraph.addEdge(taskCommFinalizeB, stateManagerCannonB);
	matrixMultiplicationGraph.addEdge(stateManagerCannonA, taskCommInitA);     //6. Push updated A/B blocks to init (step 1.)
	matrixMultiplicationGraph.addEdge(stateManagerCannonB, taskCommInitB);

	matrixMultiplicationGraph.output(stateManagerOutputBlock);

	// Execute the graph
	matrixMultiplicationGraph.executeGraph();

	// Push the matrices
	matrixMultiplicationGraph.pushData(matrixA);
	matrixMultiplicationGraph.pushData(matrixB);
	matrixMultiplicationGraph.pushData(matrixC);

	// Notify push done
	matrixMultiplicationGraph.finishPushingData();

	// Wait for the graph to terminate
	matrixMultiplicationGraph.waitForTermination();

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	matrixMultiplicationGraph.createDotFile(std::string("graph_"  + std::to_string(rank) + ".dot"), hh::ColorScheme::EXECUTION,
			hh::StructureOptions::NONE);

}




int main(int argc, char **argv)
{
	//--------------------------------- Init MPI -----------------------------------------------
	int rank, size, provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if(provided < MPI_THREAD_MULTIPLE){
		printf("MPI_THREAD_MULTIPLE not available. aborting\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	//--------------------------------- Parse int put args -----------------------------------------------

	size_t n = 0, m = 0, p = 0, blockSize = 0, numberThreadProduct = 0, numberThreadAddition = 0,  verify=0;

	try {
		TCLAP::CmdLine cmd("Matrix Multiplication parameters", ' ', "0.1");
		SizeConstraint sc;
		TCLAP::ValueArg<size_t> nArg("n", "aheight", "Matrix A Height.", false, 10, &sc);
		cmd.add(nArg);
		TCLAP::ValueArg<size_t> blockArg("b", "blocksize", "Block Size.", false, 0, &sc);
		cmd.add(blockArg);
		TCLAP::ValueArg<size_t> productArg("x", "product", "Product task's number of threads.", false, 3, &sc);
		cmd.add(productArg);
		TCLAP::ValueArg<size_t> additionArg("a", "addition", "Addition task's number of threads.", false, 3, &sc);
		cmd.add(additionArg);
	    TCLAP::ValueArg<size_t> verifyArg("v", "verify", "Verify the output", false, 0, &sc);
	    cmd.add(verifyArg);

		cmd.parse(argc, argv);

		n = nArg.getValue();
		numberThreadAddition = additionArg.getValue();
		numberThreadProduct = productArg.getValue();
		blockSize = blockArg.getValue();
		verify = verifyArg.getValue();
	} catch (TCLAP::ArgException &e)  // catch any exceptions
	{ std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; }



	//--------------------------------- error check for input arguments ---------------------------------

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int q = (int)sqrt(size);
	int N = n; //N now holds the full matrix size.

	if(q*q < size)	{
		if(rank == 0) std::cout << "number of processes should be perfect square\n";
		return(0);
	}

	if(N%q != 0){
		if(rank == 0) std::cout << "n (number of rows in matrix) should be divisible by sqrt(p)\n";
		return(0);
	}

	n = N / q; // non n holds size of the per rank matrix patch.
	m = n; //assuming square matrix. hence the same dimension.
	p = n;
	if (blockSize == 0)
		blockSize = n;  //setting a single block per rank, if blockSize is 0

	if(n%blockSize != 0){
		if(rank == 0) std::cout << "n (number of rows per rank (N/sqrt(#ranks)) should be divisible by blockSize(b)\n";
		return(0);
	}


	//--------------------------------- Create a common MPI comm -----------------------------------------------
	createCartComm(q);

	//--------------------------------- Allocate and init the matrices -----------------------------------------------
	MatrixType *dataA = initMatrix(n * m);
	MatrixType *dataB = initMatrix(m * p);
	MatrixType *dataC = initMatrix(n * p);



	  //--------------------------------- distributed DGEMM -----------------------------------------------
	  if(verify){
		  memset(dataC, 0, n*n*sizeof(MatrixType)); //reset C to verify.
		  //printf("running HH:\n");
		  matMultHH(n, q, blockSize, numberThreadProduct, numberThreadAddition, dataA, dataB, dataC);

		  MatrixType *C = initMatrix(n * p);
		  MatrixType *A = initMatrix(n * m);
		  MatrixType *B = initMatrix(m * p);
		  memset(C, 0, n*n*sizeof(MatrixType));
		  //printf("running no HH:\n");
		  matMult(A, B, C, n, q);
		  verifyProduct(C, dataC, n, rank);

		  delete[] A;
		  delete[] B;
		  delete[] C;

	  }
	  else{//measure performance
		  double seconds = 0, iterations = 20.0;
		  struct timeval  tv1, tv2;

		  for(int i=0; i<iterations; i++){
			  gettimeofday(&tv1, NULL);

			  //call matrix multiply
			  matMultHH(n, q, blockSize, numberThreadProduct, numberThreadAddition, dataA, dataB, dataC);

			  gettimeofday(&tv2, NULL);
			  seconds = seconds + (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		  }
		  double perProc[3] = {g_setupTime, g_commTime, seconds}, averageTime[3]={0, 0, 0}; //0th is setup time, 1st is comm time, 2nd is total time

		  MPI_Reduce(perProc, averageTime, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		  if(rank==0)
			  printf(" Setup time: %f\n Comm time: %f seconds\n Comp time: %f seconds\n Total time: %f seconds\n",
					  averageTime[0]/(double)size/iterations, averageTime[1]/(double)size/iterations, (averageTime[2]-averageTime[0]-averageTime[1])/(double)size/iterations, averageTime[2]/(double)size/iterations  );
	  }



	  //--------------------------------- Deallocate the Matrices ---------------------------------
	delete[] dataA;
	delete[] dataB;
	delete[] dataC;

	MPI_Finalize();

	cublasShutdown();
	return 0;
}
