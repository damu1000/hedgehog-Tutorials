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

#include "cuda_tasks/cuda_copy_in_gpu.h"
#include "cuda_tasks/cuda_copy_out_gpu.h"
#include "cuda_tasks/cuda_product_task.h"
#include "cuda_tasks/cuda_addition_task.h"

#include "state/cannon_state.h"
#include "state/cannon_state_manager.h"
#include "state/cuda_input_block_state.h"

using MatrixType = double;
constexpr Order Ord = Order::Column;

cudaStream_t *cudaStreams::streams = nullptr;
cublasHandle_t *cudaStreams::cublas_handles = nullptr;
size_t cudaStreams::n = 0;
size_t cudaStreams::p = 0;


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



double computeMatrixMultiplicationGFLOPS(size_t n, size_t m, size_t p, double duration) {
  return ((double) n * (double) m * (double) p * 2. * 1.0e-9) / duration;
}


//------------------------------------------- initialize the matrix -----------------------------------------------

// Mersenne Twister Random Generator
uint64_t timeSeed = std::chrono::system_clock::now().time_since_epoch().count();
std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> (uint64_t) 32)};
std::mt19937_64 rng(ss);
std::uniform_real_distribution<MatrixType> unif(0, 10); //choose real or int
//std::uniform_int_distribution<MatrixType> unif(0, 10);

MatrixType * initMatrix(size_t block_r, size_t block_c, size_t cellPerRank, size_t m, size_t blockSize, size_t verify=0) // n should be rows*cols
{
	size_t n = blockSize * blockSize;
	MatrixType *data = new MatrixType[n]();

	if(verify){
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//		printf("block %lu %lu:\n", block_r, block_c);
		for (size_t i = 0; i < blockSize; ++i){
			for (size_t j = 0; j < blockSize; ++j){
				//   id in cur block,     count prev ranks,        count rows from prev blcoks,       count col from prev blocks in this rank
				data[i * blockSize + j] = rank*cellPerRank    +    (block_r * blockSize + i)*m    +   (block_c * blockSize + j);
//				printf("%.2f\t", data[i * blockSize + j]);
			}
//			printf("\n");
		}
	}
	else{
		for (size_t i = 0; i < n; ++i)
			data[i] = unif(rng);
	}

	return data;
}

//n is height and m is width
template<class Type, char Id, typename MBD = MatrixBlockData<Type, Id, Order::Column>>
std::shared_ptr<std::vector<std::shared_ptr<MBD>>> initMatrixBlocks(size_t n, size_t m, size_t blockSize, size_t verify) //may need to use different blockSize for m and n for irregular matrices
{
	size_t  nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1);
	size_t  mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1);
	size_t blockSizeRemainderHeight = n % blockSize;
	size_t blockSizeRemainderWidth = m % blockSize;

	std::vector<std::shared_ptr<MBD>> matblocks;
//	printf("-------- matrix %c -----------------\n", Id);

	for (size_t i = 0; i < nBlocks; ++i) {
		size_t blockSizeHeight = blockSize;
		if (i == nBlocks-1 && blockSizeRemainderHeight > 0) {
			blockSizeHeight = blockSizeRemainderHeight;
		}
		for (size_t j = 0; j < mBlocks; ++j) {
			size_t blockSizeWidth = blockSize;
			if (j == mBlocks-1 &&  blockSizeRemainderWidth > 0) {
				blockSizeWidth = blockSizeRemainderWidth;
			}

			MatrixType *blockData = initMatrix(i, j, m*n, m, blockSize, verify);
			MatrixType *dupBlockData = nullptr;
			if(Id=='a' || Id=='b'){
				dupBlockData = new MatrixType[blockSize*blockSize]; //make a copy of original block data. Avoids 1 comm at the end to restore data
				memcpy(dupBlockData, blockData, blockSize*blockSize*sizeof(MatrixType));
			}
			//Note i and j are interchanged because of col major order
			auto block = std::make_shared<MBD>(j, i, blockSizeHeight, blockSizeWidth, blockSizeHeight, blockData, blockData, dupBlockData);
			matblocks.push_back(block);
		}
	}
	return std::make_shared<std::vector<std::shared_ptr<MBD>>>(matblocks);
}
//------------------------------------------- verify with hh and without hh -----------------------------------------------

#define RED   "\x1B[31m"
#define RESET "\x1B[0m"

void inline verifyProduct(double* correct, std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>> matC, size_t n, size_t m, size_t blockSize)
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
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	size_t  nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1);
	size_t  mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1);
//	size_t blockSizeRemainderWidth = m % blockSize;
//	size_t blockSizeRemainderHeight = n % blockSize;

	sleep(rank);
	printf("----- output of rank: %d -----\n", rank);
	size_t blockId=0;
	for (size_t c = 0; c < mBlocks; ++c) {
//		size_t blockSizeWidth = blockSize;
//		if (c == mBlocks-1 &&  blockSizeRemainderWidth > 0) {
//			blockSizeWidth = blockSizeRemainderWidth;
//		}
		for (size_t r = 0; r < nBlocks; ++r) {
//			size_t blockSizeHeight = blockSize;
//			if (r == nBlocks-1 && blockSizeRemainderHeight > 0) {
//				blockSizeHeight = blockSizeRemainderHeight;
//			}
			printf("block %lu %lu\n", r, c);
			double *hh = matC[blockId++]->blockData();
			for(size_t i=0; i<blockSize; i++){
				for(size_t j=0; j<blockSize; j++){
					MatrixType hhval = hh[j*blockSize + i];
					MatrixType correctval = correct[(c * blockSize + j)*m + (r * blockSize + i)];

					const char *color = (fabs(hhval - correctval) < 1e-9 ) ? RESET : RED; //red on error
					printf("%.1f|%s%.1f\t%s", correctval, color, hhval, RESET);
				}
				printf("\n");
			}
		}

	}
}



//----------------------------- Cannon's Algorithm. without HH. Used for verification -----------------------------

void matMult(double *A, double *B, double *C, int n, int q)
{
	CommPackage<double, 'a'> cpackA; //create and init comm packages for matrices A and B.
	CommPackage<double, 'b'> cpackB;
	cpackA.setupCommPackage(1, 1, A, n, n, 0, 0, 1); //patch is not divided into blocks. So the blockSize = leading dimension = n and rowId=colId =0
	cpackB.setupCommPackage(1, 1, B, n, n, 0, 0, 1);
	cpackA.finalizeAlign();
	cpackB.finalizeAlign();

	for(int k=0; k<q; k++)	//main loop of Cannon's Algorithm. running it from 0 to q rather than 1 to q, otherwise we miss multiplication of 1 block.
	{
		cpackA.initializeComm(); //init comm for this iteration
		cpackB.initializeComm();

//#pragma omp parallel for
		for(int i = 0; i<n; i++)
			for(int j = 0; j<n; j++)
				for(int l = 0; l<n; l++)
					C[j*n+i] += A[l*n+i]*B[j*n+l]; //TODO: verify indexing based on the row order or column order

		cpackA.waitForComm();
		cpackB.waitForComm();
		cpackA.finalizeComm(); //finalize comm
		cpackB.finalizeComm();
	}

	cpackA.destroyComm();
	cpackB.destroyComm();
}



//----------------------------------------- Cannon's Algorithm. with HH. -----------------------------------------

MatrixType *devC = nullptr;


std::shared_ptr<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>>>
asyncCopyInC(std::shared_ptr<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>>> &matC, size_t blockSize){
	std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>> cudMatC;
	auto C = (*matC);

	size_t size = C.size() * blockSize * blockSize * sizeof(MatrixType);
	checkCudaErrors(cudaMalloc(&devC, size));

	for(size_t i=0; i<C.size(); i++){
		MatrixBlockData<MatrixType, 'c', Order::Column> x(*(C[i])); //copy the object, replace the pointer with device pointer
		x.blockData(devC + i*blockSize*blockSize);
		x.fullMatrixData(nullptr);
		auto stream = cudaStreams::getStream(x.rowIdx(), x.colIdx()); //use the stream corresponding to C's block. Same stream will be used for addition and copyout. avoids explicit synchronization
		checkCudaErrors(cudaMemcpyAsync(x.blockData(), C[i]->blockData(), blockSize * blockSize * sizeof(MatrixType), cudaMemcpyHostToDevice, stream));

		cudMatC.push_back(std::make_shared<MatrixBlockData<MatrixType, 'c', Order::Column>>(x));
	}
	return std::make_shared<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>>>(cudMatC);
}

void destroyCudaC(){
	checkCudaErrors(cudaFree(devC));
}

void matMultHH(size_t n, int q, size_t blockSize, size_t numberThreadProduct, size_t numberThreadAddition,
		std::shared_ptr<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'a', Order::Column>>>> &matA,
		std::shared_ptr<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'b', Order::Column>>>> &matB,
		std::shared_ptr<std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>>> &matC
		)
{
	size_t m = n, p = n;
	size_t nBlocks = std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1);
	size_t mBlocks = std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1);
	size_t pBlocks = std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1);

	cudaStreams::createStreams(nBlocks, pBlocks);
	auto dMatC = asyncCopyInC(matC, blockSize); //initiate one time h2d copy for c using respective stream

	struct timeval  tv1, tv2;

	gettimeofday(&tv1, NULL);
	//setup Cannon's algo. Initial alignment
	for(auto &mb: *matA)
		mb->setupCommPackage(nBlocks, mBlocks, 1);
	for(auto &mb: *matB)
		mb->setupCommPackage(nBlocks, mBlocks, 1);

	for(auto &mb: *matA)
		mb->finalizeSetupComm(1);
	for(auto &mb: *matB)
		mb->finalizeSetupComm(1);

	gettimeofday(&tv2, NULL);
	g_setupTime = g_setupTime + (double) (tv2.tv_usec - tv1.tv_usec) / 1000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);

	// Graph
	auto matrixMultiplicationGraph =
			hh::Graph<MatrixBlockData<MatrixType, 'c', Ord>, //output of graph
			std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'a', Order::Column>>>, //3 inputs of the graph
			std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'b', Order::Column>>>,
			std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'c', Order::Column>>> >
			("Matrix Multiplication Graph");

	// Host Tasks
	auto taskCommSetupA = std::make_shared<commSetupTask<MatrixType, 'a', Ord>>(nBlocks, mBlocks, 1);
	auto taskCommSetupB = std::make_shared<commSetupTask<MatrixType, 'b', Ord>>(mBlocks, pBlocks, 1);

	// comm tasks
	auto taskCommInitA = std::make_shared<commInitTask<MatrixType, 'a', Ord>>(nBlocks * mBlocks);
	auto taskCommInitB = std::make_shared<commInitTask<MatrixType, 'b', Ord>>(mBlocks * pBlocks);
	auto taskCommFinalizeA = std::make_shared<commFinalizeTask<MatrixType, 'a', Ord>>(nBlocks, mBlocks);
	auto taskCommFinalizeB = std::make_shared<commFinalizeTask<MatrixType, 'b', Ord>>(mBlocks, pBlocks);

	// Cuda tasks
	// Tasks
	auto copyInATask = std::make_shared<CudaCopyInGpu<MatrixType, 'a'>>(pBlocks, blockSize, n, nBlocks * mBlocks);
	auto copyInBTask = std::make_shared<CudaCopyInGpu<MatrixType, 'b'>>(nBlocks, blockSize, m, mBlocks * pBlocks);
	auto productTask = std::make_shared<CudaProductTask<MatrixType>>(p, numberThreadProduct);
	auto additionTask = std::make_shared<CudaAdditionTask<MatrixType>>(nBlocks, mBlocks, pBlocks, q, dMatC ); //pass device block array
	auto copyOutTask = std::make_shared<CudaCopyOutGpuC<MatrixType>>(nBlocks, mBlocks, pBlocks, matC); //pass host block array

	// MemoryManagers
	auto cudaMemoryManagerA = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'a'>, size_t>>(nBlocks + 32, blockSize);
	auto cudaMemoryManagerB = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'b'>, size_t>>(pBlocks + 32, blockSize);
	auto cudaMemoryManagerProduct = std::make_shared<hh::StaticMemoryManager<CudaMatrixBlockData<MatrixType, 'p'>, size_t>>(32, blockSize);

	// Connect the memory manager
	productTask->connectMemoryManager(cudaMemoryManagerProduct);
	copyInATask->connectMemoryManager(cudaMemoryManagerA);
	copyInBTask->connectMemoryManager(cudaMemoryManagerB);

	// State
	auto stateInputBlock = std::make_shared<CudaInputBlockState<MatrixType>>(nBlocks, mBlocks, pBlocks);

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

	//output state manager to carry out loops of Cannon's algorithm
	auto stateManagerCannonA = std::make_shared<CannonStateManager<MatrixType, 'a', Ord>>(stateCannonA);
	auto stateManagerCannonB = std::make_shared<CannonStateManager<MatrixType, 'b', Ord>>(stateCannonB);

	// Build the graph
	matrixMultiplicationGraph.input(taskCommSetupA);
	matrixMultiplicationGraph.input(taskCommSetupB);

	matrixMultiplicationGraph.addEdge(taskCommSetupA, taskCommInitA);          //1. pass A / B to init. init will call MPI send-recv for these blocks
	matrixMultiplicationGraph.addEdge(taskCommSetupB, taskCommInitB);

	// Copy the blocks to the device (NVIDIA GPU)
	matrixMultiplicationGraph.addEdge(taskCommInitA, copyInATask);             //2. pass on the updated blocks to gpu for compute
	matrixMultiplicationGraph.addEdge(taskCommInitB, copyInBTask);

	matrixMultiplicationGraph.addEdge(taskCommInitA, taskCommFinalizeA);       //3. Pass on A/B blocks to finalize comm. Will be updated after finalize
	matrixMultiplicationGraph.addEdge(taskCommInitB, taskCommFinalizeB);

	// Connect to the State manager to wait for compatible block of A and B
	matrixMultiplicationGraph.addEdge(copyInATask, stateManagerInputBlock);
	matrixMultiplicationGraph.addEdge(copyInBTask, stateManagerInputBlock);

	matrixMultiplicationGraph.addEdge(copyInATask, taskCommFinalizeA);        //4. Signal finalize. Actual finalization happens only when ALL blocks are copied into GPU. Avoids premature update of A/B.
	matrixMultiplicationGraph.addEdge(copyInBTask, taskCommFinalizeA);
	matrixMultiplicationGraph.addEdge(copyInATask, taskCommFinalizeB);
	matrixMultiplicationGraph.addEdge(copyInBTask, taskCommFinalizeB);

	// Do the CUDA product -> addition -> copyout
	matrixMultiplicationGraph.addEdge(stateManagerInputBlock, productTask);
	matrixMultiplicationGraph.addEdge(productTask, additionTask);
	matrixMultiplicationGraph.addEdge(additionTask, copyOutTask);

	matrixMultiplicationGraph.addEdge(taskCommFinalizeA, stateManagerCannonA); //5. Cannon state checks if q iterations are over.
	matrixMultiplicationGraph.addEdge(taskCommFinalizeB, stateManagerCannonB);
	matrixMultiplicationGraph.addEdge(stateManagerCannonA, taskCommInitA);     //6. Push updated A/B blocks to init (step 1.)
	matrixMultiplicationGraph.addEdge(stateManagerCannonB, taskCommInitB);

//	matrixMultiplicationGraph.output(stateManagerOutputBlock);
	matrixMultiplicationGraph.output(copyOutTask);

	// Execute the graph
	matrixMultiplicationGraph.executeGraph();

	// Push the matrices
	matrixMultiplicationGraph.pushData(matA);
	matrixMultiplicationGraph.pushData(matB);

	// Notify push done
	matrixMultiplicationGraph.finishPushingData();

	// Wait for the graph to terminate
	matrixMultiplicationGraph.waitForTermination();
	checkCudaErrors(cudaDeviceSynchronize());
	
	//release comm buffers
	gettimeofday(&tv1, NULL);
	for(auto &blockA: *matA) blockA->destroyComm(); //is it needed? Will shared pointer automatically deallocate???
	for(auto &blockB: *matB) blockB->destroyComm();
	destroyCudaC();
	gettimeofday(&tv2, NULL);
	g_setupTime = g_setupTime + (double) (tv2.tv_usec - tv1.tv_usec) / 1000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);

/*
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	matrixMultiplicationGraph.createDotFile(std::string("graph_"  + std::to_string(rank) + ".dot"), hh::ColorScheme::NONE, hh::StructureOptions::ALL);
*/
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

	auto matA = initMatrixBlocks<MatrixType, 'a'>(n, m, blockSize, verify);
	auto matB = initMatrixBlocks<MatrixType, 'b'>(m, p, blockSize, verify);
	auto matC = initMatrixBlocks<MatrixType, 'c'>(n, p, blockSize, verify);

	//--------------------------------- distributed DGEMM -----------------------------------------------
	if(verify){
		for(auto &blockC: *matC)
			memset(blockC->blockData(), 0, blockSize*blockSize*sizeof(MatrixType)); //reset C to verify.
		matMultHH(n, q, blockSize, numberThreadProduct, numberThreadAddition, matA, matB, matC);

		MatrixType *C = initMatrix(0, 0, m*n, n, n, verify); //using simple 1 block for manual matmult
		MatrixType *A = initMatrix(0, 0, n*p, n, n, verify);
		MatrixType *B = initMatrix(0, 0, m*p, n, n, verify);
		memset(C, 0, n*n*sizeof(MatrixType));
		matMult(A, B, C, n, q);
		verifyProduct(C, *matC, p, n, blockSize);

		delete[] A;
		delete[] B;
		delete[] C;

	}
	else{//measure performance
		double seconds = 0, iterations = 2.0;
		struct timeval  tv1, tv2;

//		for(int i=0; i<iterations/2; i++){ //warm up
//			call matrix multiply
//			matMultHH(n, q, blockSize, numberThreadProduct, numberThreadAddition, matA, matB, matC);
//		}

		gettimeofday(&tv1, NULL);
		for(int i=0; i<iterations; i++){
			//call matrix multiply
			matMultHH(n, q, blockSize, numberThreadProduct, numberThreadAddition, matA, matB, matC);
		}
		gettimeofday(&tv2, NULL);
		seconds = seconds + (double) (tv2.tv_usec - tv1.tv_usec) / 1000000.0 + (double) (tv2.tv_sec - tv1.tv_sec);
		seconds = seconds/ iterations;
		g_setupTime = g_setupTime / iterations;

		double perProc[3] = {g_setupTime, g_commTime, seconds}, averageTime[3]={0, 0, 0}; //0th is setup time, 1st is comm time, 2nd is total time

		MPI_Reduce(perProc, averageTime, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

		if(rank==0){
			averageTime[0] /= (double) size;
			averageTime[1] /= (double) size;
			averageTime[2] /= (double) size;
									
			printf(" Setup time: %f\n Comm time: %f seconds\n Comp time: %f seconds\n Total time: %f seconds\n",
					averageTime[0], averageTime[1], (averageTime[2]-averageTime[0]-averageTime[1]), averageTime[2]  );

			printf("Comp : GFLOPS / GPU: %f\n", computeMatrixMultiplicationGFLOPS(N, N, N, (averageTime[2]-averageTime[0]-averageTime[1])) / size ); //divide by size to get per GPU time
			printf("Total: GFLOPS / GPU: %f\n", computeMatrixMultiplicationGFLOPS(N, N, N, averageTime[2]) / size ); //divide by size to get per GPU time
		}
	}



	  //--------------------------------- Deallocate the Matrices ---------------------------------
	for(auto &blockA: *matA) blockA->destroy(); //is it needed? Will shared pointer automatically deallocate???
	for(auto &blockB: *matB) blockB->destroy();
	for(auto &blockC: *matC) blockC->destroy();

	MPI_Finalize();

	cublasShutdown();
	return 0;
}
