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

#include "comm.h"
#include <unistd.h>
#include <hedgehog/hedgehog.h>
#include <random>
#include "../utils/tclap/CmdLine.h"

#include "data/matrix_data.h"
#include "data/matrix_block_data.h"

#include "task/addition_task.h"
#include "task/product_task.h"
#include "task/matrix_row_traversal_task.h"
#include "task/matrix_column_traversal_task.h"

#include "state/input_block_state.h"
#include "state/output_state.h"
#include "state/partial_computation_state.h"
#include "state/partial_computation_state_manager.h"


using MatrixType = double;
constexpr Order Ord = Order::Row;


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
		//data[i] = 1;
	}

	return data;
}


void inline verifyProduct(double* correct, double *hh, int n, int rank)
{
	for(int i=0; i<n; i++){
		for(int j=0; j<n; j++){
			if(fabs(hh[i*n + j] - correct[i*n + j]) > 1e-9 ){
				printf("########## Output does not match. Error in %s %d: at cell (%d, %d) -> correct: %f \t computed:%f \t diff: %f\n"
						, __FILE__, __LINE__, i, j, correct[i*n + j], hh[i*n + j], fabs(hh[i*n + j] - correct[i*n + j]));
				exit(1);
			}
		}
	}
	printf("output matched for rank %d!! C:\n", rank);

//	sleep(rank);
//	printf("output of rank: %d\n", rank);
//	for(int i=0; i<n; i++){
//		for(int j=0; j<n; j++)
//			printf("%f\t", correct[i*n+j]);
//		printf("\n");
//	}
}


void matMult(double *A, double *B, double *C, int n, int q) //Cannon's Algorithm. without HH
{
	CommPackage<double, 'a'> cpackA; //create and init comm packages for matrices A and B.
	CommPackage<double, 'b'> cpackB;
	cpackA.setupCommPackage(A, n, n, 0, 0); //patch is not divided into blocks. So the blockSize = leading dimension = n and rowId=colId =0
	cpackB.setupCommPackage(B, n, n, 0, 0);

	for(int k=0; k<q; k++)	//main loop of Cannon's Algorithm. running it from 0 to q rather than 1 to q, otherwise we miss multiplication of 1 block.
	{
		cpackA.initializeComm(); //init comm for this iteration
		cpackB.initializeComm();

#pragma omp parallel for
		for(int i = 0; i<n; i++)
			for(int j = 0; j<n; j++)
				for(int l = 0; l<n; l++)
					C[i*n+j] += A[i*n+l]*B[l*n+j];

		cpackA.finalizeComm(); //finalize comm
		cpackB.finalizeComm();
	}

	cpackA.destroyComm();
	cpackB.destroyComm();
}

//input from traversal task and pass it on to finalize and also product
template<class Type, char Id, Order Ord = Order::Row, class MBD = MatrixBlockData<Type, Id, Ord>>
class commInitTask : public hh::AbstractTask<MBD, MBD> {
public:
	commInitTask() : hh::AbstractTask<MBD, MBD>("commInit_" + Id) {}

	void execute(std::shared_ptr<MBD> matBlock) override {
		printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		matBlock->initializeComm();
		this->addResult(matBlock); //push result
	}

	std::shared_ptr<hh::AbstractTask<MBD, MBD>> copy() override {
		printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		return std::make_shared<commInitTask<Type, Id, Ord>>();
	}
};

//needs A / B from init task. Just save A / B passed on by init.
//Second input is partial matrix p from product Task. Decrement ttl for every input from productTask. Finalize comm when ttl reaches 0
template<class Type, char Id, Order Ord = Order::Row, class MBD = MatrixBlockData<Type, Id, Ord>>
class commFinalizeTask : public hh::AbstractTask<MBD, MBD, MatrixBlockData<Type, 'p', Ord>> {
public:
	//call finalizeComm only when ttl (time to live) reaches 0.
	//ttl for mat A is num of blocks along x dim in mat B (because each block in A will be needed those many times)
	//similary ttl for mat B is num of blocks along y dim in mat A
	int ttl, ttlCopy;
	std::shared_ptr<MBD> matBlock{NULL};
	commFinalizeTask(int _ttl) : hh::AbstractTask<MBD, MBD, MatrixBlockData<Type, 'p', Ord>>("commFinalize_" + Id) {
		ttl = _ttl;
		ttlCopy = ttl;
	}

	void execute(std::shared_ptr<MBD> _matBlock) override {
		printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		matBlock = _matBlock;
	}

	void execute(std::shared_ptr<MatrixBlockData<Type, 'p', Ord>> partialResult) override {//partialResult is unused here, but it ensures product is completed
		printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		ttl--;
		if(ttl==0){
			printf("Executing task: '%s', function '%s' at %s:%d : ttl 0, pushing data\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
			ttl = ttlCopy; //restore for the next Cannon's iteration
			matBlock->finalizeComm();
			this->addResult(matBlock); //push result
		}
	}

	std::shared_ptr<hh::AbstractTask<MBD, MBD, MatrixBlockData<Type, 'p', Ord>>> copy() override {
		printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		return std::make_shared<commFinalizeTask>(this->ttl);
	}
};

template<class Type, char Id, Order Ord = Order::Row, class MBD = MatrixBlockData<Type, Id, Ord>>
class CannonState : public hh::AbstractState<MBD, MBD> {
	int q; //q = sqrt(p);
public:
	CannonState(int _q) : q(_q-1) {}
	bool isDone() {
		printf("Executing task: '%s', function '%s' at %s:%d, q: %d\n", "cannon state", __FUNCTION__,  __FILE__, __LINE__, q ) ;
		return q == 0;
	};

	void execute(std::shared_ptr<MBD> inputparams) override {
		printf("Executing task: '%s', function '%s' at %s:%d, q: %d\n", "cannon state", __FUNCTION__,  __FILE__, __LINE__, q ) ;
		q--;
		this->push(inputparams); //push result
	}
};

template<class Type, char Id, Order Ord = Order::Row, class MBD = MatrixBlockData<Type, Id, Ord>>
class CannonStateManager : public hh::StateManager<MBD, MBD> {
public:
	explicit CannonStateManager(std::shared_ptr<CannonState<Type, Id, Ord>> const &state) :
	hh::StateManager<MBD, MBD>("output State Manager_" + Id, state, false) {}

	bool canTerminate() override {
		printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		this->state()->lock();
		auto ret = std::dynamic_pointer_cast<CannonState<Type, Id, Ord>>(this->state())->isDone();
		this->state()->unlock();
		return ret;
	}
};





void matMultHH(size_t n, int q, size_t blockSize, size_t numberThreadProduct, size_t numberThreadAddition, MatrixType *dataA, MatrixType *dataB, MatrixType * dataC){

	size_t m = n, p = n;
	// Wrap them to convenient object representing the matrices
	auto matrixA = std::make_shared<MatrixData<MatrixType, 'a', Ord>>(n, m, blockSize, dataA);
	auto matrixB = std::make_shared<MatrixData<MatrixType, 'b', Ord>>(m, p, blockSize, dataB);
	auto matrixC = std::make_shared<MatrixData<MatrixType, 'c', Ord>>(n, p, blockSize, dataC);

	//multiply num of blocks by q to gettotal number of blocks globally
//	size_t nBlocks = (n + blockSize - 1) / blockSize * q;
//	size_t mBlocks = (m + blockSize - 1) / blockSize * q;
//	size_t pBlocks = (p + blockSize - 1) / blockSize * q;

	size_t nBlocks = (std::ceil(n / blockSize) + (n % blockSize == 0 ? 0 : 1)) * q;
	size_t mBlocks = (std::ceil(m / blockSize) + (m % blockSize == 0 ? 0 : 1)) * q;
	size_t pBlocks = (std::ceil(p / blockSize) + (p % blockSize == 0 ? 0 : 1)) * q;

	// Graph
	auto matrixMultiplicationGraph = hh::Graph<MatrixBlockData<MatrixType, 'c', Ord>,
			MatrixData<MatrixType, 'a', Ord>,
			MatrixData<MatrixType, 'b', Ord>,
			MatrixData<MatrixType, 'c', Ord>> ("Matrix Multiplication Graph");

	// Tasks
	auto taskTraversalA = std::make_shared<MatrixRowTraversalTask<MatrixType, 'a', Ord>>();
	auto taskTraversalB = std::make_shared<MatrixColumnTraversalTask<MatrixType, 'b', Ord>>();
	auto taskTraversalC = std::make_shared<MatrixRowTraversalTask<MatrixType, 'c', Ord>>();

	auto taskCommInitA = std::make_shared<commInitTask<MatrixType, 'a', Ord>>();
	auto taskCommInitB = std::make_shared<commInitTask<MatrixType, 'b', Ord>>();

	auto productTask = std::make_shared<ProductTask<MatrixType, Ord>>(numberThreadProduct, p * q);

	auto taskCommFinalizeA = std::make_shared<commFinalizeTask<MatrixType, 'a', Ord>>(nBlocks/q); //ToDO: change nBlocks to correct m / p blocks
	auto taskCommFinalizeB = std::make_shared<commFinalizeTask<MatrixType, 'b', Ord>>(nBlocks/q); //ToDO: change nBlocks to correct m / p blocks

	auto additionTask = std::make_shared<AdditionTask<MatrixType, Ord>>(numberThreadAddition);

	// State
	auto stateInputBlock = std::make_shared<InputBlockState<MatrixType, Ord>>(nBlocks, mBlocks, pBlocks);
	auto statePartialComputation = std::make_shared<PartialComputationState<MatrixType, Ord>>(nBlocks, pBlocks, nBlocks * mBlocks * pBlocks);
	auto stateOutput = std::make_shared<OutputState<MatrixType, Ord>>(nBlocks, pBlocks, mBlocks);
	auto stateCannonA = std::make_shared<CannonState<MatrixType, 'a', Ord>>(q);
	auto stateCannonB = std::make_shared<CannonState<MatrixType, 'b', Ord>>(q);
	//output state and state manager required to carry out loops of Cannon's algorithm

	// StateManager
	typedef std::pair<std::shared_ptr<MatrixBlockData<MatrixType, 'a', Ord>>,
			std::shared_ptr<MatrixBlockData<MatrixType, 'b', Ord>>> stateManagerInputBlockOutput;
	auto stateManagerInputBlock = std::make_shared<hh::StateManager<stateManagerInputBlockOutput, // Pair of block as output
			MatrixBlockData<MatrixType, 'a', Ord>, MatrixBlockData<MatrixType, 'b', Ord>> // Block as Input
			>("Input State Manager", stateInputBlock);

	auto stateManagerPartialComputation = std::make_shared<PartialComputationStateManager<MatrixType, Ord>>(statePartialComputation);

	auto stateManagerCannonA = std::make_shared<CannonStateManager<MatrixType, 'a', Ord>>(stateCannonA);
	auto stateManagerCannonB = std::make_shared<CannonStateManager<MatrixType, 'b', Ord>>(stateCannonB);

	auto stateManagerOutputBlock =
			std::make_shared<hh::StateManager<
			MatrixBlockData<MatrixType, 'c', Ord>,
			MatrixBlockData<MatrixType, 'c', Ord>>>("Output State Manager", stateOutput);

	// Build the graph
	matrixMultiplicationGraph.input(taskTraversalA);
	matrixMultiplicationGraph.input(taskTraversalB);
	matrixMultiplicationGraph.input(taskTraversalC);

	matrixMultiplicationGraph.addEdge(taskTraversalA, taskCommInitA);
	matrixMultiplicationGraph.addEdge(taskTraversalB, taskCommInitB);

	matrixMultiplicationGraph.addEdge(taskCommInitA, stateManagerInputBlock);
	matrixMultiplicationGraph.addEdge(taskCommInitB, stateManagerInputBlock);
	matrixMultiplicationGraph.addEdge(taskCommInitA, taskCommFinalizeA);
	matrixMultiplicationGraph.addEdge(taskCommInitB, taskCommFinalizeB);

	matrixMultiplicationGraph.addEdge(taskTraversalC, stateManagerPartialComputation);

	matrixMultiplicationGraph.addEdge(stateManagerInputBlock, productTask);
	matrixMultiplicationGraph.addEdge(productTask, stateManagerPartialComputation);
	matrixMultiplicationGraph.addEdge(productTask, taskCommFinalizeA);
	matrixMultiplicationGraph.addEdge(productTask, taskCommFinalizeB);

	matrixMultiplicationGraph.addEdge(taskCommFinalizeA, stateManagerCannonA);
	matrixMultiplicationGraph.addEdge(taskCommFinalizeB, stateManagerCannonB);
	matrixMultiplicationGraph.addEdge(stateManagerCannonA, taskCommInitA);
	matrixMultiplicationGraph.addEdge(stateManagerCannonB, taskCommInitB);


	matrixMultiplicationGraph.addEdge(stateManagerPartialComputation, additionTask);
	matrixMultiplicationGraph.addEdge(additionTask, stateManagerPartialComputation);
	matrixMultiplicationGraph.addEdge(additionTask, stateManagerOutputBlock);

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
}

int main(int argc, char **argv) {

  //--------------------------------- Init MPI -----------------------------------------------
  int rank, size, provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if(provided < MPI_THREAD_MULTIPLE){
	  printf("MPI_THREAD_MULTIPLE not available. aborting\n");
	  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  //--------------------------------- Parse int put args -----------------------------------------------

  size_t n = 0, m = 0, p = 0, blockSize = 0, numberThreadProduct = 0, numberThreadAddition = 0, verify=0;

  try {
    TCLAP::CmdLine cmd("Matrix Multiplication parameters", ' ', "0.1");
    SizeConstraint sc;
    TCLAP::ValueArg<size_t> nArg("n", "matrix_size", "matrix size. Assuming a square matrix for now.", false, 4, &sc);
    cmd.add(nArg);
    TCLAP::ValueArg<size_t> blockArg("b", "blocksize", "Block Size.", false, 0, &sc);
    cmd.add(blockArg);
    TCLAP::ValueArg<size_t> productArg("x", "product", "Product task's number of threads.", false, 1, &sc);
    cmd.add(productArg);
    TCLAP::ValueArg<size_t> additionArg("a", "addition", "Addiction task's number of threads.", false, 1, &sc);
    cmd.add(additionArg);
    TCLAP::ValueArg<size_t> verifyArg("v", "verify", "Verify the output", false, 0, &sc);
    cmd.add(verifyArg);

    cmd.parse(argc, argv);

    n = nArg.getValue();
    numberThreadAddition = additionArg.getValue();
    numberThreadProduct = productArg.getValue();
    blockSize = blockArg.getValue();
    verify = verifyArg.getValue();
  } catch (TCLAP::ArgException &e){ // catch any exceptions
	  std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }

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
	  if(rank == 0) std::cout << "n (number of rows in matrix) should be divisible by sqrt(p)";
	  return(0);
  }

  n = N / q; // non n holds size of the per rank matrix patch.
  m = n; //assuming square matrix. hence the same dimension.
  p = n;
  if (blockSize == 0)
     blockSize = n;  //setting a single block per rank, if blockSize is 0

  if(n%blockSize != 0){
	  if(rank == 0) std::cout << "n (number of rows per rank (N/sqrt(#ranks)) should be divisible by blockSize(b)";
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
	  memset(C, 0, n*n*sizeof(MatrixType));
	  //printf("running no HH:\n");
	  matMult(dataA, dataB, C, n, q);
	  verifyProduct(C, dataC, n, rank);
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
				  averageTime[0]/size/iterations, averageTime[1]/size/iterations, (averageTime[2]-averageTime[0]-averageTime[1])/size/iterations, averageTime[2]/size/iterations  );
  }



  //--------------------------------- Deallocate the Matrices ---------------------------------
  delete[] dataA;
  delete[] dataB;
  delete[] dataC;

  MPI_Finalize();
  return 0;
}
