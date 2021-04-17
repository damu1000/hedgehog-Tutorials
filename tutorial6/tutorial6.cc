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

/*

Simple distributed matrix multiplication using Cannon's algorithm. Creates a hedgehog tasks for MPI comm.
Assigns 1 thread / task

Naming convension: p: num of ranks, q: sqrt(p), n: num of rows per rank = num of cols per rank

*/




//#define USE_HH

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <iostream>
#include <math.h>
#include <unistd.h>
#include <cstring>

#include "comm.h"
#include "io.h"

#ifdef USE_HH
#include <hedgehog/hedgehog.h>
#endif


using namespace std;


double** initialize(int n, int inittozero=0) //alocate and init 2d matrix of size nxn. n is the local dimension
{
	double** A = new double*[n];
	A[0] = new double[n*n];
	for(int i = 0; i<n; i++)
		A[i] = A[0] + i*n;

//	int rank;
//	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(inittozero==0){
//#pragma omp parallel for
		for(int i = 0; i<n; i++){
			for(int j = 0; j<n; j++)
				A[i][j] = M_PI*(rand() % 10)*0.01;
				//A[i][j] = rank+1;
				//A[i][j] = 1;
		}
	}
	else
		memset(A[0], 0, n*n*sizeof(double));

	return (A);
}

void free_array(double **A, int n)	//free 2d matrix
{
	delete []A[0];
	delete []A;
}

void matMultKernel(double **A, double **B, double **C, int n){ //main compute kernel for C += A*B
	//mat mult
  //printf("starting mult\n");
//#pragma omp parallel for
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++)
		{
			double sum=0;
			for(int l = 0; l<n; l++)
				sum += A[i][l]*B[l][j];
			C[i][j] += sum;
			//cout << rank << ": a " << sum << " b\t";
		}

}


#ifdef USE_HH

//Cannon's Algorithm. with HH
struct gemmData{
	double ** A, **B, **C;
	int n, q;
	CommPackage *comm;
	gemmData(double **_A, double **_B, double **_C, int _n, int _q, CommPackage* _comm) : A(_A), B(_B), C(_C), n(_n), q(_q), comm(_comm) {}
};

class commInitTask : public hh::AbstractTask<gemmData, gemmData> {
public:
	commInitTask(std::string_view const &name, size_t numberThreads)
			  : hh::AbstractTask<gemmData, gemmData>(name, numberThreads) {}

	void execute(std::shared_ptr<gemmData> inputparams) override {
		inputparams->comm->initializeComm(inputparams->A, inputparams->B);	
		this->addResult(inputparams); //push result
	}

	std::shared_ptr<hh::AbstractTask<gemmData, gemmData>> copy() override {
		return std::make_shared<commInitTask>(this->name(), this->numberThreads());
	}
};

class commFinalizeTask : public hh::AbstractTask<gemmData, gemmData> {
public:
	commFinalizeTask(std::string_view const &name, size_t numberThreads)
			  : hh::AbstractTask<gemmData, gemmData>(name, numberThreads) {}

	void execute(std::shared_ptr<gemmData> inputparams) override {
		inputparams->comm->finalizeComm(inputparams->A, inputparams->B);	
		this->addResult(inputparams); //push result
	}

	std::shared_ptr<hh::AbstractTask<gemmData, gemmData>> copy() override {
		return std::make_shared<commFinalizeTask>(this->name(), this->numberThreads());
	}
};

class matMultTask : public hh::AbstractTask<gemmData, gemmData> {
public:
	matMultTask(std::string_view const &name, size_t numberThreads)
			  : hh::AbstractTask<gemmData, gemmData>(name, numberThreads) {}

	void execute(std::shared_ptr<gemmData> inputparams) override {
		matMultKernel(inputparams->A,inputparams->B, inputparams->C, inputparams->n); //call the same matmult kernel for now.
		this->addResult(inputparams); //push result
	}

	std::shared_ptr<hh::AbstractTask<gemmData, gemmData>> copy() override {
		return std::make_shared<matMultTask>(this->name(), this->numberThreads());
	}
};

class outputState : public hh::AbstractState<gemmData, gemmData> {
	int q; //q = sqrt(p);	
	public:
	outputState(int _q) : q(_q-1) {}
	bool isDone() { return q == 0; };
	void execute(std::shared_ptr<gemmData> inputparams) override {
		q--;
		this->push(inputparams); //push result
	}
};

class outputStateManager : public hh::StateManager<gemmData, gemmData> {
 public:
  explicit outputStateManager(std::shared_ptr<outputState> const &state) :
      hh::StateManager<gemmData, gemmData>("output State Manager", state, false) {}

  bool canTerminate() override {
    this->state()->lock();
    auto ret = std::dynamic_pointer_cast<outputState>(this->state())->isDone();
    this->state()->unlock();
    return ret;
  }
};

void matMult(double **A, double **B, double **C, int n, int q) //Cannon's Algorithm.
{
	int numberThread=1;
	// Declaring and instantiating the graph
	hh::Graph<gemmData, gemmData> graph("Tutorial 6 : Distributed DGEMM");

	// Declaring and instantiating the task
	auto initcomm = std::make_shared<commInitTask>("Tutorial 6 : init comm for DGEMM", numberThread); // init comm task
	auto dgemm     = std::make_shared<matMultTask>("Tutorial 6 : Distributed DGEMM", numberThread);   // main mat mult task 
	auto finalizecomm = std::make_shared<commFinalizeTask>("Tutorial 6 : finalize comm for DGEMM", numberThread); //finalize comm task

	//output state and state manager required to carry out loops of Cannon's algorithm	
	auto stateOutput = std::make_shared<outputState>(q);
	auto stateManagerOutput = std::make_shared<outputStateManager>(stateOutput);

	// Set the task as the task that will be connected to the graph input
	graph.input(initcomm); //init comm at the beginning 
	graph.addEdge(initcomm, dgemm); //inti comm to dgemm
	graph.addEdge(dgemm, finalizecomm); //finalize comm before iterating
	graph.addEdge(finalizecomm, stateManagerOutput); //loop back to init comm or exit depending on the computation state
	graph.addEdge(stateManagerOutput, initcomm);

	// Set the task as the task that will be connected to the graph output
	graph.output(stateManagerOutput);

	// Execute the graph
	graph.executeGraph();

	CommPackage cpack(n, q, A, B); //create and init comm package.

	auto input = std::make_shared<gemmData>(A, B, C, n, q, &cpack); //push current A and B. pushData should start executing graph
	graph.pushData(input);
	
	// Notify the graph that no more data will be sent
	graph.finishPushingData();

	// Wait for everything to be processed
	graph.waitForTermination();


	cpack.destroyComm(A, B);
}


#else //#ifdef USE_HH


void matMult(double **A, double **B, double **C, int n, int q) //Cannon's Algorithm. without HH
{
	CommPackage cpack(n, q, A, B); //create and init comm package.

	for(int k=0; k<q; k++)	//main loop of Cannon's Algorithm. running it from 0 to q rather than 1 to q, otherwise we miss multiplication of 1 block.
	{
		cpack.initializeComm(A, B); //init comm for this iteration

		matMultKernel(A, B, C, n); //mat mult. Main compute task using hh

		cpack.finalizeComm(A, B); //finalize comm
	}

	cpack.destroyComm(A, B);
}

#endif //#ifdef USE_HH

int main(int argc, char *argv[])
{
	struct timeval  tv1, tv2;
	MPI_Init(&argc , &argv);
	int rank, size;
	MPI_Status status;

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int p = size;
	int q = (int)sqrt(p);
	int N = atoi(argv[1]);

	if(q*q < p)	{
		cout << "number of processes should be perfect square\n";
		return(0);
	}

	if(N%q != 0){
		cout << "N (number of rows in matrix) should be divisible by sqrt(p)";
		return(0);
	}

	int n = N / q;

	//initialize matrix tile assigned to rank.
	double **A = initialize(n);
	double **B = initialize(n);
	double **C = initialize(n, 1);	//initialize to 0
	
	double seconds = 0;
	double iterations = 1.0;

	for(int i=0; i<iterations; i++){

		//reset C
		memset(C[0], 0, n*n*sizeof(double));

		gettimeofday(&tv1, NULL);

		//call matrix multiply
		matMult(A, B, C, n, q);

		gettimeofday(&tv2, NULL);
		seconds = seconds + (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
	}

#ifdef USE_HH
	verify(C, n, rank);
#else
	write_to_file(C, n, rank);
#endif
	double perProc[3] = {g_setupTime, g_commTime, seconds}, averageTime[3]={0, 0, 0}; //0th is setup time, 1st is comm time, 2nd is total time

	MPI_Reduce(perProc, averageTime, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank==0)
		printf(" Setup time: %f\n Comm time: %f seconds\n Comp time: %f seconds\n Total time: %f seconds\n",
				averageTime[0]/size/iterations, averageTime[1]/size/iterations, (averageTime[2]-averageTime[0]-averageTime[1])/size/iterations, averageTime[2]/size/iterations  );


	free_array(A, n);
	free_array(B, n);
	free_array(C, n);

	MPI_Finalize();
	return (0);

}
