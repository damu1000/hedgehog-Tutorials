#include <omp.h>
#include "mpi.h"
#define ALONG_ROWS 1
#define ALONG_COLS 0

//#define OVERLAP_COMM

double g_commTime{0}, g_setupTime{0};

struct CommPackage{
public:
	MPI_Request reqs[4];
	MPI_Status stats[4];
	double *sendbuff_A, *recvbuff_A, *sendbuff_B, *recvbuff_B;
	int rank, src_A, dest_A, src_B, dest_B, n, q;
	MPI_Comm comm; //2D comm used for Cartesian arrangement
	struct timeval  tv1, tv2;

	CommPackage(int _n, int _q, double **A, double **B) : n(_n), q(_q)
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		//allocate buffers
		sendbuff_A=new double[n*n];
		recvbuff_A=new double[n*n];
		sendbuff_B=new double[n*n];
		recvbuff_B=new double[n*n];

		//create square grid of MPI processes
		int dims[2] = {q,q};
		int periods[2] = {1,1};
		MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
		MPI_Cart_shift(comm, ALONG_ROWS, -1, &src_A, &dest_A);	//get source and destination for A and B. displacement for both is -1 i.e. either to left or up and by 1 element in grid.
		MPI_Cart_shift(comm, ALONG_COLS, -1, &src_B, &dest_B);
	
	
		align(A, ALONG_ROWS, - (rank/q));	//do the initial alignment of tiles
		align(B, ALONG_COLS, - (rank%q));

//#pragma omp parallel for
		for(int i = 0; i<n; i++){ //copy values into send buffer for the first send
			for(int j = 0; j<n; j++){
				sendbuff_A[i*n+j] = A[i][j];
				sendbuff_B[i*n+j] = B[i][j];
			}
		}		
	}


	/*shift row  of A circularly by i elements to the left -> use direction ALONG_ROWS. shift_by = - rank/q : rank%q will give column of the tile among comm.
	shift col  of B circularly by j elements up ->  use direction ALONG_COLS. shift_by = - (rank%q) : rank/q will give row of the tile among comm.
	uses 2D comm. Call it only after initializing 2D grid.
	*/
	void align(double **A, int direction, int shift_by)	//shift_by should be row number (rank / q ) for
	{
		gettimeofday(&tv1, NULL);
		MPI_Status status;	//use non blocking calls to avoid hang.
		MPI_Request reqs[2];
		MPI_Status stats[2];
		double *sendbuff=sendbuff_A;
		double *recvbuff=recvbuff_A;

			//copy data to sendbuff
//#pragma omp parallel for
			for(int i = 0; i<n; i++)
				for(int j = 0; j<n; j++)
					sendbuff[i*n+j] = A[i][j];


		int source , dest;
		MPI_Cart_shift(comm, direction, shift_by, &source, &dest);
		//int rank;
		//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		//cout << rank << " " << " source: " << source <<  " des: " <<dest <<"\n";

		MPI_Isend(sendbuff, n*n, MPI_DOUBLE, dest, 1, comm, &reqs[0]);
		MPI_Irecv(recvbuff, n*n, MPI_DOUBLE, source, 1, comm, &reqs[1]);
		MPI_Waitall(2, reqs, stats);	//wait till data is sent and received

		gettimeofday(&tv2, NULL);
		double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		g_setupTime += seconds;

			//copy data from recvbuff
//#pragma omp parallel for
			for(int i = 0; i<n; i++)
				for(int j = 0; j<n; j++)
					A[i][j] = recvbuff[i*n+j];
	}

	void initializeComm(double **A, double **B) //start asynchronous communication
	{
		gettimeofday(&tv1, NULL);
		MPI_Isend(sendbuff_A, n*n, MPI_DOUBLE, dest_A, 1, comm, &reqs[0]);	//send A and B
		MPI_Isend(sendbuff_B, n*n, MPI_DOUBLE, dest_B, 1, comm, &reqs[1]);

		MPI_Irecv(recvbuff_A, n*n, MPI_DOUBLE, src_A, 1, comm, &reqs[2]);	//receive A and B
		MPI_Irecv(recvbuff_B, n*n, MPI_DOUBLE, src_B, 1, comm, &reqs[3]);

		int flag;
		MPI_Testall(4, reqs, &flag, MPI_STATUSES_IGNORE);

#ifndef OVERLAP_COMM
		//printf("waiting for comm\n");
		MPI_Waitall(4, reqs, stats);	//wait for all transfers to complete
#endif

		gettimeofday(&tv2, NULL);
		double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		g_commTime += seconds;
		//cout << rank << " " << " source A: " << src_A <<  " des A: " <<dest_A << " source B: " << src_B <<  " des B: " <<dest_B << "\n";
	}

	void finalizeComm(double **A, double **B) //wait for async comm to be over and copy values from buffer to the variable.
	{

#ifdef OVERLAP_COMM
		//printf("waiting for comm\n");
		gettimeofday(&tv1, NULL);
		MPI_Waitall(4, reqs, stats);	//wait for all transfers to complete
		gettimeofday(&tv2, NULL);
		double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		g_commTime += seconds;
#endif


		//copy received A and B from recieved buffer to A and B
	//#pragma omp parallel for
		for(int i = 0; i<n; i++)
			for(int j = 0; j<n; j++)
			{
				A[i][j] = recvbuff_A[i*n+j];
				B[i][j] = recvbuff_B[i*n+j];
			}

		//data received in this iteration is forwarded. Hence swapping send and recv buffer pointers rather than copying data
		double *temp = sendbuff_A;
		sendbuff_A = recvbuff_A;
		recvbuff_A = temp;
		temp = sendbuff_B;
		sendbuff_B = recvbuff_B;
		recvbuff_B = temp;
	}

	void destroyComm(double **A, double **B)
	{
		//After the last iteration, align A and B back to old positions - same displacement with opposite direction
		align(A, ALONG_ROWS, (rank/q));
		align(B, ALONG_COLS, (rank%q));
		
		
		delete []sendbuff_A;
		delete []recvbuff_A;
		delete []sendbuff_B;
		delete []recvbuff_B;
	}

};
