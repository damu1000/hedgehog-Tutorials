#ifndef COMM_H
#define COMM_H

#include <math.h>
#include <omp.h>
#include "mpi.h"
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>
#define ALONG_ROWS 1
#define ALONG_COLS 0

//#define OVERLAP_COMM

double g_commTime{0}, g_setupTime{0};

//TODO: Figure out how to call destroyComm from MatrixBlockData. Currently not called causing a memory leak
//TODO: Assuming square block size and Row major order for now. Fix later
template<class MatrixType, char Id> //Id should have values 'a' or 'b'
struct CommPackage{
public:
	MPI_Request reqs[2];
	MPI_Status stats[2];
	MatrixType *data;	//pointer to the block data
	MatrixType *sendbuff, *recvbuff; //send and recv buffers.
	int rank, src, dest; //self rank, source and destination ranks for this particular block
	int n, q, leadingDimension;  //n is block size, q is sqrt(num of ranks), and leadingDimension is num of columns in local matrix patch.
	int tag; //tag to be used for mpi comm;
	MPI_Comm comm; //2D comm used for Cartesian arrangement
	//struct timeval  tv1, tv2;

	CommPackage() = default;

	//this must be called before calling init and finalize comm
	void setupCommPackage(MatrixType *_data, int _n, int _leadingDimension, int _rowIdx, int _colIdx) //n and _n is blocksize.
	{
		if(Id != 'a' && Id != 'b') return; // no need to init for ids other than a and b

		data = _data;
		n = _n;
		leadingDimension = _leadingDimension;
		tag = _rowIdx * (leadingDimension / n) + _colIdx; //this is basically the position of the block in the local patch. This will give unique tags to each block. leadingDimension / n gives num of blocks along x dimension
		int size;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		q = (int)sqrt(size);

		//allocate buffers
		sendbuff=new MatrixType[n*n];
		recvbuff=new MatrixType[n*n];

		//create square grid of MPI processes. MPI_Cart_create will be repeated for every block. Fix later create a static class.
		int dims[2] = {q,q};
		int periods[2] = {1,1};
		MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);

		if(Id=='a'){
			MPI_Cart_shift(comm, ALONG_ROWS, -1, &src, &dest);	//get source and destination for A and B. displacement for both is -1 i.e. either to left or up and by 1 element in grid.
			align(ALONG_ROWS, - (rank/q));	//do the initial alignment of tiles
		}
		else if(Id=='b'){
			MPI_Cart_shift(comm, ALONG_COLS, -1, &src, &dest);
			align(ALONG_COLS, - (rank%q));
		}
		//else
		//	printf("error %s %d\n", __FILE__, __LINE__);

//#pragma omp parallel for
		for(int i = 0; i<n; i++) //copy values into send buffer for the first send
			for(int j = 0; j<n; j++)
				sendbuff[i*n+j] = data[i*leadingDimension+j];
	}


	/*shift row  of A circularly by i elements to the left -> use direction ALONG_ROWS. shift_by = - rank/q : rank%q will give column of the tile among comm.
	shift col  of B circularly by j elements up ->  use direction ALONG_COLS. shift_by = - (rank%q) : rank/q will give row of the tile among comm.
	uses 2D comm. Call it only after initializing 2D grid.
	*/
	void align(int direction, int shift_by)	//shift_by should be row number (rank / q ) for
	{
		assert(Id == 'a' || Id == 'b');
		//gettimeofday(&tv1, NULL);
		MPI_Status status;	//use non blocking calls to avoid hang.
		MPI_Request reqs[2];
		MPI_Status stats[2];

			//copy data to sendbuff
//#pragma omp parallel for
			for(int i = 0; i<n; i++)
				for(int j = 0; j<n; j++)
					sendbuff[i*n+j] = data[i*leadingDimension+j];


		int source , dest;
		MPI_Cart_shift(comm, direction, shift_by, &source, &dest);
		//int rank;
		//MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		//cout << rank << " " << " source: " << source <<  " des: " <<dest <<"\n";

		MPI_Isend(sendbuff, n*n*sizeof(MatrixType), MPI_CHAR, dest, tag, comm, &reqs[0]);
		MPI_Irecv(recvbuff, n*n*sizeof(MatrixType), MPI_CHAR, source, tag, comm, &reqs[1]);
		MPI_Waitall(2, reqs, stats);	//wait till data is sent and received

		//gettimeofday(&tv2, NULL);
		//double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		//g_setupTime += seconds;

			//copy data from recvbuff
//#pragma omp parallel for
			for(int i = 0; i<n; i++)
				for(int j = 0; j<n; j++)
					data[i*leadingDimension+j] = recvbuff[i*n+j];
	}

	void initializeComm() //start asynchronous communication
	{
		assert(Id == 'a' || Id == 'b');
		//gettimeofday(&tv1, NULL);
		MPI_Isend(sendbuff, n*n*sizeof(MatrixType), MPI_CHAR, dest, tag, comm, &reqs[0]);	//send
		MPI_Irecv(recvbuff, n*n*sizeof(MatrixType), MPI_CHAR, src, tag, comm, &reqs[1]);	//receive

#ifndef OVERLAP_COMM
		MPI_Waitall(2, reqs, stats);	//wait for all transfers to complete
#endif

		//gettimeofday(&tv2, NULL);
		//double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		//g_commTime += seconds;
		//cout << rank << " " << " source A: " << src_A <<  " des A: " <<dest_A << " source B: " << src_B <<  " des B: " <<dest_B << "\n";
	}

	void finalizeComm() //wait for async comm to be over and copy values from buffer to the variable.
	{
		assert(Id == 'a' || Id == 'b');
#ifdef OVERLAP_COMM
		//printf("waiting for comm\n");
		//gettimeofday(&tv1, NULL);
		MPI_Waitall(2, reqs, stats);	//wait for all transfers to complete
		//gettimeofday(&tv2, NULL);
		//double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		//g_commTime += seconds;
#endif

		//copy received A and B from recieved buffer to A and B
	//#pragma omp parallel for
		for(int i = 0; i<n; i++)
			for(int j = 0; j<n; j++)
				data[i*leadingDimension+j] = recvbuff[i*n+j];

		//data received in this iteration is forwarded. Hence swapping send and recv buffer pointers rather than copying data
		MatrixType *temp = sendbuff;
		sendbuff = recvbuff;
		recvbuff = temp;
	}

	void destroyComm()
	{
		if(Id != 'a' && Id != 'b') return; // no need to destroy for ids other than a and b

		//After the last iteration, align A and B back to old positions - same displacement with opposite direction
		if(Id=='a')
			align(ALONG_ROWS, (rank/q));
		else if (Id=='b')
			align(ALONG_COLS, (rank%q));
		//else
		//	printf("error %s %d\n", __FILE__, __LINE__);
		
		delete []sendbuff;
		delete []recvbuff;
	}
};


#endif //#define COMM_H
