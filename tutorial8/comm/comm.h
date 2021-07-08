#ifndef TUTORIAL8_COMM_H
#define TUTORIAL8_COMM_H

#include<iostream>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include <sys/time.h>
#include <stdio.h>
#include <assert.h>

#include "../data/data_type.h"

#define ALONG_ROWS 1
#define ALONG_COLS 0

#define OVERLAP_COMM

double g_commTime{0}, g_setupTime{0};

//A big time lesson here: Obvious factor in MPI communication is to have a common MPI_Comm in send and recv, but when
//comm is part of an object instance, a new comm will be created for every object. Sometimes new comm pair correctly
//in send-recv, some time it does not. So creating a common global MPI_Comm for everyone. Obvious but can be missed easily

MPI_Comm comm; //2D comm used for Cartesian arrangement

void createCartComm(int q){
	//create square grid of MPI processes. MPI_Cart_create will be repeated for every block. Fix later create a static class.
	int dims[2] = {q,q};
	int periods[2] = {1,1};
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
}

//TODO: Figure out how to call destroyComm from MatrixBlockData. Currently not called causing a memory leak
//TODO: Assuming square block size and Row major order for now. Fix later
template<class MatrixType, char Id, Order Ord = Order::Column> //Id should have values 'a' or 'b'
struct CommPackage{
public:
	MPI_Request reqs[2];
	MPI_Status stats[2];
	MatrixType *data{nullptr};	//pointer to the block data
	MatrixType *sendbuff{nullptr}, *recvbuff{nullptr}; //send and recv buffers.
	int rank{0}, src{0}, dest{0}; //self rank, source and destination ranks for this particular block
	int n{0}, q{0}, leadingDimension{0};  //n is block size, q is sqrt(num of ranks), and leadingDimension is num of columns in local matrix patch.
	int tag{0}; //tag to be used for mpi comm;
	int tagoffset{0};
	int iteration{0};
	int numBlocksRows{0};
	int numBlocksCols{0};
	//struct timeval  tv1, tv2;
	int rowIdx{0}, colIdx{0};

	CommPackage() = default;

	void copyBuffer(MatrixType *dest, int destLeadingDim, MatrixType *src, int srcLeadingDim, int n){
		//TODO: Ideally order should not make difference in copying block into the buffer. Try it
		if(Ord == Order::Row){
			for(int i = 0; i<n; i++) //copy values into send buffer for the first send
				for(int j = 0; j<n; j++)
					dest[i * destLeadingDim + j] = src[i * srcLeadingDim + j];
		}
		else{
			for(int i = 0; i<n; i++) //copy values into send buffer for the first send
				for(int j = 0; j<n; j++)
					dest[j * destLeadingDim + i] = src[j * srcLeadingDim + i];
		}
	}

	//this must be called before calling init and finalize comm
	void setupCommPackage(int numBlocksRows_, int numBlocksCols_, MatrixType *_data, int _n, int _leadingDimension, int _rowIdx, int _colIdx, int alignMat=0) //n and _n is blocksize.
	{
		if(Id != 'a' && Id != 'b') return; // no need to init for ids other than a and b
		rowIdx = _rowIdx;
		colIdx = _colIdx;
		numBlocksRows = numBlocksRows_;
		numBlocksCols = numBlocksCols_;
		data = _data;
		n = _n;
		leadingDimension = _leadingDimension;
		tagoffset = numBlocksRows * numBlocksCols * 2 ;//tags in one iteration. increment tag value in the next iteration by this offset to avoid conflict between iterations
		tag = _rowIdx * numBlocksCols + _colIdx; //this is basically the position of the block in the local patch. This will give unique tags to each block. leadingDimension / n gives num of blocks along x dimension
		tag = 2*tag + (Id == 'a' ? 0 : 1);
		int size;
		MPI_Comm_rank(comm, &rank);
		MPI_Comm_size(comm, &size);
		q = (int)sqrt(size);

		//allocate buffers
		sendbuff=new MatrixType[n*n];
		recvbuff=new MatrixType[n*n];

		if(Id=='a')
			MPI_Cart_shift(comm, ALONG_ROWS, -1, &src, &dest);	//get source and destination for A and B. displacement for both is -1 i.e. either to left or up and by 1 element in grid.
		else if(Id=='b')
			MPI_Cart_shift(comm, ALONG_COLS, -1, &src, &dest);

		if(alignMat==1){
			if(Id=='a')
				align(ALONG_ROWS, - (rank/q));	//do the initial alignment of tiles
			else if(Id=='b')
				align(ALONG_COLS, - (rank%q));
		}
		else
			copyBuffer(sendbuff, n, data, leadingDimension, n); //prepare send buff for first send in initComm
	}


	/*shift row  of A circularly by i elements to the left -> use direction ALONG_ROWS. shift_by = - rank/q : rank%q will give column of the tile among comm.
	shift col  of B circularly by j elements up ->  use direction ALONG_COLS. shift_by = - (rank%q) : rank/q will give row of the tile among comm.
	uses 2D comm. Call it only after initializing 2D grid.
	*/
	void align(int direction, int shift_by)	//shift_by should be row number (rank / q ) for
	{
		assert(Id == 'a' || Id == 'b');
		//gettimeofday(&tv1, NULL);

		//copy data to sendbuff
		copyBuffer(sendbuff, n, data, leadingDimension, n);

		int s , d;
		MPI_Cart_shift(comm, direction, shift_by, &s, &d);
//		int rank;
//		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//		std::cout << " Id: " << Id << " source: " << s <<  " des: " <<d << " tag: " << 10000+tag << "\n";

		//added 10k to tag. Otherwise conflicts with subsequent init comm. Do not know how, but it does.
		MPI_Isend(sendbuff, n*n*sizeof(MatrixType), MPI_BYTE, d, 10000+tag, comm, &reqs[0]);
		MPI_Irecv(recvbuff, n*n*sizeof(MatrixType), MPI_BYTE, s, 10000+tag, comm, &reqs[1]);
	}

	void finalizeAlign()
	{
		MPI_Waitall(2, reqs, stats);	//wait till data is sent and received
//		printf("%d Aligned for block %c - %d size: %d x %d\n", rank, Id, tag, n, n);

		//gettimeofday(&tv2, NULL);
		//double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		//g_setupTime += seconds;

		//copy data from recvbuff
		copyBuffer(data, leadingDimension, recvbuff, n, n);
		//data received in this iteration is forwarded. Hence swapping send and recv buffer pointers rather than copying data
		MatrixType *temp = sendbuff;
		sendbuff = recvbuff;
		recvbuff = temp;
	}


	void initializeComm() //start asynchronous communication
	{
		assert(Id == 'a' || Id == 'b');
		//gettimeofday(&tv1, NULL);
		MPI_Isend(sendbuff, n*n*sizeof(MatrixType), MPI_BYTE, dest, tag, comm, &reqs[0]);	//send
		MPI_Irecv(recvbuff, n*n*sizeof(MatrixType), MPI_BYTE, src, tag, comm, &reqs[1]);	//receive

#ifndef OVERLAP_COMM
		MPI_Waitall(2, reqs, stats);	//wait for all transfers to complete
#endif

		//gettimeofday(&tv2, NULL);
		//double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		//g_commTime += seconds;
		//cout << rank << " " << " source A: " << src_A <<  " des A: " <<dest_A << " source B: " << src_B <<  " des B: " <<dest_B << "\n";
	}

	//return true if finalize comm is successful.
	int finalizeComm() //wait for async comm to be over and copy values from buffer to the variable.
	{
		assert(Id == 'a' || Id == 'b');

#ifdef OVERLAP_COMM
		//printf("waiting for comm\n");
		//gettimeofday(&tv1, NULL);
		MPI_Waitall(2, reqs, stats);	//wait for all transfers to complete

		//ideally test all logic should work instead of wait_all, but shows race conditions. Because of out of order offload to GPU in subsequent tasks??
		//hopefully should not make so much difference considering early finalize comm call as soon as copy ins of all blocks is completed.
//		int flag=0;
//		MPI_Testall(2, reqs, &flag, stats);
//		if(flag == false) return flag;

		//gettimeofday(&tv2, NULL);
		//double seconds = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec);
		//g_commTime += seconds;
#endif

		//copy received A and B from recieved buffer to A and B
		copyBuffer(data, leadingDimension, recvbuff, n, n);

		tag += tagoffset;
		iteration++;

		//data received in this iteration is forwarded. Hence swapping send and recv buffer pointers rather than copying data
		MatrixType *temp = sendbuff;
		sendbuff = recvbuff;
		recvbuff = temp;

		return true;
	}

	void destroyComm()//(int alignMat=0)
	{
		if(Id != 'a' && Id != 'b') return; // no need to destroy for ids other than a and b

		//no need to align again. will be copied from backup copy in matrix_block_data
		//After the last iteration, align A and B back to old positions - same displacement with opposite direction
//		if(alignMat==1){
//			if(Id=='a')
//				align(ALONG_ROWS, (rank/q));
//			else if (Id=='b')
//				align(ALONG_COLS, (rank%q));
//		}
		delete []sendbuff;
		delete []recvbuff;
		sendbuff = nullptr;
		recvbuff = nullptr;
	}
};


#endif //#define COMM_H
