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

#ifndef TUTORIAL7_COMM_TASKS_H
#define TUTORIAL7_COMM_TASKS_H

#include "comm.h"
#include <hedgehog/hedgehog.h>
#include "../data/matrix_block_data.h"
#include "../data/cuda_matrix_block_data.h"
#include <atomic>

//setup comm task
template<class Type, char Id, Order Ord = Order::Column, class MBD = MatrixBlockData<Type, Id, Ord>>
class commSetupTask : public hh::AbstractTask<MBD, std::vector<std::shared_ptr<MBD>>> {
public:
	int setupComm{1};
	int numBlocksRows{0};
	int numBlocksCols{0};

	commSetupTask(int numBlocksRows_, int numBlocksCols_, int setupComm_=1) : hh::AbstractTask<MBD, std::vector<std::shared_ptr<MBD>>>("commSetup") {
		setupComm = setupComm_;
		numBlocksRows = numBlocksRows_;
		numBlocksCols = numBlocksCols_;
	}

	void execute(std::shared_ptr<std::vector<std::shared_ptr<MBD>>> matBlocksVec) override {
		for(auto &mb: *matBlocksVec)
			this->addResult(mb); //push result
	}


//	void execute(std::shared_ptr<std::vector<std::shared_ptr<MBD>>> matBlocksVec) override {
//		//printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
//		auto matBlocks = *matBlocksVec;
//
//		for(auto &mb: matBlocks)
//			mb->setupCommPackage(numBlocksRows, numBlocksCols, setupComm);
//
//		for(auto &mb: matBlocks){
//			mb->finalizeSetupComm(setupComm);
//			this->addResult(mb); //push result
//		}
//	}

	std::shared_ptr<hh::AbstractTask<MBD, std::vector<std::shared_ptr<MBD>>>> copy() override {
		return std::make_shared<commSetupTask<Type, Id, Ord>>(numBlocksRows, numBlocksCols, setupComm);
	}
};



//input from traversal task and pass it on to finalize and also product
template<class Type, char Id, Order Ord = Order::Column, class MBD = MatrixBlockData<Type, Id, Ord>>
class commInitTask : public hh::AbstractTask<MBD, MBD> {
public:
	size_t num_blocks{0};

	std::vector<std::shared_ptr<MBD>> blocks; //blocks with pending comm

	commInitTask(size_t num_blocks_) : hh::AbstractTask<MBD, MBD>("commInit") {
		num_blocks = num_blocks_;
	}

	void execute(std::shared_ptr<MBD> matBlock) override {
		//printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		matBlock->initializeComm();
		this->addResult(matBlock); //push result
		blocks.push_back(matBlock);

		if(blocks.size() == num_blocks){ //wait until all blocks complete their comm. Make comm progress
			for(auto &m : blocks)
				m->waitForComm();
			blocks.clear();
		}
	}

	std::shared_ptr<hh::AbstractTask<MBD, MBD>> copy() override {
		return std::make_shared<commInitTask<Type, Id, Ord>>(num_blocks);
	}
};

/*
  Finalize comm task:
  Output: MBD for a/b: overwrites the incoming MBD from input 1 with the latest value from MPI buffer
  Input1: MBD from Init comm task. This gets overwritten in finalize method after MPI comm is completed
  Input2: CudaMatrixBlockData for a.
  Input3: CudaMatrixBlockData for b.
  	  	  The input 2 and 3 objects (CudaMatrixBlockData) themselves are not used, these are just used as signal to indicate that the block is copied from host to device
  	  	  and the host memory can be overwritten.
  	  	  However wait until all blocks of and b are copied to device before starting MPI finalize to avoid pushing of new blocks ahead of old ones.
  	  	  Achieved by usedBlocks logic.
  */
template<class Type, char Id, Order Ord = Order::Column, class MBD = MatrixBlockData<Type, Id, Ord>>
class commFinalizeTask : public hh::AbstractTask<MBD, MBD, CudaMatrixBlockData<Type, 'a'>, CudaMatrixBlockData<Type, 'b'>> {
public:

	int rows, cols;
	std::atomic<int> usedBlocks{0};
	std::vector<std::shared_ptr<MBD>> matBlock;

	commFinalizeTask(int _rows, int _cols) : hh::AbstractTask<MBD, MBD, CudaMatrixBlockData<Type, 'a'>, CudaMatrixBlockData<Type, 'b'>>("commFinalize") {
		rows = _rows;
		cols = _cols;
		usedBlocks = 2 * rows * cols; //multiply by 2 because consider both matrices A and B
		matBlock = std::vector<std::shared_ptr<MBD>> (rows*cols, nullptr);
	}

	void execute(std::shared_ptr<MBD> _matBlock) override { //from InitComm. store the matrix in matBlock array to be used in the future. This SHOULD be always called before other executes due to linear dependency
		matBlock[_matBlock->rowIdx() * cols + _matBlock->colIdx()] = _matBlock;
	}

	void execute(std::shared_ptr<CudaMatrixBlockData<Type, 'a'>> partialResult) override {
		finalize();
	}

	void execute(std::shared_ptr<CudaMatrixBlockData<Type, 'b'>> partialResult) override {
		finalize();
	}

	void finalize() {//partialResult is unused here, but it ensures product is completed
		usedBlocks--; //decrement the number of A and B blocks used
		if(usedBlocks == 0){ //finalize the comm when ALL A and B blocks are used
			usedBlocks = 2 * rows * cols; //restore for the next iteration

			int pending;
			do{ //run until at least one block is pending
				pending = 0;
				for(auto &m : matBlock){ //iterate over blocks
					if(m){
						if(m->finalizeComm()){ //if comm completed. finalizeComm should ideally use MPI_Testall to check, but causes some race condition. So using MPI_waitall as of now. Check finalizeComm method in comm.h
							m->incCannonIteration(); //increment Cannon iteration for the new block
							this->addResult(m); //push result
							m = nullptr;
						}
						else
							pending++;
					}
				}
			}while(pending > 0);

		}
	}

	std::shared_ptr<hh::AbstractTask<MBD, MBD, CudaMatrixBlockData<Type, 'a'>, CudaMatrixBlockData<Type, 'b'>>> copy() override {
		return std::make_shared<commFinalizeTask>(this->rows, this->cols);
	}
};

#endif
