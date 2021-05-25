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


//input from traversal task and pass it on to finalize and also product
template<class Type, char Id, Order Ord = Order::Column, class MBD = MatrixBlockData<Type, Id, Ord>>
class commInitTask : public hh::AbstractTask<MBD, MBD> {
public:
	commInitTask() : hh::AbstractTask<MBD, MBD>("commInit") {}

	void execute(std::shared_ptr<MBD> matBlock) override {
		//printf("Executing task: '%s', function '%s' at %s:%d\n", std::string(this->name()).c_str(), __FUNCTION__,  __FILE__, __LINE__ ) ;
		matBlock->initializeComm();
		this->addResult(matBlock); //push result
	}

	std::shared_ptr<hh::AbstractTask<MBD, MBD>> copy() override {
		return std::make_shared<commInitTask<Type, Id, Ord>>();
	}
};



//needs A / B from init task. Just save A / B passed on by init.
//Second input is partial matrix p from product Task. Decrement ttl for every input from productTask. Finalize comm when ttl reaches 0
template<class Type, char Id, Order Ord = Order::Column, class MBD = MatrixBlockData<Type, Id, Ord>>
class commFinalizeTask : public hh::AbstractTask<MBD, MBD, MatrixBlockData<Type, 'c', Ord>> {
public:
	//call finalizeComm only when ttl (time to live) reaches 0.
	//ttl should be nBlocks*mBlocks*pBlocks i.e., num of blocks calculated locally. Restore ttl after each Cannon's iteration.
	//Do not multiply by q, because counter needs to reach 0 for every iteration to call finalizeComm routine. If multiplied by q, ttl will never reach 0.
	int ttl, ttlCopy, rows, cols;
	std::vector<std::shared_ptr<MBD>> matBlock;

	//TODO: rethink about ttl logic. no need to wait for the entire c to be computed before finalizing comm for all blocks at once
	//can it be done based on rolling basis? use indexes of matrix c to finalize comm of A and B with corresponding row and col ???
	commFinalizeTask(int _rows, int _cols, int _ttl) : hh::AbstractTask<MBD, MBD, MatrixBlockData<Type, 'c', Ord>>("commFinalize") {
		ttl = _ttl;
		ttlCopy = ttl;
		rows = _rows;
		cols = _cols;
		matBlock = std::vector<std::shared_ptr<MBD>> (rows*cols, nullptr);
	}

	void execute(std::shared_ptr<MBD> _matBlock) override {
		matBlock[_matBlock->rowIdx() * cols + _matBlock->colIdx()] = _matBlock;
	}

	void execute(std::shared_ptr<MatrixBlockData<Type, 'c', Ord>> partialResult) override {//partialResult is unused here, but it ensures product is completed
		ttl--;
		if(ttl==0){
			ttl = ttlCopy; //restore for the next Cannon's iteration
			for(auto &m : matBlock){
				if(m){
					m->finalizeComm();
					this->addResult(m); //push result
				}
			}
//			for(auto &m : matBlock){
//				this->addResult(m); //push result
//			}
		}
	}

	std::shared_ptr<hh::AbstractTask<MBD, MBD, MatrixBlockData<Type, 'c', Ord>>> copy() override {
		return std::make_shared<commFinalizeTask>(this->rows, this->cols, this->ttl);
	}
};


#endif
