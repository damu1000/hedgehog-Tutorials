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


#ifndef TUTORIAL8_CUDA_COPY_OUT_GPU_H
#define TUTORIAL8_CUDA_COPY_OUT_GPU_H
#include <hedgehog/hedgehog.h>
#include "../data/cuda_matrix_block_data.h"

template<class MatrixType>
class CudaCopyOutGpu
    : public hh::AbstractCUDATask<MatrixBlockData<MatrixType, 'p', Order::Column>,
                                  CudaMatrixBlockData<MatrixType, 'p'>> {
 private:
  size_t blockSize_, numCopies_ = 0, numResetCopies_ = 0;

  std::atomic<int> id = 0;
  std::vector<cudaEvent_t> copyEvent;
  std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'p', Order::Column>>> blocksCopied;
  std::vector<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'p'>>> managedMemory;

 public:
  explicit CudaCopyOutGpu(size_t blockSize, size_t numOfCopies)
      : hh::AbstractCUDATask<MatrixBlockData<MatrixType, 'p', Order::Column>, CudaMatrixBlockData<MatrixType, 'p'>>
            ("Copy Out GPU", 1, false, false), blockSize_(blockSize), numCopies_ (numOfCopies), numResetCopies_ (numOfCopies) {
	  copyEvent = std::vector<cudaEvent_t>(numCopies_);
	  blocksCopied = std::vector<std::shared_ptr<MatrixBlockData<MatrixType, 'p', Order::Column>>>(numCopies_, nullptr);
	  managedMemory = std::vector<std::shared_ptr<CudaMatrixBlockData<MatrixType, 'p'>>> (numCopies_, nullptr);

	  for(size_t i=0; i<numCopies_; i++)
		  checkCudaErrors(cudaEventCreate(&copyEvent[i]));
  }

  ~CudaCopyOutGpu(){
	  for(size_t i=0; i<numCopies_; i++)
	    checkCudaErrors(cudaEventDestroy(copyEvent[i]));
  }

  void execute(std::shared_ptr<CudaMatrixBlockData<MatrixType, 'p'>> ptr) override {
	int myid = id++;
	numCopies_ --;

    auto block = ptr->copyToCPUMemory(this->stream());

    checkCudaErrors(cudaEventRecord(copyEvent[myid], this->stream())); //record event and store the block
    assert(blocksCopied[myid] == nullptr);
    blocksCopied[myid] = block;
    managedMemory[myid] = ptr;

    checkProgress(myid); //check progress of earlier async copies.

	if(numCopies_ == 0){                 // if this is the last block,
		while(checkProgress(myid) > 0); // wait until all pending transfers are over
		numCopies_ = numResetCopies_;
		myid = 0;
		id = 0;
	}
  }

  std::shared_ptr<hh::AbstractTask<MatrixBlockData<MatrixType, 'p', Order::Column>,
                                   CudaMatrixBlockData<MatrixType, 'p'>>> copy() override {
    return std::make_shared<CudaCopyOutGpu>(this->blockSize_, numCopies_);
  }


  int checkProgress(int size){
	int pending = 0;
	for(int i=0; i <= size; i++){
		if(blocksCopied[i]){//query for a non NULL block and push the result if the copy is completed.
			cudaError_t status = cudaEventQuery(copyEvent[i]);
			if(status == cudaSuccess){
				managedMemory[i]->returnToMemoryManager(); //return memory
				managedMemory[i] = nullptr;
				this->addResult(blocksCopied[i]);
				blocksCopied[i] = nullptr;
			}
			else if(status == cudaErrorNotReady)//count pending transfers
				pending++;
			else //handle the error if cuda status is other than success and in progress.
				checkCudaErrors(status);
		}
	}
	return pending;
  }
};

#endif //TUTORIAL8_CUDA_COPY_OUT_GPU_H
