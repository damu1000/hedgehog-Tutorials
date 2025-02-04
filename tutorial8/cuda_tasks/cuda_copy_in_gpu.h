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


#ifndef TUTORIAL8_CUDA_COPY_IN_GPU_H
#define TUTORIAL8_CUDA_COPY_IN_GPU_H
#include <hedgehog/hedgehog.h>
#include "../data/cuda_matrix_block_data.h"
#include "../utils/cuda_utils.h"

template<class MatrixType, char Id>
class CudaCopyInGpu : public hh::AbstractCUDATask<CudaMatrixBlockData<MatrixType, Id>,
                                                  MatrixBlockData<MatrixType, Id, Order::Column>> {
 private:
  size_t
      blockTTL_ = 0,
      blockSize_ = 0,
      matrixLeadingDimension_ = 0,
	  numCopies_ = 0, numResetCopies_ = 0;

  std::atomic<int> id = 0;
  std::vector<cudaEvent_t> copyEvent;
  std::vector<std::shared_ptr<CudaMatrixBlockData<MatrixType, Id>>> blocksCopied;

 public:
  CudaCopyInGpu(size_t blockTTL, size_t blockSize, size_t matrixLeadingDimension, size_t numOfCopies)
      : hh::AbstractCUDATask<CudaMatrixBlockData<MatrixType, Id>, MatrixBlockData<MatrixType, Id, Order::Column>>
            ("Copy In GPU", 1, false, false),
        blockTTL_(blockTTL),
        blockSize_(blockSize),
        matrixLeadingDimension_(matrixLeadingDimension),
		numCopies_ (numOfCopies),
		numResetCopies_ (numOfCopies) {

	  copyEvent = std::vector<cudaEvent_t>(numCopies_);
	  blocksCopied = std::vector<std::shared_ptr<CudaMatrixBlockData<MatrixType, Id>>>(numCopies_, nullptr);

	  for(size_t i=0; i<numCopies_; i++)
		  checkCudaErrors(cudaEventCreate(&copyEvent[i]));
  }

  ~CudaCopyInGpu(){
	  for(size_t i=0; i<numCopies_; i++)
	    checkCudaErrors(cudaEventDestroy(copyEvent[i]));
  }

  void execute(std::shared_ptr<MatrixBlockData<MatrixType, Id, Order::Column>> ptr) override {
    std::shared_ptr<CudaMatrixBlockData<MatrixType, Id>> block = this->getManagedMemory();
    block->rowIdx(ptr->rowIdx());
    block->colIdx(ptr->colIdx());
    block->blockSizeHeight(ptr->blockSizeHeight());
    block->blockSizeWidth(ptr->blockSizeWidth());
    block->leadingDimension(block->blockSizeHeight());
    block->fullMatrixData(ptr->fullMatrixData());
    block->ttl(blockTTL_);

    int myid = id++;
    numCopies_ --;

    if (ptr->leadingDimension() == block->leadingDimension()) {
      checkCudaErrors(cudaMemcpyAsync(block->blockData(), ptr->blockData(),
                                          sizeof(MatrixType) * block->blockSizeHeight() * block->blockSizeWidth(),
                                          cudaMemcpyHostToDevice, this->stream()));
    } else {
      cublasSetMatrixAsync(
          (int) block->blockSizeHeight(), (int) block->blockSizeWidth(), sizeof(MatrixType),
          block->fullMatrixData()
              + IDX2C(block->rowIdx() * blockSize_, block->colIdx() * blockSize_, matrixLeadingDimension_),
          (int) matrixLeadingDimension_, block->blockData(), (int) block->leadingDimension(), this->stream());
    }

    checkCudaErrors(cudaEventRecord(copyEvent[myid], this->stream())); //record event and store the block
    assert(blocksCopied[myid] == nullptr);
    blocksCopied[myid] = block;

    checkProgress(myid);

    if(numCopies_ == 0){                 // if this is the last block,
    	while(checkProgress(myid) > 0); // wait until all pending transfers are over
    	numCopies_ = numResetCopies_;
    	myid = 0;
    	id = 0;
    }
  }

  std::shared_ptr<hh::AbstractTask<CudaMatrixBlockData<MatrixType, Id>,
                                   MatrixBlockData<MatrixType, Id, Order::Column>>> copy() override {
    return std::make_shared<CudaCopyInGpu>(blockTTL_, blockSize_, matrixLeadingDimension_, numCopies_);
  }

  int checkProgress(int size){
	int pending = 0;
	for(int i=0; i <= size; i++){
		if(blocksCopied[i]){//query for a non NULL block and push the result if the copy is completed.
			cudaError_t status = cudaEventQuery(copyEvent[i]);
			if(status == cudaSuccess){
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

#endif //TUTORIAL8_CUDA_COPY_IN_GPU_H
