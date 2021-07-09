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


#ifndef TUTORIAL8_CUDA_ADDITION_TASK_H
#define TUTORIAL8_CUDA_ADDITION_TASK_H
#include <cuda.h>
#include <cublas.h>
#include <hedgehog/hedgehog.h>
#include "../data/cuda_matrix_block_data.h"
#include "cuda_streams.h"
#include <atomic>
#include <cuda_runtime_api.h>


extern "C"{
void vectorAdd(double *dest, double *src, size_t n, cudaStream_t stream);
void vectorPrint(double *v, size_t n, cudaStream_t stream);
}

template<class Type>
class CudaAdditionTask : public hh::AbstractCUDATask<
	MatrixBlockData<Type, 'c', Order::Column>,
	CudaMatrixBlockData<Type, 'p'>> { //only takes matrix p as input. c array will be passed as input and stored locally
 private:
  size_t nBlocks=0, mBlocks=0, pBlocks=0;
  int q = 0;
  std::vector<std::atomic<int>> ttl;
  std::shared_ptr<std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>>> matC;

 public:
  explicit CudaAdditionTask(size_t nBlocks_, size_t mBlocks_, size_t pBlocks_, int q_,
		  	  	  	  	  	std::shared_ptr<std::vector<std::shared_ptr<MatrixBlockData<Type, 'c', Order::Column>>>> matC_
		  	  	  	  	  	/*, size_t numberThreadsAddition = 1*/)
  : hh::AbstractCUDATask<
	MatrixBlockData<Type, 'c', Order::Column>,
	CudaMatrixBlockData<Type, 'p'>>
	("CUDA Addition Task", 1 /*numberThreadsAddition*/, false, false) //using a single thread as of now. make it thread safe
  {
	  nBlocks=nBlocks_, mBlocks=mBlocks_, pBlocks=pBlocks_;
	  q = q_;
	  matC = matC_;

	  std::vector<std::atomic<int>> temp(nBlocks*pBlocks);
	  ttl = std::move(temp); //atomic does not allow copy / assignment in the constructor. hence using move
	  for(auto &t: ttl)
		  t =  mBlocks*q;

  }

  void initializeCuda() override {}

  void shutdownCuda() override {}

  void execute(std::shared_ptr<CudaMatrixBlockData<Type, 'p'>> P) override {

    auto C = (*(matC))[P->colIdx() * nBlocks + P->rowIdx()]; //column major order hence matP->colIdx() * nblocks + matP->rowIdx(). Is it correct?

    //assert(C->rowIdx() == P->rowIdx() && C->colIdx() == P->colIdx());

    //call the kernel to add c and p
    auto stream = cudaStreams::getStream(P->rowIdx(), P->colIdx());
    vectorAdd(C->blockData(), P->blockData(), C->blockSizeHeight() * C->blockSizeWidth(), stream);

    //return A and B blocks to memory manager asynchronously.
    CudaMatrixBlockData<Type, 'p'>::returnCudaBlock(P, stream);

    //check ttl and push the result only if all additions are completed for that block of C.
    ttl[P->colIdx() * nBlocks + P->rowIdx()]--;
    if(ttl[P->colIdx() * nBlocks + P->rowIdx()] == 0)
    	this->addResult(C);
  }


  std::shared_ptr<hh::AbstractTask<
	MatrixBlockData<Type, 'c', Order::Column>,
	CudaMatrixBlockData<Type, 'p'>>>
	copy() override {
    return std::make_shared<CudaAdditionTask>(nBlocks, mBlocks, pBlocks, q, matC /*this->numberThreads()*/);
  }
};

#endif //TUTORIAL8_CUDA_ADDITION_TASK_H
