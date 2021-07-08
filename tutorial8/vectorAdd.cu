#include <stdio.h>

//alternate to calling this kernel is to use cublasDaxpy, but needs to change cublas stream before every call. do not know the impact
extern "C"{

__global__ void vectorAddKernel(double *dest, double *src, size_t n){
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;	
#pragma unroll (32)
	for(size_t i = id; i < n; i = i + blockDim.x * gridDim.x)
		dest[i] += src[i];

}

void vectorAdd(double *dest, double *src, size_t n, cudaStream_t stream){		
		int threads = 512;
		int blocks = min(4096, (int)((n + threads - 1) / threads));
		vectorAddKernel<<<blocks, threads, 0, stream>>>(dest, src, n);
}



__global__ void vectorPrintKernel(double *v, size_t n){
	size_t id = blockIdx.x * blockDim.x + threadIdx.x;	
#pragma unroll (32)
	for(size_t i = id; i < n; i = i + blockDim.x * gridDim.x)
		printf("%.2f\t", v[i]);

}

void vectorPrint(double *v, size_t n, cudaStream_t stream){
		int threads = 512;
		int blocks = min(4096, (int)((n + threads - 1) / threads));
		vectorPrintKernel<<<blocks, threads, 0, stream>>>(v, n);
}

}
