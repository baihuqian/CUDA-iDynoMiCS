/*
 * util.cu
 *
 *  Created on: Jul 16, 2014
 *      Author: bqian
 */

#include "util.cuh"

void allocateArray(void **devPtr, size_t size)
{
	CUDA_CHECK_RETURN(cudaMalloc(devPtr, size));
}


void freeArray(void *devPtr)
{
	CUDA_CHECK_RETURN(cudaFree(devPtr));
}

void copyArrayToDevice(void *device, const void *host, int offset, int size)
{
	CUDA_CHECK_RETURN(cudaMemcpy((char *) device + offset, host, size, cudaMemcpyHostToDevice));
}

void copyArrayFromDevice(void *host, const void *device, int size)
{
	CUDA_CHECK_RETURN(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}

void setArray(void *devPtr, int value, int count)
{
	CUDA_CHECK_RETURN(cudaMemset(devPtr, value, count));
}

void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

uint inline iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
