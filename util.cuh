/*
 * util.cuh
 *
 *  Created on: Jul 16, 2014
 *      Author: bqian
 */

#ifndef UTIL_CUH_
#define UTIL_CUH_

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

void allocateArray(void **devPtr, size_t size);
void freeArray(void *devPtr);
void copyArrayToDevice(void *device, const void *host, int offset, int size);
void copyArrayFromDevice(void *host, const void *device, int size);
void setArray(void *devPtr, int value, int count);
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);

#endif /* UTIL_CUH_ */
