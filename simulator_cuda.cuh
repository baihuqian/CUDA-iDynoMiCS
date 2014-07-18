/*
 * simulator_cuda.cuh
 *
 *  Created on: Jul 18, 2014
 *      Author: bqian
 */

#ifndef SIMULATOR_CUDA_CUH_
#define SIMULATOR_CUDA_CUH_

void agentStepDevice();
void shoveAll();
void followPressure();
void solveDiffusion();
void syncDeviceToHost(); // copy device data to corresponding host memory
void syncHostToDevice(); // copy host data to device memory


#endif /* SIMULATOR_CUDA_CUH_ */
