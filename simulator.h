/*
 * simulator.h
 *
 *  Created on: Jul 18, 2014
 *      Author: bqian
 */

#ifndef SIMULATOR_H_
#define SIMULATOR_H_

#include "constant.h"

class Simulator {
	/* private domain */
	/*---------------------HOST--------------------------*/
	// dictionary
	char **speciesDic, **reactionDic, **solverDic, **particleDic, **soluteDic;
	// flags
	bool m_initialized = false;
	bool m_continueRunning = true;
	bool m_is3D = true;

	// maximal physical environment size
	float domainX, domainY, domainZ; // X, Y, Z dimension of domain in micrometer
	int nX, nY, nZ; // number of grid elements in X, Y, Z direction
	float domainResolution; // width of each side of the grid

	// indices of boundaryLayer in Z direction
	int *boundaryLayer;

	// solute parameters in bulk
	float *h_bulkSoluteList;
	int *bulkSoluteName;

	// solute parameters in domain
	float **h_soluteList; // array of soluteList, soluteList is represented by 1D array
	int *soluteName; // indices of solute names in soluteDic

	// reaction
	//TODO: add contents

	// physical conditions of agent
	int h_numAgents;
	float *h_radius, *h_totalRadius, *h_divRadius, *h_deathRadius; // radius: without capsule, with capsule, division threshold, death threshold
	float *h_position; // location of each agent, size should be 2 * MAX_NUM_AGENTS (2D) or 3 * MAX_NUM_AGENTS (3D)

	// chemical conditions of agent
	//TODO: add contents from ActiveAgent
	// biological conditions of agent
	int *h_species; // indices in speciesDic;

	//simulator control
	float agentStepSize, globalStepSize; // step size for both agent step and global step

	/*---------------------DEVICE------------------------*/
#if CUDA




#endif
	/*---------------------METHOD------------------------*/
	Simulator();





protected:/* protected domain */
#if CUDA == 0 // CPU implementation of device code
	void agentStepDevice();
	void shoveAll();
	void solveDiffusion(); // solve diffusion-reaction relaxation
	void followPressure(); // solve movement caused by pressure field
#endif
	void step(); // step function
	void checkAgentBirth();

	void agentStep();
	void afterShove();
	int gridIndex2D(int X, int Y); // 2D index to 1D index
	int gridIndex3D(int X, int Y, int Z); // 3D index to 1D index

public:/* public domain */
	Simulator(XMLParser parser); // populate data stored in XMLParser to host memory
	void start(); // start simulation
};


#endif /* SIMULATOR_H_ */
