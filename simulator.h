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
	// name
	char **speciesDic, **reactionDic, **solverDic, **particleDic, **soluteDic;
	int numSpecies, numReaction, numSolver, numParticle, numSolute;

	//reaction dictionary
	//size: # reaction * # solute/particle
	float **soluteYield; // 1st index reaction, 2nd index solute
	float **reactionKinetic; // TODO:1st index reaction, 2nd index size unknown
	float **particleYield; // 1st index reaction, 2nd index particle
	int *catalystIndex; // for each reaction, record index of catalyst particle in particleDic
	bool *autocatalytic; // equivalent to Reaction.autocatalytic

	// flags
	bool m_initialized = false;
	bool m_continueRunning = true; // set false to stop simulation
	bool m_is3D = true; // true if the world is 3D, false if 2D

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
	//TODO: add contents for reaction in domain

	/*-----------------AGENTS-------------------*/
	int h_numAgents; // number of agents, should not exceed MAX_NUM_AGENTS
	// physical conditions of agent
	float *h_radius, *h_totalRadius, *h_divRadius, *h_deathRadius; // radius: without capsule, with capsule, division threshold, death threshold
	float *h_position; // location of each agent, size should be 2 * MAX_NUM_AGENTS (2D) or 3 * MAX_NUM_AGENTS (3D)
	bool  *h_isAttached; // whether agents interact with a surface, default to false
	// chemical conditions of agent
	// Reaction[] allReaction;
	int **h_allReactions; // all possible reactions, 1st index for agent, 2nd index for reaction
	int *h_numAllReactions; // store array size for each agent
	// ArrayList<Integer> reactionActive;
	int **h_reactionActive; // same structure as h_allReactions, and size of each array are the same
	int *h_numReactionActive; // keep number of active reactions (smaller than allReactions)
	// Double[] particleMass;
	float **h_particleMass; // mass of particles that belong to agents, 1st index for agent, 2nd index for particles, size MAX_NUM_AGENTS * numParticle

	// biological conditions of agent
	int *h_species; // indices in speciesDic;
	bool *h_hasEps; // true of EPS is declared for this bacterium

	//simulator control
	float agentStepSize, globalStepSize; // step size for both agent step and global step

	/*---------------------DEVICE------------------------*/
#if CUDA




#endif
	/*---------------------METHOD------------------------*/
	Simulator();
	~Simulator();




protected:/* protected domain */
#if CUDA == 0 // CPU implementation of device code
	void agentStepDevice();
	void shoveAll();
	void solveDiffusion(); // solve diffusion-reaction relaxation
	void followPressure(); // solve movement caused by pressure field
	void grow();
	void updateSize();
	void manageEPS();
	void checkDivisionAndDeath();
	void applyDivisionAndDeath();
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
