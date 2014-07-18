/*
 * simulator.cpp
 *
 *  Created on: Jul 18, 2014
 *      Author: bqian
 *
 *  Main simulator class
 */

#include "simulator.h"
#include "simulator_cuda.cuh"
#include "constant.h"
#include <assert.h>

Simulator::Simulator()	{	}

Simulator::Simulator(XMLParser parser)
{
	//TODO: Jack: use parser to initialize all variables, allocate array space

	// when everything is initialized, set flag to true
	m_initialized = true;
}

void Simulator::start()
{
	assert(m_initialized);

	while(m_continueRunning)
	{
		step();
	}
}

void Simulator::step()
{
	// simulation time

	// check if new agents should be created
	checkAgentBirth();

	// solve diffusion-reaction relaxation
	solveDiffusion();

	// perform agent step
	agentStep();
}

void Simulator::checkAgentBirth()
{

}

void Simulator::solveDiffusion()
{

}

void Simulator::agentStep()
{
	float elapsedTime = 0.0f;

	while (elapsedTime < globalStepSize)
	{
		elapsedTime += agentStepSize;

		followPressure();

		agentStepDevice();

		shoveAll();

		afterShove();
	}
}

#if CUDA == 0 // place replacement of CUDA code here
void Simulator::agentStepDevice()
{

}

/**
 * shoving algorithm
 */
void Simulator::shoveAll()
{

}
#endif

/**
 * deal with any operation after the shoving operation
 */
void Simulator::afterShove()
{

}

/**
 * convert physical 2D index to array index
 */
int Simulator::gridIndex2D(int X, int Y)
{
	return 0;
}

/**
 * convert physical 3D index to array index
 */
int Simulator::gridIndex3D(int X, int Y, int Z)
{
	return 0;
}
