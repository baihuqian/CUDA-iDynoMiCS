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
#include "util.h"
#include <assert.h>
#include <cmath>

Simulator::~Simulator()
{

}

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

void Simulator::followPressure()
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
/**
 * perform agent step
 * growth, division, death, etc.
 */
void Simulator::agentStepDevice()
{
	respondToConditions();
	updateActiveReactions();
	grow();
	updateSize();

	manageEPS();

	// Baihu: my version of divide and die
	// TODO: implement checkDivisionAndDeath() and applyDivisionAndDeath();
	checkDivisionAndDeath();
	applyDivisionAndDeath();

}

/**
 * line-by-line replica of ActiveAgent.grow()
 */
void Simulator::grow()
{
	float deltaMass = 0.0f;
	float *deltaParticle = new float[numParticle];
	int reacIndex = 0;
	float catMass = 0.0f; //catalyst mass
	float catyield = 0.0f;
	float tStep;

	for(int index = 0; index < h_numAgents; index++)
	{
		deltaMass = 0.0f;
		for(int k = 0; k < numParticle; k++)
		{
			deltaParticle[k] = 0.0f;
		}
		reacIndex = 0;
		tStep = 0; //TODO: set up simTimer and set to SimTimer.getCurrentTimeStep();
		catMass = 0.0f; //catalyst mass
		catyield = 0.0f;

		for (int iReac = 0; iReac < h_numReactionActive[index]; iReac++)
		{
			reacIndex = h_reactionActive[index][iReac];
			catMass = h_particleMass[catalystIndex[reacIndex]];
			float growthRate = 0; //TODO:line 370. understand Reaction.computeSpecGrowthRate()

			for(int i = 0; i < numParticle; i++)
			{
				if(autocatalytic[reacIndex])
				{
					// Exponential growth/decay
					catYield = particleYield[reacIndex][catalystIndex[reacIndex]];
					deltaParticle[i] += catMass * (particleYield[reacIndex][i]/catYield)
											* (catYield * growthRate * tStep);
					deltaMass = deltaParticle[i]/tStep;
				}
				else
				{
					// Constant growth/decay
					deltaMass = catMass * particleYield[reacIndex][i] * growthRate;
					deltaParticle[i] += deltaMass * tStep;
				}
				// netGrowthRate and netVolumeRate are used only for outputting
			}
			for(int i = 0; i < numParticle; i++)
			{
				h_particleMass[index][i] += deltaParticle[i];
			}
		}
	}
	delete [] deltaParticle; // explicitly release memory
}

void Simulator::updateSize()
{
	for(int index = 0; index < h_numAgents; index++)
	{
		//updateMass();
		float totalMass = 0.0f;
		for(int k = 0; k < numParticle; k++)
		{
			totalMass += h_particleMass[index][k];
		}

		//updateVolume();
		float totalVolume = 0.0f;
		for(int k = 0; k < numParticle; k++)
		{
			totalVolume += h_particleMass[index][k]/particleDensity[i]; //TODO: check connection between particleDensity and species, and define a class variable properly
		}

		//updateRadius();
		if(m_is3D)
		{
			//TODO: not sure why they need both radius and totalRadius
			// in updateVolume() _volume and _totalVolume are equal
			// keep it here for now
			h_radius[index] = radiusOfASphere(totalVolume); // in util.cpp
			h_totalRadius[index] = radiusOfAShpere(totalVolume);
		}
		else
		{
			h_radius[index] = radiusOfACylinder(totalVolume, domainZ); // in util.cpp
			h_totalRadius[index] = radiusOfACylinder(totalVolume, domainZ);
		}

		// updateAttachment();
		//TODO: add updateAttachment()
	}

}

void Simulator::manageEPS()
{//TODO: manageEPS is not done
	for(int index = 0; index < h_numAgents; index++)
	{
		if(h_hasEPS[index])
		{
			int epsIndex = numParticle - 1;
			if(h_particleMass[index][epsIndex] != 0)
			{
				float deltaM = 1 - std::exp(0); //TODO: line 83 in BactEPS, understand kHyd
				//TODO: findCloseSiblings() include finding neighbors, implement this after the sorting and reordering is finished

			}
		}

	}
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
	return Y * nY + X;
}

/**
 * convert physical 3D index to array index
 */
int Simulator::gridIndex3D(int X, int Y, int Z)
{
	return Z * nY * nX + Y * nY + X;
}
