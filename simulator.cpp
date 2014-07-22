/*
 * simulator.cpp
 *
 *  Created on: Jul 18, 2014
 *      Author: bqian
 *
 *  Main simulator class
 */

#include "simulator.h"
#if CUDA
#include "simulator_cuda.cuh"
#endif
#include "constant.h"
#include "util.h"
#include "XMLParser/XMLParser.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

Simulator::~Simulator()
{

}

Simulator::Simulator(XMLParser *parser)
{
	m_initialized = false;
	m_continueRunning = true;
	m_is3D = true;
	// parse and initialize the simulator
	createSimulator(parser);

	// parse and initialize the domains
	parseDomain(parser);

	// parse and initialize the species
	parseSpecies(parser);

	// parse and initialize the solutes
	parseSolutes(parser);
	// when everything is initialized, set flag to true
	m_initialized = true;
}

void Simulator::createSimulator(XMLParser *parser)
{
	XMLParser localRoot(parser->getParamByType("simulator"));

	// read in parameters for the simulator from the protocol file
	isChemostat = localRoot.getParamBool("chemostat");
	isFluctEnv = localRoot.getParamBool("isFluctEnv");
	multiEpi = localRoot.getParamBool("ismultiEpi");
	invComp = localRoot.getParamBool("invComp");
	agentStepSize = localRoot.getParamTime("agentTimeStep");


	if (localRoot.getParam("attachment").empty() && !isChemostat)
	{
		attachmentMechanism = localRoot.getParam("attachment");
	}

	// read in the global timestep from the protocol file
	XMLParser timeStep(localRoot.getParamByType("timeStep"));
	globalStepSize = timeStep.getParamTime("timeStepIni");

}

/*
 * Read in the computational domain parameters from the WORLD markup
 * in the protocol file
 */
void Simulator::parseDomain(XMLParser *parser)
{
	XMLParser localRoot(parser->getParamByType("world").child("computationDomain"));

	// Extract the grid size from the protocol file
	nX = localRoot.getGrid("nI");
	nY = localRoot.getGrid("nJ");
	nZ = localRoot.getGrid("nK");

	// Extract the resolution of the grid points in the domain
	// from the protocol file
	domainResolution = localRoot.getParamLength("resolution");

	// Compute the physical size of the domain
	domainX = nX * domainResolution;
	domainY = nY * domainResolution;
	domainZ = nZ * domainResolution;

	// Extract the depth of the boundary layer from the protocol file
	boundaryLayer = localRoot.getParamLength("boundaryLayer");
}

/*
 * Read in the species parameters from the SPECIES markup
 * in the protocol file
 */
void Simulator::parseSpecies(XMLParser *parser)
{
	h_numAgents = 0;
	int numSpecies = 0;	// total number of species specified

	// allocate space to store the species and their associated
	// parameters
	h_radius = (float *) malloc(sizeof(float)*MAX_NUM_AGENTS);
	h_totalRadius = (float *) malloc(sizeof(float)*MAX_NUM_AGENTS);
	h_divRadius = (float *) malloc(sizeof(float)*MAX_NUM_AGENTS);
	h_deathRadius = (float *) malloc(sizeof(float)*MAX_NUM_AGENTS);
	h_species = (int *) malloc(sizeof(int)*MAX_NUM_SPECIES);
	speciesDic = (char **) malloc(sizeof(char *)*MAX_NUM_SPECIES);

	for (int i = 0; i < MAX_NUM_SPECIES; i++)
	{
		speciesDic[i] = (char *) malloc(sizeof(char)*MAX_SPECIES_NAME);
	}

	if (m_is3D)
		h_position = (float *) malloc(sizeof(float)*3*MAX_NUM_AGENTS);
	else
		h_position = (float *) malloc(sizeof(float)*2*MAX_NUM_AGENTS);


	// read the parameters for each species from the protocol file
	for (xml_node species : parser->_localRoot.children("species"))
	{
		int initNumAgents = 0; // number of initial agents for this specie
		float divRadius, deathRadius;
		XMLParser localRoot(species);

		// extract species parameters from the protocol file
		initNumAgents = localRoot.getParamByType("initArea").attribute("number").as_int();
		divRadius = localRoot.getParamLength("divRadius");
		deathRadius = localRoot.getParamLength("deathRadius");

		// store the name of the specie in the species dictionary
		strcpy_s(speciesDic[numSpecies], strlen(species.attribute("name").value()), species.attribute("name").value());

		// store the index of this specie in the species dictionary
		h_species[numSpecies] = numSpecies;

		// store the species parameter for each agent to be
		// created
		for (int i = 0; i < initNumAgents; i++)
		{
			h_radius[i + h_numAgents] = 0;
			h_totalRadius[i + h_numAgents] = 0;
			h_divRadius[i + h_numAgents] = divRadius;
			h_deathRadius[i + h_numAgents] = deathRadius;
		}

		// keep track of the total number of agents
		h_numAgents += initNumAgents;

		// TODO: Create the agents for this specie
	}
}

/*
 * Read in the solute parameters for the bulk region from
 * the WORLD markup in the protocol file and the solute parameters
 * for the domain from the SOLUTE markup in the protocol file
 */
void Simulator::parseSolutes(XMLParser *parser)
{
	int numSolutes = 0, numBulkSolutes = 0, numDomainSolutes = 0;
	XMLParser bulkRoot(parser->getParamByType("world").child("bulk"));

	// allocate space for the solutes in the bulk region
	bulkSoluteName = (int *) malloc(sizeof(int)*MAX_NUM_SOLUTES);

	// allocate space for the solute dictionary
	soluteDic = (char **) malloc(sizeof(char *)*MAX_NUM_SOLUTES);
	for (int i = 0; i < MAX_NUM_AGENTS; i++)
	{
		soluteDic[i] = (char *) malloc(sizeof(char)*MAX_SOLUTES_NAME);
	}

	// extract solute parameters for solutes in the bulk region
	for (xml_node param : bulkRoot._localRoot.children("solute"))
	{
		// store the name of the solute in the solute dictionary
		strcpy_s(soluteDic[numSolutes], strlen(param.attribute("name").value()), param.attribute("name").value());

		// store the index of the solute in the solute dictionary
		bulkSoluteName[numBulkSolutes] = numSolutes;

		numSolutes++;
		numBulkSolutes++;
	}

	// allocate space for the solutes in the domain
	soluteName = (int *) malloc(sizeof(int)*MAX_NUM_SOLUTES);
	for (xml_node param : parser->_localRoot.child("solute"))
	{
		// store the name of the solute in the solute dictionary
		strcpy_s(soluteDic[numSolutes], strlen(param.attribute("name").value()), param.attribute("name").value());

		// store the index of the solute in the solute dictionary
		soluteName[numDomainSolutes] = numSolutes;

		numSolutes++;
		numDomainSolutes++;
	}

	// TODO: parse the solute concentrations
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
	float catYield = 0.0f;
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
		catYield = 0.0f;

		for (int iReac = 0; iReac < h_numReactionActive[index]; iReac++)
		{
			reacIndex = h_reactionActive[index][iReac];
			catMass = h_particleMass[index][catalystIndex[reacIndex]];
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
			//totalVolume += h_particleMass[index][k]/particleDensity[i]; //TODO: check connection between particleDensity and species, and define a class variable properly
		}

		//updateRadius();
		if(m_is3D)
		{
			//TODO: not sure why they need both radius and totalRadius
			// in updateVolume() _volume and _totalVolume are equal
			// keep it here for now
			h_radius[index] = radiusOfASphere(totalVolume); // in util.cpp
			h_totalRadius[index] = radiusOfASphere(totalVolume);
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
		if(h_hasEps[index])
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
