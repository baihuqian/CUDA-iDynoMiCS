/*
 *  SimTimer.cpp
 *
 *  Created on: Jul 22, 2014
 *      Author: yhuo
 *
 *  Class to create and keep track of the timestep and simulation time course.
 */

#include "SimTimer.h"
#include "simulator.h"
#include <stdlib.h>
#include <cmath>

// in SimTimer.h static class variables are ONLY DECLARED, so it should be DEFINED in the implementation
int SimTimer::_nIter = 0;;
float SimTimer::_dT = 0.0f;
bool SimTimer::isAdaptive = false;
float SimTimer::_dTMax = 0.0f;
float SimTimer::_dTMin = 0.0f;
float *SimTimer::_oldStep = NULL;
float SimTimer::_now = 0.0f;
float SimTimer::_endOfSimulation = 0.0f;



/**
 * Parses simulation time step information from the XML protocol file
 *
 * The timeStep markup within the protocol file specified all parameters
 * for the time-stepping of the simulation. This action is controlled by an
 * object of this SimTimer class. There is a choice of using an adaptive or
 * set timestep, specified by setting the adaptive parameter to true or
 * false respectively. If an adaptive timestep is used, three timeStep
 * parameters are set to control the initial, minimum, and maximum values
 * the timestep may take. The simulation then determines during runtime the
 * timestep that should be used. On the other hand, if this is set to
 * false, the simulation runs at one timestep - that set in the parameter
 * timeStepIni. A further parameter is read in, endOfSimulation, that
 * specifies when the simulation should end.
 *
 */
SimTimer::SimTimer(XMLParser *localRoot)
{
	XMLParser parser(localRoot->getParamByType("simulator").child("timeStep"));

	// set all counters to zero
	reset();

	// initialize the timer
	_endOfSimulation = parser.getParamTime("endOfSimulation");
	_statedTimeStep = parser.getParamTime("timeStepIni");
	_dT = _statedTimeStep;

	// now determine if adaptive timesteps are being used
	isAdaptive = parser.getParamBool("adaptive");

	if (isAdaptive)
	{
		_dTMax = parser.getParamTime("timeStepMax");
		_dTMin = parser.getParamTime("timeStepMin");
		_oldStep = (float *) malloc(sizeof(float)*10);
	}
}

/**
 * Computes the current simulation time step and increments
 * the simulation clock.
 *
 * First method called by the simulation step method.
 */
void SimTimer::applyTimeStep()
{
	_now += _dT;
	_nIter++;
}

/**
 * If adaptive timesteps are enabled, updates the timestep for a 
 * given simulation world
 */
void SimTimer::updateTimeStep(Simulator *sim)
{
	if (!isAdaptive)
		return;

	float tOpt, newDeltaT;

	//tOpt = sim->getBulkTimeConstraint(); //TODO: check this function
	tOpt = 0;	// place holder

	if (!std::isfinite(tOpt) || isnan(tOpt))
		return;

	// constrain value between min and max limits
	newDeltaT = fmin(fmax(tOpt, _dTMin), _dTMax);

	// If dT needs to rise, increase it gradually. Otherwise, it must be
	// staying the same (no change) or instantly dropping to a new value.
	if (newDeltaT > _dT)
	{
		// Make the new option the average of 10 previous steps.
		for (int i = 1; i < 10; i++)
			_oldStep[i] = _oldStep[i - 1];
		_oldStep[0] = newDeltaT;

		// Again make sure the step isn't too small or too large.
		_dT = _oldStep[0] = fmin(fmax(newDeltaT, _dTMin), _dTMax);
	}
	else
	{
		_dT = newDeltaT;

		// In this case, we also need to re-populate the saved steps so
		// that we don't use too-large values to raise the step again.
		for (int i = 0; i < 10; i++)
			_oldStep[i] = _dT;
	}

	// Make the step into a nicer number.
	_dT = floor(10.0*_dT / _dTMin) * _dTMin * 0.1;
}

/**
 *	Resets the simulation timestep and counters to zero
 */
void SimTimer::reset()
{
	_now = 0.0;
	_nIter = 0;
}

/**
 *	Return the time spent since the beginning of the simulation 
 * (in hours)
 */
float SimTimer::getCurrentTime()
{
	return _now;
}

/**
 *	Returns the current iteration of the simulation
 */
int SimTimer::getCurrentIter()
{
	return _nIter;
}

/**
 * Returns the current timestep, in hours
 */
float SimTimer::getCurrentTimeStep()
{
	return _dT;
}

/**
 *	Sets the current timestep to a new value
 */
void SimTimer::setCurrentTimeStep(float dt)
{
	_dT = dt;
}

/**
 *	Set the state of the simulation timer from information in an XML file
 */
void SimTimer::setTimerState(const char *infoFile)
{
	XMLParser *fileRoot = new XMLParser();
	fileRoot->loadFile(infoFile);

	XMLParser localRoot(fileRoot->getParamByType("simulation"));
	_nIter = localRoot._localRoot.attribute("iterate").as_int();
	_now = localRoot._localRoot.attribute("time").as_float();
}

/**
 *	Determine whether or not the simulation has finished
 */
bool SimTimer::simIsFinished()
{
	return (_now >= _endOfSimulation);
}

/**
 *	Returns the length of the simulation in hours
 */
float SimTimer::returnSimulationLength()
{
	return _endOfSimulation;
}

/**
 *	Determines if a set time (such as an agent birthday) is
 *  within the next timestep
 */
bool SimTimer::isDuringNextStep(float aDate)
{
	return (aDate >= _now) && (aDate < _now + _dT);
}

/**
 *	Determines if the simulation time is within a set range of hours
 */
bool SimTimer::isCurrentTimeStepInSetInRange(float start, float end)
{
	return (start <= _now) && (end >= _now);
}

