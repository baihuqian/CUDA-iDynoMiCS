/*
 *  SimTimer.h
 *
 *  Created on: Jul 22, 2014
 *      Author: yhuo
 */

#ifndef SIMTIMER_H_
#define SIMTIMER_H_

#include "XMLParser/XMLParser.h"
#include "simulator.h"
#include <cmath>

class SimTimer
{
	static int _nIter;
	static float _dT;
	static bool isAdaptive;
	static float _dTMax;
	static float _dTMin;
	static float *_oldStep;
	static float _now;
	static float _endOfSimulation;

public:
	SimTimer(XMLParser *localRoot);
	float _statedTimeStep;
	static void applyTimeStep();
	static void updateTimeStep(Simulator *sim);
	static void reset();
	static float getCurrentTime();
	static int getCurrentIter();
	static float getCurrentTimeStep();
	static void setCurrentTimeStep(float dt);
	void setTimerState(const char *infoFile);
	static bool simIsFinished();
	float returnSimulationLength();
	static bool isDuringNextStep(float aDate);
	static bool isCurrentTimeStepInSetInRange(float start, float end);
};

#endif /* SIMTIMER_H_ */