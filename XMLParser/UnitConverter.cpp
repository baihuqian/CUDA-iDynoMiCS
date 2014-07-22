/*
 *	UnitConverter.cpp
 *
 *	Created on: Jul 20, 2014
 *	Author: yhuo
 *
 *	Static class unsed to convert units (mass, length, time, volume)
 */

#include <iostream>
#include "UnitConverter.h"
using namespace std;

/*
 *	Takes a unit of time and returns a factor to multiply a parameter by to 
 *	obtain the correct unit (hour)
 */
float UnitConverter::time(const char* timeUnit)
{
	string _timeUnit(timeUnit);
	string unit("");
	float out = 1.0;
	
	if (_timeUnit.find("day") != string::npos)
	{
		out = 24.0;
		unit.assign("day");
	}
	if (_timeUnit.find("hour") != string::npos)
	{
		out = 1.0;
		unit.assign("hour");
	}
	if (_timeUnit.find("minute") != string::npos)
	{
		out = (float) (1.0 / 60.0);
		unit.assign("minute");
	}
	if (_timeUnit.find("second") != string::npos)
	{
		out = (float) (1.0 / 3600.0);
		unit.assign("second");
	}
	if (_timeUnit.find(unit.append("-1")) != string::npos)
	{
		out = (float) (1.0 / out);
	}
		
	return out;
}

/*
 *	Takes a length unit and returns a double value of length for that unit
 */
float length(const char* lengthUnit)
{
	string _lengthUnit(lengthUnit);
	string unit("");
	float out = 1.0;

	if (_lengthUnit.find("m") != string::npos)
	{
		out = 1e6;
		unit.assign("m");
	}
	if (_lengthUnit.find("cm") != string::npos)
	{
		out = 1e4;
		unit.assign("cm");
	}
	if (_lengthUnit.find("mm") != string::npos)
	{
		out = 1e3;
		unit.assign("mm");
	}
	if (_lengthUnit.find("µm") != string::npos)
	{
		out = 1.0;
		unit.assign("microm");
	}
	if (_lengthUnit.find("um") != string::npos)
	{
		out = 1.0;
		unit.assign("microm");
	}
	if (_lengthUnit.find("fm") != string::npos)
	{
		out = (float) 1e-9;
		unit.assign("fm");
	}
	if (_lengthUnit.find(unit.append("-1")) != string::npos)
	{
		out = (float) (1.0 / out);
	}

	return out;
}

/*
 *	Takes a mass unit and returns a double value of mass for that unit
 */
float UnitConverter::mass(const char* massUnit)
{
	string _massUnit(massUnit);
	string unit("");
	float out = 1.0;

	if (_massUnit.find("g") != string::npos)
	{
		out = (float) 1e15;
		unit.assign("g");
	}
	if (_massUnit.find("kg") != string::npos)
	{
		out = (float) 1e18;
		unit.assign("kg");
	}
	if (_massUnit.find("mg") != string::npos)
	{
		out = (float) 1e12;
		unit.assign("mg");
	}
	if (_massUnit.find("µg") != string::npos)
	{
		out = (float) 1e9;
		unit.assign("µg");
	}
	if (_massUnit.find("�g") != string::npos)
	{
		out = (float) 1e9;
		unit.assign("µg");
	}
	if (_massUnit.find("fg") != string::npos)
	{
		out = 1;
		unit.assign("fg");
	}
	if (_massUnit.find(unit.append("-1")) != string::npos)
	{
		out = (float) (1.0 / out);
	}

	return out;
}

/*
 *	Takes a mass unit and returns a double value of volume for that unit
 */
float UnitConverter::volume(const char* volumeUnit)
{
	string _volumeUnit(volumeUnit);
	string unit("");
	float out = 1.0;

	if (_volumeUnit.find("m") != string::npos)
	{
		out = (float) 1e18;
		unit.assign("m");
	}
	if (_volumeUnit.find("L") != string::npos)
	{
		out = (float) 1e15;
		unit.assign("L");
	}
	if (_volumeUnit.find("µm") != string::npos)
	{
		out = 1.0;
		unit.assign("µm");
	}
	if (_volumeUnit.find("�m") != string::npos)
	{
		out = 1.0;
		unit.assign("µm");
	}
	if (_volumeUnit.find(unit.append("-1")) != string::npos)
	{
		out = (float) (1.0 / out);
	}

	return out;
}