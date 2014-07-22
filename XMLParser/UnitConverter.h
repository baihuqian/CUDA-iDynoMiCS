/*
*	UnitConverter.h
*
*	Created on: Jul 20, 2014
*	Author: yhuo
*
*	Static class unsed to convert units (mass, length, time, volume)
*/

class UnitConverter
{
public:
	static float time(const char* timeUnit);
	static float length(const char* lengthUnit);
	static float mass(const char* massUnit);
	static float volume(const char* volumeUnit);
};