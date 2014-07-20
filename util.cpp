/*
 * util.cpp
 *
 *  Created on: Jul 20, 2014
 *      Author: bqian
 */

#include <cmath>
#include "util.h"

float radiusOfASphere(float volume)
{
	return cubeRoot(volume * 0.75 / M_PI);
}

float radiusOfACylinder(float volume, float length)
{
	return std::sqrt(volume/length/M_PI);
}

float cubeRoot(float x)
{
	return std::pow(x, (1.0/3.0));
}
