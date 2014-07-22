/*
 *	XMLParser.h
 *
 *	Created on: July 18, 2014
 *  Author: yhuo
 */

#include "pugixml.hpp"
#include <iostream>
#include "XMLParser.h"
#include "UnitConverter.h"

using namespace pugi;
using namespace std;

XMLParser::XMLParser() 
{ 
	_localRoot = root.child("idynomics");
}

XMLParser::XMLParser(xml_node localRoot)
{
	_localRoot = localRoot;
}

bool XMLParser::loadFile(const char *fileName) 
{
	return root.load_file(fileName);
}

xml_node XMLParser::getParamByType(const char *paramType) 
{
	return _localRoot.child(paramType);
}

bool XMLParser::getParamBool(const char* paramName) 
{
	for (xml_node param : _localRoot.children("param")) 
	{
		if (strcmp(param.attribute("name").value(), paramName) == 0)
		{
			if (strcmp(param.child_value(), "true") == 0)
				return true;
		}
	}

	return false;
}

float XMLParser::getParamTime(const char* paramName)
{
	float out = 1.0;

	for (xml_node param : _localRoot.children("param"))
	{
		if (strcmp(param.attribute("name").value(), paramName) == 0)
		{
			out = UnitConverter::time(param.attribute("unit").value());
			out *= param.text().as_float();
		}
	}

	return out;
}


float XMLParser::getParamLength(const char* paramName)
{
	float out = 1.0;

	for (xml_node param : _localRoot.children("param"))
	{
		if (strcmp(param.attribute("name").value(), paramName) == 0)
		{
			out = UnitConverter::length(param.attribute("unit").value());
			out *= param.text().as_float();
		}
	}

	return out;
}

string XMLParser::getParam(const char* paramName)
{
	for (xml_node param : _localRoot.children("param"))
	{
		if (strcmp(param.attribute("name").value(), paramName) == 0)
		{
			return string(param.child_value());
		}
	}

	return string();
}

int XMLParser::getGrid(const char* dim)
{
	return _localRoot.child("grid").attribute("dim").as_int();
}






