/*
 *	XMLParser.h
 *
 *	Created on: July 18, 2014
 *  Author: yhuo
 *
 *	All the functions only look at child nodes
 *
 */
#ifndef XMLPARSER_H_
#define XMLPARSER_H_

#include "pugixml.hpp"

using namespace pugi;

class XMLParser {
	// owner of the entire document structure, 
	// destroying the document destroys the whole tree
	// only used to load a file
	xml_document root;

public:
	XMLParser(); // create a parser that can access the entire protocol file
	
	// create a parser the can access the localRoot part of the protocol file
	XMLParser(xml_node localRoot);
	
	xml_node _localRoot; // the root node of this parser

	bool loadFile(const char *fileName); // load a protocol file
	
	// return the node <paramType>....</paramType>
	xml_node getParamByType(const char *paramType);

	// return the parameter as a boolean
	bool getParamBool(const char* paramName);

	// return the time parameter as a float
	float getParamTime(const char* paramName);

	// return the parameter as a string
	std::string getParam(const char* paramName);

	// return the value of one of the dimensions of the grid
	//
	// @param dim - can be either x, y, or z
	int getGrid(const char* dim);
	
	// return parameters with a length unit as a float
	float getParamLength(const char* paramName);
};

#endif
