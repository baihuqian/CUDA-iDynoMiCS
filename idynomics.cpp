/*
 * idynomics.c++
 *
 *  Created on: Jul 18, 2014
 *      Author: bqian
 */
#include <iostream>


#include "simulator.h"
#include "constant.h"
#include "XMLParser/XMLParser.h"


/**
 * process input arguments and pass them to the parser
 * input: XMLParser is passed by reference
 */

void processArguments(int argc, char **argv, XMLParser *parser)
{
	for(int i = 0; i < argc; i++)
	{
		// check validation of each argument

		// pass filename to parser
		bool res = parser->loadFile(*(argv + i));
		if (res)
			std::cout << *(argv + i) << " processed successfully" << std::endl;
		else
			std::cout << "Unable to open " << *(argv + i) << std::endl;
		// parser read it

		// store data
	}
}

/**
 * main function of the simulator
 * take names of protocol file in arguments
 */
int main(int argc, char **argv)
{
	/* parse argument and pass it to XML parser */
	XMLParser *parser = new XMLParser(); // change to proper class name
	processArguments(argc, argv, parser);

	if(DEBUG)
	{
		std::cout<<"Input file processed successfully"<<std::endl;
	}

	/* initialize simulator */
	Simulator *simulator = new Simulator(parser);

	/* start simulator */
	simulator->start();

	/* clean up if simulator is stopped */

}



