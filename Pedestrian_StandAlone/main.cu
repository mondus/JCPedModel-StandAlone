
/*
* FLAME GPU v 1.4.0 for CUDA 6
* Copyright 2015 University of Sheffield.
* Author: Dr Paul Richmond
* Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
*
* University of Sheffield retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* University of Sheffield is strictly prohibited.
*
* For terms of licence agreement please attached licence or view licence
* on www.flamegpu.com website.
*
*/
#include <cuda_runtime.h>
#include <stdio.h>
#ifdef VISUALISATION
#include <GL/glew.h>
#include <GL/glut.h>
#endif
#include "header.h"
#include "cuda_profiler_api.h"
//#include "auxillaries.c"

/* IO Variables*/
char inputfile[100];          /**< Input path char buffer*/
char outputpath[100];         /**< Output path char buffer*/

#define OUTPUT_TO_XML 0

#define WRITE_TO_FILE 0

/** checkUsage
* Function to check the correct number of arguments
* @param arc	main argument count
* @param argv	main argument values
* @return true if usage is correct, otherwise false
*/
int checkUsage(int argc, char** argv){
	//Check usage
#ifdef VISUALISATION
	printf("FLAMEGPU Visualisation mode\n");
	if (argc < 2)
	{
		printf("Usage: main [XML model data] [Optional CUDA device ID]\n");
		return false;
	}
#else
	printf("FLAMEGPU Console mode\n");
	if (argc < 3)
	{
		printf("Usage: main [XML model data] [Iterations] [Optional CUDA device ID]\n");
		return false;
	}
#endif
	return true;
}


/** setFilePaths
* Function to set global variables for the input XML file and its directory location
*@param input input path of model xml file
*/
void setFilePaths(char* input){
	//Copy input file
	strcpy(inputfile, input);
	printf("Initial states: %s\n", inputfile);

	//Calculate the output path from the path of the input file
	int i = 0;
	int lastd = -1;
	while (inputfile[i] != '\0')
	{
		/* For windows directories */
		if (inputfile[i] == '\\') lastd = i;
		/* For unix directories */
		if (inputfile[i] == '/') lastd = i;
		i++;
	}
	strcpy(outputpath, inputfile);
	outputpath[lastd + 1] = '\0';
	printf("Output dir: %s\n", outputpath);
}


void initCUDA(int argc, char** argv){
	cudaError_t cudaStatus;
	int device;
	int device_count;

	//default device
	device = 0;
	cudaStatus = cudaGetDeviceCount(&device_count);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error finding CUDA devices!  Do you have a CUDA-capable GPU installed?");
		exit(0);
	}
	if (device_count == 0){
		fprintf(stderr, "Error no CUDA devices found!");
		exit(0);
	}

#ifdef VISUALISATION
	if (argc == 3){
		device = atoi(argv[2]);
	}
#else
	if (argc == 4){
		device = atoi(argv[3]);
	}
#endif

	if (device >= device_count){
		fprintf(stderr, "Error selecting CUDA device! Device id '%d' is not found?", device);
		exit(0);
	}

	// Select device
	cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error setting CUDA device!");
		exit(0);
	}
}


/**
* Program main (Handles arguments)
*/
int main(int argc, char** argv)
{
	cudaError_t cudaStatus;
	//check usage mode
	if (!checkUsage(argc, argv))
		exit(0);

	//get the directory paths
	setFilePaths(argv[1]);

	//initialise CUDA
	initCUDA(argc, argv);

#ifdef VISUALISATION
	//Init visualisation must be done before simulation init
	initVisualisation();
#endif

	//initialise the simulation
	initialise(inputfile);


#ifdef VISUALISATION
	runVisualisation();
	exit(0);
#else	
	//Benchmark simulation
	cudaEvent_t start, stop;
	cudaEvent_t loop_start, loop_stop;
	float milliseconds = 0;
	float loopmilliseconds = 0;

	/*if (WRITE_TO_FILE) {
	//Title of file for writing
	char title[] = "900 sort steps.txt";
	//File for writing into
	FILE *filewrite = openFilew(title);
	}*/

	//create timing events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&loop_start);
	cudaEventCreate(&loop_stop);

	//Get the number of iterations
	int iterations = atoi(argv[2]); //err chk...
	if (iterations == 0)
	{
		printf("Second argument must be an integer (Number of Iterations)\n");
		exit(0);
	}

	//Get the number of steps between sorts

	//start timing
	cudaEventRecord(start);

	for (int i = 0; i< iterations; i++)
	{
		//printf("Processing Simulation Step %i\n", i+1);
		//start loop timing
		cudaEventRecord(loop_start);

		//Do profiling when at 1 iteration before end
		if (i == iterations - 1)
		{
			printf("Profiling Simulation Step %i\n", i + 1);
			//start profiler
			cudaProfilerStart();
		}

		//single simulation iteration
		singleIteration();

		//Finish profiling when at 1 iteration before end
		if (i == iterations - 1)
		{
			//end profiling
			cudaProfilerStop();
		}

		//CUDA stop loop timing
		cudaEventRecord(loop_stop);
		cudaEventSynchronize(loop_stop);
		cudaEventElapsedTime(&loopmilliseconds, loop_start, loop_stop);

		/*if (WRITE_TO_FILE) {
		writeToFile(filewrite, loopmilliseconds, 1);
		}*/

		if (OUTPUT_TO_XML)
		{
			saveIterationData(outputpath, i + 1,
				//default state pedestrian agents
				get_host_pedestrian_default_agents(), get_device_pedestrian_default_agents(), get_agent_pedestrian_default_count());

			printf("Iteration %i Saved to XML\n", i + 1);
		}

	}

	//CUDA stop timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Total Processing time: %f (ms)\n", milliseconds);
#endif

	//end profiling
	cudaProfilerStop();
	/*if (WRITE_TO_FILE) {
	fclose(filewrite); //close file
	}*/
	cleanup();
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Error resetting the device!");
		return 1;
	}
}