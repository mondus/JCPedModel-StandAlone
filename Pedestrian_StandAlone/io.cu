
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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <limits.h>
	

// include header
#include "header.h"

glm::vec3 agent_maximum;
glm::vec3 agent_minimum;

void readIntArrayInput(char* buffer, int *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = atoi(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void readFloatArrayInput(char* buffer, float *array, unsigned int expected_items){
    unsigned int i = 0;
    const char s[2] = ",";
    char * token;

    token = strtok(buffer, s);
    while (token != NULL){
        if (i>=expected_items){
            printf("Error: Agent memory array has too many items, expected %d!\n", expected_items);
            exit(0);
        }
        
        array[i++] = (float)atof(token);
        
        token = strtok(NULL, s);
    }
    if (i != expected_items){
        printf("Error: Agent memory array has %d items, expected %d!\n", i, expected_items);
        exit(0);
    }
}

void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_pedestrian_list* h_pedestrians_default, xmachine_memory_pedestrian_list* d_pedestrians_default, int h_xmachine_memory_pedestrian_default_count)
{
	cudaError_t cudaStatus;
	
	//Device to host memory transfer
	
	cudaStatus = cudaMemcpy( h_pedestrians_default, d_pedestrians_default, sizeof(xmachine_memory_pedestrian_list), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr,"Error Copying pedestrian Agent default State Memory from GPU: %s\n", cudaGetErrorString(cudaStatus));
		exit(cudaStatus);
	}
	
	/* Pointer to file */
	FILE *file;
	char data[100];

	sprintf(data, "%s%i.xml", outputpath, iteration_number);
	//printf("Writing iteration %i data to %s\n", iteration_number, data);
	file = fopen(data, "w");
	fputs("<states>\n<itno>", file);
	sprintf(data, "%i", iteration_number);
	fputs(data, file);
	fputs("</itno>\n", file);
	fputs("<environment>\n" , file);
	fputs("</environment>\n" , file);

	//Write each pedestrian agent to xml
	for (int i=0; i<h_xmachine_memory_pedestrian_default_count; i++){
		fputs("<xagent>\n" , file);
		fputs("<name>pedestrian</name>\n", file);
        
		fputs("<x>", file);
        sprintf(data, "%f", h_pedestrians_default->x[i]);
		fputs(data, file);
		fputs("</x>\n", file);
        
		fputs("<y>", file);
        sprintf(data, "%f", h_pedestrians_default->y[i]);
		fputs(data, file);
		fputs("</y>\n", file);
        
		fputs("<vx>", file);
        sprintf(data, "%f", h_pedestrians_default->vx[i]);
		fputs(data, file);
		fputs("</vx>\n", file);
        
		fputs("<vy>", file);
        sprintf(data, "%f", h_pedestrians_default->vy[i]);
		fputs(data, file);
		fputs("</vy>\n", file);
        
		fputs("<desvx>", file);
        sprintf(data, "%f", h_pedestrians_default->desvx[i]);
		fputs(data, file);
		fputs("</desvx>\n", file);
        
		fputs("<desvy>", file);
        sprintf(data, "%f", h_pedestrians_default->desvy[i]);
		fputs(data, file);
		fputs("</desvy>\n", file);
        
		fputs("<count>", file);
        sprintf(data, "%i", h_pedestrians_default->count[i]);
		fputs(data, file);
		fputs("</count>\n", file);
        
		fputs("<lineFail>", file);
        sprintf(data, "%i", h_pedestrians_default->lineFail[i]);
		fputs(data, file);
		fputs("</lineFail>\n", file);
        
		fputs("<newvx>", file);
        sprintf(data, "%f", h_pedestrians_default->newvx[i]);
		fputs(data, file);
		fputs("</newvx>\n", file);
        
		fputs("<newvy>", file);
        sprintf(data, "%f", h_pedestrians_default->newvy[i]);
		fputs(data, file);
		fputs("</newvy>\n", file);
        
		fputs("<orcaLine_direction_x>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->orcaLine_direction_x[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</orcaLine_direction_x>\n", file);
        
		fputs("<orcaLine_direction_y>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->orcaLine_direction_y[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</orcaLine_direction_y>\n", file);
        
		fputs("<orcaLine_point_x>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->orcaLine_point_x[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</orcaLine_point_x>\n", file);
        
		fputs("<orcaLine_point_y>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->orcaLine_point_y[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</orcaLine_point_y>\n", file);
        
		fputs("<projLine_direction_x>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->projLine_direction_x[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</projLine_direction_x>\n", file);
        
		fputs("<projLine_direction_y>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->projLine_direction_y[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</projLine_direction_y>\n", file);
        
		fputs("<projLine_point_x>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->projLine_point_x[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</projLine_point_x>\n", file);
        
		fputs("<projLine_point_y>", file);
        for (int j=0;j<128;j++){
            fprintf(file, "%f", h_pedestrians_default->projLine_point_y[(j*xmachine_memory_pedestrian_MAX)+i]);
            if(j!=(128-1))
                fprintf(file, ",");
        }
		fputs("</projLine_point_y>\n", file);
        
		fputs("</xagent>\n", file);
	}
	
	

	fputs("</states>\n" , file);
	
	/* Close the file */
	fclose(file);
}

void readInitialStates(char* inputpath, xmachine_memory_pedestrian_list* h_pedestrians, int* h_xmachine_memory_pedestrian_count)
{

	int temp = 0;
	int* itno = &temp;

	/* Pointer to file */
	FILE *file;
	/* Char and char buffer for reading file to */
	char c = ' ';
	char buffer[10000];
	char agentname[1000];

	/* Pointer to x-memory for initial state data */
	/*xmachine * current_xmachine;*/
	/* Variables for checking tags */
	int reading, i;
	int in_tag, in_itno, in_name;
    int in_pedestrian_x;
    int in_pedestrian_y;
    int in_pedestrian_vx;
    int in_pedestrian_vy;
    int in_pedestrian_desvx;
    int in_pedestrian_desvy;
    int in_pedestrian_count;
    int in_pedestrian_lineFail;
    int in_pedestrian_newvx;
    int in_pedestrian_newvy;
    int in_pedestrian_orcaLine_direction_x;
    int in_pedestrian_orcaLine_direction_y;
    int in_pedestrian_orcaLine_point_x;
    int in_pedestrian_orcaLine_point_y;
    int in_pedestrian_projLine_direction_x;
    int in_pedestrian_projLine_direction_y;
    int in_pedestrian_projLine_point_x;
    int in_pedestrian_projLine_point_y;

	/* for continuous agents: set agent count to zero */	
	*h_xmachine_memory_pedestrian_count = 0;
	
	/* Variables for initial state data */
	float pedestrian_x;
	float pedestrian_y;
	float pedestrian_vx;
	float pedestrian_vy;
	float pedestrian_desvx;
	float pedestrian_desvy;
	int pedestrian_count;
	int pedestrian_lineFail;
	float pedestrian_newvx;
	float pedestrian_newvy;
    float pedestrian_orcaLine_direction_x[128];
    float pedestrian_orcaLine_direction_y[128];
    float pedestrian_orcaLine_point_x[128];
    float pedestrian_orcaLine_point_y[128];
    float pedestrian_projLine_direction_x[128];
    float pedestrian_projLine_direction_y[128];
    float pedestrian_projLine_point_x[128];
    float pedestrian_projLine_point_y[128];
	
	/* Open config file to read-only */
	if((file = fopen(inputpath, "r"))==NULL)
	{
		printf("Error opening initial states\n");
		exit(0);
	}
	
	/* Initialise variables */
    agent_maximum.x = 0;
    agent_maximum.y = 0;
    agent_maximum.z = 0;
    agent_minimum.x = 0;
    agent_minimum.y = 0;
    agent_minimum.z = 0;
	reading = 1;
	in_tag = 0;
	in_itno = 0;
	in_name = 0;
	in_pedestrian_x = 0;
	in_pedestrian_y = 0;
	in_pedestrian_vx = 0;
	in_pedestrian_vy = 0;
	in_pedestrian_desvx = 0;
	in_pedestrian_desvy = 0;
	in_pedestrian_count = 0;
	in_pedestrian_lineFail = 0;
	in_pedestrian_newvx = 0;
	in_pedestrian_newvy = 0;
	in_pedestrian_orcaLine_direction_x = 0;
	in_pedestrian_orcaLine_direction_y = 0;
	in_pedestrian_orcaLine_point_x = 0;
	in_pedestrian_orcaLine_point_y = 0;
	in_pedestrian_projLine_direction_x = 0;
	in_pedestrian_projLine_direction_y = 0;
	in_pedestrian_projLine_point_x = 0;
	in_pedestrian_projLine_point_y = 0;
	//set all pedestrian values to 0
	//If this is not done then it will cause errors in emu mode where undefined memory is not 0
	for (int k=0; k<xmachine_memory_pedestrian_MAX; k++)
	{	
		h_pedestrians->x[k] = 0;
		h_pedestrians->y[k] = 0;
		h_pedestrians->vx[k] = 0;
		h_pedestrians->vy[k] = 0;
		h_pedestrians->desvx[k] = 0;
		h_pedestrians->desvy[k] = 0;
		h_pedestrians->count[k] = 0;
		h_pedestrians->lineFail[k] = 0;
		h_pedestrians->newvx[k] = 0;
		h_pedestrians->newvy[k] = 0;
        for (i=0;i<128;i++){
            h_pedestrians->orcaLine_direction_x[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
        for (i=0;i<128;i++){
            h_pedestrians->orcaLine_direction_y[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
        for (i=0;i<128;i++){
            h_pedestrians->orcaLine_point_x[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
        for (i=0;i<128;i++){
            h_pedestrians->orcaLine_point_y[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
        for (i=0;i<128;i++){
            h_pedestrians->projLine_direction_x[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
        for (i=0;i<128;i++){
            h_pedestrians->projLine_direction_y[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
        for (i=0;i<128;i++){
            h_pedestrians->projLine_point_x[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
        for (i=0;i<128;i++){
            h_pedestrians->projLine_point_y[(i*xmachine_memory_pedestrian_MAX)+k] = 0;
        }
	}
	

	/* Default variables for memory */
    pedestrian_x = 0;
    pedestrian_y = 0;
    pedestrian_vx = 0;
    pedestrian_vy = 0;
    pedestrian_desvx = 0;
    pedestrian_desvy = 0;
    pedestrian_count = 0;
    pedestrian_lineFail = 0;
    pedestrian_newvx = 0;
    pedestrian_newvy = 0;
    for (i=0;i<128;i++){
        pedestrian_orcaLine_direction_x[i] = 0;
    }
    for (i=0;i<128;i++){
        pedestrian_orcaLine_direction_y[i] = 0;
    }
    for (i=0;i<128;i++){
        pedestrian_orcaLine_point_x[i] = 0;
    }
    for (i=0;i<128;i++){
        pedestrian_orcaLine_point_y[i] = 0;
    }
    for (i=0;i<128;i++){
        pedestrian_projLine_direction_x[i] = 0;
    }
    for (i=0;i<128;i++){
        pedestrian_projLine_direction_y[i] = 0;
    }
    for (i=0;i<128;i++){
        pedestrian_projLine_point_x[i] = 0;
    }
    for (i=0;i<128;i++){
        pedestrian_projLine_point_y[i] = 0;
    }

	/* Read file until end of xml */
    i = 0;
	while(reading==1)
	{
		/* Get the next char from the file */
		c = (char)fgetc(file);
		
		/* If the end of a tag */
		if(c == '>')
		{
			/* Place 0 at end of buffer to make chars a string */
			buffer[i] = 0;
			
			if(strcmp(buffer, "states") == 0) reading = 1;
			if(strcmp(buffer, "/states") == 0) reading = 0;
			if(strcmp(buffer, "itno") == 0) in_itno = 1;
			if(strcmp(buffer, "/itno") == 0) in_itno = 0;
			if(strcmp(buffer, "name") == 0) in_name = 1;
			if(strcmp(buffer, "/name") == 0) in_name = 0;
			if(strcmp(buffer, "/xagent") == 0)
			{
				if(strcmp(agentname, "pedestrian") == 0)
				{		
					if (*h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX){
						printf("ERROR: MAX Buffer size (%i) for agent pedestrian exceeded whilst reading data\n", xmachine_memory_pedestrian_MAX);
						// Close the file and stop reading
						fclose(file);
						exit(0);
					}
                    
					h_pedestrians->x[*h_xmachine_memory_pedestrian_count] = pedestrian_x;//Check maximum x value
                    if(agent_maximum.x < pedestrian_x)
                        agent_maximum.x = (float)pedestrian_x;
                    //Check minimum x value
                    if(agent_minimum.x > pedestrian_x)
                        agent_minimum.x = (float)pedestrian_x;
                    
					h_pedestrians->y[*h_xmachine_memory_pedestrian_count] = pedestrian_y;//Check maximum y value
                    if(agent_maximum.y < pedestrian_y)
                        agent_maximum.y = (float)pedestrian_y;
                    //Check minimum y value
                    if(agent_minimum.y > pedestrian_y)
                        agent_minimum.y = (float)pedestrian_y;
                    
					h_pedestrians->vx[*h_xmachine_memory_pedestrian_count] = pedestrian_vx;
					h_pedestrians->vy[*h_xmachine_memory_pedestrian_count] = pedestrian_vy;
					h_pedestrians->desvx[*h_xmachine_memory_pedestrian_count] = pedestrian_desvx;
					h_pedestrians->desvy[*h_xmachine_memory_pedestrian_count] = pedestrian_desvy;
					h_pedestrians->count[*h_xmachine_memory_pedestrian_count] = pedestrian_count;
					h_pedestrians->lineFail[*h_xmachine_memory_pedestrian_count] = pedestrian_lineFail;
					h_pedestrians->newvx[*h_xmachine_memory_pedestrian_count] = pedestrian_newvx;
					h_pedestrians->newvy[*h_xmachine_memory_pedestrian_count] = pedestrian_newvy;
                    for (int k=0;k<128;k++){
                        h_pedestrians->orcaLine_direction_x[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_orcaLine_direction_x[k];    
                    }
                    for (int k=0;k<128;k++){
                        h_pedestrians->orcaLine_direction_y[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_orcaLine_direction_y[k];    
                    }
                    for (int k=0;k<128;k++){
                        h_pedestrians->orcaLine_point_x[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_orcaLine_point_x[k];    
                    }
                    for (int k=0;k<128;k++){
                        h_pedestrians->orcaLine_point_y[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_orcaLine_point_y[k];    
                    }
                    for (int k=0;k<128;k++){
                        h_pedestrians->projLine_direction_x[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_projLine_direction_x[k];    
                    }
                    for (int k=0;k<128;k++){
                        h_pedestrians->projLine_direction_y[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_projLine_direction_y[k];    
                    }
                    for (int k=0;k<128;k++){
                        h_pedestrians->projLine_point_x[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_projLine_point_x[k];    
                    }
                    for (int k=0;k<128;k++){
                        h_pedestrians->projLine_point_y[(k*xmachine_memory_pedestrian_MAX)+(*h_xmachine_memory_pedestrian_count)] = pedestrian_projLine_point_y[k];    
                    }
					(*h_xmachine_memory_pedestrian_count) ++;	
				}
				else
				{
					printf("Warning: agent name undefined - '%s'\n", agentname);
				}
				

				
				/* Reset xagent variables */
                pedestrian_x = 0;
                pedestrian_y = 0;
                pedestrian_vx = 0;
                pedestrian_vy = 0;
                pedestrian_desvx = 0;
                pedestrian_desvy = 0;
                pedestrian_count = 0;
                pedestrian_lineFail = 0;
                pedestrian_newvx = 0;
                pedestrian_newvy = 0;
                for (i=0;i<128;i++){
                    pedestrian_orcaLine_direction_x[i] = 0;
                }
                for (i=0;i<128;i++){
                    pedestrian_orcaLine_direction_y[i] = 0;
                }
                for (i=0;i<128;i++){
                    pedestrian_orcaLine_point_x[i] = 0;
                }
                for (i=0;i<128;i++){
                    pedestrian_orcaLine_point_y[i] = 0;
                }
                for (i=0;i<128;i++){
                    pedestrian_projLine_direction_x[i] = 0;
                }
                for (i=0;i<128;i++){
                    pedestrian_projLine_direction_y[i] = 0;
                }
                for (i=0;i<128;i++){
                    pedestrian_projLine_point_x[i] = 0;
                }
                for (i=0;i<128;i++){
                    pedestrian_projLine_point_y[i] = 0;
                }

			}
			if(strcmp(buffer, "x") == 0) in_pedestrian_x = 1;
			if(strcmp(buffer, "/x") == 0) in_pedestrian_x = 0;
			if(strcmp(buffer, "y") == 0) in_pedestrian_y = 1;
			if(strcmp(buffer, "/y") == 0) in_pedestrian_y = 0;
			if(strcmp(buffer, "vx") == 0) in_pedestrian_vx = 1;
			if(strcmp(buffer, "/vx") == 0) in_pedestrian_vx = 0;
			if(strcmp(buffer, "vy") == 0) in_pedestrian_vy = 1;
			if(strcmp(buffer, "/vy") == 0) in_pedestrian_vy = 0;
			if(strcmp(buffer, "desvx") == 0) in_pedestrian_desvx = 1;
			if(strcmp(buffer, "/desvx") == 0) in_pedestrian_desvx = 0;
			if(strcmp(buffer, "desvy") == 0) in_pedestrian_desvy = 1;
			if(strcmp(buffer, "/desvy") == 0) in_pedestrian_desvy = 0;
			if(strcmp(buffer, "count") == 0) in_pedestrian_count = 1;
			if(strcmp(buffer, "/count") == 0) in_pedestrian_count = 0;
			if(strcmp(buffer, "lineFail") == 0) in_pedestrian_lineFail = 1;
			if(strcmp(buffer, "/lineFail") == 0) in_pedestrian_lineFail = 0;
			if(strcmp(buffer, "newvx") == 0) in_pedestrian_newvx = 1;
			if(strcmp(buffer, "/newvx") == 0) in_pedestrian_newvx = 0;
			if(strcmp(buffer, "newvy") == 0) in_pedestrian_newvy = 1;
			if(strcmp(buffer, "/newvy") == 0) in_pedestrian_newvy = 0;
			if(strcmp(buffer, "orcaLine_direction_x") == 0) in_pedestrian_orcaLine_direction_x = 1;
			if(strcmp(buffer, "/orcaLine_direction_x") == 0) in_pedestrian_orcaLine_direction_x = 0;
			if(strcmp(buffer, "orcaLine_direction_y") == 0) in_pedestrian_orcaLine_direction_y = 1;
			if(strcmp(buffer, "/orcaLine_direction_y") == 0) in_pedestrian_orcaLine_direction_y = 0;
			if(strcmp(buffer, "orcaLine_point_x") == 0) in_pedestrian_orcaLine_point_x = 1;
			if(strcmp(buffer, "/orcaLine_point_x") == 0) in_pedestrian_orcaLine_point_x = 0;
			if(strcmp(buffer, "orcaLine_point_y") == 0) in_pedestrian_orcaLine_point_y = 1;
			if(strcmp(buffer, "/orcaLine_point_y") == 0) in_pedestrian_orcaLine_point_y = 0;
			if(strcmp(buffer, "projLine_direction_x") == 0) in_pedestrian_projLine_direction_x = 1;
			if(strcmp(buffer, "/projLine_direction_x") == 0) in_pedestrian_projLine_direction_x = 0;
			if(strcmp(buffer, "projLine_direction_y") == 0) in_pedestrian_projLine_direction_y = 1;
			if(strcmp(buffer, "/projLine_direction_y") == 0) in_pedestrian_projLine_direction_y = 0;
			if(strcmp(buffer, "projLine_point_x") == 0) in_pedestrian_projLine_point_x = 1;
			if(strcmp(buffer, "/projLine_point_x") == 0) in_pedestrian_projLine_point_x = 0;
			if(strcmp(buffer, "projLine_point_y") == 0) in_pedestrian_projLine_point_y = 1;
			if(strcmp(buffer, "/projLine_point_y") == 0) in_pedestrian_projLine_point_y = 0;
			
			
			/* End of tag and reset buffer */
			in_tag = 0;
			i = 0;
		}
		/* If start of tag */
		else if(c == '<')
		{
			/* Place /0 at end of buffer to end numbers */
			buffer[i] = 0;
			/* Flag in tag */
			in_tag = 1;
			
			if(in_itno) *itno = atoi(buffer);
			if(in_name) strcpy(agentname, buffer);
			else
			{
				if(in_pedestrian_x){ 
                    pedestrian_x = (float) atof(buffer);    
                }
				if(in_pedestrian_y){ 
                    pedestrian_y = (float) atof(buffer);    
                }
				if(in_pedestrian_vx){ 
                    pedestrian_vx = (float) atof(buffer);    
                }
				if(in_pedestrian_vy){ 
                    pedestrian_vy = (float) atof(buffer);    
                }
				if(in_pedestrian_desvx){ 
                    pedestrian_desvx = (float) atof(buffer);    
                }
				if(in_pedestrian_desvy){ 
                    pedestrian_desvy = (float) atof(buffer);    
                }
				if(in_pedestrian_count){ 
                    pedestrian_count = (int) atoi(buffer);    
                }
				if(in_pedestrian_lineFail){ 
                    pedestrian_lineFail = (int) atoi(buffer);    
                }
				if(in_pedestrian_newvx){ 
                    pedestrian_newvx = (float) atof(buffer);    
                }
				if(in_pedestrian_newvy){ 
                    pedestrian_newvy = (float) atof(buffer);    
                }
				if(in_pedestrian_orcaLine_direction_x){ 
                    readFloatArrayInput(buffer, pedestrian_orcaLine_direction_x, 128);    
                }
				if(in_pedestrian_orcaLine_direction_y){ 
                    readFloatArrayInput(buffer, pedestrian_orcaLine_direction_y, 128);    
                }
				if(in_pedestrian_orcaLine_point_x){ 
                    readFloatArrayInput(buffer, pedestrian_orcaLine_point_x, 128);    
                }
				if(in_pedestrian_orcaLine_point_y){ 
                    readFloatArrayInput(buffer, pedestrian_orcaLine_point_y, 128);    
                }
				if(in_pedestrian_projLine_direction_x){ 
                    readFloatArrayInput(buffer, pedestrian_projLine_direction_x, 128);    
                }
				if(in_pedestrian_projLine_direction_y){ 
                    readFloatArrayInput(buffer, pedestrian_projLine_direction_y, 128);    
                }
				if(in_pedestrian_projLine_point_x){ 
                    readFloatArrayInput(buffer, pedestrian_projLine_point_x, 128);    
                }
				if(in_pedestrian_projLine_point_y){ 
                    readFloatArrayInput(buffer, pedestrian_projLine_point_y, 128);    
                }
				
			}
			
			/* Reset buffer */
			i = 0;
		}
		/* If in tag put read char into buffer */
		else if(in_tag)
		{
			buffer[i] = c;
			i++;
		}
		/* If in data read char into buffer */
		else
		{
			buffer[i] = c;
			i++;
		}
	}
	/* Close the file */
	fclose(file);
}

glm::vec3 getMaximumBounds(){
    return agent_maximum;
}

glm::vec3 getMinimumBounds(){
    return agent_minimum;
}

