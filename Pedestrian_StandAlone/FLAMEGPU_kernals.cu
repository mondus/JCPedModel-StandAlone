

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

#ifndef _FLAMEGPU_KERNELS_H_
#define _FLAMEGPU_KERNELS_H_

#include "header.h"

//#include "D:\Users\John\Documents\GitHub\cub\cub\cub.cuh"
#include "cub\cub.cuh"

/* Agent count constants */

__constant__ int d_xmachine_memory_pedestrian_count;

/* Agent state count constants */

__constant__ int d_xmachine_memory_pedestrian_default_count;


/* Message constants */

/* properties_message Message variables */
/* Non partitioned and spatial partitioned message variables  */
__constant__ int d_message_properties_message_count;         /**< message list counter*/
__constant__ int d_message_properties_message_output_type;   /**< message output type (single or optional)*/
//Spatial Partitioning Variables
__constant__ glm::vec3 d_message_properties_message_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
__constant__ glm::vec3 d_message_properties_message_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
__constant__ glm::ivec3 d_message_properties_message_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
__constant__ float d_message_properties_message_radius;                 /**< partition radius (used to determin the size of the partitions) */



//include each function file

#include "functions.c"

/* Texture bindings */
/* properties_message Message Bindings */texture<float, 1, cudaReadModeElementType> tex_xmachine_message_properties_message_x;
__constant__ int d_tex_xmachine_message_properties_message_x_offset; texture<float, 1, cudaReadModeElementType> tex_xmachine_message_properties_message_y;
__constant__ int d_tex_xmachine_message_properties_message_y_offset; texture<float, 1, cudaReadModeElementType> tex_xmachine_message_properties_message_z;
__constant__ int d_tex_xmachine_message_properties_message_z_offset; texture<float, 1, cudaReadModeElementType> tex_xmachine_message_properties_message_vx;
__constant__ int d_tex_xmachine_message_properties_message_vx_offset; texture<float, 1, cudaReadModeElementType> tex_xmachine_message_properties_message_vy;
__constant__ int d_tex_xmachine_message_properties_message_vy_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_properties_message_pbm_start;
__constant__ int d_tex_xmachine_message_properties_message_pbm_start_offset;
texture<int, 1, cudaReadModeElementType> tex_xmachine_message_properties_message_pbm_end_or_count;
__constant__ int d_tex_xmachine_message_properties_message_pbm_end_or_count_offset;



#define WRAP(x,m) (((x)<m)?(x):(x%m)) /**< Simple wrap */
#define sWRAP(x,m) (((x)<m)?(((x)<0)?(m+(x)):(x)):(m-(x))) /**<signed integer wrap (no modulus) for negatives where 2m > |x| > m */

//PADDING WILL ONLY AVOID SM CONFLICTS FOR 32BIT
//SM_OFFSET REQUIRED AS FERMI STARTS INDEXING MEMORY FROM LOCATION 0 (i.e. NULL)??
__constant__ int d_SM_START;
__constant__ int d_PADDING;

//SM addressing macro to avoid conflicts (32 bit only)
#define SHARE_INDEX(i, s) ((((s) + d_PADDING)* (i))+d_SM_START) /**<offset struct size by padding to avoid bank conflicts */

//if doubel support is needed then define the following function which requires sm_13 or later
#ifdef _DOUBLE_SUPPORT_REQUIRED_
__inline__ __device__ double tex1DfetchDouble(texture<int2, 1, cudaReadModeElementType> tex, int i)
{
	int2 v = tex1Dfetch(tex, i);
	//IF YOU HAVE AN ERROR HERE THEN YOU ARE USING DOUBLE VALUES IN AGENT MEMORY AND NOT COMPILING FOR DOUBLE SUPPORTED HARDWARE
	//To compile for double supported hardware change the CUDA Build rule property "Use sm_13 Architecture (double support)" on the CUDA-Specific Propert Page of the CUDA Build Rule for simulation.cu
	return __hiloint2double(v.y, v.x);
}
#endif

/* Helper functions */
/** next_cell
* Function used for finding the next cell when using spatial partitioning
* Upddates the relative cell variable which can have value of -1, 0 or +1
* @param relative_cell pointer to the relative cell position
* @return boolean if there is a next cell. True unless relative_Cell value was 1,1,1
*/
__device__ int next_cell3D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;

	if (relative_cell->z < 1)
	{
		relative_cell->z++;
		return true;
	}
	relative_cell->z = -1;

	return false;
}

/** next_cell2D
* Function used for finding the next cell when using spatial partitioning. Z component is ignored
* Upddates the relative cell variable which can have value of -1, 0 or +1
* @param relative_cell pointer to the relative cell position
* @return boolean if there is a next cell. True unless relative_Cell value was 1,1
*/
__device__ int next_cell2D(glm::ivec3* relative_cell)
{
	if (relative_cell->x < 1)
	{
		relative_cell->x++;
		return true;
	}
	relative_cell->x = -1;

	if (relative_cell->y < 1)
	{
		relative_cell->y++;
		return true;
	}
	relative_cell->y = -1;

	return false;
}


/** do_linear_program_3_function_filter
*	Standard agent condition function. Filters agents from one state list to the next depending on the condition
* @param currentState xmachine_memory_pedestrian_list representing agent i the current state
* @param nextState xmachine_memory_pedestrian_list representing agent i the next state
*/
__global__ void do_linear_program_3_function_filter(xmachine_memory_pedestrian_list* currentState, xmachine_memory_pedestrian_list* nextState)
{
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//check thread max
	if (index < d_xmachine_memory_pedestrian_count){

		//apply the filter
		if (currentState->lineFail[index] != -1)
		{	//copy agent data to newstate list
			nextState->x[index] = currentState->x[index];
			nextState->y[index] = currentState->y[index];
			nextState->vx[index] = currentState->vx[index];
			nextState->vy[index] = currentState->vy[index];
			nextState->desvx[index] = currentState->desvx[index];
			nextState->desvy[index] = currentState->desvy[index];
			nextState->count[index] = currentState->count[index];
			nextState->lineFail[index] = currentState->lineFail[index];
			nextState->newvx[index] = currentState->newvx[index];
			nextState->newvy[index] = currentState->newvy[index];
			nextState->orcaLine_direction_x[index] = currentState->orcaLine_direction_x[index];
			nextState->orcaLine_direction_y[index] = currentState->orcaLine_direction_y[index];
			nextState->orcaLine_point_x[index] = currentState->orcaLine_point_x[index];
			nextState->orcaLine_point_y[index] = currentState->orcaLine_point_y[index];
			nextState->projLine_direction_x[index] = currentState->projLine_direction_x[index];
			nextState->projLine_direction_y[index] = currentState->projLine_direction_y[index];
			nextState->projLine_point_x[index] = currentState->projLine_point_x[index];
			nextState->projLine_point_y[index] = currentState->projLine_point_y[index];
			//set scan input flag to 1
			nextState->_scan_input[index] = 1;
		}
		else
		{
			//set scan input flag of current state to 1 (keep agent)
			currentState->_scan_input[index] = 1;
		}

	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created pedestrian agent functions */

/** reset_pedestrian_scan_input
* pedestrian agent reset scan input function
* @param agents The xmachine_memory_pedestrian_list agent list
*/
__global__ void reset_pedestrian_scan_input(xmachine_memory_pedestrian_list* agents){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	agents->_position[index] = 0;
	agents->_scan_input[index] = 0;
}



/** scatter_pedestrian_Agents
* pedestrian scatter agents function (used after agent birth/death)
* @param agents_dst xmachine_memory_pedestrian_list agent list destination
* @param agents_src xmachine_memory_pedestrian_list agent list source
* @param dst_agent_count index to start scattering agents from
*/
__global__ void scatter_pedestrian_Agents(xmachine_memory_pedestrian_list* agents_dst, xmachine_memory_pedestrian_list* agents_src, int dst_agent_count, int number_to_scatter){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = agents_src->_scan_input[index];

	//if optional message is to be written. 
	//must check agent is within number to scatter as unused threads may have scan input = 1
	if ((_scan_input == 1) && (index < number_to_scatter)){
		int output_index = agents_src->_position[index] + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write     
		agents_dst->_position[output_index] = output_index;
		agents_dst->x[output_index] = agents_src->x[index];
		agents_dst->y[output_index] = agents_src->y[index];
		agents_dst->vx[output_index] = agents_src->vx[index];
		agents_dst->vy[output_index] = agents_src->vy[index];
		agents_dst->desvx[output_index] = agents_src->desvx[index];
		agents_dst->desvy[output_index] = agents_src->desvy[index];
		agents_dst->count[output_index] = agents_src->count[index];
		agents_dst->lineFail[output_index] = agents_src->lineFail[index];
		agents_dst->newvx[output_index] = agents_src->newvx[index];
		agents_dst->newvy[output_index] = agents_src->newvy[index];
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_point_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_point_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_point_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_point_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_point_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_point_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_point_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_point_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
	}
}

/** append_pedestrian_Agents
* pedestrian scatter agents function (used after agent birth/death)
* @param agents_dst xmachine_memory_pedestrian_list agent list destination
* @param agents_src xmachine_memory_pedestrian_list agent list source
* @param dst_agent_count index to start scattering agents from
*/
__global__ void append_pedestrian_Agents(xmachine_memory_pedestrian_list* agents_dst, xmachine_memory_pedestrian_list* agents_src, int dst_agent_count, int number_to_append){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//must check agent is within number to append as unused threads may have scan input = 1
	if (index < number_to_append){
		int output_index = index + dst_agent_count;

		//AoS - xmachine_message_location Un-Coalesced scattered memory write
		agents_dst->_position[output_index] = output_index;
		agents_dst->x[output_index] = agents_src->x[index];
		agents_dst->y[output_index] = agents_src->y[index];
		agents_dst->vx[output_index] = agents_src->vx[index];
		agents_dst->vy[output_index] = agents_src->vy[index];
		agents_dst->desvx[output_index] = agents_src->desvx[index];
		agents_dst->desvy[output_index] = agents_src->desvy[index];
		agents_dst->count[output_index] = agents_src->count[index];
		agents_dst->lineFail[output_index] = agents_src->lineFail[index];
		agents_dst->newvx[output_index] = agents_src->newvx[index];
		agents_dst->newvy[output_index] = agents_src->newvy[index];
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_point_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_point_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->orcaLine_point_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->orcaLine_point_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_point_x[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_point_x[(i*xmachine_memory_pedestrian_MAX) + index];
		}
		for (int i = 0; i<128; i++){
			agents_dst->projLine_point_y[(i*xmachine_memory_pedestrian_MAX) + output_index] = agents_src->projLine_point_y[(i*xmachine_memory_pedestrian_MAX) + index];
		}
	}
}

/** add_pedestrian_agent
* Continuous pedestrian agent add agent function writes agent data to agent swap
* @param agents xmachine_memory_pedestrian_list to add agents to
* @param x agent variable of type float
* @param y agent variable of type float
* @param vx agent variable of type float
* @param vy agent variable of type float
* @param desvx agent variable of type float
* @param desvy agent variable of type float
* @param count agent variable of type int
* @param lineFail agent variable of type int
* @param newvx agent variable of type float
* @param newvy agent variable of type float
* @param orcaLine_direction_x agent variable of type float
* @param orcaLine_direction_y agent variable of type float
* @param orcaLine_point_x agent variable of type float
* @param orcaLine_point_y agent variable of type float
* @param projLine_direction_x agent variable of type float
* @param projLine_direction_y agent variable of type float
* @param projLine_point_x agent variable of type float
* @param projLine_point_y agent variable of type float
*/
template <int AGENT_TYPE>
__device__ void add_pedestrian_agent(xmachine_memory_pedestrian_list* agents, float x, float y, float vx, float vy, float desvx, float desvy, int count, int lineFail, float newvx, float newvy){

	int index;

	//calculate the agents index in global agent list (depends on agent type)
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x* gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x*blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y*blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y* width);
	}
	else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	//for prefix sum
	agents->_position[index] = 0;
	agents->_scan_input[index] = 1;

	//write data to new buffer
	agents->x[index] = x;
	agents->y[index] = y;
	agents->vx[index] = vx;
	agents->vy[index] = vy;
	agents->desvx[index] = desvx;
	agents->desvy[index] = desvy;
	agents->count[index] = count;
	agents->lineFail[index] = lineFail;
	agents->newvx[index] = newvx;
	agents->newvy[index] = newvy;

}

//non templated version assumes DISCRETE_2D but works also for CONTINUOUS
__device__ void add_pedestrian_agent(xmachine_memory_pedestrian_list* agents, float x, float y, float vx, float vy, float desvx, float desvy, int count, int lineFail, float newvx, float newvy){
	add_pedestrian_agent<DISCRETE_2D>(agents, x, y, vx, vy, desvx, desvy, count, lineFail, newvx, newvy);
}

/** reorder_pedestrian_agents
* Continuous pedestrian agent areorder function used after key value pairs have been sorted
* @param values sorted index values
* @param unordered_agents list of unordered agents
* @ param ordered_agents list used to output ordered agents
*/
__global__ void reorder_pedestrian_agents(unsigned int* values, xmachine_memory_pedestrian_list* unordered_agents, xmachine_memory_pedestrian_list* ordered_agents)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	uint old_pos = values[index];

	//reorder agent data
	ordered_agents->x[index] = unordered_agents->x[old_pos];
	ordered_agents->y[index] = unordered_agents->y[old_pos];
	ordered_agents->vx[index] = unordered_agents->vx[old_pos];
	ordered_agents->vy[index] = unordered_agents->vy[old_pos];
	ordered_agents->desvx[index] = unordered_agents->desvx[old_pos];
	ordered_agents->desvy[index] = unordered_agents->desvy[old_pos];
	ordered_agents->count[index] = unordered_agents->count[old_pos];
	ordered_agents->lineFail[index] = unordered_agents->lineFail[old_pos];
	ordered_agents->newvx[index] = unordered_agents->newvx[old_pos];
	ordered_agents->newvy[index] = unordered_agents->newvy[old_pos];
	for (int i = 0; i<128; i++){
		ordered_agents->orcaLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->orcaLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
	for (int i = 0; i<128; i++){
		ordered_agents->orcaLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->orcaLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
	for (int i = 0; i<128; i++){
		ordered_agents->orcaLine_point_x[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->orcaLine_point_x[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
	for (int i = 0; i<128; i++){
		ordered_agents->orcaLine_point_y[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->orcaLine_point_y[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
	for (int i = 0; i<128; i++){
		ordered_agents->projLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->projLine_direction_x[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
	for (int i = 0; i<128; i++){
		ordered_agents->projLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->projLine_direction_y[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
	for (int i = 0; i<128; i++){
		ordered_agents->projLine_point_x[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->projLine_point_x[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
	for (int i = 0; i<128; i++){
		ordered_agents->projLine_point_y[(i*xmachine_memory_pedestrian_MAX) + index] = unordered_agents->projLine_point_y[(i*xmachine_memory_pedestrian_MAX) + old_pos];
	}
}

/** get_pedestrian_agent_array_value
*  Template function for accessing pedestrian agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
*  @param array Agent memory array
*  @param index to lookup
*  @return return value
*/
template<typename T>
__FLAME_GPU_FUNC__ T get_pedestrian_agent_array_value(T *array, uint index){
	return array[index*xmachine_memory_pedestrian_MAX];
}

/** set_pedestrian_agent_array_value
*  Template function for setting pedestrian agent array memory variables. Assumes array points to the first element of the agents array values (offset by agent index)
*  @param array Agent memory array
*  @param index to lookup
*  @param return value
*/
template<typename T>
__FLAME_GPU_FUNC__ void set_pedestrian_agent_array_value(T *array, uint index, T value){
	array[index*xmachine_memory_pedestrian_MAX] = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dyanamically created properties_message message functions */


/** add_properties_message_message
* Add non partitioned or spatially partitioned properties_message message
* @param messages xmachine_message_properties_message_list message list to add too
* @param x agent variable of type float
* @param y agent variable of type float
* @param z agent variable of type float
* @param vx agent variable of type float
* @param vy agent variable of type float
*/
__device__ void add_properties_message_message(xmachine_message_properties_message_list* messages, float x, float y, float z, float vx, float vy){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x + d_message_properties_message_count;

	int _position;
	int _scan_input;

	//decide output position
	if (d_message_properties_message_output_type == single_message){
		_position = index; //same as agent position
		_scan_input = 0;
	}
	else if (d_message_properties_message_output_type == optional_message){
		_position = 0;	   //to be calculated using Prefix sum
		_scan_input = 1;
	}

	//AoS - xmachine_message_properties_message Coalesced memory write
	messages->_scan_input[index] = _scan_input;
	messages->_position[index] = _position;
	messages->x[index] = x;
	messages->y[index] = y;
	messages->z[index] = z;
	messages->vx[index] = vx;
	messages->vy[index] = vy;

}

/**
* Scatter non partitioned or spatially partitioned properties_message message (for optional messages)
* @param messages scatter_optional_properties_message_messages Sparse xmachine_message_properties_message_list message list
* @param message_swap temp xmachine_message_properties_message_list message list to scatter sparse messages to
*/
__global__ void scatter_optional_properties_message_messages(xmachine_message_properties_message_list* messages, xmachine_message_properties_message_list* messages_swap){
	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	int _scan_input = messages_swap->_scan_input[index];

	//if optional message is to be written
	if (_scan_input == 1){
		int output_index = messages_swap->_position[index] + d_message_properties_message_count;

		//AoS - xmachine_message_properties_message Un-Coalesced scattered memory write
		messages->_position[output_index] = output_index;
		messages->x[output_index] = messages_swap->x[index];
		messages->y[output_index] = messages_swap->y[index];
		messages->z[output_index] = messages_swap->z[index];
		messages->vx[output_index] = messages_swap->vx[index];
		messages->vy[output_index] = messages_swap->vy[index];
	}
}

/** reset_properties_message_swaps
* Reset non partitioned or spatially partitioned properties_message message swaps (for scattering optional messages)
* @param message_swap message list to reset _position and _scan_input values back to 0
*/
__global__ void reset_properties_message_swaps(xmachine_message_properties_message_list* messages_swap){

	//global thread index
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	messages_swap->_position[index] = 0;
	messages_swap->_scan_input[index] = 0;
}

/* Message functions */

/** message_properties_message_grid_position
* Calculates the grid cell position given an glm::vec3 vector
* @param position glm::vec3 vector representing a position
*/
__device__ glm::ivec3 message_properties_message_grid_position(glm::vec3 position)
{
	glm::ivec3 gridPos;
	gridPos.x = floor((position.x - d_message_properties_message_min_bounds.x) * (float)d_message_properties_message_partitionDim.x / (d_message_properties_message_max_bounds.x - d_message_properties_message_min_bounds.x));
	gridPos.y = floor((position.y - d_message_properties_message_min_bounds.y) * (float)d_message_properties_message_partitionDim.y / (d_message_properties_message_max_bounds.y - d_message_properties_message_min_bounds.y));
	gridPos.z = floor((position.z - d_message_properties_message_min_bounds.z) * (float)d_message_properties_message_partitionDim.z / (d_message_properties_message_max_bounds.z - d_message_properties_message_min_bounds.z));

	//do wrapping or bounding


	return gridPos;
}

/** message_properties_message_hash
* Given the grid position in partition space this function calculates a hash value
* @param gridPos The position in partition space
*/
__device__ unsigned int message_properties_message_hash(glm::ivec3 gridPos)
{
	//cheap bounding without mod (within range +- partition dimension)
	gridPos.x = (gridPos.x<0) ? d_message_properties_message_partitionDim.x - 1 : gridPos.x;
	gridPos.x = (gridPos.x >= d_message_properties_message_partitionDim.x) ? 0 : gridPos.x;
	gridPos.y = (gridPos.y<0) ? d_message_properties_message_partitionDim.y - 1 : gridPos.y;
	gridPos.y = (gridPos.y >= d_message_properties_message_partitionDim.y) ? 0 : gridPos.y;
	gridPos.z = (gridPos.z<0) ? d_message_properties_message_partitionDim.z - 1 : gridPos.z;
	gridPos.z = (gridPos.z >= d_message_properties_message_partitionDim.z) ? 0 : gridPos.z;

	//unique id
	return ((gridPos.z * d_message_properties_message_partitionDim.y) * d_message_properties_message_partitionDim.x) + (gridPos.y * d_message_properties_message_partitionDim.x) + gridPos.x;
}

#ifdef FAST_ATOMIC_SORTING
/** hist_properties_message_messages
* Kernal function for performing a histogram (count) on each partition bin and saving the hash and index of a message within that bin
* @param local_bin_index output index of the message within the calculated bin
* @param unsorted_index output bin index (hash) value
* @param messages the message list used to generate the hash value outputs
* @param agent_count the current number of agents outputting messages
*/
__global__ void hist_properties_message_messages(uint* local_bin_index, uint* unsorted_index, int* global_bin_count, xmachine_message_properties_message_list* messages, int agent_count)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index >= agent_count)
		return;

	glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
	glm::ivec3 grid_position = message_properties_message_grid_position(position);
	unsigned int hash = message_properties_message_hash(grid_position);
	unsigned int bin_idx = atomicInc((unsigned int*)&global_bin_count[hash], 0xFFFFFFFF);
	local_bin_index[index] = bin_idx;
	unsorted_index[index] = hash;
}

/** reorder_properties_message_messages
* Reorders the messages accoring to the partition boundary matrix start indices of each bin
* @param local_bin_index index of the message within the desired bin
* @param unsorted_index bin index (hash) value
* @param pbm_start_index the start indices of the partition boundary matrix
* @param unordered_messages the original unordered message data
* @param ordered_messages buffer used to scatter messages into the correct order
@param agent_count the current number of agents outputting messages
*/
__global__ void reorder_properties_message_messages(uint* local_bin_index, uint* unsorted_index, int* pbm_start_index, xmachine_message_properties_message_list* unordered_messages, xmachine_message_properties_message_list* ordered_messages, int agent_count)
{
	int index = (blockIdx.x *blockDim.x) + threadIdx.x;

	if (index >= agent_count)
		return;

	uint i = unsorted_index[index];
	unsigned int sorted_index = local_bin_index[index] + pbm_start_index[i];

	//finally reorder agent data
	ordered_messages->x[sorted_index] = unordered_messages->x[index];
	ordered_messages->y[sorted_index] = unordered_messages->y[index];
	ordered_messages->z[sorted_index] = unordered_messages->z[index];
	ordered_messages->vx[sorted_index] = unordered_messages->vx[index];
	ordered_messages->vy[sorted_index] = unordered_messages->vy[index];
}

#else

/** hash_properties_message_messages
* Kernal function for calculating a hash value for each messahe depending on its position
* @param keys output for the hash key
* @param values output for the index value
* @param messages the message list used to generate the hash value outputs
*/
__global__ void hash_properties_message_messages(uint* keys, uint* values, xmachine_message_properties_message_list* messages)
{
	unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	glm::vec3 position = glm::vec3(messages->x[index], messages->y[index], messages->z[index]);
	glm::ivec3 grid_position = message_properties_message_grid_position(position);
	unsigned int hash = message_properties_message_hash(grid_position);

	keys[index] = hash;
	values[index] = index;
}

/** reorder_properties_message_messages
* Reorders the messages accoring to the ordered sort identifiers and builds a Partition Boundary Matrix by looking at the previosu threads sort id.
* @param keys the sorted hash keys
* @param values the sorted index values
* @param matrix the PBM
* @param unordered_messages the original unordered message data
* @param ordered_messages buffer used to scatter messages into the correct order
*/
__global__ void reorder_properties_message_messages(uint* keys, uint* values, xmachine_message_properties_message_PBM* matrix, xmachine_message_properties_message_list* unordered_messages, xmachine_message_properties_message_list* ordered_messages)
{
	extern __shared__ int sm_data[];

	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//load threads sort key into sm
	uint key = keys[index];
	uint old_pos = values[index];

	sm_data[threadIdx.x] = key;
	__syncthreads();

	unsigned int prev_key;

	//if first thread then no prev sm value so get prev from global memory 
	if (threadIdx.x == 0)
	{
		//first thread has no prev value so ignore
		if (index != 0)
			prev_key = keys[index - 1];
	}
	//get previous ident from sm
	else
	{
		prev_key = sm_data[threadIdx.x - 1];
	}

	//TODO: Check key is not out of bounds

	//set partition boundaries
	if (index < d_message_properties_message_count)
	{
		//if first thread then set first partition cell start
		if (index == 0)
		{
			matrix->start[key] = index;
		}

		//if edge of a boundr update start and end of partition
		else if (prev_key != key)
		{
			//set start for key
			matrix->start[key] = index;

			//set end for key -1
			matrix->end_or_count[prev_key] = index;
		}

		//if last thread then set final partition cell end
		if (index == d_message_properties_message_count - 1)
		{
			matrix->end_or_count[key] = index + 1;
		}
	}

	//finally reorder agent data
	ordered_messages->x[index] = unordered_messages->x[old_pos];
	ordered_messages->y[index] = unordered_messages->y[old_pos];
	ordered_messages->z[index] = unordered_messages->z[old_pos];
	ordered_messages->vx[index] = unordered_messages->vx[old_pos];
	ordered_messages->vy[index] = unordered_messages->vy[old_pos];
}

#endif

/** load_next_properties_message_message
* Used to load the next message data to shared memory
* Idea is check the current cell index to see if we can simply get a message from the current cell
* If we are at the end of the current cell then loop till we find the next cell with messages (this way we ignore cells with no messages)
* @param messages the message list
* @param partition_matrix the PBM
* @param relative_cell the relative partition cell position from the agent position
* @param cell_index_max the maximum index of the current partition cell
* @param agent_grid_cell the agents partition cell position
* @param cell_index the current cell index in agent_grid_cell+relative_cell
* @return true if a message has been loaded into sm false otherwise
*/
__device__ int load_next_properties_message_message(xmachine_message_properties_message_list* messages, xmachine_message_properties_message_PBM* partition_matrix, glm::ivec3 relative_cell, int cell_index_max, glm::ivec3 agent_grid_cell, int cell_index)
{
	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	int move_cell = true;
	cell_index++;

	//see if we need to move to a new partition cell
	if (cell_index < cell_index_max)
		move_cell = false;

	while (move_cell)
	{
		//get the next relative grid position 
		if (next_cell2D(&relative_cell))
		{
			//calculate the next cells grid position and hash
			glm::ivec3 next_cell_position = agent_grid_cell + relative_cell;
			int next_cell_hash = message_properties_message_hash(next_cell_position);
			//use the hash to calculate the start index
			int cell_index_min = tex1Dfetch(tex_xmachine_message_properties_message_pbm_start, next_cell_hash + d_tex_xmachine_message_properties_message_pbm_start_offset);
			cell_index_max = tex1Dfetch(tex_xmachine_message_properties_message_pbm_end_or_count, next_cell_hash + d_tex_xmachine_message_properties_message_pbm_end_or_count_offset);
			//check for messages in the cell (cell index max is the count for atomic sorting)
#ifdef FAST_ATOMIC_SORTING
			if (cell_index_max > 0)
			{
				//when using fast atomics value represents bin count not last index!
				cell_index_max += cell_index_min; //when using fast atomics value represents bin count not last index!
#else
			if (cell_index_min != 0xffffffff)
			{
#endif
				//start from the cell index min
				cell_index = cell_index_min;
				//exit the loop as we have found a valid cell with message data
				move_cell = false;
			}
			}
		else
		{
			//we have exhausted all the neighbouring cells so there are no more messages
			return false;
		}
		}

	//get message data using texture fetch
	xmachine_message_properties_message temp_message;
	temp_message._relative_cell = relative_cell;
	temp_message._cell_index_max = cell_index_max;
	temp_message._cell_index = cell_index;
	temp_message._agent_grid_cell = agent_grid_cell;

	//Using texture cache
	temp_message.x = tex1Dfetch(tex_xmachine_message_properties_message_x, cell_index + d_tex_xmachine_message_properties_message_x_offset); temp_message.y = tex1Dfetch(tex_xmachine_message_properties_message_y, cell_index + d_tex_xmachine_message_properties_message_y_offset); temp_message.z = tex1Dfetch(tex_xmachine_message_properties_message_z, cell_index + d_tex_xmachine_message_properties_message_z_offset); temp_message.vx = tex1Dfetch(tex_xmachine_message_properties_message_vx, cell_index + d_tex_xmachine_message_properties_message_vx_offset); temp_message.vy = tex1Dfetch(tex_xmachine_message_properties_message_vy, cell_index + d_tex_xmachine_message_properties_message_vy_offset);

	//load it into shared memory (no sync as no sharing between threads)
	int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_properties_message));
	xmachine_message_properties_message* sm_message = ((xmachine_message_properties_message*)&message_share[message_index]);
	sm_message[0] = temp_message;

	return true;
	}

/*
* get first spatial partitioned properties_message message (first batch load into shared memory)
*/
__device__ xmachine_message_properties_message* get_first_properties_message_message(xmachine_message_properties_message_list* messages, xmachine_message_properties_message_PBM* partition_matrix, float x, float y, float z){

	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	glm::ivec3 relative_cell = glm::ivec3(-2, -1, -1);
	int cell_index_max = 0;
	int cell_index = 0;
	glm::vec3 position = glm::vec3(x, y, z);
	glm::ivec3 agent_grid_cell = message_properties_message_grid_position(position);

	if (load_next_properties_message_message(messages, partition_matrix, relative_cell, cell_index_max, agent_grid_cell, cell_index))
	{
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_properties_message));
		return ((xmachine_message_properties_message*)&message_share[message_index]);
	}
	else
	{
		return false;
	}
}

/*
* get next spatial partitioned properties_message message (either from SM or next batch load)
*/
__device__ xmachine_message_properties_message* get_next_properties_message_message(xmachine_message_properties_message* message, xmachine_message_properties_message_list* messages, xmachine_message_properties_message_PBM* partition_matrix){

	extern __shared__ int sm_data[];
	char* message_share = (char*)&sm_data[0];

	//TODO: check message count

	if (load_next_properties_message_message(messages, partition_matrix, message->_relative_cell, message->_cell_index_max, message->_agent_grid_cell, message->_cell_index))
	{
		//get conflict free address of 
		int message_index = SHARE_INDEX(threadIdx.y*blockDim.x + threadIdx.x, sizeof(xmachine_message_properties_message));
		return ((xmachine_message_properties_message*)&message_share[message_index]);
	}
	else
		return false;

}




/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Dynamically created GPU kernels  */



/**
*
*/
__global__ void GPUFLAME_output_properties(xmachine_memory_pedestrian_list* agents, xmachine_message_properties_message_list* properties_message_messages){

	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//For agents not using non partitioned message input check the agent bounds
	if (index >= d_xmachine_memory_pedestrian_count)
		return;


	//SoA to AoS - xmachine_memory_output_properties Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_pedestrian agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.vx = agents->vx[index];
	agent.vy = agents->vy[index];
	agent.desvx = agents->desvx[index];
	agent.desvy = agents->desvy[index];
	agent.count = agents->count[index];
	agent.lineFail = agents->lineFail[index];
	agent.newvx = agents->newvx[index];
	agent.newvy = agents->newvy[index];
	agent.orcaLine_direction_x = &(agents->orcaLine_direction_x[index]);
	agent.orcaLine_direction_y = &(agents->orcaLine_direction_y[index]);
	agent.orcaLine_point_x = &(agents->orcaLine_point_x[index]);
	agent.orcaLine_point_y = &(agents->orcaLine_point_y[index]);
	agent.projLine_direction_x = &(agents->projLine_direction_x[index]);
	agent.projLine_direction_y = &(agents->projLine_direction_y[index]);
	agent.projLine_point_x = &(agents->projLine_point_x[index]);
	agent.projLine_point_y = &(agents->projLine_point_y[index]);

	//FLAME function call
	int dead = !output_properties(&agent, properties_message_messages);

	//continuous agent: set reallocation flag
	agents->_scan_input[index] = dead;

	//AoS to SoA - xmachine_memory_output_properties Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->vx[index] = agent.vx;
	agents->vy[index] = agent.vy;
	agents->desvx[index] = agent.desvx;
	agents->desvy[index] = agent.desvy;
	agents->count[index] = agent.count;
	agents->lineFail[index] = agent.lineFail;
	agents->newvx[index] = agent.newvx;
	agents->newvy[index] = agent.newvy;
}

/**
*
*/
__global__ void GPUFLAME_make_orcaLines(xmachine_memory_pedestrian_list* agents, xmachine_message_properties_message_list* properties_message_messages, xmachine_message_properties_message_PBM* partition_matrix){

	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//For agents not using non partitioned message input check the agent bounds
	if (index >= d_xmachine_memory_pedestrian_count)
		return;


	//SoA to AoS - xmachine_memory_make_orcaLines Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_pedestrian agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.vx = agents->vx[index];
	agent.vy = agents->vy[index];
	agent.desvx = agents->desvx[index];
	agent.desvy = agents->desvy[index];
	agent.count = agents->count[index];
	agent.lineFail = agents->lineFail[index];
	agent.newvx = agents->newvx[index];
	agent.newvy = agents->newvy[index];
	agent.orcaLine_direction_x = &(agents->orcaLine_direction_x[index]);
	agent.orcaLine_direction_y = &(agents->orcaLine_direction_y[index]);
	agent.orcaLine_point_x = &(agents->orcaLine_point_x[index]);
	agent.orcaLine_point_y = &(agents->orcaLine_point_y[index]);
	agent.projLine_direction_x = &(agents->projLine_direction_x[index]);
	agent.projLine_direction_y = &(agents->projLine_direction_y[index]);
	agent.projLine_point_x = &(agents->projLine_point_x[index]);
	agent.projLine_point_y = &(agents->projLine_point_y[index]);

	//FLAME function call
	int dead = !make_orcaLines(&agent, properties_message_messages, partition_matrix);

	//continuous agent: set reallocation flag
	agents->_scan_input[index] = dead;

	//AoS to SoA - xmachine_memory_make_orcaLines Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->vx[index] = agent.vx;
	agents->vy[index] = agent.vy;
	agents->desvx[index] = agent.desvx;
	agents->desvy[index] = agent.desvy;
	agents->count[index] = agent.count;
	agents->lineFail[index] = agent.lineFail;
	agents->newvx[index] = agent.newvx;
	agents->newvy[index] = agent.newvy;
}

/*My version*/

//number of threads in a block
#define blockdimsize 160

//get agent from index
__device__ xmachine_memory_pedestrian getAgent(xmachine_memory_pedestrian_list* agents, int index) {
	xmachine_memory_pedestrian agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.vx = agents->vx[index];
	agent.vy = agents->vy[index];
	agent.desvx = agents->desvx[index];
	agent.desvy = agents->desvy[index];
	agent.count = agents->count[index];
	agent.lineFail = agents->lineFail[index];
	agent.newvx = agents->newvx[index];
	agent.newvy = agents->newvy[index];
	agent.orcaLine_direction_x = &(agents->orcaLine_direction_x[index]);
	agent.orcaLine_direction_y = &(agents->orcaLine_direction_y[index]);
	agent.orcaLine_point_x = &(agents->orcaLine_point_x[index]);
	agent.orcaLine_point_y = &(agents->orcaLine_point_y[index]);
	agent.projLine_direction_x = &(agents->projLine_direction_x[index]);
	agent.projLine_direction_y = &(agents->projLine_direction_y[index]);
	agent.projLine_point_x = &(agents->projLine_point_x[index]);
	agent.projLine_point_y = &(agents->projLine_point_y[index]);
	return agent;
}

//update agent values that can be changed within lp2
//could make newVel and prefVel, count shared memory variables...
__device__ void storeAgent(xmachine_memory_pedestrian_list* agents, int index, xmachine_memory_pedestrian* agent) {
	agents->lineFail[index] = agent->lineFail;
	agents->newvx[index] = agent->newvx;
	agents->newvy[index] = agent->newvy;
}

__global__ void GPUFLAME_do_linear_program_2_COMP(xmachine_memory_pedestrian_list* agents) {
	//shared memory of compressed array
	__shared__ int compArr[blockdimsize];

	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//For agents not using non partitioned message input check the agent bounds
	if (index >= d_xmachine_memory_pedestrian_count) {
		return;
	}

	//SoA to AoS - xmachine_memory_do_linear_program_2 Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_pedestrian agent;

	// Specialize BlockReduce for a 1D block of blockdimsize threads on type int
	typedef cub::BlockReduce<int, blockdimsize> BlockReduce;
	// Allocate shared memory for BlockReduce
	__shared__ typename BlockReduce::TempStorage temp_storageR;
	// Specialize BlockScan for a 1D block of blockdimsize threads on type int
	typedef cub::BlockScan<int, blockdimsize> BlockScan;
	// Allocate shared memory for BlockScan
	__shared__ typename BlockScan::TempStorage temp_storageS;

	//Loop over max possible size AGENTNO
	for (int i = 0; i < AGENTNO; i++) {

		//needed for next loop for agents to be written to ok
		__syncthreads();

		//point thread back to the appropriate agent
		agent = getAgent(agents, index);

		const glm::vec2 lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent.orcaLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent.orcaLine_direction_y, i));
		const glm::vec2 lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent.orcaLine_point_x, i), get_pedestrian_agent_array_value<float>(agent.orcaLine_point_y, i));
		const glm::vec2 prefVelocity = glm::vec2(agent.desvx, agent.desvy);
		const glm::vec2 newVelocity = glm::vec2(agent.newvx, agent.newvy);


		//early out - reduce over block
		// Each thread obtains an input item - whether i has surpassed agent neighbour count
		int thread_dataR = (int)(agent.count <= i);
		// Compute the block-wide sum for threads
		int aggregateCount = BlockReduce(temp_storageR).Reduce(thread_dataR, cub::Sum());
		// Each thread obtains an input item - whether i has surpassed agent neighbour count
		thread_dataR = (int)(agent.count != -1);
		//A subsequent __syncthreads() threadblock barrier should be invoked after calling this method if the collective's temporary storage (e.g., temp_storage) is to be reused or repurposed. Does this apply here??
		__syncthreads();
		// Compute the block-wide sum for threads
		int aggregateFail = BlockReduce(temp_storageR).Reduce(thread_dataR, cub::Sum());
		//all threads in block contain same aggregate
		if (aggregateCount == blockdimsize || aggregateFail == blockdimsize)
			return;

		//Compress//

		//exclusive scan
		int thread_data_out;
		// Obtain input item for each thread
		int thread_data;
		//If i surpases agent number or lineFailed in previous loop iteration dont do calculation
		if (thread_dataR == 1 || agent.lineFail != -1) {
			thread_data = 0;
		}
		else {
			thread_data = (det(lines_direction_i, lines_point_i - newVelocity) > 0.0f) ? 1 : 0;
		}
		// Collectively compute the block-wide exclusive prefix sum
		int block_aggregate;
		BlockScan(temp_storageS).ExclusiveSum(thread_data, thread_data_out, block_aggregate);

		//scatter
		if (thread_data == 1) {
			compArr[thread_data_out] = threadIdx.x;
		}

		//threadIdx.x = {0,1,2,3,4,5,6,7,8}
		//thread_data = {0,1,1,0,1,1,1,0,0}
		//thread_out  = {0,0,1,2,2,3,4,5,5}
		//block_aggragate = 5;
		//compArr = {1,2,4,5,6}

		//run lp1
		/*if (blockIdx.x == 0) {
		printf("%i \t i:%i b:%i \t o:%i c:%i\n", threadIdx.x, i, thread_data, thread_data_out, compArr[threadIdx.x]);
		}*/
		//For compArr to be filled properly
		__syncthreads();

		//threads too high are ignored
		if (threadIdx.x < block_aggregate) {

			//Get new agent values
			agent = getAgent(agents, (blockDim.x*blockIdx.x) + compArr[threadIdx.x]);
			const glm::vec2 newprefVelocity = glm::vec2(agent.desvx, agent.desvy);
			glm::vec2 newnewVelocity = glm::vec2(agent.newvx, agent.newvy);

			const glm::vec2 tempResult = newnewVelocity;
			if (!linearProgram1(&agent, i, maxSpeed_, newprefVelocity, false, newnewVelocity, 0)) {
				//failed here
				newnewVelocity = tempResult;
				agent.lineFail = i;
				//printf("Agent failed linear program 1!\n");
			}
			//write new velocity into agent memory
			agent.newvx = newnewVelocity.x;
			agent.newvy = newnewVelocity.y;

			//store it back
			storeAgent(agents, (blockDim.x*blockIdx.x) + compArr[threadIdx.x], &agent);
		}
	}
}

/**
*
*/
__global__ void GPUFLAME_do_linear_program_2(xmachine_memory_pedestrian_list* agents){

	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//For agents not using non partitioned message input check the agent bounds
	if (index >= d_xmachine_memory_pedestrian_count)
		return;


	//SoA to AoS - xmachine_memory_do_linear_program_2 Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_pedestrian agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.vx = agents->vx[index];
	agent.vy = agents->vy[index];
	agent.desvx = agents->desvx[index];
	agent.desvy = agents->desvy[index];
	agent.count = agents->count[index];
	agent.lineFail = agents->lineFail[index];
	agent.newvx = agents->newvx[index];
	agent.newvy = agents->newvy[index];
	agent.orcaLine_direction_x = &(agents->orcaLine_direction_x[index]);
	agent.orcaLine_direction_y = &(agents->orcaLine_direction_y[index]);
	agent.orcaLine_point_x = &(agents->orcaLine_point_x[index]);
	agent.orcaLine_point_y = &(agents->orcaLine_point_y[index]);
	agent.projLine_direction_x = &(agents->projLine_direction_x[index]);
	agent.projLine_direction_y = &(agents->projLine_direction_y[index]);
	agent.projLine_point_x = &(agents->projLine_point_x[index]);
	agent.projLine_point_y = &(agents->projLine_point_y[index]);

	//FLAME function call
	int dead = !do_linear_program_2(&agent);

	//continuous agent: set reallocation flag
	agents->_scan_input[index] = dead;

	//AoS to SoA - xmachine_memory_do_linear_program_2 Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->vx[index] = agent.vx;
	agents->vy[index] = agent.vy;
	agents->desvx[index] = agent.desvx;
	agents->desvy[index] = agent.desvy;
	agents->count[index] = agent.count;
	agents->lineFail[index] = agent.lineFail;
	agents->newvx[index] = agent.newvx;
	agents->newvy[index] = agent.newvy;
}

/**
*
*/
__global__ void GPUFLAME_do_linear_program_3(xmachine_memory_pedestrian_list* agents){

	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//For agents not using non partitioned message input check the agent bounds
	if (index >= d_xmachine_memory_pedestrian_count)
		return;


	//SoA to AoS - xmachine_memory_do_linear_program_3 Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_pedestrian agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.vx = agents->vx[index];
	agent.vy = agents->vy[index];
	agent.desvx = agents->desvx[index];
	agent.desvy = agents->desvy[index];
	agent.count = agents->count[index];
	agent.lineFail = agents->lineFail[index];
	agent.newvx = agents->newvx[index];
	agent.newvy = agents->newvy[index];
	agent.orcaLine_direction_x = &(agents->orcaLine_direction_x[index]);
	agent.orcaLine_direction_y = &(agents->orcaLine_direction_y[index]);
	agent.orcaLine_point_x = &(agents->orcaLine_point_x[index]);
	agent.orcaLine_point_y = &(agents->orcaLine_point_y[index]);
	agent.projLine_direction_x = &(agents->projLine_direction_x[index]);
	agent.projLine_direction_y = &(agents->projLine_direction_y[index]);
	agent.projLine_point_x = &(agents->projLine_point_x[index]);
	agent.projLine_point_y = &(agents->projLine_point_y[index]);

	//FLAME function call
	int dead = !do_linear_program_3(&agent);

	//continuous agent: set reallocation flag
	agents->_scan_input[index] = dead;

	//AoS to SoA - xmachine_memory_do_linear_program_3 Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->vx[index] = agent.vx;
	agents->vy[index] = agent.vy;
	agents->desvx[index] = agent.desvx;
	agents->desvy[index] = agent.desvy;
	agents->count[index] = agent.count;
	agents->lineFail[index] = agent.lineFail;
	agents->newvx[index] = agent.newvx;
	agents->newvy[index] = agent.newvy;
}

/**
*
*/
__global__ void GPUFLAME_move_agents(xmachine_memory_pedestrian_list* agents, RNG_rand48* rand48){

	//continuous agent: index is agent position in 1D agent list
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	//For agents not using non partitioned message input check the agent bounds
	if (index >= d_xmachine_memory_pedestrian_count)
		return;


	//SoA to AoS - xmachine_memory_move_agents Coalesced memory read (arrays point to first item for agent index)
	xmachine_memory_pedestrian agent;
	agent.x = agents->x[index];
	agent.y = agents->y[index];
	agent.vx = agents->vx[index];
	agent.vy = agents->vy[index];
	agent.desvx = agents->desvx[index];
	agent.desvy = agents->desvy[index];
	agent.count = agents->count[index];
	agent.lineFail = agents->lineFail[index];
	agent.newvx = agents->newvx[index];
	agent.newvy = agents->newvy[index];
	agent.orcaLine_direction_x = &(agents->orcaLine_direction_x[index]);
	agent.orcaLine_direction_y = &(agents->orcaLine_direction_y[index]);
	agent.orcaLine_point_x = &(agents->orcaLine_point_x[index]);
	agent.orcaLine_point_y = &(agents->orcaLine_point_y[index]);
	agent.projLine_direction_x = &(agents->projLine_direction_x[index]);
	agent.projLine_direction_y = &(agents->projLine_direction_y[index]);
	agent.projLine_point_x = &(agents->projLine_point_x[index]);
	agent.projLine_point_y = &(agents->projLine_point_y[index]);

	//FLAME function call
	int dead = !move_agents(&agent, rand48);

	//continuous agent: set reallocation flag
	agents->_scan_input[index] = dead;

	//AoS to SoA - xmachine_memory_move_agents Coalesced memory write (ignore arrays)
	agents->x[index] = agent.x;
	agents->y[index] = agent.y;
	agents->vx[index] = agent.vx;
	agents->vy[index] = agent.vy;
	agents->desvx[index] = agent.desvx;
	agents->desvy[index] = agent.desvy;
	agents->count[index] = agent.count;
	agents->lineFail[index] = agent.lineFail;
	agents->newvx[index] = agent.newvx;
	agents->newvy[index] = agent.newvy;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* Rand48 functions */

__device__ static glm::uvec2 RNG_rand48_iterate_single(glm::uvec2 Xn, glm::uvec2 A, glm::uvec2 C)
{
	unsigned int R0, R1;

	// low 24-bit multiplication
	const unsigned int lo00 = __umul24(Xn.x, A.x);
	const unsigned int hi00 = __umulhi(Xn.x, A.x);

	// 24bit distribution of 32bit multiplication results
	R0 = (lo00 & 0xFFFFFF);
	R1 = (lo00 >> 24) | (hi00 << 8);

	R0 += C.x; R1 += C.y;

	// transfer overflows
	R1 += (R0 >> 24);
	R0 &= 0xFFFFFF;

	// cross-terms, low/hi 24-bit multiplication
	R1 += __umul24(Xn.y, A.x);
	R1 += __umul24(Xn.x, A.y);

	R1 &= 0xFFFFFF;

	return glm::uvec2(R0, R1);
}

//Templated function
template <int AGENT_TYPE>
__device__ float rnd(RNG_rand48* rand48){

	int index;

	//calculate the agents index in global agent list
	if (AGENT_TYPE == DISCRETE_2D){
		int width = (blockDim.x * gridDim.x);
		glm::ivec2 global_position;
		global_position.x = (blockIdx.x * blockDim.x) + threadIdx.x;
		global_position.y = (blockIdx.y * blockDim.y) + threadIdx.y;
		index = global_position.x + (global_position.y * width);
	}
	else//AGENT_TYPE == CONTINOUS
		index = threadIdx.x + blockIdx.x*blockDim.x;

	glm::uvec2 state = rand48->seeds[index];
	glm::uvec2 A = rand48->A;
	glm::uvec2 C = rand48->C;

	int rand = (state.x >> 17) | (state.y << 7);

	// this actually iterates the RNG
	state = RNG_rand48_iterate_single(state, A, C);

	rand48->seeds[index] = state;

	return (float)rand / 2147483647;
}

__device__ float rnd(RNG_rand48* rand48){
	return rnd<DISCRETE_2D>(rand48);
}

#endif //_FLAMEGPU_KERNELS_H_