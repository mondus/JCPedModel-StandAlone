
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

//Disable internal thrust warnings about conversions
#pragma warning(push)
#pragma warning (disable : 4267)
#pragma warning (disable : 4244)

// includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cmath>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

// include FLAME kernels
#include "FLAMEGPU_kernals.cu"


#pragma warning(pop)

/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line, bool abort = true)
{
	gpuAssert(cudaPeekAtLastError(), file, line);
#ifdef _DEBUG
	gpuAssert(cudaDeviceSynchronize(), file, line);
#endif

}

/* SM padding and offset variables */
int SM_START;
int PADDING;

/* Agent Memory */

/* pedestrian Agent variables these lists are used in the agent function where as the other lists are used only outside the agent functions*/
xmachine_memory_pedestrian_list* d_pedestrians;      /**< Pointer to agent list (population) on the device*/
xmachine_memory_pedestrian_list* d_pedestrians_swap; /**< Pointer to agent list swap on the device (used when killing agents)*/
xmachine_memory_pedestrian_list* d_pedestrians_new;  /**< Pointer to new agent list on the device (used to hold new agents bfore they are appended to the population)*/
int h_xmachine_memory_pedestrian_count;   /**< Agent population size counter */
uint * d_xmachine_memory_pedestrian_keys;	  /**< Agent sort identifiers keys*/
uint * d_xmachine_memory_pedestrian_values;  /**< Agent sort identifiers value */

/* pedestrian state variables */
xmachine_memory_pedestrian_list* h_pedestrians_default;      /**< Pointer to agent list (population) on host*/
xmachine_memory_pedestrian_list* d_pedestrians_default;      /**< Pointer to agent list (population) on the device*/
int h_xmachine_memory_pedestrian_default_count;   /**< Agent population size counter */


/* Message Memory */

/* properties_message Message variables */
xmachine_message_properties_message_list* h_properties_messages;         /**< Pointer to message list on host*/
xmachine_message_properties_message_list* d_properties_messages;         /**< Pointer to message list on device*/
xmachine_message_properties_message_list* d_properties_messages_swap;    /**< Pointer to message swap list on device (used for holding optional messages)*/

/* Non partitioned and spatial partitioned message variables  */
int h_message_properties_message_count;         /**< message list counter*/
int h_message_properties_message_output_type;   /**< message output type (single or optional)*/

/* Spatial Partitioning Variables*/
#ifdef FAST_ATOMIC_SORTING
uint * d_xmachine_message_properties_message_local_bin_index;	  /**< index offset within the assigned bin */
uint * d_xmachine_message_properties_message_unsorted_index;		/**< unsorted index (hash) value for message */
#else
uint * d_xmachine_message_properties_message_keys;	  /**< message sort identifier keys*/
uint * d_xmachine_message_properties_message_values;  /**< message sort identifier values */
#endif
xmachine_message_properties_message_PBM * d_properties_message_partition_matrix;  /**< Pointer to PCB matrix */
glm::vec3 h_message_properties_message_min_bounds;           /**< min bounds (x,y,z) of partitioning environment */
glm::vec3 h_message_properties_message_max_bounds;           /**< max bounds (x,y,z) of partitioning environment */
glm::ivec3 h_message_properties_message_partitionDim;           /**< partition dimensions (x,y,z) of partitioning environment */
float h_message_properties_message_radius;                 /**< partition radius (used to determin the size of the partitions) */

/* Texture offset values for host */
int h_tex_xmachine_message_properties_message_x_offset;

int h_tex_xmachine_message_properties_message_y_offset;

int h_tex_xmachine_message_properties_message_z_offset;

int h_tex_xmachine_message_properties_message_vx_offset;

int h_tex_xmachine_message_properties_message_vy_offset;

int h_tex_xmachine_message_properties_message_pbm_start_offset;
int h_tex_xmachine_message_properties_message_pbm_end_or_count_offset;


/* CUDA Streams for function layers */
cudaStream_t stream1;


/*Global condition counts*/

/* RNG rand48 */
RNG_rand48* h_rand48;    /**< Pointer to RNG_rand48 seed list on host*/
RNG_rand48* d_rand48;    /**< Pointer to RNG_rand48 seed list on device*/

/* CUDA Parallel Primatives variables */
int scan_last_sum;           /**< Indicates if the position (in message list) of last message*/
int scan_last_included;      /**< Indicates if last sum value is included in the total sum count*/

/* Agent function prototypes */

/** pedestrian_output_properties
* Agent function prototype for output_properties function of pedestrian agent
*/
void pedestrian_output_properties(cudaStream_t &stream);

/** pedestrian_make_orcaLines
* Agent function prototype for make_orcaLines function of pedestrian agent
*/
void pedestrian_make_orcaLines(cudaStream_t &stream);

/** pedestrian_do_linear_program_2
* Agent function prototype for do_linear_program_2 function of pedestrian agent
*/
void pedestrian_do_linear_program_2(cudaStream_t &stream);

/** pedestrian_do_linear_program_3
* Agent function prototype for do_linear_program_3 function of pedestrian agent
*/
void pedestrian_do_linear_program_3(cudaStream_t &stream);

/** pedestrian_move_agents
* Agent function prototype for move_agents function of pedestrian agent
*/
void pedestrian_move_agents(cudaStream_t &stream);


void setPaddingAndOffset()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	int x64_sys = 0;

	// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("Error: There is no device supporting CUDA.\n");
		exit(0);
	}

	//check if double is used and supported
#ifdef _DOUBLE_SUPPORT_REQUIRED_
	printf("Simulation requires full precision double values\n");
	if ((deviceProp.major < 2) && (deviceProp.minor < 3)){
		printf("Error: Hardware does not support full precision double values!\n");
		exit(0);
	}

#endif

	//check 32 or 64bit
	x64_sys = (sizeof(void*) == 8);
	if (x64_sys)
	{
		printf("64Bit System Detected\n");
	}
	else
	{
		printf("32Bit System Detected\n");
	}

	SM_START = 0;
	PADDING = 0;

	//copy padding and offset to GPU
	gpuErrchk(cudaMemcpyToSymbol(d_SM_START, &SM_START, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(d_PADDING, &PADDING, sizeof(int)));
}

int is_sqr_pow2(int x){
	int r = (int)pow(4, ceil(log(x) / log(4)));
	return (r == x);
}

int lowest_sqr_pow2(int x){
	int l;

	//escape early if x is square power of 2
	if (is_sqr_pow2(x))
		return x;

	//lower bound
	l = (int)pow(4, floor(log(x) / log(4)));

	return l;
}

/* Unary function required for cudaOccupancyMaxPotentialBlockSizeVariableSMem to avoid warnings */
int no_sm(int b){
	return 0;
}

/* Unary function to return shared memory size for reorder message kernels */
int reorder_messages_sm_size(int blockSize)
{
	return sizeof(unsigned int)*(blockSize + 1);
}


void initialise(char * inputfile){

	//set the padding and offset values depending on architecture and OS
	setPaddingAndOffset();


	printf("Allocating Host and Device memeory\n");

	/* Agent memory allocation (CPU) */
	int xmachine_pedestrian_SoA_size = sizeof(xmachine_memory_pedestrian_list);
	h_pedestrians_default = (xmachine_memory_pedestrian_list*)malloc(xmachine_pedestrian_SoA_size);


	/* Message memory allocation (CPU) */
	int message_properties_message_SoA_size = sizeof(xmachine_message_properties_message_list);
	h_properties_messages = (xmachine_message_properties_message_list*)malloc(message_properties_message_SoA_size);


	//Exit if agent or message buffer sizes are to small for function outpus

	/* Set spatial partitioning properties_message message variables (min_bounds, max_bounds)*/
	h_message_properties_message_radius = (float)5;
	gpuErrchk(cudaMemcpyToSymbol(d_message_properties_message_radius, &h_message_properties_message_radius, sizeof(float)));
	h_message_properties_message_min_bounds = glm::vec3((float)-150, (float)-150, (float)0);
	gpuErrchk(cudaMemcpyToSymbol(d_message_properties_message_min_bounds, &h_message_properties_message_min_bounds, sizeof(glm::vec3)));
	h_message_properties_message_max_bounds = glm::vec3((float)450, (float)450, (float)5);
	gpuErrchk(cudaMemcpyToSymbol(d_message_properties_message_max_bounds, &h_message_properties_message_max_bounds, sizeof(glm::vec3)));
	h_message_properties_message_partitionDim.x = (int)ceil((h_message_properties_message_max_bounds.x - h_message_properties_message_min_bounds.x) / h_message_properties_message_radius);
	h_message_properties_message_partitionDim.y = (int)ceil((h_message_properties_message_max_bounds.y - h_message_properties_message_min_bounds.y) / h_message_properties_message_radius);
	h_message_properties_message_partitionDim.z = (int)ceil((h_message_properties_message_max_bounds.z - h_message_properties_message_min_bounds.z) / h_message_properties_message_radius);
	gpuErrchk(cudaMemcpyToSymbol(d_message_properties_message_partitionDim, &h_message_properties_message_partitionDim, sizeof(glm::ivec3)));


	//read initial states
	readInitialStates(inputfile,
		h_pedestrians_default, &h_xmachine_memory_pedestrian_default_count);


	/* pedestrian Agent memory allocation (GPU) */
	gpuErrchk(cudaMalloc((void**)&d_pedestrians, xmachine_pedestrian_SoA_size));
	gpuErrchk(cudaMalloc((void**)&d_pedestrians_swap, xmachine_pedestrian_SoA_size));
	gpuErrchk(cudaMalloc((void**)&d_pedestrians_new, xmachine_pedestrian_SoA_size));

	//continuous agent sort identifiers
	gpuErrchk(cudaMalloc((void**)&d_xmachine_memory_pedestrian_keys, xmachine_memory_pedestrian_MAX* sizeof(uint)));
	gpuErrchk(cudaMalloc((void**)&d_xmachine_memory_pedestrian_values, xmachine_memory_pedestrian_MAX* sizeof(uint)));

	/* default memory allocation (GPU) */
	gpuErrchk(cudaMalloc((void**)&d_pedestrians_default, xmachine_pedestrian_SoA_size));
	gpuErrchk(cudaMemcpy(d_pedestrians_default, h_pedestrians_default, xmachine_pedestrian_SoA_size, cudaMemcpyHostToDevice));

	/* properties_message Message memory allocation (GPU) */
	gpuErrchk(cudaMalloc((void**)&d_properties_messages, message_properties_message_SoA_size));
	gpuErrchk(cudaMalloc((void**)&d_properties_messages_swap, message_properties_message_SoA_size));
	gpuErrchk(cudaMemcpy(d_properties_messages, h_properties_messages, message_properties_message_SoA_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMalloc((void**)&d_properties_message_partition_matrix, sizeof(xmachine_message_properties_message_PBM)));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaMalloc((void**)&d_xmachine_message_properties_message_local_bin_index, xmachine_message_properties_message_MAX* sizeof(uint)));
	gpuErrchk(cudaMalloc((void**)&d_xmachine_message_properties_message_unsorted_index, xmachine_message_properties_message_MAX* sizeof(uint)));
#else
	gpuErrchk(cudaMalloc((void**)&d_xmachine_message_properties_message_keys, xmachine_message_properties_message_MAX* sizeof(uint)));
	gpuErrchk(cudaMalloc((void**)&d_xmachine_message_properties_message_values, xmachine_message_properties_message_MAX* sizeof(uint)));
#endif



	/*Set global condition counts*/

	/* RNG rand48 */
	int h_rand48_SoA_size = sizeof(RNG_rand48);
	h_rand48 = (RNG_rand48*)malloc(h_rand48_SoA_size);
	//allocate on GPU
	gpuErrchk(cudaMalloc((void**)&d_rand48, h_rand48_SoA_size));
	// calculate strided iteration constants
	static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;
	int seed = 123;
	unsigned long long A, C;
	A = 1LL; C = 0LL;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		C += A*c;
		A *= a;
	}
	h_rand48->A.x = A & 0xFFFFFFLL;
	h_rand48->A.y = (A >> 24) & 0xFFFFFFLL;
	h_rand48->C.x = C & 0xFFFFFFLL;
	h_rand48->C.y = (C >> 24) & 0xFFFFFFLL;
	// prepare first nThreads random numbers from seed
	unsigned long long x = (((unsigned long long)seed) << 16) | 0x330E;
	for (unsigned int i = 0; i < buffer_size_MAX; ++i) {
		x = a*x + c;
		h_rand48->seeds[i].x = x & 0xFFFFFFLL;
		h_rand48->seeds[i].y = (x >> 24) & 0xFFFFFFLL;
	}
	//copy to device
	gpuErrchk(cudaMemcpy(d_rand48, h_rand48, h_rand48_SoA_size, cudaMemcpyHostToDevice));

	/* Call all init functions */


	/* Init CUDA Streams for function layers */

	gpuErrchk(cudaStreamCreate(&stream1));

}


void sort_pedestrians_default(void(*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_pedestrian_list* agents))
{
	int blockSize;
	int minGridSize;
	int gridSize;

	//generate sort keys
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, generate_key_value_pairs, no_sm, h_xmachine_memory_pedestrian_default_count);
	gridSize = (h_xmachine_memory_pedestrian_default_count + blockSize - 1) / blockSize;    // Round up according to array size
	generate_key_value_pairs << <gridSize, blockSize >> >(d_xmachine_memory_pedestrian_keys, d_xmachine_memory_pedestrian_values, d_pedestrians_default);
	gpuErrchkLaunch();

	//updated Thrust sort
	thrust::sort_by_key(thrust::device_pointer_cast(d_xmachine_memory_pedestrian_keys), thrust::device_pointer_cast(d_xmachine_memory_pedestrian_keys) + h_xmachine_memory_pedestrian_default_count, thrust::device_pointer_cast(d_xmachine_memory_pedestrian_values));
	gpuErrchkLaunch();

	//reorder agents
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorder_pedestrian_agents, no_sm, h_xmachine_memory_pedestrian_default_count);
	gridSize = (h_xmachine_memory_pedestrian_default_count + blockSize - 1) / blockSize;    // Round up according to array size
	reorder_pedestrian_agents << <gridSize, blockSize >> >(d_xmachine_memory_pedestrian_values, d_pedestrians_default, d_pedestrians_swap);
	gpuErrchkLaunch();

	//swap
	xmachine_memory_pedestrian_list* d_pedestrians_temp = d_pedestrians_default;
	d_pedestrians_default = d_pedestrians_swap;
	d_pedestrians_swap = d_pedestrians_temp;
}


void cleanup(){

	/* Call all exit functions */


	/* Agent data free*/

	/* pedestrian Agent variables */
	gpuErrchk(cudaFree(d_pedestrians));
	gpuErrchk(cudaFree(d_pedestrians_swap));
	gpuErrchk(cudaFree(d_pedestrians_new));

	free(h_pedestrians_default);
	gpuErrchk(cudaFree(d_pedestrians_default));


	/* Message data free */

	/* properties_message Message variables */
	free(h_properties_messages);
	gpuErrchk(cudaFree(d_properties_messages));
	gpuErrchk(cudaFree(d_properties_messages_swap));
	gpuErrchk(cudaFree(d_properties_message_partition_matrix));
#ifdef FAST_ATOMIC_SORTING
	gpuErrchk(cudaFree(d_xmachine_message_properties_message_local_bin_index));
	gpuErrchk(cudaFree(d_xmachine_message_properties_message_unsorted_index));
#else
	gpuErrchk(cudaFree(d_xmachine_message_properties_message_keys));
	gpuErrchk(cudaFree(d_xmachine_message_properties_message_values));
#endif



	/* CUDA Streams for function layers */

	gpuErrchk(cudaStreamDestroy(stream1));

}

void singleIteration(){

	/* set all non partitioned and spatial partitionded message counts to 0*/
	h_message_properties_message_count = 0;
	//upload to device constant
	gpuErrchk(cudaMemcpyToSymbol(d_message_properties_message_count, &h_message_properties_message_count, sizeof(int)));


	/* Call agent functions in order itterating through the layer functions */

	/* Layer 1*/
	pedestrian_output_properties(stream1);
	cudaDeviceSynchronize();

	/* Layer 2*/
	pedestrian_make_orcaLines(stream1);
	cudaDeviceSynchronize();

	/* Layer 3*/
	//pedestrian_do_linear_program_2(stream1);
	cudaDeviceSynchronize();

	/* Layer 4*/
	//pedestrian_do_linear_program_3(stream1);
	cudaDeviceSynchronize();

	/* Layer 5*/
	//pedestrian_move_agents(stream1);
	cudaDeviceSynchronize();


	/* Call all step functions */
	//sort_func();

}

/* Environment functions */

//host constant declaration



/* Agent data access functions*/


int get_agent_pedestrian_MAX_count(){
	return xmachine_memory_pedestrian_MAX;
}


int get_agent_pedestrian_default_count(){

	//continuous agent
	return h_xmachine_memory_pedestrian_default_count;

}

xmachine_memory_pedestrian_list* get_device_pedestrian_default_agents(){
	return d_pedestrians_default;
}

xmachine_memory_pedestrian_list* get_host_pedestrian_default_agents(){
	return h_pedestrians_default;
}



/*  Analytics Functions */

float reduce_pedestrian_default_x_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->x), thrust::device_pointer_cast(d_pedestrians_default->x) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_y_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->y), thrust::device_pointer_cast(d_pedestrians_default->y) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_vx_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->vx), thrust::device_pointer_cast(d_pedestrians_default->vx) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_vy_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->vy), thrust::device_pointer_cast(d_pedestrians_default->vy) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_desvx_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->desvx), thrust::device_pointer_cast(d_pedestrians_default->desvx) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_desvy_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->desvy), thrust::device_pointer_cast(d_pedestrians_default->desvy) + h_xmachine_memory_pedestrian_default_count);
}

int reduce_pedestrian_default_count_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->count), thrust::device_pointer_cast(d_pedestrians_default->count) + h_xmachine_memory_pedestrian_default_count);
}

int count_pedestrian_default_count_variable(int count_value){
	//count in default stream
	return (int)thrust::count(thrust::device_pointer_cast(d_pedestrians_default->count), thrust::device_pointer_cast(d_pedestrians_default->count) + h_xmachine_memory_pedestrian_default_count, count_value);
}
int reduce_pedestrian_default_lineFail_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->lineFail), thrust::device_pointer_cast(d_pedestrians_default->lineFail) + h_xmachine_memory_pedestrian_default_count);
}

int count_pedestrian_default_lineFail_variable(int count_value){
	//count in default stream
	return (int)thrust::count(thrust::device_pointer_cast(d_pedestrians_default->lineFail), thrust::device_pointer_cast(d_pedestrians_default->lineFail) + h_xmachine_memory_pedestrian_default_count, count_value);
}
float reduce_pedestrian_default_newvx_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->newvx), thrust::device_pointer_cast(d_pedestrians_default->newvx) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_newvy_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->newvy), thrust::device_pointer_cast(d_pedestrians_default->newvy) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_orcaLine_direction_x_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->orcaLine_direction_x), thrust::device_pointer_cast(d_pedestrians_default->orcaLine_direction_x) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_orcaLine_direction_y_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->orcaLine_direction_y), thrust::device_pointer_cast(d_pedestrians_default->orcaLine_direction_y) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_orcaLine_point_x_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->orcaLine_point_x), thrust::device_pointer_cast(d_pedestrians_default->orcaLine_point_x) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_orcaLine_point_y_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->orcaLine_point_y), thrust::device_pointer_cast(d_pedestrians_default->orcaLine_point_y) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_projLine_direction_x_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->projLine_direction_x), thrust::device_pointer_cast(d_pedestrians_default->projLine_direction_x) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_projLine_direction_y_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->projLine_direction_y), thrust::device_pointer_cast(d_pedestrians_default->projLine_direction_y) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_projLine_point_x_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->projLine_point_x), thrust::device_pointer_cast(d_pedestrians_default->projLine_point_x) + h_xmachine_memory_pedestrian_default_count);
}

float reduce_pedestrian_default_projLine_point_y_variable(){
	//reduce in default stream
	return thrust::reduce(thrust::device_pointer_cast(d_pedestrians_default->projLine_point_y), thrust::device_pointer_cast(d_pedestrians_default->projLine_point_y) + h_xmachine_memory_pedestrian_default_count);
}




/* Agent functions */



/* Shared memory size calculator for agent function */
int pedestrian_output_properties_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;

	return sm_size;
}

/** pedestrian_output_properties
* Agent function prototype for output_properties function of pedestrian agent
*/
void pedestrian_output_properties(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************

	/*//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_default;
	d_pedestrians_default = pedestrians_default_temp;
	//set working count to current state count
	h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));
	//set current state count to 0
	h_xmachine_memory_pedestrian_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/

	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));

	//******************************** AGENT FUNCTION *******************************


	//CONTINUOUS AGENT CHECK FUNCTION OUTPUT BUFFERS FOR OUT OF BOUNDS
	//if (h_message_properties_message_count + h_xmachine_memory_pedestrian_count > xmachine_message_properties_message_MAX){
	if (h_message_properties_message_count + h_xmachine_memory_pedestrian_default_count > xmachine_message_properties_message_MAX) {
		printf("Error: Buffer size of properties_message message will be exceeded in function output_properties\n");
		exit(0);
	}


	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_output_properties, pedestrian_output_properties_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;

	sm_size = pedestrian_output_properties_sm_size(blockSize);



	//SET THE OUTPUT MESSAGE TYPE FOR CONTINUOUS AGENTS

	//Set the message_type for non partitioned and spatially partitioned message outputs
	h_message_properties_message_output_type = single_message;
	gpuErrchk(cudaMemcpyToSymbol(d_message_properties_message_output_type, &h_message_properties_message_output_type, sizeof(int)));


	//MAIN XMACHINE FUNCTION CALL (output_properties)
	//Reallocate   : false
	//Input        : 
	//Output       : properties_message
	//Agent Output : 
	GPUFLAME_output_properties << <g, b, sm_size, stream >> >(d_pedestrians_default
		, d_properties_messages);
	gpuErrchkLaunch();


	//CONTINUOUS AGENTS SCATTER NON PARTITIONED OPTIONAL OUTPUT MESSAGES

	//UPDATE MESSAGE COUNTS FOR CONTINUOUS AGENTS WITH NON PARTITIONED MESSAGE OUTPUT 
	//h_message_properties_message_count += h_xmachine_memory_pedestrian_count;
	h_message_properties_message_count += h_xmachine_memory_pedestrian_default_count;
	//Copy count to device
	gpuErrchk(cudaMemcpyToSymbol(d_message_properties_message_count, &h_message_properties_message_count, sizeof(int)));

	//reset partition matrix
	gpuErrchk(cudaMemset((void*)d_properties_message_partition_matrix, 0, sizeof(xmachine_message_properties_message_PBM)));
	//PR Bug fix: Second fix. This should prevent future problems when multiple agents write the same message as now the message structure is completely rebuilt after an output.
	if (h_message_properties_message_count > 0){
#ifdef FAST_ATOMIC_SORTING
		//USE ATOMICS TO BUILD PARTITION BOUNDARY
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, hist_properties_message_messages, no_sm, h_message_properties_message_count);
		gridSize = (h_message_properties_message_count + blockSize - 1) / blockSize;
		hist_properties_message_messages << <gridSize, blockSize, 0, stream >> >(d_xmachine_message_properties_message_local_bin_index, d_xmachine_message_properties_message_unsorted_index, d_properties_message_partition_matrix->end_or_count, d_properties_messages, h_message_properties_message_count);
		gpuErrchkLaunch();

		thrust::device_ptr<int> ptr_count = thrust::device_pointer_cast(d_properties_message_partition_matrix->end_or_count);
		thrust::device_ptr<int> ptr_index = thrust::device_pointer_cast(d_properties_message_partition_matrix->start);
		thrust::exclusive_scan(thrust::cuda::par.on(stream), ptr_count, ptr_count + xmachine_message_properties_message_grid_size, ptr_index); // scan

		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorder_properties_message_messages, no_sm, h_message_properties_message_count);
		gridSize = (h_message_properties_message_count + blockSize - 1) / blockSize; 	// Round up according to array size
		reorder_properties_message_messages << <gridSize, blockSize, 0, stream >> >(d_xmachine_message_properties_message_local_bin_index, d_xmachine_message_properties_message_unsorted_index, d_properties_message_partition_matrix->start, d_properties_messages, d_properties_messages_swap, h_message_properties_message_count);
		gpuErrchkLaunch();
#else
		//HASH, SORT, REORDER AND BUILD PMB FOR SPATIAL PARTITIONING MESSAGE OUTPUTS
		//Get message hash values for sorting
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, hash_properties_message_messages, no_sm, h_message_properties_message_count);
		gridSize = (h_message_properties_message_count + blockSize - 1) / blockSize;
		hash_properties_message_messages << <gridSize, blockSize, 0, stream >> >(d_xmachine_message_properties_message_keys, d_xmachine_message_properties_message_values, d_properties_messages);
		gpuErrchkLaunch();
		//Sort
		thrust::sort_by_key(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_xmachine_message_properties_message_keys), thrust::device_pointer_cast(d_xmachine_message_properties_message_keys) + h_message_properties_message_count, thrust::device_pointer_cast(d_xmachine_message_properties_message_values));
		gpuErrchkLaunch();
		//reorder and build pcb
		gpuErrchk(cudaMemset(d_properties_message_partition_matrix->start, 0xffffffff, xmachine_message_properties_message_grid_size* sizeof(int)));
		cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reorder_properties_message_messages, reorder_messages_sm_size, h_message_properties_message_count);
		gridSize = (h_message_properties_message_count + blockSize - 1) / blockSize;
		int reorder_sm_size = reorder_messages_sm_size(blockSize);
		reorder_properties_message_messages << <gridSize, blockSize, reorder_sm_size, stream >> >(d_xmachine_message_properties_message_keys, d_xmachine_message_properties_message_values, d_properties_message_partition_matrix, d_properties_messages, d_properties_messages_swap);
		gpuErrchkLaunch();
#endif
	}
	//swap ordered list
	xmachine_message_properties_message_list* d_properties_messages_temp = d_properties_messages;
	d_properties_messages = d_properties_messages_swap;
	d_properties_messages_swap = d_properties_messages_temp;


	//************************ MOVE AGENTS TO NEXT STATE ****************************

	/*//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count+h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX){
	printf("Error: Buffer size of output_properties agents in state default will be exceeded moving working agents to next state in function output_properties\n");
	exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents<<<gridSize, blockSize, 0, stream>>>(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/


}




/* Shared memory size calculator for agent function */
int pedestrian_make_orcaLines_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;

	//Continuous agent and message input is spatially partitioned
	sm_size += (blockSize * sizeof(xmachine_message_properties_message));

	//all continuous agent types require single 32bit word per thread offset (to avoid sm bank conflicts)
	sm_size += (blockSize * PADDING);

	return sm_size;
}

/** pedestrian_make_orcaLines
* Agent function prototype for make_orcaLines function of pedestrian agent
*/
void pedestrian_make_orcaLines(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************

	/*//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_default;
	d_pedestrians_default = pedestrians_default_temp;*/
	//set working count to current state count
	/*h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));
	//set current state count to 0
	h_xmachine_memory_pedestrian_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/

	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));

	//******************************** AGENT FUNCTION *******************************



	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_make_orcaLines, pedestrian_make_orcaLines_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;

	sm_size = pedestrian_make_orcaLines_sm_size(blockSize);



	//BIND APPROPRIATE MESSAGE INPUT VARIABLES TO TEXTURES (to make use of the texture cache)

	//any agent with discrete or partitioned message input uses texture caching

	size_t tex_xmachine_message_properties_message_x_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_properties_message_x_byte_offset, tex_xmachine_message_properties_message_x, d_properties_messages->x, sizeof(float)*xmachine_message_properties_message_MAX));
	h_tex_xmachine_message_properties_message_x_offset = (int)tex_xmachine_message_properties_message_x_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_properties_message_x_offset, &h_tex_xmachine_message_properties_message_x_offset, sizeof(int)));

	size_t tex_xmachine_message_properties_message_y_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_properties_message_y_byte_offset, tex_xmachine_message_properties_message_y, d_properties_messages->y, sizeof(float)*xmachine_message_properties_message_MAX));
	h_tex_xmachine_message_properties_message_y_offset = (int)tex_xmachine_message_properties_message_y_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_properties_message_y_offset, &h_tex_xmachine_message_properties_message_y_offset, sizeof(int)));

	size_t tex_xmachine_message_properties_message_z_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_properties_message_z_byte_offset, tex_xmachine_message_properties_message_z, d_properties_messages->z, sizeof(float)*xmachine_message_properties_message_MAX));
	h_tex_xmachine_message_properties_message_z_offset = (int)tex_xmachine_message_properties_message_z_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_properties_message_z_offset, &h_tex_xmachine_message_properties_message_z_offset, sizeof(int)));

	size_t tex_xmachine_message_properties_message_vx_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_properties_message_vx_byte_offset, tex_xmachine_message_properties_message_vx, d_properties_messages->vx, sizeof(float)*xmachine_message_properties_message_MAX));
	h_tex_xmachine_message_properties_message_vx_offset = (int)tex_xmachine_message_properties_message_vx_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_properties_message_vx_offset, &h_tex_xmachine_message_properties_message_vx_offset, sizeof(int)));

	size_t tex_xmachine_message_properties_message_vy_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_properties_message_vy_byte_offset, tex_xmachine_message_properties_message_vy, d_properties_messages->vy, sizeof(float)*xmachine_message_properties_message_MAX));
	h_tex_xmachine_message_properties_message_vy_offset = (int)tex_xmachine_message_properties_message_vy_byte_offset / sizeof(float);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_properties_message_vy_offset, &h_tex_xmachine_message_properties_message_vy_offset, sizeof(int)));

	//bind pbm start and end indices to textures
	size_t tex_xmachine_message_properties_message_pbm_start_byte_offset;
	size_t tex_xmachine_message_properties_message_pbm_end_or_count_byte_offset;
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_properties_message_pbm_start_byte_offset, tex_xmachine_message_properties_message_pbm_start, d_properties_message_partition_matrix->start, sizeof(int)*xmachine_message_properties_message_grid_size));
	h_tex_xmachine_message_properties_message_pbm_start_offset = (int)tex_xmachine_message_properties_message_pbm_start_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_properties_message_pbm_start_offset, &h_tex_xmachine_message_properties_message_pbm_start_offset, sizeof(int)));
	gpuErrchk(cudaBindTexture(&tex_xmachine_message_properties_message_pbm_end_or_count_byte_offset, tex_xmachine_message_properties_message_pbm_end_or_count, d_properties_message_partition_matrix->end_or_count, sizeof(int)*xmachine_message_properties_message_grid_size));
	h_tex_xmachine_message_properties_message_pbm_end_or_count_offset = (int)tex_xmachine_message_properties_message_pbm_end_or_count_byte_offset / sizeof(int);
	gpuErrchk(cudaMemcpyToSymbol(d_tex_xmachine_message_properties_message_pbm_end_or_count_offset, &h_tex_xmachine_message_properties_message_pbm_end_or_count_offset, sizeof(int)));



	//MAIN XMACHINE FUNCTION CALL (make_orcaLines)
	//Reallocate   : false
	//Input        : properties_message
	//Output       : 
	//Agent Output : 
	GPUFLAME_make_orcaLines << <g, b, sm_size, stream >> >(d_pedestrians_default, d_properties_messages
		, d_properties_message_partition_matrix
		);
	gpuErrchkLaunch();


	//UNBIND MESSAGE INPUT VARIABLE TEXTURES

	//any agent with discrete or partitioned message input uses texture caching

	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_properties_message_x));

	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_properties_message_y));

	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_properties_message_z));

	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_properties_message_vx));

	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_properties_message_vy));

	//unbind pbm indices
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_properties_message_pbm_start));
	gpuErrchk(cudaUnbindTexture(tex_xmachine_message_properties_message_pbm_end_or_count));


	//************************ MOVE AGENTS TO NEXT STATE ****************************

	/*//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count+h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX){
	printf("Error: Buffer size of make_orcaLines agents in state default will be exceeded moving working agents to next state in function make_orcaLines\n");
	exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents<<<gridSize, blockSize, 0, stream>>>(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/


}




/* Shared memory size calculator for agent function */
int pedestrian_do_linear_program_2_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;

	return sm_size;
}

//My version
void pedestrian_do_linear_program_2(cudaStream_t &stream) {

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************

	/*//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_default;
	d_pedestrians_default = pedestrians_default_temp;
	//set working count to current state count
	h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));
	//set current state count to 0
	h_xmachine_memory_pedestrian_default_count = 0;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/

	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));
	//******************************** AGENT FUNCTION *******************************



	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_do_linear_program_2, pedestrian_do_linear_program_2_sm_size, state_list_size);
	//cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_do_linear_program_2_COMP, pedestrian_do_linear_program_2_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;

	sm_size = pedestrian_do_linear_program_2_sm_size(blockSize);


	b.x = blockdimsize;
	g.x = 80000 / blockdimsize + 1;

	//MAIN XMACHINE FUNCTION CALL (do_linear_program_2)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_do_linear_program_2_COMP << <g, b, sm_size, stream >> >(d_pedestrians_default);
	gpuErrchkLaunch();



	//************************ MOVE AGENTS TO NEXT STATE ****************************

	/*//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count + h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX) {
	printf("Error: Buffer size of do_linear_program_2 agents in state default will be exceeded moving working agents to next state in function do_linear_program_2\n");
	exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/


}






/* Shared memory size calculator for agent function */
int pedestrian_do_linear_program_3_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;

	return sm_size;
}

/** pedestrian_do_linear_program_3
* Agent function prototype for do_linear_program_3 function of pedestrian agent
*/
void pedestrian_do_linear_program_3(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************

	//CONTINUOUS AGENT FUNCTION AND THERE IS A FUNCTION CONDITION

	//COPY CURRENT STATE COUNT TO WORKING COUNT (host and device)
	h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));

	//RESET SCAN INPUTS
	//reset scan input for currentState
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, reset_pedestrian_scan_input, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	reset_pedestrian_scan_input << <gridSize, blockSize, 0, stream >> >(d_pedestrians_default);
	gpuErrchkLaunch();
	//reset scan input for working lists
	reset_pedestrian_scan_input << <gridSize, blockSize, 0, stream >> >(d_pedestrians);
	gpuErrchkLaunch();

	//APPLY FUNCTION FILTER
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, do_linear_program_3_function_filter, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	do_linear_program_3_function_filter << <gridSize, blockSize, 0, stream >> >(d_pedestrians_default, d_pedestrians);
	gpuErrchkLaunch();

	//GRID AND BLOCK SIZE FOR COMPACT
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, scatter_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;

	//COMPACT CURRENT STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_pedestrians_default->_scan_input), thrust::device_pointer_cast(d_pedestrians_default->_scan_input) + h_xmachine_memory_pedestrian_count, thrust::device_pointer_cast(d_pedestrians_default->_position));
	//reset agent count
	gpuErrchk(cudaMemcpy(&scan_last_sum, &d_pedestrians_default->_position[h_xmachine_memory_pedestrian_count - 1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&scan_last_included, &d_pedestrians_default->_scan_input[h_xmachine_memory_pedestrian_count - 1], sizeof(int), cudaMemcpyDeviceToHost));
	if (scan_last_included == 1)
		h_xmachine_memory_pedestrian_default_count = scan_last_sum + 1;
	else
		h_xmachine_memory_pedestrian_default_count = scan_last_sum;
	//Scatter into swap
	scatter_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians_swap, d_pedestrians_default, 0, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
	//use a temp pointer change working swap list with current state list
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians_default;
	d_pedestrians_default = d_pedestrians_swap;
	d_pedestrians_swap = pedestrians_default_temp;
	//update the device count
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));

	//COMPACT WORKING STATE LIST
	thrust::exclusive_scan(thrust::cuda::par.on(stream), thrust::device_pointer_cast(d_pedestrians->_scan_input), thrust::device_pointer_cast(d_pedestrians->_scan_input) + h_xmachine_memory_pedestrian_count, thrust::device_pointer_cast(d_pedestrians->_position));
	//reset agent count
	gpuErrchk(cudaMemcpy(&scan_last_sum, &d_pedestrians->_position[h_xmachine_memory_pedestrian_count - 1], sizeof(int), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&scan_last_included, &d_pedestrians->_scan_input[h_xmachine_memory_pedestrian_count - 1], sizeof(int), cudaMemcpyDeviceToHost));
	//Scatter into swap
	scatter_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians_swap, d_pedestrians, 0, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
	//update working agent count after the scatter
	if (scan_last_included == 1)
		h_xmachine_memory_pedestrian_count = scan_last_sum + 1;
	else
		h_xmachine_memory_pedestrian_count = scan_last_sum;
	//use a temp pointer change working swap list with current state list
	xmachine_memory_pedestrian_list* pedestrians_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_swap;
	d_pedestrians_swap = pedestrians_temp;
	//update the device count
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));

	//CHECK WORKING LIST COUNT IS NOT EQUAL TO 0
	if (h_xmachine_memory_pedestrian_count == 0)
	{
		return;
	}


	//Update the state list size for occupancy calculations
	state_list_size = h_xmachine_memory_pedestrian_count;



	//******************************** AGENT FUNCTION *******************************



	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_do_linear_program_3, pedestrian_do_linear_program_3_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;

	sm_size = pedestrian_do_linear_program_3_sm_size(blockSize);




	//MAIN XMACHINE FUNCTION CALL (do_linear_program_3)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_do_linear_program_3 << <g, b, sm_size, stream >> >(d_pedestrians);
	gpuErrchkLaunch();



	//************************ MOVE AGENTS TO NEXT STATE ****************************

	//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count + h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX){
		printf("Error: Buffer size of do_linear_program_3 agents in state default will be exceeded moving working agents to next state in function do_linear_program_3\n");
		exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents << <gridSize, blockSize, 0, stream >> >(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));


}




/* Shared memory size calculator for agent function */
int pedestrian_move_agents_sm_size(int blockSize){
	int sm_size;
	sm_size = SM_START;

	return sm_size;
}

/** pedestrian_move_agents
* Agent function prototype for move_agents function of pedestrian agent
*/
void pedestrian_move_agents(cudaStream_t &stream){

	int sm_size;
	int blockSize;
	int minGridSize;
	int gridSize;
	int state_list_size;
	dim3 g; //grid for agent func
	dim3 b; //block for agent func


	//CHECK THE CURRENT STATE LIST COUNT IS NOT EQUAL TO 0

	if (h_xmachine_memory_pedestrian_default_count == 0)
	{
		return;
	}


	//SET SM size to 0 and save state list size for occupancy calculations
	sm_size = SM_START;
	state_list_size = h_xmachine_memory_pedestrian_default_count;



	//******************************** AGENT FUNCTION CONDITION *********************

	/*//THERE IS NOT A FUNCTION CONDITION
	//currentState maps to working list
	xmachine_memory_pedestrian_list* pedestrians_default_temp = d_pedestrians;
	d_pedestrians = d_pedestrians_default;
	d_pedestrians_default = pedestrians_default_temp;
	//set working count to current state count
	h_xmachine_memory_pedestrian_count = h_xmachine_memory_pedestrian_default_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_count, sizeof(int)));
	//set current state count to 0
	h_xmachine_memory_pedestrian_default_count = 0;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/

	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));
	gpuErrchk(cudaMemcpyToSymbol(d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));

	//******************************** AGENT FUNCTION *******************************



	//calculate the grid block size for main agent function
	cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, GPUFLAME_move_agents, pedestrian_move_agents_sm_size, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	b.x = blockSize;
	g.x = gridSize;

	sm_size = pedestrian_move_agents_sm_size(blockSize);




	//MAIN XMACHINE FUNCTION CALL (move_agents)
	//Reallocate   : false
	//Input        : 
	//Output       : 
	//Agent Output : 
	GPUFLAME_move_agents << <g, b, sm_size, stream >> >(d_pedestrians_default, d_rand48);
	gpuErrchkLaunch();



	//************************ MOVE AGENTS TO NEXT STATE ****************************

	/*//check the working agents wont exceed the buffer size in the new state list
	if (h_xmachine_memory_pedestrian_default_count+h_xmachine_memory_pedestrian_count > xmachine_memory_pedestrian_MAX){
	printf("Error: Buffer size of move_agents agents in state default will be exceeded moving working agents to next state in function move_agents\n");
	exit(0);
	}
	//append agents to next state list
	cudaOccupancyMaxPotentialBlockSizeVariableSMem( &minGridSize, &blockSize, append_pedestrian_Agents, no_sm, state_list_size);
	gridSize = (state_list_size + blockSize - 1) / blockSize;
	append_pedestrian_Agents<<<gridSize, blockSize, 0, stream>>>(d_pedestrians_default, d_pedestrians, h_xmachine_memory_pedestrian_default_count, h_xmachine_memory_pedestrian_count);
	gpuErrchkLaunch();
	//update new state agent size
	h_xmachine_memory_pedestrian_default_count += h_xmachine_memory_pedestrian_count;
	gpuErrchk( cudaMemcpyToSymbol( d_xmachine_memory_pedestrian_default_count, &h_xmachine_memory_pedestrian_default_count, sizeof(int)));*/


}



extern void reset_pedestrian_default_count()
{
	h_xmachine_memory_pedestrian_default_count = 0;
}