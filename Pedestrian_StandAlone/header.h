
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

#ifndef __HEADER
#define __HEADER
#define GLM_FORCE_NO_CTOR_INIT
#include <glm/glm.hpp>

/* General standard definitions */
//Threads per block (agents per block)
#define THREADS_PER_TILE 64
//Definition for any agent function or helper function
#define __FLAME_GPU_FUNC__ __device__
//Definition for a function used to initialise environment variables
#define __FLAME_GPU_INIT_FUNC__
#define __FLAME_GPU_STEP_FUNC__
#define __FLAME_GPU_EXIT_FUNC__

#define USE_CUDA_STREAMS
#define FAST_ATOMIC_SORTING

typedef unsigned int uint;


	

/* Agent population size definifions must be a multiple of THREADS_PER_TILE (defualt 64) */
//Maximum buffer size (largest agent buffer size)
#define buffer_size_MAX 131072

//Maximum population size of xmachine_memory_pedestrian
#define xmachine_memory_pedestrian_MAX 131072
  
  
/* Message poulation size definitions */
//Maximum population size of xmachine_mmessage_properties_message
#define xmachine_message_properties_message_MAX 131072



/* Spatial partitioning grid size definitions */
//xmachine_message_properties_message partition grid size (gridDim.X*gridDim.Y*gridDim.Z)
#define xmachine_message_properties_message_grid_size 14400
  
  
/* enum types */

/**
 * MESSAGE_OUTPUT used for all continuous messaging
 */
enum MESSAGE_OUTPUT{
	single_message,
	optional_message,
};

/**
 * AGENT_TYPE used for templates device message functions
 */
enum AGENT_TYPE{
	CONTINUOUS,
	DISCRETE_2D
};


/* Agent structures */

/** struct xmachine_memory_pedestrian
 * continuous valued agent
 * Holds all agent variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_memory_pedestrian
{
    float x;    /**< X-machine memory variable x of type float.*/
    float y;    /**< X-machine memory variable y of type float.*/
    float vx;    /**< X-machine memory variable vx of type float.*/
    float vy;    /**< X-machine memory variable vy of type float.*/
    float desvx;    /**< X-machine memory variable desvx of type float.*/
    float desvy;    /**< X-machine memory variable desvy of type float.*/
    int count;    /**< X-machine memory variable count of type int.*/
    int lineFail;    /**< X-machine memory variable lineFail of type int.*/
    float newvx;    /**< X-machine memory variable newvx of type float.*/
    float newvy;    /**< X-machine memory variable newvy of type float.*/
    float *orcaLine_direction_x;    /**< X-machine memory variable orcaLine_direction_x of type float.*/
    float *orcaLine_direction_y;    /**< X-machine memory variable orcaLine_direction_y of type float.*/
    float *orcaLine_point_x;    /**< X-machine memory variable orcaLine_point_x of type float.*/
    float *orcaLine_point_y;    /**< X-machine memory variable orcaLine_point_y of type float.*/
    float *projLine_direction_x;    /**< X-machine memory variable projLine_direction_x of type float.*/
    float *projLine_direction_y;    /**< X-machine memory variable projLine_direction_y of type float.*/
    float *projLine_point_x;    /**< X-machine memory variable projLine_point_x of type float.*/
    float *projLine_point_y;    /**< X-machine memory variable projLine_point_y of type float.*/
};



/* Message structures */

/** struct xmachine_message_properties_message
 * Spatial Partitioning
 * Holds all message variables and is aligned to help with coalesced reads on the GPU
 */
struct __align__(16) xmachine_message_properties_message
{	
    /* Spatial Partitioning Variables */
    glm::ivec3 _relative_cell;    /**< Relative cell position from agent grid cell poistion range -1 to 1 */
    int _cell_index_max;    /**< Max boundary value of current cell */
    glm::ivec3 _agent_grid_cell;  /**< Agents partition cell position */
    int _cell_index;        /**< Index of position in current cell */  
      
    float x;        /**< Message variable x of type float.*/  
    float y;        /**< Message variable y of type float.*/  
    float z;        /**< Message variable z of type float.*/  
    float vx;        /**< Message variable vx of type float.*/  
    float vy;        /**< Message variable vy of type float.*/
};



/* Agent lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_memory_pedestrian_list
 * continuous valued agent
 * Variables lists for all agent variables
 */
struct xmachine_memory_pedestrian_list
{	
    /* Temp variables for agents. Used for parallel operations such as prefix sum */
    int _position [xmachine_memory_pedestrian_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_memory_pedestrian_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list x of type float.*/
    float y [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list y of type float.*/
    float vx [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list vx of type float.*/
    float vy [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list vy of type float.*/
    float desvx [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list desvx of type float.*/
    float desvy [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list desvy of type float.*/
    int count [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list count of type int.*/
    int lineFail [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list lineFail of type int.*/
    float newvx [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list newvx of type float.*/
    float newvy [xmachine_memory_pedestrian_MAX];    /**< X-machine memory variable list newvy of type float.*/
    float orcaLine_direction_x [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list orcaLine_direction_x of type float.*/
    float orcaLine_direction_y [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list orcaLine_direction_y of type float.*/
    float orcaLine_point_x [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list orcaLine_point_x of type float.*/
    float orcaLine_point_y [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list orcaLine_point_y of type float.*/
    float projLine_direction_x [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list projLine_direction_x of type float.*/
    float projLine_direction_y [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list projLine_direction_y of type float.*/
    float projLine_point_x [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list projLine_point_x of type float.*/
    float projLine_point_y [xmachine_memory_pedestrian_MAX*128];    /**< X-machine memory variable list projLine_point_y of type float.*/
};



/* Message lists. Structure of Array (SoA) for memory coalescing on GPU */

/** struct xmachine_message_properties_message_list
 * Spatial Partitioning
 * Structure of Array for memory coalescing 
 */
struct xmachine_message_properties_message_list
{
    /* Non discrete messages have temp variables used for reductions with optional message outputs */
    int _position [xmachine_message_properties_message_MAX];    /**< Holds agents position in the 1D agent list */
    int _scan_input [xmachine_message_properties_message_MAX];  /**< Used during parallel prefix sum */
    
    float x [xmachine_message_properties_message_MAX];    /**< Message memory variable list x of type float.*/
    float y [xmachine_message_properties_message_MAX];    /**< Message memory variable list y of type float.*/
    float z [xmachine_message_properties_message_MAX];    /**< Message memory variable list z of type float.*/
    float vx [xmachine_message_properties_message_MAX];    /**< Message memory variable list vx of type float.*/
    float vy [xmachine_message_properties_message_MAX];    /**< Message memory variable list vy of type float.*/
    
};



/* Spatialy Partitioned Message boundary Matrices */

/** struct xmachine_message_properties_message_PBM
 * Partition Boundary Matrix (PBM) for xmachine_message_properties_message 
 */
struct xmachine_message_properties_message_PBM
{
	int start[xmachine_message_properties_message_grid_size];
	int end_or_count[xmachine_message_properties_message_grid_size];
};



  /* Random */
  /** struct RNG_rand48
  *	structure used to hold list seeds
  */
  struct RNG_rand48
  {
  glm::uvec2 A, C;
  glm::uvec2 seeds[buffer_size_MAX];
  };


/** getOutputDir
* Gets the output directory of the simulation. This is the same as the 0.xml input directory.
* @return a const char pointer to string denoting the output directory
*/
const char* getOutputDir();

  /* Random Functions (usable in agent functions) implemented in FLAMEGPU_Kernels */

  /**
  * Templated random function using a DISCRETE_2D template calculates the agent index using a 2D block
  * which requires extra processing but will work for CONTINUOUS agents. Using a CONTINUOUS template will
  * not work for DISCRETE_2D agent.
  * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
  * @return			returns a random float value
  */
  template <int AGENT_TYPE> __FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);
/**
 * Non templated random function calls the templated version with DISCRETE_2D which will work in either case
 * @param	rand48	an RNG_rand48 struct which holds the seeds sued to generate a random number on the GPU
 * @return			returns a random float value
 */
__FLAME_GPU_FUNC__ float rnd(RNG_rand48* rand48);

/* Agent function prototypes */

/**
 * output_properties FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
 * @param properties_message_messages Pointer to output message list of type xmachine_message_properties_message_list. Must be passed as an argument to the add_properties_message_message function ??.
 */
__FLAME_GPU_FUNC__ int output_properties(xmachine_memory_pedestrian* agent, xmachine_message_properties_message_list* properties_message_messages);

/**
 * make_orcaLines FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
 * @param properties_message_messages  properties_message_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_properties_message_message and get_next_properties_message_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_properties_message_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
 */
__FLAME_GPU_FUNC__ int make_orcaLines(xmachine_memory_pedestrian* agent, xmachine_message_properties_message_list* properties_message_messages, xmachine_message_properties_message_PBM* partition_matrix);

/**
 * do_linear_program_2 FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int do_linear_program_2(xmachine_memory_pedestrian* agent);

/**
 * do_linear_program_3 FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int do_linear_program_3(xmachine_memory_pedestrian* agent);

/**
 * move_agents FLAMEGPU Agent Function
 * @param agent Pointer to an agent structre of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
 * @param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an arument to the rand48 function for genertaing random numbers on the GPU.
 */
__FLAME_GPU_FUNC__ int move_agents(xmachine_memory_pedestrian* agent, RNG_rand48* rand48);

  
/* Message Function Prototypes for Spatially Partitioned properties_message message implemented in FLAMEGPU_Kernels */

/** add_properties_message_message
 * Function for all types of message partitioning
 * Adds a new properties_message agent to the xmachine_memory_properties_message_list list using a linear mapping
 * @param agents	xmachine_memory_properties_message_list agent list
 * @param x	message variable of type float
 * @param y	message variable of type float
 * @param z	message variable of type float
 * @param vx	message variable of type float
 * @param vy	message variable of type float
 */
 
 __FLAME_GPU_FUNC__ void add_properties_message_message(xmachine_message_properties_message_list* properties_message_messages, float x, float y, float z, float vx, float vy);
 
/** get_first_properties_message_message
 * Get first message function for spatially partitioned messages
 * @param properties_message_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @param agentx x position of the agent
 * @param agenty y position of the agent
 * @param agentz z position of the agent
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_properties_message * get_first_properties_message_message(xmachine_message_properties_message_list* properties_message_messages, xmachine_message_properties_message_PBM* partition_matrix, float x, float y, float z);

/** get_next_properties_message_message
 * Get first message function for discrete partitioned messages. Template function will call either shared memeory or texture cache implementation depending on AGENT_TYPE
 * @param current the current message struct
 * @param properties_message_messages message list
 * @param partition_matrix the boundary partition matrix for the spatially partitioned message list
 * @return        returns the first message from the message list (offset depending on agent block)
 */
__FLAME_GPU_FUNC__ xmachine_message_properties_message * get_next_properties_message_message(xmachine_message_properties_message* current, xmachine_message_properties_message_list* properties_message_messages, xmachine_message_properties_message_PBM* partition_matrix);
  
  
  
/* Agent Function Prototypes implemented in FLAMEGPU_Kernels */

/** add_pedestrian_agent
 * Adds a new continuous valued pedestrian agent to the xmachine_memory_pedestrian_list list using a linear mapping. Note that any agent variables with an arrayLength are ommited and not support during the creation of new agents on the fly.
 * @param agents xmachine_memory_pedestrian_list agent list
 * @param x	agent agent variable of type float
 * @param y	agent agent variable of type float
 * @param vx	agent agent variable of type float
 * @param vy	agent agent variable of type float
 * @param desvx	agent agent variable of type float
 * @param desvy	agent agent variable of type float
 * @param count	agent agent variable of type int
 * @param lineFail	agent agent variable of type int
 * @param newvx	agent agent variable of type float
 * @param newvy	agent agent variable of type float
 */
__FLAME_GPU_FUNC__ void add_pedestrian_agent(xmachine_memory_pedestrian_list* agents, float x, float y, float vx, float vy, float desvx, float desvy, int count, int lineFail, float newvx, float newvy);

/** get_pedestrian_agent_array_value
 *  Template function for accessing pedestrian agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @return return value
 */
template<typename T>
__FLAME_GPU_FUNC__ T get_pedestrian_agent_array_value(T *array, unsigned int index);

/** set_pedestrian_agent_array_value
 *  Template function for setting pedestrian agent array memory variables.
 *  @param array Agent memory array
 *  @param index to lookup
 *  @param return value
 */
template<typename T>
__FLAME_GPU_FUNC__ void set_pedestrian_agent_array_value(T *array, unsigned int index, T value);


  


  
/* Simulation function prototypes implemented in simulation.cu */

/** initialise
 * Initialise the simulation. Allocated host and device memory. Reads the initial agent configuration from XML.
 * @param input	XML file path for agent initial configuration
 */
extern void initialise(char * input);

/** cleanup
 * Function cleans up any memory allocations on the host and device
 */
extern void cleanup();

/** singleIteration
 *	Performs a single itteration of the simulation. I.e. performs each agent function on each function layer in the correct order.
 */
extern void singleIteration();

/** saveIterationData
 * Reads the current agent data fromt he device and saves it to XML
 * @param	outputpath	file path to XML file used for output of agent data
 * @param	itteration_number
 * @param h_pedestrians Pointer to agent list on the host
 * @param d_pedestrians Pointer to agent list on the GPU device
 * @param h_xmachine_memory_pedestrian_count Pointer to agent counter
 */
extern void saveIterationData(char* outputpath, int iteration_number, xmachine_memory_pedestrian_list* h_pedestrians_default, xmachine_memory_pedestrian_list* d_pedestrians_default, int h_xmachine_memory_pedestrian_default_count);


/** readInitialStates
 * Reads the current agent data fromt he device and saves it to XML
 * @param	inputpath	file path to XML file used for input of agent data
 * @param h_pedestrians Pointer to agent list on the host
 * @param h_xmachine_memory_pedestrian_count Pointer to agent counter
 */
extern void readInitialStates(char* inputpath, xmachine_memory_pedestrian_list* h_pedestrians, int* h_xmachine_memory_pedestrian_count);


/* Return functions used by external code to get agent data from device */

    
/** get_agent_pedestrian_MAX_count
 * Gets the max agent count for the pedestrian agent type 
 * @return		the maximum pedestrian agent count
 */
extern int get_agent_pedestrian_MAX_count();



/** get_agent_pedestrian_default_count
 * Gets the agent count for the pedestrian agent type in state default
 * @return		the current pedestrian agent count in state default
 */
extern int get_agent_pedestrian_default_count();

/** reset_default_count
 * Resets the agent count of the pedestrian in state default to 0. This is usefull for interacting with some visualisations.
 */
extern void reset_pedestrian_default_count();

/** get_device_pedestrian_default_agents
 * Gets a pointer to xmachine_memory_pedestrian_list on the GPU device
 * @return		a xmachine_memory_pedestrian_list on the GPU device
 */
extern xmachine_memory_pedestrian_list* get_device_pedestrian_default_agents();

/** get_host_pedestrian_default_agents
 * Gets a pointer to xmachine_memory_pedestrian_list on the CPU host
 * @return		a xmachine_memory_pedestrian_list on the CPU host
 */
extern xmachine_memory_pedestrian_list* get_host_pedestrian_default_agents();


/** sort_pedestrians_default
 * Sorts an agent state list by providing a CUDA kernal to generate key value pairs
 * @param		a pointer CUDA kernal function to generate key value pairs
 */
void sort_pedestrians_default(void (*generate_key_value_pairs)(unsigned int* keys, unsigned int* values, xmachine_memory_pedestrian_list* agents));


  
  
/* Analytics functions for each varible in each state*/
typedef enum {
  REDUCTION_MAX,
  REDUCTION_MIN,
  REDUCTION_SUM
}reduction_operator;


/** float reduce_pedestrian_default_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_x_variable();



/** float reduce_pedestrian_default_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_y_variable();



/** float reduce_pedestrian_default_vx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_vx_variable();



/** float reduce_pedestrian_default_vy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_vy_variable();



/** float reduce_pedestrian_default_desvx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_desvx_variable();



/** float reduce_pedestrian_default_desvy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_desvy_variable();



/** int reduce_pedestrian_default_count_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_pedestrian_default_count_variable();



/** int count_pedestrian_default_count_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_pedestrian_default_count_variable(int count_value);

/** int reduce_pedestrian_default_lineFail_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
int reduce_pedestrian_default_lineFail_variable();



/** int count_pedestrian_default_lineFail_variable(int count_value){
 * Count can be used for integer only agent variables and allows unique values to be counted using a reduction. Useful for generating histograms.
 * @param count_value The unique value which should be counted
 * @return The number of unique values of the count_value found in the agent state varaible list
 */
int count_pedestrian_default_lineFail_variable(int count_value);

/** float reduce_pedestrian_default_newvx_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_newvx_variable();



/** float reduce_pedestrian_default_newvy_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_newvy_variable();



/** float reduce_pedestrian_default_orcaLine_direction_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_orcaLine_direction_x_variable();



/** float reduce_pedestrian_default_orcaLine_direction_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_orcaLine_direction_y_variable();



/** float reduce_pedestrian_default_orcaLine_point_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_orcaLine_point_x_variable();



/** float reduce_pedestrian_default_orcaLine_point_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_orcaLine_point_y_variable();



/** float reduce_pedestrian_default_projLine_direction_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_projLine_direction_x_variable();



/** float reduce_pedestrian_default_projLine_direction_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_projLine_direction_y_variable();



/** float reduce_pedestrian_default_projLine_point_x_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_projLine_point_x_variable();



/** float reduce_pedestrian_default_projLine_point_y_variable();
 * Reduction functions can be used by visualisations, step and exit functions to gather data for plotting or updating global variables
 * @return the reduced variable value of the specified agent name and state
 */
float reduce_pedestrian_default_projLine_point_y_variable();




  
/* global constant variables */


/** getMaximumBound
 * Returns the maximum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the maximum x, y and z positions of all agents
 */
glm::vec3 getMaximumBounds();

/** getMinimumBounds
 * Returns the minimum agent positions determined from the initial loading of agents
 * @return 	a three component float indicating the minimum x, y and z positions of all agents
 */
glm::vec3 getMinimumBounds();
    
    
#ifdef VISUALISATION
/** initVisualisation
 * Prototype for method which initialises the visualisation. Must be implemented in seperate file
 * @param argc	the argument count from the main function used with GLUT
 * @param argv	the argument values fromt the main function used with GLUT
 */
extern void initVisualisation();

extern void runVisualisation();


#endif

#endif //__HEADER

