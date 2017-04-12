
/*
* Copyright 2011 University of Sheffield.
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

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#include "header.h"

//Environment Bounds
#define MIN_POSITION 0.8f
#define MAX_POSITION 300.0f

#define timeStep_ 0.8f
#define timeHorizon_ 5.0f
#define timeHorizonObst_ 5.0f
#define RVO_EPSILON 0.00001f
#define AGENTNO 128 //Adjust//Size of orcaLines max array size (same as within XMLmodel file array length)
#define maxSpeed_ 1.0f
#define RADIUS 0.5f
#define LOOKRADIUS 5.0f //equal to environment radius bin size.

#define NELEMS(x) (sizeof(x) / sizeof((x)[0]))


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Helper functions/////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//If agent is past bounds return a new random velocity vector back into the bounds
__FLAME_GPU_FUNC__ glm::vec2 isAtEdge(glm::vec2 agent_position, glm::vec2 agent_desv, RNG_rand48* rand48)
{
	float rand = rnd<CONTINUOUS>(rand48)*0.5f;

	//If the agent is close to the edge, give it a new random velocity
	if (agent_position.x < MIN_POSITION)
	{
		agent_desv.x = rand;
		//agent_desv.x = fabs(agent_desv.x);
		//agent_desv.x = 0.3f;
	}
	else if (agent_position.x > MAX_POSITION) 
	{
		agent_desv.x = -rand;
		//agent_desv.x = -fabs(agent_desv.x);
		//agent_desv.x = -0.3f;
	}
	if (agent_position.y < MIN_POSITION)
	{
		agent_desv.y = rand;
		//agent_desv.y = fabs(agent_desv.y);
		//agent_desv.y = 0.3f;
	}
	else if (agent_position.y > MAX_POSITION)
	{
		agent_desv.y = -rand;
		//agent_desv.y = -fabs(agent_desv.y);
		//agent_desv.y = -0.3f;
	}

	
	return agent_desv;
}

//Keeps agents within bounds defined
__FLAME_GPU_FUNC__ glm::vec2 boundAgents(glm::vec2 agent_position)
{
	agent_position.x = (agent_position.x < MIN_POSITION) ? MAX_POSITION : agent_position.x;
	agent_position.x = (agent_position.x > MAX_POSITION) ? MIN_POSITION : agent_position.x;

	agent_position.y = (agent_position.y < MIN_POSITION) ? MAX_POSITION : agent_position.y;
	agent_position.y = (agent_position.y > MAX_POSITION) ? MIN_POSITION : agent_position.y;

	return agent_position;
}

//Determinant of 2 2d vectors
__FLAME_GPU_FUNC__ float det(glm::vec2 v1, glm::vec2 v2)
{
	return (v1.x * v2.y) - (v1.y * v2.x);
}

//Vector doted with itself
__FLAME_GPU_FUNC__ float absSq(const glm::vec2 &vector)
{
	return glm::dot(vector, vector);
}

//Squar of value a
__FLAME_GPU_FUNC__ float sqr(float a)
{
	return a * a;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Linear Program functions/////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Versions used agent memory

//1D minimization to find closest possible solution of @optVelocity with computed orcaLines
//agent: pointer to agent of interest for calculations
//@lineNo: which line failed within linearProgram2
//@radius: maximum cutoff for consideration, usually set to agent's maximum velocity
//@optVelocity: the agents optimized velocity, usually set to its prefered veloicty. Saves a solution in @result that is the closest possible 
//@directionOpt: whether the optimized velocity needs normalizing
//@result: the agent's velocity at the end of the calculations
//@useProjLines: whether to use orcaLines or projLines (i.e. if running from makeOrcaLines or from LP3 function respectively)
//return -- false if failed to find solution, true otherwise
__FLAME_GPU_FUNC__ bool linearProgram1(xmachine_memory_pedestrian* agent, int lineNo, float radius, const glm::vec2 &optVelocity, bool directionOpt, glm::vec2 &result, bool useProjLines = false)
{
	glm::vec2 lines_direction_lineNo;
	glm::vec2 lines_point_lineNo;
	if (useProjLines) {
		lines_direction_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_direction_x, lineNo), get_pedestrian_agent_array_value<float>(agent->projLine_direction_y, lineNo));
		lines_point_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_point_x, lineNo), get_pedestrian_agent_array_value<float>(agent->projLine_point_y, lineNo));
	}
	else {
		lines_direction_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, lineNo), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, lineNo));
		lines_point_lineNo = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, lineNo), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, lineNo));
	}

	const float dotProduct = glm::dot(lines_point_lineNo, lines_direction_lineNo);
	const float discriminant = sqr(dotProduct) + sqr(radius) - absSq(lines_point_lineNo);

	if (discriminant < 0.0f) {
		// Max speed circle fully invalidates line lineNo. 
		return false;
	}

	float tLeft = -dotProduct - std::sqrt(discriminant);
	float tRight = -dotProduct + std::sqrt(discriminant);

	for (int i = 0; i < lineNo; ++i) {
		glm::vec2 lines_direction_i;
		glm::vec2 lines_point_i;
		if (useProjLines) {
			lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_direction_y, i));
			lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_point_y, i));
		}
		else {
			lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, i));
			lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, i));
		}

		const float denominator = det(lines_direction_lineNo, lines_direction_i);
		const float numerator = det(lines_direction_i, lines_point_lineNo - lines_point_i);

		if (fabsf(denominator) <= RVO_EPSILON) {
			// Lines lineNo and i are (almost) parallel. 
			if (numerator < 0.0f) {
				return false;
			}
			else {
				continue;
			}
		}

		const float t = numerator / denominator;

		if (denominator >= 0.0f) {
			// Line i bounds line lineNo on the right. 
			tRight = fminf(tRight, t);
		}
		else {
			// Line i bounds line lineNo on the left. 
			tLeft = fmaxf(tLeft, t);
		}

		if (tLeft > tRight) {
			return false;
		}
	}

	if (directionOpt) {
		// Optimize direction. 
		if (glm::dot(optVelocity, lines_direction_lineNo) > 0.0f) {
			/* Take right extreme. */
			result = lines_point_lineNo + tRight * lines_direction_lineNo;
		}
		else {
			// Take left extreme. 
			result = lines_point_lineNo + tLeft * lines_direction_lineNo;
		}
	}
	else {
		// Optimize closest point. 
		const float t = glm::dot(lines_direction_lineNo, optVelocity - lines_point_lineNo);

		if (t < tLeft) {
			result = lines_point_lineNo + tLeft * lines_direction_lineNo;
		}
		else if (t > tRight) {
			result = lines_point_lineNo + tRight * lines_direction_lineNo;
		}
		else {
			result = lines_point_lineNo + t * lines_direction_lineNo;
		}
	}

	return true;
}

//Checks to see if current velocity satisfies the constrains of orcaLines. Calls |linearProgram1| if not satisfied
//@agent: pointer to agent of interest for calculations
//@count: actual number of elements within arrays to consider
//@radius: maximum cutoff for consideration, usually set to agent's maximum velocity
//@optVelocity: the agents optimized velocity, usually set to its prefered veloicty. Saves a solution in @result that is the closest possible 
//@directionOpt: whether the optimized velocity needs normalizing
//@result: the velocity at the end of the calculations
//@useProjLines: whether to use orcaLines or projLines (i.e. if running from makeOrcaLines or from LP3 function respectively)
//return -- count if solution is found. -1<i<count otherwise where i is the constraint that could not be solved
//template <bool useProjLines>
__FLAME_GPU_FUNC__ int linearProgram2(xmachine_memory_pedestrian* agent, const int count, float radius, const glm::vec2 optVelocity, bool directionOpt, glm::vec2 &result, bool useProjLines = false)
{
	if (directionOpt) {
		// Optimize direction. Note that the optimization velocity is of unit length in this case.
		result = optVelocity * radius;
	}
	else if (absSq(optVelocity) > sqr(radius)) {
		// Optimize closest point and outside circle. 
		result = glm::normalize(optVelocity) * radius;
	}
	else {
		// Optimize closest point and inside circle. 
		result = optVelocity;
	}
	//If first time calling lp2
	if (useProjLines == false) {
		for (int i = 0; i < count; ++i) {
			glm::vec2 lines_direction_i;
			glm::vec2 lines_point_i;

			lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, i));
			lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, i));

			if (det(lines_direction_i, lines_point_i - result) > 0.0f) {
				// Result does not satisfy constraint i. Compute new optimal result. 
				int index = threadIdx.x + (blockDim.x * blockIdx.x);
				if (index < 10) {
					printf("agent: %i \t i: %i\n", index,i);
				}

				const glm::vec2 tempResult = result;

				if (!linearProgram1(agent, i, radius, optVelocity, directionOpt, result, useProjLines)) {
					result = tempResult;
					return i;
				}
			}
		}
	}
	else { //calling from lp3
		for (int i = 0; i < count; ++i) {
			glm::vec2 lines_direction_i;
			glm::vec2 lines_point_i;

			lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_direction_y, i));
			lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->projLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->projLine_point_y, i));

			if (det(lines_direction_i, lines_point_i - result) > 0.0f) {
				// Result does not satisfy constraint i. Compute new optimal result. 
				const glm::vec2 tempResult = result;

				if (!linearProgram1(agent, i, radius, optVelocity, directionOpt, result, useProjLines)) {
					result = tempResult;
					return i;
				}
			}
		}
	}

	return count;
}

//Finds best solution to satisfy constrants of orcaLines
//@agent: pointer to agent of interest for calculations
//@numObstLines: the number of orcaLines which correspond to obstacles (0 for me as no obstacles implemented)
//@beginLine: first line to not have solution from running linearProgram1 and 2
//@radius: maximum cutoff for consideration, usually set to agent's maximum velocity
//@result: the velocity at the end of the calculations
__FLAME_GPU_FUNC__ void linearProgram3(xmachine_memory_pedestrian* agent, int numObstLines, int beginLine, float radius, glm::vec2 &result)
{
	const int count = agent->count;
	float distance = 0.0f;

	for (int i = beginLine; i < count; ++i) {
		glm::vec2 lines_direction_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, i));
		glm::vec2 lines_point_i = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, i), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, i));

		// Result does not satisfy constraint of line i.
		if (det(lines_direction_i, lines_point_i - result) > distance) {
			//number of elements within projLines arrays 						   
			int countlp3 = 0;
			for (int j = numObstLines; j < i; ++j) {

				glm::vec2 lines_direction_j = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, j), get_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, j));
				glm::vec2 lines_point_j = glm::vec2(get_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, j), get_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, j));


				glm::vec2 line_direction;
				glm::vec2 line_point;

				float determinant = det(lines_direction_i, lines_direction_j);

				if (fabsf(determinant) <= RVO_EPSILON) {
					// Line i and line j are parallel. 
					if (glm::dot(lines_direction_i, lines_direction_j) > 0.0f) {
						// Line i and line j point in the same direction. 
						continue;
					}
					else {
						// Line i and line j point in opposite direction. 
						line_point = 0.5f * (lines_point_i + lines_point_j);
					}
				}
				else {
					line_point = lines_point_i + (det(lines_direction_j, lines_point_i - lines_point_j) / determinant) * lines_direction_i;
				}
				
				line_direction = normalize(lines_direction_j - lines_direction_i);
				//Agent memory version
				set_pedestrian_agent_array_value<float>(agent->projLine_direction_x, countlp3, line_direction.x);
				set_pedestrian_agent_array_value<float>(agent->projLine_direction_y, countlp3, line_direction.y);
				set_pedestrian_agent_array_value<float>(agent->projLine_point_x, countlp3, line_point.x);
				set_pedestrian_agent_array_value<float>(agent->projLine_point_y, countlp3, line_point.y);

				countlp3++;
			}

			const glm::vec2 tempResult = result;

			//If there is no solution found to the linear program minimization
			if (linearProgram2(agent, countlp3, radius, glm::vec2(-lines_direction_i.y, lines_direction_i.x), true, result, true) < countlp3) {
				// This should in principle not happen.  The result is by definition already in the feasible region of this linear program. If it fails, it is due to small floating point error, and the current result kept.
				//printf("Shouldn't happen\n");
				result = tempResult;
			}

			distance = det(lines_direction_i, lines_point_i - result);

		}
	}
}







////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Main function functions/////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// output properties message
// @param agent Pointer to an agent structure of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
// @param properties_message_messages Pointer to output message list of type xmachine_message_properties_message_list. Must be passed as an argument to the add_properties_message_message function ??.
__FLAME_GPU_FUNC__ int output_properties(xmachine_memory_pedestrian* agent, xmachine_message_properties_message_list* properties_messages) {
	add_properties_message_message(properties_messages, agent->x, agent->y, 0, agent->vx, agent->vy);

	return 0;
}

// Read in partitioned messaged and calculate the corresponding orcaLines which are saved to agent memory
// @param agent Pointer to an agent structure of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
// @param properties_message_messages  properties_message_messages Pointer to input message list of type xmachine_message__list. Must be passed as an argument to the get_first_properties_message_message and get_next_properties_message_message functions.* @param partition_matrix Pointer to the partition matrix of type xmachine_message_properties_message_PBM. Used within the get_first__message and get_next__message functions for spatially partitioned message access.
__FLAME_GPU_FUNC__ int make_orcaLines(xmachine_memory_pedestrian* agent, xmachine_message_properties_message_list* properties_messages, xmachine_message_properties_message_PBM* partition_matrix) {
	glm::vec2 position_ = glm::vec2(agent->x, agent->y);
	glm::vec2 velocity_ = glm::vec2(agent->vx, agent->vy);
	const glm::vec2 prefVelocity_ = glm::vec2(agent->desvx, agent->desvy);
	glm::vec2 newVelocity_ = glm::vec2(0.0, 0.0);
	const float radius_ = RADIUS;

	const float invTimeHorizon = 1.0f / timeHorizon_;

	int count = 0;

	// Create agent ORCA lines. 
	xmachine_message_properties_message* current_message = get_first_properties_message_message(properties_messages, partition_matrix, agent->x, agent->y, 0);//Partitioned version
																																							  //xmachine_message_properties_message* current_message = get_first_properties_message_message(properties_messages); //BFversion
	while (current_message)
	{
		const glm::vec2 other_position_ = glm::vec2(current_message->x, current_message->y);
		const glm::vec2 other_velocity_ = glm::vec2(current_message->vx, current_message->vy);
		const float other_radius_ = RADIUS;


		const glm::vec2 relativePosition = other_position_ - position_;
		const glm::vec2 relativeVelocity = velocity_ - other_velocity_;
		const float distSq = absSq(relativePosition);
		const float combinedRadius = radius_ + other_radius_;
		const float combinedRadiusSq = sqr(combinedRadius);
	
		if ((glm::length(relativePosition) < LOOKRADIUS) && (position_ != other_position_)) // message outside the radius of interest and not the same message as agent
		{
			glm::vec2 line_direction = glm::vec2(0.0, 0.0);
			glm::vec2 line_point = glm::vec2(0.0, 0.0);
			glm::vec2 u = glm::vec2(0.0, 0.0);

			if (distSq >= combinedRadiusSq)
			{
				// No collision.
				const glm::vec2 w = relativeVelocity - invTimeHorizon * relativePosition;
				// Vector from cutoff center to relative velocity.
				const float wLengthSq = absSq(w);

				const float dotProduct1 = glm::dot(w, relativePosition);

				if (dotProduct1 < 0.0f && sqr(dotProduct1) > combinedRadiusSq * wLengthSq) {
					// Project on cut-off circle.
					const float wLength = std::sqrt(wLengthSq);
					const glm::vec2 unitW = w / wLength;

					line_direction = glm::vec2(unitW.y, -unitW.x);
					u = (combinedRadius * invTimeHorizon - wLength) * unitW;
				}
				else
				{
					// Project on legs.
					const float leg = std::sqrt(distSq - combinedRadiusSq);

					if (det(relativePosition, w) > 0.0f) {
						// Project on left leg.
						line_direction = glm::vec2(relativePosition.x * leg - relativePosition.y * combinedRadius, relativePosition.x * combinedRadius + relativePosition.y * leg) / distSq;
					}
					else {
						// Project on right leg.
						line_direction = -glm::vec2(relativePosition.x * leg + relativePosition.y * combinedRadius, -relativePosition.x * combinedRadius + relativePosition.y * leg) / distSq;
					}

					const float dotProduct2 = glm::dot(relativeVelocity, line_direction);

					u = dotProduct2 * line_direction - relativeVelocity;
				}
			}
			else
			{
				// Collision. Project on cut-off circle of time timeStep.
				const float invTimeStep = 1.0f / timeStep_;

				// Vector from cutoff center to relative velocity.
				const glm::vec2 w = relativeVelocity - invTimeStep * relativePosition;

				const float wLength = glm::length(w);
				const glm::vec2 unitW = w / wLength;

				line_direction = glm::vec2(unitW.y, -unitW.x);
				u = (combinedRadius * invTimeStep - wLength) * unitW;

				//If deep within eachother collision, report it. Cannot be exactly equal to radius due to floating point errors calling false-positive collisions
				if (distSq < 0.49990*0.49990)
				{
				//	printf("Collision at x:%f y:%f!\n", agent->x, agent->y);
				}
			}

			line_point = velocity_ + (0.5f * u);

			//Set the values into the array
			set_pedestrian_agent_array_value<float>(agent->orcaLine_direction_x, count, line_direction.x);
			set_pedestrian_agent_array_value<float>(agent->orcaLine_direction_y, count, line_direction.y);
			set_pedestrian_agent_array_value<float>(agent->orcaLine_point_x, count, line_point.x);
			set_pedestrian_agent_array_value<float>(agent->orcaLine_point_y, count, line_point.y);


			//Move onto next message
			count++;
			if (count >= AGENTNO) {
				count = AGENTNO - 1;
				printf("warning: More agents than allowed for in max size at x:%f y:%f\n", agent->x, agent->y);
			}

		}
		current_message = get_next_properties_message_message(current_message, properties_messages, partition_matrix);

	}
	agent->count = count;
	//initialize agent velocity as desired velocity initialy. Will change through lp2
	agent->newvx = agent->desvx;
	agent->newvy = agent->desvy;

	return 0;
}


// run linearProgram2 function
// @param agent Pointer to an agent structure of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
__FLAME_GPU_FUNC__ int do_linear_program_2(xmachine_memory_pedestrian* agent) {
	//useful values
	const glm::vec2 prefVelocity_ = glm::vec2(agent->desvx, agent->desvy);
	int count = agent->count;
	//Each new iteration take newVelocity_ set to 0
	glm::vec2 newVelocity_ = glm::vec2(0.0, 0.0);
	
	//Make sure lineFail is set to default success
	agent->lineFail = -1;

	//Run LP2
	int lineFail = linearProgram2(agent, count, maxSpeed_, prefVelocity_, false, newVelocity_);

	//If not successful, flag agent
	if (lineFail < count)
	{
		agent->lineFail = lineFail;
	}

	//write new velocity into agent memory
	agent->newvx = newVelocity_.x;
	agent->newvy = newVelocity_.y;

	return 0;
}


// run linearProgram3 function if lineFail from agent memory is not equal to -1 having ran linearProgram 2
//@param agent Pointer to an agent structure of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
__FLAME_GPU_FUNC__ int do_linear_program_3(xmachine_memory_pedestrian* agent) {
	/*//Read in current new velocity from agent memory
	glm::vec2 newVelocity_ = glm::vec2(agent->newvx, agent->newvy);

	//Run lp3
	linearProgram3(agent, 0, agent->lineFail, maxSpeed_, newVelocity_);

	//write new velocity into agent memory
	agent->newvx = newVelocity_.x;
	agent->newvy = newVelocity_.y;*/

	return 0;
}


// Read agent memory variables and update position, speed and desired speed accordingly
//@param agent Pointer to an agent structure of type xmachine_memory_pedestrian. This represents a single agent instance and can be modified directly.
//@param rand48 Pointer to the seed list of type RNG_rand48. Must be passed as an argument to the rand48 function for generating random numbers on the GPU.
__FLAME_GPU_FUNC__ int move_agents(xmachine_memory_pedestrian* agent, RNG_rand48* rand48) {

	//useful values
	glm::vec2 position_ = glm::vec2(agent->x, agent->y);
	//glm::vec2 velocity_ = glm::vec2(agent->vx, agent->vy);
	glm::vec2 prefVelocity_ = glm::vec2(agent->desvx, agent->desvy);
	//Read in current new velocity from agent memory
	glm::vec2 newVelocity_ = glm::vec2(agent->newvx, agent->newvy);
	
	if (threadIdx.x + (blockDim.x * blockIdx.x) < 20) {
		printf("agent %i, \t vx = %f \t vy = %f\n", threadIdx.x + (blockDim.x * blockIdx.x), newVelocity_.x, newVelocity_.y);
	}

	//Update the holder values
	position_ += newVelocity_*timeStep_;

	//Change desired velocity if outside soft boundary
	prefVelocity_ = isAtEdge(position_, prefVelocity_, rand48);

	//update
	agent->x = position_.x;
	agent->y = position_.y;
	agent->vx = newVelocity_.x;
	agent->vy = newVelocity_.y;

	agent->desvx = prefVelocity_.x;
	agent->desvy = prefVelocity_.y;

	return 0;
}




//Generate key value pairs function kernel for sorting by density
__global__ void gen_keyval_pairs_density(unsigned int* keys, unsigned int* values, xmachine_memory_pedestrian_list* agents) {
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	
	//Number of agents
	const int n = xmachine_memory_pedestrian_MAX;

	if (index < n)
	{
		//set key
		keys[index] = (unsigned)agents->count[index];
		//set value
		values[index] = index;
	}
}

//Generate key value pairs function kernel for sorting by partitioning bin
__global__ void gen_keyval_pairs_spatial(unsigned int* keys, unsigned int* values, xmachine_memory_pedestrian_list* agents) {
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;

	//Number of agents
	const int n = xmachine_memory_pedestrian_MAX;

	if (index < n)
	{
		//set value
		values[index] = index;

		//calculate agent bin

		//Agent position
		float xpos = agents->x[index];
		float ypos = agents->y[index];
		//Bin size as defined in xmlmodelfile
		int binRadius = 5;
		int envLength = 600;
		//Lower bound of environment
		int envStart = -150;

		//bin no in x direction
		int binNoX = floor((xpos + envStart) / binRadius);
		//bin no in y direction
		int binNoY = floor((ypos + envStart) / binRadius);
		//The actual bin number
		int binNo = binNoX + binNoY*(envLength / binRadius);

		//set key
		keys[index] = binNo;
	}
}

//Use a defined __global__ function to define the key-value pairs used within an agent sort function
__FLAME_GPU_STEP_FUNC__ void sort_func() {
	//Do step only every k iterations
	const int sortCycle = 900;
	static int iterations = 0;
	iterations++;

	if (iterations % sortCycle == 0)
	{
		//Pointer function taking arguments specified within sort_pedestrians_default
		void(*func_ptr)(unsigned int*, unsigned int*, xmachine_memory_pedestrian_list*) = &gen_keyval_pairs_spatial;

		//sort the key value pairs initialized within argument function
		sort_pedestrians_default(func_ptr);
		cudaDeviceSynchronize();

	}
}




#endif //_FLAMEGPU_FUNCTIONS
