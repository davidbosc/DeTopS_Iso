#include <iostream>   //Standard input output
#include <fstream>    //Read input and write output files
#include <vector>     //Provides access to vector object, for flexibly sized arrays
#include <math.h>     //Provides math functions. pow, log, ceil, floor
#include <stdlib.h>   //Provides size_t datatype
#include <string>     //Provides string object
#include <sstream>    //Provides methods for working with strings
#include <limits>     //Used to derive minFloat
#include <ctime>      //Used for CPU timing code
#include <pthread.h>  //Used for parallel CPU threads
#include <mutex>      //Used for synchronization of parallel cpu code

static void CheckCudaErrorAux(const char*, unsigned, const char*, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

unsigned SETS = 10;    //How many subsets to load in (for testing)
#define STREAMS 500    //How many streams to launch intersectKernels in
typedef unsigned long long bitString;

bool emptySetCheck = false;
//Most negative float value, used as a null in arrays
const float minFloat = (-1) * (std::numeric_limits<float>::max());
//Maximum depth of intersections (max number of sets that can take place in an intersection)
unsigned maxDepth = 0;
unsigned F_SUBSET_COUNT = 4;  //Number of input sets
unsigned VECTORS_PER_SUBSET = 3;  //Width of each fundamental subset
unsigned VECTOR_SIZE = 2;         //Features per feature vector, defines shared memory tile length
unsigned WIDTH = VECTORS_PER_SUBSET * (1+F_SUBSET_COUNT);               //Total width of the output set
unsigned CORES = 1;           //How many cores to run cpu on
unsigned TILE_WIDTH;          //Tile width of intersectKernel

using namespace std;

//instead of pointers to pointers, try pointers to arrays?
template<typename T>
using metric_t = T(*) (T*, T*, T*, unsigned, unsigned, unsigned, unsigned, float, unsigned);

template<typename T>
__device__ T desc_jaccard_dist(
	T* A_desc, 
	T* B_desc, 
	T* desc_intersection, 
	unsigned index_A, 
	unsigned index_B, 
	unsigned size_A, 
	unsigned size_B,
	float minFloat,
	unsigned VECTOR_SIZE
) {

	float descriptiveIntersectionCardinality = 0.0f; 
	
	//starting at index_B * size_A + index_A of the array containing all descriptive intersections (in row major layout), get all the vectors that aren't minFloat
	int desc_intersections_index = index_B * size_A + index_A;
	for (int i = 0; i < size_A; i += VECTOR_SIZE) { 	//I think size_A should be the subscript of the family within As
		if (desc_intersection[desc_intersections_index + i] != minFloat) {
			descriptiveIntersectionCardinality += 1.0f;
		}
	}
	float unionCardinality = 0.0f;

	//get the number of vectors in the description of A
	for (int i = 0; i < size_A; i += VECTOR_SIZE) {
		if (A_desc[i] != minFloat) {
			unionCardinality += 1.0f;
		}
	}

	//get the number of vectors in the description of B, not in A
	for (int i = 0; i < size_B; i += VECTOR_SIZE) {
		//for every vector in B's description that's not the initilized negative number
		if (B_desc[i] != minFloat) {
			bool isUnique = true;
			for (int j = 0; j < size_A && isUnique; j += VECTOR_SIZE) {
				//Check it against every term of the vector in the description of A
				for (int k = 0; k < VECTOR_SIZE; k++) {
					if (B_desc[i+k] == A_desc[j+k]) {
						isUnique = false;
					}
				}
			}
			if (isUnique) {
				unionCardinality += 1.0f;
			}
		}	
	}
	return desc_intersection[desc_intersections_index];
	//return (1.0f - descriptiveIntersectionCardinality / unionCardinality);
}

template <typename T>
__device__ metric_t<T> p_desc_jaccard_dist = desc_jaccard_dist<T>;

template <typename T>
__global__ void runMetricOnGPU(metric_t<T> metric, T* d_A, T* d_B, T* d_inter, T* result, float minFloat, unsigned VECTOR_SIZE, unsigned VECTORS_PER_SUBSET)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int size = VECTOR_SIZE * VECTORS_PER_SUBSET;
	*result = (*metric)(d_A, d_B, d_inter, i, i, size, size, minFloat, VECTOR_SIZE);
}

template <typename T>
void dIteratedPseudometric(T* A_desc, T* B_desc, T* desc_intersection, unsigned size) {

	metric_t<T> h_desc_jaccard_dist;

	T *d_A;
	T *d_B;
	T *d_inter;
	
	//desc_intersection will be calculated in a kernel in the future rather than a param
	
	for (unsigned i = 0; i < size; i++) {
		cout << "desc_int[" << i << "]=" << desc_intersection[i] << endl;
	}

	cudaMalloc((void**)&d_A, sizeof(T) * size);
	cudaMalloc((void**)&d_B, sizeof(T) * size);
	cudaMalloc((void**)&d_inter, sizeof(T) * size);
	cudaMemcpy(d_A, &A_desc, sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, &B_desc, sizeof(T) * size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inter, &desc_intersection, sizeof(T) * size, cudaMemcpyHostToDevice);

	T result;
	T* d_result, * h_result;
	cudaMalloc(&d_result, sizeof(T));
	h_result = &result;

	// Copy device function pointer to host side
	cudaMemcpyFromSymbol(&h_desc_jaccard_dist, p_desc_jaccard_dist<T>, sizeof(metric_t<T>));

	runMetricOnGPU<T> << <1, 1 >> > (h_desc_jaccard_dist, d_A, d_B, d_inter, d_result, minFloat, VECTOR_SIZE, VECTORS_PER_SUBSET);
	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);
	std::cout << "d Iterated Pseudometric Distance: " << result << " (Should be 0.5)" << std::endl;
}

void initNegative(float* data, unsigned size) {

	for (unsigned i = 0; i < size; ++i) {
		data[i] = minFloat;
	}
}

void setDifference(float* a, float* b, float* out) {
	//TODO
}

void getSetDescription(float* input, float* output, unsigned size) {
	
	//initialize the description as the minFloat
	initNegative(output, size);

	//The first vector in input is trivially added to output
	for (unsigned i = 0; input[i] != minFloat && i < VECTOR_SIZE; i++) {
		output[i] = input[i];
	}

	bool inputRemaining = true;
	unsigned outputIndex = VECTOR_SIZE;
	//For each vector in input after the first, check 
	for (unsigned i = 1; inputRemaining && i < size / VECTOR_SIZE; i++) {
		bool isUnique = true;
		if (input[i*VECTOR_SIZE] == minFloat) {
			inputRemaining = false;
		}
		if (inputRemaining) {
			//check output array for repeated vector
			for (unsigned j = 0; j < outputIndex; j += VECTOR_SIZE) {
				bool isVectorPartiallyIdentical = true;
				for (unsigned k = 0; k < VECTOR_SIZE; k++) {
					if (input[i*VECTOR_SIZE + k] != output[j + k]) {
						isVectorPartiallyIdentical = false;
					}
				}
				if (isVectorPartiallyIdentical) {
					isUnique = false;
				}
			}
			//if it is unique, add it to output and increment index
			if (isUnique) {
				for (unsigned m = 0; m < VECTOR_SIZE; m++) {
					output[outputIndex + m] = input[i*VECTOR_SIZE + m];
				}
				outputIndex += outputIndex;
			}
		}
	}

	////print input for debugging
	//for (int k = 0; k < size; k++) {
	//	cout << "in[" << k << "]= " << input[k] << endl;
	//}

	////print out the description for debugging
	//for (int k = 0; k < size; k++) {
	//	cout << "desc[" << k << "]= " << output[k] << endl;
	//}
}

__global__ void descriptiveIntersectionGPU(float* desc_A_d, float* desc_B_d) {
	//TODO
}

int main(void) {
	unsigned device = 0;
	cout << "\033[2J\033[1;1H";

	CUDA_CHECK_RETURN(cudaSetDevice(device));

	//Get device properties
	cudaDeviceProp deviceProp;
	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProp, device));

	float a_0[6] = {
		2,1,
		3,3,
		3,2
	};
	float a_1[6] = {
		1,0,
		3,2,
		2,1
	};
	float b_0[6] = {
		2,1,
		3,3,
		3,0
	};
	float b_1[6] = {
		2,1,
		3,3,
		4,3
	};

	unsigned size = VECTORS_PER_SUBSET * VECTOR_SIZE;

	float* desc_a0 = new float[size];
	float* desc_b0 = new float[size];
	float* desc_a1 = new float[size];
	float* desc_b1 = new float[size];
	
	//setup array for desc intersection kernel(?) <-- this should be done in template
	//Hard coding intersections for now
	float *desc_inter_h = new float[size];
	initNegative(desc_inter_h, size);

	desc_inter_h[0] = 2;
	desc_inter_h[1] = 1;
	desc_inter_h[2] = 3;
	desc_inter_h[3] = 3;

	for (unsigned i = 0; i < size; i++) {
		cout << "dih[" << i << "]=" << desc_inter_h[i] << endl;
	}

	getSetDescription(a_0, desc_a0, size);
	getSetDescription(b_0, desc_b0, size);
	getSetDescription(a_1, desc_a1, size);
	getSetDescription(b_1, desc_b1, size);

	dIteratedPseudometric<float>(desc_a0, desc_b0, desc_inter_h, size);
	
	//cout << "done.\nFreeing memory ...";
	//delete[] a_0;
	//delete[] a_1;
	//delete[] b_0;
	//delete[] b_1;
	//delete[] desc_a0;
	//delete[] desc_b0;
	//delete[] desc_inter_h;
	//cout << "done.\nExiting program\n";

	return 0;
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed
 */
static void CheckCudaErrorAux(const char* file, unsigned line,
	const char* statement, cudaError_t err) {
	if (err == cudaSuccess)
		return;
	std::cerr << statement << " returned " << cudaGetErrorString(err) << "(" << err << ") at "
		<< file << ":" << line << std::endl;
	exit(1);
}
