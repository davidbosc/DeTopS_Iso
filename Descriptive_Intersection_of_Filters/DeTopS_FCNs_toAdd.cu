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
using metric_t = T(*) (T*, T*, T*, unsigned, unsigned, unsigned, unsigned, float, unsigned, unsigned);

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
	unsigned VECTOR_SIZE,
	unsigned VECTORS_PER_SUBSET
) {

	float descriptiveIntersectionCardinality = 0.0f; 
	float unionCardinality = 0.0f;
	
	//starting at index_B * size_A + index_A of the array containing all descriptive intersections (in row major layout), 
	//get all the vectors that aren't minFloat
	unsigned desc_intersections_index = index_A * 2 + index_B; //0,1,2,3
	unsigned inputSetVectorOffset = desc_intersections_index * VECTOR_SIZE * VECTORS_PER_SUBSET; //0,6,12,18
	unsigned inputAVectorOffset = index_A * VECTOR_SIZE * VECTORS_PER_SUBSET;
	unsigned inputBVectorOffset = index_B * VECTOR_SIZE * VECTORS_PER_SUBSET;

	for (int i = 0; i < size_A; i += VECTOR_SIZE) { 
		if (desc_intersection[inputSetVectorOffset + i] != minFloat) {
			descriptiveIntersectionCardinality += 1.0f;
		}
	}

	//get the number of vectors in the description of A
	for (int i = 0; i < size_A; i += VECTOR_SIZE) {
		if (A_desc[inputAVectorOffset + i] != minFloat) {
			unionCardinality += 1.0f;
		}
	}

	//get the number of vectors in the description of B, not in A
	for (int i = 0; i < size_B; i += VECTOR_SIZE) {
		//for every vector in B's description that's not the initilized minFloat
		if (B_desc[inputBVectorOffset + i] != minFloat) {
			bool isUnique = true;
			for (int j = 0; isUnique && j < size_A; j += VECTOR_SIZE) {
				bool termIsRepeated = true;
				//Check it against every term of the vector in the description of A
				for (int k = 0; termIsRepeated && k < VECTOR_SIZE; k++) {
					if (B_desc[inputBVectorOffset + i + k] != A_desc[inputAVectorOffset + j + k]) {
						termIsRepeated = false;
					}
				}
				if (termIsRepeated) {
					isUnique = false;
				}
			}
			if (isUnique) {
				unionCardinality += 1.0f;
			}
		}	
	}
	//return descriptiveIntersectionCardinality;
	//return unionCardinality;
	return (1.0f - descriptiveIntersectionCardinality / unionCardinality);
}

template <typename T>
__device__ metric_t<T> p_desc_jaccard_dist = desc_jaccard_dist<T>;

template <typename T>
__global__ void runMetricOnGPU(
	metric_t<T> metric, 
	T* d_A, 
	T* d_B, 
	T* d_inter, 
	T* result, 
	float minFloat, 
	unsigned VECTOR_SIZE, 
	unsigned VECTORS_PER_SUBSET
	//add size_A, size_B
) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	//TODO: incorporate shared memory (MATRIX MUL SHOULD GIVE A GOOD CLUE FOR THIS)

	//replaced when size_A, size_B implemented
	int size = VECTOR_SIZE * VECTORS_PER_SUBSET;
	result[row * VECTOR_SIZE + col] = (*metric)(d_A, d_B, d_inter, row, col, size, size, minFloat, VECTOR_SIZE, VECTORS_PER_SUBSET);
}

//FIX Right now it's treating the families as 2 big sets
//Kernel needs to be launch with blocks per number of vectors
__global__ void setDifferenceOfFamilies(
	float* d_A,
	float* d_B,
	float* d_output,
	unsigned size,
	unsigned VECTOR_SIZE,
	float minFloat
) {
	int vectorStartingIndex = (blockIdx.x * blockDim.x + threadIdx.x) * VECTOR_SIZE;

	if (vectorStartingIndex < size) {
		float* termToCheck = new float[VECTOR_SIZE];
		for (int i = 0; i < VECTOR_SIZE; i++) {
			termToCheck[i] = d_A[vectorStartingIndex + i];
		}
		bool termIsInB = false;
		//check our term against each term in d_B
		for (int i = 0; i < size; i++) {
			bool termIsPartiallyInB = true;
			//check each vector term by term
			for (int j = 0; termIsPartiallyInB && j < VECTOR_SIZE; j++) {
				if (termToCheck[j] != d_B[i + j]) {
					termIsPartiallyInB = false;
				}
			}
			if (termIsPartiallyInB) {
				termIsInB = true;
			}
		}
		//if term is not found, write to output based on term index
		if (termIsInB == false) {
			for (int i = 0; i < VECTOR_SIZE; i++) {
				d_output[vectorStartingIndex + i] = termToCheck[i];
			}
		}
		else {
			for (int i = 0; i < VECTOR_SIZE; i++) {
				d_output[vectorStartingIndex + i] = minFloat;
			}
		}
	}
}

template <typename T>
void dIteratedPseudometric(T* family_A, T* family_B, T* desc_intersection, unsigned size) {

	metric_t<T> h_desc_jaccard_dist;

	T *d_A;
	T *d_B;
	T *d_inter;
	T* family_A_less_B = new T[size/2];
	T* family_B_less_A = new T[size/2];
	T* d_output;

	cudaMalloc((void**)&d_A, sizeof(T) * size/2);
	cudaMalloc((void**)&d_B, sizeof(T) * size/2);
	cudaMalloc((void**)&d_output, sizeof(T) * size/2);
	cudaMemcpy(d_A, family_A, sizeof(T) * size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * size/2, cudaMemcpyHostToDevice);

	//TODO: fix kernel parameters
	setDifferenceOfFamilies << <1, size/2 >> > (
		d_A,
		d_B,
		d_output,
		size,
		VECTOR_SIZE,
		minFloat
	);

	cudaMemcpy(family_A_less_B, d_output, sizeof(T) * size/2, cudaMemcpyDeviceToHost);

	for (unsigned i = 0; i < size/2; i++) {
		cout << "family_A_less_B[" << i << "]=" << family_A_less_B[i] << endl;
	}

	//TODO: Fix kernel parameters
	setDifferenceOfFamilies << <1, size/2 >> > (
		d_B,
		d_A,
		d_output,
		size,
		VECTOR_SIZE,
		minFloat
	);

	cudaMemcpy(family_B_less_A, d_output, sizeof(T) * size/2, cudaMemcpyDeviceToHost);

	for (unsigned i = 0; i < size/2; i++) {
		cout << "family_B_less_A[" << i << "]=" << family_B_less_A[i] << endl;
	}


	//TODO: desc_intersection will be calculated in a kernel in the future rather than a param
	
	for (unsigned i = 0; i < size; i++) {
		cout << "desc_int[" << i << "]=" << desc_intersection[i] << endl;
	}

	cudaMalloc((void**)&d_A, sizeof(T) * size/2);
	cudaMalloc((void**)&d_B, sizeof(T) * size/2);
	cudaMalloc((void**)&d_inter, sizeof(T) * size);
	cudaMemcpy(d_A, family_A, sizeof(T) * size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inter, desc_intersection, sizeof(T) * size, cudaMemcpyHostToDevice);

	T* d_result;
	T* h_result = new T[(F_SUBSET_COUNT * F_SUBSET_COUNT / 4)];
	//this is assuming our families have the same amount of sets (or one has less)
	cudaMalloc(&d_result, sizeof(T)*(F_SUBSET_COUNT* F_SUBSET_COUNT / 4));

	// Copy device function pointer to host side
	cudaMemcpyFromSymbol(&h_desc_jaccard_dist, p_desc_jaccard_dist<T>, sizeof(metric_t<T>));

	//TODO: fix kernel parameters (F_SUBSET_COUNT * F_SUBSET_COUNT / 4)
	//NOT THE CASE ANYMORE
	//CARDINALITY OF A, CARD OF B

	dim3 jaccardGrid(1, 1);
	dim3 jaccardBlock(2, 2);

	runMetricOnGPU<T> <<<jaccardGrid, jaccardBlock>>> (h_desc_jaccard_dist, d_A, d_B, d_inter, d_result, minFloat, VECTOR_SIZE, VECTORS_PER_SUBSET);

	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(T) * (F_SUBSET_COUNT * F_SUBSET_COUNT / 4), cudaMemcpyDeviceToHost);
	T result = 0;
	for (unsigned i = 0; i < (F_SUBSET_COUNT * F_SUBSET_COUNT / 4); i++) {
		cout << "h_result[" << i << "]=" << h_result[i] << endl;
		result += h_result[i];
	}
	std::cout << "d Iterated Pseudometric Distance: " << result << " (Should be 2.4)" << std::endl;
}

void initNegative(float* data, unsigned size) {

	for (unsigned i = 0; i < size; ++i) {
		data[i] = minFloat;
	}
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

	//might need to change this to size_A and size_B
	unsigned size = VECTORS_PER_SUBSET * VECTOR_SIZE * (F_SUBSET_COUNT / 2);

	float* family_A = new float[size];
	//first set
	family_A[0] =	2;
	family_A[1] =	1;
	family_A[2] =	3;
	family_A[3] =	3;
	family_A[4] =	3;
	family_A[5] =	2;
	//second set
	family_A[6] =	1;
	family_A[7] =	0;
	family_A[8] =	3;
	family_A[9] =	2;
	family_A[10] =	2;
	family_A[11] =	1;

	float* family_B = new float[size];
	//first set
	family_B[0] =	2;
	family_B[1] =	1;
	family_B[2] =	3;
	family_B[3] =	3;
	family_B[4] =	3;
	family_B[5] =	0;
	//second set
	family_B[6] =	2;		
	family_B[7] =	1;
	family_B[8] =	3;
	family_B[9] =	3;
	family_B[10] =	4;
	family_B[11] =	3;

	//setup array for desc intersection kernel(?) <-- this should be done in template
	//Hard coding intersections for now

	//if we use the two different sizes, they need to be added here, not multiplied
	unsigned intersectionSize = size * 2;

	float *desc_inter_h = new float[intersectionSize];
	initNegative(desc_inter_h, intersectionSize);

	desc_inter_h[0] =	2;
	desc_inter_h[1] =	1;
	desc_inter_h[2] =	3;
	desc_inter_h[3] =	3;
	desc_inter_h[6] =	2;
	desc_inter_h[7] =	1;
	desc_inter_h[8] =	3;
	desc_inter_h[9] =	2;
	desc_inter_h[12] =	2;
	desc_inter_h[13] =	1;
	desc_inter_h[18] =	2;
	desc_inter_h[19] =	1;
	desc_inter_h[20] =	3;
	desc_inter_h[21] =	2;
	
	//might need to pass size_A and size_B
	dIteratedPseudometric<float>(family_A, family_B, desc_inter_h, intersectionSize);

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