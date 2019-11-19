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
	//unsigned sizeOfA,
	//unsigned sizeOfB,
	unsigned VECTOR_SIZE, 
	unsigned VECTORS_PER_SUBSET
) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	//TODO: incorporate shared memory (MATRIX MUL SHOULD GIVE A GOOD CLUE FOR THIS)

	//TODO: This will need to be changed based on the number of sets passed to this metric 
	//can be thought of like a 2x1 matrix, for example
	unsigned size = VECTOR_SIZE * VECTORS_PER_SUBSET;
	result[row * VECTOR_SIZE + col] = (*metric)(
		d_A, 
		d_B, 
		d_inter, 
		row, 
		col, 
		size, 
		size, 
		minFloat, 
		VECTOR_SIZE, 
		VECTORS_PER_SUBSET
	);
}

//TODO: Utilize a lambda or template to call any metric with this function
template <typename T>
void dIteratedPseudometric(T* family_A, T* family_B, T* desc_intersection, unsigned size) {

	metric_t<T> d_metric;

	T *d_A;
	T *d_B;
	T* d_inter;
	T* d_family_A_less_B;
	T* d_family_B_less_A;
	T* d_output;

	T* h_family_A_less_B = new T[size / 2];
	T* h_family_B_less_A = new T[size / 2];
	unsigned sizeOfFamilyALessB;
	unsigned sizeOfFamilyBLessA;

	cudaMalloc((void**)&d_A, sizeof(T) * size/2);
	cudaMalloc((void**)&d_B, sizeof(T) * size/2);
	cudaMalloc((void**)&d_output, sizeof(T) * size/2);
	cudaMemcpy(d_A, family_A, sizeof(T) * size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * size/2, cudaMemcpyHostToDevice);

	unsigned subsetSize = VECTOR_SIZE * VECTORS_PER_SUBSET;

	//TODO: fix kernel parameters
	setDifferenceOfFamilies << <1, size/2 >> > (
		d_A,
		d_B,
		d_output,
		F_SUBSET_COUNT, 
		VECTORS_PER_SUBSET, 
		VECTOR_SIZE,
		minFloat
	);

	cudaMemcpy(h_family_A_less_B, d_output, sizeof(T) * size/2, cudaMemcpyDeviceToHost);

	sizeOfFamilyALessB = getFamilyCardinality(h_family_A_less_B, size);

	cout << "|family_A_less_B| = " << sizeOfFamilyALessB << endl;
	for (unsigned i = 0; i < size/2; i++) {
		cout << "family_A_less_B[" << i << "]=" << h_family_A_less_B[i] << endl;
	}

	//TODO: Fix kernel parameters
	setDifferenceOfFamilies << <1, size/2 >> > (
		d_B,
		d_A,
		d_output,
		F_SUBSET_COUNT, 
		VECTORS_PER_SUBSET, 
		VECTOR_SIZE,
		minFloat
	);

	cudaMemcpy(h_family_B_less_A, d_output, sizeof(T) * size/2, cudaMemcpyDeviceToHost);

	sizeOfFamilyBLessA = getFamilyCardinality(h_family_B_less_A, size);

	cout << "\n|family_B_less_A| = " << sizeOfFamilyBLessA << endl;
	for (unsigned i = 0; i < size/2; i++) {
		cout << "family_B_less_A[" << i << "]=" << h_family_B_less_A[i] << endl;
	}

	//TODO: desc_intersection will be calculated in a kernel in the future rather than a param
	
	for (unsigned i = 0; i < size; i++) {
		cout << "desc_int[" << i << "]=" << desc_intersection[i] << endl;
	}

	//allocate to device
	cudaMalloc((void**)&d_A, sizeof(T) * size / 2);
	cudaMalloc((void**)&d_B, sizeof(T) * size / 2);
	cudaMalloc((void**)&d_family_A_less_B, sizeof(T) * size / 2);
	cudaMalloc((void**)&d_family_B_less_A, sizeof(T) * size / 2);
	cudaMalloc((void**)&d_inter, sizeof(T) * size);

	//copy to device
	cudaMemcpy(d_A, family_A, sizeof(T) * size / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * size / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_A_less_B, h_family_A_less_B, sizeof(T) * size / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_B_less_A, h_family_B_less_A, sizeof(T) * size / 2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_inter, desc_intersection, sizeof(T) * size, cudaMemcpyHostToDevice);

	T* d_result;
	T* h_result = new T[(F_SUBSET_COUNT * F_SUBSET_COUNT / 4)];
	//this is assuming our families have the same amount of sets (or one has less)
	cudaMalloc(&d_result, sizeof(T)*(F_SUBSET_COUNT* F_SUBSET_COUNT / 4));

	// Copy device function pointer to host side
	cudaMemcpyFromSymbol(&d_metric, p_desc_jaccard_dist<T>, sizeof(metric_t<T>));
	//TODO: The below code is causing issues when passing a metric_t<T> as an arguement; try a lambda?
	//cudaMemcpy(d_metric, metric, sizeof(metric_t<T>), cudaMemcpyHostToDevice);

	//TODO: fix kernel parameters (F_SUBSET_COUNT * F_SUBSET_COUNT / 4)
	//NOT THE CASE ANYMORE
	//CARDINALITY OF A, CARD OF B
	//each thread to take care of one set within family

	dim3 jaccardGrid(1, 1);
	dim3 jaccardBlock(2, 2);

	runMetricOnGPU<T> << <jaccardGrid, jaccardBlock >> > (
		d_metric,
		d_A,
		d_family_B_less_A,
		d_inter,
		d_result,
		minFloat,
		VECTOR_SIZE,
		VECTORS_PER_SUBSET);

	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(T) * (F_SUBSET_COUNT * F_SUBSET_COUNT / 4), cudaMemcpyDeviceToHost);
	T result1 = 0;
	for (unsigned i = 0; i < (F_SUBSET_COUNT * F_SUBSET_COUNT / 4); i++) {
		cout << "h_result1[" << i << "]=" << h_result[i] << endl;
		result1 += h_result[i];
	}
	result1 /= ((F_SUBSET_COUNT / 2) * sizeOfFamilyBLessA);

	runMetricOnGPU<T> << <jaccardGrid, jaccardBlock >> > (
		d_metric,
		d_family_A_less_B,
		d_B,
		d_inter,
		d_result,
		minFloat,
		VECTOR_SIZE,
		VECTORS_PER_SUBSET);

	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(T) * (F_SUBSET_COUNT * F_SUBSET_COUNT / 4), cudaMemcpyDeviceToHost);
	T result2 = 0;
	for (unsigned i = 0; i < (F_SUBSET_COUNT * F_SUBSET_COUNT / 4); i++) {
		cout << "h_result2[" << i << "]=" << h_result[i] << endl;
		result2 += h_result[i];
	}
	result2 /= ((F_SUBSET_COUNT / 2) * sizeOfFamilyALessB);

	T result = result1 + result2;

	cout << "d Iterated Pseudometric Distance: " << result << " (Should be 1.2)" << endl;
}

void initNegative(float* data, unsigned size) {

	for (unsigned i = 0; i < size; ++i) {
		data[i] = minFloat;
	}
}

//TODO: this needs a rework like mad (I don't even want to use the word efficiency here...)
//Kernel needs to be launch with treads per number of families
__global__ void setDifferenceOfFamilies(
	float* d_A,
	float* d_B,
	float* d_output,
	unsigned F_SUBSET_COUNT,
	unsigned VECTORS_PER_SUBSET,
	unsigned VECTOR_SIZE,
	float minFloat
) {
	unsigned numberOfSubsets = (F_SUBSET_COUNT / 2);			//2
	unsigned subsetSize = VECTOR_SIZE * VECTORS_PER_SUBSET;		//6
	unsigned familySize = subsetSize * numberOfSubsets;			//12
	unsigned subsetStartingIndex = (blockIdx.x * blockDim.x + threadIdx.x) * subsetSize;	//0, 6

	if (subsetStartingIndex < familySize) {

		bool isSubsetOfAinB = false;
		//each subset of B
		for (int i = 0; i < numberOfSubsets; i++) { //2
			unsigned matchedVectorCount = 0;
			//each vector of subset of B
			for (int j = 0; j < VECTORS_PER_SUBSET; j++) { //3
				bool vectorsMatchWithinSubset = false;
				//check if terms in vectors equal, otherwise move onto the next vector
				for(int m  = 0; m < VECTORS_PER_SUBSET; m++) {
					bool vectorsMatch = true;
					//check each term in vector from subset of B with the terms in our subset to check
					for (int k = 0; vectorsMatch && k < VECTOR_SIZE; k++) { //2
						if (d_B[(i * subsetSize) + (j * VECTOR_SIZE) + k] != d_A[subsetStartingIndex + (m * VECTOR_SIZE) + k]) {
							vectorsMatch = false;
						}
					}
					if (vectorsMatch) {
						vectorsMatchWithinSubset = true;
					}
				}
				//if the vectors have match, then increase the count of matching vectors within subsets
				if (vectorsMatchWithinSubset) {
					matchedVectorCount++;
				}
			}
			//if the matched vector count isn't the same as the numbers of vectors in the subset,
			//then we move onto the next subset
			if (matchedVectorCount == VECTORS_PER_SUBSET) {
				isSubsetOfAinB = true;
			}
		}

		for (int i = 0; i < subsetSize; i++) {
			if (isSubsetOfAinB) {
				d_output[subsetStartingIndex + i] = minFloat;
			}
			else {
				d_output[subsetStartingIndex + i] = d_A[subsetStartingIndex + i];
			}
		}
	}
}

unsigned getFamilyCardinality(float* input, unsigned size) {
	unsigned setSize = F_SUBSET_COUNT / 2;
	unsigned index = 0;
	//for each subset in input family of sets
	while(index < setSize) {
		//if we encounter a subset with a vector that starts with minFloat
		//swap the subsets with the row-major index of our final subset based on our running setSize
		//decrease the set size if this is the case (we have a 'nulled' out subset from set difference on the families)
		if (input[index * VECTOR_SIZE * VECTORS_PER_SUBSET] == minFloat) {
			for (int i = 0; i < VECTOR_SIZE * VECTORS_PER_SUBSET; i++) {
				float temp = input[(index * VECTOR_SIZE * VECTORS_PER_SUBSET) + i];
				input[(index * VECTOR_SIZE * VECTORS_PER_SUBSET) + i] 
					= input[(setSize * VECTOR_SIZE * VECTORS_PER_SUBSET) - ((VECTOR_SIZE * VECTORS_PER_SUBSET) - i)];
				input[(setSize * VECTOR_SIZE * VECTORS_PER_SUBSET) - ((VECTOR_SIZE * VECTORS_PER_SUBSET) - i)] = temp;
			}
			setSize--;
		} else {
			index++;
		}
	}
	return setSize;
}

//needs to be reworked into a descriptive intersection kernel
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
	
	//Total size of each family
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
	family_B[0] = 2;
	family_B[1] = 1;
	family_B[2] = 3;
	family_B[3] = 3;
	family_B[4] = 3;
	family_B[5] = 0;
	//second set
	family_B[6] = 2;
	family_B[7] = 1;
	family_B[8] = 3;
	family_B[9] = 3;
	family_B[10] = 4;
	family_B[11] = 3;

	float* family_C = new float[size];
	//first set
	family_C[0] = 2;
	family_C[1] = 1;
	family_C[2] = 3;
	family_C[3] = 3;
	family_C[4] = 3;
	family_C[5] = 2;
	//second set
	family_C[6] = 4;
	family_C[7] = 3;
	family_C[8] = 2;
	family_C[9] = 1;
	family_C[10] = 3;
	family_C[11] = 3;

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

	cout << "Testing setDifferenceOfFamilies and getFamilyCardinality methods..." << endl;

	float* d_A;
	float* d_B;
	float* family_A_less_A = new float[size];
	float* family_A_less_B = new float[size];
	float* family_B_less_C = new float[size];
	float* d_output;

	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_A, sizeof(float) * size));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_B, sizeof(float) * size));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_output, sizeof(float) * size));
	CUDA_CHECK_RETURN(cudaMemcpy(d_A, family_A, sizeof(float) * size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_B, family_A, sizeof(float) * size, cudaMemcpyHostToDevice));

	unsigned subsetSize = VECTOR_SIZE * VECTORS_PER_SUBSET;

	//TODO: fix kernel parameters
	setDifferenceOfFamilies << <1, 2 >> > (
		d_A,
		d_B,
		d_output,
		F_SUBSET_COUNT,
		VECTORS_PER_SUBSET,
		VECTOR_SIZE,
		minFloat
	);

	CUDA_CHECK_RETURN(cudaMemcpy(family_A_less_A, d_output, sizeof(float) * size, cudaMemcpyDeviceToHost));

	cout << "\n|family_A_less_A| = " << getFamilyCardinality(family_A_less_A, size) << " (Should be 0)\n" << endl;

	for (unsigned i = 0; i < size; i++) {
		cout << "family_A_less_A[" << i << "]=" << family_A_less_A[i] << endl;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(d_B, family_B, sizeof(float) * size, cudaMemcpyHostToDevice));

	setDifferenceOfFamilies << <1, 2 >> > (
		d_A,
		d_B,
		d_output,
		F_SUBSET_COUNT,
		VECTORS_PER_SUBSET,
		VECTOR_SIZE,
		minFloat
	);

	CUDA_CHECK_RETURN(cudaMemcpy(family_A_less_B, d_output, sizeof(float) * size, cudaMemcpyDeviceToHost));

	cout << "\n|family_A_less_B| = " << getFamilyCardinality(family_A_less_B, size) << " (Should be 2)\n" << endl;

	for (unsigned i = 0; i < size; i++) {
		cout << "family_A_less_B[" << i << "]=" << family_A_less_B[i] << endl;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(d_A, family_B, sizeof(float) * size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_B, family_C, sizeof(float) * size, cudaMemcpyHostToDevice));
	
	setDifferenceOfFamilies << <1, 2 >> > (
		d_A,
		d_B,
		d_output,
		F_SUBSET_COUNT,
		VECTORS_PER_SUBSET,
		VECTOR_SIZE,
		minFloat
	);

	CUDA_CHECK_RETURN(cudaMemcpy(family_B_less_C, d_output, sizeof(float) * size, cudaMemcpyDeviceToHost));

	cout << "\n|family_B_less_C| = " << getFamilyCardinality(family_B_less_C, size) << " (Should be 1)\n" << endl;

	for (unsigned i = 0; i < size; i++) {
		cout << "family_B_less_C[" << i << "]=" << family_B_less_C[i] << endl;
	}

	cout << "\nTesting Complete!\n" << endl;

	//dIteratedPseudometric<float>(family_A, family_B, desc_inter_h, p_desc_jaccard_dist<float>, intersectionSize);
	dIteratedPseudometric<float>(family_A, family_B, desc_inter_h, intersectionSize);

	CUDA_CHECK_RETURN(cudaFree((void*)d_A));
	CUDA_CHECK_RETURN(cudaFree((void*)d_B));
	CUDA_CHECK_RETURN(cudaFree((void*)d_output));
	CUDA_CHECK_RETURN(cudaDeviceReset());

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