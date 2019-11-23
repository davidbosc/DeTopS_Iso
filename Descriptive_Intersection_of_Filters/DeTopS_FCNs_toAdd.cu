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
unsigned SUBSETS_PER_FAMILY = F_SUBSET_COUNT / 2;

using namespace std;

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
	
	//starting at index_B * size_A + index_A of the array containing all descriptive intersections
	//(in row major layout), get all the vectors that aren't minFloat
	unsigned desc_intersections_index = index_A * 2 + index_B; //0,1,2,3
	unsigned inputSetVectorOffset = desc_intersections_index * VECTOR_SIZE * VECTORS_PER_SUBSET; //0,6,12,18
	unsigned inputAVectorOffset = index_A * VECTOR_SIZE * VECTORS_PER_SUBSET;
	unsigned inputBVectorOffset = index_B * VECTOR_SIZE * VECTORS_PER_SUBSET;
	float maxUnionSize = VECTORS_PER_SUBSET * 2;

	for (int i = 0; i < size_A; i += VECTOR_SIZE) { 
		if (desc_intersection[inputSetVectorOffset + i] != minFloat) {
			descriptiveIntersectionCardinality += 1.0f;
		}
	}

	unionCardinality = maxUnionSize - descriptiveIntersectionCardinality;
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
void dIteratedPseudometric(T* family_A, T* family_B) {

	metric_t<T> d_metric;

	T* d_A;
	T* d_B;
	T* d_inter;
	T* d_family_A_less_B;
	T* d_family_B_less_A;
	T* d_output;
	unsigned sizeOfFamilyALessB;
	unsigned sizeOfFamilyBLessA;

	unsigned numberOfVectorsPerFamily = SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET;
	unsigned intersectionSize = pow(SUBSETS_PER_FAMILY, 2) * VECTORS_PER_SUBSET * VECTOR_SIZE;
	unsigned sizeOfSets = VECTORS_PER_SUBSET * VECTOR_SIZE * (F_SUBSET_COUNT / 2);
	unsigned subsetSize = VECTOR_SIZE * VECTORS_PER_SUBSET;
	T* h_family_A_less_B = new T[sizeOfSets];
	T* h_family_B_less_A = new T[sizeOfSets];

	cudaMalloc((void**)&d_A, sizeof(T) * sizeOfSets);
	cudaMalloc((void**)&d_B, sizeof(T) * sizeOfSets);
	cudaMalloc((void**)&d_output, sizeof(T) * sizeOfSets);
	cudaMemcpy(d_A, family_A, sizeof(T) * sizeOfSets, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * sizeOfSets, cudaMemcpyHostToDevice);

	//TODO: fix kernel parameters
	setDifferenceOfFamilies << <1, sizeOfSets >> > (
		d_A,
		d_B,
		d_output,
		F_SUBSET_COUNT, 
		VECTORS_PER_SUBSET, 
		VECTOR_SIZE,
		minFloat
	);

	cudaMemcpy(h_family_A_less_B, d_output, sizeof(T) * sizeOfSets, cudaMemcpyDeviceToHost);

	sizeOfFamilyALessB = getFamilyCardinality(h_family_A_less_B, sizeOfSets);

	//TODO: Fix kernel parameters
	setDifferenceOfFamilies << <1, sizeOfSets >> > (
		d_B,
		d_A,
		d_output,
		F_SUBSET_COUNT, 
		VECTORS_PER_SUBSET, 
		VECTOR_SIZE,
		minFloat
	);

	cudaMemcpy(h_family_B_less_A, d_output, sizeof(T) * sizeOfSets, cudaMemcpyDeviceToHost);

	sizeOfFamilyBLessA = getFamilyCardinality(h_family_B_less_A, sizeOfSets);

	//allocate to device
	cudaMalloc((void**)&d_A, sizeof(T) * sizeOfSets);
	cudaMalloc((void**)&d_B, sizeof(T) * sizeOfSets);
	cudaMalloc((void**)&d_family_A_less_B, sizeof(T) * sizeOfSets);
	cudaMalloc((void**)&d_family_B_less_A, sizeof(T) * sizeOfSets);
	cudaMalloc((void**)&d_inter, sizeof(T) * intersectionSize);

	//copy to device
	cudaMemcpy(d_A, family_A, sizeof(T) * sizeOfSets, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * sizeOfSets, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_A_less_B, h_family_A_less_B, sizeof(T) * sizeOfSets, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_B_less_A, h_family_B_less_A, sizeof(T) * sizeOfSets, cudaMemcpyHostToDevice);
	

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

	descriptiveIntersectionGPU << <1, numberOfVectorsPerFamily, intersectionSize * sizeof(float) >> > (
		d_A,
		d_family_B_less_A,
		d_inter,
		intersectionSize,
		minFloat,
		SUBSETS_PER_FAMILY,
		VECTORS_PER_SUBSET,
		VECTOR_SIZE
	);

	runMetricOnGPU<T> << <jaccardGrid, jaccardBlock >> > (
		d_metric,
		d_A,
		d_family_B_less_A,
		d_inter,
		d_result,
		minFloat,
		VECTOR_SIZE,
		VECTORS_PER_SUBSET
	);

	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(T) * (F_SUBSET_COUNT * F_SUBSET_COUNT / 4), cudaMemcpyDeviceToHost);
	T result1 = 0;
	for (unsigned i = 0; i < (F_SUBSET_COUNT * F_SUBSET_COUNT / 4); i++) {
		//cout << "h_result1[" << i << "]=" << h_result[i] << endl;
		result1 += h_result[i];
	}
	//TODO: GET CARDINALITY OF UNION OF BOTH FAMILIES
	result1 /= ((F_SUBSET_COUNT / 2) * 4);

	descriptiveIntersectionGPU << <1, numberOfVectorsPerFamily, intersectionSize * sizeof(float) >> > (
		d_family_A_less_B,
		d_B,
		d_inter,
		intersectionSize,
		minFloat,
		SUBSETS_PER_FAMILY,
		VECTORS_PER_SUBSET,
		VECTOR_SIZE
	);

	runMetricOnGPU<T> << <jaccardGrid, jaccardBlock >> > (
		d_metric,
		d_family_A_less_B,
		d_B,
		d_inter,
		d_result,
		minFloat,
		VECTOR_SIZE,
		VECTORS_PER_SUBSET
	);

	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(T) * (F_SUBSET_COUNT * F_SUBSET_COUNT / 4), cudaMemcpyDeviceToHost);
	T result2 = 0;
	for (unsigned i = 0; i < (F_SUBSET_COUNT * F_SUBSET_COUNT / 4); i++) {
		//cout << "h_result2[" << i << "]=" << h_result[i] << endl;
		result2 += h_result[i];
	}
	result2 /= ((F_SUBSET_COUNT / 2) * 4);

	T result = result1 + result2;

	cout << "d Iterated Pseudometric Distance: " << result << endl;

	CUDA_CHECK_RETURN(cudaFree((void*)d_A));
	CUDA_CHECK_RETURN(cudaFree((void*)d_B));
	CUDA_CHECK_RETURN(cudaFree((void*)d_inter));
	CUDA_CHECK_RETURN(cudaFree((void*)d_family_A_less_B));
	CUDA_CHECK_RETURN(cudaFree((void*)d_family_B_less_A));
	CUDA_CHECK_RETURN(cudaFree((void*)d_output));
	CUDA_CHECK_RETURN(cudaFree((void*)d_result));
	CUDA_CHECK_RETURN(cudaDeviceReset());
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
	unsigned numberOfSubsets = (F_SUBSET_COUNT / 2);
	unsigned subsetSize = VECTOR_SIZE * VECTORS_PER_SUBSET;
	unsigned familySize = subsetSize * numberOfSubsets;
	unsigned subsetStartingIndex = (blockIdx.x * blockDim.x + threadIdx.x) * subsetSize;

	if (subsetStartingIndex < familySize) {

		bool isSubsetOfAinB = false;
		//each subset of B
		for (int i = 0; i < numberOfSubsets; i++) {
			unsigned matchedVectorCount = 0;
			//each vector of subset of B
			for (int j = 0; j < VECTORS_PER_SUBSET; j++) {
				bool vectorsMatchWithinSubset = false;
				//check if terms in vectors equal, otherwise move onto the next vector
				for(int m  = 0; m < VECTORS_PER_SUBSET; m++) {
					bool vectorsMatch = true;
					//check each term in vector from subset of B with the terms in our subset to check
					for (int k = 0; vectorsMatch && k < VECTOR_SIZE; k++) {
						if (d_B[(i * subsetSize) + (j * VECTOR_SIZE) + k] != 
							d_A[subsetStartingIndex + (m * VECTOR_SIZE) + k]) {
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
		//decrease the set size if this is the case (we have a 'nulled'
		//out subset from set difference on the families)
		if (input[index * VECTOR_SIZE * VECTORS_PER_SUBSET] == minFloat) {
			for (int i = 0; i < VECTOR_SIZE * VECTORS_PER_SUBSET; i++) {
				float temp = input[(index * VECTOR_SIZE * VECTORS_PER_SUBSET) + i];
				input[(index * VECTOR_SIZE * VECTORS_PER_SUBSET) + i] =
					input[(setSize * VECTOR_SIZE * VECTORS_PER_SUBSET) - 
						((VECTOR_SIZE * VECTORS_PER_SUBSET) - i)];
				input[(setSize * VECTOR_SIZE * VECTORS_PER_SUBSET) - 
					((VECTOR_SIZE * VECTORS_PER_SUBSET) - i)] = temp;
			}
			setSize--;
		} else {
			index++;
		}
	}
	return setSize;
}

__global__ void descriptiveIntersectionGPU(
	float* d_A, 
	float* d_B, 
	float* d_output,
	unsigned size,
	float minFloat,
	unsigned SUBSETS_PER_FAMILY,
	unsigned VECTORS_PER_SUBSET,
	unsigned VECTOR_SIZE
) {
	
	extern __shared__ float shared[];

	float* ds_A = &shared[0];
	float* ds_B = &shared[size];

	unsigned vectorInFamily = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned setSubscript = floorf((float)vectorInFamily / VECTORS_PER_SUBSET);

	//Load into Shared Memory
	for (unsigned i = 0; i < VECTOR_SIZE; i++) {
		ds_A[vectorInFamily * VECTOR_SIZE + i] = d_A[vectorInFamily * VECTOR_SIZE + i];
		ds_B[vectorInFamily * VECTOR_SIZE + i] = d_B[vectorInFamily * VECTOR_SIZE + i];
	}
	__syncthreads();

	//Perform Intersections
	//for each subset in B...
	if (vectorInFamily < SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET) {
		for (unsigned i = 0; i < SUBSETS_PER_FAMILY; i++) {
			//for each vector in subset of B...
			bool vectorIsInSubset = false;
			for (unsigned j = 0; !vectorIsInSubset && j < VECTORS_PER_SUBSET; j++) {
				bool vectorsMatch = true;
				for (unsigned k = 0; vectorsMatch && k < VECTOR_SIZE; k++) {
					if (ds_B[i * VECTORS_PER_SUBSET * VECTOR_SIZE + j * VECTOR_SIZE + k] !=
						ds_A[vectorInFamily * VECTOR_SIZE + k] ) {
							vectorsMatch = false;
					}
				}
				//if the vector is found within subset, don't check the rest of the subset
				if (vectorsMatch) {
					vectorIsInSubset = true;
				}
			}
			for (unsigned j = 0; j < VECTOR_SIZE; j++) {
				if (vectorIsInSubset) {
					d_output[(i * VECTOR_SIZE * VECTORS_PER_SUBSET) + (vectorInFamily * VECTOR_SIZE) +
						(setSubscript * (SUBSETS_PER_FAMILY - 1) * VECTOR_SIZE * VECTORS_PER_SUBSET) +j] =													
							ds_A[vectorInFamily * VECTOR_SIZE + j];
				}
				else {
					d_output[(i * VECTOR_SIZE * VECTORS_PER_SUBSET) + (vectorInFamily * VECTOR_SIZE) +
						(setSubscript * (SUBSETS_PER_FAMILY - 1) * VECTOR_SIZE * VECTORS_PER_SUBSET) + j] =													
							minFloat;
				}
			}
		}
	}
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
	family_A[0] = 2;
	family_A[1] = 1;
	family_A[2] = 3;
	family_A[3] = 3;
	family_A[4] = 3;
	family_A[5] = 2;
	//second set
	family_A[6] = 1;
	family_A[7] = 0;
	family_A[8] = 3;
	family_A[9] = 2;
	family_A[10] = 2;
	family_A[11] = 1;

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
	family_B[9] = 2;
	family_B[10] = 4;
	family_B[11] = 3;

	float* family_C = new float[size];
	//first set
	family_C[0] = 4;
	family_C[1] = 3;
	family_C[2] = 3;
	family_C[3] = 0;
	family_C[4] = 3;
	family_C[5] = 3;
	//second set
	family_C[6] = 3;
	family_C[7] = 1;
	family_C[8] = 3;
	family_C[9] = 2;
	family_C[10] = 4;
	family_C[11] = 4;

	//setup array for desc intersection kernel(?) <-- this should be done in template

	//dIteratedPseudometric<float>(family_A, family_B, p_desc_jaccard_dist<float>);
	dIteratedPseudometric<float>(family_A, family_B);
	dIteratedPseudometric<float>(family_A, family_C);
	dIteratedPseudometric<float>(family_B, family_C);
	
	delete[] family_A;
	delete[] family_B;
	delete[] family_C;

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