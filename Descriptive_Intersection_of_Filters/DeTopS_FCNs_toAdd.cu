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
using metric_t = T(*) (T*, T*, unsigned, unsigned, unsigned, float);

template<typename T>
using pseudometric_t = T(*) (T*, T*, T*, unsigned, unsigned, unsigned, float, unsigned, unsigned, metric_t<T>);

template<typename T>
__host__ __device__ T vectorHammingDistance(
	T* d_A,
	T* d_B,
	unsigned index_A,
	unsigned index_B,
	unsigned VECTOR_SIZE,
	float minFloat
) {
	unsigned distance = 0;
	for (unsigned k = 0; k < VECTOR_SIZE; k++) {
		if (d_A[index_A + k] != minFloat &&
			d_B[index_B + k] != minFloat) {
			if (d_A[index_A + k] != d_B[index_B + k]) {
				distance++;
			}
		}
	}
	return distance;
}

template<typename T>
__host__ __device__ T descJaccardDistance(
	T* A_desc, 
	T* B_desc, 
	T* desc_intersection, 
	unsigned index_A, 
	unsigned index_B, 
	unsigned size, 
	float minFloat,
	unsigned VECTOR_SIZE,
	unsigned VECTORS_PER_SUBSET,
	metric_t<T> embeddedMetric
) {

	float descriptiveIntersectionCardinality = 0.0f; 
	float unionCardinality = 0.0f;
	
	//starting at index_B * size_A + index_A of the array containing all descriptive intersections
	//(in row major layout), get all the vectors that aren't minFloat
	unsigned desc_intersections_index = index_A * 2 + index_B;
	unsigned inputSetVectorOffset = desc_intersections_index * VECTOR_SIZE * VECTORS_PER_SUBSET;
	
	float maxUnionSize = VECTORS_PER_SUBSET * 2;

	for (int i = 0; i < size; i += VECTOR_SIZE) { 
		if (desc_intersection[inputSetVectorOffset + i] != minFloat) {
			descriptiveIntersectionCardinality += 1.0f;
		}
	}

	unionCardinality = maxUnionSize - descriptiveIntersectionCardinality;
	return 1.0f - (descriptiveIntersectionCardinality / unionCardinality);
}

template<typename T>
__host__ __device__ T descHausdorffDistance(
	T* A_desc,
	T* B_desc,
	T* desc_intersection,	//unused
	unsigned index_A,
	unsigned index_B,
	unsigned size,			//unused
	float minFloat,
	unsigned VECTOR_SIZE,
	unsigned VECTORS_PER_SUBSET,
	metric_t<T> embeddedMetric
) {
	unsigned* distanceBetweenEachVector = new unsigned[VECTORS_PER_SUBSET * VECTORS_PER_SUBSET];
	unsigned* minOfCols = new unsigned[VECTORS_PER_SUBSET];
	unsigned* minOfRows = new unsigned[VECTORS_PER_SUBSET];

	unsigned subsetAIndex = index_A * VECTOR_SIZE * VECTORS_PER_SUBSET;
	unsigned subsetBIndex = index_B * VECTOR_SIZE * VECTORS_PER_SUBSET;

	//Build a matrix of distances
	//for each a in A_y
	for (unsigned i = 0; i < VECTORS_PER_SUBSET; i++) {
		//take the distance with each b in B_x
		for (unsigned j = 0; j < VECTORS_PER_SUBSET; j++) {
			//embedded distance function (hard coded for now)
			unsigned distance =	embeddedMetric(
				A_desc,
				B_desc,
				subsetAIndex + j * VECTOR_SIZE,
				subsetBIndex + i * VECTOR_SIZE,
				VECTOR_SIZE,
				minFloat
			);
			distanceBetweenEachVector[i * VECTORS_PER_SUBSET + j] = distance;
		}
	}

	//Find the min of each row and column
	//for each col
	for (unsigned i = 0; i < VECTORS_PER_SUBSET; i++) {
		//go through each row and find the min
		unsigned minOfCol = distanceBetweenEachVector[i];
		unsigned minOfRow = distanceBetweenEachVector[i * VECTORS_PER_SUBSET];
		for (unsigned j = 1; j < VECTORS_PER_SUBSET; j++) {
			minOfCol = minOfCol < distanceBetweenEachVector[j * VECTORS_PER_SUBSET + i] ?
				 minOfCol : distanceBetweenEachVector[j * VECTORS_PER_SUBSET + i];
			minOfCols[i] = minOfCol;

			minOfRow = minOfRow < distanceBetweenEachVector[i * VECTORS_PER_SUBSET + j] ?
				 minOfRow : distanceBetweenEachVector[i * VECTORS_PER_SUBSET + j];
			minOfRows[i] = minOfRow;
		}
	}

	//Find the max
	unsigned maxOfMinCols = minOfCols[0];
	unsigned maxOfMinRows = minOfRows[0];
	for (int i = 1; i < VECTORS_PER_SUBSET; i++) {
		maxOfMinCols = maxOfMinCols > minOfCols[i] ?
			maxOfMinCols : minOfCols[i];
		maxOfMinRows = maxOfMinRows > minOfRows[i] ?
			maxOfMinRows : minOfRows[i];
	}

	return max(maxOfMinCols, maxOfMinRows);
}

template <typename T>
__device__ pseudometric_t<T> p_descJaccardDistance = descJaccardDistance<T>;

template <typename T>
__device__ pseudometric_t<T> p_descHausdorffDistance = descHausdorffDistance<T>;

template <typename T>
__device__ metric_t<T> p_no_embeddedMetric;

template <typename T>
__device__ metric_t<T> p_vectorHammingDistance = vectorHammingDistance<T>;

template <typename T>
__global__ void runMetricOnGPU(
	pseudometric_t<T> pseudometric,
	T* d_A, 
	T* d_B,
	T* d_inter, 
	T* result, 
	float minFloat, 
	unsigned VECTOR_SIZE, 
	unsigned VECTORS_PER_SUBSET,
	unsigned SUBSETS_PER_FAMILY,
	unsigned INTERSECTION_SM_OFFSET,
	metric_t<T> embeddedMetric
) {
	unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned size = VECTOR_SIZE * VECTORS_PER_SUBSET;
	unsigned familySize = size * SUBSETS_PER_FAMILY;
	extern __shared__ T shared[];

	T* ds_A = &shared[0];
	T* ds_B = &shared[familySize];
	T* ds_inter = &shared[INTERSECTION_SM_OFFSET];

	//each thread loads a subset based on row and column
	if (row < SUBSETS_PER_FAMILY && col < SUBSETS_PER_FAMILY) {
		for (unsigned i = 0; i < size; i++) {
			ds_A[row * size + i] = d_A[row * size + i];
			ds_B[col * size + i] = d_B[col * size + i];
			ds_inter[row * familySize + col * size + i] = d_inter[row * familySize + col * size + i];
		}

		__syncthreads();

		result[row * SUBSETS_PER_FAMILY + col] = (*pseudometric)(
			ds_A,
			ds_B,
			ds_inter,
			row,
			col,
			size,
			minFloat,
			VECTOR_SIZE,
			VECTORS_PER_SUBSET,
			embeddedMetric
		);
	}
}

//Version of d-iterated pseudometric that uses GPU for metric caluculations and descriptive intersections
template <typename T>
T dIteratedPseudometricGPU(
	T* family_A,
	T* family_B, 
	pseudometric_t<T>* pseudometric,
	metric_t<T>* embeddedMetric = &p_no_embeddedMetric<T>
) {
	//Device Variables
	pseudometric_t<T> d_pseudometric;
	metric_t<T> d_metric;
	T* d_A;
	T* d_B;
	T* d_inter;
	T* d_family_A_less_B;
	T* d_family_B_less_A;

	//Host Variables
	unsigned sizeOfFamilyAUnionFamilyB;
	bool familiesAreDisjoint = true;
	unsigned numberOfVectorsPerFamily = SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET;
	unsigned intersectionSize = pow(SUBSETS_PER_FAMILY, 2) * VECTORS_PER_SUBSET * VECTOR_SIZE;
	unsigned indiciesPerSet = VECTORS_PER_SUBSET * VECTOR_SIZE * (F_SUBSET_COUNT / 2);
	unsigned subsetSize = VECTOR_SIZE * VECTORS_PER_SUBSET;
	unsigned metricSharedMemorySize = 2 * indiciesPerSet + intersectionSize;
	T* h_inter = new T[intersectionSize];
	T result = 0;
	T* h_family_A_less_B = setDifferenceOfFamilies(family_A, family_B);
	T* h_family_B_less_A = setDifferenceOfFamilies(family_B, family_A);
	unsigned sizeOfFamilyALessB = getFamilyCardinality(h_family_A_less_B, indiciesPerSet);
	unsigned sizeOfFamilyBLessA = getFamilyCardinality(h_family_B_less_A, indiciesPerSet);

	if (sizeOfFamilyALessB == SUBSETS_PER_FAMILY && sizeOfFamilyBLessA == SUBSETS_PER_FAMILY) {
		//If the families A and B are disjoint, then the cardinality of their union 
		//is the sum of their cardinalities
		sizeOfFamilyAUnionFamilyB = 2 * SUBSETS_PER_FAMILY;
	} else {
		//Otherwise, take the cardinality of B, and sum it with the cardinality of A less B 
		sizeOfFamilyAUnionFamilyB = SUBSETS_PER_FAMILY + sizeOfFamilyALessB;
		familiesAreDisjoint = false;
	}

	//allocate to device
	cudaMalloc((void**)&d_A, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_B, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_family_A_less_B, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_family_B_less_A, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_inter, sizeof(T) * intersectionSize);

	//copy to device
	cudaMemcpy(d_A, family_A, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_A_less_B, h_family_A_less_B, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_B_less_A, h_family_B_less_A, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);
	

	T* d_result;
	T* h_result = new T[(SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY)];
	cudaMalloc(&d_result, sizeof(T)*(SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY));

	// Copy device function pointer to host side
	cudaMemcpyFromSymbol(&d_pseudometric, *pseudometric, sizeof(pseudometric_t<T>));
	cudaMemcpyFromSymbol(&d_metric, *embeddedMetric, sizeof(metric_t<T>));

	//TODO: fix kernel parameters
	//NOT THE CASE ANYMORE
	//CARDINALITY OF A, CARD OF B
	//each thread to take care of one set within family

	dim3 metricGrid(1, 1);
	dim3 metricBlock(2, 2);

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

	runMetricOnGPU<T> << <metricGrid, metricBlock, metricSharedMemorySize * sizeof(T) >> > (
		d_pseudometric,
		d_A,
		d_family_B_less_A,
		d_inter,
		d_result,
		minFloat,
		VECTOR_SIZE,
		VECTORS_PER_SUBSET,
		SUBSETS_PER_FAMILY,
		intersectionSize,
		d_metric
	);

	cudaDeviceSynchronize();
	cudaMemcpy(h_result, d_result, sizeof(T) * (SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY), 
		cudaMemcpyDeviceToHost);
	T result1 = 0;
	for (unsigned i = 0; i < (SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY); i++) {
		result1 += h_result[i];
	}

	result1 /= (SUBSETS_PER_FAMILY * sizeOfFamilyAUnionFamilyB);

	if (!familiesAreDisjoint) {

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

		runMetricOnGPU<T> << <metricGrid, metricBlock, metricSharedMemorySize * sizeof(T) >> > (
			d_pseudometric,
			d_family_A_less_B,
			d_B,
			d_inter,
			d_result,
			minFloat,
			VECTOR_SIZE,
			VECTORS_PER_SUBSET,
			SUBSETS_PER_FAMILY,
			intersectionSize,
			d_metric
		);

		cudaDeviceSynchronize();
		cudaMemcpy(h_result, d_result, sizeof(T) * (SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY),
			cudaMemcpyDeviceToHost);
		T result2 = 0;
		for (unsigned i = 0; i < (SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY); i++) {
			result2 += h_result[i];
		}
		result2 /= (SUBSETS_PER_FAMILY * sizeOfFamilyAUnionFamilyB);

		result = result1 + result2;
	}
	else {
		result = result1 * 2;
	}

	CUDA_CHECK_RETURN(cudaFree((void*)d_A));
	CUDA_CHECK_RETURN(cudaFree((void*)d_B));
	CUDA_CHECK_RETURN(cudaFree((void*)d_inter));
	CUDA_CHECK_RETURN(cudaFree((void*)d_family_A_less_B));
	CUDA_CHECK_RETURN(cudaFree((void*)d_family_B_less_A));
	CUDA_CHECK_RETURN(cudaFree((void*)d_result));
	CUDA_CHECK_RETURN(cudaDeviceReset());
	return result;
}

template <typename T>
T* runMetricOnCPU(
	pseudometric_t<T> pseudometric,
	T* desc_A,
	T* desc_B,
	T* desc_inter,
	metric_t<T> embeddedMetric
) {
	unsigned size = VECTOR_SIZE * VECTORS_PER_SUBSET;
	T* result = new T[SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY];

	for (unsigned i = 0; i < SUBSETS_PER_FAMILY; i++) {
		for (unsigned j = 0; j < SUBSETS_PER_FAMILY; j++) {
			result[i * SUBSETS_PER_FAMILY + j] = (*pseudometric)(
				desc_A,
				desc_B,
				desc_inter,
				i,
				j,
				size,
				minFloat,
				VECTOR_SIZE,
				VECTORS_PER_SUBSET,
				embeddedMetric
			);
		}
	}

	return result;
}

//Version of d-iterated pseudometric that uses GPU for descriptive intersections
template <typename T>
T dIteratedPseudometric(
	T* family_A,
	T* family_B,
	pseudometric_t<T> pseudometric,
	metric_t<T> embeddedMetric = p_no_embeddedMetric<T>
) {
	//Device Variables
	T* d_A;
	T* d_B;
	T* d_inter;
	T* d_family_A_less_B;
	T* d_family_B_less_A;

	//Host Variables
	unsigned sizeOfFamilyAUnionFamilyB;
	bool familiesAreDisjoint = true;
	unsigned numberOfVectorsPerFamily = SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET;
	unsigned intersectionSize = pow(SUBSETS_PER_FAMILY, 2) * VECTORS_PER_SUBSET * VECTOR_SIZE;
	unsigned indiciesPerSet = VECTORS_PER_SUBSET * VECTOR_SIZE * (F_SUBSET_COUNT / 2);
	unsigned subsetSize = VECTOR_SIZE * VECTORS_PER_SUBSET;
	unsigned metricSharedMemorySize = 2 * indiciesPerSet + intersectionSize;
	T* h_inter = new T[intersectionSize];
	T result = 0;
	T* h_family_A_less_B = setDifferenceOfFamilies(family_A, family_B);
	T* h_family_B_less_A = setDifferenceOfFamilies(family_B, family_A);
	unsigned sizeOfFamilyALessB = getFamilyCardinality(h_family_A_less_B, indiciesPerSet);
	unsigned sizeOfFamilyBLessA = getFamilyCardinality(h_family_B_less_A, indiciesPerSet);

	if (sizeOfFamilyALessB == SUBSETS_PER_FAMILY && sizeOfFamilyBLessA == SUBSETS_PER_FAMILY) {
		//If the families A and B are disjoint, then the cardinality of their union 
		//is the sum of their cardinalities
		sizeOfFamilyAUnionFamilyB = 2 * SUBSETS_PER_FAMILY;
	}
	else {
		//Otherwise, take the cardinality of B, and sum it with the cardinality of A less B 
		sizeOfFamilyAUnionFamilyB = SUBSETS_PER_FAMILY + sizeOfFamilyALessB;
		familiesAreDisjoint = false;
	}

	//allocate to device
	cudaMalloc((void**)&d_A, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_B, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_family_A_less_B, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_family_B_less_A, sizeof(T) * indiciesPerSet);
	cudaMalloc((void**)&d_inter, sizeof(T) * intersectionSize);

	//copy to device
	cudaMemcpy(d_A, family_A, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, family_B, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_A_less_B, h_family_A_less_B, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);
	cudaMemcpy(d_family_B_less_A, h_family_B_less_A, sizeof(T) * indiciesPerSet, cudaMemcpyHostToDevice);

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

	cudaMemcpy(h_inter, d_inter, sizeof(T) * intersectionSize, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	
	T* metricValues1 = runMetricOnCPU<T> (
		pseudometric,
		family_A,
		h_family_B_less_A,
		h_inter,
		embeddedMetric
	);

	T result1 = 0;
	for (unsigned i = 0; i < (SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY); i++) {
		result1 += metricValues1[i];
	}

	result1 /= (SUBSETS_PER_FAMILY * sizeOfFamilyAUnionFamilyB);

	if (!familiesAreDisjoint) {

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
	
		cudaMemcpy(h_inter, d_inter, sizeof(T) * intersectionSize, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	
		T* metricValues2 = runMetricOnCPU<T>(
			pseudometric,
			h_family_A_less_B,
			family_B,
			h_inter,
			embeddedMetric
		);

		T result2 = 0;
		for (unsigned i = 0; i < (SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY); i++) {
			result2 += metricValues2[i];
		}
		result2 /= (SUBSETS_PER_FAMILY * sizeOfFamilyAUnionFamilyB);

		result = result1 + result2;
	}
	else {
		result = result1 * 2;
	}

	CUDA_CHECK_RETURN(cudaFree((void*)d_A));
	CUDA_CHECK_RETURN(cudaFree((void*)d_B));
	CUDA_CHECK_RETURN(cudaFree((void*)d_inter));
	CUDA_CHECK_RETURN(cudaFree((void*)d_family_A_less_B));
	CUDA_CHECK_RETURN(cudaFree((void*)d_family_B_less_A));
	CUDA_CHECK_RETURN(cudaDeviceReset());
	return result;
}

template <typename T>
T* setDifferenceOfFamilies(
	T* familyA,
	T* familyB
) {
	T* setDifferenceResult = new T[SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET * VECTOR_SIZE];
	unsigned* vectorsInCommonCounts = new unsigned[SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY];

	//initilize counts to 0.  These will be incremented as vectors that match are found
	for (unsigned i = 0; i < SUBSETS_PER_FAMILY * SUBSETS_PER_FAMILY; i++) {
		vectorsInCommonCounts[i] = 0;
	}

	//find vectors in common
	//for each vector in A
	for (unsigned i = 0; i < SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET; i++) {
		//get the subset index to index into vectorsInCommonCounts
		unsigned vectorInASubsetIndex = floor(i / VECTORS_PER_SUBSET);
		//for each vector in B
		for (unsigned j = 0; j < SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET; j++) {
			//get the subset index to index into vectorsInCommonCounts
			unsigned vectorInBSubsetIndex = floor(j / VECTORS_PER_SUBSET);
			bool vectorsMatch = true;
			for (unsigned k = 0; vectorsMatch && k < VECTOR_SIZE; k++) {
				if (familyA[i * VECTOR_SIZE + k] != familyB[j * VECTOR_SIZE + k]) {
					vectorsMatch = false;
				}
			}
			if (vectorsMatch) {
				vectorsInCommonCounts[vectorInASubsetIndex * SUBSETS_PER_FAMILY + vectorInBSubsetIndex]++;
			}
		}
	}

	//write to output array
	//for each subset in A
	for (unsigned i = 0; i < SUBSETS_PER_FAMILY; i++) {
		//if the vectorsInCommonCounts of any element in the ith row is VECTOR_SIZE...
		bool subsetsMatch = false;
		for (unsigned j = 0; !subsetsMatch && j < SUBSETS_PER_FAMILY; j++) {
			if (vectorsInCommonCounts[i * SUBSETS_PER_FAMILY + j] == VECTORS_PER_SUBSET) {
				subsetsMatch = true;
			}
		}
		//write each term of the subset as minFloat.  Otherwise, preserve the value
		for (unsigned j = 0; j < VECTORS_PER_SUBSET * VECTOR_SIZE; j++) {
			if (subsetsMatch) {
				setDifferenceResult[i * VECTORS_PER_SUBSET * VECTOR_SIZE + j] = minFloat;
			} else {
				setDifferenceResult[i * VECTORS_PER_SUBSET * VECTOR_SIZE + j] = 
					familyA[i * VECTORS_PER_SUBSET * VECTOR_SIZE + j];
			}
		}
	}

	return setDifferenceResult;
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
			for (unsigned i = 0; i < VECTOR_SIZE * VECTORS_PER_SUBSET; i++) {
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
	unsigned numberOfVectorsToLoad = SUBSETS_PER_FAMILY * VECTORS_PER_SUBSET;

	//Load d_A into Shared Memory: each thread will only need access to the one element in d_A
	if(vectorInFamily < numberOfVectorsToLoad) {
		for (unsigned i = 0; i < VECTOR_SIZE; i++) {
			ds_A[vectorInFamily * VECTOR_SIZE + i] = d_A[vectorInFamily * VECTOR_SIZE + i];
		}
	}

	//Load d_B into Shared Memory: if out number of vectors exceeds the number of blocks,
	//threads will have to load in mutliple vectors
	unsigned numberOfVectorsInBToLoad = numberOfVectorsToLoad;
	for (unsigned i = 0; numberOfVectorsInBToLoad > 0; i++) {
		// (vector number + offset of unloaded vectors in B) % number of blocks * size of blocks
		if((vectorInFamily + (i * blockDim.x)) % (gridDim.x * blockDim.x) 
			< numberOfVectorsToLoad) {
			for (unsigned j = 0; j < VECTOR_SIZE; j++) {
				ds_B[((vectorInFamily + (i * blockDim.x)) % (gridDim.x * blockDim.x)) * VECTOR_SIZE + j] =
					d_B[((vectorInFamily + (i * blockDim.x)) % (gridDim.x * blockDim.x)) * VECTOR_SIZE + j];
			}
		}
		numberOfVectorsInBToLoad -= blockDim.x;
	}

	__syncthreads();

	//TODO: Get descriptions before intersecting

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

	float* family = new float[size];
	//first set
	family[0] = 2;
	family[1] = 1;
	family[2] = 3;
	family[3] = 3;
	family[4] = 3;
	family[5] = 2;
	//second set
	family[6] = 3;
	family[7] = 1;
	family[8] = 3;
	family[9] = 2;
	family[10] = 4;
	family[11] = 4;

	/////////////////////////////////////////////////////////////
	cout << "d-iterated Pseudometric with Descriptive Jaccard Distance (CPU):" << endl;
	/////////////////////////////////////////////////////////////
	cout << "DELTA_d_J(A,A) = " << 
		dIteratedPseudometric<float>(family_A, family_A, &descJaccardDistance<float>) << endl;
	cout << "DELTA_d_J(A,B) = " << 
		dIteratedPseudometric<float>(family_A, family_B, &descJaccardDistance<float>) << endl;
	cout << "DELTA_d_J(A,C) = " << 
		dIteratedPseudometric<float>(family_A, family_C, &descJaccardDistance<float>) << endl;
	cout << "DELTA_d_J(B,C) = " << 
		dIteratedPseudometric<float>(family_B, family_C, &descJaccardDistance<float>) << endl;
	cout << endl;
	/////////////////////////////////////////////////////////////
	cout << "d-iterated Pseudometric with Descriptive Hausdorff (with Hamming) Distance (CPU):" << endl;
	/////////////////////////////////////////////////////////////
	cout << "DELTA_d_H(A,A) = " <<
		dIteratedPseudometric<float>(
			family_A, 
			family_A, 
			&descHausdorffDistance<float>, 
			&vectorHammingDistance<float>
		) << endl;
	cout << "DELTA_d_H(A,B) = " <<
		dIteratedPseudometric<float>(
			family_A, 
			family_B, 
			&descHausdorffDistance<float>,
			&vectorHammingDistance<float>
		) << endl;
	cout << "DELTA_d_H(A,C) = " <<
		dIteratedPseudometric<float>(
			family_A, 
			family_C, 
			&descHausdorffDistance<float>,
			&vectorHammingDistance<float>
		) << endl;
	cout << "DELTA_d_H(B,C) = " <<
		dIteratedPseudometric<float>(
			family_B, 
			family_C,
			&descHausdorffDistance<float>,
			&vectorHammingDistance<float>
		) << endl;
	cout << endl;
	/////////////////////////////////////////////////////////////
	cout << "d-iterated Pseudometric with Descriptive Jaccard Distance (GPU):" << endl;
	/////////////////////////////////////////////////////////////
	cout << "DELTA_d_J(A,A) = " << 
		dIteratedPseudometricGPU<float>(family_A, family_A, &p_descJaccardDistance<float>) << endl;
	cout << "DELTA_d_J(A,B) = " << 
		dIteratedPseudometricGPU<float>(family_A, family_B, &p_descJaccardDistance<float>) << endl;
	cout << "DELTA_d_J(A,C) = " << 
		dIteratedPseudometricGPU<float>(family_A, family_C, &p_descJaccardDistance<float>) << endl;
	cout << "DELTA_d_J(B,C) = " << 
		dIteratedPseudometricGPU<float>(family_B, family_C, &p_descJaccardDistance<float>) << endl;
	cout << endl;
	/////////////////////////////////////////////////////////////
	cout << "d-iterated Pseudometric with Descriptive Hausdorff (with Hamming) Distance (GPU):" << endl;
	/////////////////////////////////////////////////////////////
	cout << "DELTA_d_H(A,A) = " <<
		dIteratedPseudometricGPU<float>(
			family_A, 
			family_A, 
			&p_descHausdorffDistance<float>, 
			&p_vectorHammingDistance<float>
		) << endl;
	cout << "DELTA_d_H(A,B) = " <<
		dIteratedPseudometricGPU<float>(
			family_A, 
			family_B, 
			&p_descHausdorffDistance<float>,
			&p_vectorHammingDistance<float>
		) << endl;
	cout << "DELTA_d_H(A,C) = " <<
		dIteratedPseudometricGPU<float>(
			family_A, 
			family_C, 
			&p_descHausdorffDistance<float>,
			&p_vectorHammingDistance<float>
		) << endl;
	cout << "DELTA_d_H(B,C) = " <<
		dIteratedPseudometricGPU<float>(
			family_B, 
			family_C,
			&p_descHausdorffDistance<float>,
			&p_vectorHammingDistance<float>
		) << endl;
	cout << endl;
	/////////////////////////////////////////////////////////////

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