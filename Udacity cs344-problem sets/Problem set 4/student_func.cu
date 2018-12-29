//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

#include "device_launch_parameters.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

const int MAX_THREADS_PER_BLOCK = 512;


__global__ void split_array(unsigned int* d_inputVals, unsigned int* d_splitVals,
							const size_t numElems, unsigned int mask, unsigned int ibit)
{
	int array_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (array_idx >= numElems)
		return;
	d_splitVals[array_idx] = !(d_inputVals[array_idx] & mask);
	//why this ? 18/12/28
}


__global__ void blelloch_scan_single_block(unsigned int* d_in_array, const size_t numBins, unsigned normalization = 0)
{
	int tid = threadIdx.x;
	extern __shared__ float sdata[];
	if (tid < numBins)
		sdata[tid] = d_in_array[tid];
	else
		sdata[tid] = 0;
	if ((tid + numBins / 2) < numBins)
		sdata[tid + numBins / 2] = d_in_array[tid + numBins / 2];
	else
		sdata[tid + numBins / 2] = 0;
	__syncthreads();
	//reduction
	int offset = 1;
	for (int d = numBins / 2; d > 0; d >>= 1)
	{
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			sdata[bi] += sdata[ai];
		}
		offset *= 2;
		__syncthreads();
	}
	//clear last element
	if (tid == 0)
		sdata[numBins - 1] = 0;
	//down sweep
	for (int d = 1; d < numBins; d *= 2)
	{
		offset >>= 1;
		if (tid < d)
		{
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			float t = sdata[ai];
			sdata[ai] = sdata[bi];
			sdata[bi] += t;
		}
		__syncthreads();
	}
	//write to original memory
	if (tid < numBins)
		d_in_array[tid] = sdata[tid] + normalization;
	if ((tid + numBins / 2) < numBins)
		d_in_array[tid + numBins / 2] = sdata[tid + numBins / 2] + normalization;
}

__global__ void compute_outputPos(const unsigned int* d_inputVals, unsigned int* d_outputVals,
	                                unsigned int* d_outputPos, unsigned int* d_tVals,
									const unsigned int* d_splitVals, const unsigned int* d_cdf,
                                 	const unsigned int totalFalses, const unsigned int numElems)
{
	int tid = threadIdx.x;
	int myId = blockIdx.x * blockDim.x + tid;
	if (myId >= numElems)
		return;
	d_tVals[myId] = myId - d_cdf[myId] + totalFalses;
	unsigned int scatter = (!(d_splitVals[myId]) ? d_tVals[myId] : d_cdf[myId]);
	d_outputPos[myId] = scatter;
}

__global__ void do_scatter(unsigned int* d_outputVals, const unsigned int* d_inputVals,
							unsigned int* d_outputPos, unsigned int* d_inputPos,
							unsigned int* d_scatterAddr, const unsigned int numElems)
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId >= numElems)
		return;
	d_outputVals[d_outputPos[myId]] = d_inputVals[myId];
	d_scatterAddr[d_outputPos[myId]] = d_inputPos[myId];
	//这个地方是很不理解
}
//上面几个函数除了scan的使用剩下的三个都还没有理解在算法中的具体应用过程
void full_blelloch_exclusive_scan(unsigned int* d_binScan, const size_t totalNumElems)
{
	int nthreads = MAX_THREADS_PER_BLOCK;
	int nblocksTotal = (totalNumElems / 2 - 1) / nthreads + 1;
	int partialBins = 2 * nthreads;
	int smSize = partialBins * sizeof(unsigned);
	// Need a balanced d_binScan array so that on final block, correct
	// values are given to d_partialBinScan.
	// 1. define balanced bin scan
	// 2. set all values to zero
	// 3. copy all of binScan into binScanBalanced.

	unsigned int* d_binScanBalanced;
	unsigned int balanced_size = nblocksTotal * partialBins * sizeof(unsigned);
	checkCudaErrors(cudaMalloc((void**)&d_binScanBalanced, balanced_size));
	checkCudaErrors(cudaMemset(d_binScanBalanced, 0, balanced_size));
	checkCudaErrors(cudaMemcpy(d_binScanBalanced, d_binScan, totalNumElems * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	
	unsigned int* d_partialBinScan;
	checkCudaErrors(cudaMalloc((void**)&d_partialBinScan, partialBins * sizeof(unsigned)));

	unsigned int* normalization = (unsigned*)malloc(sizeof(unsigned));
	unsigned int* lastVal = (unsigned*)malloc(sizeof(unsigned));
	for (unsigned iblock = 0; iblock < nblocksTotal; iblock++)
	{
		unsigned offset = iblock * partialBins;
		//copy binScan partition into partialBinScan
		checkCudaErrors(cudaMemcpy(d_partialBinScan, (d_binScanBalanced + offset), smSize, cudaMemcpyDeviceToDevice));
		if (iblock > 0)
		{
			//get normalization - final value in last cdf bin + last value in original
			checkCudaErrors(cudaMemcpy(normalization, (d_binScanBalanced + (offset - 1)), sizeof(unsigned), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaMemcpy(lastVal, (d_binScan + (offset - 1)), sizeof(unsigned), cudaMemcpyDeviceToHost));
			*normalization += (*lastVal);
		}
		else
			*normalization = 0;
		blelloch_scan_single_block << <1, nthreads, smSize >> > (d_partialBinScan, partialBins, *normalization);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

		//copy partialBinScan back into binScanBalanced
		checkCudaErrors(cudaMemcpy((d_binScanBalanced + offset), d_partialBinScan, smSize, cudaMemcpyDeviceToDevice));
	}
	// ONE BLOCK WORKING HERE! ↑
	checkCudaErrors(cudaMemcpy(d_binScan, d_binScanBalanced, totalNumElems * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	free(normalization);
	free(lastVal);
	checkCudaErrors(cudaFree(d_binScanBalanced));
	checkCudaErrors(cudaFree(d_partialBinScan));
}


void compute_scatter_addresses(const unsigned int* d_inputVals, unsigned int* d_outputVals,
								unsigned int* d_inputPos, unsigned int* d_outputPos, unsigned int* d_scatterAddr,
								const unsigned int* const d_splitVals, const unsigned int* const d_cdf, const unsigned totalFalses, const size_t numElems)
//Modifies d_outputVals and d_outputPos
{
	unsigned int* d_tVals;
	checkCudaErrors(cudaMalloc((void**)&d_tVals, numElems * sizeof(unsigned)));
	int nthreads = MAX_THREADS_PER_BLOCK;
	int nblocks = (numElems - 1) / nthreads + 1;
	compute_outputPos << <nblocks, nthreads >> > (d_inputVals, d_outputVals, d_outputPos, d_tVals,
													d_splitVals, d_cdf, totalFalses, numElems);
	// testing purposes - Remove in production  
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	do_scatter << <nblocks, nthreads >> > (d_outputVals, d_inputVals, d_outputPos, d_inputPos, d_scatterAddr, numElems);
	// testing purposes - Remove in production
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_tVals));
	//最后两步的移除没看懂，整个kernel都不需要了；其实到这已经完全看不懂了，只是凭意思来猜
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE

	// set up
	const int numBits = 1;
	unsigned int* d_splitVals;
	checkCudaErrors(cudaMalloc((void**)&d_splitVals, numElems * sizeof(unsigned)));
	unsigned int* d_cdf;
	checkCudaErrors(cudaMalloc((void**)&d_cdf, numElems * sizeof(unsigned)));

	// d_scatterAddr keeps track of the scattered original address at every pass
	unsigned int* d_scatterAddr;
	checkCudaErrors(cudaMalloc((void**)&d_scatterAddr, numElems * sizeof(unsigned)));
	checkCudaErrors(cudaMemcpy(d_scatterAddr, d_inputPos, numElems * sizeof(unsigned), cudaMemcpyDeviceToDevice));

	// need a global device array for blelloch scan
	const int nBlellochBins = 1 << unsigned(log((long double)numElems) / log((long double)2) + 0.5);
	unsigned int* d_blelloch;
	checkCudaErrors(cudaMalloc((void**)&d_blelloch, nBlellochBins * sizeof(unsigned)));

	unsigned int* d_inVals = d_inputVals;
	unsigned int* d_inPos = d_inputPos;
	unsigned int* d_outVals = d_outputVals;
	unsigned int* d_outPos = d_outputPos;

	//testing putpose?
	unsigned int* h_splitVals = (unsigned*)malloc(numElems * sizeof(unsigned));
	unsigned int* h_cdf = (unsigned*)malloc(numElems * sizeof(unsigned));
	unsigned int* h_inVals = (unsigned*)malloc(numElems * sizeof(unsigned));
	unsigned int* h_outVals = (unsigned*)malloc(numElems * sizeof(unsigned));
	unsigned int* h_inPos = (unsigned*)malloc(numElems * sizeof(unsigned));
	unsigned int* h_outPos = (unsigned*)malloc(numElems * sizeof(unsigned));

	/*
	parallel radix sort - for each pass(each bit)
	1) split values based on current bit
	2) scan values of split array
	3) compute scatter output position
	4) scatter output values using inputVals and outputPos
	*/
	for (unsigned ibit = 0; ibit < 8 * sizeof(unsigned); ibit += numBits)  //understand can't
	{
		checkCudaErrors(cudaMemset(d_splitVals, 0, numElems * sizeof(unsigned)));
		checkCudaErrors(cudaMemset(d_cdf, 0, numElems * sizeof(unsigned)));
		checkCudaErrors(cudaMemset(d_blelloch, 0, nBlellochBins * sizeof(unsigned)));
		//step 1: split values on true if bit matches 0 in the given bit
		unsigned int mask = 1 << ibit;
		int nthreads = MAX_THREADS_PER_BLOCK;
		int nblocks = (numElems - 1) / nthreads + 1;
		split_array << <nblocks, nthreads >> > (d_inVals, d_splitVals, numElems, mask, ibit);
		//testing purpos ? ↑？
		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaMemcpy(d_cdf, d_splitVals, numElems * sizeof(unsigned), cudaMemcpyDeviceToDevice));
		// step 2:scan values of split array Use Blelloch exclusive scan
		full_blelloch_exclusive_scan(d_cdf, numElems);

		//step 3: compute scatter addresses
		unsigned totalFalses = 0;
		checkCudaErrors(cudaMemcpy(h_splitVals, d_splitVals + (numElems - 1), sizeof(unsigned), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_cdf, d_cdf + (numElems - 1), sizeof(unsigned), cudaMemcpyDeviceToHost));
		totalFalses = h_splitVals[0] + h_cdf[0];
		compute_scatter_addresses(d_inVals, d_outVals, d_inPos, d_outPos, d_scatterAddr, d_splitVals, d_cdf, totalFalses, numElems);
		std::swap(d_inVals, d_outVals);
		std::swap(d_inPos, d_scatterAddr);
	}
	// need this?
	checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inPos, numElems * sizeof(unsigned), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaFree(d_splitVals));
	checkCudaErrors(cudaFree(d_cdf));
	checkCudaErrors(cudaFree(d_blelloch));
	free(h_splitVals);
	free(h_cdf);
	free(h_inVals);
	free(h_outVals);
	free(h_inPos);
	free(h_outPos);
}
