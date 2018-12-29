/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "device_atomic_functions.h"

__global__ void reduce_minimum(float* d_out, const float* const d_in, int numItem)
{
	extern __shared__ float sdata[];
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	if (myId < numItem)
		sdata[tid] = d_in[myId];
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
			sdata[tid] = min(sdata[tid], sdata[tid + offset]);
		__syncthreads();
	}
	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

__global__ void reduce_maximum(float* d_out, const float* const d_in, int numItem)
{
	extern __shared__ float sdata[];
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	if (myId < numItem)
		sdata[tid] = d_in[myId];
	__syncthreads();
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (tid < offset)
			sdata[tid] = max(sdata[tid], sdata[tid + offset]);
		__syncthreads();
	}
	if (tid == 0)
		d_out[blockIdx.x] = sdata[0];
}

__global__ void histogram(int* d_bins, const size_t numBins, const float* const d_in, float min_logLum, float range, const size_t numRows, const size_t numCols)
{
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId >= numRows * numCols)
		return;
	float myItem = d_in[myId];
	int myBin = (myItem - min_logLum) / range * numBins;
	atomicAdd(&(d_bins[myBin]), 1);
}

__global__ void scan(unsigned int* const d_out, int* d_sums, int* d_in, const size_t numBins, int numElems)
{
	extern __shared__ float sdata[];
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	int offset = 1;
	// load two items per thread into shared memory
	if ((2 * myId) < numBins)
		sdata[2 * tid] = d_in[2 * myId];
	else
		sdata[2 * tid] = 0;
	if ((2 * myId + 1) < numBins)
		sdata[2 * tid + 1] = d_in[2 * myId + 1];
	else
		sdata[2 * tid + 1] = 0;
	// reduce
	for (int d = numElems / 2; d > 0; d >>= 1)
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
	// clear the last element
	if (tid == 0)
	{
		d_sums[blockIdx.x] = sdata[numElems - 1];
		sdata[numElems - 1] = 0;
	}
	// down sweep
	for (int d = 1; d < numElems; d *= 2)
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
	// write the output to global memory
	if ((2 * myId) < numBins)
		d_out[2 * myId] = sdata[2 * tid];
	if ((2 * myId + 1) < numBins)
		d_out[2 * myId + 1] = sdata[2 * tid + 1];
}

__global__ void scan2(int* d_out, int* d_in, const size_t numBins, int numElems)
{
	extern __shared__ float sdata[];
	int tid = threadIdx.x;
	int offset = 1;
	// load two items per thread into shared memory
	if ((2 * tid) < numBins)
		sdata[2 * tid] = d_in[2 * tid];
	else
		sdata[2 * tid] = 0;
	if ((2 * tid + 1) < numBins)
		sdata[2 * tid + 1] = d_in[2 * tid + 1];
	else
		sdata[2 * tid + 1] = 0;
	//reduce
	for (int d = numElems / 2; d > 0; d >>= 1)
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
	//clear the last element
	if (tid == 0)
		sdata[numElems - 1] = 0;
	//down sweep
	for (int d = 1; d < numElems; d *= 2)
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
	//write the output to global memory
	if ((2 * tid) < numBins)
		d_out[2 * tid] = sdata[2 * tid];
	if ((2 * tid + 1) < numBins)
		d_out[2 * tid + 1] = sdata[2 * tid + 1];
}

__global__ void add_scan(unsigned int* const d_out, int* d_in, const size_t numBins)
{
	if (blockIdx.x == 0)
		return;
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	int myOffset = d_in[blockIdx.x];
	if ((2 * myId) < numBins)
		d_out[2 * myId] += myOffset;
	if ((2 * myId + 1) < numBins)
		d_out[2 * myId + 1] += myOffset;
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	int numItem = numRows * numCols;
	dim3 block(256, 1);
	dim3 grid(numItem / block.x + 1, 1);
	float* d_inter_min;
	float* d_inter_max;
	int* d_histogram;
	int* d_sums;
	int* d_incr;
	checkCudaErrors(cudaMalloc(&d_inter_min, sizeof(float) * grid.x));
	checkCudaErrors(cudaMalloc(&d_inter_max, sizeof(float) * grid.x));
	checkCudaErrors(cudaMalloc(&d_histogram, sizeof(int) * numBins));
	checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(int) * numBins));
	// step1: reduce(min and max). It could be done in one step only!  
	///(In fact, I'm not so clear how to do this in a single step; except first find the value we need in all different blocks and then push all of these in the first block, I think)
	reduce_minimum << <grid, block, sizeof(float) * block.x >> > (d_inter_min, d_logLuminance, numItem);
	reduce_maximum << <grid, block, sizeof(float) * block.x >> > (d_inter_max, d_logLuminance, numItem);
	numItem = grid.x;
	grid.x = numItem / block.x + 1;
	while (numItem > 1)
	{
		reduce_minimum << <grid, block, sizeof(float) * block.x >> > (d_inter_min, d_inter_min, numItem);
		reduce_maximum << <grid, block, sizeof(float) * block.x >> > (d_inter_max, d_inter_max, numItem);
		numItem = grid.x;
		grid.x = numItem / block.x + 1;
	}
	//step 2: range
	checkCudaErrors(cudaMemcpy(&min_logLum, d_inter_min, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_inter_max, sizeof(float), cudaMemcpyDeviceToHost));
	float range = max_logLum - min_logLum;
	// step 3: histogram
	grid.x = numRows * numCols / block.x + 1;
	histogram << <grid, block >> > (d_histogram, numBins, d_logLuminance, min_logLum, range, numRows, numCols);
	//step 4: Exclusive scan - Blelloch
	int numElems = 256;
	block.x = numElems / 2;
	grid.x = numBins / numElems;
	if (numBins % numElems != 0)
		grid.x++;
	checkCudaErrors(cudaMalloc(&d_sums, sizeof(int) * grid.x));
	checkCudaErrors(cudaMemset(d_sums, 0, sizeof(int) * grid.x));

	//first level scan to obtain the scanned blocks
	scan << <grid, block, sizeof(float) * numElems >> > (d_cdf, d_sums, d_histogram, numBins, numElems);
	//second level scan to obtain the scannen blocks sums
	/****************************************************************************/
	numElems = grid.x;
	unsigned int nextPow = numElems;
	nextPow--;
	nextPow = (nextPow >> 1) | nextPow;
	nextPow = (nextPow >> 2) | nextPow;
	nextPow = (nextPow >> 4) | nextPow;
	nextPow = (nextPow >> 8) | nextPow;
	nextPow = (nextPow >> 16) | nextPow;
	nextPow++;
	// Not so clear about the nextPow operations, some confused!
	/*****************************************************************************/
	block.x = nextPow / 2;
	grid.x = 1;
	checkCudaErrors(cudaMalloc(&d_incr, sizeof(int) * numElems));
	checkCudaErrors(cudaMemset(d_incr, 0, sizeof(int) * numElems));
	scan2 << <grid, block, sizeof(float) * nextPow >> > (d_incr, d_sums, numElems, nextPow);
	// add scanned block sum i to all value of scanned block
	numElems = 256;
	block.x = numElems / 2;
	grid.x = numBins / numElems;
	if (numBins % numElems != 0)
		grid.x++;
	add_scan << <grid, block >> > (d_cdf, d_incr, numBins);

    //clean memory
	checkCudaErrors(cudaFree(d_inter_min));
	checkCudaErrors(cudaFree(d_inter_max));
	checkCudaErrors(cudaFree(d_histogram));
	checkCudaErrors(cudaFree(d_sums));
	checkCudaErrors(cudaFree(d_incr));
}
