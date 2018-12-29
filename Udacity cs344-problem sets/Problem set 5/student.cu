/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"
#include "device_launch_parameters.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
	int myId = blockIdx.x * blockDim.x + threadIdx.x;
	if (myId > numVals)
		return;
	atomicAdd(&histo[vals[myId]], 1);  //this kernel run 3.310368ms
	
}


__global__ void sharedHisto(const unsigned int* const vals, unsigned int* const histo, unsigned int numVals, unsigned int numBins)
{
	extern __shared__ unsigned int sdata[];
	for (int i = threadIdx.x; i < numBins; i += blockDim.x)
		sdata[i] = 0;
	__syncthreads();
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalId < numVals)
		atomicAdd(&sdata[vals[globalId]], 1);
	__syncthreads();
	for (int i = threadIdx.x; i < numBins; i += blockDim.x)
	{
		atomicAdd(&histo[i], sdata[i]);
	}
	//I always think the shared memory is the memory per block, but in fact it should be whole memory size ?
	//And this use share memory's kernel run much time about 5ms
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free
	dim3 block(1024, 1);
	dim3 grid((numElems - 1) / block.x + 1, 1);
	//yourHisto << <grid, block >> > (d_vals, d_histo, numElems);
	int sSize = numBins * sizeof(unsigned int);
	sharedHisto << <grid, block, sSize >> > (d_vals, d_histo, numElems, numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
