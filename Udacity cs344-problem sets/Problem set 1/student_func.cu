// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"

#include "device_launch_parameters.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset
	/*int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	int tid = tidY * numCols + tidX;
	if (tidX < numCols && tidY < numRows)
	{
		greyImage[tid] = 0.299 * (float)rgbaImage[tid].x + 0.587 * (float)rgbaImage[tid].y + 0.114 * (float)rgbaImage[tid].z;
	}
	else
		return;*/
	/*************************************************************
	The output of previous code has a little different with reference img;
	For me, two points are not so clear. First is the setting of the condition, why y < numCols but not x < numCols. Second is that if the data type influence the output;
	Follow code find in github: https://github.com/ibebrett/CUDA-CS344/blob/master/Problem%20Sets/Problem%20Set%201/student_func.cu
	***************************************************************/
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x < numCols && y < numRows)  //change this condition with the same as previous code also OK ? A little curious
	{
		//int index = numRows * y + x;
		int index = y * numCols + x;
		uchar4 color = rgbaImage[index];
		unsigned char grey = (unsigned char)(0.299f * color.x + 0.587f * color.y + 0.114f * color.z);
		greyImage[index] = grey;
	}

}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
	// add the memcpy operation !
	checkCudaErrors(cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numRows * numCols, cudaMemcpyHostToDevice));
  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  const dim3 blockSize(32, 32, 1);  //TODO
  const dim3 gridSize( (numCols - 1) / blockSize.x + 1, (numRows - 1) / blockSize.y + 1, 1);  //TODO
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
