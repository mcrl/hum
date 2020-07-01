/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2020 Seoul National University.                             */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE           */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   Center for Manycore Programming                                         */
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 08826, Korea                           */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jaehoon Jung, Jungho Park, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <random>

__global__ void vectorAdd(float *A, float *B, float *C, int num_elements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < num_elements) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  struct timeval program_start, program_end;
  double program_time = 0;

  cudaError_t err = cudaSuccess;

  int num_elements = 104857600;
  size_t size = num_elements * sizeof(float);
  printf("Vector addition sample: %d elements\n", num_elements);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);
  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  std::random_device rd;
  std::mt19937 mersenne(rd());
  std::uniform_int_distribution<> die(1, 6);
  for (int i = 0; i < num_elements; ++i)
  {
      h_A[i] = die(mersenne)/(float)RAND_MAX;
      h_B[i] = die(mersenne)/(float)RAND_MAX;
  }

  float *d_A = NULL;
  err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *d_B = NULL;
  err = cudaMalloc((void **)&d_B, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *d_C = NULL;
  err = cudaMalloc((void **)&d_C, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  gettimeofday(&program_start, NULL);

  printf("Copy input data from the host memory to the GPU\n");
  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector A from host to device (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector B from host to device (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int block_dim = 256;
  int grid_dim =(num_elements + block_dim - 1) / block_dim;
  printf("Launching CUDA kernel with %d blocks of %d threads\n", grid_dim, block_dim);
  
  vectorAdd<<<grid_dim, block_dim>>>(d_A, d_B, d_C, num_elements);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch CUDA kernel (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy result data from the device memory to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector C from device to host (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  gettimeofday(&program_end, NULL);

  for (int i = 0; i < num_elements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Verification: True\n");

  err = cudaFree(d_A);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector A (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_B);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector B (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaFree(d_C);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device vector C (error %s)!\n",
        cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  free(h_A);
  free(h_B);
  free(h_C);

  program_time += (program_end.tv_sec - program_start.tv_sec) 
    + (program_end.tv_usec - program_start.tv_usec) / 1000000.0;
  printf("=== Program Elapsed Time = %.6lfs ===\n", program_time);

  return 0;
}
