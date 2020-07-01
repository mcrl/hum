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

#include "NVLibs.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>

#include <dirent.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <assert.h>

#define CHECK_SYMBOL(x)                             \
	do {                                              \
		if (x == NULL) {                                      \
			printf("Failed to load symbol " #x "\n");     \
			assert(0);																		\
			return;                                       \
		}                                               \
	} while (0)


NVLibs::NVLibs() {
	OpenCUDARTLib();
	OpenCUDADrvLib();
}

NVLibs::~NVLibs() {
	dlclose(cudart_handle_);
	dlclose(cudadrv_handle_);
}

void NVLibs::OpenCUDARTLib() {
	cudart_handle_ = dlopen("/usr/local/cuda/lib64/libcudart.so", RTLD_NOW);
	if (!cudart_handle_) {
		printf("Failed to open cuda runtime library file\n");
		return;
	}

	__cudaInitModule = (void(*)(void**))dlsym(cudart_handle_, "__cudaInitModule");
	CHECK_SYMBOL(__cudaInitModule);

	__cudaRegisterFunction = (void(*)(void **fatCubinHandle, const char *hostFun, char *deviceFun,
				const char *deviceName, int thread_limit, uint3 *tid,
				uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize))dlsym(cudart_handle_, "__cudaRegisterFunction");
	CHECK_SYMBOL(__cudaRegisterFunction);

	__cudaRegisterVar = (void(*) (void **fatCubinHandle, char *hostVar, char *deviceAddress,
				const char *deviceName, int ext, size_t size,
				int constant, int global))dlsym(cudart_handle_, "__cudaRegisterVar");
	CHECK_SYMBOL(__cudaRegisterVar);

	__cudaRegisterTexture = (void(*) (void **fatCubinHandle, const struct textureReference *hostVar,
				const void **deviceAddress, const char *deviceName,
				int dim, int norm, int ext))dlsym(cudart_handle_, "__cudaRegisterTexture");
	CHECK_SYMBOL(__cudaRegisterTexture);

	__cudaRegisterFatBinary = (void**(*)(void *fatCubin))dlsym(cudart_handle_, "__cudaRegisterFatBinary");
	CHECK_SYMBOL(__cudaRegisterFatBinary);

	__cudaRegisterFatBinaryEnd = (void(*)(void **fatCubinHandle))dlsym(cudart_handle_, "__cudaRegisterFatBinaryEnd");
	CHECK_SYMBOL(__cudaRegisterFatBinaryEnd);

	__cudaUnregisterFatBinary = (void(*)(void **fatCubinHandle))dlsym(cudart_handle_, "__cudaUnregisterFatBinary");
	CHECK_SYMBOL(__cudaUnregisterFatBinary);

	__cudaPopCallConfiguration = (cudaError_t(*)(dim3 *gridDim,
				dim3 *blockDim,  size_t *sharedMem, void *stream))dlsym(cudart_handle_, "__cudaPopCallConfiguration");
	CHECK_SYMBOL(__cudaPopCallConfiguration);

	__cudaPushCallConfiguration = (unsigned(*)(dim3 gridDim,
				dim3 blockDim,  size_t sharedMem, void *stream))dlsym(cudart_handle_, "__cudaPushCallConfiguration");
	CHECK_SYMBOL(__cudaPushCallConfiguration);

	cudaGetDevice = (cudaError_t(*)(int*))dlsym(cudart_handle_, "cudaGetDevice");
	CHECK_SYMBOL(cudaGetDevice);

	cudaSetDevice = (cudaError_t(*)(int))dlsym(cudart_handle_, "cudaSetDevice");
	CHECK_SYMBOL(cudaSetDevice);

  cudaSetDeviceFlags = (cudaError_t(*)(unsigned int))dlsym(cudart_handle_, "cudaSetDeviceFlags");
  CHECK_SYMBOL(cudaSetDeviceFlags);

	cudaGetDeviceCount = (cudaError_t (*) ( int* ))
		dlsym(cudart_handle_, "cudaGetDeviceCount");
	CHECK_SYMBOL(cudaGetDeviceCount);

	cudaGetDeviceProperties = (cudaError_t(*)(struct cudaDeviceProp*, int))
		dlsym(cudart_handle_, "cudaGetDeviceProperties");
	CHECK_SYMBOL(cudaGetDeviceProperties);

	cudaDeviceSynchronize = (cudaError_t(*)(void))
		dlsym(cudart_handle_, "cudaDeviceSynchronize");
	CHECK_SYMBOL(cudaDeviceSynchronize);

	cudaStreamSynchronize = (cudaError_t(*)(cudaStream_t))
		dlsym(cudart_handle_, "cudaStreamSynchronize");
	CHECK_SYMBOL(cudaStreamSynchronize);

	cudaMalloc = (cudaError_t(*)(void**, size_t))
		dlsym(cudart_handle_, "cudaMalloc");
	CHECK_SYMBOL(cudaMalloc);

	cudaMallocHost = (cudaError_t(*)(void**, size_t))
		dlsym(cudart_handle_, "cudaMallocHost");
	CHECK_SYMBOL(cudaMallocHost);

	cudaHostAlloc = (cudaError_t(*)(void**, size_t, unsigned int))
		dlsym(cudart_handle_, "cudaHostAlloc");
	CHECK_SYMBOL(cudaHostAlloc);

	cudaMallocManaged = (cudaError_t(*)(void**, size_t, unsigned int))
		dlsym(cudart_handle_, "cudaMallocManaged");
	CHECK_SYMBOL(cudaMallocManaged);

	cudaFree = (cudaError_t(*)(void*))
		dlsym(cudart_handle_, "cudaFree");
	CHECK_SYMBOL(cudaFree);

	cudaFreeHost = (cudaError_t(*)(void*))
		dlsym(cudart_handle_, "cudaFreeHost");
	CHECK_SYMBOL(cudaFreeHost);

	cudaMemset = (cudaError_t(*)(void*, int, size_t))
		dlsym(cudart_handle_, "cudaMemset");
	CHECK_SYMBOL(cudaMemset);

	cudaMemsetAsync = (cudaError_t(*)(void*, int, size_t, cudaStream_t))
		dlsym(cudart_handle_, "cudaMemsetAsync");
	CHECK_SYMBOL(cudaMemsetAsync);

	cudaMemcpy = (cudaError_t(*)(void*, const void*, size_t, enum cudaMemcpyKind))
		dlsym(cudart_handle_, "cudaMemcpy");
	CHECK_SYMBOL(cudaMemcpy);

	cudaMemcpyAsync = (cudaError_t(*)(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t))
		dlsym(cudart_handle_, "cudaMemcpyAsync");
	CHECK_SYMBOL(cudaMemcpyAsync);
  
  cudaMemcpyToArrayAsync = (cudaError_t(*)(cudaArray_t, size_t, size_t,
        const void *, size_t, enum cudaMemcpyKind, cudaStream_t))
    dlsym(cudart_handle_, "cudaMemcpyToArrayAsync");
  CHECK_SYMBOL(cudaMemcpyToArrayAsync);

	cudaMemcpyToSymbolAsync = (cudaError_t(*)(const void*, const void*, size_t,
        size_t, enum cudaMemcpyKind, cudaStream_t))
		dlsym(cudart_handle_, "cudaMemcpyToSymbolAsync");
	CHECK_SYMBOL(cudaMemcpyToSymbolAsync);

	cudaMemcpyFromSymbolAsync = (cudaError_t(*)(void*, const void*, size_t, size_t, enum cudaMemcpyKind, cudaStream_t))
		dlsym(cudart_handle_, "cudaMemcpyFromSymbolAsync");
	CHECK_SYMBOL(cudaMemcpyFromSymbolAsync);

	cudaMemGetInfo = (cudaError_t(*)(size_t*, size_t*))
		dlsym(cudart_handle_, "cudaMemGetInfo");
	CHECK_SYMBOL(cudaMemGetInfo);
  
  cudaHostGetDevicePointer = (cudaError_t(*)(void**, void*, unsigned int))
    dlsym(cudart_handle_, "cudaHostGetDevicePointer");
  CHECK_SYMBOL(cudaHostGetDevicePointer);

	cudaMemPrefetchAsync = (cudaError_t(*)(const void*, size_t, int dstDevice, cudaStream_t stream))
		dlsym(cudart_handle_, "cudaMemPrefetchAsync");
	CHECK_SYMBOL(cudaMemPrefetchAsync);

  cudaFuncGetAttributes = (cudaError_t(*)(struct cudaFuncAttributes *, const void *))
    dlsym(cudart_handle_, "cudaFuncGetAttributes");
  CHECK_SYMBOL(cudaFuncGetAttributes);

  cudaFuncSetAttribute = (cudaError_t(*)(const void*, enum cudaFuncAttribute, int))
    dlsym(cudart_handle_, "cudaFuncSetAttribute");
  CHECK_SYMBOL(cudaFuncSetAttribute);

	cudaLaunchKernel = (cudaError_t(*)( const void* func, 
				dim3 gridDim, dim3 blockDim, 
				void** args, size_t sharedMem, 
				cudaStream_t stream))
		dlsym(cudart_handle_, "cudaLaunchKernel");
	CHECK_SYMBOL(cudaLaunchKernel);
  
  cudaMalloc3DArray = (cudaError_t(*)(cudaArray_t*, const cudaChannelFormatDesc*, cudaExtent, unsigned int))
    dlsym(cudart_handle_, "cudaMalloc3DArray");
  CHECK_SYMBOL(cudaMalloc3DArray);

  cudaMallocArray = (cudaError_t(*)(cudaArray_t*, const cudaChannelFormatDesc*, size_t, size_t, unsigned int))
    dlsym(cudart_handle_, "cudaMallocArray");
  CHECK_SYMBOL(cudaMallocArray);

  cudaFreeArray = (cudaError_t(*)(cudaArray_t))
    dlsym(cudart_handle_, "cudaFreeArray");
  CHECK_SYMBOL(cudaFreeArray);

  cudaMemcpy2DToArray = (cudaError_t(*)(cudaArray_t, size_t, size_t,
      const void*, size_t, size_t, size_t, cudaMemcpyKind))
    dlsym(cudart_handle_, "cudaMemcpy2DToArray");
  CHECK_SYMBOL(cudaMemcpy2DToArray);
  
  cudaMemcpy2DToArrayAsync = (cudaError_t(*)(cudaArray_t, size_t, size_t,
      const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t))
    dlsym(cudart_handle_, "cudaMemcpy2DToArrayAsync");
  CHECK_SYMBOL(cudaMemcpy2DToArrayAsync);

  cudaMemcpy3DAsync = (cudaError_t(*)(const cudaMemcpy3DParms*, cudaStream_t))
    dlsym(cudart_handle_, "cudaMemcpy3DAsync");
  CHECK_SYMBOL(cudaMemcpy3DAsync);
  
  cudaBindTextureToArray = (cudaError_t(*)(const struct textureReference *,
        cudaArray_const_t, const struct cudaChannelFormatDesc *))
    dlsym(cudart_handle_, "cudaBindTextureToArray");
  CHECK_SYMBOL(cudaBindTextureToArray);

	cudaDeviceCanAccessPeer = (cudaError_t(*)(int*, int, int))
		dlsym(cudart_handle_, "cudaDeviceCanAccessPeer");
	CHECK_SYMBOL(cudaDeviceCanAccessPeer);

	cudaDeviceEnablePeerAccess = (cudaError_t(*)(int, unsigned int))
		dlsym(cudart_handle_, "cudaDeviceEnablePeerAccess");
	CHECK_SYMBOL(cudaDeviceEnablePeerAccess);

	cudaDeviceDisablePeerAccess = (cudaError_t(*)(int))
		dlsym(cudart_handle_, "cudaDeviceDisablePeerAccess");
	CHECK_SYMBOL(cudaDeviceDisablePeerAccess);

	cudaCreateChannelDesc = (cudaChannelFormatDesc (*) ( int  x, int  y, 
				int  z, int  w, cudaChannelFormatKind f ))
		dlsym(cudart_handle_, "cudaCreateChannelDesc");
	CHECK_SYMBOL(cudaCreateChannelDesc);

  cudaGetChannelDesc = (cudaError_t(*)(cudaChannelFormatDesc*, cudaArray_const_t))
    dlsym(cudart_handle_, "cudaGetChannelDesc");
  CHECK_SYMBOL(cudaGetChannelDesc);

	cudaBindTexture = (cudaError_t (*)(size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size))
		dlsym(cudart_handle_, "cudaBindTexture");
	CHECK_SYMBOL(cudaBindTexture);

	cudaUnbindTexture = (cudaError_t (*)( const textureReference* texref ))
		dlsym(cudart_handle_, "cudaUnbindTexture");
	CHECK_SYMBOL(cudaUnbindTexture);
}

void NVLibs::OpenCUDADrvLib() {
	cudadrv_handle_ = dlopen("/usr/lib/x86_64-linux-gnu/libcuda.so", RTLD_NOW);
	if (!cudadrv_handle_) {
		printf("Failed to open cuda driver library file\n");
		return;
	}

	cuDeviceGetCount = (CUresult (*) ( int* ))
		dlsym(cudadrv_handle_, "cuDeviceGetCount");
	CHECK_SYMBOL(cuDeviceGetCount);

	cuDeviceGet = (CUresult (*) (CUdevice*, int))
		dlsym(cudadrv_handle_, "cuDeviceGet");
	CHECK_SYMBOL(cuDeviceGet);

	cuDevicePrimaryCtxRetain = (CUresult (*) (CUcontext*, CUdevice))
		dlsym(cudadrv_handle_, "cuDevicePrimaryCtxRetain");
	CHECK_SYMBOL(cuDevicePrimaryCtxRetain);

	cuDevicePrimaryCtxRelease = (CUresult (*) (CUdevice))
		dlsym(cudadrv_handle_, "cuDevicePrimaryCtxRelease");
	CHECK_SYMBOL(cuDevicePrimaryCtxRelease);

	cuCtxSetCurrent = (CUresult (*) (CUcontext))
		dlsym(cudadrv_handle_, "cuCtxSetCurrent");
	CHECK_SYMBOL(cuCtxSetCurrent);

	cuStreamCreate = (CUresult (*) ( CUstream* , unsigned int ))
		dlsym(cudadrv_handle_, "cuStreamCreate");
	CHECK_SYMBOL(cuStreamCreate);

	cuStreamDestroy = (CUresult (*) ( CUstream ))
		dlsym(cudadrv_handle_, "cuStreamDestroy");
	CHECK_SYMBOL(cuStreamDestroy);

}
