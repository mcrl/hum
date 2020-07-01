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

#ifndef __HUM_NV_LIBS_H__
#define __HUM_NV_LIBS_H__

#include <map>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

class NVLibs {
	public:
		NVLibs();
		~NVLibs();

		void OpenCUDARTLib();
		void OpenCUDADrvLib();

		// cuda runtime
		void (*__cudaInitModule)(void **fatCubinHandle);
		void (*__cudaRegisterFunction)(void **fatCubinHandle, const char *hostFun, char *deviceFun,
				const char *deviceName, int thread_limit, uint3 *tid,
				uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
		void (*__cudaRegisterVar)(void **fatCubinHandle, char *hostVar, char *deviceAddress,
				const char *deviceName, int ext, size_t size,
				int constant, int global);
		void (*__cudaRegisterTexture)(void **fatCubinHandle, const struct textureReference *hostVar,
				const void **deviceAddress, const char *deviceName,
				int dim, int norm, int ext);

		void** (*__cudaRegisterFatBinary)(void *fatCubin);
		void (*__cudaRegisterFatBinaryEnd)(void **fatCubinHandle);
		void (*__cudaUnregisterFatBinary)(void **fatCubinHandle);
		cudaError_t (*__cudaPopCallConfiguration)(dim3 *gridDim,
				dim3 *blockDim,  size_t *sharedMem, void *stream);
		unsigned (*__cudaPushCallConfiguration)(dim3 gridDim,
				dim3 blockDim,  size_t sharedMem, void *stream);


		cudaError_t (*cudaGetDevice)(int*);
		cudaError_t (*cudaSetDevice)(int);
    cudaError_t (*cudaSetDeviceFlags)(unsigned int);
		cudaError_t	(*cudaGetDeviceCount) ( int* count );
		cudaError_t (*cudaGetDeviceProperties)(struct cudaDeviceProp*, int);
		cudaError_t (*cudaDeviceSynchronize)(void);
		cudaError_t (*cudaStreamSynchronize)(cudaStream_t);
		cudaError_t (*cudaMalloc)(void**, size_t);
		cudaError_t (*cudaMallocHost)(void**, size_t);
		cudaError_t (*cudaHostAlloc)(void**, size_t, unsigned int);
		cudaError_t (*cudaMallocManaged)(void**, size_t, unsigned int);
		cudaError_t (*cudaFree)(void*);
		cudaError_t (*cudaFreeHost)(void*);
		cudaError_t (*cudaMemset)(void*, int, size_t);
		cudaError_t (*cudaMemsetAsync)(void*, int, size_t, cudaStream_t);
		cudaError_t (*cudaMemcpy)(void*, const void*, size_t, enum cudaMemcpyKind);
		cudaError_t (*cudaMemcpyAsync)(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t stream);
    cudaError_t (*cudaMemcpyToArrayAsync)(cudaArray_t, size_t, size_t, const void *,
        size_t, enum cudaMemcpyKind, cudaStream_t);
		cudaError_t (*cudaMemcpyToSymbolAsync)(const void*, const void*, size_t, size_t, enum cudaMemcpyKind, cudaStream_t stream);
		cudaError_t (*cudaMemcpyFromSymbolAsync)(void*, const void*, size_t, size_t,
        enum cudaMemcpyKind, cudaStream_t stream);
		cudaError_t (*cudaMemGetInfo)(size_t*, size_t*);
    cudaError_t (*cudaHostGetDevicePointer)(void**, void*, unsigned int);
		cudaError_t (*cudaMemPrefetchAsync) (const void* devPtr, size_t count, int  dstDevice, cudaStream_t stream);
    
    cudaError_t (*cudaMalloc3DArray)(cudaArray_t*, const cudaChannelFormatDesc*,
        cudaExtent, unsigned int);
    cudaError_t (*cudaMallocArray)(cudaArray_t*, const cudaChannelFormatDesc*, size_t, size_t, unsigned int);
    cudaError_t (*cudaFreeArray)(cudaArray_t);
    cudaError_t (*cudaMemcpy2DToArray)(cudaArray_t, size_t, size_t,
        const void*, size_t, size_t, size_t, cudaMemcpyKind);
    cudaError_t (*cudaMemcpy2DToArrayAsync)(cudaArray_t, size_t, size_t,
        const void*, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
    cudaError_t (*cudaMemcpy3DAsync)(const cudaMemcpy3DParms*, cudaStream_t);
    cudaError_t (*cudaBindTextureToArray)(const struct textureReference *,
        cudaArray_const_t, const struct cudaChannelFormatDesc *);
    
    cudaError_t (*cudaFuncGetAttributes)(struct cudaFuncAttributes *, const void *);
    cudaError_t (*cudaFuncSetAttribute)(const void*, enum cudaFuncAttribute, int);
		cudaError_t (*cudaLaunchKernel)( const void* func, 
				dim3 gridDim, dim3 blockDim, 
				void** args, size_t sharedMem, 
				cudaStream_t stream );


		cudaError_t (*cudaDeviceCanAccessPeer)(int*, int, int);
		cudaError_t (*cudaDeviceEnablePeerAccess)(int, unsigned int);
		cudaError_t (*cudaDeviceDisablePeerAccess)(int);


		cudaChannelFormatDesc (*cudaCreateChannelDesc) ( int  x, int  y, 
				int  z, int  w, cudaChannelFormatKind f );
    cudaError_t (*cudaGetChannelDesc)(cudaChannelFormatDesc*, cudaArray_const_t);
		cudaError_t (*cudaBindTexture)(size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size);
		cudaError_t (*cudaUnbindTexture)( const textureReference* texref );

		// Driver API functions
		CUresult (*cuDeviceGetCount)(int* count);
		CUresult (*cuDeviceGet)(CUdevice* device, int ordinal);
		CUresult (*cuDevicePrimaryCtxRetain)(CUcontext* pctx, CUdevice dev);
		CUresult (*cuDevicePrimaryCtxRelease)(CUdevice dev);

		CUresult (*cuCtxSetCurrent)(CUcontext ctx);
		CUresult (*cuStreamCreate) ( CUstream* phStream, unsigned int  Flags );
		CUresult (*cuStreamDestroy) ( CUstream hStream );

	private:
		void* cudart_handle_;
		void* cudadrv_handle_;
};

#endif
