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

#ifndef __HUM_CUDAWRAPPER_H__
#define __HUM_CUDAWRAPPER_H__

#include "Wrapper.h"
#include <cuda_runtime.h>
#include <cuda.h>

#include <stdarg.h>
extern __thread unsigned int cuda_last_error_;
extern __thread int current_device_id_;

class NVLibs;

class CUDAWrapper: public Wrapper {
	public:
		CUDAWrapper();
		~CUDAWrapper();

		NVLibs* nvlibs_;
		pthread_mutex_t mutex_;


		void InitModule(void **fatCubinHandle);
		void RegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
				const char *deviceName, int thread_limit, uint3 *tid,
				uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
		void RegisterVar(void **fatCubinHandle, 	char  *hostVar,	char  *deviceAddress,
				const char *deviceName, int ext, size_t size,
				int constant, int global);
		void RegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
				const void **deviceAddress, const char *deviceName,
				int dim, int norm, int ext);
		void ** RegisterFatBinary(void * fatCubin);
    void RegisterFatBinaryEnd(void **fatCubinHandle);
		void UnregisterFatBinary(void **fatCubinHandle);
		cudaError_t PopCallConfiguration(dim3 *gridDim,
				dim3 *blockDim,  size_t *sharedMem, void *stream);
		unsigned PushCallConfiguration(dim3 gridDim,
				dim3 blockDim, size_t sharedMem, void *stream);



		cudaError_t DeviceSynchronize(void);
		cudaError_t DeviceReset(void);
		cudaError_t DeviceGetAttribute(int* pi, cudaDeviceAttr attr, int deviceId);
    cudaError_t SetDeviceFlags(unsigned int flags);



		cudaError_t SetDevice(int deviceId);
		cudaError_t GetDevice(int* device);
		cudaError_t GetDeviceCount(int* count);
		cudaError_t GetDeviceProperties(cudaDeviceProp* prop, int device);

		cudaError_t FuncGetAttributes( cudaFuncAttributes *attr, const void *func);
    cudaError_t FuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value);

		const char* GetErrorName(cudaError_t cudaError);
		const char* GetErrorString(cudaError_t cudaError);
		cudaError_t GetLastError(void);


		cudaError_t StreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
		cudaError_t StreamDestroy(cudaStream_t stream);
		cudaError_t StreamSynchronize(cudaStream_t stream);
		cudaError_t StreamWaitEvent(cudaStream_t stream, cudaEvent_t _event, unsigned int flags); 


		cudaError_t EventCreateWithFlags(cudaEvent_t* event, unsigned flags);
		cudaError_t EventRecord(cudaEvent_t _event, cudaStream_t stream);
		cudaError_t EventDestroy(cudaEvent_t _event);
		cudaError_t EventSynchronize(cudaEvent_t event);
		cudaError_t EventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop);
		cudaError_t EventQuery(cudaEvent_t event);

		cudaError_t Free(void* devPtr);
		cudaError_t Malloc(void** devPtr, size_t size);

		cudaError_t MallocHost(void** ptr, size_t size);
		cudaError_t HostAlloc(void** ptr, size_t size, unsigned int flags);
		cudaError_t FreeHost(void* ptr);

		cudaError_t MallocManaged(void** devPtr, size_t size,
				unsigned int flags/* = cudaMemAttachGlobal*/);
		cudaError_t MemPrefetchAsync(const void* devPtr, size_t count,
				int dstDevice, cudaStream_t stream/* = 0*/);

		cudaError_t Memcpy(void* dst, const void* src, size_t count,
				cudaMemcpyKind kind, cudaStream_t stream, bool async);
    cudaError_t MemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
        const void *src, size_t count, enum cudaMemcpyKind kind,
        cudaStream_t stream, bool async);
		cudaError_t MemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
				cudaMemcpyKind kind, cudaStream_t stream, bool async);
		cudaError_t MemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
				cudaMemcpyKind kind, cudaStream_t stream, bool async);

		cudaError_t Memset(void* dst, int value, size_t sizeBytes, 
				cudaStream_t _stream, bool async); 

		cudaError_t MemGetInfo(size_t* free_mem, size_t* total_mem);
    cudaError_t HostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);

    cudaError_t Malloc3DArray(cudaArray_t* array,
        const cudaChannelFormatDesc* desc, cudaExtent extent,
        unsigned int flags);
    cudaError_t MallocArray(cudaArray_t* array,
        const cudaChannelFormatDesc* desc, size_t width, size_t height,
        unsigned int flags);
    cudaError_t FreeArray(cudaArray_t array);
    cudaError_t Memcpy2DToArray(cudaArray_t dst, size_t wOffset,
        size_t hOffset, const void* src, size_t spitch, size_t width,
        size_t height, cudaMemcpyKind kind, cudaStream_t stream, bool async);
    cudaError_t Memcpy3D(const cudaMemcpy3DParms* p, cudaStream_t stream,
        bool async);
    cudaError_t BindTextureToArray(const struct textureReference *texref,
        cudaArray_const_t array, const struct cudaChannelFormatDesc *desc);


		cudaError_t LaunchKernel( const void* func, 
				dim3 gridDim, dim3 blockDim, 
				void** args, size_t sharedMem, 
				cudaStream_t stream );

		void LaunchKernel(const char* kernelName, 
				dim3 numBlocks3D, dim3 blockDim3D, 
				size_t sharedMem, cudaStream_t stream, 
				int numArgs, va_list args);
		//void LaunchKernel(const char* kernelName, dim3 numBlocks3D,
		//    dim3 blockDim3D, size_t sharedMem, cudaStream_t stream, int numArgs, ...);

		cudaChannelFormatDesc CreateChannelDesc ( int  x, int  y, 
		int  z, int  w, cudaChannelFormatKind f );
    cudaError_t GetChannelDesc(cudaChannelFormatDesc* desc,
        cudaArray_const_t array);

		cudaError_t BindTexture(size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size);
		cudaError_t UnbindTexture( const textureReference* texref );





		////CUDA Driver APIs
		CUresult DeviceGet(CUdevice* device, int ordinal);
		CUresult DeviceGetAttribute_drv(int* pi, CUdevice_attribute attrib, CUdevice dev);
		CUresult DeviceGetCount(int* count);
		CUresult DeviceGetName(char* name, int len, CUdevice dev);
		CUresult DeviceTotalMem(size_t* bytes, CUdevice dev);

		CUresult DeviceComputeCapability(int* major, int* minor, CUdevice dev);

};

#endif // __HUM_CUDAWRAPPER_H__
