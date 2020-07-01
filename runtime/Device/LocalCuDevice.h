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

#ifndef __HUM_LOCAL_CUDA_DEVICE_H__
#define __HUM_LOCAL_CUDA_DEVICE_H__

#include "NVLibs.h"
#include "HUMDevice.h"
#include "HUMKernel.h"

class HUMCommand;
class HUMKernel;
class HUMMem;
class CudaProgramBinary;

class LocalCuDevice: public HUMDevice {
	public:
		LocalCuDevice(HUMPlatform* platform, NVLibs* dispatch, int cu_device_id, CudaProgramBinary* cu_program);
		virtual ~LocalCuDevice();

		void SetupDevInfo();

		virtual bool IsVirtual() { return false; }

		virtual void LaunchKernel(HUMCommand* command, HUMKernel* kernel,
				hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
				std::map<hum_uint, HUMKernelArg*>* kernel_args);
		virtual void LaunchCudaKernel(HUMCommand* command, const char* kernel_name, const void* kernel_func,
				hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
				std::map<hum_uint, HUMKernelArg*>* kernel_args);
		virtual void LaunchCudaDirectKernel(HUMCommand* command, const char* kernel_name, const void* kernel_func,
				hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
				std::map<hum_uint, HUMKernelArg*>* kernel_args);

		virtual void ReadBuffer(HUMCommand* command, HUMMem* mem_src, size_t off_src,
				size_t size, void* ptr);
		virtual void WriteBuffer(HUMCommand* command, HUMMem* mem_dst, size_t off_dst,
				size_t size, void* ptr, bool protect_src);
		virtual void WriteBufferToSymbol(HUMCommand* command, const void* symbol, size_t off_dst,
				size_t size, void* ptr);
    virtual void ReadBufferFromSymbol(HUMCommand* command, const void* symbol, size_t off_dst,
        size_t size, void* ptr);
		virtual void CopyBuffer(HUMCommand* command, HUMMem* mem_src, HUMMem* mem_dst,
				size_t off_src, size_t off_dst, size_t size);
		virtual void FillBuffer(HUMCommand* command, HUMMem* mem_dst, void* pattern,
				size_t pattern_size, size_t off_dst,
				size_t size);

		virtual void ExecuteFunc(HUMCommand* command, hum_int func_type, void* data, size_t data_size);

		static void CreateDevices();

		virtual HUMMicroMem* AllocMem(HUMMem* mem);
		virtual void FreeMem(HUMMicroMem* umem);

		virtual void* AllocKernel(HUMKernel* kernel);
		virtual void FreeKernel(HUMKernel* kernel, void* dev_specific);

		NVLibs* dispatch() const { return dispatch_; }

		//For cuda functions
		void InitModule(void **fatCubinHandle);
		void RegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
				const char *deviceName, int thread_limit, uint3 *tid,
				uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);
		void ** RegisterFatBinary(void * fatCubin);
		void UnregisterFatBinary(void **fatCubinHandle);
		cudaError_t PopCallConfiguration(dim3 *gridDim,
				dim3 *blockDim,  size_t *sharedMem, void *stream);
		unsigned PushCallConfiguration(dim3 gridDim,
				dim3 blockDim, size_t sharedMem, void *stream);

		CUstream GetMemStream() { return mem_stream_; }

		void SetNumDevices(int num_devices) { num_devices_ = num_devices; }
		int GetNumDevices(void);
		int GetNumNodes(void);

	private:
		NVLibs* dispatch_;

		CUdevice cu_device_;
		CUcontext cu_context_;
		CUmodule cu_module_;

		CUstream kernel_stream_;
		CUstream mem_stream_;
		CUstream misc_stream_;

		int version_;
		int num_devices_;
		
		cudaDeviceProp cuda_prop_;
		char error_string_[1024];

		CudaProgramBinary* cu_program_;

		std::map<std::string, CUfunction> function_map_;
    
#ifdef USE_MEM_PREFETCH
    bool threads_running_;
    void Run();
    static void* ThreadFunc(void* argp);

    pthread_mutex_t control_block_list_mutex_;
    std::list<memcpy_control_block_t*> control_block_list_;

    pthread_mutex_t memcpy_mutex_[NUM_MEMCPY_THREAD];
    pthread_t memcpy_thread_[NUM_MEMCPY_THREAD];
    sem_t sem_memcpy_[NUM_MEMCPY_THREAD];
#endif
};

#endif //__HUM_LOCAL_CUDA_DEVICE_H__
