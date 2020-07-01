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

#ifndef __HUM_DEVICE_H__
#define __HUM_DEVICE_H__

#include "HUMObject.h"
#include "Utils.h"
#include <semaphore.h>
#include <map>
#include <vector>
#include "HUMMem.h"
#include "HUMKernel.h"

#define HUM_MODEL_TYPE_NONE 0
#define HUM_MODEL_TYPE_OPENCL 1
#define HUM_MODEL_TYPE_CUDA 2

#ifdef DISABLE_MEMCPY_SCHEDULING
#define NUM_MEMCPY_THREAD 1
#else
#define NUM_MEMCPY_THREAD 8
#endif

class HUMPlatform;
class HUMCommand;
class HUMCommandQueue;
class HUMKernel;
class HUMMem;
class HUMMicroMem;
class HUMScheduler;
class HUMIssuer;
class HUMEvent;

struct cuda_dev_info_t
{
	char name[256];
	char	  uuid[16];                       /**< 16-byte unique identifier */
	char         luid[8];                    /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
	unsigned int luidDeviceNodeMask;         /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */

	size_t totalGlobalMem;
	size_t sharedMemPerBlock;
	int regsPerBlock;
	int warpSize;
	size_t memPitch;
	int maxThreadsPerBlock;
	int maxThreadsDim[3];
	int maxGridSize[3];
	int clockRate;
	size_t totalConstMem;
	int major;
	int minor;
	size_t textureAlignment;
	size_t texturePitchAlignment;
	int deviceOverlap;
	int multiProcessorCount;
	int kernelExecTimeoutEnabled;
	int integrated;
	int canMapHostMemory;
	int computeMode;
	int maxTexture1D;
	int maxTexture1DMipmap;
	int maxTexture1DLinear;
	int maxTexture2D[2];
	int maxTexture2DMipmap[2];
	int maxTexture2DLinear[3];
	int maxTexture2DGather[2];
	int maxTexture3D[3];
	int maxTexture3DAlt[3];
	int maxTextureCubemap;
	int maxTexture1DLayered[2];
	int maxTexture2DLayered[3];
	int maxTextureCubemapLayered[2];
	int maxSurface1D;
	int maxSurface2D[2];
	int maxSurface3D[3];
	int maxSurface1DLayered[2];
	int maxSurface2DLayered[3];
	int maxSurfaceCubemap;
	int maxSurfaceCubemapLayered[2];
	size_t surfaceAlignment;
	int concurrentKernels;
	int ECCEnabled;
	int pciBusID;
	int pciDeviceID;
	int pciDomainID;
	int tccDriver;
	int asyncEngineCount;
	int unifiedAddressing;
	int memoryClockRate;
	int memoryBusWidth;
	int l2CacheSize;
	int maxThreadsPerMultiProcessor;
	int streamPrioritiesSupported;
	int globalL1CacheSupported;
	int localL1CacheSupported;
	size_t sharedMemPerMultiprocessor;
	int regsPerMultiprocessor;
	int managedMemSupported;
	int isMultiGpuBoard;
	int multiGpuBoardGroupID;
	int singleToDoublePrecisionPerfRatio;
	int pageableMemoryAccess;
	int concurrentManagedAccess;
};


class HUMDevice: public HUMObject<HUMDevice>
{
	public:
		HUMDevice(HUMPlatform* platform, int node_id, bool add_device_to_platform = true);
		virtual ~HUMDevice();
		virtual void Cleanup();

		hum_device_type type() const { return device_type_; }
		hum_int model() const { return model_type_; }
		int node_id() const { return node_id_; }
		HUMPlatform* platform() const { return platform_; }

		virtual bool IsVirtual() { return true; }

		hum_int GetDeviceInfo(hum_device_info param_name, size_t param_value_size,
                       void* param_value, size_t* param_value_size_ret);

  
		bool IsAvailable() const { return available_; }
		bool IsCudaAvailable() const { return model_type_ == HUM_MODEL_TYPE_CUDA; }
		
		void AddCommandQueue(HUMCommandQueue* queue);
		void RemoveCommandQueue(HUMCommandQueue* queue);
		size_t GetNumCommandQueues();
		HUMCommandQueue* GetCommandQueue(int idx);

		void InvokeScheduler();

		void EnqueueReadyQueue(HUMCommand* command);
		HUMCommand* DequeueReadyQueue();
		void InvokeReadyQueue();
		void WaitReadyQueue();

		virtual void LaunchKernel(HUMCommand* command, HUMKernel* kernel,
				hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
				std::map<hum_uint, HUMKernelArg*>* kernel_args) = 0;
		virtual void LaunchCudaKernel(HUMCommand* command, const char* kernel_name, const void* kernel_func,
				hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
				std::map<hum_uint, HUMKernelArg*>* kernel_args) = 0;
		virtual void LaunchCudaDirectKernel(HUMCommand* command, const char* kernel_name, const void* kernel_func,
				hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
				std::map<hum_uint, HUMKernelArg*>* kernel_args) { assert(0); };

		virtual void ReadBuffer(HUMCommand* command, HUMMem* mem_src, size_t off_src,
				size_t size, void* ptr) = 0;
		virtual void ReadBufferFromLocalGPU(HUMCommand* command, HUMMem* mem_src, size_t off_src,
                          size_t size, void* ptr, int target_worker) { assert(0); }

		virtual void WriteBuffer(HUMCommand* command, HUMMem* mem_dst, size_t off_dst,
				size_t size, void* ptr, bool protect_src) = 0;
		virtual void WriteBufferToSymbol(HUMCommand* command, const void* symbol, size_t off_dst,
				size_t size, void* ptr) = 0;
		virtual void WriteBufferToLocalGPU(HUMCommand* command, HUMMem* mem_dst, size_t off_dst,
                           size_t size, void* ptr, int target_worker) { assert(0); };
    virtual void ReadBufferFromSymbol(HUMCommand* command, const void* symbol, size_t off_dst,
        size_t size, void* ptr) = 0;

		virtual void BroadcastBuffer(HUMCommand* command, HUMMem* mem_dst, size_t off_dst,
				size_t size, void* ptr) { assert(0); };

		virtual void CopyBuffer(HUMCommand* command, HUMMem* mem_src, HUMMem* mem_dst,
				size_t off_src, size_t off_dst, size_t size) = 0;
		virtual void CopyBroadcastBuffer(HUMCommand* command, HUMMem* mem_src, HUMMem* mem_dst,
				size_t off_src, size_t off_dst, size_t size) { assert(0); };
		virtual void FillBuffer(HUMCommand* command, HUMMem* mem_dst, void* pattern,
				size_t pattern_size, size_t off_dst,
				size_t size) = 0;

		virtual void ExecuteFunc(HUMCommand* command, hum_int func_type, void* data, size_t data_size) = 0; 

		int GetDistance(HUMDevice* other) const;

		virtual HUMMicroMem* AllocMem(HUMMem* mem) = 0;
		virtual void FreeMem(HUMMicroMem* mem) = 0;

		virtual void* AllocKernel(HUMKernel* kernel);
		virtual void FreeKernel(HUMKernel* kernel, void* dev_specific);

		virtual bool IsComplete(HUMCommand* command);

		const cuda_dev_info_t* cuda_dev_info() const { return &cuda_dev_info_; }

		hum_uint GetDeviceID() const { return device_id_; }

#if defined(USE_MEM_PREFETCH)
		std::list<memcpy_command_t*> memcpy_list[NUM_MEMCPY_THREAD];
#endif

	protected:
		int node_id_;
		hum_uint device_id_;	//device_id in a node

		hum_device_type device_type_;

		HUMPlatform* platform_;
		HUMScheduler* scheduler_;
		sem_t sem_ready_queue_;

		LockFreeQueueMS<HUMCommand> ready_queue_;
	
		hum_bool available_;	
		struct cuda_dev_info_t cuda_dev_info_;

		hum_int model_type_;
};

#endif //__HUM_DEVICE_H__
