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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/time.h>
#include "LocalCuDevice.h"
#include "HUMPlatform.h"
#include "HUMMem.h"
#include "HUMCommand.h"
#include "HUMProgram.h"
#include "MemoryRegion.h"
#include "cuda_func.h"
#include "ioctl.h"


extern int gsize_;
extern int grank_;
extern HUMComm* g_HUMComm;
extern std::map<const void*, std::string> g_cuda_func_map_h2d;

using namespace std;

#define MAX(x, y) ((x) < (y) ? (y) : (x))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define CHECK_DEV_ERROR(cond, err)                                             \
  if (cond) {                                                                  \
    cuGetErrorString(err, (const char**)&error_string_);                       \
    HUM_ERROR("CUDA driver error: %s (%d)", error_string_, err);              \
    assert(0);                                                                 \
    return;                                                                    \
  }

#define UPDATE_DEV_ERROR(err)                                                  \
  if (err != CUDA_SUCCESS) {                                                   \
    cuGetErrorString(err, (const char**)&error_string_);                       \
    HUM_ERROR("CUDA driver error: %s (%d)", error_string_, err);              \
    assert(0);                                                                 \
    return;                                                                    \
  }

#define CHECK_ERROR(cond, err)                                                 \
  if (cond) {                                                                  \
    HUM_ERROR("Runtime error: %d", err);                                      \
    assert(0);                                                                 \
    command->SetError(err);                                                    \
    return;                                                                    \
  }

#define UPDATE_ERROR(err)                                                      \
  if (err != cudaSuccess) {                                                    \
    HUM_ERROR("Runtime error: %d", err);                                      \
    assert(0);                                                                 \
    command->SetError(err);                                                    \
    return;                                                                    \
  }

#include <iostream>
#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << " at " << __FILE__ << "::" << __LINE__ << std::endl; \
  assert(0); \
  } \
}

void LocalCuDevice::CreateDevices()
{
  HUM_DEV("%s", "CreateDevices begin");

  char* error_string;
  CUresult error = cuInit(0);

  struct stat fstat;
  if (stat("/proc/driver/nvidia/version", &fstat)) {
    // this node does not have nvidia driver, just pass it
    cuGetErrorString(error, (const char**)&error_string);
    fprintf(stderr, "CUDA driver is not installed: %s (%d)\n", error_string, error);
    return;
  }

  if (error != CUDA_SUCCESS) {
    cuGetErrorString(error, (const char**)&error_string);
    fprintf(stderr, "Failed to init CUDA driver API: %s (%d)\n", error_string, error);
    return;
  }

  HUMPlatform* platform = HUMPlatform::GetPlatform();
  NVLibs* nv_libs = new NVLibs();
  assert(nv_libs != NULL);

  int num_devices = 0;

  error = nv_libs->cuDeviceGetCount(&num_devices);
  //printf("error = %d", error);
  if (error != CUDA_SUCCESS) {
    cuGetErrorString(error, (const char**)&error_string);
    fprintf(stderr, "Failed to get device count: %s (%d)\n", error_string, error);
    return;
  }

  //if (num_devices > 2) num_devices = 2;
  HUM_DEV("cuda device cnt = %d", num_devices);

  //CudaProgramBinary* cuProgram = new CudaProgramBinary("kernels.cubin"); 
  CudaProgramBinary* cuProgram = new CudaProgramBinary("kernels.ptx"); 
  //CudaProgramBinary* cuProgram = NULL; 
  for (hum_uint i = 0; i < num_devices; ++i) {
    LocalCuDevice* device = new LocalCuDevice(platform, nv_libs, i, cuProgram);
    device->SetNumDevices(num_devices);
  }

  for (hum_uint i = 0; i < num_devices; ++i) {
    for (hum_uint j = 0; j < num_devices; ++j) {
      if (i == j) continue;

      int canpeer = 0;
      nv_libs->cudaSetDevice(i);
      nv_libs->cudaDeviceSynchronize();

      nv_libs->cudaDeviceCanAccessPeer(&canpeer, i, j);

      if (canpeer)
        nv_libs->cudaDeviceEnablePeerAccess(j, 0);
    }
  }

  HUM_DEV("%s", "CreateDevices end");
}

LocalCuDevice::LocalCuDevice(HUMPlatform* platform, NVLibs* dispatch, int cu_device_id, CudaProgramBinary* cu_program)
:HUMDevice(platform, grank_, false)
{
  HUM_DEV("LocalCuDevice %p is creating...", this);

  dispatch_ = dispatch;
  device_id_ = cu_device_id;
  cu_program_ = cu_program;

  CUresult error = dispatch_->cuDeviceGet(&cu_device_, device_id_);
  UPDATE_DEV_ERROR(error);

  error = dispatch_->cuDevicePrimaryCtxRetain(&cu_context_, cu_device_);
  UPDATE_DEV_ERROR(error);

  dispatch_->cudaGetDeviceProperties(&cuda_prop_, device_id_);

  memcpy(&cuda_dev_info_, &cuda_prop_, sizeof(cuda_dev_info_));

  dispatch_->cuCtxSetCurrent(cu_context_);

  if (cu_program_->size() > 0) {
    error = cuModuleLoadData(&cu_module_, cu_program->binary());
    //error = cuModuleLoad(&cu_module_, "kernels.ptx");
    UPDATE_DEV_ERROR(error);

    HUM_DEV("load module from %s for dev_id(%d)", "kernels.ptx", device_id_);
  }
  else {
    HUM_DEV("Dose not load module for dev_id(%d)", device_id_);
    //assert(0);
  }
  cudaError_t err = dispatch_->cudaSetDevice(device_id_);
  //UPDATE_DEV_ERROR(error);

  error = dispatch_->cuStreamCreate(&kernel_stream_, CU_STREAM_NON_BLOCKING);
  UPDATE_DEV_ERROR(error);
  error = dispatch_->cuStreamCreate(&mem_stream_, CU_STREAM_NON_BLOCKING);
  UPDATE_DEV_ERROR(error);
  error = dispatch_->cuStreamCreate(&misc_stream_, CU_STREAM_NON_BLOCKING);
  UPDATE_DEV_ERROR(error);

  SetupDevInfo();

  platform_->AddDevice(this);
  
#ifdef USE_MEM_PREFETCH
  threads_running_ = true;

  pthread_mutex_init(&control_block_list_mutex_, NULL);

  for (unsigned int i = 0; i < NUM_MEMCPY_THREAD; ++i) {
    sem_init(&sem_memcpy_[i], 0, 0);
    pthread_mutex_init(&memcpy_mutex_[i], NULL);
    pthread_create(&memcpy_thread_[i], NULL, &LocalCuDevice::ThreadFunc, this);
  }
#endif
}

LocalCuDevice::~LocalCuDevice()
{
  dispatch_->cuCtxSetCurrent(cu_context_);
  cuModuleUnload(cu_module_);
  dispatch_->cuDevicePrimaryCtxRelease(cu_device_);

  dispatch_->cuStreamDestroy(kernel_stream_);
  dispatch_->cuStreamDestroy(mem_stream_);
  dispatch_->cuStreamDestroy(misc_stream_);

#ifdef USE_MEM_PREFETCH
  threads_running_ = false;

  pthread_mutex_destroy(&control_block_list_mutex_);
  for (unsigned int i = 0; i < NUM_MEMCPY_THREAD; ++i) {
    sem_post(&sem_memcpy_[i]);
    pthread_join(memcpy_thread_[i], NULL);
    pthread_mutex_destroy(&memcpy_mutex_[i]);
    sem_destroy(&sem_memcpy_[i]);
  }
#endif
}

void LocalCuDevice::SetupDevInfo() 
{
  available_ = true;
  device_type_ = HUM_DEVICE_TYPE_GPU;
  model_type_ = HUM_MODEL_TYPE_CUDA;
}

void LocalCuDevice::LaunchKernel(HUMCommand* command, HUMKernel* kernel,
                                hum_uint work_dim, size_t gwo[3], size_t gws[3],
                                size_t lws[3], size_t nwg[3],
                                std::map<hum_uint, HUMKernelArg*>* kernel_args) 
{
  assert(0);
}

void LocalCuDevice::LaunchCudaKernel(HUMCommand* command, const char* kernel_name, const void* kernel_func,
    hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
    std::map<hum_uint, HUMKernelArg*>* kernel_args)
{
  HUMKernelCommand* kcommand = (HUMKernelCommand*)command;
  const char* funcname = NULL;

  cudaError_t err = dispatch_->cudaSetDevice(device_id_);
  if(err != 0) {
    HUM_ERROR("cudaSetDevice(%d) Failed", device_id_);
    assert(0);
  }


  if(kernel_func == NULL) {
    assert(0); //TODO;
    CUresult error;

    CUfunction kernel_function;
    std::string kernel_str(kernel_name);
    std::map<std::string, CUfunction>::iterator Itr;
    Itr = function_map_.find(kernel_str);

    // no function found, make new one
    if (Itr == function_map_.end()) {
      error = cuModuleGetFunction(&kernel_function, cu_module_, kernel_name);
      if (error != CUDA_SUCCESS) {
        cuGetErrorString(error, (const char**)&error_string_);
        HUM_ERROR("Failed to get function \"%s\" from module in device %d: %s (%d)",
            kernel_name, device_id_, error_string_, error);
        exit(1);
        return;
      }
      function_map_[kernel_str] = kernel_function;
    }
    else {
      kernel_function = Itr->second;
    }

    // copy kernel arguments to kernel arg region
    size_t num_args = (*kernel_args).size();
    void* args[num_args];
    uint64_t arg_value[num_args];

    for (int i = 0; i < num_args; ++i) {
      HUMKernelArg* karg = (*kernel_args)[i];

      if(karg->mem != NULL) {
        HUMMem* mem = karg->mem;
        arg_value[i] = ((uint64_t)mem->GetDevSpecific(this) + karg->cuda_offset);
        args[i] = (void*)&arg_value[i];

        HUM_DEV("arg_value[%d] = %p (base=%p, offset=%ld)", i, mem->GetDevSpecific(this), arg_value[i], karg->cuda_offset); 
      }
      else {
        arg_value[i] = (size_t)(karg->value);
        //memcpy(&arg_value[i], karg->value, karg->size);
        args[i] = (void*)arg_value[i];

        HUM_DEV("arg_value[%d] = %p", i, arg_value[i]); 
      }
    }

    size_t shared_mem = ((HUMKernelCommand*)command)->cuda_shared_mem_;

    HUM_DEV("gws_(%d, %d, %d) : lws_(%d,%d,%d) : nwg_(%d,%d,%d)", 
        gws[0], gws[1], gws[2],
        lws[0], lws[1], lws[2],
        nwg[0], nwg[1], nwg[2]);

    error = cuLaunchKernel(kernel_function,
        nwg[0], nwg[1], nwg[2],
        lws[0], lws[1], lws[2],
        shared_mem, kernel_stream_, args, NULL);
    if (error != CUDA_SUCCESS) {
      cuGetErrorString(error, (const char**)&error_string_);
      HUM_ERROR("Failed to launch kernel in device %d: %s (%d)",
          device_id_, error_string_, error);
      exit(1);
      return;
    }
  }
  else {
    // copy kernel arguments to kernel arg region
    size_t num_args = (*kernel_args).size();
    HUM_DEV("num_args = %d", num_args);
    void* args[num_args];
    uint64_t arg_value[num_args];

    HUM_DEV("LaunchCudaKernel start numargs=%d", num_args);

    for (int i = 0; i < num_args; ++i) {
      HUMKernelArg* karg = (*kernel_args)[i];

      if(karg->mem != NULL) {
        HUMMem* mem = karg->mem;
        arg_value[i] = ((uint64_t)mem->GetDevSpecific(this) + karg->cuda_offset);
        args[i] = (void*)&arg_value[i];

        HUM_DEV("arg_value[%d] = %p (base=%p, offset=%ld, mem_size=%ld)",
            i, mem->GetDevSpecific(this), arg_value[i], karg->cuda_offset, mem->size()); 
      }
      else {
        karg->BuildValue(this);
        arg_value[i] = (size_t)(karg->value);
        //memcpy(&arg_value[i], karg->value, karg->size);
        args[i] = (void*)arg_value[i];

        HUM_DEV("arg_value[%d] = %p size=%ld", i, arg_value[i], karg->size); 
      }
    }
  
    size_t shared_mem = ((HUMKernelCommand*)command)->cuda_shared_mem_;

    HUM_DEV("gws_(%d, %d, %d) : lws_(%d,%d,%d) : nwg_(%d,%d,%d)", 
        gws[0], gws[1], gws[2],
        lws[0], lws[1], lws[2],
        nwg[0], nwg[1], nwg[2]);
  
    dim3 gridDim;  
    dim3 blockDim;
    gridDim.x = nwg[0]; gridDim.y = nwg[1]; gridDim.z = nwg[2];
    blockDim.x = lws[0]; blockDim.y = lws[1]; blockDim.z = lws[2];

    cudaError_t error = dispatch()->cudaLaunchKernel(kernel_func,
        gridDim, blockDim,
        args, shared_mem, kernel_stream_);
    if (error != 0) {
      HUM_ERROR("Failed to launch kernel in device %d for kerenl_func=%p: err=%d",
          device_id_, kernel_func, error);
      assert(0);
      exit(1);
      return;
    }
    else {
      HUM_DEV("Success to launch kernel in device %d for kerenl_func=%p",
          device_id_, kernel_func);
    }
    
  }

  // wait for kernel to finish
  err = dispatch_->cudaStreamSynchronize(kernel_stream_);
  UPDATE_ERROR(err);
}

void LocalCuDevice::LaunchCudaDirectKernel(HUMCommand* command, const char* kernel_name, const void* kernel_func,
    hum_uint work_dim, size_t gwo[3], size_t gws[3], size_t lws[3], size_t nwg[3],
    std::map<hum_uint, HUMKernelArg*>* kernel_args)
{
  HUMKernelCommand* kcommand = (HUMKernelCommand*)command;
  const char* funcname = NULL;

  cudaError_t err = dispatch_->cudaSetDevice(device_id_);
  if(err != 0) {
    HUM_ERROR("cudaSetDevice(%d) Failed", device_id_);
    assert(0);
  }

  {
    funcname = g_cuda_func_map_h2d[kernel_func].c_str();

    // copy kernel arguments to kernel arg region
    size_t num_args = (*kernel_args).size();
    HUM_DEV("num_args = %d", num_args);
    void* args[num_args];
    uint64_t arg_value[num_args];

    HUM_DEV("LaunchCudaDirectKernel start numargs=%d", num_args);

    for (int i = 0; i < num_args; ++i) {
      HUMKernelArg* karg = (*kernel_args)[i];

      if(karg->mem != NULL) {
        assert(0);
        HUMMem* mem = karg->mem;
        arg_value[i] = ((uint64_t)mem->GetDevSpecific(this) + karg->cuda_offset);
        args[i] = (void*)&arg_value[i];

        HUM_DEV("arg_value[%d] = %p (base=%p, offset=%ld, mem_size=%ld)", i, mem->GetDevSpecific(this), arg_value[i], karg->cuda_offset, mem->size()); 
      }
      else {
        karg->BuildValue(this);
        arg_value[i] = (size_t)(karg->value);
        //memcpy(&arg_value[i], karg->value, karg->size);
        args[i] = (void*)arg_value[i];

        HUM_DEV("arg_value[%d] = %p size=%ld", i, arg_value[i], karg->size); 
      }
    }

    size_t shared_mem = ((HUMKernelCommand*)command)->cuda_shared_mem_;

    HUM_DEV("gws_(%d, %d, %d) : lws_(%d,%d,%d) : nwg_(%d,%d,%d)", 
        gws[0], gws[1], gws[2],
        lws[0], lws[1], lws[2],
        nwg[0], nwg[1], nwg[2]);
  
    dim3 gridDim;  
    dim3 blockDim;
    gridDim.x = nwg[0]; gridDim.y = nwg[1]; gridDim.z = nwg[2];
    blockDim.x = lws[0]; blockDim.y = lws[1]; blockDim.z = lws[2];

    cudaError_t error = dispatch()->cudaLaunchKernel(kernel_func,
        gridDim, blockDim,
        args, shared_mem, kernel_stream_);
    if (error != 0) {
      HUM_ERROR("Failed to direct launch kernel in device %d for kerenl_func=%p: err=%d",
          device_id_, kernel_func, error);
      assert(0);
      exit(1);
      return;
    }
    else {
      HUM_DEV("Success to direct launch kernel in device %d for kerenl_func=%p",
          device_id_, kernel_func);
    }
    
  }

  HUM_DEV("%s direct kernel launch done in dev_id(%d)", funcname, device_id_);
  // wait for kernel to finish
  err = dispatch_->cudaStreamSynchronize(kernel_stream_);
  UPDATE_ERROR(err);
  HUM_DEV("%s direct kernel execution done in dev_id(%d)", funcname, device_id_);
}



void LocalCuDevice::ReadBuffer(HUMCommand* command, HUMMem* mem_src,
                              size_t off_src, size_t size, void* ptr) 
{
   cudaError_t err;

  CHECK_ERROR(available_ == HUM_FALSE, HUM_DEVICE_NOT_AVAILABLE);

  err = dispatch_->cudaSetDevice(device_id_);
  UPDATE_ERROR(err);

  void* mem_src_dev = (void*)mem_src->GetDevSpecific(this);
   HUM_DEV("node_id=%d, dev=%p, ReadBuffer mem_src=%p, mem_src_dev=%p",
      node_id(), this, mem_src, mem_src_dev);
 
  
  CHECK_ERROR(mem_src_dev == NULL, HUM_INVALID_MEM_OBJECT);
  mem_src_dev = (void*)((size_t)mem_src_dev + off_src);

  err = dispatch_->cudaMemcpyAsync(ptr, mem_src_dev, size, cudaMemcpyDeviceToHost, mem_stream_);
  UPDATE_ERROR(err);
//  err = dispatch_->cudaMemPrefetchAsync(mem_src_dev, size, device_id_, mem_stream_);
//  UPDATE_ERROR(err);

  // wait for memcpy to finish
  err = dispatch_->cudaStreamSynchronize(mem_stream_);
  UPDATE_ERROR(err);
}

void LocalCuDevice::WriteBuffer(HUMCommand* command, HUMMem* mem_dst,
                               size_t off_dst, size_t size, void* ptr, bool protect_src) 
{
  HUM_DEV("WriteBuffer begin off_dst=%ld, size=%ld, mem_size=%ld", off_dst, size, mem_dst->size());
  assert(off_dst+size <= mem_dst->size());
  cudaError_t err;

  CHECK_ERROR(available_ == HUM_FALSE, HUM_DEVICE_NOT_AVAILABLE);
  err = dispatch_->cudaSetDevice(device_id_);
  UPDATE_ERROR(err);


  void* mem_dst_dev = (void*)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_dst_dev == NULL, HUM_INVALID_MEM_OBJECT);
  mem_dst_dev = (void*)((size_t)mem_dst_dev + off_dst);
  HUM_DEV("node_id=%d, dev=%p, WriteBuffer mem_dst_dev=%p",
      node_id(), this, mem_dst_dev);

#if defined(USE_MEM_PREFETCH)
  {
    // information for calculating page-aligned user pointer
    size_t align_mod = (uint64_t)ptr % 4096;
    size_t aligned_size = size + align_mod;
    // TODO
    if (aligned_size > 4096)
      aligned_size -= aligned_size % 4096;
    void* aligned_ptr = (void*)((uint64_t)ptr - align_mod);

    memcpy_control_block_t *control_block;
    control_block = new memcpy_control_block_t;
    control_block->mprotect_start = aligned_ptr;
    control_block->mprotect_size = aligned_size;
    control_block->mprotected = false;
    control_block->num_chunks = 0;
    control_block->finished_chunks = 0;
    pthread_mutex_init(&control_block->block_mutex, NULL);

    bool wait_for_pred = false;
    uint64_t this_start = (uint64_t)mem_dst_dev;
    uint64_t this_end = (uint64_t)mem_dst_dev + size;
    std::list<memcpy_control_block_t*>::iterator LI;
    std::list<memcpy_control_block_t*>::iterator LE;
    memcpy_control_block_t *pred_block;

    // fill in more info
    control_block->mem_start_dev = this_start;
    control_block->mem_end_dev = this_end;
    control_block->ref_count = 1;

    // check dendencies between commands that are in-flight
    pthread_mutex_lock(&control_block_list_mutex_);
    for (LI = control_block_list_.begin(), LE = control_block_list_.end();
        LI != LE; ++LI) {
      pred_block = *LI;
      uint64_t pred_start = pred_block->mem_start_dev;
      uint64_t pred_end = pred_block->mem_end_dev;

      if (!((pred_end <= this_start) || (pred_start >= this_end))) {
        printf("pred_start: %lx, pred_end: %lx\n", pred_start, pred_end);
        printf("this_start: %lx, this_end: %lx\n", this_start, this_end);
        wait_for_pred = true;
        pred_block->ref_count++;
        break;
      }
    }

    control_block_list_.push_back(control_block);
    pthread_mutex_unlock(&control_block_list_mutex_);

    if (wait_for_pred == true) {
      while (pred_block->ref_count != 1) {
        // busy-wait
//        printf("Waiting for dependence command!!!!\n");
      }

      // delete control block cos memcpy thread does not need it anymore
      // i.e. whole memcpy is finished
      pthread_mutex_destroy(&control_block->block_mutex);
      delete pred_block;

      pthread_mutex_lock(&control_block_list_mutex_);
      control_block_list_.erase(LI);
      pthread_mutex_unlock(&control_block_list_mutex_);
    }

    // unmap from GPU to trigger page fault when kernel is executed
    struct map_command map_cmd;
    map_cmd.mem_start = (uint64_t)mem_dst_dev;
    map_cmd.mem_length = size;
    map_cmd.gpu_id = device_id_;
    ioctl(driver_fd_, IOCTL_UNMAP_FROM_GPU, &map_cmd);

    // make memcpy commands and enqueue to memcpy list
    HUMUnifiedMem* umbuf = (HUMUnifiedMem*)mem_dst;
    size_t block_per_thread =
      ceil((double)size / NUM_MEMCPY_THREAD / BLOCK_SIZE);
    size_t left_size = size;
    size_t curr_off = off_dst;
    uint64_t next_user = (uint64_t)ptr;
    uint64_t next_off = off_dst;

    for (unsigned int i = 0; i < NUM_MEMCPY_THREAD; ++i) {
      size_t copy_size = 0;
      size_t tmp_size;

      for (unsigned int j = 0; j < block_per_thread; ++j) {
        tmp_size = BLOCK_SIZE - (curr_off % BLOCK_SIZE);
        tmp_size = tmp_size > left_size ? left_size : tmp_size;

        copy_size += tmp_size;
        left_size -= tmp_size;
        curr_off += tmp_size;
      }

      if (copy_size == 0)
        continue;

      memcpy_command_t* memcpy_command = new memcpy_command_t;
      memcpy_command->user_src = (void*)next_user;
      memcpy_command->umbuf = umbuf;
      memcpy_command->offset = next_off;
      memcpy_command->size = copy_size;
      memcpy_command->copied_size = 0;
      memcpy_command->control_block = control_block;
      control_block->num_chunks += 1;

      pthread_mutex_lock(&memcpy_mutex_[i]);
      this->memcpy_list[i].push_back(memcpy_command);
      pthread_mutex_unlock(&memcpy_mutex_[i]);
      
      // wake up memcpy thread
      sem_post(&sem_memcpy_[i]);

      next_user += copy_size;
      next_off += copy_size;
    }

    if (left_size != 0) {
      memcpy_command_t* memcpy_command = new memcpy_command_t;
      memcpy_command->user_src = (void*)next_user;
      memcpy_command->umbuf = umbuf;
      memcpy_command->offset = next_off;
      memcpy_command->size = left_size;
      memcpy_command->copied_size = 0;
      memcpy_command->control_block = control_block;
      control_block->num_chunks += 1;

      pthread_mutex_lock(&memcpy_mutex_[0]);
      this->memcpy_list[0].push_back(memcpy_command);
      pthread_mutex_unlock(&memcpy_mutex_[0]);

      sem_post(&sem_memcpy_[0]);
    }
    
    pthread_mutex_lock(&control_block->block_mutex);
    
    if (control_block->num_chunks != control_block->finished_chunks) {
      if (protect_src) {
        // set protection of region to read-only for write check
        if (mprotect(aligned_ptr, aligned_size, PROT_READ | PROT_EXEC)) {
          HUM_ERROR("mprotect Failed errno=%d", errno);
          assert(0);
        }
        control_block->mprotected = true;
      }
      pthread_mutex_unlock(&control_block->block_mutex);
    }
    else {
      // if memcpy finishes before we enter here, we do not need to change
      // memory protection. in that case, just wipe it clean
      pthread_mutex_unlock(&control_block->block_mutex);
      pthread_mutex_destroy(&control_block->block_mutex);
      delete control_block;

      pthread_mutex_lock(&control_block_list_mutex_);
      control_block_list_.erase(LI);
      pthread_mutex_unlock(&control_block_list_mutex_);
    }
  }
#else
  err = dispatch_->cudaMemcpyAsync(mem_dst_dev, ptr, size, cudaMemcpyHostToDevice, mem_stream_);
  UPDATE_ERROR(err);

  //err = dispatch_->cudaMemPrefetchAsync(mem_dst_dev, size, device_id_, mem_stream_);
  //UPDATE_ERROR(err);

  // wait for memcpy to finish
  err = dispatch_->cudaStreamSynchronize(mem_stream_);
  UPDATE_ERROR(err);
#endif
  HUM_DEV("WriteBuffer end off_dst=%ld, size=%ld", off_dst, size);

}

void LocalCuDevice::WriteBufferToSymbol(HUMCommand* command, const void* symbol, size_t off_dst,
        size_t size, void* ptr)
{
  cudaError_t err;

  CHECK_ERROR(available_ == HUM_FALSE, HUM_DEVICE_NOT_AVAILABLE);
  err = dispatch_->cudaSetDevice(device_id_);
  UPDATE_ERROR(err);

  HUM_DEV("node_id=%d, dev=%p, WriteBufferToSymbol symbol=%p",
      node_id(), this, symbol);

  err = dispatch_->cudaMemcpyToSymbolAsync(symbol, ptr, size, off_dst, cudaMemcpyHostToDevice, mem_stream_);
  UPDATE_ERROR(err);

  // wait for memcpy to finish
  err = dispatch_->cudaStreamSynchronize(mem_stream_);
  UPDATE_ERROR(err);
}

void LocalCuDevice::ReadBufferFromSymbol(HUMCommand* command, const void* symbol, size_t off_dst,
        size_t size, void* ptr)
{
  cudaError_t err;

  CHECK_ERROR(available_ == HUM_FALSE, HUM_DEVICE_NOT_AVAILABLE);
  err = dispatch_->cudaSetDevice(device_id_);
  UPDATE_ERROR(err);

  err = dispatch_->cudaMemcpyFromSymbolAsync(ptr, symbol, size, off_dst, cudaMemcpyDeviceToHost, mem_stream_);
  UPDATE_ERROR(err);

  // wait for memcpy to finish
  err = dispatch_->cudaStreamSynchronize(mem_stream_);
  UPDATE_ERROR(err);
}


void LocalCuDevice::CopyBuffer(HUMCommand* command, HUMMem* mem_src,
                              HUMMem* mem_dst, size_t off_src, size_t off_dst,
                              size_t size) 
{
  CHECK_ERROR(available_ == HUM_FALSE, HUM_DEVICE_NOT_AVAILABLE);
   
  cudaError_t err;
  err = dispatch_->cudaSetDevice(device_id_);
  UPDATE_ERROR(err);

  void* mem_src_dev = (void*)mem_src->GetDevSpecific(this);
  CHECK_ERROR(mem_src_dev == NULL, HUM_INVALID_MEM_OBJECT);

   void* mem_dst_dev = (void*)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_dst_dev == NULL, HUM_INVALID_MEM_OBJECT);

  mem_src_dev = (void*)((size_t)mem_src_dev + off_src);
  mem_dst_dev = (void*)((size_t)mem_dst_dev + off_dst);
  
  err = dispatch_->cudaMemcpyAsync(mem_dst_dev, mem_src_dev, size, cudaMemcpyDeviceToDevice, mem_stream_);
  UPDATE_ERROR(err);

  // wait for memcpy to finish
  err = dispatch_->cudaStreamSynchronize(mem_stream_);
  UPDATE_ERROR(err);
}

void LocalCuDevice::FillBuffer(HUMCommand* command, HUMMem* mem_dst, void* pattern,
    size_t pattern_size, size_t off_dst,
    size_t size)
{
  CHECK_ERROR(available_ == HUM_FALSE, HUM_DEVICE_NOT_AVAILABLE);
   
  cudaError_t err;
  err = dispatch_->cudaSetDevice(device_id_);
  UPDATE_ERROR(err);

  void* mem_dst_dev = (void*)mem_dst->GetDevSpecific(this);
  CHECK_ERROR(mem_dst_dev == NULL, HUM_INVALID_MEM_OBJECT);
  mem_dst_dev = (void*)((size_t)mem_dst_dev + off_dst);

  //CHECK_ERROR(pattern_size == 4, HUM_INVALID_ARG_SIZE);
  int value = *(char*)pattern;

  HUM_DEV("node_id=%d, dev=%p, FillBuffer mem_dst_dev=%p",
      node_id(), this, mem_dst_dev);


  err = dispatch_->cudaMemsetAsync(mem_dst_dev, value, size, mem_stream_);
  UPDATE_ERROR(err);
  //err = dispatch_->cudaMemPrefetchAsync(mem_dst_dev, size, device_id_, mem_stream_);
  //UPDATE_ERROR(err);

  // wait for memset to finish
  err = dispatch_->cudaStreamSynchronize(mem_stream_);
  UPDATE_ERROR(err);
}

HUMMicroMem* LocalCuDevice::AllocMem(HUMMem* mem) 
{
  HUM_DEV("Alloc Mem begin for mem(%p)", mem);
  hum_int err = HUM_SUCCESS;
  hum_mem_flags flags = mem->flags() & (HUM_MEM_READ_WRITE | HUM_MEM_READ_ONLY |
                                       HUM_MEM_WRITE_ONLY);
  bool cuda_alloc = false;

  err = dispatch_->cudaSetDevice(device_id_);
  if(err != 0) {
    HUM_ERROR("cudaSetDevice(%d) Failed", device_id_);
    assert(0);
  }

  void* dev_ptr = NULL;
  HUMMicroMem* umem = NULL;

  if(((HUMUnifiedMem*)mem)->dev_ptr_) {
    dev_ptr = ((HUMUnifiedMem*)mem)->dev_ptr_;
  }
  else {
    assert(0);
    err = dispatch_->cudaMallocManaged(&dev_ptr, mem->size(), cudaMemAttachGlobal);
    if(err != 0) {
      HUM_ERROR("cudaMallocManaged(size=%ld) Failed", mem->size());
      assert(0);
    }
    cuda_alloc = true;
  }
  
#ifdef USE_MEM_PREFETCH
  // initial residence of the memory is GPU
  struct map_command cmd;
  cmd.mem_start = (uint64_t)dev_ptr;
  cmd.mem_length = mem->size();
  cmd.gpu_id = device_id_;
  ioctl(driver_fd_, IOCTL_MAP_TO_GPU_WRITE_PROT, &cmd);
#endif

  umem = new HUMMicroMem(mem->context(), this, mem);
  assert(umem != NULL);
  assert(umem->get_obj() == NULL);
  umem->set_obj((void*)dev_ptr);
  umem->Retain();
  umem->cuda_alloc_ = cuda_alloc;

  HUM_DEV("Alloc Mem end for mem(%p) umem = %p <- devPtr=%p", mem, umem, dev_ptr);
  return umem;
}

void LocalCuDevice::FreeMem(HUMMicroMem* umem) 
{
  HUM_DEV("Free umem %p begin", umem);
  if (umem != NULL && umem->get_obj() != NULL) {
    if(umem->cuda_alloc_) 
      dispatch_->cudaFree(umem->get_obj());
    umem->set_obj(NULL);
  }
  umem->Release();
  HUM_DEV("Free umem %p end", umem);
}

void* LocalCuDevice::AllocKernel(HUMKernel* kernel) {
  return NULL;
}

void LocalCuDevice::FreeKernel(HUMKernel* kernel, void* dev_specific) {
}

void LocalCuDevice::ExecuteFunc(HUMCommand* command, hum_int func_type, void* data, size_t data_size)
{
  hum_int err = HUM_SUCCESS;
  err = dispatch_->cudaSetDevice(device_id_);
  if(err != 0) {
    HUM_ERROR("cudaSetDevice(%d) Failed", device_id_);
    assert(0);
  }

  switch(func_type) {
  case HUM_CUDA_API_FUNC_BIND_TEXTURE:
    {
      cuda_func_bind_texture_t* params =
        (cuda_func_bind_texture_t*)data;

      void* mem_dev = (void*)params->mem->GetDevSpecific(this);
      mem_dev = (void*)((size_t)mem_dev + params->mem_offset);

      HUM_DEV("cudaBindTexture offset=%p, texref=%p, devPtr=%p, desc=%p, size=%ld",
          params->offset, params->texref, mem_dev, params->desc, params->size);

      err = dispatch_->cudaBindTexture(
          params->offset,
          params->texref,
          mem_dev,
          params->desc,
          params->size
          );
      if(err != 0) {
        HUM_ERROR("cudaBindTexture(%d) Failed", device_id_);
        assert(0);  
      }
    }
    break;
  case HUM_CUDA_API_FUNC_UNBIND_TEXTURE:
    {
      cuda_func_unbind_texture_t* params = 
        (cuda_func_unbind_texture_t*)data;

      err = dispatch_->cudaUnbindTexture(params->texref);
      if(err != 0) {
        HUM_ERROR("cudaUnbindTexture(%d) Failed", device_id_);
        assert(0);
      }

    }
    break;
  case HUM_CUDA_API_FUNC_BIND_TEXTURE_TO_ARRAY:
    {
      cuda_func_bind_texture_to_array_t *params =
        (cuda_func_bind_texture_to_array_t*)data;

      err = dispatch_->cudaBindTextureToArray(
          params->texref, params->array, params->desc);
      if (err != 0) {
        HUM_ERROR("cudaBindTextureToArray(%d) Failed: %d",
            device_id_, err);
        assert(0);
      }
    }
    break;
  case HUM_CUDA_API_FUNC_MEMCPY_TO_ARRAY:
    {
      cuda_func_memcpy_to_array_t *params =
        (cuda_func_memcpy_to_array_t*)data;

      err = dispatch_->cudaMemcpyToArrayAsync(
          params->dst, params->wOffset, params->hOffset, params->src,
          params->count, params->kind, mem_stream_);
      if (err != 0) {
        HUM_ERROR("cudaMemcpyToArray(%d) Failed", device_id_);
        assert(0);
      }
    }
    break;
  case HUM_CUDA_API_FUNC_MEMCPY2D_TO_ARRAY:
    {
      cuda_func_memcpy2d_to_array_t *params =
        (cuda_func_memcpy2d_to_array_t*)data;

      err = dispatch_->cudaMemcpy2DToArrayAsync(
          params->dst, params->wOffset, params->hOffset, params->src,
          params->spitch, params->width, params->height, params->kind, mem_stream_);
      if (err != 0) {
        HUM_ERROR("cudaMemcpy2DToArray(%d) Failed", device_id_);
        assert(0);
      }
    }
    break;
  case HUM_CUDA_API_FUNC_MEMCPY3D:
    {
      const cudaMemcpy3DParms *params =
        (const cudaMemcpy3DParms*)data;

      err = dispatch_->cudaMemcpy3DAsync(params, mem_stream_);
      if (err != 0) {
        HUM_ERROR("cudaMemcpy3D(%d) Failed", device_id_);
        assert(0);
      }
    }
    break;
  default:
    HUM_ASSERT(0);
  }
}

void LocalCuDevice::InitModule(void **fatCubinHandle) {
  dispatch()->__cudaInitModule(fatCubinHandle);
}

void LocalCuDevice::RegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid,
    uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
  dispatch()->__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
    deviceName, thread_limit, tid,
    bid, bDim, gDim, wSize);
}

void** LocalCuDevice::RegisterFatBinary(void * fatCubin) {
  return dispatch()->__cudaRegisterFatBinary(fatCubin);
}

void LocalCuDevice::UnregisterFatBinary(void **fatCubinHandle){
  dispatch()->__cudaUnregisterFatBinary(fatCubinHandle);
}

cudaError_t LocalCuDevice::PopCallConfiguration(dim3 *gridDim,
    dim3 *blockDim,  size_t *sharedMem, void *stream) {
  return dispatch()->__cudaPopCallConfiguration(gridDim,
    blockDim, sharedMem, stream);
}

unsigned LocalCuDevice::PushCallConfiguration(dim3 gridDim,
    dim3 blockDim, size_t sharedMem, void *stream) {
  return dispatch()->__cudaPushCallConfiguration(gridDim,
    blockDim, sharedMem, stream);
}

int LocalCuDevice::GetNumDevices()
{
  return num_devices_;
}

int LocalCuDevice::GetNumNodes()
{
  return gsize_;
}

#ifdef USE_MEM_PREFETCH
void LocalCuDevice::Run() {
  int id = -1;

  for (unsigned int i = 0; i < NUM_MEMCPY_THREAD; ++i) {
    if (pthread_equal(pthread_self(), memcpy_thread_[i])) {
      id = i;
      break;
    }
  }
  if (id == -1) {
    printf("Failed to get thread id of memcpy thread\n");
    assert(0);
    return;
  }

  while(threads_running_) {
    pthread_mutex_lock(&memcpy_mutex_[id]);
    std::list<memcpy_command_t*>::iterator I = memcpy_list[id].begin();
    std::list<memcpy_command_t*>::iterator E = memcpy_list[id].end();
    pthread_mutex_unlock(&memcpy_mutex_[id]);

    if (I == E) {
      sem_wait(&sem_memcpy_[id]);
      continue;
    }

    for (; I != E; ++I) {
      memcpy_command_t* memcpy_command = *I;

      HUMUnifiedMem* umbuf = memcpy_command->umbuf;
      void* mem_dst_dev = umbuf->GetDevSpecific(this, false);
      mem_dst_dev = (void*)((size_t)mem_dst_dev + memcpy_command->offset);
      void* src_ptr = (void*)((size_t)memcpy_command->user_src + memcpy_command->copied_size);

#ifdef DISABLE_MEMCPY_SCHEDULING
      size_t copy_size = memcpy_command->size;
#else
      size_t copy_size =
        min((BLOCK_SIZE - memcpy_command->offset % BLOCK_SIZE), memcpy_command->size);
#endif
      struct memcpy_direct_command command;
      command.dst_addr = (uint64_t)mem_dst_dev;
      command.src_addr = (uint64_t)src_ptr;
      command.copy_size = copy_size;
      command.gpu_mask.bitmap[0] = (unsigned long)(1 << device_id_);
      ioctl(driver_fd_, IOCTL_MEMCPY_H2D_PREFETCH, &command);

      // update variables
      memcpy_command->offset += copy_size;
      memcpy_command->size -= copy_size;
      memcpy_command->copied_size += copy_size;

      // when this copy command is completely finished
      if (memcpy_command->size == 0) {
        // handle mprotect logic
        memcpy_control_block_t *control_block = memcpy_command->control_block;
        pthread_mutex_lock(&control_block->block_mutex);
        control_block->finished_chunks += 1;
        
        // when all the threads finished their job
        if (control_block->num_chunks == control_block->finished_chunks) {
          if (control_block->mprotected == true) {
            // restore protection of the region
            if (mprotect(control_block->mprotect_start,
                  control_block->mprotect_size,
                  PROT_READ | PROT_WRITE | PROT_EXEC)) {
              HUM_ERROR("mprotect Failed errno=%d", errno);
              assert(0);
            }
          }

          control_block->ref_count--;
          if (control_block->ref_count == 0) {
            std::list<memcpy_control_block_t*>::iterator LI;
            std::list<memcpy_control_block_t*>::iterator LE;
            memcpy_control_block_t* target_block;
            
            pthread_mutex_lock(&control_block_list_mutex_);
            for (LI = control_block_list_.begin(), LE = control_block_list_.end();
                LI != LE; ++LI) {
              target_block = *LI;
              if (control_block == target_block) {
                control_block_list_.erase(LI);
                break;
              }
            }
            pthread_mutex_unlock(&control_block_list_mutex_);

            pthread_mutex_unlock(&control_block->block_mutex);
            pthread_mutex_destroy(&control_block->block_mutex);
            delete control_block;
          }
        }
        else {
          pthread_mutex_unlock(&control_block->block_mutex);
        }

        pthread_mutex_lock(&memcpy_mutex_[id]);
        I = memcpy_list[id].erase(I);
        pthread_mutex_unlock(&memcpy_mutex_[id]);
        continue;
      }
    }
  }
}

void* LocalCuDevice::ThreadFunc(void* argp) {
  ((LocalCuDevice*)argp)->Run();
  return NULL;
}
#endif
