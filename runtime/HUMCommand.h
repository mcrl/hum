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

#ifndef __HUM_COMMAND_H__
#define __HUM_COMMNAD_H__

#include "hum.h"
#include <vector>
#include <map>
#include <string>
#include <assert.h>
#include "HUMKernel.h"

#define HUM_COMMAND_CATEGORY_NONE			0
#define HUM_COMMAND_CATEGORY_MEM				1
#define HUM_COMMAND_CATEGORY_KERNEL		2
#define HUM_COMMAND_CATEGORY_PROGRAM		3
#define HUM_COMMAND_CATEGORY_DRIVER		4

class HUMCommandQueue;
class HUMContext;
class HUMDevice;
class HUMEvent;
class HUMMicroMem;
class HUMMem;
class HUMMemCommand;
class HUMKernelCommand;
class HUMDriverCommand;

class HUMCudaCommand;

class HUMCommand {
	public:
		HUMCommand(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
            hum_command_type type, HUMEvent* event = NULL);
		virtual ~HUMCommand();

  hum_command_type type() const { return type_; }
  hum_uint category() const { return category_; }
  HUMContext* context() const { return context_; }
  HUMDevice* device() const { return device_; }

	//For CUDA Command
	virtual size_t shared_mem() const { return 0; }

  HUMDevice* source_device() const { return dev_src_; }
  HUMDevice* destination_device() const { return dev_dst_; }
  int source_node() const { return node_src_; }
  int destination_node() const { return node_dst_; }
  unsigned long event_id() const { return event_id_; }

	bool IsPartialCommand() const; 
  void SetAsPartialCommand(HUMCommand* root);


  void SetWaitList(hum_uint num_events_in_wait_list,
                   const hum_event_handle* event_wait_list);
  void AddWaitEvent(HUMEvent* event);
  HUMEvent* ExportEvent();

  bool IsExecutable();
  void Submit();
  void SetError(hum_int error);
  void SetAsRunning();
  void SetAsComplete();

  virtual void Execute();
	virtual bool ResolveConsistency() { return true; }

	protected:
	void GetCopyPattern(HUMDevice* dev_src, HUMDevice* dev_dst, bool& use_read,
			bool& use_write, bool& use_copy, bool& use_send,
			bool& use_recv, bool& use_rcopy, bool& alloc_ptr,
			bool& use_host_ptr);
	HUMEvent* CloneMem(HUMDevice* dev_src, HUMDevice* dev_dst, HUMMem* mem);

	bool LocateMemOnDevice(HUMMem* mem);
	void AccessMemOnDevice(HUMMem* mem, bool write);
	bool ChangeDeviceToReadMem(HUMMem* mem, HUMDevice*& device);

	hum_command_type type_;

	HUMCommandQueue* queue_;
	HUMContext* context_;
	HUMDevice* device_;
	HUMEvent* event_;
	std::vector<HUMEvent*> wait_events_;

	bool wait_events_complete_;
	bool wait_events_good_;
	bool consistency_resolved_;
	hum_int error_;

	HUMDevice* dev_src_;
  HUMDevice* dev_dst_;
  int node_src_;
  int node_dst_;
  size_t event_id_;

	protected:
	hum_uint category_;

	public:
		static HUMMemCommand*
		CreateReadBuffer(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
		    HUMMem* buffer, size_t offset, size_t size, void* ptr);

		static HUMMemCommand*
		CreateReadBufferFromLocalGPU(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
		    HUMMem* buffer, size_t offset, size_t size, void* ptr, int target_worker);

		static HUMMemCommand*
		CreateWriteBuffer(HUMContext* context, HUMDevice* device,	HUMCommandQueue* queue, 
					HUMMem* buffer, size_t offset,	size_t size, void* ptr, bool protect_src);
		static HUMMemCommand*
		CreateWriteBufferToLocalGPU(HUMContext* context, HUMDevice* device,	HUMCommandQueue* queue, 
					HUMMem* buffer, size_t offset,	size_t size, void* ptr, int target_worker);

		static HUMMemCommand*
		CreateBroadcastBuffer(HUMContext* context, HUMDevice* device,	HUMCommandQueue* queue, 
					HUMMem* buffer, size_t offset,	size_t size, void* ptr);

		static HUMMemCommand*
		CreateWriteBufferToSymbol(HUMContext* context, HUMDevice* device,	HUMCommandQueue* queue, 
					const void* symbol, size_t offset,	size_t size, void* ptr);

		static HUMMemCommand*
		CreateReadBufferFromSymbol(HUMContext* context, HUMDevice* device,HUMCommandQueue* queue, 
					const void* symbol, size_t offset,	size_t size, void* ptr);

		static HUMMemCommand*
		CreateCopyBuffer(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
					HUMMem* src_buffer, HUMMem* dst_buffer, size_t src_offset,
					size_t dst_offset, size_t size);

		static HUMMemCommand*
		CreateCopyBroadcastBuffer(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
					HUMMem* src_buffer, HUMMem* dst_buffer, size_t src_offset,
					size_t dst_offset, size_t size);

		static HUMMemCommand*
		CreateFillBuffer(HUMContext* context, HUMDevice* device,
                            HUMCommandQueue* queue, HUMMem* buffer,
                            const void* pattern, size_t pattern_size,
                            size_t offset, size_t size); 

		static HUMKernelCommand*
		CreateNDRangeKernel(HUMContext* context, HUMDevice* device,
					HUMCommandQueue* queue, HUMKernel* kernel,
					hum_uint work_dim, const size_t* global_work_offset,
					const size_t* global_work_size,
					const size_t* local_work_size);

		static HUMKernelCommand*
    CreateLaunchCudaKernel(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
        const char* kernel_name, const void* kernel_func, const size_t* num_blocks_3d, const size_t* block_dim_3d,
				size_t shared_mem, int num_args, HUMKernelArg** args);
		static HUMKernelCommand*
    CreateLaunchCudaDirectKernel(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
        const char* kernel_name, const void* kernel_func, const size_t* num_blocks_3d, const size_t* block_dim_3d,
				size_t shared_mem, int num_args, HUMKernelArg** args);


		static HUMCommand*
		CreateNop(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue);

		static HUMCommand*
		CreateMarker(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue);

		static HUMCommand*
		CreateEventRecord(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue, HUMEvent* event);

		static HUMDriverCommand*
		CreateDriver(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue, hum_int func_type, void* data, size_t data_size);
};


class HUMMemCommand : public HUMCommand
{
	public:
	HUMMemCommand(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
			hum_command_type type);
	virtual ~HUMMemCommand();

  virtual void Execute();
	virtual bool ResolveConsistency();
 
	bool ResolveConsistencyOfReadMem();
  bool ResolveConsistencyOfWriteMem();
  bool ResolveConsistencyOfCopyMem();
  void UpdateConsistencyOfReadMem();
  void UpdateConsistencyOfWriteMem();
  void UpdateConsistencyOfCopyMem();
 

	void AnnotateSourceDevice(HUMDevice* device) { dev_src_ = device; }
  void AnnotateDestinationDevice(HUMDevice* device) { dev_dst_ = device; }
  void AnnotateSourceNode(int node) { node_src_ = node; }
  void AnnotateDestinationNode(int node) { node_dst_= node; }

  HUMDevice* dev_src_;
  HUMDevice* dev_dst_;
  int node_src_;
  int node_dst_;
  size_t event_id_;


	HUMMem* mem_src_;
  HUMMem* mem_dst_;
  size_t off_src_;
  size_t off_dst_;
  size_t size_;
  void* ptr_;
  bool protect_src_;

	int target_worker_;

	void* pattern_;
	size_t pattern_size_;

	const void* symbol_;

  void* temp_buf_;
};

class HUMKernelCommand : public HUMCommand
{
	public:
	HUMKernelCommand(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
			hum_command_type type);
	virtual ~HUMKernelCommand();

  virtual void Execute();
 	virtual bool ResolveConsistency();
  
	bool ResolveConsistencyOfLaunchKernel();
	void UpdateConsistencyOfLaunchKernel();

	HUMKernel* kernel_;

	//for CUDA Kernel
	std::string cuda_kernel_name_;
	void* cuda_kernel_func_;
	size_t cuda_shared_mem_;
	virtual size_t shared_mem() const { return cuda_shared_mem_; }

  hum_uint work_dim_;
  size_t gwo_[3];
  size_t gws_[3];
  size_t lws_[3];
  size_t nwg_[3];
  std::map<hum_uint, HUMKernelArg*>* kernel_args_;
};

class HUMDriverCommand : public HUMCommand
{
	public:
	HUMDriverCommand(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
			hum_command_type type, hum_int func_type, void* data, size_t data_size);
	virtual ~HUMDriverCommand();

  virtual void Execute();
 	virtual bool ResolveConsistency() { return true; }

	hum_int func_type_;
	void* data_;
	size_t data_size_;
};

#endif //__HUM_COMMAND_H__
