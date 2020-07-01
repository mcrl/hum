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

#include "HUMCommand.h"
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <malloc.h>
#include "Callbacks.h"
#include "HUMCommandQueue.h"
#include "HUMContext.h"
#include "HUMDevice.h"
#include "HUMEvent.h"
#include "HUMKernel.h"
#include "HUMMem.h"

using namespace std;

HUMCommand::HUMCommand(HUMContext* context, HUMDevice* device, HUMCommandQueue* queue,
            hum_command_type type, HUMEvent* event)
{
  type_ = type;
	category_ = HUM_COMMAND_CATEGORY_NONE;
  queue_ = queue;
  if (queue_ != NULL) {
    queue_->Retain();
    context = queue_->context();
    device = queue_->device();
  }
  context_ = context;
  context_->Retain();
  device_ = device;

	if(event == NULL) {
		if (queue_ != NULL) {
			event_ = new HUMEvent(queue, this);
		} 
		else {
			event_ = new HUMEvent(context, this);
		}
	}
	else {
		event_ = event;
		event_->SetStatus(HUM_QUEUED);
		event_->Retain();
	}

  wait_events_complete_ = false;
  wait_events_good_ = true;
  consistency_resolved_ = false;
  error_ = HUM_SUCCESS;

  dev_src_ = NULL;
  dev_dst_ = NULL;
  node_src_ = -1;
  node_dst_ = -1;
  event_id_ = event_->id();
}

HUMCommand::~HUMCommand()
{
  if (queue_) queue_->Release();
  context_->Release();
  for (vector<HUMEvent*>::iterator it = wait_events_.begin();
       it != wait_events_.end();
       ++it) {
    (*it)->Release();
  }

  event_->Release();
}

void HUMCommand::SetWaitList(hum_uint num_events_in_wait_list,
                            const hum_event_handle* event_wait_list) {
  wait_events_.reserve(num_events_in_wait_list);
  for (hum_uint i = 0; i < num_events_in_wait_list; i++) {
    HUMEvent* e = event_wait_list[i]->c_obj;
		AddWaitEvent(e);
  }
}

void HUMCommand::AddWaitEvent(HUMEvent* event) {
  event->Retain();
  wait_events_.push_back(event);
  wait_events_complete_ = false;
}


bool HUMCommand::IsExecutable() {
  if (wait_events_complete_)
    return true;

  for (vector<HUMEvent*>::iterator it = wait_events_.begin();
       it != wait_events_.end();
       ++it) {
    if (!(*it)->IsComplete())
      return false;
  }

  for (vector<HUMEvent*>::iterator it = wait_events_.begin();
       it != wait_events_.end();
       ++it) {
    HUMEvent* event = *it;
    if (event->IsError())
      wait_events_good_ = false;
    event->Release();
  }
  wait_events_.clear();
  wait_events_complete_ = true;
  return true;
}

HUMEvent* HUMCommand::ExportEvent() {
  event_->Retain();
  return event_;
}


void HUMCommand::Submit() {
	HUM_DEV("Command %p %x SetAsSubmit ", this, type());
  event_->SetStatus(HUM_SUBMITTED);
  device_->EnqueueReadyQueue(this);
}


void HUMCommand::SetError(hum_int error) {
	HUM_DEV("SetError(%d -> %d)", error_, error);
#ifdef HUM_DEBUG
	if(error < 0) assert(0);
#endif
  error_ = error;
}

void HUMCommand::SetAsRunning() {
	HUM_DEV("Command %p %x SetAsRunning ", this, type());
  event_->SetStatus(HUM_RUNNING);
}

void HUMCommand::SetAsComplete() {
	HUM_DEV("Command %p %x SetAsComplete ", this, type());
  if (error_ != HUM_SUCCESS && error_ < 0) {
    event_->SetStatus(error_);
  } else {
    event_->SetStatus(HUM_COMPLETE);
  }
}

bool HUMCommand::IsPartialCommand() const {
  return event_id_ != event_->id();
}

void HUMCommand::SetAsPartialCommand(HUMCommand* root) {
  event_id_ = root->event_id_;
}


void HUMCommand::GetCopyPattern(HUMDevice* dev_src, HUMDevice* dev_dst,
                               bool& use_read, bool& use_write, bool& use_copy,
                               bool& use_send, bool& use_recv, bool& use_rcopy,
                               bool& alloc_ptr, bool& use_host_ptr) {
  /*
   * GetCopyPattern() decides a method to copy a memory object in dev_src to a
   * memory object in dev_dst. Possible results are as follows:
   *
   * (1) dev_dst -> Copy -> dev_dst (dev_src == dev_dst)
   * (2) host pointer -> Write -> dev_dst
   * (3) dev_src -> Read -> a temporary buffer in host -> Write -> dev_dst
   * (4) dev_src -> ClusterDriver -> dev_dst
   * (5) dev_src -> Read -> MPI_Send -> MPI_Recv -> Write -> dev_dst
   *
   * No.  Required Commands                     Intermediate Buffer
   *      read  write  copy  send  recv  rcopy  alloc  use_host
   * (1)               TRUE
   * (2)        TRUE                                   TRUE
   * (3)  TRUE  TRUE                            TRUE
   * (4)                                 TRUE
   * (5)                     TRUE  TRUE
   */

  use_read = false;
  use_write = false;
  use_copy = false;
  use_send = false;
  use_recv = false;
  use_rcopy = false;
  alloc_ptr = false;
  use_host_ptr = false;

  if (dev_src == dev_dst) { // (1)
    use_copy = true;
  } 
	else if (dev_src == LATEST_HOST) { // (2)
    use_write = true;
    use_host_ptr = true;
  } 
	else if (dev_src->node_id() == dev_dst->node_id()) {
    if (dev_src->node_id() == 0) { // (3)
      use_read = true;
      use_write = true;
      alloc_ptr = true;
    } 
		else { // (4)
      use_rcopy = true;
    }
  } 
	else { // (5)
    use_send = true;
    use_recv = true;
  }
}

static void HUM_CALLBACK IssueCommandCallback(HUMEvent* event, hum_int status,
                                             void* user_data) {
  HUMCommand* command = (HUMCommand*)user_data;
  command->Submit();
}

HUMEvent* HUMCommand::CloneMem(HUMDevice* dev_src, HUMDevice* dev_dst,
                             HUMMem* mem) {

	assert(0);
  bool use_read, use_write, use_copy, use_send, use_recv, use_rcopy;
  bool alloc_ptr, use_host_ptr;
  GetCopyPattern(dev_src, dev_dst, use_read, use_write, use_copy, use_send,
                 use_recv, use_rcopy, alloc_ptr, use_host_ptr);

  void* ptr = NULL;
  if (alloc_ptr)
    ptr = memalign(4096, mem->size());
  if (use_host_ptr)
    ptr = mem->GetHostPtr();

  HUMMemCommand* read = NULL;
  HUMMemCommand* write = NULL;
  HUMMemCommand* copy = NULL;
  if (mem->IsImage()) {
		HUM_ERROR("%s", "TODO: HUM does not support Image type yet\n");
		assert(0);
	} 
	else {
    if (use_read || use_send)
      read = CreateReadBuffer(context_, dev_src, NULL, mem, 0, mem->size(),
                              ptr);
    if (use_write || use_recv)
      write = CreateWriteBuffer(context_, dev_dst, NULL, mem, 0, mem->size(),
                                ptr, false);
    if (use_copy || use_rcopy)
      copy = CreateCopyBuffer(context_, dev_dst, NULL, mem, mem, 0, 0,
                              mem->size());
  }
  if (use_send) {
    read->AnnotateDestinationNode(dev_dst->node_id());
    read->SetAsPartialCommand(write);
  }
  if (use_recv)
    write->AnnotateSourceNode(dev_src->node_id());
  if (use_rcopy) {
    copy->AnnotateSourceDevice(dev_src);
    copy->AnnotateDestinationDevice(dev_dst);
  }
  if (alloc_ptr)
    write->temp_buf_ = ptr;

	HUMEvent* last_event = NULL;
  if (copy != NULL)
    last_event = copy->ExportEvent();
  else
    last_event = write->ExportEvent();

  if (use_read && use_write) {
    read->event_->AddCallback(new EventCallback(IssueCommandCallback, write, HUM_COMPLETE));
    write = NULL;
  }
  if (read != NULL)
    read->Submit();
  if (write != NULL)
    write->Submit();
  if (copy != NULL)
    copy->Submit();
  mem->AddLatest(dev_dst);

  return last_event;
}

bool HUMCommand::LocateMemOnDevice(HUMMem* mem) 
{
  if (mem->HasLatest(device_) || mem->EmptyLatest())
    return true;
  HUMDevice* source = mem->GetNearestLatest(device_);
  HUMEvent* last_event = CloneMem(source, device_, mem);
  AddWaitEvent(last_event);
  last_event->Release();
  return false;
}

void HUMCommand::AccessMemOnDevice(HUMMem* mem, bool write) 
{
  if (write)
    mem->SetLatest(device_);
  else
    mem->AddLatest(device_);
}

bool HUMCommand::ChangeDeviceToReadMem(HUMMem* mem, HUMDevice*& device) 
{
  if (mem->HasLatest(device) || mem->EmptyLatest())
    return true;
  HUMDevice* source = mem->GetNearestLatest(device);
  if (source == LATEST_HOST) {
    HUMEvent* last_event = CloneMem(source, device, mem);
    AddWaitEvent(last_event);
    last_event->Release();
    return false;
  }
  device = source;
  return true;
}


void HUMCommand::Execute() {
  switch (type_) {

    case HUM_COMMAND_MARKER:
      break;
    case HUM_COMMAND_BARRIER:
      break;
    case HUM_COMMAND_MIGRATE_MEM_OBJECTS:
			assert(0);
      //device_->MigrateMemObjects(this, num_mem_objects_, mem_list_,
      //                           migration_flags_);
      break;
    case HUM_COMMAND_WAIT_FOR_EVENTS:
      break;
		case HUM_COMMAND_NOP:
      break;
			/*
    case HUM_COMMAND_ALLTOALL_BUFFER:
			assert(0);
      //device_->AlltoAllBuffer(this, mem_src_, mem_dst_, off_src_, off_dst_,
      //                        size_);
      break;
    case HUM_COMMAND_BROADCAST_BUFFER:
			assert(0);
      //device_->BroadcastBuffer(this, mem_src_, mem_dst_, off_src_, off_dst_,
      //                         size_);
      break;
    case HUM_COMMAND_LOCAL_FILE_OPEN:
			assert(0);
      //device_->LocalFileOpen(this, file_dst_, filename_, file_open_flags_);
      break;
    case HUM_COMMAND_COPY_BUFFER_TO_FILE:
			assert(0);
      //device_->CopyBufferToFile(this, mem_src_, file_dst_, off_src_, off_dst_,
      //                          size_);
      break;
    case HUM_COMMAND_COPY_FILE_TO_BUFFER:
			assert(0);
      //device_->CopyFileToBuffer(this, file_src_, mem_dst_, off_src_, off_dst_,
      //                          size_);
      break;
			*/
    default:
			assert(0);
      HUM_ERROR("Unsupported command [%x]", type_);
      break;
  }
}

HUMMemCommand*
HUMCommand::CreateReadBuffer(HUMContext* context, HUMDevice* device,
                            HUMCommandQueue* queue, HUMMem* buffer,
                            size_t offset, size_t size, void* ptr) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_READ_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = buffer;
  command->mem_src_->Retain();
  command->off_src_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
  return command;
}

HUMMemCommand*
HUMCommand::CreateReadBufferFromLocalGPU(HUMContext* context, HUMDevice* device,
                            HUMCommandQueue* queue, HUMMem* buffer,
                            size_t offset, size_t size, void* ptr, int target_worker) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_READ_LOCAL_GPU_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = buffer;
  command->mem_src_->Retain();
  command->off_src_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
	command->target_worker_ = target_worker;

  return command;
}


HUMMemCommand*
HUMCommand::CreateWriteBuffer(HUMContext* context, HUMDevice* device,
                             HUMCommandQueue* queue, HUMMem* buffer,
                             size_t offset, size_t size, void* ptr, bool protect_src) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_WRITE_BUFFER);
  if (command == NULL) return NULL;
  command->mem_dst_ = buffer;
  command->mem_dst_->Retain();
  command->off_dst_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
  command->protect_src_ = protect_src;

  return command;
}

HUMMemCommand*
HUMCommand::CreateWriteBufferToLocalGPU(HUMContext* context, HUMDevice* device,
                             HUMCommandQueue* queue, HUMMem* buffer,
                             size_t offset, size_t size, void* ptr, int target_worker) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_WRITE_LOCAL_GPU_BUFFER);
  if (command == NULL) return NULL;
  command->mem_dst_ = buffer;
  command->mem_dst_->Retain();
  command->off_dst_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
	command->target_worker_ = target_worker;

  return command;
}



HUMMemCommand*
HUMCommand::CreateBroadcastBuffer(HUMContext* context, HUMDevice* device,
                             HUMCommandQueue* queue, HUMMem* buffer,
                             size_t offset, size_t size, void* ptr) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_BROADCAST_BUFFER);
  if (command == NULL) return NULL;
  command->mem_dst_ = buffer;
  command->mem_dst_->Retain();
  command->off_dst_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;

  return command;
}


HUMMemCommand*
HUMCommand::CreateWriteBufferToSymbol(HUMContext* context, HUMDevice* device,
                             HUMCommandQueue* queue, const void* symbol,
                             size_t offset, size_t size, void* ptr) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_WRITE_BUFFER_TO_SYMBOL);
  if (command == NULL) return NULL;
  command->symbol_ = symbol;
  command->off_dst_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
  return command;
}

HUMMemCommand*
HUMCommand::CreateReadBufferFromSymbol(HUMContext* context, HUMDevice* device,
                             HUMCommandQueue* queue, const void* symbol,
                             size_t offset, size_t size, void* ptr) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_READ_BUFFER_FROM_SYMBOL);
  if (command == NULL) return NULL;
  command->symbol_ = symbol;
  command->off_dst_ = offset;
  command->size_ = size;
  command->ptr_ = ptr;
  return command;
}

HUMMemCommand*
HUMCommand::CreateCopyBuffer(HUMContext* context, HUMDevice* device,
                            HUMCommandQueue* queue, HUMMem* src_buffer,
                            HUMMem* dst_buffer, size_t src_offset,
                            size_t dst_offset, size_t size) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_COPY_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = src_buffer;
  command->mem_src_->Retain();
  command->mem_dst_ = dst_buffer;
  command->mem_dst_->Retain();
  command->off_src_ = src_offset;
  command->off_dst_ = dst_offset;
  command->size_ = size;
  return command;
}

HUMMemCommand*
HUMCommand::CreateCopyBroadcastBuffer(HUMContext* context, HUMDevice* device,
                            HUMCommandQueue* queue, HUMMem* src_buffer,
                            HUMMem* dst_buffer, size_t src_offset,
                            size_t dst_offset, size_t size) {
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_COPY_BUFFER);
  if (command == NULL) return NULL;
  command->mem_src_ = src_buffer;
  command->mem_src_->Retain();
  command->mem_dst_ = dst_buffer;
  command->mem_dst_->Retain();
  command->off_src_ = src_offset;
  command->off_dst_ = dst_offset;
  command->size_ = size;
  return command;
}

HUMMemCommand*
HUMCommand::CreateFillBuffer(HUMContext* context, HUMDevice* device,
                            HUMCommandQueue* queue, HUMMem* buffer,
                            const void* pattern, size_t pattern_size,
                            size_t offset, size_t size) 
{
  HUMMemCommand* command = new HUMMemCommand(context, device, queue,
                                     HUM_COMMAND_FILL_BUFFER);
  if (command == NULL) return NULL;
  command->mem_dst_ = buffer;
  command->mem_dst_->Retain();
  command->pattern_ = malloc(pattern_size);
  memcpy(command->pattern_, pattern, pattern_size);
  command->pattern_size_ = pattern_size;
  command->off_dst_ = offset;
  command->size_ = size;
  return command;
}



HUMKernelCommand*
HUMCommand::CreateNDRangeKernel(HUMContext* context, HUMDevice* device,
                               HUMCommandQueue* queue, HUMKernel* kernel,
                               hum_uint work_dim,
                               const size_t* global_work_offset,
                               const size_t* global_work_size,
                               const size_t* local_work_size) {
  HUMKernelCommand* command = new HUMKernelCommand(context, device, queue,
                                     HUM_COMMAND_NDRANGE_KERNEL);
  if (command == NULL) return NULL;
  command->kernel_ = kernel;
  command->kernel_->Retain();
  command->work_dim_ = work_dim;
  for (hum_uint i = 0; i < work_dim; i++) {
    command->gwo_[i] = (global_work_offset != NULL) ? global_work_offset[i] :
                                                      0;
    command->gws_[i] = global_work_size[i];
    command->lws_[i] = (local_work_size != NULL) ? local_work_size[i] :
                           ((global_work_size[i] % 4 == 0) ? 4 : 1);
    command->nwg_[i] = command->gws_[i] / command->lws_[i];
  }
  for (hum_uint i = work_dim; i < 3; i++) {
    command->gwo_[i] = 0;
    command->gws_[i] = 1;
    command->lws_[i] = 1;
    command->nwg_[i] = 1;
  }
  command->kernel_args_ = kernel->ExportArgs();
  return command;
}

HUMKernelCommand*
HUMCommand::CreateLaunchCudaKernel(HUMContext* context, HUMDevice* device, 
															HUMCommandQueue* queue,
															const char* kernel_name, const void* kernel_func, 
															const size_t* num_blocks_3d, const size_t* block_dim_3d,
															size_t shared_mem, 
															int num_args, HUMKernelArg** args)
{
	HUM_DEV("CreateLaunchCudaKernel kernel_name=%s kernel_func=%p",
			kernel_name, kernel_func);

  HUMKernelCommand* command = new HUMKernelCommand(context, device, queue, HUM_COMMAND_CUDA_KERNEL);
  if (command == NULL) return NULL;
	command->cuda_kernel_name_ = kernel_name;
	command->cuda_kernel_func_ = (void*)kernel_func;
	command->cuda_shared_mem_ = shared_mem;
  command->work_dim_ = 3;

  for (hum_uint i = 0; i < 3; i++) {
    command->gwo_[i] = 0;
    command->gws_[i] = num_blocks_3d[i] * block_dim_3d[i];
    command->lws_[i] = block_dim_3d[i]; 
		command->nwg_[i] = num_blocks_3d[i];
  }

	HUM_DEV("gws_(%d, %d, %d) : lws_(%d,%d,%d) : nwg_(%d,%d,%d)", 
			command->gws_[0], command->gws_[1], command->gws_[2],
			command->lws_[0], command->lws_[1], command->lws_[2],
			command->nwg_[0], command->nwg_[1], command->nwg_[2]);

	std::map<hum_uint, HUMKernelArg*>* args_map = new std::map<hum_uint, HUMKernelArg*>;
	for(int i = 0; i < num_args; i++) {

		(*args_map)[i] = args[i];
		HUM_DEV("cmd(%p): kernel_args_[%d] = %p, offset=%ld", command, i, args[i], args[i]->cuda_offset);

	}
	command->kernel_args_ = args_map;

  return command;
}

HUMKernelCommand*
HUMCommand::CreateLaunchCudaDirectKernel(HUMContext* context, HUMDevice* device, 
															HUMCommandQueue* queue,
															const char* kernel_name, const void* kernel_func, 
															const size_t* num_blocks_3d, const size_t* block_dim_3d,
															size_t shared_mem, 
															int num_args, HUMKernelArg** args)
{
	HUM_DEV("CreateLaunchCudaDirectKernel kernel_name=%s kernel_func=%p",
			kernel_name, kernel_func);

  HUMKernelCommand* command = new HUMKernelCommand(context, device, queue, HUM_COMMAND_CUDA_DIRECT_KERNEL);
  if (command == NULL) return NULL;
	command->cuda_kernel_name_ = kernel_name;
	command->cuda_kernel_func_ = (void*)kernel_func;
	command->cuda_shared_mem_ = shared_mem;
  command->work_dim_ = 3;

  for (hum_uint i = 0; i < 3; i++) {
    command->gwo_[i] = 0;
    command->gws_[i] = num_blocks_3d[i] * block_dim_3d[i];
    command->lws_[i] = block_dim_3d[i]; 
		command->nwg_[i] = num_blocks_3d[i];
  }

	HUM_DEV("gws_(%d, %d, %d) : lws_(%d,%d,%d) : nwg_(%d,%d,%d)", 
			command->gws_[0], command->gws_[1], command->gws_[2],
			command->lws_[0], command->lws_[1], command->lws_[2],
			command->nwg_[0], command->nwg_[1], command->nwg_[2]);

	std::map<hum_uint, HUMKernelArg*>* args_map = new std::map<hum_uint, HUMKernelArg*>;
	for(int i = 0; i < num_args; i++) {

		(*args_map)[i] = args[i];
		HUM_DEV("cmd(%p): kernel_args_[%d] = %p, offset=%ld", command, i, args[i], args[i]->cuda_offset);

	}
	command->kernel_args_ = args_map;

  return command;
}


HUMCommand*
HUMCommand::CreateNop(HUMContext* context, HUMDevice* device,
                     HUMCommandQueue* queue) {
  HUMCommand* command = new HUMCommand(context, device, queue, HUM_COMMAND_NOP);
  return command;
}

HUMCommand*
HUMCommand::CreateMarker(HUMContext* context, HUMDevice* device,
                        HUMCommandQueue* queue) 
{
  HUMCommand* command = new HUMCommand(context, device, queue,
                                     HUM_COMMAND_MARKER);
  return command;
}

HUMCommand*
HUMCommand::CreateEventRecord(HUMContext* context, HUMDevice* device,
                        HUMCommandQueue* queue, HUMEvent* event) 
{
  HUMCommand* command = new HUMCommand(context, device, queue,
                                     HUM_COMMAND_NOP, event);
  return command;
}

HUMDriverCommand*
HUMCommand::CreateDriver(HUMContext* context, HUMDevice* device,
                        HUMCommandQueue* queue, hum_int func_type, void* data, size_t data_size) 
{
  HUMDriverCommand* command = new HUMDriverCommand(context, device, queue,
                                     HUM_COMMAND_DRIVER, func_type, data, data_size);
  return command;
}


HUMMemCommand::HUMMemCommand(HUMContext* context,
		HUMDevice* device, HUMCommandQueue* queue,
		hum_command_type type)
: HUMCommand(context, device, queue, type)
{
	category_ = HUM_COMMAND_CATEGORY_MEM;

	mem_src_ = NULL;
	mem_dst_ = NULL;
	off_src_ = 0;
	off_dst_ = 0;
	size_ = 0;
	ptr_ = NULL;

	pattern_ = NULL;
	pattern_size_ = 0;

	temp_buf_ = NULL;
}

HUMMemCommand::~HUMMemCommand()
{ 
  if (mem_src_) mem_src_->Release();
	if (mem_dst_) mem_dst_->Release();

	if (pattern_) free(pattern_);

  if (temp_buf_) free(temp_buf_);
}

bool HUMMemCommand::ResolveConsistency() {
	return true;	//TODO: it is not need for CUDA

	HUM_DEV("ResolveConsistency Cmd(%p) type_=%x begin", this, type_);
	bool resolved = consistency_resolved_;

  if (!resolved) {
    switch (type_) {
      case HUM_COMMAND_READ_BUFFER:
      case HUM_COMMAND_READ_IMAGE:
      case HUM_COMMAND_READ_BUFFER_RECT:
        resolved = ResolveConsistencyOfReadMem();
        break;
      case HUM_COMMAND_WRITE_BUFFER:
      case HUM_COMMAND_WRITE_IMAGE:
      case HUM_COMMAND_WRITE_BUFFER_RECT:
      case HUM_COMMAND_FILL_BUFFER:
      case HUM_COMMAND_FILL_IMAGE:
        resolved = ResolveConsistencyOfWriteMem();
        break;
      case HUM_COMMAND_COPY_BUFFER:
      case HUM_COMMAND_COPY_IMAGE:
      case HUM_COMMAND_COPY_IMAGE_TO_BUFFER:
      case HUM_COMMAND_COPY_BUFFER_TO_IMAGE:
      case HUM_COMMAND_COPY_BUFFER_RECT:
        resolved = ResolveConsistencyOfCopyMem();
		break;
	  case HUM_COMMAND_WRITE_BUFFER_TO_SYMBOL:
		resolved = true;
		break;
	  case HUM_COMMAND_COPY_BROADCAST_BUFFER:
		resolved = true;
		break;
	  case HUM_COMMAND_BROADCAST_BUFFER:
		resolved = true;
		break;

#if 0
      case HUM_COMMAND_MAP_BUFFER:
      case HUM_COMMAND_MAP_IMAGE:
        resolved = ResolveConsistencyOfMap();
        break;
      case HUM_COMMAND_UNMAP_MEM_OBJECT:
        resolved = ResolveConsistencyOfUnmap();
        break;
      case HUM_COMMAND_BROADCAST_BUFFER:
        resolved = ResolveConsistencyOfBroadcast();
        break;
      case HUM_COMMAND_ALLTOALL_BUFFER:
        resolved = ResolveConsistencyOfAlltoAll();
        break;
      case HUM_COMMAND_COPY_BUFFER_TO_FILE:
        resolved = ResolveConsistencyOfCopyMemToFile();
        break;
      case HUM_COMMAND_COPY_FILE_TO_BUFFER:
        resolved = ResolveConsistencyOfCopyFileToMem();
        break;
#else
      case HUM_COMMAND_MAP_BUFFER:
      case HUM_COMMAND_MAP_IMAGE:
      case HUM_COMMAND_UNMAP_MEM_OBJECT:
      //case HUM_COMMAND_BROADCAST_BUFFER:
      //case HUM_COMMAND_ALLTOALL_BUFFER:
      //case HUM_COMMAND_COPY_BUFFER_TO_FILE:
      //case HUM_COMMAND_COPY_FILE_TO_BUFFER:
				assert(0);
        break;

#endif
      default:
				assert(0);
        resolved = true;
        break;
    }
  }
  if (resolved) {
    switch (type_) {
      case HUM_COMMAND_READ_BUFFER:
      case HUM_COMMAND_READ_IMAGE:
      case HUM_COMMAND_READ_BUFFER_RECT:
        UpdateConsistencyOfReadMem();
        break;
      case HUM_COMMAND_WRITE_BUFFER:
      case HUM_COMMAND_WRITE_IMAGE:
      case HUM_COMMAND_WRITE_BUFFER_RECT:
      case HUM_COMMAND_FILL_BUFFER:
      case HUM_COMMAND_FILL_IMAGE:
        UpdateConsistencyOfWriteMem();
        break;
      case HUM_COMMAND_COPY_BUFFER:
      case HUM_COMMAND_COPY_IMAGE:
      case HUM_COMMAND_COPY_IMAGE_TO_BUFFER:
      case HUM_COMMAND_COPY_BUFFER_TO_IMAGE:
      case HUM_COMMAND_COPY_BUFFER_RECT:
        UpdateConsistencyOfCopyMem();
        break;
			case HUM_COMMAND_WRITE_BUFFER_TO_SYMBOL:
				break;
			case HUM_COMMAND_BROADCAST_BUFFER:
				break;

#if 0
      case HUM_COMMAND_MAP_BUFFER:
      case HUM_COMMAND_MAP_IMAGE:
        UpdateConsistencyOfMap();
        break;
      case HUM_COMMAND_UNMAP_MEM_OBJECT:
        UpdateConsistencyOfUnmap();
        break;
      case HUM_COMMAND_BROADCAST_BUFFER:
        UpdateConsistencyOfBroadcast();
        break;
      case HUM_COMMAND_ALLTOALL_BUFFER:
        UpdateConsistencyOfAlltoAll();
        break;
      case HUM_COMMAND_COPY_BUFFER_TO_FILE:
        UpdateConsistencyOfCopyMemToFile();
        break;
      case HUM_COMMAND_COPY_FILE_TO_BUFFER:
        UpdateConsistencyOfCopyFileToMem();
        break;
#else
      case HUM_COMMAND_MAP_BUFFER:
      case HUM_COMMAND_MAP_IMAGE:
      case HUM_COMMAND_UNMAP_MEM_OBJECT:
      //case HUM_COMMAND_BROADCAST_BUFFER:
      //case HUM_COMMAND_ALLTOALL_BUFFER:
      //case HUM_COMMAND_COPY_BUFFER_TO_FILE:
      //case HUM_COMMAND_COPY_FILE_TO_BUFFER:
				assert(0);
        break;
#endif
      default:
        break;
    }
  }

	HUM_DEV("ResolveConsistency Cmd(%p) type_=%x end", this, type_);
  return resolved;
}

bool HUMMemCommand::ResolveConsistencyOfReadMem() 
{
  bool already_resolved = ChangeDeviceToReadMem(mem_src_, device_);
  consistency_resolved_ = true;
  return already_resolved;
}

bool HUMMemCommand::ResolveConsistencyOfWriteMem() 
{
  bool already_resolved = true;
  bool write_all = false;
  switch (type_) {
    case HUM_COMMAND_WRITE_BUFFER:
      write_all = (off_dst_ == 0 && size_ == mem_dst_->size());
      break;
    case HUM_COMMAND_WRITE_IMAGE: {
			assert(0);
			/*
      size_t* region = mem_dst_->GetImageRegion();
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 && region_[0] == region[0] &&
                   region_[1] == region[1] && region_[2] == region[2]);
									 */
      break;
    }
    case HUM_COMMAND_WRITE_BUFFER_RECT:
			assert(0);
			/*
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 &&
                   region_[0] * region_[1] * region_[2] == mem_dst_->size() &&
                   (dst_row_pitch_ == 0 || dst_row_pitch_ == region_[0]) &&
                   (dst_slice_pitch_ == 0 ||
                    dst_slice_pitch_ == region_[0] * region_[1]));
										*/
      break;
    case HUM_COMMAND_FILL_BUFFER:
      write_all = true;
			break;
    case HUM_COMMAND_FILL_IMAGE:
			assert(0);
      write_all = true;
      break;
    default:
			assert(0);
      HUM_ERROR("Unsupported command [%x]", type_);
      break;
  }
  if (!write_all)
    already_resolved &= LocateMemOnDevice(mem_dst_);
  consistency_resolved_ = true;
  return already_resolved;
}

bool HUMMemCommand::ResolveConsistencyOfCopyMem() 
{
  HUMDevice* source = device_;
  if (!ChangeDeviceToReadMem(mem_src_, source))
    return false;

  bool already_resolved = true;
  bool write_all = false;
  switch (type_) {
    case HUM_COMMAND_COPY_BUFFER:
      write_all = (off_dst_ == 0 && size_ == mem_dst_->size());
      break;
    case HUM_COMMAND_COPY_IMAGE_TO_BUFFER:
			assert(0);
			/*
      write_all = (off_dst_ == 0 &&
                   region_[0] * region_[1] * region_[2] == mem_dst_->size());
      */
			break;
    case HUM_COMMAND_COPY_IMAGE:
    case HUM_COMMAND_COPY_BUFFER_TO_IMAGE: {
			assert(0);
			/*
      size_t* region = mem_dst_->GetImageRegion();
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 && region_[0] == region[0] &&
                   region_[1] == region[1] && region_[2] == region[2]);
									 */
      break;
    }
    case HUM_COMMAND_COPY_BUFFER_RECT:
			assert(0);
			/*
      write_all = (dst_origin_[0] == 0 && dst_origin_[1] == 0 &&
                   dst_origin_[2] == 0 &&
                   region_[0] * region_[1] * region_[2] == mem_dst_->size() &&
                   (dst_row_pitch_ == 0 || dst_row_pitch_ == region_[0]) &&
                   (dst_slice_pitch_ == 0 ||
                    dst_slice_pitch_ == region_[0] * region_[1]));
										*/
      break;
    default:
			assert(0);
      HUM_ERROR("Unsupported command [%x]", type_);
      break;
  }
  if (!write_all)
    already_resolved &= LocateMemOnDevice(mem_dst_);

  bool use_read, use_write, use_copy, use_send, use_recv, use_rcopy;
  bool alloc_ptr, use_host_ptr;
  GetCopyPattern(source, device_, use_read, use_write, use_copy, use_send,
                 use_recv, use_rcopy, alloc_ptr, use_host_ptr /* unused */);

  void* ptr = NULL;
  if (alloc_ptr) {
    size_t size;
    switch (type_) {
      case HUM_COMMAND_COPY_BUFFER:
        size = size_;
        break;
      case HUM_COMMAND_COPY_IMAGE:
      case HUM_COMMAND_COPY_BUFFER_TO_IMAGE:
      case HUM_COMMAND_COPY_BUFFER_RECT:
        assert(0);
				//size = mem_dst_->GetRegionSize(region_);
        break;
      case HUM_COMMAND_COPY_IMAGE_TO_BUFFER:
        assert(0);
				//size = mem_src_->GetRegionSize(region_);
        break;
      default:
				assert(0);
        HUM_ERROR("Unsupported command [%x]", type_);
        break;
    }
    ptr = memalign(4096, size);
  }

  HUMMemCommand* read = NULL;
  switch (type_) {
    case HUM_COMMAND_COPY_BUFFER:
      if (use_read || use_send) {
        read = CreateReadBuffer(context_, source, NULL, mem_src_, off_src_,
                                size_, ptr);
      }
      if (use_write || use_recv) {
        type_ = HUM_COMMAND_WRITE_BUFFER;
        ptr_ = ptr;
      }
      break;
    case HUM_COMMAND_COPY_IMAGE:
			assert(0);
			/*
      if (use_read || use_send) {
        read = CreateReadImage(context_, source, NULL, mem_src_, src_origin_,
                               region_, 0, 0, ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_IMAGE;
        src_row_pitch_ = 0;
        src_slice_pitch_ = 0;
        ptr_ = ptr;
      }
      break;
			*/
    case HUM_COMMAND_COPY_IMAGE_TO_BUFFER:
      assert(0);
			/*
			if (use_read || use_send) {
        read = CreateReadImage(context_, source, NULL, mem_src_, src_origin_,
                               region_, 0, 0, ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_BUFFER;
        size_ = mem_src_->GetRegionSize(region_);
        ptr_ = ptr;
      }
			*/
      break;
    case HUM_COMMAND_COPY_BUFFER_TO_IMAGE:
      assert(0);
			/*
			if (use_read || use_send) {
        read = CreateReadBuffer(context_, source, NULL, mem_src_, off_src_,
                                mem_src_->GetRegionSize(region_), ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_IMAGE;
        src_row_pitch_ = 0;
        src_slice_pitch_ = 0;
        ptr_ = ptr;
      }
			*/
      break;
    case HUM_COMMAND_COPY_BUFFER_RECT:
			assert(0);
			/*
			if (use_read || use_send) {
        size_t host_origin[3] = {0, 0, 0};
        read = CreateReadBufferRect(context_, source, NULL, mem_src_,
                                    src_origin_, host_origin, region_,
                                    src_row_pitch_, src_slice_pitch_, 0, 0,
                                    ptr);
      }
      if (use_write || use_recv) {
        type_ = CL_COMMAND_WRITE_BUFFER_RECT;
        src_origin_[0] = 0;
        src_origin_[1] = 0;
        src_origin_[2] = 0;
        src_row_pitch_ = 0;
        src_slice_pitch_ = 0;
        ptr_ = ptr;
      }
			*/
      break;
    default:
			assert(0);
      HUM_ERROR("Unsupported command [%x]", type_);
      break;
  }
  if (use_send) {
    read->AnnotateDestinationNode(device_->node_id());
    read->SetAsPartialCommand(this);
  }
  if (use_recv)
    AnnotateSourceNode(source->node_id());
  if (use_rcopy) {
    AnnotateSourceDevice(source);
    AnnotateDestinationDevice(device_);
  }
  if (alloc_ptr)
    temp_buf_ = ptr;

  if (use_read && use_write) {
    HUMEvent* last_event = read->ExportEvent();
    AddWaitEvent(last_event);
    last_event->Release();
    already_resolved = false;
  }
  if (read != NULL)
    read->Submit();
  consistency_resolved_ = true;
  return already_resolved;
}

void HUMMemCommand::UpdateConsistencyOfReadMem() {
  AccessMemOnDevice(mem_src_, false);
}

void HUMMemCommand::UpdateConsistencyOfWriteMem() {
  AccessMemOnDevice(mem_dst_, true);
}

void HUMMemCommand::UpdateConsistencyOfCopyMem() {
  AccessMemOnDevice(mem_dst_, true);
}


void HUMMemCommand::Execute()
{
	HUM_DEV("HUMMemCommand %p Execute type %x begin", this, type_);
	switch (type_) {
	case HUM_COMMAND_READ_BUFFER:
		device_->ReadBuffer(this, mem_src_, off_src_, size_, ptr_);
		break;
	case HUM_COMMAND_READ_LOCAL_GPU_BUFFER:
		device_->ReadBufferFromLocalGPU(this, mem_src_, off_src_, size_, ptr_, target_worker_);
		break;

	case HUM_COMMAND_WRITE_BUFFER:
		device_->WriteBuffer(this, mem_dst_, off_dst_, size_, ptr_, protect_src_);
		break;
	case HUM_COMMAND_WRITE_LOCAL_GPU_BUFFER:
		device_->WriteBufferToLocalGPU(this, mem_dst_, off_dst_, size_, ptr_, target_worker_);
		break;

	case HUM_COMMAND_BROADCAST_BUFFER:
		device_->BroadcastBuffer(this, mem_dst_, off_dst_, size_, ptr_);
		break;
	case HUM_COMMAND_WRITE_BUFFER_TO_SYMBOL:
		device_->WriteBufferToSymbol(this, symbol_, off_dst_, size_, ptr_);
		break;
	case HUM_COMMAND_READ_BUFFER_FROM_SYMBOL:
		device_->ReadBufferFromSymbol(this, symbol_, off_dst_, size_, ptr_);
		break;
	case HUM_COMMAND_COPY_BUFFER:
		device_->CopyBuffer(this, mem_src_, mem_dst_, off_src_, off_dst_, size_);
		break;
	case HUM_COMMAND_COPY_BROADCAST_BUFFER:
		device_->CopyBroadcastBuffer(this, mem_src_, mem_dst_, off_src_, off_dst_, size_);
		break;
	case HUM_COMMAND_READ_IMAGE:
		assert(0);
		//device_->ReadImage(this, mem_src_, src_origin_, region_, dst_row_pitch_,
		//                   dst_slice_pitch_, ptr_);
		break;
	case HUM_COMMAND_WRITE_IMAGE:
		assert(0);
		//device_->WriteImage(this, mem_dst_, dst_origin_, region_, src_row_pitch_,
		//                    src_slice_pitch_, ptr_);
		break;
	case HUM_COMMAND_COPY_IMAGE:
		assert(0);
		//device_->CopyImage(this, mem_src_, mem_dst_, src_origin_, dst_origin_,
		//                   region_);
		break;
	case HUM_COMMAND_COPY_IMAGE_TO_BUFFER:
		assert(0);
		//device_->CopyImageToBuffer(this, mem_src_, mem_dst_, src_origin_,
		//                           region_, off_dst_);
		break;
	case HUM_COMMAND_COPY_BUFFER_TO_IMAGE:
		assert(0);
		//device_->CopyBufferToImage(this, mem_src_, mem_dst_, off_src_,
		//                           dst_origin_, region_);
		break;
	case HUM_COMMAND_MAP_BUFFER:
		assert(0);
		//device_->MapBuffer(this, mem_src_, map_flags_, off_src_, size_, ptr_);
		break;
	case HUM_COMMAND_MAP_IMAGE:
		assert(0);
		//device_->MapImage(this, mem_src_, map_flags_, src_origin_, region_,
		//                  ptr_);
		break;
	case HUM_COMMAND_UNMAP_MEM_OBJECT:
		assert(0);
		//device_->UnmapMemObject(this, mem_src_, ptr_);
		break;
	case HUM_COMMAND_READ_BUFFER_RECT:
		assert(0);
		//device_->ReadBufferRect(this, mem_src_, src_origin_, dst_origin_,
		//                        region_, src_row_pitch_, src_slice_pitch_,
		//                        dst_row_pitch_, dst_slice_pitch_, ptr_);
		break;
	case HUM_COMMAND_WRITE_BUFFER_RECT:
		assert(0);
		//device_->WriteBufferRect(this, mem_dst_, src_origin_, dst_origin_,
		//                         region_, src_row_pitch_, src_slice_pitch_,
		//                         dst_row_pitch_, dst_slice_pitch_, ptr_);
		break;
	case HUM_COMMAND_COPY_BUFFER_RECT:
		assert(0);
		//device_->CopyBufferRect(this, mem_src_, mem_dst_, src_origin_,
		//                        dst_origin_, region_, src_row_pitch_,
		//                        src_slice_pitch_, dst_row_pitch_,
		//                        dst_slice_pitch_);
		break;
	case HUM_COMMAND_FILL_BUFFER:
		device_->FillBuffer(this, mem_dst_, pattern_, pattern_size_, off_dst_,
		                    size_);
		break;
	case HUM_COMMAND_FILL_IMAGE:
		assert(0);
		//device_->FillImage(this, mem_dst_, ptr_, dst_origin_, region_);
		break;

	default:
		assert(0);
		HUM_ERROR("Unsupported command [%x]", type_);
		break;
	}

	HUM_DEV("HUMMemCommand %p Execute type %x end", this, type_);
}









HUMKernelCommand::HUMKernelCommand(HUMContext* context,
		HUMDevice* device, HUMCommandQueue* queue,
		hum_command_type type)
: HUMCommand(context, device, queue, type)
{
	category_ = HUM_COMMAND_CATEGORY_KERNEL;
	kernel_ = NULL;
	work_dim_ = 0;

	cuda_kernel_func_ = NULL;
}

HUMKernelCommand::~HUMKernelCommand()
{ 
  if (kernel_) kernel_->Release();
  if (kernel_args_) {
    for (map<hum_uint, HUMKernelArg*>::iterator it = kernel_args_->begin();
         it != kernel_args_->end();
         ++it) {
      HUMKernelArg* arg = it->second;
      delete(arg);
    }
    delete kernel_args_;
  }
}

bool HUMKernelCommand::ResolveConsistency() {

	return true; //TODO: it is not needed for CUDA

  bool resolved = consistency_resolved_;

	HUM_DEV("ResolveConsistency Cmd(%p) type_=%x begin", this, type_);

  if (!resolved) {
    switch (type_) {
      case HUM_COMMAND_NDRANGE_KERNEL:
      case HUM_COMMAND_TASK:
        resolved = ResolveConsistencyOfLaunchKernel();
        break;
      case HUM_COMMAND_NATIVE_KERNEL:
        assert(0);
				//resolved = ResolveConsistencyOfLaunchNativeKernel();
        break;
			case HUM_COMMAND_CUDA_KERNEL:
			case HUM_COMMAND_CUDA_DIRECT_KERNEL:
        //resolved = ResolveConsistencyOfLaunchKernel();
				resolved = 1;
				break;
      default:
				assert(0);
        resolved = true;
        break;
    }
  }
  if (resolved) {
    switch (type_) {
      case HUM_COMMAND_NDRANGE_KERNEL:
      case HUM_COMMAND_TASK:
        UpdateConsistencyOfLaunchKernel();
        break;
      case HUM_COMMAND_NATIVE_KERNEL:
       assert(0);
				//UpdateConsistencyOfLaunchNativeKernel();
        break;
			case HUM_COMMAND_CUDA_KERNEL:
			case HUM_COMMAND_CUDA_DIRECT_KERNEL:
        UpdateConsistencyOfLaunchKernel();
        //resolved = ResolveConsistencyOfLaunchKernel();
				resolved = 1;
				break;
      default:
        break;
    }
  }

	HUM_DEV("ResolveConsistency Cmd(%p) type_=%x end", this, type_);
  return resolved;
}


bool HUMKernelCommand::ResolveConsistencyOfLaunchKernel() 
{
	HUM_DEV("%s func begin", __func__);
  bool already_resolved = true;
  for (std::map<hum_uint, HUMKernelArg*>::iterator it = kernel_args_->begin();
       it != kernel_args_->end();
       ++it) {
    HUMKernelArg* arg = it->second;
    if (arg->mem != NULL)
      already_resolved &= LocateMemOnDevice(arg->mem);
  }
  consistency_resolved_ = true;

	HUM_DEV("%s func end", __func__);
  return already_resolved;
}

void HUMKernelCommand::UpdateConsistencyOfLaunchKernel() {

	HUM_DEV("%s func begin", __func__);
  for (std::map<hum_uint, HUMKernelArg*>::iterator it = kernel_args_->begin();
       it != kernel_args_->end();
       ++it) {
    HUMKernelArg* arg = it->second;
    if (arg->mem != NULL)
      AccessMemOnDevice(arg->mem, arg->mem->IsWritable());
  }

	HUM_DEV("%s func end", __func__);
}


void HUMKernelCommand::Execute()
{
	HUM_DEV("HUMKernelCommand %p Execute type: %x begin", this, type_);
	switch (type_) {
	case HUM_COMMAND_NDRANGE_KERNEL:
	case HUM_COMMAND_TASK:
		device_->LaunchKernel(this, kernel_, work_dim_, gwo_, gws_, lws_, nwg_,
				kernel_args_);
		break;
	case HUM_COMMAND_NATIVE_KERNEL:
		assert(0);
		//device_->LaunchNativeKernel(this, user_func_, native_args_, size_,
		//                            num_mem_objects_, mem_list_, mem_offsets_);
		break;
	case HUM_COMMAND_CUDA_KERNEL:
		device_->LaunchCudaKernel(this, cuda_kernel_name_.c_str(), cuda_kernel_func_, work_dim_, gwo_, gws_, lws_, nwg_,
				kernel_args_);
		break;
	case HUM_COMMAND_CUDA_DIRECT_KERNEL:
		device_->LaunchCudaDirectKernel(this, cuda_kernel_name_.c_str(), cuda_kernel_func_, work_dim_, gwo_, gws_, lws_, nwg_,
				kernel_args_);
		break;
	default:
		assert(0);
		HUM_ERROR("Unsupported command [%x]", type_);
		break;
	}
	HUM_DEV("HUMKernelCommand %p Execute type: %x end", this, type_);
}

HUMDriverCommand::HUMDriverCommand(HUMContext* context,
		HUMDevice* device, HUMCommandQueue* queue,
		hum_command_type type, hum_int func_type, void* data, size_t data_size)
: HUMCommand(context, device, queue, type)
{
	category_ = HUM_COMMAND_CATEGORY_DRIVER;
	func_type_ = func_type;
	data_ = malloc(data_size);
	memcpy(data_, data, data_size);
	data_size_ = data_size;
}

HUMDriverCommand::~HUMDriverCommand()
{
	if(data_) {
		free(data_);
	}
}

void HUMDriverCommand::Execute()
{
  switch (type_) {
    case HUM_COMMAND_DRIVER:
			{
				device_->ExecuteFunc(this, func_type_, data_, data_size_);
			}
      break;
    default:
			assert(0);
      HUM_ERROR("Unsupported command [%x]", type_);
      break;
  }
}
