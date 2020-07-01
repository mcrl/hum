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

#include "HUMKernel.h"
#include "HUMContext.h"
#include "HUMKernelInfo.h"
#include "HUMDevice.h"
#include "HUMMem.h"
#include <map>
#include <assert.h>

HUMKernel::HUMKernel(HUMContext* context, HUMKernelInfo* kernel_info) {
  context_ = context;
  context_->Retain();
  kernel_info_ = kernel_info;

  name_ = kernel_info->name();
  args_dirty_ = false;

  pthread_mutex_init(&mutex_dev_specific_, NULL);
}

void HUMKernel::Cleanup() {
  for (std::map<HUMDevice*, void*>::iterator it = dev_specific_.begin();
       it != dev_specific_.end();
       ++it) {
    (it->first)->FreeKernel(this, it->second);
  }
}

HUMKernel::~HUMKernel() {
  for (std::map<hum_uint, HUMKernelArg*>::iterator it = args_.begin();
       it != args_.end();
       ++it) {
    free(it->second);
  }
  context_->Release();

  pthread_mutex_destroy(&mutex_dev_specific_);
}

hum_uint HUMKernel::num_args() const {
  return kernel_info_->num_args();
}

hum_int HUMKernel::GetKernelInfo(hum_kernel_info param_name,
                               size_t param_value_size, void* param_value,
                               size_t* param_value_size_ret) {
  switch (param_name) {
    GET_OBJECT_INFO_S(HUM_KERNEL_FUNCTION_NAME, name_);
    GET_OBJECT_INFO_T(HUM_KERNEL_REFERENCE_COUNT, hum_uint, ref_cnt());
/*  
 *  TODO
 *  GET_OBJECT_INFO_T(CL_KERNEL_CONTEXT, cl_context, context_->st_obj());
    GET_OBJECT_INFO_T(CL_KERNEL_PROGRAM, cl_program, program_->st_obj());
		*/
    GET_OBJECT_INFO_T(HUM_KERNEL_NUM_ARGS, hum_uint, kernel_info_->num_args());
    GET_OBJECT_INFO_S(HUM_KERNEL_ATTRIBUTES, kernel_info_->attributes());
    default: return HUM_INVALID_VALUE;
  }
  return HUM_SUCCESS;
}

hum_int HUMKernel::GetKernelWorkGroupInfo(HUMDevice* device,
                                        hum_kernel_work_group_info param_name,
                                        size_t param_value_size,
                                        void* param_value,
                                        size_t* param_value_size_ret) {
  return kernel_info_->GetKernelWorkGroupInfo(device, param_name,
                                              param_value_size, param_value,
                                              param_value_size_ret);
}

hum_int HUMKernel::GetKernelArgInfo(hum_uint arg_index,
                                  hum_kernel_arg_info param_name,
                                  size_t param_value_size, void* param_value,
                                  size_t* param_value_size_ret) {
  return kernel_info_->GetKernelArgInfo(arg_index, param_name,
                                        param_value_size, param_value,
                                        param_value_size_ret);
}

hum_int HUMKernel::SetKernelArg(hum_uint arg_index, 
			HUMKernelArg* _arg)
{

	HUMKernelArg* arg;
	std::map<hum_uint, HUMKernelArg*>::iterator old = args_.find(arg_index);
  if (old != args_.end()) {
    arg = old->second;
		delete(arg);
  } 
	args_[arg_index] = _arg;
  args_dirty_ = true;
	return HUM_SUCCESS;
}

hum_int HUMKernel::SetKernelArg(hum_uint arg_index, 
		size_t arg_size,
    const void* arg_value) {
  if (arg_size > 256 && arg_value != NULL)
    return HUM_INVALID_ARG_SIZE;

  HUMKernelArg* arg;
	std::map<hum_uint, HUMKernelArg*>::iterator old = args_.find(arg_index);
  if (old != args_.end()) {
    arg = old->second;
  } 
	else {
    arg = new HUMKernelArg();
    if (arg == NULL)
      return HUM_OUT_OF_HOST_MEMORY;
    args_[arg_index] = arg;
  }
	arg->cuda_offset = 0;
  arg->size = arg_size;
  if (arg_value != NULL) {
    memcpy(arg->value, arg_value, arg_size);
    arg->local = false;
    hum_mem_handle m = *((hum_mem_handle*)arg_value);
    arg->mem = context_->IsValidMem(m) ? m->c_obj : NULL;
  } 
	else {
    arg->local = true;
    arg->mem = NULL;
  }

  args_dirty_ = true;

  return HUM_SUCCESS;
}


std::map<hum_uint, HUMKernelArg*>* HUMKernel::ExportArgs() 
{
	std::map<hum_uint, HUMKernelArg*>* new_args = new std::map<hum_uint, HUMKernelArg*>();
  for (std::map<hum_uint, HUMKernelArg*>::iterator it = args_.begin();
       it != args_.end();
       ++it) {
    HUMKernelArg* arg = it->second;
    HUMKernelArg* new_arg = new HUMKernelArg();
    new_arg->size = arg->size;
    if (!arg->local)
      memcpy(new_arg->value, arg->value, arg->size);
    new_arg->local = arg->local;
    new_arg->mem = arg->mem;
    if (new_arg->mem != NULL)
      new_arg->mem->Retain();
		/*
    new_arg->sampler = arg->sampler;
    if (new_arg->sampler != NULL)
      new_arg->sampler->Retain();
		*/
    (*new_args)[it->first] = new_arg;
  }
  return new_args;
}

bool HUMKernel::HasDevSpecific(HUMDevice* device) 
{
  pthread_mutex_lock(&mutex_dev_specific_);
  bool alloc = (dev_specific_.count(device) > 0);
  pthread_mutex_unlock(&mutex_dev_specific_);
  return alloc;
}

void* HUMKernel::GetDevSpecific(HUMDevice* device) 
{
  pthread_mutex_lock(&mutex_dev_specific_);
  void* dev_specific;
  if (dev_specific_.count(device) > 0) {
    dev_specific = dev_specific_[device];
  } 
	else {
    pthread_mutex_unlock(&mutex_dev_specific_);
    dev_specific = device->AllocKernel(this);
    pthread_mutex_lock(&mutex_dev_specific_);
    dev_specific_[device] = dev_specific;
  }
  pthread_mutex_unlock(&mutex_dev_specific_);
  return dev_specific;
}

HUMKernelArg::HUMKernelArg()
{
	size = 0;
	local = false;
	mem = NULL;
	cuda_offset = 0;
	flags = 0;
	location = 0;
}

HUMKernelArg::HUMKernelArg(HUMContext* context, size_t arg_size, const void* arg_value, size_t _cuda_offset)
{
	location = 0;
	cuda_offset = 0;
  size = arg_size;
  if (arg_value != NULL) {
    memcpy(value, arg_value, arg_size);
    local = false;
		if(context) {
	    hum_mem_handle m = *((hum_mem_handle*)arg_value);
		  mem = context->IsValidMem(m) ? m->c_obj : NULL;
		}
		else {
			mem = NULL;
		}
		if(mem) mem->Retain();
		cuda_offset = _cuda_offset;
  } 
	else {
    local = true;
    mem = NULL;
  }
}


HUMKernelArg::~HUMKernelArg() {
	if (mem) 
		mem->Release();

	std::list<HUMKernelArg*>::iterator it = struct_members.begin();
	for(;it != struct_members.end();it++) {
		HUMKernelArg* arg = *it;
		delete(arg);
	}
	struct_members.clear();
}

void HUMKernelArg::BuildValue(HUMDevice* device) {
	std::list<HUMKernelArg*>::iterator it = struct_members.begin();
	for(;it != struct_members.end();it++) {
		HUMKernelArg* sub_arg = *it;

		if(sub_arg->mem != NULL) {
			HUMMem* mem = sub_arg->mem;
			*(uint64_t*)&value[sub_arg->location] = ((uint64_t)mem->GetDevSpecific(device) + sub_arg->cuda_offset);

			HUM_DEV("BuildValue loc=%d, <- %p", sub_arg->location, (void*)((uint64_t)mem->GetDevSpecific(device) + sub_arg->cuda_offset));

		}
	}
}


HUMKernelArg* CreateKernelArg(HUMContext* context, size_t arg_size, const void* arg_value, size_t cuda_offset, int location)
{
	//assert(context);
  if (arg_size > 2048 && arg_value != NULL) {
		HUM_ERROR("Invalid Argument size = %ld\n", arg_size);
		return NULL;
	}

  HUMKernelArg* arg = new HUMKernelArg(context, arg_size, arg_value, cuda_offset);
	if (arg == NULL) {
		HUM_ERROR("%s", "HUM OUT OF HOST MEMORY\n");
		return NULL;
	}
	arg->location = location;

	HUM_DEV("%p offset = %ld", arg, arg->cuda_offset);
	return arg;
}

HUMKernelArg* CreateKernelArg(HUMKernelArg* karg)
{
	HUMKernelArg* arg = new HUMKernelArg();
	arg->size = karg->size;
	memcpy(arg->value, karg->value, arg->size);
	arg->local = karg->local;
	arg->mem = karg->mem;
	if(arg->mem)
		arg->mem->Retain();
	arg->cuda_offset = karg->cuda_offset;
	arg->flags = karg->flags;
	arg->location = karg->location;

	std::list<HUMKernelArg*>::iterator it = karg->struct_members.begin();
	for(;it != karg->struct_members.end();it++) {
		HUMKernelArg* sub_karg = *it;
		HUMKernelArg* sub_arg = CreateKernelArg(sub_karg);
		arg->struct_members.push_back(sub_arg);
	}
	return arg;
}

