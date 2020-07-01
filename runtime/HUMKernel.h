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

#ifndef __HUM_KERNEL_H__
#define __HUM_KERNEL_H__

#include <map>
#include <list>
#include <pthread.h>
#include "HUMObject.h"

class HUMContext;
class HUMDevice;
class HUMKernelInfo;
class HUMMem;
class HUMSampler;

class HUMKernelArg {
public:
	HUMKernelArg();
	HUMKernelArg(HUMContext* context, size_t arg_size, const void* arg_value, size_t cuda_offset);
	~HUMKernelArg();

  size_t size;
  char value[2048];
  bool local; //Is it local mem?
  HUMMem* mem;
	size_t cuda_offset;	//offset for cuda
  /*HUMSampler* sampler;*/
  hum_mem_flags flags;

	int location; //for struct members
	std::list<HUMKernelArg*> struct_members;

	void BuildValue(HUMDevice* device);
	void Copy(HUMKernelArg* karg);
};

extern HUMKernelArg* CreateKernelArg(HUMContext* context, size_t arg_size, const void* arg_value, size_t cuda_offset, int location);
extern HUMKernelArg* CreateKernelArg(HUMKernelArg* karg);

class HUMKernel: public HUMObject<HUMKernel>
{
public:
  HUMKernel(HUMContext* context, HUMKernelInfo* kernel_info);
  virtual void Cleanup();
  virtual ~HUMKernel();

  HUMContext* context() const { return context_; }
  const char* name() const { return name_; }
  hum_uint num_args() const;

  hum_int GetKernelInfo(hum_kernel_info param_name, size_t param_value_size,
                       void* param_value, size_t* param_value_size_ret);
  hum_int GetKernelWorkGroupInfo(HUMDevice* device,
                                hum_kernel_work_group_info param_name,
                                size_t param_value_size, void* param_value,
                                size_t* param_value_size_ret);
  hum_int GetKernelArgInfo(hum_uint arg_index, hum_kernel_arg_info param_name,
                          size_t param_value_size, void* param_value,
                          size_t* param_value_size_ret);

  hum_int SetKernelArg(hum_uint arg_index, HUMKernelArg* arg);
  hum_int SetKernelArg(hum_uint arg_index, size_t arg_size, const void* arg_value);
  
	bool IsArgsDirty() { return args_dirty_; }

  std::map<hum_uint, HUMKernelArg*>* ExportArgs();

  bool HasDevSpecific(HUMDevice* device);
  void* GetDevSpecific(HUMDevice* device);

 private:
  HUMContext* context_;
  HUMKernelInfo* kernel_info_;

  const char* name_;
  std::map<hum_uint, HUMKernelArg*> args_;
  bool args_dirty_;

  std::map<HUMDevice*, void*> dev_specific_;

  pthread_mutex_t mutex_dev_specific_;
};

#endif //__HUM_KERNEL_H__
