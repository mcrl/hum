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

#ifndef __HUM_KERNEL_INFO_H__
#define __HUM_KERNEL_INFO_H__

#include "hum.h"
#include <set>
#include <map>
#include <string.h>
#include <stdlib.h>

class HUMDevice;

class HUMKernelInfo {
 public:
  HUMKernelInfo(const char* name, hum_uint num_args, const char* attributes);
  virtual ~HUMKernelInfo();

  const char* name() const { return name_; }
  hum_uint num_args() const { return num_args_; }
  const char* attributes() const { return attributes_; }
  int hum_kernel_id() const { return hum_kernel_id_; }

  hum_int GetKernelWorkGroupInfo(HUMDevice* device,
                                hum_kernel_work_group_info param_name,
                                size_t param_value_size, void* param_value,
                                size_t* param_value_size_ret);
  hum_int GetKernelArgInfo(hum_uint arg_index, hum_kernel_arg_info param_name,
                          size_t param_value_size, void* param_value,
                          size_t* param_value_size_ret);

  bool IsValid() const { return valid_; }

  void Update(const char* name, hum_uint num_args, const char* attributes);
  void SetWorkGroupInfo(HUMDevice* device, size_t work_group_size,
                        size_t compile_work_group_size[3],
                        hum_ulong local_mem_size,
                        size_t preferred_work_group_size_multiple,
                        hum_ulong private_mem_size);
  void SetArgInfo(hum_uint arg_index,
                  hum_kernel_arg_address_qualifier arg_address_qualifier,
                  hum_kernel_arg_access_qualifier arg_access_qualifier,
                  const char* arg_type_name,
                  hum_kernel_arg_type_qualifier arg_type_qualifier,
                  const char* arg_name);
  void SetKernelID(int kernel_id);
  void Invalidate();

  size_t GetSerializationSize(HUMDevice* device);
  void* SerializeKernelInfo(void* buffer, HUMDevice* device);
  // Deserialization is implemented in Program

 private:
  char* name_;
  hum_uint num_args_;
  char* attributes_;

  bool valid_;
  std::set<HUMDevice*> devices_;
  std::set<hum_uint> indexes_;

  std::map<HUMDevice*, size_t> work_group_size_;
  size_t compile_work_group_size_[3];
  std::map<HUMDevice*, hum_ulong> local_mem_size_;
  std::map<HUMDevice*, size_t> preferred_work_group_size_multiple_;
  std::map<HUMDevice*, hum_ulong> private_mem_size_;

  hum_kernel_arg_address_qualifier* arg_address_qualifiers_;
  hum_kernel_arg_access_qualifier* arg_access_qualifiers_;
  char** arg_type_names_;
  hum_kernel_arg_type_qualifier* arg_type_qualifiers_;
  char** arg_names_;

  int hum_kernel_id_;
};


#endif //__HUM_KERNEL_INFO_H__

