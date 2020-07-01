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

#include "HUMKernelInfo.h"
#include "Utils.h"

HUMKernelInfo::HUMKernelInfo(const char* name, hum_uint num_args,
                           const char* attributes) {
  name_ = (char*)malloc(sizeof(char) * (strlen(name) + 1));
  strcpy(name_, name);
  num_args_ = num_args;
  attributes_ = (char*)malloc(sizeof(char) * (strlen(attributes) + 1));
  strcpy(attributes_, attributes);
  valid_ = true;

  arg_address_qualifiers_ =
    (hum_kernel_arg_address_qualifier*)malloc(
      sizeof(hum_kernel_arg_address_qualifier) * num_args_);
  arg_access_qualifiers_ =
    (hum_kernel_arg_access_qualifier*)malloc(
      sizeof(hum_kernel_arg_access_qualifier) * num_args_);
  arg_type_names_ = (char**)malloc(sizeof(char*) * num_args_);
  for (hum_uint i = 0; i < num_args_; i++)
    arg_type_names_[i] = NULL;
  arg_type_qualifiers_ =
    (hum_kernel_arg_type_qualifier*)malloc(
      sizeof(hum_kernel_arg_type_qualifier) * num_args_);
  arg_names_ = (char**)malloc(sizeof(char*) * num_args_);
  for (hum_uint i = 0; i < num_args_; i++)
    arg_names_[i] = NULL;

  hum_kernel_id_ = -1;
}

HUMKernelInfo::~HUMKernelInfo() {
  free(name_);
  free(attributes_);
  free(arg_address_qualifiers_);
  free(arg_access_qualifiers_);
  for (hum_uint i = 0; i < num_args_; i++)
    if (arg_type_names_[i] != NULL)
      free(arg_type_names_[i]);
  free(arg_type_names_);
  free(arg_type_qualifiers_);
  for (hum_uint i = 0; i < num_args_; i++)
    if (arg_names_[i] != NULL)
      free(arg_names_[i]);
  free(arg_names_);
}

hum_int HUMKernelInfo::GetKernelWorkGroupInfo(
    HUMDevice* device, hum_kernel_work_group_info param_name,
    size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
  if (device == NULL) {
    if (devices_.size() > 1)
      return HUM_INVALID_DEVICE;
    device = *devices_.begin();
  } else {
    if (devices_.count(device) == 0)
      return HUM_INVALID_DEVICE;
  }
  switch (param_name) {
    case HUM_KERNEL_GLOBAL_WORK_SIZE:
      return HUM_INVALID_VALUE;
    GET_OBJECT_INFO_T(HUM_KERNEL_WORK_GROUP_SIZE, size_t,
                      work_group_size_[device]);
    GET_OBJECT_INFO_A(HUM_KERNEL_COMPILE_WORK_GROUP_SIZE, size_t,
                      compile_work_group_size_, 3);
    GET_OBJECT_INFO_T(HUM_KERNEL_LOCAL_MEM_SIZE, hum_ulong,
                      local_mem_size_[device]);
    GET_OBJECT_INFO_T(HUM_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_t,
                      preferred_work_group_size_multiple_[device]);
    GET_OBJECT_INFO_T(HUM_KERNEL_PRIVATE_MEM_SIZE, hum_ulong,
                      private_mem_size_[device]);
    default: return HUM_INVALID_VALUE;
  }
  return HUM_SUCCESS;
}

hum_int HUMKernelInfo::GetKernelArgInfo(hum_uint arg_index,
                                      hum_kernel_arg_info param_name,
                                      size_t param_value_size,
                                      void* param_value,
                                      size_t* param_value_size_ret) {
  if (indexes_.count(arg_index) == 0)
    return HUM_KERNEL_ARG_INFO_NOT_AVAILABLE;
  switch (param_name) {
    GET_OBJECT_INFO(HUM_KERNEL_ARG_ADDRESS_QUALIFIER,
                    hum_kernel_arg_address_qualifier,
                    arg_address_qualifiers_[arg_index]);
    GET_OBJECT_INFO(HUM_KERNEL_ARG_ACCESS_QUALIFIER,
                    hum_kernel_arg_access_qualifier,
                    arg_access_qualifiers_[arg_index]);
    GET_OBJECT_INFO_S(HUM_KERNEL_ARG_TYPE_NAME, arg_type_names_[arg_index]);
    GET_OBJECT_INFO(HUM_KERNEL_ARG_TYPE_QUALIFIER, hum_kernel_arg_type_qualifier,
                    arg_type_qualifiers_[arg_index]);
    GET_OBJECT_INFO_S(HUM_KERNEL_ARG_NAME, arg_names_[arg_index]);
  }
  return HUM_SUCCESS;
}

void HUMKernelInfo::Update(const char* name, hum_uint num_args,
                          const char* attributes) {
  if (strcmp(name_, name) != 0 || num_args_ != num_args ||
      strcmp(attributes_, attributes) != 0)
    valid_ = false;
}

void HUMKernelInfo::SetWorkGroupInfo(HUMDevice* device, size_t work_group_size,
                                    size_t compile_work_group_size[3],
                                    hum_ulong local_mem_size,
                                    size_t preferred_work_group_size_multiple,
                                    hum_ulong private_mem_size) {
  if (devices_.count(device) > 0) {
/*
 * Currently we don't invalidate the kernel info because OpenCL compilers of
 * different vendors may return different results.
 */
#if 0
    if (work_group_size_[device] != work_group_size ||
        compile_work_group_size_[0] != compile_work_group_size[0] ||
        compile_work_group_size_[1] != compile_work_group_size[1] ||
        compile_work_group_size_[2] != compile_work_group_size[2] ||
        local_mem_size_[device] != local_mem_size ||
        preferred_work_group_size_multiple_[device] !=
            preferred_work_group_size_multiple ||
        private_mem_size_[device] != private_mem_size) {
      valid_ = false;
    }
#endif
    return;
  }
  devices_.insert(device);
  work_group_size_[device] = work_group_size;
  memcpy(compile_work_group_size_, compile_work_group_size,
         sizeof(size_t) * 3);
  local_mem_size_[device] = local_mem_size;
  preferred_work_group_size_multiple_[device] =
      preferred_work_group_size_multiple;
  private_mem_size_[device] = private_mem_size;
}

void HUMKernelInfo::SetArgInfo(
    hum_uint arg_index, hum_kernel_arg_address_qualifier arg_address_qualifier,
    hum_kernel_arg_access_qualifier arg_access_qualifier,
    const char* arg_type_name, hum_kernel_arg_type_qualifier arg_type_qualifier,
    const char* arg_name) {
  if (indexes_.count(arg_index) > 0) {
/*
 * Currently we don't invalidate the kernel info because OpenCL compilers of
 * different vendors may return different results.
 */
#if 0
    if (arg_address_qualifiers_[arg_index] != arg_address_qualifier ||
        arg_access_qualifiers_[arg_index] != arg_access_qualifier ||
        strcmp(arg_type_names_[arg_index], arg_type_name) != 0 ||
        arg_type_qualifiers_[arg_index] != arg_type_qualifier ||
        strcmp(arg_names_[arg_index], arg_name) != 0) {
      valid_ = false;
    }
#endif
    return;
  }
  indexes_.insert(arg_index);
  arg_address_qualifiers_[arg_index] = arg_address_qualifier;
  arg_access_qualifiers_[arg_index] = arg_access_qualifier;
  arg_type_names_[arg_index] =
      (char*)malloc(sizeof(char) * (strlen(arg_type_name) + 1));
  strcpy(arg_type_names_[arg_index], arg_type_name);
  arg_type_qualifiers_[arg_index] = arg_type_qualifier;
  arg_names_[arg_index] =
      (char*)malloc(sizeof(char) * (strlen(arg_name) + 1));
  strcpy(arg_names_[arg_index], arg_name);
}

void HUMKernelInfo::SetKernelID(int kernel_id) 
{
  hum_kernel_id_ = kernel_id;
}

void HUMKernelInfo::Invalidate() {
  valid_ = false;
}

size_t HUMKernelInfo::GetSerializationSize(HUMDevice* device) {
  size_t size = 0;

#define SERIALIZE_INFO(type, value) size += sizeof(type);
#define SERIALIZE_INFO_T(type, value) size += sizeof(type);
#define SERIALIZE_INFO_A(type, value, length) size += sizeof(type) * length;
#define SERIALIZE_INFO_S(value) size += strlen(value) + 1;

  SERIALIZE_INFO_S(name_);
  SERIALIZE_INFO(hum_uint, num_args_);
  SERIALIZE_INFO_S(attributes_);
  SERIALIZE_INFO_T(size_t, work_group_size_[device]);
  SERIALIZE_INFO_A(size_t, compile_work_group_size_[device], 3);
  SERIALIZE_INFO_T(hum_ulong, local_mem_size_[device]);
  SERIALIZE_INFO_T(size_t, preferred_work_group_size_multiple_[device]);
  SERIALIZE_INFO_T(hum_ulong, private_mem_size_[device]);
  for (hum_uint i = 0; i < num_args_; i++) {
    if (indexes_.count(i) > 0) {
      SERIALIZE_INFO(hum_uint, i);
      SERIALIZE_INFO(hum_kernel_arg_address_qualifier,
                     arg_address_qualifiers_[i]);
      SERIALIZE_INFO(hum_kernel_arg_access_qualifier,
                     arg_access_qualifiers_[i]);
      SERIALIZE_INFO_S(arg_type_names_[i]);
      SERIALIZE_INFO(hum_kernel_arg_type_qualifier, arg_type_qualifiers_[i]);
      SERIALIZE_INFO_S(arg_names_[i]);
    }
  }
  SERIALIZE_INFO_T(hum_uint, num_args_);
  SERIALIZE_INFO(bool, valid_);

#undef SERIALIZE_INFO
#undef SERIALIZE_INFO_T
#undef SERIALIZE_INFO_A
#undef SERIALIZE_INFO_S

  return size;
}

void* HUMKernelInfo::SerializeKernelInfo(void* buffer, HUMDevice* device) {
  char* p = (char*)buffer;

#define SERIALIZE_INFO(type, value) \
  memcpy(p, &value, sizeof(type));  \
  p += sizeof(type);
#define SERIALIZE_INFO_T(type, value) \
  {                                   \
    type temp = value;                \
    memcpy(p, &temp, sizeof(type));   \
    p += sizeof(type);                \
  }
#define SERIALIZE_INFO_A(type, value, length) \
  memcpy(p, value, sizeof(type) * length);    \
  p += sizeof(type) * length;
#define SERIALIZE_INFO_S(value)    \
  {                                \
    size_t length = strlen(value); \
    strcpy(p, value);              \
    p += length + 1;               \
  }

  SERIALIZE_INFO_S(name_);
  SERIALIZE_INFO(hum_uint, num_args_);
  SERIALIZE_INFO_S(attributes_);
  SERIALIZE_INFO_T(size_t, work_group_size_[device]);
  SERIALIZE_INFO_A(size_t, compile_work_group_size_, 3);
  SERIALIZE_INFO_T(hum_ulong, local_mem_size_[device]);
  SERIALIZE_INFO_T(size_t, preferred_work_group_size_multiple_[device]);
  SERIALIZE_INFO_T(hum_ulong, private_mem_size_[device]);
  for (hum_uint i = 0; i < num_args_; i++) {
    if (indexes_.count(i) > 0) {
      SERIALIZE_INFO(hum_uint, i);
      SERIALIZE_INFO(hum_kernel_arg_address_qualifier,
                     arg_address_qualifiers_[i]);
      SERIALIZE_INFO(hum_kernel_arg_access_qualifier,
                     arg_access_qualifiers_[i]);
      SERIALIZE_INFO_S(arg_type_names_[i]);
      SERIALIZE_INFO(hum_kernel_arg_type_qualifier, arg_type_qualifiers_[i]);
      SERIALIZE_INFO_S(arg_names_[i]);
    }
  }
  SERIALIZE_INFO_T(hum_uint, num_args_);
  SERIALIZE_INFO(bool, valid_);

#undef SERIALIZE_INFO
#undef SERIALIZE_INFO_T
#undef SERIALIZE_INFO_A
#undef SERIALIZE_INFO_S

  return p;
}
