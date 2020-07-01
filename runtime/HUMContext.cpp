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

#include "HUMContext.h"
#include "HUMDevice.h"
#include "HUMMem.h"
#include <algorithm>

using namespace std;

HUMContext::HUMContext(const std::vector<HUMDevice*>& devices, 
				size_t num_properties,
        const hum_context_properties* properties)
: devices_(devices)
{
  for (vector<HUMDevice*>::iterator it = devices_.begin();
       it != devices_.end();
       ++it) {
    (*it)->Retain();
  }

  num_properties_ = num_properties;
  if (num_properties > 0) {
    properties_ = (hum_context_properties*)malloc(
        sizeof(hum_context_properties) * num_properties);
    memcpy(properties_, properties,
           sizeof(hum_context_properties) * num_properties);
  } 
	else {
    properties_ = NULL;
  }

	callback_ = NULL;

  pthread_mutex_init(&mutex_mems_, NULL);
}

HUMContext::~HUMContext()
{
  for (vector<HUMDevice*>::iterator it = devices_.begin();
       it != devices_.end();
       ++it) {
    (*it)->Release();
  }
  if (properties_ != NULL)
    free(properties_);
	if(callback_ != NULL)
		delete callback_;

  pthread_mutex_destroy(&mutex_mems_);
}

bool HUMContext::IsValidDevice(HUMDevice* device) {
  return (find(devices_.begin(), devices_.end(), device) != devices_.end());
}

bool HUMContext::IsValidDevices(hum_uint num_devices,
                               const hum_device_handle* device_list) {
  for (hum_uint i = 0; i < num_devices; i++) {
    if (device_list[i] == NULL || !IsValidDevice(device_list[i]->c_obj))
      return false;
  }
  return true;
}

bool HUMContext::IsValidMem(hum_mem_handle mem) {
  bool valid = false;
  pthread_mutex_lock(&mutex_mems_);
  for (vector<HUMMem*>::iterator it = mems_.begin();
       it != mems_.end();
       ++it) {
    if ((*it)->get_handle() == mem) {
      valid = true;
      break;
    }
  }
  pthread_mutex_unlock(&mutex_mems_);
  return valid;
}

void HUMContext::AddMem(HUMMem* mem)
{
	pthread_mutex_lock(&mutex_mems_);
	mems_.push_back(mem);
	pthread_mutex_unlock(&mutex_mems_);
}

void HUMContext::RemoveMem(HUMMem* mem)
{
	pthread_mutex_lock(&mutex_mems_);
	std::vector<HUMMem*>::iterator it = std::find(mems_.begin(), mems_.end(), mem);
	if(it != mems_.end())
		mems_.erase(it);
	pthread_mutex_unlock(&mutex_mems_);
}

void HUMContext::SetErrorNotificationCallback(
    ContextErrorNotificationCallback* callback) 
{
  if (callback_ != NULL)
    delete callback_;
  callback_ = callback;
}

void HUMContext::NotifyError(const char* errinfo, 
		const void* private_info, size_t cb) 
{
  if (callback_ != NULL)
    callback_->run(errinfo, private_info, cb);
}

