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

#include "HUMPlatform.h"
#include <iostream>
#include <algorithm>
#include <signal.h>
#include <malloc.h>
#include <dlfcn.h>

#include "HUMContext.h"
#include "HUMCommandQueue.h"
#include "HUMScheduler.h"
#include "HUMIssuer.h"
#include "Callbacks.h"
#include <assert.h>
#include "Device/LocalCuDevice.h"

using namespace std;

HUMPlatform* HUMPlatform::singleton_ = NULL;
mutex_t HUMPlatform::mutex_;
extern int grank_;

#ifdef USE_MEM_PREFETCH
//extern "C" void *__libc_malloc(size_t size);
//void *malloc(size_t size) {
//  return memalign(4096, size);
//}

extern "C" void __libc_free(void* ptr);
void free(void* ptr) {
  char* buf = (char*)ptr;
  if (ptr != NULL) {
    // write dummy value to buffer so that SIGSEGV handler can catch it
    buf[0] = 1;
  }
  __libc_free(ptr);
}

void segv_handler(int signal_number, siginfo_t *info, void *ucontext) {
  // continuously sleep until memory copy is finished
  usleep(1);
}
#endif

HUMPlatform* HUMPlatform::GetPlatform(int rank) {
	bool init = false;

	mutex_.lock();
	if (singleton_ == NULL) {
		init = true;
		assert(rank >= 0);
		singleton_ = new HUMPlatform();
	}
	mutex_.unlock();

	if(init)
		singleton_->Init();


	return singleton_;
}

CUDAPlatform* HUMPlatform::GetCudaPlatform(int rank) {
	bool init = false;
	mutex_.lock();

	if (singleton_ == NULL) {

		init = true;
		singleton_ = new CUDAPlatform();
	}
	mutex_.unlock();

	if(init) {
		if(grank_ == 0) {
			((CUDAPlatform*)singleton_)->InitCuda();
		}
		else {
			singleton_->Init();
			hum_run();
		}
	}

	return (CUDAPlatform*)singleton_;
}

HUMPlatform::HUMPlatform() {
	profile_ = "FULL_PROFILE";
	version_ = "HUM 1.0 rev01";
	name_ = "HUM (Hidden Unified Memory)";
	vendor_ = "Seoul National University";
	extensions_ = "";
	suffix_= "HUM";

	default_device_type_ = HUM_DEVICE_TYPE_GPU;

	is_host_ = true;
	is_cuda_ = false;

  pthread_mutex_init(&mutex_devices_, NULL);
  pthread_mutex_init(&mutex_issuers_, NULL);

	HUM_DEV("Create New HUMPlatform %p", this);
}

HUMPlatform::~HUMPlatform() {
  for (vector<HUMDevice*>::iterator it = devices_.begin();
       it != devices_.end();
       ++it) {
    delete (*it);
  }
  for (vector<HUMScheduler*>::iterator it = schedulers_.begin();
       it != schedulers_.end();
       ++it) {
    delete (*it);
  }
  for (vector<HUMIssuer*>::iterator it = issuers_.begin();
       it != issuers_.end();
       ++it) {
    delete (*it);
  }
  pthread_mutex_destroy(&mutex_devices_);
  pthread_mutex_destroy(&mutex_issuers_);
}

void HUMPlatform::Init() {
	HUM_DEV("Host Platform(rank %d):%p Initializing....", 0, this);
	
	// SCHEDULER_MULTIPLE
	InitSchedulers(4, false);
	// ISSUER_GLOBAL
	//AddIssuer(new HUMIssuer(false));

	LocalCuDevice::CreateDevices();	//SINGLE NODE MODE

#ifdef HUM_DEBUG
  unsigned int num_cpu = 0;
  unsigned int num_gpu = 0;
  unsigned int num_accelerator = 0;
  unsigned int num_custom = 0;
	unsigned int num_cuda = 0;
	unsigned int num_opencl = 0;


  for (std::vector<HUMDevice*>::const_iterator it = devices_.begin();
       it != devices_.end();
       ++it) {
    HUMDevice* device = *it;
    switch (device->type()) {
      case HUM_DEVICE_TYPE_CPU:
        num_cpu++;
        break;
      case HUM_DEVICE_TYPE_GPU:
        num_gpu++;
        break;
      case HUM_DEVICE_TYPE_ACCELERATOR:
        num_accelerator++;
        break;
      case HUM_DEVICE_TYPE_CUSTOM:
        num_custom++;
        break;
      default:
        HUM_ERROR("Unsupported device type [%lx]", device->type());
    }
    switch (device->model()) {
      case HUM_MODEL_TYPE_OPENCL:
        num_opencl++;
        break;
      case HUM_MODEL_TYPE_CUDA:
        num_cuda++;
        break;
			default:
        HUM_ERROR("Unsupported model type [%x]", device->model());
    }

  }

  HUM_INFO("%s", "HUM platform has been initialized.");
  HUM_INFO("Total %lu devices (%u CPUs, %u GPUs, %u accelerators, "
             "and %u custom devices) (%u CUDAs, %u OpenCLs) are in the platform.",
             devices_.size(), num_cpu, num_gpu, num_accelerator, num_custom,
						 num_cuda, num_opencl);
#endif // SNUCL_DEBUG

#ifdef USE_MEM_PREFETCH
  // install handler for SIGSEGV
  struct sigaction sa;
  memset(&sa, 0, sizeof (sa));
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = &segv_handler;
  sigaction(SIGSEGV, &sa, NULL);
#endif

}

hum_int HUMPlatform::GetPlatformInfo(hum_platform_info param_name,
                                   size_t param_value_size,
                                   void* param_value,
                                   size_t* param_value_size_ret) {
  switch (param_name) {
    GET_OBJECT_INFO_S(HUM_PLATFORM_PROFILE, profile_);
    GET_OBJECT_INFO_S(HUM_PLATFORM_VERSION, version_);
    GET_OBJECT_INFO_S(HUM_PLATFORM_NAME, name_);
    GET_OBJECT_INFO_S(HUM_PLATFORM_VENDOR, vendor_);
    GET_OBJECT_INFO_S(HUM_PLATFORM_EXTENSIONS, extensions_);
    //GET_OBJECT_INFO_S(HUM_PLATFORM_ICD_SUFFIX_KHR, suffix_);
    default: return HUM_INVALID_VALUE;
  }
  return HUM_SUCCESS;
}


hum_int HUMPlatform::GetDeviceIDs(hum_device_type device_type,
                                hum_uint num_entries, hum_device_handle* devices,
                                hum_uint* num_devices) {
  if (device_type == HUM_DEVICE_TYPE_DEFAULT)
    device_type = default_device_type_;
  if (device_type == HUM_DEVICE_TYPE_ALL)
    device_type ^= HUM_DEVICE_TYPE_CUSTOM;

	std::vector<HUMDevice*> all_devices;
  pthread_mutex_lock(&mutex_devices_);
  all_devices = devices_;
  pthread_mutex_unlock(&mutex_devices_);

  hum_uint num_devices_ret = 0;
  for (std::vector<HUMDevice*>::iterator it = all_devices.begin();
       it != all_devices.end();
       ++it) {
    HUMDevice* device = *it;
    /*if (device->IsSubDevice()) continue;*/
		HUM_DEV("Platform(%p): device(%p): device->type() = %d vs device_type = %d",
			this,	device, device->type(), device_type);

    if (device->type() & device_type) {
      if (devices != NULL && num_devices_ret < num_entries) {
        devices[num_devices_ret] = device->get_handle();
      }
      num_devices_ret++;
    }
  }
  if (num_devices) {
    if (num_entries > 0 && num_devices_ret > num_entries)
      *num_devices = num_entries;
    else
      *num_devices = num_devices_ret;
  }

	HUM_DEV("NUM_devices_ret = %d", num_devices_ret);
  if (num_devices_ret == 0) {
    return HUM_DEVICE_NOT_FOUND;
	}
  else {
    return HUM_SUCCESS;
	}
}

size_t CheckContextProperties(
    HUMPlatform* platform,
		const hum_context_properties* properties, 
		hum_int* err) {
  if (properties == NULL) 
		return 0;

  size_t idx = 0;
  bool set_platform = false;
  bool set_sync = false;
  while (properties[idx] > 0) {
    if (properties[idx] == HUM_CONTEXT_PLATFORM) {
      if (set_platform) {
        *err = HUM_INVALID_PROPERTY;
        return 0;
      }
      set_platform = true;
      if ((hum_platform_handle)properties[idx + 1] != platform->get_handle()) {
        *err = HUM_INVALID_PLATFORM;
        return 0;
      }
      idx += 2;
    } 
		else if (properties[idx] == HUM_CONTEXT_INTEROP_USER_SYNC) {
      if (set_sync) {
        *err = HUM_INVALID_PROPERTY;
        return 0;
      }
      set_sync = true;
    } 
		else {
      *err = HUM_INVALID_PROPERTY;
      return 0;
    }
  }
  return idx + 1;
}

HUMContext* HUMPlatform::CreateContextFromDevices(
    const hum_context_properties* properties, hum_uint num_devices,
    const hum_device_handle* devices, ContextErrorNotificationCallback* callback,
    hum_int* err)  {
  *err = HUM_SUCCESS;
  size_t num_properties = CheckContextProperties(this, properties, err);
  if (*err != HUM_SUCCESS) 
		return NULL;

	std::vector<HUMDevice*> selected_devices;
  selected_devices.reserve(num_devices);

  for (hum_uint i = 0; i < num_devices; i++) {
    if (devices[i] == NULL) {
      *err = HUM_INVALID_DEVICE;
      return NULL;
    }
    if (!devices[i]->c_obj->IsAvailable()) {
      *err = HUM_DEVICE_NOT_AVAILABLE;
      return NULL;
    }
    selected_devices.push_back(devices[i]->c_obj);
  }

  HUMContext* context = new HUMContext(selected_devices, num_properties,
                                     properties);
  if (context == NULL) {
    *err = HUM_OUT_OF_HOST_MEMORY;
    return NULL;
  }

  if (callback != NULL)
    context->SetErrorNotificationCallback(callback);
  return context;
}

void HUMPlatform::GetDevices(std::vector<HUMDevice*>& devices) {
  pthread_mutex_lock(&mutex_devices_);
  devices = devices_;
  pthread_mutex_unlock(&mutex_devices_);
}

HUMDevice* HUMPlatform::GetFirstDevice() {
  pthread_mutex_lock(&mutex_devices_);
  HUMDevice* device = devices_.front();
  pthread_mutex_unlock(&mutex_devices_);
  return device;
}

void HUMPlatform::AddDevice(HUMDevice* device) {
	HUM_DEV("platform(%p)->AddDevice(%p)", this, device);

	AttachDevice(device);

	if(device->IsVirtual()) {
		// ISSUER_GLOBAL
		HUM_DEV("AddDevice: Device %p is virtual", device);
		AddDeviceToFirstIssuer(device);
	}
	else {

		HUM_DEV("AddDevice: Device %p is real", device);
		// ISSUER_PER_DEVICE_BLOCKING
		AddIssuer(new HUMIssuer(device, true));
	}

	//AddDeviceToFirstIssuer(device);
}

void HUMPlatform::RemoveDevice(HUMDevice* device) {
	HUM_DEV("platform(%p)->RemoveDevice(%p)", this, device);
	if(device->IsVirtual()) {

		HUM_DEV("RemoveDevice: Device %p is virtual", device);
		// ISSUER_GLOBAL
		RemoveDeviceFromFirstIssuer(device);
	}
	else {
		HUM_DEV("RemoveDevice: Device %p is real",  device);
		// ISSUER_PER_DEVICE_BLOCKING
		RemoveIssuerOfDevice(device);
	}
	//RemoveDeviceFromFirstIssuer(device);
	DetachDevice(device);
}

void HUMPlatform::AttachDevice(HUMDevice* device) {
  pthread_mutex_lock(&mutex_devices_);
  devices_.push_back(device);
  pthread_mutex_unlock(&mutex_devices_);
}

void HUMPlatform::DetachDevice(HUMDevice* device) {
	pthread_mutex_lock(&mutex_devices_);
  devices_.erase(remove(devices_.begin(), devices_.end(), device),
                 devices_.end());
  pthread_mutex_unlock(&mutex_devices_);
}

HUMScheduler* HUMPlatform::AllocIdleScheduler() {
  // Round-robin manner
  static size_t next = 0;
  HUMScheduler* scheduler = schedulers_[next];
  next = (next + 1) % schedulers_.size();
  return scheduler;
}

void HUMPlatform::InvokeAllSchedulers() {
  for (vector<HUMScheduler*>::iterator it = schedulers_.begin();
       it != schedulers_.end();
       ++it) {
    (*it)->Invoke();
  }
}

void HUMPlatform::InitSchedulers(unsigned int num_schedulers,
                                bool busy_waiting) {
  schedulers_.reserve(num_schedulers);
  for (unsigned int i = 0; i < num_schedulers; i++) {
    HUMScheduler* scheduler = new HUMScheduler(this, busy_waiting);
    schedulers_.push_back(scheduler);
    scheduler->Start();
  }
}

void HUMPlatform::AddIssuer(HUMIssuer* issuer) {
  pthread_mutex_lock(&mutex_issuers_);
	HUM_DEV("Issuer %p is added", issuer);
  issuers_.push_back(issuer);
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->Start();
}

void HUMPlatform::RemoveIssuerOfDevice(HUMDevice* device) {
  pthread_mutex_lock(&mutex_issuers_);
  HUMIssuer* issuer = NULL;
  for (vector<HUMIssuer*>::iterator it = issuers_.begin();
       it != issuers_.end();) {

		HUM_DEV("RemoveIssuerOfDevice issuer %p device %p", *it, device);
    if ((*it)->GetFirstDevice() == device) {
      issuer = *it;
      it = issuers_.erase(it);
      break;
    }
		else {
			++it;
		}
  }
  pthread_mutex_unlock(&mutex_issuers_);
  if (issuer != NULL)
    delete issuer;
}

void HUMPlatform::AddDeviceToFirstIssuer(HUMDevice* device) {

	HUM_DEV("dev %p AddDeviceToFirstIsseur", device);

  pthread_mutex_lock(&mutex_issuers_);
  HUMIssuer* issuer = issuers_.front();
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->AddDevice(device);
}

void HUMPlatform::AddDeviceToSecondIssuer(HUMDevice* device) {
	HUM_DEV("dev %p AddDeviceToSecondIsseur", device);

  pthread_mutex_lock(&mutex_issuers_);
  HUMIssuer* issuer = issuers_[1];
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->AddDevice(device);
}
void HUMPlatform::RemoveDeviceFromFirstIssuer(HUMDevice* device) {
  pthread_mutex_lock(&mutex_issuers_);
  HUMIssuer* issuer = issuers_.front();
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->RemoveDevice(device);
}

void HUMPlatform::RemoveDeviceFromSecondIssuer(HUMDevice* device) {
  pthread_mutex_lock(&mutex_issuers_);
  HUMIssuer* issuer = issuers_[1];
  pthread_mutex_unlock(&mutex_issuers_);
  issuer->RemoveDevice(device);
}


CUDAPlatform::CUDAPlatform() {
	is_cuda_ = true;
	context_ = NULL;
	cuda_devices_ = NULL;
	num_cuda_devices_ = 0;
}

CUDAPlatform::~CUDAPlatform() {
	if(context_)
		context_->Release();

	if(cuda_devices_)
		delete(cuda_devices_);
};

void CUDAPlatform::InitCuda() {
	Init();

	for(unsigned int i=0; i<devices_.size(); i++) {
		if(devices_[i]->IsCudaAvailable()) {
			num_cuda_devices_++;
		}
		else {
			assert(0);
		}
	}

	cuda_devices_ = new hum_device_handle[num_cuda_devices_];
	for(int i=0; i< num_cuda_devices_; i++) {
		if(devices_[i]->IsCudaAvailable()) {
			cuda_devices_[i] = devices_[i]->get_handle();
		}
	}

	hum_int err;
	context_ = CreateContextFromDevices(
			NULL, num_cuda_devices_,
			cuda_devices_, NULL, &err);
	if(err != HUM_SUCCESS) {
		HUM_ERROR("CUDA create context failed err = %d", err);
		assert(0);
	}
	context_->Retain();

	for(int i=0; i< num_cuda_devices_; i++) {
		HUMDevice* device = cuda_devices_[i]->c_obj;
		if(device->IsCudaAvailable()) {
			HUMCommandQueue* default_queue = 
				HUMCommandQueue::CreateCommandQueue(context_, device, 
						(HUM_QUEUE_CUDA_STREAM|HUM_QUEUE_CUDA_DEFAULT), 
						&err);
			assert(default_queue != NULL);
			default_queues_[device] = (HUMCudaStream*)default_queue;
		}
	}
}

HUMDevice* CUDAPlatform::GetDeviceById(int device_id) {
	if(device_id >= num_cuda_devices_) {
		HUM_ERROR("device_id %d is invalid id", device_id);
		exit(0);
		return NULL;
	}
	mutex_default_queue_.lock();
	HUMDevice* ret = cuda_devices_[device_id]->c_obj;
	mutex_default_queue_.unlock();
	return ret;
}


HUMCommandQueue* CUDAPlatform::GetDefaultQueue(HUMDevice* device) {
	mutex_default_queue_.lock();

	HUMCommandQueue* queue = NULL;
	if(default_queues_.count(device) > 0) {
		queue = default_queues_[device];
	}
	else {
		HUM_ERROR("Cannot find cuda device(%p)", device);
		assert(0);
	}
	mutex_default_queue_.unlock();

	return queue;
}

void CUDAPlatform::AddBlockingQueue(HUMDevice* device,
		HUMCommandQueue* stream) {
	assert(stream->IsCudaBlockingStream());
	std::list<HUMCommandQueue*>* queues = &blocking_queues_[device];
	mutex_blocking_queue_.lock();
	queues->push_back(stream);
	mutex_blocking_queue_.unlock();
}

void CUDAPlatform::RemoveBlockingQueue(HUMDevice* device,
		HUMCommandQueue* stream) {
	assert(stream->IsCudaBlockingStream());
	std::list<HUMCommandQueue*>* queues = &blocking_queues_[device];
	mutex_blocking_queue_.lock();
	queues->remove(stream);
	mutex_blocking_queue_.unlock();

}

std::list<HUMCommandQueue*>* CUDAPlatform::GetBlockingQueue(HUMDevice* device) {
	return &blocking_queues_[device];
}
