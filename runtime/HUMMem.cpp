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

#include <assert.h>
#include <malloc.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include "HUMMem.h"
#include "HUMContext.h"
#include "HUMDevice.h"
#include "HUMPlatform.h"
#include "ioctl.h"

#include "MemoryRegion.h"
extern MemoryRegion<HUMMem>* g_MemRegion_;


HUMMem::HUMMem(HUMContext* context, hum_mem_object_type type,
				hum_mem_flags flags, size_t size, void* host_ptr) 
: context_(context)
{
	use_host_ = false;
	alloc_host_ = false;
	alloc_host_cuda_ = false;

  type_ = type;
  flags_ = flags;
  size_ = size;
  SetHostPtr(host_ptr);

	if(context_) {
	  context_->Retain();
		context_->AddMem(this);
	}

  pthread_mutex_init(&mutex_dev_specific_, NULL);
  pthread_mutex_init(&mutex_dev_latest_, NULL);
}

HUMMem::~HUMMem()
{
	if(context_)
		context_->Release();
	if(alloc_host_) {
		if(alloc_host_cuda_) {
			cudaFreeHost(host_ptr_);
		}
		else {
			free(host_ptr_);
		}
	}

  pthread_mutex_destroy(&mutex_dev_specific_);
  pthread_mutex_destroy(&mutex_dev_latest_);
}

void HUMMem::Cleanup() {
  for (std::map<HUMDevice*, HUMMicroMem*>::iterator it = umems_.begin();
       it != umems_.end();
       ++it) {
    (it->first)->FreeMem(it->second);
  }
  context_->RemoveMem(this);
}

bool HUMMem::HasDevSpecific(HUMDevice* device) {
  pthread_mutex_lock(&mutex_dev_specific_);
  bool has = (umems_.count(device) > 0);
  pthread_mutex_unlock(&mutex_dev_specific_);
  return has;
}

void* HUMMem::GetDevSpecific(HUMDevice* device, bool access_lock) {
  void* dev_specific = NULL;
	HUMMicroMem* umem = GetMicroMem(device, access_lock);
	dev_specific = umem->GetDevSpecific(device);
  return dev_specific;
}


bool HUMMem::IsWithinRange(size_t offset, size_t cb) const {
  return (offset < size_ && (offset + cb) <= size_);
}

bool HUMMem::IsWritable() const {
  return (flags_ & (HUM_MEM_WRITE_ONLY | HUM_MEM_READ_WRITE));
}

bool HUMMem::IsHostReadable() const {
  return !(flags_ & (HUM_MEM_HOST_WRITE_ONLY | HUM_MEM_HOST_NO_ACCESS));
}

bool HUMMem::IsHostWritable() const {
  return !(flags_ & (HUM_MEM_HOST_READ_ONLY | HUM_MEM_HOST_NO_ACCESS));
}

HUMMicroMem* HUMMem::GetMicroMem(HUMDevice* device, bool access_lock)
{
	HUM_DEV("GetMicroMem for device(%p), mem(%p)", device, this);
  pthread_mutex_lock(&mutex_dev_specific_);
	HUMMicroMem* umem = NULL;
  if (umems_.count(device) > 0) {
    umem = umems_[device];
  }
	else {	
		umem = device->AllocMem(this);
		//assert(umem->get_obj());
		umems_[device] = umem;
	}
	pthread_mutex_unlock(&mutex_dev_specific_);
	HUM_DEV("GetMicroMem for device(%p), mem(%p) return umem(%p:devptr=%p), cnt=%d", device, this, umem, umem->get_obj(), umems_.size());

  return umem;
}

HUMMicroMem* HUMMem::AddMicroMem(HUMDevice* device, HUMMicroMem* umem)
{
	HUMMicroMem* old = NULL;
  pthread_mutex_lock(&mutex_dev_specific_); 
	if (umems_.count(device) > 0) {
    old = umems_[device];
	}
	umems_[device] = umem;
	pthread_mutex_unlock(&mutex_dev_specific_);

	return old;
}

void HUMMem::RemoveMicroMem(HUMDevice* device)
{
	pthread_mutex_lock(&mutex_dev_specific_); 
	if (umems_.count(device) > 0) {
		HUMMicroMem* umem = umems_[device];
		delete(umem);
		umems_.erase(device);
	}
	pthread_mutex_unlock(&mutex_dev_specific_);
}

bool HUMMem::EmptyLatest() {
  pthread_mutex_lock(&mutex_dev_latest_);
  bool empty = dev_latest_.empty();
  pthread_mutex_unlock(&mutex_dev_latest_);
  return empty;
}

bool HUMMem::HasLatest(HUMDevice* device) {
  pthread_mutex_lock(&mutex_dev_latest_);
	std::set<HUMDevice*>::iterator it = dev_latest_.find(device);
  bool find = (it != dev_latest_.end());
  pthread_mutex_unlock(&mutex_dev_latest_);
  return find;
}

HUMDevice* HUMMem::FrontLatest() {
  HUMDevice* device = NULL;
  pthread_mutex_lock(&mutex_dev_latest_);
  if (!dev_latest_.empty())
    device = *(dev_latest_.begin());
  pthread_mutex_unlock(&mutex_dev_latest_);
  return device;
}

void HUMMem::AddLatest(HUMDevice* device) {
  pthread_mutex_lock(&mutex_dev_latest_);
  dev_latest_.insert(device);
  pthread_mutex_unlock(&mutex_dev_latest_);
}

void HUMMem::SetLatest(HUMDevice* device) {
  pthread_mutex_lock(&mutex_dev_latest_);
  dev_latest_.clear();
  dev_latest_.insert(device);
  pthread_mutex_unlock(&mutex_dev_latest_);
}

HUMDevice* HUMMem::GetNearestLatest(HUMDevice* device) {
  HUMDevice* nearest = NULL;
  int min_distance = 10; // INF
  pthread_mutex_lock(&mutex_dev_latest_);
  for (std::set<HUMDevice*>::iterator it = dev_latest_.begin();
       it != dev_latest_.end();
       ++it) {
    int distance = device->GetDistance(*it);
    if (distance < min_distance) {
      nearest = *it;
      min_distance = distance;
    } 
  }
  pthread_mutex_unlock(&mutex_dev_latest_);
  return nearest;
}

void HUMMem::SetHostPtr(void* host_ptr) {
	assert(host_ptr == 0);
  if (flags_ & HUM_MEM_USE_HOST_PTR) {
    host_ptr_ = host_ptr;
    use_host_ = true;
  } 
	else if ((flags_ & HUM_MEM_ALLOC_HOST_PTR) &&
             (flags_ & HUM_MEM_COPY_HOST_PTR)) {
		cudaMallocHost(&host_ptr_, size_);
		if(host_ptr_ == NULL) {
	    host_ptr_ = memalign(4096, size_);
		}
		else {
			alloc_host_cuda_ = true;
		}

		memcpy(host_ptr_, host_ptr, size_);
    alloc_host_ = true;
    use_host_ = true;
  } 
	else if (flags_ & HUM_MEM_ALLOC_HOST_PTR) {
		cudaMallocHost(&host_ptr_, size_);
		if(host_ptr_ == NULL) {
	    host_ptr_ = memalign(4096, size_);
		}
		else {
			alloc_host_cuda_ = true;
		}

		alloc_host_ = true;
    use_host_ = true;
  } 
	else if (flags_ & HUM_MEM_COPY_HOST_PTR) {
 		cudaMallocHost(&host_ptr_, size_);
		if(host_ptr_ == NULL) {
	    host_ptr_ = memalign(4096, size_);
		}
		else {
			alloc_host_cuda_ = true;
		}
    memcpy(host_ptr_, host_ptr, size_);
    alloc_host_ = true;
    SetLatest(LATEST_HOST);
  }
}





HUMMem* HUMMem::CreateBuffer(HUMContext* context, HUMDevice* device,
    hum_mem_flags flags, size_t size, void* host_ptr, hum_int* err) {
	if (!(flags & (HUM_MEM_READ_WRITE | HUM_MEM_READ_ONLY | HUM_MEM_WRITE_ONLY)))
		flags |= flags & HUM_MEM_READ_WRITE;

	HUMMem* mem = new HUMUnifiedMem(context, device, HUM_MEM_OBJECT_BUFFER, 
			flags, size, host_ptr /*for UnifiedMemory it is dev_ptr*/);

	if (mem == NULL) {
		*err = HUM_OUT_OF_HOST_MEMORY;
		return NULL;
	}

	return mem;
}


HUMUnifiedMem::HUMUnifiedMem(HUMContext* context, HUMDevice* device,
    hum_mem_object_type type, hum_mem_flags flags, size_t _size, void* dev_ptr)
: HUMMem(context, type, flags, _size, NULL)
{
	dev_ptr_ = dev_ptr;
  caller_device_id_ = device->GetDeviceID();

	if(grank_ == 0) {
		HUM_DEV("Send Create UM Buffer(mem=%p, id=%d) to all nodes(%d)", this, id(), gsize_);
		//Create memory objs for all of nodes in this cluster
		int num_nodes = gsize_;
	}
  
  void* cuda_ptr = this->GetDevSpecific(device);
  assert(dev_ptr_ == cuda_ptr);
}

HUMUnifiedMem::~HUMUnifiedMem()
{
	if(grank_ == 0) {
		//Create memory objs for all of nodes in this cluster
		int num_nodes = gsize_;
		//munmap(cluster_um_host_ptr_, size());
	}
}


HUMMicroMem* HUMUnifiedMem::GetMicroMem(HUMDevice* device, bool access_lock)
{
	//Memobj of UM has only one uMemobj
	HUM_DEV("GetMicroMem for device(%p), mem(%p)", device, this);
  if (access_lock)
    pthread_mutex_lock(&mutex_dev_specific_);
	HUMMicroMem* umem = NULL;
	if (umems_.size() > 0) {
		umem = umems_.begin()->second;
	}
	else {
		umem = device->AllocMem(this);
		if(grank_== 0 && umem->get_obj() == NULL) {
			assert(0);
			//umem->set_obj(cluster_um_host_ptr_);
		}
		assert(umem->get_obj());
		umems_[device] = umem;
	}
  if (access_lock)
    pthread_mutex_unlock(&mutex_dev_specific_);
	HUM_DEV("GetMicroMem for device(%p), mem(%p) return umem(%p:devptr=%p), cnt=%d", device, this, umem, umem->get_obj(), umems_.size());

  return umem;
}

HUMMicroMem::HUMMicroMem(HUMContext* context, HUMDevice* dev, HUMMem* mem) 
: context_(context), dev_(dev), mem_(mem)
{
  context_->Retain();
	real_mem_obj_ = NULL;
	cuda_alloc_ = false;
}

HUMMicroMem::~HUMMicroMem()
{
	context_->Release();
/*
	if(real_mem_obj_ && cuda_alloc_) {
		cudaFree(real_mem_obj_);
	}
*/
}

void* HUMMicroMem::GetDevSpecific(HUMDevice* device)
{
	return this->get_obj();
}


std::map<std::string, HUMCudaSymbol*> g_cuda_symbol_map;
HUMCudaSymbol::HUMCudaSymbol(HUMContext* context, const char* symbol)
: HUMMem(context, HUM_MEM_OBJECT_CUDA_SYMBOL, HUM_MEM_READ_ONLY, 0, NULL)
{
	assert(0);
	symbol_ = symbol;
	g_cuda_symbol_map[symbol_] = this;
}

HUMCudaSymbol::~HUMCudaSymbol() {
}

