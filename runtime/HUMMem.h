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

#ifndef __HUM_MEM_H__
#define __HUM_MEM_H__

#include "HUMObject.h"
#include <map>
#include <set>
#include <list>
#include <pthread.h>
#include <string>

class HUMContext;
class HUMDevice;
class HUMMicroMem;
class HUMUnifiedMem;


#define LATEST_HOST ((HUMDevice*)1)

#define BLOCK_SIZE (4096*512)

typedef struct {
  uint64_t mem_start_dev;
  uint64_t mem_end_dev;
  void* mprotect_start;
  size_t mprotect_size;
  bool mprotected;
  int num_chunks;
  int finished_chunks;
  volatile int ref_count;
  pthread_mutex_t block_mutex;
} memcpy_control_block_t;

typedef struct {
  void* user_src;
  HUMUnifiedMem* umbuf;
	size_t offset;
	size_t size;
  size_t copied_size;
  memcpy_control_block_t *control_block;
} memcpy_command_t;


class HUMMem: public HUMObject<HUMMem>
{
	public:
		HUMMem(HUMContext* context, hum_mem_object_type type,
				hum_mem_flags flags, size_t size, void* host_ptr);
		virtual ~HUMMem();

		virtual void Cleanup();
		bool HasDevSpecific(HUMDevice* device);
		void* GetDevSpecific(HUMDevice* device, bool access_lock = true);

		bool IsImage() const { return (type_ >= HUM_MEM_OBJECT_IMAGE2D && type_ < HUM_MEM_OBJECT_PIPE); }
		bool IsBuffer() const { return type_ == HUM_MEM_OBJECT_BUFFER; }
		bool IsSymbol() const { return type_ == HUM_MEM_OBJECT_CUDA_SYMBOL; }
		bool IsSubBuffer() const { return false; /*return parent_ != NULL*/ }
		bool IsWithinRange(size_t offset, size_t cb) const;
		bool IsWritable() const;
		bool IsHostReadable() const;
		bool IsHostWritable() const;

		virtual HUMMicroMem* GetMicroMem(HUMDevice* device, bool access_lock);
		HUMMicroMem* AddMicroMem(HUMDevice* device, HUMMicroMem* umem);
		void RemoveMicroMem(HUMDevice* device);
		size_t GetNumUMems() const { return umems_.size(); }

		bool EmptyLatest();
		bool HasLatest(HUMDevice* device);
		HUMDevice* FrontLatest();
		void AddLatest(HUMDevice* device);
		void SetLatest(HUMDevice* device);
		HUMDevice* GetNearestLatest(HUMDevice* device);

		HUMContext* context() const { return context_; }
		hum_mem_object_type type() const { return type_; }
		hum_mem_flags flags() const { return flags_; }
		size_t size() const { return size_; }
		//size_t offset() const { return offset_; }

		void SetHostPtr(void* host_ptr);
		void* GetHostPtr() const { return host_ptr_; }

    int GetCallerDeviceID() { return caller_device_id_; }

	private:
		HUMContext* context_;
		hum_mem_object_type type_;
		hum_mem_flags flags_;
		size_t size_;
		//size_t offset_;

		void* host_ptr_;
		bool use_host_;
		bool alloc_host_;
		bool alloc_host_cuda_;

	protected:
		std::map<HUMDevice*, HUMMicroMem*> umems_;
		std::set<HUMDevice*> dev_latest_;

		pthread_mutex_t mutex_dev_specific_;
		pthread_mutex_t mutex_dev_latest_;

    int caller_device_id_;
	public:
		static HUMMem* CreateBuffer(HUMContext* context, HUMDevice* device,
        hum_mem_flags flags, size_t size, void* host_ptr, hum_int* err);
};


// for single node UM
class HUMUnifiedMem: public HUMMem
{
	public:
		HUMUnifiedMem(HUMContext* context, HUMDevice* device,
        hum_mem_object_type type,
				hum_mem_flags flags, size_t size, void* dev_ptr);
		virtual ~HUMUnifiedMem();

		virtual HUMMicroMem* GetMicroMem(HUMDevice* device, bool access_lock);

		void* GetUMHostPtr(HUMDevice* device) { 
			return dev_ptr_;
		}
#if defined(USE_MEM_PREFETCH)
		void* tmp_cpy_buf_;
#endif

		void* dev_ptr_;
};



class HUMMicroMem : public HUMObject<HUMMicroMem>
{
	public:
		HUMMicroMem(HUMContext* context, HUMDevice* dev, HUMMem* mem);
		virtual ~HUMMicroMem();

		HUMContext* context() const { return context_; }

		void* GetDevSpecific(HUMDevice* device);

		hum_mem_object_type type() const { return mem_->type(); }
		hum_mem_flags flags() const { return mem_->flags(); }

		void set_obj(void* obj) { real_mem_obj_ = obj; }
		void* get_obj() const { return real_mem_obj_; };
			
		size_t size() const { return mem_->size(); }
		//size_t offset() const { return mem_->offset(); }

		HUMMem* get_parent() { return mem_; }
		bool cuda_alloc_;

	private:
		HUMContext* context_;
		HUMDevice* dev_;
		HUMMem* mem_;

		void * real_mem_obj_;
		/*
			 hum_mem_object_type type_;
			 hum_mem_flags flags_;
			 size_t size_;
			 */
};

class HUMCudaSymbol: public HUMMem
{
	public:
		//HUM_MEM_OBJECT_CUDA_SYMBOL
		HUMCudaSymbol(HUMContext* context, const char* symbol);
		virtual ~HUMCudaSymbol();

		std::string symbol_;
};
extern std::map<std::string, HUMCudaSymbol*> g_cuda_symbol_map;

#endif //__HUM_MEM_H__
