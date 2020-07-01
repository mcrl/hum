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

#ifndef __HUM_MEMORY_REGION_H__
#define __HUM_MEMORY_REIGON_H__

#include <list>
#include <assert.h>
#include <errno.h>
#include <sys/mman.h>
#include "Device/NVLibs.h"
#include "Utils.h"
#include "HUMMem.h"
#include "HUMDevice.h"

#define NUM_CHUNK_CAT 18
//#define DEFAULT_PAGE_SIZE (4L*KB)
#define DEFAULT_PAGE_SIZE (2L*MB)
#define CHUNK_MULTIPLY_FACTOR 2 //

#define MAGIC_WORD	0x1234FEDC

extern NVLibs* g_nvlibs;


template<typename T>
struct PageChunk {
		uint64_t magic_;
		int64_t id_;
		size_t size_;
		void* page_ptr_;
		uint64_t category_;

		uint64_t offset_; //offset from um base addr

		T*  obj_;

		void set(uint64_t id, void* page_ptr, size_t size, uint64_t category, T* obj, size_t offset) {
			magic_ = MAGIC_WORD;
			id_ = id;
			page_ptr_ = page_ptr;
			size_ = size;
			category_ = category;
			obj_ = obj;
			offset_ = offset;
		}

		void* page_ptr() const { return page_ptr_; }
		size_t size() const { return size_; }
		uint64_t category() const { return category_; }
		inline T* obj() const { return (T*)obj_; }
		void set_obj(T* obj) { obj_ = obj; }
};

template<typename T>
class MemoryRegion
{
	public:
		MemoryRegion(size_t size) {
			{
        cudaError_t err = g_nvlibs->cudaGetDeviceCount(&dev_count_);
        assert(err == cudaSuccess);

        for (unsigned int i = 0; i < dev_count_; ++i) {
          err = g_nvlibs->cudaSetDevice(i);
          assert(err == cudaSuccess);

          err = g_nvlibs->cudaMallocManaged(&um_alloc_[i], (16L*1024L*GB), cudaMemAttachGlobal);
          assert(err == cudaSuccess);
        }

				if(grank_ == 0) {
					size_ = size;

					obj_region_ = mmap(0, size_,
							PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
					if (obj_region_ == MAP_FAILED) {
						fprintf(stderr, "Unable to reserve memory region. ERROR : %d, %s\n", errno, strerror(errno));
						assert(0);
					}
					PrepareMemChunk();

				}

        for (unsigned int i = 0; i < dev_count_; ++i) {
          um_base_[i] = (void*)((uint64_t)um_alloc_[i] - ((uint64_t)um_alloc_[i] & ((4096L * GB) -1)));
          um_base_[i] = (void*)((uint64_t)um_base_[i] + (4096L * GB));
        }

				pthread_mutex_init(&mutex_mem_region_, NULL);

			}
		}

		~MemoryRegion() {
			if(grank_ == 0) {
				munmap(obj_region_, size_);

			}
			{
/* TODO: i don't know why this code occurs segfault
				g_nvlibs->cudaSetDevice(0);

				if(um_alloc_)
					g_nvlibs->cudaFree(um_alloc_);
*/
				pthread_mutex_destroy(&mutex_mem_region_);
			}
		}
	
		int IsWithin(const void* ptr) const {
			HUM_ASSERT(grank_ == 0);

      for (unsigned int i = 0; i < dev_count_; ++i) {
        if ((uint64_t)ptr >= (uint64_t)um_base_[i] &&
            (uint64_t)ptr < (uint64_t)um_base_[i] + size_)
          return i;
      }
      return -1;
		}

		T* GetMemObj(const void* ptr, uint64_t* offset) {
      int um_region_id = IsWithin(ptr);
      if(um_region_id == -1)
        return NULL;

			pthread_mutex_lock(&mutex_mem_region_);

			struct PageChunk<T>* page = GetPageChunk(um_region_id, ptr);
			assert(page != NULL);
		
			if(offset != NULL) {
				*offset = (uint64_t)ptr - ((uint64_t)um_base_[um_region_id] + (uint64_t)page->offset_) ;
			}
			
			pthread_mutex_unlock(&mutex_mem_region_);

			return page->obj();
		}

		size_t GetPageChunkSize(const void* ptr) const {
			size_t ret;
      int um_region_id = IsWithin(ptr);
      if(um_region_id == -1)
        return NULL;

			pthread_mutex_lock(&mutex_mem_region_);

			struct PageChunk<T>* page = GetPageChunk(um_region_id, ptr);
			assert(page != NULL);
		
			ret = page->size();

			pthread_mutex_unlock(&mutex_mem_region_);
			return ret;
		}

		void* CreateHUMDevPtr(HUMContext* context, HUMDevice* device, T** obj, size_t size) {
			HUM_ASSERT(grank_ == 0);

			pthread_mutex_lock(&mutex_mem_region_);
			struct PageChunk<T>* page = CreatePageChunk(size);
			assert(page != NULL);

			void* dev_ptr = (void*)((uint64_t)um_base_[device->GetDeviceID()] + page->offset_);

			hum_mem_flags flags = HUM_MEM_READ_WRITE;
			hum_int err = HUM_SUCCESS;

			if(page->obj()) {
				HUMMem* mem = page->obj();
				*obj = mem;
				assert(mem->size() == size);
				if(((HUMUnifiedMem*)mem)->dev_ptr_ != dev_ptr) {
					HUM_ERROR("memobj(%p)->dev_ptr_ = %p vs dev_ptr = %p",
							mem, ((HUMUnifiedMem*)mem)->dev_ptr_, dev_ptr);
				}
				assert(((HUMUnifiedMem*)mem)->dev_ptr_ == dev_ptr);
			}
			else {
				T* mem = HUMMem::CreateBuffer(context, device, flags, size, dev_ptr, &err);
				*obj = mem;
				page->set_obj(mem);
			}
			pthread_mutex_unlock(&mutex_mem_region_);
		
			//return (void*)page;
			return dev_ptr;
		}

		void FreeHUMDevPtr(void* ptr) {
			HUM_ASSERT(grank_ == 0);

      int um_region_id = IsWithin(ptr);
      if(um_region_id == -1)
        return;

			pthread_mutex_lock(&mutex_mem_region_);
			struct PageChunk<T>* page = GetPageChunk(um_region_id, ptr);
			assert(page != NULL);
			//page->set_obj(NULL);
			free_chunks_[page->category()].push_back(page);
			pthread_mutex_unlock(&mutex_mem_region_);
		}


	private:
		void PrepareMemChunk() {
			HUM_ASSERT(grank_ == 0);

			int64_t unit_size = DEFAULT_PAGE_SIZE; //4L * KB;
			for(int i=0;i<NUM_CHUNK_CAT;i++) {
				chunk_cnts_[i] = 0;
				chunk_sizes_[i] = unit_size;
				//chunk_max_cnts_[i] = (128L * GB) / unit_size;
				chunk_max_cnts_[i] = (size_ / NUM_CHUNK_CAT) / unit_size;

				chunk_masks_[i] = 0xFFFFFFFFFFFFFFFF - (chunk_sizes_[i]-1);
				unit_size *= CHUNK_MULTIPLY_FACTOR;

				HUM_DEV("[%d] size=%ld, max_cnt=%ld, mask=%lX",
						i, chunk_sizes_[i], chunk_max_cnts_[i], chunk_masks_[i]);

			}
		}

		struct PageChunk<T>* CreatePageChunk(size_t size) {
			HUM_ASSERT(grank_ == 0);

			uint64_t chunk_category = NUM_CHUNK_CAT;
			struct PageChunk<T>* page = NULL;

			for(int i=0;i<NUM_CHUNK_CAT;i++) {
				if(size <= chunk_sizes_[i]) {
					chunk_category = i;
					break;
				}
			}

			if(chunk_category == NUM_CHUNK_CAT) {
				fprintf(stderr, "Allocation failed: too large allocation memory size %ld\n", size);
				return NULL;
			}
		

			if(!free_chunks_[chunk_category].empty()) {
				typename std::list<struct PageChunk<T>*>::iterator it = free_chunks_[chunk_category].begin();
				for(;it != free_chunks_[chunk_category].end();it++) {
					struct PageChunk<T>* cur_page = *it;
					HUMMem* mem = (HUMMem*)cur_page->obj();
					assert(mem);
					if(mem->size() == size) {
						page = cur_page;
						free_chunks_[chunk_category].erase(it);
						break;
					}
				}
			}

			if(page == NULL) {
/*				
				if((free_chunks_[chunk_category].size() > 16) ||
					(free_chunks_[chunk_category].size() > chunk_max_cnts_[chunk_category]/4))	{*/
					
				
				//if(!free_chunks_[chunk_category].empty()) {
				
				if(free_chunks_[chunk_category].size() > chunk_max_cnts_[chunk_category]/2)	{
					page = free_chunks_[chunk_category].front();

					//TODO:::
					{
						HUMMem* mem = (HUMMem*)page->obj();
						mem->Release();
					}

					page->set_obj(NULL);
					free_chunks_[chunk_category].pop_front();
				}
				else {
					if(chunk_cnts_[chunk_category] >= chunk_max_cnts_[chunk_category]) {
						fprintf(stderr, "Allocation failed: no enough space %ld\n", size);

						return NULL;
					}

					uint64_t chunk_offset = (chunk_category << 38) 
						+	chunk_cnts_[chunk_category] * chunk_sizes_[chunk_category];

					uint64_t chunk_addr = chunk_offset + (uint64_t)obj_region_;
					page = (struct PageChunk<T>*)chunk_addr;
					page->set_obj(NULL);
					page->set(chunk_cnts_[chunk_category], page, chunk_sizes_[chunk_category], chunk_category, NULL, chunk_offset);
					chunk_cnts_[chunk_category]++;
				}
			}

			return page;
		}

		struct PageChunk<T>* GetPageChunk(int um_region_id, const void* ptr) {
			HUM_ASSERT(grank_ == 0);

			uint64_t addr_offset = (uint64_t)ptr - (uint64_t)um_base_[um_region_id];
			uint64_t chunk_category = addr_offset >> 38;
			
			uint64_t page_ptr = (uint64_t)((uint64_t)obj_region_ + (addr_offset & chunk_masks_[chunk_category]));
			struct PageChunk<T>* page = (struct PageChunk<T>*)page_ptr;
			HUM_DEV("page = %p", page);

			if(page->magic_ != MAGIC_WORD) {
				HUM_ERROR("%p is not a valid address", ptr);
				return NULL;
			}
			return page;
		}

		void* obj_region_;	//page object base region_
		void* um_base_[8];	//unified memory base region_
		void* um_alloc_[8];	//unified memory base region_
    int dev_count_;

		size_t size_;

		int64_t chunk_cnts_[NUM_CHUNK_CAT];
		int64_t chunk_max_cnts_[NUM_CHUNK_CAT];
		size_t chunk_sizes_[NUM_CHUNK_CAT];
		uint64_t chunk_masks_[NUM_CHUNK_CAT];
		std::list<struct PageChunk<T>*> free_chunks_[NUM_CHUNK_CAT];

		pthread_mutex_t mutex_mem_region_;
};


#endif //_HUM_MEMORY_REGION_H__
