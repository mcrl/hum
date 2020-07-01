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

#ifndef __HUM_UTILS_H__
#define __HUM_UTILS_H__

#include "hum.h"
#include "string.h"
#include "stdlib.h"
#include "assert.h"
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <sys/types.h>

extern int grank_;

#ifdef HUM_DEBUG
#include <pthread.h>
#define HUM_ERROR(fmt, ...) fprintf(stdout, "** ERR ** <rank:%d,tid:%lx> [%s:%d] " fmt "\n",  grank_, pthread_self(),  __FILE__, __LINE__, __VA_ARGS__)
#define HUM_INFO(fmt, ...) fprintf(stdout, "** INF ** <rank:%d,tid:%lx> [%s:%d] " fmt "\n", grank_, pthread_self(),  __FILE__, __LINE__, __VA_ARGS__)
#define HUM_DEV(fmt, ...) fprintf(stdout, "** DEV ** <rank:%d,tid:%lx> [%s:%d] " fmt "\n", grank_, pthread_self(),  __FILE__, __LINE__, __VA_ARGS__)
#else
#include <stdio.h>
#define HUM_ERROR(fmt, ...) fprintf(stdout, "** ERR ** <rank:%d,tid:%lx> [%s:%d] " fmt "\n",  grank_, pthread_self(),  __FILE__, __LINE__, __VA_ARGS__)
#define HUM_INFO(fmt, ...) fprintf(stdout, "** INF ** <rank:%d,tid:%lx> [%s:%d] " fmt "\n", grank_, pthread_self(),  __FILE__, __LINE__, __VA_ARGS__)
#define HUM_DEV(fmt, ...) 
#endif //HUM_DEBUG

#define HUM_PRINT(fmt, ...) fprintf(stdout, "** PRN ** <rank:%d,pid:%lx,tid:%lx> [%s:%d] " fmt "\n", grank_, (uint64_t)getpid(), (uint64_t)pthread_self(),  __FILE__, __LINE__, __VA_ARGS__)

#ifdef HUM_DEBUG
#include <assert.h>
#define HUM_ASSERT(input) assert(input)
#else
#define HUM_ASSERT(input) 
#endif

#define GET_OBJECT_INFO(param, type, value)                 \
  case param: {                                             \
    size_t size = sizeof(type);                             \
    if (param_value) {                                      \
      if (param_value_size < size) return HUM_INVALID_VALUE; \
      memcpy(param_value, &(value), size);                  \
    }                                                       \
    if (param_value_size_ret) *param_value_size_ret = size; \
    break;                                                  \
  }

#define GET_OBJECT_INFO_T(param, type, value)               \
  case param: {                                             \
    size_t size = sizeof(type);                             \
    if (param_value) {                                      \
      if (param_value_size < size) return HUM_INVALID_VALUE; \
      type temp = value;                                    \
      memcpy(param_value, &temp, size);                     \
    }                                                       \
    if (param_value_size_ret) *param_value_size_ret = size; \
    break;                                                  \
  }

#define GET_OBJECT_INFO_A(param, type, value, length)       \
  case param: {                                             \
    size_t size = sizeof(type) * length;                    \
    if (value == NULL) {                                    \
      if (param_value_size_ret) *param_value_size_ret = 0;  \
      break;                                                \
    }                                                       \
    if (param_value) {                                      \
      if (param_value_size < size) return HUM_INVALID_VALUE; \
      memcpy(param_value, value, size);                     \
    }                                                       \
    if (param_value_size_ret) *param_value_size_ret = size; \
    break;                                                  \
  }

#define GET_OBJECT_INFO_S(param, value)                     \
  case param: {                                             \
    if (value == NULL) {                                    \
      if (param_value_size_ret) *param_value_size_ret = 0;  \
      break;                                                \
    }                                                       \
    size_t size = sizeof(char) * (strlen(value) + 1);       \
    if (param_value) {                                      \
      if (param_value_size < size) return HUM_INVALID_VALUE; \
      memcpy(param_value, value, size);                     \
    }                                                       \
    if (param_value_size_ret) *param_value_size_ret = size; \
    break;                                                  \
  }


// Single Producer & Single Comsumer Queue
template<typename T>
class LockFreeQueue {
	public:
		LockFreeQueue(unsigned long size) 
			: size_(size), idx_r_(0), idx_w_(0) {
			elements_ = (volatile T**)(new T*[size]);
		}

		virtual ~LockFreeQueue() {
			delete[] elements_;
		}

		virtual bool Enqueue(T* element) {
			unsigned long next_idx_w = (idx_w_ + 1) % size_;
			if(next_idx_w == idx_r_) return false;
			elements_[idx_w_] = element;
			__sync_synchronize();
			idx_w_ = next_idx_w;
			return true;
		}

		bool Dequeue(T** element) {
			if(idx_r_ == idx_w_) return false;
			unsigned long next_idx_r = (idx_r_ + 1) % size_;
			*element = (T*)elements_[idx_r_];
			idx_r_ = next_idx_r;
			return true;
		}

		bool Peek(T** element) {
			if(idx_r_ == idx_w_) return false;
			*element = (T*) elements_[idx_r_];
			return true;
		}

		unsigned long Size() {
			if(idx_w_ >= idx_r_) return idx_w_ - idx_r_;
			return size_ - idx_r_ + idx_w_;
		}
	
	protected:
		unsigned long size_;

		volatile T** elements_;
		volatile unsigned long idx_r_;
		volatile unsigned long idx_w_;
		
};

// Multiple Producers & Single Consumer Queue
template <typename T>
class LockFreeQueueMS: public LockFreeQueue<T>
{
	public:
		LockFreeQueueMS(unsigned long size) 
			: LockFreeQueue<T>(size), idx_w_cas_(0) {
		}

		~LockFreeQueueMS() {
		}

		bool Enqueue(T* element) {
			while(true) {
				unsigned long prev_idx_w = idx_w_cas_;
				unsigned long next_idx_w = (prev_idx_w + 1) % this->size_;
				if(next_idx_w == this->idx_r_) return false;
				if(__sync_bool_compare_and_swap(&idx_w_cas_, 
							prev_idx_w, next_idx_w)) {
					this->elements_[prev_idx_w] = element;
					while(!__sync_bool_compare_and_swap(&this->idx_w_, 
								prev_idx_w, next_idx_w)) {}
					break;
				}
			}
			return true;
		}

		bool Enqueue(T* element, int& idx) {
			while(true) {
				unsigned long prev_idx_w = idx_w_cas_;
				unsigned long next_idx_w = (prev_idx_w + 1) % this->size_;
				if(next_idx_w == this->idx_r_) return false;
				if(__sync_bool_compare_and_swap(&idx_w_cas_, 
							prev_idx_w, next_idx_w)) {
					this->elements_[prev_idx_w] = element;
					idx = prev_idx_w;
					//printf("idx = %d\n", idx);
					while(!__sync_bool_compare_and_swap(&this->idx_w_, 
								prev_idx_w, next_idx_w)) {}
					break;
				}
			}
			return true;
		}


	private:
		volatile unsigned long idx_w_cas_;	
};

class mutex_t {
	public:
		mutex_t() { pthread_mutex_init(&mutex_, 0); }
		~mutex_t() { pthread_mutex_destroy(&mutex_); }
		void lock() { HUM_DEV("%s", "Lock"); pthread_mutex_lock(&mutex_); }
		void unlock() {  HUM_DEV("%s", "Unlock");pthread_mutex_unlock(&mutex_); }
	private:
		pthread_mutex_t mutex_;
};



#endif //__HUM_UTILS_H__
