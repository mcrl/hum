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

#ifndef __HUM_COMMAND_QUEUE_H__
#define __HUM_COMMAND_QUEUE_H__

#include <list>
#include <pthread.h>
#include "HUMObject.h"
#include "Utils.h"


class HUMCommand;
class HUMContext;
class HUMDevice;
class HUMEvent;


class HUMCommandQueue: public HUMObject<HUMCommandQueue> {
 protected:
  HUMCommandQueue(HUMContext* context, HUMDevice* device,
                 hum_command_queue_properties properties);

 public:
  virtual void Cleanup();
  virtual ~HUMCommandQueue();

  HUMContext* context() const { return context_; }
  HUMDevice* device() const { return device_; }

  hum_int GetCommandQueueInfo(hum_command_queue_info param_name,
                             size_t param_value_size, void* param_value,
                             size_t* param_value_size_ret);

  bool IsProfiled() const {
    return (properties_ & HUM_QUEUE_PROFILING_ENABLE);
  }
	bool IsCudaStream() const {
    return (properties_ & HUM_QUEUE_CUDA_STREAM);
	}
	bool IsCudaBlockingStream() const {
    return (properties_ & (HUM_QUEUE_CUDA_STREAM | HUM_QUEUE_CUDA_BLOCKING));
	}
	bool IsCudaDefaultStream() const {
    return (properties_ & (HUM_QUEUE_CUDA_STREAM | HUM_QUEUE_CUDA_DEFAULT));
	}

  virtual bool IsEmpty() = 0;
  virtual HUMCommand* Peek() = 0;
  virtual void Enqueue(HUMCommand* command) = 0;
  virtual void Dequeue(HUMCommand* command) = 0;
  void Flush() {}

	HUMDevice* device() { return device_; }

 protected:
  void InvokeScheduler();

 private:
  HUMContext* context_;
  HUMDevice* device_;
  hum_command_queue_properties properties_;

 public:
  static HUMCommandQueue* CreateCommandQueue(
      HUMContext* context, HUMDevice* device,
      hum_command_queue_properties properties, hum_int* err);
};

class HUMInOrderCommandQueue: public HUMCommandQueue {
 public:
  HUMInOrderCommandQueue(HUMContext* context, HUMDevice* device,
                        hum_command_queue_properties properties);
  virtual ~HUMInOrderCommandQueue();

  virtual bool IsEmpty();
  virtual HUMCommand* Peek();
  virtual void Enqueue(HUMCommand* command);
  virtual void Dequeue(HUMCommand* command);

 private:
  LockFreeQueueMS<HUMCommand> queue_;
  HUMEvent* last_event_;
};

class HUMOutOfOrderCommandQueue: public HUMCommandQueue {
 public:
  HUMOutOfOrderCommandQueue(HUMContext* context, HUMDevice* device,
                           hum_command_queue_properties properties);
  virtual ~HUMOutOfOrderCommandQueue();

  virtual bool IsEmpty();
  virtual HUMCommand* Peek();
  virtual void Enqueue(HUMCommand* command);
  virtual void Dequeue(HUMCommand* command);

 private:
  std::list<HUMCommand*> commands_;
  pthread_mutex_t mutex_commands_;
};

class HUMCudaStream: public HUMInOrderCommandQueue {
	public:
	HUMCudaStream(HUMContext* context, HUMDevice* device,
                        hum_command_queue_properties properties);
  virtual ~HUMCudaStream();

  //virtual HUMCommand* Peek();
  virtual void Enqueue(HUMCommand* command);
  //virtual void Dequeue(HUMCommand* command);

	bool is_default_stream() const { return default_stream_; };
	bool is_blocking_stream() const { return blocking_stream_; };

	private:
	bool default_stream_;
	bool blocking_stream_;
};

#endif // _HUM_COMMAND_QUEUE_H__
