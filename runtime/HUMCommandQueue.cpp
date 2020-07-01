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

#include "HUMCommandQueue.h"
#include <list>
#include <pthread.h>
#include "HUMPlatform.h"
#include "HUMCommand.h"
#include "HUMContext.h"
#include "HUMDevice.h"
#include "HUMEvent.h"
#include "HUMObject.h"
#include "Utils.h"
#include <stdio.h> 

using namespace std;

#define COMMAND_QUEUE_SIZE 4096

HUMCommandQueue::HUMCommandQueue(HUMContext *context, HUMDevice* device,
                               hum_command_queue_properties properties) {
  context_ = context;
  context_->Retain();
  device_ = device;
  device_->AddCommandQueue(this);
  properties_ = properties;
}

void HUMCommandQueue::Cleanup() {
  device_->RemoveCommandQueue(this);
}

HUMCommandQueue::~HUMCommandQueue() {
  context_->Release();
}

hum_int HUMCommandQueue::GetCommandQueueInfo(hum_command_queue_info param_name,
                                           size_t param_value_size,
                                           void* param_value,
                                           size_t* param_value_size_ret) {
  switch (param_name) {
    GET_OBJECT_INFO_T(HUM_QUEUE_CONTEXT, HUMContext*, context_);
    GET_OBJECT_INFO_T(HUM_QUEUE_DEVICE, HUMDevice*, device_);
    GET_OBJECT_INFO_T(HUM_QUEUE_REFERENCE_COUNT, hum_uint, ref_cnt());
    GET_OBJECT_INFO(HUM_QUEUE_PROPERTIES, hum_command_queue_properties,
                    properties_);
    default: return HUM_INVALID_VALUE;
  }
  return HUM_SUCCESS;
}

void HUMCommandQueue::InvokeScheduler() {
  device_->InvokeScheduler();
}

HUMCommandQueue* HUMCommandQueue::CreateCommandQueue(
    HUMContext* context, HUMDevice* device,
    hum_command_queue_properties properties, hum_int* err) {
  HUMCommandQueue* queue;
  if (properties & HUM_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    queue = new HUMOutOfOrderCommandQueue(context, device, properties);
	else if (properties & HUM_QUEUE_CUDA_STREAM)
    queue = new HUMCudaStream(context, device, properties);
	else {
    queue = new HUMInOrderCommandQueue(context, device, properties);
	}
  if (queue == NULL) {
    *err = HUM_OUT_OF_HOST_MEMORY;
    return NULL;
  }
	else {
		*err = HUM_SUCCESS;
	}
  return queue;
}

HUMInOrderCommandQueue::HUMInOrderCommandQueue(
    HUMContext* context, HUMDevice* device,
    hum_command_queue_properties properties)
    : HUMCommandQueue(context, device, properties),
      queue_(COMMAND_QUEUE_SIZE) {
	HUM_DEV("Create HUMInOrderCommandQueue %p", this);
  last_event_ = NULL;
}

HUMInOrderCommandQueue::~HUMInOrderCommandQueue() {
  if (last_event_)
    last_event_->Release();
}

bool HUMInOrderCommandQueue::IsEmpty() {
  return queue_.Size() == 0;
}

HUMCommand* HUMInOrderCommandQueue::Peek() {
  if (queue_.Size() == 0) return NULL;
  HUMCommand* command;
  if (queue_.Peek(&command) && command->IsExecutable())
    return command;
  else
    return NULL;
}

void HUMInOrderCommandQueue::Enqueue(HUMCommand* command) {
  if (last_event_ != NULL) {
    command->AddWaitEvent(last_event_);
    last_event_->Release();
  }
  last_event_ = command->ExportEvent();
  while (!queue_.Enqueue(command)) {}
  InvokeScheduler();
}

void HUMInOrderCommandQueue::Dequeue(HUMCommand* command) {
  HUMCommand* dequeued_command;
  queue_.Dequeue(&dequeued_command);
#ifdef HUM_DEBUG
  if (command != dequeued_command)
    HUM_ERROR("%s", "Invalid dequeue request");
#endif // HUM_DEBUG
}

HUMOutOfOrderCommandQueue::HUMOutOfOrderCommandQueue(
    HUMContext* context, HUMDevice* device,
    hum_command_queue_properties properties)
    : HUMCommandQueue(context, device, properties) {
  pthread_mutex_init(&mutex_commands_, NULL);
}

HUMOutOfOrderCommandQueue::~HUMOutOfOrderCommandQueue() {
  pthread_mutex_destroy(&mutex_commands_);
}

bool HUMOutOfOrderCommandQueue::IsEmpty() {
  return commands_.empty();
}

HUMCommand* HUMOutOfOrderCommandQueue::Peek() {
  if (commands_.empty()) return NULL;

  HUMCommand* result = NULL;
  pthread_mutex_lock(&mutex_commands_);
  for (list<HUMCommand*>::iterator it = commands_.begin();
       it != commands_.end();
       ++it) {
    HUMCommand* command = *it;
    if (!command->IsExecutable()) continue;

    if (command->type() == HUM_COMMAND_MARKER ||
        command->type() == HUM_COMMAND_BARRIER) {
      if (it == commands_.begin())
        result = command;
    } else {
      result = command;
    }
    break;
  }
  pthread_mutex_unlock(&mutex_commands_);
  return result;
}

void HUMOutOfOrderCommandQueue::Enqueue(HUMCommand* command) {
  pthread_mutex_lock(&mutex_commands_);
  commands_.push_back(command);
  pthread_mutex_unlock(&mutex_commands_);
  InvokeScheduler();
}

void HUMOutOfOrderCommandQueue::Dequeue(HUMCommand* command) {
  pthread_mutex_lock(&mutex_commands_);
  commands_.remove(command);
  pthread_mutex_unlock(&mutex_commands_);
}

HUMCudaStream::HUMCudaStream(
    HUMContext* context, HUMDevice* device,
    hum_command_queue_properties properties)
    : HUMInOrderCommandQueue(context, device, properties)
{
	HUM_DEV("Create HUMCudaStream %p", this);

	CUDAPlatform* platform = HUMPlatform::GetCudaPlatform();
	assert(platform == device->platform());

	default_stream_ = false;
	blocking_stream_ = false;

	if(!(properties & HUM_QUEUE_CUDA_STREAM)) {
		HUM_ERROR("Invalid CUDA Stream properties(%lX)", properties);
		assert(0);
	}
	if(properties & HUM_QUEUE_CUDA_DEFAULT) {
		default_stream_ = true;	
	}
	if(properties & HUM_QUEUE_CUDA_BLOCKING) {
		blocking_stream_ = true;
		platform->AddBlockingQueue(device, this);
	}

	assert(default_stream_ != true || blocking_stream_ != true);
}


HUMCudaStream::~HUMCudaStream()
{
	CUDAPlatform* platform = HUMPlatform::GetCudaPlatform();

	if(blocking_stream_) {
		platform->RemoveBlockingQueue(device(), this);
	}
}


void HUMCudaStream::Enqueue(HUMCommand* command) 
{
	CUDAPlatform* platform = HUMPlatform::GetCudaPlatform();
	assert(platform == device()->platform());

	HUM_DEV("Enqueue: cmd %p type=%x", command, command->type());

	if(command->type() != HUM_COMMAND_MARKER) {
		if(default_stream_) {
			// default stream must be synchronous 
			// with all blocking streams

			std::list<HUMCommandQueue*>* blocking_streams
				= platform->GetBlockingQueue(device());
			size_t num_streams = blocking_streams->size();

			int i=0;
			if(num_streams > 0) {

				HUM_DEV("default_stream(%p) sync begin", this);

				HUMEvent* ev[num_streams];

				for(std::list<HUMCommandQueue*>::iterator it = blocking_streams->begin();
						it != blocking_streams->end(); it++) {
					HUMCommandQueue* queue = (*it);
					HUMCommand* mk = HUMCommand::CreateMarker(NULL, NULL, queue);
					if (mk == NULL) assert(0);

					ev[i] = mk->ExportEvent();
					command->AddWaitEvent(ev[i]);
					ev[i]->Release();
					queue->Enqueue(mk);
					i++;
				}

				HUM_DEV("default_stream(%p) sync end", this);
			}
		}
		else if(blocking_stream_) {
			// blocking stream must be synchronous 
			// with default stream

			HUM_DEV("blocking_stream(%p) sync begin", this);
			HUMCommandQueue* queue = platform->GetDefaultQueue(device());
			assert(queue != NULL);

			HUMEvent* ev;
			HUMCommand* mk = HUMCommand::CreateMarker(NULL, NULL, queue);
			if (mk == NULL) assert(0);
			ev = mk->ExportEvent();
			command->AddWaitEvent(ev);
			ev->Release();
			queue->Enqueue(mk);

			HUM_DEV("blocking_stream(%p) sync end", this);
		}
	}
	HUMInOrderCommandQueue::Enqueue(command);
}

