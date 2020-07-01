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

#include "HUMEvent.h"
#include <vector>
#include <pthread.h>
#include <time.h>
#include "Callbacks.h"
#include "HUMCommand.h"
#include "HUMCommandQueue.h"
#include "HUMContext.h"
#include "HUMObject.h"
#include "HUMPlatform.h"
#include "Utils.h"

using namespace std;

HUMEvent::HUMEvent(HUMCommandQueue* queue, HUMCommand* command) {
  context_ = queue->context();
  context_->Retain();

  queue_ = queue;
  queue_->Retain();
  command_type_ = command->type();
  status_ = HUM_QUEUED;

  profiled_ = queue->IsProfiled();
  if (profiled_)
    profile_[HUM_QUEUED] = GetTimestamp();

  pthread_mutex_init(&mutex_complete_, NULL);
  pthread_cond_init(&cond_complete_, NULL);
  pthread_mutex_init(&mutex_callbacks_, NULL);
}

HUMEvent::HUMEvent(HUMContext* context, HUMCommand* command) {
  context_ = context;
  context_->Retain();

  queue_ = NULL;
  command_type_ = command->type();
  status_ = HUM_SUBMITTED;

  profiled_ = false;

  pthread_mutex_init(&mutex_complete_, NULL);
  pthread_cond_init(&cond_complete_, NULL);
  pthread_mutex_init(&mutex_callbacks_, NULL);
}

HUMEvent::HUMEvent(HUMContext* context, bool profiled) 
{
  context_ = context;
  context_->Retain();

  queue_ = NULL;
  command_type_ = HUM_COMMAND_USER;
  status_ = HUM_SUBMITTED;

  profiled_ = profiled;

  pthread_mutex_init(&mutex_complete_, NULL);
  pthread_cond_init(&cond_complete_, NULL);
  pthread_mutex_init(&mutex_callbacks_, NULL);

}

HUMEvent::~HUMEvent() {
  if (queue_) queue_->Release();
  context_->Release();

  for (vector<EventCallback*>::iterator it = callbacks_.begin();
       it != callbacks_.end();
       ++it) {
    delete (*it);
  }

  pthread_mutex_destroy(&mutex_complete_);
  pthread_cond_destroy(&cond_complete_);
  pthread_mutex_destroy(&mutex_callbacks_);
}

hum_int HUMEvent::GetEventInfo(hum_event_info param_name, size_t param_value_size,
                             void* param_value, size_t* param_value_size_ret) {
  switch (param_name) {
    GET_OBJECT_INFO_T(HUM_EVENT_COMMAND_QUEUE, HUMCommandQueue*, queue_);
    GET_OBJECT_INFO_T(HUM_EVENT_CONTEXT, HUMContext*, context_);
    GET_OBJECT_INFO(HUM_EVENT_COMMAND_TYPE, hum_command_type, command_type_);
    GET_OBJECT_INFO_T(HUM_EVENT_COMMAND_EXECUTION_STATUS, hum_int, status_);
    GET_OBJECT_INFO_T(HUM_EVENT_REFERENCE_COUNT, hum_uint, ref_cnt());
    default: return HUM_INVALID_VALUE;
  }
  return HUM_SUCCESS;
}

hum_int HUMEvent::GetEventProfilingInfo(hum_profiling_info param_name,
                                      size_t param_value_size,
                                      void* param_value,
                                      size_t* param_value_size_ret) {
  if (!profiled_) return HUM_PROFILING_INFO_NOT_AVAILABLE;
  if (status_ != HUM_COMPLETE) return HUM_PROFILING_INFO_NOT_AVAILABLE;
  switch (param_name) {
    GET_OBJECT_INFO(HUM_PROFILING_COMMAND_QUEUED, hum_ulong,
                    profile_[HUM_QUEUED]);
    GET_OBJECT_INFO(HUM_PROFILING_COMMAND_SUBMIT, hum_ulong,
                    profile_[HUM_SUBMITTED]);
    GET_OBJECT_INFO(HUM_PROFILING_COMMAND_START, hum_ulong,
                    profile_[HUM_RUNNING]);
    GET_OBJECT_INFO(HUM_PROFILING_COMMAND_END, hum_ulong, profile_[HUM_COMPLETE]);
    default: return HUM_INVALID_VALUE;
  }
  return HUM_SUCCESS;
}

hum_int HUMEvent::SetUserEventStatus(hum_int execution_status) {
  if (command_type_ != HUM_COMMAND_USER) return HUM_INVALID_EVENT;
  if (status_ == HUM_COMPLETE || status_ < 0)
    return HUM_INVALID_OPERATION;
  SetStatus(execution_status);
  return HUM_SUCCESS;
}

void HUMEvent::SetStatus(hum_int status) {
	HUM_DEV("SetStatus(%d -> %d)", status_, status);

  if (status == HUM_COMPLETE || status < 0) {
    pthread_mutex_lock(&mutex_complete_);
    status_ = status;
    pthread_cond_broadcast(&cond_complete_);
    pthread_mutex_unlock(&mutex_complete_);
  } else {
    status_ = status;
  }

  vector<EventCallback*> target_callbacks;
  target_callbacks.reserve(callbacks_.size());
  pthread_mutex_lock(&mutex_callbacks_);
  for (vector<EventCallback*>::iterator it = callbacks_.begin();
       it != callbacks_.end();
       ++it) {
    if ((*it)->hit(status))
      target_callbacks.push_back(*it);
  }
  pthread_mutex_unlock(&mutex_callbacks_);

  for (vector<EventCallback*>::iterator it = target_callbacks.begin();
       it != target_callbacks.end();
       ++it) {
    (*it)->run(this, status);
  }

  if (status == HUM_COMPLETE || status < 0)
    HUMPlatform::GetPlatform()->InvokeAllSchedulers();
}

hum_int HUMEvent::Wait() {
  pthread_mutex_lock(&mutex_complete_);
  if (status_ != HUM_COMPLETE && status_ > 0)
    pthread_cond_wait(&cond_complete_, &mutex_complete_);
  pthread_mutex_unlock(&mutex_complete_);
  return status_;
}

void HUMEvent::AddCallback(EventCallback* callback) {
  pthread_mutex_lock(&mutex_callbacks_);
  bool passed = callback->passed(status_);
  if (!passed)
    callbacks_.push_back(callback);
  pthread_mutex_unlock(&mutex_callbacks_);
  if (passed)
    callback->run(this, status_);
}

hum_ulong HUMEvent::GetTimestamp() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  hum_ulong ret = t.tv_sec;
  ret *= 1000000000;
  ret += t.tv_nsec;
  return ret;
}
