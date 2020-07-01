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

#ifndef __HUM_EVENT_H__
#define __HUM_EVENT_H_

#include <vector>
#include <pthread.h>
#include "HUMObject.h"

class HUMCommand;
class HUMCommandQueue;
class HUMContext;
class EventCallback;

class HUMEvent: public HUMObject<HUMEvent> 
{
 public:
  HUMEvent(HUMCommandQueue* queue, HUMCommand* command);
  HUMEvent(HUMContext* context, HUMCommand* command);
  HUMEvent(HUMContext* context, bool profiled = false);
  virtual ~HUMEvent();

  HUMContext* context() const { return context_; }
  HUMCommandQueue* queue() const { return queue_; }

  hum_int GetEventInfo(hum_event_info param_name, size_t param_value_size,
                      void* param_value, size_t* param_value_size_ret);
  hum_int GetEventProfilingInfo(hum_profiling_info param_name,
                               size_t param_value_size, void* param_value,
                               size_t* param_value_size_ret);

  hum_int SetUserEventStatus(hum_int execution_status);

  bool IsComplete() const {
		assert(status_ >= 0);
    return (status_ == HUM_COMPLETE || status_ < 0);
  }
  bool IsError() const {
    return (status_ < 0);
  }

	bool IsProfiling() const {
		return profiled_;
	}

	hum_ulong GetCompleteTime() const {
		return profile_[HUM_COMPLETE];
	}

	hum_int GetStatus() const {
		return status_;
	}

  void SetStatus(hum_int status);
  hum_int Wait();

  void AddCallback(EventCallback* callback);

 private:
  hum_ulong GetTimestamp();

  HUMContext* context_;
  HUMCommandQueue* queue_;
  hum_command_type command_type_;
  hum_int status_;

  std::vector<EventCallback*> callbacks_;

  bool profiled_;
  hum_ulong profile_[4];

  pthread_mutex_t mutex_complete_;
  pthread_cond_t cond_complete_;
  pthread_mutex_t mutex_callbacks_;
};


#endif //__HUM_EVENT_H__
