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

#ifndef __HUM_CALLBACKS_H__
#define __HUM_CALLBACKS_H__


#include "hum.h"
#include "HUMMem.h"

class HUMEvent;

class ContextErrorNotificationCallback {
 public:
  ContextErrorNotificationCallback(
      void (HUM_CALLBACK *pfn_notify)(const char*, const void*, size_t, void*),
      void* user_data) {
    pfn_notify_ = pfn_notify;
    user_data_ = user_data;
  }

  void run(const char* errinfo, const void* private_info, size_t cb) {
    pfn_notify_(errinfo, private_info, cb, user_data_);
  }

 private:
  void (HUM_CALLBACK *pfn_notify_)(const char*, const void*, size_t, void*);
  void* user_data_;
};

class EventCallback {
 public:
  EventCallback(void (HUM_CALLBACK *pfn_notify)(HUMEvent*, hum_int, void*),
                void* user_data, hum_int command_exec_callback_type) {
    pfn_notify_ = pfn_notify;
    user_data_ = user_data;
    command_exec_callback_type_ = command_exec_callback_type;
  }

  bool passed(hum_int status) {
    return (status <= command_exec_callback_type_);
  }

  bool hit(hum_int status) {
    hum_int compare = (status < 0 ? HUM_COMPLETE : status);
    return (command_exec_callback_type_ == compare);
  }

  void run_if(HUMEvent* event, hum_int status) {
    hum_int compare = (status < 0 ? HUM_COMPLETE : status);
    if (command_exec_callback_type_ == compare)
      pfn_notify_(event, status, user_data_);
  }

  void run(HUMEvent* event, hum_int status) {
    pfn_notify_(event, status, user_data_);
  }

 private:
  void (HUM_CALLBACK* pfn_notify_)(HUMEvent*, hum_int, void*);
  void* user_data_;
  hum_int command_exec_callback_type_;
};

class MemObjectDestructorCallback {
 public:
  MemObjectDestructorCallback(void (HUM_CALLBACK *pfn_notify)(HUMMem*, void*),
                              void* user_data) {
    pfn_notify_ = pfn_notify;
    user_data_ = user_data;
  }

  void run(HUMMem* mem) {
    pfn_notify_(mem, user_data_);
  }

 private:
  void (HUM_CALLBACK *pfn_notify_)(HUMMem*, void*);
  void* user_data_;
};

#endif // __HUM_CALLBACKS_H__
