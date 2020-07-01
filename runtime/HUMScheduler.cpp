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

#include "HUMScheduler.h"
#include <algorithm>
#include <vector>
#include <pthread.h>
#include <semaphore.h>
#include "HUMCommand.h"
#include "HUMCommandQueue.h"
#include "HUMDevice.h"
#include "HUMEvent.h"
#include "HUMPlatform.h"

using namespace std;

HUMScheduler::HUMScheduler(HUMPlatform* platform, bool busy_waiting) {
  platform_ = platform;
  busy_waiting_ = busy_waiting;
  queues_updated_ = false;
  thread_ = (pthread_t)NULL;
  thread_running_ = false;
  if (!busy_waiting_)
    sem_init(&sem_schedule_, 0, 0);
  pthread_mutex_init(&mutex_queues_, NULL);
  pthread_cond_init(&cond_queues_remove_, NULL);
}

HUMScheduler::~HUMScheduler() {
  Stop();
  if (!busy_waiting_)
    sem_destroy(&sem_schedule_);
  pthread_mutex_destroy(&mutex_queues_);
  pthread_cond_destroy(&cond_queues_remove_);
}

void HUMScheduler::Start() {
  if (!thread_) {
    thread_running_ = true;
    pthread_create(&thread_, NULL, &HUMScheduler::ThreadFunc, this);
  }
}

void HUMScheduler::Stop() {
  if (thread_) {
    thread_running_ = false;
    Invoke();
    pthread_join(thread_, NULL);
    thread_ = (pthread_t)NULL;
  }
}

void HUMScheduler::Invoke() {
  if (!busy_waiting_)
    sem_post(&sem_schedule_);
}

void HUMScheduler::AddCommandQueue(HUMCommandQueue* queue) {
  pthread_mutex_lock(&mutex_queues_);
  queues_updated_ = true;
  queues_.push_back(queue);
  pthread_mutex_unlock(&mutex_queues_);
}

void HUMScheduler::RemoveCommandQueue(HUMCommandQueue* queue) {
  pthread_mutex_lock(&mutex_queues_);
  vector<HUMCommandQueue*>::iterator it = find(queues_.begin(), queues_.end(),
                                              queue);
  if (it != queues_.end()) {
    queues_updated_ = true;
    *it = NULL;
    if (!busy_waiting_)
      sem_post(&sem_schedule_);
    pthread_cond_wait(&cond_queues_remove_, &mutex_queues_);
  }
  pthread_mutex_unlock(&mutex_queues_);
}

size_t HUMScheduler::GetNumCommandQueues() {
  pthread_mutex_lock(&mutex_queues_);
	size_t ret = queues_.size();
  pthread_mutex_unlock(&mutex_queues_);
	return ret;
}

HUMCommandQueue* HUMScheduler::GetCommandQueue(int idx) {
  pthread_mutex_lock(&mutex_queues_);
	HUMCommandQueue* ret =queues_[idx];
  pthread_mutex_unlock(&mutex_queues_);
	return ret;
}



void HUMScheduler::Run() {
  vector<HUMCommandQueue*> target_queues;
  int chance_count = 0;

  while (thread_running_) {
    if (!busy_waiting_) {
      if (chance_count == 10) {
        chance_count = 0;
        sem_wait(&sem_schedule_);
      }
      else {
        chance_count++;
      }
    }

    if (queues_updated_) {
      pthread_mutex_lock(&mutex_queues_);
      pthread_cond_broadcast(&cond_queues_remove_);
      queues_.erase(remove(queues_.begin(), queues_.end(),
                           (HUMCommandQueue*)NULL),
                    queues_.end());
      target_queues = queues_;
      queues_updated_ = false;
      pthread_mutex_unlock(&mutex_queues_);
    }

    for (vector<HUMCommandQueue*>::iterator it = target_queues.begin();
         it != target_queues.end();
         ++it) {
      HUMCommandQueue* queue = *it;
      HUMCommand* command = queue->Peek();
      if (command != NULL && command->ResolveConsistency()) {
        command->Submit();
        queue->Dequeue(command);
      }
    }
  }
}

void* HUMScheduler::ThreadFunc(void* argp) {
  ((HUMScheduler*)argp)->Run();
  return NULL;
}
