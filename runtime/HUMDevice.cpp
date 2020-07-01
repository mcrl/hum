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

#include "HUMDevice.h"
#include "HUMPlatform.h"
#include "HUMScheduler.h"
#include "HUMCommand.h"
#include "HUMEvent.h"
#include "Utils.h"
#include <assert.h>

using namespace std;

HUMDevice::HUMDevice(HUMPlatform* platform, int node_id, bool add_device_to_platform /*=true*/) 
:	node_id_(node_id),
	ready_queue_(READY_QUEUE_SIZE)
{
	device_type_ = 0;

	platform_ = platform;

	if(add_device_to_platform) {
	  platform_->AddDevice(this);
	}

  scheduler_ = platform_->AllocIdleScheduler();

	available_ = false;
	model_type_ = HUM_MODEL_TYPE_NONE;

  sem_init(&sem_ready_queue_, 0, 0);
}

HUMDevice::~HUMDevice() 
{
	sem_destroy(&sem_ready_queue_);

}

void HUMDevice::Cleanup() {
  platform_->RemoveDevice(this);
}

int HUMDevice::GetDistance(HUMDevice* other) const 
{
  if (this == other) {
    return 0;
  } 
	else if (other == LATEST_HOST) {
    if (node_id_ == 0) {
      if (type() == HUM_DEVICE_TYPE_CPU)
        return 1;
      else
        return 2;
    } 
		else if (type() == HUM_DEVICE_TYPE_CPU) {
      return 4;
    } 
		else {
      return 5;
    }
  } 
	else if (node_id_ == other->node_id_) {
    if (type() == HUM_DEVICE_TYPE_CPU && other->type() == HUM_DEVICE_TYPE_CPU)
      return 1;
    else if (type() == HUM_DEVICE_TYPE_CPU || other->type() == HUM_DEVICE_TYPE_CPU)
      return 2;
    else
      return 3;
  } 
	else if (type() == HUM_DEVICE_TYPE_CPU &&
             other->type() == HUM_DEVICE_TYPE_CPU) {
    return 4;
  } 
	else if (type() == HUM_DEVICE_TYPE_CPU ||
             other->type() == HUM_DEVICE_TYPE_CPU) {
    return 5;
  } 
	else {
    return 6;
  }
}

void* HUMDevice::AllocKernel(HUMKernel* kernel) {
  return NULL;
}

void HUMDevice::FreeKernel(HUMKernel* kernel, void* dev_specific) {
  // Do nothing
}

void HUMDevice::AddCommandQueue(HUMCommandQueue* queue) {
  scheduler_->AddCommandQueue(queue);
}

void HUMDevice::RemoveCommandQueue(HUMCommandQueue* queue) {
  scheduler_->RemoveCommandQueue(queue);
}

size_t HUMDevice::GetNumCommandQueues() {
	return scheduler_->GetNumCommandQueues();
}

HUMCommandQueue* HUMDevice::GetCommandQueue(int idx) {
	return scheduler_->GetCommandQueue(idx);
}



void HUMDevice::InvokeScheduler() {
  scheduler_->Invoke();
}

void HUMDevice::EnqueueReadyQueue(HUMCommand* command) {
  while (!ready_queue_.Enqueue(command)) {}
  sem_post(&sem_ready_queue_);
}

HUMCommand* HUMDevice::DequeueReadyQueue() {
  HUMCommand* command;
  if (ready_queue_.Dequeue(&command))
    return command;
  else
    return NULL;
}

void HUMDevice::InvokeReadyQueue() {
  sem_post(&sem_ready_queue_);
}

void HUMDevice::WaitReadyQueue() {
  sem_wait(&sem_ready_queue_);
}

bool HUMDevice::IsComplete(HUMCommand* command) {
  return true;
}
