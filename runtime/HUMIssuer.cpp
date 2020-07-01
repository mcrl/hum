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

#include "HUMIssuer.h"
#include <algorithm>
#include <list>
#include <vector>
#include <pthread.h>
#include "HUMCommand.h"
#include "HUMDevice.h"
#include "HUMEvent.h"

using namespace std;

HUMIssuer::HUMIssuer(HUMDevice* device, bool blocking) {
  blocking_ = blocking;

	HUM_DEV("Issuer %p add device %p", this, device);
  devices_.push_back(device);
  devices_updated_ = true;

  thread_ = (pthread_t)NULL;
  thread_running_ = false;

  pthread_mutex_init(&mutex_devices_, NULL);
  pthread_cond_init(&cond_devices_remove_, NULL);
}

HUMIssuer::HUMIssuer(bool blocking) {
  blocking_ = blocking;

  thread_ = (pthread_t)NULL;
  thread_running_ = false;

  pthread_mutex_init(&mutex_devices_, NULL);
  pthread_cond_init(&cond_devices_remove_, NULL);
}

HUMIssuer::~HUMIssuer() {
  Stop();

  pthread_mutex_destroy(&mutex_devices_);
  pthread_cond_destroy(&cond_devices_remove_);
}

void HUMIssuer::Start() {
  if (!thread_) {
    thread_running_ = true;
    pthread_create(&thread_, NULL, &HUMIssuer::ThreadFunc, this);
  }
}

void HUMIssuer::Stop() {
  if (thread_) {
    thread_running_ = false;
    pthread_mutex_lock(&mutex_devices_);
    for (vector<HUMDevice*>::iterator it = devices_.begin();
         it != devices_.end();
         ++it) {
      (*it)->InvokeReadyQueue();
    }
    pthread_mutex_unlock(&mutex_devices_);
    pthread_join(thread_, NULL);
    thread_ = (pthread_t)NULL;
  }
}

HUMDevice* HUMIssuer::GetFirstDevice() {
	HUMDevice* device = NULL;
  pthread_mutex_lock(&mutex_devices_);
	if(devices_.size() > 0) {
	  device = devices_.front();
	}
  pthread_mutex_unlock(&mutex_devices_);
  return device;
}

void HUMIssuer::AddDevice(HUMDevice* device) {
  if (blocking_) return;

  pthread_mutex_lock(&mutex_devices_);
  devices_updated_ = true;
  devices_.push_back(device);
  pthread_mutex_unlock(&mutex_devices_);
}

void HUMIssuer::RemoveDevice(HUMDevice* device) {
  if (blocking_) return;

  pthread_mutex_lock(&mutex_devices_);
  vector<HUMDevice*>::iterator it = find(devices_.begin(), devices_.end(),
                                        device);
  if (it != devices_.end()) {
    devices_updated_ = true;
    *it = NULL;
    for (vector<HUMDevice*>::iterator other_it = devices_.begin();
         other_it != devices_.end();
         ++other_it) {
      if (*other_it != NULL)
        (*other_it)->InvokeReadyQueue();
    }
    pthread_cond_wait(&cond_devices_remove_, &mutex_devices_);
  }
  pthread_mutex_unlock(&mutex_devices_);
}

void HUMIssuer::Run() {
	vector<HUMDevice*> target_devices;
	int cnt = 0;
	while (thread_running_) {
		if (devices_updated_) {
			pthread_mutex_lock(&mutex_devices_);
			pthread_cond_broadcast(&cond_devices_remove_);
			devices_.erase(remove(devices_.begin(), devices_.end(), (HUMDevice*)NULL),
					devices_.end());
			target_devices = devices_;
			devices_updated_ = false;
			pthread_mutex_unlock(&mutex_devices_);
		}

		for (vector<HUMDevice*>::iterator it = target_devices.begin();
				it != target_devices.end();
				++it) {

			HUMDevice* device = *it;
			if (blocking_) {
				device->WaitReadyQueue();
			}
			HUMCommand* command = device->DequeueReadyQueue();

			if (command != NULL) {
				//SingleDevice* sd = dynamic_cast<SingleDevice*>(device);

				command->SetAsRunning();
				if (!blocking_) {
					running_commands_.push_back(command);
					command->Execute();
					HUM_DEV("HUMIssuer(%p) Get Command %p cmd=%x end", this, command, command->type());
				} 
				else {
					command->Execute();
					command->SetAsComplete();
					HUM_DEV("HUMIssuer(%p) Get Command %p cmd=%x end", this, command, command->type());
					delete command;
				}
			}
		}

		if (!blocking_) {
			list<HUMCommand*>::iterator it = running_commands_.begin();
			while (it != running_commands_.end()) {
				HUMCommand* command = *it;
				if (command->device()->IsComplete(command)) {
					command->SetAsComplete();
					it = running_commands_.erase(it);

					HUM_DEV("cmd %p type=%x is completed", command, command->type());
					delete command;
				}
				else {
					//HUM_DEV("Cmd %p is not completed", command);
					++it;
				}
			}
		}
	}
}

void* HUMIssuer::ThreadFunc(void *argp) {
  ((HUMIssuer*)argp)->Run();
  return NULL;
}
