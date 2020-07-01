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

#ifndef __HUM_ISSUER_H__
#define __HUM_ISSUER_H__

#include <list>
#include <vector>
#include <pthread.h>

class HUMCommand;
class HUMDevice;

class HUMIssuer {
 public:
  HUMIssuer(HUMDevice* device, bool blocking);
  HUMIssuer(bool blocking);
  ~HUMIssuer();

  void Start();
  void Stop();

  HUMDevice* GetFirstDevice();
  void AddDevice(HUMDevice* device);
  void RemoveDevice(HUMDevice* device);

 private:
  void Run();

  bool blocking_;
  std::vector<HUMDevice*> devices_;
  bool devices_updated_;

  pthread_t thread_;
  bool thread_running_;

  std::list<HUMCommand*> running_commands_;

  pthread_mutex_t mutex_devices_;
  pthread_cond_t cond_devices_remove_;

  static void* ThreadFunc(void* argp);
};

#endif // __HUM_ISSUER_H__