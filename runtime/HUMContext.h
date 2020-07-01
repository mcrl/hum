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

#ifndef __HUM_CONTEXT_H__
#define __HUM_CONTEXT_H__

#include "HUMObject.h"
#include <vector>
#include <pthread.h>
#include "Callbacks.h"

class HUMDevice;
class HUMMem;
class HUMMicroMem;
class ContextErrorNotificationCallback;

class HUMContext: public HUMObject<HUMContext>
{
	public:
		HUMContext(const std::vector<HUMDevice*>& devices, 
				size_t num_properties,
        const hum_context_properties* properties);
		virtual ~HUMContext();

		const std::vector<HUMDevice*>& devices() const { return devices_; }

		bool IsValidDevice(HUMDevice* device);
		bool IsValidDevices(hum_uint num_devices, const hum_device_handle* device_list);
		bool IsValidMem(hum_mem_handle mem);

		void AddMem(HUMMem* mem);
		void RemoveMem(HUMMem* mem);

		void SetErrorNotificationCallback(
				ContextErrorNotificationCallback* callback);
		void NotifyError(const char* errinfo, 
				const void* private_info, size_t cb);

	private:
		std::vector<HUMDevice*> devices_;
		std::vector<HUMMem*> mems_;

		size_t num_properties_;
		hum_context_properties* properties_;

		ContextErrorNotificationCallback* callback_;
		pthread_mutex_t mutex_mems_;
};


#endif //__HUM_CONTEXT_H__
