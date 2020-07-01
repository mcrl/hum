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

#ifndef __HUM_PLATFORM_H__
#define __HUM_PLATFORM_H__

#include <vector>
#include <map>
#include "HUMObject.h"
#include <pthread.h>
#include <list>

class HUMDevice;
class HUMScheduler;
class HUMIssuer;
class HUMContext;
//class HUMCudaStream;
class ContextErrorNotificationCallback;

class CUDAPlatform;


class HUMPlatform: public HUMObject<HUMPlatform> {
	public:
		HUMPlatform();
		virtual ~HUMPlatform();

		virtual void Init();

		hum_int GetPlatformInfo(hum_platform_info param_name, size_t param_value_size,
				void* param_value, size_t* param_value_size_ret);
		hum_int GetDeviceIDs(hum_device_type device_type, hum_uint num_entries,
				hum_device_handle* devices, hum_uint* num_devices);
		HUMContext* CreateContextFromDevices(
				const hum_context_properties* properties, hum_uint num_devices,
				const hum_device_handle* devices, ContextErrorNotificationCallback* callback,
				hum_int* err);
		HUMContext* CreateContextFromType(const hum_context_properties* properties,
				hum_device_type device_type,
				ContextErrorNotificationCallback* callback,
				hum_int* err);


		//Devices
		void GetDevices(std::vector<HUMDevice*>& devices);
		HUMDevice* GetFirstDevice();
		virtual void AddDevice(HUMDevice* device);
		virtual void RemoveDevice(HUMDevice* device);
		virtual void AttachDevice(HUMDevice* device);
		virtual void DetachDevice(HUMDevice* device);

		//Schedulers
		HUMScheduler* AllocIdleScheduler();
		void InvokeAllSchedulers();


		bool IsHost() const { return is_host_; }
		bool IsCuda() const { return is_cuda_; }
	protected:
		//Platform information
		const char* profile_;
		const char* version_;
		const char* name_;
		const char* vendor_;
		const char* extensions_;
		const char* suffix_;

		hum_device_type default_device_type_;

		//Host or Compute
		bool is_host_;
		bool is_cuda_;

		std::vector<HUMDevice*> devices_;
		std::vector<HUMScheduler*> schedulers_;
		std::vector<HUMIssuer*> issuers_;
		pthread_mutex_t mutex_devices_;
		pthread_mutex_t mutex_issuers_;

		void InitSchedulers(unsigned int num_scheduler, bool busy_waiting);
		void AddIssuer(HUMIssuer* issuer);
		void RemoveIssuerOfDevice(HUMDevice* device);
		void AddDeviceToFirstIssuer(HUMDevice* device);
		void AddDeviceToSecondIssuer(HUMDevice* device);
		void RemoveDeviceFromFirstIssuer(HUMDevice* device);
		void RemoveDeviceFromSecondIssuer(HUMDevice* device);

		//Singleton
	public:
		static HUMPlatform* GetPlatform(int rank = -1);
		static CUDAPlatform* GetCudaPlatform(int rank = -1);

	private:
		static HUMPlatform* singleton_;
		static mutex_t mutex_;
};

class CUDAPlatform: public HUMPlatform 
{
	public:
		CUDAPlatform();
		virtual ~CUDAPlatform();

		void InitCuda();

		HUMContext* context() const { return context_; }
		HUMDevice* GetDeviceById(int device_id);	
		HUMCommandQueue* GetDefaultQueue(HUMDevice* device);

		hum_int GetNumCudaDevices() { return num_cuda_devices_; }

		void AddBlockingQueue(HUMDevice* device, HUMCommandQueue* stream);
		void RemoveBlockingQueue(HUMDevice* device, HUMCommandQueue* stream);
		std::list<HUMCommandQueue*>* GetBlockingQueue(HUMDevice* device);

	private:
		HUMContext* context_;
		hum_device_handle* cuda_devices_;
		int num_cuda_devices_;

		std::map<HUMDevice*, HUMCommandQueue*> default_queues_;
		std::map<HUMDevice*, std::list<HUMCommandQueue*> > blocking_queues_;

		mutex_t mutex_default_queue_;
		mutex_t mutex_blocking_queue_;
};


#endif //__HUM_PLATFORM_H__
