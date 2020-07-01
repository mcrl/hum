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

#include "CUDAWrapper.h"
#include <cuda_runtime.h>
#include <cuda.h>

#include "HUMCommand.h"
#include "HUMCommandQueue.h"
#include "HUMDevice.h"
#include "HUMPlatform.h"
#include "HUMEvent.h"
#include "MemoryRegion.h"
#include "Device/NVLibs.h"

#include <cxxabi.h>
#include <omp.h>

#include <stdarg.h>
#include <malloc.h>
#include <map>

#define TODO() \
	HUM_ERROR("*** %s function is not implemented yet ***", __FUNCTION__); assert(0);\
	return cudaSuccess;

#define NYI_ERROR(funcname) \
  printf("*** Function \"%s\" not yet implemented! ***\n", funcname);

#define SET_AND_RETURN_ERROR(X) \
  cuda_last_error_ = X;          \
	assert(X == 0); \
  return X;

//#define LOCK_STEP

std::map<std::string, const void*> g_cuda_func_map_d2h;
std::map<const void*, std::string> g_cuda_func_map_h2d;
std::map<std::string, const void*> g_cuda_var_map_d2h;
std::map<const void*, std::string> g_cuda_var_map_h2d;



__thread unsigned int cuda_last_error_;
__thread int current_device_id_;

__thread void *thread_stack_;
__thread size_t thread_stacksize_ = 0;

extern MemoryRegion<HUMMem>* g_MemRegion_;

void set_stack_info() {
  if (thread_stacksize_ == 0) {
    pthread_t self = pthread_self();
    pthread_attr_t attr;
    pthread_getattr_np(self, &attr);
    pthread_attr_getstack(&attr, &thread_stack_, &thread_stacksize_);
  }
}

int is_on_stack(const void *ptr) {
  set_stack_info();
  return ((uintptr_t) ptr >= (uintptr_t) thread_stack_
          && (uintptr_t) ptr < (uintptr_t) thread_stack_ + thread_stacksize_);
}

CUDAWrapper::CUDAWrapper()
{
	nvlibs_ = new NVLibs();
  pthread_mutex_init(&mutex_, NULL);

}

CUDAWrapper::~CUDAWrapper()
{
	delete(nvlibs_);
  pthread_mutex_destroy(&mutex_);
}

void CUDAWrapper::InitModule(void **fatCubinHandle)
{
	HUM_DEV("Call InitModule fatCubinHandle=%p", fatCubinHandle);
	nvlibs_->__cudaInitModule(fatCubinHandle);
}

void CUDAWrapper::RegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
		const char *deviceName, int thread_limit, uint3 *tid,
		uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) 
{
	HUM_DEV("Call RegisterFunction fatCubinHandle=%p, hostFun=%p, devFunc=%s", fatCubinHandle, hostFun, deviceFun);

	g_cuda_func_map_h2d[(void*)hostFun] = deviceFun;
	g_cuda_func_map_d2h[deviceFun] = (void*)hostFun;

	nvlibs_->__cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
		deviceName, thread_limit, tid,
		bid, bDim, gDim, wSize);
}

void CUDAWrapper::RegisterVar(void **fatCubinHandle, 	char  *hostVar,	char  *deviceAddress,
				const char *deviceName, int ext, size_t size,
				int constant, int global) 
{
	HUM_DEV("Call RegisterVar fatCubinHandle=%p, hostVar=%p, devAddress=%s, deviceName=%s", fatCubinHandle, hostVar, deviceAddress, deviceName);

	g_cuda_var_map_h2d[(void*)hostVar] = deviceAddress;
	g_cuda_var_map_d2h[deviceAddress] = (void*)hostVar;

	nvlibs_->__cudaRegisterVar(fatCubinHandle, hostVar, deviceAddress,
				deviceName, ext, size, constant, global);
}

void CUDAWrapper::RegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
				const void **deviceAddress, const char *deviceName,
				int dim, int norm, int ext)
{
	HUM_DEV("Call RegisterTexture fatCubinHandle=%p, hostVar=%p, devAddress=%p, deviceName=%s", fatCubinHandle, hostVar, deviceAddress, deviceName);

	nvlibs_->__cudaRegisterTexture(fatCubinHandle, hostVar,
		deviceAddress, deviceName, dim, norm, ext);
}

void** CUDAWrapper::RegisterFatBinary(void * fatCubin) {
	HUM_DEV("Call RegisterFatBinary fatCubin=%p", fatCubin);
	return nvlibs_->__cudaRegisterFatBinary(fatCubin);
}

void CUDAWrapper::RegisterFatBinaryEnd(void **fatCubinHandle) {
  nvlibs_->__cudaRegisterFatBinaryEnd(fatCubinHandle);
}

void CUDAWrapper::UnregisterFatBinary(void **fatCubinHandle)
{
	HUM_DEV("Call UnregisterFatBinary fatCubinHandle=%p", fatCubinHandle);
	nvlibs_->__cudaUnregisterFatBinary(fatCubinHandle);
}

cudaError_t CUDAWrapper::PopCallConfiguration(dim3 *gridDim,
  dim3 *blockDim,  size_t *sharedMem, void *stream)
{
	HUM_DEV("Call PopCallConfiguration: stream=%p", stream);

	return nvlibs_->__cudaPopCallConfiguration(gridDim,
		blockDim, sharedMem, stream);
}

unsigned CUDAWrapper::PushCallConfiguration(dim3 gridDim,
  dim3 blockDim,  size_t sharedMem, void *stream)
{
	HUM_DEV("Call PushCallConfiguration: stream=%p", stream);

	return nvlibs_->__cudaPushCallConfiguration(gridDim,
		blockDim, sharedMem, stream);
}


//=============================================================================
// CUDA Runtime API
// 5.1. Device Management
//=============================================================================
cudaError_t CUDAWrapper::DeviceSynchronize(void) 
{
  CUDAPlatform* platform = HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);

	size_t num_queues = device->GetNumCommandQueues();
	HUMEvent* blocking[num_queues];

	for(int i=0;i<num_queues;i++) {
		HUMCommandQueue* queue = device->GetCommandQueue(i);
		HUMCommand* command = HUMCommand::CreateMarker(NULL, NULL, queue);
		if (command == NULL) return cudaErrorMemoryAllocation;

		blocking[i] = command->ExportEvent();
		queue->Enqueue(command);
	}

	for(int i=0;i<num_queues;i++) {
		blocking[i]->Wait();
		blocking[i]->Release();
	}

  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::DeviceReset(void) {
	TODO();
}

cudaError_t CUDAWrapper::DeviceGetAttribute(int* pi, cudaDeviceAttr attr, int deviceId)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(deviceId);
	const cuda_dev_info_t* dev_info = device->cuda_dev_info();

	switch(attr) {
	case cudaDevAttrMultiProcessorCount:
		*pi = dev_info->multiProcessorCount;
		printf("cudaDevAttrMultiProcessorCount: %d\n", *pi);
		break;
	case cudaDevAttrMaxSharedMemoryPerMultiprocessor:
		*pi = dev_info->sharedMemPerMultiprocessor;
		printf("cudaDevAttrMaxSharedMemoryPerMultiprocessor: %d\n", *pi);
		break;
	case cudaDevAttrMaxThreadsPerMultiProcessor:
		*pi = dev_info->maxThreadsPerMultiProcessor;
		printf("cudaDevAttrMaxThreadsPerMultiProcessor: %d\n", *pi);
		break;
	case cudaDevAttrMaxThreadsPerBlock:
		*pi = dev_info->maxThreadsPerBlock;
		printf("cudaDevAttrMaxThreadsPerBlock: %d\n", *pi);
		break;
	case cudaDevAttrMaxRegistersPerBlock:
		*pi = dev_info->regsPerBlock;
		printf("cudaDevAttrMaxRegistersPerBlock: %d\n", *pi);
		break;
	case cudaDevAttrWarpSize:
		*pi = dev_info->warpSize;
		printf("cudaDevAttrWarpSize: %d\n", *pi);
		break;
	case cudaDevAttrMaxGridDimX:
		*pi = dev_info->maxGridSize[0];
		printf("cudaDevAttrMaxGridDimX: %d\n", *pi);
		break;
	case cudaDevAttrMaxGridDimY:
		*pi = dev_info->maxGridSize[1];
		printf("cudaDevAttrMaxGridDimY: %d\n", *pi);
		break;
	case cudaDevAttrMaxGridDimZ:
		*pi = dev_info->maxGridSize[2];
		printf("cudaDevAttrMaxGridDimZ: %d\n", *pi);
		break;
	case cudaDevAttrMaxSharedMemoryPerBlock:
		*pi = dev_info->sharedMemPerBlock;
		printf("cudaDevAttrMaxSharedMemoryPerBlock: %d\n", *pi);
		break;
	case cudaDevAttrEccEnabled:
		*pi = dev_info->ECCEnabled;
		printf("cudaDevAttrEccEnabled: %d\n", *pi);
		break;
	case cudaDevAttrMemoryClockRate:
		*pi = dev_info->memoryClockRate;
		printf("cudaDevAttrMemoryClockRate: %d\n", *pi);
		break;
	case cudaDevAttrGlobalMemoryBusWidth:
		*pi = dev_info->memoryBusWidth;
		printf("cudaDevAttrGlobalMemoryBusWidth: %d\n", *pi);
		break;
	case cudaDevAttrMaxRegistersPerMultiprocessor:
		*pi = dev_info->regsPerMultiprocessor;
		printf("cudaDevAttrMaxRegistersPerMultiprocessor: %d\n", *pi);
		break;
	default:
		printf("Unsupported cudaDeviceAttr: %d\n", (int)attr);
		assert(0);
		break;
	}


  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::SetDeviceFlags(unsigned int flags) {
  return nvlibs_->cudaSetDeviceFlags(flags);
}

cudaError_t CUDAWrapper::SetDevice(int deviceId)
{
	HUM_DEV("SetDevice deviceId = %d", deviceId);
  CUDAPlatform* platform = HUMPlatform::GetCudaPlatform(0);
  set_stack_info();
	if(platform->GetNumCudaDevices() <= deviceId) {
		printf("Error while trying to set device to %d\n", deviceId);
    	SET_AND_RETURN_ERROR(cudaErrorInvalidDevice);
	}
	
	current_device_id_ = deviceId;
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::GetDevice(int* deviceId)
{
  CUDAPlatform* platform = HUMPlatform::GetCudaPlatform(0);
  set_stack_info();
  *deviceId = current_device_id_;
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::GetDeviceCount(int* count)
{
  CUDAPlatform* platform = HUMPlatform::GetCudaPlatform(0);
  *count = platform->GetNumCudaDevices();
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::GetDeviceProperties(cudaDeviceProp* prop, int device_id)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(device_id);
  memcpy(prop, device->cuda_dev_info(), sizeof(cuda_dev_info_t));
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::FuncGetAttributes( cudaFuncAttributes *attr, const void *func) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	return nvlibs_->cudaFuncGetAttributes(attr, func);
}

cudaError_t CUDAWrapper::FuncSetAttribute(const void *func,
    enum cudaFuncAttribute attr, int value) {
  TODO();
//  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
//	return nvlibs_->cudaFuncSetAttribute(func, attr, value);
}


//=============================================================================
// CUDA Runtime API
// 5.3. Error Handling
//=============================================================================
const char* CUDAWrapper::GetErrorName(cudaError_t cudaError)
{
  switch (cudaError) {
    case cudaSuccess                           : return "cudaSuccess";
    case cudaErrorMissingConfiguration         : return "cudaErrorMissingConfiguration";
    case cudaErrorMemoryAllocation             : return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError          : return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure                : return "cudaErrorLaunchFailure";
    case cudaErrorPriorLaunchFailure           : return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchTimeout                : return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources         : return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction        : return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration         : return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice                : return "cudaErrorInvalidDevice";
    case cudaErrorInvalidValue                 : return "cudaErrorInvalidValue";
    case cudaErrorInvalidPitchValue            : return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol                : return "cudaErrorInvalidSymbol";
    case cudaErrorMapBufferObjectFailed        : return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed      : return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorInvalidHostPointer           : return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer         : return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture               : return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding        : return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor     : return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection       : return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant            : return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed           : return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound              : return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError         : return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting         : return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting           : return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution         : return "cudaErrorMixedDeviceExecution";
    case cudaErrorCudartUnloading              : return "cudaErrorCudartUnloading";
    case cudaErrorUnknown                      : return "cudaErrorUnknown";
    case cudaErrorNotYetImplemented            : return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge          : return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInvalidResourceHandle        : return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady                     : return "cudaErrorNotReady";
    case cudaErrorInsufficientDriver           : return "cudaErrorInsufficientDriver";
    case cudaErrorSetOnActiveProcess           : return "cudaErrorSetOnActiveProcess";
    case cudaErrorInvalidSurface               : return "cudaErrorInvalidSurface";
    case cudaErrorNoDevice                     : return "cudaErrorNoDevice";
    case cudaErrorECCUncorrectable             : return "cudaErrorECCUncorrectable";
    case cudaErrorSharedObjectSymbolNotFound   : return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed       : return "cudaErrorSharedObjectInitFailed";
    case cudaErrorUnsupportedLimit             : return "cudaErrorUnsupportedLimit";
    case cudaErrorDuplicateVariableName        : return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName         : return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName         : return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable           : return "cudaErrorDevicesUnavailable";
    case cudaErrorInvalidKernelImage           : return "cudaErrorInvalidKernelImage";
    case cudaErrorNoKernelImageForDevice       : return "cudaErrorNoKernelImageForDevice";
    case cudaErrorIncompatibleDriverContext    : return "cudaErrorIncompatibleDriverContext";
    case cudaErrorPeerAccessAlreadyEnabled     : return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled         : return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorDeviceAlreadyInUse           : return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorProfilerDisabled             : return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized       : return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted       : return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped       : return "cudaErrorProfilerAlreadyStopped";
    case cudaErrorAssert                       : return "cudaErrorAssert";
    case cudaErrorTooManyPeers                 : return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered  : return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered      : return "cudaErrorHostMemoryNotRegistered";
    case cudaErrorOperatingSystem              : return "cudaErrorOperatingSystem";
    case cudaErrorPeerAccessUnsupported        : return "cudaErrorPeerAccessUnsupported";
    case cudaErrorLaunchMaxDepthExceeded       : return "cudaErrorLaunchMaxDepthExceeded";
    case cudaErrorLaunchFileScopedTex          : return "cudaErrorLaunchFileScopedTex";
    case cudaErrorLaunchFileScopedSurf         : return "cudaErrorLaunchFileScopedSurf";
    case cudaErrorSyncDepthExceeded            : return "cudaErrorSyncDepthExceeded";
    case cudaErrorLaunchPendingCountExceeded   : return "cudaErrorLaunchPendingCountExceeded";
    case cudaErrorNotPermitted                 : return "cudaErrorNotPermitted";
    case cudaErrorNotSupported                 : return "cudaErrorNotSupported";
    case cudaErrorHardwareStackError           : return "cudaErrorHardwareStackError";
    case cudaErrorIllegalInstruction           : return "cudaErrorIllegalInstruction";
    case cudaErrorMisalignedAddress            : return "cudaErrorMisalignedAddress";
    case cudaErrorInvalidAddressSpace          : return "cudaErrorInvalidAddressSpace";
    case cudaErrorInvalidPc                    : return "cudaErrorInvalidPc";
    case cudaErrorIllegalAddress               : return "cudaErrorIllegalAddress";
    case cudaErrorInvalidPtx                   : return "cudaErrorInvalidPtx";
    case cudaErrorInvalidGraphicsContext       : return "cudaErrorInvalidGraphicsContext";
    case cudaErrorNvlinkUncorrectable          : return "cudaErrorNvlinkUncorrectable";
    case cudaErrorStartupFailure               : return "cudaErrorStartupFailure";
    case cudaErrorApiFailureBase               : return "cudaErrorApiFailureBase";
    default                                    : return "cudaInvalidError";
  }
}

const char* CUDAWrapper::GetErrorString(cudaError_t cudaError)
{
  switch (cudaError) {
    case cudaSuccess                           : return "cudaSuccess";
    case cudaErrorMissingConfiguration         : return "cudaErrorMissingConfiguration";
    case cudaErrorMemoryAllocation             : return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError          : return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure                : return "cudaErrorLaunchFailure";
    case cudaErrorPriorLaunchFailure           : return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchTimeout                : return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources         : return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction        : return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration         : return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice                : return "cudaErrorInvalidDevice";
    case cudaErrorInvalidValue                 : return "cudaErrorInvalidValue";
    case cudaErrorInvalidPitchValue            : return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol                : return "cudaErrorInvalidSymbol";
    case cudaErrorMapBufferObjectFailed        : return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed      : return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorInvalidHostPointer           : return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer         : return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture               : return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding        : return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor     : return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection       : return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant            : return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed           : return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound              : return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError         : return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting         : return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting           : return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution         : return "cudaErrorMixedDeviceExecution";
    case cudaErrorCudartUnloading              : return "cudaErrorCudartUnloading";
    case cudaErrorUnknown                      : return "cudaErrorUnknown";
    case cudaErrorNotYetImplemented            : return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge          : return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInvalidResourceHandle        : return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady                     : return "cudaErrorNotReady";
    case cudaErrorInsufficientDriver           : return "cudaErrorInsufficientDriver";
    case cudaErrorSetOnActiveProcess           : return "cudaErrorSetOnActiveProcess";
    case cudaErrorInvalidSurface               : return "cudaErrorInvalidSurface";
    case cudaErrorNoDevice                     : return "cudaErrorNoDevice";
    case cudaErrorECCUncorrectable             : return "cudaErrorECCUncorrectable";
    case cudaErrorSharedObjectSymbolNotFound   : return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed       : return "cudaErrorSharedObjectInitFailed";
    case cudaErrorUnsupportedLimit             : return "cudaErrorUnsupportedLimit";
    case cudaErrorDuplicateVariableName        : return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName         : return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName         : return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable           : return "cudaErrorDevicesUnavailable";
    case cudaErrorInvalidKernelImage           : return "cudaErrorInvalidKernelImage";
    case cudaErrorNoKernelImageForDevice       : return "cudaErrorNoKernelImageForDevice";
    case cudaErrorIncompatibleDriverContext    : return "cudaErrorIncompatibleDriverContext";
    case cudaErrorPeerAccessAlreadyEnabled     : return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled         : return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorDeviceAlreadyInUse           : return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorProfilerDisabled             : return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized       : return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted       : return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped       : return "cudaErrorProfilerAlreadyStopped";
    case cudaErrorAssert                       : return "cudaErrorAssert";
    case cudaErrorTooManyPeers                 : return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered  : return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered      : return "cudaErrorHostMemoryNotRegistered";
    case cudaErrorOperatingSystem              : return "cudaErrorOperatingSystem";
    case cudaErrorPeerAccessUnsupported        : return "cudaErrorPeerAccessUnsupported";
    case cudaErrorLaunchMaxDepthExceeded       : return "cudaErrorLaunchMaxDepthExceeded";
    case cudaErrorLaunchFileScopedTex          : return "cudaErrorLaunchFileScopedTex";
    case cudaErrorLaunchFileScopedSurf         : return "cudaErrorLaunchFileScopedSurf";
    case cudaErrorSyncDepthExceeded            : return "cudaErrorSyncDepthExceeded";
    case cudaErrorLaunchPendingCountExceeded   : return "cudaErrorLaunchPendingCountExceeded";
    case cudaErrorNotPermitted                 : return "cudaErrorNotPermitted";
    case cudaErrorNotSupported                 : return "cudaErrorNotSupported";
    case cudaErrorHardwareStackError           : return "cudaErrorHardwareStackError";
    case cudaErrorIllegalInstruction           : return "cudaErrorIllegalInstruction";
    case cudaErrorMisalignedAddress            : return "cudaErrorMisalignedAddress";
    case cudaErrorInvalidAddressSpace          : return "cudaErrorInvalidAddressSpace";
    case cudaErrorInvalidPc                    : return "cudaErrorInvalidPc";
    case cudaErrorIllegalAddress               : return "cudaErrorIllegalAddress";
    case cudaErrorInvalidPtx                   : return "cudaErrorInvalidPtx";
    case cudaErrorInvalidGraphicsContext       : return "cudaErrorInvalidGraphicsContext";
    case cudaErrorNvlinkUncorrectable          : return "cudaErrorNvlinkUncorrectable";
    case cudaErrorStartupFailure               : return "cudaErrorStartupFailure";
    case cudaErrorApiFailureBase               : return "cudaErrorApiFailureBase";
    default                                    : return "cudaInvalidError";
  }
}

cudaError_t CUDAWrapper::GetLastError(void)
{
  cudaError_t ret_error = (cudaError_t)cuda_last_error_;
  cuda_last_error_ = cudaSuccess;
  return ret_error;
}


//=============================================================================
// CUDA Runtime API
// 5.4. Stream Management
//=============================================================================
cudaError_t CUDAWrapper::StreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
  CUDAPlatform* platform = HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);

	HUMCommandQueue* queue = NULL;
	hum_int err;

  if (flags == cudaStreamDefault) {
    queue = HUMCommandQueue::CreateCommandQueue(platform->context(),
				device, (HUM_QUEUE_CUDA_STREAM|HUM_QUEUE_CUDA_BLOCKING), &err);

		if(err != HUM_SUCCESS) {
			HUM_ERROR("CreateCommandQueue Error: %d", err);
			SET_AND_RETURN_ERROR(cudaErrorInvalidValue);
		}
  }
  else if (flags == cudaStreamNonBlocking) {
    queue = HUMCommandQueue::CreateCommandQueue(platform->context(),
				device, (HUM_QUEUE_CUDA_STREAM), &err);
		if(err != HUM_SUCCESS) {
			HUM_ERROR("CreateCommandQueue Error: %d", err);
			SET_AND_RETURN_ERROR(cudaErrorInvalidValue);
		}
  }
  else {
		assert(0);
    SET_AND_RETURN_ERROR(cudaErrorInvalidValue);
  }

  *stream = (cudaStream_t)queue->get_handle();

  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::StreamDestroy(cudaStream_t stream) {
  CUDAPlatform* platform = HUMPlatform::GetCudaPlatform(0);

	hum_command_queue_handle command_queue = (hum_command_queue_handle)stream;
	if(stream == NULL) {
		HUM_ERROR("Cannot destory default stream %p", stream);
	}
	else {
		HUMCommandQueue* cmdq = (HUMCommandQueue*)command_queue->c_obj;
		cmdq->Release();
	}

  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::StreamSynchronize(cudaStream_t _stream)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);

	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
	
	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(device);
  }
  else {
    queue = stream->c_obj;
  }
  HUMCommand* command = HUMCommand::CreateMarker(NULL, NULL, queue);
  if (command == NULL) return cudaErrorMemoryAllocation;

  HUMEvent* blocking = command->ExportEvent();
  queue->Enqueue(command);
  blocking->Wait();
  blocking->Release();

	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::StreamWaitEvent(cudaStream_t stream, cudaEvent_t _event, unsigned int flags) 
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
 	hum_command_queue_handle command_queue = (hum_command_queue_handle)stream;
	hum_event_handle event = (hum_event_handle)_event;
	
	assert(stream);
	assert(event);

	HUMCommandQueue* queue = command_queue->c_obj;
	HUMEvent* ev = event->c_obj;

  HUMCommand* command = HUMCommand::CreateMarker(NULL, NULL, queue);
  if (command == NULL) return cudaErrorMemoryAllocation;

	command->AddWaitEvent(ev);
  queue->Enqueue(command);

  SET_AND_RETURN_ERROR(cudaSuccess);
}

//=============================================================================
// CUDA Runtime API
// 5.5. Event Management
//=============================================================================
cudaError_t CUDAWrapper::EventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	HUMContext* context = platform->context();

	bool profiling = !(flags & cudaEventDisableTiming);
		
	HUMEvent* E = new HUMEvent(context, profiling);
  *event = (cudaEvent_t)E->get_handle();
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::EventRecord(cudaEvent_t _event, cudaStream_t _stream) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
	
	bool use_default_queue = true;

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(device);
  }
  else {
    queue = stream->c_obj;
		use_default_queue = false;
  }

	hum_event_handle event = (hum_event_handle)_event;
  HUMEvent* E = event->c_obj;

  HUMCommand* command = HUMCommand::CreateEventRecord(platform->context(), 
			queue->device(), queue, E);

  queue->Enqueue(command);

  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::EventDestroy(cudaEvent_t _event) {
	hum_event_handle event = (hum_event_handle)_event;
  HUMEvent* E = event->c_obj;
  E->Release();
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::EventSynchronize(cudaEvent_t event) {
  HUMEvent* E = ((hum_event_handle)event)->c_obj;

	if(E->Wait() < 0) {
		SET_AND_RETURN_ERROR(cudaErrorInvalidValue);
	}
  
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::EventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop) {

	*ms = 0.0;

  HUMEvent* SE = ((hum_event_handle)start)->c_obj;
  HUMEvent* EE = ((hum_event_handle)stop)->c_obj;

  if (SE->GetStatus() != HUM_COMPLETE ||
      EE->GetStatus() != HUM_COMPLETE) {
    SET_AND_RETURN_ERROR(cudaErrorNotReady);
  }

	if(!SE->IsProfiling() || !EE->IsProfiling()) {
    SET_AND_RETURN_ERROR(cudaErrorNotReady);
	}

	hum_ulong start_ns, stop_ns;
	hum_int err;
	size_t ret_size;
	err = SE->GetEventProfilingInfo(HUM_PROFILING_COMMAND_END,
			sizeof(hum_ulong), &start_ns, &ret_size);
	HUM_ASSERT(sizeof(hum_ulong) == ret_size);
	HUM_ASSERT(err == 0);
	err = EE->GetEventProfilingInfo(HUM_PROFILING_COMMAND_END,
			sizeof(hum_ulong), &stop_ns, &ret_size);
	HUM_ASSERT(sizeof(hum_ulong) == ret_size);
	HUM_ASSERT(err == 0);
	
	*ms = (float)(stop_ns - start_ns) / 1000000.0f;

  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::EventQuery(cudaEvent_t event) {
  HUMEvent* E = ((hum_event_handle)event)->c_obj;

	if(E->GetStatus() == HUM_COMPLETE) {
		return cudaSuccess;
	}
	else
		return cudaErrorNotReady;
}



//=============================================================================
// CUDA Runtime API
// 5.10. Memory Management
//=============================================================================
cudaError_t CUDAWrapper::Malloc(void** devPtr, size_t size)
{
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	HUMContext* context = platform->context();
  HUMDevice* device = platform->GetDeviceById(current_device_id_);

	*devPtr = NULL;

  if (size == 0) {
    SET_AND_RETURN_ERROR(cudaErrorMemoryAllocation);
	}

	hum_mem_flags flags = HUM_MEM_READ_WRITE;

	hum_int err = HUM_SUCCESS;

  size_t aligned_size = size;
  size_t size_mod = aligned_size % (2 * 1024 * 1024);
  if (size_mod != 0)
    aligned_size += ((2 * 1024 * 1024) - size_mod);

	/*
	pthread_mutex_lock(&mutex_);
	HUMMem* mem = HUMMem::CreateBuffer(context, flags, aligned_size, NULL, &err);
	pthread_mutex_unlock(&mutex_);
	*/
	HUMMem* mem = NULL;
	*devPtr = g_MemRegion_->CreateHUMDevPtr(context, device, &mem, aligned_size);
	((HUMUnifiedMem*)mem)->dev_ptr_ = *devPtr;
	assert(((HUMUnifiedMem*)mem)->dev_ptr_);
	HUM_DEV("devPtr(%p) for mem=%p", *devPtr, mem);

	assert(mem->IsHostReadable());
	assert(mem->IsHostWritable());

	if(err != HUM_SUCCESS) {
		assert(0);
		SET_AND_RETURN_ERROR(cudaErrorMemoryAllocation);
	}
	
	HUM_DEV("Malloc vDev Memory %p", *devPtr);

	SET_AND_RETURN_ERROR(cudaSuccess);
}


cudaError_t CUDAWrapper::Free(void* devPtr)
{
	//Impicitly synchronize for Free
	DeviceSynchronize();

  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	HUMMem* mem = g_MemRegion_->GetMemObj(devPtr, NULL);
	HUM_DEV("cudaFree(%p) mem=%p begin", devPtr, mem);

  if(mem == NULL) {
		HUM_ERROR("mem = %p", mem);
		SET_AND_RETURN_ERROR(cudaErrorInvalidDevicePointer);
	}

	g_MemRegion_->FreeHUMDevPtr(devPtr);
	//mem->Release();

	HUM_DEV("cudaFree(%p) mem=%p end", devPtr, mem);
	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::MallocHost(void** ptr, size_t size) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	//*ptr = memalign(4096, size);
	nvlibs_->cudaMallocHost(ptr, size);
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::HostAlloc(void** ptr, size_t size, unsigned int flags)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	//*ptr = memalign(4096, size);
	nvlibs_->cudaHostAlloc(ptr, size, flags);
  SET_AND_RETURN_ERROR(cudaSuccess);
}


cudaError_t CUDAWrapper::FreeHost(void* ptr) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	//free(ptr);
	nvlibs_->cudaFreeHost(ptr);
  SET_AND_RETURN_ERROR(cudaSuccess);
}


cudaError_t CUDAWrapper::MallocManaged(void** devPtr, size_t size,
		unsigned int flags/* = cudaMemAttachGlobal*/)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  nvlibs_->cudaMallocManaged(devPtr, size, flags);
	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::MemPrefetchAsync(const void* devPtr, size_t count,
		int dstDevice, cudaStream_t stream/* = 0*/)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	TODO();
	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::Memcpy(void* dst, const void* src, size_t count,
		cudaMemcpyKind kind, cudaStream_t _stream, bool async)
{
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
  HUMDevice* cur_device = platform->GetDeviceById(current_device_id_);

	//HUM_PRINT("Memcpy begin, stream=%p", _stream);

	size_t dst_offset = 0;
	HUMMem* dst_mem = g_MemRegion_->GetMemObj(dst, &dst_offset);
	size_t src_offset = 0;
	HUMMem* src_mem = g_MemRegion_->GetMemObj(src, &src_offset);

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(cur_device);
  }
  else {
    queue = stream->c_obj;
	}

	assert(kind != cudaMemcpyDefault);

  if(kind ==  cudaMemcpyDefault) {
    if (dst_mem == NULL && src_mem == NULL) {
			kind = cudaMemcpyHostToHost;
		}
		else if(dst_mem == NULL && src_mem != NULL) {
			kind = cudaMemcpyDeviceToHost;
		}
		else if(dst_mem != NULL && src_mem == NULL) {
			kind = cudaMemcpyHostToDevice;
		}
		else if(dst_mem != NULL && src_mem != NULL) {
			kind = cudaMemcpyDeviceToDevice;
		}
		else {
			assert(0); //?
		}
	}

	HUMMemCommand* command = NULL;
	switch(kind) {
  case cudaMemcpyHostToHost:
		TODO();
    break;
  case cudaMemcpyHostToDevice: //WriteBuffer
		if(!dst_mem->IsHostWritable()) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorInvalidDevicePointer);
		}
		if(src_mem != NULL) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorInvalidHostPointer);
		}
		assert(dst_mem);

		//dst_mem->Retain();
		command = HUMCommand::CreateWriteBuffer(NULL, NULL,
				queue, dst_mem, dst_offset, count, (void*)src, !is_on_stack(src)); 
    break;
  case cudaMemcpyDeviceToHost:	//ReadBuffer
		if(!src_mem->IsHostReadable()) {
			HUM_ERROR("src_mem %p is not host readable", src_mem);
			SET_AND_RETURN_ERROR(cudaErrorInvalidDevicePointer);
		}
		if(dst_mem != NULL) {
			SET_AND_RETURN_ERROR(cudaErrorInvalidHostPointer);
		}
		assert(src_mem);

		//src_mem->Retain();
		command =  HUMCommand::CreateReadBuffer(NULL, NULL,
				queue, src_mem, src_offset, count, (void*)dst); 
    break;
  case cudaMemcpyDeviceToDevice: //CopyBuffer

		if(dst_mem == NULL || src_mem == NULL) {
			SET_AND_RETURN_ERROR(cudaErrorInvalidDevicePointer);
			assert(0);
		}
		assert(src_mem);
		assert(dst_mem);

		//src_mem->Retain();
		//dst_mem->Retain();

		command = HUMCommand::CreateCopyBuffer(
				NULL, NULL, queue, src_mem, dst_mem, src_offset, dst_offset, count);
   break;
	}
	if(command == NULL) {
		assert(0);
    SET_AND_RETURN_ERROR(cudaErrorUnknown);
	}

  HUMEvent* blocking;
  if (async == false) blocking = command->ExportEvent();
  queue->Enqueue(command);
  if (async == false) {
    hum_int ret = blocking->Wait();
    blocking->Release();
    if (ret < 0) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorUnknown);
		}
  }
	HUM_DEV("%s", "Memcpy end");

	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::MemcpyToArray(cudaArray_t dst, size_t wOffset,
    size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind,
    cudaStream_t _stream, bool async) {
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
  HUMDevice* cur_device = platform->GetDeviceById(current_device_id_);

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(cur_device);
  }
  else {
    queue = stream->c_obj;
	}

  cuda_func_memcpy_to_array_t params;
  params.dst = dst;
  params.wOffset = wOffset;
  params.hOffset = hOffset;
  params.src = src;
  params.count = count;
  params.kind = kind;

  HUMCommand* command = HUMCommand::CreateDriver(platform->context(),
      cur_device, queue, HUM_CUDA_API_FUNC_MEMCPY_TO_ARRAY,
      &params, sizeof(cuda_func_memcpy_to_array_t));

  HUMEvent* blocking;
  if (async == false) blocking = command->ExportEvent();
  queue->Enqueue(command);
  if (async == false) {
    hum_int ret = blocking->Wait();
    blocking->Release();
    if (ret < 0) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorUnknown);
		}
  }

	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::MemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset,
		cudaMemcpyKind kind, cudaStream_t _stream, bool async)
{
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
  HUMDevice* cur_device = platform->GetDeviceById(current_device_id_);

	HUM_DEV("MemcpyToSymbol symbol=%p begin", symbol);

	size_t src_offset = 0;
	HUMMem* src_mem = g_MemRegion_->GetMemObj(src, &src_offset);

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(cur_device);
  }
  else {
    queue = stream->c_obj;
	}

	HUMMemCommand* command = NULL;
	switch(kind) {
  case cudaMemcpyHostToHost:
		assert(0);
    break;
  case cudaMemcpyHostToDevice: //WriteBuffer
		if(src_mem != NULL) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorInvalidHostPointer);
		}

		//dst_mem->Retain();
		command = HUMCommand::CreateWriteBufferToSymbol(NULL, NULL,
				queue, symbol, offset, count, (void*)src); 
    break;
  case cudaMemcpyDeviceToHost:	//ReadBuffer
		assert(0);
		break;
  case cudaMemcpyDeviceToDevice: //CopyBuffer
		TODO();
		break;
	}
	if(command == NULL) {
		assert(0);
    SET_AND_RETURN_ERROR(cudaErrorUnknown);
	}

  HUMEvent* blocking;
  if (async == false) blocking = command->ExportEvent();
  queue->Enqueue(command);
  if (async == false) {
    hum_int ret = blocking->Wait();
    blocking->Release();
    if (ret < 0) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorUnknown);
		}
  }
	HUM_DEV("%s", "MemcpyToSymbol end");
	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::MemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset,
		cudaMemcpyKind kind, cudaStream_t _stream, bool async) {
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
  HUMDevice* cur_device = platform->GetDeviceById(current_device_id_);

	size_t dst_offset = 0;
	HUMMem* dst_mem = g_MemRegion_->GetMemObj(dst, &dst_offset);

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(cur_device);
  }
  else {
    queue = stream->c_obj;
	}

	HUMMemCommand* command = NULL;
	switch(kind) {
  case cudaMemcpyHostToHost:
		assert(0);
    break;
  case cudaMemcpyHostToDevice: //WriteBuffer
		assert(0);
    break;
  case cudaMemcpyDeviceToHost:	//ReadBuffer
		if(dst_mem != NULL) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorInvalidHostPointer);
		}

		command = HUMCommand::CreateReadBufferFromSymbol(NULL, NULL,
				queue, symbol, offset, count, dst); 
		break;
  case cudaMemcpyDeviceToDevice: //CopyBuffer
		TODO();
		break;
	}
	if(command == NULL) {
		assert(0);
    SET_AND_RETURN_ERROR(cudaErrorUnknown);
	}

  HUMEvent* blocking;
  if (async == false) blocking = command->ExportEvent();
  queue->Enqueue(command);
  if (async == false) {
    hum_int ret = blocking->Wait();
    blocking->Release();
    if (ret < 0) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorUnknown);
		}
  }
	SET_AND_RETURN_ERROR(cudaSuccess);
}



cudaError_t CUDAWrapper::Memset(void* dst, int value, size_t sizeBytes, 
		cudaStream_t _stream, bool async) 
{

 	//HUM_PRINT("Memset begin, stream=%p", _stream);

	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;

	size_t dst_offset = 0;
	HUMMem* dst_mem = g_MemRegion_->GetMemObj(dst, &dst_offset);

  HUMDevice* cur_device = platform->GetDeviceById(current_device_id_);

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(cur_device);
  }
  else {
    queue = stream->c_obj;
	}

	HUM_DEV("%s", "Memset begin");

	HUMMemCommand* command = NULL;
  command = HUMCommand::CreateFillBuffer(NULL, NULL, queue, dst_mem, &value, sizeof(value), dst_offset, sizeBytes);

  HUMEvent* blocking;
  if (async == false) blocking = command->ExportEvent();
  queue->Enqueue(command);
  if (async == false) {
    hum_int ret = blocking->Wait();
    blocking->Release();
    if (ret < 0) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorUnknown);
		}
  }
	HUM_DEV("%s", "Memset end");

  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::MemGetInfo(size_t* free_mem, size_t* total_mem)
{
	return nvlibs_->cudaMemGetInfo(free_mem, total_mem);
}

cudaError_t CUDAWrapper::HostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
  return nvlibs_->cudaHostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t CUDAWrapper::Malloc3DArray(cudaArray_t* array,
    const cudaChannelFormatDesc* desc, cudaExtent extent,
    unsigned int flags) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  cudaError_t err = nvlibs_->cudaMalloc3DArray(array, desc, extent, flags);
  return err;
}

cudaError_t CUDAWrapper::MallocArray(cudaArray_t* array,
    const cudaChannelFormatDesc* desc, size_t width, size_t height,
    unsigned int flags) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  cudaError_t err = nvlibs_->cudaMallocArray(array, desc, width, height, flags);
  return err;
}

cudaError_t CUDAWrapper::FreeArray(cudaArray_t array) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  cudaError_t err = nvlibs_->cudaFreeArray(array);
  return err;
}

cudaError_t CUDAWrapper::Memcpy2DToArray(cudaArray_t dst, size_t wOffset,
    size_t hOffset, const void* src, size_t spitch, size_t width,
    size_t height, cudaMemcpyKind kind, cudaStream_t _stream, bool async) {
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
  HUMDevice* cur_device = platform->GetDeviceById(current_device_id_);

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(cur_device);
  }
  else {
    queue = stream->c_obj;
	}

  cuda_func_memcpy2d_to_array_t params;
  params.dst = dst;
  params.wOffset = wOffset;
  params.hOffset = hOffset;
  params.src = src;
  params.spitch = spitch;
  params.width = width;
  params.height = height;
  params.kind = kind;

  HUMCommand* command = HUMCommand::CreateDriver(platform->context(),
      cur_device, queue, HUM_CUDA_API_FUNC_MEMCPY2D_TO_ARRAY,
      &params, sizeof(cuda_func_memcpy2d_to_array_t));

  HUMEvent* blocking;
  if (async == false) blocking = command->ExportEvent();
  queue->Enqueue(command);
  if (async == false) {
    hum_int ret = blocking->Wait();
    blocking->Release();
    if (ret < 0) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorUnknown);
		}
  }

	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::Memcpy3D(const cudaMemcpy3DParms* p,
    cudaStream_t _stream, bool async) {
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
  HUMDevice* cur_device = platform->GetDeviceById(current_device_id_);

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(cur_device);
  }
  else {
    queue = stream->c_obj;
	}

  HUMCommand* command = HUMCommand::CreateDriver(platform->context(),
      cur_device, queue, HUM_CUDA_API_FUNC_MEMCPY3D,
      (void*)p, sizeof(cudaMemcpy3DParms));

  HUMEvent* blocking;
  if (async == false) blocking = command->ExportEvent();
  queue->Enqueue(command);
  if (async == false) {
    hum_int ret = blocking->Wait();
    blocking->Release();
    if (ret < 0) {
			assert(0);
			SET_AND_RETURN_ERROR(cudaErrorUnknown);
		}
  }

	SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::BindTextureToArray(const struct textureReference *texref,
    cudaArray_const_t array, const struct cudaChannelFormatDesc *desc) {
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);

	HUMCommandQueue* queue = platform->GetDefaultQueue(device);

	cuda_func_bind_texture_to_array_t params;
	params.texref = texref;
  params.array = array;
	params.desc = desc;

	HUMCommand* command = HUMCommand::CreateDriver(platform->context(), device, queue,
			HUM_CUDA_API_FUNC_BIND_TEXTURE_TO_ARRAY,
			&params, sizeof(cuda_func_bind_texture_to_array_t));

	HUMEvent* blocking;
	blocking = command->ExportEvent();
	queue->Enqueue(command);
	hum_int ret = blocking->Wait();
	blocking->Release();
	if (ret < 0) {
		assert(0);
		SET_AND_RETURN_ERROR(cudaErrorUnknown);
	}
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::LaunchKernel ( const void* func, 
		dim3 gridDim, dim3 blockDim, 
		void** args, size_t sharedMem, 
		cudaStream_t _stream )
{
	const char* funcname = g_cuda_func_map_h2d[func].c_str();

	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;

	bool use_default_queue = true;

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(device);
  }
  else {
    queue = stream->c_obj;
		use_default_queue = false;
  }

	HUMKernelArg* kernel_args[64];

	size_t numBlocks3D[3];
	size_t blockDim3D[3];

	numBlocks3D[0] = gridDim.x;
	numBlocks3D[1] = gridDim.y;
	numBlocks3D[2] = gridDim.z;
	blockDim3D[0] = blockDim.x;
	blockDim3D[1] = blockDim.y;
	blockDim3D[2] = blockDim.z;
	HUM_DEV("nwg = (%d, %d, %d), lws = (%d, %d, %d)",
			numBlocks3D[0], numBlocks3D[1], numBlocks3D[2],
			blockDim3D[0], blockDim3D[1], blockDim3D[2]);

	size_t magic_number = *(size_t*)args[0];
	size_t num_args = *(size_t*)args[1];
	HUM_DEV("func=%p, funcname=%s, magic_number=%lx, num_args=%d, queue=%p", func, funcname, magic_number, num_args, queue);

	if(magic_number != 0xFFFFFFABCDEF) {
		HUM_DEV("%s launch in one GPU", funcname);

		HUMCommand* command = HUMCommand::CreateMarker(NULL, NULL, queue);
		if (command == NULL) return cudaErrorMemoryAllocation;

		HUMEvent* blocking = command->ExportEvent();
		queue->Enqueue(command);
		blocking->Wait();
		blocking->Release();

		cudaError_t err = nvlibs_->cudaLaunchKernel(func, 
				gridDim, blockDim, args, sharedMem, 0);
		nvlibs_->cudaDeviceSynchronize();
		pthread_mutex_unlock(&mutex_);

		return err;
	}

	uint64_t value[1024];

	int idx = 3; //arg_idx
	int arg_idx = 0; //arg_idx

	value[0] = 0;
	kernel_args[arg_idx++] = CreateKernelArg(platform->context(),
			8, value, 0, 0);	//magic_number
	kernel_args[arg_idx++] = CreateKernelArg(platform->context(),
			8, value, 0, 0);	//num args

	uint64_t arg_infos_addr = *(uint64_t*)args[arg_idx];
	kernel_args[arg_idx++] = CreateKernelArg(platform->context(),
			8, value, 0, 0);	//arg_info_addr

	int* arg_infos = (int*)arg_infos_addr;
	HUM_DEV("arg_infos = %p", (int*)arg_infos);

	int arg_infos_idx = 0;
	void* prev_arg = args[idx];
	for(int i=0;i<num_args;i++) {
		size_t offset = 0;

		void* arg_value_ptr = args[idx++];
		int arg_type = arg_infos[arg_infos_idx++];
		int arg_size = arg_infos[arg_infos_idx++];

		HUM_DEV("[%d] value_ptr=%p, type=%d, size=%d",
				i, arg_value_ptr, arg_type, arg_size);

		if(arg_type == 0 || arg_type == 1) 
		{
			
			memcpy(&value[0], arg_value_ptr, arg_size);
			kernel_args[arg_idx++] = CreateKernelArg(platform->context(),
					arg_size, value, offset, 0);
		}
		else { //structure
		
			memcpy(&value[0], arg_value_ptr, arg_size);
			HUMKernelArg* karg =	CreateKernelArg(platform->context(),
					arg_size, value, offset, 0);

			//if(arg_size > 1) 
			{
				int num_members =	arg_infos[arg_infos_idx++];
				HUM_DEV("num_members = %d", num_members);

				for(int m=0;m<num_members;m++) {
					int member_loc = arg_infos[arg_infos_idx++];
					int member_type = arg_infos[arg_infos_idx++];
					int member_size = arg_infos[arg_infos_idx++];
				}
			}
			kernel_args[arg_idx++] = karg;
		}
	}

	HUMCommand* command = HUMCommand::CreateLaunchCudaKernel(platform->context(), device, queue,
      "", func, numBlocks3D, blockDim3D, 
			sharedMem, arg_idx, kernel_args);

#ifdef LOCK_STEP
	HUMEvent* blocking;
	blocking = command->ExportEvent();
#endif

  queue->Enqueue(command);

#ifdef LOCK_STEP
	hum_int ret = blocking->Wait();
	blocking->Release();
	if (ret < 0) {
		assert(0);
		SET_AND_RETURN_ERROR(cudaErrorUnknown);
	}
#endif
//	pthread_mutex_unlock(&mutex_);

  SET_AND_RETURN_ERROR(cudaSuccess);
}

void CUDAWrapper::LaunchKernel(const char* kernelName, dim3 _numBlocks3D,
    dim3 _blockDim3D, size_t sharedMem, cudaStream_t _stream, int numArgs, va_list args)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);
	hum_command_queue_handle stream = (hum_command_queue_handle)_stream;
	
	bool use_default_queue = true;

	HUMCommandQueue* queue = NULL;
  if (stream == NULL) {
    queue = platform->GetDefaultQueue(device);
  }
  else {
    queue = stream->c_obj;
		use_default_queue = false;
  }

	HUMKernelArg* kernel_args[numArgs];
	for (unsigned int i = 0; i < numArgs; ++i) {
		size_t offset = 0;
    uint64_t arg_size = va_arg(args, uint64_t);
		HUM_DEV("[%d]: arg_size=%ld", i, arg_size);
		uint64_t value[(arg_size+7)/8];
	
    bool is_pointer = false;
    bool is_floating = false;
    char* arg_typename = va_arg(args, char*);
		HUM_DEV("[%d] arg_typename = %s", i, arg_typename);
    if (!strcmp(arg_typename, "f") || !strcmp(arg_typename, "d")) {
      is_floating = true;
    }
    else if (!strcmp(arg_typename, "p")) {
      is_pointer = true;
    }

		if(is_pointer) {
			assert(arg_size == 8);
			uint64_t arg_p = va_arg(args, uint64_t);
			if(g_MemRegion_->IsWithin((void*)arg_p) > -1) {
				HUMMem* memobj = g_MemRegion_->GetMemObj((void*)arg_p, &offset);
				value[0] = (uint64_t)memobj->get_handle();
				//memobj->Retain();
			}
			else {
				value[0] = arg_p;
			}
		}
		else if(is_floating) {
			value[0] = 0;
			if(arg_size == 4) {
        float arg_f = va_arg(args, double);
        memcpy(&value[0], &arg_f, 4);
				/*
				printf("arg_f = %f vs %f\n", arg_f, *(float*)&value[0]);

				{
					unsigned char* test = (unsigned char*)&arg_f;
					
					printf("arg_f printf\n");
					for(int j=0;j<arg_size;j++) {
						if(j > 0 && j % 16 == 0) 
							printf("\n");
						printf("%02x ", test[j]);
					}
					printf("\n");
				}
				*/

			}
			else if(arg_size == 8) {
        double arg_d = va_arg(args, double);
        memcpy(&value[0], &arg_d, 8);
				//printf("arg_d = %lf vs %lf\n", arg_d, *(double*)&value[0]);
				/*
				{
					unsigned char* test = (unsigned char*)&arg_d;
					printf("arg_d printf\n");
					for(int j=0;j<arg_size;j++) {
						if(j > 0 && j % 16 == 0) 
							printf("\n");
						printf("%02x ", test[j]);
					}
					printf("\n");
				}
				*/

			}
			else {
				assert(0);
			}
		}
		else {
			for (int k = 0; k < ((arg_size+7)/8); k++) {
				value[k] = 0;
			}
			for (int k = 0; k < ((arg_size+7)/8); k++) {
				value[k] = va_arg(args, uint64_t);
			}
		}
/*
		{
			unsigned char* test = (unsigned char*)value;
			printf("arg test printf\n");
			for(int j=0;j<arg_size;j++) {
				if(j > 0 && j % 16 == 0) 
					printf("\n");
				printf("%02x ", test[j]);
			}
			printf("\n");
		}
		*/

		kernel_args[i] = CreateKernelArg(platform->context(),
				arg_size, value, offset, 0);
	}

	size_t numBlocks3D[3];
	size_t blockDim3D[3];

	numBlocks3D[0] = _numBlocks3D.x;
	numBlocks3D[1] = _numBlocks3D.y;
	numBlocks3D[2] = _numBlocks3D.z;
	blockDim3D[0] = _blockDim3D.x;
	blockDim3D[1] = _blockDim3D.y;
	blockDim3D[2] = _blockDim3D.z;

	HUM_DEV("nwg = (%d, %d, %d), lws = (%d, %d, %d)",
			numBlocks3D[0], numBlocks3D[1], numBlocks3D[2],
			blockDim3D[0], blockDim3D[1], blockDim3D[2]);


	HUMCommand* command = HUMCommand::CreateLaunchCudaKernel(platform->context(), device, queue,
      kernelName, NULL, numBlocks3D, blockDim3D, 
			sharedMem, numArgs, kernel_args);

	/*
	HUMEvent* blocking;
  if (use_default_queue == HUM_TRUE) 
		blocking = command->ExportEvent();
	*/
  queue->Enqueue(command);
	
	/*
  if (use_default_queue == HUM_TRUE) {
    hum_int ret = blocking->Wait();
    blocking->Release();
		if(ret < 0) {
			HUM_ERROR("command waiting error %d", ret);
		}
	}
	*/
}

cudaChannelFormatDesc CUDAWrapper::CreateChannelDesc ( int  x, int  y, 
		int  z, int  w, cudaChannelFormatKind f )
{
	cudaChannelFormatDesc desc = nvlibs_->cudaCreateChannelDesc(x,y,z,w,f);
	
	HUM_DEV("desc = %p", desc);

	return desc;
}

cudaError_t CUDAWrapper::GetChannelDesc(cudaChannelFormatDesc* desc,
    cudaArray_const_t array) {
  return nvlibs_->cudaGetChannelDesc(desc, array);
}

cudaError_t CUDAWrapper::BindTexture(size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);

	HUMCommandQueue* queue =platform->GetDefaultQueue(device);

	cuda_func_bind_texture_t params;
	params.offset = offset;
	params.texref = texref;
	params.devPtr = devPtr;
	params.desc = desc;
	params.size = size;

	HUM_DEV("cudaBindTexture offset=%p, texref=%p, devPtr=%p, desc=%p, size=%ld",
			params.offset, params.texref, params.devPtr, params.desc, params.size);
	if(g_MemRegion_->IsWithin(devPtr) > -1) {
		size_t mem_offset = 0;
		HUMMem* memobj = g_MemRegion_->GetMemObj((void*)devPtr, &mem_offset);
		assert(memobj);

		params.mem = memobj;
		params.mem_offset = mem_offset;
	}
	else {
		assert(0);
	}

	HUMCommand* command = HUMCommand::CreateDriver(platform->context(), device, queue,
			HUM_CUDA_API_FUNC_BIND_TEXTURE,
			&params, sizeof(cuda_func_bind_texture_t));

	HUMEvent* blocking;
	blocking = command->ExportEvent();
	queue->Enqueue(command);
	hum_int ret = blocking->Wait();
	blocking->Release();
	if (ret < 0) {
		assert(0);
		SET_AND_RETURN_ERROR(cudaErrorUnknown);
	}
/*
	device->ExecuteFunc(
			HUM_CUDA_API_FUNC_BIND_TEXTURE,
			&params);
			*/
  SET_AND_RETURN_ERROR(cudaSuccess);
}

cudaError_t CUDAWrapper::UnbindTexture( const textureReference* texref )
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(current_device_id_);
	HUMCommandQueue* queue =platform->GetDefaultQueue(device);

	cuda_func_unbind_texture_t params;
	params.texref = texref;

	HUMCommand* command = HUMCommand::CreateDriver(platform->context(), device, queue,
			HUM_CUDA_API_FUNC_UNBIND_TEXTURE,
			&params, sizeof(cuda_func_unbind_texture_t));

	HUMEvent* blocking;
	blocking = command->ExportEvent();
	queue->Enqueue(command);
	hum_int ret = blocking->Wait();
	blocking->Release();
	if (ret < 0) {
		assert(0);
		SET_AND_RETURN_ERROR(cudaErrorUnknown);
	}

/*
	device->ExecuteCUDAFunc(
			HUM_CUDA_API_FUNC_UNBIND_TEXTURE,
			&params);
*/
  SET_AND_RETURN_ERROR(cudaSuccess);
}



CUresult CUDAWrapper::DeviceGet(CUdevice* _device, int ordinal)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(ordinal);

	*_device = (CUdevice)ordinal;

	return CUDA_SUCCESS;
}


CUresult CUDAWrapper::DeviceGetAttribute_drv(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
  HUMDevice* device = platform->GetDeviceById(dev);
	const cuda_dev_info_t* dev_info = device->cuda_dev_info();

	switch(attrib) {
	case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
		*pi = dev_info->multiProcessorCount;
		printf("CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR:
		*pi = dev_info->sharedMemPerMultiprocessor;
		printf("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
		*pi = dev_info->maxThreadsPerMultiProcessor;
		printf("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
		*pi = dev_info->maxThreadsPerBlock;
		printf("CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
		*pi = dev_info->regsPerBlock;
		printf("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
		*pi = dev_info->warpSize;
		printf("CU_DEVICE_ATTRIBUTE_WARP_SIZE: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
		*pi = dev_info->maxGridSize[0];
		printf("CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
		*pi = dev_info->maxGridSize[1];
		printf("CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
		*pi = dev_info->maxGridSize[2];
		printf("CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
		*pi = dev_info->sharedMemPerBlock;
		printf("CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
		*pi = dev_info->ECCEnabled;
		printf("CU_DEVICE_ATTRIBUTE_ECC_ENABLED: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
		*pi = dev_info->memoryClockRate;
		printf("CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
		*pi = dev_info->memoryBusWidth;
		printf("CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: %d\n", *pi);
		break;
	case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR:
		*pi = dev_info->regsPerMultiprocessor;
		printf("CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR: %d\n", *pi);
		break;
	default:
		HUM_ERROR("Unsupported CUdevice_attribute: %d", (int)attrib);
		assert(0);
		break;
	}


	return CUDA_SUCCESS;
}

CUresult CUDAWrapper::DeviceGetCount(int* count)
{
  CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	*count = platform->GetNumCudaDevices();
	printf("cuDeviceGetCount = %d\n", *count);

	return CUDA_SUCCESS;
}

CUresult CUDAWrapper::DeviceGetName(char* name, int len, CUdevice dev)
{
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	HUMDevice* device = platform->GetDeviceById(dev);
	const cuda_dev_info_t* dev_info = device->cuda_dev_info();

	strcpy(name, dev_info->name); 
	printf("cuDeviceGetName = %s\n", name);

	return CUDA_SUCCESS;
}

CUresult CUDAWrapper::DeviceTotalMem(size_t* bytes, CUdevice dev)
{
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	HUMDevice* device = platform->GetDeviceById(dev);
	const cuda_dev_info_t* dev_info = device->cuda_dev_info();

	*bytes = dev_info->totalGlobalMem;
	printf("cuDeviceTotalMem = %ld\n", *bytes);

	return CUDA_SUCCESS;
}


CUresult CUDAWrapper::DeviceComputeCapability(int* major, int* minor, CUdevice dev)
{
	CUDAPlatform* platform = (CUDAPlatform*)HUMPlatform::GetCudaPlatform(0);
	HUMDevice* device = platform->GetDeviceById(dev);
	const cuda_dev_info_t* dev_info = device->cuda_dev_info();
	*major = dev_info->major;
	*minor = dev_info->minor;

	printf("cuDeviceComputeCapability: major=%d, minor=%d\n", 
			*major, *minor);

	return CUDA_SUCCESS;
}

