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

#include "CUDAAPI.h"
#include "CUDAWrapper.h"

#include "Utils.h"

#include <stdint.h>
#include <stdarg.h>
#include <malloc.h>
#include <map>

#include <cuda_runtime.h>
#include <cuda.h>

#define TODO() \
	HUM_ERROR("*** %s function is not implemented yet ***", __FUNCTION__); assert(0);\
	return cudaSuccess;

CUDAWrapper cuda_wrapper;

extern "C"
void __cudaInitModule(void **fatCubinHandle) {
	return cuda_wrapper.InitModule(fatCubinHandle);
}

extern "C"
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
		const char *deviceName, int thread_limit, uint3 *tid,
		uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
	return cuda_wrapper.RegisterFunction(fatCubinHandle, hostFun, deviceFun,
			deviceName, thread_limit, tid, bid, bDim, gDim, wSize);
}

extern "C"
void  __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
				const char *deviceName, int ext, size_t size,
        int constant, int global) {
	return cuda_wrapper.RegisterVar(fatCubinHandle, hostVar, deviceAddress,
				deviceName, ext, size, constant, global);
}

extern "C"
void __cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
		const void **deviceAddress, const char *deviceName,
		int dim, int norm, int ext) {
	return cuda_wrapper.RegisterTexture(fatCubinHandle, hostVar,
		deviceAddress, deviceName, dim, norm, ext);
}

extern "C"
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
		dim3 *blockDim,  size_t *sharedMem, void *stream) {

	return cuda_wrapper.PopCallConfiguration(gridDim,
			blockDim, sharedMem, stream);
}

extern "C"
unsigned __cudaPushCallConfiguration(dim3 gridDim,
		dim3 blockDim, size_t sharedMem, void *stream) {
	return cuda_wrapper.PushCallConfiguration(gridDim,
			blockDim, sharedMem, stream);
}

extern "C"
void**  __cudaRegisterFatBinary(void *fatCubin) {
	return cuda_wrapper.RegisterFatBinary(fatCubin);
}

extern "C"
void  __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
	return cuda_wrapper.RegisterFatBinaryEnd(fatCubinHandle);
}

extern "C"
void __cudaUnregisterFatBinary(void **fatCubinHandle) {
	return cuda_wrapper.UnregisterFatBinary(fatCubinHandle);
}


//=============================================================================
// CUDA Runtime API
// 5.1. Device Management
//=============================================================================
cudaError_t cudaDeviceSynchronize(void) {
	return cuda_wrapper.DeviceSynchronize();
}

cudaError_t cudaThreadSynchronize(void) {
	return cuda_wrapper.DeviceSynchronize();
}

cudaError_t cudaDeviceReset(void) {
	return cuda_wrapper.DeviceReset();
}

cudaError_t cudaSetDevice(int deviceId)
{
	return cuda_wrapper.SetDevice(deviceId);
}

cudaError_t cudaGetDevice(int* device)
{
	return cuda_wrapper.GetDevice(device);
}

cudaError_t cudaGetDeviceCount(int* count)
{
	return cuda_wrapper.GetDeviceCount(count);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device)
{
	return cuda_wrapper.GetDeviceProperties(prop, device);
}

cudaError_t cudaDeviceGetAttribute(int* pi, cudaDeviceAttr attr, int deviceId) {
	//TODO();
	return cuda_wrapper.DeviceGetAttribute(pi, attr, deviceId);
}

cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig) {
	TODO();
}

cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache *cacheConfig) {
	TODO();
}

cudaError_t cudaDeviceGetLimit(size_t *pValue, cudaLimit limit) {
	TODO();
}

cudaError_t cudaFuncSetCacheConfig(const void* func, cudaFuncCache config) {
	return cudaSuccess;
}

cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig * pConfig) {
	TODO();
}

cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig config) {
	TODO();
}

cudaError_t cudaSetDeviceFlags(unsigned flags) {
  return cuda_wrapper.SetDeviceFlags(flags);
}

cudaError_t cudaChooseDevice(int *device, const cudaDeviceProp* prop) {
	TODO();
}

//=============================================================================
// CUDA Runtime API
// 5.3. Error Handling
//=============================================================================
cudaError_t cudaPeekAtLastError(void) {
  return (cudaError_t)cuda_last_error_;
}

const char* cudaGetErrorName(cudaError_t cudaError)
{
	return cuda_wrapper.GetErrorName(cudaError);
}

const char* cudaGetErrorString(cudaError_t cudaError)
{
	return cuda_wrapper.GetErrorString(cudaError);
}

cudaError_t cudaGetLastError(void)
{
	return cuda_wrapper.GetLastError();
}


//=============================================================================
// CUDA Runtime API
// 5.4. Stream Management
//=============================================================================
cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void *userData,
                                  unsigned int flags) {
  TODO();
}

cudaError_t cudaStreamCreate(cudaStream_t *stream) {
  return cuda_wrapper.StreamCreateWithFlags(stream, cudaStreamDefault);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags) {
	return cuda_wrapper.StreamCreateWithFlags(stream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
	return cuda_wrapper.StreamDestroy(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
	TODO();
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
	return cuda_wrapper.StreamSynchronize(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
	return cuda_wrapper.StreamWaitEvent(stream, event, flags);
}

cudaError_t cudaStreamAttachMemAsync ( cudaStream_t stream, void* devPtr, size_t length /*= 0*/, unsigned int  flags /*= cudaMemAttachSingle*/ ) {
	TODO();
}
#if 0
cudaError_t cudaStreamBeginCapture ( cudaStream_t stream, cudaStreamCaptureMode mode ) {
	TODO();
}
#endif

cudaError_t cudaStreamCreateWithPriority ( cudaStream_t* pStream, unsigned int  flags, int  priority ) {
	TODO();
}

#if 0
cudaError_t cudaStreamEndCapture ( cudaStream_t stream, cudaGraph_t* pGraph ) {
	TODO();
}

cudaError_t cudaStreamGetCaptureInfo ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus, unsigned long long* pId ) {
	TODO();
}
#endif

cudaError_t cudaStreamGetFlags ( cudaStream_t hStream, unsigned int* flags ) {
	TODO();
}

cudaError_t cudaStreamGetPriority ( cudaStream_t hStream, int* priority ) {
	TODO();
}
#if 0
cudaError_t cudaStreamIsCapturing ( cudaStream_t stream, cudaStreamCaptureStatus ** pCaptureStatus ) {
	TODO();
}

cudaError_t cudaThreadExchangeStreamCaptureMode ( cudaStreamCaptureMode ** mode ) {
	TODO();
}
#endif 
//=============================================================================
// CUDA Runtime API
// 5.5. Event Management
//=============================================================================
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
	return cuda_wrapper.EventCreateWithFlags(event, flags);
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
  return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
	return cuda_wrapper.EventRecord(event, stream);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
	return cuda_wrapper.EventDestroy(event);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
	return cuda_wrapper.EventSynchronize(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t stop) {
	return cuda_wrapper.EventElapsedTime(ms, start, stop);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
	return cuda_wrapper.EventQuery(event);
}

//=============================================================================
// CUDA Runtime API
// 5.7. Execution Control
//=============================================================================
cudaError_t cudaFuncGetAttributes( cudaFuncAttributes* attr, const void* func )
{
	return cuda_wrapper.FuncGetAttributes( attr, func);
}

cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value) {
  return cuda_wrapper.FuncSetAttribute(func, attr, value);
}

cudaError_t cudaLaunchKernel( const void* func,
		dim3 gridDim, dim3 blockDim,
		void** args, size_t sharedMem,
		cudaStream_t stream )
{
	return cuda_wrapper.LaunchKernel(func,
			gridDim, blockDim, args, sharedMem,
			stream);
}

//=============================================================================
// CUDA Runtime API
// 5.9. Memory Management
//=============================================================================
cudaError_t cudaFree(void* devPtr)
{
	return cuda_wrapper.Free(devPtr);
}

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
	return cuda_wrapper.Malloc(devPtr, size);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
	return cuda_wrapper.MallocHost(ptr, size);
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
	return cuda_wrapper.HostAlloc(ptr, size, flags);
}


cudaError_t cudaFreeHost(void* ptr) {
	return cuda_wrapper.FreeHost(ptr);
}

cudaError_t cudaMallocManaged(void** devPtr, size_t size,
		unsigned int flags/* = cudaMemAttachGlobal*/)
{
	return cuda_wrapper.MallocManaged(devPtr, size, flags);
}

cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count,
		int dstDevice, cudaStream_t stream/* = 0*/)
{
	return cuda_wrapper.MemPrefetchAsync(devPtr, count, dstDevice, stream);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count,
		cudaMemcpyKind kind)
{
	return cuda_wrapper.Memcpy(dst, src, count, kind, 0, false);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,
		cudaMemcpyKind kind, cudaStream_t stream/* = 0*/)
{
	return cuda_wrapper.Memcpy(dst, src, count, kind, stream, true);
}

cudaError_t cudaMemset(void* dst, int value, size_t sizeBytes) {
	return cuda_wrapper.Memset(dst, value, sizeBytes, NULL, false);
}

cudaError_t cudaMemsetAsync(void* dst, int  value, size_t sizeBytes, cudaStream_t stream/* = 0*/)
{
	return cuda_wrapper.Memset(dst, value, sizeBytes, stream, true);
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
    const void *src, size_t count, enum cudaMemcpyKind kind) {
  return cuda_wrapper.MemcpyToArray(dst, wOffset, hOffset, src, count, kind, 0, false);
}

cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset/*=0*/,
		cudaMemcpyKind kind/*=cudaMemcpyHostToDevice*/) {
	return cuda_wrapper.MemcpyToSymbol(symbol, src, count, offset, kind, 0, false);
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset/*=0*/,
		cudaMemcpyKind kind/*= cudaMemcpyHostToDevice*/, cudaStream_t stream /*= 0*/) {
	return cuda_wrapper.MemcpyToSymbol(symbol, src, count, offset, kind, stream, false);
}

cudaError_t cudaArrayGetInfo ( cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array ) {
	TODO();
}

cudaError_t cudaFreeArray ( cudaArray_t array ) {
	return cuda_wrapper.FreeArray(array);
}
cudaError_t cudaFreeMipmappedArray ( cudaMipmappedArray_t mipmappedArray ) {
	TODO();
}

cudaError_t cudaGetMipmappedArrayLevel ( cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level ) {
	TODO();
}

cudaError_t cudaGetSymbolAddress ( void** devPtr, const void* symbol ) {
	TODO();
}

cudaError_t cudaGetSymbolSize ( size_t* size, const void* symbol ) {
	TODO();
}

cudaError_t cudaHostGetDevicePointer ( void** pDevice, void* pHost, unsigned int  flags ) {
  return cuda_wrapper.HostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t cudaHostGetFlags ( unsigned int* pFlags, void* pHost ) {
	TODO();
}

cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags ) {
	TODO();
}

cudaError_t cudaHostUnregister ( void* ptr ) {
	TODO();
}

cudaError_t cudaMalloc3D ( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent ) {
	TODO();
}

cudaError_t cudaMalloc3DArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  flags /*= 0*/ ) {
  return cuda_wrapper.Malloc3DArray(array, desc, extent, flags);
}

cudaError_t cudaMallocArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height /*= 0*/, unsigned int  flags /*= 0*/ ) {
	return cuda_wrapper.MallocArray(array, desc, width, height, flags);
}

cudaError_t cudaMallocMipmappedArray ( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height ) {
	TODO();
}

cudaError_t cudaMemAdvise ( const void* devPtr, size_t count, cudaMemoryAdvise advice, int  device ) {
	TODO();
}

cudaError_t cudaMemGetInfo ( size_t* free, size_t* total ) {
	return cuda_wrapper.MemGetInfo(free, total);
}

cudaError_t cudaMemRangeGetAttribute ( void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count ) {
	TODO();
}

cudaError_t cudaMemRangeGetAttributes ( void** data, size_t* dataSizes, cudaMemRangeAttribute ** attributes, size_t numAttributes, const void* devPtr, size_t count ) {
	TODO();
}

cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) {
	TODO();
}

cudaError_t cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind /*= cudaMemcpyDeviceToDevice*/ ) {
	TODO();
}

cudaError_t	cudaMemcpy2DAsync ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMemcpy2DFromArray ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind ) {
	TODO();
}

cudaError_t cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMemcpy2DToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind ) {
  return cuda_wrapper.Memcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind, 0, false);
}

cudaError_t cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMemcpy3D ( const cudaMemcpy3DParms* p ) {
  return cuda_wrapper.Memcpy3D(p, 0, false);
}

cudaError_t	cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p ) {
	TODO();
}

cudaError_t cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMemcpyFromSymbol ( void* dst, const void* symbol, size_t count, size_t offset /*= 0*/, cudaMemcpyKind kind /*= cudaMemcpyDeviceToHost*/ ) {
	return cuda_wrapper.MemcpyFromSymbol(dst, symbol, count, offset, kind, 0, false);
}

cudaError_t cudaMemcpyFromSymbolAsync ( void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream /*= 0*/ ) {
	return cuda_wrapper.MemcpyFromSymbol(dst, symbol, count, offset, kind, stream, false);
}

cudaError_t cudaMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count ) {
	TODO();
}

cudaError_t cudaMemcpyPeerAsync ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMemset2D ( void* devPtr, size_t pitch, int  value, size_t width, size_t height ) {
	TODO();
}

cudaError_t	cudaMemset2DAsync ( void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

cudaError_t cudaMemset3D ( cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent ) {
	TODO();
}

cudaError_t	cudaMemset3DAsync ( cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent, cudaStream_t stream /*= 0*/ ) {
	TODO();
}

#if 0
cudaExtent make_cudaExtent ( size_t w, size_t h, size_t d ) {
	TODO();
}

cudaPitchedPtr make_cudaPitchedPtr ( void* d, size_t p, size_t xsz, size_t ysz ) {
	TODO();
}

cudaPos make_cudaPos ( size_t x, size_t y, size_t z ) {
	TODO();
}
#endif

//=============================================================================
// CUDA Runtime API
// 5.24. Texture Reference Management
//=============================================================================
cudaChannelFormatDesc cudaCreateChannelDesc ( int  x, int  y,
		int  z, int  w, cudaChannelFormatKind f ) {
	return cuda_wrapper.CreateChannelDesc(x, y, z, w, f);
}

cudaError_t cudaGetChannelDesc(cudaChannelFormatDesc* desc, cudaArray_const_t array) {
  return cuda_wrapper.GetChannelDesc(desc, array);
}

cudaError_t cudaBindTexture(size_t* offset, const textureReference* texref, const void* devPtr, const cudaChannelFormatDesc* desc, size_t size)
{
	return cuda_wrapper.BindTexture(offset, texref, devPtr, desc, size);
}
cudaError_t cudaUnbindTexture( const textureReference* texref )
{
	return cuda_wrapper.UnbindTexture(texref);
}

cudaError_t cudaBindTextureToArray(const struct textureReference *texref, cudaArray_const_t array,
    const struct cudaChannelFormatDesc *desc) {
  return cuda_wrapper.BindTextureToArray(texref, array, desc);
}

//=============================================================================
// CUDA Runtime API
// 5.30. C++ API Routines (Note: should be compiled with `nvcc`
//=============================================================================
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize,
		size_t dynamicSMemSize, unsigned int flags) {
	TODO();
}


//=============================================================================
// CUDA Driver API
// 5.5. Device Management
//=============================================================================
CUresult cuDeviceGet(CUdevice* device, int ordinal)
{
	printf("cuDeviceGet\n"); 
	return cuda_wrapper.DeviceGet(device, ordinal);	
}

CUresult cuDeviceGetCount(int* count)
{
	printf("cuDeviceGetCount\n"); 
	return cuda_wrapper.DeviceGetCount(count);
}

CUresult cuDeviceGetAttribute(int* pi, CUdevice_attribute attrib, CUdevice dev)
{
	printf("cuDeviceGetAttribute\n"); 
	return cuda_wrapper.DeviceGetAttribute_drv(pi, attrib, dev);
}

CUresult cuDeviceGetName(char* name, int len, CUdevice dev)
{
	printf("cuDeviceGetName\n"); 
	return cuda_wrapper.DeviceGetName(name, len, dev);	
}

CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev)
{
	printf("cuDeviceTotalMem\n");
	return cuda_wrapper.DeviceTotalMem(bytes, dev);
}

//=============================================================================
// CUDA Driver API
// 5.6. Device Management [DEPRECATED]
//=============================================================================
CUresult cuDeviceComputeCapability(int* major, int* minor, CUdevice dev)
{
	printf("cuDeviceComputeCapability\n"); 
	return cuda_wrapper.DeviceComputeCapability(major, minor, dev);
}

/*
//=============================================================================
// CUDA Driver API
// 5.7. Primary Context Management
//=============================================================================
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active)
{
  printf("cuDevicePrimaryCtxGetState\n"); 
	
		
	return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
{
  printf("cuDevicePrimaryCtxRelease\n"); return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev)
{
  printf("cuDevicePrimaryCtxRetain\n"); return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags)
{
  printf("cuDevicePrimaryCtxSetFlags\n"); return CUDA_SUCCESS;
}
*/
/*
//=============================================================================
// CUDA Driver API
// 5.8. Context Management
//=============================================================================
CUresult cuCtxGetCurrent(CUcontext* pctx)
{
  printf("cuCtxGetCurrent\n"); return CUDA_SUCCESS;
}

CUresult cuCtxGetDevice(CUdevice* device)
{
  printf("cuCtxGetDevice\n"); return CUDA_SUCCESS;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig* pConfig)
{
  printf("cuCtxGetSharedMemConfig\n"); return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx)
{
  printf("cuCtxSetCurrent\n"); return CUDA_SUCCESS;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config)
{
  printf("cuCtxSetSharedMemConfig\n"); return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void)
{
  printf("cuCtxSynchronize\n"); return CUDA_SUCCESS;
}

//=============================================================================
// CUDA Driver API
// 5.11. Memory Management
//=============================================================================
CUresult cuDeviceGetPCIBusId( char* pciBusId, int len, CUdevice dev)
{
  printf("cuDeviceGetPCIBusId\n"); return CUDA_SUCCESS;
}

//=============================================================================
// CUDA Driver API
// 5.25. Peer Context Memory Access
//=============================================================================
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags)
{
  printf("cuCtxEnablePeerAccess\n"); return CUDA_SUCCESS;
}

CUresult cuDeviceCanAccessPeer(int* canAccessPeer, CUdevice dev, CUdevice peerDev)
{
  printf("cuDeviceCanAccessPeer\n"); return CUDA_SUCCESS;
}
*/

////////////////////////////////////////////////////////////////////////////////
// Additional
void HUM_CUDALaunchKernel(const char* kernelName, dim3 numBlocks3D,
    dim3 blockDim3D, size_t sharedMem, cudaStream_t stream, int numArgs, ...)
{
  va_list args;
  va_start(args, numArgs);

	cuda_wrapper.LaunchKernel(kernelName, numBlocks3D,
    blockDim3D, sharedMem, stream, numArgs, args);

  va_end(args);
}
