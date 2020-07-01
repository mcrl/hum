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

#ifndef __CUDA_FUNC_H__
#define __CUDA_FUNC_H__

//CUDA API FUNCTION TYPES
#define HUM_CUDA_API_FUNC														0x5000
#define HUM_CUDA_API_FUNC_BIND_TEXTURE								0x5001
#define HUM_CUDA_API_FUNC_UNBIND_TEXTURE							0x5002
#define HUM_CUDA_API_FUNC_BIND_TEXTURE_TO_ARRAY      0x5003
#define HUM_CUDA_API_FUNC_MEMCPY_TO_ARRAY            0x5004
#define HUM_CUDA_API_FUNC_MEMCPY2D_TO_ARRAY          0x5005
#define HUM_CUDA_API_FUNC_MEMCPY3D                   0x5006

#define HUM_DRIVER_FUNC_INVALIDATE										0x6000
#define HUM_DRIVER_FUNC_PINFO_UPDATE									0x6001
#define HUM_DRIVER_FUNC_TOUCH												0x6002
#define HUM_DRIVER_FUNC_UPLOAD_BUFFER								0x6003

#include <cuda_runtime.h>

struct textureReference;
struct cudaChannelFormatDesc;
class HUMMem;

typedef struct _cuda_func_bind_texture_t {
	size_t* offset;
	const textureReference* texref;
	const void* devPtr;
	const cudaChannelFormatDesc* desc;
	size_t size;
	HUMMem* mem;
	size_t mem_offset;
} cuda_func_bind_texture_t;

typedef struct _cuda_func_unbind_texture_t {
	const textureReference* texref;
} cuda_func_unbind_texture_t;

typedef struct _cuda_func_bind_texture_to_array_t {
	const textureReference* texref;
  cudaArray_const_t array;
	const cudaChannelFormatDesc* desc;
} cuda_func_bind_texture_to_array_t;

typedef struct _cuda_func_memcpy_to_array_t {
  cudaArray_t dst;
  size_t wOffset;
  size_t hOffset;
  const void* src;
  size_t count;
  cudaMemcpyKind kind;
} cuda_func_memcpy_to_array_t;

typedef struct _cuda_func_memcpy2d_to_array_t {
  cudaArray_t dst;
  size_t wOffset;
  size_t hOffset;
  const void* src;
  size_t spitch;
  size_t width;
  size_t height;
  cudaMemcpyKind kind;
} cuda_func_memcpy2d_to_array_t;

#endif //__CUDA_FUNC_H__
