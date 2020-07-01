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

#ifndef HUM_DRIVER_IOCTL
#define HUM_DRIVER_IOCTL

#include <linux/ioctl.h>

#define IOCTL_HOOK                      _IOW(0xFF, 0, unsigned long)
#define IOCTL_RESTORE                   _IOW(0xFF, 1, unsigned long)

#define IOCTL_MAP_TO_GPU_WRITE_PROT     _IOW(0xff, 14, unsigned long)
#define IOCTL_UNMAP_FROM_GPU            _IOW(0xff, 17, unsigned long)

#define IOCTL_MEMCPY_H2D_PREFETCH       _IOW(0xff, 22, unsigned long)

#define IOCTL_MARKER                    _IOW(0xff, 40, unsigned long)

#ifdef COMPILE_RUNTIME
struct list_head {
  struct list_head *next, *prev;
};

#define BITS_PER_BYTE           8
#define DIV_ROUND_UP(n, d) (((n) + (d) - 1) / (d))
#define BITS_TO_LONGS(nr)       DIV_ROUND_UP(nr, BITS_PER_BYTE * sizeof(long))

#define DECLARE_BITMAP(name,bits) \
        unsigned long name[BITS_TO_LONGS(bits)]

typedef struct uvm_va_block_struct uvm_va_block_t;
#endif


struct map_command {
  uint64_t mem_start;
  uint64_t mem_length;
  int gpu_id;
};

#define MAX_CHUNK_SIZE  4194304
#define MAX_DMA_ADDRS   64
#define MAX_NUM_NODES   64

#define MAX_NUM_GPUS  4
#define MAX_NUM_PROCS MAX_NUM_GPUS + 1
typedef struct {
  DECLARE_BITMAP(bitmap, MAX_NUM_GPUS);
} memcpy_gpu_mask;

struct memcpy_direct_command {
  uint64_t dst_addr;
  uint64_t src_addr;
  size_t copy_size;
  memcpy_gpu_mask gpu_mask;
  uint64_t flag_ptr;
};

#endif
