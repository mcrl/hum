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

#ifndef HUM_DRIVER_NV
#define HUM_DRIVER_NV

#include "nv.h"
#include "uvm8_global.h"
#include "uvm8_processors.h"
#include "uvm8_hal.h"
#include "uvm8_va_space.h"
#include "uvm8_va_block.h"
#include "uvm8_va_range.h"
#include "uvm8_kvmalloc.h"
#include "uvm8_va_space_mm.h"
#include "clc369.h"
#include "hwref/volta/gv100/dev_fault.h"
#include "hwref/volta/gv100/dev_fb.h"
#include "uvm8_volta_fault_buffer.h"
#include "uvm8_perf_prefetch.h"

extern uvm_global_t *g_uvm_global_p;
extern bool *g_uvm_perf_prefetch_enable_p;
extern bool *g_uvm_perf_thrashing_enable_p;

extern uvm_gpu_t *(*_hum_uvm_gpu_get_by_uuid_locked)(const NvProcessorUuid *);
extern NV_STATUS (*_hum_uvm_va_block_find_create)(uvm_va_space_t *,
    NvU64, uvm_va_block_t **);
extern NV_STATUS (*_hum_uvm_gpu_fault_entry_to_va_space)(uvm_gpu_t *,
    uvm_fault_buffer_entry_t *, uvm_va_space_t **);
extern NV_STATUS (*_hum_preprocess_fault_batch)(uvm_gpu_t *,
    uvm_fault_service_batch_context_t *);
extern NV_STATUS (*_hum_block_populate_page_cpu)(uvm_va_block_t *,
    uvm_page_index_t, bool);
extern NV_STATUS (*_hum_uvm_va_block_make_resident)(uvm_va_block_t *,
    uvm_va_block_retry_t *, uvm_va_block_context_t *, uvm_processor_id_t,
    uvm_va_block_region_t, const uvm_page_mask_t *, const uvm_page_mask_t *,
    uvm_make_resident_cause_t);
extern NV_STATUS (*_hum_uvm_va_block_make_resident_read_duplicate)(
    uvm_va_block_t *, uvm_va_block_retry_t *, uvm_va_block_context_t *,
    uvm_processor_id_t, uvm_va_block_region_t, const uvm_page_mask_t *,
    const uvm_page_mask_t *, uvm_make_resident_cause_t);
extern NV_STATUS (*_hum_uvm_va_block_unmap)(uvm_va_block_t *,
    uvm_va_block_context_t *, uvm_processor_id_t, uvm_va_block_region_t,
    const uvm_page_mask_t *, uvm_tracker_t *);
extern NV_STATUS (*_hum_uvm_va_block_map)(uvm_va_block_t *,
    uvm_va_block_context_t *, uvm_processor_id_t, uvm_va_block_region_t,
    const uvm_page_mask_t *, uvm_prot_t, UvmEventMapRemoteCause,
    uvm_tracker_t *);
extern NV_STATUS (*_hum_fault_buffer_flush_locked)(uvm_gpu_t *,
    uvm_gpu_buffer_flush_mode_t, uvm_fault_replay_type_t,
    uvm_fault_service_batch_context_t *);
extern NV_STATUS (*_hum_uvm_va_block_service_locked)(
    uvm_processor_id_t, uvm_va_block_t *, uvm_va_block_retry_t *,
    uvm_service_block_context_t *);
extern uvm_page_mask_t *(*_hum_uvm_va_block_resident_mask_get)(
    uvm_va_block_t *, uvm_processor_id_t);

extern void (*_hum_uvm_gpu_replayable_faults_isr_unlock)(uvm_gpu_t *);
extern NV_STATUS (*_hum_service_batch_managed_faults_in_block_locked)(
    uvm_gpu_t *, uvm_va_block_t *, uvm_va_block_retry_t *, NvU32,
    uvm_fault_service_batch_context_t *, NvU32 *);
extern void (*_hum_uvm_va_block_retry_deinit)(
    uvm_va_block_retry_t *, uvm_va_block_t *);
extern NV_STATUS (*_hum_push_replay_on_gpu)(uvm_gpu_t *,
    uvm_fault_replay_type_t, uvm_fault_service_batch_context_t *);
extern const char *(*_hum_uvm_fault_access_type_string)(uvm_fault_access_type_t);

extern bool (*_hum_uvm_file_is_nvidia_uvm)(struct file *filp);
extern struct mm_struct *(*_hum_uvm_va_space_mm_retain)(uvm_va_space_t *);
extern void (*_hum_uvm_va_space_mm_release)(uvm_va_space_t *);

extern uvm_va_range_t *(*_hum_uvm_va_range_find)(
    uvm_va_space_t *va_space, NvU64 addr);

extern void (*_hum_uvm_va_block_retry_init)(uvm_va_block_retry_t *);
extern void (*_hum_uvm_tracker_deinit)(uvm_tracker_t *);
extern NV_STATUS (*_hum_uvm_tracker_add_tracker_safe)(
    uvm_tracker_t *, uvm_tracker_t *);
extern NV_STATUS (*_hum_uvm_tracker_wait)(uvm_tracker_t *);

extern void (*_hum_uvm_gpu_kref_put)(uvm_gpu_t *);

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// uvm8_volta_fault_buffer.c
///////////////////////////////////////////////////////////////////////////////
NvU32 *get_fault_buffer_entry(uvm_gpu_t *gpu, NvU32 index);
void parse_fault_entry_common(uvm_gpu_t *gpu, NvU32 *fault_entry,
    uvm_fault_buffer_entry_t *buffer_entry);

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// uvm8_gpu_replayable_faults.c
///////////////////////////////////////////////////////////////////////////////
typedef enum {
  // Fetch a batch of faults from the buffer.
  FAULT_FETCH_MODE_BATCH_ALL,

  // Fetch a batch of faults from the buffer. Stop at the first entry that is
  // not ready yet
  FAULT_FETCH_MODE_BATCH_READY,

  // Fetch all faults in the buffer before PUT. Wait for all faults to become
  // ready
  FAULT_FETCH_MODE_ALL,
} fault_fetch_mode_t;

void write_get(uvm_gpu_t *gpu, NvU32 get);

int cmp_fault_instance_ptr(const uvm_fault_buffer_entry_t *a,
    const uvm_fault_buffer_entry_t *b);
bool fetch_fault_buffer_try_merge_entry(
    uvm_fault_buffer_entry_t *current_entry,
    uvm_fault_service_batch_context_t *batch_context,
    uvm_fault_utlb_info_t *current_tlb, bool is_same_instance_ptr);

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// uvm8_va_block.h
///////////////////////////////////////////////////////////////////////////////
#define _HUM_UVM_VA_BLOCK_RETRY_LOCKED(va_block, block_retry, call) ({   \
    NV_STATUS status;                                                     \
    uvm_va_block_t *__block = (va_block);                                 \
    uvm_va_block_retry_t *__retry = (block_retry);                        \
                                                                          \
    _hum_uvm_va_block_retry_init(__retry);                               \
                                                                          \
    uvm_assert_mutex_locked(&__block->lock);                              \
                                                                          \
    do {                                                                  \
        status = (call);                                                  \
    } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);                  \
                                                                          \
    _hum_uvm_va_block_retry_deinit(__retry, __block);                    \
                                                                          \
    status;                                                               \
})

#endif
