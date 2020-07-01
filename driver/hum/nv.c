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

#include "config.h"
#include "isr_gpu.h"
#include "nv.h"

uvm_global_t *g_uvm_global_p = NULL;
bool *g_uvm_perf_prefetch_enable_p = NULL;
bool *g_uvm_perf_thrashing_enable_p = NULL;

uvm_gpu_t *(*_hum_uvm_gpu_get_by_uuid_locked)(const NvProcessorUuid *) = NULL;
NV_STATUS (*_hum_uvm_va_block_find_create)(uvm_va_space_t *,
    NvU64, uvm_va_block_t **) = NULL;
NV_STATUS (*_hum_uvm_gpu_fault_entry_to_va_space)(uvm_gpu_t *,
    uvm_fault_buffer_entry_t *, uvm_va_space_t **) = NULL;
NV_STATUS (*_hum_preprocess_fault_batch)(uvm_gpu_t *,
    uvm_fault_service_batch_context_t *) = NULL;
NV_STATUS (*_hum_block_populate_page_cpu)(uvm_va_block_t *,
    uvm_page_index_t, bool) = NULL;
NV_STATUS (*_hum_uvm_va_block_unmap)(uvm_va_block_t *,
    uvm_va_block_context_t *, uvm_processor_id_t, uvm_va_block_region_t,
    const uvm_page_mask_t *, uvm_tracker_t *) = NULL;
NV_STATUS (*_hum_uvm_va_block_map)(uvm_va_block_t *,
    uvm_va_block_context_t *, uvm_processor_id_t, uvm_va_block_region_t,
    const uvm_page_mask_t *, uvm_prot_t, UvmEventMapRemoteCause,
    uvm_tracker_t *) = NULL;
NV_STATUS (*_hum_fault_buffer_flush_locked)(uvm_gpu_t *,
    uvm_gpu_buffer_flush_mode_t, uvm_fault_replay_type_t,
    uvm_fault_service_batch_context_t *) = NULL;
NV_STATUS (*_hum_uvm_va_block_service_locked)(
    uvm_processor_id_t, uvm_va_block_t *, uvm_va_block_retry_t *,
    uvm_service_block_context_t *) = NULL;
uvm_page_mask_t *(*_hum_uvm_va_block_resident_mask_get)(
    uvm_va_block_t *, uvm_processor_id_t) = NULL;

void (*_hum_uvm_gpu_replayable_faults_isr_unlock)(uvm_gpu_t *) = NULL;
NV_STATUS (*_hum_service_batch_managed_faults_in_block_locked)(
    uvm_gpu_t *, uvm_va_block_t *, uvm_va_block_retry_t *, NvU32,
    uvm_fault_service_batch_context_t *, NvU32 *) = NULL;
void (*_hum_uvm_va_block_retry_deinit)(
    uvm_va_block_retry_t *, uvm_va_block_t *) = NULL;
NV_STATUS (*_hum_push_replay_on_gpu)(uvm_gpu_t *,
    uvm_fault_replay_type_t, uvm_fault_service_batch_context_t *) = NULL;
const char *(*_hum_uvm_fault_access_type_string)(uvm_fault_access_type_t) = NULL;

bool (*_hum_uvm_file_is_nvidia_uvm)(struct file *filp) = NULL;
struct mm_struct *(*_hum_uvm_va_space_mm_retain)(uvm_va_space_t *) = NULL;
void (*_hum_uvm_va_space_mm_release)(uvm_va_space_t *) = NULL;

uvm_va_range_t *(*_hum_uvm_va_range_find)(
    uvm_va_space_t *va_space, NvU64 addr) = NULL;

void (*_hum_uvm_va_block_retry_init)(uvm_va_block_retry_t *) = NULL;
void (*_hum_uvm_tracker_deinit)(uvm_tracker_t *) = NULL;
NV_STATUS (*_hum_uvm_tracker_add_tracker_safe)(
    uvm_tracker_t *, uvm_tracker_t *) = NULL;
NV_STATUS (*_hum_uvm_tracker_wait)(uvm_tracker_t *) = NULL;

void (*_hum_uvm_gpu_kref_put)(uvm_gpu_t *) = NULL;

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// uvm_common.c
///////////////////////////////////////////////////////////////////////////////
static int uvm_debug_prints = UVM_IS_DEBUG() || UVM_IS_DEVELOP();
bool uvm_debug_prints_enabled()
{
    return uvm_debug_prints != 0;
}

// TODO: Bug 1710855: Tweak this number through benchmarks
#define UVM_SPIN_LOOP_SCHEDULE_TIMEOUT_NS   (10*1000ULL)
#define UVM_SPIN_LOOP_PRINT_TIMEOUT_SEC     30ULL
NV_STATUS uvm_spin_loop(uvm_spin_loop_t *spin)
{
    NvU64 curr = NV_GETTIME();

    // This schedule() is required for functionality, not just system
    // performance. It allows RM to run and unblock the UVM driver:
    //
    // - UVM must service faults in order for RM to idle/preempt a context
    // - RM must service interrupts which stall UVM (SW methods, stalling CE
    //   interrupts, etc) in order for UVM to service faults
    //
    // Even though UVM's bottom half is preemptable, we have encountered cases
    // in which a user thread running in RM won't preempt the UVM driver's
    // thread unless the UVM driver thread gives up its timeslice. This is also
    // theoretically possible if the RM thread has a low nice priority.
    //
    // TODO: Bug 1710855: Look into proper prioritization of these threads as a longer-term
    //       solution.
    if (curr - spin->start_time_ns >= UVM_SPIN_LOOP_SCHEDULE_TIMEOUT_NS && NV_MAY_SLEEP()) {
        schedule();
        curr = NV_GETTIME();
    }

    cpu_relax();

    // TODO: Bug 1710855: Also check fatal_signal_pending() here if the caller can handle it.

    if (curr - spin->print_time_ns >= 1000*1000*1000*UVM_SPIN_LOOP_PRINT_TIMEOUT_SEC) {
        spin->print_time_ns = curr;
        return NV_ERR_TIMEOUT_RETRY;
    }

    return NV_OK;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// uvm8_volta_fault_buffer.c
///////////////////////////////////////////////////////////////////////////////
typedef struct {
  NvU8 bufferEntry[NVC369_BUF_SIZE];
} fault_buffer_entry_c369_t;

NvU32 *get_fault_buffer_entry(uvm_gpu_t *gpu, NvU32 index) {
  fault_buffer_entry_c369_t *buffer_start;
  NvU32 *fault_entry;

  UVM_ASSERT(index < gpu->fault_buffer_info.replayable.max_faults);

  buffer_start = (fault_buffer_entry_c369_t *)gpu->fault_buffer_info.rm_info.replayable.bufferAddress;
  fault_entry = (NvU32 *)&buffer_start[index];
  return fault_entry;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// uvm8_gpu_replayable_faults.c
///////////////////////////////////////////////////////////////////////////////
void write_get(uvm_gpu_t *gpu, NvU32 get) {
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;

    UVM_ASSERT(mutex_is_locked(&gpu->isr.replayable_faults.service_lock.m));

    // Write get on the GPU only if it's changed.
    if (replayable_faults->cached_get == get)
        return;

    replayable_faults->cached_get = get;

    // Update get pointer on the GPU
    gpu->fault_buffer_hal->write_get(gpu, get);
}

int cmp_fault_instance_ptr(const uvm_fault_buffer_entry_t *a,
                                         const uvm_fault_buffer_entry_t *b) {
  int result = uvm_gpu_phys_addr_cmp(a->instance_ptr, b->instance_ptr);
  // On Volta+ we need to sort by {instance_ptr + subctx_id} pair since it can
  // map to a different VA space
  if (result != 0)
    return result;
  return UVM_CMP_DEFAULT(a->fault_source.ve_id, b->fault_source.ve_id);
}

static void fetch_fault_buffer_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                           uvm_fault_buffer_entry_t *last_entry) {
  UVM_ASSERT(last_entry->num_instances > 0);

  ++last_entry->num_instances;
  uvm_fault_access_type_mask_set(&last_entry->access_type_mask, current_entry->fault_access_type);
  
  if (current_entry->fault_access_type > last_entry->fault_access_type) {
    // If the new entry has a higher access type, it becomes the
    // fault to be serviced. Add the previous one to the list of instances
    current_entry->access_type_mask = last_entry->access_type_mask;
    current_entry->num_instances = last_entry->num_instances;
    last_entry->filtered = true;

    // We only merge faults from different uTLBs if the new fault has an
    // access type with the same or lower level of intrusiveness.
    UVM_ASSERT(current_entry->fault_source.utlb_id == last_entry->fault_source.utlb_id);

    list_replace(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
    list_add(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
  }
  else {
    // Add the new entry to the list of instances for reporting purposes
    current_entry->filtered = true;
    list_add(&current_entry->merged_instances_list, &last_entry->merged_instances_list);
  }
}

bool fetch_fault_buffer_try_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                               uvm_fault_service_batch_context_t *batch_context,
                                               uvm_fault_utlb_info_t *current_tlb,
                                               bool is_same_instance_ptr) {
  uvm_fault_buffer_entry_t *last_tlb_entry = current_tlb->last_fault;
  uvm_fault_buffer_entry_t *last_global_entry = batch_context->last_fault;

  // Check the last coalesced fault and the coalesced fault that was
  // originated from this uTLB
  const bool is_last_tlb_fault = current_tlb->num_pending_faults > 0 &&
                                 cmp_fault_instance_ptr(current_entry, last_tlb_entry) == 0 &&
                                 current_entry->fault_address == last_tlb_entry->fault_address;

  // We only merge faults from different uTLBs if the new fault has an
  // access type with the same or lower level of intrusiveness. This is to
  // avoid having to update num_pending_faults on both uTLBs and recomputing
  // last_fault.
  const bool is_last_fault = is_same_instance_ptr &&
                             current_entry->fault_address == last_global_entry->fault_address &&
                             current_entry->fault_access_type <= last_global_entry->fault_access_type;

  if (is_last_tlb_fault) {
    fetch_fault_buffer_merge_entry(current_entry, last_tlb_entry);
    if (current_entry->fault_access_type > last_tlb_entry->fault_access_type)
      current_tlb->last_fault = current_entry;

    return true;
  }
  else if (is_last_fault) {
    fetch_fault_buffer_merge_entry(current_entry, last_global_entry);
    if (current_entry->fault_access_type > last_global_entry->fault_access_type)
      batch_context->last_fault = current_entry;

    return true;
  }

  return false;
}
