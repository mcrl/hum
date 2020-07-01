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

#include <linux/module.h>
#include <linux/time.h>
#include <linux/rtc.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/irq.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/kthread.h>
#include <asm/tlbflush.h>
#include "config.h"
#include "isr_gpu.h"
#include "nv.h"

// for saving original bottom_half function of nvidia driver
nv_q_func_t replayable_faults_isr_bottom_half;

wait_queue_head_t gpu_mem_sem[MAX_NUM_PROCS];

static void hooked_replayable_faults_isr_bottom_half(void *args) {
  uvm_gpu_t *gpu = (uvm_gpu_t *)args;
  unsigned int cpu;

  UVM_ASSERT(gpu->replayable_faults_supported);

  // Multiple bottom halves for replayable faults can be running
  // concurrently, but only one can be running this function for a given GPU
  // since we enter with the replayable_faults.service_lock held.
  cpu = get_cpu();
  ++gpu->isr.replayable_faults.stats.bottom_half_count;
  cpumask_set_cpu(cpu, &gpu->isr.replayable_faults.stats.cpus_used_mask);
  ++gpu->isr.replayable_faults.stats.cpu_exec_count[cpu];
  put_cpu();

  //uvm_gpu_service_replayable_faults(gpu);
  hum_gpu_service_replayable_faults(gpu);

  _hum_uvm_gpu_replayable_faults_isr_unlock(gpu);
  _hum_uvm_gpu_kref_put(gpu);

  return;
}

int hook_gpu_data(void *dev_id) {
  nv_linux_state_t *nvl = (void *)dev_id;
  nv_state_t *nv = NV_STATE_PTR(nvl);
  uvm_gpu_t *gpu;

  const NvProcessorUuid* gpu_uuid =
    (const NvProcessorUuid *)nv_get_cached_uuid(nv);
  if (!gpu_uuid) {
    LOGE("Failed to get uuid of GPU\n");
    return -1;
  }

  if (g_uvm_global_p == NULL) {
    LOGE("Failed to get g_uvm_global pointer\n");
    return -1;
  }
  
  uvm_spin_lock_irqsave(&g_uvm_global_p->gpu_table_lock);

  gpu = _hum_uvm_gpu_get_by_uuid_locked(gpu_uuid);
  if (gpu == NULL) {
    LOGE("Failed to get GPU when hooking gpu data\n");
    uvm_spin_unlock_irqrestore(&g_uvm_global_p->gpu_table_lock);
    return -1;
  }

  nv_kref_get(&gpu->gpu_kref);
  uvm_spin_unlock_irqrestore(&g_uvm_global_p->gpu_table_lock);

  uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);
  
  init_waitqueue_head(&gpu_mem_sem[uvm_id_value(gpu->id)]);

  // hook bottom_half function of nvidia driver
  // we do not want nvidia driver worker threads to handle the GPU faults
  // so replace function pointer of worker thread to do nothing
  nv_kthread_q_item_t *q_item =
    &gpu->isr.replayable_faults.bottom_half_q_item;
  replayable_faults_isr_bottom_half = q_item->function_to_run;
  q_item->function_to_run = hooked_replayable_faults_isr_bottom_half;

  uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);
  _hum_uvm_gpu_kref_put(gpu);

  return 0;
}

void unhook_gpu_data(void *dev_id) {
  nv_linux_state_t *nvl = (void *)dev_id;
  nv_state_t *nv = NV_STATE_PTR(nvl);
  uvm_gpu_t *gpu;

  const NvProcessorUuid* gpu_uuid =
    (const NvProcessorUuid *)nv_get_cached_uuid(nv);
  if (!gpu_uuid) {
    LOGE("Failed to get uuid of GPU\n");
    return;
  }

  if (g_uvm_global_p == NULL) {
    LOGE("Failed to get g_uvm_global pointer\n");
    return;
  }
  
  uvm_spin_lock_irqsave(&g_uvm_global_p->gpu_table_lock);

  gpu = _hum_uvm_gpu_get_by_uuid_locked(gpu_uuid);
  if (gpu == NULL) {
    LOGE("Failed to get GPU when hooking gpu data\n");
    uvm_spin_unlock_irqrestore(&g_uvm_global_p->gpu_table_lock);
    return;
  }

  nv_kref_get(&gpu->gpu_kref);
  uvm_spin_unlock_irqrestore(&g_uvm_global_p->gpu_table_lock);

  uvm_spin_lock_irqsave(&gpu->isr.interrupts_lock);

  // restore bottom_half function of nvidia driver we previously hooked
  nv_kthread_q_item_t *q_item =
    &gpu->isr.replayable_faults.bottom_half_q_item;
  q_item->function_to_run = replayable_faults_isr_bottom_half;

  uvm_spin_unlock_irqrestore(&gpu->isr.interrupts_lock);
  _hum_uvm_gpu_kref_put(gpu);
}

void hum_gpu_service_replayable_faults(uvm_gpu_t *gpu) {
  NvU32 num_replays = 0;
  NV_STATUS status = NV_OK;
  uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;
  uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

  UVM_ASSERT(gpu->replayable_faults_supported);

  uvm_tracker_init(&batch_context->tracker);
  
  while (1) {
    batch_context->num_invalid_prefetch_faults = 0;
    batch_context->num_duplicate_faults        = 0;
    batch_context->num_replays                 = 0;
    batch_context->has_fatal_faults            = false;
    batch_context->has_throttled_faults        = false;
  
    // find faulting page
    fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
    if (batch_context->num_cached_faults == 0)
      break;
    
    ++batch_context->batch_id;

    status = _hum_preprocess_fault_batch(gpu, batch_context);
    
    num_replays += batch_context->num_replays;
    
    if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
      continue;
    else if (status != NV_OK)
      break;
  
    status = service_fault_batch(gpu, batch_context);
    
    num_replays += batch_context->num_replays;

    if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
      continue;

    if (status != NV_OK) {
      LOGE("Failed to service fault batch. status=%d\n", status);
      break;
    }

    if (replayable_faults->replay_policy ==
        UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
      status = _hum_push_replay_on_gpu(
          gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
      if (status != NV_OK)
        return;
      ++num_replays;
    }
    else if (replayable_faults->replay_policy ==
        UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
      uvm_gpu_buffer_flush_mode_t flush_mode =
        UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT;
      
      if (batch_context->num_duplicate_faults * 100 >
          batch_context->num_cached_faults *
          replayable_faults->replay_update_put_ratio) {
        flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
      }
      
      status = _hum_fault_buffer_flush_locked(gpu, flush_mode,
          UVM_FAULT_REPLAY_TYPE_START, batch_context);
      if (status != NV_OK)
        return;
      ++num_replays;
      
      status = _hum_uvm_tracker_wait(&replayable_faults->replay_tracker);
      if (status != NV_OK)
        return;
    }
  }
  
  // Make sure that we issue at least one replay if no replay has been
  // issued yet to avoid dropping faults that do not show up in the buffer
  if (num_replays == 0) {
    status = _hum_push_replay_on_gpu(
        gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
  }

  _hum_uvm_tracker_deinit(&batch_context->tracker);

  if (status != NV_OK)
    LOGE("Error servicing replayable faults on GPU%d\n", uvm_id_value(gpu->id));
}

static unsigned uvm_perf_fault_coalesce = 1;
void fetch_fault_buffer_entries(uvm_gpu_t *gpu,
    uvm_fault_service_batch_context_t *batch_context, fault_fetch_mode_t fetch_mode) {
  NvU32 get;
  NvU32 put;
  NvU32 fault_index;
  NvU32 num_coalesced_faults;
  NvU32 utlb_id;
  uvm_fault_buffer_entry_t *fault_cache;
  uvm_spin_loop_t spin;
  uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->fault_buffer_info.replayable;
  const bool in_pascal_cancel_path = (!gpu->fault_cancel_va_supported && fetch_mode == FAULT_FETCH_MODE_ALL);
  const bool may_filter = uvm_perf_fault_coalesce && !in_pascal_cancel_path;

  // TODO: Bug 1766600: right now uvm locks do not support the synchronization
  //       method used by top and bottom ISR. Add uvm lock assert when it's
  //       supported. Use plain mutex kernel utilities for now.
  UVM_ASSERT(mutex_is_locked(&gpu->isr.replayable_faults.service_lock.m));
  UVM_ASSERT(gpu->replayable_faults_supported);

  fault_cache = batch_context->fault_cache;

  get = replayable_faults->cached_get;
  
  // TODO
  // Read put pointer from GPU and cache it
//  if (get == replayable_faults->cached_put)
    replayable_faults->cached_put = gpu->fault_buffer_hal->read_put(gpu);

  put = replayable_faults->cached_put;

  batch_context->is_single_instance_ptr = true;
  batch_context->last_fault = NULL;

  fault_index = 0;
  num_coalesced_faults = 0;

  // Clear uTLB counters
  for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
    batch_context->utlbs[utlb_id].num_pending_faults = 0;
    batch_context->utlbs[utlb_id].has_fatal_faults = false;
  }
  batch_context->max_utlb_id = 0;

  if (get == put)
    goto done;

  // Parse until get != put and have enough space to cache.
  while ((get != put) &&
      (fetch_mode == FAULT_FETCH_MODE_ALL || fault_index < gpu->fault_buffer_info.max_batch_size)) {
    bool is_same_instance_ptr = true;
    uvm_fault_buffer_entry_t *current_entry = &fault_cache[fault_index];
    uvm_fault_utlb_info_t *current_tlb;

    // We cannot just wait for the last entry (the one pointed by put) to
    // become valid, we have to do it individually since entries can be
    // written out of order
    UVM_SPIN_WHILE(!gpu->fault_buffer_hal->entry_is_valid(gpu, get), &spin) {
      // We have some entry to work on. Let's do the rest later.
      if (fetch_mode != FAULT_FETCH_MODE_ALL &&
          fetch_mode != FAULT_FETCH_MODE_BATCH_ALL &&
          fault_index > 0)
        goto done;
    }

    // Prevent later accesses being moved above the read of the valid bit
    smp_mb__after_atomic();

    // Got valid bit set. Let's cache.
    gpu->fault_buffer_hal->parse_entry(gpu, get, current_entry);

    // The GPU aligns the fault addresses to 4k, but all of our tracking is
    // done in PAGE_SIZE chunks which might be larger.
    current_entry->fault_address = UVM_PAGE_ALIGN_DOWN(current_entry->fault_address);
    
    // Make sure that all fields in the entry are properly initialized
    current_entry->is_fatal = (current_entry->fault_type >= UVM_FAULT_TYPE_FATAL);
    
    if (current_entry->is_fatal) {
      // Record the fatal fault event later as we need the va_space locked
      current_entry->fatal_reason = UvmEventFatalReasonInvalidFaultType;
    }
    else {
      current_entry->fatal_reason = UvmEventFatalReasonInvalid;
    }

    current_entry->va_space = NULL;
    current_entry->filtered = false;

    if (current_entry->fault_source.utlb_id > batch_context->max_utlb_id) {
        UVM_ASSERT(current_entry->fault_source.utlb_id < replayable_faults->utlb_count);
        batch_context->max_utlb_id = current_entry->fault_source.utlb_id;
    }

    current_tlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
    
    if (fault_index > 0) {
      UVM_ASSERT(batch_context->last_fault);
      is_same_instance_ptr = cmp_fault_instance_ptr(current_entry, batch_context->last_fault) == 0;
      
      // Coalesce duplicate faults when possible
      if (may_filter && !current_entry->is_fatal) {
        bool merged = fetch_fault_buffer_try_merge_entry(current_entry,
                                                         batch_context,
                                                         current_tlb,
                                                         is_same_instance_ptr);
        if (merged)
          goto next_fault;
      }
    }

    if (batch_context->is_single_instance_ptr && !is_same_instance_ptr)
        batch_context->is_single_instance_ptr = false;

//    PRINT("GPU%d fault_addr=%lx, type=%s\n", uvm_id_value(gpu->id),
//        (long unsigned int)current_entry->fault_address,
//        _hum_uvm_fault_access_type_string(current_entry->fault_access_type));

    current_entry->num_instances = 1;
    current_entry->access_type_mask = uvm_fault_access_type_mask_bit(current_entry->fault_access_type);
    INIT_LIST_HEAD(&current_entry->merged_instances_list);

    ++current_tlb->num_pending_faults;
    current_tlb->last_fault = current_entry;
    batch_context->last_fault = current_entry;

    ++num_coalesced_faults;
  next_fault:
    ++fault_index;
    ++get;
    if (get == replayable_faults->max_faults)
      get = 0;
  }

done:
  write_get(gpu, get);

  batch_context->num_cached_faults = fault_index;
  batch_context->num_coalesced_faults = num_coalesced_faults;

  if (fault_index == 0)
    return;

  LOGD("GPU%d %s cached_faults=%d, coalesced_faults=%d\n",
      uvm_id_value(gpu->id), __FUNCTION__, fault_index, num_coalesced_faults);
}

bool is_pages_resident_on(uvm_va_block_t *va_block,
    uvm_processor_id_t processor_id, uvm_page_mask_t* fault_page_mask) {
  uvm_page_mask_t prefetched_mask;
  uvm_page_mask_t *resident_mask =
    _hum_uvm_va_block_resident_mask_get(va_block, processor_id);
  
  if (resident_mask == NULL)
    return false;

  uvm_page_mask_and(&prefetched_mask, resident_mask, fault_page_mask);

  return bitmap_equal(prefetched_mask.bitmap, fault_page_mask->bitmap,
      PAGES_PER_UVM_VA_BLOCK);
}

NV_STATUS service_fault_batch(uvm_gpu_t *gpu,
    uvm_fault_service_batch_context_t *batch_context) {
  unsigned int i, j;
  NV_STATUS status = NV_OK;
  NV_STATUS tracker_status;
  uvm_va_space_t *va_space = NULL;
  struct mm_struct *mm = NULL;
  uvm_fault_buffer_entry_t **ordered_fault_cache =
    batch_context->ordered_fault_cache;
#ifdef ENABLE_MEM_PREFETCH
  uvm_page_mask_t fault_page_mask;
  uvm_page_index_t page_index;
#endif

  // traverse faults and send REQUEST_PTE message
  for (i = 0; i < batch_context->num_coalesced_faults;) {
    uvm_va_block_t *va_block;
    uvm_fault_buffer_entry_t *current_entry = ordered_fault_cache[i];
    //uvm_fault_utlb_info_t *utlb =
    //  &batch_context->utlbs[current_entry->fault_source.utlb_id];

    UVM_ASSERT(current_entry->va_space);
    
    if (va_space != current_entry->va_space) {
      if (va_space) {
        uvm_va_space_up_read(va_space);
        
        if (mm) {
          uvm_up_read_mmap_sem(&mm->mmap_sem);
          _hum_uvm_va_space_mm_release(va_space);
          mm = NULL;
        }
      }

      va_space = current_entry->va_space;
      mm = _hum_uvm_va_space_mm_retain(va_space);
      if (mm)
        uvm_down_read_mmap_sem(&mm->mmap_sem);
      uvm_va_space_down_read(va_space);
    }

    status = _hum_uvm_va_block_find_create(current_entry->va_space,
        current_entry->fault_address, &va_block);
    if (status != NV_OK) {
      LOGE("Failed to find va_block of GPU address %lx. status=%d\n",
          (long unsigned int)current_entry->fault_address, status);
      return status;
    }

#ifdef ENABLE_MEM_PREFETCH
    uvm_page_mask_zero(&fault_page_mask);
    
    for (j = i;
        j < batch_context->num_coalesced_faults &&
        ordered_fault_cache[j]->fault_address <= va_block->end;
        ++j) {
      if (ordered_fault_cache[j]->fault_access_type ==
          UVM_FAULT_ACCESS_TYPE_READ) {
        page_index = uvm_va_block_cpu_page_index(va_block,
            ordered_fault_cache[j]->fault_address);
        uvm_page_mask_set(&fault_page_mask, page_index);
      }
    }

    if (!uvm_page_mask_empty(&fault_page_mask)) {
      // wait until the pages in this block are resident in this GPU
//      wait_event_interruptible(gpu_mem_sem[uvm_id_value(gpu->id)],
//          is_pages_resident_on(va_block, gpu->id, &fault_page_mask));
      
      while (is_pages_resident_on(
            va_block, gpu->id, &fault_page_mask) == false) {
        schedule();
      }

      // this block no longer needs to be processed
      // move on to the next block
      i = j;
      continue;
    }

#endif

    // now we can lock this va_block
    uvm_mutex_lock(&va_block->lock);

    /////////////////////////////////////////////////////////////////
    NvU32 nv_block_faults;
    uvm_va_block_retry_t va_block_retry;
    uvm_service_block_context_t *fault_block_context =
      &gpu->fault_buffer_info.replayable.block_service_context;
  
    fault_block_context->operation =
      UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;
    fault_block_context->num_retries = 0;
    //fault_block_context->block_context.mm =
    //  uvm_va_space_mm_get_mm(va_space_mm);
    fault_block_context->block_context.mm = NULL;
  
    status = _HUM_UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
        _hum_service_batch_managed_faults_in_block_locked(
          gpu, va_block, &va_block_retry, i,
          batch_context, &nv_block_faults));
  
    tracker_status = _hum_uvm_tracker_add_tracker_safe(
        &batch_context->tracker, &va_block->tracker);
    /////////////////////////////////////////////////////////////////
  
    uvm_mutex_unlock(&va_block->lock);

    i += nv_block_faults;
  }

  if (va_space) {
    uvm_va_space_up_read(va_space);
    
    if (mm) {
      uvm_up_read_mmap_sem(&mm->mmap_sem);
      _hum_uvm_va_space_mm_release(va_space);
    }
  }

  return status;
}
