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
#include <asm/tlbflush.h>
#include "config.h"
#include "data_structure.h"

uvm_va_space_t *get_va_space_from_mm(struct mm_struct* mm) {
  struct vm_area_struct *vma;
  uvm_va_space_t *va_space = NULL;

  if (mm == NULL) {
    LOGE("mm of current is NULL\n");
  }
  else {
    vma = mm->mmap;
    while (vma != NULL) {
      if (_hum_uvm_file_is_nvidia_uvm(vma->vm_file)) {
        va_space = uvm_va_space_get(vma->vm_file);
        break;
      }
      vma = vma->vm_next;
    }
  }
  return va_space;
}

void ioctl_map_to_gpu_write_prot(struct map_command *cmd) {
  uint64_t mem_start = cmd->mem_start;
  uint64_t mem_next = mem_start;
  int64_t mem_left = cmd->mem_length;

  NV_STATUS status = NV_OK;
  uvm_va_space_t *va_space;
  uvm_va_block_t *va_block;
  uvm_service_block_context_t *service_context;
  uvm_processor_id_t gpu_id;
  uvm_page_index_t page_index;
  uvm_page_index_t first_page_index;
  uvm_page_index_t last_page_index;
  uvm_va_block_retry_t block_retry;
  uvm_perf_thrashing_hint_t thrashing_hint;

  va_space = get_va_space_from_mm(current->mm);
  if (va_space == NULL) {
    LOGE("Failed to get va_space from mm\n");
    return;
  }

  service_context = vmalloc(sizeof(uvm_service_block_context_t));
  thrashing_hint.type = UVM_PERF_THRASHING_HINT_TYPE_NONE;

  // processor id of GPU is "GPU id in runtime" + 1
  gpu_id = uvm_id(cmd->gpu_id + 1);

  while (mem_left > 0) {
    // find va block
    status = _hum_uvm_va_block_find_create(
        va_space, mem_next, &va_block);
    if (status != NV_OK) {
      LOGE("Failed to find va_block of address %lx. status=%d\n",
          (long unsigned int)mem_next, status);
      return;
    }

    // reset per block variables
    first_page_index = PAGES_PER_UVM_VA_BLOCK;
    last_page_index = 0;

    service_context->num_retries = 0;
    service_context->read_duplicate_count = 0;
    service_context->thrashing_pin_count = 0;
    service_context->cpu_fault.did_migrate = false;
    service_context->block_context.mm =
      uvm_va_range_vma(va_block->va_range)->vm_mm;
    uvm_processor_mask_zero(
        &service_context->cpu_fault.gpus_to_check_for_ecc);
    uvm_processor_mask_zero(&service_context->resident_processors);
    uvm_page_mask_zero(&service_context->read_duplicate_mask);
    
    uvm_processor_mask_set(
        &service_context->resident_processors, gpu_id);
    uvm_page_mask_zero(
        &service_context->per_processor_masks[uvm_id_value(gpu_id)].new_residency);

    // lock this block and let's start the job
    uvm_mutex_lock(&va_block->lock);
    _hum_uvm_va_block_retry_init(&block_retry);
    
    for (; mem_next <= va_block->end && mem_left > 0;
        mem_next += PAGE_SIZE, mem_left -= PAGE_SIZE) {
      page_index = uvm_va_block_cpu_page_index(va_block, mem_next);
      uvm_page_mask_set(
          &service_context->per_processor_masks[uvm_id_value(gpu_id)].new_residency,
          page_index);

      // map the page with WRITE prot
      service_context->access_type[page_index] =
        UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG;

      // calculate the first and last page index for creating region
      if (page_index < first_page_index)
        first_page_index = page_index;
      if (page_index > last_page_index)
        last_page_index = page_index;
    }
    
    uvm_va_block_region_t region =
      uvm_va_block_region(first_page_index, last_page_index + 1);
    service_context->region = region;

    do {
      status = _hum_uvm_va_block_service_locked(
          gpu_id, va_block, &block_retry, service_context);
      if (status != NV_OK &&
          status != NV_ERR_MORE_PROCESSING_REQUIRED) {
        LOGE("Failed to service block locked. status=%d\n", status);
      }
    } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);

    _hum_uvm_va_block_retry_deinit(&block_retry, va_block);
    uvm_mutex_unlock(&va_block->lock);
  }

  vfree(service_context);
}

void ioctl_unmap(struct map_command *cmd) {
  uint64_t mem_start = cmd->mem_start;
  uint64_t mem_end = mem_start + cmd->mem_length - 1;
  uint64_t mem_next = mem_start;

  NV_STATUS status = NV_OK;
  uvm_va_space_t *va_space;
  uvm_va_block_t *va_block;
  uvm_processor_id_t gpu_id;
  uvm_page_index_t first_page_index;
  uvm_page_index_t last_page_index;
  uvm_va_block_context_t block_context;

  va_space = get_va_space_from_mm(current->mm);
  if (va_space == NULL) {
    LOGE("Failed to get va_space from mm\n");
    return;
  }

  // processor id of GPU is "GPU id in runtime" + 1
  gpu_id = uvm_id(cmd->gpu_id + 1);

  while (mem_end > mem_next) {
    status = _hum_uvm_va_block_find_create(
        va_space, mem_next, &va_block);
    if (status != NV_OK) {
      LOGE("Failed to find va_block of address %lx. status=%d\n",
          (long unsigned int)mem_next, status);
      return;
    }
    
    first_page_index =
      uvm_va_block_cpu_page_index(va_block, mem_next);
    last_page_index = uvm_va_block_cpu_page_index(va_block,
        min(va_block->end, mem_end));
    uvm_va_block_region_t region =
      uvm_va_block_region(first_page_index, last_page_index + 1);

    uvm_mutex_lock(&va_block->lock);
    status = _hum_uvm_va_block_unmap(va_block, &block_context, gpu_id,
        region, NULL, &va_block->tracker);
    if (status != NV_OK) {
      LOGE("Failed to unmap va_block from GPU. status=%d\n", status);
      return;
    }

    uvm_page_mask_t *resident_mask =
      _hum_uvm_va_block_resident_mask_get(va_block, gpu_id);
    if (resident_mask == NULL) {
      LOGE("Failed to get resident mask of GPU%d\n", uvm_id_value(gpu_id));
    }
    else {
      uvm_page_mask_region_clear(resident_mask, region);
    }
    uvm_mutex_unlock(&va_block->lock);

    mem_next = va_block->end + 1;
  }
}
