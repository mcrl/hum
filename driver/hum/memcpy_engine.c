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
#include <linux/hashtable.h>
#include <linux/kthread.h>
#include "config.h"
#include "nv.h"
#include "data_structure.h"
#include "isr_gpu.h"
#include "memcpy_engine.h"

extern wait_queue_head_t gpu_mem_sem[MAX_NUM_PROCS];

struct task_struct *memcpy_engine[MAX_NUM_PROCS] = {NULL};
wait_queue_head_t memcpy_engine_sem[MAX_NUM_PROCS];
bool memcpy_engine_ignition[MAX_NUM_PROCS] = {false};

volatile struct service_command memcpy_queue[MAX_NUM_PROCS][MEMCPY_COMMAND_QUEUE_SIZE];
atomic_t memcpy_queue_head[MAX_NUM_PROCS];
unsigned int memcpy_queue_tail[MAX_NUM_PROCS];

int memcpy_engine_func(void *data) {
  struct service_command *command;
  int id = (uint64_t)data;

  NV_STATUS status = NV_OK;
  uvm_service_block_context_t *service_context;
  uvm_va_block_retry_t block_retry;
  uvm_va_block_t *va_block;
  uvm_page_mask_t *resident_mask;

  while (!kthread_should_stop()) {
    // fetch available command
    command = get_service_command(id);
    
    if (command) {
      service_context = command->service_context;
      va_block = command->va_block;

      _hum_uvm_va_block_retry_init(&block_retry);

      // mark as pages are resident on CPU so that driver copy pages from
      // CPU to GPU
      resident_mask = &(va_block->cpu.resident);
      if (resident_mask == NULL) {
        LOGE("Failed to get resident mask of CPU\n");
        goto next_command;
      }
      uvm_page_mask_or(resident_mask, resident_mask,
          &service_context->per_processor_masks[id].new_residency);

      // add CPU to the resident list
      uvm_processor_mask_set(&va_block->resident, UVM_ID_CPU);

      // finally, upload this block to GPU
      uvm_processor_mask_set(
          &service_context->resident_processors, uvm_id(id));
      do {
        status = _hum_uvm_va_block_service_locked(
            uvm_id(id), va_block, &block_retry, service_context);
        if (status != NV_OK &&
            status != NV_ERR_MORE_PROCESSING_REQUIRED) {
          LOGE("Failed to service block locked. status=%d\n", status);
          goto next_command;
        }
      } while (status == NV_ERR_MORE_PROCESSING_REQUIRED);

next_command:
      _hum_uvm_va_block_retry_deinit(&block_retry, va_block);
      uvm_mutex_unlock(&va_block->lock);
      vfree(service_context);

      // fetch next command
      continue;
    }
    
    // when no command exists, sleep once
    schedule();

    // goto sleep when suspend mode is on
    wait_event_interruptible(memcpy_engine_sem[id],
        memcpy_engine_ignition[id] == true);
  }

  return 0;
}

void handle_h2d_prefetch(struct memcpy_command *command) {
#ifdef ENABLE_MEM_PREFETCH
  uint64_t dst_addr, src_addr, dst_kern_addr;
  uint64_t page_offset;
  size_t left_copy_size;
  size_t copy_size;
  unsigned long ret;
  
  NV_STATUS status = NV_OK;
  uvm_va_space_t *va_space;
  uvm_va_block_t *va_block;
  uvm_service_block_context_t *service_context;
  uvm_page_index_t page_index;
  uvm_page_index_t first_page_index;
  uvm_page_index_t last_page_index;
  uvm_perf_thrashing_hint_t thrashing_hint;
  uvm_processor_id_t gpu_id;
  struct page* target_page;
  
  va_space = get_va_space_from_mm(command->mm);
  if (!va_space) {
    LOGE("Failed to memcpy direct h2d due to NULL va_space\n");
    return;
  }

  dst_addr = command->dst_addr;
  src_addr = command->src_addr;
  left_copy_size = command->copy_size;
  gpu_id = uvm_id(find_first_bit(command->gpu_mask.bitmap, MAX_NUM_GPUS) + 1);

  thrashing_hint.type = UVM_PERF_THRASHING_HINT_TYPE_NONE;

  while (left_copy_size > 0) {
    service_context = vmalloc(sizeof(uvm_service_block_context_t));
    service_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;

    // find va_block
    status = _hum_uvm_va_block_find_create(
        va_space, dst_addr, &va_block);
    if (status != NV_OK) {
      LOGE("Failed to find va_block of address %lx. status=%d\n",
          (long unsigned int)dst_addr, status);
      return;
    }
    
    // reset per block variables
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
    first_page_index = PAGES_PER_UVM_VA_BLOCK;
    last_page_index = 0;
    
    uvm_mutex_lock(&va_block->lock);
    
    page_index = uvm_va_block_cpu_page_index(va_block, dst_addr);
    page_offset = dst_addr & (PAGE_SIZE - 1);

    // traverse pages in a uvm_va_block and copy pages to managed
    for (; dst_addr <= va_block->end && left_copy_size > 0;) {
      uvm_page_mask_set(
          &service_context->per_processor_masks[uvm_id_value(gpu_id)].new_residency,
          page_index);
      
      // upload with WRITE prot
      service_context->access_type[page_index] =
        UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG;

      target_page = va_block->cpu.pages[page_index];
      if (target_page == NULL) {
        status = _hum_block_populate_page_cpu(
            va_block, page_index, false);
        if (status != NV_OK) {
          LOGE("Failed to populate page cpu in a block. status=%d\n",
              status);
          return;
        }
        target_page = va_block->cpu.pages[page_index];
      }

      unsigned long kern_addr =
        (unsigned long)pfn_to_kaddr(page_to_pfn(target_page));

      dst_kern_addr = kern_addr + page_offset;
      copy_size = page_offset + left_copy_size > PAGE_SIZE ?
        PAGE_SIZE - page_offset : left_copy_size;
      
      // copy to managed memory
      ret = copy_from_user(
          (void*)dst_kern_addr, (void*)src_addr, copy_size);
      if (ret)
        LOGE("Failed to copy %lubytes of %lubytes from user\n",
            ret, copy_size);

      // calculate the first and last page index for creating region
      if (page_index < first_page_index)
        first_page_index = page_index;
      if (page_index > last_page_index)
        last_page_index = page_index;

      // update variables
      page_index += 1;
      page_offset = 0;
      dst_addr += copy_size;
      src_addr += copy_size;
      left_copy_size -= copy_size;
    }

    uvm_va_block_region_t region =
      uvm_va_block_region(first_page_index, last_page_index + 1);
    service_context->region = region;

    add_service_command(uvm_id_value(gpu_id), va_block, service_context);
  }
#endif
}

void add_service_command(int id, uvm_va_block_t *va_block,
    uvm_service_block_context_t *service_context) {
  // get a ticket atomically
  unsigned int queue_head = atomic_inc_return(&memcpy_queue_head[id]);
  queue_head = queue_head % MEMCPY_COMMAND_QUEUE_SIZE;

  volatile struct service_command *command = &memcpy_queue[id][queue_head];
  command->service_context = service_context;
  command->va_block = va_block;
  command->valid = true;

  if (memcpy_engine_ignition[id] == false) {
    memcpy_engine_ignition[id] = true;
    wake_up_interruptible(&memcpy_engine_sem[id]);
  }
}

struct service_command *get_service_command(int id) {
  int queue_tail = memcpy_queue_tail[id] % MEMCPY_COMMAND_QUEUE_SIZE;

  // return only when the item is valid
  if (memcpy_queue[id][queue_tail].valid == true) {
    memcpy_queue[id][queue_tail].valid = false;
    memcpy_queue_tail[id]++;
    return (struct service_command*)(&memcpy_queue[id][queue_tail]);
  }

  return NULL;
}

void setup_memcpy_engine(void) {
  unsigned long int i, j;
  char thread_name[256];

  for (i = 1; i < MAX_NUM_PROCS; ++i) {
    // initialize the queues
    for (j = 0; j < MEMCPY_COMMAND_QUEUE_SIZE; ++j)
      memcpy_queue[i][j].valid = false;

    atomic_set(&memcpy_queue_head[i], 0);
    memcpy_queue_tail[i] = 1;
    
    init_waitqueue_head(&memcpy_engine_sem[i]);

    sprintf(thread_name, "memcpy_engine%lu", i);
    memcpy_engine[i] = kthread_run(
      memcpy_engine_func, (void*)i, thread_name);
  }
}

void suspend_memcpy_engine(void) {
  unsigned int i;
  for (i = 1; i < MAX_NUM_PROCS; ++i) {
    memcpy_engine_ignition[i] = false;
  }
}

void clean_memcpy_engine(void) {
  unsigned int i;
  for (i = 1; i < MAX_NUM_PROCS; ++i) {
    memcpy_engine_ignition[i] = true;
    wake_up_interruptible(&memcpy_engine_sem[i]);
    kthread_stop(memcpy_engine[i]);
  }
}
