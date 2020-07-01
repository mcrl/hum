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
#include <linux/vmalloc.h>
#include <linux/slab.h>
#include <linux/syscalls.h>
#include <linux/kallsyms.h>
#include <linux/kthread.h>
#include "config.h"
#include "isr_gpu.h"
#include "ioctl.h"
#include "memcpy_engine.h"

#define DEV_NAME "hum"

MODULE_LICENSE("GPL");
 
char* nvidia_name = NULL;
bool g_uvm_perf_prefetch_enable_old;
bool g_uvm_perf_thrashing_enable_old;

atomic_t use_count = ATOMIC_INIT(0);
bool ioctl_hook_called = false;
static dev_t virtual_dev_t;
static struct cdev* virtual_cdev;
static struct class* virtual_class;

void hook_idesc(void) {
  unsigned int i;
  int ret;
  struct irq_desc* idesc;
  struct irqaction* iaction;

  printk(KERN_INFO "================================================\n");
  printk(KERN_INFO "Hooking nvidia interrupt request descriptor...\n");

  for (i = 0; i < nr_irqs; ++i) {
    idesc = irq_to_desc(i);
    if (!idesc)
      continue;

    iaction = idesc->action;
    if (!iaction)
      continue;

    if (!strcmp(iaction->name, "nvidia")) {
      printk(KERN_INFO "Found nvidia irq descriptor IRQ#%d\n", iaction->irq);
      nvidia_name = iaction->name;

      ret = hook_gpu_data(iaction->dev_id);
      if (ret == 0) {
        iaction->name = "nvidia_hum";
      }
      else {
        printk(KERN_INFO "Failed to hook nvidia irq descriptor for IRQ#%d\n",
            iaction->irq);
      }
    }
  }
}

void restore_idesc(void) {
  unsigned int i;
  struct irq_desc* idesc;
  struct irqaction* iaction;

  printk(KERN_INFO "================================================\n");
  printk(KERN_INFO "Restoring nvidia interrupt request descriptor...\n");

  for (i = 0; i < nr_irqs; ++i) {
    idesc = irq_to_desc(i);
    if (!idesc)
      continue;

    iaction = idesc->action;
    if (!iaction)
      continue;

    if (!strcmp(iaction->name, "nvidia_hum")) {
      iaction->name = nvidia_name;
      unhook_gpu_data(iaction->dev_id);
    }
  }
}

void hook_nvidia_params(void) {
  #ifdef DISABLE_NVIDIA_UVM_PERF_PREFETCH
  g_uvm_perf_prefetch_enable_old = *g_uvm_perf_prefetch_enable_p;
  *g_uvm_perf_prefetch_enable_p = 0;
  #endif

  #ifdef DISABLE_NVIDIA_UVM_PERF_THRASHING
  g_uvm_perf_thrashing_enable_old = *g_uvm_perf_thrashing_enable_p;
  *g_uvm_perf_thrashing_enable_p = 0;
  #endif
}

void restore_nvidia_params(void) {
  #ifdef DISABLE_NVIDIA_UVM_PERF_PREFETCH
  *g_uvm_perf_prefetch_enable_p = g_uvm_perf_prefetch_enable_old;
  #endif

  #ifdef DISABLE_NVIDIA_UVM_PERF_THRASHING
  *g_uvm_perf_thrashing_enable_p = g_uvm_perf_thrashing_enable_old;
  #endif
}

unsigned long get_symbol_addr(char* symbol_name) {
  unsigned long symbol_addr = kallsyms_lookup_name(symbol_name);
  if (symbol_addr == 0) {
    LOGE("Failed to get symbol %s\n", symbol_name);
    return 0;
  }
  return symbol_addr;
}

void ioctl_hook(void) {
  if (ioctl_hook_called == false) {
    hook_idesc();
    hook_nvidia_params();
    ioctl_hook_called = true;
  }
}

void ioctl_restore(void) {
  if (ioctl_hook_called == true) {
    restore_nvidia_params();
    restore_idesc();
    ioctl_hook_called = false;
  }
}

int virtual_cdev_open(struct inode* inode, struct file* filep) {
  atomic_inc(&use_count);
  return 0;
}
 
int virtual_cdev_release(struct inode* inode, struct file* filep) {
  LOGD("Releasing device\n");
  
  unsigned int post_count = atomic_dec_return(&use_count);
  if (post_count == 0) {
#ifdef ENABLE_MEM_PREFETCH
    suspend_memcpy_engine();
#endif
    ioctl_restore();
  }
  LOGD("Release done\n");

  return 0;
}
 
ssize_t virtual_cdev_write(struct file* filep, const char* buf, size_t count, loff_t* f_pos) {
  return count;
}
 
ssize_t virtual_cdev_read(struct file* filep, char* buf, size_t count, loff_t* f_pos) {
  return count;
}

char *ioctl_to_string(unsigned int cmd) {
  switch (cmd) {
  case IOCTL_HOOK: return "IOCTL_HOOK";
  case IOCTL_RESTORE: return "IOCTL_RESTORE";
  case IOCTL_MAP_TO_GPU_WRITE_PROT: return "IOCTL_MAP_TO_GPU_WRITE_PROT";
  case IOCTL_UNMAP_FROM_GPU: return "IOCTL_UNMAP_FROM_GPU";
  case IOCTL_MEMCPY_H2D_PREFETCH: return "IOCTL_MEMCPY_H2D_PREFETCH";
  case IOCTL_MARKER: return "IOCTL_MARKER";
  default: return "Unknown";
  }
}

long virtual_cdev_ioctl(struct file* filep, unsigned int cmd, unsigned long arg) {
  switch (cmd) {
  case IOCTL_HOOK:
    ioctl_hook();
    break;

  case IOCTL_RESTORE:
    ioctl_restore();
    break;

  case IOCTL_MAP_TO_GPU_WRITE_PROT:
    {
      struct map_command cmd;
      copy_from_user(&cmd, (void*)arg, sizeof(struct map_command));
      ioctl_map_to_gpu_write_prot(&cmd);
    }
    break;

  case IOCTL_UNMAP_FROM_GPU:
    {
      struct map_command cmd;
      copy_from_user(&cmd, (void*)arg, sizeof(struct map_command));
      ioctl_unmap(&cmd);
    }
    break;
  
  case IOCTL_MEMCPY_H2D_PREFETCH:
    {
      struct memcpy_direct_command user_cmd;
      copy_from_user(&user_cmd, (void*)arg, sizeof(struct memcpy_direct_command));
      
      struct memcpy_command cmd;
      cmd.type = COMMAND_H2D;
      cmd.dst_addr = user_cmd.dst_addr;
      cmd.src_addr = user_cmd.src_addr;
      cmd.copy_size = user_cmd.copy_size;
      cmd.gpu_mask = user_cmd.gpu_mask;
      cmd.mm = current->mm;
      handle_h2d_prefetch(&cmd);
    }
    break;

  case IOCTL_MARKER:
    PRINT("============ IOCTL_MARKER %lu ============\n", arg);
    break;
  }

  return 0;
}

static char* virtual_class_devnode(struct device* dev, umode_t* mode) {
  if (mode)
    *mode = 0666;
  return NULL;
}

struct file_operations virtual_cdev_fops = {
  .open           = virtual_cdev_open,
  .release        = virtual_cdev_release,
  .read           = virtual_cdev_read,
  .write          = virtual_cdev_write,
//  .mmap           = virtual_cdev_mmap,
  .unlocked_ioctl = virtual_cdev_ioctl
};

static int __init hum_init(void) {
  int dev_minor = 0;
  int result;

  result = alloc_chrdev_region(&virtual_dev_t, dev_minor, 1, DEV_NAME);
  if (result < 0) {
    printk(KERN_ERR "Failed to alloc chrdev region\n");
    return result;
  }

  virtual_cdev = cdev_alloc();
  if (!virtual_cdev) {
    printk(KERN_ERR "Failed to alloc cdev\n");
    unregister_chrdev_region(virtual_dev_t, 1);
    return -ENOMEM;
  }

  cdev_init(virtual_cdev, &virtual_cdev_fops);
  virtual_cdev->owner = THIS_MODULE;

  result = cdev_add(virtual_cdev, virtual_dev_t, 1);
  if (result < 0) {
    printk(KERN_ERR "Failed to add cdev\n");

    unregister_chrdev_region(virtual_dev_t, 1);
    return result;
  }

  virtual_class = class_create(THIS_MODULE, DEV_NAME);
  if (!virtual_class) {
    printk(KERN_ERR "Failed to create class\n");
    
    cdev_del(virtual_cdev);
    unregister_chrdev_region(virtual_dev_t, 1);
    return -EEXIST;
  }
  virtual_class->devnode = virtual_class_devnode;

  if (!device_create(virtual_class, NULL, virtual_dev_t, NULL, DEV_NAME)) {
    printk(KERN_ERR "Failed to create device\n");

    class_destroy(virtual_class);
    cdev_del(virtual_cdev);
    unregister_chrdev_region(virtual_dev_t, 1);
    return -EINVAL;
  }

  // get global variable symbol from nvidia kernel module
  g_uvm_global_p = (uvm_global_t*)get_symbol_addr("g_uvm_global");
  g_uvm_perf_prefetch_enable_p =
    (bool*)get_symbol_addr("g_uvm_perf_prefetch_enable");
  g_uvm_perf_thrashing_enable_p = 
    (bool*)get_symbol_addr("g_uvm_perf_thrashing_enable");
  _hum_uvm_gpu_get_by_uuid_locked =
    (void*)get_symbol_addr("uvm_gpu_get_by_uuid_locked");
  _hum_uvm_va_block_find_create =
    (void*)get_symbol_addr("uvm_va_block_find_create");
  _hum_uvm_gpu_fault_entry_to_va_space =
    (void*)get_symbol_addr("uvm_gpu_fault_entry_to_va_space");
  _hum_preprocess_fault_batch =
    (void*)get_symbol_addr("preprocess_fault_batch");
  _hum_block_populate_page_cpu =
    (void*)get_symbol_addr("block_populate_page_cpu");
  _hum_uvm_va_block_unmap =
    (void*)get_symbol_addr("uvm_va_block_unmap");
  _hum_uvm_va_block_map =
    (void*)get_symbol_addr("uvm_va_block_map");
  _hum_fault_buffer_flush_locked =
    (void*)get_symbol_addr("fault_buffer_flush_locked");
  _hum_uvm_va_block_service_locked =
    (void*)get_symbol_addr("uvm_va_block_service_locked");
  _hum_uvm_va_block_resident_mask_get =
    (void*)get_symbol_addr("uvm_va_block_resident_mask_get");
  _hum_uvm_gpu_replayable_faults_isr_unlock =
    (void*)get_symbol_addr("uvm_gpu_replayable_faults_isr_unlock");
  _hum_service_batch_managed_faults_in_block_locked =
    (void*)get_symbol_addr("service_batch_managed_faults_in_block_locked");
  _hum_uvm_va_block_retry_deinit =
    (void*)get_symbol_addr("uvm_va_block_retry_deinit");
  _hum_push_replay_on_gpu =
    (void*)get_symbol_addr("push_replay_on_gpu");
  _hum_uvm_fault_access_type_string =
    (void*)get_symbol_addr("uvm_fault_access_type_string");
  _hum_uvm_file_is_nvidia_uvm =
    (void*)get_symbol_addr("uvm_file_is_nvidia_uvm");
  _hum_uvm_va_space_mm_retain =
    (void*)get_symbol_addr("uvm_va_space_mm_retain");
  _hum_uvm_va_space_mm_release =
    (void*)get_symbol_addr("uvm_va_space_mm_release");
  _hum_uvm_va_range_find =
    (void*)get_symbol_addr("uvm_va_range_find");
  _hum_uvm_va_block_retry_init =
    (void*)get_symbol_addr("uvm_va_block_retry_init");
  _hum_uvm_tracker_deinit =
    (void*)get_symbol_addr("uvm_tracker_deinit");
  _hum_uvm_tracker_add_tracker_safe =
    (void*)get_symbol_addr("uvm_tracker_add_tracker_safe");
  _hum_uvm_tracker_wait =
    (void*)get_symbol_addr("uvm_tracker_wait");
  _hum_uvm_gpu_kref_put =
    (void*)get_symbol_addr("uvm_gpu_kref_put");

  if (g_uvm_global_p == NULL ||
      g_uvm_perf_prefetch_enable_p == NULL ||
      g_uvm_perf_thrashing_enable_p == NULL ||
      _hum_uvm_gpu_get_by_uuid_locked == NULL ||
      _hum_uvm_va_block_find_create == NULL ||
      _hum_uvm_gpu_fault_entry_to_va_space == NULL ||
      _hum_preprocess_fault_batch == NULL ||
      _hum_block_populate_page_cpu == NULL ||
      _hum_uvm_va_block_unmap == NULL ||
      _hum_uvm_va_block_map == NULL ||
      _hum_fault_buffer_flush_locked == NULL ||
      _hum_uvm_va_block_service_locked == NULL ||
      _hum_uvm_va_block_resident_mask_get == NULL ||
      _hum_uvm_gpu_replayable_faults_isr_unlock == NULL ||
      _hum_service_batch_managed_faults_in_block_locked == NULL ||
      _hum_uvm_va_block_retry_deinit == NULL ||
      _hum_push_replay_on_gpu == NULL ||
      _hum_uvm_fault_access_type_string == NULL ||
      _hum_uvm_file_is_nvidia_uvm == NULL ||
      _hum_uvm_va_space_mm_retain == NULL ||
      _hum_uvm_va_space_mm_release == NULL ||
      _hum_uvm_va_range_find == NULL ||
      _hum_uvm_va_block_retry_init == NULL ||
      _hum_uvm_tracker_deinit == NULL ||
      _hum_uvm_tracker_add_tracker_safe == NULL ||
      _hum_uvm_tracker_wait == NULL ||
      _hum_uvm_gpu_kref_put == NULL) {
    device_destroy(virtual_class, virtual_dev_t);
    class_destroy(virtual_class);
    cdev_del(virtual_cdev);
    unregister_chrdev_region(virtual_dev_t, 1);
    return -EINVAL;
  }

#ifdef ENABLE_MEM_PREFETCH
  setup_memcpy_engine();
#endif

  printk(KERN_INFO "Initialized hum kernel module\n");

  return 0;
}
 
static void __exit hum_exit(void) {
  printk(KERN_INFO "Exiting hum kernel module\n");

#ifdef ENABLE_MEM_PREFETCH
  clean_memcpy_engine();
#endif

  device_destroy(virtual_class, virtual_dev_t);
  class_destroy(virtual_class);
  cdev_del(virtual_cdev);
  unregister_chrdev_region(virtual_dev_t, 1);
}
 
module_init(hum_init);
module_exit(hum_exit);

