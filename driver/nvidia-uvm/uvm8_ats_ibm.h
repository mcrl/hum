/*******************************************************************************
    Copyright (c) 2018-2019 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM8_ATS_IBM_H__
#define __UVM8_ATS_IBM_H__

#include "uvm_linux.h"
#include "uvm8_forward_decl.h"
#include "uvm8_hal_types.h"

// The powerpc kernel APIs to enable ATS were present prior to this callback
// change, but they were still in development. Various bug fixes were needed in
// the kernel and they all went in before this callback change. We can use the
// callback signature as a flag to indicate whether the kernel can support ATS
// in production.
#if defined(NV_PNV_NPU2_INIT_CONTEXT_CALLBACK_RETURNS_VOID)
    #define UVM_KERNEL_SUPPORTS_IBM_ATS() 1
#else
    #define UVM_KERNEL_SUPPORTS_IBM_ATS() 0
#endif

#if UVM_KERNEL_SUPPORTS_IBM_ATS()
    // Enables ATS access for the gpu_va_space on current->mm.
    //
    // If another VA space has already been associated with current->mm, or
    // another mm has already been associated with the VA space,
    // NV_ERR_NOT_SUPPORTED is returned.
    //
    // LOCKING: mmap_sem and the VA space lock must be held in exclusive mode.
    NV_STATUS uvm_ats_ibm_register_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space);

    // Disables ATS access for the gpu_va_space. Prior to calling this function,
    // the caller must guarantee that the GPU will no longer make any ATS
    // accesses in this GPU VA space, and that no ATS fault handling for this
    // GPU will be attempted.
    //
    // LOCKING: This function may block on mmap_sem and the VA space lock, so
    //          neither must be held.
    void uvm_ats_ibm_unregister_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space);

    // Request the kernel to handle a fault.
    //
    // LOCKING: mmap_sem must be held.
    NV_STATUS uvm_ats_ibm_service_fault(uvm_gpu_va_space_t *gpu_va_space,
                                        NvU64 fault_addr,
                                        uvm_fault_access_type_t access_type);

#else
    static NV_STATUS uvm_ats_ibm_register_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space)
    {
        return NV_OK;
    }

    static void uvm_ats_ibm_unregister_gpu_va_space(uvm_gpu_va_space_t *gpu_va_space)
    {

    }

    static NV_STATUS uvm_ats_ibm_service_fault(uvm_gpu_va_space_t *gpu_va_space,
                                               NvU64 fault_addr,
                                               uvm_fault_access_type_t access_type)
    {
        return NV_ERR_NOT_SUPPORTED;
    }
#endif // UVM_KERNEL_SUPPORTS_IBM_ATS

#endif // __UVM8_ATS_IBM_H__
