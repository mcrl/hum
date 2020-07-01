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

#ifndef __HUM_H__
#define __HUM_H__

#include <stdint.h>
#include "parameters.h"
#include <stddef.h>

#include "cuda_func.h"

class HUMComm;

extern void hum_run();

extern int grank_;
extern int gsize_;
extern HUMComm* g_HUMComm;
#ifdef USE_MEM_PREFETCH
extern int driver_fd_;
#endif

#if defined(_WIN32)
#define HUM_API_ENTRY
#define HUM_API_CALL	__stdcall
#define HUM_CALLBACK	__stdcall
#else
#define HUM_API_ENTRY
#define HUM_API_CALL
#define HUM_CALLBACK
#endif

template<class c_obj_type>
struct _hum_handle_t {
	void* dummy;
	c_obj_type* c_obj;
};

struct hum_dim3_t { 
	unsigned int x,y,z; 
};

class HUMPlatform;
class HUMDevice;
class HUMContext;
class HUMCommandQueue;
class HUMMem;
class HUMKernel;
class HUMEvent;

typedef struct _hum_handle_t<HUMPlatform>*			hum_platform_handle;
typedef struct _hum_handle_t<HUMDevice>*			  hum_device_handle;
typedef struct _hum_handle_t<HUMContext>*				hum_context_handle;
typedef struct _hum_handle_t<HUMCommandQueue>*	hum_command_queue_handle;
typedef struct _hum_handle_t<HUMMem>*				    hum_mem_handle;
typedef struct _hum_handle_t<HUMKernel>*		    hum_kernel_handle;
typedef struct _hum_handle_t<HUMEvent>*			    hum_event_handle;
typedef struct _hum_handle_t<void>*					    hum_sampler_handle;


typedef uint32_t						hum_uint;
typedef int32_t							hum_int;
typedef uint64_t						hum_ulong;
typedef int64_t							hum_long;

typedef hum_uint						hum_command_type;
typedef hum_uint 					hum_command_status;

typedef hum_uint             hum_bool;                     
typedef hum_ulong            hum_bitfield;
typedef hum_bitfield         hum_device_type;
typedef hum_uint             hum_platform_info;
typedef hum_uint             hum_device_info;
typedef hum_bitfield         hum_device_fp_config;
typedef hum_uint             hum_device_mem_cache_type;
typedef hum_uint             hum_device_local_mem_type;
typedef hum_bitfield         hum_device_exec_capabilities;
typedef hum_bitfield         hum_command_queue_properties;
typedef intptr_t            hum_device_partition_property;
typedef hum_bitfield         hum_device_affinity_domain;

typedef intptr_t            hum_context_properties;
typedef hum_uint             hum_context_info;
typedef hum_uint             hum_command_queue_info;
typedef hum_uint             hum_channel_order;
typedef hum_uint             hum_channel_type;
typedef hum_bitfield         hum_mem_flags;
typedef hum_uint             hum_mem_object_type;
typedef hum_uint             hum_mem_info;
typedef hum_bitfield         hum_mem_migration_flags;
typedef hum_uint             hum_image_info;
typedef hum_uint             hum_buffer_create_type;
typedef hum_uint             hum_addressing_mode;
typedef hum_uint             hum_filter_mode;
typedef hum_uint             hum_sampler_info;
typedef hum_bitfield         hum_map_flags;
typedef hum_uint             hum_program_info;
typedef hum_uint             hum_program_build_info;
typedef hum_uint             hum_program_binary_type;
typedef hum_int              hum_build_status;
typedef hum_uint             hum_kernel_info;
typedef hum_uint             hum_kernel_arg_info;
typedef hum_uint             hum_kernel_arg_address_qualifier;
typedef hum_uint             hum_kernel_arg_access_qualifier;
typedef hum_bitfield         hum_kernel_arg_type_qualifier;
typedef hum_uint             hum_kernel_work_group_info;
typedef hum_uint             hum_event_info;
typedef hum_uint             hum_command_type;
typedef hum_uint             hum_profiling_info;

typedef struct _hum_image_format {
    hum_channel_order        image_channel_order;
    hum_channel_type         image_channel_data_type;
} hum_image_format;

typedef struct _hum_image_desc {
    hum_mem_object_type      image_type;
    size_t                  image_width;
    size_t                  image_height;
    size_t                  image_depth;
    size_t                  image_array_size;
    size_t                  image_row_pitch;
    size_t                  image_slice_pitch;
    hum_uint               num_mip_levels;
    hum_uint               num_samples;
    hum_mem_handle         buffer;
} hum_image_desc;

typedef struct _hum_buffer_region {
    size_t                  origin;
    size_t                  size;
} hum_buffer_region;





/* Error Codes */
#define HUM_SUCCESS                                  0
#define HUM_DEVICE_NOT_FOUND                         -1
#define HUM_DEVICE_NOT_AVAILABLE                     -2
#define HUM_COMPILER_NOT_AVAILABLE                   -3
#define HUM_MEM_OBJECT_ALLOCATION_FAILURE            -4
#define HUM_OUT_OF_RESOURCES                         -5
#define HUM_OUT_OF_HOST_MEMORY                       -6
#define HUM_PROFILING_INFO_NOT_AVAILABLE             -7
#define HUM_MEM_COPY_OVERLAP                         -8
#define HUM_IMAGE_FORMAT_MISMATCH                    -9
#define HUM_IMAGE_FORMAT_NOT_SUPPORTED               -10
#define HUM_BUILD_PROGRAM_FAILURE                    -11
#define HUM_MAP_FAILURE                              -12
#define HUM_MISALIGNED_SUB_BUFFER_OFFSET             -13
#define HUM_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST -14
#define HUM_COMPILE_PROGRAM_FAILURE                  -15
#define HUM_LINKER_NOT_AVAILABLE                     -16
#define HUM_LINK_PROGRAM_FAILURE                     -17
#define HUM_DEVICE_PARTITION_FAILED                  -18
#define HUM_KERNEL_ARG_INFO_NOT_AVAILABLE            -19

#define HUM_INVALID_VALUE                            -30
#define HUM_INVALID_DEVICE_TYPE                      -31
#define HUM_INVALID_PLATFORM                         -32
#define HUM_INVALID_DEVICE                           -33
#define HUM_INVALID_CONTEXT                          -34
#define HUM_INVALID_QUEUE_PROPERTIES                 -35
#define HUM_INVALID_COMMAND_QUEUE                    -36
#define HUM_INVALID_HOST_PTR                         -37
#define HUM_INVALID_MEM_OBJECT                       -38
#define HUM_INVALID_IMAGE_FORMAT_DESCRIPTOR          -39
#define HUM_INVALID_IMAGE_SIZE                       -40
#define HUM_INVALID_SAMPLER                          -41
#define HUM_INVALID_BINARY                           -42
#define HUM_INVALID_BUILD_OPTIONS                    -43
#define HUM_INVALID_PROGRAM                          -44
#define HUM_INVALID_PROGRAM_EXECUTABLE               -45
#define HUM_INVALID_KERNEL_NAME                      -46
#define HUM_INVALID_KERNEL_DEFINITION                -47
#define HUM_INVALID_KERNEL                           -48
#define HUM_INVALID_ARG_INDEX                        -49
#define HUM_INVALID_ARG_VALUE                        -50
#define HUM_INVALID_ARG_SIZE                         -51
#define HUM_INVALID_KERNEL_ARGS                      -52
#define HUM_INVALID_WORK_DIMENSION                   -53
#define HUM_INVALID_WORK_GROUP_SIZE                  -54
#define HUM_INVALID_WORK_ITEM_SIZE                   -55
#define HUM_INVALID_GLOBAL_OFFSET                    -56
#define HUM_INVALID_EVENT_WAIT_LIST                  -57
#define HUM_INVALID_EVENT                            -58
#define HUM_INVALID_OPERATION                        -59
#define HUM_INVALID_GL_OBJECT                        -60
#define HUM_INVALID_BUFFER_SIZE                      -61
#define HUM_INVALID_MIP_LEVEL                        -62
#define HUM_INVALID_GLOBAL_WORK_SIZE                 -63
#define HUM_INVALID_PROPERTY                         -64
#define HUM_INVALID_IMAGE_DESCRIPTOR                 -65
#define HUM_INVALID_COMPILER_OPTIONS                 -66
#define HUM_INVALID_LINKER_OPTIONS                   -67
#define HUM_INVALID_DEVICE_PARTITION_COUNT           -68
#define HUM_INVALID_PIPE_SIZE                        -69
#define HUM_INVALID_DEVICE_QUEUE                     -70

/* OpenCL Version */
#define HUM_CL_VERSION_1_0                              1
#define HUM_CL_VERSION_1_1                              1
#define HUM_CL_VERSION_1_2                              1
#define HUM_CL_VERSION_2_0                              1

/* hum_bool */
#define HUM_FALSE                                    0
#define HUM_TRUE                                     1
#define HUM_BLOCKING                                 HUM_TRUE
#define HUM_NON_BLOCKING                             HUM_FALSE

/* cl_platform_info */
#define HUM_PLATFORM_PROFILE                         0x0900
#define HUM_PLATFORM_VERSION                         0x0901
#define HUM_PLATFORM_NAME                            0x0902
#define HUM_PLATFORM_VENDOR                          0x0903
#define HUM_PLATFORM_EXTENSIONS                      0x0904


/* cl_device_type - bitfield */
#define HUM_DEVICE_TYPE_DEFAULT                      (1 << 0)
#define HUM_DEVICE_TYPE_CPU                          (1 << 1)
#define HUM_DEVICE_TYPE_GPU                          (1 << 2)
#define HUM_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#define HUM_DEVICE_TYPE_CUSTOM                       (1 << 4)
#define HUM_DEVICE_TYPE_ALL                          0xFFFFFFFF

/* cl_device_info */
#define HUM_DEVICE_TYPE                                  0x1000
#define HUM_DEVICE_VENDOR_ID                             0x1001
#define HUM_DEVICE_MAX_COMPUTE_UNITS                     0x1002
#define HUM_DEVICE_MAX_WORK_ITEM_DIMENSIONS              0x1003
#define HUM_DEVICE_MAX_WORK_GROUP_SIZE                   0x1004
#define HUM_DEVICE_MAX_WORK_ITEM_SIZES                   0x1005
#define HUM_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR           0x1006
#define HUM_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT          0x1007
#define HUM_DEVICE_PREFERRED_VECTOR_WIDTH_INT            0x1008
#define HUM_DEVICE_PREFERRED_VECTOR_WIDTH_LONG           0x1009
#define HUM_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT          0x100A
#define HUM_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE         0x100B
#define HUM_DEVICE_MAX_CLOCK_FREQUENCY                   0x100C
#define HUM_DEVICE_ADDRESS_BITS                          0x100D
#define HUM_DEVICE_MAX_READ_IMAGE_ARGS                   0x100E
#define HUM_DEVICE_MAX_WRITE_IMAGE_ARGS                  0x100F
#define HUM_DEVICE_MAX_MEM_ALLOC_SIZE                    0x1010
#define HUM_DEVICE_IMAGE2D_MAX_WIDTH                     0x1011
#define HUM_DEVICE_IMAGE2D_MAX_HEIGHT                    0x1012
#define HUM_DEVICE_IMAGE3D_MAX_WIDTH                     0x1013
#define HUM_DEVICE_IMAGE3D_MAX_HEIGHT                    0x1014
#define HUM_DEVICE_IMAGE3D_MAX_DEPTH                     0x1015
#define HUM_DEVICE_IMAGE_SUPPORT                         0x1016
#define HUM_DEVICE_MAX_PARAMETER_SIZE                    0x1017
#define HUM_DEVICE_MAX_SAMPLERS                          0x1018
#define HUM_DEVICE_MEM_BASE_ADDR_ALIGN                   0x1019
#define HUM_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE              0x101A
#define HUM_DEVICE_SINGLE_FP_CONFIG                      0x101B
#define HUM_DEVICE_GLOBAL_MEM_CACHE_TYPE                 0x101C
#define HUM_DEVICE_GLOBAL_MEM_CACHELINE_SIZE             0x101D
#define HUM_DEVICE_GLOBAL_MEM_CACHE_SIZE                 0x101E
#define HUM_DEVICE_GLOBAL_MEM_SIZE                       0x101F
#define HUM_DEVICE_MAX_CONSTANT_BUFFER_SIZE              0x1020
#define HUM_DEVICE_MAX_CONSTANT_ARGS                     0x1021
#define HUM_DEVICE_LOCAL_MEM_TYPE                        0x1022
#define HUM_DEVICE_LOCAL_MEM_SIZE                        0x1023
#define HUM_DEVICE_ERROR_CORRECTION_SUPPORT              0x1024
#define HUM_DEVICE_PROFILING_TIMER_RESOLUTION            0x1025
#define HUM_DEVICE_ENDIAN_LITTLE                         0x1026
#define HUM_DEVICE_AVAILABLE                             0x1027
#define HUM_DEVICE_COMPILER_AVAILABLE                    0x1028
#define HUM_DEVICE_EXECUTION_CAPABILITIES                0x1029
#define HUM_DEVICE_QUEUE_PROPERTIES                      0x102A    /* deprecated */
#define HUM_DEVICE_QUEUE_ON_HOST_PROPERTIES              0x102A
#define HUM_DEVICE_NAME                                  0x102B
#define HUM_DEVICE_VENDOR                                0x102C
#define HUM_DRIVER_VERSION                               0x102D
#define HUM_DEVICE_PROFILE                               0x102E
#define HUM_DEVICE_VERSION                               0x102F
#define HUM_DEVICE_EXTENSIONS                            0x1030
#define HUM_DEVICE_PLATFORM                              0x1031
#define HUM_DEVICE_DOUBLE_FP_CONFIG                      0x1032
/* 0x1033 reserved for HUM_DEVICE_HALF_FP_CONFIG */
#define HUM_DEVICE_PREFERRED_VECTOR_WIDTH_HALF           0x1034
#define HUM_DEVICE_HOST_UNIFIED_MEMORY                   0x1035   /* deprecated */
#define HUM_DEVICE_NATIVE_VECTOR_WIDTH_CHAR              0x1036
#define HUM_DEVICE_NATIVE_VECTOR_WIDTH_SHORT             0x1037
#define HUM_DEVICE_NATIVE_VECTOR_WIDTH_INT               0x1038
#define HUM_DEVICE_NATIVE_VECTOR_WIDTH_LONG              0x1039
#define HUM_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT             0x103A
#define HUM_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE            0x103B
#define HUM_DEVICE_NATIVE_VECTOR_WIDTH_HALF              0x103C
#define HUM_DEVICE_OPENHUM_C_VERSION                      0x103D
#define HUM_DEVICE_LINKER_AVAILABLE                      0x103E
#define HUM_DEVICE_BUILT_IN_KERNELS                      0x103F
#define HUM_DEVICE_IMAGE_MAX_BUFFER_SIZE                 0x1040
#define HUM_DEVICE_IMAGE_MAX_ARRAY_SIZE                  0x1041
#define HUM_DEVICE_PARENT_DEVICE                         0x1042
#define HUM_DEVICE_PARTITION_MAX_SUB_DEVICES             0x1043
#define HUM_DEVICE_PARTITION_PROPERTIES                  0x1044
#define HUM_DEVICE_PARTITION_AFFINITY_DOMAIN             0x1045
#define HUM_DEVICE_PARTITION_TYPE                        0x1046
#define HUM_DEVICE_REFERENCE_COUNT                       0x1047
#define HUM_DEVICE_PREFERRED_INTEROP_USER_SYNC           0x1048
#define HUM_DEVICE_PRINTF_BUFFER_SIZE                    0x1049
#define HUM_DEVICE_IMAGE_PITCH_ALIGNMENT                 0x104A
#define HUM_DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT          0x104B
#define HUM_DEVICE_MAX_READ_WRITE_IMAGE_ARGS             0x104C
#define HUM_DEVICE_MAX_GLOBAL_VARIABLE_SIZE              0x104D
#define HUM_DEVICE_QUEUE_ON_DEVICE_PROPERTIES            0x104E
#define HUM_DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE        0x104F
#define HUM_DEVICE_QUEUE_ON_DEVICE_MAX_SIZE              0x1050
#define HUM_DEVICE_MAX_ON_DEVICE_QUEUES                  0x1051
#define HUM_DEVICE_MAX_ON_DEVICE_EVENTS                  0x1052
#define HUM_DEVICE_SVM_CAPABILITIES                      0x1053
#define HUM_DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE  0x1054
#define HUM_DEVICE_MAX_PIPE_ARGS                         0x1055
#define HUM_DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS          0x1056
#define HUM_DEVICE_PIPE_MAX_PACKET_SIZE                  0x1057
#define HUM_DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT   0x1058
#define HUM_DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT     0x1059
#define HUM_DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT      0x105A

/* cl_device_local_mem_type */
#define HUM_LOCAL                                    0x1
#define HUM_GLOBAL                                   0x2

/* cl_device_exec_capabilities - bitfield */
#define HUM_EXEC_KERNEL                              (1 << 0)
#define HUM_EXEC_NATIVE_KERNEL                       (1 << 1)

/* cl_command_queue_properties - bitfield */
#define HUM_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      (1 << 0)
#define HUM_QUEUE_PROFILING_ENABLE                   (1 << 1)
#define HUM_QUEUE_ON_DEVICE                          (1 << 2)
#define HUM_QUEUE_ON_DEVICE_DEFAULT                  (1 << 3)
#define HUM_QUEUE_CUDA_STREAM					              (1 << 4)
#define HUM_QUEUE_CUDA_DEFAULT					              (1 << 5)
#define HUM_QUEUE_CUDA_BLOCKING				              (1 << 6)

/* cl_context_info  */
#define HUM_CONTEXT_REFERENCE_COUNT                  0x1080
#define HUM_CONTEXT_DEVICES                          0x1081
#define HUM_CONTEXT_PROPERTIES                       0x1082
#define HUM_CONTEXT_NUM_DEVICES                      0x1083

/* cl_context_properties */
#define HUM_CONTEXT_PLATFORM                         0x1084
#define HUM_CONTEXT_INTEROP_USER_SYNC                0x1085
 

/* cl_command_queue_info */
#define HUM_QUEUE_CONTEXT                            0x1090
#define HUM_QUEUE_DEVICE                             0x1091
#define HUM_QUEUE_REFERENCE_COUNT                    0x1092
#define HUM_QUEUE_PROPERTIES                         0x1093
#define HUM_QUEUE_SIZE                               0x1094


/* cl_mem_flags and cl_svm_mem_flags - bitfield */
#define HUM_MEM_READ_WRITE                           (1 << 0)
#define HUM_MEM_WRITE_ONLY                           (1 << 1)
#define HUM_MEM_READ_ONLY                            (1 << 2)
#define HUM_MEM_USE_HOST_PTR                         (1 << 3)
#define HUM_MEM_ALLOC_HOST_PTR                       (1 << 4)
#define HUM_MEM_COPY_HOST_PTR                        (1 << 5)
/* reserved                                         (1 << 6)    */
#define HUM_MEM_HOST_WRITE_ONLY                      (1 << 7)
#define HUM_MEM_HOST_READ_ONLY                       (1 << 8)
#define HUM_MEM_HOST_NO_ACCESS                       (1 << 9)
#define HUM_MEM_SVM_FINE_GRAIN_BUFFER                (1 << 10)   /* used by cl_svm_mem_flags only */
#define HUM_MEM_SVM_ATOMICS                          (1 << 11)   /* used by cl_svm_mem_flags only */
#define HUM_MEM_KERNEL_READ_AND_WRITE                (1 << 12)


/* cl_mem_object_type */
#define HUM_MEM_OBJECT_BUFFER                        0x10F0
#define HUM_MEM_OBJECT_IMAGE2D                       0x10F1
#define HUM_MEM_OBJECT_IMAGE3D                       0x10F2
#define HUM_MEM_OBJECT_IMAGE2D_ARRAY                 0x10F3
#define HUM_MEM_OBJECT_IMAGE1D                       0x10F4
#define HUM_MEM_OBJECT_IMAGE1D_ARRAY                 0x10F5
#define HUM_MEM_OBJECT_IMAGE1D_BUFFER                0x10F6
#define HUM_MEM_OBJECT_PIPE                          0x10F7

#define HUM_MEM_OBJECT_CUDA_SYMBOL                        0x10F8

/* cl_mem_info */
#define HUM_MEM_TYPE                                 0x1100
#define HUM_MEM_FLAGS                                0x1101
#define HUM_MEM_SIZE                                 0x1102
#define HUM_MEM_HOST_PTR                             0x1103
#define HUM_MEM_MAP_COUNT                            0x1104
#define HUM_MEM_REFERENCE_COUNT                      0x1105
#define HUM_MEM_CONTEXT                              0x1106
#define HUM_MEM_ASSOCIATED_MEMOBJECT                 0x1107
#define HUM_MEM_OFFSET                               0x1108
#define HUM_MEM_USES_SVM_POINTER                     0x1109

/* cl_program_info */
#define HUM_PROGRAM_REFERENCE_COUNT                  0x1160
#define HUM_PROGRAM_CONTEXT                          0x1161
#define HUM_PROGRAM_NUM_DEVICES                      0x1162
#define HUM_PROGRAM_DEVICES                          0x1163
#define HUM_PROGRAM_SOURCE                           0x1164
#define HUM_PROGRAM_BINARY_SIZES                     0x1165
#define HUM_PROGRAM_BINARIES                         0x1166
#define HUM_PROGRAM_NUM_KERNELS                      0x1167
#define HUM_PROGRAM_KERNEL_NAMES                     0x1168

/* cl_program_build_info */
#define HUM_PROGRAM_BUILD_STATUS                     0x1181
#define HUM_PROGRAM_BUILD_OPTIONS                    0x1182
#define HUM_PROGRAM_BUILD_LOG                        0x1183
#define HUM_PROGRAM_BINARY_TYPE                      0x1184
#define HUM_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE 0x1185
    
/* cl_program_binary_type */
#define HUM_PROGRAM_BINARY_TYPE_NONE                 0x0
#define HUM_PROGRAM_BINARY_TYPE_COMPILED_OBJECT      0x1
#define HUM_PROGRAM_BINARY_TYPE_LIBRARY              0x2
#define HUM_PROGRAM_BINARY_TYPE_EXECUTABLE           0x4
#define HUM_PROGRAM_BINARY_TYPE_CUDA				          0x8

/* cl_build_status */
#define HUM_BUILD_SUCCESS                            0
#define HUM_BUILD_NONE                               -1
#define HUM_BUILD_ERROR                              -2
#define HUM_BUILD_IN_PROGRESS                        -3


/* cl_kernel_info */
#define HUM_KERNEL_FUNCTION_NAME                     0x1190
#define HUM_KERNEL_NUM_ARGS                          0x1191
#define HUM_KERNEL_REFERENCE_COUNT                   0x1192
#define HUM_KERNEL_CONTEXT                           0x1193
#define HUM_KERNEL_PROGRAM                           0x1194
#define HUM_KERNEL_ATTRIBUTES                        0x1195


/* cl_kernel_arg_info */
#define HUM_KERNEL_ARG_ADDRESS_QUALIFIER             0x1196
#define HUM_KERNEL_ARG_ACCESS_QUALIFIER              0x1197
#define HUM_KERNEL_ARG_TYPE_NAME                     0x1198
#define HUM_KERNEL_ARG_TYPE_QUALIFIER                0x1199
#define HUM_KERNEL_ARG_NAME                          0x119A

/* cl_kernel_arg_address_qualifier */
#define HUM_KERNEL_ARG_ADDRESS_GLOBAL                0x119B
#define HUM_KERNEL_ARG_ADDRESS_LOCAL                 0x119C
#define HUM_KERNEL_ARG_ADDRESS_CONSTANT              0x119D
#define HUM_KERNEL_ARG_ADDRESS_PRIVATE               0x119E

/* cl_kernel_arg_access_qualifier */
#define HUM_KERNEL_ARG_ACCESS_READ_ONLY              0x11A0
#define HUM_KERNEL_ARG_ACCESS_WRITE_ONLY             0x11A1
#define HUM_KERNEL_ARG_ACCESS_READ_WRITE             0x11A2
#define HUM_KERNEL_ARG_ACCESS_NONE                   0x11A3
    
/* cl_kernel_arg_type_qualifer */
#define HUM_KERNEL_ARG_TYPE_NONE                     0
#define HUM_KERNEL_ARG_TYPE_CONST                    (1 << 0)
#define HUM_KERNEL_ARG_TYPE_RESTRICT                 (1 << 1)
#define HUM_KERNEL_ARG_TYPE_VOLATILE                 (1 << 2)


/* cl_kernel_work_group_info */
#define HUM_KERNEL_WORK_GROUP_SIZE                   0x11B0
#define HUM_KERNEL_COMPILE_WORK_GROUP_SIZE           0x11B1
#define HUM_KERNEL_LOCAL_MEM_SIZE                    0x11B2
#define HUM_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x11B3
#define HUM_KERNEL_PRIVATE_MEM_SIZE                  0x11B4
#define HUM_KERNEL_GLOBAL_WORK_SIZE                  0x11B5

/* cl_event_info  */
#define HUM_EVENT_COMMAND_QUEUE                      0x11D0
#define HUM_EVENT_COMMAND_TYPE                       0x11D1
#define HUM_EVENT_REFERENCE_COUNT                    0x11D2
#define HUM_EVENT_COMMAND_EXECUTION_STATUS           0x11D3
#define HUM_EVENT_CONTEXT                            0x11D4


/* hum_command_type */

// Kernel
#define HUM_COMMAND_NDRANGE_KERNEL                   0x11F0
#define HUM_COMMAND_TASK                             0x11F1
#define HUM_COMMAND_NATIVE_KERNEL                    0x11F2

// Memory
#define HUM_COMMAND_READ_BUFFER                      0x11F3
#define HUM_COMMAND_WRITE_BUFFER                     0x11F4
#define HUM_COMMAND_COPY_BUFFER                      0x11F5
#define HUM_COMMAND_READ_IMAGE                       0x11F6
#define HUM_COMMAND_WRITE_IMAGE                      0x11F7
#define HUM_COMMAND_COPY_IMAGE                       0x11F8
#define HUM_COMMAND_COPY_IMAGE_TO_BUFFER             0x11F9
#define HUM_COMMAND_COPY_BUFFER_TO_IMAGE             0x11FA
#define HUM_COMMAND_MAP_BUFFER                       0x11FB
#define HUM_COMMAND_MAP_IMAGE                        0x11FC
#define HUM_COMMAND_UNMAP_MEM_OBJECT                 0x11FD
#define HUM_COMMAND_MARKER                           0x11FE
#define HUM_COMMAND_ACQUIRE_GL_OBJECTS               0x11FF
#define HUM_COMMAND_RELEASE_GL_OBJECTS               0x1200
#define HUM_COMMAND_READ_BUFFER_RECT                 0x1201
#define HUM_COMMAND_WRITE_BUFFER_RECT                0x1202
#define HUM_COMMAND_COPY_BUFFER_RECT                 0x1203

//ETC (TODO)
#define HUM_COMMAND_USER                             0x1204
#define HUM_COMMAND_BARRIER                          0x1205
#define HUM_COMMAND_MIGRATE_MEM_OBJECTS              0x1206
#define HUM_COMMAND_FILL_BUFFER                      0x1207
#define HUM_COMMAND_FILL_IMAGE                       0x1208
#define HUM_COMMAND_SVM_FREE                         0x1209
#define HUM_COMMAND_SVM_MEMCPY                       0x120A
#define HUM_COMMAND_SVM_MEMFILL                      0x120B
#define HUM_COMMAND_SVM_MAP                          0x120C
#define HUM_COMMAND_SVM_UNMAP                        0x120D
#define HUM_COMMAND_BROADCAST_BUFFER                 0x120E
#define HUM_COMMAND_COPY_BROADCAST_BUFFER            0x120F

//HUM INTERNAL
#define HUM_COMMAND_BUILD_PROGRAM                    0x1210
#define HUM_COMMAND_COMPILE_PROGRAM                  0x1211
#define HUM_COMMAND_LINK_PROGRAM                     0x1212
#define HUM_COMMAND_WAIT_FOR_EVENTS                  0x1213
#define HUM_COMMAND_CUSTOM                           0x1214
#define HUM_COMMAND_NOP                              0x1215

//CUDA COMMAND
#define HUM_COMMAND_CUDA_KERNEL                      0x1216
#define HUM_COMMAND_WRITE_BUFFER_TO_SYMBOL           0x1217
#define HUM_COMMAND_READ_BUFFER_FROM_SYMBOL          0x1218

//Driver COMMAND
#define HUM_COMMAND_DRIVER                           0x1219
#define HUM_COMMAND_CUDA_DIRECT_KERNEL               0x1220

/* command execution status */
#define HUM_COMPLETE                                 0x0
#define HUM_RUNNING                                  0x1
#define HUM_SUBMITTED                                0x2
#define HUM_QUEUED                                   0x3



/* cl_profiling_info  */
#define HUM_PROFILING_COMMAND_QUEUED                 0x2280
#define HUM_PROFILING_COMMAND_SUBMIT                 0x2281
#define HUM_PROFILING_COMMAND_START                  0x2282
#define HUM_PROFILING_COMMAND_END                    0x2283
#define HUM_PROFILING_COMMAND_COMPLETE               0x2284

#define HUM_COMMAND_READ_LOCAL_GPU_BUFFER										0x3300
#define HUM_COMMAND_WRITE_LOCAL_GPU_BUFFER										0x3301



#endif //__HUM_H__
