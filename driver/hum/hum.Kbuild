###########################################################################
# Kbuild fragment for hum.ko
###########################################################################

#
# Define HUM_{SOURCES,OBJECTS}
#

HUM_OBJECTS =

include $(src)/hum/hum-sources.Kbuild
HUM_OBJECTS += $(patsubst %.c,%.o,$(HUM_SOURCES))

obj-m += hum.o
hum-y := $(HUM_OBJECTS)

HUM_KO = hum/hum.ko

#
# Define hum.ko-specific CFLAGS.
#

HUM_CFLAGS+=-Wno-declaration-after-statement -Wno-unused-function -Wno-uninitialized -Wno-unused-result
HUM_CFLAGS+=-I$(src)/nvidia-common/inc -I$(src)/nvidia-uvm -I$(src)/hum \
        -DNVRM -DNV_UVM_ENABLE -DNV_BUILD_MODULE_INSTANCES=0
HUM_CFLAGS+=-DUSE_HOST_NODE_DEVICES

$(call ASSIGN_PER_OBJ_CFLAGS, $(HUM_OBJECTS), $(HUM_CFLAGS))

#
# Register the conftests needed by hum.ko
#

NV_OBJECTS_DEPEND_ON_CONFTEST += $(HUM_OBJECTS)

NV_CONFTEST_FUNCTION_COMPILE_TESTS += address_space_init_once
NV_CONFTEST_FUNCTION_COMPILE_TESTS += kbasename
NV_CONFTEST_FUNCTION_COMPILE_TESTS += vzalloc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += wait_on_bit_lock_argument_count
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pde_data
NV_CONFTEST_FUNCTION_COMPILE_TESTS += proc_remove
NV_CONFTEST_FUNCTION_COMPILE_TESTS += bitmap_clear
NV_CONFTEST_FUNCTION_COMPILE_TESTS += usleep_range
NV_CONFTEST_FUNCTION_COMPILE_TESTS += radix_tree_empty
NV_CONFTEST_FUNCTION_COMPILE_TESTS += radix_tree_replace_slot
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pnv_npu2_init_context
NV_CONFTEST_FUNCTION_COMPILE_TESTS += do_gettimeofday
NV_CONFTEST_FUNCTION_COMPILE_TESTS += kthread_create_on_node
NV_CONFTEST_FUNCTION_COMPILE_TESTS += vmf_insert_pfn
NV_CONFTEST_FUNCTION_COMPILE_TESTS += cpumask_of_node
NV_CONFTEST_FUNCTION_COMPILE_TESTS += list_is_first
NV_CONFTEST_FUNCTION_COMPILE_TESTS += timer_setup
NV_CONFTEST_FUNCTION_COMPILE_TESTS += acquire_console_sem
NV_CONFTEST_FUNCTION_COMPILE_TESTS += console_lock
NV_CONFTEST_FUNCTION_COMPILE_TESTS += pci_bus_address
NV_CONFTEST_FUNCTION_COMPILE_TESTS += set_memory_uc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += set_pages_uc
NV_CONFTEST_FUNCTION_COMPILE_TESTS += acpi_walk_namespace

NV_CONFTEST_TYPE_COMPILE_TESTS += outer_flush_all
NV_CONFTEST_TYPE_COMPILE_TESTS += file_operations
NV_CONFTEST_TYPE_COMPILE_TESTS += kuid_t
NV_CONFTEST_TYPE_COMPILE_TESTS += address_space
NV_CONFTEST_TYPE_COMPILE_TESTS += backing_dev_info
NV_CONFTEST_TYPE_COMPILE_TESTS += mm_context_t
NV_CONFTEST_TYPE_COMPILE_TESTS += get_user_pages_remote
NV_CONFTEST_TYPE_COMPILE_TESTS += get_user_pages
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_has_address
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_ops_fault_removed_vma_arg
NV_CONFTEST_TYPE_COMPILE_TESTS += node_states_n_memory
NV_CONFTEST_TYPE_COMPILE_TESTS += kmem_cache_has_kobj_remove_work
NV_CONFTEST_TYPE_COMPILE_TESTS += sysfs_slab_unlink
NV_CONFTEST_TYPE_COMPILE_TESTS += vm_fault_t
