; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i686-- -mattr=+lzcnt | FileCheck %s --check-prefixes=CHECK,X86
; RUN: llc < %s -mtriple=x86_64-linux-gnux32  -mattr=+lzcnt | FileCheck %s --check-prefixes=CHECK,X32
; RUN: llc < %s -mtriple=x86_64-- -mattr=+lzcnt | FileCheck %s --check-prefixes=CHECK,X64

declare i8 @llvm.ctlz.i8(i8, i1) nounwind readnone
declare i16 @llvm.ctlz.i16(i16, i1) nounwind readnone
declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone

define i8 @t1(i8 %x) nounwind  {
; X86-LABEL: t1:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    lzcntl %eax, %eax
; X86-NEXT:    addl $-24, %eax
; X86-NEXT:    # kill: def $al killed $al killed $eax
; X86-NEXT:    retl
;
; X32-LABEL: t1:
; X32:       # %bb.0:
; X32-NEXT:    movzbl %dil, %eax
; X32-NEXT:    lzcntl %eax, %eax
; X32-NEXT:    addl $-24, %eax
; X32-NEXT:    # kill: def $al killed $al killed $eax
; X32-NEXT:    retq
;
; X64-LABEL: t1:
; X64:       # %bb.0:
; X64-NEXT:    movzbl %dil, %eax
; X64-NEXT:    lzcntl %eax, %eax
; X64-NEXT:    addl $-24, %eax
; X64-NEXT:    # kill: def $al killed $al killed $eax
; X64-NEXT:    retq
	%tmp = tail call i8 @llvm.ctlz.i8( i8 %x, i1 false )
	ret i8 %tmp
}

define i16 @t2(i16 %x) nounwind  {
; X86-LABEL: t2:
; X86:       # %bb.0:
; X86-NEXT:    lzcntw {{[0-9]+}}(%esp), %ax
; X86-NEXT:    retl
;
; X32-LABEL: t2:
; X32:       # %bb.0:
; X32-NEXT:    lzcntw %di, %ax
; X32-NEXT:    retq
;
; X64-LABEL: t2:
; X64:       # %bb.0:
; X64-NEXT:    lzcntw %di, %ax
; X64-NEXT:    retq
	%tmp = tail call i16 @llvm.ctlz.i16( i16 %x, i1 false )
	ret i16 %tmp
}

define i32 @t3(i32 %x) nounwind  {
; X86-LABEL: t3:
; X86:       # %bb.0:
; X86-NEXT:    lzcntl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    retl
;
; X32-LABEL: t3:
; X32:       # %bb.0:
; X32-NEXT:    lzcntl %edi, %eax
; X32-NEXT:    retq
;
; X64-LABEL: t3:
; X64:       # %bb.0:
; X64-NEXT:    lzcntl %edi, %eax
; X64-NEXT:    retq
	%tmp = tail call i32 @llvm.ctlz.i32( i32 %x, i1 false )
	ret i32 %tmp
}

define i64 @t4(i64 %x) nounwind  {
; X86-LABEL: t4:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    testl %eax, %eax
; X86-NEXT:    jne .LBB3_1
; X86-NEXT:  # %bb.2:
; X86-NEXT:    lzcntl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    addl $32, %eax
; X86-NEXT:    xorl %edx, %edx
; X86-NEXT:    retl
; X86-NEXT:  .LBB3_1:
; X86-NEXT:    lzcntl %eax, %eax
; X86-NEXT:    xorl %edx, %edx
; X86-NEXT:    retl
;
; X32-LABEL: t4:
; X32:       # %bb.0:
; X32-NEXT:    lzcntq %rdi, %rax
; X32-NEXT:    retq
;
; X64-LABEL: t4:
; X64:       # %bb.0:
; X64-NEXT:    lzcntq %rdi, %rax
; X64-NEXT:    retq
	%tmp = tail call i64 @llvm.ctlz.i64( i64 %x, i1 false )
	ret i64 %tmp
}

define i8 @t5(i8 %x) nounwind  {
; X86-LABEL: t5:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    lzcntl %eax, %eax
; X86-NEXT:    addl $-24, %eax
; X86-NEXT:    # kill: def $al killed $al killed $eax
; X86-NEXT:    retl
;
; X32-LABEL: t5:
; X32:       # %bb.0:
; X32-NEXT:    movzbl %dil, %eax
; X32-NEXT:    lzcntl %eax, %eax
; X32-NEXT:    addl $-24, %eax
; X32-NEXT:    # kill: def $al killed $al killed $eax
; X32-NEXT:    retq
;
; X64-LABEL: t5:
; X64:       # %bb.0:
; X64-NEXT:    movzbl %dil, %eax
; X64-NEXT:    lzcntl %eax, %eax
; X64-NEXT:    addl $-24, %eax
; X64-NEXT:    # kill: def $al killed $al killed $eax
; X64-NEXT:    retq
	%tmp = tail call i8 @llvm.ctlz.i8( i8 %x, i1 true )
	ret i8 %tmp
}

define i16 @t6(i16 %x) nounwind  {
; X86-LABEL: t6:
; X86:       # %bb.0:
; X86-NEXT:    lzcntw {{[0-9]+}}(%esp), %ax
; X86-NEXT:    retl
;
; X32-LABEL: t6:
; X32:       # %bb.0:
; X32-NEXT:    lzcntw %di, %ax
; X32-NEXT:    retq
;
; X64-LABEL: t6:
; X64:       # %bb.0:
; X64-NEXT:    lzcntw %di, %ax
; X64-NEXT:    retq
	%tmp = tail call i16 @llvm.ctlz.i16( i16 %x, i1 true )
	ret i16 %tmp
}

define i32 @t7(i32 %x) nounwind  {
; X86-LABEL: t7:
; X86:       # %bb.0:
; X86-NEXT:    lzcntl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    retl
;
; X32-LABEL: t7:
; X32:       # %bb.0:
; X32-NEXT:    lzcntl %edi, %eax
; X32-NEXT:    retq
;
; X64-LABEL: t7:
; X64:       # %bb.0:
; X64-NEXT:    lzcntl %edi, %eax
; X64-NEXT:    retq
	%tmp = tail call i32 @llvm.ctlz.i32( i32 %x, i1 true )
	ret i32 %tmp
}

define i64 @t8(i64 %x) nounwind  {
; X86-LABEL: t8:
; X86:       # %bb.0:
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    testl %eax, %eax
; X86-NEXT:    jne .LBB7_1
; X86-NEXT:  # %bb.2:
; X86-NEXT:    lzcntl {{[0-9]+}}(%esp), %eax
; X86-NEXT:    addl $32, %eax
; X86-NEXT:    xorl %edx, %edx
; X86-NEXT:    retl
; X86-NEXT:  .LBB7_1:
; X86-NEXT:    lzcntl %eax, %eax
; X86-NEXT:    xorl %edx, %edx
; X86-NEXT:    retl
;
; X32-LABEL: t8:
; X32:       # %bb.0:
; X32-NEXT:    lzcntq %rdi, %rax
; X32-NEXT:    retq
;
; X64-LABEL: t8:
; X64:       # %bb.0:
; X64-NEXT:    lzcntq %rdi, %rax
; X64-NEXT:    retq
	%tmp = tail call i64 @llvm.ctlz.i64( i64 %x, i1 true )
	ret i64 %tmp
}
