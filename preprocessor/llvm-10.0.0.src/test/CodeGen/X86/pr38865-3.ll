; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -show-mc-encoding < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnux32"

define void @foo(i8* %x) optsize {
; CHECK-LABEL: foo:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl $707406378, %eax # encoding: [0xb8,0x2a,0x2a,0x2a,0x2a]
; CHECK-NEXT:    # imm = 0x2A2A2A2A
; CHECK-NEXT:    movl $32, %ecx # encoding: [0xb9,0x20,0x00,0x00,0x00]
; CHECK-NEXT:    rep;stosl %eax, %es:(%edi) # encoding: [0xf3,0x67,0xab]
; CHECK-NEXT:    retq # encoding: [0xc3]
  call void @llvm.memset.p0i8.i32(i8* align 4 %x, i8 42, i32 128, i1 false)
  ret void
}
declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)
