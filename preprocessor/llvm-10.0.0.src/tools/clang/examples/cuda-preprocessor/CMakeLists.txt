set(LLVM_LINK_COMPONENTS
  Core
  ExecutionEngine
  MC
  MCJIT
  Support
  native
  )

add_clang_executable(cuda-preprocessor
  main.cpp
  )

add_dependencies(cuda-preprocessor
  clang-headers
  )

target_link_libraries(cuda-preprocessor
  PUBLIC
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangRewrite
  clangTooling
  )
