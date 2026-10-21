# Maya's Metal build (Apple Silicon): maya-metal runs GLM-5.3-Flash through the pinned llama.cpp's ggml with its Metal
# backend.  It needs none of the CUDA/HIP engine and none of the x86 CPU kernels (AVX-512/AVX2, cpuid), so this build
# is its own short list of targets; the top-level CMakeLists.txt returns right after including it.
#
#   -DMAYA_ENABLE_METAL=ON            this build
#   -DMAYA_GGML_DIR=<llama.cpp>       a checkout at the pinned commit (default: fetched)
#   -DMAYA_METAL_TESTS=OFF            leave out the parity tests (the setup does)
#   -DMAYA_METAL_PORTABLE=ON          no -mcpu=native (a build for other Macs)
if(NOT APPLE)
  message(FATAL_ERROR "MAYA_ENABLE_METAL builds for macOS on Apple Silicon")
endif()
enable_language(C OBJC)

option(MAYA_METAL_TESTS "Build maya-metal's parity tests" ON)
option(MAYA_METAL_PORTABLE "Build maya-metal for any Apple Silicon Mac" OFF)
set(MAYA_GGML_DIR "" CACHE PATH "llama.cpp checkout to take ggml from (empty: fetch the pinned commit)")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL ON CACHE BOOL "" FORCE)
set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)   # the shaders live inside the binary: nothing to install
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(MAYA_METAL_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()

if(MAYA_GGML_DIR)
  set(_maya_llamacpp_dir ${MAYA_GGML_DIR})
else()
  include(FetchContent)
  FetchContent_Declare(maya_llamacpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG 3cf03257f219afbe7334045ff7c6a06ac68c627d
    GIT_SHALLOW FALSE)
  FetchContent_GetProperties(maya_llamacpp)
  if(NOT maya_llamacpp_POPULATED)
    FetchContent_Populate(maya_llamacpp)
  endif()
  set(_maya_llamacpp_dir ${maya_llamacpp_SOURCE_DIR})
endif()
add_subdirectory(${_maya_llamacpp_dir}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

# the engine reports Maya's release (VERSION) as its version
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/VERSION _maya_version LIMIT_COUNT 1)
add_library(maya_metal STATIC src/metal/glm_metal.cpp src/metal/glm_metal_load.cpp src/metal/expert_pool.cpp
                              src/metal/glm_metal_dflash.cpp src/metal/spec.cpp)
target_include_directories(maya_metal PUBLIC src/metal)
target_compile_definitions(maya_metal PUBLIC MAYA_VERSION="${_maya_version}")
target_link_libraries(maya_metal PUBLIC ggml ggml-base ggml-cpu ggml-metal)

add_executable(maya-metal src/metal/main.cpp src/metal/serve.cpp)
target_link_libraries(maya-metal PRIVATE maya_metal)

if(MAYA_METAL_TESTS)
  enable_testing()
  # the CUDA engine's GLM gates (src/core/glm_model_test.cpp, glm_model_big_test.cpp) on maya-metal
  add_executable(metal_model_test src/metal/metal_model_test.cpp)
  target_link_libraries(metal_model_test PRIVATE maya_metal)
  add_test(NAME metal_model_test COMMAND metal_model_test small)
  # the same gates on ggml-cpu (the oracle backend), and with the expert pool at its floor and small slices
  add_test(NAME metal_model_test_cpu COMMAND metal_model_test small)
  add_test(NAME metal_model_test_evict COMMAND metal_model_test small)
  set_tests_properties(metal_model_test_cpu PROPERTIES ENVIRONMENT MAYA_METAL_CPU=1)
  set_tests_properties(metal_model_test_evict PROPERTIES
                       ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1;MAYA_METAL_UBATCH=3;MAYA_METAL_ATTN_CHUNK=2")
  # verify windows (speculative decoding): partly wrong drafts must decode exactly as plain decoding does
  add_test(NAME metal_model_window COMMAND metal_model_test window data/glm5-synth.gguf 40)
  set_tests_properties(metal_model_test metal_model_test_cpu metal_model_test_evict metal_model_window PROPERTIES
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  # the 45-layer fixture: its 1.2 GB GGUF is generated into the build tree (tools/glm_big_fixture.py needs numpy)
  find_program(MAYA_PYTHON3 python3)
  if(MAYA_PYTHON3)
    add_test(NAME metal_model_big_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_big_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    set_tests_properties(metal_model_big_gen PROPERTIES FIXTURES_SETUP metal_big_gguf)
    add_test(NAME metal_model_big_test COMMAND metal_model_test big ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    add_test(NAME metal_model_big_test_evict
             COMMAND metal_model_test big ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf)
    set_tests_properties(metal_model_big_test_evict PROPERTIES
                         ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1;MAYA_METAL_UBATCH=64")
    add_test(NAME metal_model_big_window COMMAND metal_model_test window ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf 48)
    add_test(NAME metal_model_big_window_evict
             COMMAND metal_model_test window ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf 48)
    set_tests_properties(metal_model_big_window_evict PROPERTIES ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1")
    # MTP drafting (the NextN block, tools/glm_mtp_fixture.py adds one to the fixtures) decodes as plain decoding does
    add_test(NAME metal_mtp_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-small.gguf)
    add_test(NAME metal_mtp_big_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py --big
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-big.gguf)
    set_tests_properties(metal_mtp_gen metal_mtp_big_gen PROPERTIES FIXTURES_SETUP metal_mtp_gguf)
    add_test(NAME metal_model_mtp COMMAND metal_model_test mtp ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-small.gguf 40)
    add_test(NAME metal_model_big_mtp COMMAND metal_model_test mtp ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-big.gguf 48)
    add_test(NAME metal_model_big_mtp_evict COMMAND metal_model_test mtp ${CMAKE_CURRENT_BINARY_DIR}/glm5-mtp-big.gguf 48)
    set_tests_properties(metal_model_big_mtp_evict PROPERTIES ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1")
    # DFlash2 drafting (tools/glm_dflash_fixture.py: a drafter for the 45-layer fixture, its KV ring wrapping)
    add_test(NAME metal_dflash_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_dflash_fixture.py
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-dflash-drafter.gguf)
    set_tests_properties(metal_dflash_gen PROPERTIES FIXTURES_SETUP metal_mtp_gguf)
    add_test(NAME metal_model_big_dflash COMMAND metal_model_test dflash ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-fixture.gguf
                                                 ${CMAKE_CURRENT_BINARY_DIR}/glm5-dflash-drafter.gguf 120)
    # quantized experts (Q8_0): a 512-token prompt slice takes ggml-metal's matrix-matrix expert kernel (at most 1024
    # experts a tensor), all resident and with a small pool
    add_test(NAME metal_q8_gen
             COMMAND ${MAYA_PYTHON3} ${CMAKE_CURRENT_SOURCE_DIR}/tools/glm_mtp_fixture.py --big --no-nextn --q8-experts
                     --out ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-q8.gguf)
    set_tests_properties(metal_q8_gen PROPERTIES FIXTURES_SETUP metal_mtp_gguf)
    add_test(NAME metal_model_big_prefill COMMAND metal_model_test prefill ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-q8.gguf 512)
    add_test(NAME metal_model_big_prefill_evict
             COMMAND metal_model_test prefill ${CMAKE_CURRENT_BINARY_DIR}/glm5-big-q8.gguf 512)
    set_tests_properties(metal_model_big_prefill_evict PROPERTIES ENVIRONMENT "MAYA_METAL_POOL_SLOTS=1")
    set_tests_properties(metal_model_big_prefill metal_model_big_prefill_evict PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_mtp_gguf TIMEOUT 3600)
    set_tests_properties(metal_model_mtp metal_model_big_mtp metal_model_big_mtp_evict PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_mtp_gguf TIMEOUT 3600)
    set_tests_properties(metal_model_big_dflash PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                         FIXTURES_REQUIRED "metal_big_gguf;metal_mtp_gguf" TIMEOUT 3600)
    set_tests_properties(metal_model_big_test metal_model_big_test_evict metal_model_big_window metal_model_big_window_evict PROPERTIES
                         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} FIXTURES_REQUIRED metal_big_gguf TIMEOUT 3600)
  endif()
endif()
