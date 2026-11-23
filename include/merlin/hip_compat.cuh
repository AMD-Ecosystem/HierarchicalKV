/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Lets the CUDA sources of this library build with HIP for AMD GPUs. On CUDA
// this header only defines the lane-mask helpers, as plain wrappers around
// the CUDA intrinsics they replace.

#if defined(__HIP__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#define MERLIN_HIP 1
#endif

#ifdef MERLIN_HIP

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <type_traits>

// HIP's __shfl takes only arithmetic types, while CUDA's thread_block_tile
// shfl takes any type. These overloads cover the pointers and enums the tile
// shuffles here pass; they are declared before the cooperative-groups headers
// so that the tile shfl template can find them.
template <typename T>
__device__ inline T* __shfl(T* var, int src_lane, int width) {
  return reinterpret_cast<T*>(
      __shfl(reinterpret_cast<unsigned long long>(var), src_lane, width));
}

template <typename T,
          typename std::enable_if<std::is_enum<T>::value, int>::type = 0>
__device__ inline T __shfl(T var, int src_lane, int width) {
  using U = typename std::underlying_type<T>::type;
  return static_cast<T>(__shfl(static_cast<U>(var), src_lane, width));
}

#include <hip/cooperative_groups/hip_reduce.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdint>
#include <hipcub/device/device_radix_sort.hpp>
#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_scan.hpp>

namespace cub = hipcub;

#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaError hipError_t
#define cudaError_t hipError_t
#define cudaErrorInvalidResourceHandle hipErrorInvalidResourceHandle
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeAsync hipFreeAsync
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaMalloc hipMalloc
#define cudaMallocAsync hipMallocAsync
#define cudaMallocHost hipHostMalloc
#define cudaMallocManaged hipMallocManaged
#define cudaMemAttachGlobal hipMemAttachGlobal
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyFromSymbol(dst, symbol, ...) \
  hipMemcpyFromSymbol(dst, HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyToSymbol(symbol, ...) \
  hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaMemoryTypeHost hipMemoryTypeHost
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#endif  // MERLIN_HIP

namespace nv {
namespace merlin {

// Lane masks of warp votes. A HIP wavefront has 32 or 64 lanes, so HIP masks
// are 64 bits wide.
#ifdef MERLIN_HIP
using warp_mask_t = unsigned long long;
constexpr warp_mask_t kFullWarpMask = ~0ull;
#else
using warp_mask_t = unsigned int;
constexpr warp_mask_t kFullWarpMask = 0xffffffffu;
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)

// Warp votes and barriers whose mask is always the set of lanes that reach
// the call. HIP checks such masks against the lanes actually present, so it
// uses the maskless forms, which act on exactly those lanes.
__forceinline__ __device__ warp_mask_t warp_ballot(warp_mask_t mask,
                                                   bool predicate) {
#ifdef MERLIN_HIP
  return __ballot(predicate);
#else
  return __ballot_sync(mask, predicate);
#endif
}

__forceinline__ __device__ bool warp_any(warp_mask_t mask, bool predicate) {
#ifdef MERLIN_HIP
  return __any(predicate);
#else
  return __any_sync(mask, predicate);
#endif
}

__forceinline__ __device__ void warp_sync(warp_mask_t mask) {
#ifdef MERLIN_HIP
  __syncwarp();
#else
  __syncwarp(mask);
#endif
}

#ifdef MERLIN_HIP

// Nanosecond device clock. The HIP wall clock runs at a constant 100 MHz
// (hipDeviceAttributeWallClockRate) on AMD GPUs.
template <class S>
__forceinline__ __device__ S global_timer_ns() {
  return static_cast<S>(wall_clock64() * 10);
}

__forceinline__ __device__ unsigned int umax(unsigned int a, unsigned int b) {
  return a > b ? a : b;
}

// Per-byte equality of two 32-bit words: 0xff in each byte that matches,
// 0x00 elsewhere.
__forceinline__ __device__ unsigned int __vcmpeq4(unsigned int a,
                                                  unsigned int b) {
  const unsigned int x = a ^ b;
  const unsigned int nonzero = ((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x;
  const unsigned int zero_high = ~nonzero & 0x80808080u;
  return (zero_high >> 7) * 0xffu;
}

// Global-to-shared copies. HIP has no asynchronous copy instruction for
// these, so the copy completes before returning and commit/wait are no-ops.
__forceinline__ __device__ void __pipeline_memcpy_async(void* dst,
                                                        const void* src,
                                                        size_t size) {
  switch (size) {
    case 16:
      *static_cast<uint4*>(dst) = *static_cast<const uint4*>(src);
      break;
    case 8:
      *static_cast<uint2*>(dst) = *static_cast<const uint2*>(src);
      break;
    case 4:
      *static_cast<uint32_t*>(dst) = *static_cast<const uint32_t*>(src);
      break;
    case 2:
      *static_cast<uint16_t*>(dst) = *static_cast<const uint16_t*>(src);
      break;
    default:
      for (size_t i = 0; i < size; ++i) {
        static_cast<char*>(dst)[i] = static_cast<const char*>(src)[i];
      }
  }
}

__forceinline__ __device__ void __pipeline_commit() {}

__forceinline__ __device__ void __pipeline_wait_prior(size_t) {}

// Streaming loads and stores. HIP defines these only for half types.
template <typename T>
__forceinline__ __device__ T __ldcs(const T* ptr) {
  if constexpr (std::is_arithmetic<T>::value) {
    return __builtin_nontemporal_load(ptr);
  } else {
    return *ptr;
  }
}

template <typename T>
__forceinline__ __device__ void __stcs(T* ptr, T value) {
  if constexpr (std::is_arithmetic<T>::value) {
    __builtin_nontemporal_store(value, ptr);
  } else {
    *ptr = value;
  }
}

#endif  // MERLIN_HIP

#endif  // defined(__CUDACC__) || defined(__HIPCC__)

}  // namespace merlin
}  // namespace nv
