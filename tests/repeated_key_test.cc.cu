/*
 * A batch that repeats a key so that several thread groups of one warp
 * contend for the same slot must complete, with every copy applied. The
 * operation runs under a watchdog: a kernel that never returns fails the test
 * binary instead of hanging it.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include "merlin_hashtable.cuh"
#include "test_util.cuh"

namespace {

using K = uint64_t;
using V = float;
using S = uint64_t;
using EvictStrategy = nv::merlin::EvictStrategy;
using TableOptions = nv::merlin::HashTableOptions;
using Table = nv::merlin::HashTable<K, V, S, EvictStrategy::kCustomized>;

constexpr int kTimeoutSeconds = 120;

// Runs `op` and waits for `stream`, on a worker thread. A kernel that never
// finishes cannot be cancelled, so on timeout the process exits with a failure
// rather than blocking in the device teardown.
void run_with_timeout(const std::string& what, cudaStream_t stream,
                      const std::function<void()>& op) {
  std::atomic<bool> done{false};
  cudaError_t err = cudaSuccess;
  std::thread worker([&] {
    op();
    err = cudaStreamSynchronize(stream);
    done = true;
  });
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kTimeoutSeconds);
  while (!done) {
    if (std::chrono::steady_clock::now() > deadline) {
      std::fprintf(stderr, "FAILED: %s did not finish within %d s\n",
                   what.c_str(), kTimeoutSeconds);
      std::fflush(stderr);
      std::_Exit(EXIT_FAILURE);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  worker.join();
  CUDA_CHECK(err);
}

enum class Op { kAssign, kAssignScores, kAssignValues };

const char* op_name(Op op) {
  switch (op) {
    case Op::kAssign:
      return "assign";
    case Op::kAssignScores:
      return "assign_scores";
    default:
      return "assign_values";
  }
}

// 1024 buckets of 128 slots, 256-byte values and a load factor above 0.6
// select the pipeline update kernels, whose thread groups have 16 threads.
// Each 128-key block of the batch carries the repeated key at the first key
// of `copies` consecutive groups, which are in one warp of 64 threads (two
// warps of 32 for more than 2 copies).
void test_repeated_key(Op op, int copies) {
  constexpr size_t kDim = 64;
  constexpr size_t kBucketSize = 128;
  constexpr size_t kCapacity = 1024 * kBucketSize;
  constexpr size_t kInsertNum = kCapacity * 3 / 4;
  constexpr size_t kBatch = 1024;
  constexpr size_t kGroupSize = 16;
  constexpr size_t kBlock = 128;

  TableOptions options;
  options.init_capacity = kCapacity;
  options.max_capacity = kCapacity;
  options.max_bucket_size = kBucketSize;
  options.dim = kDim;
  options.max_hbm_for_vectors = nv::merlin::GB(16);
  Table table;
  table.init(options);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<K> h_keys(kInsertNum);
  std::vector<S> h_scores(kInsertNum);
  std::vector<V> h_values(kInsertNum * kDim);
  for (size_t i = 0; i < kInsertNum; i++) {
    h_keys[i] = 1 + i * 7919;
    h_scores[i] = 100;
    for (size_t j = 0; j < kDim; j++) h_values[i * kDim + j] = 0.0f;
  }

  K* d_keys;
  S* d_scores;
  V* d_values;
  bool* d_found;
  CUDA_CHECK(cudaMalloc(&d_keys, kInsertNum * sizeof(K)));
  CUDA_CHECK(cudaMalloc(&d_scores, kInsertNum * sizeof(S)));
  CUDA_CHECK(cudaMalloc(&d_values, kInsertNum * kDim * sizeof(V)));
  CUDA_CHECK(cudaMalloc(&d_found, kInsertNum * sizeof(bool)));
  CUDA_CHECK(cudaMemcpy(d_keys, h_keys.data(), kInsertNum * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), kInsertNum * sizeof(S),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, h_values.data(),
                        kInsertNum * kDim * sizeof(V),
                        cudaMemcpyHostToDevice));
  table.insert_or_assign(kInsertNum, d_keys, d_values, d_scores, stream);

  // Overflowing buckets evict some keys; build the batch from keys present.
  table.find(kInsertNum, d_keys, d_values, d_found, nullptr, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<uint8_t> h_found(kInsertNum);
  CUDA_CHECK(cudaMemcpy(h_found.data(), d_found, kInsertNum * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  std::vector<K> present;
  for (size_t i = 0; i < kInsertNum; i++) {
    if (h_found[i]) present.push_back(h_keys[i]);
  }
  ASSERT_GT(present.size(), kBatch + 1);

  const K repeated = present[kBatch];
  std::vector<K> b_keys(present.begin(), present.begin() + kBatch);
  for (size_t base = 0; base < kBatch; base += kBlock) {
    for (int g = 0; g < copies; g++) b_keys[base + g * kGroupSize] = repeated;
  }
  std::vector<S> b_scores(kBatch);
  std::vector<V> b_values(kBatch * kDim);
  for (size_t i = 0; i < kBatch; i++) {
    const bool rep = b_keys[i] == repeated;
    b_scores[i] = rep ? 7 : 1000 + i;
    for (size_t j = 0; j < kDim; j++) {
      b_values[i * kDim + j] = rep ? -1.0f : static_cast<V>(i + 1);
    }
  }
  CUDA_CHECK(cudaMemcpy(d_keys, b_keys.data(), kBatch * sizeof(K),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scores, b_scores.data(), kBatch * sizeof(S),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, b_values.data(), kBatch * kDim * sizeof(V),
                        cudaMemcpyHostToDevice));

  const std::string what = std::string(op_name(op)) + " with " +
                           std::to_string(copies) + " copies of a key";
  run_with_timeout(what, stream, [&] {
    switch (op) {
      case Op::kAssign:
        table.assign(kBatch, d_keys, d_values, d_scores, stream);
        break;
      case Op::kAssignScores:
        table.assign_scores(kBatch, d_keys, d_scores, stream);
        break;
      case Op::kAssignValues:
        table.assign_values(kBatch, d_keys, d_values, stream);
        break;
    }
  });

  CUDA_CHECK(cudaMemset(d_values, 0, kBatch * kDim * sizeof(V)));
  CUDA_CHECK(cudaMemset(d_scores, 0, kBatch * sizeof(S)));
  table.find(kBatch, d_keys, d_values, d_found, d_scores, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<V> f_values(kBatch * kDim);
  std::vector<S> f_scores(kBatch);
  CUDA_CHECK(cudaMemcpy(h_found.data(), d_found, kBatch * sizeof(bool),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(f_values.data(), d_values, kBatch * kDim * sizeof(V),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(f_scores.data(), d_scores, kBatch * sizeof(S),
                        cudaMemcpyDeviceToHost));

  const bool values_assigned = op != Op::kAssignScores;
  const bool scores_assigned = op != Op::kAssignValues;
  size_t bad = 0;
  for (size_t i = 0; i < kBatch; i++) {
    if (!h_found[i]) {
      bad++;
      continue;
    }
    const V want_value = values_assigned ? b_values[i * kDim] : 0.0f;
    const S want_score = scores_assigned ? b_scores[i] : 100;
    bool ok = f_scores[i] == want_score;
    for (size_t j = 0; j < kDim; j++) ok &= f_values[i * kDim + j] == want_value;
    if (!ok) bad++;
  }
  EXPECT_EQ(bad, 0) << what;

  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_found));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace

TEST(RepeatedKeyTest, assign_2_copies) { test_repeated_key(Op::kAssign, 2); }
TEST(RepeatedKeyTest, assign_3_copies) { test_repeated_key(Op::kAssign, 3); }
TEST(RepeatedKeyTest, assign_4_copies) { test_repeated_key(Op::kAssign, 4); }
TEST(RepeatedKeyTest, assign_scores_2_copies) {
  test_repeated_key(Op::kAssignScores, 2);
}
TEST(RepeatedKeyTest, assign_scores_3_copies) {
  test_repeated_key(Op::kAssignScores, 3);
}
TEST(RepeatedKeyTest, assign_scores_4_copies) {
  test_repeated_key(Op::kAssignScores, 4);
}
TEST(RepeatedKeyTest, assign_values_2_copies) {
  test_repeated_key(Op::kAssignValues, 2);
}
TEST(RepeatedKeyTest, assign_values_3_copies) {
  test_repeated_key(Op::kAssignValues, 3);
}
TEST(RepeatedKeyTest, assign_values_4_copies) {
  test_repeated_key(Op::kAssignValues, 4);
}
