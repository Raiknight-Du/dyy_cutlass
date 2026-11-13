#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace m72a_dataset {

enum class DataKind { kPackedFp4, kUe4m3Scale };

inline uint64_t splitmix64(uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

inline std::filesystem::path shape_directory(
    std::string const& root, int m, int n, int k) {
  return std::filesystem::path(root) /
      ("M" + std::to_string(m) + "_N" + std::to_string(n) +
       "_K" + std::to_string(k));
}

inline void fill_chunk(
    std::vector<uint8_t>& chunk, uint64_t& state, DataKind kind) {
  size_t offset = 0;
  while (offset < chunk.size()) {
    uint64_t value = splitmix64(state);
    for (int byte = 0; byte < 8 && offset < chunk.size(); ++byte, ++offset) {
      uint8_t bits = static_cast<uint8_t>(value >> (byte * 8));
      if (kind == DataKind::kUe4m3Scale) {
        // UE4M3 is unsigned and has one NaN encoding (0x7f). Restrict scales
        // to finite, nonzero encodings while retaining irregular bit patterns.
        bits = static_cast<uint8_t>(0x18u + (bits % 0x47u)); // [0x18, 0x5e]
      }
      chunk[offset] = bits;
    }
  }
}

inline void generate_file(
    std::filesystem::path const& path, size_t bytes, uint64_t seed,
    DataKind kind) {
  std::filesystem::create_directories(path.parent_path());
  if (std::filesystem::exists(path) &&
      std::filesystem::file_size(path) == bytes) {
    return;
  }

  auto temporary = path;
  temporary += ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("Cannot create dataset file: " + temporary.string());
  }

  constexpr size_t kChunkBytes = 16u * 1024u * 1024u;
  std::vector<uint8_t> chunk(std::min(kChunkBytes, bytes));
  uint64_t state = seed;
  size_t written = 0;
  while (written < bytes) {
    chunk.resize(std::min(kChunkBytes, bytes - written));
    fill_chunk(chunk, state, kind);
    output.write(reinterpret_cast<char const*>(chunk.data()), chunk.size());
    if (!output) {
      throw std::runtime_error("Failed writing dataset file: " + temporary.string());
    }
    written += chunk.size();
  }
  output.close();
  std::filesystem::rename(temporary, path);

  std::ofstream metadata(path.string() + ".meta", std::ios::trunc);
  metadata << "bytes=" << bytes << "\nseed=0x" << std::hex << seed << std::dec
           << "\nkind=" << (kind == DataKind::kPackedFp4 ? "packed_fp4" : "ue4m3_scale")
           << "\n";
}

inline void write_cutlass_tensor_file(
    std::filesystem::path const& path, void const* host_ptr, size_t bytes,
    uint64_t seed) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("Cannot create CUTLASS dataset file: " + path.string());
  }
  output.write(static_cast<char const*>(host_ptr), bytes);
  if (!output) {
    throw std::runtime_error("Failed writing CUTLASS dataset file: " + path.string());
  }
  output.close();

  std::ofstream metadata(path.string() + ".meta", std::ios::trunc);
  metadata << "bytes=" << bytes << "\nseed=" << seed
           << "\ngenerator=CUTLASS_TensorFillRandomUniform\n";
}

inline void copy_file_to_device(
    std::filesystem::path const& path, void* device_ptr, size_t bytes) {
  if (!std::filesystem::exists(path) ||
      std::filesystem::file_size(path) != bytes) {
    throw std::runtime_error("Missing or wrong-sized dataset file: " + path.string());
  }

  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot open dataset file: " + path.string());
  }

  constexpr size_t kChunkBytes = 64u * 1024u * 1024u;
  std::vector<uint8_t> chunk(std::min(kChunkBytes, bytes));
  size_t copied = 0;
  auto* destination = static_cast<uint8_t*>(device_ptr);
  while (copied < bytes) {
    size_t count = std::min(kChunkBytes, bytes - copied);
    chunk.resize(count);
    input.read(reinterpret_cast<char*>(chunk.data()), count);
    if (static_cast<size_t>(input.gcount()) != count) {
      throw std::runtime_error("Short read from dataset file: " + path.string());
    }
    cudaError_t status = cudaMemcpy(
        destination + copied, chunk.data(), count, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          "cudaMemcpy failed for " + path.string() + ": " +
          cudaGetErrorString(status));
    }
    copied += count;
  }
}

inline void load_operand(
    std::string const& root, int m, int n, int k, std::string const& name,
    void* device_ptr, size_t bytes, uint64_t seed, DataKind kind,
    bool generate) {
  auto path = shape_directory(root, m, n, k) / (name + ".bin");
  if (generate) {
    generate_file(path, bytes, seed, kind);
  }
  copy_file_to_device(path, device_ptr, bytes);
}

inline void fill_device_random(void* device_ptr, size_t bytes, uint64_t seed) {
  constexpr size_t kChunkBytes = 64u * 1024u * 1024u;
  std::vector<uint8_t> chunk(std::min(kChunkBytes, bytes));
  uint64_t state = seed;
  size_t copied = 0;
  auto* destination = static_cast<uint8_t*>(device_ptr);
  while (copied < bytes) {
    size_t count = std::min(kChunkBytes, bytes - copied);
    chunk.resize(count);
    fill_chunk(chunk, state, DataKind::kPackedFp4);
    cudaError_t status = cudaMemcpy(
        destination + copied, chunk.data(), count, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy random fill failed: ") +
                               cudaGetErrorString(status));
    }
    copied += count;
  }
}

} // namespace m72a_dataset
