#pragma once

#include "ssb_utils.h"
//#include "cub/test/test_util.h"
using namespace cub;

template<typename T>
T* loadColumnPinned(string col_name, int num_entries) {
  T* h_col;
  CubDebugExit(cudaMallocHost(&h_col, num_entries * sizeof(T)));
  string filename = DATA_DIR + lookup(col_name);
  ifstream colData (filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    return NULL;
  }

  colData.read((char*)h_col, num_entries * sizeof(T));
  return h_col;
}

/***
 * Loads encoding from disk into memory
 * encoding: bin | dbin
 **/
encoded_column loadEncodedColumnPinned(string col_name, string encoding, int num_entries) {
  if (!(encoding == "bin" || encoding == "dbin" || encoding == "pbin")) {
    cout << "Encoding has to be bin or dbin" << endl;
    exit(1);
  }

  // Open file
  string filename = DATA_DIR + lookup(col_name) + "." + encoding;
  string offsets_filename = DATA_DIR + lookup(col_name) + "." + encoding + "off";

  int fd = open(filename.c_str(), O_RDONLY);

  // Get size of file
  struct stat s;
  int status = fstat(fd, &s);
  int filesize = s.st_size;

  encoded_column col;

  ifstream colData (filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    cout << "Unable to open encoded column file" << filename << endl;
    exit(1);
  }

  CubDebugExit(cudaMallocHost(&col.data, filesize));
  colData.read((char*)col.data, filesize);
  colData.close();

  col.data_size = filesize;

  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((num_entries + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  CubDebugExit(cudaMallocHost(&col.block_start, (num_blocks + 1) * sizeof(uint)));

  ifstream offsetsData (offsets_filename.c_str(), ios::in | ios::binary);
  if (!offsetsData) {
    cout << "Unable to open encoded column file" << offsets_filename << endl;
    exit(1);
  }

  offsetsData.read((char*)col.block_start, (num_blocks + 1) * sizeof(int));
  offsetsData.close();

  return col;
}

encoded_column loadEncodedColumnPinnedRLE(string col_name, string encoding, int num_entries) {
  if (!(encoding == "valbin" || encoding == "rlbin")) {
    cout << "Encoding has to be valbin or rlbin" << endl;
    exit(1);
  }

  // Open file
  string filename = DATA_DIR + lookup(col_name) + "." + encoding;
  string offsets_filename = DATA_DIR + lookup(col_name) + "." + encoding + "off";

  int fd = open(filename.c_str(), O_RDONLY);

  // Get size of file
  struct stat s;
  int status = fstat(fd, &s);
  int filesize = s.st_size;

  encoded_column col;

  ifstream colData (filename.c_str(), ios::in | ios::binary);
  if (!colData) {
    cout << "Unable to open encoded column file" << filename << endl;
    exit(1);
  }

  CubDebugExit(cudaMallocHost(&col.data, filesize));
  colData.read((char*)col.data, filesize);
  colData.close();

  col.data_size = filesize;

  int block_size = 512;
  int elem_per_thread = 1; //the only difference for RLE
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((num_entries + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  CubDebugExit(cudaMallocHost(&col.block_start, (num_blocks + 1) * sizeof(uint)));

  ifstream offsetsData (offsets_filename.c_str(), ios::in | ios::binary);
  if (!offsetsData) {
    cout << "Unable to open encoded column file" << offsets_filename << endl;
    exit(1);
  }

  offsetsData.read((char*)col.block_start, (num_blocks + 1) * sizeof(int));
  offsetsData.close();

  return col;
}

template<typename T>
T* loadColumnToGPU(T* src, int len, CachingDeviceAllocator& g_allocator) {
  T* dest = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &dest, sizeof(T) * len));
  CubDebugExit(cudaMemcpy(dest, src, sizeof(T) * len, cudaMemcpyHostToDevice));
  return dest;
}

encoded_column loadEncodedColumnToGPU(string col_name, string encoding, int len, CachingDeviceAllocator& g_allocator) {
  if (!(encoding == "bin" || encoding == "dbin" || encoding == "pbin")) {
    cout << "Encoding has to be bin or dbin" << endl;
    exit(1);
  }

  encoded_column h_col = loadEncodedColumn(col_name, encoding, len);

  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint* d_col_block_start = loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1, g_allocator);
  uint* d_col_data = loadColumnToGPU<uint>(h_col.data, h_col.data_size/4, g_allocator);

  cout << "Encoded Col Size: " << h_col.data_size << " " << num_blocks + 1 << endl;

  encoded_column d_col;
  d_col.block_start = d_col_block_start;
  d_col.data = d_col_data;
  return d_col;
}

encoded_column loadEncodedColumnToGPURLE(string col_name, string encoding, int len, CachingDeviceAllocator& g_allocator) {
  if (!(encoding == "valbin" || encoding == "rlbin")) {
    cout << "Encoding has to be valbin or rlbin" << endl;
    exit(1);
  }

  encoded_column h_col = loadEncodedColumnRLE(col_name, encoding, len);

  // for (int i = 0; i < 5; i++) {
  //   cout << h_col.block_start[i] << endl;
  // }

  int block_size = 512;
  int elem_per_thread = 1; //the only difference for RLE
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint* d_col_block_start = loadColumnToGPU<uint>(h_col.block_start, num_blocks + 1, g_allocator);
  uint* d_col_data = loadColumnToGPU<uint>(h_col.data, h_col.data_size/4, g_allocator);

  cout << "Encoded Col Size: " << h_col.data_size << " " << num_blocks + 1 << " " << h_col.data_size + num_blocks + 1 << endl;

  encoded_column d_col;
  d_col.block_start = d_col_block_start;
  d_col.data = d_col_data;
  return d_col;
}

encoded_column allocateEncodedColumnOnGPU(encoded_column h_col, int len, CachingDeviceAllocator& g_allocator) {
  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint* d_col_block_start = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_col_block_start, (num_blocks + 1) * sizeof(uint)));

  uint* d_col_data = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_col_data, h_col.data_size));

  encoded_column d_col;
  d_col.block_start = d_col_block_start;
  d_col.data = d_col_data;
  d_col.data_size = h_col.data_size;
  return d_col;
}

encoded_column allocateEncodedColumnOnGPURLE(encoded_column h_col, int len, CachingDeviceAllocator& g_allocator) {
  int block_size = 512;
  int elem_per_thread = 1; //the only difference for RLE
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  uint* d_col_block_start = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_col_block_start, (num_blocks + 1) * sizeof(uint)));

  uint* d_col_data = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**) &d_col_data, h_col.data_size));

  encoded_column d_col;
  d_col.block_start = d_col_block_start;
  d_col.data = d_col_data;
  d_col.data_size = h_col.data_size;
  return d_col;
}

template<typename T>
void copyColumn(const T* h_col, T* d_col, int len, cudaStream_t stream = nullptr) {
  if (stream == nullptr) {
    CubDebugExit(cudaMemcpy(d_col, h_col, sizeof(T) * len, cudaMemcpyHostToDevice));
  } else {
    CubDebugExit(cudaMemcpyAsync(d_col, h_col, sizeof(T) * len, cudaMemcpyHostToDevice, stream));
  }
}

void copyEncodedColumn(encoded_column h_col, encoded_column d_col, int len, cudaStream_t stream = nullptr) {
  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  if (stream == nullptr) {
    CubDebugExit(cudaMemcpy(d_col.block_start, h_col.block_start, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_col.data, h_col.data, h_col.data_size, cudaMemcpyHostToDevice));
  } else {
    CubDebugExit(cudaMemcpyAsync(d_col.block_start, h_col.block_start, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice, stream));
    CubDebugExit(cudaMemcpyAsync(d_col.data, h_col.data, h_col.data_size, cudaMemcpyHostToDevice, stream));
  }
}

void copyEncodedColumnRLE(encoded_column h_col, encoded_column d_col, int len, cudaStream_t stream = nullptr) {
  int block_size = 512;
  int elem_per_thread = 1; //the only difference for RLE
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  if (stream == nullptr) {
    CubDebugExit(cudaMemcpy(d_col.block_start, h_col.block_start, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(d_col.data, h_col.data, h_col.data_size, cudaMemcpyHostToDevice));
  } else {
    CubDebugExit(cudaMemcpyAsync(d_col.block_start, h_col.block_start, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice, stream));
    CubDebugExit(cudaMemcpyAsync(d_col.data, h_col.data, h_col.data_size, cudaMemcpyHostToDevice, stream));
  }
}

size_t sizeOfEncodedColumn(encoded_column h_col, int len) {
  int block_size = 128;
  int elem_per_thread = 4;
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  return h_col.data_size + (num_blocks + 1) * sizeof(uint);
}

size_t sizeOfEncodedColumnRLE(encoded_column h_col, int len) {
  int block_size = 512;
  int elem_per_thread = 1; //the only difference for RLE
  int tile_size = block_size * elem_per_thread;
  int adjusted_len = ((len + tile_size - 1)/tile_size) * tile_size;
  int num_blocks = adjusted_len / block_size;

  return h_col.data_size + (num_blocks + 1) * sizeof(uint);
}

template <typename T, typename ST, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoadDirect(const unsigned int tid,
                                                    T*                 block_itr,
                                                    T                  (&items)[ITEMS_PER_THREAD],
                                                    ST                 (&selection_flags)[ITEMS_PER_THREAD]) {
  T* thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) { items[ITEM] = thread_itr[ITEM * BLOCK_THREADS]; }
  }
}

template <typename T, typename ST, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoadDirect(const unsigned int tid,
                                                    T*                 block_itr,
                                                    T                  (&items)[ITEMS_PER_THREAD],
                                                    int                num_items,
                                                    ST                 (&selection_flags)[ITEMS_PER_THREAD]) {
  T* thread_itr = block_itr + tid;

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      if (tid + (ITEM * BLOCK_THREADS) < num_items) { items[ITEM] = thread_itr[ITEM * BLOCK_THREADS]; }
    }
  }
}

template <typename T, typename ST, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLoad(T* inp, T (&items)[ITEMS_PER_THREAD], int num_items, ST (&selection_flags)[ITEMS_PER_THREAD]) {
  T* block_itr = inp;

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockPredLoadDirect<T, ST, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, selection_flags);
  } else {
    BlockPredLoadDirect<T, ST, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, block_itr, items, num_items, selection_flags);
  }
}

