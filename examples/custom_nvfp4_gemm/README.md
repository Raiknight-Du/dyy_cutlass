# Custom NVFP4 GEMM Example

This example demonstrates a custom NVFP4 GEMM implementation for the NVIDIA Blackwell SM100 architecture using CUTLASS.

## Features

- Blockscaled NVFP4 (NVIDIA 4-bit floating point) data type support
- Blackwell SM100 architecture optimization
- Warp-specialized kernel design
- Tensor Memory (TMEM) utilization
- Block-scaled BF16 output

## Building

From the CUTLASS build directory:

```bash
cd /path/to/cutlass/build

# Build the example
make custom_nvfp4_gemm -j16
```

## Running

After successful compilation:

```bash
./examples/custom_nvfp4_gemm/custom_nvfp4_gemm [options]
```

### Command Line Options

- `--help` - Display usage information
- `--m=<int>` - M dimension of GEMM (default: 1024)
- `--n=<int>` - N dimension of GEMM (default: 1024)
- `--k=<int>` - K dimension of GEMM (default: 1024)
- `--alpha=<float>` - Epilogue scalar alpha (default: 1.0)
- `--beta=<float>` - Epilogue scalar beta (default: 0.0)
- `--iterations=<int>` - Number of profiling iterations (default: 10)

### Example Usage

```bash
./examples/custom_nvfp4_gemm/custom_nvfp4_gemm --m=2048 --n=2048 --k=2048
```

## Implementation Details

The example uses:

1. **ElementA/B**: `cutlass::nv_float4_t<cutlass::float_e2m1_t>` (NVFP4 with block scaling)
2. **ElementC/D**: `cutlass::bfloat16_t` (BF16 output)
3. **MMA Tile Shape**: 128x128x256
4. **Cluster Shape**: 1x1x1
5. **Architecture**: SM100 (Blackwell)

## References

- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/)
- [Blackwell Architecture](https://docs.nvidia.com/cuda/parallel-thread-execution/)
