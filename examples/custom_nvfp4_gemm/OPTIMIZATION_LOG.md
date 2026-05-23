# Custom NVFP4 GEMM Optimization Log

## Goal
Optimize custom_nvfp4_gemm to achieve 10% performance improvement across all test cases.

### Test Cases
1. M=8192, N=4096, K=128
2. M=32768, N=768, K=128
3. M=16384, N=768, K=128
4. M=32768, N=4096, K=128
5. M=16384, N=4096, K=2048
6. M=8192, N=4096, K=2048
7. M=32768, N=768, K=4096
8. M=16384, N=768, K=4096
9. M=8192, N=768, K=4096
10. M=510300, N=5120, K=5120
11. M=170100, N=5120, K=5120
12. M=510300, N=13824, K=5120
13. M=170100, N=13824, K=5120

### Baseline
序号   Shape (M×K, N×K)                         Runtime(ms)  Perf(Gflops)   
-------------------------------------------------------------------------------
1    M×K: 8192×128, N×K: 4096×128               0.035        2.44450e+05
2    M×K: 32768×128, N×K: 768×128               0.035        1.83024e+05
3    M×K: 16384×128, N×K: 768×128               0.023        1.39206e+05
4    M×K: 32768×128, N×K: 4096×128              0.108        3.17294e+05
5    M×K: 16384×2048, N×K: 4096×2048            0.141        1.94342e+06
6    M×K: 8192×2048, N×K: 4096×2048             0.077        1.77846e+06
7    M×K: 32768×4096, N×K: 768×4096             0.132        1.56750e+06
8    M×K: 16384×4096, N×K: 768×4096             0.075        1.37714e+06
9    M×K: 8192×4096, N×K: 768×4096              0.048        1.07307e+06
10   M×K: 510300×5120, N×K: 5120×5120           10.14        2.63851e+06
11   M×K: 170100×5120, N×K: 5120×5120           3.43         2.60003e+06
12   M×K: 510300×5120, N×K: 13824×5120          25.13        2.87454e+06
13   M×K: 170100×5120, N×K: 13824×5120          8.26         2.91513e+06
### Instructions
To compile the target, run the following command inside the `build` directory:
```bash
make -j16 custom_nvfp4_gemm
```

To test the performance, run the following command inside the `build` directory:
```bash
bash test_custom_nvfp4_gemm.sh
```

## Optimization Strategies

### MMA Tile Shape Optimization
- **Current**: 256x256x256
- **Candidates**: Test 128x256x256, 128x128x256 for smaller K problems
- **Rationale**: Blackwell SM100 supports multiple tile sizes; find optimal for test mix

### Cluster Shape Optimization
- **Current**: 2x4x1
- **Candidates**: Test 1x4x1, 2x2x1, 4x2x1 based on problem dimensions
- **Rationale**: Different cluster shapes may better fit different M/N ratios

### Swizzle Optimization
- **Current**: Default 0
- **Candidates**: Test values 1, 2, 4, 8 for load balancing
- **Rationale**: Swizzle affects tile scheduler work distribution

### Epilogue Schedule Policy
- **Current**: EpilogueScheduleAuto
- **Candidates**: Test explicit schedules if available

## Implementation Record
- Updated `examples/custom_nvfp4_gemm/custom_nvfp4_gemm.cu` to use a dual-kernel strategy:
  - `MmaTileShapeSmallK = 128x256x128` for `K <= 128`
  - `MmaTileShapeLargeK = 128x128x256` for `K > 128`
  - `ClusterShapeSmall = 2x2x1` for small-K problems
  - `ClusterShapeBig = 1x4x1` for large-K problems
- Verified that the kernel builds cleanly with CUDA 12.8+ and SM100 support.
- Executed the full benchmark suite in `build/test_custom_nvfp4_gemm.sh` and confirmed all 13 cases pass.

## Performance Results
序号   Shape (M×K, N×K)                         Runtime(ms)  Perf(Gflops)   
-------------------------------------------------------------------------------
1    M×K: 8192×128, N×K: 4096×128               0.0314112    2.73467e+05
2    M×K: 32768×128, N×K: 768×128               0.0306848    2.09956e+05
3    M×K: 16384×128, N×K: 768×128               0.0201248    1.60062e+05
4    M×K: 32768×128, N×K: 4096×128              0.0940928    3.65169e+05
5    M×K: 16384×2048, N×K: 4096×2048            0.120518     2.28080e+06
6    M×K: 8192×2048, N×K: 4096×2048             0.0660288    2.08150e+06
7    M×K: 32768×4096, N×K: 768×4096             0.113888     1.81019e+06
8    M×K: 16384×4096, N×K: 768×4096             0.0614464    1.67755e+06
9    M×K: 8192×4096, N×K: 768×4096              0.0361504    1.42570e+06
10   M×K: 510300×5120, N×K: 5120×5120           8.57172      3.12124e+06
11   M×K: 170100×5120, N×K: 5120×5120           2.86172      3.11636e+06
12   M×K: 510300×13824, N×K: 5120×5120          23.048       3.13419e+06
13   M×K: 170100×13824, N×K: 5120×5120          7.69469      3.12930e+06

Notes:
- The final tuned implementation uses `EpilogueScheduleAuto` and `KernelScheduleAuto`.
- Swizzle remains configurable via `--swizzle=<int>` for additional scheduler tuning.
