# Custom NVFP4 GEMM Optimization Log

## Goal
Optimize custom_nvfp4_gemm to achieve 10% performance improvement across all test cases.

## Test Cases
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

## Optimization Strategies

### 1. Parameter Parsing Support
- **Status**: TODO
- **Details**: Modify argument parsing to support positional parameters "M N K" format
  for test script compatibility

### 2. MMA Tile Shape Optimization
- **Current**: 256x256x256
- **Candidates**: Test 128x256x256, 128x128x256 for smaller K problems
- **Rationale**: Blackwell SM100 supports multiple tile sizes; find optimal for test mix

### 3. Cluster Shape Optimization
- **Current**: 2x4x1
- **Candidates**: Test 1x4x1, 2x2x1, 4x2x1 based on problem dimensions
- **Rationale**: Different cluster shapes may better fit different M/N ratios

### 4. Swizzle Optimization
- **Current**: Default 0
- **Candidates**: Test values 1, 2, 4, 8 for load balancing
- **Rationale**: Swizzle affects tile scheduler work distribution

### 5. Epilogue Schedule Policy
- **Current**: EpilogueScheduleAuto
- **Candidates**: Test explicit schedules if available

## Implementation Record
[Changes will be documented as they are made]

## Performance Results
[Results will be recorded here]
