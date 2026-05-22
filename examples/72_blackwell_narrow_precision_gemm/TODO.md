### 1. Problem
I encountered a `std::bad_alloc` error when running the following command with large dimensions:

```bash
./examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm --m=510300 --n=5120 --k=5120
```

**Output:**
```text
terminate called after throwing an instance of 'std::bad_alloc'
  what():  std::bad_alloc
Aborted (core dumped)
```

The example `72a_blackwell_nvfp4_bf16_gemm` fails to allocate memory for the shape `(510300, 5120, 5120)`.

### 2. Reference & Analysis Request
Interestingly, another example—`examples/84_blackwell_narrow_precision_sparse_gemm/84a_blackwell_nvfp4_bf16_sparse_gemm.cu`—**successfully runs** with the exact same large dimensions `(510300, 5120, 5120)`.

Could you please analyze how the reference example (No. 84) handles this differently? I would like to understand why it passes while `72a` fails.

### 3. Goal
Based on your analysis, please help me fix the memory allocation issue in example `72a` with **minimal code changes**.

### 4. Build Instructions
To compile the target, run the following command inside the `build` directory:
```bash
make -j16 72a_blackwell_nvfp4_bf16_gemm
```