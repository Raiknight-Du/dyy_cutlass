#include <torch/extension.h>

void gdn_prefill_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor cu_seqlens,
    torch::Tensor initial_state,
    torch::Tensor output_state,
    torch::Tensor state_checkpoints,
    torch::Tensor checkpoint_cu_starts,
    torch::Tensor state_indices,
    double scale,
    int64_t checkpoint_every_n_tokens,
    bool use_qk_l2norm);

void gdn_prefill_cp_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor cu_seqlens,
    torch::Tensor initial_state,
    torch::Tensor output_state,
    torch::Tensor workspace_m,
    torch::Tensor workspace_n,
    torch::Tensor workspace_boundaries,
    double scale,
    int64_t cp_chunk_len,
    int64_t max_partitions);

void gdn_prefill_wmma_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor gate,
    torch::Tensor beta,
    torch::Tensor output,
    torch::Tensor cu_seqlens,
    torch::Tensor initial_state,
    torch::Tensor output_state,
    torch::Tensor state_checkpoints,
    torch::Tensor checkpoint_cu_starts,
    torch::Tensor state_indices,
    double scale,
    int64_t checkpoint_every_n_tokens);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_prefill_cuda", &gdn_prefill_cuda, "op12 native CUDA GDN prefill");
  m.def("gdn_prefill_cp_cuda", &gdn_prefill_cp_cuda, "op12 native CUDA CP GDN prefill");
  m.def("gdn_prefill_wmma_cuda", &gdn_prefill_wmma_cuda, "op12 native CUDA WMMA GDN prefill");
}
