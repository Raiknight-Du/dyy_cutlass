#define main op1_experiment_main
#include "op1_packed.cu"
#undef main

#include <cstddef>

template <class Config>
void print_config(char const* name) {
  using Mainloop = typename Config::CollectiveMainloop;
  using Epilogue = typename Config::CollectiveEpilogue;
  using Kernel = typename Config::GemmKernel;
  using KernelTensorStorage = typename Kernel::SharedStorage::TensorStorage;
  std::printf("\n%s\n", name);
  std::printf(
      "stages=%d A=%zu B=%zu SFA=%zu SFB=%zu "
      "mainloop_tensors=%zu align=%zu mainloop_offset=%zu "
      "mainloop_pipeline=%zu epilogue=%zu kernel_shared=%zu\n",
      Mainloop::DispatchPolicy::Stages,
      size_t(cute::cosize_v<typename Mainloop::SmemLayoutA>),
      size_t(cute::cosize_v<typename Mainloop::SmemLayoutB>),
      size_t(cute::cosize_v<typename Mainloop::SmemLayoutSFA>),
      size_t(cute::cosize_v<typename Mainloop::SmemLayoutSFB>),
      sizeof(typename Mainloop::TensorStorage),
      alignof(typename Mainloop::TensorStorage),
      offsetof(KernelTensorStorage, mainloop),
      sizeof(typename Mainloop::PipelineStorage),
      sizeof(typename Epilogue::SharedStorage),
      sizeof(typename Kernel::SharedStorage));
  cute::print("A:   "); cute::print(typename Mainloop::SmemLayoutA{}); cute::print("\n");
  cute::print("B:   "); cute::print(typename Mainloop::SmemLayoutB{}); cute::print("\n");
  cute::print("SFA: "); cute::print(typename Mainloop::SmemLayoutSFA{}); cute::print("\n");
  cute::print("SFB: "); cute::print(typename Mainloop::SmemLayoutSFB{}); cute::print("\n");
}

int main() {
  print_config<Config_00>("Config_00");
  print_config<Config_02>("Config_02");
  print_config<Config_03>("Config_03");
  print_config<Config_04>("Config_04");
  print_config<Config_05>("Config_05");
  print_config<Config_06>("Config_06");
  print_config<Config_07>("Config_07");
  print_config<Config_08>("Config_08");
  print_config<Config_09>("Config_09");
  print_config<Config_10>("Config_10");
  return 0;
}
