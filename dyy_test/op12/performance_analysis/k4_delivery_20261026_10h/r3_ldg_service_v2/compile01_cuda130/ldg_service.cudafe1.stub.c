#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "ldg_service.fatbin.c"
static void __device_stub__Z13r3_ldg_kernelILb0EEvPKjjjPjP5Trace(const uint32_t *, unsigned, unsigned, uint32_t *, struct Trace *);
static void __device_stub__Z13r3_ldg_kernelILb1EEvPKjjjPjP5Trace(const uint32_t *, unsigned, unsigned, uint32_t *, struct Trace *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
static void __device_stub__Z13r3_ldg_kernelILb0EEvPKjjjPjP5Trace(const uint32_t *__par0, unsigned __par1, unsigned __par2, uint32_t *__par3, struct Trace *__par4){__cudaLaunchPrologue(5);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaSetupArgSimple(__par3, 16UL);__cudaSetupArgSimple(__par4, 24UL);__cudaLaunch(((char *)((void ( *)(const uint32_t *, unsigned, unsigned, uint32_t *, struct Trace *))r3_ldg_kernel<(bool)0> )), 0U);}
template<> __specialization_static void __wrapper__device_stub_r3_ldg_kernel<false>( const ::uint32_t *&__cuda_0,unsigned &__cuda_1,unsigned &__cuda_2,::uint32_t *&__cuda_3,struct ::Trace *&__cuda_4){__device_stub__Z13r3_ldg_kernelILb0EEvPKjjjPjP5Trace( (const ::uint32_t *&)__cuda_0,(unsigned &)__cuda_1,(unsigned &)__cuda_2,(::uint32_t *&)__cuda_3,(struct ::Trace *&)__cuda_4);}
static void __device_stub__Z13r3_ldg_kernelILb1EEvPKjjjPjP5Trace(const uint32_t *__par0, unsigned __par1, unsigned __par2, uint32_t *__par3, struct Trace *__par4){__cudaLaunchPrologue(5);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 12UL);__cudaSetupArgSimple(__par3, 16UL);__cudaSetupArgSimple(__par4, 24UL);__cudaLaunch(((char *)((void ( *)(const uint32_t *, unsigned, unsigned, uint32_t *, struct Trace *))r3_ldg_kernel<(bool)1> )), 0U);}
template<> __specialization_static void __wrapper__device_stub_r3_ldg_kernel<true>( const ::uint32_t *&__cuda_0,unsigned &__cuda_1,unsigned &__cuda_2,::uint32_t *&__cuda_3,struct ::Trace *&__cuda_4){__device_stub__Z13r3_ldg_kernelILb1EEvPKjjjPjP5Trace( (const ::uint32_t *&)__cuda_0,(unsigned &)__cuda_1,(unsigned &)__cuda_2,(::uint32_t *&)__cuda_3,(struct ::Trace *&)__cuda_4);}
static void __nv_cudaEntityRegisterCallback(void **__T33){__nv_dummy_param_ref(__T33);__nv_save_fatbinhandle_for_managed_rt(__T33);__cudaRegisterEntry(__T33, ((void ( *)(const uint32_t *, unsigned, unsigned, uint32_t *, struct Trace *))r3_ldg_kernel<(bool)1> ), _Z13r3_ldg_kernelILb1EEvPKjjjPjP5Trace, 128);__cudaRegisterEntry(__T33, ((void ( *)(const uint32_t *, unsigned, unsigned, uint32_t *, struct Trace *))r3_ldg_kernel<(bool)0> ), _Z13r3_ldg_kernelILb0EEvPKjjjPjP5Trace, 128);}
static void __sti____cudaRegisterAll(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}

#pragma GCC diagnostic pop
