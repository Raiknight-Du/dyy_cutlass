# Graph edge host API 诊断 CPU 方案（未重跑）

R4 首次 graph screen 保留失败：一张 N128/CONTROL 图通过3节点、2边、event句柄与拓扑检查，但绑定返回 type=[36,0]，在 kernel params 查询前停止。仅直接 lazy CONTROL128 一次；没有 instantiate/replay/warm/sample。原进程已退出，完整原 tuple 未在断言前序列化，不能补造；原 DOT 与 stderr 是现存事实。

本机为 cuda-bindings13.2.0，扩展 SHA256 `25d06a1d312bb285ba79644110830c271f0eaef1c37e5d444c99e211ee289f74`。libcuda.so.1 同时导出 legacy cuGraphGetEdges 和 cuGraphGetEdges_v2。现有 .pxd 中公开函数有 edgeData 参数，底层声明名为 `_cuGraphGetEdges_v2`；wheel 未提供实际 .pyx wrapper 实现，因此不能仅据声明排除 resolver 错配或返回结构视图生存期问题。

CPU ctypes 与 g++ 对本机 CUDA13.0 cuda.h 的静态编译交叉验证：CUgraphEdgeData sizeof8，from_port/to_port/type/reserved offset为0/1/2/3，reserved长度5；cuGraphGetEdges宏对应五参数v2原型。CPU二进制不链接/调用CUDA。仅dlsym了原生v2，未调用 cuInit 或 graph API，torch context 前后均未初始化。证据在 graph_binding_cpu_audit01。

准备的 `graph_edge_native_audit.py` 将直接绑定原生 v2 五参数ABI，先查询边数，再用零初始化的 from/to 与8字节结构数组读取；随后以 edgeData=NULL、同一from/to数组独立复查。三次原生调用都必须CUDA_SUCCESS，边数/两条边必须精确匹配，type/ports/reserved全部0。NULL查询若遇非默认依赖应报lossy-query，不能忽略非0。

拟单独有限 audit-only 子进程：复用原metadata/CUBIN gate，明确记录一次CONTROL128 lazy；只capture同一三节点图，立即保存Python完整tuple类型、节点句柄、每struct字段/原始8字节/地址。保留tuple并作CPU分配+GC后再保存，另重复Python query；与原生v2零init及NULL结果逐字段比较。读取kernel params/REG并保存DOT，随后退出。禁止 instantiate、graph replay、20warm或任何计时样本，不自动开启新的screen/formal。

若原生也出现非0、NULL检查失败、拓扑不符，直接停止；若原生全0而绑定不同，最多证明host binding路径异常，不自动宣称已定位视图失效或legacy错配。Lead复核后才能决定同一测量外壳的安全实现。内核、计时边界及失败原始证据都不改。

接口行为依据本机cuda.h和官方[CUDA Driver graph API](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__GRAPH.html)；审查不把36当0，也不通过删除type门禁让计时准入。
