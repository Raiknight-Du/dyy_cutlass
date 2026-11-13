#include <cuda.h>
#include <cstddef>
#include <iostream>
#include <type_traits>
using Expected = CUresult (*)(CUgraph, CUgraphNode*, CUgraphNode*, CUgraphEdgeData*, size_t*);
static_assert(std::is_same_v<decltype(&cuGraphGetEdges), Expected>, "CUDA header GraphGetEdges ABI must use edgeData/v2");
static_assert(sizeof(CUgraphEdgeData)==8);
static_assert(offsetof(CUgraphEdgeData,from_port)==0);
static_assert(offsetof(CUgraphEdgeData,to_port)==1);
static_assert(offsetof(CUgraphEdgeData,type)==2);
static_assert(offsetof(CUgraphEdgeData,reserved)==3);
int main(){std::cout << "{\"sizeof\":"<<sizeof(CUgraphEdgeData)<<",\"from_port\":"<<offsetof(CUgraphEdgeData,from_port)<<",\"to_port\":"<<offsetof(CUgraphEdgeData,to_port)<<",\"type\":"<<offsetof(CUgraphEdgeData,type)<<",\"reserved\":"<<offsetof(CUgraphEdgeData,reserved)<<",\"prototype_5args_header_verified\":true,\"CUDA_link_or_calls\":false}\n";}
