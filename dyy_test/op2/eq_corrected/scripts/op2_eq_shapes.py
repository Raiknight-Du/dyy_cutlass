#!/usr/bin/env python3

# Binary coordinates for D^T[N_s, M_s] = B_sparse^T[N_s, K_s] *
# A_dense^T[K_s, M_s]. The logical sparse shapes are recorded in
# ../equivalent_shapes.csv with FLOP and theoretical-byte deltas.
SHAPES = {
    1: (13824, 154636, 11264),
    2: (13696, 468245, 11264),
    3: (5120, 126000, 13824),
    4: (4992, 387692, 13824),
    5: (1664, 3559, 8704),
    6: (1536, 7944, 8448),
    7: (1536, 16913, 7936),
    8: (4096, 6899, 4864),
    9: (3840, 14717, 4864),
    10: (5248, 25575, 256),
    11: (1408, 8937, 256),
    12: (1408, 17873, 256),
    13: (4096, 8192, 256),
}
