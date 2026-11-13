#!/usr/bin/env python3
import csv
import math
from fractions import Fraction
from pathlib import Path


ORIGINAL_SHAPES = [
    (1, 170100, 13824, 5120),
    (2, 510300, 13824, 5120),
    (3, 170100, 5120, 5120),
    (4, 510300, 5120, 5120),
    (5, 8192, 768, 4096),
    (6, 16384, 768, 4096),
    (7, 32768, 768, 4096),
    (8, 8192, 4096, 2048),
    (9, 16384, 4096, 2048),
    (10, 32768, 4096, 128),
    (11, 16384, 768, 128),
    (12, 32768, 768, 128),
    (13, 8192, 4096, 128),
]

# Current CUTLASS NVFP4 implementation:
# value = 4 bits, scale = one byte per 32 logical elements,
# sparse values = half of dense values, metadata = one byte per 16 elements.
A_BYTES = Fraction(1, 2) + Fraction(1, 32)
B_DENSE_BYTES = Fraction(1, 2) + Fraction(1, 32)
B_SPARSE_BYTES = (
    Fraction(1, 4) + Fraction(1, 32) + Fraction(1, 16)
)
D_BYTES = Fraction(2, 1)

N_ALIGNMENT = 128
K_ALIGNMENT = 256


def dense_bytes(m, n, k):
    return A_BYTES * m * k + B_DENSE_BYTES * n * k + D_BYTES * m * n


def sparse_bytes(m, n, k):
    return A_BYTES * m * k + B_SPARSE_BYTES * n * k + D_BYTES * m * n


def continuous_roots(m, n, k):
    product = 2 * n * k
    target = dense_bytes(m, n, k)
    # D*M*N_s^2 - (Q_d - b_s*P)*N_s + a*M*P = 0
    qa = float(D_BYTES * m)
    qb = -float(target - B_SPARSE_BYTES * product)
    qc = float(A_BYTES * m * product)
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return []
    return sorted([
        (-qb - math.sqrt(disc)) / (2 * qa),
        (-qb + math.sqrt(disc)) / (2 * qa),
    ])


def divisors(value):
    result = set()
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate == 0:
            result.add(candidate)
            result.add(value // candidate)
    return sorted(result)


def choose_fixed_m_shape(m, n, k):
    product = 2 * n * k
    unit_product, remainder = divmod(product, N_ALIGNMENT * K_ALIGNMENT)
    if remainder:
        raise ValueError(f"Product cannot use required alignments: {(m, n, k)}")

    target = dense_bytes(m, n, k)
    roots = continuous_roots(m, n, k)
    near_root = (
        min(roots, key=lambda root: abs(math.log(root / n)))
        if roots
        else float(n)
    )
    candidates = []
    for n_units in divisors(unit_product):
        ns = n_units * N_ALIGNMENT
        ks = (unit_product // n_units) * K_ALIGNMENT
        q_sparse = sparse_bytes(m, ns, ks)
        byte_error = abs(float(q_sparse / target - 1))
        root_distance = abs(math.log(ns / near_root))
        shape_distance = abs(math.log(ns / n)) + abs(math.log(ks / (2 * k)))
        candidates.append((
            byte_error,
            root_distance,
            shape_distance,
            ns,
            ks,
            q_sparse,
        ))
    candidates.sort()
    return roots, near_root, candidates[0], candidates[:8]


def choose_joint_shape(m, n, k):
    product = 2 * m * n * k
    target = dense_bytes(m, n, k)
    all_divisors = divisors(product)
    candidates = []

    for ks in all_divisors:
        if ks % K_ALIGNMENT != 0:
            continue
        if not (k // 4 <= ks <= 16 * k):
            continue
        mn_product = product // ks
        for ns in divisors(mn_product):
            if ns % N_ALIGNMENT != 0:
                continue
            if not (max(N_ALIGNMENT, n // 8) <= ns <= 8 * n):
                continue
            ms = mn_product // ns
            if not (max(1, m // 8) <= ms <= 8 * m):
                continue

            q_sparse = sparse_bytes(ms, ns, ks)
            byte_error = abs(float(q_sparse / target - 1))
            shape_distance = (
                abs(math.log(ms / m))
                + abs(math.log(ns / n))
                + abs(math.log(ks / (2 * k)))
            )
            candidates.append((
                byte_error,
                shape_distance,
                ms,
                ns,
                ks,
                q_sparse,
            ))

    if not candidates:
        raise ValueError(f"No jointly aligned candidate for {(m, n, k)}")

    candidates.sort()
    # A tiny byte-error improvement is not worth turning a model-like shape
    # into an extreme aspect ratio. Prefer the closest shape under 2% byte
    # error; relax only when the integer/alignment constraints require it.
    tolerance = next(
        limit
        for limit in (0.02, 0.05, 0.10, 0.20, 1.0)
        if any(candidate[0] <= limit for candidate in candidates)
    )
    acceptable = [
        candidate for candidate in candidates if candidate[0] <= tolerance
    ]
    acceptable.sort(key=lambda candidate: (candidate[1], candidate[0]))
    selected = acceptable[0]
    return selected, candidates[:12], tolerance


def choose_practical_shape(m, n, k):
    target_flops = 2 * m * n * k
    target_bytes = dense_bytes(m, n, k)
    candidates = []

    n_begin = max(N_ALIGNMENT, math.ceil((n / 2) / N_ALIGNMENT) * N_ALIGNMENT)
    n_end = 4 * n
    k_begin = max(K_ALIGNMENT, math.ceil((k / 2) / K_ALIGNMENT) * K_ALIGNMENT)
    k_end = 8 * k

    for ns in range(n_begin, n_end + 1, N_ALIGNMENT):
        for ks in range(k_begin, k_end + 1, K_ALIGNMENT):
            ms = max(1, round(target_flops / (ns * ks)))
            if not (m / 4 <= ms <= 4 * m):
                continue
            actual_flops = ms * ns * ks
            flop_error = abs(actual_flops / target_flops - 1)
            if flop_error > 0.001:
                continue
            q_sparse = sparse_bytes(ms, ns, ks)
            byte_error = abs(float(q_sparse / target_bytes - 1))
            shape_distance = (
                abs(math.log(ms / m))
                + abs(math.log(ns / n))
                + abs(math.log(ks / (2 * k)))
            )
            candidates.append((
                byte_error,
                flop_error,
                shape_distance,
                ms,
                ns,
                ks,
                q_sparse,
                actual_flops,
            ))

    if not candidates:
        raise ValueError(f"No practical candidate for {(m, n, k)}")

    tolerance = next(
        limit
        for limit in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0)
        if any(candidate[0] <= limit for candidate in candidates)
    )
    acceptable = [
        candidate for candidate in candidates if candidate[0] <= tolerance
    ]
    acceptable.sort(key=lambda candidate: (
        candidate[2],
        candidate[0],
        candidate[1],
    ))
    candidates.sort(key=lambda candidate: (
        candidate[0],
        candidate[1],
        candidate[2],
    ))
    return acceptable[0], candidates[:12], tolerance


def main():
    output = Path(__file__).resolve().parent.parent
    rows = []
    candidate_rows = []
    for shape_id, m, n, k in ORIGINAL_SHAPES:
        roots, near_root, fixed_best, fixed_candidates = choose_fixed_m_shape(
            m, n, k
        )
        joint_best, joint_candidates, joint_tolerance = choose_joint_shape(
            m, n, k
        )
        practical_best, practical_candidates, tolerance = (
            choose_practical_shape(m, n, k)
        )
        (
            byte_error,
            flop_error,
            shape_distance,
            ms,
            ns,
            ks,
            q_sparse,
            actual_flops,
        ) = practical_best
        q_dense = dense_bytes(m, n, k)
        rows.append({
            "shape": shape_id,
            "dense_m": m,
            "dense_n": n,
            "dense_k": k,
            "sparse_m": ms,
            "sparse_n": ns,
            "sparse_k": ks,
            "op2_m": ns,
            "op2_n": ms,
            "op2_k": ks,
            "continuous_solution_exists": bool(roots),
            "continuous_root_near_n": near_root if roots else "",
            "selection_byte_tolerance_pct": tolerance * 100,
            "shape_log_distance": shape_distance,
            "dense_actual_flops": 2 * m * n * k,
            "sparse_actual_flops": actual_flops,
            "flop_difference_pct": 100 * (
                actual_flops / (2 * m * n * k) - 1
            ),
            "dense_bytes": float(q_dense),
            "sparse_bytes": float(q_sparse),
            "byte_difference_pct": 100 * float(q_sparse / q_dense - 1),
        })
        fixed_error, _, _, fixed_n, fixed_k, fixed_q = fixed_best
        candidate_rows.append({
            "shape": shape_id,
            "candidate_kind": "fixed_m_best",
            "rank": 1,
            "sparse_m": m,
            "sparse_n": fixed_n,
            "sparse_k": fixed_k,
            "byte_difference_pct": 100 * float(fixed_q / q_dense - 1),
            "byte_error_pct": 100 * fixed_error,
            "shape_distance": "",
            "flop_difference_pct": 0.0,
            "flop_error_pct": 0.0,
        })
        (
            joint_error,
            joint_distance,
            joint_m,
            joint_n,
            joint_k,
            joint_q,
        ) = joint_best
        candidate_rows.append({
            "shape": shape_id,
            "candidate_kind": "joint_exact_selected",
            "rank": 1,
            "sparse_m": joint_m,
            "sparse_n": joint_n,
            "sparse_k": joint_k,
            "byte_difference_pct": 100 * float(joint_q / q_dense - 1),
            "byte_error_pct": 100 * joint_error,
            "shape_distance": joint_distance,
            "flop_difference_pct": 0.0,
            "flop_error_pct": 0.0,
        })
        for rank, candidate in enumerate(joint_candidates, 1):
            error, candidate_distance, cm, cn, ck, cq = candidate
            candidate_rows.append({
                "shape": shape_id,
                "candidate_kind": "joint_byte_best",
                "rank": rank,
                "sparse_m": cm,
                "sparse_n": cn,
                "sparse_k": ck,
                "byte_difference_pct": 100 * float(cq / q_dense - 1),
                "byte_error_pct": 100 * error,
                "shape_distance": candidate_distance,
                "flop_difference_pct": 0.0,
                "flop_error_pct": 0.0,
            })
        for rank, candidate in enumerate(practical_candidates, 1):
            (
                error,
                candidate_flop_error,
                candidate_distance,
                cm,
                cn,
                ck,
                cq,
                candidate_flops,
            ) = candidate
            candidate_rows.append({
                "shape": shape_id,
                "candidate_kind": "practical_byte_best",
                "rank": rank,
                "sparse_m": cm,
                "sparse_n": cn,
                "sparse_k": ck,
                "byte_difference_pct": 100 * float(cq / q_dense - 1),
                "byte_error_pct": 100 * error,
                "shape_distance": candidate_distance,
                "flop_difference_pct": 100 * (
                    candidate_flops / (2 * m * n * k) - 1
                ),
                "flop_error_pct": 100 * candidate_flop_error,
            })

    with (output / "equivalent_shapes.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "equivalent_shape_candidates.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=candidate_rows[0].keys())
        writer.writeheader()
        writer.writerows(candidate_rows)

    print(
        f"coefficients: a={float(A_BYTES):.7f}, "
        f"b_dense={float(B_DENSE_BYTES):.7f}, "
        f"b_sparse={float(B_SPARSE_BYTES):.7f}, d={float(D_BYTES):.1f}"
    )
    print("shape  dense(M,N,K)                 sparse(M,N,K)                bytes delta")
    for row in rows:
        print(
            f"{row['shape']:>2}     "
            f"({row['dense_m']:>6},{row['dense_n']:>5},{row['dense_k']:>5})  "
            f"({row['sparse_m']:>6},{row['sparse_n']:>5},{row['sparse_k']:>5})  "
            f"{row['byte_difference_pct']:+.4f}%"
        )


if __name__ == "__main__":
    main()
