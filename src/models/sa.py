from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from numba import njit, prange # type: ignore
from src.shared import (
    Result, Schedule, Span, heads_tails,
    get_makespan, get_critical_path,
    bidir, get_head, get_tail, move,
    get_job_sequence
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from src.shared import (
        Instance, Job, ScheduleType,
        OpID, OpArray, JobArray, MacArray,
        IntArray
    )

@njit(cache = True) # type: ignore
def get_random_n5_neighbor(
    cpath: OpArray,
    ranges: npt.NDArray[np.int32],
    num_blocks: int
) -> tuple[int, int]:

    max_cands = num_blocks * 2
    candidates = np.zeros((max_cands, 2), dtype=np.int32)
    c_count = 0

    max_fallbacks = len(cpath)
    fallback = np.zeros((max_fallbacks, 2), dtype=np.int32)
    f_count = 0

    for b_idx in range(num_blocks):
        start, end = int(ranges[b_idx, 0]), int(ranges[b_idx, 1])
        block = cpath[start:end]

        if len(block) < 2:
            continue

        for i in range(len(block) - 1):
            fallback[f_count, 0] = block[i]
            fallback[f_count, 1] = block[i+1]
            f_count += 1

        if b_idx > 0:
            candidates[c_count, 0] = block[0]
            candidates[c_count, 1] = block[1]
            c_count += 1

        if b_idx < num_blocks - 1:
            if c_count == 0 or not (candidates[c_count-1, 0] == block[-2] and candidates[c_count-1, 1] == block[-1]):
                candidates[c_count, 0] = block[-2]
                candidates[c_count, 1] = block[-1]
                c_count += 1

    if c_count > 0:
        idx = np.random.randint(0, c_count)
        return candidates[idx, 0], candidates[idx, 1]
    elif f_count > 0:
        idx = np.random.randint(0, f_count)
        return fallback[idx, 0], fallback[idx, 1]

    return -1, -1

@njit(cache = True) # type: ignore
def sa_loop(
    max_iterations: int, N: int, total_ops: int,
    op_ptime: IntArray, op_job_prev: OpArray, op_job_next: OpArray,
    op_mac_prev: OpArray, op_mac_next: OpArray, op_pos_in_mac: IntArray,
    op_mac: MacArray, initial_makespan: int
) -> tuple[int, int, int, OpArray, OpArray, IntArray, IntArray]:

    warmup_iters = min(100, max_iterations)
    diffs = np.zeros(warmup_iters, dtype=np.float64)

    curr_r, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)

    for i in range(warmup_iters):
        cp = get_critical_path(N, curr_r, op_ptime, op_job_prev, op_mac_prev)
        rg, nb = get_blocks(cp, op_mac)
        u, v = get_random_n5_neighbor(cp, rg, nb)

        if u == -1:
            diffs[i] = 1.0
            continue

        oblock = np.array([u, v], dtype=np.int32)
        new_Q = np.array([v, u], dtype=np.int32)

        move(oblock, new_Q, op_mac_prev, op_mac_next, op_pos_in_mac)
        r_new, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
        diffs[i] = abs(float(get_makespan(N, r_new, op_ptime) - initial_makespan))
        move(new_Q, oblock, op_mac_prev, op_mac_next, op_pos_in_mac)

    diffs_sorted = np.sort(diffs)
    idx_97 = int(warmup_iters * 0.97)
    d97 = diffs_sorted[idx_97]
    if d97 <= 0.0:
        d97 = 1.0

    best_makespan = initial_makespan
    best_mac_prev = op_mac_prev.copy()
    best_mac_next = op_mac_next.copy()
    best_pos_in_mac = op_pos_in_mac.copy()

    curr_makespan = initial_makespan

    t_0 = 20.0
    theta_0 = 0.5
    t = t_0
    gamma = 0.85

    r = curr_r.copy()
    iters_run = max_iterations

    for n in range(1, max_iterations + 1):
        cpath = get_critical_path(N, r, op_ptime, op_job_prev, op_mac_prev)
        ranges, num_blocks = get_blocks(cpath, op_mac)
        u, v = get_random_n5_neighbor(cpath, ranges, num_blocks)

        if u == -1:
            iters_run = n
            break

        oblock = np.array([u, v], dtype=np.int32)
        new_Q = np.array([v, u], dtype=np.int32)

        move(oblock, new_Q, op_mac_prev, op_mac_next, op_pos_in_mac)
        r_new, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
        new_makespan = get_makespan(N, r_new, op_ptime)

        diff = new_makespan - curr_makespan
        d_hat = diff / d97
        delta_n = (t_0 - theta_0) / (n ** gamma)

        accepted = False
        d_n1 = 0.0

        if diff <= 0:
            accepted = True
            d_n1 = float(diff)
        else:
            prob = np.exp(-diff / t) if t > 0 else 0.0
            if np.random.random() < prob:
                accepted = True
                d_n1 = (1.0 / prob) - 1.0 - d_hat
            else:
                accepted = False
                d_n1 = - max(0.0, 1.0 + d_hat)

        t = max(theta_0, t - d_n1 * delta_n)

        if accepted:
            curr_makespan = new_makespan
            r = r_new
            if curr_makespan < best_makespan:
                best_makespan = curr_makespan
                best_mac_prev = op_mac_prev.copy()
                best_mac_next = op_mac_next.copy()
                best_pos_in_mac = op_pos_in_mac.copy()
        else:
            move(new_Q, oblock, op_mac_prev, op_mac_next, op_pos_in_mac)

    best_r, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, best_mac_prev, best_mac_next)

    return iters_run, 0, 0, best_mac_prev, best_mac_next, best_pos_in_mac, best_r

@njit(parallel = True) # type: ignore
def optimize_population_parallel(
    pop_size: int, sa_iters: int, N: int, total_ops: int,
    op_ptime: IntArray, op_job_prev: OpArray, op_job_next: OpArray,
    op_mac: MacArray,
    pop_prev: npt.NDArray[np.int32], # 2D matrix: (pop_size, total_ops)
    pop_next: npt.NDArray[np.int32],
    pop_pos: npt.NDArray[np.int32],
    pop_mks: npt.NDArray[np.int32],  # 1D array: (pop_size)
    pop_r: npt.NDArray[np.int32],    # 2D matrix: (pop_size, total_ops)
    pop_iters: npt.NDArray[np.int32],
):

    for i in prange(pop_size):

        p_prev = pop_prev[i]
        p_next = pop_next[i]
        p_pos = pop_pos[i]
        mks_initial = pop_mks[i]

        iters_run, _, _, best_prev, best_next, best_pos, best_r = sa_loop(
            sa_iters, N, total_ops, op_ptime, op_job_prev, op_job_next,
            p_prev, p_next, p_pos, op_mac, mks_initial
        )

        pop_prev[i] = best_prev
        pop_next[i] = best_next
        pop_pos[i] = best_pos
        pop_r[i] = best_r
        pop_mks[i] = get_makespan(N, best_r, op_ptime)
        pop_iters[i] = iters_run

@njit(cache = True) # type: ignore
def crossover_kolonko(
    N: int, total_ops: int, num_macs: int, op_mac: MacArray,
    p1_mac_prev: OpArray, p1_mac_next: OpArray, r1: IntArray,
    p2_mac_prev: OpArray, p2_mac_next: OpArray, r2: IntArray,
    op_ptime: IntArray
) -> tuple[OpArray, OpArray, IntArray]:

    mks1 = get_makespan(N, r1, op_ptime)
    mks2 = get_makespan(N, r2, op_ptime)
    T = np.random.uniform(0, max(mks1, mks2))

    in_I = np.zeros(total_ops, dtype=np.bool_)
    for i in range(1, N + 1):
        if r1[i] <= T:
            in_I[i] = True

    child_mac_prev = np.full(total_ops, -1, dtype=np.int32)
    child_mac_next = np.full(total_ops, -1, dtype=np.int32)
    child_pos_in_mac = np.full(total_ops, -1, dtype=np.int32)

    for m in range(num_macs):

        curr1 = -1
        for i in range(1, N + 1):
            if op_mac[i] == m and p1_mac_prev[i] == -1:
                curr1 = i
                break

        seq_I = np.zeros(N, dtype=np.int32)
        idx_I = 0
        while curr1 != -1:
            if in_I[curr1]:
                seq_I[idx_I] = curr1
                idx_I += 1
            curr1 = p1_mac_next[curr1]

        curr2 = -1
        for i in range(1, N + 1):
            if op_mac[i] == m and p2_mac_prev[i] == -1:
                curr2 = i
                break

        seq_not_I = np.zeros(N, dtype=np.int32)
        idx_not_I = 0
        while curr2 != -1:
            if not in_I[curr2]:
                seq_not_I[idx_not_I] = curr2
                idx_not_I += 1
            curr2 = p2_mac_next[curr2]

        total_m_ops = idx_I + idx_not_I
        final_seq = np.zeros(total_m_ops, dtype=np.int32)

        for i in range(idx_I):
            final_seq[i] = seq_I[i]
        for i in range(idx_not_I):
            final_seq[idx_I + i] = seq_not_I[i]

        for i in range(total_m_ops):
            curr = final_seq[i]
            child_pos_in_mac[curr] = i
            if i > 0:
                prev = final_seq[i - 1]
                child_mac_next[prev] = curr
                child_mac_prev[curr] = prev

    return child_mac_prev, child_mac_next, child_pos_in_mac

@njit(cache = True) # type: ignore
def get_blocks(cpath: OpArray, op_mac: MacArray) -> tuple[np.ndarray, int]:

    n_cpath = len(cpath)
    ranges = np.zeros((n_cpath, 2), dtype=np.int32)
    num_blocks = 0

    if n_cpath == 0:
        return ranges, num_blocks

    start_idx = 0
    for i in range(1, n_cpath):
        curr_op = cpath[i]
        prev_op = cpath[i - 1]

        if op_mac[curr_op] != op_mac[prev_op]:
            if i - start_idx > 1:
                ranges[num_blocks, 0] = start_idx
                ranges[num_blocks, 1] = i
                num_blocks += 1
            start_idx = i

    if n_cpath - start_idx > 1:
        ranges[num_blocks, 0] = start_idx
        ranges[num_blocks, 1] = n_cpath
        num_blocks += 1

    return ranges, num_blocks

def get_schedule(N: int, r: IntArray, op_job: JobArray, op_mac: MacArray, op_ptime: IntArray) -> Schedule:
    schedule: ScheduleType = {}
    for op_id in range(1, N + 1):
        job_id = int(op_job[op_id])
        mac_id = int(op_mac[op_id])
        schedule[job_id, mac_id] = Span(int(r[op_id]), int(op_ptime[op_id]))
    return Schedule(schedule)

def init_simple_schedule(
    _: int,  total_ops: int, num_macs: int,
    op_mac: MacArray, job_sequence_dict: dict[Job, list[OpID]]
) -> tuple[OpArray, OpArray, IntArray]:

    op_mac_prev: OpArray = np.full(total_ops, -1, dtype=np.int32)
    op_mac_next: OpArray = np.full(total_ops, -1, dtype=np.int32)
    op_pos_in_mac: IntArray = np.full(total_ops, -1, dtype=np.int32)

    mac_last: IntArray = np.full(num_macs, -1, dtype=np.int32)
    mac_count: IntArray = np.zeros(num_macs, dtype=np.int32)

    for job_ops in job_sequence_dict.values():
        for op in job_ops:
            m = int(op_mac[op])

            if mac_last[m] != -1:
                op_mac_prev[op] = mac_last[m]
                op_mac_next[mac_last[m]] = op

            mac_last[m] = op
            op_pos_in_mac[op] = mac_count[m]
            mac_count[m] += 1

    return op_mac_prev, op_mac_next, op_pos_in_mac

@njit(cache = True) # type: ignore
def calc_pos_in_mac(N: int, num_macs: int, op_mac: MacArray, mac_prev: OpArray, mac_next: OpArray) -> IntArray:
    total_ops = N + 2
    pos_in_mac = np.full(total_ops, -1, dtype=np.int32)

    for m in range(num_macs):
        curr = -1
        for i in range(1, N + 1):
            if op_mac[i] == m and mac_prev[i] == -1:
                curr = i
                break

        idx = 0
        while curr != -1:
            pos_in_mac[curr] = idx
            idx += 1
            curr = mac_next[curr]

    return pos_in_mac

def solve(instance: Instance, pop_size: int = 12, generations: int = 25, sa_iters: int = 50000, use_bidir: bool = True, bidir_c: int = 3, time_limit: Optional[int] = None) -> Result:

    N = sum(len(seq) for seq in instance.jobseq.values()); total_ops = N + 2
    job_seq_dict, op_job_list, op_mac_list, op_ptime_list, op_job_prev_list, op_job_next_list = get_job_sequence(instance, N)
    op_job, op_mac, op_ptime = np.array(op_job_list, dtype=np.int32), np.array(op_mac_list, dtype=np.int32), np.array(op_ptime_list, dtype=np.int32)
    op_job_prev, op_job_next = np.array(op_job_prev_list, dtype=np.int32), np.array(op_job_next_list, dtype=np.int32)

    p_prev = np.zeros((pop_size, total_ops), dtype=np.int32)
    p_next = np.zeros((pop_size, total_ops), dtype=np.int32)
    p_pos = np.zeros((pop_size, total_ops), dtype=np.int32)
    p_r = np.zeros((pop_size, total_ops), dtype=np.int32)
    p_mks = np.zeros(pop_size, dtype=np.int32)

    if use_bidir:
        print(">> Initializing population with BIDIR...")

        job_head = np.zeros(total_ops, dtype=np.int32)
        job_tail = np.zeros(total_ops, dtype=np.int32)
        for i in range(1, N + 1):
            job_head[i] = get_head(i, op_job_prev, op_ptime)
            job_tail[i] = get_tail(i, N, op_job_next, op_ptime)

        mac_counts = np.zeros(instance.macs, dtype=np.int32)
        for i in range(1, N + 1):
            mac_counts[op_mac[i]] += 1
        ops_per_mac = np.max(mac_counts)

        mac_sorted_tail = np.full((instance.macs, ops_per_mac), -1, dtype=np.int32)
        mac_sorted_head = np.full((instance.macs, ops_per_mac), -1, dtype=np.int32)

        for m in range(instance.macs):
            ops_in_m = [i for i in range(1, N + 1) if op_mac[i] == m]

            ops_by_tail = sorted(ops_in_m, key=lambda x: job_tail[x], reverse=True)
            for i, op in enumerate(ops_by_tail): mac_sorted_tail[m, i] = op

            ops_by_head = sorted(ops_in_m, key=lambda x: job_head[x], reverse=True)
            for i, op in enumerate(ops_by_head): mac_sorted_head[m, i] = op

        initial_S = np.array([seq[0] for seq in job_seq_dict.values()], dtype=np.int32)
        initial_T = np.array([seq[-1] for seq in job_seq_dict.values()], dtype=np.int32)

        for i in range(pop_size):
            b_prev, b_next = bidir(
                N, instance.macs, bidir_c, op_mac, op_ptime, op_job_prev, op_job_next,
                job_head, job_tail, mac_sorted_tail, mac_sorted_head, int(ops_per_mac),
                initial_S, initial_T
            )
            b_pos = calc_pos_in_mac(N, instance.macs, op_mac, b_prev, b_next)
            r_init, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, b_prev, b_next)

            p_prev[i], p_next[i], p_pos[i], p_r[i], p_mks[i] = b_prev, b_next, b_pos, r_init, get_makespan(N, r_init, op_ptime)

    else:
        print(">> Initializing population with SIMPLE SCHEDULE...")
        b_prev, b_next, b_pos = init_simple_schedule(N, total_ops, instance.macs, op_mac, job_seq_dict)
        r_init, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, b_prev, b_next)
        mks_init = get_makespan(N, r_init, op_ptime)

        for i in range(pop_size):
            p_prev[i], p_next[i], p_pos[i], p_r[i], p_mks[i] = b_prev.copy(), b_next.copy(), b_pos.copy(), r_init.copy(), mks_init

    b_prev, b_next, b_pos = init_simple_schedule(N, total_ops, instance.macs, op_mac, job_seq_dict)
    r_init, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, b_prev, b_next)
    mks_init = get_makespan(N, r_init, op_ptime)

    for i in range(pop_size):
        p_prev[i], p_next[i], p_pos[i], p_r[i], p_mks[i] = b_prev.copy(), b_next.copy(), b_pos.copy(), r_init.copy(), mks_init

    p_iters = np.zeros(pop_size, dtype=np.int32)
    total_iters_run = 0
    history_best = np.zeros(generations, dtype=np.int32)
    history_mean = np.zeros(generations, dtype=np.float64)
    print(f">> Starting SAGen Parallel: {pop_size} cores | {generations} gens")
    t_start = time.perf_counter()

    for gen in range(1, generations + 1):
        optimize_population_parallel(pop_size, sa_iters, N, total_ops, op_ptime, op_job_prev, op_job_next, op_mac, p_prev, p_next, p_pos, p_mks, p_r, p_iters)
        total_iters_run += int(np.sum(p_iters))

        best_idx = np.argmin(p_mks)
        print(f"Gen {gen}/{generations} | Best: {p_mks[best_idx]}")

        history_best[gen - 1] = p_mks[best_idx]
        history_mean[gen - 1] = np.mean(p_mks)

        if instance.optimum and p_mks[best_idx] == instance.optimum:
            print(f">> Optimal solution found: {p_mks[best_idx]}! Stopping early at generation {gen}.")

            history_best = history_best[:gen]
            history_mean = history_mean[:gen]
            break

        if time_limit and time.perf_counter() - t_start >= time_limit:
            print(f">> Time limite reached ({time_limit}s) at gen {gen}. Stopping search...")
            history_best = history_best[:gen]
            history_mean = history_mean[:gen]
            break

        new_prev, new_next, new_pos, new_r, new_mks = np.zeros_like(p_prev), np.zeros_like(p_next), np.zeros_like(p_pos), np.zeros_like(p_r), np.zeros_like(p_mks)
        new_prev[0], new_next[0], new_pos[0], new_r[0], new_mks[0] = p_prev[best_idx], p_next[best_idx], p_pos[best_idx], p_r[best_idx], p_mks[best_idx]

        for i in range(1, pop_size):
            idx1, idx2 = np.random.randint(0, pop_size), np.random.randint(0, pop_size)
            c_prev, c_next, c_pos = crossover_kolonko(N, total_ops, instance.macs, op_mac, p_prev[idx1], p_next[idx1], p_r[idx1], p_prev[idx2], p_next[idx2], p_r[idx2], op_ptime)
            new_prev[i], new_next[i], new_pos[i] = c_prev, c_next, c_pos
            r_c, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, c_prev, c_next)
            new_r[i], new_mks[i] = r_c, get_makespan(N, r_c, op_ptime)

        p_prev, p_next, p_pos, p_r, p_mks = new_prev, new_next, new_pos, new_r, new_mks

    t_end = time.perf_counter()
    best_final = np.argmin(p_mks)
    return Result(
        model = "SA",
        instance = instance,
        schedule = get_schedule(N, p_r[best_final], op_job, op_mac, op_ptime),
        history_LB = [round(makespan) for makespan in history_mean],
        history_UB = history_best.tolist(),
        history_time_s = [t_end - t_start],
    )
