from __future__ import annotations

import time
import numpy as np

from numba import njit # type: ignore
from src.shared import (
    Result, get_job_sequence,
    heads_tails, INT32_MAX,
    get_critical_path, get_makespan,
    get_schedule, bidir, get_head,
    get_tail, move
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from src.shared import (
        Instance, Mac, OpID,
        IntArray, OpArray, MacArray,
    )

@njit(cache = True) # type: ignore
def lpath2(
    q0: int, q1: int, o0: int, o1: int,
    r: IntArray, q: IntArray,
    op_ptime: IntArray, op_job_prev: OpArray, op_job_next: OpArray,
    op_mac_prev: OpArray, op_mac_next: OpArray
) -> int:

    mac_prev_first = op_mac_prev[o0]
    r_mac_prev = r[mac_prev_first] + op_ptime[mac_prev_first] if mac_prev_first != -1 else 0
    prev_job_0 = op_job_prev[q0]
    r_job_0 = r[prev_job_0] + op_ptime[prev_job_0] if prev_job_0 != -1 else 0
    l0 = max(r_job_0, r_mac_prev)

    prev_job_1 = op_job_prev[q1]
    r_job_1 = r[prev_job_1] + op_ptime[prev_job_1] if prev_job_1 != -1 else 0
    l1 = max(r_job_1, l0 + op_ptime[q0])

    mac_next_last = op_mac_next[o1]
    q_mac_ext = q[mac_next_last] + op_ptime[mac_next_last] if mac_next_last != -1 else 0

    next_job_0 = op_job_next[q0]
    q_job_0 = q[next_job_0] + op_ptime[next_job_0] if next_job_0 != -1 else 0
    cost0 = l0 + op_ptime[q0] + q_job_0

    next_job_1 = op_job_next[q1]
    q_job_1 = q[next_job_1] + op_ptime[next_job_1] if next_job_1 != -1 else 0
    q_eff_1 = max(q_job_1, q_mac_ext)
    cost1 = l1 + op_ptime[q1] + q_eff_1

    return max(cost0, cost1)

@njit(cache = True) # type: ignore
def lpath3(
    q0: int, q1: int, q2: int,
    o0: int, o1: int, o2: int,
    r: IntArray, q: IntArray,
    op_ptime: IntArray, op_job_prev: OpArray, op_job_next: OpArray,
    op_mac_prev: OpArray, op_mac_next: OpArray
) -> int:

    mac_prev_first = op_mac_prev[o0]
    r_mac_prev = r[mac_prev_first] + op_ptime[mac_prev_first] if mac_prev_first != -1 else 0
    prev_job_0 = op_job_prev[q0]
    r_job_0 = r[prev_job_0] + op_ptime[prev_job_0] if prev_job_0 != -1 else 0
    l0 = max(r_job_0, r_mac_prev)

    prev_job_1 = op_job_prev[q1]
    r_job_1 = r[prev_job_1] + op_ptime[prev_job_1] if prev_job_1 != -1 else 0
    l1 = max(r_job_1, l0 + op_ptime[q0])

    prev_job_2 = op_job_prev[q2]
    r_job_2 = r[prev_job_2] + op_ptime[prev_job_2] if prev_job_2 != -1 else 0
    l2 = max(r_job_2, l1 + op_ptime[q1])

    mac_next_last = op_mac_next[o2]
    q_mac_ext = q[mac_next_last] + op_ptime[mac_next_last] if mac_next_last != -1 else 0

    next_job_0 = op_job_next[q0]
    q_job_0 = q[next_job_0] + op_ptime[next_job_0] if next_job_0 != -1 else 0
    cost0 = l0 + op_ptime[q0] + q_job_0

    next_job_1 = op_job_next[q1]
    q_job_1 = q[next_job_1] + op_ptime[next_job_1] if next_job_1 != -1 else 0
    cost1 = l1 + op_ptime[q1] + q_job_1

    next_job_2 = op_job_next[q2]
    q_job_2 = q[next_job_2] + op_ptime[next_job_2] if next_job_2 != -1 else 0
    q_eff_2 = max(q_job_2, q_mac_ext)
    cost2 = l2 + op_ptime[q2] + q_eff_2

    return max(cost0, max(cost1, cost2))

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

@njit(cache = True) # type: ignore
def estim(
    u: int, v: int, u_mac_prev: int, v_mac_next: int,
    r: IntArray, q: IntArray, op_ptime: IntArray,
    op_job_prev: OpArray, op_job_next: OpArray,
    op_mac_prev: OpArray, op_mac_next: OpArray
) -> tuple[int, OpArray, OpArray]:

    best_cost = INT32_MAX
    best_tb = 999
    best_case = 0

    e1 = lpath2(v, u, u, v, r, q, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
    if e1 < best_cost or (e1 == best_cost and 1 < best_tb):
        best_cost, best_tb, best_case = e1, 1, 1

    if u_mac_prev != -1:
        e2 = lpath3(v, u_mac_prev, u, u_mac_prev, u, v, r, q, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
        if e2 < best_cost or (e2 == best_cost and 2 < best_tb):
            best_cost, best_tb, best_case = e2, 2, 2

        e3 = lpath3(v, u, u_mac_prev, u_mac_prev, u, v, r, q, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
        if e3 < best_cost or (e3 == best_cost and 3 < best_tb):
            best_cost, best_tb, best_case = e3, 3, 3

    if v_mac_next != -1:
        e4 = lpath3(v, v_mac_next, u, u, v, v_mac_next, r, q, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
        if e4 < best_cost or (e4 == best_cost and 4 < best_tb):
            best_cost, best_tb, best_case = e4, 4, 4

        e5 = lpath3(v_mac_next, v, u, u, v, v_mac_next, r, q, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
        if e5 < best_cost or (e5 == best_cost and 5 < best_tb):
            best_cost, best_tb, best_case = e5, 5, 5

    if best_case == 1:
        best_oblock = np.array([u, v], dtype = np.int32)
        best_Q = np.array([v, u], dtype = np.int32)
    elif best_case == 2:
        best_oblock = np.array([u_mac_prev, u, v], dtype = np.int32)
        best_Q = np.array([v, u_mac_prev, u], dtype = np.int32)
    elif best_case == 3:
        best_oblock = np.array([u_mac_prev, u, v], dtype = np.int32)
        best_Q = np.array([v, u, u_mac_prev], dtype = np.int32)
    elif best_case == 4:
        best_oblock = np.array([u, v, v_mac_next], dtype = np.int32)
        best_Q = np.array([v, v_mac_next, u], dtype = np.int32)
    elif best_case == 5:
        best_oblock = np.array([u, v, v_mac_next], dtype = np.int32)
        best_Q = np.array([v_mac_next, v, u], dtype = np.int32)
    else:
        raise RuntimeError("UNREACHABLE")

    return best_cost, best_oblock, best_Q

def get_move_id(u: OpID, v: OpID) -> tuple[int, int]:
    return min(u, v), max(u, v)

@njit(cache = True) # type: ignore
def ts_loop(
    max_iterations: int, N: int, total_ops: int,
    op_ptime: IntArray, op_job_prev: OpArray, op_job_next: OpArray,
    op_mac_prev: OpArray, op_mac_next: OpArray, op_pos_in_mac: IntArray,
    op_mac: MacArray, initial_makespan: int,
    tenure_min: int, tenure_max: int, max_fails: int, optimum: Optional[int]
) -> tuple[int, IntArray, IntArray, IntArray, IntArray, OpArray, OpArray]:

    best_makespan = initial_makespan

    best_mac_prev: OpArray = op_mac_prev.copy()
    best_mac_next: OpArray = op_mac_next.copy()
    best_pos_in_mac: IntArray = op_pos_in_mac.copy()

    curr_history = np.zeros(max_iterations + 1, dtype = np.int32)
    best_history = np.zeros(max_iterations + 1, dtype = np.int32)
    curr_history[0] = initial_makespan
    best_history[0] = initial_makespan

    tabu_u = np.full(tenure_max, -1, dtype = np.int32)
    tabu_v = np.full(tenure_max, -1, dtype = np.int32)
    tabu_expire = np.zeros(tenure_max, dtype = np.int32)
    tabu_ptr = 0

    current_tenure = np.random.randint(tenure_min, tenure_max)
    fails = 0
    iters_run = 0

    r, q = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)

    five_percent = max_iterations * 0.05
    for iteration in range(1, max_iterations + 1):

        if not iteration % five_percent:
            print(iteration, "/", max_iterations)

        cpath = get_critical_path(N, r, op_ptime, op_job_prev, op_mac_prev)
        ranges, num_blocks = get_blocks(cpath, op_mac)

        best_move_cost = INT32_MAX
        best_u, best_v = -1, -1
        best_oblock = np.empty(0, dtype = np.int32)
        best_Q = np.empty(0, dtype = np.int32)

        fallback_cost = INT32_MAX
        fallback_u, fallback_v = -1, -1
        fallback_oblock = np.empty(0, dtype = np.int32)
        fallback_Q = np.empty(0, dtype = np.int32)

        for b_idx in range(num_blocks):
            start, end = ranges[b_idx, 0], ranges[b_idx, 1]
            block = cpath[start:end]

            is_first_block = (b_idx == 0)
            is_last_block = (b_idx == num_blocks - 1)
            force_eval = (num_blocks == 1)
            eval_start = (not is_first_block) or force_eval
            eval_end = (not is_last_block) or force_eval

            if eval_start:
                u, v = int(block[0]), int(block[1])
                cost, oblock, bQ = estim(u, v, op_mac_prev[u], op_mac_next[v], r, q, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)

                is_tabu = False
                for t_idx in range(tenure_max):
                    if tabu_expire[t_idx] >= iteration:
                        if (tabu_u[t_idx] == u and tabu_v[t_idx] == v) or (tabu_u[t_idx] == v and tabu_v[t_idx] == u):
                            is_tabu = True
                            break

                if not is_tabu or cost < best_makespan:
                    if cost < best_move_cost:
                        best_move_cost, best_u, best_v, best_oblock, best_Q = cost, u, v, oblock, bQ

                if cost < fallback_cost:
                    fallback_cost, fallback_u, fallback_v, fallback_oblock, fallback_Q = cost, u, v, oblock, bQ

            if eval_end:
                if len(block) > 2 or (len(block) == 2 and not eval_start):
                    w, z = int(block[-2]), int(block[-1])
                    cost, oblock, bQ = estim(w, z, op_mac_prev[w], op_mac_next[z], r, q, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)

                    is_tabu = False
                    for t_idx in range(tenure_max):
                        if tabu_expire[t_idx] >= iteration:
                            if (tabu_u[t_idx] == w and tabu_v[t_idx] == z) or (tabu_u[t_idx] == z and tabu_v[t_idx] == w):
                                is_tabu = True
                                break

                    if not is_tabu or cost < best_makespan:
                        if cost < best_move_cost:
                            best_move_cost, best_u, best_v, best_oblock, best_Q = cost, w, z, oblock, bQ

                    if cost < fallback_cost:
                        fallback_cost, fallback_u, fallback_v, fallback_oblock, fallback_Q = cost, w, z, oblock, bQ

        if best_move_cost == INT32_MAX:
            if fallback_cost != INT32_MAX:
                best_move_cost, best_u, best_v, best_oblock, best_Q = fallback_cost, fallback_u, fallback_v, fallback_oblock, fallback_Q
            else:
                print(">> Breaking out of ts_loop")
                break

        tabu_u[tabu_ptr] = best_u
        tabu_v[tabu_ptr] = best_v
        tabu_expire[tabu_ptr] = iteration + current_tenure
        tabu_ptr = (tabu_ptr + 1) % tenure_max

        if np.random.random() < 0.1:
            current_tenure = np.random.randint(tenure_min, tenure_max)

        move(best_oblock, best_Q, op_mac_prev, op_mac_next, op_pos_in_mac)
        r, q = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
        curr_makespan = get_makespan(N, r, op_ptime)

        if curr_makespan < best_makespan:
            best_makespan = curr_makespan
            best_mac_prev = op_mac_prev.copy()
            best_mac_next = op_mac_next.copy()
            best_pos_in_mac = op_pos_in_mac.copy()
            fails = 0
        else:
            fails += 1

        curr_history[iteration] = curr_makespan
        best_history[iteration] = best_makespan
        iters_run = iteration

        if optimum and optimum == best_makespan:
            print(">> Breaking out of ts_loop :: found optimum")
            break

        if fails >= max_fails:
            op_mac_prev = best_mac_prev.copy()
            op_mac_next = best_mac_next.copy()
            op_pos_in_mac = best_pos_in_mac.copy()
            r, q = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)

            tabu_expire.fill(0)
            current_tenure = np.random.randint(tenure_min, tenure_max)
            fails = 0

    best_r, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, best_mac_prev, best_mac_next)

    return iters_run, curr_history[:iters_run+1], best_history[:iters_run+1], best_mac_prev, best_mac_next, best_pos_in_mac, best_r

# c is the bidir's hiperparameter
# tenure_min/max are the min and max lengths possible for the tabu list
# max_fails is the max number of iters w/o improvement till resetting to best solution
def solve(
    instance: Instance, max_iterations: int, c: int = 3,
    tenure_min: int = 10, tenure_max: int = 15, max_fails: int = 5000
) -> Result:

    N = sum(len(seq) for seq in instance.jobseq.values())
    total_ops = N + 2
    num_macs = instance.macs

    job_seq_dict, op_job_list, op_mac_list, op_ptime_list, op_job_prev_list, op_job_next_list = get_job_sequence(instance, N)

    op_job = np.array(op_job_list, dtype = np.int32)
    op_mac = np.array(op_mac_list, dtype = np.int32)
    op_ptime = np.array(op_ptime_list, dtype = np.int32)
    op_job_prev = np.array(op_job_prev_list, dtype = np.int32)
    op_job_next = np.array(op_job_next_list, dtype = np.int32)

    job_head = np.zeros(total_ops, dtype = np.int32)
    job_tail = np.zeros(total_ops, dtype = np.int32)
    for i in range(1, N + 1):
        job_head[i] = get_head(i, op_job_prev, op_ptime)
        job_tail[i] = get_tail(i, N, op_job_next, op_ptime)

    mac_ops: dict[Mac, list[OpID]] = {m: [] for m in range(num_macs)}
    for op in range(1, N + 1):
        mac_ops[op_mac[op]].append(op)

    ops_per_mac = max(len(ops) for ops in mac_ops.values())
    mac_sorted_tail = np.full((num_macs, ops_per_mac), -1, dtype = np.int32)
    mac_sorted_head = np.full((num_macs, ops_per_mac), -1, dtype = np.int32)

    for m in range(num_macs):
        sorted_by_tail = sorted(mac_ops[m], key = lambda o: op_ptime[o] + job_tail[o], reverse = True)
        sorted_by_head = sorted(mac_ops[m], key = lambda o: op_ptime[o] + job_head[o], reverse = True)

        for i, op in enumerate(sorted_by_tail):
            mac_sorted_tail[m, i] = op
        for i, op in enumerate(sorted_by_head):
            mac_sorted_head[m, i] = op

    initial_S = np.array([seq[0] for seq in job_seq_dict.values() if seq], dtype = np.int32)
    initial_T = np.array([seq[-1] for seq in job_seq_dict.values() if seq], dtype = np.int32)

    t0 = time.perf_counter()
    op_mac_prev, op_mac_next = bidir(
        N, num_macs, c, op_mac, op_ptime, op_job_prev, op_job_next,
        job_head, job_tail, mac_sorted_tail, mac_sorted_head,
        ops_per_mac, initial_S, initial_T
    )
    t1 = time.perf_counter()
    bidir_t = t1 - t0

    op_pos_in_mac = np.full(total_ops, -1, dtype = np.int32)
    for m in range(num_macs):
        curr = -1
        for op in mac_ops[m]:
            if op_mac_prev[op] == -1:
                curr = op
                break

        pos = 0
        while curr != -1:
            op_pos_in_mac[curr] = pos
            pos += 1
            curr = op_mac_next[curr]

    r, _ = heads_tails(total_ops, op_ptime, op_job_prev, op_job_next, op_mac_prev, op_mac_next)
    initial_makespan = get_makespan(N, r, op_ptime)

    print(">> Starting Tabu Search...")
    t0 = time.perf_counter()
    _, curr_hist, best_hist, _, _, _, best_r = ts_loop(
        max_iterations, N, total_ops, op_ptime, op_job_prev, op_job_next,
        op_mac_prev, op_mac_next, op_pos_in_mac, op_mac, initial_makespan,
        tenure_min, tenure_max, max_fails, instance.optimum
    )
    t1 = time.perf_counter()
    ts_loop_t = t1 - t0

    return Result(
        model = "TS",
        instance = instance,
        schedule = get_schedule(N, best_r, op_job, op_mac, op_ptime),
        history_LB = curr_hist.tolist(),
        history_UB = best_hist.tolist(),
        history_time_s = [bidir_t + ts_loop_t],
    )
