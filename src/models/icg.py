# type: ignore

import math
import time
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from src.shared import (
    get_job_sequence, INT32_MAX, Instance, Result, get_schedule,
    build_mac_links, topological_sort, heads_tails, get_makespan, Schedule,
    Mac
)

def safe_floor_lb(raw: float) -> int:
    floored = math.floor(raw)
    if abs(raw - (floored + 1)) < 1e-6:
        return floored + 1
    return floored

def z_expr(model, oi: int, oj: int, z_idx_set: set):
    if (oi, oj) in z_idx_set:
        return model.Z[oi, oj]
    if (oj, oi) in z_idx_set:
        return 1 - model.Z[oj, oi]
    raise KeyError(f"Variable Z for ({oi}, {oj}) doesn't exist")

def get_adaptive_gap(best_lb: int, best_ub: int) -> float:

    relative_gap = (best_ub - best_lb) / max(best_lb, 1)

    if relative_gap < 0.01:
        return 1e-4
    if relative_gap < 0.05:
        return 0.01

    return min(0.05, max(0.005, relative_gap * 0.3))

def add_cuts(
    m: Mac, start_times: dict, model,
    mac_to_ops: dict, op_ptime: list,
    EST: dict, LST: dict,
    z_idx_set: set,
    active_pairs: set,
    transitivity_triples_added: set
) -> int:

    ops = mac_to_ops[m]
    ops_sorted = sorted(ops, key=lambda o: (start_times[o], o))
    new_pairs = []

    for i in range(len(ops_sorted)):
        for j in range(i + 1, len(ops_sorted)):
            oi, oj = ops_sorted[i], ops_sorted[j]
            key = (min(oi, oj), max(oi, oj))

            if key in active_pairs:
                continue

            collision = (start_times[oi] + int(op_ptime[oi]) > start_times[oj] + 1e-6)

            if collision:
                M_ij = max(0, LST[oi] + int(op_ptime[oi]) - EST[oj])
                M_ji = max(0, LST[oj] + int(op_ptime[oj]) - EST[oi])
                z = z_expr(model, oi, oj, z_idx_set)

                model.constraints.add(model.S[oj] >= model.S[oi] + int(op_ptime[oi]) - M_ij * (1 - z))
                model.constraints.add(model.S[oi] >= model.S[oj] + int(op_ptime[oj]) - M_ji * z)

                active_pairs.add(key)
                new_pairs.append((oi, oj))

    for (oi, oj) in new_pairs:
        for ok in ops:
            if ok == oi or ok == oj:
                continue

            a, b, c = sorted([oi, oj, ok])
            triple = (a, b, c)

            if triple in transitivity_triples_added:
                continue

            has_ab = (a, b) in active_pairs
            has_bc = (b, c) in active_pairs
            has_ac = (a, c) in active_pairs

            if has_ab and has_bc and has_ac:
                z_ab = z_expr(model, a, b, z_idx_set)
                z_bc = z_expr(model, b, c, z_idx_set)
                z_ac = z_expr(model, a, c, z_idx_set)

                model.constraints.add(z_ab + z_bc - z_ac <= 1)
                model.constraints.add(z_ab + z_bc - z_ac >= 0)
                transitivity_triples_added.add(triple)

    return len(new_pairs)

def add_full_machine_constraints(
    m: Mac, model, mac_to_ops: dict,
    op_ptime: list, EST: dict, LST: dict,
    z_idx_set: set,
    active_pairs: set,
    transitivity_triples_added: set
) -> None:

    ops = mac_to_ops[m]
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            oi, oj = ops[i], ops[j]
            key = (min(oi, oj), max(oi, oj))
            if key in active_pairs:
                continue

            M_ij = max(0, LST[oi] + int(op_ptime[oi]) - EST[oj])
            M_ji = max(0, LST[oj] + int(op_ptime[oj]) - EST[oi])
            z = z_expr(model, oi, oj, z_idx_set)

            model.constraints.add(model.S[oj] >= model.S[oi] + int(op_ptime[oi]) - M_ij * (1 - z))
            model.constraints.add(model.S[oi] >= model.S[oj] + int(op_ptime[oj]) - M_ji * z)
            active_pairs.add(key)

            for k in range(j + 1, len(ops)):
                ok = ops[k]

                a, b, c = sorted([oi, oj, ok])
                triple = (a, b, c)

                if triple not in transitivity_triples_added:
                    z_ab = z_expr(model, a, b, z_idx_set)
                    z_bc = z_expr(model, b, c, z_idx_set)
                    z_ac = z_expr(model, a, c, z_idx_set)
                    model.constraints.add(z_ab + z_bc - z_ac <= 1)
                    model.constraints.add(z_ab + z_bc - z_ac >= 0)
                    transitivity_triples_added.add(triple)

def warmstart_z_from_heuristic(model, mac_seq: dict, z_idx_set: set) -> None:
    for _mac, ops_ordered in mac_seq.items():
        for rank_i, oi in enumerate(ops_ordered):
            for rank_j in range(rank_i + 1, len(ops_ordered)):
                oj = ops_ordered[rank_j]
                if (oi, oj) in z_idx_set:
                    model.Z[oi, oj].set_value(1)
                elif (oj, oi) in z_idx_set:
                    model.Z[oj, oi].set_value(0)

def solve(instance: Instance, time_limit: float, solver_name: str) -> Result:

    start_time_global = time.perf_counter()

    N = sum(len(seq) for seq in instance.jobseq.values())
    (job_sequence, op_job, op_mac, op_ptime,
     op_job_prev, op_job_next) = get_job_sequence(instance, N)
    total_ops = len(op_job)

    np_op_job_next = np.array(op_job_next, dtype=np.int32)
    np_op_job_prev = np.array(op_job_prev, dtype=np.int32)
    np_op_ptime    = np.array(op_ptime,    dtype=np.int32)

    mac_to_ops: dict[int, list[int]] = {}
    for op in range(1, total_ops - 1):
        mac = int(op_mac[op])
        mac_to_ops.setdefault(mac, []).append(op)

    C_UB = sum(int(op_ptime[op]) for op in range(1, total_ops - 1))
    EST: dict[int, int] = {}
    LST: dict[int, int] = {}

    for _job, seq in job_sequence.items():
        acc = 0
        for op in seq:
            EST[op] = acc
            acc += int(op_ptime[op])
        tail = 0
        for op in reversed(seq):
            tail += int(op_ptime[op])
            LST[op] = C_UB - tail

    workloads = {m: sum(int(op_ptime[o]) for o in ops) for m, ops in mac_to_ops.items()}
    jobloads = {j: sum(int(op_ptime[o]) for o in s) for j, s in job_sequence.items()}
    global_lower_bound = max(max(workloads.values()), max(jobloads.values()))

    model = pyo.ConcreteModel()
    model.S = pyo.Var(range(1, total_ops - 1), within=pyo.NonNegativeReals)
    model.C_max = pyo.Var(within=pyo.NonNegativeReals)
    model.constraints = pyo.ConstraintList()
    model.constraints.add(model.C_max >= global_lower_bound)

    for ops in job_sequence.values():
        for i in range(len(ops) - 1):
            model.constraints.add(model.S[ops[i + 1]] >= model.S[ops[i]] + int(op_ptime[ops[i]]))
        model.constraints.add(model.C_max >= model.S[ops[-1]] + int(op_ptime[ops[-1]]))

    z_idx = [(ops[i], ops[j]) for ops in mac_to_ops.values() for i in range(len(ops)) for j in range(i + 1, len(ops))]
    z_idx_set = set(z_idx)
    model.Z = pyo.Var(z_idx, within=pyo.Binary)
    model.obj = pyo.Objective(expr=model.C_max, sense=pyo.minimize)

    active_pairs: set[tuple] = set()
    transitivity_triples_added: set[tuple] = set()
    all_mac_ids = list(mac_to_ops.keys())

    sorted_macs = sorted(all_mac_ids, key=lambda m: workloads[m], reverse=True)
    for m in sorted_macs[:2]:
        add_full_machine_constraints(
            m, model, mac_to_ops, op_ptime, EST, LST,
            z_idx_set, active_pairs, transitivity_triples_added
        )
        print(f">> Fully adding machine {m}, load (w): {workloads[m]}")

    initial_mac_seq = {m: sorted(ops) for m, ops in mac_to_ops.items()}
    mac_links_init = build_mac_links(initial_mac_seq, total_ops)
    _, op_mac_next_init, _ = mac_links_init
    topo_init, _ = topological_sort(total_ops, np_op_job_next, op_mac_next_init)
    r_init, _ = heads_tails(
        total_ops, np_op_ptime, np_op_job_prev, np_op_job_next,
        *build_mac_links(initial_mac_seq, total_ops)[:2], topo_init
    )

    best_upper_bound = get_makespan(N, r_init, np_op_ptime)
    best_feasible_r  = r_init.copy()

    print(f">> Initial LB: {global_lower_bound}")
    print(f">> Initial UB: {best_upper_bound}")

    warmstart_z_from_heuristic(model, initial_mac_seq, z_idx_set)

    iteration = 0
    best_lower_bound = global_lower_bound
    solver = SolverFactory(solver_name)
    forced_gap = None

    while True:
        iteration += 1
        rem_time = time_limit - (time.perf_counter() - start_time_global)

        print(f"\n{'='*70}")
        print(f"Iteration {iteration}")
        print(f"  Active pairs: {len(active_pairs)}")
        print(f"  LB: {best_lower_bound} | UB: {best_upper_bound}")
        print(f"  GAP: {100*(best_upper_bound-best_lower_bound)/max(best_lower_bound,1):.2f}%")
        print(f"  Rem time (s): {rem_time:.1f}s")

        if rem_time <= 2.0:
            print(">> ICG timeout before master problem")
            break

        gap_tolerance = forced_gap if forced_gap else get_adaptive_gap(best_lower_bound, best_upper_bound)
        forced_gap = None
        print(f">> GAP set to: {gap_tolerance*100:.2f}%")

        if solver_name.upper() == "MOSEK":
            solver.options['dparam.optimizer_max_time'] = rem_time
            solver.options['dparam.mio_max_time']       = rem_time
            solver.options['dparam.mio_tol_rel_gap']    = gap_tolerance
        elif solver_name.upper() == "APPSI_HIGHS":
            solver.options['time_limit']   = rem_time
            solver.options['mip_rel_gap']  = gap_tolerance
        else:
            raise RuntimeError("Solver not implemented")

        if best_upper_bound < INT32_MAX:
            model.C_max.setub(best_upper_bound + 0.1)

        results = solver.solve(model, tee=False, warmstart=True)
        status  = results.solver.termination_condition

        timeout_hit = (status == pyo.TerminationCondition.maxTimeLimit) or \
                      (status == pyo.TerminationCondition.unknown and "time" in str(results.solver.message).lower())

        if timeout_hit:
            try:
                rescued_lb = results.problem[0].lower_bound
                best_lower_bound = max(best_lower_bound, safe_floor_lb(rescued_lb))
            except Exception: pass
            print(f">> ICG timeout on master problem")
            break

        try:
            master_primal = safe_floor_lb(pyo.value(model.C_max))
        except (ValueError, TypeError):
            break

        raw = master_primal
        try:
            if len(results.problem) > 0 and getattr(results.problem[0], 'lower_bound', None) is not None:
                raw = results.problem[0].lower_bound
        except Exception: pass

        master_dual = safe_floor_lb(raw)
        if master_dual > best_lower_bound:
            best_lower_bound = master_dual

        print(f">> Master: {master_primal}")
        print(f">> Master LB: {best_lower_bound}")

        if best_lower_bound >= best_upper_bound:
            print(f">> Found optimum: LB = UB = {best_upper_bound}")
            return Result(
                model = "ICG",
                solver = solver_name,
                instance = instance,
                schedule = get_schedule(N, best_feasible_r, op_job, op_mac, op_ptime) if best_feasible_r is not None else Schedule({}),
                history_LB = [best_lower_bound],
                history_time_s = [time.perf_counter() - start_time_global]
            )

        start_times = {op: pyo.value(model.S[op]) for op in range(1, total_ops - 1)}

        to_add = []
        overlapping_ops = set()

        for m in all_mac_ids:
            ops_s = sorted(mac_to_ops[m], key=lambda o: (start_times[o], o))
            n_conflicts_nuevos = 0
            for i in range(len(ops_s)):
                for j in range(i + 1, len(ops_s)):
                    oi, oj = ops_s[i], ops_s[j]

                    if start_times[oi] + int(op_ptime[oi]) > start_times[oj] + 1e-6:
                        overlapping_ops.add(oi)
                        overlapping_ops.add(oj)

                        key = (min(oi, oj), max(oi, oj))
                        if key not in active_pairs:
                            n_conflicts_nuevos += 1

            if n_conflicts_nuevos > 0:
                to_add.append((m, n_conflicts_nuevos))

        all_ops = set(range(1, total_ops - 1))
        non_overlapping_ops = all_ops - overlapping_ops

        highlight_sanas = [(int(op_job[o]), int(op_mac[o])) for o in non_overlapping_ops]

        mac_seq = {m: sorted(ops, key=lambda o: start_times[o]) for m, ops in mac_to_ops.items()}
        _, op_mac_next, _ = build_mac_links(mac_seq, total_ops)
        topo, has_cycle = topological_sort(total_ops, np_op_job_next, op_mac_next)

        if not has_cycle:
            r, _ = heads_tails(
                total_ops, np_op_ptime, np_op_job_prev, np_op_job_next,
                *build_mac_links(mac_seq, total_ops)[:2], topo
            )
            curr_ub = get_makespan(N, r, np_op_ptime)

            if curr_ub < best_upper_bound:
                best_upper_bound, best_feasible_r = curr_ub, r.copy()
                print(f">> New UB by heuristic: {best_upper_bound}")

        if not to_add:
            if gap_tolerance > 1e-3:
                print(f">> Found topology at {gap_tolerance*100:.1f}%.")
                print(">> Forcing gap to 0.01% to certify...")
                forced_gap = 1e-4
                continue

            r_final = np.zeros(total_ops, dtype=np.int32)
            for op in range(1, total_ops - 1):
                r_final[op] = int(round(start_times[op]))

            if status == pyo.TerminationCondition.optimal or gap_tolerance <= 1e-3:
                print(f"\n>> Master found optimum: {best_lower_bound}")
            else:
                print(f"\n>> Feasible solution on: {master_primal}")

            return Result(
                model = "ICG",
                solver = solver_name,
                instance = instance,
                schedule = get_schedule(N, r_final, op_job, op_mac, op_ptime),
                history_LB = [best_lower_bound],
                history_time_s = [time.perf_counter() - start_time_global]
            )

        to_add.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(3, len(to_add))):
            m_id, n_conf = to_add[i]
            n_pairs = add_cuts(
                m_id, start_times, model, mac_to_ops, op_ptime, EST, LST,
                z_idx_set, active_pairs, transitivity_triples_added
            )
            print(f">> {n_conf} collisions detected on machine {m_id}, {n_pairs} Big-M added")

    elapsed = time.perf_counter() - start_time_global
    print(f">> ICG total timeout: LB: {best_lower_bound} | UB: {best_upper_bound}")

    return Result(
        model = "ICG",
        solver = solver_name,
        instance = instance,
        schedule = get_schedule(N, best_feasible_r, op_job, op_mac, op_ptime) if best_feasible_r is not None else Schedule({}),
        history_LB = [best_lower_bound],
        history_time_s = [elapsed]
    )
