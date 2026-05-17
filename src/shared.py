from __future__ import annotations

import os
import glob
import json
import shutil
import textwrap
import colorsys
import itertools

import numpy as np
import numpy.typing as npt
import matplotlib.colors
import matplotlib.pyplot as plt

from numba import njit # type: ignore

if shutil.which("latex"):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (5.9, 3.5),
        "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage[utf8]{inputenc} \usepackage{amsmath}"
    })
else:
    print(f"WARNING: {__file__}: LaTeX was not detected in system's PATH, using regular plot configuration instead")
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif", 
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (5.9, 3.5),
    })

from typing import Optional, TextIO
from dataclasses import dataclass
from matplotlib.colors import ListedColormap

INT32_MAX = 2147483647
GOLDEN_RATIO_CONJUGATE = 0.618033988749895
BASE_PALETTE: list[str] = ["#FFE8A1", "#FFDAB9", "#F5A9A9", "#7BA1C7", "#D4EDDA", "#FFF3CD", "#E2E2E2"]

@dataclass
class Span:
    start: int
    duration: int

@dataclass
class DotNode:
    j: Job
    m: Mac

Job = int
Mac = int
Op = tuple[Job, Mac]
OpID = int
ScheduleType = dict[Op, Span]
OpArray = npt.NDArray[np.int32]
JobArray = npt.NDArray[np.int32]
MacArray = npt.NDArray[np.int32]
IntArray = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]

def get_extended_palette(base_palette: list[str] = BASE_PALETTE, n: int = 100):
    result = list(base_palette)
    for i in range(n - len(base_palette)):
        h = (i * GOLDEN_RATIO_CONJUGATE) % 1
        l = 0.75 + (i % 3) * 0.05
        s = 0.65 + (i % 2) * 0.25
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        result.append(matplotlib.colors.to_hex((r, g, b)))
    return result

class Sequence(list[Mac]):
    def prev(self, m: Mac) -> Optional[Mac]:
        assert m in self
        return self[self.index(m) - 1] if not (m == self[0]) else None

class Instance:

    def __init__(self, name: str, root: str = "jsp-instances") -> None:

        match = None

        datafiles = glob.glob(os.path.join(root, "*.json"))
        for dfile in datafiles:
            with open(dfile, 'r') as f:
                data = json.load(f)
                if match := next((instance for instance in data if instance["name"] == name), None):
                    break
        if not match:
            raise ValueError(f"No match was found within {datafiles}")

        self.path: str = os.path.join(root, os.path.normpath(match["path"]))
        self.name: str = match["name"]
        self.jobs: int = match["jobs"]
        self.macs: int = match["machines"]
        self.optimum: Optional[int] = match["optimum"]
        self.jobseq: dict[Job, Sequence] = {}
        self.ptimes: dict[Op, int] = {}
        self.isrect: bool = True

        with open(self.path, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
            lines = lines[1:]

        for j, line in enumerate(lines):

            nums = list(map(int, line.split()))
            macs = list(map(int, nums[0::2]))

            self.jobseq[j] = Sequence(macs)
            if len(macs) != self.macs:
                self.isrect = False

            ptimes = nums[1::2]
            self.ptimes.update({(j, m): p for m, p in zip(macs, ptimes)})

    def digraph_write(self, fp: str, schedule: Optional[Schedule] = None, showcpath: bool = False) -> None:

        assert fp.endswith(".dot")
        if showcpath:
            assert schedule

        ranks  = self.digraph_get_ranks()
        groups = self.digraph_get_groups()
        conjarcs = self.digraph_get_conjarcs()

        with open(fp, "w") as f:
            f.write(self._digraph_get_header())
            self._digraph_write_ranks(f, ranks)
            f.write('\n')
            if not schedule: self._digraph_write_disjarcs(f, self.digraph_get_disjarcs())
            else: self._write_oriented_disjarcs(f, self.digraph_get_oriented_disjarcs(schedule), schedule if showcpath else None)
            f.write('\n')
            self._digraph_write_groups(f, groups)
            f.write('\n')
            self._digraph_write_conjarcs(f, conjarcs, schedule if showcpath else None)
            f.write(self._digraph_get_footer())

    def digraph_get_ranks(self) -> list[list[DotNode]]:
        ranks: list[list[DotNode]] = [[] for _ in range(max(len(seq) for seq in self.jobseq.values()))]
        for j in range(self.jobs):
            for i, m in enumerate(self.jobseq[j]):
                ranks[i].append(DotNode(j = j, m = m))
        return ranks

    def digraph_get_oriented_disjarcs(self, schedule: Schedule) -> list[tuple[DotNode, DotNode]]:
        mac_ops: dict[int, list[tuple[int, int]]] = {}
        for (j, m), span in schedule.schedule.items():
            if m not in mac_ops:
                mac_ops[m] = []
            mac_ops[m].append((span.start, j))
        oriented_arcs: list[tuple[DotNode, DotNode]] = []
        for m, ops in mac_ops.items():
            ops.sort()
            for i in range(len(ops) - 1):
                src = DotNode(j = ops[i][1], m = m)
                dst = DotNode(j = ops[i+1][1], m = m)
                oriented_arcs.append((src, dst))
        return oriented_arcs

    def digraph_get_disjarcs(self) -> list[tuple[DotNode, DotNode]]:
        groups: list[list[DotNode]] = [[] for _ in range(self.macs)]
        for j in range(self.jobs):
            for m in self.jobseq[j]:
                groups[m].append(DotNode(j = j, m = m))
        return [pair for group in groups if len(group) >= 2 for pair in itertools.combinations(group, 2)]

    def digraph_get_groups(self) -> list[list[DotNode]]:
        return [[DotNode(j, m) for m in self.jobseq[j]] for j in range(self.jobs)]

    def digraph_get_conjarcs(self) -> list[tuple[DotNode, DotNode, int]]:
        conjarcs: list[tuple[DotNode, DotNode, int]] = []
        for j in range(self.jobs):
            for m in self.jobseq[j]:
                prev_m = self.jobseq[j].prev(m)
                src = DotNode(j = j, m = prev_m) if prev_m is not None else DotNode(-1, -1)
                dst = DotNode(j = j, m = m)
                lab = 0 if prev_m is None else self.ptimes[j, prev_m]
                conjarcs.append((src, dst, lab))
            m = self.jobseq[j][-1]
            conjarcs.append((DotNode(j = j, m = m), DotNode(j = -1, m = -1), self.ptimes[j, m]))
        return conjarcs

    def _write_oriented_disjarcs(self, f: TextIO, disjarcs: list[tuple[DotNode, DotNode]], schedule: Optional[Schedule]) -> None:
        f.write('    edge [style=solid, color=black];\n')
        all_disjarcs = self.digraph_get_disjarcs()
        oriented_set = {(src.j, src.m, dst.j, dst.m) for src, dst in disjarcs}
        for src, dst in all_disjarcs:
            if (src.j, src.m, dst.j, dst.m) in oriented_set:
                real_src, real_dst = src, dst
                is_visible = True
            elif (dst.j, dst.m, src.j, src.m) in oriented_set:
                real_src, real_dst = dst, src
                is_visible = True
            else:
                real_src, real_dst = src, dst
                is_visible = False
            if not is_visible:
                f.write(f'    "{real_src.j},{real_src.m}" -> "{real_dst.j},{real_dst.m}" [style=invis];\n')
            else:
                if schedule:
                    if ((real_src.j, real_src.m), (real_dst.j, real_dst.m)) in zip(schedule.cpath, schedule.cpath[1:]):
                        f.write(f'    "{real_src.j},{real_src.m}" -> "{real_dst.j},{real_dst.m}" [color="#F08080", penwidth=1.25, arrowsize=0.6, arrowhead=normal];\n')
                    else:
                        f.write(f'    "{real_src.j},{real_src.m}" -> "{real_dst.j},{real_dst.m}" [color="#00000030"];\n')
                else:
                    f.write(f'    "{real_src.j},{real_src.m}" -> "{real_dst.j},{real_dst.m}";\n')

    def _digraph_write_ranks(self, f: TextIO, ranks: list[list[DotNode]]) -> None:
        f.write('    { rank=min; "s"; }\n')
        f.write('    { rank=max; "t"; }\n')
        for rank in ranks:
            f.write('    { rank=same; ')
            for node in rank:
                duration = self.ptimes[node.j, node.m]
                label_html = f'<<FONT POINT-SIZE="10">{duration}</FONT><BR/>{node.j},{node.m}>'
                f.write(f'"{node.j},{node.m}" [label={label_html}]; ')
            f.write('}\n')

    def _digraph_write_disjarcs(self, f: TextIO, disjarcs: list[tuple[DotNode, DotNode]]) -> None:
        f.write(f'    edge [style=dashed, color="#7BA1C7"];\n')
        for jump in disjarcs:
            f.write(f'    "{jump[0].j},{jump[0].m}" -> "{jump[1].j},{jump[1].m}" [dir=both, style=dashed];\n')

    def _digraph_write_groups(self, f: TextIO, groups: list[list[DotNode]]) -> None:
        for j, group in enumerate(groups):
            f.write('    ')
            for node in group:
                f.write(f'"{node.j},{node.m}" [group=j{j}]; ')
            f.write('\n')

    def _digraph_write_conjarcs(self, f: TextIO, conjarcs: list[tuple[DotNode, DotNode, int]], schedule: Optional[Schedule]) -> None:
        f.write(f'    edge [style=solid, color=black];\n');
        for arc in conjarcs:
            src = "s" if arc[0].j == -1 else f"{arc[0].j},{arc[0].m}"
            dst = "t" if arc[1].j == -1 else f"{arc[1].j},{arc[1].m}"
            if schedule:
                if ((arc[0].j, arc[0].m), (arc[1].j, arc[1].m)) in zip(schedule.cpath, schedule.cpath[1:]):
                    f.write(f'    "{src}" -> "{dst}" [color="#F08080", penwidth=1.25, arrowsize=0.6, arrowhead=normal];\n')
                else:
                    f.write(f'    "{src}" -> "{dst}" [color="#00000030"];\n')
            else:
                f.write(f'    "{src}" -> "{dst}";\n')

    def _digraph_get_header(self) -> str:
        return textwrap.dedent(f'''
            digraph {self.name} {r"{"}

                rankdir="LR";

                "s" [shape=doublecircle];
                "t" [shape=doublecircle];

                edge [penwidth=0.5, arrowsize=0.3, arrowhead=vee, arrowtail=vee];
                node [shape=circle, width=0.55, fixedsize=true, style=filled, fillcolor="#f5f5f5"];\n
        ''')

    def _digraph_get_footer(self) -> str:
        return '}\n'

class Schedule:

    def __init__(self, schedule: ScheduleType, verify: bool = True) -> None:

        if verify: Schedule.verify(schedule)
        makespan = Schedule.get_makespan(schedule)
        cpath = Schedule.get_cpath(schedule, makespan)
        cblocks = Schedule.get_cblocks(cpath)

        self.schedule = schedule
        self.makespan = makespan
        self.cpath = cpath
        self.cblocks = cblocks

    @staticmethod
    def get_makespan(schedule: Schedule | ScheduleType) -> int:
        if isinstance(schedule, Schedule):
            schedule = schedule.schedule
        return max(span.start + span.duration for span in schedule.values()) if schedule else 0

    @staticmethod
    def verify(schedule: Schedule | ScheduleType) -> None:

        if isinstance(schedule, Schedule):
            schedule = schedule.schedule

        machine_ops: dict[Mac, list[tuple[int, int, Job]]] = {}
        job_ops: dict[Job, list[tuple[int, int, Mac]]] = {}

        for (job, mac), span in schedule.items():
            if span.start < 0 or span.duration < 0:
                raise ValueError
            machine_ops.setdefault(mac, []).append((span.start, span.duration, job))
            job_ops.setdefault(job, []).append((span.start, span.duration, mac))

        for mac, ops in machine_ops.items():
            ops.sort(key = lambda x: x[0])
            for i in range(len(ops) - 1):
                curr_start, curr_dur, curr_job = ops[i]
                next_start, _, next_job = ops[i+1]
                if curr_start + curr_dur > next_start:
                    raise ValueError(f"Machine {mac} overload: Job {curr_job} ends at {curr_start + curr_dur} and Job {next_job} starts at {next_start}.")

        for job, ops in job_ops.items():
            ops.sort(key = lambda x: x[0])
            for i in range(len(ops) - 1):
                curr_start, curr_dur, curr_mac = ops[i]
                next_start, _, next_mac = ops[i+1]
                if curr_start + curr_dur > next_start:
                    raise ValueError(f"Job {job} overload: Processed in {curr_mac} until {curr_start + curr_dur}, but starts in {next_mac} at {next_start}.")

    @staticmethod
    def get_cpath(schedule: Schedule | ScheduleType, makespan: Optional[int] = None) -> list[tuple[Job, Mac]]:

        if isinstance(schedule, Schedule):
            schedule = schedule.schedule
        if not schedule:
            return []
        if not makespan:
            makespan = Schedule.get_makespan(schedule)

        endops = [op for op, span in schedule.items() if span.start + span.duration == makespan]
        if not endops:
            raise ValueError("When trying to find the beginning (from the end) of the cpath, no critical ops where found")

        job_sequences: dict[Job, list[tuple[int, Mac]]] = {}
        mac_sequences: dict[Mac, list[tuple[int, Job]]] = {}
        for (j, m), span in schedule.items():
            job_sequences.setdefault(j, []).append((span.start, m))
            mac_sequences.setdefault(m, []).append((span.start, j))

        for j in job_sequences: job_sequences[j].sort()
        for m in mac_sequences: mac_sequences[m].sort()

        def get_predecessors(op: Op, curr_start: int) -> list[Op]:

            j, m = op
            result: list[Op] = []

            s_job = job_sequences[j]
            idx_j = next(i for i, (_, machine) in enumerate(s_job) if machine == m)
            if idx_j > 0:
                prev_m = s_job[idx_j - 1][1]
                prev_span = schedule[j, prev_m]
                if prev_span.start + prev_span.duration == curr_start:
                    result.append((j, prev_m))

            s_mac = mac_sequences[m]
            idx_m = next(i for i, (_, job) in enumerate(s_mac) if job == j)
            if idx_m > 0:
                prev_j = s_mac[idx_m - 1][1]
                prev_span = schedule[prev_j, m]
                if prev_span.start + prev_span.duration == curr_start:
                    result.append((prev_j, m))

            return result

        def find_path_to_start(current_op: Op, current_path: list[Op]) -> Optional[list[Op]]:
            span = schedule[current_op]
            if span.start == 0:
                return current_path
            preds = get_predecessors(current_op, span.start)
            for pred in preds:
                result_path = find_path_to_start(pred, current_path + [pred])
                if result_path:
                    return result_path
            return None

        for endop in endops:
            path = find_path_to_start(endop, [endop])
            if path:
                return path[::-1]

        raise ValueError("No cpath found")

    @staticmethod
    def get_cblocks(cpath: list[Op]) -> list[list[Op]]:

        if not cpath:
            return []

        blocks: list[list[tuple[Job, Mac]]] = []
        current_block = [cpath[0]]

        for i in range(1, len(cpath)):
            if cpath[i][1] == current_block[-1][1]:
                current_block.append(cpath[i])
            else:
                blocks.append(current_block)
                current_block = [cpath[i]]

        blocks.append(current_block)
        return blocks

@dataclass
class Result:

    model: str
    instance: Instance
    schedule: Schedule

    solver: Optional[str] = None
    history_LB: Optional[list[int]] = None
    history_UB: Optional[list[int]] = None
    history_time_s: Optional[list[float]] = None

    def plot(
        self, figsize: tuple[int, int] = (6, 4), dpi: int = 100,
        showoptimum: bool = True, showlegend: bool = False,
        showcpath: bool = False, showjoblabels: bool = True,
        highlight: Optional[list[tuple[Job, Mac]]] = None,
        save_name: Optional[str] = None, show: bool = True,
    ) -> None:

        assert not (highlight and showcpath)

        fig, ax = plt.subplots(figsize = figsize, dpi = dpi) # type: ignore
        palette = ListedColormap(get_extended_palette(n = self.instance.jobs)).colors

        labeled_jobs: set[Job] = set()

        if showcpath:
            for block in self.schedule.cblocks:
                start = self.schedule.schedule[block[0]].start
                last_op = self.schedule.schedule[block[-1]]
                end = last_op.start + last_op.duration
                ax.barh( # type: ignore
                    y = block[0][1], width = end - start,
                    left = start, height = 0.8,
                    color = "none", edgecolor = "black",
                    linewidth = 1, zorder = 5,
                )

        for (j, m), span in self.schedule.schedule.items():

            label = None
            if j not in labeled_jobs:
                label = f"$j_{{{j}}}$"
                labeled_jobs.add(j)

            alpha = 1
            if showcpath and (j, m) not in self.schedule.cpath:
                alpha = 0.3
            if highlight and (j, m) not in highlight:
                alpha = 0.3

            ax.barh( # type: ignore
                y = m, width = span.duration,
                left = span.start, height = 0.8,
                label = label, color = palette[j % len(palette)], # type: ignore
                alpha = alpha,
            )

            if showjoblabels:
                ax.text( # type: ignore
                    x = span.start + span.duration / 2, y = m,
                    s = f"$j_{{{j}}}$", ha = "center", va = "center",
                    alpha = alpha,
                )

        if self.instance.optimum is not None and showoptimum and self.instance.optimum != self.schedule.makespan:
            ax.axvline( # type: ignore
                x = self.instance.optimum, linewidth = 1,
                linestyle = "--", color = "red",
                alpha = 0.5, label = f"$C_{r'\text{opt}'} = {self.instance.optimum}$",
            )

        if showlegend:
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = zip(*sorted(zip(handles, labels), key = lambda x: int(x[1].split()[1]) if "Job" in x[1] else -1))
            num_cols = max(1, len(labels) // 30 + 1)
            ax.legend(handles = handles, labels = labels, loc = 'upper left', bbox_to_anchor = (1, 1.03), ncol = num_cols) # type: ignore

        ax.invert_yaxis()
        if self.schedule.makespan == self.instance.optimum:
            ax.set_title(f"$C_\\text{{{"opt"}}} = {self.schedule.makespan}$") # type: ignore
        ax.set_xlim((0, self.schedule.makespan + 3))
        ax.set_yticks(range(self.instance.macs), labels = [f"$m_{{{i}}}$" for i in range(self.instance.macs)]) # type: ignore
        ax.set_xlabel("Tiempo ($t$)") # type: ignore
        ax.set_ylabel("Máquina") # type: ignore
        fig.tight_layout()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if save_name: plt.savefig(save_name, bbox_inches = "tight") # type: ignore
        if show: plt.show() # type: ignore

def get_job_sequence(instance: Instance, N: int) -> tuple[dict[Job, list[OpID]], list[Job], list[Mac], list[int], list[OpID], list[OpID]]:

    total_ops = N + 2
    op_start = 0
    op_end = N + 1

    op_job = [-1] * total_ops
    op_mac = [-1] * total_ops
    op_ptime = [0] * total_ops
    op_job_prev = [-1] * total_ops
    op_job_next = [-1] * total_ops

    curr_id = 0
    job_sequence: dict[Job, list[int]] = {}

    for job, sequence in instance.jobseq.items():
        job_ops: list[int] = []
        for mac in sequence:
            curr_id += 1

            op_job[curr_id] = job
            op_mac[curr_id] = mac
            op_ptime[curr_id] = instance.ptimes[job, mac]
            job_ops.append(curr_id)

        for i in range(len(job_ops)):
            curr_op = job_ops[i]
            op_job_prev[curr_op] = job_ops[i - 1] if i > 0 else op_start
            op_job_next[curr_op] = job_ops[i + 1] if i < len(job_ops) - 1 else op_end

        job_sequence[job] = job_ops

    return job_sequence, op_job, op_mac, op_ptime, op_job_prev, op_job_next

@njit(cache = True) # type: ignore
def topological_sort_no_cycles(total_ops: int, op_job_next: OpArray, op_mac_next: OpArray) -> OpArray:
    topo, cycle = topological_sort(total_ops=total_ops, op_job_next=op_job_next, op_mac_next=op_mac_next)
    if cycle: raise ValueError("Cycle in graph")
    return topo

@njit(cache = True) # type: ignore
def topological_sort(total_ops: int, op_job_next: OpArray, op_mac_next: OpArray) -> tuple[OpArray, bool]:

    in_degree = np.zeros(total_ops, dtype = np.int32)
    adj = np.full((total_ops, 2), -1, dtype = np.int32)

    for u in range(total_ops):
        idx = 0
        v_mac = op_mac_next[u]
        if v_mac != -1:
            adj[u, idx] = v_mac
            in_degree[v_mac] += 1
            idx += 1

        v_job = op_job_next[u]
        if v_job != -1 and v_job != v_mac:
            adj[u, idx] = v_job
            in_degree[v_job] += 1

    head, tail = 0, 0
    queue = np.empty(total_ops, dtype = np.int32)

    for i in range(total_ops):
        if in_degree[i] == 0:
            queue[tail] = i
            tail += 1

    topo_idx = 0
    topo_order = np.empty(total_ops, dtype = np.int32)

    while head < tail:
        u = queue[head]
        head += 1

        topo_order[topo_idx] = u
        topo_idx += 1

        for i in range(2):
            v = adj[u, i]
            if v != -1:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue[tail] = v
                    tail += 1

    return topo_order, topo_idx != total_ops

@njit(cache = True) # type: ignore
def heads_tails(
    total_ops: int,
    op_ptime: IntArray,
    op_job_prev: OpArray, op_job_next: OpArray,
    op_mac_prev: OpArray, op_mac_next: OpArray,
    topo_order: Optional[OpArray] = None,
) -> tuple[IntArray, IntArray]:

    if topo_order is None:
        topo_order = topological_sort_no_cycles(total_ops=total_ops, op_job_next=op_job_next, op_mac_next=op_mac_next)
    r = np.zeros(total_ops, dtype = np.int32)

    for i in range(total_ops):
        op_id = topo_order[i]

        prev_job = op_job_prev[op_id]
        val_job = r[prev_job] + op_ptime[prev_job] if prev_job != -1 else 0

        prev_mac = op_mac_prev[op_id]
        val_mac = r[prev_mac] + op_ptime[prev_mac] if prev_mac != -1 else 0

        r[op_id] = max(val_job, val_mac)

    q = np.zeros(total_ops, dtype = np.int32)

    for i in range(total_ops - 1, -1, -1):
        op_id = topo_order[i]

        next_job = op_job_next[op_id]
        val_job = q[next_job] + op_ptime[next_job] if next_job != -1 else 0

        next_mac = op_mac_next[op_id]
        val_mac = q[next_mac] + op_ptime[next_mac] if next_mac != -1 else 0

        q[op_id] = max(val_job, val_mac)

    return r, q

@njit(cache = True) # type: ignore
def get_makespan(N: int, r: IntArray, op_ptime: IntArray) -> int:
    best = 0
    for i in range(1, N + 1):
        val = r[i] + op_ptime[i]
        if val > best:
            best = val
    return best

def get_schedule(N: int, r: IntArray, op_job: JobArray, op_mac: MacArray, op_ptime: IntArray, verify: bool = True) -> Schedule:
    schedule: ScheduleType = {}
    for op_id in range(1, N + 1):
        job_id = int(op_job[op_id])
        mac_id = int(op_mac[op_id])
        schedule[job_id, mac_id] = Span(int(r[op_id]), int(op_ptime[op_id]))
    return Schedule(schedule, verify=verify)

@njit(cache = True) # type: ignore
def get_critical_path(
    N: int, r: IntArray,
    op_ptime: IntArray,
    op_job_prev: OpArray, op_mac_prev: OpArray,
) -> OpArray:

    makespan = get_makespan(N=N, r=r, op_ptime=op_ptime)

    curr = -1
    for i in range(1, N + 1):
        if r[i] + op_ptime[i] == makespan:
            curr = i
            break

    idx = 0
    cpath_temp = np.empty(N + 2, dtype = np.int32)

    while curr >= 1:
        cpath_temp[idx] = curr
        idx += 1
        target = r[curr]

        prev_job = op_job_prev[curr]
        prev_mac = op_mac_prev[curr]

        if prev_job != -1 and r[prev_job] + op_ptime[prev_job] == target:
            curr = prev_job
        elif prev_mac != -1 and r[prev_mac] + op_ptime[prev_mac] == target:
            curr = prev_mac
        else:
            curr = -1

    return cpath_temp[:idx][::-1]

@njit
def bidir(
    N: int, num_macs: int, c: int,
    op_mac: MacArray, op_ptime: IntArray,
    op_job_prev: OpArray, op_job_next: OpArray,
    job_head: IntArray, job_tail: IntArray,
    mac_sorted_tail: IntArray,
    mac_sorted_head: IntArray,
    ops_per_mac: int,
    initial_S: OpArray, initial_T: OpArray
) -> tuple[OpArray, OpArray]:

    total_ops = N + 2
    in_L = np.zeros(total_ops, dtype = np.bool_)
    in_R = np.zeros(total_ops, dtype = np.bool_)
    in_S = np.zeros(total_ops, dtype = np.bool_)
    in_T = np.zeros(total_ops, dtype = np.bool_)

    in_L[0] = True
    in_R[N + 1] = True

    for op in initial_S: in_S[op] = True
    for op in initial_T: in_T[op] = True

    r = np.zeros(total_ops, dtype = np.int32)
    t = np.zeros(total_ops, dtype = np.int32)
    free_L = np.zeros(num_macs, dtype = np.int32)
    free_R = np.zeros(num_macs, dtype = np.int32)

    mac_first_L = np.full(num_macs, -1, dtype = np.int32)
    mac_last_L = np.full(num_macs, -1, dtype = np.int32)
    mac_first_R = np.full(num_macs, -1, dtype = np.int32)
    mac_last_R = np.full(num_macs, -1, dtype = np.int32)
    next_in_mac = np.full(total_ops, -1, dtype = np.int32)

    scheduled_count = 2
    five_percent = (total_ops - 2) * 0.05
    while scheduled_count < total_ops:

        if not (scheduled_count-2) % five_percent:
            print(scheduled_count-2, "/", total_ops-2)

        if scheduled_count < total_ops:
            candidates_S: list[OpID] = []
            for i in range(1, N + 1):
                if in_S[i]: candidates_S.append(i)

            if candidates_S:
                costs: list[tuple[OpID, int]] = []
                for cand in candidates_S:
                    cost = est(cand, in_L, in_R, r, t, op_mac, op_ptime, job_head, job_tail, mac_sorted_tail, mac_sorted_head, ops_per_mac, True)
                    costs.append((cand, cost))

                costs.sort(key = lambda x: x[1])
                limit = min(c, len(costs))
                idx = np.random.randint(0, limit)
                best = costs[idx][0]

                m = op_mac[best]
                in_L[best] = True
                in_S[best] = False
                in_T[best] = False

                if mac_first_L[m] == -1: mac_first_L[m] = best
                if mac_last_L[m] != -1: next_in_mac[mac_last_L[m]] = best
                mac_last_L[m] = best

                free_L[m] = r[best] + op_ptime[best]
                scheduled_count += 1

                nxt = op_job_next[best]
                if nxt <= N and not in_R[nxt]: in_S[nxt] = True

                for op_id in range(1, N + 1):
                    if in_S[op_id]:
                        prv_j = op_job_prev[op_id]
                        r_job = r[prv_j] + op_ptime[prv_j] if (prv_j != -1 and in_L[prv_j]) else 0
                        r_mac = free_L[op_mac[op_id]]
                        r[op_id] = max(r_job, r_mac)

        if scheduled_count >= total_ops: break

        if scheduled_count < total_ops:
            candidates_T: list[OpID] = []
            for i in range(1, N + 1):
                if in_T[i]: candidates_T.append(i)

            if candidates_T:
                costs: list[tuple[OpID, int]] = []
                for cand in candidates_T:
                    cost = est(cand, in_L, in_R, r, t, op_mac, op_ptime, job_head, job_tail, mac_sorted_tail, mac_sorted_head, ops_per_mac, False)
                    costs.append((cand, cost))

                costs.sort(key=lambda x: x[1])
                idx = np.random.randint(0, min(c, len(costs)))
                best = costs[idx][0]

                m = op_mac[best]
                in_R[best] = True
                in_T[best] = False
                in_S[best] = False

                if mac_last_R[m] == -1: mac_last_R[m] = best
                next_in_mac[best] = mac_first_R[m]
                mac_first_R[m] = best

                free_R[m] = t[best] + op_ptime[best]
                scheduled_count += 1

                prv = op_job_prev[best]
                if prv >= 1 and not in_L[prv]: in_T[prv] = True

                for op_id in range(1, N + 1):
                    if in_T[op_id]:
                        nxt_j = op_job_next[op_id]
                        t_job = t[nxt_j] + op_ptime[nxt_j] if (nxt_j != -1 and in_R[nxt_j]) else 0
                        t_mac = free_R[op_mac[op_id]]
                        t[op_id] = max(t_job, t_mac)

    final_mac_prev = np.full(total_ops, -1, dtype=np.int32)
    final_mac_next = np.full(total_ops, -1, dtype=np.int32)

    for m in range(num_macs):
        if mac_last_L[m] != -1 and mac_first_R[m] != -1:
            next_in_mac[mac_last_L[m]] = mac_first_R[m]

        curr = mac_first_L[m] if mac_first_L[m] != -1 else mac_first_R[m]
        while curr != -1:
            nxt = next_in_mac[curr]
            if nxt != -1:
                final_mac_next[curr] = nxt
                final_mac_prev[nxt] = curr
            curr = nxt

    return final_mac_prev, final_mac_next

@njit # type: ignore
def est(
    op_id: int, in_L: BoolArray, in_R: BoolArray,
    r: IntArray, t: IntArray,
    op_mac: MacArray, op_ptime: IntArray,
    job_head: IntArray, job_tail: IntArray,
    mac_sorted_tail: OpArray,
    mac_sorted_head: OpArray,
    ops_per_mac: int, is_S: bool
) -> int:

    mac_id = op_mac[op_id]

    if is_S:
        term_job = job_tail[op_id]
        term_mac = 0
        for i in range(ops_per_mac):
            o = mac_sorted_tail[mac_id, i]
            if o == -1: break
            if not in_L[o] and o != op_id:
                term_mac = op_ptime[o] + job_tail[o]
                break
        return r[op_id] + op_ptime[op_id] + max(term_job, term_mac)
    else:
        term_job = job_head[op_id]
        term_mac = 0
        for i in range(ops_per_mac):
            o = mac_sorted_head[mac_id, i]
            if o == -1: break
            if not in_R[o] and o != op_id:
                term_mac = op_ptime[o] + job_head[o]
                break
        return max(term_job, term_mac) + op_ptime[op_id] + t[op_id]

@njit(cache = True) # type: ignore
def get_tail(op_id: OpID, N: int, op_job_next: OpArray, op_ptime: IntArray) -> int:
    result = 0
    curr = op_job_next[op_id]
    while curr <= N:
        assert curr != -1
        result += op_ptime[curr]
        curr = op_job_next[curr]
    return result # type: ignore

@njit(cache = True) # type: ignore
def get_head(op_id: OpID, op_job_prev: OpArray, op_ptime: IntArray) -> int:
    result = 0
    curr = op_job_prev[op_id]
    while curr >= 1:
        assert curr != -1
        result += op_ptime[curr]
        curr = op_job_prev[curr]
    return result # type: ignore

@njit(cache = True) # type: ignore
def move(
    oblock: OpArray, new_Q: OpArray,
    op_mac_prev: OpArray, op_mac_next: OpArray,
    op_pos_in_mac: IntArray,
) -> None:

    before_block = op_mac_prev[oblock[0]]
    after_block = op_mac_next[oblock[-1]]
    start_pos = op_pos_in_mac[oblock[0]]

    if before_block != -1:
        op_mac_next[before_block] = new_Q[0]
    op_mac_prev[new_Q[0]] = before_block

    for i in range(len(new_Q)):
        curr_op = new_Q[i]
        op_pos_in_mac[curr_op] = start_pos + i

        if i > 0:
            prev_op = new_Q[i - 1]
            op_mac_next[prev_op] = curr_op
            op_mac_prev[curr_op] = prev_op

    if after_block != -1:
        op_mac_prev[after_block] = new_Q[-1]
    op_mac_next[new_Q[-1]] = after_block
