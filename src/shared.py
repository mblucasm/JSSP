from __future__ import annotations

import os
import glob
import json
import shutil
import textwrap
import colorsys
import itertools

import matplotlib.colors
import matplotlib.pyplot as plt

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
ScheduleType = dict[Op, Span]

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