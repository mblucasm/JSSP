from __future__ import annotations

import time
import random

from bisect import insort
from itertools import chain
from dataclasses import dataclass
from alive_progress import alive_bar

from src.shared import Mac, Result, Schedule, Span
from typing import Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from src.shared import Job, Instance, ScheduleType

@dataclass
class Op:
    job: Job
    mac: Mac
    start: int
    end: int

@dataclass
class AbstractOp:
    job: Job
    mac: Mac
    ptime: int

Antigen = list[int]
Antibody = list[int]
Body = Antigen | Antibody
Plan = dict[Mac, list[Op]]

class AIS:

    def __init__(self, instance: Instance, r: int = 2, delta: float = 0.5) -> None:

        self.instance = instance
        self.mp = 1 / sum(len(self.instance.jobseq[j]) for j in range(self.instance.jobs))
        self._hint: Body | None = None

        assert 0 <= delta <= 1
        self.delta = delta
        self.r = r

    @staticmethod
    def flip(delta: float) -> bool:
        assert 0 <= delta <= 1
        return random.random() < delta

    def generate_body(self) -> Body:
        antigen = [j for j in range(self.instance.jobs) for _ in self.instance.jobseq[j]]
        random.shuffle(antigen)
        return antigen

    def mutate_B(self, body: Body) -> Body:

        result = body.copy()

        while body == result:
            for i in range(len(body)):
                if AIS.flip(self.mp):
                    target = i
                    while target == i:
                        target = random.randint(0, len(body) - 1)
                    result[i], result[target] = result[target], result[i]

        assert sorted(result) == sorted([j for j in range(self.instance.jobs) for _ in self.instance.jobseq[j]])
        return result

    def mutate_C(self, body: Body) -> Body:

        result = body.copy()

        while body == result:
            for i in range(len(body)):
                if AIS.flip(self.mp):
                    target = i
                    job = result.pop(i)
                    while target == i:
                        target = random.randint(0, len(body) - 1)
                    result.insert(target, job)

        assert sorted(result) == sorted([j for j in range(self.instance.jobs) for _ in self.instance.jobseq[j]])
        return result

    def solve(self, generations: int, clones_per_generation: int, mutation: Literal['A', 'B', 'C'], time_limit: int = 3600) -> Result:

        t0 = time.perf_counter()

        assert mutation != 'A', "Unimplemented yet"
        assert mutation in ('B', 'C')
        mutation_function = self.mutate_B if mutation == 'B' else self.mutate_C

        ab = self.generate_body() if self._hint is None else self._hint
        aga = self.generate_body()
        agb = aga.copy()

        ab_makespan, ab_plan = self.decode(ab)
        self.sort_body(ab, ab_plan)

        aga_makespan, aga_plan = self.decode(aga)
        self.sort_body(aga, aga_plan)

        agb_makespan = aga_makespan

        curr_history: list[int] = [agb_makespan]
        best_history: list[int] = [agb_makespan]

        current_gen = 0
        with alive_bar(generations, title = "AIS") as bar: # type: ignore
            for current_gen in range(generations):

                if time.perf_counter() - t0 >= time_limit:
                    print(f"Límite de tiempo ({time_limit}s) alcanzado en la generación {current_gen}. Deteniendo búsqueda...")
                    break

                if self.instance.optimum is not None and agb_makespan == self.instance.optimum:
                    print("Early stop, found known optimum")
                    curr_history.append(agb_makespan)
                    best_history.append(agb_makespan)
                    break

                if ab_makespan + self.r <= aga_makespan:

                    aga = ab.copy()
                    aga_makespan = ab_makespan

                    if aga_makespan < agb_makespan:
                        agb = aga.copy()
                        agb_makespan = aga_makespan

                else:
                    if AIS.flip(self.delta):
                        ab = aga.copy()
                        ab_makespan = aga_makespan

                clones = [mutation_function(clone) for clone in [ab.copy() for _ in range(clones_per_generation)]]
                info = [(i, self.decode(clone)) for i, clone in enumerate(clones)]

                best_i, best_clone_info = min(info, key = lambda x: x[1][0])
                self.sort_body(clones[best_i], best_clone_info[1])
                ab = clones[best_i].copy()
                ab_makespan = best_clone_info[0]

                curr_history.append(ab_makespan)
                best_history.append(agb_makespan)

                bar.text(f'Best makespan: {agb_makespan}') # type: ignore
                bar()

        if ab_makespan < agb_makespan:
            agb = ab.copy()
            agb_makespan = ab_makespan

        agb_makespan, agb_plan = self.decode(agb)

        return Result(
            model = "AIS",
            instance = self.instance,
            schedule = AIS.get_schedule(agb_plan),
            history_LB = curr_history,
            history_UB = best_history,
            history_time_s = [time.perf_counter() - t0]
        )

    def _get_op_info(self, bodyi: Job, current_op: dict[Job, int]) -> AbstractOp:

        job = bodyi
        opi = current_op.get(job, 0)
        mac = self.instance.jobseq[job][opi]
        ptime = self.instance.ptimes[job, mac]

        current_op[job] = current_op.get(job, 0) + 1

        return AbstractOp(job = job, mac = mac, ptime = ptime)

    def _find_position(self, plan: Plan, tjobisfree: dict[Job, int], tmacisfree: dict[Mac, int], absop: AbstractOp) -> Op:

        mac_plan = plan.get(absop.mac, []).copy()
        assert mac_plan == sorted(mac_plan, key = lambda x: x.start)

        mac_plan.insert(0, Op(job = -1, mac = -1, start = 0, end = 0))

        jobstart: int | None = None
        for op1, op2 in zip(mac_plan, mac_plan[1:]):

            earliest_start = max(tjobisfree.get(absop.job, 0), op1.end)

            if earliest_start + absop.ptime <= op2.start:
                jobstart = earliest_start
                break

        start = max(tjobisfree.get(absop.job, 0), tmacisfree.get(absop.mac, 0)) if jobstart is None else jobstart
        return Op(job = absop.job, mac = absop.mac, start = start, end = start + absop.ptime)

    def _update(self, plan: Plan, op: Op, tjobisfree: dict[Job, int], tmacisfree: dict[Mac, int]) -> None:

        mac_plan = plan.setdefault(op.mac, [])
        assert mac_plan == sorted(mac_plan, key = lambda x: x.start)

        insort(mac_plan, op, key = lambda x: x.start)
        tjobisfree[op.job] = op.end
        tmacisfree[op.mac] = max(tmacisfree.get(op.mac, 0), op.end)

    def sort_body(self, body: Body, plan: Plan) -> None:

        ops = sorted(chain(*plan.values()), key = lambda x: (x.start, x.job, x.mac))

        assert len(ops) == len(body)

        for i, op in enumerate(ops):
            body[i] = op.job

    def decode(self, body: Body) -> tuple[int, Plan]:

        plan: Plan = {}
        tjobisfree: dict[Job, int] = {}
        tmacisfree: dict[Mac, int] = {}
        current_op: dict[Job, int] = {}

        for gen in body:
            absop = self._get_op_info(bodyi = gen, current_op = current_op)
            op = self._find_position(plan = plan, tjobisfree = tjobisfree, tmacisfree = tmacisfree, absop = absop)
            self._update(plan = plan, op = op, tjobisfree = tjobisfree, tmacisfree = tmacisfree)

        makespan = max(tmacisfree.values())
        return makespan, plan

    @staticmethod
    def get_schedule(plan: Plan) -> Schedule:
        schedule: ScheduleType = {}
        for ops in plan.values():
            for op in ops:
                schedule[op.job, op.mac] = Span(op.start, op.end - op.start)
        return Schedule(schedule)

    @staticmethod
    def encode(schedule: Schedule) -> Body:
        sorted_schedule = list(sorted(schedule.schedule.items(), key = lambda x: x[1].start))
        return [jm[0] for jm, _ in sorted_schedule]

    def hint(self, schedule: Schedule) -> None:
        self._hint = AIS.encode(schedule)
