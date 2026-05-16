from __future__ import annotations

from src.shared import Result, Schedule, Span
from ortools.sat.python.cp_model import CpModel, CpSolver, CpSolverSolutionCallback

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from src.shared import Job, Mac, JSSPInstance, ScheduleType
    from ortools.sat.python.cp_model import IntVar, IntervalVar

class HistoryCallback(CpSolverSolutionCallback):

    def __init__(self):
        super().__init__()
        self.history_LB: list[int] = []
        self.history_UB: list[int] = []
        self.history_time_s: list[float] = []

    def on_solution_callback(self):
        self.history_LB.append(int(self.BestObjectiveBound()))
        self.history_UB.append(int(self.ObjectiveValue()))
        self.history_time_s.append(self.WallTime())

class _BaseModel(CpModel):

    def __init__(self, instance: JSSPInstance) -> None:
        super().__init__()
        self.instance = instance
        self.J = range(instance.jobs)
        self.M = range(instance.macs)
        self._calculate_bounds()
        self.C = self.new_int_var(self.C_LB, self.C_UB, "C")
        self.minimize(self.C)

    def _calculate_bounds(self) -> None:
        self.C_LB = 0
        self.C_UB = sum(self.instance.ptimes.values())

    def solve(
        self, time_limit: Optional[int] = None,
        num_search_workers: Optional[int] = None,
        tee: bool = False
    ) -> Result:

        assert not time_limit or 0 <= time_limit
        assert not num_search_workers or 0 <= num_search_workers

        solver = CpSolver()
        history_cb = HistoryCallback()

        if num_search_workers: solver.parameters.num_search_workers = num_search_workers
        if time_limit: solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.log_search_progress = tee

        solver.solve(self, history_cb)

        if history_cb.history_UB:
            final_LB = int(solver.best_objective_bound)
            final_UB = int(solver.objective_value)
            if solver.wall_time > history_cb.history_time_s[-1]:
                history_cb.history_LB.append(final_LB)
                history_cb.history_UB.append(final_UB)
                history_cb.history_time_s.append(solver.wall_time)

        return Result(
            model = "CP", solver = "CP-SAT",
            instance = self.instance, schedule = self.get_schedule(solver),
            history_LB = history_cb.history_LB, history_UB = history_cb.history_UB,
            history_time_s = history_cb.history_time_s
        )

    def get_schedule(self, solver: CpSolver) -> Schedule: ...

class Disjunctive(_BaseModel):

    def __init__(self, instance: JSSPInstance) -> None:

        super().__init__(instance)
        self._calculate_disjunctive_bounds()

        self.x = {(j, m): self.new_int_var(self.x_LB[j, m], self.x_UB[j, m], f"x_{j}_{m}") for j in self.J for m in self.instance.jobseq[j]}

        self.ends: dict[tuple[Job, Mac], IntVar] = {}
        self.intervals: dict[tuple[Job, Mac], IntervalVar] = {}

        for j in self.J:
            for m in self.instance.jobseq[j]:
                start = self.x[j, m]
                duration = self.instance.ptimes[j, m]
                end = self.new_int_var(self.x_LB[j, m] + duration, self.x_UB[j, m] + duration, f"end_{j}_{m}")
                self.ends[j, m] = end
                self.intervals[j, m] = self.new_interval_var(start, duration, end, f"int_{j}_{m}")

        for m in self.M:
            intervals_m = [self.intervals[j, m] for j in self.J if m in self.instance.jobseq[j]]
            if len(intervals_m) >= 2:
                self.add_no_overlap(intervals_m)

        self._add_correct_sequence()
        self._add_total_makespan()

    def _calculate_disjunctive_bounds(self) -> None:
        self.x_LB = {(j, m): sum(self.instance.ptimes[j, o] for o in self.instance.jobseq[j][:self.instance.jobseq[j].index(m)]) for j in self.J for m in self.instance.jobseq[j]}
        self.x_UB = {(j, m) : self.C_UB - sum(self.instance.ptimes[j, o] for o in self.instance.jobseq[j][self.instance.jobseq[j].index(m):]) for j in self.J for m in self.instance.jobseq[j]}

    def _add_correct_sequence(self) -> None:
        for j in self.J:
            seq = self.instance.jobseq[j]
            for prev_m, curr_m in zip(seq, seq[1:]):
                self.add(self.ends[j, prev_m] <= self.x[j, curr_m])

    def _add_total_makespan(self) -> None:
        for j in self.J:
            last_m = self.instance.jobseq[j][-1]
            end_last_m = self.ends[j, last_m]
            self.add(self.C >= end_last_m)

    def get_schedule(self, solver: CpSolver) -> Schedule:
        schedule: ScheduleType = {}
        for j in self.J:
            for m in self.instance.jobseq[j]:
                schedule[(j, m)] = Span(solver.value(self.x[j, m]), self.instance.ptimes[j, m])
        return Schedule(schedule)

    def hint(self, schedule: Schedule) -> None:

        starts = {key: span.start for key, span in schedule.schedule.items()}
        makespan = max(starts[j, m] + self.instance.ptimes[j, m] for j in self.J for m in self.instance.jobseq[j])

        for j in self.J:
            for m in self.instance.jobseq[j]:
                self.add_hint(self.x[j, m], starts[j, m])

        self.add_hint(self.C, makespan)
        self.add(self.C <= makespan)
