import time
import pyomo.environ as pyo # type: ignore

from src.shared import Result, Schedule, Span, Instance
from pyomo.environ import SolverFactory # type: ignore

from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from src.shared import ScheduleType

class _BaseModel(pyo.ConcreteModel): # type: ignore

    def __init__(self, instance: Instance):

        super().__init__() # type: ignore

        self.washinted = False

        self.instance = instance
        self.python_J = range(instance.jobs)
        self.python_M = range(instance.macs)
        self.python_JM = [(j, m) for j, seq in self.instance.jobseq.items() for m in seq]

        self.pyo_J  = pyo.Set(dimen = 1, initialize = self.python_J) # type: ignore
        self.pyo_M  = pyo.Set(dimen = 1, initialize = self.python_M) # type: ignore
        self.pyo_JM = pyo.Set(dimen = 2, initialize = self.python_JM) # type: ignore
        self.pyo_p  = pyo.Param(self.pyo_JM, initialize = self.instance.ptimes) # type: ignore

        self._calculateBounds()
        self.pyo_C = pyo.Var(within = pyo.NonNegativeIntegers, bounds = (0, self.python_C_UB)) # type: ignore
        self.pyo_obj = pyo.Objective(expr = self.pyo_C, sense = pyo.minimize) # type: ignore

    def _calculateBounds(self):
        self.python_C_UB = sum(self.instance.ptimes.values())

    def solve(self, solver_name: str, time_limit: Optional[int] = None, tee: bool = False):

        with SolverFactory(solver_name) as solver: # type: ignore
            if time_limit:
                if solver_name.upper() == "APPSI_HIGHS":
                    solver.options['time_limit'] = time_limit # type: ignore
                elif solver_name.upper() == "MOSEK":
                    solver.options['dparam.optimizer_max_time'] = time_limit # type: ignore
                    solver.options['dparam.mio_max_time'] = time_limit # type: ignore
                else:
                    raise ValueError("Solver not implemented")

            solver.solve(self, tee = tee) # type: ignore

            start_time = time.perf_counter()
            results = solver.solve(self, tee = tee) # type: ignore
            time_run = time.perf_counter() - start_time

        milp_best_bound = None

        if len(results.problem) > 0: # type: ignore
            milp_best_bound = results.problem[0].lower_bound # type: ignore

        return Result(
            model = self.__class__.__qualname__, solver = solver_name,
            instance = self.instance, schedule = self.get_schedule(),
            history_LB = [milp_best_bound if milp_best_bound else -1],
            history_time_s = [time_run]
        )

    def get_schedule(self, tol: float = 0.5) -> Schedule: ...

class Disjunctive(_BaseModel):

    def __init__(self, instance: Instance):

        super().__init__(instance)
        self._calculateDisjunctiveBounds()

        self.python_IJM = [
            (i, j, m)
            for i in self.python_J
            for j in self.python_J
            if i < j
            for m in self.instance.jobseq[i]
            if  m in self.instance.jobseq[j] and (i, m) in self.python_JM and (j, m) in self.python_JM
        ]

        self.pyo_IJM = pyo.Set(dimen = 3, initialize = self.python_IJM) # type: ignore

        self.python_M1 = {(i, j, m): self.python_x_UB[(i, m)] + self.instance.ptimes[(i, m)] - self.python_x_LB[(j, m)] for (i, j, m) in self.python_IJM}
        self.python_M2 = {(i, j, m): self.python_x_UB[(j, m)] + self.instance.ptimes[(j, m)] - self.python_x_LB[(i, m)] for (i, j, m) in self.python_IJM}

        self.pyo_M1 = pyo.Param( # type: ignore
            self.pyo_IJM, # type: ignore
            initialize = self.python_M1,
            within = pyo.NonNegativeIntegers, # type: ignore
        )

        self.pyo_M2 = pyo.Param( # type: ignore
            self.pyo_IJM, # type: ignore
            initialize = self.python_M2,
            within = pyo.NonNegativeIntegers, # type: ignore
        )

        self.pyo_x = pyo.Var( # type: ignore
            self.pyo_JM, # type: ignore
            within = pyo.NonNegativeIntegers, # type: ignore
            bounds = lambda _, j, m: (self.python_x_LB[j, m], self.python_x_UB[j, m]), # type: ignore
        )

        self.pyo_z = pyo.Var( # type: ignore
            self.pyo_IJM, # type: ignore
            within = pyo.Binary, # type: ignore
        )

        self.constraint1 = pyo.Constraint(self.pyo_JM, rule = self._constraintCorrectSequence) # type: ignore
        self.constraint2 = pyo.Constraint(self.pyo_J, rule = self._constraintTotalMakespan) # type: ignore
        self.constraint3 = pyo.Constraint(self.pyo_IJM, rule = self._constraintCorrectPrecedence1) # type: ignore
        self.constraint4 = pyo.Constraint(self.pyo_IJM, rule = self._constraintCorrectPrecedence2) # type: ignore

    def _calculateDisjunctiveBounds(self):
        self.python_x_LB = {(j, m): sum(self.instance.ptimes[j, o] for o in self.instance.jobseq[j][:self.instance.jobseq[j].index(m)]) for j, m in self.python_JM}
        self.python_x_UB = {(j, m) : self.python_C_UB - sum(self.instance.ptimes[j, o] for o in self.instance.jobseq[j][self.instance.jobseq[j].index(m):]) for j, m in self.python_JM}

    def _constraintCorrectSequence(self, model, j, m): # type: ignore
        o = model.instance.jobseq[j].prev(m) # type: ignore
        if o is None:
            return pyo.Constraint.Skip # type: ignore
        return (model.pyo_x[j, o] + model.pyo_p[j, o] <= model.pyo_x[j, m]) # type: ignore

    def _constraintTotalMakespan(self, model, j): # type: ignore
        o = model.instance.jobseq[j][-1] # type: ignore
        return (model.pyo_x[j, o] + model.pyo_p[j, o] <= model.pyo_C) # type: ignore

    def _constraintCorrectPrecedence1(self, model, i, j, m): # type: ignore
        return model.pyo_x[i, m] + model.pyo_p[i, m] <= model.pyo_x[j, m] + model.pyo_M1[i, j, m] * (1 - model.pyo_z[i, j, m]) # type: ignore

    def _constraintCorrectPrecedence2(self, model, i, j, m): # type: ignore
        return model.pyo_x[j, m] + model.pyo_p[j, m] <= model.pyo_x[i, m] + model.pyo_M2[i, j, m] * model.pyo_z[i, j, m] # type: ignore

    def get_schedule(self, tol: float = 0.5): # type: ignore
        schedule: ScheduleType = {}
        for j, m in self.python_JM:
            schedule[j, m] = Span(round(pyo.value(self.pyo_x[j, m])), self.instance.ptimes[j, m]) # type: ignore
        return Schedule(schedule)

    def hint(self, schedule: Schedule) -> None:

        if self.washinted:
            raise RuntimeError("Model was already hinted")
        self.washinted = True

        starts = {key: value.start for key, value in schedule.schedule.items()}
        makespan = max(starts[j, m] + self.instance.ptimes[j, m] for j, m in self.python_JM)

        for j, m in self.python_JM:
            self.pyo_x[j, m].value = starts[j, m] # type: ignore

        self.pyo_C.value = makespan # type: ignore
        self.constraint5 = pyo.Constraint(expr = self.pyo_C <= makespan) # type: ignore

class TimeIndex(_BaseModel):

    def __init__(self, instance: Instance):

        super().__init__(instance)

        self.python_T = range(self.python_C_UB)
        self.pyo_T = pyo.RangeSet(0, self.python_C_UB - 1) # type: ignore

        self.pyo_x = pyo.Var(self.pyo_JM, self.pyo_T, within = pyo.Binary) # type: ignore
        self._fix_infeasible_starts()

        self.constraint1 = pyo.Constraint(self.pyo_JM, rule = self._constraintUniqueStart) # type: ignore
        self.constraint2 = pyo.Constraint(self.pyo_J, rule = self._constraintTotalMakespan) # type: ignore
        self.constraint3 = pyo.Constraint(self.pyo_JM, rule = self._constraintCorrectSequence) # type: ignore
        self.constraint4 = pyo.Constraint(self.pyo_M, self.pyo_T, rule = self._constraintUniqueMac) # type: ignore

    def _fix_infeasible_starts(self):
        for j, m in self.python_JM:
            for t in self.python_T:
                if t + self.instance.ptimes[j, m] > self.python_C_UB:
                    self.pyo_x[j, m, t].fix(0) # type: ignore

    def _constraintUniqueStart(self, model, j, m): # type: ignore
        return sum(model.pyo_x[j, m, t] for t in model.pyo_T) == 1 # type: ignore

    def _constraintTotalMakespan(self, model, j): # type: ignore
        m = model.instance.jobseq[j][-1] # type: ignore
        return sum((t + model.pyo_p[j, m]) * model.pyo_x[j, m, t] for t in model.pyo_T) <= model.pyo_C # type: ignore

    def _constraintCorrectSequence(self, model, j, m): # type: ignore
        o = model.instance.jobseq[j].prev(m) # type: ignore
        if o is None:
            return pyo.Constraint.Skip # type: ignore
        return sum((t + model.pyo_p[j, o]) * model.pyo_x[j, o, t] for t in model.pyo_T) <= sum(t * model.pyo_x[j, m, t] for t in model.pyo_T) # type: ignore

    def _constraintUniqueMac(self, model, m, t): # type: ignore
        expr = 0
        for j in model.pyo_J: # type: ignore
            if (j, m) not in model.pyo_JM: # type: ignore
                continue
            lo = max(0, t - model.pyo_p[j, m] + 1) # type: ignore
            for tau in range(lo, t + 1): # type: ignore
                expr += model.pyo_x[j, m, tau] # type: ignore
        return expr <= 1 # type: ignore

    def get_schedule(self, tol: float = 0.5):
        schedule: ScheduleType = {}
        for j, m in self.python_JM:
            starts = [int(t) for t in self.python_T if pyo.value(self.pyo_x[j, m, t]) > tol] # type: ignore
            if len(starts) == 0:
                raise ValueError(f"No start found for ({j = }, {m = })")
            if len(starts) > 1:
                raise ValueError(f"Only expected 1 start for ({j = }, {m = }) but got {len(starts)} starts: {starts}")
            start = starts[0]
            duration = self.instance.ptimes[j, m]
            schedule[(j, m)] = Span(start, duration)
        return Schedule(schedule)

    def hint(self) -> None:
        raise RuntimeError("Unimplemented")
