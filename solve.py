import os
import sys

import src.models.cp as cp
import src.models.pyomo as pyomo
import src.models.ais as ais
import src.models.ts as ts
import src.models.sa as sa
import src.models.icg as icg

from src.shared import Instance

class Program:

    MODELS = ["TI-HiGHS", "TI-MOSEK", "DI-HiGHS", "DI-MOSEK", "CP", "AIS", "TS", "SA", "ICG-HiGHS", "ICG-MOSEK"]

    def __init__(self, argv: list[str]) -> None:

        self.program = argv[0]
        self.file = os.path.basename(self.program)

        if len(argv) != 3:
            self.usage(1)

        self.model = argv[1]
        if self.model not in self.MODELS:
            self.usage(1)

        self.instance_name = argv[2]

    def usage(self, code: int) -> None:
        print(f"usage: {self.program} <model> <instance>")
        print()
        print(f"model:")
        for model in self.MODELS:
            print(f"  {model}")
        sys.exit(code)

def main(argv: list[str]) -> None:

    program = Program(argv)
    instance = Instance(program.instance_name)

    assert len(program.MODELS) == 10
    if program.model == "TI-HiGHS":
        model = pyomo.TimeIndex(instance)
        result = model.solve("appsi_highs", tee = True)
    elif program.model == "TI-MOSEK":
        model = pyomo.TimeIndex(instance)
        result = model.solve("mosek", tee = True)
    elif program.model == "DI-HiGHS":
        model = pyomo.Disjunctive(instance)
        result = model.solve("appsi_highs", tee = True)
    elif program.model == "DI-MOSEK":
        model = pyomo.Disjunctive(instance)
        result = model.solve("mosek", tee = True)
    elif program.model == "CP":
        model = cp.Disjunctive(instance)
        result = model.solve(tee = True)
    elif program.model == "AIS":
        model = ais.AIS(instance)
        result = model.solve(10000, 3, 'C')
    elif program.model == "TS":
        result = ts.solve(instance, 100000)
    elif program.model == "SA":
        result = sa.solve(instance, use_bidir = False)
    elif program.model == "ICG-HiGHS":
        result = icg.solve(instance, 3600, "appsi_highs")
    elif program.model == "ICG-MOSEK":
        result = icg.solve(instance, 3600, "mosek")
    else:
        raise RuntimeError("UNREACHABLE")

    print("======= RESULT =======")
    print(f"Name: {result.instance.name}")
    print(f"Path: {result.instance.path}")
    print(f"Size: {result.instance.jobs} x {result.instance.macs}")
    print(f"Rect: {result.instance.isrect}")
    print(f"Model: {result.model}")
    print(f"Solver: {result.solver}")
    print(f"Makespan: {result.schedule.makespan}")
    print(f"Optimum: {result.instance.optimum}")
    result.plot()

if __name__ == "__main__":
    main(sys.argv)
