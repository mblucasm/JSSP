import os
import sys

import src.models.cp
import src.models.pyomo
import src.models.ais

from src.shared import Instance

class Program:

    MODELS = ["TI-HiGHS", "TI-MOSEK", "DI-HiGHS", "DI-MOSEK", "CP", "AIS"]

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

    assert len(program.MODELS) == 6
    if program.model == "TI-HiGHS":
        model = src.models.pyomo.TimeIndex(instance)
        result = model.solve("appsi_highs", tee = True)
    elif program.model == "TI-MOSEK":
        model = src.models.pyomo.TimeIndex(instance)
        result = model.solve("mosek", tee = True)
    elif program.model == "DI-HiGHS":
        model = src.models.pyomo.Disjunctive(instance)
        result = model.solve("appsi_highs", tee = True)
    elif program.model == "DI-MOSEK":
        model = src.models.pyomo.Disjunctive(instance)
        result = model.solve("mosek", tee = True)
    elif program.model == "CP":
        model = src.models.cp.Disjunctive(instance)
        result = model.solve(tee = True)
    elif program.model == "AIS":
        model = src.models.ais.AIS(instance)
        result = model.solve(10000, 3, 'C')
    else:
        raise RuntimeError("UNREACHABLE")

    result.plot()

if __name__ == "__main__":
    main(sys.argv)
