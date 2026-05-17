import os
import sys

import src.models.cp
import src.models.pyomo

from src.shared import Instance

class Program:

    MODELS = ["TI-HiGHS", "TI-MOSEK", "DI-HiGHS", "DI-MOSEK", "CP"]

    def __init__(self, argv: list[str]) -> None:

        self.program = argv[0]
        self.file = os.path.basename(self.program)

        if len(argv) != 3 and len(argv) != 4:
            self.usage(1)

        self.instance_name = argv[1]
        self.out_path = argv[2]
        self.showcpath = "-cpath" in self.out_path

        if len(argv) == 4:
            self.model = argv[3]
            if self.model not in self.MODELS:
                self.usage(1)
        else:
            self.model = None

        if self.showcpath and not self.model:
            print(f"error: {self.program}: a model must be provided if -cpath is present")
            self.usage(1)

    def usage(self, code: int) -> None:
        print(f"usage: {self.program} <instance> <<out-path>.dot [model] | <out-path>-cpath.dot <model>")
        print()
        print(f"model:")
        for model in self.MODELS:
            print(f"  {model}")
        sys.exit(code)

def main(argv: list[str]) -> None:

    program = Program(argv)
    instance = Instance(program.instance_name)

    result = None
    if program.model:
        assert len(program.MODELS) == 5
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
        else:
            raise RuntimeError("UNREACHABLE")
        
    instance.digraph_write(program.out_path, result.schedule if result else None, program.showcpath)

if __name__ == "__main__":
    main(sys.argv)