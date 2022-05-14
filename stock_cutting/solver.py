import logging
import math
import numpy as np
import pandas as pd
import itertools

from ortools.sat.python import cp_model
from typing import List, Tuple

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_CP_SAT_OPTIMAL = "OPTIMAL"


class StockCuttingSolver:

    _waste_epsilon = .0001

    def __init__(
        self, 
        needed_lengths: List[float], 
        needed_quantities: List[int],
        total_length: float,
        cut_slippage: float = 0,
    ):
        self.needed_lengths = needed_lengths
        self.needed_quantities = needed_quantities
        self.total_length = total_length
        self._check_well_posed()
        self.cut_slippage = cut_slippage
        self.solution_dataframe = None
        self.solved = False

    def _check_well_posed(self):
        """ basic consistency checks """
        if self.total_length <= 0:
            raise ValueError(f"supplied `total_length` must be greater than 0 (got `{self.total_length}`)")
        
        _exceed_lengths = [x for x in self.needed_lengths if x > self.total_length]
        if _exceed_lengths:
            raise ValueError(f"{len(_exceed_lengths)} requested lengths exceed the `total_length`.")

    def _check_solved(self):
        if not self.solved:
            raise ValueError("Solver has not yet been invoked. Ensure that `StockCuttingSolver.solve()` has been invoked.")

    @staticmethod
    def generate_possible_cuts(needed_lengths: List[float], total_length: float, cut_slippage: float = 0) -> List[Tuple[float]]:
        # Establish a bound on the number of combinations that we need
        max_combinations = math.floor(total_length / min(needed_lengths))
        cuts = []
        for i in range(1, max_combinations + 1):
            for comb in itertools.combinations_with_replacement(needed_lengths, i):
                if len(comb) == 1 and comb[0] == base_length:
                    cuts.append(comb)
                    continue
    
                waste = len(comb) * cut_slippage
                if sum(comb) <= base_length - waste:
                    cuts.append(comb)
        return cuts
        
    def _build_model(self):
        self.model = cp_model.CpModel()
        self.naive_max_cuts = sum(self.needed_quantities)
        self.possible_cuts = self.generate_possible_cuts(
            needed_lengths=self.needed_lengths,
            total_length=self.total_length,
            cut_slippage=self.cut_slippage,
        )

        self.cut_produced_quantities = {}
        for i, cut in enumerate(self.possible_cuts):
            for j, length in enumerate(self.needed_lengths):
                self.cut_produced_quantities[i, j] = cut.count(length)

        self.cut_wastes = {}
        for i, cut in enumerate(self.possible_cuts):
            self.cut_wastes[i] = self.base_length - sum(cut) + self._waste_epsilon
        
        self.cut_vars = {}
        for i, cut in enumerate(self.possible_cuts):
            self.cut_vars[i] = self.model.NewIntVar(
                0, 
                self.naive_max_cuts, 
                f"Cut into segments of lengths {cut}"
            )

        # Constraint: cuts must fufill needed quantities
        for (j, length), needed in zip(enumerate(self.needed_lengths), self.needed_quantities):
            self.model.Add(
                sum(var * self.cut_produced_quantities[i, j] for i, var in self.cut_vars.items())
                    == needed
            )

        # Objective: Minimize total cut waste
        total_cut_waste = sum(
            var * self.cut_wastes[i] for i, var in self.cut_vars.items()
        )
        self.model.Minimize(total_cut_waste)

    def solve(self, **solver_kwargs):
        self._build_model()
        self.solver = cp_model.CpSolver(**solver_kwargs)
        self.status = self.solver.Solve(self.model)
        _status_name = self.solver.StatusName()
        if _status_name != _CP_SAT_OPTIMAL:
            logger.warning(
                f"Solver responded with non-optimal status of `{_status_name}`. You may wish to invoke `StockCuttingSolver.print_solution_stats()`"
            )

    def print_solution_stats(self):
        print(self.solver.ResponseStats())

    def _build_solution_dataframe(self):
        solution_df = []
        for i, var in self.cut_vars.items():
            solution_df.append(
                {
                    "cut_index": i,
                    "cut": self.possible_cuts[i],
                    "number_of_cuts": self.solver.Value(var),
                    "cut_waste": self.cut_wastes[i] - self.waste_epsilon
                    "total_waste": (self.cut_wastes[i] - self.waste_epsilon) * self.solver.Value(var),
                }
            )
        self.solution_dataframe = pd.DataFrame(solution_df)

    def get_solution_frame(self) -> pd.DataFrame:
        self._build_solution_dataframe()
        return self.solution_dataframe

    def compute_waste(self) -> float:
        df = self.get_solution_frame()
        return df["total_waste"].sum()

    def compute_efficiency(self) -> float:
        waste = self.compute_waste()
        total_length_needed = sum(
            q * l for q, l in zip(self.needed_quantities, self.needed_lengths)
        )
        return 1 - waste / total_length_needed