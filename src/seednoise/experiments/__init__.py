"""The four experiments, in the order the pre-registration runs them."""

from seednoise.experiments.e1_screen import run_screen
from seednoise.experiments.e2_primary import run_primary
from seednoise.experiments.e3_nulls import run_nulls
from seednoise.experiments.e4_bakeoff import run_bakeoff
from seednoise.experiments.e5_sensitivity import (
    run_gain_calibration, run_leave_one_out, run_resplit,
)
from seednoise.experiments.e6_splitsweep import run_splitsweep

__all__ = ["run_screen", "run_primary", "run_nulls", "run_bakeoff",
           "run_gain_calibration", "run_resplit", "run_leave_one_out",
           "run_splitsweep"]
