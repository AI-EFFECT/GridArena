"""Shared type aliases used across multiple schema modules."""

from typing import Literal

Phase = Literal["R", "S", "T"]
GridLevel = Literal["MV", "LV", "Feeder"]
# What a historical database's power values are: "load" (consumption -- the
# default, and what PF data generation sources injections from), or a kind of
# generation injected into the grid (positive = generated): "pv" (solar PV),
# "chp" (cogeneration), "wind", "hydro", "other_generation" (other
# technologies) and "injection_total" (all generation injected into the grid).
HistoricalDataType = Literal["load", "pv", "chp", "wind", "hydro", "other_generation", "injection_total"]
HISTORICAL_DATA_TYPES = ("load", "pv", "chp", "wind", "hydro", "other_generation", "injection_total")
