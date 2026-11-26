"""Per-unit bases for the MV PF data-generation output.

Three-phase system bases: V_base is the grid's line-to-line nominal voltage and
S_base the run's three-phase power base (PFDataGenRequest.s_base_mva). The
derived impedance base is the same whether read line-to-line or on the
per-phase equivalent the solve runs on (see pf_datagen/three_phase.py):

    Z_base = V_LL^2 / S_3ph = (V_LL / sqrt(3))^2 / (S_3ph / 3)

so per-phase ohms and siemens (cable R/X, Y-bus G/B) divide by Z_base / Y_base
directly, and three-phase powers divide by S_base.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PerUnitBases:
    v_base_v: float   # line-to-line (V)
    s_base_va: float  # three-phase (VA)

    @classmethod
    def for_grid(cls, nominal_voltage_kv: float, s_base_mva: float) -> "PerUnitBases":
        return cls(v_base_v=nominal_voltage_kv * 1e3, s_base_va=s_base_mva * 1e6)

    @property
    def z_base_ohm(self) -> float:
        return self.v_base_v ** 2 / self.s_base_va

    @property
    def y_base_s(self) -> float:
        return 1.0 / self.z_base_ohm

    def as_dict(self) -> dict:
        return {
            "v_base_v": self.v_base_v,
            "s_base_va": self.s_base_va,
            "z_base_ohm": self.z_base_ohm,
            "y_base_s": self.y_base_s,
        }
