"""Adapter package for harmonization methods."""
from .base import HarmonizerBase
from .site_covar import SiteCovarHarmonizer
from .neuroharmonize_wrapper import NeuroHarmonizeAdapter
from .neurocombat_wrapper import NeuroCombatAdapter
from .covbat_wrapper import CovBatAdapter


def build(method: str, batch: str, protected: tuple[str, ...]) -> HarmonizerBase:
    """Factory: map --method strings to adapter instances."""
    method = method.lower()
    if method == "none":
        from .base import HarmonizerBase as _Base
        # Identity adapter for the "none" baseline
        import numpy as np
        import pandas as pd

        class _Identity(_Base):
            def fit(self, features, meta):
                self.fitted_state["fitted"] = True
                return self

            def transform(self, features, meta):
                return features

            def method_name(self):
                return "none"

        return _Identity(batch=batch, protected=protected)
    if method == "site-covar":
        return SiteCovarHarmonizer(batch=batch, protected=protected)
    if method == "combat":
        return NeuroHarmonizeAdapter(batch=batch, protected=protected, method="combat")
    if method == "combat-gam":
        return NeuroHarmonizeAdapter(
            batch=batch, protected=protected, method="combat-gam"
        )
    if method == "combat-raw":
        return NeuroCombatAdapter(batch=batch, protected=protected)
    if method == "covbat":
        return CovBatAdapter(batch=batch, protected=protected)
    raise ValueError(
        f"Unknown method '{method}'. Expected one of: "
        "none, site-covar, combat, combat-gam, combat-raw, covbat"
    )


__all__ = [
    "HarmonizerBase",
    "SiteCovarHarmonizer",
    "NeuroHarmonizeAdapter",
    "NeuroCombatAdapter",
    "CovBatAdapter",
    "build",
]
