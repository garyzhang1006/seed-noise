"""The rectangular array a reduced population becomes, and nothing more.

Every arm of the paper reduces to the same object: for each configuration, each
replicate run and each trait, one score computed on item half A and one computed
on half B, in each of the two phenotypes, plus a run-level gain covariate on each
half.  Once a population is in this shape the estimator does not know or care
whether it came from DataDecide's released per-instance files, from a forward
pass over PolyPythias checkpoints, or from a simulation, which is what lets the
nulls run through byte-identical code.

Shapes, fixed throughout:

    yA, yB      (N, R, K)   trait scores per half
    gainA, gainB (N, R)     mean per-byte log-likelihood over all choices
    batch       (N, R)      0 for the default run, 1 and 2 for the aux batch
    recipe      (N,)        cluster label for the wild cluster bootstrap
    size        (N,)        size-band label
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Phenotype", "Population", "MARGIN", "ACCURACY", "PHENOTYPES"]

MARGIN = "margin"
ACCURACY = "accuracy"
PHENOTYPES = (MARGIN, ACCURACY)


@dataclass(frozen=True)
class Phenotype:
    """One phenotype's half scores.  ``A`` and ``B`` are ``(N, R, K)``."""

    name: str
    A: np.ndarray
    B: np.ndarray

    def __post_init__(self) -> None:
        if self.A.shape != self.B.shape:
            raise ValueError(
                f"phenotype {self.name!r}: half A is {self.A.shape} and half B is "
                f"{self.B.shape}; the two halves must be scored on the same runs"
            )
        if self.A.ndim != 3:
            raise ValueError(
                f"phenotype {self.name!r}: expected (N, R, K), got {self.A.shape}"
            )
        if not (np.isfinite(self.A).all() and np.isfinite(self.B).all()):
            raise ValueError(
                f"phenotype {self.name!r} contains non-finite scores; a run that "
                "failed to score must be dropped with its whole configuration, "
                "because a partially observed cell breaks the deviation identity"
            )


class Population:
    """A reduced population of replicate runs, ready for the estimator."""

    def __init__(
        self,
        phenotypes: dict[str, Phenotype],
        gainA: np.ndarray | None = None,
        gainB: np.ndarray | None = None,
        batch: np.ndarray | None = None,
        recipe: np.ndarray | None = None,
        size: np.ndarray | None = None,
        traits: list[str] | None = None,
        config_ids: list | None = None,
        n_items: np.ndarray | None = None,
    ) -> None:
        if not phenotypes:
            raise ValueError("a population needs at least one phenotype")
        self.phenotypes = dict(phenotypes)
        shapes = {p.A.shape for p in self.phenotypes.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"phenotypes disagree on shape: {sorted(shapes)}; every phenotype "
                "must be computed on the same runs and the same traits"
            )
        self.N, self.R, self.K = next(iter(shapes))
        if self.R < 2:
            raise ValueError(
                f"R={self.R}: a configuration with one run has no within-"
                "configuration deviation and contributes nothing"
            )
        if self.N < 1:
            raise ValueError("population has no configurations")

        self.gainA = self._vec(gainA, "gainA")
        self.gainB = self._vec(gainB, "gainB")
        self.batch = (np.tile(np.arange(self.R), (self.N, 1))
                      if batch is None else np.asarray(batch, dtype=np.int64))
        if self.batch.shape != (self.N, self.R):
            raise ValueError(f"batch must be (N, R) = {(self.N, self.R)}")
        self.recipe = (np.arange(self.N) if recipe is None
                       else np.asarray(recipe, dtype=np.int64))
        self.size = (np.zeros(self.N, dtype=np.int64) if size is None
                     else np.asarray(size, dtype=np.int64))
        for name, arr in (("recipe", self.recipe), ("size", self.size)):
            if arr.shape != (self.N,):
                raise ValueError(f"{name} must be (N,) = {(self.N,)}, got {arr.shape}")
        self.traits = list(traits) if traits is not None else [
            f"trait{j}" for j in range(self.K)
        ]
        if len(self.traits) != self.K:
            raise ValueError(f"{len(self.traits)} trait names for K={self.K}")
        self.config_ids = list(config_ids) if config_ids is not None else list(
            range(self.N)
        )
        self.n_items = None if n_items is None else np.asarray(n_items, dtype=np.int64)

    def _vec(self, x, name):
        if x is None:
            return np.zeros((self.N, self.R), dtype=np.float64)
        a = np.asarray(x, dtype=np.float64)
        if a.shape != (self.N, self.R):
            raise ValueError(f"{name} must be (N, R) = {(self.N, self.R)}, got {a.shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"{name} contains non-finite values")
        return a

    # -- accessors ---------------------------------------------------------

    def pheno(self, name: str) -> Phenotype:
        try:
            return self.phenotypes[name]
        except KeyError:
            raise KeyError(
                f"no phenotype {name!r}; this population has "
                f"{sorted(self.phenotypes)}"
            ) from None

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.recipe).size)

    @property
    def size_bands(self) -> np.ndarray:
        return np.unique(self.size)

    def __len__(self) -> int:
        return self.N

    def subset(self, idx) -> "Population":
        """Configurations by index, for splits, bands and the configuration bootstrap."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            raise ValueError("empty configuration subset")
        if idx.min() < 0 or idx.max() >= self.N:
            raise ValueError(f"configuration index out of range for N={self.N}")
        ph = {k: Phenotype(v.name, v.A[idx], v.B[idx])
              for k, v in self.phenotypes.items()}
        return Population(
            ph, self.gainA[idx], self.gainB[idx], self.batch[idx],
            self.recipe[idx], self.size[idx], self.traits,
            [self.config_ids[i] for i in idx], self.n_items,
        )

    def subset_clusters(self, recipes) -> "Population":
        """Whole recipe clusters, drawn with replacement for the cluster bootstrap.

        A recipe drawn twice contributes two independent clusters, so the drawn
        copies receive fresh labels rather than sharing one.
        """
        idx, lab = [], []
        for new, r in enumerate(recipes):
            members = np.flatnonzero(self.recipe == int(r))
            if members.size == 0:
                raise ValueError(f"recipe {r} has no configurations")
            idx.extend(members.tolist())
            lab.extend([new] * members.size)
        out = self.subset(np.asarray(idx))
        out.recipe = np.asarray(lab, dtype=np.int64)
        return out

    def with_phenotype(self, name: str, A: np.ndarray, B: np.ndarray) -> "Population":
        """Same population with one phenotype's half scores replaced.

        Used by the mediation step, which recomputes every headline on residuals,
        and by the nulls, which replace scores while keeping the design fixed.
        """
        ph = dict(self.phenotypes)
        ph[name] = Phenotype(name, np.asarray(A, float), np.asarray(B, float))
        return Population(ph, self.gainA, self.gainB, self.batch, self.recipe,
                          self.size, self.traits, self.config_ids, self.n_items)

    def summary(self) -> dict:
        return {
            "N": self.N, "R": self.R, "K": self.K,
            "clusters": self.n_clusters,
            "size_bands": self.size_bands.tolist(),
            "phenotypes": sorted(self.phenotypes),
            "contrasts": self.N * (self.R - 1),
        }
