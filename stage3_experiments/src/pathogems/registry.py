"""A small generic registry for models, losses, and optimizers.

Motivation
----------
The training loop used to branch on `config.model`, `config.loss`,
`config.optimizer` via a chain of `if/elif/raise` inside `train.py`.
That pattern has two concrete problems:

1. **Every new experiment touches a central file.** Adding a linear
   Cox model or a new loss meant editing a branch in `train.py`.
   That's a merge-conflict magnet and makes "change one thing at a
   time" reviews noisier than they need to be.
2. **Silent typos.** `config.optimizer = "adm"` would raise a
   generic `ValueError("Unknown optimizer: 'adm'.")` that doesn't
   tell the user the thing they actually want to know: "did you mean
   'adam'?"

The `Registry[T]` class here is deliberately minimal (~40 lines) — it
is a dict plus a decorator plus a `difflib`-powered error message. The
module-level registries (`MODEL_REGISTRY`, `LOSS_REGISTRY`,
`OPTIMIZER_REGISTRY`) live next to the things they register, so adding
a new loss is a one-line change in `loss.py` rather than a rewrite of
`train.py`.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A named store of objects (or factory callables) of type T.

    Example:
        MODEL_REGISTRY: Registry[Callable[..., nn.Module]] = Registry("model")

        @MODEL_REGISTRY.register("omics_mlp")
        def build_omics_mlp(**kwargs) -> nn.Module:
            return OmicsMLP(OmicsMLPConfig(**kwargs))

        model = MODEL_REGISTRY.get("omics_mlp")(in_features=500, ...)
    """
    __slots__ = ("_kind", "_entries")

    def __init__(self, kind: str) -> None:
        # `kind` appears in error messages; keep it short and human-readable.
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: register `obj` under `name`. Duplicate names raise.

        The decorator returns the object unchanged so `@register(...)`
        composes with other decorators and can be applied to plain
        functions, classes, or pre-constructed values.
        """

        def decorator(obj: T) -> T:
            if name in self._entries:
                raise ValueError(
                    f"{self._kind}: name {name!r} is already registered "
                    f"(existing: {self._entries[name]!r})."
                )
            self._entries[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        """Look up `name`. On miss, suggest close matches via difflib."""
        if name not in self._entries:
            close = difflib.get_close_matches(name, self._entries.keys(), n=3, cutoff=0.5)
            hint = f" Did you mean: {close}?" if close else ""
            raise KeyError(
                f"Unknown {self._kind}: {name!r}. "
                f"Registered: {sorted(self._entries)}.{hint}"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        """Sorted list of registered names — handy for --help and tests."""
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self._kind!r}, names={self.names()})"
