from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from alphagen_style.dsl import OP_SPECS, WINDOW_CANDIDATES


SlotKind = Literal["series", "scalar", "window"]
ActionKind = Literal["op", "col", "scalar", "window"]


@dataclass(frozen=True)
class Slot:
    kind: SlotKind


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    value: Union[str, float, int]


@dataclass
class ExprState:
    # stack of pending slots with depth budget
    stack: List[Tuple[Slot, int]]


@dataclass(frozen=True)
class Node:
    def to_expr(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Col(Node):
    name: str

    def to_expr(self) -> str:
        return self.name


@dataclass(frozen=True)
class Scalar(Node):
    value: float

    def to_expr(self) -> str:
        # Keep stable repr for caching.
        v = float(self.value)
        if abs(v - int(v)) < 1e-12:
            return str(int(v))
        return repr(v)


@dataclass(frozen=True)
class Call(Node):
    name: str
    args: List[Node]

    def to_expr(self) -> str:
        inner = ", ".join(a.to_expr() for a in self.args)
        return f"{self.name}({inner})"


class ActionSpace:
    def __init__(
        self,
        *,
        columns: Sequence[str],
        scalar_candidates: Sequence[float] = (-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0),
        window_candidates: Sequence[int] = tuple(WINDOW_CANDIDATES),
        ops: Optional[Sequence[str]] = None,
    ):
        self.columns = list(columns)
        self.scalar_candidates = list(map(float, scalar_candidates))
        self.window_candidates = list(map(int, window_candidates))
        self.ops = list(ops) if ops is not None else sorted(OP_SPECS.keys())

        self.actions: List[Action] = []
        # op actions
        for op in self.ops:
            self.actions.append(Action("op", op))
        # terminal actions
        for c in self.columns:
            self.actions.append(Action("col", c))
        for s in self.scalar_candidates:
            self.actions.append(Action("scalar", float(s)))
        for w in self.window_candidates:
            self.actions.append(Action("window", int(w)))

        self._index: Dict[Tuple[ActionKind, Union[str, float, int]], int] = {
            (a.kind, a.value): i for i, a in enumerate(self.actions)
        }

        # precomputed index ranges
        self._op_idx = [i for i, a in enumerate(self.actions) if a.kind == "op"]
        self._col_idx = [i for i, a in enumerate(self.actions) if a.kind == "col"]
        self._scalar_idx = [i for i, a in enumerate(self.actions) if a.kind == "scalar"]
        self._window_idx = [i for i, a in enumerate(self.actions) if a.kind == "window"]

    @property
    def n(self) -> int:
        return len(self.actions)

    def mask(self, state: ExprState) -> np.ndarray:
        """Return boolean mask for current state."""
        mask = np.zeros(self.n, dtype=bool)
        if not state.stack:
            return mask

        slot, depth = state.stack[-1]
        if slot.kind == "window":
            mask[self._window_idx] = True
            return mask
        if slot.kind == "scalar":
            mask[self._scalar_idx] = True
            return mask

        # series
        mask[self._col_idx] = True
        if depth > 0:
            mask[self._op_idx] = True
        return mask


def initial_state(*, max_depth: int) -> ExprState:
    return ExprState(stack=[(Slot("series"), int(max_depth))])


def step(
    state: ExprState,
    action: Action,
    *,
    max_depth: int,
) -> None:
    """Apply action to state (in-place) and update pending slots."""
    if not state.stack:
        raise ValueError("State is terminal")

    slot, depth = state.stack.pop()

    if slot.kind == "window":
        if action.kind != "window":
            raise ValueError("Expected window action")
        return

    if slot.kind == "scalar":
        if action.kind != "scalar":
            raise ValueError("Expected scalar action")
        return

    # series
    if action.kind == "col":
        return

    if action.kind == "scalar":
        # allow scalar broadcast to series for convenience
        return

    if action.kind != "op":
        raise ValueError("Expected op/col action")

    op = str(action.value)
    spec = OP_SPECS[op]
    if depth <= 0:
        raise ValueError("No depth remaining for op")

    child_depth = depth - 1

    # push args in reverse (stack LIFO)
    for i in range(len(spec.arg_kinds) - 1, -1, -1):
        kind = spec.arg_kinds[i]
        if spec.window_arg is not None and i == spec.window_arg:
            state.stack.append((Slot("window"), child_depth))
        elif kind == "scalar":
            state.stack.append((Slot("scalar"), child_depth))
        else:
            state.stack.append((Slot("series"), child_depth))


def sample_actions(
    space: ActionSpace,
    *,
    max_depth: int,
    rng: np.random.Generator,
) -> List[Action]:
    st = initial_state(max_depth=max_depth)
    actions: List[Action] = []

    while st.stack:
        m = space.mask(st)
        idxs = np.flatnonzero(m)
        if len(idxs) == 0:
            break
        i = int(rng.choice(idxs))
        a = space.actions[i]
        actions.append(a)
        step(st, a, max_depth=max_depth)

    return actions


@dataclass
class _CallFrame:
    name: str
    remaining: int
    args: List[Node]


def actions_to_expr(
    space: ActionSpace,
    actions: Sequence[Action],
    *,
    max_depth: int,
) -> str:
    """Deterministically convert an action sequence into a DSL expression.

    Assumes actions were generated following the same stack discipline as `step()`.
    """

    st = initial_state(max_depth=max_depth)
    call_stack: List[_CallFrame] = []
    root: Optional[Node] = None

    for a in actions:
        if not st.stack:
            break

        slot, depth = st.stack.pop()
        node: Optional[Node] = None

        if slot.kind == "window":
            if a.kind != "window":
                raise ValueError("Expected window action")
            node = Scalar(float(int(a.value)))
        elif slot.kind == "scalar":
            if a.kind != "scalar":
                raise ValueError("Expected scalar action")
            node = Scalar(float(a.value))
        else:
            # series
            if a.kind == "col":
                node = Col(str(a.value))
            elif a.kind == "scalar":
                node = Scalar(float(a.value))
            elif a.kind == "op":
                if depth <= 0:
                    raise ValueError("No depth remaining for op")
                op = str(a.value)
                spec = OP_SPECS[op]
                call_stack.append(_CallFrame(op, len(spec.arg_kinds), []))

                child_depth = depth - 1
                for i in range(len(spec.arg_kinds) - 1, -1, -1):
                    kind = spec.arg_kinds[i]
                    if spec.window_arg is not None and i == spec.window_arg:
                        st.stack.append((Slot("window"), child_depth))
                    elif kind == "scalar":
                        st.stack.append((Slot("scalar"), child_depth))
                    else:
                        st.stack.append((Slot("series"), child_depth))
                continue
            else:
                raise ValueError("Invalid action for series slot")

        # Attach node to call frames
        while True:
            if call_stack:
                fr = call_stack[-1]
                fr.args.append(node)
                fr.remaining -= 1
                if fr.remaining == 0:
                    call_stack.pop()
                    node = Call(fr.name, fr.args)
                    continue
                break
            else:
                root = node
                break

    if call_stack:
        raise ValueError("Unfinished call frames")
    if root is None:
        raise ValueError("No root expression")
    return root.to_expr()


def sample_expression(
    space: ActionSpace,
    *,
    max_depth: int,
    rng: np.random.Generator,
) -> Tuple[str, List[Action]]:
    actions = sample_actions(space, max_depth=max_depth, rng=rng)
    expr = actions_to_expr(space, actions, max_depth=max_depth)
    return expr, actions
