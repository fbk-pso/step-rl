# Copyright 2021-2023 AIPlan4EU project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import List, Set, Optional, Tuple, Iterable

from unified_planning.environment import get_environment
from unified_planning.model import FNode, TimepointKind, Timing, StartTiming, EndTiming
from unified_planning.model.walkers import AnyChecker


class TemporalConstraints:
    """A set of temporal constraints on a set of timepoints.
    `TotalOrder` and `PartialOrder` are two special cases that are represented as subclasses.
    """

    def __init__(self, constraints: List[FNode]):
        self.constraints = constraints

    def __repr__(self):
        return str(self.constraints)


class PartialOrder(TemporalConstraints):
    """A purely qualitative set of constraints that define a partial order on its elements."""

    def __init__(self, precedences: List[Tuple[str, str]]):
        self.precedences = precedences
        constraints = [
            get_environment().expression_manager.LT(
                EndTiming(container=a), StartTiming(container=b)
            )
            for (a, b) in precedences
        ]
        super().__init__(constraints)

    def __repr__(self):
        precs = map(lambda p: f"{p[0]} < {p[1]}", self.precedences)
        return f"[{', '.join(precs)}]"


class TotalOrder(PartialOrder):
    """A purely qualitative set of constraints that define a total order on its elements."""

    def __init__(self, order: List[str]):
        self.order = order
        precedences = [(order[i - 1], order[i]) for i in range(1, len(order))]
        super().__init__(precedences)

    def __repr__(self):
        return f"[{', '.join(self.order)}]"


def ordering(
    task_ids: Iterable[str], time_constraints: List[FNode]
) -> TemporalConstraints:
    has_time = AnyChecker(predicate=lambda e: e.is_timing_exp())
    assert all(
        has_time.any(c) for c in time_constraints
    ), "A temporal constraint has no time expression"

    precedences = []
    # try transforming all constraints into precedences
    for c in time_constraints:
        if not c.is_lt():
            break
        lhs = c.arg(0)
        rhs = c.arg(1)
        if not lhs.is_timing_exp() or not rhs.is_timing_exp():
            break
        lhs = lhs.timing()  # type: ignore
        rhs = rhs.timing()  # type: ignore
        if lhs.delay != 0 or rhs.delay != 0:  # type: ignore
            break
        lhs = lhs.timepoint  # type: ignore
        rhs = rhs.timepoint  # type: ignore
        if lhs.kind != TimepointKind.END or rhs.kind != TimepointKind.START:  # type: ignore
            break
        if lhs.container is None or rhs.container is None:  # type: ignore
            break
        precedences.append((lhs.container, rhs.container))  # type: ignore

    qualitative = len(precedences) == len(time_constraints)
    if not qualitative:
        # At least one constraint cannot be encoded as a precedence
        return TemporalConstraints(time_constraints)
    else:
        to = _build_total_order(set(task_ids), precedences)
        if to is not None:
            return TotalOrder(to)
        else:
            return PartialOrder(precedences)


def _build_total_order(
    tasks: Set[str], precedences: List[Tuple[str, str]]
) -> Optional[List[str]]:
    """Returns a total order over all elements, or None if the given precedences are not sufficient to impose a total order."""
    order = []
    pending_tasks = tasks.copy()
    pending_precedences = precedences.copy()

    while len(pending_tasks) > 0:
        # find all elements with no predecessors
        firsts = [
            t
            for t in pending_tasks
            if all(tgt != t for (src, tgt) in pending_precedences)
        ]
        if len(firsts) != 1:
            return None  # not exactly one leading element => not a total order
        first = firsts[0]
        order.append(first)
        # remove `first` from the pending tasks/constraints before continuing
        pending_tasks.remove(first)
        pending_precedences = [
            (src, tgt) for (src, tgt) in pending_precedences if src != first
        ]

    assert len(pending_tasks) == 0
    assert len(order) == len(tasks)
    assert set(order) == tasks
    return order
