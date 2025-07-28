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

from warnings import warn
import unified_planning as up
import unified_planning.engines.mixins as mixins
from unified_planning.model import ProblemKind
from unified_planning.engines.engine import Engine
from unified_planning.engines.meta_engine import MetaEngine
from unified_planning.engines.results import (
    PlanGenerationResultStatus,
    PlanGenerationResult,
)
from unified_planning.engines.mixins.oneshot_planner import OptimalityGuarantee
from unified_planning.exceptions import UPUsageError
from typing import Type, IO, Callable, Optional, Union, List, Tuple
from fractions import Fraction


class Replanner(MetaEngine, mixins.ReplannerMixin):
    """
    This :class:`~unified_planning.engines.MetaEngine` implements the :func:`~unified_planning.engines.Factory.Replanner>` operation mode starting
    a new oneshot planning query with the updated :class:`~unified_planning.model.AbstractProblem` instance.
    """

    def __init__(
        self,
        problem: "up.model.AbstractProblem",
        error_on_failed_checks: "bool",
        *args,
        **kwargs,
    ):
        MetaEngine.__init__(self, *args, **kwargs)
        mixins.ReplannerMixin.__init__(
            self, problem=problem, error_on_failed_checks=error_on_failed_checks
        )

    @property
    def name(self) -> str:
        return f"Replanner[{self.engine.name}]"

    @staticmethod
    def is_compatible_engine(engine: Type[Engine]) -> bool:
        return engine.is_oneshot_planner() and engine.supports(ProblemKind({"ACTION_BASED"}))  # type: ignore

    @staticmethod
    def satisfies(optimality_guarantee: OptimalityGuarantee) -> bool:
        return False

    @staticmethod
    def _supported_kind(engine: Type[Engine]) -> "ProblemKind":
        engine_supported_kind = engine.supported_kind()
        supported_kind = engine_supported_kind.clone()
        supported_kind.unset_problem_class("HIERARCHICAL")
        return supported_kind

    @staticmethod
    def _supports(problem_kind: "ProblemKind", engine: Type[Engine]) -> bool:
        return problem_kind <= Replanner._supported_kind(engine)

    def _resolve(
        self,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
    ) -> "up.engines.results.PlanGenerationResult":
        assert isinstance(self._problem, up.model.Problem)
        assert isinstance(self.engine, mixins.OneshotPlannerMixin)
        return self.engine.solve(
            self._problem, timeout=timeout, output_stream=output_stream
        )

    def _update_initial_value(
        self,
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.object.Object",
            bool,
            int,
            float,
            Fraction,
        ],
    ):
        assert isinstance(self._problem, up.model.Problem)
        self._problem.set_initial_value(fluent, value)

    def _add_goal(
        self, goal: Union["up.model.fnode.FNode", "up.model.fluent.Fluent", bool]
    ):
        assert isinstance(self._problem, up.model.Problem)
        self._problem.add_goal(goal)

    def _remove_goal(
        self, goal: Union["up.model.fnode.FNode", "up.model.fluent.Fluent", bool]
    ):
        assert isinstance(self._problem, up.model.Problem)
        (goal_exp,) = self._problem.environment.expression_manager.auto_promote(goal)
        goals = self._problem.goals
        self._problem.clear_goals()
        removed = False
        for g in goals:
            if not g is goal_exp:
                self._problem.add_goal(g)
            else:
                removed = True
        if not self._skip_checks and not removed:
            msg = f"goal to remove: {goal_exp} not found inside the problem goals: {goals}"
            if self._error_on_failed_checks:
                raise UPUsageError(msg)
            else:
                warn(msg)

    def _add_action(self, action: "up.model.action.Action"):
        assert isinstance(self._problem, up.model.Problem)
        self._problem.add_action(action)

    def _remove_action(self, name: str):
        assert isinstance(self._problem, up.model.Problem)
        actions = self._problem.actions
        self._problem.clear_actions()
        removed = False
        for a in actions:
            if a.name != name:
                self._problem.add_action(a)
            else:
                removed = True
        if not self._skip_checks and not removed:
            msg = f"action to remove: {name} not found inside the problem actions: {list(map(lambda a: a.name, actions))}"
            if self._error_on_failed_checks:
                raise UPUsageError(msg)
            else:
                warn(msg)
