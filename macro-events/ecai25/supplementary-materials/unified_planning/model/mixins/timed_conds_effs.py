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


from typing import Optional, Dict, List, Set, Union, Iterable

import unified_planning as up
from unified_planning.environment import Environment, get_environment
from unified_planning.exceptions import UPTypeError, UPUsageError
from unified_planning.model.timing import Timing


class TimedCondsEffs:
    """A set of timed conditions of effects."""

    def __init__(self, _env: Optional[Environment] = None):
        self._environment = get_environment(_env)
        self._conditions: Dict[
            "up.model.timing.TimeInterval", List["up.model.fnode.FNode"]
        ] = {}
        self._effects: Dict[
            "up.model.timing.Timing", List["up.model.effect.Effect"]
        ] = {}
        self._simulated_effects: Dict[
            "up.model.timing.Timing", "up.model.effect.SimulatedEffect"
        ] = {}
        self._fluents_assigned: Dict[
            "up.model.timing.Timing",
            Dict["up.model.fnode.FNode", "up.model.fnode.FNode"],
        ] = {}
        self._fluents_inc_dec: Dict[
            "up.model.timing.Timing", Set["up.model.fnode.FNode"]
        ] = {}

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, TimedCondsEffs):
            if self._environment != oth._environment:
                return False
            if len(self._conditions) != len(oth._conditions):
                return False
            for i, cl in self._conditions.items():
                oth_cl = oth._conditions.get(i, None)
                if oth_cl is None:
                    return False
                elif set(cl) != set(oth_cl):
                    return False
            if len(self._effects) != len(oth._effects):
                return False
            for t, el in self._effects.items():
                oth_el = oth._effects.get(t, None)
                if oth_el is None:
                    return False
                elif set(el) != set(oth_el):
                    return False
            for t, se in self._simulated_effects.items():
                oth_se = oth._simulated_effects.get(t, None)
                if oth_se is None:
                    return False
                elif se != oth_se:
                    return False
            return True
        else:
            return False

    def __hash__(self) -> int:
        res = 0
        for i, cl in self._conditions.items():
            res += hash(i)
            for c in cl:
                res += hash(c)
        for t, el in self._effects.items():
            res += hash(t)
            for e in el:
                res += hash(e)
        for t, se in self._simulated_effects.items():
            res += hash(t) + hash(se)
        return res

    def clone(self) -> "TimedCondsEffs":
        new = TimedCondsEffs(self._environment)
        self._clone_to(new)
        return new

    def _clone_to(self, other: "TimedCondsEffs"):
        """Transfers deep copies of all `self` attributes into `other`"""
        other._conditions = {t: cl[:] for t, cl in self._conditions.items()}
        other._effects = {t: [e.clone() for e in el] for t, el in self._effects.items()}
        other._simulated_effects = {t: se for t, se in self._simulated_effects.items()}
        other._fluents_assigned = {
            t: d.copy() for t, d in self._fluents_assigned.items()
        }
        other._fluents_inc_dec = {
            t: fs.copy() for t, fs in self._fluents_inc_dec.items()
        }

    @property
    def conditions(
        self,
    ) -> Dict["up.model.timing.TimeInterval", List["up.model.fnode.FNode"]]:
        """
        Returns the `action conditions`; a map from `TimeInterval` to a `list` of `Expressions`
        indicating that for this `action` to be applicable, during the whole `TimeInterval`
        set as `key`, all the `expression` in the `mapped list` must evaluate to `True`.
        """
        return self._conditions

    def clear_conditions(self):
        """Removes all `conditions`."""
        self._conditions = {}

    @property
    def effects(self) -> Dict["up.model.timing.Timing", List["up.model.effect.Effect"]]:
        """
        Returns the all the `action's effects`; a map from `Timing` to `list` of `Effects`
        indicating that, when the action is applied, all the `Effects` must be applied at the
        `Timing` set as `key` in the map.
        """
        return self._effects

    def clear_effects(self):
        """Removes all `effects` from the `Action`."""
        self._effects = {}
        self._fluents_assigned = {}
        self._fluents_inc_dec = {}
        self._simulated_effects = {}

    @property
    def conditional_effects(
        self,
    ) -> Dict["up.model.timing.Timing", List["up.model.effect.Effect"]]:
        """
        Return the `action` `conditional effects`.

        IMPORTANT NOTE: this property does some computation, so it should be called as
        seldom as possible.
        """
        retval: Dict[up.model.timing.Timing, List[up.model.effect.Effect]] = {}
        for timing, effect_list in self._effects.items():
            cond_effect_list = [e for e in effect_list if e.is_conditional()]
            if len(cond_effect_list) > 0:
                retval[timing] = cond_effect_list
        return retval

    @property
    def unconditional_effects(
        self,
    ) -> Dict["up.model.timing.Timing", List["up.model.effect.Effect"]]:
        """
        Return the `action` `unconditional effects`.

        IMPORTANT NOTE: this property does some computation, so it should be called as
        seldom as possible.
        """
        retval: Dict[up.model.timing.Timing, List[up.model.effect.Effect]] = {}
        for timing, effect_list in self._effects.items():
            uncond_effect_list = [e for e in effect_list if not e.is_conditional()]
            if len(uncond_effect_list) > 0:
                retval[timing] = uncond_effect_list
        return retval

    def is_conditional(self) -> bool:
        """Returns `True` if the `action` has `conditional effects`, `False` otherwise."""
        return any(e.is_conditional() for l in self._effects.values() for e in l)

    def add_condition(
        self,
        interval: Union[
            "up.model.expression.TimeExpression", "up.model.timing.TimeInterval"
        ],
        condition: Union[
            "up.model.fnode.FNode",
            "up.model.fluent.Fluent",
            "up.model.parameter.Parameter",
            bool,
        ],
    ):
        """
        Adds the given expression to the `action's conditions`. For this `action` to be applicable
        the given expression must evaluate to `True` during the whole given `interval`.

        :param interval: The `interval` in which the given expression must evaluate to `True` for this
            `action` to be applicable.
        :param condition: The expression that must be `True` in the given `interval` for this
            `action` to be applicable.
        """
        if not isinstance(interval, up.model.TimeInterval):
            # transform from int/float/timepoint... to Timing
            timing = Timing.from_time(interval)
            interval = up.model.TimePointInterval(timing)  # and from Timing to Interval
        (condition_exp,) = self._environment.expression_manager.auto_promote(condition)
        assert self._environment.type_checker.get_type(condition_exp).is_bool_type()
        conditions = self._conditions.setdefault(interval, [])
        if condition_exp not in conditions:
            conditions.append(condition_exp)

    def _set_conditions(
        self,
        interval: "up.model.timing.TimeInterval",
        conditions: List["up.model.fnode.FNode"],
    ):
        self._conditions[interval] = conditions

    def add_effect(
        self,
        timing: "up.model.expression.TimeExpression",
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
        forall: Iterable["up.model.variable.Variable"] = tuple(),
    ):
        """
        At the given time, adds the given assignment to the `action's effects`.

        :param timing: The exact time in which the assignment is applied.
        :param fluent: The `fluent` which value is modified by the assignment.
        :param value: The `value` to assign to the given `fluent`.
        :param condition: The `condition` in which this `effect` is applied; the default
            value is `True`.
        :param forall: The 'Variables' that are universally quantified in this
            effect; the default value is empty.
        """
        (
            fluent_exp,
            value_exp,
            condition_exp,
        ) = self._environment.expression_manager.auto_promote(fluent, value, condition)
        if not fluent_exp.is_fluent_exp():
            raise UPUsageError(
                "fluent field of add_effect must be a Fluent or a FluentExp"
            )
        if not self._environment.type_checker.get_type(condition_exp).is_bool_type():
            raise UPTypeError("Effect condition is not a Boolean condition!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            raise UPTypeError(
                f"DurativeAction effect has an incompatible value type. Fluent type: {fluent_exp.type} // Value type: {value_exp.type}"
            )
        self._add_effect_instance(
            timing,
            up.model.effect.Effect(fluent_exp, value_exp, condition_exp, forall=forall),
        )

    def add_increase_effect(
        self,
        timing: "up.model.expression.TimeExpression",
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
        forall: Iterable["up.model.variable.Variable"] = tuple(),
    ):
        """
        At the given time, adds the given `increment` to the `action's effects`.

        :param timing: The exact time in which the increment is applied.
        :param fluent: The `fluent` which value is incremented by the added `effect`.
        :param value: The given `fluent` is incremented by the given `value`.
        :param condition: The `condition` in which this effect is applied; the default
            value is `True`.
        :param forall: The 'Variables' that are universally quantified in this
            effect; the default value is empty.
        """
        (
            fluent_exp,
            value_exp,
            condition_exp,
        ) = self._environment.expression_manager.auto_promote(fluent, value, condition)
        if not fluent_exp.is_fluent_exp():
            raise UPUsageError(
                "fluent field of add_increase_effect must be a Fluent or a FluentExp"
            )
        if not condition_exp.type.is_bool_type():
            raise UPTypeError("Effect condition is not a Boolean condition!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            raise UPTypeError(
                f"DurativeAction effect has an incompatible value type. Fluent type: {fluent_exp.type} // Value type: {value_exp.type}"
            )
        if not fluent_exp.type.is_int_type() and not fluent_exp.type.is_real_type():
            raise UPTypeError("Increase effects can be created only on numeric types!")
        self._add_effect_instance(
            timing,
            up.model.effect.Effect(
                fluent_exp,
                value_exp,
                condition_exp,
                kind=up.model.effect.EffectKind.INCREASE,
                forall=forall,
            ),
        )

    def add_decrease_effect(
        self,
        timing: "up.model.expression.TimeExpression",
        fluent: Union["up.model.fnode.FNode", "up.model.fluent.Fluent"],
        value: "up.model.expression.Expression",
        condition: "up.model.expression.BoolExpression" = True,
        forall: Iterable["up.model.variable.Variable"] = tuple(),
    ):
        """
        At the given time, adds the given `decrement` to the `action's effects`.

        :param timing: The exact time in which the `decrement` is applied.
        :param fluent: The `fluent` which value is decremented by the added effect.
        :param value: The given `fluent` is decremented by the given `value`.
        :param condition: The `condition` in which this effect is applied; the default
            value is `True`.
        :param forall: The 'Variables' that are universally quantified in this
            effect; the default value is empty.
        """
        (
            fluent_exp,
            value_exp,
            condition_exp,
        ) = self._environment.expression_manager.auto_promote(fluent, value, condition)
        if not fluent_exp.is_fluent_exp():
            raise UPUsageError(
                "fluent field of add_decrease_effect must be a Fluent or a FluentExp"
            )
        if not condition_exp.type.is_bool_type():
            raise UPTypeError("Effect condition is not a Boolean condition!")
        if not fluent_exp.type.is_compatible(value_exp.type):
            raise UPTypeError(
                f"DurativeAction effect has an incompatible value type. Fluent type: {fluent_exp.type} // Value type: {value_exp.type}"
            )
        if not fluent_exp.type.is_int_type() and not fluent_exp.type.is_real_type():
            raise UPTypeError("Decrease effects can be created only on numeric types!")
        self._add_effect_instance(
            timing,
            up.model.effect.Effect(
                fluent_exp,
                value_exp,
                condition_exp,
                kind=up.model.effect.EffectKind.DECREASE,
                forall=forall,
            ),
        )

    def _add_effect_instance(
        self,
        timing: "up.model.expression.TimeExpression",
        effect: "up.model.effect.Effect",
    ):
        timing = Timing.from_time(timing)
        assert (
            self._environment == effect.environment
        ), "effect does not have the same environment of the action"
        fluents_assigned = self._fluents_assigned.setdefault(timing, {})
        fluents_inc_dec = self._fluents_inc_dec.setdefault(timing, set())
        simulated_effect = self._simulated_effects.get(timing, None)
        up.model.effect.check_conflicting_effects(
            effect,
            timing,
            simulated_effect,
            fluents_assigned,
            fluents_inc_dec,
            f"action or problem: {self.name}",  # type: ignore[attr-defined]
        )
        self._effects.setdefault(timing, []).append(effect)

    @property
    def simulated_effects(
        self,
    ) -> Dict["up.model.timing.Timing", "up.model.effect.SimulatedEffect"]:
        """Returns the `action` `simulated effects`."""
        return self._simulated_effects

    def set_simulated_effect(
        self,
        timing: "up.model.timing.Timing",
        simulated_effect: "up.model.effect.SimulatedEffect",
    ):
        """
        Sets the given `simulated effect` at the specified `timing`.

        :param timing: The time in which the `simulated effect` must be applied.
        :param simulated_effect: The `simulated effect` that must be applied at the given `timing`.
        """
        up.model.effect.check_conflicting_simulated_effects(
            simulated_effect,
            timing,
            self._fluents_assigned.setdefault(timing, {}),
            self._fluents_inc_dec.get(timing, set()),
            f"action or problem: {self.name}",  # type: ignore[attr-defined]
        )
        if simulated_effect.environment != self._environment:
            raise UPUsageError(
                "The added SimulatedEffect does not have the same environment of the Action"
            )
        self._simulated_effects[timing] = simulated_effect
