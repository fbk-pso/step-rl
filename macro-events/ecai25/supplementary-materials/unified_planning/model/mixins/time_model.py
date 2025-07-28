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

from decimal import Decimal
from fractions import Fraction
from unified_planning.exceptions import UPProblemDefinitionError
from typing import Optional, Union


class TimeModelMixin:
    """
    This class defines the problem's mixin for the epsilon separation and the
    time-kind, that can be continuous or discrete.

    When this mixin is initialized, a default for both fields must be set.
    Then, the epsilon value or the time_kind can be changed based on the user's
    request.
    """

    def __init__(
        self,
        epsilon_default: Optional[Fraction],
        discrete_time: bool,
        self_overlapping: bool,
    ):
        self._epsilon = epsilon_default
        self._discrete_time = discrete_time
        self._self_overlapping = self_overlapping

    @property
    def epsilon(self) -> Optional[Fraction]:
        """
        This parameter has meaning only in temporal problems: it defines the minimum
        amount of time that can elapse between 2 temporal events. A temporal event can
        be, for example, the start of an action, the end of an action, an intermediate
        step of an action, a timed effect of the problem.

        When None, it means that this minimum step is chosen by the Engine the Problem
        is given to.
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_value: Optional[Union[float, Decimal, Fraction, str]]):
        if new_value is not None:
            if not isinstance(new_value, Fraction):
                try:
                    new_value = Fraction(new_value)
                except ValueError:
                    raise UPProblemDefinitionError(
                        "The epsilon of a problem must be convertible to a Fraction."
                    )
            if new_value < 0:
                raise UPProblemDefinitionError("The epsilon must be a positive value!")
        self._epsilon = new_value

    @property
    def discrete_time(self) -> bool:
        """Returns True if the problem time is discrete, False if it is continuous."""
        return self._discrete_time

    @discrete_time.setter
    def discrete_time(self, new_value: bool):
        self._discrete_time = new_value

    @property
    def self_overlapping(self) -> bool:
        """
        The ``self_overlapping`` flag determines if 2 (or more) different instances of the same
        action grounded with the same parameters can be running at the same time.
        """
        return self._self_overlapping

    @self_overlapping.setter
    def self_overlapping(self, new_value: bool):
        self._self_overlapping = new_value

    def _clone_to(self, other: "TimeModelMixin"):
        other.epsilon = self._epsilon
        other.discrete_time = self._discrete_time
        other.self_overlapping = self._self_overlapping
