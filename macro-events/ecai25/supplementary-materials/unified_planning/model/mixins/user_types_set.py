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
from unified_planning.model.types import _UserType
from unified_planning.exceptions import UPProblemDefinitionError, UPValueError
from typing import List, Dict, Optional, cast


class UserTypesSetMixin:
    """
    This class is a mixin that contains a `set` of `user types` with some related methods.

    NOTE: when this mixin is used in combination with other mixins that share some
    of the attributes (e.g. `has_name_method`), it is required to pass the very same
    arguments to the mixins constructors.
    """

    def __init__(self, env, has_name_method):
        self._env = env
        self._has_name_method = has_name_method
        self._user_types: List["up.model.types.Type"] = []
        # The field _user_types_hierarchy stores the information about the types and the list of their sons.
        self._user_types_hierarchy: Dict[
            Optional["up.model.types.Type"], List["up.model.types.Type"]
        ] = {}

    def _add_user_type(self, type: "up.model.types.Type"):
        """This method adds a Type, together with all it's ancestors, to the user_types_hierarchy"""
        assert type.is_user_type()
        if type not in self._user_types:
            ut = cast(_UserType, type)
            if self._has_name_method(ut.name):
                msg = f"The type name {ut.name} is already used in the problem! Different elements of a problem can have the same name if the environment flag error_used_name is disabled."
                if self._env.error_used_name or any(
                    ut.name == cast(_UserType, t).name for t in self._user_types
                ):
                    raise UPProblemDefinitionError(msg)
                else:
                    warn(msg)
            if ut.father is not None:
                self._add_user_type(ut.father)
            self._user_types.append(type)

    @property
    def user_types(self) -> List["up.model.types.Type"]:
        """Returns the `list` of all the `user types` in the `problem`."""
        return self._user_types

    def user_type(self, name: str) -> "up.model.types.Type":
        """
        Returns the `user type` in the `problem` with the given `name`.

        :param name: The target `name` for the `type`.
        :return: The `type` in the `problem` with the given `name`.
        """
        for ut in self.user_types:
            assert ut.is_user_type()
            if cast(_UserType, ut).name == name:
                return ut
        raise UPValueError(f"UserType {name} is not defined!")

    def has_type(self, name: str) -> bool:
        """
        Returns `True` if the `type` with the given `name` is defined in the
        `problem`, `False`, otherwise.

        :param name: The target `name` for the `type`.
        :return: `True` if a `type` with the given `name` is in the `problem`,
            `False` otherwise.
        """
        for ut in self.user_types:
            assert ut.is_user_type()
            if cast(_UserType, ut).name == name:
                return True
        return False

    @property
    def user_types_hierarchy(
        self,
    ) -> Dict[Optional["up.model.types.Type"], List["up.model.types.Type"]]:
        """
        Returns a `Dict` where every `key` represents an `Optional Type` and the `value`
        associated to the `key` is the `List` of the `direct sons` of the `Optional Type`.

        All the `user types` corresponding to the 'None' key are fatherless.
        """
        res: Dict[Optional["up.model.types.Type"], List["up.model.types.Type"]] = {}
        for t in self._user_types:
            if t not in res:
                res[t] = []
            f = cast(_UserType, t).father
            if f not in res:
                res[f] = [t]
            else:
                res[f].append(t)
        return res

    def __eq__(self, other):
        return isinstance(other, UserTypesSetMixin) and set(self._user_types) == set(
            other._user_types
        )

    def __hash__(self):
        return sum(map(hash, self._user_types))

    def _clone_to(self, other: "UserTypesSetMixin"):
        other._user_types = self._user_types[:]
        other._user_types_hierarchy = self._user_types_hierarchy.copy()
