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
"""
This module defines the Method class.
A Method has a name, a list of Parameters, a list of conditions
and a list of subtasks.
"""
from collections import OrderedDict
from typing import List, Union, Optional

import unified_planning as up
from unified_planning.environment import Environment
from unified_planning.exceptions import UPUnboundedVariablesError, UPValueError
from unified_planning.model.htn.task_network import AbstractTaskNetwork
from unified_planning.model.parameter import Parameter
from unified_planning.model.htn.task import Task
from unified_planning.model.expression import Expression


class ParameterizedTask:
    """A task instantiated with some parameters."""

    def __init__(self, task: Task, *params: Parameter):
        self._task = task
        self._params: List[Parameter] = list(params)
        assert len(self._task.parameters) == len(self._params)
        # TODO #153: check that the type of each parameter is compatible with the task' signature

    def __repr__(self):
        return str(self._task.name) + "(" + ", ".join(map(str, self.parameters)) + ")"

    def __eq__(self, other):
        return (
            isinstance(other, ParameterizedTask)
            and self._task == other._task
            and self._params == other._params
        )

    def __hash__(self):
        return hash(self._task) + sum(map(hash, self._params))

    @property
    def task(self) -> Task:
        return self._task

    @property
    def parameters(self) -> List["up.model.parameter.Parameter"]:
        return self._params


class Method(AbstractTaskNetwork):
    """HTN Method: encoding of a procedure for achieving a high-level task."""

    def __init__(
        self,
        _name: str,
        _parameters: Optional[
            "Union[OrderedDict[str, up.model.types.Type], List[Parameter]]"
        ] = None,
        _env: Optional[Environment] = None,
        **kwargs: "up.model.types.Type",
    ):
        super(Method, self).__init__(_env)
        self._task: Optional[ParameterizedTask] = None
        self._name = _name
        self._parameters: "OrderedDict[str, Parameter]" = OrderedDict()
        self._preconditions: List[up.model.fnode.FNode] = []
        if _parameters is None:
            for n, t in kwargs.items():
                self._parameters[n] = Parameter(n, t, self._env)
        elif isinstance(_parameters, List):
            assert len(kwargs) == 0
            for p in _parameters:
                self._parameters[p.name] = p
        else:
            assert isinstance(_parameters, OrderedDict)
            assert len(kwargs) == 0
            for n, t in _parameters.items():
                self._parameters[n] = Parameter(n, t, self._env)

    def __repr__(self) -> str:
        s = []
        s.append(f"method {self.name}")
        first = True
        for p in self.parameters:
            if first:
                s.append("(")
                first = False
            else:
                s.append(", ")
            s.append(str(p))
        if not first:
            s.append(")")
        s.append(" {\n")
        s.append(f"  task = {self._task}\n")
        if len(self.preconditions) > 0:
            s.append("  preconditions = [\n")
            for c in self.preconditions:
                s.append(f"    {str(c)}\n")
            s.append("  ]\n")
        if len(self.constraints) > 0:
            s.append("  constraints = [\n")
            for c in self.constraints:
                s.append(f"    {str(c)}\n")
            s.append("  ]\n")
        if len(self.subtasks) > 0:
            s.append("  subtasks = [\n")
            for st in self.subtasks:
                s.append(f"      {str(st)}\n")
            s.append("  ]\n")
        s.append("}")
        return "".join(s)

    def __eq__(self, oth: object) -> bool:
        if not isinstance(oth, Method):
            return False
        return (
            self._env == oth._env
            and self._name == oth._name
            and self._parameters == oth._parameters
            and self._task == oth._task
            and set(self._preconditions) == set(oth._preconditions)
            and set(self.subtasks) == set(oth.subtasks)
            and set(self.constraints) == set(oth.constraints)
        )

    def __hash__(self) -> int:
        res = hash(self._name)
        res += hash(self._task)
        res += sum(map(hash, self.parameters))
        res += sum(map(hash, self._preconditions))
        res += sum(map(hash, self.constraints))
        res += sum(map(hash, self.subtasks))
        return res

    @property
    def name(self) -> str:
        """Returns the action name."""
        return self._name

    @property
    def achieved_task(self) -> ParameterizedTask:
        """Returns the task that this method achieves."""
        assert (
            self._task is not None
        ), "The achieved task was previously set (see the set_task method)."
        return self._task

    def set_task(self, task: Union[Task, ParameterizedTask], *arguments: Parameter):
        """Defines the task that is method achieves.

        It expects a Task and its arguments, either bundle in a `ParameterizedTask` instance of
        passed separetly.
        It is assumed that each parameter of the achieved task is a parameter of the method.

        # Examples
        >>> from unified_planning.shortcuts import *
        >>> from unified_planning.model.htn import *
        >>> Location = UserType("Location")
        >>> go = Task("go", target=Location)
        >>> m1 = Method("m-go1", target=Location)
        >>> task_achieved = ParameterizedTask(go, m1.parameter("target"))
        >>> m1.set_task(task_achieved)
        >>> m2 = Method("m-go2", source=Location, target=Location)
        >>> m2.set_task(go, m2.parameter("target"))
        >>> m3 = Method("m-go3", source=Location, target=Location)
        >>> m3.set_task(go) # Infer the parameters of the `go` task from the parameters of m3 with the same name
        """
        assert self._task is None, f"Method {self.name} was already assigned a task"
        if isinstance(task, ParameterizedTask):
            assert (
                len(arguments) == 0
            ), "Unexpected arguments passed along a ParameterizedTask"
            assert all(
                p in self.parameters for p in task.parameters
            ), "A parameter of the task does not appear as a parameter of the method."
            self._task = task
        elif isinstance(task, Task) and len(arguments) == 0:
            for task_param in task.parameters:
                assert (
                    task_param.name in self._parameters
                ), f"Missing task parameter '{task_param.name}' in method {self._name}. Please pass all parameters explicitly."
            self._task = ParameterizedTask(task, *task.parameters)
        else:
            assert all(
                p in self.parameters for p in arguments
            ), "An argument passed to the task does not appear as a parameter of the method."
            self._task = ParameterizedTask(task, *arguments)

    @property
    def parameters(self) -> List[Parameter]:
        """Returns the list of the method's parameters."""
        return list(self._parameters.values())

    def parameter(self, name: str) -> Parameter:
        """
        Returns the `parameter` of the `Method` with the given `name`.

        Example
        -------
        >>> from unified_planning.shortcuts import *
        >>> from unified_planning.model.htn import *
        >>> location_type = UserType("Location")
        >>> robot_type = UserType("Robot")
        >>> goto = Method("goto", robot=robot_type, target=location_type)
        >>> goto.parameter("robot")  # return the "robot" parameter of the method, with type "Robot"
        Robot robot
        >>> goto.parameter("target")
        Location target

        If a parameter's name (1) does not conflict with an existing attribute of `Method` and (2) does not start with '_'
        it can also be accessed as if it was an attribute of the method. For instance:

        >>> goto.target
        Location target

        :param name: The `name` of the target `parameter`.
        :return: The `parameter` of the `Method` with the given `name`.
        """
        for param in self.parameters:
            if param.name == name:
                return param
        raise UPValueError(f"Unknown parameter name: {name}")

    def __getattr__(self, parameter_name: str) -> "up.model.parameter.Parameter":
        if parameter_name.startswith("_"):
            # guard access as pickling relies on attribute error to be thrown even when
            # no attributes of the object have been set.
            # In this case accessing `self._name` or `self._parameters`, would re-invoke __getattr__
            raise AttributeError(f"Method has no attribute '{parameter_name}'")
        if parameter_name not in self._parameters:
            raise AttributeError(
                f"Method '{self.name}' has no attribute or parameter '{parameter_name}'"
            )
        return self._parameters[parameter_name]

    @property
    def preconditions(self) -> List["up.model.fnode.FNode"]:
        """Returns the list of the method's preconditions."""
        return self._preconditions

    def add_precondition(self, precondition: Expression):
        """Adds the given method precondition."""
        (precondition_exp,) = self._env.expression_manager.auto_promote(precondition)
        assert self._env.type_checker.get_type(precondition_exp).is_bool_type()
        if precondition_exp == self._env.expression_manager.TRUE():
            return
        free_vars = self._env.free_vars_oracle.get_free_variables(precondition_exp)
        if len(free_vars) != 0:
            raise UPUnboundedVariablesError(
                f"The precondition {str(precondition_exp)} has unbounded variables:\n{str(free_vars)}"
            )
        if precondition_exp not in self._preconditions:
            self._preconditions.append(precondition_exp)
