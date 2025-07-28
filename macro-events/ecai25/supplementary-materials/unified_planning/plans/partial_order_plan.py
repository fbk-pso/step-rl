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


import networkx as nx
import unified_planning as up
import unified_planning.plans as plans
from unified_planning.environment import Environment
from unified_planning.exceptions import UPUsageError
from unified_planning.plans.plan import ActionInstance
from unified_planning.plans.sequential_plan import SequentialPlan
from typing import Callable, Dict, Iterator, List, Optional


class PartialOrderPlan(plans.plan.Plan):
    """Represents a partial order plan. Actions are represent as an adjacency list graph."""

    def __init__(
        self,
        adjacency_list: Dict[
            "plans.plan.ActionInstance", List["plans.plan.ActionInstance"]
        ],
        environment: Optional["Environment"] = None,
        _graph: Optional[nx.DiGraph] = None,
    ):
        """
        Constructs the PartialOrderPlan using the adjacency list representation.

        :param adjacency_list: The Dictionary representing the adjacency list for this PartialOrderPlan.
        :param environment: The environment in which the ActionInstances in the adjacency_list are created.
        :param _graph: The graph that is semantically equivalent to the adjacency_list.
            NOTE: This parameter is for internal use only and it's maintainance is not guaranteed by any means.
        :return: The created PartialOrderPlan.
        """
        # if we have a specific environment or we don't have any actions
        if environment is not None or not adjacency_list:
            plans.plan.Plan.__init__(
                self, plans.plan.PlanKind.PARTIAL_ORDER_PLAN, environment
            )
        # If we don't have a specific environment, use the environment of the first action
        else:
            assert len(adjacency_list) > 0
            for ai in adjacency_list.keys():
                plans.plan.Plan.__init__(
                    self, plans.plan.PlanKind.PARTIAL_ORDER_PLAN, ai.action.environment
                )
                break
        if _graph is not None:
            # sanity checks
            assert len(adjacency_list) == 0
            assert all(isinstance(n, ActionInstance) for n in _graph.nodes)
            assert all(
                isinstance(f, ActionInstance) and isinstance(t, ActionInstance)
                for f, t in _graph.edges
            )
            self._graph = _graph
        else:
            for (
                ai_k,
                ai_v_list,
            ) in (
                adjacency_list.items()
            ):  # check that given environment and the environment in the actions is the same
                if ai_k.action.environment != self._environment:
                    raise UPUsageError(
                        "The environment given to the plan is not the same of the actions in the plan."
                    )
                for ai in ai_v_list:
                    if ai.action.environment != self._environment:
                        raise UPUsageError(
                            "The environment given to the plan is not the same of the actions in the plan."
                        )
            self._graph = nx.convert.from_dict_of_lists(
                adjacency_list, create_using=nx.DiGraph
            )

    def __repr__(self) -> str:
        return f"PartialOrderPlan({repr(self.get_adjacency_list)})"

    def __str__(self) -> str:
        ret = ["PartialOrderPlan:", "  actions:"]

        # give an ID, starting from 0, to every ActionInstance in the Plan
        swap_couple = lambda x: (x[1], x[0])
        id: Dict[ActionInstance, int] = dict(
            map(swap_couple, enumerate(nx.topological_sort(self._graph)))
        )
        convert_action_id = lambda action_id: f"    {action_id[1]}) {action_id[0]}"
        ret.extend(map(convert_action_id, id.items()))

        ret.append("  constraints:")
        adj_list = self.get_adjacency_list

        def convert_action_adjlist(action_adjlist):
            action = action_adjlist[0]
            adj_list = action_adjlist[1]
            get_id_as_str = lambda ai: str(id[ai])
            adj_list_str = " ,".join(map(get_id_as_str, adj_list))
            return f"    {id[action]} < {adj_list_str}"

        ret.extend(
            map(
                convert_action_adjlist,
                ((act, adj) for act, adj in adj_list.items() if adj),
            )
        )

        return "\n".join(ret)

    def __eq__(self, oth: object) -> bool:
        if isinstance(oth, PartialOrderPlan):
            return nx.is_isomorphic(
                self._graph,
                oth._graph,
                node_match=_semantically_equivalent_action_instances,
            )
        else:
            return False

    def __hash__(self) -> int:
        return hash(nx.weisfeiler_lehman_graph_hash(self._graph))

    def __contains__(self, item: object) -> bool:
        if isinstance(item, ActionInstance):
            return any(item.is_semantically_equivalent(a) for a in self._graph.nodes)
        else:
            return False

    @property
    def get_adjacency_list(
        self,
    ) -> Dict["plans.plan.ActionInstance", List["plans.plan.ActionInstance"]]:
        """Returns the graph of action instances as an adjacency list."""
        return nx.convert.to_dict_of_lists(self._graph)

    def replace_action_instances(
        self,
        replace_function: Callable[
            ["plans.plan.ActionInstance"], Optional["plans.plan.ActionInstance"]
        ],
    ) -> "plans.plan.Plan":
        """
        Returns a new `PartialOrderPlan` where every `ActionInstance` of the current plan is replaced using the given `replace_function`.

        :param replace_function: The function that applied to an `ActionInstance A` returns the `ActionInstance B`; `B`
            replaces `A` in the resulting `Plan`.
        :return: The `PartialOrderPlan` where every `ActionInstance` is replaced using the given `replace_function`.
        """
        # first replace all nodes and store the mapping, then use the mapping to
        # recreate the adjacency list representing the new graph
        # ai = action_instance
        original_to_replaced_ai: Dict[
            "plans.plan.ActionInstance", "plans.plan.ActionInstance"
        ] = {}
        for ai in self._graph.nodes:
            replaced_ai = replace_function(ai)
            if replaced_ai is not None:
                original_to_replaced_ai[ai] = replaced_ai

        new_adj_list: Dict[
            "plans.plan.ActionInstance", List["plans.plan.ActionInstance"]
        ] = {}

        # Populate the new adjacency list with the replaced action instances

        for ai in self._graph.nodes:
            replaced_ai = original_to_replaced_ai.get(ai, None)
            if replaced_ai is not None:
                replaced_ai = original_to_replaced_ai[ai]
                replaced_neighbors = []
                for successor in self._graph.neighbors(ai):
                    replaced_successor = original_to_replaced_ai.get(successor, None)
                    if replaced_successor is not None:
                        replaced_neighbors.append(replaced_successor)
                new_adj_list[replaced_ai] = replaced_neighbors

        new_env = self._environment
        for ai in new_adj_list.keys():
            new_env = ai.action.environment
            break
        return up.plans.PartialOrderPlan(new_adj_list, new_env)

    def convert_to(
        self,
        plan_kind: "plans.plan.PlanKind",
        problem: "up.model.AbstractProblem",
    ) -> "plans.plan.Plan":
        """
        This function takes a `PlanKind` and returns the representation of `self`
        in the given `plan_kind`. If the conversion does not make sense, raises
        an exception.

        For the conversion to `SequentialPlan`, returns one  all possible
        `SequentialPlans` that respects the ordering constraints given by
        this `PartialOrderPlan`.

        :param plan_kind: The plan_kind of the returned plan.
        :param problem: The `Problem` of which this plan is referring to.
        :return: The plan equivalent to self but represented in the kind of
            `plan_kind`.
        """
        if plan_kind == self._kind:
            return self
        elif plan_kind == plans.plan.PlanKind.SEQUENTIAL_PLAN:
            return SequentialPlan(
                list(nx.topological_sort(self._graph)), self._environment
            )
        else:
            raise UPUsageError(f"{type(self)} can't be converted to {plan_kind}.")

    def all_sequential_plans(self) -> Iterator[SequentialPlan]:
        """Returns all possible `SequentialPlans` that respects the ordering constraints given by this `PartialOrderPlan`."""
        for sorted_plan in nx.all_topological_sorts(self._graph):
            yield SequentialPlan(list(sorted_plan), self._environment)

    def get_neighbors(
        self, action_instance: ActionInstance
    ) -> Iterator[ActionInstance]:
        """
        Returns an `Iterator` over all the neighbors of the given `ActionInstance`.

        :param action_instance: The `ActionInstance` of which neighbors must be retrieved.
        :return: The `Iterator` over all the neighbors of the given `action_instance`.
        """
        try:
            retval = self._graph.neighbors(action_instance)
        except nx.NetworkXError:
            raise UPUsageError(
                f"The action instance {str(action_instance)} does not belong to this Partial Order Plan. \n Note that 2 Action Instances are equals if and only if they are the exact same object."
            )
        return retval


def _semantically_equivalent_action_instances(
    action_instance_1: ActionInstance, action_instance_2: ActionInstance
) -> bool:
    return action_instance_1.is_semantically_equivalent(action_instance_2)
