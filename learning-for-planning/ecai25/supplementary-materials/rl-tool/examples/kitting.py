import itertools
from examples import ProblemGenerator
from typing import Iterator, Optional, Tuple, Dict, List, Union
import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.model.walkers import AnyChecker

class KittingGenerator(ProblemGenerator):
    def __init__(self, n_components: int, kit_size: Union[int,Tuple[int,int]], n_kit: Union[int,Tuple[int,int]], max_kit: int, n_robots: Union[int,Tuple[int,int]]) -> None:
        super().__init__()
        self._n_components = n_components
        self._min_kit_size = kit_size[0] if isinstance(kit_size, tuple) else 0
        self._max_kit_size = kit_size[1] if isinstance(kit_size, tuple) else kit_size
        self._min_n_kit = n_kit[0] if isinstance(n_kit, tuple) else 1
        self._max_n_kit = n_kit[1] if isinstance(n_kit, tuple) else n_kit
        self._min_n_robots = n_robots[0] if isinstance(n_robots, tuple) else 1
        self._max_n_robots = n_robots[1] if isinstance(n_robots, tuple) else n_robots
        self._max_kit = max_kit
        self._domain = self._build_domain()

    def _build_domain(self) -> up.model.Problem:
        domain = Problem()

        Robot = UserType("Robot")
        Location = UserType("Location")
        Component = UserType("Component")
        Kit = UserType("Kit")

        l0 = Object("l0", Location)
        k1 = Object("k1", Kit)
        EMPTY = Object("EMPTY", Component)
        domain.add_objects([l0, k1, EMPTY])
        for i in range(1, self._n_components + 1):
            l = Object(f"l{i}", Location)
            c = Object(f"c{i}", Component)
            domain.add_objects([l, c])

        for i in range(self._max_n_robots):
            r = Object(f"r{i}", Robot)
            domain.add_object(r)

        distance = Fluent("distance", IntType(0, 1), a=Location, b=Location)
        is_present = Fluent("is_present", BoolType(), c=Component, l=Location)
        components_on_kit = Fluent("components_on_kit", Component, k=Kit, i=IntType(0, self._max_kit_size-1))
        robot_busy = Fluent("robot_busy", r=Robot)
        human_busy = Fluent("human_busy")
        ready_to_receive = Fluent("ready_to_receive", BoolType(), i=IntType(0, self._max_kit-1))
        robot_at = Fluent("robot_at", BoolType(), r=Robot, l=Location)
        components_on_robot = Fluent("components_on_robot", Component, r=Robot, i=IntType(0, self._max_kit_size-1))
        completed = Fluent("completed", BoolType(), i=IntType(0, self._max_kit-1), k=Kit)
        robot_cnt = Fluent("robot_cnt", IntType(0, self._max_kit_size), r=Robot)
        kit_cnt = Fluent("kit_cnt", IntType(0, self._max_kit))
        battery = Fluent("battery", IntType(0, 10), r=Robot)

        domain.add_fluent(distance, default_initial_value=1)
        domain.add_fluent(is_present, default_initial_value=False)
        domain.add_fluent(components_on_kit, default_initial_value=EMPTY)
        domain.add_fluent(robot_busy, default_initial_value=False)
        domain.add_fluent(human_busy, default_initial_value=False)
        domain.add_fluent(ready_to_receive, default_initial_value=False)
        domain.add_fluent(robot_at, default_initial_value=False)
        domain.add_fluent(components_on_robot, default_initial_value=EMPTY)
        domain.add_fluent(completed, default_initial_value=False)
        domain.add_fluent(robot_cnt, default_initial_value=0)
        domain.add_fluent(kit_cnt, default_initial_value=0)
        domain.add_fluent(battery, default_initial_value=self._max_kit_size+1)

        move = DurativeAction('move', r=Robot, l_from=Location, l_to=Location)
        r = move.parameter('r')
        l_from = move.parameter('l_from')
        l_to = move.parameter('l_to')
        move.set_fixed_duration(distance(l_from, l_to))
        move.add_condition(StartTiming(), Not(robot_busy(r)))
        move.add_effect(StartTiming(), robot_busy(r), True)
        move.add_effect(EndTiming(), robot_busy(r), False)
        move.add_condition(StartTiming(), Not(Equals(l_from, l_to)))
        move.add_condition(StartTiming(), robot_at(r, l_from))
        move.add_condition(StartTiming(), GT(distance(l_from, l_to), 0))
        move.add_condition(StartTiming(), GT(battery(r), 0))
        move.add_decrease_effect(StartTiming(), battery(r), 1)
        move.add_effect(StartTiming(), robot_at(r, l_from), False)
        move.add_effect(EndTiming(), robot_at(r, l_to), True)
        domain.add_action(move)

        load = DurativeAction("load", r=Robot, l=Location, c=Component, k=Kit, i=IntType(0, self._max_kit_size-1))
        r = load.parameter("r")
        l = load.parameter("l")
        c = load.parameter("c")
        k = load.parameter("k")
        i = load.parameter("i")
        load.set_fixed_duration(5)
        load.add_condition(StartTiming(), Not(robot_busy(r)))
        load.add_effect(StartTiming(), robot_busy(r), True)
        load.add_effect(EndTiming(), robot_busy(r), False)
        load.add_condition(StartTiming(), robot_at(r, l))
        load.add_condition(StartTiming(), is_present(c, l))
        load.add_condition(StartTiming(), Equals(robot_cnt(r), i))
        load.add_condition(StartTiming(), Equals(components_on_robot(r, i), EMPTY))
        load.add_condition(StartTiming(), Equals(components_on_kit(k, i), c))
        load.add_effect(EndTiming(), components_on_robot(r, i), c)
        load.add_effect(EndTiming(), robot_cnt(r), Plus(i, 1))
        domain.add_action(load)

        prepare_unload = DurativeAction("prepare_unload", i=IntType(0, self._max_kit-1))
        i = prepare_unload.parameter("i")
        prepare_unload.set_fixed_duration(30)
        prepare_unload.add_condition(StartTiming(), Not(human_busy))
        prepare_unload.add_effect(StartTiming(), human_busy, True)
        prepare_unload.add_effect(EndTiming(), human_busy, False)
        prepare_unload.add_condition(StartTiming(), Equals(kit_cnt, i))
        prepare_unload.add_effect(StartTiming(10), ready_to_receive(i), True)
        prepare_unload.add_effect(StartTiming(20), ready_to_receive(i), False)
        domain.add_action(prepare_unload)

        unload = DurativeAction("unload", r=Robot, k=Kit, i=IntType(0, self._max_kit-1))
        r = unload.parameter("r")
        k = unload.parameter("k")
        i = unload.parameter("i")
        unload.set_fixed_duration(5)
        unload.add_condition(StartTiming(), Not(robot_busy(r)))
        unload.add_effect(StartTiming(), robot_busy(r), True)
        unload.add_effect(EndTiming(), robot_busy(r), False)
        unload.add_condition(StartTiming(), robot_at(r, l0))
        unload.add_condition(StartTiming(), Equals(kit_cnt, i))
        unload.add_condition(ClosedTimeInterval(StartTiming(), EndTiming()), ready_to_receive(i))
        for j in range(self._max_kit_size):
            unload.add_condition(StartTiming(), Equals(components_on_robot(r, j), components_on_kit(k, j)))
            unload.add_effect(EndTiming(), components_on_robot(r, j), EMPTY)
        unload.add_effect(EndTiming(), robot_cnt(r), 0)
        unload.add_effect(EndTiming(), completed(i, k), True)
        unload.add_effect(EndTiming(), kit_cnt, Plus(i, 1))
        unload.add_effect(EndTiming(), battery(r), self._max_kit_size+1)
        domain.add_action(unload)

        return domain

    def domain(self) -> up.model.Problem:
        return self._domain

    def size(self) -> Optional[int]:
        res = 0
        for length in range(self._min_kit_size, self._max_kit_size + 1):
            for _ in itertools.product(range(self._n_components), repeat=length):
                res += (self._max_n_kit + 1 - self._min_n_kit)
        return res * (self._max_n_robots + 1 - self._min_n_robots)

    def _get_problem_items(self, n_robots, length, combination, n) -> Tuple[Dict["up.model.FNode", "up.model.FNode"], List["up.model.Object"], List["up.model.FNode"]]:
        robot_at = self._domain.fluent("robot_at")
        completed = self._domain.fluent("completed")
        components_on_kit = self._domain.fluent("components_on_kit")
        is_present = self._domain.fluent("is_present")
        l0 = self._domain.object("l0")
        k1 = self._domain.object("k1")
        EMPTY = self._domain.object("EMPTY")
        objects = [l0, k1, EMPTY]
        TRUE = self._domain.environment.expression_manager.TRUE()
        l_c = {}
        for i in range(1, self._n_components + 1):
            l = self._domain.object(f"l{i}")
            c = self._domain.object(f"c{i}")
            l_c[l] = c
            objects.extend([l,c])
        robots = [self._domain.object(f"r{i}") for i in range(n_robots)]
        objects.extend(robots)
        
        initial_values = {}
        c = AnyChecker(lambda x : x.is_object_exp() and x.object() not in objects)
        for k, v in self._domain.initial_values.items():
            if c.any(k):
                continue
            initial_values[k] = v
        for k,v in l_c.items():
            initial_values[is_present(v, k)] = TRUE
        for r in robots:
            initial_values[robot_at(r, l0)] = TRUE
        for i, c in enumerate(combination):
            initial_values[components_on_kit(k1, i)] = self._domain.environment.expression_manager.ObjectExp(c)
        for i in range(length, self._max_kit_size):
            initial_values[components_on_kit(k1, i)] = self._domain.environment.expression_manager.ObjectExp(EMPTY) 
        goals = [completed(i, k1) for i in range(n)]
        return initial_values, objects, goals
        
    def get_problems(self) -> Iterator[Tuple[Dict["up.model.FNode", "up.model.FNode"], List["up.model.Object"], List["up.model.FNode"]]]:
        components = [self._domain.object(f"c{i}") for i in range(1, self._n_components+1)]
        for n_robots in range(self._min_n_robots, self._max_n_robots+1):
            for length in range(self._min_kit_size, self._max_kit_size+1):
                for combination in itertools.product(components, repeat=length):
                    for n in range(self._min_n_kit, self._max_n_kit+1):
                        yield self._get_problem_items(n_robots, length, combination, n)

    def get_problem(self, n_robots, length, combination, n) -> Problem:
        problem = self.domain().clone()
        problem.all_objects.clear()
        init, objects, goals = self._get_problem_items(n_robots, length, combination, n)
        for o in objects:
            problem.add_object(o)
        for k, v in init.items():
            problem.set_initial_value(k, v)
        for g in goals:
            problem.add_goal(g)
        return problem

    def dump_problem_params(self, filename: str):
        with open(filename, "w") as f:
            components = [self._domain.object(f"c{i}") for i in range(1, self._n_components+1)]
            for n_robots in range(self._min_n_robots, self._max_n_robots+1):
                for length in range(self._min_kit_size, self._max_kit_size+1):
                    for combination in itertools.product(components, repeat=length):
                        for n in range(self._min_n_kit, self._max_n_kit+1):
                            comb_str = ";".join([str(c) for c in combination])
                            f.write(f"{n_robots},{length},({comb_str}),{n}\n")