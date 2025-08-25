import itertools
from examples import ProblemGenerator
from typing import Iterator, Optional, Tuple, List, Dict, Union
import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.model.walkers import AnyChecker

class MaJSPGenerator(ProblemGenerator):
    def __init__(self, n_robots: Union[int,Tuple[int,int]], n_pallets: Union[int,Tuple[int,int]], n_treatments: Union[int,Tuple[int,int]], n_positions: int) -> None:
        super().__init__()
        self._min_n_robots = n_robots[0] if isinstance(n_robots, tuple) else 1
        self._max_n_robots = n_robots[1] if isinstance(n_robots, tuple) else n_robots
        self._min_n_pallets = n_pallets[0] if isinstance(n_pallets, tuple) else 1
        self._max_n_pallets = n_pallets[1] if isinstance(n_pallets, tuple) else n_pallets
        self._min_n_treatments = n_treatments[0] if isinstance(n_treatments, tuple) else 1
        self._max_n_treatments = n_treatments[1] if isinstance(n_treatments, tuple) else n_treatments
        self._n_positions = n_positions
        self._domain = self._build_domain()

    def _build_domain(self):
        domain = Problem("MaJSP")

        #Setting up Types
        Robot = UserType('Robot')
        Pallet = UserType('Pallet')
        Position = UserType('Position')

        UNKNOWN = Object("UNKNOWN", Position)
        DEPOT = Object("DEPOT", Position)
        NOPALLET = Object("NOPALLET", Pallet)
        domain.add_objects([UNKNOWN, DEPOT, NOPALLET])

        #Setting up Fluents
        robot_at = Fluent('robot_at', Position, r=Robot)
        domain.add_fluent(robot_at, default_initial_value=DEPOT)

        robot_has = Fluent('robot_has', Pallet, r=Robot)
        domain.add_fluent(robot_has, default_initial_value=NOPALLET)

        position_has = Fluent('position_has', Pallet, p=Position)
        domain.add_fluent(position_has, default_initial_value=NOPALLET)

        at_depot = Fluent("at_depot", BoolType(), b=Pallet)
        domain.add_fluent(at_depot, default_initial_value=True)

        treated = Fluent('treated', BoolType(), b=Pallet, p=Position)
        domain.add_fluent(treated, default_initial_value=False)

        ready = Fluent('ready', BoolType(), b=Pallet, p=Position)
        domain.add_fluent(ready, default_initial_value=False)

        battery_level = Fluent('battery_level', IntType(0, 100), r=Robot)
        domain.add_fluent(battery_level, default_initial_value=100)

        #Setting up Actions:
        move = DurativeAction('move', r=Robot, to=Position)
        r = move.parameter('r')
        to = move.parameter('to')
        move.set_fixed_duration(1)
        move.add_condition(StartTiming(), Not(Equals(to, UNKNOWN)))
        move.add_condition(StartTiming(), Not(Equals(robot_at(r), to)))
        move.add_condition(StartTiming(), Not(Equals(robot_at(r), UNKNOWN)))
        move.add_condition(StartTiming(), GE(battery_level(r), 1))
        move.add_decrease_effect(StartTiming(), battery_level(r), 1)
        move.add_effect(StartTiming(), robot_at(r), UNKNOWN)
        move.add_effect(EndTiming(), robot_at(r), to)

        unload_at_depot = InstantaneousAction('unload_at_depot', r=Robot)
        r = unload_at_depot.parameter('r')
        unload_at_depot.add_precondition(Not(Equals(robot_has(r), NOPALLET)))
        unload_at_depot.add_precondition(Equals(robot_at(r), DEPOT))
        unload_at_depot.add_effect(position_has(DEPOT), robot_has(r))
        unload_at_depot.add_effect(robot_has(r), NOPALLET)

        load_at_depot = InstantaneousAction('load_at_depot', r=Robot, p=Pallet)
        r = load_at_depot.parameter('r')
        p = load_at_depot.parameter('p')
        load_at_depot.add_precondition(Not(Equals(p, NOPALLET)))
        load_at_depot.add_precondition(Equals(robot_has(r), NOPALLET))
        load_at_depot.add_precondition(at_depot(p))
        load_at_depot.add_precondition(Equals(robot_at(r), DEPOT))
        load_at_depot.add_effect(robot_has(r), p)
        load_at_depot.add_effect(at_depot(p), False)

        make_treat = DurativeAction('make_treatment', r=Robot, b=Pallet, p=Position)
        r = make_treat.parameter('r')
        b = make_treat.parameter('b')
        p = make_treat.parameter('p')
        make_treat.set_fixed_duration(20)
        make_treat.add_condition(StartTiming(), Not(Equals(p, UNKNOWN)))
        make_treat.add_condition(StartTiming(), Not(Equals(b, NOPALLET)))
        make_treat.add_condition(StartTiming(), Not(Equals(p, DEPOT)))
        make_treat.add_condition(StartTiming(), Equals(position_has(p), NOPALLET))
        make_treat.add_condition(StartTiming(), Equals(robot_at(r), p))
        make_treat.add_condition(StartTiming(), Equals(robot_has(r), b))
        make_treat.add_condition(StartTiming(), Not(treated(b, p)))
        make_treat.add_effect(StartTiming(), position_has(p), b)
        make_treat.add_effect(StartTiming(), robot_has(r), NOPALLET)
        make_treat.add_effect(StartTiming(10), ready(b, p), True)
        make_treat.add_condition(EndTiming(), treated(b, p))
        make_treat.add_condition(EndTiming(), Equals(position_has(p), NOPALLET))

        load = DurativeAction('load', r=Robot, b=Pallet, p=Position)
        r = load.parameter('r')
        b = load.parameter('b')
        p = load.parameter('p')
        load.set_fixed_duration(1)
        load.add_condition(StartTiming(), Not(Equals(p, UNKNOWN)))
        load.add_condition(StartTiming(), Not(Equals(b, NOPALLET)))
        load.add_condition(StartTiming(), Equals(position_has(p), b))
        load.add_condition(StartTiming(), Equals(robot_has(r), NOPALLET))
        load.add_condition(StartTiming(), ready(b, p))
        load.add_condition(StartTiming(), Equals(robot_at(r), p))
        load.add_effect(StartTiming(), ready(b, p), False)
        load.add_effect(StartTiming(), position_has(p), NOPALLET)
        load.add_effect(EndTiming(), robot_has(r), b)
        load.add_effect(EndTiming(), treated(b, p), True)

        domain.add_actions([move, load_at_depot, unload_at_depot, make_treat, load])

        for i in range(self._max_n_robots):
            r = Object(f"r{i}", Robot)
            domain.add_object(r)

        for i in range( self._max_n_pallets):
            b = Object(f"b{i}", Pallet)
            domain.add_object(b)

        for i in range(self._n_positions):
            p = Object(f"p{i}", Position)
            domain.add_object(p)

        return domain

    def domain(self) -> up.model.Problem:
        return self._domain

    def size(self) -> Optional[int]:
        size = 0
        for r in range(self._min_n_robots, self._max_n_robots+1):
            for b in range(self._min_n_pallets, self._max_n_pallets+1):
                for i in range(self._min_n_treatments, self._max_n_treatments+1):
                    size += len(list(itertools.combinations(range(self._n_positions), i)))
        return size

    def _get_problem_items(self, b, i, ps, rob) -> Tuple[Dict["up.model.FNode", "up.model.FNode"], List["up.model.Object"], List["up.model.FNode"]]:
        goals = []
        objects = [self._domain.object(f"r{r}") for r in range(rob)]
        objects.extend([self._domain.object("UNKNOWN"), self._domain.object("DEPOT"), self._domain.object("NOPALLET")])
        for p in ps:
            po = self._domain.object(f"p{p}")
            objects.append(po)
        for pallets in range(b):
            bo = self._domain.object(f"b{pallets}")
            objects.append(bo)
        for p in ps:
            po = self._domain.object(f"p{p}")
            for pallets in range(b):
                bo = self._domain.object(f"b{pallets}")
                goals.append(self._domain.fluent("treated")(bo, po))
        initial_values = {}
        c = AnyChecker(lambda x : x.is_object_exp() and x.object() not in objects)
        for k, v in self._domain.initial_values.items():
            if c.any(k):
                continue
            initial_values[k] = v
        return initial_values, objects, goals

    def get_problems(self) -> Iterator[Tuple[Dict["up.model.FNode", "up.model.FNode"], List["up.model.Object"], List["up.model.FNode"]]]:
        for b in range(self._min_n_pallets, self._max_n_pallets+1):
            for i in range(self._min_n_treatments, self._max_n_treatments+1):
                for ps in itertools.combinations(range(self._n_positions), i):
                    for rob in range(self._min_n_robots, self._max_n_robots+1):
                        yield self._get_problem_items(b, i, ps, rob)

    def get_problem(self, b, i, ps, rob) -> Problem:
        problem = self.domain().clone()
        problem.all_objects.clear()
        init, objects, goals = self._get_problem_items(b, i, ps, rob)
        for o in objects:
            problem.add_object(o)
        for k, v in init.items():
            problem.set_initial_value(k, v)
        for g in goals:
            problem.add_goal(g)
        return problem

    def dump_problem_params(self, filename: str):
        with open(filename, "w") as f:
            for b in range(self._min_n_pallets, self._max_n_pallets+1):
                for i in range(self._min_n_treatments, self._max_n_treatments+1):
                    for ps in itertools.combinations(range(self._n_positions), i):
                        for rob in range(self._min_n_robots, self._max_n_robots+1):
                            ps_str = ";".join([str(p) for p in ps])
                            f.write(f"{b},{i},({ps_str}),{rob}\n")