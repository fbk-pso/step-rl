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


import unified_planning
from unified_planning.shortcuts import *
from unified_planning.test import unittest_TestCase, main
from unified_planning.test.examples import get_example_problems
from unified_planning.engines.compilers import BoundedTypesRemover
from unified_planning.engines import CompilationKind


class TestBoundedTypesRemover(unittest_TestCase):
    def setUp(self):
        unittest_TestCase.setUp(self)
        self.problems = get_example_problems()

    def test_counter(self):
        problem = self.problems["counter"].problem
        kind = problem.kind
        added_conditions = 6  # 3 integers, all with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))

        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )

    def test_counter_to_50(self):
        problem = self.problems["counter_to_50"].problem
        kind = problem.kind
        added_conditions = 2  # 1 integer with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))
        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )

    def test_robot(self):
        problem = self.problems["robot"].problem
        kind = problem.kind
        added_conditions = 2  # 1 integer with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))
        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )

    def test_robot_decrease(self):
        problem = self.problems["robot_decrease"].problem
        kind = problem.kind
        added_conditions = 2  # 1 real with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))
        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )

    def test_robot_locations_connected(self):
        problem = self.problems["robot_locations_connected"].problem
        kind = problem.kind
        added_conditions = 2  # 1 real with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))
        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )

    def test_robot_locations_visited(self):
        problem = self.problems["robot_locations_visited"].problem
        kind = problem.kind
        added_conditions = 2  # 1 real with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))
        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )

    def test_robot_real_constants(self):
        problem = self.problems["robot_real_constants"].problem
        kind = problem.kind
        added_conditions = 2  # 1 real with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))
        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )

    def test_robot_int_battery(self):
        problem = self.problems["robot_int_battery"].problem
        kind = problem.kind
        added_conditions = 2  # 1 integer with upper and lower bound
        with Compiler(
            problem_kind=kind,
            compilation_kind=CompilationKind.BOUNDED_TYPES_REMOVING,
        ) as cer:
            res = cer.compile(problem, CompilationKind.BOUNDED_TYPES_REMOVING)
        compiled_problem = res.problem
        compiled_kind = compiled_problem.kind
        self.assertTrue(kind.has_bounded_types())
        self.assertFalse(compiled_kind.has_bounded_types())
        self.assertEqual(len(problem.actions), len(compiled_problem.actions))
        for a, ca in zip(problem.actions, compiled_problem.actions):
            assert isinstance(a, InstantaneousAction)
            assert isinstance(ca, InstantaneousAction)
            self.assertEqual(
                len(a.preconditions) + added_conditions, len(ca.preconditions)
            )
            self.assertEqual(len(a.effects), len(ca.effects))
        self.assertEqual(
            len(problem.goals) + added_conditions, len(compiled_problem.goals)
        )
