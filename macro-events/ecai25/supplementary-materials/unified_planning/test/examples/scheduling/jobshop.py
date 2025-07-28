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
from unified_planning.shortcuts import *
from unified_planning.model.scheduling import *
from typing import List

# Text representation of the FT-O6 instance (Fisher and Thompson)
FT06 = """nb_jobs nb_machines
6 6 0 0 0 0
Times
1 3 6 7 3 6
8 5 10 10 10 4
5 4 8 9 1 7
5 5 5 3 8 9
9 3 5 4 3 1
3 3 9 10 4 1
Machines
3 1 2 4 6 5
2 3 5 6 1 4
3 4 6 1 2 5
2 1 3 4 5 6
3 2 5 6 1 4
2 4 6 1 5 3
"""


def parse(instance: str, instance_name: str) -> SchedulingProblem:
    """Parses a job instance and return the corresponding JobShop with 3 operators instance."""
    lines = instance.splitlines()

    def ints(line: str) -> List[int]:
        return list(map(int, line.rstrip().split()))

    def int_matrix(lines) -> List[List[int]]:
        return list(map(ints, lines))

    header = lines.pop(0)
    sizes = ints(lines.pop(0))
    num_jobs = sizes[0]
    num_machines = sizes[1]

    first_times_line = 1
    last_times_line = first_times_line + num_jobs - 1
    times = int_matrix(lines[first_times_line : last_times_line + 1])
    # print("Times: ", times)

    first_machine_line = last_times_line + 2
    last_machine_line = first_machine_line + num_jobs - 1
    machines = int_matrix(lines[first_machine_line : last_machine_line + 1])
    # print("Machines: ", machines)

    problem = unified_planning.model.scheduling.SchedulingProblem(
        f"sched:jobshop-{instance_name}-operators"
    )
    machine_objects = [
        problem.add_resource(f"m{i}", capacity=1) for i in range(1, num_machines + 1)
    ]

    # use the jobshop with operators extension: each activity requires an operator
    # for its duration
    num_operators = 3
    operators = problem.add_resource("operators", capacity=num_operators)

    for j in range(num_jobs):
        prev_in_job: Optional[Activity] = None

        for t in range(num_machines):
            act = problem.add_activity(f"t_{j}_{t}", duration=times[j][t])
            machine = machine_objects[machines[j][t] - 1]
            act.uses(machine)
            act.uses(operators, amount=1)

            if prev_in_job is not None:
                problem.add_constraint(LE(prev_in_job.end, act.start))
            prev_in_job = act

    problem.add_quality_metric(unified_planning.model.metrics.MinimizeMakespan())
    return problem


if __name__ == "__main__":
    pb = parse(FT06, "ft06-operators")
    print(pb)
    print(pb.kind)
