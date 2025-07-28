#!/usr/bin/env python3

import argparse
import importlib
import subprocess
import tempfile

import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.io import ANMLReader, ANMLWriter, PDDLReader, PDDLWriter

ENGINE_PARAMETERS_VARIABLE_NAME = "ENGINE_PARAMETERS"
PDDL_DOMAIN_ENVIRON_VAR_NAME = "PDDL_DOMAIN_PATH"
PDDL_PROBLEM_ENVIRON_VAR_NAME = "PDDL_PROBLEM_PATH"

ANML_PROBLEMS_ENVIRON_VAR_NAME = "ANML_PROBLEMS_PATH" # a sequence of paths separated by a an empty space

set_credits_stream(None)


def convert_value(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def get_python_problem(args):
    module = importlib.import_module(args.problem_package)
    problem_parameters = map(convert_value, args.problem_params.split(","))
    problem = getattr(module, args.problem_function)(*problem_parameters)
    return problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner', type=str, required=True)
    parser.add_argument('--executable', type=str, default=None)
    parser.add_argument('--supported-kind', type=str)
    parser.add_argument('--pddl-domain', type=str)
    parser.add_argument('--pddl-problem', type=str)
    parser.add_argument('--anml-files', nargs='+')
    parser.add_argument('--problem-package', type=str)
    parser.add_argument('--problem-function', type=str)
    parser.add_argument('--problem-params', type=str)

    args, unknown_args = parser.parse_known_args()

    if args.executable is None:
        if args.pddl_domain:
            reader = PDDLReader()
            problem = reader.parse_problem(args.pddl_domain, args.pddl_problem)
        elif args.anml_files:
            reader = ANMLReader()
            problem = reader.parse_problem(args.anml_files)
        else:
            problem = get_python_problem(args)

        params = {}
        for i in range(0, len(unknown_args), 2):
            key = unknown_args[i].lstrip('-')
            value = convert_value(unknown_args[i + 1])
            params[key] = value

        with OneshotPlanner(name=args.planner, params=params) as planner:
            res = planner.solve(problem)

        print("status:", res.status)
        print("plan-found:", res.plan is not None)
        print("plan:", repr(str(res.plan)))

        if res.metrics:
            for k, v in res.metrics.items():
                print(f"{k}: {v}")

    else:
        with tempfile.TemporaryDirectory() as tempdir:
            supported_kind = args.supported_kind
            domain_problem_paths = []
            env = {ENGINE_PARAMETERS_VARIABLE_NAME: " ".join(unknown_args)}

            if supported_kind == "pddl":
                if args.pddl_domain:
                    domain_problem_paths.extend((args.pddl_domain, args.pddl_problem))
                else:
                    if args.anml_files:
                        reader = ANMLReader()
                        problem = reader.parse_problem(args.anml_files)
                    else:
                        problem = get_python_problem(args)
                    writer = PDDLWriter(problem)
                    domain_file = tempfile.NamedTemporaryFile(dir=tempdir, delete=False, suffix=".pddl")
                    problem_file = tempfile.NamedTemporaryFile(dir=tempdir, delete=False, suffix=".pddl")
                    domain_problem_paths.extend((domain_file.name, problem_file.name))
                    writer.write_domain(domain_file.name)
                    writer.write_problem(problem_file.name)
                env[PDDL_DOMAIN_ENVIRON_VAR_NAME] = domain_problem_paths[0]
                env[PDDL_PROBLEM_ENVIRON_VAR_NAME] = domain_problem_paths[1]
            elif supported_kind == "anml":
                if args.anml_files:
                    domain_problem_paths.extend(args.anml_files)
                else:
                    if args.pddl_domain:
                        reader = PDDLReader()
                        problem = reader.parse_problem(args.pddl_domain, args.pddl_problem)
                    else:
                        problem = get_python_problem(args)
                    writer = ANMLWriter(problem)
                    problem_file = tempfile.NamedTemporaryFile(dir=tempdir, delete=False, suffix=".anml")
                    domain_problem_paths.append(problem_file.name)
                    writer.write_problem(problem_file.name)
                env[ANML_PROBLEMS_ENVIRON_VAR_NAME] = " ".join(domain_problem_paths)
            else:
                raise NotImplementedError(f"The support to the kind {supported_kind} is not implemented yet.")

            subprocess.run(
                [args.executable],
                text=True,
                env=env,
            )


if __name__ == '__main__':
    main()
