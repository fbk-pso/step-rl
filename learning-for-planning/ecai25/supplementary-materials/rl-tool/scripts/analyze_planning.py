import tarfile
import re
import os
import csv
import argparse

KEYS = ["solved", "overall_time", "expanded_states", "plan_size"]

# Function to extract tarballs and read files
def extract_tarballs_and_read(directory):
    instances = {}
    for filename in os.listdir(directory):
        if filename.endswith(".tar.bz2"):
            filepath = os.path.join(directory, filename)
            with tarfile.open(filepath, "r:bz2") as tar:
                tar.extractall(path=directory)
                for member in tar.getmembers():
                    if member.isfile() and (member.name.endswith(".stats") or member.name.endswith(".stderr") or member.name.endswith(".stdout")):
                        instance_name = os.path.splitext(member.name)[0]
                        if not instance_name.endswith(".anml") :
                            continue
                        if instance_name not in instances:
                            group = instance_name.split(os.sep)[0]
                            set_part = re.search(r'(set_[^_]+)', group)
                            config_part = re.search(r'(config_[^_]+)', group)
                            run_part = re.search(r'(run_[^_]+)', group)
                            set = set_part.group(1) if set_part else None
                            config = config_part.group(1) if config_part else None
                            run = run_part.group(1) if run_part else None
                            instance_info = group.split("_")
                            if len(instance_info) == 2:
                                set = instance_info
                            instances[instance_name] = {"instance": instance_name.split(os.sep)[-1], "group": group, "set": set, "config": config, "run": run, "memory-out": False, "timeout": False, "solved": False}
                        with open(os.path.join(directory, member.name), "r") as file:
                            content = file.read()

                        # Parsing key-value pairs and adding them to instances
                        lines = content.splitlines()
                        for line in lines:
                            if ": " in line:
                                key, value = line.split(": ", 1)
                                if key in KEYS:
                                    instances[instance_name][key] = value
                            if "out of memory" in line:
                                instances[instance_name]["memory-out"] = True
                            if "out of time" in line:
                                instances[instance_name]["timeout"] = True
                            if "wall-clock time " in line:
                                instances[instance_name]["wc-time"] = line[16:]
                            if "retval " in line:
                                instances[instance_name]["retval"] = line[7:]
                            if "maximum resident set size " in line:
                                instances[instance_name]["memory"] = line[26:]

    return instances

# Function to write instances to a CSV file
def write_instances_to_csv(instances, output_csv):
    # Get the header from the first instance (unique keys from all instances)
    headers = set()
    for instance in instances.values():
        headers.update(instance.keys())
    headers = list(headers)

    with open(output_csv, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(headers))
        writer.writeheader()
        for instance in instances.values():
            writer.writerow(instance)

# Main function
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-i", type=str, required=True)
    parser.add_argument("-d", type=str, default='planning_res')
    args = parser.parse_args()

    instances = extract_tarballs_and_read(os.path.join(args.i, args.d))
    write_instances_to_csv(instances, os.path.join(args.i, f"{args.d}.csv"))

if __name__ == "__main__":
    main()
