import argparse
import json
import os
import re

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description="Plot Loss vs kimg from multiple stats.jsonl runs."
)
parser.add_argument(
    "paths",
    nargs="+",
    help="Paths to dirs containing stats.jsonl or parent dirs with date-time subfolders.",
)

date_time_pattern = re.compile(r"^\d{8}_\d{6}$")


def find_latest_date_time_dir(directory):
    """Return the latest directory in 'directory' that matches the date_time_pattern,
    or None if none exist or no stats.jsonl is found."""
    if not os.path.isdir(directory):
        return None
    subdirs = [
        d
        for d in os.listdir(directory)
        if date_time_pattern.match(d) and os.path.isdir(os.path.join(directory, d))
    ]
    if not subdirs:
        return None
    subdirs.sort(reverse=True)  # latest first
    for subdir in subdirs:
        candidate = os.path.join(directory, subdir)
        if os.path.isfile(os.path.join(candidate, "stats.jsonl")):
            return candidate
    return None


def main(args):

    fig, ax = plt.subplots(figsize=(5, 3))

    for input_path in args.paths:
        input_path = os.path.normpath(input_path)  # Clean up the path
        base = os.path.basename(input_path)
        parent = os.path.basename(os.path.dirname(input_path))

        if date_time_pattern.match(base):
            stats_file = os.path.join(input_path, "stats.jsonl")
            if not os.path.isfile(stats_file):
                print(f"Warning: No stats.jsonl in {input_path}, skipping.")
                continue
            target_dir = input_path
            label = parent  # e.g. "standard", "edm", "cfm"
        else:
            latest_dir = find_latest_date_time_dir(input_path)
            if latest_dir is None:
                print(
                    f"Warning: No valid date_time subdirectory with stats.jsonl in {input_path}, skipping."
                )
                continue
            target_dir = latest_dir
            label = os.path.basename(input_path)

        stats_path = os.path.join(target_dir, "stats.jsonl")
        if not os.path.isfile(stats_path):
            print(f"Warning: No stats.jsonl in {target_dir}, skipping.")
            continue

        kimgs, losses = [], []
        with open(stats_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                kimgs.append(data["Progress/kimg"]["mean"])
                losses.append(data["Loss/loss"]["mean"])

        ax.plot(kimgs[::5], losses[::5], lw=1, label=label)

    # y axis on log scale
    ax.set_yscale("log")

    ax.set_xlabel("kimg")
    ax.set_ylabel("loss")
    ax.legend()

    fig.tight_layout()
    fig.savefig("media/loss.png", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    # python loss.py ../results/standard/ ../results/edm/ ../results/cfm/
    args = parser.parse_args()
    main(args)
