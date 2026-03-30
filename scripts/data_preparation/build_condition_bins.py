import argparse
import os

import yaml

import utils.misc as misc
from datasets import get_dataset
from utils.condition_bins import build_condition_bin_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Dataset config yaml, e.g. configs/training_dock_guide.yml")
    parser.add_argument("--output", type=str, default=None, help="Where to write the condition bin spec")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of equal-frequency bins")
    args = parser.parse_args()

    config = misc.load_config(args.config)
    dataset, subsets = get_dataset(config=config.data, transform=None, index_path=getattr(config.data, "index_path", None))

    train_indices = subsets["train"].indices
    test_indices = subsets["test"].indices
    spec = build_condition_bin_spec(dataset, train_indices=train_indices, test_indices=test_indices, num_bins=args.num_bins)

    output = args.output
    if output is None:
        base_dir = os.path.dirname(args.config)
        output = os.path.join(base_dir, "condition_bins.yml")

    with open(output, "w") as f:
        yaml.safe_dump(spec, f, sort_keys=False)

    print(f"Wrote condition bin spec to: {output}")
    for prop, entry in spec["properties"].items():
        print(f"{prop}: field={entry['field']}, edges={entry['edges']}")


if __name__ == "__main__":
    main()

