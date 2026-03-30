import argparse
import os
import pickle
from collections import Counter

import lmdb


def load_index(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            return None


def infer_processed_lmdb(index_path):
    base_dir = os.path.dirname(index_path)
    if "dock_guide" in index_path:
        return os.path.join(base_dir, os.path.basename(base_dir) + "_processed_dock_guide_final.lmdb")
    return os.path.join(base_dir, os.path.basename(base_dir) + "_processed_final.lmdb")


def load_lmdb_records(lmdb_path, limit=None):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    records = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (_, raw) in enumerate(cursor):
            records.append(pickle.loads(raw))
            if limit is not None and i + 1 >= limit:
                break
    env.close()
    return records


def row_key(row):
    if isinstance(row, dict):
        protein_fn = row.get("protein_filename")
        ligand_fn = row.get("ligand_filename")
        if protein_fn is not None and ligand_fn is not None:
            return (protein_fn, ligand_fn)
        return tuple(sorted(row.keys()))
    if len(row) >= 2:
        return (row[0], row[1])
    return tuple(row)


def describe_index(name, index):
    lengths = Counter(len(row) for row in index)
    print(f"[{name}] rows: {len(index)}")
    print(f"[{name}] tuple lengths: {dict(sorted(lengths.items()))}")
    for i, row in enumerate(index[:3]):
        print(f"[{name}] sample[{i}]: {row}")

    prop_rows = 0
    prop_keys = Counter()
    for row in index:
        if len(row) >= 1 and isinstance(row[-1], dict):
            prop_rows += 1
            prop_keys.update(row[-1].keys())
    print(f"[{name}] rows with trailing prop dict: {prop_rows}")
    if prop_keys:
        print(f"[{name}] prop keys: {dict(sorted(prop_keys.items()))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_index", type=str, default=None)
    parser.add_argument("--guide_index", type=str, default=None)
    parser.add_argument("--base_lmdb", type=str, default=None)
    parser.add_argument("--guide_lmdb", type=str, default=None)
    args = parser.parse_args()

    base_index = load_index(args.base_index) if args.base_index else None
    guide_index = load_index(args.guide_index) if args.guide_index else None

    if base_index is None:
        base_lmdb = args.base_lmdb
        if base_lmdb is None:
            if args.base_index is None:
                raise ValueError("Provide either --base_index or --base_lmdb")
            base_lmdb = infer_processed_lmdb(args.base_index)
        print(f"[base] index is not pickle; falling back to LMDB: {base_lmdb}")
        base_index = load_lmdb_records(base_lmdb)

    if guide_index is None:
        guide_lmdb = args.guide_lmdb
        if guide_lmdb is None:
            if args.guide_index is None:
                raise ValueError("Provide either --guide_index or --guide_lmdb")
            guide_lmdb = infer_processed_lmdb(args.guide_index)
        print(f"[guide] index is not pickle; falling back to LMDB: {guide_lmdb}")
        guide_index = load_lmdb_records(guide_lmdb)

    base_keys = [row_key(row) for row in base_index]
    guide_keys = [row_key(row) for row in guide_index]

    describe_index("base", base_index)
    print("---")
    describe_index("guide", guide_index)

    base_set = set(base_keys)
    guide_set = set(guide_keys)

    print("---")
    print(f"shared pair keys: {len(base_set & guide_set)}")
    print(f"base-only pair keys: {len(base_set - guide_set)}")
    print(f"guide-only pair keys: {len(guide_set - base_set)}")
    print(f"exact same order: {base_keys == guide_keys}")

    if base_index and guide_index:
        print(f"base first key: {base_keys[0] if base_keys else 'n/a'}")
        print(f"guide first key: {guide_keys[0] if guide_keys else 'n/a'}")


if __name__ == "__main__":
    main()
