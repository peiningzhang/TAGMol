import argparse
import glob
import json
import os
import subprocess

_DEFAULT_GENBENCH_PY = os.environ.get(
    "GENBENCH_PYTHON",
    "/home/phz24002/anaconda3/envs/genbench3d/bin/python",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run GenBench3D sb_benchmark_mols.py on *_generated.sdf under gen_dir."
    )
    parser.add_argument(
        "--gen_dir",
        default="/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped",
        help="Directory containing *_generated.sdf",
    )
    parser.add_argument(
        "--gb3d_dir",
        default="/shared/healthinfolab/phz24002/genbench3d",
        help="GenBench3D repo root (sb_benchmark_mols.py, config/default.yaml)",
    )
    parser.add_argument(
        "--test_set_dir",
        default="/shared/healthinfolab/phz24002/TAGMol/data/test_set",
        help="Native ligand tree (pocket subdirs with reference .sdf)",
    )
    parser.add_argument(
        "--genbench_python",
        default=_DEFAULT_GENBENCH_PY,
        help=(
            "Python that has GenBench3D deps (MDAnalysis, etc.). "
            "Do not use tagmol env here unless MDAnalysis is installed there. "
            "Override with env GENBENCH_PYTHON."
        ),
    )
    parser.add_argument(
        "--do_conf_analysis",
        action="store_true",
        help="Forward --do_conf_analysis to sb_benchmark_mols.py",
    )
    parser.add_argument(
        "--no_vina",
        action="store_true",
        help="Forward --no_vina to sb_benchmark_mols.py",
    )
    args = parser.parse_args()

    gen_dir = os.path.abspath(args.gen_dir)
    gb3d_dir = os.path.abspath(args.gb3d_dir)
    test_set_dir = os.path.abspath(args.test_set_dir)

    if args.genbench_python.startswith("/") and not os.path.isfile(args.genbench_python):
        print(
            "Warning: --genbench_python path not found: %s\n"
            "Use the conda env where GenBench3D is installed (e.g. .../envs/genbench3d/bin/python)."
            % args.genbench_python
        )

    output_dir = os.path.join(gen_dir, "genbench_results")
    os.makedirs(output_dir, exist_ok=True)

    sdf_files = glob.glob(os.path.join(gen_dir, "*_generated.sdf"))
    print("Using GenBench Python: %s" % args.genbench_python)
    print("Found %d grouped SDF files to evaluate." % len(sdf_files))

    all_native_ligands = glob.glob(os.path.join(test_set_dir, "*", "*.sdf"))
    ligand_map = {}
    for np_path in all_native_ligands:
        lig_base = os.path.basename(np_path).replace(".sdf", "")
        pocket_dir_name = os.path.basename(os.path.dirname(np_path))
        combined_name = "%s_%s" % (pocket_dir_name, lig_base)
        ligand_map[combined_name] = np_path

    failed = []
    success = 0

    for sdf in sdf_files:
        basename = os.path.basename(sdf).replace("_generated.sdf", "")

        native_ligand = ligand_map.get(basename)
        if not native_ligand:
            print("Warning: native ligand not found in test_set for %s" % basename)
            failed.append(basename)
            continue

        pocket_dir = os.path.dirname(native_ligand)

        proteins = [f for f in os.listdir(pocket_dir) if f.endswith("_rec.pdb")]
        if not proteins:
            print("Warning: protein not found in %s" % pocket_dir)
            failed.append(basename)
            continue

        protein_pdb = os.path.abspath(os.path.join(pocket_dir, proteins[0]))
        sdf_abs = os.path.abspath(sdf)
        native_abs = os.path.abspath(native_ligand)
        out_json = os.path.abspath(os.path.join(output_dir, "results_%s.json" % basename))

        cmd = [
            args.genbench_python,
            os.path.join(gb3d_dir, "sb_benchmark_mols.py"),
            "-c",
            os.path.join(gb3d_dir, "config/default.yaml"),
            "-i",
            sdf_abs,
            "-p",
            protein_pdb,
            "-n",
            native_abs,
            "-o",
            out_json,
            "-s",
            "ligboundconf",
        ]
        if args.do_conf_analysis:
            cmd.append("--do_conf_analysis")
        if args.no_vina:
            cmd.append("--no_vina")

        print("Evaluating %s..." % basename)
        try:
            subprocess.run(cmd, cwd=gb3d_dir, check=True)
            success += 1
        except subprocess.CalledProcessError as e:
            print("Failed GenBench3D on %s, error: %s" % (basename, e))
            failed.append(basename)

    print("===========================================================")
    print("Evaluation Completed! Success: %d, Failed: %d" % (success, len(failed)))
    print("Results saved in: %s" % output_dir)

    results_merged = []
    for jf in sorted(glob.glob(os.path.join(output_dir, "results_*.json"))):
        try:
            with open(jf, "r") as f:
                res = json.load(f)
            res["pocket"] = os.path.basename(jf)
            results_merged.append(res)
        except (json.JSONDecodeError, OSError):
            pass
    if results_merged:
        agg_path = os.path.join(output_dir, "all_results_aggregated.json")
        with open(agg_path, "w") as outf:
            json.dump(results_merged, outf)
        print("Aggregated JSON created: all_results_aggregated.json (%d pockets)" % len(results_merged))


if __name__ == "__main__":
    main()
