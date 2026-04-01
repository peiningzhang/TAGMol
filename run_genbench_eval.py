import os
import glob
import subprocess
import json

gen_dir = "/shared/healthinfolab/phz24002/TAGMol/experiments/trained_176000/sdfs_grouped"
gb3d_dir = "/shared/healthinfolab/phz24002/genbench3d"
test_set_dir = "/shared/healthinfolab/phz24002/TAGMol/data/test_set"

output_dir = os.path.join(gen_dir, "genbench_results")
os.makedirs(output_dir, exist_ok=True)

sdf_files = glob.glob(os.path.join(gen_dir, "*_generated.sdf"))
print(f"Found {len(sdf_files)} grouped SDF files to evaluate.")

# Find all test set ligands to map basename -> dir
all_native_ligands = glob.glob(os.path.join(test_set_dir, "*", "*.sdf"))
ligand_map = {}
for np_path in all_native_ligands:
    lig_base = os.path.basename(np_path).replace('.sdf', '')
    pocket_dir_name = os.path.basename(os.path.dirname(np_path))
    # In TAGMol's metrics_-1.pt, LIGAND_NAME is the folder_name + "_" + ligand_name
    combined_name = f"{pocket_dir_name}_{lig_base}"
    ligand_map[combined_name] = np_path

failed = []
success = 0

for sdf in sdf_files:
    basename = os.path.basename(sdf).replace("_generated.sdf", "")
    
    native_ligand = ligand_map.get(basename)
    if not native_ligand:
        print(f"Warning: native ligand not found in test_set for {basename}")
        failed.append(basename)
        continue
        
    pocket_dir = os.path.dirname(native_ligand)
    
    # Find protein. In CrossDocked, protein usually ends with '_rec.pdb'
    proteins = [f for f in os.listdir(pocket_dir) if f.endswith('_rec.pdb')]
    if not proteins:
        print(f"Warning: protein not found in {pocket_dir}")
        failed.append(basename)
        continue
        
    protein_pdb = os.path.join(pocket_dir, proteins[0])
    out_json = os.path.join(output_dir, f"results_{basename}.json")
    
    # Construct GenBench3D command
    cmd = [
        "python",
        os.path.join(gb3d_dir, "sb_benchmark_mols.py"),
        "-c", os.path.join(gb3d_dir, "config/default.yaml"),
        "-i", sdf,
        "-p", protein_pdb,
        "-n", native_ligand,
        "-o", out_json,
        "-s", "ligboundconf"
    ]
    
    print(f"Evaluating {basename}...")
    try:
        subprocess.run(cmd, cwd=gb3d_dir, check=True)
        success += 1
    except subprocess.CalledProcessError as e:
        print(f"Failed GenBench3D on {basename}, error: {e}")
        failed.append(basename)

print(f"===========================================================")
print(f"Evaluation Completed! Success: {success}, Failed: {len(failed)}")
print(f"Results saved in: {output_dir}")

# Quick snippet to aggregate JSON results later
results_merged = []
for jf in glob.glob(os.path.join(output_dir, "*.json")):
    try:
        with open(jf, 'r') as f:
            res = json.load(f)
            # You can inject the pocket name if you need to, using filename
            res['pocket'] = os.path.basename(jf)
            results_merged.append(res)
    except:
        pass
if results_merged:
    with open(os.path.join(output_dir, "all_results_aggregated.json"), 'w') as outf:
        json.dump(results_merged, outf)
    print("Aggregated JSON created: all_results_aggregated.json")
