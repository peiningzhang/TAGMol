import itertools
import subprocess
import os
import csv
import yaml
import tempfile
import copy

# Cord and Categorical guidance scale ranges to test
cords = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.1, -1.0, -5.0]
categs = [0.0, -0.1, -1.0, -5.0, -10.0, -20.0, -50.0, -100.0, -200.0]

output_csv = "BA_fixed_QED_SA_search.csv"

# The base sample config provided by the user
base_sample_config = "configs/noise_guide_multi/sampling_guided_qed_0.33_sa_0.33_ba_0.34_veda.yml"
with open(base_sample_config, 'r') as f:
    base_cfg = yaml.safe_load(f)

combinations = list(itertools.product(cords, categs))
print(f"Total combinations to evaluate: {len(combinations)}")

# Fixed BA parameters as requested
BA_FIXED_CORD = 10.0
BA_FIXED_CATEG = -100.0

# We will write/append to CSV
file_exists = os.path.isfile(output_csv)

with open(output_csv, mode='a', newline='') as f:
    writer = csv.writer(f)
    headers_written = file_exists

    for idx, (cord, categ) in enumerate(combinations):
        print(f"[{idx+1}/{len(combinations)}] Running with QED/SA_cord={cord}, QED/SA_categ={categ} (Fix BA=10/-100)")
        
        # Prepare a temporary sample config with these scales
        current_cfg = copy.deepcopy(base_cfg)
        for g in current_cfg.get('guide_models', []):
            if g['name'] in ['qed', 'sa']:
                g['gradient_scale_cord'] = cord
                g['gradient_scale_categ'] = categ
            elif g['name'] == 'binding_affinity':
                g['gradient_scale_cord'] = BA_FIXED_CORD
                g['gradient_scale_categ'] = BA_FIXED_CATEG
            
        fd, tmp_cfg_path = tempfile.mkstemp(suffix='.yml', dir='.')
        try:
            with os.fdopen(fd, 'w') as tmp_f:
                yaml.dump(current_cfg, tmp_f)
            cmd = [
                "python", "scripts/quick_evaluate.py",
                "--config", "configs/training.yml",
                "--checkpoint", "logs_diffusion/training_2026_03_17__15_30_20/checkpoints/last.pt",
                "--sample_config", tmp_cfg_path,
                "--device", "cuda:0",
                "--num_proteins", "10",
                "--num_ligands_per_protein", "10",
                "--docking_mode", "vina_score",
                "--num_steps", "100"
            ]
            
            # Run command with real-time logging
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            complete_output = []
            metrics_head = None
            metrics_val = None
            
            for line in process.stdout:
                complete_output.append(line)
                
                # Show sampling progress to user
                if "Sampling for protein index" in line:
                    print(f"    {line.strip()}")
                elif "Running evaluation script on the generated samples" in line:
                    print("    Starting results evaluation...")
                
                # Parse stdout for one-line metrics
                if "METRICS_ONE_LINE_HEAD" in line:
                    idx_head = line.find("METRICS_ONE_LINE_HEAD")
                    parts = line[idx_head + len("METRICS_ONE_LINE_HEAD"):].split('\t')
                    metrics_head = [p.strip() for p in parts if p.strip()]
                elif "METRICS_ONE_LINE_VAL" in line:
                    idx_val = line.find("METRICS_ONE_LINE_VAL")
                    parts = line[idx_val + len("METRICS_ONE_LINE_VAL"):].split('\t')
                    metrics_val = [p.strip() for p in parts if p.strip()]

            process.wait()
            
            if metrics_head and metrics_val:
                if not headers_written:
                    full_headers = ["qed_sa_scale_cord", "qed_sa_scale_categ"] + metrics_head
                    writer.writerow(full_headers)
                    headers_written = True
                    f.flush()
                    
                full_values = [cord, categ] + metrics_val
                writer.writerow(full_values)
                f.flush()
                # Find mean score index for display
                try:
                    score_idx = metrics_head.index('Vina_score_mean')
                    print(f"  -> Success! Vina_score_mean: {metrics_val[score_idx]}")
                except (ValueError, IndexError):
                    print(f"  -> Success!")
            else:
                print(f"  -> Failed to parse metrics.")
                print("  --- STDOUT (last 10 lines) ---")
                print(''.join(complete_output[-10:]))
                
        finally:
            if os.path.exists(tmp_cfg_path):
                os.remove(tmp_cfg_path)
            
print(f"All evaluations finished. Results saved to {output_csv}")
