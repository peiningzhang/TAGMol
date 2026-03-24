import itertools
import subprocess
import os
import csv

cords = [0.0, 0.1]
categs = [-100.0, -200.0]

output_csv = "BA_guide_test_w1.csv"

combinations = list(itertools.product(cords, categs))
print(f"Total combinations to evaluate: {len(combinations)}")

# We will write/append to CSV
file_exists = os.path.isfile(output_csv)

with open(output_csv, mode='a', newline='') as f:
    writer = csv.writer(f)
    
    # We don't know the exact headers until the first run, but we will store 
    # guide_scale_cord, guide_scale_categ, and then all HEADERS from stdout.
    headers_written = file_exists

    for idx, (cord, categ) in enumerate(combinations):
        print(f"[{idx+1}/{len(combinations)}] Running evaluation with guide_scale_cord={cord}, guide_scale_categ={categ}")
        
        cmd = [
            "python", "scripts/quick_evaluate.py",
            "--config", "logs_diffusion/training_2026_03_17__15_30_20/sampling.yml",
            "--checkpoint", "logs_diffusion/training_2026_03_17__15_30_20/checkpoints/last.pt",
            "--num_proteins", "10",
            "--num_ligands_per_protein", "10",
            "--docking_mode", "vina_score",
            "--guide_checkpoint", "logs/training_dock_guide_veda_2026_03_20__10_30_47/checkpoints/last.pt",
            "--guide_scale_cord", str(cord),
            "--guide_scale_categ", str(categ),
            "--num_steps", "100"
        ]
        
        # Run command
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Parse stdout for one-line metrics
        metrics_head = None
        metrics_val = None
        
        for line in result.stdout.split('\n'):
            if "METRICS_ONE_LINE_HEAD" in line:
                # Find the location of METRICS_ONE_LINE_HEAD and take everything after it
                idx_head = line.find("METRICS_ONE_LINE_HEAD")
                parts = line[idx_head + len("METRICS_ONE_LINE_HEAD"):].split('\t')
                metrics_head = [p.strip() for p in parts if p.strip()]
            elif "METRICS_ONE_LINE_VAL" in line:
                idx_val = line.find("METRICS_ONE_LINE_VAL")
                parts = line[idx_val + len("METRICS_ONE_LINE_VAL"):].split('\t')
                metrics_val = [p.strip() for p in parts if p.strip()]

        
        if metrics_head and metrics_val:
            metrics_head = [h.strip() for h in metrics_head if h.strip()]
            metrics_val = [v.strip() for v in metrics_val if v.strip()]
            
            if not headers_written:
                full_headers = ["guide_scale_cord", "guide_scale_categ"] + metrics_head
                writer.writerow(full_headers)
                headers_written = True
                f.flush()
                
            full_values = [cord, categ] + metrics_val
            writer.writerow(full_values)
            f.flush()
            print(f"  -> Success! Vina_score_mean: {metrics_val[metrics_head.index('Vina_score_mean')] if 'Vina_score_mean' in metrics_head else 'N/A'}")
        else:
            print(f"  -> Failed to parse metrics.")
            print("  --- STDOUT (last 20 lines) ---")
            print('\n'.join(result.stdout.split('\n')[-20:]))
            
print("All evaluations finished.")
