#!/bin/bash
# 检查SLURM分区和节点的CPU资源

echo "=========================================="
echo "SLURM Resource Check"
echo "=========================================="

echo ""
echo "1. Partition Information:"
scontrol show partition general-gpu | grep -E "MaxNodes|MaxTime|MaxCPUs|DefCpuPerGPU|MaxCpusPerNode"

echo ""
echo "2. Available Nodes with A100:"
sinfo -p general-gpu -o "%N %c %m %G" | grep -i a100 | head -10

echo ""
echo "3. Node Details (first A100 node):"
FIRST_NODE=$(sinfo -p general-gpu -o "%N" | grep -i a100 | head -1 | awk '{print $1}' | cut -d'[' -f1)
if [ ! -z "$FIRST_NODE" ]; then
    scontrol show node $FIRST_NODE | grep -E "CPUAlloc|CPUTot|CPULoad|RealMemory"
fi

echo ""
echo "4. Current Job Status:"
squeue -p general-gpu -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %.4C %.8m %.8e" | head -20

echo ""
echo "5. Test Resource Request:"
echo "Testing if 8 CPUs are available..."
srun --partition=general-gpu --gres=gpu:1 --cpus-per-task=8 --constraint="a100" --time=1:00:00 --test-only echo "8 CPUs available" 2>&1 | head -5

echo ""
echo "=========================================="
