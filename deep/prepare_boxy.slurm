#!/bin/bash
#SBATCH --job-name=DEIN_USERNAME_boxy_prepare
#SBATCH --partition=cpu  # CPU-Partition für I/O-intensive Tasks
#SBATCH --cpus-per-task=4  # Moderate CPU-Anzahl für Downloads/Extraktion
#SBATCH --mem=8G  # Ausreichend Memory für JSON-Verarbeitung
#SBATCH --time=02:00:00  # 2 Stunden sollten ausreichen
#SBATCH --output=slurm-boxy-prepare-%j.out
#SBATCH --error=slurm-boxy-prepare-%j.err

# Wechsle in das richtige Verzeichnis
cd ~/neural-navi/deep/

# Python-Umgebung laden (falls nötig)
# module load python/3.10

# Skript ausführen
srun python3 boxy_prepare.py

echo "Boxy dataset preparation completed!"