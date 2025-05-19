#!/bin/bash
#SBATCH --job-name=inf22086_boxy_prepare  # Ersetze #USERNAME mit deinem LDAP-Namen
#SBATCH --gres=gpu:0                        # Keine GPUs anfordern für CPU-Task
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=boxy_prepare-%j.out
#SBATCH --error=boxy_prepare-%j.err

# Erstelle virtuelle Umgebung falls nicht vorhanden
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Aktiviere virtuelle Umgebung
source venv/bin/activate

# Installiere Requirements falls nötig
# pip install -r ../requirements.txt

# Führe das Skript aus
srun python3 -u boxy_prepare.py

# Deaktiviere virtuelle Umgebung
deactivate

echo "Boxy dataset preparation completed successfully!"