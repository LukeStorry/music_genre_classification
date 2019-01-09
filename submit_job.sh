#!/bin/bash
#SBATCH -t 0-3:00 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=15000
#SBATCH --account=comsm0018       # use the course account
#SBATCH -J music_classification_cwk  # name
#SBATCH -o hostname_%j.out # File to which STDOUT will be written
#SBATCH -e hostname_%j.err # File to which STDERR will be written
#SBATCH --mail-type=ALL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sd14814@bristol.ac.uk # Email to which notifications will be sent

module add languages/anaconda2/5.0.1.tensorflow-1.6.0

srun python main.py --epochs 100 --depth shallow
srun python main.py --epochs 200 --depth shallow
#srun python main.py --epochs 100 --depth shallow --augment True
#srun python main.py --epochs 200 --depth shallow --augment True

srun python main.py --epochs 100 --depth deep
srun python main.py --epochs 200 --depth deep
#srun python main.py --epochs 100 --depth deep --augment True
#srun python main.py --epochs 200 --depth deep --augment True


wait
