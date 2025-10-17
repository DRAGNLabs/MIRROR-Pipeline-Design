#!/bin/bash

# run with `sbatch --nodes=# --ntasks-per-node=# --gpus-per-node=# --export=ALL,HIDDEN_SIZE=#,BATCH_SIZE=# launch_experiment.sh` replacing each # with a number. `ntasks-per-gpu` and `gpus-per-node` should be the same.

#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --signal=SIGHUP@90

scratch_dir=$(pwd | sed -E "s/\/home\/(\w+)/\/home\/\1\/nobackup\/autodelete/")

srun python launch_experiment.py --trainer.devices=$SLURM_NTASKS_PER_NODE --trainer.num_nodes=$SLURM_NNODES --trainer.default_root_dir="./experiments/experiment-$SLURM_NNODES-$SLURM_NTASKS_PER_NODE-$BATCH_SIZE-$HIDDEN_SIZE" --model.hidden_size=$HIDDEN_SIZE --data.batch_size=$BATCH_SIZE
