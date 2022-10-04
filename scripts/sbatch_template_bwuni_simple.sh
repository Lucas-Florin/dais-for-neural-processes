#!/bin/bash
#SBATCH -p %%partition%%
# #SBATCH -A %%account%%
#SBATCH -J %%job-name%%
#SBATCH --array 0-%%last_job_idx%%%%%num_parallel_jobs%%
# Please use the complete path details :
#SBATCH -D %%experiment_execution_dir%%
#SBATCH -o %%slurm_log%%/out_%A_%a.log
#SBATCH -e %%slurm_log%%/err_%A_%a.log
# Cluster Settings
#SBATCH -n %%ntasks%%         # Number of tasks
#SBATCH -c %%cpus-per-task%%  # Number of cores per task
#SBATCH -t %%time%%             # 1:00:00 Hours, minutes and seconds, or '#SBATCH -t 10' - only minutes
#SBATCH --mem=10G
# -------------------------------
# Load the required modules
# module load devel/python/3.8.1_gnu_9.2-pipenv
# module load devel/miniconda
# module load python/3
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export OMP_NUM_THREADS=1
# echo $PYTHONPATH
# export PYTHONPATH=/home/kit/anthropomatik/ak4831/code/stochastic_search:$PYTHONPATH
# Export Pythonpath
# echo %%pythonpath%%
# export PYTHONPATH=$HOME/code/stochastic_search:$PYTHONPATH
# export PYTHONPATH=/pfs/work7/workspace/scratch/vs2244-gradientMORE-0/test_exp_1/exp_1/code:$PYTHONPATH
# export PYTHONPATH=/pfs/work6/workspace/scratch/ak4831-optuna_holereach-0/cas_more_lr_25/code:$PYTHONPATH
# echo $PYTHONPATH
# Activate the virtualenv / conda environment
# source /home/kit/anthropomatik/ak4831/venvs/stoch_search/bin/activate
# conda activate test
# echo $PYTHONPATH
# cd into the working directory
# cd %%experiment_root%%
# srun hostname > hostfile.$SLURM_JOB_ID
# mpiexec -map-by core -bind-to core python3 -c "%%python_script%%" %%path_to_yaml_config%% -m -g 1 -l WARN -j $SLURM_ARRAY_TASK_ID %%exp_name%%
# mpiexec -map-by node -hostfile hostfile.$SLURM_JOB_ID --mca mpi_warn_on_fork 0 --display-allocation --display-map python -m mpi4py -c "%%python_script%%" %%path_to_yaml_config%%
#python3 -c "%%python_script%%" %%path_to_yaml_config%% -l INFO -L INFO -j $SLURM_ARRAY_TASK_ID %%exp_name%%
python3 %%python_script%% %%path_to_yaml_config%% -j $SLURM_ARRAY_TASK_ID %%cw_args%%
