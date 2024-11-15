#!/bin/bash

####### specify cluster configuration
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --account=cse446

####### select resources (here we specify required memory)
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1

####### make sure no other jobs are assigned to your nodes
#SBATCH --exclusive

####### further customizations
### name of your job
#SBATCH --job-name="CSE_446_Final Project"
#SBATCH --mail-user=henrylee@buffalo.edu
#SBATCH --mail-type=all

### files to store stdout and stderr (%j will be replaced by the job id)
#SBATCH --output=%j.stdout
#SBATCH --error=%j.stderr

### how many nodes to allocate for the job
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=8

### max time the job will run
#SBATCH --time=12:00:00

# here go regular commands, these commands will be run
# on the first node allocated to the job
echo "Hostname:"
hostname
echo "g++ version:"
g++ --version
echo "OS:"
cat /etc/os-release
echo "Memory:"
free -g
echo "HW info:"
lscpu

# Load venv and python
source ./setup.sh

