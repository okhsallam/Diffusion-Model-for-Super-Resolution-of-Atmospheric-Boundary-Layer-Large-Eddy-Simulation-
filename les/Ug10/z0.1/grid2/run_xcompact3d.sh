#!/bin/bash 
#SBATCH --ntasks-per-node=64 # Tasks to be run
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --time=03:00:00   # Required, estimate 5 minutes
#SBATCH --account=<allocation> # Required
#SBATCH --exclusive
#SBATCH --error=inf-%j.err
#SBATCH --output=inf-%j.out
#SBATCH --mem=96000

module purge
module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-oneapi-mkl  # Adding MKL module



mpirun -np 64  /path/to/project/xcompact3d/Incompact3d/build/bin/xcompact3d

rm data/snapshot*

### For interactive session
### salloc --time=4:00:00    --account=<allocation> --nodes=1 --ntasks-per-node=16 --cpus-per-task 1  --mem=96000M
### dos2unix run_xcompact3d.sh