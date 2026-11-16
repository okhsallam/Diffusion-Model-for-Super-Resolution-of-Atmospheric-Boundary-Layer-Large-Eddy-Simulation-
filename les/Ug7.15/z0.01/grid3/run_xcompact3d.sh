#!/bin/bash 
#SBATCH --ntasks-per-node=96 # Tasks to be run
#SBATCH --nodes=4  # Run the tasks on the same node
#SBATCH --time=25:00:00   # Required, estimate 5 minutes
#SBATCH --account=<allocation> # Required
#SBATCH --exclusive
#SBATCH --error=inf-%j.err
#SBATCH --output=inf-%j.out
#SBATCH --mem=96000

module purge
module load intel-oneapi-compilers
module load intel-oneapi-mpi
module load intel-oneapi-mkl  # Adding MKL module



mpirun -np 384  /path/to/project/xcompact3d/Incompact3d/build/bin/xcompact3d

rm -r statistics

for i in {1..100}; do
    # Delete all four file types for each number from the data folder
    rm -f data/pp-$i.bin
    rm -f data/ux-$i.bin
    rm -f data/uy-$i.bin
    rm -f data/uz-$i.bin
    rm -f turb-data/nut_smag-$i.bin
done

### For interactive session
### salloc --time=4:00:00    --account=<allocation> --nodes=1 --ntasks-per-node=16 --cpus-per-task 1  --mem=96000M
### dos2unix run_xcompact3d.sh
### du -sh folder_name
