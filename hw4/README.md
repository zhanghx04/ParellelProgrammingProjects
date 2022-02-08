# HPC 2021 - Homework - 4 OpenMPI

## homework4 using SLURM to run on **MIZZOU Lewis Server**

## Steps:

1. cd {path}/hw4
2. mkdir build && cd build
3. module load cmake gcc opencv openmpi
4. cmake ..
5. make
6. cp ../hw4.sh .
7. sbatch ./hw4.sh
8. cat hpc-hw4-%j.out

> To change the parallel level (number of nodes)
>
> edit build/hw4.sh
>
> find    #SBATCH --ntasks=3
>
> change the number of nodes

## NOTE: the master node will not do parallel work. if ntasks=7, there is only 6 nodes are doing parallel work