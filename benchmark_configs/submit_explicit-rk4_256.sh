
#!/bin/bash 
### Job Name 
#PBS -N dendrtic_cuda_explicit-rk4_256 
### required runtime 
#PBS -l walltime=00:30:00 
### queue for submission 
#PBS -q gpuA
 
### Merge output and error files 
#PBS -j oe 
 
#PBS -l select=1:mem=10G:ncpus=1:ngpus=1 
 
### start job in the directory it was submitted from 
cd $PBS_O_WORKDIR 
 
### load the necessary software modules 
module load cuda/12.4.1 
module load gcc/13.3
 
### run the application 
./build/main.out benchmark_configs/config_explicit-rk4_256.ini
