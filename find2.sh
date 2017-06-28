#!/bin/sh
#PBS -q batch
#PBS -N test2
#PBS -l mem=64gb
#PBS -l vmem=64gb
#PBS -l walltime=144:00:00
#PBS -l nodes=1 :ppn=42:intel                                                   
#PBS -M sc.sanabria1984@uniandes.edu.co
#PBS -m abe
module load openmpi/1.8.5
module load rocks-openmpi_ib
module load libs/gsl/1.15
module load hpcstats
cd $PBS_O_WORKDIR
NPROCS=`wc -l < $PBS_NODEFILE`
RUNSTATS 1
rm /hpcfs/home/sc.sanabria1984/momografia/rockstar/Rockstar-0.99.9-RC3/HALOS_50/auto-rockstar.cfg
echo 1 > pasos.txt 
mpiexec -n 1 ./rockstar -c ./parallel2.cfg >& server.dat &    
echo 2 > pasos.txt
sleep 10
echo 3 > pasos.txt
mpiexec -n 42 ./rockstar -c /hpcfs/home/sc.sanabria1984/momografia/rockstar/Rockstar-0.99.9-RC3/HALOS_50/auto-rockstar.cfg >& server2.dat
echo 4 > pasos.txt

