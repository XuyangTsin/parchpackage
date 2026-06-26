#!/bin/sh
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH -J evp1
#SBATCH --partition=normal,longjobs  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --exclude=node1181,node[1141-1143]

module load gromacs
module unload anaconda3/2023.9

#-------------------------------EM-------------------------------------------------------

gmx grompp -f em.mdp -c W_init.gro -p W_topol.top -o em.tpr -maxwarn 5 -r W_init.gro -n W_ind.ndx
mpirun gmx_mpi mdrun -pin on -deffnm em

#-------------------------------annealing process-------------------------------------

gmx grompp -f heat_nvt_1.mdp -c em.gro -p W_topol.top -o w_h.tpr -maxwarn 5 -r em.gro -n W_ind.ndx
mpirun gmx_mpi mdrun -pin on -deffnm w_h -cpi w_h -dlb yes


rm -rf \#*     # delete hashpack files with #


