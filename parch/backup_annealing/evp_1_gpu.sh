#!/bin/sh
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH -J evp_1
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=gpu_zone2


module load gromacs gnu12 pmix
module unload anaconda3/2023.9

#-------------------------------EM-------------------------------------------------------

gmx grompp -f em.mdp -c W_init.gro -p W_topol.top -o em.tpr -maxwarn 5 -r W_init.gro -n W_ind.ndx
mpirun gmx_mpi mdrun -pin on -deffnm em

#-------------------------------annealing process-------------------------------------

gmx grompp -f heat_nvt_1.mdp -c em.gro -p W_topol.top -o w_h.tpr -maxwarn 5 -r em.gro -n W_ind.ndx
mpirun -np 4  gmx_mpi mdrun -pin on -deffnm w_h -cpi w_h -dlb yes -pme gpu -nb gpu -npme 1


rm -rf \#*     # delete hashpack files with #
