#!/bin/sh
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH -J evp_3
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --exclusive


#-------------------------------EM-------------------------------------------------------

gmx grompp -f em.mdp -c W_init.gro -p W_topol.top -o em.tpr -maxwarn 5 -r W_init.gro -n W_ind.ndx
srun mdrun_mpi -pin on -deffnm em

#-------------------------------annealing process-------------------------------------

gmx grompp -f heat_nvt_3.mdp -c em.gro -p W_topol.top -o w_h.tpr -maxwarn 5 -r em.gro -n W_ind.ndx
srun mdrun_mpi -pin on -deffnm w_h -cpi w_h -dlb yes -v


rm -rf \#*     # delete hashpack files with #
