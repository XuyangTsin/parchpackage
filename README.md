# parch

Command-line tools for the PARCH water-shell workflow: extract a solute from an
equilibrated system, build a hydration shell for annealing, submit the annealing
runs, analyse the per-residue dehydration, and compute averaged PARCH values.

The full pipeline is five commands:

```
parch prep  ->  parch shellsetup  ->  parch submit  ->  parch analysis  ->  parch calpv
```

## Installation

### Prerequisites

| Requirement | Needed for | Notes |
| ----------- | ---------- | ----- |
| Python ≥ 3.8 with `pip` | everything | a conda env is recommended |
| `numpy`, `MDAnalysis ≥ 2.0` | everything | installed automatically by `pip` |
| **GROMACS** (`gmx`) | `shellsetup`, and running the annealing | external; not installable via pip. Put it on `PATH` (e.g. `source /path/to/gromacs/bin/GMXRC`) |
| **SLURM** (`sbatch`) | `submit` only | external; only needed to launch annealing jobs on a cluster |

`prep`, `analysis` and `calpv` are pure Python/MDAnalysis and need neither
GROMACS nor SLURM.

### Install from GitHub

Install straight from the repository into the Python environment where you want
the `parch` commands to live:

```bash
python -m pip install git+https://github.com/<your-username>/parch.git
```

or from a local clone:

```bash
git clone https://github.com/<your-username>/parch.git
cd parch
python -m pip install .          # add -e for an editable/dev install
```

The bundled data (`parch/backup_annealing/` — the mdp/itp/INP/evp templates) is
shipped as package data, so a normal install includes everything the tools need
at runtime — no editable install required. (Force fields are **not** bundled;
use your own for the upstream equilibration / topology.) This puts the umbrella
command `parch` plus per-tool entry points on the environment's `PATH`:

| Umbrella            | Direct command   | Purpose                                              |
| ------------------- | ---------------- | --------------------------------------------------- |
| `parch prep`        | `parch_prep`     | extract selected molecules from an equilibrated system |
| `parch shellsetup`  | `parch_shell`    | build a hydration shell (+ counterions) and set up the box |
| `parch submit`      | `parch_submit`   | stage simulation files into `mid_*` and submit jobs |
| `parch analysis`    | `parch_analysis` | per-residue water-shell hydration over the ramp     |
| `parch calpv`       | `parch_calpv`    | PARCH values, averaged over `mid_*` runs            |

Verify the install:

```bash
parch -h
parch shellsetup -h
```

All commands can be run from any directory; paths are resolved against where you
launch them.

### Cluster configuration (important for `parch submit`)

`parch submit` copies SLURM job scripts (`evp_<i>_zest3.sh` for cpu,
`evp_<i>_gpu.sh` for gpu) into each run directory and submits them. **These
scripts are written for the original author's cluster** — partitions
(`--partition=normal,longjobs`), node excludes, and `module load gromacs` /
`mpirun gmx_mpi`. Before using `parch submit` on your own cluster, **edit the
`#SBATCH` headers and the `module`/`mpirun` lines** in the installed
`parch/backup_annealing/evp_*.sh` templates (run
`python -c "import parch, os; print(os.path.join(os.path.dirname(parch.__file__), 'backup_annealing'))"`
to find them) to match your scheduler, partitions, and GROMACS build. (You can
also skip `submit` entirely and launch the `shell/mid_*` runs with your own
scripts; each just needs `gmx grompp`/`mdrun` on `W_init.gro`, `W_topol.top`,
`W_ind.ndx` with `em.mdp` then `heat_nvt_<i>.mdp`.)

### Layout

`parch/` is a proper Python package; its modules
(`prep`, `shellsetup`, `submit`, `analysis`, `calpv`, `cli`) and the
`backup_annealing/` data directory are installed together, so the tools find
their templates relative to the installed package on any machine.

---

## 1. `parch prep` — extract the solute

This tool is useful for preparing the structure (`.gro`) and topology (`.top`) files 
before setting up the shell. It is most useful in interactive mode, but a non-interactive 
mode has also been implemented to support automated workflows and scripting.

From an equilibrated system (`md.gro` + `md.tpr` + `.top`), list every molecule
type / chain, choose which to keep, and write a trimmed structure + topology
ready for `shellsetup`. The molecule→atom mapping comes from the `.tpr`; the
coordinates come from the `.gro`.

**IMPORTANT:** Before running the `setup` command, ensure that **all heavy atoms** of the 
molecules you want to run with PARCH have position restraints defined in their 
corresponding `.itp` files. 

The position restraints should be enclosed by the `POSRES` preprocessor directive, for example:

```
#ifdef POSRES
[ position_restraints ]
    1     1    POSRES_FC_BB    POSRES_FC_BB    POSRES_FC_BB   
    5     1    POSRES_FC_BB    POSRES_FC_BB    POSRES_FC_BB   
    7     1    POSRES_FC_SC    POSRES_FC_SC    POSRES_FC_SC   
   10     1    POSRES_FC_SC    POSRES_FC_SC    POSRES_FC_SC   
   11     1    POSRES_FC_SC    POSRES_FC_SC    POSRES_FC_SC   
   12     1    POSRES_FC_SC    POSRES_FC_SC    POSRES_FC_SC   
   13     1    POSRES_FC_BB    POSRES_FC_BB    POSRES_FC_BB   
   14     1    POSRES_FC_SC    POSRES_FC_SC    POSRES_FC_SC   
   15     1    POSRES_FC_SC    POSRES_FC_SC    POSRES_FC_SC
   .......
   .......  
#endif
```

In default, these position restraints are activated in the mdp options for later annealing process as:

```
define                  = -DPOSRES -DPOSRES_FC_BB=10000 -DPOSRES_FC_SC=10000
```

`POSRES_FC_BB` and `POSRES_FC_SC` correspond to the restraint force constants 
(in kJ mol<sup>−1</sup> nm<sup>−2</sup>) applied to backbone and side-chain heavy atoms, respectively. 

```bash
parch prep -f md.gro -s md.tpr -pi topol.top -o solute.gro -po solute.top
```
**!!Please Read the NOTICE printed when excuting the command for guidance!!**

First, it prints a numbered table of molecules and asks which to KEEP (keyboard, e.g. 1 2 4 5-10),

```
#group  molecule        number   atoms
1       PROA            1        1418
2       HEME            1        73
3       SOD             88       88
4       CLA             77       77
5       TIP3            27337    82011
```

**IMPORTANT**: After selecting, it modifies the `.top` file, and reports the renumbered 
residue ranges (Consecutive, and starting from 1) in the output (useful for if you want to use `-shelldef` later):

```
Output residue ranges (renumbered) -- useful for -separateshell / -shelldef:
    PROA           1:90
    HEME           91:91
```

Then, it prints `#include` (force-field) lines, and asks which to keep (keyboard).
DO NOT KEEP the `.itp` files for IONS used during equilibration, as PARCH uses its own tailored ones.
Keeping the original ion `.itp` files may result in conflicting topology entries.
If you plan to use a DIFFERENT water model than the one used during equilibration, DO NOT keep the water's `.itp` file.
```
The following itp files are in your .top:
1     toppar/forcefield.itp
2     toppar/PROA.itp
3     toppar/HEME.itp
4     toppar/SOD.itp
5     toppar/CLA.itp
6     /home/xqin10/parch_platform/backup_eqb/charmm36-jul2021.ff/tip3p.itp
```

After selection, the kept `.itp` files will be overwritten with their full path 
in the `.top`, to avoid additional copying and moving the files:
```
; Include forcefield parameters
#include "/home/xqin10/parch_platform/test_shellsetup/uniform/toppar/forcefield.itp"
#include "/home/xqin10/parch_platform/test_shellsetup/uniform/toppar/PROA.itp"
#include "/home/xqin10/parch_platform/backup_eqb/charmm36-jul2021.ff/tip3p.itp"
```
Additionally, the tailored `.itp` of ions are automatically included.
```
; PARCH counter-ion topologies
#include "/home/xqin10/parch_platform/parch_package/backup_annealing/CLA.itp"
#include "/home/xqin10/parch_platform/parch_package/backup_annealing/SOD.itp"
```

**IMPORTANT**: The net charge of the retained molecules will be reported at the end. This 
value should be provided to the `shellsetup` to properly place counterions.

| Option       | Meaning                                                                 |
| ------------ | ----------------------------------------------------------------------- |
| `-f`         | input equilibrated structure (`md.gro`)                                 |
| `-s`         | run input file (`md.tpr`) — supplies the molecule typing                |
| `-pi`        | input topology (`.top`)                                                 |
| `-o`         | output structure (`.gro`) with kept molecules                          |
| `-po`        | output topology (`.top`) with kept molecules / includes                |
| `-renumber`  | `yes` (default) renumbers kept residues from 1 (shellsetup-ready);     |
|                 `no` keeps original ids, the structure will not work well for shell setup |
| `-keep`      | non-interactive selection, e.g. `"1 2 4"` or `"1-3,5"`                     |
| `-keepff`    | non-interactive `#include` selection (default with `-keep`: keep all)      |

---

## 2. `parch shellsetup` — build the hydration shell and place the counter ions

Reads a prepared solute structure + topology (e.g. from `prep`) and
builds a hydrated shell with optional neutralising counterions.

**Uniform shell** — one thickness applied to every non-solvent residue:

```bash
parch shellsetup -f solute.gro -p solute.top -o W_init.gro -separateshell no -dshell 4.15 -netcharge 0
```

**Multiple shell regions** — any number of `(thickness  residue-range)` groups
from a shell-definition file:

```bash
parch shellsetup \
    -f solute.gro -p solute.top -o W_init.gro \
    -separateshell yes -shelldef shelldef_setup.txt -netcharge 0
```

`shelldef_setup.txt` (one group per line; `#` comments and blank lines ignored):

```text
# shell_thickness(Å)  residue_range
4.15  1:195
4.50  196:200
4.60  201:220
```

**Shell thickness for setup:**
For protein: 4.15 Å.
For DNA and RNA: 4.80 Å


Validation (separateshell): the file must exist, ranges must be valid, must not
overlap, every referenced residue ID must exist in the structure, and all
thicknesses must be positive — otherwise the run aborts with a clear message.

Inputs are read from the current directory; all generated files are written to a
`shell/` subdirectory (the input directory stays clean). The key outputs are the
hydrated structure (`-o`), `W_topol.top`, and `W_ind.ndx`, which are also copied
into independent run directories `shell/mid_1` … `shell/mid_N`.

| Option            | Meaning                                                            |
| ----------------- | ------------------------------------------------------------------ |
| `-f` / `-p` / `-o`| input structure / input topology / output structure name          |
| `-dshell`         | uniform shell thickness (Å) when `-separateshell no` (default 4.15)|
| `-separateshell`  | `no` (uniform `-dshell`) or `yes` (per-group from `-shelldef`)     |
| `-shelldef`       | shell-definition file (required with `-separateshell yes`)         |
| `-netcharge`      | system net charge; sets the number of counterions                 |
| `-nmids`          | number of `mid_*` run dirs to create (integer ≥ 3; default 5)      |
| `-water`          | water model `tip3p`/`tip4p`/`tip5p` (selects the solvate box)      |
| `-dsi`/`-dii`/`-db` | ion–solute / ion–ion / solute–box distances (nm)                |

Auxiliary files (hydrated-ion `.gro`, INP make_ndx templates) come from
`-datadir` (default: `../backup_annealing`).

---

## 3. `parch submit` — stage inputs and launch the annealing

For each `mid_*` run, copies in the simulation inputs (energy-min + annealing
`.mdp`, and the SLURM job script for the chosen partition) and — by
default — submits the job with `sbatch`.

```bash
# cpu partition, all mid_*, submit immediately
parch submit -path shell -partition cpu

# gpu partition, first 3 runs, stage only (don't submit)
parch submit -path shell -nmids 3 -partition gpu -launch no
```

| Option        | Meaning                                                              |
| ------------- | ------------------------------------------------------------------- |
| `-path`       | the `shell/` directory holding the `mid_*` runs                      |
| `-nmids`      | submit `mid_1..mid_N`; if omitted, all `mid_*` under `-path`         |
| `-launch`     | `yes` (default) submit with `sbatch`; `no` stage only               |
| `-partition`  | node partition for the jobs, `cpu` (default) → `evp_<i>_zest3.sh`; `gpu` → `evp_<i>_gpu.sh`       |

Each `mid_i` uses its own annealing protocol (`heat_nvt_<i>.mdp`) and job script
(`evp_<i>_zest3.sh` / `evp_<i>_gpu.sh`). **Edit the `#SBATCH` lines in those
`mid_*/evp_<i>_*.sh` scripts to change your job requests** (partition, nodes,
walltime, …). Runs that are already finished (`w_h.gro` present) or not yet set
up (`W_init.gro` missing) are skipped. The protocol/job templates ship for
indices 1–5, so this supports up to 5 `mid_*` runs.

Each run produces `em.tpr/.gro` and `w_h.tpr/.xtc`, which the analysis step reads.

---

## 4. `parch analysis` — hydration over the annealing ramp

For each `mid_*` run, counts how many solvent molecules sit within a cutoff of
each solute residue, in the energy-minimised structure and across the heating
trajectory, and writes b-factor-coloured PDBs at 11 temperature points. Outputs
go to `mid_*/analysis/`.

```bash
# uniform cutoff
parch analysis -path shell -da 3.15 -separateshell no

# per-residue cutoffs from a shell-definition file
parch analysis -path shell -separateshell yes -shelldef shelldef_analysis.txt
```

| Option           | Meaning                                                             |
| ---------------- | ------------------------------------------------------------------- |
| `-path`          | the `shell/` directory holding the `mid_*` runs                     |
| `-da`            | uniform cutoff (Å) when `-separateshell no` (default 3.15)          |
| `-separateshell` | `no` (uniform `-da`) or `yes` (per-residue from `-shelldef`)        |
| `-shelldef`      | shell-definition file (required with `-separateshell yes`)          |
| `-newwater`      | extra solvent residue name(s) beyond `SOL OPC TP3 T4D T4E`          |
| `-blocks`        | block-mapping file to split residues into sub-units (e.g. DNA sugar/base) |
| `-overwrite`     | re-run even if a `mid_*/analysis/` already looks complete           |


**`-shelldef shelldef_analysis.txt`** the format is consistent with the one used for setup. However,
the shell cutoff for analysis can be **different than setup**.

**Shell cutoff for setup:**
For protein: 3.15 Å.
For DNA and RNA: 4.80 Å

**`-blocks blocks.txt`** file — split chosen residues into named atom sub-units (one or
more lines per residue range; whitespace-separated, no commas):

```text
# resid_range  block_name  atom1 atom2 ...
1:24   PSU   O5' H5T C5' H5'1 ... P O1P O2P
1:24   NBP   N9 C4 N2 N3 ... N6 H61 H62 H41 H42
```

Each named residue then yields one result per block (`5_PSU`, `5_NBP`, …);
residues not named fall back to whole-residue analysis.

---

## 5. `parch calpv` — PARCH values, averaged over runs

Reads the `num_ww_temp_<tag>.txt` from each `mid_*/analysis/`, computes the PARCH
value per analysis unit (time-correlation integral of the dehydration series
scaled by `hard_ref`), and averages over the runs.

```bash
parch calpv -path shell -water_ff charmm_tip3p -nmids 5 -mean_without_min_max yes
```

| Option                  | Meaning                                                       |
| ----------------------- | ------------------------------------------------------------- |
| `-path`                 | the `shell/` directory with analysed `mid_*` runs            |
| `-water_ff`             | selects `hard_ref` (default `charmm&tip3p`)                   |
| `-newhardref`           | user `hard_ref` value, overrides `-water_ff`                  |
| `-nmids`                | number of runs to include (integer ≥ 3; default 5)           |
| `-mean_without_min_max` | `yes` (default) trims lowest+highest run per unit; auto-disabled when < 5 runs |
| `-blocks`               | block-mapping file (only needed to colour the output PDBs per sub-unit) |

Supported `-water_ff` values: `charmm_tip3p` (828602.27), `charmm_tip4p`
(834204.55), `charmm_tip4pew` (793704.55), `charmm_tip5p` (886392.05),
`amber_opc` (548924.2424), `amber_tip3p` (813643.94).

Outputs under `shell/` (the two averaged flavours are mutually exclusive):

| `-mean_without_min_max` | files written                                                     |
| ----------------------- | ---------------------------------------------------------------- |
| `no`                    | `ave_pv_<tag>.txt`, `std_pv_<tag>.txt`, `ave_correlation_pv.pdb`  |
| `yes`                   | `ave_excl_pv_<tag>.txt`, `std_excl_pv_<tag>.txt`, `ave_excl_correlation_pv.pdb` |
| (always)                | `parch_summary_<tag>.txt` + per-run `mid_*/analysis/pv_<tag>.txt`, `correlation_pv.pdb` |

`parch_summary_<tag>.txt` is a per-unit table: `unit  name  mid_1 … mid_N  ave
std` — the residue id, the residue name (`GLN`, or `DG_PSU` with blocks), the
PARCH value from each run, and the across-run average and std.

---

## Notes

- Run any command with `-h` for the full option list.
- `prep`/`analysis`/`calpv` need only MDAnalysis; `shellsetup` also needs `gmx`;
  `submit` needs SLURM (`sbatch`).
- The `-shelldef` format is identical across `shellsetup`, `analysis` and (for
  `prep`) the residue ranges it prints, so the same file/ranges carry through.
