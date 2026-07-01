#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parch shellsetup -- build a hydration shell (+ counterions) for PARCH annealing.

Standalone, argument-driven refactor of ``2_TIP3p_batch_pp_refill_annealing.py``.

Given a pre-equilibrated solute structure, this tool:
  1. strips existing ions/water from the input topology,
  2. refills a cubic box with the chosen water model and re-numbers residues,
  3. carves a water shell of a given thickness around explicitly-specified
     solute residues (optionally a different thickness per residue group),
  4. optionally places neutralising counterions on a sphere around the solute,
  5. writes the hydrated structure (``-o``) and an updated topology
     (``W_topol.top``) holding the correct number of shell waters / ions.

The input structure must already have sequentially-numbered residue IDs.

  -separateshell no  : apply a single ``-dshell`` thickness to every non-solvent
                       residue in the input structure.
  -separateshell yes : read an arbitrary number of (thickness, residue-range)
                       groups from the ``-shelldef`` file.

Usage (also callable as ``parch shellsetup ...`` / ``parch_shell ...``):

    # uniform shell
    parch shellsetup -f input.gro -p topol.top -o W_init.gro \
        -separateshell no -dshell 4.15 -netcharge 0

    # multiple shell regions (one 'thickness_A start:end' per line in the file)
    parch shellsetup -f input.gro -p topol.top -o W_init.gro \
        -separateshell yes -shelldef separateshell.txt -netcharge 0
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.distances as Mdistance


# --------------------------------------------------------------------------- #
# Water-model -> auxiliary file table.
#
#   solvent : the pre-equilibrated solvent box handed to `gmx solvate`
#             (spc216.gro / tip4p.gro / tip5p.gro ship with GROMACS).
#   na / cl : a hydrated Na+ / Cl- cluster (ion + waters) used for counterions.
#
# `na` and `cl` are copied from --datadir.
# --------------------------------------------------------------------------- #
WATER_MODELS = {
    "tip3p": {"solvent": "spc216.gro", "na": "hydrated_na.gro", "cl": "hydrated_cl.gro"},
    "tip4p": {"solvent": "tip4p.gro",  "na": "tip4p_na.gro",    "cl": "tip4p_cl.gro"},
    "opc": {"solvent": "tip4p.gro",  "na": "tip4p_na.gro",    "cl": "tip4p_cl.gro"},
    "tip5p": {"solvent": "tip5p.gro",  "na": "tip5p_na.gro",    "cl": "tip5p_cl.gro"},
}

# Solvent / ion molecule names whose [ molecules ] count lines are replaced by
# parch_shell. Every OTHER line of the user's topology is preserved verbatim.
SOLVENT_ION_NAMES = ["SOL", "WAT", "HOH",
                     "TIP3", "TIP3P", "TIP4", "TIP4P", "TIP5", "TIP5P",
                     "NA", "CL", "SOD", "CLA", "NA+", "CL-"]

DEFAULT_DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_annealing")


# =========================================================================== #
# Small helpers
# =========================================================================== #
def die(msg):
    sys.exit("ERROR: " + msg)


def run(cmd, allow_fail=False):
    """Run a shell command, echoing it; abort on non-zero exit unless allowed."""
    print(">> " + cmd, flush=True)
    rc = subprocess.run(cmd, shell=True).returncode
    if rc != 0 and not allow_fail:
        die("command failed (exit %d):\n    %s" % (rc, cmd))
    return rc


def parse_range(text):
    """Parse 'a:b' (or a single 'a') into an inclusive (start, end) int tuple."""
    text = text.strip()
    try:
        if ":" in text:
            a, b = text.split(":")
            a, b = int(a), int(b)
        else:
            a = b = int(text)
    except ValueError:
        die("could not parse residue range %r (expected 'start:end', e.g. 1:159)" % text)
    if a > b:
        die("residue range %r has start > end" % text)
    return (a, b)


def range_ids(rng):
    return set(range(rng[0], rng[1] + 1))


def _min3_int(text):
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if v < 3:
        raise argparse.ArgumentTypeError("must be >= 3")
    return v


# =========================================================================== #
# Argument parsing
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(
        prog="parch_shell",
        description="Build a hydration shell (+ counterions) for PARCH annealing.",
        allow_abbrev=False,
    )
    p.add_argument("-f", dest="structure", required=True,
                   help="Input structure file (.gro or .pdb) with sequential residue IDs.")
    p.add_argument("-p", dest="topology", required=True,
                   help="Input topology file (.top). Updated copy saved as W_topol.top.")
    p.add_argument("-o", dest="output", default="W_init.gro",
                   help="Output hydrated structure file name (default: W_init.gro).")

    p.add_argument("-dshell", type=float, default=4.15,
                   help="Uniform water-shell thickness, in Angstrom, applied to the whole "
                        "solute when -separateshell no (default: 4.15).")
    p.add_argument("-dsi", type=float, default=4.0,
                   help="Min distance between ions and solute surface, nm (default: 4.0).")
    p.add_argument("-dii", type=float, default=3.0,
                   help="Min distance between any two ions, nm (default: 3.0).")
    p.add_argument("-db", type=float, default=3.0,
                   help="Distance between ions and box boundary, nm (default: 3.0).")

    p.add_argument("-netcharge", type=int, default=0,
                   help="Net charge of the system; sets the number of counterions (default: 0).")
    p.add_argument("-nmids", type=_min3_int, default=5,
                   help="Number of independent run directories (mid_1..mid_N) to create "
                        "with the key outputs (integer >= 3; default: 5).")
    p.add_argument("-water", choices=sorted(WATER_MODELS), default="tip3p",
                   help="Water model (tip3p|tip4p|tip5p). Selects the solvate box (default: tip3p).")
    p.add_argument("-watername", default="SOL",
                   help="Water molecule name written to the [ molecules ] section of "
                        "W_topol.top (default: SOL; use e.g. TIP3 if your topology's "
                        "water moleculetype is named differently).")

    p.add_argument("-separateshell", choices=["yes", "no"], default="no",
                   help="no: apply -dshell uniformly to every non-solvent residue. "
                        "yes: read per-group thicknesses from -shelldef.")
    p.add_argument("-shelldef",
                   help="Shell-definition file (required with -separateshell yes). Each "
                        "non-comment line is 'thickness_A start:end', e.g. '4.15 1:195'.")

    p.add_argument("-datadir", default=DEFAULT_DATADIR,
                   help="Directory holding auxiliary files (hydrated-ion .gro and "
                        "INP make_ndx templates). Default: %(default)s")
    return p


def parse_shelldef(path):
    """Parse a shell-definition file into a list of ((start, end), thickness).

    Format: one group per line, '#'-prefixed comments and blank lines ignored:
        # shell_thickness(A) residue_range
        4.15 1:195
        4.50 196:200
    """
    if not os.path.isfile(path):                                   # rule 1
        die("-shelldef file not found: %s" % path)

    groups = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()       # drop inline / full comments
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                die("shelldef line %d: expected 'thickness start:end', got %r"
                    % (lineno, raw.strip()))
            tpart, rpart = parts
            try:
                thickness = float(tpart)
            except ValueError:
                die("shelldef line %d: invalid thickness %r" % (lineno, tpart))
            rng = parse_range(rpart)                               # rule 2
            groups.append((rng, thickness))

    if not groups:
        die("-shelldef file %s contains no shell definitions." % path)
    return groups


def build_groups(args, present_resids, solute_range):
    """Return a validated list of ((start, end), thickness) shell groups.

    -separateshell no  -> one group: the whole solute (solute_range) at -dshell.
    -separateshell yes -> the groups listed in the -shelldef file.
    """
    if args.separateshell == "yes":
        if not args.shelldef:
            die("-separateshell yes requires -shelldef <file>.")
        groups = parse_shelldef(args.shelldef)
    else:
        if args.shelldef:
            print("WARNING: -shelldef is ignored unless -separateshell yes is set.",
                  file=sys.stderr)
        if args.dshell is None or args.dshell <= 0:
            die("-separateshell no requires a positive -dshell.")
        groups = [(solute_range, args.dshell)]

    # --- rule 5: positive thicknesses -------------------------------------- #
    for rng, d in groups:
        if d is None or d <= 0:
            die("shell thickness for residues %d:%d must be positive (got %r)." % (rng[0], rng[1], d))

    # --- rule 3: ranges must not overlap ----------------------------------- #
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = range_ids(groups[i][0]) & range_ids(groups[j][0])
            if overlap:
                lo, hi = min(overlap), max(overlap)
                die("residue ranges %d:%d and %d:%d overlap (e.g. residues %d..%d)." % (
                    groups[i][0][0], groups[i][0][1],
                    groups[j][0][0], groups[j][0][1], lo, hi))

    # --- rule 4: every referenced residue ID must exist in the structure --- #
    for rng, _ in groups:
        missing = sorted(range_ids(rng) - present_resids)
        if missing:
            preview = ", ".join(str(m) for m in missing[:10])
            more = " ..." if len(missing) > 10 else ""
            die("residues %d:%d include IDs not present in the input structure: %s%s"
                % (rng[0], rng[1], preview, more))
    return groups


# =========================================================================== #
# Topology editing
#
# The user-provided topology is assumed fully prepared: all #include forcefield statements
# and all non-solvent molecule counts (protein, ligand, cofactor, ...) are kept
# EXACTLY as given. parch_shell only rewrites the solvent (water) and counter-
# ion count lines in [ molecules ].
# =========================================================================== #
def _strip_molecule_lines(lines, names):
    pat = re.compile(r"^\s*(" + "|".join(re.escape(n) for n in names) + r")\s+\d+\s*$")
    return [l for l in lines if not pat.match(l)]


def write_final_topology(src_top, dst_top, water_resname, sol_count,
                         ion_resname=None, ion_count=0):
    """Write W_topol.top: copy the input topology verbatim, drop any existing
    solvent/ion [ molecules ] count lines, then append the final water (and
    counter-ion) counts. Nothing else in the topology is touched."""
    with open(src_top) as fh:
        lines = fh.readlines()

    lines = _strip_molecule_lines(lines, SOLVENT_ION_NAMES)

    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.append("%-16s %d\n" % (water_resname, sol_count))
    if ion_resname and ion_count:
        lines.append("%-16s %d\n" % (ion_resname, ion_count))

    with open(dst_top, "w") as fh:
        fh.writelines(lines)


# =========================================================================== #
# Counterion placement (sphere of random, mutually-distant points)
# =========================================================================== #
def random_vector(radius):
    rand_i, rand_j = np.random.rand(2)
    theta = np.pi * rand_i
    phi = (2.0 * np.pi) * rand_j
    return np.array([radius * np.sin(theta) * np.cos(phi),
                     radius * np.sin(theta) * np.sin(phi),
                     radius * np.cos(theta)])


def cal_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def place_ions(n_ions, shell_radius, dis_ion_ion):
    """Return `n_ions` points on a sphere of radius `shell_radius` (nm), each
    pair separated by at least `dis_ion_ion` (nm)."""
    position = []
    for i in range(0, n_ions * 10000):
        point = random_vector(shell_radius)
        if i == 0:
            position.append(point)
        else:
            if all(cal_distance(p, point) >= dis_ion_ion for p in position):
                position.append(point)
        if len(position) == n_ions:
            break
    if len(position) < n_ions:
        die("could not place %d ions with min separation %.2f nm on a %.2f nm "
            "sphere; try lowering -dii or raising -dsi." % (n_ions, dis_ion_ion, shell_radius))
    return np.array(position)


# =========================================================================== #
# Main workflow
# =========================================================================== #
def main(argv=None):
    args = build_parser().parse_args(argv)

    # ---- resolve auxiliary files ------------------------------------------ #
    datadir = os.path.abspath(args.datadir)
    if not os.path.isdir(datadir):
        die("--datadir not found: %s" % datadir)
    wm = WATER_MODELS[args.water]
    solvent_box = wm["solvent"]
    hydrated = {"NA": wm["na"], "CL": wm["cl"]}

    structure = os.path.abspath(args.structure)
    topology = os.path.abspath(args.topology)
    for path in (structure, topology):
        if not os.path.isfile(path):
            die("input file not found: %s" % path)
    # Resolve the shell-definition file relative to the *input* directory now,
    # before we switch into the shell/ output directory below.
    if args.shelldef:
        args.shelldef = os.path.abspath(args.shelldef)

    # ---- inspect the input: present residues + solute (non-solvent) extent  #
    U_in = mda.Universe(structure)
    present_resids = set(int(r) for r in np.unique(U_in.atoms.resids))
    solvent_sel_names = [n for n in SOLVENT_ION_NAMES if n.isalnum()]
    non_solvent = U_in.select_atoms("not resname " + " ".join(solvent_sel_names))
    if len(non_solvent) == 0:
        die("input structure %s contains no non-solvent (solute) residues."
            % os.path.basename(structure))
    solute_resids = np.unique(non_solvent.resids)
    solute_range = (int(solute_resids.min()), int(solute_resids.max()))

    # ---- shell groups + validation (rules 1-6) ---------------------------- #
    groups = build_groups(args, present_resids, solute_range)

    print("Shell groups (residues -> thickness A):")
    for rng, d in groups:
        print("    %d:%d -> %.3f" % (rng[0], rng[1], d))

    # MDAnalysis selection fragments shared throughout.
    solute_sel = " or ".join("(resid %d:%d)" % (r[0], r[1]) for r, _ in groups)
    r_line = "r %d-%d" % (min(r[0] for r, _ in groups), max(r[1] for r, _ in groups))

    # ---- create the shell/ output directory and work inside it ------------ #
    # Inputs are read from the launch directory (absolute paths above); every
    # generated file is written under ./shell so the input directory stays clean.
    input_dir = os.getcwd()
    shell_dir = os.path.join(input_dir, "shell")
    os.makedirs(shell_dir, exist_ok=True)
    os.chdir(shell_dir)
    print("Output directory: %s" % shell_dir)

    # ---- stage the user structure ----------------------------------------- #
    shutil.copy(structure, os.path.join(shell_dir, "pp_out.gro"))

    # ---- refill a cubic box with solvent ---------------------------------- #
    # Residue numbering is the user's responsibility, so the original single-
    # water `gmx insert-molecules` reordering step is no longer needed: solvate
    # runs directly on the boxed structure.
    run("gmx editconf -f pp_out.gro -o pp_newbox.gro -c -d 1.5 -bt cubic")
    run("gmx solvate -cp pp_newbox.gro -cs %s -o box_solv.gro" % solvent_box)
    run("gmx editconf -f box_solv.gro -resnr 1 -o box_solv.gro")

    # ---- carve the water shell -------------------------------------------- #
    U2 = mda.Universe("box_solv.gro")
    shell_parts = ["(around %g (%s)) and (resname SOL)" % (d, "resid %d:%d" % (r[0], r[1]))
                   for r, d in groups]
    shell_sel = "(" + ") or (".join(shell_parts) + ")"

    sol_shell = U2.select_atoms(shell_sel)
    sol_shell_prot = U2.select_atoms("(%s) or (%s)" % (solute_sel, shell_sel))
    sol_shell_prot.residues.atoms.write("W_shell.gro", reindex=False)

    n_shell_waters = len(sol_shell.residues)
    print("Shell waters carved: %d" % n_shell_waters)

    # ---- geometry: solute extent for the ion sphere ----------------------- #
    run("gmx editconf -f W_shell.gro -center 0 0 0 -o W_init_c.gro > inform.txt")
    U0 = mda.Universe("W_init_c.gro")
    pp_cog = U0.select_atoms(solute_sel).center_of_geometry()
    solute_resids = sorted(set().union(*[range_ids(r) for r, _ in groups]))
    dist = []
    for rid in solute_resids:
        res = U0.select_atoms("resid %d" % rid)
        if len(res) == 0:
            continue
        dist.append(Mdistance.distance_array(pp_cog, res.center_of_geometry())[0][0])
    dist_max = np.max(np.array(dist)) / 10.0          # angstrom -> nm
    shell_radius = dist_max + args.dsi                # nm
    box_length = (shell_radius + args.db) * 2.0       # nm

    net_charge = args.netcharge

    # ---- build the make_ndx instruction file (PP / Solv / total groups) --- #
    inp_template = os.path.join(datadir, "INP_origin_ion" if net_charge != 0 else "INP_origin_no_ion")
    if not os.path.isfile(inp_template):
        die("INP template missing from datadir: %s" % inp_template)
    with open(inp_template) as fh:
        inp_lines = fh.readlines()
    inp_lines.insert(1, r_line + "\n")        # original: sed '2i r 1-bb'
    with open("INP", "w") as fh:
        fh.writelines(inp_lines)

    if net_charge != 0:
        # ---- place neutralising counterions ------------------------------- #
        ion_resname = "NA" if net_charge < 0 else "CL"
        ion_file = hydrated[ion_resname]
        shutil.copy(os.path.join(datadir, ion_file), os.path.join(os.getcwd(), ion_file))

        positions = place_ions(abs(net_charge), shell_radius, args.dii)
        np.savetxt("position.dat", positions)
        print("ion sphere radius (nm) = %.3f" % shell_radius)

        run("gmx insert-molecules -f W_init_c.gro -ci %s -o pp_ion.gro -ip position.dat -try 20000 -scale 0.01"
            % ion_file)

        # gmx insert-molecules adds each ion together with its hydration
        # waters, so ion atoms end up interleaved among the waters. Reorder the
        # atoms to match the topology's [ molecules ] block, which lists all
        # ions AFTER the complete water section: solute -> all SOL -> all ions.
        U1 = mda.Universe("pp_ion.gro")
        solvent_names = " ".join(n for n in SOLVENT_ION_NAMES if n.isalnum())
        solute_atoms = U1.select_atoms("not resname " + solvent_names)
        water_atoms = U1.select_atoms("resname SOL")
        ion_atoms = U1.select_atoms("resname " + ion_resname)
        ordered = solute_atoms + water_atoms + ion_atoms   # '+' keeps this order
        ordered.write("W_all.gro", reindex=True)

        # Final SOL count includes the ions' hydration waters (counted directly).
        n_sol_final = len(water_atoms.residues)

        run("gmx make_ndx -f W_all.gro -o W_ind.ndx < INP")
        run("echo 0 0 | gmx editconf -f W_all.gro -o %s -box %g %g %g -n W_ind.ndx -c"
            % (args.output, box_length, box_length, box_length))

        write_final_topology(topology, "W_topol.top", args.watername, n_sol_final,
                             ion_resname=ion_resname, ion_count=abs(net_charge))
    else:
        # ---- neutral system: no ions -------------------------------------- #
        run("gmx make_ndx -f W_init_c.gro -o W_ind.ndx < INP")
        run("echo 0 0 | gmx editconf -f W_init_c.gro -o %s -box %g %g %g -n W_ind.ndx -c"
            % (args.output, box_length, box_length, box_length))

        write_final_topology(topology, "W_topol.top", args.watername, n_shell_waters)

    run("rm -rf ./#*", allow_fail=True)

    # ---- replicate key outputs into independent run directories ----------- #
    # mid_1 .. mid_<nmids> each get the three files needed to launch a run.
    key_files = [args.output, "W_topol.top", "W_ind.ndx"]
    n_runs = args.nmids
    for i in range(1, n_runs + 1):
        mid = "mid_%d" % i
        os.makedirs(mid, exist_ok=True)
        for kf in key_files:
            shutil.copy(kf, os.path.join(mid, os.path.basename(kf)))

    print("\nDone.")
    print("  output dir : %s" % shell_dir)
    print("  structure  : %s" % args.output)
    print("  topology   : W_topol.top")
    print("  index      : W_ind.ndx")
    print("  box (nm)   : %.3f cubic" % box_length)
    print("  run dirs   : %s/{%s}  (each holds %s)"
          % (os.path.basename(shell_dir),
             ",".join("mid_%d" % i for i in range(1, n_runs + 1)),
             ", ".join(os.path.basename(k) for k in key_files)))


if __name__ == "__main__":
    # Allow invocation as `parch shellsetup ...` / `parch_shell shellsetup ...`:
    # transparently drop a leading 'shellsetup' subcommand token.
    argv = sys.argv[1:]
    if argv and argv[0] == "shellsetup":
        argv = argv[1:]
    main(argv)
