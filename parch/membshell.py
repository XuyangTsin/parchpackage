#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parch membshell -- build a hydration shell (+ counterions) for PARCH annealing
of an entire lipid membrane.

Membrane counterpart of ``parch shellsetup``. Where ``shellsetup`` hydrates a
globular solute inside a cubic box and scatters counterions on a sphere around
it, ``membshell`` is designed for a slab-like membrane that keeps its periodic
XY dimensions:

  1. solvate the membrane in its own *equilibrated* box (whose vectors already
     size the aqueous region, so only a physical amount of water is added) and
     carve a hydration shell of a given thickness around the explicitly-specified
     membrane (lipid) residues,
  2. (``-delewater``) delete the waters that the solvation step inserted *inside*
     the membrane core, using leaflet reference atoms to locate the two
     head-group planes,
  3. resize to the requested box (XY padded by ``-dbxy``; Z set absolutely to
     ``2*dbz`` with the system centred) and optionally place neutralising
     counterions in two planar regions at ``-dsi`` above and below the system's Z
     centre (never scattered on a sphere -- membranes are planar),
  4. write the hydrated structure (``-o``) and an updated topology
     (``W_topol.top``) holding the correct number of shell waters / ions.

The input structure must be membrane-only (lipids, no water/ions -- e.g. the
output of ``parch prep``) with sequentially-numbered residue IDs, and must carry
the equilibrated box vectors so its periodic XY size is preserved.

All command-line arguments match ``parch shellsetup`` except:
  -dsi        distance from the system's Z CENTRE to the INNERMOST ion plane (nm).
              Measured from the centre (not a surface extremum) so a protruding
              membrane protein cannot displace the planes; requires dsi > memb_z/2.
  -ionplanes  number of ion planes, even: 2 (default), 4, 6 or 8, split evenly
              above/below. Each side stacks ionplanes/2 planes outwards from
              centre+/-dsi, spaced -dii apart in Z -- so ions in DIFFERENT planes
              satisfy -dii by construction and only in-plane packing is checked.
  -dbxy       replaces the cubic -db in XY: box_xy = membrane_xy + 2*dbxy.
  -dbz        gap from the OUTERMOST ion plane to the box wall (nm). The box height
              follows from the plane layout, giving an elongated (non-cubic) box
              independent of how far any protein protrudes:
                  box_z = 2 * (dbz + (ionplanes/2 - 1)*dii + dsi)
  -delewater  leaflet-reference-atom file used to delete intra-membrane waters.

Usage (also callable as ``parch membshell ...``):

    parch membshell -f membrane.gro -p topol.top -o W_init.gro \
        -dshell 4.15 -dsi 4.0 -dbxy 0.0 -dbz 6.0 \
        -delewater leaflet_atoms.txt -netcharge 0
"""

import argparse
import math
import os
import shutil
import sys

import numpy as np
import MDAnalysis as mda

# Reuse the shared machinery from shellsetup so the two tools stay in lock-step.
from .shellsetup import (
    die,
    run,
    range_ids,
    _min3_int,
    build_groups,
    write_final_topology,
    WATER_MODELS,
    SOLVENT_ION_NAMES,
    DEFAULT_DATADIR,
)


# =========================================================================== #
# Argument parsing
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(
        prog="parch_membshell",
        description="Build a hydration shell (+ counterions) for PARCH annealing "
                    "of a lipid membrane.",
        allow_abbrev=False,
    )
    p.add_argument("-f", dest="structure", required=True,
                   help="Input membrane structure (.gro or .pdb): lipids only, with "
                        "sequential residue IDs and equilibrated box vectors.")
    p.add_argument("-p", dest="topology", required=True,
                   help="Input topology file (.top). Updated copy saved as W_topol.top.")
    p.add_argument("-o", dest="output", default="W_init.gro",
                   help="Output hydrated structure file name (default: W_init.gro).")

    p.add_argument("-dshell", type=float, default=4.15,
                   help="Uniform water-shell thickness, in Angstrom, applied to the whole "
                        "membrane when -separateshell no (default: 4.15).")
    p.add_argument("-dsi", type=float, default=4.0,
                   help="Distance from the system's Z CENTRE to the INNERMOST ion plane, nm. "
                        "Must exceed the solute's Z half-extent, or the ions land inside the "
                        "membrane/protein (default: 4.0).")
    p.add_argument("-dii", type=float, default=3.0,
                   help="Min distance between any two ions in the same plane, nm (default: 3.0).")
    p.add_argument("-dbxy", type=float, default=0.0,
                   help="Padding between the membrane and the box boundary in X and Y, nm. "
                        "box_xy = membrane_xy + 2*dbxy. Use 0 to preserve the input periodic "
                        "XY dimensions (default: 0.0).")
    p.add_argument("-dbz", type=float, default=2.0,
                   help="Distance from the OUTERMOST ion plane to the box boundary in Z, nm "
                        "(default: 2.0). The full box height follows from the plane layout: "
                        "box_z = 2*(dbz + (ionplanes/2 - 1)*dii + dsi), with the system "
                        "centred inside it.")
    p.add_argument("-ionplanes", type=int, choices=[2, 4, 6, 8], default=2,
                   help="Number of ion planes, split evenly above and below the membrane. "
                        "Must be even: 2, 4, 6 or 8 (default: 2). Planes on each side are "
                        "stacked outwards from centre+/-dsi, spaced -dii apart in Z (so ions "
                        "in different planes automatically satisfy the -dii minimum). Raise "
                        "this when one plane cannot hold all the counterions.")

    p.add_argument("-delewater",
                   help="File defining the leaflet reference atoms. Either a bare "
                        "whitespace-separated atom-name list (e.g. 'P', or 'C1 C2 C3' for "
                        "glycerol carbons) or a full MDAnalysis selection when names alone "
                        "are ambiguous (e.g. 'resname POPC POPE POPS and name C1 C2 C3' to "
                        "exclude cholesterol/sugar carbons of the same name). Their mean Z "
                        "sets the membrane mid-plane; the mean Z of the atoms above/below it "
                        "defines the upper and lower head-group planes, and every water "
                        "between those planes (i.e. inside the membrane core) is deleted.")

    p.add_argument("-netcharge", type=int, default=0,
                   help="Net charge of the system; sets the number of counterions (default: 0).")
    p.add_argument("-nmids", type=_min3_int, default=5,
                   help="Number of independent run directories (mid_1..mid_N) to create "
                        "with the key outputs (integer >= 3; default: 5).")
    p.add_argument("-water", choices=sorted(WATER_MODELS), default="tip3p",
                   help="Water model (tip3p|tip4p|tip5p). Selects the solvate box (default: tip3p).")
    p.add_argument("-watername", default="SOL",
                   help="Water molecule name written to the [ molecules ] section of "
                        "W_topol.top (default: SOL).")

    p.add_argument("-separateshell", choices=["yes", "no"], default="no",
                   help="no: apply -dshell uniformly to every non-solvent residue. "
                        "yes: read per-group thicknesses from -shelldef.")
    p.add_argument("-shelldef",
                   help="Shell-definition file (required with -separateshell yes). Each "
                        "non-comment line is 'thickness_A start:end', e.g. '4.15 1:128'.")

    p.add_argument("-datadir", default=DEFAULT_DATADIR,
                   help="Directory holding auxiliary files (hydrated-ion .gro and "
                        "INP make_ndx templates). Default: %(default)s")
    return p


# =========================================================================== #
# Membrane-specific helpers
# =========================================================================== #
# MDAnalysis selection keywords: if any appears in the -delewater file, the file
# is treated as a full selection string rather than a bare atom-name list.
_MDA_SELECT_KEYWORDS = frozenset((
    "resname", "resid", "resnum", "resnums", "name", "type", "and", "or", "not",
    "around", "prop", "segid", "same", "byres", "bynum", "index", "group",
    "global", "point", "sphlayer", "cylayer", "sphzone", "cyzone",
))


def read_reference_selection(path):
    """Build an MDAnalysis selection string from the -delewater file.

    The file may contain either:
      * a bare whitespace-separated atom-name list (e.g. ``P`` or ``C1 C2 C3``),
        which becomes ``name C1 C2 C3``; or
      * a full MDAnalysis selection (e.g. ``resname POPC POPE and name C1 C2 C3``,
        detected by the presence of a selection keyword), used verbatim.
    ``#`` comments and blank lines are ignored.
    """
    if not os.path.isfile(path):
        die("-delewater file not found: %s" % path)
    tokens = []
    with open(path) as fh:
        for raw in fh:
            tokens.extend(raw.split("#", 1)[0].split())
    if not tokens:
        die("-delewater file %s contains no atom names / selection." % path)
    if any(t.lower() in _MDA_SELECT_KEYWORDS for t in tokens):
        return " ".join(tokens)          # full selection, used verbatim
    return "name " + " ".join(tokens)    # bare atom-name list


def delete_interior_waters(in_gro, out_gro, ref_selection, water_resname="SOL"):
    """Delete waters that sit inside the membrane core.

    The reference atoms (``ref_selection``, e.g. lipid phosphates or glycerol
    carbons) are split into an upper and lower leaflet at their mean Z (the
    mid-plane). The mean Z of each leaflet defines the upper / lower head-group
    planes; any water whose residue centre lies strictly between those two planes
    is removed.

    Returns (n_deleted, z_lower, z_upper) with the plane heights in Angstrom.
    """
    U = mda.Universe(in_gro)
    ref = U.select_atoms(ref_selection)
    if len(ref) == 0:
        die("-delewater: selection '%s' matched no atoms in %s."
            % (ref_selection, os.path.basename(in_gro)))

    ref_z = ref.positions[:, 2]
    midplane = float(ref_z.mean())
    upper = ref_z[ref_z > midplane]
    lower = ref_z[ref_z <= midplane]
    if len(upper) == 0 or len(lower) == 0:
        die("-delewater: reference atoms all fall on one side of the mid-plane "
            "(%.2f A); check the atom-name list." % midplane)
    z_upper = float(upper.mean())
    z_lower = float(lower.mean())

    water = U.select_atoms("resname %s" % water_resname)
    if len(water) == 0:
        print("WARNING: -delewater found no '%s' waters to inspect." % water_resname,
              file=sys.stderr)
        shutil.copy(in_gro, out_gro)
        return 0, z_lower, z_upper

    res = water.residues
    zc = np.array([r.atoms.center_of_geometry()[2] for r in res])
    interior = res[(zc > z_lower) & (zc < z_upper)]

    kept = U.atoms - interior.atoms
    kept.write(out_gro, reindex=True)
    return len(interior), z_lower, z_upper


def _hex_lattice_pbc(Lx, Ly, dii):
    """Commensurate hexagonal lattice filling a PERIODIC Lx x Ly domain with all
    (minimum-image) separations >= dii. Rows are forced even so the periodic seam
    keeps the half-row offset -- with an odd row count rows 0 and n-1 would sit
    directly on top of each other, only 0.866*dii apart.
    Returns [(x, y), ...] in [0, Lx) x [0, Ly)."""
    ncols, nrows = _hex_lattice_shape(Lx, Ly, dii)
    if ncols < 1 or nrows < 2:
        return []
    sx, sy = Lx / ncols, Ly / nrows
    sites = []
    for r in range(nrows):
        xoff = sx / 2.0 if r % 2 else 0.0
        for c in range(ncols):
            sites.append(((xoff + c * sx) % Lx, r * sy))
    return sites


def _hex_lattice_shape(Lx, Ly, dii):
    """(ncols, nrows) of the densest commensurate hex lattice with separation >= dii.

    Within a row the spacing is Lx/ncols >= dii; rows are Ly/nrows >= dii*sqrt(3)/2
    apart, so a neighbour in the adjacent (half-offset) row is at
    sqrt((sx/2)^2 + sy^2) >= dii.

    Counted analytically: never build the lattice just to size it. Site count grows
    as 1/dii^2, so a small -dii explodes it -- dii=0.001 nm over a 25 x 25 nm plane
    is ~7e8 points, which exhausts memory before returning a number nobody needed.
    """
    if dii <= 0 or Lx <= 0 or Ly <= 0:
        return 0, 0
    ncols = int(math.floor(Lx / dii + 1e-9))
    nrows = int(math.floor(Ly / (dii * math.sqrt(3.0) / 2.0) + 1e-9))
    if nrows % 2:                      # even row count -> periodic seam stays valid
        nrows -= 1
    return max(ncols, 0), max(nrows, 0)


def _plane_capacity(Lx, Ly, dii):
    """Max ions per plane at separation >= dii over a periodic Lx x Ly footprint."""
    ncols, nrows = _hex_lattice_shape(Lx, Ly, dii)
    return ncols * nrows


def _plane_counts(n_ions, n_planes):
    """Split n_ions over n_planes as evenly as possible (extras to the inner planes)."""
    base, rem = divmod(n_ions, n_planes)
    return [base + (1 if i < rem else 0) for i in range(n_planes)]


def check_ion_feasibility(n_ions, Lx, Ly, dii, n_planes):
    """Fail fast if the ions cannot be packed into the requested planes.

    Called BEFORE solvation so an impossible request costs milliseconds rather
    than a full solvate/carve cycle. Capacity is the periodic hexagonal bound --
    no arrangement of min-distance points beats it.

    Only -ionplanes is quoted as a concrete value, because it is exact integer
    arithmetic. -dii and -dbxy are mentioned qualitatively on purpose: their
    capacity thresholds are step functions (a whole lattice row/column at a time),
    so a rounded-off number can be one cliff short and silently still fail -- a
    confident recommendation that does not work is worse than no number at all.
    """
    per_plane = max(_plane_counts(n_ions, n_planes))
    cap = _plane_capacity(Lx, Ly, dii)
    if per_plane <= cap:
        return cap

    msg = ["cannot place %d ions in %d plane(s) with -dii %.2f nm over the "
           "%.2f x %.2f nm periodic footprint:" % (n_ions, n_planes, dii, Lx, Ly),
           "  each plane holds at most %d ions (ideal hexagonal packing), so %d "
           "plane(s) hold %d -- %d needed (%d per plane)."
           % (cap, n_planes, cap * n_planes, n_ions, per_plane)]
    if cap > 0:
        need = -(-n_ions // cap)              # ceil
        need += need % 2                      # -ionplanes must be even
        if need <= 8:
            msg.append("  Increase -ionplanes to %d (allowed: 2, 4, 6, 8)." % need)
        else:
            msg.append("  Even -ionplanes 8 (the maximum) is not enough here.")
    msg.append("  Decreasing -dii or increasing -dbxy may also help, but can be dangerous "
               "with them below 3 nm.")
    die("\n".join(msg))


def _lattice_plane(n, Lx, Ly, dii):
    """Lay n ions on a hexagonal lattice spanning the whole periodic Lx x Ly plane.

    The spacing is the LARGEST that still yields n sites -- never below dii -- so
    the ions spread over the full footprint. Opening the spacing up is the whole
    point: a random n-subset of a tight dii-spaced lattice is no better spread than
    uniform random sampling (measured: slightly worse, since random sampling at
    least has hard-core rejection pushing points apart). Picking the subset at
    random supplies the run-to-run variation.

    Every separation is >= the lattice spacing >= dii by construction, so no
    distance test is needed and there is no jamming/retry: placement is O(n).
    """
    if n <= 0:
        return []
    lo, hi = dii, max(Lx, Ly)
    for _ in range(60):                  # capacity is monotone-decreasing in spacing
        mid = 0.5 * (lo + hi)
        if _plane_capacity(Lx, Ly, mid) >= n:
            lo = mid
        else:
            hi = mid
    sites = _hex_lattice_pbc(Lx, Ly, lo)
    if len(sites) < n:                   # bisection may land a hair above the step
        sites = _hex_lattice_pbc(Lx, Ly, dii)
    if len(sites) < n:
        return None
    idx = np.random.permutation(len(sites))[:n]
    return [sites[i] for i in idx]


def place_ions_planes(n_ions, x_origin, y_origin, Lx, Ly, z_planes, dii):
    """Distribute n_ions over the given ion planes (z coordinates, nm).

    Each plane is filled independently over the periodic Lx x Ly footprint from a
    hexagonal lattice opened up to span the plane (see _lattice_plane), giving an
    even spread with every in-plane separation >= ``dii``.

    Ions in DIFFERENT planes need no check: consecutive planes are dii apart in z
    by construction, so the 3-D separation is already >= dii.
    """
    counts = _plane_counts(n_ions, len(z_planes))
    check_ion_feasibility(n_ions, Lx, Ly, dii, len(z_planes))

    position = []
    for n_plane, z in zip(counts, z_planes):
        pts = _lattice_plane(n_plane, Lx, Ly, dii)
        if pts is None:
            die("could not place %d ions in a plane at -dii %.2f nm." % (n_plane, dii))
        position.extend([np.array([x_origin + px, y_origin + py, z]) for px, py in pts])
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

    if args.dbz <= 0:
        die("-dbz (%.2f nm) must be positive: it is the gap between the outermost "
            "ion plane and the box boundary." % args.dbz)

    structure = os.path.abspath(args.structure)
    topology = os.path.abspath(args.topology)
    for path in (structure, topology):
        if not os.path.isfile(path):
            die("input file not found: %s" % path)
    if args.shelldef:
        args.shelldef = os.path.abspath(args.shelldef)
    if args.delewater:
        args.delewater = os.path.abspath(args.delewater)
        delewater_sel = read_reference_selection(args.delewater)

    # ---- inspect the input: residues, membrane extent, box vectors -------- #
    U_in = mda.Universe(structure)
    present_resids = set(int(r) for r in np.unique(U_in.atoms.resids))
    solvent_sel_names = [n for n in SOLVENT_ION_NAMES if n.isalnum()]
    membrane = U_in.select_atoms("not resname " + " ".join(solvent_sel_names))
    if len(membrane) == 0:
        die("input structure %s contains no non-solvent (membrane) residues."
            % os.path.basename(structure))
    memb_resids = np.unique(membrane.resids)
    solute_range = (int(memb_resids.min()), int(memb_resids.max()))

    box = U_in.dimensions
    if box is None or not np.all(box[:3] > 0):
        die("input structure %s has no valid box vectors; a membrane must carry "
            "its equilibrated periodic XY dimensions." % os.path.basename(structure))
    box_x_nm = box[0] / 10.0
    box_y_nm = box[1] / 10.0
    memb_z_nm = (membrane.positions[:, 2].max() - membrane.positions[:, 2].min()) / 10.0

    # Final (non-cubic) box. XY keeps the periodic size (+ -dbxy padding). Z is
    # built OUTWARD FROM THE ION PLANES: each side carries ionplanes/2 planes,
    # the innermost at centre +/- dsi and each next one -dii further out, then
    # -dbz of clearance to the wall. Nothing here depends on how far a membrane
    # protein protrudes. The system is centred in the box by the final editconf -c.
    n_side = args.ionplanes // 2
    box_lx = box_x_nm + 2.0 * args.dbxy
    box_ly = box_y_nm + 2.0 * args.dbxy
    box_lz = 2.0 * (args.dbz + (n_side - 1) * args.dii + args.dsi)

    half_z = memb_z_nm / 2.0
    if args.netcharge != 0 and args.dsi <= half_z:
        die("-dsi (%.2f nm) is measured from the system Z centre, but the solute's "
            "half-extent is %.2f nm: the innermost ion plane would fall INSIDE the "
            "membrane / protein. Increase -dsi above %.2f."
            % (args.dsi, half_z, half_z))
    if box_lz / 2.0 <= half_z:
        die("box_z = 2*(dbz + (ionplanes/2-1)*dii + dsi) = %.2f nm cannot contain the "
            "solute (Z extent %.2f nm). Increase -dsi or -dbz."
            % (box_lz, memb_z_nm))

    # Fail fast: check the ions can actually be packed BEFORE solvating/carving,
    # so an impossible request costs milliseconds instead of a full cycle.
    if args.netcharge != 0:
        check_ion_feasibility(abs(args.netcharge), box_lx, box_ly,
                              args.dii, args.ionplanes)

    # ---- shell groups + validation ---------------------------------------- #
    groups = build_groups(args, present_resids, solute_range)

    print("Shell groups (residues -> thickness A):")
    for rng, d in groups:
        print("    %d:%d -> %.3f" % (rng[0], rng[1], d))

    solute_sel = " or ".join("(resid %d:%d)" % (r[0], r[1]) for r, _ in groups)
    r_line = "r %d-%d" % (min(r[0] for r, _ in groups), max(r[1] for r, _ in groups))

    # ---- create the shell/ output directory and work inside it ------------ #
    input_dir = os.getcwd()
    shell_dir = os.path.join(input_dir, "shell")
    os.makedirs(shell_dir, exist_ok=True)
    os.chdir(shell_dir)
    print("Output directory: %s" % shell_dir)
    print("Box (nm): %.3f x %.3f x %.3f  (membrane XY %.3f x %.3f, Z %.3f)"
          % (box_lx, box_ly, box_lz, box_x_nm, box_y_nm, memb_z_nm))

    # ---- stage the user structure ----------------------------------------- #
    shutil.copy(structure, os.path.join(shell_dir, "pp_out.gro"))

    # ---- solvate in the *equilibrated* box, then carve the shell ---------- #
    # Solvation is done in the input's own equilibrated box, whose vectors
    # already size the aqueous region: only a physically reasonable amount of
    # water is added before the shell is carved. The requested box padding
    # (-dbxy / -dbz) is applied LATER, once the shell has been cut, so we never
    # generate (and immediately discard) a large excess of bulk water.
    run("gmx solvate -cp pp_out.gro -cs %s -o box_solv.gro" % solvent_box)
    run("gmx editconf -f box_solv.gro -resnr 1 -o box_solv.gro")

    # ---- carve the water shell around the membrane ------------------------ #
    U2 = mda.Universe("box_solv.gro")
    shell_parts = ["(around %g (%s)) and (resname SOL)" % (d, "resid %d:%d" % (r[0], r[1]))
                   for r, d in groups]
    shell_sel = "(" + ") or (".join(shell_parts) + ")"

    sol_shell_prot = U2.select_atoms("(%s) or (%s)" % (solute_sel, shell_sel))
    sol_shell_prot.residues.atoms.write("W_shell.gro", reindex=False)

    # ---- centre, then delete intra-membrane waters ------------------------ #
    run("gmx editconf -f W_shell.gro -center 0 0 0 -o W_shell_c.gro > inform.txt")

    if args.delewater:
        n_del, z_lo, z_up = delete_interior_waters("W_shell_c.gro", "W_init_c.gro",
                                                   delewater_sel)
        print("Intra-membrane waters deleted: %d  (core Z band %.2f .. %.2f A)"
              % (n_del, z_lo, z_up))
    else:
        shutil.copy("W_shell_c.gro", "W_init_c.gro")

    U0 = mda.Universe("W_init_c.gro")
    n_shell_waters = len(U0.select_atoms("resname SOL").residues)
    print("Shell waters retained: %d" % n_shell_waters)

    # ---- ion plane geometry ----------------------------------------------- #
    memb0 = U0.select_atoms(solute_sel)
    memb_z = memb0.positions[:, 2] / 10.0            # angstrom -> nm
    # Planes are referenced from the system's Z centre (the mid-point of the solute's
    # Z extent), NOT from a surface extremum: the centre is stable even when a
    # membrane protein protrudes past the bilayer faces.
    z_center = float(memb_z.max() + memb_z.min()) / 2.0
    # Stack ionplanes/2 planes per side, innermost at centre +/- dsi and each next
    # one -dii further out. Ordered inner-to-outer, alternating up/down, so any
    # remainder from the even split lands on the innermost planes symmetrically.
    z_planes = []
    for k in range(n_side):
        offset = args.dsi + k * args.dii
        z_planes.append(z_center + offset)
        z_planes.append(z_center - offset)

    # Ions are sampled over the PERIODIC final box footprint (not the solute's
    # atom extent, which exceeds the box once prep has made molecules whole),
    # centred on the membrane's XY centre.
    x_origin = float(memb0.positions[:, 0].mean() / 10.0) - box_lx / 2.0
    y_origin = float(memb0.positions[:, 1].mean() / 10.0) - box_ly / 2.0

    net_charge = args.netcharge

    # ---- build the make_ndx instruction file (PP / Solv / total groups) --- #
    inp_template = os.path.join(datadir, "INP_origin_ion" if net_charge != 0 else "INP_origin_no_ion")
    if not os.path.isfile(inp_template):
        die("INP template missing from datadir: %s" % inp_template)
    with open(inp_template) as fh:
        inp_lines = fh.readlines()
    inp_lines.insert(1, r_line + "\n")
    with open("INP", "w") as fh:
        fh.writelines(inp_lines)

    if net_charge != 0:
        # ---- place neutralising counterions in the ion planes -------------- #
        ion_resname = "NA" if net_charge < 0 else "CL"
        ion_file = hydrated[ion_resname]
        shutil.copy(os.path.join(datadir, ion_file), os.path.join(os.getcwd(), ion_file))

        positions = place_ions_planes(abs(net_charge), x_origin, y_origin,
                                      box_lx, box_ly, z_planes, args.dii)
        np.savetxt("position.dat", positions)
        counts = _plane_counts(abs(net_charge), len(z_planes))
        print("ion planes (nm): centre Z = %.3f, %d plane(s), %d ions"
              % (z_center, len(z_planes), abs(net_charge)))
        for z, c in sorted(zip(z_planes, counts)):
            print("    z = %+8.3f : %d ions" % (z, c))

        run("gmx insert-molecules -f W_init_c.gro -ci %s -o pp_ion.gro -ip position.dat -try 20000 -scale 0.01"
            % ion_file)

        # Reorder atoms to match the topology's [ molecules ] block:
        # membrane -> all SOL -> all ions.
        U1 = mda.Universe("pp_ion.gro")
        solvent_names = " ".join(n for n in SOLVENT_ION_NAMES if n.isalnum())
        solute_atoms = U1.select_atoms("not resname " + solvent_names)
        water_atoms = U1.select_atoms("resname SOL")
        ion_atoms = U1.select_atoms("resname " + ion_resname)
        ordered = solute_atoms + water_atoms + ion_atoms
        ordered.write("W_all.gro", reindex=True)

        n_sol_final = len(water_atoms.residues)

        run("gmx make_ndx -f W_all.gro -o W_ind.ndx < INP")
        run("echo 0 0 | gmx editconf -f W_all.gro -o %s -box %g %g %g -n W_ind.ndx -c"
            % (args.output, box_lx, box_ly, box_lz))

        write_final_topology(topology, "W_topol.top", args.watername, n_sol_final,
                             ion_resname=ion_resname, ion_count=abs(net_charge))
    else:
        # ---- neutral system: no ions -------------------------------------- #
        run("gmx make_ndx -f W_init_c.gro -o W_ind.ndx < INP")
        run("echo 0 0 | gmx editconf -f W_init_c.gro -o %s -box %g %g %g -n W_ind.ndx -c"
            % (args.output, box_lx, box_ly, box_lz))

        write_final_topology(topology, "W_topol.top", args.watername, n_shell_waters)

    run("rm -rf ./#*", allow_fail=True)

    # ---- replicate key outputs into independent run directories ----------- #
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
    print("  box (nm)   : %.3f x %.3f x %.3f" % (box_lx, box_ly, box_lz))
    print("  run dirs   : %s/{%s}  (each holds %s)"
          % (os.path.basename(shell_dir),
             ",".join("mid_%d" % i for i in range(1, n_runs + 1)),
             ", ".join(os.path.basename(k) for k in key_files)))


if __name__ == "__main__":
    # Allow invocation as `parch membshell ...`: drop a leading 'membshell' token.
    argv = sys.argv[1:]
    if argv and argv[0] == "membshell":
        argv = argv[1:]
    main(argv)
