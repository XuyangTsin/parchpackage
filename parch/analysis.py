#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parch analysis -- water-shell hydration analysis over the annealing ramp.

For each ``mid_*`` run, counts how many solvent molecules sit within a cutoff of
each solute residue (or ``-blocks`` sub-unit), then builds the monotonic
dehydration trend (``num_ww_temp``) with the FULL-TRAJECTORY envelope
(SORT-THEN-SAMPLE): enforce the non-increasing envelope over EVERY trajectory
frame first (all ~1000 frames), then read off the 11 temperature points. This is
less sensitive to which 11 frames are sampled than the alternative
SAMPLE-THEN-SORT (sample 11 frames, then enforce monotonicity among only those),
which remains available as ``parch analysis_old``. The full-trajectory envelope
matches ``sorting_1st_array3D`` in the old DNA/protein analysis scripts.

Given a completed PARCH workspace::

    <launch>/
    |- input.gro
    |- topol.top
    `- shell/                 <- pass this path with -path
       |- mid_1/  (em.tpr/.gro, w_h.tpr/.xtc, ...)
       |- mid_2/
       `- ...

it locates every ``mid_*`` directory and, for each one independently, counts how
many solvent molecules sit within a cutoff of each solute residue -- first in
the energy-minimised structure (``init_ww``) and then across the heating
trajectory (``NUM_w``) -- builds the full-trajectory monotonic dehydration trend
(``num_ww_temp``), and writes b-factor-coloured PDBs of the solute at 11 points
along the temperature ramp. All outputs are written into a fresh
``mid_x/analysis/`` directory.

Shell thickness:
  -separateshell no  : apply a single -da cutoff to every solute residue.
  -separateshell yes : per-residue cutoffs from -shelldef (same file format as
                       ``parch shellsetup``); residues not listed use -da.
"""

import argparse
import glob
import os
import sys

import numpy as np
import MDAnalysis as mda

# Shared helpers/conventions from the shellsetup module (same package).
from .shellsetup import die, parse_range, parse_shelldef, range_ids


# Solvent residue names recognised by default; -newwater appends to this list.
DEFAULT_SOLVENT = ["SOL", "OPC", "TP3", "T4D", "T4E"]

# Per-mid input files (annealing outputs) and the output sub-directory.
EM_TPR, EM_GRO = "em.tpr", "em.gro"
TRAJ_TPR, TRAJ_XTC = "w_h.tpr", "w_h.xtc"
ANALYSIS_DIRNAME = "analysis"

# Number of temperature points sampled along the heating trajectory.
TEMP_POINTS = 11


# =========================================================================== #
# Argument parsing
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(
        prog="parch_analysis",
        description="Water-shell hydration analysis over the PARCH annealing ramp "
                    "(full-trajectory monotonic trend).",
        allow_abbrev=False,
    )
    p.add_argument("-path", dest="path", required=True,
                   help="Path to the 'shell' directory holding the mid_* run dirs.")
    p.add_argument("-da", type=float, default=3.15,
                   help="Uniform water-shell cutoff in angstrom, applied to every solute "
                        "residue when -separateshell no (default: 3.15).")
    p.add_argument("-separateshell", choices=["yes", "no"], default="no",
                   help="no: apply -da uniformly. yes: per-residue cutoffs from -shelldef.")
    p.add_argument("-shelldef",
                   help="Shell-definition file (required with -separateshell yes); each "
                        "non-comment line is 'thickness_A start:end', e.g. '3.15 1:195'.")
    p.add_argument("-newwater", nargs="+", default=[],
                   help="Extra solvent residue name(s) to treat as water, in addition to "
                        "the defaults (%s)." % " ".join(DEFAULT_SOLVENT))
    p.add_argument("-blocks",
                   help="Block-mapping file. Each non-comment line is "
                        "'resid_range block_name atom1 atom2 ...' (all whitespace-"
                        "separated, no commas); a residue may appear on several lines to "
                        "be split into multiple blocks (e.g. DNA sugar vs base). Residues "
                        "not named fall back to whole-residue analysis.")
    p.add_argument("-overwrite", action="store_true",
                   help="Re-run analysis even if a mid_*/analysis/ already looks complete.")
    return p


# =========================================================================== #
# Helpers
# =========================================================================== #
def tag_for(da):
    """'3-15A' for da=3.15 -- matches the original output naming."""
    return ("%g" % da) + "A"


def _compact(resids):
    """Compress a sorted list of ints into 'a-b, c, e-f' for readable messages."""
    resids = sorted(resids)
    if not resids:
        return "(none)"
    spans, start, prev = [], resids[0], resids[0]
    for r in resids[1:]:
        if r == prev + 1:
            prev = r
        else:
            spans.append((start, prev))
            start = prev = r
    spans.append((start, prev))
    return ", ".join("%d" % a if a == b else "%d-%d" % (a, b) for a, b in spans)


def resolve_cutoffs(separateshell, da, shelldef, aa, bb, present_resids):
    """Return (cutoff_per_resid dict for aa..bb, validated groups list).

    Uniform mode -> every residue maps to `da`, no groups.
    Separate mode -> residues in a -shelldef group take that group's thickness,
                     residues not covered fall back to `da`.
    """
    cutoffs = {k: da for k in range(aa, bb + 1)}
    if separateshell != "yes":
        return cutoffs, []

    if not shelldef:
        die("-separateshell yes requires -shelldef <file>.")
    groups = parse_shelldef(shelldef)                      # rules 1,2,5

    # rule 3: no overlap
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            overlap = range_ids(groups[i][0]) & range_ids(groups[j][0])
            if overlap:
                lo, hi = min(overlap), max(overlap)
                die("residue ranges %d:%d and %d:%d overlap (e.g. residues %d..%d)." % (
                    groups[i][0][0], groups[i][0][1],
                    groups[j][0][0], groups[j][0][1], lo, hi))
    # rule 4: residues exist in the structure
    for rng, _ in groups:
        missing = sorted(range_ids(rng) - present_resids)
        if missing:
            preview = ", ".join(str(m) for m in missing[:10])
            more = " ..." if len(missing) > 10 else ""
            die("-shelldef residues %d:%d not present in the structure: %s%s"
                % (rng[0], rng[1], preview, more))
    # assign per-residue cutoffs
    for (a, b), thick in groups:
        for k in range(a, b + 1):
            cutoffs[k] = thick
    return cutoffs, groups


def shell_selection(cutoffs, aa, bb, solvent_sel):
    """MDAnalysis selection string for all solvent within each residue's cutoff
    of the solute -- union over distinct cutoff values."""
    # group solute residues by their cutoff so the selection stays compact
    by_cut = {}
    for k in range(aa, bb + 1):
        by_cut.setdefault(cutoffs[k], []).append(k)
    parts = []
    for cut, resids in by_cut.items():
        resid_tokens = " ".join(str(r) for r in resids)
        parts.append("(around %g (resid %s)) and (%s)" % (cut, resid_tokens, solvent_sel))
    return "(" + ") or (".join(parts) + ")"


def parse_blocks(path):
    """Parse a block-mapping file into ordered specs [((start,end), name, [atoms])].

    Each non-comment line: 'resid_range block_name atom1 atom2 ...'
    (all whitespace-separated; '#' starts a comment; blank lines ignored).
    The same resid range may appear on multiple lines (one per block).
    """
    if not os.path.isfile(path):
        die("-blocks file not found: %s" % path)
    specs = []
    with open(path) as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            toks = line.split()
            if len(toks) < 3:
                die("blocks line %d: expected 'resid_range block_name atom...', got %r"
                    % (lineno, raw.strip()))
            rng = parse_range(toks[0])
            specs.append((rng, toks[1], toks[2:]))
    if not specs:
        die("-blocks file %s contains no block definitions." % path)
    return specs


def build_units(U0, aa, bb, specs):
    """Turn block specs into an ordered list of analysis units.

    A unit is a dict {resid, block, atoms, label}:
      - block None / atoms None  -> whole residue (label "<resid>")
      - otherwise                -> a named atom subset (label "<resid>_<block>")

    Residues named in `specs` are split into their block(s) (in file order);
    residues not named fall back to a single whole-residue unit. Returns
    (units, fallback_resids, ignored_atoms) where ignored_atoms maps a covered
    resid to the present atom names left unassigned by its blocks.
    """
    present = set(int(r) for r in np.unique(U0.atoms.resids))

    block_map = {}                                   # resid -> [(name, [atoms]), ...]
    for (a, b), name, atoms in specs:
        if a < aa or b > bb:
            die("blocks range %d:%d lies outside the solute range %d:%d." % (a, b, aa, bb))
        missing = sorted(range_ids((a, b)) - present)
        if missing:
            preview = ", ".join(str(m) for m in missing[:10])
            die("blocks range %d:%d references residues not in the structure: %s%s"
                % (a, b, preview, " ..." if len(missing) > 10 else ""))
        for k in range(a, b + 1):
            block_map.setdefault(k, []).append((name, atoms))

    # validation: within a residue, no atom name may belong to two blocks
    for k, blist in block_map.items():
        owner = {}
        for name, atoms in blist:
            for at in atoms:
                if at in owner and owner[at] != name:
                    die("residue %d: atom %r is assigned to both blocks %r and %r."
                        % (k, at, owner[at], name))
                owner[at] = name

    units, fallback, ignored = [], [], {}
    for k in range(aa, bb + 1):
        if k in block_map:
            assigned = set()
            for name, atoms in block_map[k]:
                units.append({"resid": k, "block": name, "atoms": atoms,
                              "label": "%d_%s" % (k, name)})
                assigned |= set(atoms)
            present_names = set(U0.select_atoms("resid %d" % k).names)
            leftover = present_names - assigned
            if leftover:
                ignored[k] = sorted(leftover)
        else:
            units.append({"resid": k, "block": None, "atoms": None, "label": "%d" % k})
            fallback.append(k)
    return units, fallback, ignored


def make_refs(universe, units):
    """Reference AtomGroup for each unit (the atoms its hydration is measured
    around): the whole residue, or the named atom subset within it."""
    refs = []
    for u in units:
        ag = universe.select_atoms("resid %d" % u["resid"])
        if u["atoms"] is not None:
            ag = ag[np.isin(ag.names, u["atoms"])]
        refs.append(ag)
    return refs


def water_counts(universe, units, refs, cutoffs, solvent_sel):
    """Number of distinct solvent residues within each unit's cutoff of its
    reference atoms, for the universe's current frame. Length == len(units).

    Atom names (primes, leading digits) are matched via pre-built AtomGroups
    and the `group` selection keyword, never embedded in a selection string."""
    counts = []
    for u, ref in zip(units, refs):
        if len(ref) == 0:
            counts.append(0)
            continue
        wat = universe.select_atoms("(around %g group ref) and (%s)"
                                    % (cutoffs[u["resid"]], solvent_sel), ref=ref)
        counts.append(len(np.unique(wat.resids)))
    return np.array(counts)


# =========================================================================== #
# Per-mid analysis
# =========================================================================== #
def analyse_mid(mid_dir, args, solvent_sel, tag, block_specs):
    name = os.path.basename(mid_dir.rstrip("/"))

    needed = [EM_TPR, EM_GRO, TRAJ_TPR, TRAJ_XTC]
    missing = [f for f in needed if not os.path.isfile(os.path.join(mid_dir, f))]
    if missing:
        print("  [%s] skipped -- missing %s" % (name, ", ".join(missing)))
        return False

    outdir = os.path.join(mid_dir, ANALYSIS_DIRNAME)
    os.makedirs(outdir, exist_ok=True)

    done_marker = os.path.join(outdir, "pp_for_bfactor_%d.pdb" % (TEMP_POINTS - 1))
    if os.path.isfile(done_marker) and not args.overwrite:
        print("  [%s] skipped -- analysis already complete (use -overwrite)" % name)
        return False

    print("  [%s] analysing ..." % name)

    # ---- reference structure: solute extent + per-residue cutoffs --------- #
    U0 = mda.Universe(os.path.join(mid_dir, EM_TPR), os.path.join(mid_dir, EM_GRO))
    present = set(int(r) for r in np.unique(U0.atoms.resids))
    aa = int(np.unique(U0.atoms.resids)[0])
    solv = U0.select_atoms(solvent_sel)
    if len(solv) == 0:
        die("[%s] no solvent atoms found (looked for: %s). Use -newwater." % (name, solvent_sel))
    bb = int(np.unique(solv.resids)[0]) - 1               # last solute residue
    cutoffs, _groups = resolve_cutoffs(args.separateshell, args.da, args.shelldef,
                                       aa, bb, present)
    print("      solute residues %d:%d (%d residues)" % (aa, bb, bb - aa + 1))

    # ---- analysis units (whole residues, or residue sub-blocks) ----------- #
    units, fallback, ignored = build_units(U0, aa, bb, block_specs)
    n_units = len(units)
    if block_specs:
        n_split = len([u for u in units if u["block"] is not None])
        print("      blocks: %d sub-units across %d residues; %d residue(s) "
              "fall back to whole-residue analysis"
              % (n_split, len(set(u["resid"] for u in units if u["block"])), len(fallback)))
        if fallback:
            print("      NOTE: residues not named in -blocks are analysed whole-residue: %s"
                  % _compact(fallback))
        if ignored:
            ex = sorted(ignored)[0]
            print("      WARNING: %d block-covered residue(s) have atoms assigned to no "
                  "block (ignored); e.g. resid %d: %s"
                  % (len(ignored), ex, " ".join(ignored[ex])))

    # column/row labels for the per-unit output tables
    labels = [u["label"] for u in units]
    with open(os.path.join(outdir, "block_labels.txt"), "w") as fh:
        fh.write("\n".join(labels) + "\n")

    refs0 = make_refs(U0, units)

    # ---- initial hydration (em structure) --------------------------------- #
    init_num = water_counts(U0, units, refs0, cutoffs, solvent_sel)
    np.savetxt(os.path.join(outdir, "init_ww_%s.txt" % tag), init_num)
    print("      fully-dehydrated units at start (count==0): %d"
          % len(np.argwhere(init_num == 0)))

    # ---- hydration across the heating trajectory -------------------------- #
    U1 = mda.Universe(os.path.join(mid_dir, TRAJ_TPR), os.path.join(mid_dir, TRAJ_XTC))
    refs1 = make_refs(U1, units)
    n_frames = len(U1.trajectory)
    num_per_frame = np.empty((n_frames, n_units), dtype=int)
    for i, _ts in enumerate(U1.trajectory):
        num_per_frame[i] = water_counts(U1, units, refs1, cutoffs, solvent_sel)
    np.savetxt(os.path.join(outdir, "NUM_w.txt"), num_per_frame)

    # temperature-point frame indices, robust to the actual trajectory length
    step = (n_frames - 1) / float(TEMP_POINTS - 1)
    frame_idx = [min(int(round(t * step)), n_frames - 1) for t in range(TEMP_POINTS)]

    # ---- monotonic dehydration trend: full-trajectory envelope, then sample  #
    # Enforce the non-increasing envelope over EVERY frame first (all frames),
    # then read off the temperature points -- applied to ALL units (whether or
    # not -blocks is used). More robust than sampling 11 frames and then
    # enforcing monotonicity among only those (that is `parch analysis_old`).
    full = num_per_frame.astype(float).copy()                 # frames x units
    for i in range(1, n_frames):
        m = full[i - 1] < full[i]
        full[i, m] = full[i - 1, m]
    temp_cols = full[frame_idx, :].T                          # units x temp_points
    np.savetxt(os.path.join(outdir, "num_ww_temp_%s.txt" % tag), temp_cols)

    # ---- b-factor-coloured solute PDBs along the ramp --------------------- #
    if not hasattr(U1.atoms, "tempfactors"):
        U1.add_TopologyAttr("tempfactors")

    solute = U1.select_atoms("resid %d:%d" % (aa, bb))
    shell_sel = shell_selection(cutoffs, aa, bb, solvent_sel)
    for t, fi in enumerate(frame_idx):
        U1.trajectory[fi]
        tbf = num_per_frame[fi]                            # per-unit water count

        # pp_for_bfactor_<t>.pdb : the plain, complete solute system (resid
        # aa:bb), atom order preserved, no per-residue colouring.
        solute.atoms.tempfactors = 0.0
        solute.write(os.path.join(outdir, "pp_for_bfactor_%d.pdb" % t))

        # bfactor_ww_<t>.pdb : solute coloured by per-unit hydration number
        # (sugar vs base get different colours) + its shell waters (b=0).
        # Atoms in covered residues not assigned to any block stay at 0.
        for u_i, ref in zip(range(n_units), refs1):
            if len(ref):
                ref.tempfactors = float(tbf[u_i])
        # .residues.atoms expands to WHOLE water molecules (not just the atoms
        # that fell inside the cutoff), matching the original output.
        shell = U1.select_atoms(shell_sel).residues.atoms
        shell.tempfactors = 0.0
        (solute + shell).write(os.path.join(outdir, "bfactor_ww_%d.pdb" % t))

    print("      wrote outputs to %s" % outdir)
    return True


# =========================================================================== #
# Main
# =========================================================================== #
def main(argv=None):
    args = build_parser().parse_args(argv)

    shell_path = os.path.abspath(args.path)
    if not os.path.isdir(shell_path):
        die("-path is not a directory: %s" % shell_path)
    if args.shelldef:
        args.shelldef = os.path.abspath(args.shelldef)

    # Parse the block-mapping once (same definitions apply to every mid_*).
    block_specs = parse_blocks(os.path.abspath(args.blocks)) if args.blocks else []

    solvent_names = DEFAULT_SOLVENT + [w for w in args.newwater if w not in DEFAULT_SOLVENT]
    solvent_sel = "resname " + " ".join(solvent_names)
    tag = tag_for(args.da)

    mid_dirs = sorted(d for d in glob.glob(os.path.join(shell_path, "mid_*"))
                      if os.path.isdir(d))
    if not mid_dirs:
        die("no mid_* directories found under %s" % shell_path)

    print("Found %d mid_* directories under %s" % (len(mid_dirs), shell_path))
    print("Solvent residues: %s" % " ".join(solvent_names))
    print("Mode: %s  (cutoff tag %s)"
          % ("separateshell" if args.separateshell == "yes" else "uniform da=%g" % args.da, tag))
    if block_specs:
        print("Blocks: %d definition(s) from %s" % (len(block_specs), args.blocks))
    print()

    n_done = 0
    for mid in mid_dirs:
        if analyse_mid(mid, args, solvent_sel, tag, block_specs):
            n_done += 1

    print("\nDone. Analysed %d of %d mid_* directories." % (n_done, len(mid_dirs)))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "analysis":
        argv = argv[1:]
    main(argv)
