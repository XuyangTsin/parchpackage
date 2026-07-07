#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parch calpv -- PARCH values from parch-analysis results, averaged over mid_* runs.

Reads the per-unit dehydration trend (``num_ww_temp_<tag>.txt``) written by
``parch analysis`` in each ``mid_*/analysis/`` directory, computes the PARCH
value (pv) of every analysis unit via the time-correlation integral of its
no-recross hydration series scaled by a reference ``hard_ref``, and averages the
pv across the mid_* runs (optionally as a trimmed mean that drops the lowest and
highest run per unit).

Directory it operates on (pass the shell/ path with -path)::

    shell/
    |- mid_1/analysis/num_ww_temp_<tag>.txt , pp_for_bfactor_0.pdb
    |- mid_2/analysis/...
    `- ...

Outputs (written under the shell/ directory). The averaged flavour depends on
-mean_without_min_max (the two are mutually exclusive):
    -mean_without_min_max no   -> ave_pv_<tag>.txt, std_pv_<tag>.txt,
                                  ave_correlation_pv.pdb
    -mean_without_min_max yes  -> ave_excl_pv_<tag>.txt, std_excl_pv_<tag>.txt,
                                  ave_excl_correlation_pv.pdb   (trimmed mean)
    parch_summary_<tag>.txt    -> labelled per-run table (always)
and, per run, mid_*/analysis/pv_<tag>.txt and correlation_pv.pdb.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import MDAnalysis as mda

from .shellsetup import die
from .analysis import parse_blocks, build_units, make_refs, ANALYSIS_DIRNAME


# Reference 'hard_ref' per water model + force field (from 4_pv_correlation.py).
HARD_REF = {
    "charmm_tip3p":   557187.5,
    "charmm_tip4p":   595454.55,
    "charmm_tip4pew": 568142.05,
    "charmm_tip5p":   609215.91,
    "amber_opc":      455075.76,
    "amber_tip3p":    361196.97,
    #the old reference values are kept for backward compatibility, 
    #which used for sort-then-sample in the old analysis,
    #the new ones above are used by default
    "charmm_tip3p_old":   828602.27,
    "charmm_tip4p_old":   834204.55,
    "charmm_tip4pew_old": 793704.55,
    "charmm_tip5p_old":   886392.05,
    "amber_opc_old":      548924.2424,
    "amber_tip3p_old":    813643.94,
}

# 11 annealing temperature points (time in ps, temperature in K).
TIME_VEC = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
TEMP_VEC = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])


# =========================================================================== #
# PARCH-value core (faithful to backup_analysis/4_pv_correlation.py)
# =========================================================================== #
def get_no_recross_series(vec):
    """Non-increasing 'no-recross' series: out[i] = min(vec[i], out[i-1])."""
    out = np.zeros_like(vec, dtype=float)
    out[0] = vec[0]
    for i in range(1, len(vec)):
        out[i] = min(vec[i], out[i - 1])
    return out


def calc_time_corr_fn(skip_step, fvec):
    """Single time-correlation value at lag `skip_step` (zero-padded)."""
    ndata = len(fvec)
    npt, total = 0, 0.0
    for i in range(ndata):
        j = i + skip_step
        fj = fvec[j] if j < ndata else 0.0
        npt += 1
        total += fvec[i] * fj
    return total / npt if npt > 0 else 0.0


def calc_time_corr_series(wt_vec):
    """Time-correlation series of the no-recross hydration vector."""
    nr = get_no_recross_series(wt_vec)
    return np.array([calc_time_corr_fn(t, nr) for t in range(len(wt_vec))])


def calc_time_integral(time_vec, ctau_vec, norm_factor=1.0):
    """Trapezoidal integral of the correlation series over time."""
    area = 0.0
    for i in range(1, len(time_vec)):
        area += (time_vec[i] - time_vec[i - 1]) * (ctau_vec[i - 1] + ctau_vec[i]) * 0.5
    return area / norm_factor


def parch_values(mat, hard_ref):
    """PARCH value per row of `mat` (units x ndata): scaled correlation integral."""
    n_units, ndata = mat.shape
    if ndata != len(TIME_VEC):
        die("num_ww_temp has %d columns but %d temperature points are expected; "
            "was -da/TEMP_POINTS changed?" % (ndata, len(TIME_VEC)))
    pv = np.empty(n_units)
    for r in range(n_units):
        ctau = calc_time_corr_series(mat[r])
        pv[r] = round((calc_time_integral(TIME_VEC, ctau) / hard_ref) * 10.0, 2)
    return pv


# =========================================================================== #
# Argument parsing
# =========================================================================== #
def positive_min3_int(text):
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if v < 3:
        raise argparse.ArgumentTypeError("must be >= 3")
    return v


def build_parser():
    p = argparse.ArgumentParser(
        prog="parch_calpv",
        description="Compute PARCH values from parch-analysis results and average over mid_* runs.",
        allow_abbrev=False,
    )
    p.add_argument("-path", dest="path", required=True,
                   help="Path to the 'shell' directory holding the analysed mid_* run dirs.")
    p.add_argument("-water_ff", choices=sorted(HARD_REF), default="charmm_tip3p",
                   help="Water model & force field used in annealing; selects hard_ref "
                        "(default: charmm_tip3p). Ignored if -newhardref is given.")
    p.add_argument("-newhardref", type=float,
                   help="User-defined hard_ref value; overrides -water_ff.")
    p.add_argument("-nmids", type=positive_min3_int, default=5,
                   help="Number of mid_* runs to include (integer >= 3; default: 5).")
    p.add_argument("-mean_without_min_max", choices=["yes", "no"], default="yes",
                   help="Exclude the lowest and highest pv per unit before averaging "
                        "(trimmed mean); automatically disabled when fewer than 5 runs "
                        "are used. Default: yes.")
    p.add_argument("-blocks",
                   help="Block-mapping file used during analysis; needed to colour the "
                        "output PDBs per sub-unit (sugar/base etc.).")
    return p


# =========================================================================== #
# Helpers
# =========================================================================== #
def find_num_ww_temp(analysis_dir):
    """Return (path, tag) of the single num_ww_temp_<tag>.txt in analysis_dir, or None."""
    hits = sorted(glob.glob(os.path.join(analysis_dir, "num_ww_temp_*.txt")))
    if not hits:
        return None
    if len(hits) > 1:
        die("multiple num_ww_temp_*.txt files in %s; keep only the one to analyse:\n  %s"
            % (analysis_dir, "\n  ".join(os.path.basename(h) for h in hits)))
    m = re.match(r"num_ww_temp_(.+)\.txt$", os.path.basename(hits[0]))
    return hits[0], (m.group(1) if m else "")


def colour_pdb(ref_pdb, out_pdb, pv, block_specs):
    """Write `ref_pdb` with each atom's b-factor set to its unit's pv value.

    Units are rebuilt with the same logic parch analysis used, so the pv row
    order matches. Returns True on success, False (with a note) if the unit
    layout does not match the pv vector (e.g. -blocks not supplied)."""
    u = mda.Universe(ref_pdb)
    resids = np.unique(u.atoms.resids)
    aa, bb = int(resids[0]), int(resids[-1])
    try:
        units, _fallback, _ignored = build_units(u, aa, bb, block_specs)
    except SystemExit:
        return False
    if len(units) != len(pv):
        print("      note: %d units rebuilt but %d pv values -- pass the same -blocks "
              "used for analysis to colour PDBs; skipping %s"
              % (len(units), len(pv), os.path.basename(out_pdb)))
        return False
    if not hasattr(u.atoms, "tempfactors"):
        u.add_TopologyAttr("tempfactors")
    u.atoms.tempfactors = 0.0
    for ref, val in zip(make_refs(u, units), pv):
        if len(ref):
            ref.tempfactors = float(val)
    u.atoms.write(out_pdb)
    return True


# =========================================================================== #
# Main
# =========================================================================== #
def main(argv=None):
    args = build_parser().parse_args(argv)

    shell_path = os.path.abspath(args.path)
    if not os.path.isdir(shell_path):
        die("-path is not a directory: %s" % shell_path)

    hard_ref = args.newhardref if args.newhardref is not None else HARD_REF[args.water_ff]
    if hard_ref <= 0:
        die("hard_ref must be positive (got %r)." % hard_ref)
    block_specs = parse_blocks(os.path.abspath(args.blocks)) if args.blocks else []

    # ---- locate analysed mid_* runs --------------------------------------- #
    mid_dirs = sorted(d for d in glob.glob(os.path.join(shell_path, "mid_*"))
                      if os.path.isdir(d))
    valid = []                                   # (mid_dir, num_ww_temp_path, tag)
    for mid in mid_dirs:
        found = find_num_ww_temp(os.path.join(mid, ANALYSIS_DIRNAME))
        if found:
            valid.append((mid, found[0], found[1]))
        else:
            print("  [%s] no analysis results (num_ww_temp_*.txt) -- skipped"
                  % os.path.basename(mid))
    if len(valid) < 3:
        die("found only %d analysed mid_* run(s); need at least 3." % len(valid))

    if len(valid) < args.nmids:
        print("WARNING: requested -nmids %d but only %d analysed runs available; "
              "using all %d." % (args.nmids, len(valid), len(valid)))
    used = valid[:args.nmids]
    n_used = len(used)

    trim = args.mean_without_min_max == "yes"
    if trim and n_used < 5:
        print("NOTE: -mean_without_min_max needs >= 5 runs; using %d, so the trimmed "
              "mean is disabled for this run." % n_used)
        trim = False

    tags = set(t for _, _, t in used)
    if len(tags) > 1:
        die("mixed cutoff tags across runs: %s -- analyse all runs with the same -da."
            % ", ".join(sorted(tags)))
    tag = used[0][2]

    print("hard_ref = %.4f  (%s)" % (hard_ref, "newhardref" if args.newhardref is not None
                                     else args.water_ff))
    print("using %d run(s): %s\n" % (n_used, ", ".join(os.path.basename(m) for m, _, _ in used)))

    # ---- per-run PARCH values --------------------------------------------- #
    pv_runs, labels = [], None
    for mid, ntw_path, _ in used:
        name = os.path.basename(mid)
        mat = np.loadtxt(ntw_path)
        if mat.ndim == 1:                        # single-unit solute -> (1, ndata)
            mat = mat.reshape(1, -1)
        pv = parch_values(mat, hard_ref)
        if labels is None:
            labels = _load_labels(os.path.dirname(ntw_path), len(pv))
        elif len(pv) != len(labels):
            die("[%s] has %d units but a previous run had %d -- inconsistent analysis."
                % (name, len(pv), len(labels)))
        np.savetxt(os.path.join(os.path.dirname(ntw_path), "pv_%s.txt" % tag), pv, fmt="%.2f")
        ref_pdb = os.path.join(os.path.dirname(ntw_path), "pp_for_bfactor_0.pdb")
        if os.path.isfile(ref_pdb):
            colour_pdb(ref_pdb, os.path.join(os.path.dirname(ntw_path),
                       "correlation_pv.pdb"), pv, block_specs)
        pv_runs.append(pv)
        print("  [%s] pv computed (%d units)" % (name, len(pv)))

    pv_mat = np.vstack(pv_runs)                   # (n_used, n_units)

    # ---- average: trimmed OR plain, depending on -mean_without_min_max ----- #
    # The two output flavours are mutually exclusive: only the requested one is
    # written (ave_excl_* when trimming, ave_* otherwise).
    if trim:
        srt = np.sort(pv_mat, axis=0)[1:-1, :]    # drop lowest & highest per unit
        ave = np.average(srt, axis=0)
        std = np.std(srt, axis=0)
        pv_file = "ave_excl_pv_%s.txt" % tag
        std_file = "std_excl_pv_%s.txt" % tag
        pdb_file = "ave_excl_correlation_pv.pdb"
        kind = "trimmed mean (lowest & highest run excluded per unit)"
    else:
        ave = np.average(pv_mat, axis=0)
        std = np.std(pv_mat, axis=0)
        pv_file = "ave_pv_%s.txt" % tag
        std_file = "std_pv_%s.txt" % tag
        pdb_file = "ave_correlation_pv.pdb"
        kind = "mean over all included runs"

    np.savetxt(os.path.join(shell_path, pv_file), ave, fmt="%.2f")
    np.savetxt(os.path.join(shell_path, std_file), std, fmt="%.2f")

    # ---- labelled summary table ------------------------------------------- #
    # Split each unit label into the residue id and an optional block name, and
    # attach the residue name from the reference structure:
    #   no blocks -> unit "5",  name "GLN"
    #   blocks    -> unit "5",  name "DG_PSU" / "DG_NBP"
    resmap = _resname_map(os.path.join(used[0][0], ANALYSIS_DIRNAME, "pp_for_bfactor_0.pdb"))
    unit_col, name_col = [], []
    for lab in labels:
        resid, _, block = lab.partition("_")
        unit_col.append(resid)
        rn = resmap.get(int(resid), "?") if resid.isdigit() else "?"
        name_col.append("%s_%s" % (rn, block) if block else rn)
    _write_summary(os.path.join(shell_path, "parch_summary_%s.txt" % tag),
                   unit_col, name_col, [os.path.basename(m) for m, _, _ in used],
                   pv_mat, ave, std, kind)

    # ---- pv-coloured average PDB (from mid_1's reference structure) -------- #
    ref_pdb = os.path.join(used[0][0], ANALYSIS_DIRNAME, "pp_for_bfactor_0.pdb")
    if os.path.isfile(ref_pdb):
        colour_pdb(ref_pdb, os.path.join(shell_path, pdb_file), ave, block_specs)

    # ---- report ----------------------------------------------------------- #
    print("\nDone. Averaged PARCH value over %d run(s)  [%s]:" % (n_used, kind))
    print("  per-unit results : %s" % os.path.join(shell_path, pv_file))
    print("  coloured PDB     : %s" % os.path.join(shell_path, pdb_file))
    print("  summary table    : %s" % os.path.join(shell_path, "parch_summary_%s.txt" % tag))


def _load_labels(analysis_dir, n):
    """Unit labels from block_labels.txt if present, else 1..n."""
    lp = os.path.join(analysis_dir, "block_labels.txt")
    if os.path.isfile(lp):
        labs = open(lp).read().split()
        if len(labs) == n:
            return labs
    return [str(i + 1) for i in range(n)]


def _resname_map(pdb_path):
    """resid -> residue name from a structure file (empty dict if unavailable)."""
    if not os.path.isfile(pdb_path):
        return {}
    u = mda.Universe(pdb_path)
    return {int(r.resid): r.resname for r in u.residues}


def _write_summary(path, unit_col, name_col, run_names, pv_mat, ave, std, kind):
    cols = ["unit", "name"] + run_names + ["ave", "std"]
    with open(path, "w") as fh:
        fh.write("# averaging: %s\n" % kind)
        fh.write("# " + "  ".join(cols) + "\n")
        for i in range(len(unit_col)):
            row = [unit_col[i], name_col[i]] + ["%.2f" % v for v in pv_mat[:, i]] \
                  + ["%.2f" % ave[i], "%.2f" % std[i]]
            fh.write("  ".join(row) + "\n")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] in ("calpv", "calval"):
        argv = argv[1:]
    main(argv)
