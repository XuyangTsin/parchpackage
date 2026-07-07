#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parch cval -- raw correlation values from parch-analysis results, averaged over mid_* runs.

Same inputs as ``parch calpv``, but reports each unit's time-correlation
integral as-is, without dividing by a hard_ref (i.e. the value ``parch calpv``
computes right before it scales by hard_ref to produce the PARCH value).

Directory it operates on (pass the shell/ path with -path)::

    shell/
    |- mid_1/analysis/num_ww_temp_<tag>.txt , pp_for_bfactor_0.pdb
    |- mid_2/analysis/...
    `- ...

Outputs (written under the shell/ directory). The averaged flavour depends on
-mean_without_min_max (the two are mutually exclusive):
    -mean_without_min_max no   -> ave_cval_<tag>.txt, std_cval_<tag>.txt
    -mean_without_min_max yes  -> ave_excl_cval_<tag>.txt, std_excl_cval_<tag>.txt
                                  (trimmed mean)
    cval_summary_<tag>.txt -> labelled per-run table (always)
and, per run, mid_*/analysis/cval_<tag>.txt.
"""

import argparse
import glob
import os
import sys

import numpy as np

from .shellsetup import die
from .analysis import ANALYSIS_DIRNAME
from .calpv import (
    TIME_VEC, calc_time_corr_series, calc_time_integral,
    positive_min3_int, find_num_ww_temp,
    _load_labels, _resname_map, _write_summary,
)


# =========================================================================== #
# Raw correlation-integral core (no hard_ref normalisation)
# =========================================================================== #
def cval_values(mat):
    """Raw time-correlation integral per row of `mat` (units x ndata)."""
    n_units, ndata = mat.shape
    if ndata != len(TIME_VEC):
        die("num_ww_temp has %d columns but %d temperature points are expected; "
            "was -da/TEMP_POINTS changed?" % (ndata, len(TIME_VEC)))
    cval = np.empty(n_units)
    for r in range(n_units):
        ctau = calc_time_corr_series(mat[r])
        cval[r] = round(calc_time_integral(TIME_VEC, ctau), 2)
    return cval


# =========================================================================== #
# Argument parsing
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(
        prog="parch_cval",
        description="Compute raw correlation values (before divided by hard_ref) from parch-analysis "
                     "results and average over mid_* runs.",
        allow_abbrev=False,
    )
    p.add_argument("-path", dest="path", required=True,
                   help="Path to the 'shell' directory holding the analysed mid_* run dirs.")
    p.add_argument("-nmids", type=positive_min3_int, default=5,
                   help="Number of mid_* runs to include (integer >= 3; default: 5).")
    p.add_argument("-mean_without_min_max", choices=["yes", "no"], default="yes",
                   help="Exclude the lowest and highest cval per unit before averaging "
                        "(trimmed mean); automatically disabled when fewer than 5 runs "
                        "are used. Default: yes.")
    return p


# =========================================================================== #
# Main
# =========================================================================== #
def main(argv=None):
    args = build_parser().parse_args(argv)

    shell_path = os.path.abspath(args.path)
    if not os.path.isdir(shell_path):
        die("-path is not a directory: %s" % shell_path)

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

    print("using %d run(s): %s\n" % (n_used, ", ".join(os.path.basename(m) for m, _, _ in used)))

    # ---- per-run raw correlation values ------------------------------------ #
    cval_runs, labels = [], None
    for mid, ntw_path, _ in used:
        name = os.path.basename(mid)
        mat = np.loadtxt(ntw_path)
        if mat.ndim == 1:                        # single-unit solute -> (1, ndata)
            mat = mat.reshape(1, -1)
        cval = cval_values(mat)
        if labels is None:
            labels = _load_labels(os.path.dirname(ntw_path), len(cval))
        elif len(cval) != len(labels):
            die("[%s] has %d units but a previous run had %d -- inconsistent analysis."
                % (name, len(cval), len(labels)))
        np.savetxt(os.path.join(os.path.dirname(ntw_path), "cval_%s.txt" % tag), cval, fmt="%.2f")
        cval_runs.append(cval)
        print("  [%s] correlation values computed (%d units)" % (name, len(cval)))

    cval_mat = np.vstack(cval_runs)               # (n_used, n_units)

    # ---- average: trimmed OR plain, depending on -mean_without_min_max ----- #
    # The two output flavours are mutually exclusive: only the requested one is
    # written (ave_excl_* when trimming, ave_* otherwise).
    if trim:
        srt = np.sort(cval_mat, axis=0)[1:-1, :]  # drop lowest & highest per unit
        ave = np.average(srt, axis=0)
        std = np.std(srt, axis=0)
        cval_file = "ave_excl_cval_%s.txt" % tag
        std_file = "std_excl_cval_%s.txt" % tag
        kind = "trimmed mean (lowest & highest run excluded per unit)"
    else:
        ave = np.average(cval_mat, axis=0)
        std = np.std(cval_mat, axis=0)
        cval_file = "ave_cval_%s.txt" % tag
        std_file = "std_cval_%s.txt" % tag
        kind = "mean over all included runs"

    np.savetxt(os.path.join(shell_path, cval_file), ave, fmt="%.2f")
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
    _write_summary(os.path.join(shell_path, "cval_summary_%s.txt" % tag),
                   unit_col, name_col, [os.path.basename(m) for m, _, _ in used],
                   cval_mat, ave, std, kind)

    # ---- report ----------------------------------------------------------- #
    print("\nDone. Averaged raw correlation value over %d run(s)  [%s]:" % (n_used, kind))
    print("  per-run results : %s" % os.path.join(shell_path, cval_file))
    print("  summary table    : %s" % os.path.join(shell_path, "cval_summary_%s.txt" % tag))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "cval":
        argv = argv[1:]
    main(argv)