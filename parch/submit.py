#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parch submit -- stage and launch the PARCH annealing runs.

For each ``mid_*`` directory produced by ``parch shellsetup`` (which already
holds W_init.gro / W_topol.top / W_ind.ndx), this copies in the simulation
inputs (energy-min + annealing .mdp files, ion .itp files, and the SLURM job
script for the chosen partition) and -- unless told not to -- submits the job
with ``sbatch``.

    parch submit -path shell -partition cpu
    parch submit -path shell -nmids 3 -partition gpu -launch no

Each mid_i uses its own annealing protocol (heat_nvt_<i>.mdp) and job script
(evp_<i>_zest3.sh for cpu, evp_<i>_gpu.sh for gpu); edit those #SBATCH lines to
change your job requests.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

from .shellsetup import die


PACKAGE_DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_annealing")

# partition -> evp job-script suffix
PARTITION_SUFFIX = {"cpu": "zest3", "gpu": "gpu"}

# files copied into every mid_* (shared across runs)
SHARED_FILES = ["em.mdp"]


def positive_int(text):
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer")
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def build_parser():
    p = argparse.ArgumentParser(
        prog="parch_submit",
        description="Stage simulation files into mid_* runs and submit the PARCH annealing jobs.",
        allow_abbrev=False,
    )
    p.add_argument("-path", dest="path", required=True,
                   help="Path to the 'shell' directory holding the mid_* run dirs.")
    p.add_argument("-nmids", type=positive_int,
                   help="Number of mid_* runs to submit (mid_1..mid_N). "
                        "If omitted, all mid_* under -path are used.")
    p.add_argument("-launch", choices=["yes", "no"], default="yes",
                   help="Immediately submit the jobs with sbatch (default: yes).")
    p.add_argument("-partition", choices=sorted(PARTITION_SUFFIX), default="cpu",
                   help="cpu -> evp_<i>_zest3.sh, gpu -> evp_<i>_gpu.sh (default: cpu).")
    return p


def mid_index(mid_dir):
    """Integer i from a 'mid_<i>' directory name, or None."""
    m = re.match(r"mid_(\d+)$", os.path.basename(mid_dir.rstrip("/")))
    return int(m.group(1)) if m else None


def main(argv=None):
    args = build_parser().parse_args(argv)

    shell_path = os.path.abspath(args.path)
    if not os.path.isdir(shell_path):
        die("-path is not a directory: %s" % shell_path)
    if not os.path.isdir(PACKAGE_DATADIR):
        die("data directory not found: %s" % PACKAGE_DATADIR)

    suffix = PARTITION_SUFFIX[args.partition]

    # ---- decide which mid_* to process ------------------------------------ #
    found = {}
    for d in glob.glob(os.path.join(shell_path, "mid_*")):
        i = mid_index(d)
        if i is not None and os.path.isdir(d):
            found[i] = d
    if not found:
        die("no mid_* directories found under %s" % shell_path)

    if args.nmids:
        targets = list(range(1, args.nmids + 1))
        missing = [i for i in targets if i not in found]
        if missing:
            print("WARNING: requested mid_%s not present under %s"
                  % ("/mid_".join(map(str, missing)), shell_path))
    else:
        targets = sorted(found)

    if args.launch == "yes" and not shutil.which("sbatch"):
        die("sbatch not found on PATH; rerun with -launch no, or submit from a "
            "node that has SLURM.")

    print("Shell: %s" % shell_path)
    print("Partition: %s  (job script evp_<i>_%s.sh)" % (args.partition, suffix))
    print("Targets: %s\n" % ", ".join("mid_%d" % i for i in targets))

    submitted, staged = [], []
    for i in targets:
        mid = found.get(i)
        if mid is None:
            continue
        tag = os.path.basename(mid)

        if not os.path.isfile(os.path.join(mid, "W_init.gro")):
            print("  [%s] skipped -- no W_init.gro (run parch shellsetup first)" % tag)
            continue
        if os.path.isfile(os.path.join(mid, "w_h.gro")):
            print("  [%s] skipped -- annealing already done (w_h.gro present)" % tag)
            continue

        # required per-run inputs must exist in the data directory
        evp = "evp_%d_%s.sh" % (i, suffix)
        needed = SHARED_FILES + ["heat_nvt_%d.mdp" % i, evp]
        missing = [f for f in needed if not os.path.isfile(os.path.join(PACKAGE_DATADIR, f))]
        if missing:
            print("  [%s] skipped -- missing in datadir: %s" % (tag, ", ".join(missing)))
            continue

        for f in needed:
            shutil.copy(os.path.join(PACKAGE_DATADIR, f), os.path.join(mid, f))
        staged.append((tag, evp))

        if args.launch == "yes":
            rc = subprocess.run(["sbatch", evp], cwd=mid).returncode
            if rc == 0:
                print("  [%s] staged + submitted (%s)" % (tag, evp))
                submitted.append(tag)
            else:
                print("  [%s] staged, but sbatch failed (exit %d)" % (tag, rc))
        else:
            print("  [%s] staged (%s) -- not submitted" % (tag, evp))

    # ---- summary + guidance ----------------------------------------------- #
    print()
    if args.launch == "yes":
        print("Submitted %d job(s)." % len(submitted))
    else:
        print("Staged %d run(s); submit manually, e.g.:" % len(staged))
        for tag, evp in staged:
            print("    cd %s && sbatch %s" % (os.path.join(shell_path, tag), evp))
    if staged:
        print("\nTo change job requests (partition, nodes, walltime, ...), edit the "
              "#SBATCH lines in each mid_*/evp_<i>_%s.sh before submitting." % suffix)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "submit":
        argv = argv[1:]
    main(argv)
