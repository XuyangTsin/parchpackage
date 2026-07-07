#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Console entry point for the ``parch`` package.

Thin dispatcher around the standalone analysis modules so that those modules
stay plain scripts. Installed (via pyproject) as the single ``parch`` command,
e.g. ``parch shellsetup ...`` / ``parch analysis -path ...``.
"""

import sys

from .shellsetup import main, die
from .analysis import main as analysis_main
from .analysis_old import main as analysis_old_main
from .calpv import main as calpv_main
from .cval import main as cval_main
from .prep import main as prep_main
from .submit import main as submit_main


def cli_parch(argv=None):
    """Console entry point for the ``parch`` umbrella command.

    Dispatches sub-commands, e.g. ``parch shellsetup -f ...`` / ``parch analysis -path ...``.
    """
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: parch <command> [options]\n\n"
              "commands:\n"
              "  prep         Extract selected molecules from an equilibrated system.\n"
              "  shellsetup   Build a hydration shell (+ counterions) for PARCH annealing.\n"
              "  submit       Stage simulation files into mid_* and submit the annealing jobs.\n"
              "  analysis     Water-shell hydration analysis (full-trajectory trend).\n"
              "  analysis_old Same as analysis but sample-then-sort trend (original).\n"
              "  calpv        PARCH values from analysis results, averaged over mid_* runs.\n"
              "  cval         Raw correlation values (pre-hard_ref) from analysis results, "
              "averaged over mid_* runs.\n\n"
              "Run 'parch <command> -h' for command options.")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "prep":
        prep_main(rest)
    elif cmd == "shellsetup":
        main(rest)
    elif cmd == "submit":
        submit_main(rest)
    elif cmd == "analysis":
        analysis_main(rest)
    elif cmd == "analysis_old":
        analysis_old_main(rest)
    elif cmd in ("calpv", "calval"):
        calpv_main(rest)
    elif cmd == "cval":
        cval_main(rest)
    else:
        die("unknown command %r (available commands: prep, shellsetup, submit, "
            "analysis, analysis_old, calpv, cval)" % cmd)


if __name__ == "__main__":
    cli_parch()