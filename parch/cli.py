#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Console entry points for the ``parch`` package.

Thin wrappers around the standalone analysis modules so that those modules stay
plain scripts. Installed (via pyproject) as the ``parch``, ``parch_shell`` and
``parch_analysis`` commands.
"""

import sys

from .shellsetup import main, die
from .analysis import main as analysis_main
from .calpv import main as calpv_main
from .prep import main as prep_main
from .submit import main as submit_main


def cli_prep(argv=None):
    """Console entry point for ``parch_prep``.

    Accepts an optional leading ``prep`` token.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "prep":
        argv = argv[1:]
    prep_main(argv)


def cli_submit(argv=None):
    """Console entry point for ``parch_submit``.

    Accepts an optional leading ``submit`` token.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "submit":
        argv = argv[1:]
    submit_main(argv)


def cli_shell(argv=None):
    """Console entry point for ``parch_shell``.

    Accepts an optional leading ``shellsetup`` token so that both
    ``parch_shell -f ...`` and ``parch_shell shellsetup -f ...`` work.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "shellsetup":
        argv = argv[1:]
    main(argv)


def cli_analysis(argv=None):
    """Console entry point for ``parch_analysis``.

    Accepts an optional leading ``analysis`` token so that both
    ``parch_analysis -path ...`` and ``parch_analysis analysis -path ...`` work.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "analysis":
        argv = argv[1:]
    analysis_main(argv)


def cli_calpv(argv=None):
    """Console entry point for ``parch_calpv``.

    Accepts an optional leading ``calpv``/``calval`` token.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("calpv", "calval"):
        argv = argv[1:]
    calpv_main(argv)


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
              "  analysis     Water-shell hydration analysis over the annealing ramp.\n"
              "  calpv        PARCH values from analysis results, averaged over mid_* runs.\n\n"
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
    elif cmd in ("calpv", "calval"):
        calpv_main(rest)
    else:
        die("unknown command %r (available commands: prep, shellsetup, submit, analysis, calpv)" % cmd)


if __name__ == "__main__":
    cli_shell()
