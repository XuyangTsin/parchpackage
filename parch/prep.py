#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parch prep -- extract selected molecules from an equilibrated system.

Preparation step *before* ``parch shellsetup``: from an equilibrated GROMACS
system (``md.gro`` + ``md.tpr`` + ``.top``) it lists every molecule type / chain,
lets you choose which ones to keep, then writes a trimmed structure (``-o``) and
a matching topology (``-po``) containing only the kept molecules. You are also
asked which force-field/``#include`` lines to keep, and optionally walked through
building a ``shellsetup -shelldef`` file (per-molecule shell thicknesses).

The molecule -> atom mapping (which the .gro alone does not carry) is taken from
the .tpr via MDAnalysis; the coordinates are taken from the .gro.

    parch prep -f md.gro -s md.tpr -pi topol.top -o solute.gro -po solute.top

Selection is interactive (keyboard) by default; use -keep / -keepff to run
non-interactively, e.g.  -keep "1 2 4"  -keepff "1 2 3".
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import MDAnalysis as mda

from .shellsetup import die


# PARCH's own counter-ion .itp files (shipped in the package), added to the
# output topology so the shell-setup step's NA/CL counterions are defined.
PACKAGE_DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backup_annealing")
ION_ITPS = {"charmm": ["CLA.itp", "SOD.itp"],
            "amber":  ["Cl-_AMBER.itp", "Na+_AMBER.itp"]}


# =========================================================================== #
# Selection parsing
# =========================================================================== #
def parse_index_selection(text, n_max):
    """Parse '1 2 4', '1-3,5', etc. into a sorted unique list of ints in 1..n_max."""
    out = set()
    for tok in text.replace(",", " ").split():
        if "-" in tok[1:]:                       # a range like 2-5 (not a lone '-')
            a, b = tok.split("-", 1)
            try:
                a, b = int(a), int(b)
            except ValueError:
                die("invalid selection token %r" % tok)
            if a > b:
                a, b = b, a
            out.update(range(a, b + 1))
        else:
            try:
                out.add(int(tok))
            except ValueError:
                die("invalid selection token %r" % tok)
    bad = [i for i in out if i < 1 or i > n_max]
    if bad:
        die("selection out of range 1..%d: %s" % (n_max, ", ".join(map(str, sorted(bad)))))
    if not out:
        die("empty selection.")
    return sorted(out)


def ask_selection(prompt, n_max):
    """Prompt for an index selection on the terminal; abort cleanly on EOF."""
    try:
        text = input(prompt)
    except EOFError:
        die("no selection provided (running non-interactively?). Use -keep / -keepff.")
    return parse_index_selection(text, n_max)


def ask_ionff():
    """Prompt for the counter-ion force field (charmm/amber); abort on EOF."""
    while True:
        try:
            ans = input("Which forcefield are you using? (charmm/amber): ").strip().lower()
        except EOFError:
            die("no forcefield provided (running non-interactively?). Use -ionff charmm|amber.")
        if ans in ION_ITPS:
            return ans
        print("Please type 'charmm' or 'amber'.")


def ask_yesno(prompt):
    """Prompt for a yes/no answer; loop on invalid input, abort on EOF."""
    while True:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            die("no yes/no answer provided (running non-interactively?). Use -separateshell yes|no.")
        if ans in ("yes", "y"):
            return True
        if ans in ("no", "n"):
            return False
        print("Please type 'yes' or 'no'.")


def ask_group_indices(prompt, n_max):
    """Prompt for an optional index selection; Enter (blank) means 'none'."""
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            die("no selection provided (running non-interactively?).")
        if not raw:
            return []
        return parse_index_selection(raw, n_max)


def ask_thickness_value(prompt):
    """Prompt for a positive shell thickness (Angstrom); loop until valid."""
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            die("no thickness provided (running non-interactively?).")
        try:
            v = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if v <= 0:
            print("Thickness must be positive.")
            continue
        return v


# =========================================================================== #
# Molecule grouping
#
# Each [ molecules ] LINE of the topology is one selectable group (1:1), so a
# moleculetype appearing on several lines gives several independent groups and
# is edited positionally -- keeping one line never drags in the others. The
# .tpr supplies atoms-per-moleculetype; the [ molecules ] order is the system
# (atom/coordinate) order, so each line maps to a contiguous atom range.
# =========================================================================== #
def parse_molecules_block(top_lines):
    """Return [(file_line_index, name, count), ...] for each [ molecules ] entry."""
    mols, in_mol = [], False
    pat = re.compile(r"^\s*(\S+)\s+(\d+)\s*$")
    for i, l in enumerate(top_lines):
        if re.match(r"\s*\[\s*molecules\s*\]", l, re.IGNORECASE):
            in_mol = True
            continue
        if in_mol:
            s = l.strip()
            if s.startswith("["):                 # next section ends [molecules]
                break
            if not s or s.startswith(";"):
                continue
            m = pat.match(l)
            if m:
                mols.append((i, m.group(1), int(m.group(2))))
    return mols


def moltype_info(universe):
    """{moltype: (atoms_per_molecule, n_molecules)} from the .tpr."""
    if not hasattr(universe.atoms, "moltypes"):
        die("the .tpr did not provide molecule types; is -s a GROMACS .tpr?")
    mt = np.asarray(universe.atoms.moltypes)
    mn = np.asarray(universe.atoms.molnums)
    info = {}
    for name in np.unique(mt):
        mask = mt == name
        n_atoms = int(mask.sum())
        n_mol = int(len(np.unique(mn[mask])))
        info[str(name)] = (n_atoms // n_mol if n_mol else 0, n_mol)
    return info


def build_groups(top_mols, info, universe):
    """Map each [ molecules ] line to a contiguous atom range (system order),
    validating that the topology and the .tpr/.gro correspond."""
    resindices = np.asarray(universe.atoms.resindices)
    resids = np.asarray(universe.atoms.resids)
    total = len(universe.atoms)

    # counts per moleculetype in the .top must match the .tpr
    top_counts = {}
    for _, name, count in top_mols:
        top_counts[name] = top_counts.get(name, 0) + count
    for name, c in top_counts.items():
        if name not in info:
            die("[molecules] entry %r is not in the .tpr; do the .top and .tpr match?" % name)
        if c != info[name][1]:
            die("molecule count mismatch for %r: .top has %d, .tpr has %d."
                % (name, c, info[name][1]))

    groups, cursor = [], 0
    for line_idx, name, count in top_mols:
        n_atoms = count * info[name][0]
        s, e = cursor, cursor + n_atoms
        groups.append({"line_idx": line_idx, "name": name, "count": count,
                       "start": s, "end": e,
                       "n_res": int(len(np.unique(resindices[s:e]))),
                       "res_first": int(resids[s:e].min()) if e > s else 0,
                       "res_last": int(resids[s:e].max()) if e > s else 0})
        cursor = e
    if cursor != total:
        die("[molecules] expands to %d atoms but the structure has %d; do the "
            ".gro/.tpr/.top correspond?" % (cursor, total))
    return groups


def print_group_table(groups):
    print("\n#group  molecule        num_mol  atoms")
    for i, g in enumerate(groups, 1):
        print("%-6d  %-14s  %-7d  %d" % (i, g["name"], g["count"], g["end"] - g["start"]))
    print()


# =========================================================================== #
# Topology editing
# =========================================================================== #
def collect_includes(lines):
    """Return [(line_index, path_text)] for every #include line, in order."""
    inc = []
    for i, l in enumerate(lines):
        m = re.match(r'\s*#include\s+"?([^"\n]+)"?', l)
        if m:
            inc.append((i, m.group(1).strip()))
    return inc


def resolve_itp(path, top_dir):
    """Absolute path of an include if the file exists (resolved against top_dir,
    the current directory, then each $GMXLIB entry), else None."""
    if os.path.isabs(path):
        cands = [path]
    else:
        cands = [os.path.join(top_dir, path), os.path.join(os.getcwd(), path)]
        for d in os.environ.get("GMXLIB", "").split(os.pathsep):
            if d:
                cands.append(os.path.join(d, path))
    for c in cands:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def write_topology(src_lines, dst_top, keep_mol_line_idx, keep_include_idx,
                   include_rewrite=None, extra_includes=None):
    """Write dst_top keeping only the requested #include lines and the EXACT
    [ molecules ] data lines in keep_mol_line_idx (by file-line index, so a
    repeated moleculetype name keeps only the selected line). Everything else
    (comments, [ system ], counts) is preserved verbatim in original order.
    include_rewrite maps a kept include's line index to a replacement line
    (e.g. the same include rewritten with a full path). extra_includes (paths)
    are inserted as #include lines just before the [ system ] / [ molecules ]
    section."""
    include_rewrite = include_rewrite or {}
    mol_pat = re.compile(r"^\s*(\S+)\s+\d+\s*$")
    out, in_molecules = [], False
    for i, l in enumerate(src_lines):
        stripped = l.strip()

        if re.match(r"\s*#include\b", l):
            if i in keep_include_idx:
                out.append(include_rewrite.get(i, l))
            continue

        if re.match(r"\s*\[\s*molecules\s*\]", l, re.IGNORECASE):
            in_molecules = True
            out.append(l)
            continue

        if in_molecules:
            if stripped.startswith("["):          # a new section ends [molecules]
                in_molecules = False
                out.append(l)
                continue
            if stripped == "" or stripped.startswith(";"):
                out.append(l)                     # blank / comment -> keep
                continue
            if mol_pat.match(l):
                if i in keep_mol_line_idx:
                    out.append(l)                 # this specific entry was selected
                continue
            out.append(l)                         # anything unexpected -> keep
            continue

        out.append(l)

    if extra_includes:
        inc_lines = ['#include "%s"\n' % p for p in extra_includes]
        ins = next((i for i, l in enumerate(out)
                    if re.match(r"\s*\[\s*(system|molecules)\s*\]", l, re.IGNORECASE)),
                   len(out))
        out = out[:ins] + ["\n; PARCH counter-ion topologies\n"] + inc_lines + ["\n"] + out[ins:]

    with open(dst_top, "w") as fh:
        fh.writelines(out)


def write_shelldef(path, groups):
    """Write a shellsetup -shelldef file: one 'thickness  start:end' line per
    (start, end, thickness) group, e.g.:
        # shell_thickness, resid range
        4.15  1:298
        4.80  401:450
    """
    with open(path, "w") as fh:
        fh.write("# shell_thickness, resid range\n")
        for a, b, t in groups:
            fh.write("%-6.2f%d:%d\n" % (t, a, b))


# =========================================================================== #
# Argument parsing
# =========================================================================== #
def build_parser():
    p = argparse.ArgumentParser(
        prog="parch_prep",
        description="Extract selected molecules from an equilibrated system "
                    "(structure + topology) for use with parch shellsetup.",
        allow_abbrev=False,
    )
    p.add_argument("-f", dest="structure", required=True, help="Input equilibrated structure (md.gro).")
    p.add_argument("-s", dest="tpr", required=True, help="Input run input file (md.tpr) for molecule typing.")
    p.add_argument("-pi", dest="topology", required=True, help="Input topology (.top).")
    p.add_argument("-o", dest="out_gro", required=True, help="Output structure (.gro) with kept molecules.")
    p.add_argument("-po", dest="out_top", required=True, help="Output topology (.top) with kept molecules.")
    p.add_argument("-keep", help="Non-interactive: molecule group numbers to keep, e.g. '1 2 4' or '1-3,5'.")
    p.add_argument("-keepff", help="Non-interactive: #include line numbers to keep "
                                   "(default when -keep is given: keep all includes).")
    p.add_argument("-renumber", choices=["yes", "no"], default="yes",
                   help="Renumber kept residues sequentially from 1 in the output "
                        "structure (default: yes), which is REQUIRED for parch shellsetup.")
    p.add_argument("-ionff", choices=sorted(ION_ITPS),
                   help="Force field for the PARCH counter-ion .itp includes added to the "
                        "output topology (charmm -> CLA/SOD, amber -> Cl-/Na+). "
                        "If omitted, you are prompted.")
    p.add_argument("-separateshell", choices=["yes", "no"],
                   help="Whether to interactively build a shellsetup -shelldef file "
                        "(per-molecule shell thicknesses). If omitted, you are prompted.")
    p.add_argument("-shelldefout", default="shelldef.txt",
                   help="Output shell-definition file written when -separateshell yes "
                        "(default: %(default)s).")
    return p


# =========================================================================== #
# Main
# =========================================================================== #
def main(argv=None):
    args = build_parser().parse_args(argv)
    for path, label in ((args.structure, "-f"), (args.tpr, "-s"), (args.topology, "-pi")):
        if not os.path.isfile(path):
            die("%s file not found: %s" % (label, path))

    # ---- molecule groups: one per [ molecules ] line, mapped to atom ranges  #
    u = mda.Universe(args.tpr, args.structure)
    if len(u.atoms) == 0:
        die("no atoms read from %s." % args.structure)
    with open(args.topology) as fh:
        top_lines = fh.readlines()
    top_mols = parse_molecules_block(top_lines)
    if not top_mols:
        die("no [ molecules ] entries found in %s." % args.topology)
    groups = build_groups(top_mols, moltype_info(u), u)
    print_group_table(groups)

    # ---- choose molecules to keep ----------------------------------------- #
    print("Please select the molecules you would like to KEEP for the PARCH process;\n"
          "the extraction will be applied on the .gro and .top files.\n"
          "!! DO NOT KEEP waters or ions from your equilibration structure (e.g., SOL, TP3, OPC, NA, CL, K, CA).\n"
          "")
    if args.keep:
        keep_idx = parse_index_selection(args.keep, len(groups))
    else:
        keep_idx = ask_selection("Enter molecule group numbers to KEEP (e.g. 1 2 4 5-10): ", len(groups))
    keep_groups = [groups[i - 1] for i in keep_idx]
    keep_mol_line_idx = set(g["line_idx"] for g in keep_groups)   # positional, exact lines
    print("Keeping: %s" % ", ".join("%s(%d)" % (g["name"], g["count"]) for g in keep_groups))

    # ---- extract atoms (system order) into the output structure ----------- #
    atom_idx = np.concatenate([np.arange(g["start"], g["end"]) for g in keep_groups])
    atom_idx.sort()
    kept = u.atoms[atom_idx]

    # ---- make molecules whole across PBC and centre them in the box ------- #
    # Required for parch shellsetup: a solute split across the periodic boundary
    # would be carved/boxed incorrectly. unwrap() uses bonds from the .tpr.
    try:
        kept.unwrap(compound="fragments")
    except Exception as exc:
        die("could not make molecules whole across PBC (%s); the .tpr must "
            "provide bond information." % exc)
    box = u.dimensions
    if box is not None and np.all(box[:3] > 0):
        kept.translate(box[:3] / 2.0 - kept.center_of_geometry())

    # ---- renumber kept residues sequentially from 1 (default) ------------- #
    # Build the output resid range of each kept group (renumbered or original).
    res_map, cursor = [], 1
    for g in keep_groups:
        if args.renumber == "yes":
            res_map.append((g["name"], cursor, cursor + g["n_res"] - 1))
        else:
            res_map.append((g["name"], g["res_first"], g["res_last"]))
        cursor += g["n_res"]
    if args.renumber == "yes":
        kept.residues.resids = np.arange(1, len(kept.residues) + 1)

    kept.write(args.out_gro)
    print("Wrote %s  (%d atoms, %d residues%s)"
          % (args.out_gro, len(kept), len(kept.residues),
             ", renumbered from 1" if args.renumber == "yes" else ""))

    # ---- output residue map (use these ranges for -shelldef later) -------- #
    print("\nNOTICE: Output residue ranges%s -- useful for -separateshell or -shelldef:"
          % (" (renumbered)" if args.renumber == "yes" else " (original)"))
    for name, a, b in res_map:
        print("    %-14s %d:%d" % (name, a, b))
    time.sleep(2)

    # ---- choose force-field / #include lines to keep ---------------------- #
    includes = collect_includes(top_lines)
    keep_inc_line_idx = set(i for i, _ in includes)          # default: keep all
    if includes:
        print("\nThe following itp files are in your .top:")
        for n, (_, path) in enumerate(includes, 1):
            print("%-4d  %s" % (n, path))
        time.sleep(1)
        print("\n!! Please READ the notice and SELECT the .itp files to KEEP for PARCH (e.g. 1 2 4 5-10):\n"
              "Which should #include the following compents:\n"
              "!!   i.   the total forcefield file (e.g. forcefield.itp)\n"
              "!!   ii.  itps corresponding to molecules you kept (e.g. PROA.itp, PROB.itp, DNAA.itp)\n"
              "!!   iii. the water's itp (e.g. SOL.itp)\n"
              "NOTICE: If you plan to use a DIFFERENT water model than the one used during equilibration, "
              "DO NOT keep the  water .itp file.\n"
              "You will need to MANUALLY include the new water .itp AFTER the output .top is generated.\n"
              "NOTICE: DO NOT KEEP the .itp files for IONS used during equilibration, As PARCH uses its own tailored ion itp.\n"
              "NOTICE: Keeping the original ion .itp files may result in duplicate and conflicting topology entries.\n")
        if args.keep and args.keepff is None:
            print("(keeping all includes; pass -keepff to choose)")
        else:
            if args.keepff is not None:
                sel = parse_index_selection(args.keepff, len(includes))
            else:
                try:
                    raw = input("\nEnter numbers to KEEP for #include (Enter = keep all itps): ").strip()
                except EOFError:
                    raw = ""
                sel = parse_index_selection(raw, len(includes)) if raw else list(range(1, len(includes) + 1))
            keep_inc_line_idx = set(includes[n - 1][0] for n in sel)

    # ---- rewrite kept includes to absolute paths; error if any is missing - #
    top_dir = os.path.dirname(os.path.abspath(args.topology))
    include_rewrite, unresolved = {}, []
    for i, path in includes:
        if i not in keep_inc_line_idx:
            continue
        full = resolve_itp(path, top_dir)
        if full:
            include_rewrite[i] = '#include "%s"\n' % full
        else:
            unresolved.append(path)
    if unresolved:
        die("the following kept .itp include(s) were not found:\n    %s\n"
            "Please provide the correct path for these .itp files (edit the "
            "topology's #include lines,\n or run from the directory that contains them)."
            % "\n    ".join(unresolved))

    # ---- add PARCH's counter-ion .itp includes ---------------------------- #
    ionff = args.ionff if args.ionff else ask_ionff()
    extra_includes = []
    for fn in ION_ITPS[ionff]:
        p = os.path.join(PACKAGE_DATADIR, fn)
        if not os.path.isfile(p):
            die("counter-ion itp not found: %s" % p)
        extra_includes.append(p)

    # ---- write trimmed topology ------------------------------------------- #
    write_topology(top_lines, args.out_top, keep_mol_line_idx, keep_inc_line_idx,
                   include_rewrite=include_rewrite, extra_includes=extra_includes)
    kept_desc = ", ".join("%s(%d)" % (g["name"], g["count"]) for g in keep_groups)
    print("Wrote %s  (kept [molecules]: %s)" % (args.out_top, kept_desc))
    print("The counter ions' itp were added for %s forcefield." % ionff)

    # ---- optional: build a shellsetup -shelldef file ----------------------- #
    if args.separateshell is not None:
        want_shelldef = args.separateshell == "yes"
    else:
        want_shelldef = ask_yesno(
            "\nWhether you need to define different shell thickness among your molecules "
            "(e.g. parch shellsetup -separateshell yes)? It will help you prepare the file "
            "for -shelldef %s.\nType yes/no: " % args.shelldefout)

    if want_shelldef:
        print("\nKept molecule groups (index  name  resid_range):")
        for i, (name, a, b) in enumerate(res_map, 1):
            print("%-3d %-14s %d:%d" % (i, name, a, b))

        n_groups = len(res_map)
        thickness = {}

        protein_sel = ask_group_indices(
            "\nSelect the PROTEIN molecules (shell thickness of 4.15 A)? ", n_groups)
        for i in protein_sel:
            thickness[i] = 4.15

        dna_sel = ask_group_indices(
            "Select the DNA/RNA molecules? (shell thickness of 4.8 A) ", n_groups)
        for i in dna_sel:
            thickness[i] = 4.8

        first = True
        while True:
            prompt = ("Select the molecules you want to specify shell thickness besides above: "
                      if first else
                      "Select the molecules you want to specify except above types: ")
            sel = ask_group_indices(prompt, n_groups)
            first = False
            if not sel:
                break
            t = ask_thickness_value("Please specify the shell thickness: ")
            for i in sel:
                thickness[i] = t

        shelldef_groups = [(res_map[i - 1][1], res_map[i - 1][2], thickness[i])
                           for i in sorted(thickness)]
        if shelldef_groups:
            write_shelldef(args.shelldefout, shelldef_groups)
            print("Wrote %s  (%d shell group(s)) -- use with 'parch shellsetup -separateshell "
                  "yes -shelldef %s'." % (args.shelldefout, len(shelldef_groups), args.shelldefout))
        else:
            print("No shell thickness specified for any group; %s was not written."
                  % args.shelldefout)

    # ---- net charge of the kept molecules (from the .tpr charges) ---------- #
    if hasattr(kept, "charges"):
        net_q = float(np.sum(kept.charges))
        print("\nNOTICE: net charge of the kept molecules = %+.2f e  ->  use "
              "'-netcharge %d' in parch shellsetup." % (net_q, round(net_q)))
        time.sleep(1)

    print("\nDone. But check before running parch shellsetup on %s / %s.\n"
          "!! 1. that the kept residues are sequentially numbered\n"
          "!! 2. visually, the moelculs should be centered in the box, and unbroken\n" % (args.out_gro, args.out_top))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "prep":
        argv = argv[1:]
    main(argv)
