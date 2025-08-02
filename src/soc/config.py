# config.py
import numpy as np
import os, sys, re  
import argparse 

## --- Global Constants ---
NTHREADS = os.cpu_count() or 1
H2EV = 27.211386245988
BOHR_PER_ANG = 1.8897259886
valence_electrons = {
    "H": 1, "C": 4, "N": 5, "O": 6,
    "P": 5, "S": 6, "Br": 7, "Cl": 7, "F": 7, "In": 13,
    "Bi": 5, "Pb": 4, "Cs": 9, "Cd": 12, "Zn": 12, 
    "Hg": 12, "Se": 6, "Te":6, "As": 5, "Sb": 5
}

# -----------------------------------------------------------------------------
## --- Runtime SOC Calculation Settings ---
# -----------------------------------------------------------------------------
# Compulsory variables 
XYZ_PATH = "geom.xyz" # Must be specified, no default
BASIS_TXT = "BASIS_MOLOPT" # Must be specified, no default
BASIS_NAME = "DZVP-MOLOPT-SR-GTH" # Must be specified, no default
GTH_POTENTIALS_PATH = "GTH_SOC_POTENTIALS.txt" # Must be specified, no default
# --- Spin treatment ---
UKS = False  # This is the default. set True for unrestricted runs
# RKS
MO_PATH = "MOs.txt" # Must be specified, no default 
# UKS
MO_ALPHA_PATH = "MOs_alpha.txt" # if UKS=True, Must be specified, no default
MO_BETA_PATH  = "MOs_beta.txt" # if UKS=True, Must be specified, no default

# --- Output of SOC eigenvectors ---
WRITE_SPINORS = False                 # off by default
SPINORS_SUBSET = "printed"            # "printed" (= the 'to_print' selection) or "all"
OUTDIR = "."                           # where to write files
SPINORS_BASENAME = "soc_spinors"       # base filename (-> soc_spinors.npz, soc_spinors.csv)

## --- File Paths ---
PERIODIC = False # This is the default 
# PERIODIC=True , then specify lattice . It must be specified as -A1 -A2 -A3 
LATTICE  = np.array([
    [25.128664117,   0.00000000,  0.00000000],
    [-12.564332058, 21.76206150,  0.00000000],
    [ 0.000000000,   0.00000000, 30.61909053],
])

# A list of atomic symbols for which to calculate on-site SOC.
# If this list is empty, all atoms in the system will be treated as active by default.
# Example: soc_active_atoms = ['Cs', 'Bi', 'Br', 'Pb']
soc_active_atoms = [] # This is the default 

# Whether to include two-center SOC integrals between an AO on one atom
# and a projector on another.
calculate_offsite_soc = True #This is the default. It can be hidden

# Energy window in eV for filtering SOC matrix elements around the Fermi level.
# Set to None to disable the filter.
energy_window_eV = 20.0 # This is the default. It can be hidden.

# Number of MOs to print on screen
N_print = 30 # Default but must be specified 

# --- SOC subspace selection around the Fermi level (HOMO/LUMO) ---
# mo_indices = [n_occ, n_virt] means: take the last n_occ occupied + first n_virt virtual MOs
# Set to [0, 0] to disable subspace solve (i.e., do full AO spinor diagonalization).
mo_indices = [1000, 1000]
# Per-spin subspace counts (UKS only): [NOCC, NVIRT]
MO_WINDOW_ALPHA = None   # e.g. [232, 620]
MO_WINDOW_BETA  = None   # e.g. [232, 620]

def _parse_span_to_pair(s: str):
    """
    Accepts formats like '233:619', '233,619', '[233,619]'.
    Returns [i0, i1] as ints (0-based, inclusive).
    """
    if s is None:
        return None
    s = s.strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]
    s = s.replace(' ', '')
    if ':' in s:
        a, b = s.split(':', 1)
    elif ',' in s:
        a, b = s.split(',', 1)
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid span '{s}'. Use i0:i1 or i0,i1 (0-based, inclusive)."
        )
    try:
        i0, i1 = int(a), int(b)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid integers in span '{s}'."
        )
    if i0 < 0 or i1 < 0 or i0 > i1:
        raise argparse.ArgumentTypeError(
            f"Invalid span [{i0}, {i1}]. Must satisfy 0 <= i0 <= i1."
        )
    return [i0, i1]

def _parse_vec3(tokens, name):
    if tokens is None:
        return None
    if len(tokens) != 3:
        raise ValueError(f"{name} must have 3 numbers (got {tokens})")
    return [float(x) for x in tokens]


def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="soc",
        description="Spin–Orbit Coupling driver (SOC). "
                    "CLI overrides values in soc.config before running."
    )

    # --- Required-ish paths (some depend on spin mode) ---
    g_paths = p.add_argument_group("Paths & basis/potentials")
    g_paths.add_argument("--xyz", dest="XYZ_PATH",
                         help="Path to geometry XYZ (default: config.XYZ_PATH)")
    g_paths.add_argument("--basis-txt", dest="BASIS_TXT",
                         help="Path to BASIS_MOLOPT file (default: config.BASIS_TXT)")
    g_paths.add_argument("--basis-name", dest="BASIS_NAME",
                         help="Basis set name, e.g. DZVP-MOLOPT-SR-GTH (default: config.BASIS_NAME)")
    g_paths.add_argument("--gth-potentials", dest="GTH_POTENTIALS_PATH",
                         help="Path to GTH SOC potentials (default: config.GTH_POTENTIALS_PATH)")

    # --- Spin mode & MO files (RKS/UKS) ---
    g_spin = p.add_argument_group("Spin treatment & MOs")
    g_spin.add_argument("--uks", action="store_true", default=None,
                        help="Use UKS spin-unrestricted mode (default: config.UKS=False)")
    # RKS
    g_spin.add_argument("--mo", dest="MO_PATH",
                        help="MOs file for RKS (default: config.MO_PATH)")
    # UKS
    g_spin.add_argument("--mo-alpha", dest="MO_ALPHA_PATH",
                        help="Alpha MOs file for UKS (required if --uks)")
    g_spin.add_argument("--mo-beta", dest="MO_BETA_PATH",
                        help="Beta  MOs file for UKS (required if --uks)")
    
    # Only meaningful for UKS; validated after parse
    g_spin.add_argument("--mo-window-alpha", dest="MO_WINDOW_ALPHA",
                        nargs=2, type=int, metavar=("NOCC","NVIRT"), default=None,
                        help="Alpha subspace counts: last NOCC occ + first NVIRT virt (UKS only).")
    g_spin.add_argument("--mo-window-beta", dest="MO_WINDOW_BETA",
                        nargs=2, type=int, metavar=("NOCC","NVIRT"), default=None,
                        help="Beta subspace counts: last NOCC occ + first NVIRT virt (UKS only).")

    # --- Periodicity & lattice ---
    g_cell = p.add_argument_group("Periodicity & lattice")
    g_cell.add_argument("--periodic", action="store_true", default=None,
                        help="Enable periodic mode (default: config.PERIODIC=False)")
    g_cell.add_argument("--A1", nargs=3, type=float, metavar=("AX", "AY", "AZ"),
                        help="Lattice vector A1 (Å) when --periodic")
    g_cell.add_argument("--A2", nargs=3, type=float, metavar=("BX", "BY", "BZ"),
                        help="Lattice vector A2 (Å) when --periodic")
    g_cell.add_argument("--A3", nargs=3, type=float, metavar=("CX", "CY", "CZ"),
                        help="Lattice vector A3 (Å) when --periodic")

    # --- SOC selection & options ---
    g_soc = p.add_argument_group("SOC options")
    g_soc.add_argument("--soc-active", type=str, default=None,
                       help="Comma-separated atomic symbols to restrict on-site SOC (e.g. 'Cs,Bi,Br,Pb'). "
                            "Empty/omitted means all atoms.")
    g_soc.add_argument("--offsite", dest="calculate_offsite_soc",
                       action=argparse.BooleanOptionalAction, default=None,
                       help="Include two-center SOC AO–projector terms (default: True). "
                            "Use --no-offsite to disable.")
    g_soc.add_argument("--energy-window", dest="energy_window_eV", type=float, default=None,
                       help="Energy window (eV) around Fermi for filtering SOC matrix elements. "
                            "Use 0 or negative to disable.")

    g_out = p.add_argument_group("Spinor output")
    g_out.add_argument("--write-spinors", action=argparse.BooleanOptionalAction, default=None,
                       help="Write SOC eigenvectors/eigenvalues/occupations to NPZ+CSV (default: False). "
                            "Use --no-write-spinors to force off.")
    g_out.add_argument("--spinors-subset", choices=["printed", "all"], default=None,
                       help="Which spinors to write: 'printed' (selection used in analysis) or 'all'.")
    g_out.add_argument("--outdir", dest="OUTDIR", default=None,
                       help="Output directory for spinor files (default: '.').")
    g_out.add_argument("--basename", dest="SPINORS_BASENAME", default=None,
                       help="Base filename without extension (default: 'soc_spinors').")

    # --- Misc runtime knobs ---
    g_misc = p.add_argument_group("Runtime & output")
    g_misc.add_argument("--nthreads", type=int, default=None,
                        help="Override number of threads (default: config.NTHREADS)")
    g_misc.add_argument("--n-print", dest="N_print", type=int, default=None,
                        help="Number of MOs to print (default: config.N_print)")
    g_misc.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v, -vv)")
    g_misc.add_argument("--dry-run", action="store_true",
                        help="Parse and show final settings, then exit")

    g_misc.add_argument("--mo-window", nargs=2, type=int, metavar=("NOCC","NVIRT"),
                        help="SOC subspace window around HOMO: last NOCC occ + first NVIRT virt (default from config.mo_indices)")
    
    p.add_argument("--version", action="version", version="soc 0.1.0")
    return p


def apply_cli_overrides(args: argparse.Namespace):
    """
    Overwrite config.* globals with CLI overrides, validating combinations.
    """
    global XYZ_PATH, BASIS_TXT, BASIS_NAME, GTH_POTENTIALS_PATH
    global UKS, MO_PATH, MO_ALPHA_PATH, MO_BETA_PATH
    global PERIODIC, LATTICE
    global soc_active_atoms, calculate_offsite_soc, energy_window_eV, NTHREADS, N_print
    global WRITE_SPINORS, SPINORS_SUBSET, OUTDIR, SPINORS_BASENAME
    global mo_indices

    # Basic paths / basis
    if args.XYZ_PATH is not None:
        XYZ_PATH = args.XYZ_PATH
    if args.BASIS_TXT is not None:
        BASIS_TXT = args.BASIS_TXT
    if args.BASIS_NAME is not None:
        BASIS_NAME = args.BASIS_NAME
    if args.GTH_POTENTIALS_PATH is not None:
        GTH_POTENTIALS_PATH = args.GTH_POTENTIALS_PATH

    # Spin mode
    if args.uks is not None:
        UKS = bool(args.uks)

    # RKS vs UKS MO files
    if UKS:
        # Prefer explicit CLI, else keep config defaults
        if args.MO_ALPHA_PATH is not None:
            MO_ALPHA_PATH = args.MO_ALPHA_PATH
        if args.MO_BETA_PATH is not None:
            MO_BETA_PATH = args.MO_BETA_PATH
        # Validate
        for label, path in (("MO_ALPHA_PATH", MO_ALPHA_PATH),
                            ("MO_BETA_PATH",  MO_BETA_PATH)):
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"--uks requires valid {label} (got '{path}')")
    else:
        if args.MO_PATH is not None:
            MO_PATH = args.MO_PATH
        if not MO_PATH or not os.path.exists(MO_PATH):
            raise FileNotFoundError(f"RKS mode requires valid --mo (got '{config.MO_PATH}')")

    # Periodicity & lattice
    if args.periodic is not None:
        PERIODIC = bool(args.periodic)
    if PERIODIC:
        # If user provided lattice vectors, overwrite
        A1 = _parse_vec3(args.A1, "A1") if args.A1 is not None else None
        A2 = _parse_vec3(args.A2, "A2") if args.A2 is not None else None
        A3 = _parse_vec3(args.A3, "A3") if args.A3 is not None else None
        if any(v is None for v in (A1, A2, A3)):
            # keep existing config.LATTICE if fully specified there
            if not (isinstance(LATTICE, np.ndarray) and LATTICE.shape == (3, 3)):
                raise ValueError("Periodic mode enabled but lattice not fully specified "
                                 "(provide --A1 --A2 --A3 or set config.LATTICE).")
        else:
            LATTICE = np.array([A1, A2, A3], dtype=float)

    # SOC selection & options
    if args.soc_active is not None:
        # empty string -> [] meaning "all atoms"
        cleaned = [s.strip() for s in args.soc_active.split(",") if s.strip()] if args.soc_active else []
        soc_active_atoms = cleaned

    if args.calculate_offsite_soc is not None:
        calculate_offsite_soc = bool(args.calculate_offsite_soc)

    if args.energy_window_eV is not None:
        ew = float(args.energy_window_eV)
        energy_window_eV = None if ew <= 0.0 else ew

    if args.write_spinors is not None:
        WRITE_SPINORS = bool(args.write_spinors)
    if args.SPINORS_BASENAME is not None:
        SPINORS_BASENAME = args.SPINORS_BASENAME
    if args.OUTDIR is not None:
        OUTDIR = args.OUTDIR
    if args.spinors_subset is not None:
        SPINORS_SUBSET = args.spinors_subset

    if args.mo_window is not None:
        mo_indices = [max(0, int(args.mo_window[0])), max(0, int(args.mo_window[1]))]
    
    # Runtime
    if args.nthreads is not None:
        NTHREADS = max(1, int(args.nthreads))
    if args.N_print is not None:
        N_print = max(1, int(args.N_print))

    # ---- Per-spin subspace counts (UKS only) ----
    global MO_WINDOW_ALPHA, MO_WINDOW_BETA

    if args.MO_WINDOW_ALPHA is not None or args.MO_WINDOW_BETA is not None:
        if not UKS:
            raise ValueError("The options --mo-window-alpha/--mo-window-beta require --uks.")
        if args.MO_WINDOW_ALPHA is not None:
            a_occ, a_vir = map(int, args.MO_WINDOW_ALPHA)
            MO_WINDOW_ALPHA = [max(0, a_occ), max(0, a_vir)]
        if args.MO_WINDOW_BETA is not None:
            b_occ, b_vir = map(int, args.MO_WINDOW_BETA)
            MO_WINDOW_BETA = [max(0, b_occ), max(0, b_vir)]
        print(f"[CONFIG] UKS per-spin subspace counts: "
              f"alpha={MO_WINDOW_ALPHA}, beta={MO_WINDOW_BETA}")
    else:
        MO_WINDOW_ALPHA = None
        MO_WINDOW_BETA  = None


def parse_and_apply_cli(argv=None):
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # Fill in defaults from config first, then overwrite
    apply_cli_overrides(args)

    # Optional echo of final settings
    if args.verbose:
        import pprint
        settings = {
            "XYZ_PATH": config.XYZ_PATH,
            "BASIS_TXT": config.BASIS_TXT,
            "BASIS_NAME": config.BASIS_NAME,
            "GTH_POTENTIALS_PATH": config.GTH_POTENTIALS_PATH,
            "UKS": config.UKS,
            "MO_PATH": getattr(config, "MO_PATH", None),
            "MO_ALPHA_PATH": getattr(config, "MO_ALPHA_PATH", None),
            "MO_BETA_PATH": getattr(config, "MO_BETA_PATH", None),
            "PERIODIC": config.PERIODIC,
            "LATTICE": config.LATTICE.tolist() if isinstance(config.LATTICE, np.ndarray) else config.LATTICE,
            "soc_active_atoms": config.soc_active_atoms,
            "calculate_offsite_soc": config.calculate_offsite_soc,
            "energy_window_eV": config.energy_window_eV,
            "NTHREADS": config.NTHREADS,
            "N_print": config.N_print,
            "WRITE_SPINORS": WRITE_SPINORS,
            "SPINORS_SUBSET": SPINORS_SUBSET,
            "OUTDIR": OUTDIR,
            "SPINORS_BASENAME": SPINORS_BASENAME,
        }
        print("[SOC] Resolved settings:")
        pprint.pprint(settings)
    return args



