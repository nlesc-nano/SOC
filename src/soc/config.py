# config.py
import numpy as np
import os, sys 
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

    # Runtime
    if args.nthreads is not None:
        NTHREADS = max(1, int(args.nthreads))
    if args.N_print is not None:
        N_print = max(1, int(args.N_print))


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
        }
        print("[SOC] Resolved settings:")
        pprint.pprint(settings)
    return args



