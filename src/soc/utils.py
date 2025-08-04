# utils.py
import numpy as np
import os, csv  
from scipy.linalg import fractional_matrix_power
from scipy.linalg import cholesky, solve_triangular
from . import config

def build_shell_dicts(syms, coords_ang, basis_dict):
    """Creates a list of shell dictionaries from parsed basis data."""
    shells = []
    for atom_idx, (sym, xyz_ang) in enumerate(zip(syms, coords_ang)):
        if sym not in basis_dict: raise KeyError(f"No basis found for element {sym}")
        xyz_bohr = np.asarray(xyz_ang, dtype=float) * config.BOHR_PER_ANG
        for l, exps, coefs in basis_dict[sym]:
            shells.append(dict(sym=sym, atom_idx=atom_idx, l=l, exps=exps, coefs=coefs, center=xyz_bohr))
    return shells

def build_projector_params(syms, coords, soc_tbl):
    """Prepares HGH projector parameters for the C++ overlap calculation."""
    params = []
    for atom_idx, sym in enumerate(syms):
        if sym not in soc_tbl: continue
        center_bohr = coords[atom_idx] * config.BOHR_PER_ANG
        for blk in soc_tbl[sym].get('so', []):
            if not blk.get('k_coeffs'): continue
            for i in range(1, blk['nprj'] + 1):
                params.append({
                    'sym': sym, 'atom_idx': atom_idx, 'l': blk['l'], 'i': i,
                    'r_l': blk['r'], 'center': center_bohr
                })
    return params

def build_L_matrices(lmax=3):
    """Builds the angular momentum matrices Lx, Ly, Lz."""
    L = {}
    for l in range(1, lmax + 1):
        m = np.arange(-l, l + 1)
        Lz = np.diag(m)
        Lp = np.diag(np.sqrt(l * (l + 1) - m[:-1] * (m[:-1] + 1)), 1)
        Lm = np.diag(np.sqrt(l * (l + 1) - m[1:] * (m[1:] - 1)), -1)
        Lx = 0.5 * (Lp + Lm)
        Ly = -0.5j * (Lp - Lm)
        L[l] = {'x': Lx, 'y': Ly, 'z': Lz}
    return L

def unpack_symmetric(k_packed, n):
    """Unpacks a list of upper-triangle elements into a symmetric matrix."""
    K = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = k_packed[idx]
            idx += 1
    return K

def precompute_soc_operators(soc_tbl, L_matrices):
    """Pre-calculates SOC operators K⊗L for each element and angular momentum."""
    k_big_cache = {}
    for sym, pdata in soc_tbl.items():
        if not pdata.get('so'): continue
        k_big_cache[sym] = {}
        for blk in pdata['so']:
            if not blk.get('k_coeffs'): continue
            l, nprj, k_coeffs = blk['l'], blk['nprj'], blk['k_coeffs']
            K_mat = unpack_symmetric(k_coeffs, nprj)
            k_big_cache[sym][l] = {
                comp: np.kron(K_mat, L_mat)
                for comp, L_mat in L_matrices[l].items()
            }
    return k_big_cache

def make_ao_info(shell_dicts_spherical):
    """Creates a list of dictionaries, one for each AO."""
    return [
        {'sym': sh['sym'], 'atom_idx': sh['atom_idx'], 'l': sh['l'], 'm': m}
        for sh in shell_dicts_spherical
        for m in range(-sh['l'], sh['l'] + 1)
    ]

def unique_atoms_syms(syms):
    """Returns a list of unique symbols, preserving order of first appearance."""
    return list(dict.fromkeys(syms))


def ao_population_mulliken(alpha, beta, S):
    """Calculates Mulliken population for a given spinor."""
    alpha = np.atleast_2d(alpha).T if alpha.ndim == 1 else alpha
    beta = np.atleast_2d(beta).T if beta.ndim == 1 else beta
    a = alpha.conj() * (S @ alpha)
    b = beta.conj() * (S @ beta)
    pop = (a + b).real
    pop_sum = pop.sum(axis=0, keepdims=True)
    pop_sum[pop_sum < 1e-16] = 1.0
    return np.squeeze(pop / pop_sum)

def ao_fock_from_mos(S, C, eps):
    """Builds the AO Fock/H0 from MO eigenvectors+eigenvalues."""
    return S @ C @ np.diag(eps) @ C.T @ S

def build_density_matrix(C, occ):
    # C: (n_ao, n_mo), occ: (n_mo,)
    return (C * occ) @ C.T

def build_density_matrix_uks(C_a, occ_a, C_b, occ_b):
    D_a = (C_a * occ_a) @ C_a.T
    D_b = (C_b * occ_b) @ C_b.T
    return D_a + D_b, D_a, D_b

def compute_nuclear_repulsion_from_list(coords_bohr, Z_list):
    """E_nn for pseudopotentials: coords in Bohr, Z_list are ionic valence charges (q)."""
    E_nn = 0.0
    n = len(Z_list)
    for i in range(n):
        Zi = Z_list[i]
        ri = coords_bohr[i]
        for j in range(i):
            Zj = Z_list[j]
            rj = coords_bohr[j]
            Rij = np.linalg.norm(ri - rj)
            if Rij > 1e-12:
                E_nn += Zi * Zj / Rij
    return E_nn

def compute_total_energy(D, F, E_nuc):
    # D, F: AO basis, E_nuc: scalar
    return 0.5 * np.sum(D * (F + F.T)) + E_nuc

def select_mo_subspace_indices(
    eps_Ha,
    occ_vector,
    n_occ_keep,
    n_virt_keep,
    *,
    explicit_idx=None,
    occ_threshold=1e-3,
    respect_degeneracy=True,
    degeneracy_tol_Ha=1e-8,
):
    """
    Build MO subspace indices as: last n_occ_keep occupied (ending at HOMO)
    + first n_virt_keep virtual (starting at LUMO).

    If respect_degeneracy is True, we enlarge the selection to include
    the full degenerate block at the selection boundaries.

    Parameters
    ----------
    eps_Ha : (n_mo,) array
        MO energies in Hartree (canonical MO order).
    occ_vector : (n_mo,) or (n_mo,2)
        Occupations; if 2D we sum across spin.
    n_occ_keep, n_virt_keep : int
        Counts to keep on each side of the Fermi.
    explicit_idx : iterable or None
        If provided, return sorted(unique(explicit_idx)) and ignore counts.
    occ_threshold : float
        Occupation cutoff to decide occupied vs virtual.
    respect_degeneracy : bool
        If True, pad selection at HOMO/LUMO boundaries to include full degenerate blocks.
    degeneracy_tol_Ha : float
        Energy tolerance for degeneracy detection in Hartree.

    Returns
    -------
    idx_sel : (k,) np.ndarray[int]
        Sorted unique indices for the subspace.
    """
    if explicit_idx is not None:
        return np.array(sorted(set(int(i) for i in explicit_idx)), dtype=int)

    eps_Ha = np.asarray(eps_Ha, dtype=float)
    occ = np.asarray(occ_vector)
    if occ.ndim == 2:
        occ = occ.sum(axis=1)
    occ_mask = occ > occ_threshold

    occ_idx = np.where(occ_mask)[0]
    vir_idx = np.where(~occ_mask)[0]

    # Base selection: highest-energy occupieds + lowest-energy virtuals
    sel_occ = occ_idx[-int(n_occ_keep):] if int(n_occ_keep) > 0 else np.array([], dtype=int)
    sel_vir = vir_idx[:int(n_virt_keep)] if int(n_virt_keep) > 0 else np.array([], dtype=int)

    if respect_degeneracy:
        # Pad at occupied boundary (top of selection)
        if sel_occ.size:
            E_top_occ = eps_Ha[sel_occ[-1]]
            same = np.where(np.abs(eps_Ha[occ_idx] - E_top_occ) <= degeneracy_tol_Ha)[0]
            if same.size:
                deg_block = occ_idx[same]  # all with energy ≈ E_top_occ
                sel_occ = np.union1d(sel_occ, deg_block)

        # Pad at virtual boundary (bottom of selection)
        if sel_vir.size:
            E_bot_vir = eps_Ha[sel_vir[0]]
            same = np.where(np.abs(eps_Ha[vir_idx] - E_bot_vir) <= degeneracy_tol_Ha)[0]
            if same.size:
                deg_block = vir_idx[same]
                sel_vir = np.union1d(sel_vir, deg_block)

    idx_sel = np.union1d(sel_occ, sel_vir).astype(int)
    return idx_sel

    ## --- Analysis 3: SOC Spinors in Spin-Free MO Basis ---
    from scipy.linalg import cholesky, solve_triangular

def chol_orthonormalize(C, S_AO):
    """
    Return C_ortho with C_ortho† S_AO C_ortho = I.
    Uses Cholesky of S_mo = C† S C (upper R), solves triangular instead of forming inv(R).
    """
    # S_mo = R† R
    S_mo = C.conj().T @ (S_AO @ C)
    R = cholesky(S_mo, lower=False, check_finite=False, overwrite_a=False)
    # Solve R X = C†  => X = R^{-1} C† ;  then C_ortho = (X)†
    X = solve_triangular(R, C.conj().T, lower=False, trans='N',
                         overwrite_b=False, check_finite=False)
    return X.conj().T

def lowdin_Ssqrt(S, eps=1e-10):
    """Robust symmetric orthogonalization via S^(1/2)."""
    w, U = np.linalg.eigh(S)
    w_clipped = np.clip(w, eps, None)
    return (U * np.sqrt(w_clipped)) @ U.T

def build_lowdin_ops(S, eps=1e-10):
    """
    Precompute eigendecomposition of S for repeated Lowdin uses.
    Returns a dict with apply_shalf(X) and apply_sminushalf(X).
    """
    w, U = np.linalg.eigh(S)
    w_clipped = np.clip(w, eps, None)
    sqrt_w = np.sqrt(w_clipped)
    inv_sqrt_w = 1.0 / sqrt_w

    # For speed: make U Fortran-contiguous for BLAS-friendly gemms
    U = np.asfortranarray(U)

    def _proj(X):       # U^T @ X
        return U.T @ X

    def _lift(Y):       # U @ Y
        return U @ Y

    def apply_shalf(X):
        # U diag(sqrt_w) U^T X  == U * (sqrt_w * (U^T X))
        Y = _proj(X)
        Y *= sqrt_w[:, None]
        return _lift(Y)

    def apply_sminushalf(X):
        Y = _proj(X)
        Y *= inv_sqrt_w[:, None]
        return _lift(Y)

    return {
        "U": U,
        "sqrt_w": sqrt_w,
        "inv_sqrt_w": inv_sqrt_w,
        "apply_shalf": apply_shalf,
        "apply_sminushalf": apply_sminushalf,
    }


def ao_population_lowdin_with_ops(alpha, beta, ops):
    """Löwdin AO population using precomputed apply_shalf (normalized columns)."""
    apply_shalf = ops["apply_shalf"]
    alpha = np.atleast_2d(alpha).T if alpha.ndim == 1 else alpha
    beta  = np.atleast_2d(beta).T  if beta.ndim  == 1 else beta
    a, b  = apply_shalf(alpha), apply_shalf(beta)
    pop   = (np.abs(a)**2 + np.abs(b)**2).real
    pop_sum = pop.sum(axis=0, keepdims=True)
    pop_sum[pop_sum < 1e-16] = 1.0
    return np.squeeze(pop / pop_sum)


def ao_population_lowdin_raw_with_ops(alpha, beta, ops):
    """UN-normalized Löwdin AO population using precomputed apply_shalf."""
    apply_shalf = ops["apply_shalf"]
    alpha = np.atleast_2d(alpha).T if alpha.ndim == 1 else alpha
    beta  = np.atleast_2d(beta).T  if beta.ndim  == 1 else beta
    a, b  = apply_shalf(alpha), apply_shalf(beta)
    pop   = (np.abs(a)**2 + np.abs(b)**2).real
    return np.squeeze(pop)

def _save_spinors_npz_csv(outdir, basename, indices, energies_Ha, occ_vec, U_cols):
    """
    Save selected spinors to:
      - NPZ: complex matrix U (columns = selected spinors) + energies/occ/indices
      - CSV: summary table (idx, energy_Ha, energy_eV, occupation)
    """
    os.makedirs(outdir, exist_ok=True)
    npz_path = os.path.join(outdir, f"{basename}.npz")
    csv_path = os.path.join(outdir, f"{basename}.csv")
    energies_eV = energies_Ha * config.H2EV
        
    # Save arrays (compressed)
    np.savez_compressed(
        npz_path,
        indices=np.asarray(indices, dtype=np.int64),
        energies_Ha=np.asarray(energies_Ha, dtype=np.float64),
        energies_eV=np.asarray(energies_eV, dtype=np.float64),
        occupations=np.asarray(occ_vec, dtype=np.float64),
        U=U_cols.astype(np.complex128),      # shape: (2*n_ao, len(indices))
    )       
            
    # Save a readable summary
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spinor_index", "energy_Ha", "energy_eV", "occupation"])
        for i, EH, EV, occ in zip(indices, energies_Ha, energies_eV, occ_vec):
            w.writerow([int(i), float(EH), float(EV), float(occ)])

    print(f"[WRITE] Spinors: {npz_path}")
    print(f"[WRITE] Summary: {csv_path}")

