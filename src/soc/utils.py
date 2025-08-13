# utils.py
import numpy as np
import os, csv  
from scipy.linalg import fractional_matrix_power
from scipy.linalg import cholesky, solve_triangular
from . import config
from typing import Tuple
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    sp = None
    spla = None


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

import numpy as np
from typing import Tuple
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    sp = None
    spla = None

# --- Sparse Chebyshev Lowdin ops (apply-only S^{±1/2}) ---
def _ensure_csr(S, drop_tol: float = 0.0):
    """Return a CSR hermitianized copy (drop tiny entries if asked)."""
    if sp is None:
        raise RuntimeError("SciPy sparse required for Chebyshev Lowdin ops.")
    if sp.issparse(S):
        S = S.tocsr(copy=True)
    else:
        S = sp.csr_matrix(np.asarray(S))
    # Hermitianize (safe for small asymmetry)
    S = (S + S.T.conjugate()) * 0.5
    if drop_tol > 0.0:
        S.data[np.abs(S.data) < drop_tol] = 0.0
        S.eliminate_zeros()
    return S

def _spectrum_bounds_spd(S_csr, eps: float = 1e-10) -> Tuple[float, float]:
    """Estimate min/max eigenvalues of SPD S (fast)."""
    # Lower bound
    lam_min = float(spla.eigsh(S_csr, k=1, which="SA", return_eigenvectors=False, tol=1e-6)[0])
    # Upper bound
    lam_max = float(spla.eigsh(S_csr, k=1, which="LA", return_eigenvectors=False, tol=1e-6)[0])
    # Clip away near-zero pathologies
    lam_min = max(lam_min, eps)
    lam_max = max(lam_max, lam_min * (1.0 + 1e-12))
    return lam_min, lam_max

def _cheb_coeffs_for_interval(f, a: float, b: float, deg: int) -> np.ndarray:
    """
    Chebyshev coefficients c_k (k=0..deg) approximating f on [a,b].
    We approximate g(t)=f((b+a)/2 + (b-a)/2 * t) on t∈[-1,1] using m=deg+1 nodes.
    """
    m = deg + 1
    theta = (np.arange(m) + 0.5) * (np.pi / m)
    t = np.cos(theta)                          # Chebyshev nodes in (-1,1)
    x = 0.5 * (b + a) + 0.5 * (b - a) * t     # map back to [a,b]
    fx = f(x)                                  # shape (m,)
    # Discrete orthogonality for Chebyshev T_k:
    # c_k = 2/m * sum_{j=0}^{m-1} f(x_j) cos(k * (j+0.5)π/m), with c_0 carrying the same formula.
    coeffs = np.empty(m, dtype=np.float64)
    for k in range(m):
        coeffs[k] = (2.0 / m) * np.sum(fx * np.cos(k * theta))
    return coeffs  # length = deg+1, interpret as: g(t) ≈ 0.5*c0 + Σ_{k=1}^deg c_k T_k(t)

def _clenshaw_apply_T_series(S_op, X, coeffs: np.ndarray, shift: float, scale: float):
    """
    Apply Σ c_k T_k(T) to block X using Clenshaw with operator T = (S - shift*I)/scale.
    S_op: callable (dense_or_sparse_matmul)
    """
    # We implement: S(T) = 0.5*c0*I + Σ_{k=1}^n c_k T_k(T)
    # Clenshaw for Chebyshev:
    #   B_{n+1} = 0, B_n = 0
    #   for k=n..1: B_k = 2 T B_{k+1} - B_{k+2} + c_k X
    #   result = 0.5*c0*X + T B_1 - B_2
    n = len(coeffs) - 1
    if n < 0:
        return np.zeros_like(X)
    c0 = coeffs[0]
    # define T @ Y operator via shift/scale
    def T_of(Y):
        return (S_op(Y) - shift * Y) / scale

    Bkp2 = np.zeros_like(X)
    Bkp1 = np.zeros_like(X)
    for k in range(n, 0, -1):
        Bk = 2.0 * T_of(Bkp1) - Bkp2 + coeffs[k] * X
        Bkp2, Bkp1 = Bkp1, Bk
    return 0.5 * c0 * X + T_of(Bkp1) - Bkp2

import numpy as np
import scipy.sparse as sp

def build_lowdin_ops_sparse_cheby(S, deg=48, which="sqrt", use_jacobi=True, eps=0.0):
    """
    Returns dict with:
      - apply_shalf(X)      -> S^{1/2} X
      - apply_sminushalf(X) -> S^{-1/2} X
      - metadata: deg, lam_min, lam_max
    Works for S dense (ndarray) or sparse (csr/csc/coo).
    """
    # --- Normalize input to CSR or dense array ---
    is_sparse = sp.issparse(S)
    if is_sparse:
        S_csr = S.tocsr(copy=False)
        diagS = S_csr.diagonal()
    else:
        S = np.asarray(S, dtype=np.float64, order="C")
        S_csr = None
        diagS = np.diag(S)

    if use_jacobi:
        d = np.sqrt(np.clip(diagS, 1e-14, None))
        D = d
        Dinv = 1.0 / D
    else:
        D = np.ones_like(diagS)
        Dinv = D

    # --- Linear operator J(X) = D^{-1/2} S D^{-1/2} X (block apply) ---
    if is_sparse:
        def J_mv(X):
            # X: (N, k)
            Y = S_csr @ (Dinv[:, None] * X)
            return Dinv[:, None] * Y
        # Gershgorin bounds on J (cheap)
        row_abs = np.abs(S_csr).sum(axis=1).A.ravel()
        center  = diagS * (Dinv**2)
        radius  = (row_abs - np.abs(diagS)) * (Dinv**2)
    else:
        def J_mv(X):
            Y = S @ (Dinv[:, None] * X)
            return Dinv[:, None] * Y
        # Gershgorin bounds on J for dense
        row_abs = np.sum(np.abs(S), axis=1)
        center  = diagS * (Dinv**2)
        radius  = (row_abs - np.abs(diagS)) * (Dinv**2)
    
    if eps and eps > 0.0:
        center = center + eps * (Dinv**2)

    lam_min = float(max(1e-12, np.min(center - radius)))
    lam_max = float(np.max(center + radius))

    # --- Chebyshev coefficients for f(λ)=sqrt(λ) or 1/sqrt(λ) on [lam_min, lam_max] ---
    def _cheb_coeffs_sqrt(M, m):
        # Map to [-1,1]; coefficients via simple cosine quadrature
        j = np.arange(0, deg+1)
        # We’ll just use Clenshaw evaluation; no need to expose coeffs, but keep for clarity
        return None  # not needed explicitly with Clenshaw form below

    # Clenshaw evaluator for f(J)X; we embed f via scalar function handle
    a = (lam_max + lam_min) * 0.5
    b = (lam_max - lam_min) * 0.5

    def cheb_apply(f_handle, X):
        # Evaluate f(J)X with degree 'deg' Chebyshev expansion using Clenshaw recurrence
        # Scale spectrum: J = (a + b t), t in [-1,1]
        # We need Chebyshev coeffs of f(a + b t). Use scalar callback with standard Clenshaw form.
        # Implement vectorized Clenshaw via three-term recurrence on matrices.
        # Precompute nodes’ coeffs via barycentric trick: we’ll use a fixed set of Cheb moments
        # For compactness, approximate with Horner-like three-term recurrence:
        #   Initialize:
        T0 = X.copy()
        T1 = (J_mv(X) - a*T0) / b
        # Use truncated series f(J)X ≈ c0*T0 + c1*T1 + ...; we approximate coefficients by sampling f at T_n(0) basis:
        # In practice, fixed coefficients are better; to stay compact, we use the well-known recursion for sqrt:
        # Use a few-term Lanczos-like stabilization by re-scaling result after loop.
        # For robustness and compactness here, fall back to the simpler power-series-emulating Cheb sum with weights:
        #   Y = α0*T0 + α1*T1 + Σ αn*Tn
        # Coeffs via cosine integral are lengthy; instead, precompute numerically once per build:
        Nprobe = 256
        theta = (np.arange(Nprobe) + 0.5) * (np.pi / Nprobe)
        tvals = np.cos(theta)                        # in (-1,1)
        lam   = a + b * tvals
        if which == "sqrt":
            fvals = np.sqrt(lam)
        else:
            fvals = 1.0 / np.sqrt(lam)
        # Cheb coeffs (type I) c_n = (2/N) Σ f(cosθ_k) cos(n θ_k); c0 halved later
        cosnt = np.cos(np.outer(np.arange(deg+1), theta))  # (deg+1, Nprobe)
        cn = (2.0 / Nprobe) * (cosnt @ fvals)
        cn[0] *= 0.5

        Y = cn[0] * T0 + cn[1] * T1
        Tnm2, Tnm1 = T0, T1
        for n in range(2, deg+1):
            Tn = (J_mv(Tnm1) - a*Tnm1) / b - Tnm2
            Y += cn[n] * Tn
            Tnm2, Tnm1 = Tnm1, Tn
        return Y

    def apply_shalf(X):
        Xs = Dinv[:, None] * X
        Y  = cheb_apply(np.sqrt, Xs)
        return D[:, None] * Y

    def apply_sminushalf(X):
        Xs = D[:, None] * X
        Y  = cheb_apply(lambda z: 1.0/np.sqrt(z), Xs)
        return Dinv[:, None] * Y

    print(f"[LOWDIN/CHEB] deg={deg} λ∈[{lam_min:.3e}, {lam_max:.3e}] (Jacobi+Gershgorin; {'sparse' if is_sparse else 'dense'} S)")
    return {
        "apply_shalf": apply_shalf,
        "apply_sminushalf": apply_sminushalf,
        "deg": deg, "lam_min": lam_min, "lam_max": lam_max,
        "is_sparse": is_sparse
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

def lowdin_apply_shalf(X, lowdin_ops):
    """
    Apply S^(1/2) to matrix X using precomputed lowdin_ops.
    Works for both Chebyshev operator dicts and dense Shalf arrays.
    """
    # Chebyshev ops dict path
    if isinstance(lowdin_ops, dict) and "apply_shalf" in lowdin_ops:
        return lowdin_ops["apply_shalf"](X)
    # Dense matrix path
    Shalf = np.asarray(lowdin_ops, dtype=np.complex128 if np.iscomplexobj(X) else np.float64)
    return Shalf @ X

