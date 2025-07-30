# utils.py
import numpy as np
from scipy.linalg import fractional_matrix_power
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

def lowdin_Ssqrt(S, eps=1e-10):
    """Robust symmetric orthogonalization via S^(1/2)."""
    w, U = np.linalg.eigh(S)
    w_clipped = np.clip(w, eps, None)
    return (U * np.sqrt(w_clipped)) @ U.T

def ao_population_lowdin(alpha, beta, S):
    """Calculates Löwdin population for a given spinor."""
    Shalf = lowdin_Ssqrt(S)
    alpha = np.atleast_2d(alpha).T if alpha.ndim == 1 else alpha
    beta = np.atleast_2d(beta).T if beta.ndim == 1 else beta
    a, b = Shalf @ alpha, Shalf @ beta
    pop = (np.abs(a)**2 + np.abs(b)**2).real
    pop_sum = pop.sum(axis=0, keepdims=True)
    pop_sum[pop_sum < 1e-16] = 1.0
    return np.squeeze(pop / pop_sum)

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



