# hamiltonian.py
import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

from . import config
from . import utils

import time
from collections import defaultdict

def assemble_soc_matrices(n_ao, projector_params, shell_dicts_spherical,
                          soc_tbl, k_big_cache, soc_active_atoms,
                          calculate_offsite_soc, ao_to_atom_map, libint_cpp):
    """
    Builds the SOC matrices (Hx, Hy, Hz) with performance optimizations:
      - per-atom row slicing when offsite SOC is disabled (n -> n_loc)
      - distance cutoff for offsite SOC (R_cut in Å; defaults to 7.5 Å)
      - optional norm-based pruning after distance cut (keeps ~all mass)
      - automatic sparse path if B_loc is sufficiently sparse
      - HERK-like update when K_big is Hermitian (via cached eigendecomp)
    Also prints a profiler log per (atom,l) block and a summary.

    Returns:
        B_raw            : (n_ao, n_cols) complex128
        B_modified_full  : (n_ao, n_cols) complex128 (zeroed where masked)
        Hx, Hy, Hz       : (n_ao, n_ao) complex128 Hermitian
        proj_info        : list of dicts
    """
    # ---- Tunables (overridable via config) ----
    SPARSE_DENSITY_THRESHOLD = getattr(config, "SOC_SPARSE_DENSITY_THRESHOLD", 0.10)  # density < 10% -> sparse
    USE_HERK_LIKE_UPDATE     = getattr(config, "SOC_USE_HERK", True)
    KEEP_B_MODIFIED_FULL     = getattr(config, "SOC_KEEP_B_MODIFIED", True)
    ZERO_EPS                 = getattr(config, "SOC_ZERO_EPS", 1e-15)     # threshold for "structural zero"
    HERMIT_TOL               = getattr(config, "SOC_HERMIT_TOL", 1e-12)   # hermiticity tolerance for K
    # Distance cutoff (Å) for offsite rows; if ATOM_COORDS missing, falls back to norm pruning only
    R_CUT_ANG                = getattr(config, "SOC_R_CUT_ANG", 7.5)
    # Norm-based pruning: keep rows until cumulative row-norm mass >= 1 - tol (after distance filter)
    ROWNORM_CUM_TOL          = getattr(config, "SOC_ROW_CUM_TOL", 1e-5)    # 1e-5 -> keep 99.999% mass
    # -------------------------------------------------------------

    # Optional SciPy sparse support
    try:
        import scipy.sparse as sp
        SCIPY_AVAILABLE = True
    except Exception:
        SCIPY_AVAILABLE = False

    # ---- Early exit if no projectors ----
    if not projector_params:
        print("\n[WARNING] No SOC projectors found. SOC contribution will be zero.")
        zeros = np.zeros((n_ao, n_ao), dtype=np.complex128)
        return np.zeros((n_ao, 0), dtype=np.complex128), np.zeros((n_ao, 0), dtype=np.complex128), zeros, zeros, zeros, []

    # ---- Compute B (HGH overlaps) ----
    print("\n[INFO] Calling C++ wrapper to compute HGH overlaps...")
    t0 = time.perf_counter()
    B_raw = libint_cpp.compute_hgh_overlaps(shell_dicts_spherical, projector_params, config.NTHREADS)
    tB = time.perf_counter() - t0
    if B_raw.dtype != np.complex128:
        B_raw = B_raw.astype(np.complex128, copy=False)
    n_cols = B_raw.shape[1]
    print(f"  ...done in {tB:.3f}s. Overlap matrix B has shape: {B_raw.shape}")

    # ---- Accumulators ----
    Hx = np.zeros((n_ao, n_ao), dtype=np.complex128)
    Hy = np.zeros((n_ao, n_ao), dtype=np.complex128)
    Hz = np.zeros((n_ao, n_ao), dtype=np.complex128)

    if KEEP_B_MODIFIED_FULL:
        B_modified_full = np.zeros_like(B_raw)
    else:
        B_modified_full = np.zeros((n_ao, n_cols), dtype=np.complex128)

    # ---- Group projectors by (atom_idx, l) ----
    proj_groups = {}
    for i, p in enumerate(projector_params):
        key = (p['atom_idx'], p['l'])
        if key not in proj_groups:
            proj_groups[key] = {'params': [], 'col_indices': []}
        proj_groups[key]['params'].append(p)
        proj_groups[key]['col_indices'].append(i)

    # ---- Precompute AO rows per atom ----
    ao_rows_by_atom = defaultdict(list)
    for ao_idx, at in enumerate(ao_to_atom_map):
        ao_rows_by_atom[at].append(ao_idx)
    for at in ao_rows_by_atom:
        ao_rows_by_atom[at] = np.asarray(ao_rows_by_atom[at], dtype=np.int32)

    # ---- Coordinates (for distance cutoff) ----
    # Expect config.ATOM_COORDS as (n_atoms, 3) in Å; AO centers assumed at their atom coords
    HAVE_COORDS = hasattr(config, "ATOM_COORDS")
    if HAVE_COORDS:
        atom_coords = np.asarray(getattr(config, "ATOM_COORDS"), dtype=float)  # Å
        # quick sanity
        if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
            HAVE_COORDS = False

    # ---- Cache eigendecompositions of K_big per (sym,l,comp) ----
    K_evd_cache = {}
    for sym, lmap in k_big_cache.items():
        for l, comps in lmap.items():
            for comp_key, K in comps.items():  # 'x','y','z'
                Kc = K.astype(np.complex128, copy=False)
                if USE_HERK_LIKE_UPDATE and np.allclose(Kc, Kc.conj().T, atol=HERMIT_TOL, rtol=HERMIT_TOL):
                    w, U = np.linalg.eigh(Kc)
                    K_evd_cache[(sym, l, comp_key)] = (w, U)
                else:
                    K_evd_cache[(sym, l, comp_key)] = None

    # ---- Profiler containers ----
    prof = {
        'blocks': [],               # list of per-block dicts
        'counts': defaultdict(int), # path counters
        'time_total_blocks': 0.0,
        'time_B': tB
    }

    # ---- Helpers ----
    def _estimate_density(B_loc, thr=ZERO_EPS):
        return (np.count_nonzero(np.abs(B_loc) > thr) / B_loc.size) if B_loc.size else 1.0

    def _prune_rows_by_norm(B_mat, keep_mass=1.0 - ROWNORM_CUM_TOL):
        """Return sorted unique row indices that retain 'keep_mass' of row-norm^2."""
        if B_mat.size == 0:
            return np.empty((0,), dtype=np.int32)
        rn2 = np.sum(np.abs(B_mat)**2, axis=1)
        if rn2.sum() == 0.0:
            return np.arange(B_mat.shape[0], dtype=np.int32)
        order = np.argsort(rn2)[::-1]
        cum = np.cumsum(rn2[order]) / rn2.sum()
        k = np.searchsorted(cum, keep_mass) + 1
        sel = np.sort(order[:k].astype(np.int32))
        return sel

    def _update_component(H_comp, sym, l, comp_key, K_big, B_loc, rows, use_sparse):
        # Return (time_seconds, path_label)
        t0c = time.perf_counter()

        # Try cached EVD path
        evd = K_evd_cache.get((sym, l, comp_key))
        if USE_HERK_LIKE_UPDATE and evd is not None:
            w, U = evd
            pos = w >  1e-14
            neg = w < -1e-14

            H_add = None
            if np.any(pos):
                Up = U[:, pos] * np.sqrt(w[pos])[np.newaxis, :]
                if SCIPY_AVAILABLE and use_sparse:
                    Bsp = sp.csr_matrix(B_loc)
                    Yp = (Bsp @ Up); Yp = np.asarray(Yp)
                else:
                    Yp = B_loc @ Up
                H_add = Yp @ Yp.conj().T
            if np.any(neg):
                Un = U[:, neg] * np.sqrt(-w[neg])[np.newaxis, :]
                if SCIPY_AVAILABLE and use_sparse:
                    Bsp = sp.csr_matrix(B_loc)
                    Yn = (Bsp @ Un); Yn = np.asarray(Yn)
                else:
                    Yn = B_loc @ Un
                H_add = (-Yn @ Yn.conj().T) if H_add is None else (H_add - Yn @ Yn.conj().T)

            if H_add is not None:
                if rows is None:
                    H_comp += 0.5 * H_add
                else:
                    H_comp[np.ix_(rows, rows)] += 0.5 * H_add
            return (time.perf_counter() - t0c, 'herk_sparse' if (SCIPY_AVAILABLE and use_sparse) else 'herk_dense')

        # Generic GEMM path
        K = K_big if K_big.dtype == np.complex128 else K_big.astype(np.complex128, copy=False)
        if SCIPY_AVAILABLE and use_sparse:
            Bsp = sp.csr_matrix(B_loc)         # (n_loc, m)
            M = Bsp @ K                        # -> dense (n_loc, m)
            M = np.asarray(M)
            H_loc = M @ Bsp.conjugate().transpose()  # dense @ sparse^H -> dense
            H_loc = np.asarray(H_loc)
        else:
            M = B_loc @ K
            H_loc = M @ B_loc.conj().T

        if rows is None:
            H_comp += 0.5 * H_loc
        else:
            H_comp[np.ix_(rows, rows)] += 0.5 * H_loc
        return (time.perf_counter() - t0c, 'gemm_sparse' if (SCIPY_AVAILABLE and use_sparse) else 'gemm_dense')

    # ---- Iterate blocks in a stable order ----
    proj_col_offset = 0
    sorted_keys = sorted(proj_groups.keys())

    for key in sorted_keys:
        atom_idx, l = key
        params = proj_groups[key]['params']
        sym = params[0]['sym']

        potential_block = next((b for b in soc_tbl[sym]['so'] if b['l'] == l and b['k_coeffs']), None)
        if not potential_block:
            continue

        nprj = potential_block['nprj']
        m = nprj * (2 * l + 1)

        j0 = proj_col_offset
        j1 = proj_col_offset + m
        B_block = B_raw[:, j0:j1]  # (n_ao, m)

        is_proj_on_active_atom = (sym in soc_active_atoms)
        block_info = {
            'atom_idx': atom_idx, 'sym': sym, 'l': l,
            'm': m, 'n_ao': n_ao, 'offsite': bool(calculate_offsite_soc),
            'active': bool(is_proj_on_active_atom)
        }

        if not is_proj_on_active_atom:
            if KEEP_B_MODIFIED_FULL:
                B_modified_full[:, j0:j1] = 0.0
            block_info['path'] = 'skipped_inactive_atom'
            prof['blocks'].append(block_info)
            proj_col_offset += m
            continue

        # ---- Determine contributing rows ----
        rows = None
        rows_note = "all"
        rows_pre = n_ao

        if not calculate_offsite_soc:
            rows = ao_rows_by_atom[atom_idx]
            rows_pre = rows.size
            rows_note = "on-site"
            B_loc = B_block[rows, :]
            if KEEP_B_MODIFIED_FULL:
                B_mod = np.zeros_like(B_block)
                B_mod[rows, :] = B_loc
                B_modified_full[:, j0:j1] = B_mod
        else:
            # Offsite enabled
            if HAVE_COORDS:
                # Distance-based near field: atoms within R_CUT_ANG of projector's atom
                r0 = atom_coords[atom_idx]          # Å
                d = np.linalg.norm(atom_coords - r0, axis=1)
                atoms_keep = np.where(d <= R_CUT_ANG)[0]
                # Gather AO rows belonging to these atoms
                if atoms_keep.size:
                    rows_list = [ao_rows_by_atom[a] for a in atoms_keep if a in ao_rows_by_atom]
                    if rows_list:
                        rows = np.unique(np.concatenate(rows_list))
                if rows is None or rows.size == 0:
                    # nothing in cutoff -> skip block
                    if KEEP_B_MODIFIED_FULL:
                        B_modified_full[:, j0:j1] = 0.0
                    block_info.update({'rows_pre': 0, 'rows_post': 0, 'rows_note': 'empty_cutoff'})
                    prof['blocks'].append(block_info)
                    proj_col_offset += m
                    continue
                rows_pre = rows.size
                rows_note = f"dist<= {R_CUT_ANG:.2f} Å"
                B_loc = B_block[rows, :]
                if KEEP_B_MODIFIED_FULL:
                    B_mod = np.zeros_like(B_block)
                    B_mod[rows, :] = B_loc
                    B_modified_full[:, j0:j1] = B_mod
            else:
                # No coordinates available: keep all rows (will prune by norm below)
                B_loc = B_block
                if KEEP_B_MODIFIED_FULL:
                    B_modified_full[:, j0:j1] = B_block

        # Optional norm-based pruning after initial selection (distance or on-site/all)
        rows_post = None
        if rows is None:
            # currently all rows selected (no coords case or full offsite); prune by norm
            sel = _prune_rows_by_norm(B_loc, keep_mass=1.0 - ROWNORM_CUM_TOL)
            if sel.size < B_loc.shape[0]:
                rows_note = (rows_note + " + norm-pruned") if rows_note != "all" else "norm-pruned"
                rows_pre = B_loc.shape[0]
                rows_post = sel.size
                B_loc = B_loc[sel, :]
                # build 'rows' index in global space
                rows = sel if not HAVE_COORDS and not (not calculate_offsite_soc) else sel
                # when we had no rows previously (rows=None), we need rows to scatter-add;
                # here we choose to work in the reduced subspace and then scatter later via index map.
                # For simplicity, use rows=None when we truly used the full n_ao (we didn't here).
            else:
                rows_post = B_loc.shape[0]
        else:
            # We have a global 'rows' index (on-site or distance-cut); apply norm pruning within it
            sel = _prune_rows_by_norm(B_loc, keep_mass=1.0 - ROWNORM_CUM_TOL)
            if sel.size < B_loc.shape[0]:
                rows = rows[sel]           # map back to global AO indices
                B_loc = B_loc[sel, :]
                rows_post = rows.size
                rows_note = rows_note + " + norm-pruned"
            else:
                rows_post = rows.size

        n_loc = B_loc.shape[0]
        block_info['n_loc'] = n_loc
        block_info['rows_pre'] = rows_pre
        block_info['rows_post'] = rows_post if rows_post is not None else n_loc
        block_info['rows_note'] = rows_note

        # Density estimate for sparsity decision
        if SCIPY_AVAILABLE:
            density = _estimate_density(B_loc, ZERO_EPS)
            use_sparse = (density < SPARSE_DENSITY_THRESHOLD)
            block_info['density'] = float(density)
        else:
            use_sparse = False
            block_info['density'] = None

        t0blk = time.perf_counter()

        # Update x,y,z (use cached EVD where possible)
        times = {}
        paths = {}
        for comp_key, H_comp in (('x', Hx), ('y', Hy), ('z', Hz)):
            dt, path = _update_component(H_comp, sym, l, comp_key,
                                         k_big_cache[sym][l][comp_key],
                                         B_loc, rows, use_sparse)
            times[comp_key] = dt
            paths[comp_key] = path
            prof['counts'][path] += 1

        dt_blk = time.perf_counter() - t0blk
        prof['time_total_blocks'] += dt_blk

        block_info.update({
            'time_block': dt_blk,
            'time_x': times['x'], 'time_y': times['y'], 'time_z': times['z'],
            'path_x': paths['x'], 'path_y': paths['y'], 'path_z': paths['z']
        })
        prof['blocks'].append(block_info)

        proj_col_offset += m

        # Per-block log line
        dens_str = f"{block_info['density']:.4f}" if block_info['density'] is not None else "n/a"
        print(f"[SOC blk] atom={atom_idx:3d} sym={sym:>2} l={l} "
              f"n_loc={n_loc:4d} m={m:3d} dens={dens_str} "
              f"rows({block_info['rows_pre']}→{block_info['rows_post']}:{rows_note}) "
              f"paths=({paths['x']},{paths['y']},{paths['z']}) "
              f"time={dt_blk:.3f}s")

    # ---- Symmetrize to clean numerical noise ----
    Hx = 0.5 * (Hx + Hx.conj().T)
    Hy = 0.5 * (Hy + Hy.conj().T)
    Hz = 0.5 * (Hz + Hz.conj().T)

    # ---- Summary ----
    n_blocks = len(prof['blocks'])
    print("\n[PROFILE] SOC assembly summary")
    print(f"  B build time:       {prof['time_B']:.3f}s")
    print(f"  # blocks processed: {n_blocks}")
    print(f"  Block compute time: {prof['time_total_blocks']:.3f}s")
    if prof['counts']:
        print("  Path counts:")
        for k, v in sorted(prof['counts'].items()):
            print(f"    {k:>12s}: {v}")
    if n_blocks:
        slow = sorted(prof['blocks'], key=lambda d: d.get('time_block', 0.0), reverse=True)[:5]
        print("  Slowest blocks:")
        for b in slow:
            dens_str = f"{b['density']:.4f}" if b['density'] is not None else "n/a"
            print(f"    atom={b['atom_idx']:3d} sym={b['sym']:>2} l={b['l']} "
                  f"n_loc={b.get('n_loc','?'):4d} m={b['m']:3d} dens={dens_str} "
                  f"rows({b.get('rows_pre','?')}→{b.get('rows_post','?')}:{b.get('rows_note','-')}) "
                  f"time={b.get('time_block',0.0):.3f}s "
                  f"paths=({b.get('path_x')},{b.get('path_y')},{b.get('path_z')})")

    # ---- proj_info (unchanged) ----
    proj_info = [{'sym': p['sym'], 'atom_idx': p['atom_idx'], 'l': p['l'], 'i': p['i'], 'm': m}
                 for p in projector_params for m in range(-p['l'], p['l'] + 1)]

    return B_raw, B_modified_full, Hx, Hy, Hz, proj_info


def assemble_soc_matricesi_old(n_ao, projector_params, shell_dicts_spherical,
                          soc_tbl, k_big_cache, soc_active_atoms,
                          calculate_offsite_soc, ao_to_atom_map, libint_cpp):
    """
    Builds the SOC matrices (Hx, Hy, Hz) by calling the C++ integral engine
    and assembling the contributions.
    """
    if not projector_params:
        print("\n[WARNING] No SOC projectors found. SOC contribution will be zero.")
        zeros = np.zeros((n_ao, n_ao), dtype=complex)
        return np.zeros((n_ao, 0)), np.zeros((n_ao, 0)), zeros, zeros, zeros, []

    print("\n[INFO] Calling C++ wrapper to compute HGH overlaps...")
    B_raw = libint_cpp.compute_hgh_overlaps(shell_dicts_spherical, projector_params, config.NTHREADS)
    print(f"  ...done. Overlap matrix B has shape: {B_raw.shape}")

    Hx, Hy, Hz = np.zeros((n_ao, n_ao), dtype=complex), np.zeros((n_ao, n_ao), dtype=complex), np.zeros((n_ao, n_ao), dtype=complex)
    B_modified_full = np.zeros_like(B_raw)
    
    # Group projectors by atom and l to process them in blocks
    proj_groups = {}
    for i, p in enumerate(projector_params):
        key = (p['atom_idx'], p['l'])
        if key not in proj_groups:
            proj_groups[key] = {'params': [], 'col_indices': []}
        proj_groups[key]['params'].append(p)
        proj_groups[key]['col_indices'].append(i)

    # Process each block
    proj_col_offset = 0
    sorted_keys = sorted(proj_groups.keys())

    for key in sorted_keys:
        atom_idx, l = key
        params = proj_groups[key]['params']
        sym = params[0]['sym']
        
        # Find the correct potential block
        potential_block = next((b for b in soc_tbl[sym]['so'] if b['l'] == l and b['k_coeffs']), None)
        if not potential_block: continue

        nprj = potential_block['nprj']
        num_funcs_in_block = nprj * (2 * l + 1)
        
        B_block = B_raw[:, proj_col_offset : proj_col_offset + num_funcs_in_block]
        B_modified = B_block.copy()

        # Apply SOC active atom and offsite rules
        is_proj_on_active_atom = (sym in soc_active_atoms)
        if not is_proj_on_active_atom:
             B_modified[:,:] = 0.0 # Zero out the entire block if projector is not on an active atom
        elif not calculate_offsite_soc:
            # Zero out rows corresponding to AOs on different atoms
            for ao_idx in range(n_ao):
                if ao_to_atom_map[ao_idx] != atom_idx:
                    B_modified[ao_idx, :] = 0.0

        B_modified_full[:, proj_col_offset : proj_col_offset + num_funcs_in_block] = B_modified

        # Assemble Hamiltonian components
        for comp, H_comp in zip(('x', 'y', 'z'), (Hx, Hy, Hz)):
            K_big = k_big_cache[sym][l][comp]
            H_comp += (B_modified @ K_big @ B_modified.T) * 0.5

        proj_col_offset += num_funcs_in_block

    proj_info = [{'sym': p['sym'], 'atom_idx': p['atom_idx'], 'l': p['l'], 'i': p['i'], 'm': m}
                 for p in projector_params for m in range(-p['l'], p['l']+1)]

    return B_raw, B_modified_full, Hx, Hy, Hz, proj_info

def apply_soc_energy_window(H_soc, S_ao, C_ao, eps_Ha, homo_idx, energy_window_eV):
    """
    Filters a SOC component (Hx, Hy, Hz) using an energy window.
    Supports:
      - RKS: C_ao: (nAO, nMO), eps_Ha: (nMO,), homo_idx: int
      - UKS: C_ao: (C_alpha, C_beta), eps_Ha: (eps_a, eps_b), homo_idx: (h_a, h_b)
    """
    if energy_window_eV is None:
        return H_soc

    # UKS: sum the two spin-channel filtered projections in AO space
    if isinstance(C_ao, (tuple, list)):
        C_list, eps_list, h_list = C_ao, eps_Ha, homo_idx
        Hf = np.zeros_like(H_soc, dtype=complex)
        for C, eps, h in zip(C_list, eps_list, h_list):
            Hf += _filter_single(H_soc, S_ao, C, eps, h, energy_window_eV)
        return Hf

    # RKS: single channel
    return _filter_single(H_soc, S_ao, C_ao, eps_Ha, homo_idx, energy_window_eV)


def _filter_single(H_soc, S_ao, C, eps, homo, energy_window_eV):
    H_mo = C.T @ H_soc @ C
    energy_window_Ha = energy_window_eV / config.H2EV
    # Fermi level from adjacent levels around HOMO
    E_fermi_Ha = (eps[homo] + eps[min(homo + 1, len(eps) - 1)]) / 2.0
    min_E = E_fermi_Ha - 0.5 * energy_window_Ha
    max_E = E_fermi_Ha + 0.5 * energy_window_Ha

    sel = (eps >= min_E) & (eps <= max_E)
    mask = np.outer(sel, sel)
    H_mo_filtered = H_mo * mask
    return S_ao @ C @ H_mo_filtered @ C.T @ S_ao

def _orthonormalize_in_S(C_AO_sel, S_AO):
    """Return C_ortho with C_ortho^† S C_ortho = I_k."""
    S_sub = C_AO_sel.conj().T @ S_AO @ C_AO_sel
    X = fractional_matrix_power(S_sub, -0.5)
    return C_AO_sel @ X

def solve_soc_in_mo_subspace_rks(C_AO, eps_Ha, S_AO, Hx_AO, Hy_AO, Hz_AO, idx_sel, *, debug=False):
    """
    RKS: subspace solve in the span of selected canonical MOs (k of them).
    Returns (E_soc_Ha, U_soc_mo, U_soc_ao).
    """
    # 1) Orthonormalize subspace in S
    Csel = C_AO[:, idx_sel]                                          # (n_ao, k)
    Csel_ortho = _orthonormalize_in_S(Csel, S_AO)                    # S-orthonormal

    # 2) MO-space SOC blocks (k x k)
    Hx_mo = Csel_ortho.conj().T @ Hx_AO @ Csel_ortho
    Hy_mo = Csel_ortho.conj().T @ Hy_AO @ Csel_ortho
    Hz_mo = Csel_ortho.conj().T @ Hz_AO @ Csel_ortho

    # 3) Spin-free baseline in MO-space
    eps_sel = np.asarray(eps_Ha)[idx_sel]
    H0_mo = np.diag(eps_sel)                                         # (k, k)

    # 4) Build spinor Hamiltonian in the subspace
    k = len(idx_sel)
    I2 = np.eye(2)
    sigma_x = np.array([[0, 1],[1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0],[0, -1]], dtype=complex)

    H0_total = np.kron(I2, H0_mo)                                    # (2k, 2k)
    H_SO_total = 0.5 * (np.kron(sigma_x, Hx_mo) +
                        np.kron(sigma_y, Hy_mo) +
                        np.kron(sigma_z, Hz_mo))
    H_total = H0_total - H_SO_total                                  # SIGN!

    if debug:
        # With SOC off, eigenvalues must be eps_sel duplicated
        w0 = np.linalg.eigvalsh(H0_total)
        ref = np.sort(np.r_[eps_sel, eps_sel])
        assert np.allclose(np.sort(w0), ref, atol=1e-10), "RKS subspace baseline check failed"

    # 5) Solve (standard Hermitian problem; metric = I in this basis)
    E, U_mo = eigh(H_total)                                          # (2k,), (2k, 2k)

    # 6) Map spinors back to AO space for analysis:
    U_ao = np.zeros((2*C_AO.shape[0], 2*k), dtype=complex)
    # top (alpha) block in AO:
    U_ao[:C_AO.shape[0], :] = Csel_ortho @ U_mo[:k, :]
    # bottom (beta) block in AO:
    U_ao[C_AO.shape[0]:, :] = Csel_ortho @ U_mo[k:, :]

    return E, U_mo, U_ao

def solve_soc_in_mo_subspace_uks(Ca_AO, eps_a, Cb_AO, eps_b, S_AO, Hx_AO, Hy_AO, Hz_AO,
                                 idx_sel_a, idx_sel_b, *, debug=False):
    """
    UKS: α and β can have different MO subspaces (kα and kβ).
    We build Φ = diag(Ca_ortho, Cb_ortho) and project the full spinor operator.
    Returns (E_soc_Ha, U_soc_mo, U_soc_ao).
    """
    # 1) Orthonormalize α, β subspaces in S
    Ca_sel = Ca_AO[:, idx_sel_a]                                    # (n_ao, kα)
    Cb_sel = Cb_AO[:, idx_sel_b]                                    # (n_ao, kβ)
    Ca_ortho = _orthonormalize_in_S(Ca_sel, S_AO)
    Cb_ortho = _orthonormalize_in_S(Cb_sel, S_AO)
    ka, kb = Ca_ortho.shape[1], Cb_ortho.shape[1]
    k_tot = ka + kb

    # 2) All projected SOC blocks (note: not a simple Kron if ka != kb)
    # Diagonal (σ_z part uses + on α, − on β):
    Hz_aa = Ca_ortho.conj().T @ Hz_AO @ Ca_ortho                    # (ka, ka)
    Hz_bb = Cb_ortho.conj().T @ Hz_AO @ Cb_ortho                    # (kb, kb)
    # Off-diagonal (σ_x, σ_y couple α<->β):
    Hx_ab = Ca_ortho.conj().T @ Hx_AO @ Cb_ortho                    # (ka, kb)
    Hy_ab = Ca_ortho.conj().T @ Hy_AO @ Cb_ortho                    # (ka, kb)
    Hx_ba = Hx_ab.conj().T                                          # (kb, ka)
    Hy_ba = Hy_ab.conj().T                                          # (kb, ka)

    # 3) Spin-free baseline (block-diagonal α ⊕ β)
    Ha_mo = np.diag(np.asarray(eps_a)[idx_sel_a])                   # (ka, ka)
    Hb_mo = np.diag(np.asarray(eps_b)[idx_sel_b])                   # (kb, kb)
    H0_spinor = np.block([
        [Ha_mo,            np.zeros((ka, kb))],
        [np.zeros((kb, ka)), Hb_mo           ],
    ])                                                              # (ka+kb, ka+kb)

    # 4) SOC spinor operator in this basis:
    # [ +Hz_aa      Hx_ab - i Hy_ab ]
    # [ Hx_ba + i Hy_ba   -Hz_bb    ]
    H_SO_spinor = 0.5 * np.block([
        [ +Hz_aa,                 Hx_ab - 1j*Hy_ab ],
        [ Hx_ba + 1j*Hy_ba,      -Hz_bb            ],
    ])

    H_spinor = H0_spinor - H_SO_spinor

    if debug:
        # With SOC off, eigenvalues must be [eps_a_sel, eps_b_sel]
        w0 = np.linalg.eigvalsh(H0_spinor)
        ref = np.sort(np.r_[np.asarray(eps_a)[idx_sel_a], np.asarray(eps_b)[idx_sel_b]])
        assert np.allclose(np.sort(w0), ref, atol=1e-10), "UKS subspace baseline check failed"

    # 5) Solve in the spinor subspace
    E, U_mo = eigh(H_spinor)                                        # (ka+kb,), (ka+kb, ka+kb)

    # 6) Map to AO spinor (2*n_ao rows): top=α AO, bottom=β AO
    n_ao = Ca_AO.shape[0]
    U_ao = np.zeros((2*n_ao, k_tot), dtype=complex)
    # split columns of U_mo: first ka rows are α block in subspace, last kb rows are β
    U_ao[:n_ao, :] = Ca_ortho @ U_mo[:ka, :]
    U_ao[n_ao:, :] = Cb_ortho @ U_mo[ka:, :]

    return E, U_mo, U_ao


def reconstruct_fock_from_mos(C: np.ndarray, eps: np.ndarray, S_AO: np.ndarray) -> np.ndarray:
    """Reconstruct the spin-free AO operator F from canonical MOs & energies.
       Correct for the generalized problem F C = S C ε with C^† S C = I."""
    eps = np.asarray(eps)
    return S_AO @ C @ np.diag(eps) @ C.conj().T @ S_AO


def project_ao_operator_to_mo_subspace(C: np.ndarray,
                                       S_AO: np.ndarray,
                                       idx_sel: list[int],
                                       A_AO: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Project a Hermitian AO operator A_AO to the S-orthonormalized MO subspace spanned by idx_sel.

    Returns:
      A_sub : k x k matrix = Φ^† A_AO Φ
      Csel_ortho : n_ao x k matrix Φ = C_sel (C_sel^† S C_sel)^(-1/2)
    """
    from .hamiltonian import _orthonormalize_in_S  # reuse your function
    C_sel = C[:, idx_sel]
    Csel_ortho = _orthonormalize_in_S(C_sel, S_AO)     # Φ with Φ^† S Φ = I_k
    A_sub = Csel_ortho.conj().T @ A_AO @ Csel_ortho
    return A_sub, Csel_ortho

