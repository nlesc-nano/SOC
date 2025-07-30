# hamiltonian.py
import numpy as np
from . import config
from . import utils

def assemble_soc_matrices(n_ao, projector_params, shell_dicts_spherical,
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



