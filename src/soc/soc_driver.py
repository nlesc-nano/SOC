#!/usr/bin/env python3
"""
SOC Driver - High-Level Workflow:
* Reads quantum chemistry data (geometry, basis, MOs, potentials).
* Builds the spin-orbit coupling (SOC) Hamiltonian.
* Solves the generalized eigenvalue problem for SOC-corrected energies.
* Performs post-analysis and visualization.
"""
import numpy as np
from scipy.linalg import eigh, fractional_matrix_power

# Import refactored modules
from . import libint_cpp  # Assuming this C++ wrapper module is available
from . import config
from . import parsers
from . import utils
from . import hamiltonian
from . import analysis
from .config import parse_and_apply_cli  # <-- only this import for CLI

def main():
    """Main execution workflow."""
    print("== SOC Driver (Refactored Workflow) ==")
    # Parse CLI and apply any overrides to config
    parse_and_apply_cli()

    # 1. ---------- Load Data and Set Up System ----------
    print("\n[STEP 1] Loading geometry, basis, and potentials...")
    syms, coords = parsers.read_xyz(config.XYZ_PATH)
    basis_dict = parsers.parse_basis(config.BASIS_TXT, config.BASIS_NAME)
    elements_in_xyz = {sym: config.valence_electrons.get(sym) for sym in set(syms)}

    coords_bohr = coords * config.BOHR_PER_ANG
    Z_val = [config.valence_electrons[s] for s in syms] 
    E_nuc = utils.compute_nuclear_repulsion_from_list(coords_bohr, Z_val)

    soc_tbl = parsers.parse_gth_soc_potentials(config.GTH_POTENTIALS_PATH, elements_in_xyz)

    shell_dicts = utils.build_shell_dicts(syms, coords, basis_dict)
    shell_dicts_spherical = [{**sh, 'pure': True} for sh in shell_dicts]
    n_ao = sum(2 * sh['l'] + 1 for sh in shell_dicts_spherical)
    print(f"System setup complete. Total AOs: {n_ao}")

    ao_info = utils.make_ao_info(shell_dicts_spherical)
    ao_to_atom_map = [ao['atom_idx'] for ao in ao_info]

    # 2. ---------- Load QM Data & Build H0 ----------
    print("\n[STEP 2] Loading MOs and constructing H0...")
    
    if not config.UKS:
        # --- RKS path (unchanged) ---
        C_AO, eps_Ha, occ = parsers.read_mos_txt(config.MO_PATH, n_ao)
        n_mo = C_AO.shape[1]
    
        S_AO = libint_cpp.overlap(shell_dicts_spherical, config.NTHREADS) if not config.PERIODIC else \
               libint_cpp.overlap_pbc(shell_dicts_spherical, config.LATTICE * config.BOHR_PER_ANG,
                                      cutoff_A=14, nthreads=config.NTHREADS)
    
        orth_err = np.linalg.norm(C_AO.T @ S_AO @ C_AO - np.eye(n_mo))
        print(f"[CHECK] MO Orthogonality ||CᵀSC - I|| = {orth_err:.3e}")
    
        # H0 from eigen-decomposition (AO representation)
        H0_ao = S_AO @ C_AO @ np.diag(eps_Ha) @ C_AO.T @ S_AO
        # or: H0_ao = utils.ao_fock_from_mos(S_AO, C_AO, eps_Ha)
    
        # total spinor overlap/H0
        S_total = np.kron(np.eye(2), S_AO)
        H0_total = np.kron(np.eye(2), H0_ao)
    
        # For SOC windowing downstream
        HOMO_idx = np.where(occ > 1e-3)[0][-1]
        C_for_filter = C_AO
        eps_for_filter = eps_Ha
        homo_for_filter = HOMO_idx
#        C_AO, eps_Ha, occ = C_a, eps_a, occ_a
        
        E_band = float(np.dot(occ, eps_Ha))  # Ha
        print(f"\n[ENERGY] Band energy (Σ f ε) RKS (spin‑free): {E_band:.8f} Ha")
        print(f"[ENERGY] Band + E_nuc(q_val):               {E_band + E_nuc:.8f} Ha")
 
    else:
        # --- UKS path ---
        C_a, eps_a, occ_a = parsers.read_mos_txt(config.MO_ALPHA_PATH, n_ao)
        C_b, eps_b, occ_b = parsers.read_mos_txt(config.MO_BETA_PATH,  n_ao)
        n_mo_a, n_mo_b = C_a.shape[1], C_b.shape[1]
    
        S_AO = libint_cpp.overlap(shell_dicts_spherical, config.NTHREADS) if not config.PERIODIC else \
               libint_cpp.overlap_pbc(shell_dicts_spherical, config.LATTICE * config.BOHR_PER_ANG,
                                      cutoff_A=14, nthreads=config.NTHREADS)
    
        err_a = np.linalg.norm(C_a.T @ S_AO @ C_a - np.eye(n_mo_a))
        err_b = np.linalg.norm(C_b.T @ S_AO @ C_b - np.eye(n_mo_b))
        print(f"[CHECK] ||CᵀSC - I|| (alpha) = {err_a:.3e} | (beta) = {err_b:.3e}")
    
        H0_a = S_AO @ C_a @ np.diag(eps_a) @ C_a.T @ S_AO   # utils.ao_fock_from_mos(S_AO, C_a, eps_a)
        H0_b = S_AO @ C_b @ np.diag(eps_b) @ C_b.T @ S_AO   # utils.ao_fock_from_mos(S_AO, C_b, eps_b)
    
        # Spinor overlap is unchanged; H0 becomes block-diagonal (α ⊕ β)
        S_total = np.kron(np.eye(2), S_AO)
        H0_total = np.zeros((2 * n_ao, 2 * n_ao), dtype=complex)
        H0_total[:n_ao, :n_ao] = H0_a
        H0_total[n_ao:, n_ao:] = H0_b
    
        # For SOC windowing downstream: pass both channels
        def last_occ(occ): 
            idx = np.where(occ > 1e-3)[0]
            return int(idx[-1]) if len(idx) else 0
    
        C_for_filter = (C_a, C_b)
        eps_for_filter = (eps_a, eps_b)
        homo_for_filter = (last_occ(occ_a), last_occ(occ_b))
        occ = np.concatenate([occ_a, occ_b])

        E_band = float(np.dot(occ_a, eps_a) + np.dot(occ_b, eps_b))
        print(f"\n[ENERGY] Band energy (Σ f ε) UKS (spin‑free): {E_band:.8f} Ha")
        print(f"[ENERGY] Band + E_nuc(q_val):                {E_band + E_nuc:.8f} Ha")

    # 3. ---------- Build SOC Hamiltonian (H_SO) ----------
    print("\n[STEP 3] Assembling the SOC Hamiltonian...")
    L_matrices = utils.build_L_matrices()
    k_big_cache = utils.precompute_soc_operators(soc_tbl, L_matrices)
    projector_params = utils.build_projector_params(syms, coords, soc_tbl)

    # Determine which atoms are active for SOC based on config.py
    if config.soc_active_atoms:
        active_atoms_list = config.soc_active_atoms
        print(f"[INFO] Using user-defined active atoms for SOC: {active_atoms_list}")
    else:
        active_atoms_list = utils.unique_atoms_syms(syms)
        print(f"[INFO] Defaulting to all system atoms as active for SOC: {active_atoms_list}")

    _, _, Hx, Hy, Hz, _ = hamiltonian.assemble_soc_matrices(
        n_ao, projector_params, shell_dicts_spherical, soc_tbl, k_big_cache,
        active_atoms_list, config.calculate_offsite_soc, ao_to_atom_map, libint_cpp
    )
    analysis.hermiticity_checks(Hx, Hy, Hz)

    # 4. ---------- Apply Energy Window Filter ----------
    print(f"\n[STEP 4] Applying energy window filter ({config.energy_window_eV} eV)...")
    Hx_filt = hamiltonian.apply_soc_energy_window(Hx, S_AO, C_for_filter, eps_for_filter, homo_for_filter, config.energy_window_eV)
    Hy_filt = hamiltonian.apply_soc_energy_window(Hy, S_AO, C_for_filter, eps_for_filter, homo_for_filter, config.energy_window_eV)
    Hz_filt = hamiltonian.apply_soc_energy_window(Hz, S_AO, C_for_filter, eps_for_filter, homo_for_filter, config.energy_window_eV)

    # 5. ---------- Solve Full Spinor Hamiltonian ----------
    print("\n[STEP 5] Building and diagonalizing the full spinor Hamiltonian...")
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

#    S_total = np.kron(np.eye(2), S_AO)
#    H0_total = np.kron(np.eye(2), H0_ao)
    H_SO_total = 0.5 * (np.kron(sigma_x, Hx_filt) + np.kron(sigma_y, Hy_filt) + np.kron(sigma_z, Hz_filt))

    H_total = H0_total - H_SO_total

    E_soc_Ha, U_soc_ao = eigh(H_total, b=S_total)

    # Total electrons (works for RKS and UKS; make sure 'occ' exists in UKS as occ = np.r_[occ_a, occ_b])
    Ne = int(round(np.sum(occ)))
    order = np.argsort(E_soc_Ha)
    f_spinor = np.zeros_like(E_soc_Ha)
    f_spinor[order[:Ne]] = 1.0
    
    E_band_soc = float(np.dot(f_spinor, E_soc_Ha))
    print(f"\n[ENERGY] Band energy with SOC (fill {Ne} spinors): {E_band_soc:.8f} Ha")
    print(f"[ENERGY] Band + E_nuc(q_val) with SOC:            {E_band_soc + E_nuc:.8f} Ha")

    print("Solver finished.")

    # 6. ---------- Post-Analysis and Results ----------
    print("\n[STEP 6] Performing population analysis...")
    l_labels = {0: "s", 1: "p", 2: "d", 3: "f"}
    l_list = sorted(set(ao['l'] for ao in ao_info))
    unique_syms = utils.unique_atoms_syms(syms)

    ## --- Analysis 1: Spin-Free MOs (Löwdin AO basis) ---
    N_print = config.N_print
    
    if not config.UKS:
        print("\n--- Population Analysis of Spin-Free MOs (Löwdin AO basis) ---")
        mo_indices = list(range(max(0, HOMO_idx - N_print + 1), HOMO_idx + 1)) + \
                     list(range(HOMO_idx + 1, min(n_mo, HOMO_idx + 1 + N_print)))
        mo_coeffs_mat = C_AO[:, mo_indices]
        pop_lowdin_mat_sf = utils.ao_population_lowdin(mo_coeffs_mat, np.zeros_like(mo_coeffs_mat), S_AO)
        for i, idx in enumerate(mo_indices):
            E_eV = eps_Ha[idx] * config.H2EV
            pop_lowdin = pop_lowdin_mat_sf[:, i] if pop_lowdin_mat_sf.ndim > 1 else pop_lowdin_mat_sf
            contrib_lowd = analysis.decompose_pop(pop_lowdin, ao_info, l_list, unique_syms)
            contrib_str = analysis.format_contrib(contrib_lowd, l_labels)
            print(f"MO {idx+1:3d} | E = {E_eV:10.6f} eV | Occ = {occ[idx]:4.2f} | {contrib_str}")
    
    else:
        # Alpha
        print("\n--- Population Analysis of Spin-Free Alpha MOs (Löwdin AO basis) ---")
        mo_indices_a = list(range(max(0, homo_for_filter[0] - N_print + 1), homo_for_filter[0] + 1)) + \
                       list(range(homo_for_filter[0] + 1, min(n_mo_a, homo_for_filter[0] + 1 + N_print)))
        mo_coeffs_mat_a = C_a[:, mo_indices_a]
        pop_lowdin_mat_sf_a = utils.ao_population_lowdin(mo_coeffs_mat_a, np.zeros_like(mo_coeffs_mat_a), S_AO)
        for i, idx in enumerate(mo_indices_a):
            E_eV = eps_a[idx] * config.H2EV
            pop_lowdin = pop_lowdin_mat_sf_a[:, i] if pop_lowdin_mat_sf_a.ndim > 1 else pop_lowdin_mat_sf_a
            contrib_lowd = analysis.decompose_pop(pop_lowdin, ao_info, l_list, unique_syms)
            contrib_str = analysis.format_contrib(contrib_lowd, l_labels)
            print(f"MO(α) {idx+1:3d} | E = {E_eV:10.6f} eV | Occ = {occ_a[idx]:4.2f} | {contrib_str}")
    
        # Beta
        print("\n--- Population Analysis of Spin-Free Beta MOs (Löwdin AO basis) ---")
        mo_indices_b = list(range(max(0, homo_for_filter[1] - N_print + 1), homo_for_filter[1] + 1)) + \
                       list(range(homo_for_filter[1] + 1, min(n_mo_b, homo_for_filter[1] + 1 + N_print)))
        mo_coeffs_mat_b = C_b[:, mo_indices_b]
        pop_lowdin_mat_sf_b = utils.ao_population_lowdin(mo_coeffs_mat_b, np.zeros_like(mo_coeffs_mat_b), S_AO)
        for i, idx in enumerate(mo_indices_b):
            E_eV = eps_b[idx] * config.H2EV
            pop_lowdin = pop_lowdin_mat_sf_b[:, i] if pop_lowdin_mat_sf_b.ndim > 1 else pop_lowdin_mat_sf_b
            contrib_lowd = analysis.decompose_pop(pop_lowdin, ao_info, l_list, unique_syms)
            contrib_str = analysis.format_contrib(contrib_lowd, l_labels)
            print(f"MO(β) {idx+1:3d} | E = {E_eV:10.6f} eV | Occ = {occ_b[idx]:4.2f} | {contrib_str}")


    ## --- Analysis 2: SOC Spinors in AO Basis ---
    print("\n--- Population Analysis of SOC Spinors (Löwdin AO basis) ---")
    total_electrons = int(round(sum(occ)))
    idx_sorted = np.argsort(E_soc_Ha)
    N_print_soc = N_print * 2 
    to_print = list(idx_sorted[max(0, total_electrons - N_print_soc): total_electrons]) + \
               list(idx_sorted[total_electrons: total_electrons + N_print_soc])

    norms = np.einsum('ij,ji->j', U_soc_ao.conj().T, S_total @ U_soc_ao).real
    U_soc_norm = U_soc_ao / np.sqrt(norms)[None, :]

    alpha_mat = U_soc_norm[:n_ao, to_print]
    beta_mat  = U_soc_norm[n_ao:, to_print]
    pop_mat_soc = utils.ao_population_lowdin(alpha_mat, beta_mat, S_AO)

    for i, idx in enumerate(to_print):
        E_eV = E_soc_Ha[idx] * config.H2EV
        is_occ = (idx in idx_sorted[:total_electrons])
        pop_ao = pop_mat_soc[:, i] if pop_mat_soc.ndim > 1 else pop_mat_soc
        # Split total α and β population
        pop_alpha = np.sum(pop_ao[:n_ao])
        pop_beta  = np.sum(pop_ao[n_ao:])
        contrib = analysis.decompose_pop(pop_ao, ao_info, l_list, unique_syms)
        contrib_str = analysis.format_contrib(contrib, l_labels)
        print(f"Spinor {idx+1:4d} | E = {E_eV:9.4f} eV | Occ = {int(is_occ):d} | α={pop_alpha:.3f}, β={pop_beta:.3f} | {contrib_str}")

    ## --- Analysis 3: SOC Spinors in Spin-Free MO Basis ---
    if config.UKS:
        print("\n--- Population Analysis of SOC Spinors in terms of UKS Spin‑Free MOs ---")
    
        # 1) Orthonormalize C_alpha, C_beta in the AO metric S_AO
        #    (so Φ† S_total Φ = I)
        S_mo_a = C_a.conj().T @ S_AO @ C_a
        S_mo_b = C_b.conj().T @ S_AO @ C_b
        C_a_ortho = C_a @ fractional_matrix_power(S_mo_a, -0.5)
        C_b_ortho = C_b @ fractional_matrix_power(S_mo_b, -0.5)
    
        n_mo_a = C_a_ortho.shape[1]
        n_mo_b = C_b_ortho.shape[1]
    
        # 2) Build the spinor spin‑free basis Φ = diag(Ca, Cb)
        sf_basis = np.zeros((2*n_ao, n_mo_a + n_mo_b), dtype=complex)
        sf_basis[:n_ao, :n_mo_a] = C_a_ortho
        sf_basis[n_ao:, n_mo_a:] = C_b_ortho
    
        # 3) Sanity: orthonormality in the spinor metric
        err = np.linalg.norm(sf_basis.conj().T @ S_total @ sf_basis - np.eye(n_mo_a + n_mo_b))
        print(f"[CHECK] ||Φ† S Φ - I|| = {err:.3e}")
    
        # 4) Project (use your already S-normalized spinors: U_soc_norm)
        #    Shape: proj = (n_mo_a + n_mo_b) x (#states in to_print)
        proj = sf_basis.conj().T @ (S_total @ U_soc_norm[:, to_print])
    
        # 5) Weights per spin‑free MO; column-normalize
        w_sf = (proj.conj() * proj).real
        w_sf /= w_sf.sum(axis=0, keepdims=True)
        # Make these available for plotting after the if/else:
        proj_abs2_for_conn = w_sf                             # shape: (n_mo_a+n_mo_b, len(to_print))
        E_sf_all_Ha_for_conn = np.concatenate([eps_a, eps_b]) # spin‑free energies in Φ order
    
        labels = [f"α{j+1}" for j in range(n_mo_a)] + [f"β{j+1}" for j in range(n_mo_b)]
    
        # 6) Pretty print: α/β totals + top 5 contributing spin‑free MOs
        for col, idx in enumerate(to_print):
            E_eV = E_soc_Ha[idx] * config.H2EV
            is_occ = (idx in idx_sorted[:total_electrons])
    
            wcol    = w_sf[:, col]
            w_alpha = wcol[:n_mo_a].sum()
            w_beta  = wcol[n_mo_a:].sum()
    
            top = np.argsort(-wcol)[:5]
            comp = ", ".join(f"{labels[k]} ({wcol[k]*100:.1f}%)" for k in top)
    
            print(f"Spinor {idx+1:4d} | E = {E_eV:9.4f} eV | Occ = {int(is_occ):d} "
                  f"| α={w_alpha*100:5.1f}%, β={w_beta*100:5.1f}% | Comp: {comp}")
    
    else:
        # --- RKS path (your existing block, unchanged) ---
        print("\n--- Population Analysis of SOC Spinors in terms of Spin‑Free MOs ---")
        S_mo = C_AO.conj().T @ S_AO @ C_AO
        S_mo_mhalf = fractional_matrix_power(S_mo, -0.5)
        C_AO_ortho = C_AO @ S_mo_mhalf
    
        sf_basis = np.zeros((2*n_ao, 2*n_mo), dtype=complex)
        sf_basis[:n_ao,   :n_mo] = C_AO_ortho
        sf_basis[n_ao:, n_mo:  ] = C_AO_ortho
    
        proj = sf_basis.conj().T @ (S_total @ U_soc_norm[:, to_print])
        w_sf = np.abs(proj[:n_mo, :])**2 + np.abs(proj[n_mo:, :])**2
        w_sf /= w_sf.sum(axis=0, keepdims=True)
        # w_sf here is already summed over the duplicated spin blocks → shape: (n_mo, len(to_print))
        proj_abs2_for_conn = w_sf
        E_sf_all_Ha_for_conn = eps_Ha
 
        for i, idx in enumerate(to_print):
            E_eV = E_soc_Ha[idx] * config.H2EV
            is_occ = (idx in idx_sorted[:total_electrons])
            idx_top = np.argsort(-w_sf[:, i])[:5]
            sfmo_line = ", ".join([f"MO {k+1} ({w_sf[k, i]*100:.1f}%)" for k in idx_top])
            print(f"Spinor {idx+1:4d} | E = {E_eV:9.4f} eV | Occ = {int(is_occ):d} | Comp: {sfmo_line}")

    # === PLOTS: spin‑free vs SOC (stacked bars), PDOS overlay, and connections ===
    
    # 1) Choose atom‑type order for colors/stacking
    type_order = getattr(config, "TYPE_ORDER", None) or list(unique_syms)
    
    # 2) Prepare spin‑free selection (energies + Löwdin populations) for the stacked bars / PDOS
    if not config.UKS:
        # RKS: you already have 'mo_indices' and 'pop_lowdin_mat_sf' (AO Löwdin pops for selected MOs)
        E_sf_sel_Ha = np.array([eps_Ha[i] for i in mo_indices])
        pop_sf_sel = pop_lowdin_mat_sf
    else:
        # UKS: combine selected alpha and beta sets
        E_sf_sel_Ha = np.concatenate([eps_a[mo_indices_a], eps_b[mo_indices_b]])
        pop_sf_sel  = np.hstack([pop_lowdin_mat_sf_a,      pop_lowdin_mat_sf_b])
    
    # Fold spin blocks (alpha+beta) down to n_ao for type fractions
    # (In spin‑free alpha/beta matrices one half is zero, so this is safe.)
    if pop_sf_sel.shape[0] == 2 * n_ao:
        pop_sf_sel = pop_sf_sel[:n_ao, :] + pop_sf_sel[n_ao:, :]
    
    # 3) Prepare SOC selection (energies + Löwdin populations)
    E_soc_sel_Ha = E_soc_Ha[to_print]
    pop_soc_sel  = pop_mat_soc
    if pop_soc_sel.shape[0] == 2 * n_ao:
        pop_soc_sel = pop_soc_sel[:n_ao, :] + pop_soc_sel[n_ao:, :]
    
    # 4) Convert AO populations → per‑type fractions, then plot stacked bars and PDOS
    frac_sf  = analysis.fractions_from_pop_matrix(pop_sf_sel,  ao_info, syms, type_order)
    frac_soc = analysis.fractions_from_pop_matrix(pop_soc_sel, ao_info, syms, type_order)

    print(f"[DBG] spin-free selection: E {E_sf_sel_Ha.shape}, pop {pop_sf_sel.shape}")
    print(f"[DBG] SOC selection:       E {E_soc_sel_Ha.shape}, pop {pop_soc_sel.shape}")
    
    analysis.plot_elevels_stackedbars_sidebyside(
        energies1_Ha=E_sf_sel_Ha, frac1=frac_sf,
        energies2_Ha=E_soc_sel_Ha, frac2=frac_soc,
        type_order=type_order, label1="Spin-free", label2="SOC",
        outfile="electronic_structure_spinfree_vs_soc.png",
        H2EV=config.H2EV, bar_height_eV=0.06
    )
    print("[PLOT] Wrote: electronic_structure_spinfree_vs_soc.png")
     
    analysis.plot_pdos_mirror_fill(
        energies1_Ha=E_sf_sel_Ha, frac1=frac_sf,
        energies2_Ha=E_soc_sel_Ha, frac2=frac_soc,
        type_order=type_order, sigma_eV=0.10, npts=2000,
        outfile="pdos_spinfree_vs_soc.png", H2EV=config.H2EV
    )
    print("[PLOT] Wrote: pdos_spinfree_vs_soc.png")

 
    # 5) Connections plot: uses the weights prepared in Analysis 3
    analysis.plot_soc_connections(
        energies_sf_Ha=E_sf_all_Ha_for_conn,          # Φ energies (RKS: eps_Ha; UKS: [eps_a, eps_b])
        energies_soc_Ha=E_soc_sel_Ha,                 # SOC energies for 'to_print'
        proj_abs2=proj_abs2_for_conn,                 # |Φ† S U|^2 (columns correspond to 'to_print')
        topk=3, weight_min=0.05,
        outfile="connections_spinfree_to_soc.png",
        H2EV=config.H2EV
    )
    print("[PLOT] Wrote: connections_spinfree_to_soc.png")


    print("\n==== Done ====")

if __name__ == "__main__":
    main()

