# analysis.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from . import utils

def decompose_pop(pop_AO, ao_info, l_list, unique_syms):
    contrib = {sym: {l: 0.0 for l in l_list} for sym in unique_syms}
    tot = float(np.sum(pop_AO).real)
    if tot <= 1e-16:
        tot = 1.0
    for k, ao in enumerate(ao_info):
        contrib[ao['sym']][ao['l']] += float(pop_AO[k])
    # normalize to fractions
    for sym in contrib:
        for l in contrib[sym]:
            contrib[sym][l] /= tot
    return contrib

def format_contrib(contrib, l_labels):
    """Formats the decomposed population into a printable string."""
    parts = []
    for sym, l_contribs in contrib.items():
        sub_parts = [f"{l_labels.get(l, f'l={l}')}: {100*v:.1f}%"
                     for l, v in l_contribs.items() if v > 0.02]
        if sub_parts:
            parts.append(f"{sym} [{' + '.join(sub_parts)}]")
    return " | ".join(parts)

def hermiticity_checks(Hx, Hy, Hz):
    """Checks the hermiticity of the SOC matrices."""
    print(f"[Herm] ||Hx - Hx†|| = {np.linalg.norm(Hx - Hx.conj().T):.3e}")
    print(f"[Herm] ||Hy - Hy†|| = {np.linalg.norm(Hy - Hy.conj().T):.3e}")
    print(f"[Herm] ||Hz - Hz†|| = {np.linalg.norm(Hz - Hz.conj().T):.3e}")
    print(f"[Herm] ||Re(Hy)||   = {np.linalg.norm(Hy.real):.3e}") # Should be pure imaginary

# ===== analysis.py additions =====

def _type_order_default(unique_syms):
    # Stable default order
    return list(unique_syms)

def _color_map_for_types(type_order):
    # Deterministic colors; add more if needed
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]
    cmap = {}
    for i, t in enumerate(type_order):
        cmap[t] = base[i % len(base)]
    return cmap

def fractions_from_pop_matrix(pop_mat, ao_info, syms, type_order):
    """
    Convert AO Löwdin population matrix (n_ao x n_states) into
    per-state fractions by atom type (n_states x n_types), in type_order.
    """
    n_ao, n_states = pop_mat.shape
    # Map AO -> element symbol
    ao_sym = [syms[ao_info[k]['atom_idx']] for k in range(n_ao)]
    type_to_idx = {t: i for i, t in enumerate(type_order)}
    frac = np.zeros((n_states, len(type_order)), dtype=float)
    for j in range(n_states):
        col = pop_mat[:, j]
        tot = float(col.sum())
        if tot <= 0.0:
            continue
        # accumulate by type
        acc = np.zeros(len(type_order), dtype=float)
        for k in range(n_ao):
            t = ao_sym[k]
            if t in type_to_idx:
                acc[type_to_idx[t]] += col[k]
        frac[j, :] = acc / tot
    return frac  # (n_states, n_types)

def plot_elevels_stackedbars_sidebyside(
    energies1_Ha, frac1, energies2_Ha, frac2, type_order,
    label1="spin-free", label2="SOC", outfile=None,
    H2EV=27.211386245988, bar_height_eV=0.05, alpha1=0.85, alpha2=0.85
):

    E1 = np.asarray(energies1_Ha) * H2EV
    E2 = np.asarray(energies2_Ha) * H2EV
    frac1 = np.asarray(frac1)
    frac2 = np.asarray(frac2)

    # Guards
    if E1.size == 0 or E2.size == 0 or frac1.size == 0 or frac2.size == 0:
        print("[WARN] plot_elevels_stackedbars_sidebyside: empty inputs; nothing to draw.")
        fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
        if outfile: fig.savefig(outfile, dpi=200)
        return fig, axes
    if frac1.shape[0] != E1.size or frac2.shape[0] != E2.size:
        raise ValueError(f"Shape mismatch: frac1 rows={frac1.shape[0]} vs len(E1)={E1.size}, "
                         f"frac2 rows={frac2.shape[0]} vs len(E2)={E2.size}")

    colors = _color_map_for_types(type_order)
    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True, gridspec_kw=dict(width_ratios=[1, 1]))

    def draw(ax, E, frac, label, alpha):
        h = bar_height_eV
        for j, Eev in enumerate(E):
            left = 0.0
            for t_idx, t in enumerate(type_order):
                width = float(frac[j, t_idx])
                if width <= 1e-12: 
                    continue
                ax.add_patch(Rectangle((left, Eev - 0.5*h), width, h,
                                       facecolor=colors[t], edgecolor='none', alpha=alpha))
                left += width
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Fraction by atom type")
        ax.set_title(label)
        ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.4)

    draw(axes[0], E1, frac1, label1, alpha1)
    draw(axes[1], E2, frac2, label2, alpha2)

    # Set y-limits from data (with a little margin)
    all_E = np.concatenate([E1, E2])
    if all_E.size:
        ymin, ymax = float(all_E.min()), float(all_E.max())
        pad = max(0.5, 0.05 * max(1.0, ymax - ymin))
        for ax in axes:
            ax.set_ylim(ymin - pad, ymax + pad)

    axes[0].set_ylabel("Energy (eV)")
    fig.suptitle("Electronic structure: stacked type fractions")

    # Atom-type legend below
    handles_types = [Rectangle((0, 0), 1, 1, color=colors[t]) for t in type_order]
    labels_types = [f"{t}" for t in type_order]
    fig.legend(handles_types, labels_types, loc="lower center",
               ncol=min(len(type_order), 6), frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if outfile:
        fig.savefig(outfile, dpi=200)
    return fig, axes


def _gaussian(x, mu, sigma):
    s = sigma
    return np.exp(-0.5 * ((x - mu) / s)**2) / (s * np.sqrt(2.0*np.pi))

def plot_pdos_mirror_fill(
    energies1_Ha, frac1, energies2_Ha, frac2, type_order,
    sigma_eV=0.10, npts=2000, E_min_eV=None, E_max_eV=None,
    label1="spin-free", label2="SOC", outfile=None,
    H2EV=27.211386245988
):
    """
    Mirrored PDOS with fill: total spin-free PDOS filled up (y>0), SOC PDOS filled down (y<0).
    Optionally overlays per-type lines.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    E1 = np.asarray(energies1_Ha) * H2EV
    E2 = np.asarray(energies2_Ha) * H2EV
    frac1 = np.asarray(frac1)
    frac2 = np.asarray(frac2)
    if frac1.shape[0] != E1.size or frac2.shape[0] != E2.size:
        raise ValueError("PDOS mirror: frac rows must match number of states in each dataset.")

    E_all_min = float(np.min(E1)) if E1.size else 0.0
    E_all_max = float(np.max(E1)) if E1.size else 0.0
    if E2.size:
        E_all_min = min(E_all_min, float(np.min(E2)))
        E_all_max = max(E_all_max, float(np.max(E2)))
    if E_min_eV is None: E_min_eV = E_all_min - 0.5
    if E_max_eV is None: E_max_eV = E_all_max + 0.5
    x = np.linspace(E_min_eV, E_max_eV, int(npts))

    colors = _color_map_for_types(type_order)

    def _gauss(x, mu, sigma):
        s = sigma
        return np.exp(-0.5 * ((x - mu) / s)**2) / (s * np.sqrt(2.0*np.pi))

    # Compute per-type PDOS (matrix: n_types x len(x))
    y_sf_types = []
    y_soc_types = []
    for t_idx, t in enumerate(type_order):
        # Spin-free PDOS (up)
        y1 = np.zeros_like(x)
        for j, Ej in enumerate(E1):
            w = float(frac1[j, t_idx])
            if w <= 1e-12: continue
            y1 += w * _gauss(x, Ej, sigma_eV)
        y_sf_types.append(y1)

        # SOC PDOS (down)
        y2 = np.zeros_like(x)
        for j, Ej in enumerate(E2):
            w = float(frac2[j, t_idx])
            if w <= 1e-12: continue
            y2 += w * _gauss(x, Ej, sigma_eV)
        y_soc_types.append(y2)
    y_sf_types = np.array(y_sf_types)
    y_soc_types = np.array(y_soc_types)

    # Sum over atom types for total
    y_sf_total = y_sf_types.sum(axis=0)
    y_soc_total = y_soc_types.sum(axis=0)

    ymax = max(y_sf_total.max(), y_soc_total.max()) * 1.05

    fig, ax = plt.subplots(figsize=(9, 6))
    # Fill total
    ax.fill_between(x, 0, y_sf_total, color="dodgerblue", alpha=0.28, label=f"{label1} (total)")
    ax.fill_between(x, 0, -y_soc_total, color="orangered", alpha=0.28, label=f"{label2} (total)")

    # Overlay per-type lines (optional but helps for interpretation)
    for t_idx, t in enumerate(type_order):
        ax.plot(x,  y_sf_types[t_idx], linestyle="--", color=colors[t], linewidth=1.1, alpha=0.9, label=f"{t} ({label1})")
        ax.plot(x, -y_soc_types[t_idx], linestyle="-",  color=colors[t], linewidth=1.1, alpha=0.9, label=f"{t} ({label2})")

    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.65)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("PDOS (arb. units)\n(up: spin-free, down: SOC)")
    ax.set_title(f"Mirrored PDOS (σ = {sigma_eV:.2f} eV), fill = total, lines = per type")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.4)
    # Compact legend
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
    return fig, ax


def plot_soc_connections(
    energies_sf_Ha, energies_soc_Ha, proj_abs2,
    topk=3, weight_min=0.05, outfile=None, H2EV=27.211386245988
):
    """
    Connection map between spin-free MOs (left column at x=0) and
    SOC spinors (right column at x=1). Draw lines for the top-k
    contributions per spinor with |c|^2 >= weight_min.

    Parameters
    ----------
    energies_sf_Ha : array-like, shape (N_sf,)
        Spin-free MO energies (alpha+beta concatenated for UKS).
    energies_soc_Ha : array-like, shape (N_soc,)
        SOC spinor energies (subset you want to display).
    proj_abs2 : ndarray, shape (N_sf, N_soc)
        |Φ† S_total U|^2 (weights) between spin-free basis Φ and SOC spinors U.
        Columns correspond to the SOC states you are showing.
    """
    E_sf = np.asarray(energies_sf_Ha) * H2EV
    E_soc = np.asarray(energies_soc_Ha) * H2EV
    N_sf = E_sf.shape[0]
    N_sc = E_soc.shape[0]

    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    # Scatter the two columns
    ax.scatter(np.zeros(N_sf), E_sf, marker='_', s=80, linewidths=2, color="#555555", label="spin-free MOs")
    ax.scatter(np.ones(N_sc),  E_soc, marker='_', s=80, linewidths=2, color="#111111", label="SOC spinors")

    # Draw connections
    for j in range(N_sc):
        wcol = proj_abs2[:, j]
        if wcol.size != N_sf:
            raise ValueError("proj_abs2 has wrong shape.")
        idxs = np.argsort(-wcol)[:topk]
        for i in idxs:
            w = float(wcol[i])
            if w < weight_min: 
                continue
            # line alpha and width scale with weight
            alpha = min(0.15 + 0.85*w, 1.0)
            lw    = 0.5 + 3.0*w
            ax.plot([0, 1], [E_sf[i], E_soc[j]], color="#1f77b4", alpha=alpha, linewidth=lw)

    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["spin-free", "SOC"])
    ax.set_ylabel("Energy (eV)")
    ax.set_title("Spin-free ↔ SOC connections (top contributors)")
    ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.legend(frameon=False, loc="lower left")

    if outfile:
        fig.tight_layout()
        fig.savefig(outfile, dpi=200)
    return fig, ax


