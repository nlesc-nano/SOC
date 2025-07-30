# parsers.py
import numpy as np
import collections

def _collect_coeffs_from_iterator(line_iter, n_coeffs_to_get, initial_coeffs):
    """Helper to read coefficients that may span multiple lines."""
    coeffs = list(initial_coeffs)
    while len(coeffs) < n_coeffs_to_get:
        line = next(line_iter).strip()
        if line and not line.startswith('#'):
            coeffs.extend([float(x) for x in line.split()])
    return coeffs

def read_mos_txt(path, n_ao_total):
    """
    A robust parser for CP2K MO output files, designed to handle
    molecular systems with multiple atoms and basis functions.
    """
    with open(path) as f:
        lines = f.readlines()

    all_eps, all_occ, all_c_cols = [], [], []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        tokens = line.split()

        if len(tokens) > 0 and all(t.isdigit() for t in tokens):
            n_cols = len(tokens)
            
            i += 1; eps = [float(e) for e in lines[i].strip().split()]; all_eps.extend(eps)
            i += 1; occ = [float(o) for o in lines[i].strip().split()]; all_occ.extend(occ)
            i += 1

            c_block = np.zeros((n_ao_total, n_cols))
            ao_lines_read = 0
            while ao_lines_read < n_ao_total and i < len(lines):
                coeff_line = lines[i].strip()
                i += 1
                if not coeff_line: continue
                
                parts = coeff_line.split()
                try:
                    coeffs = [float(c) for c in parts[-n_cols:]]
                    if len(coeffs) != n_cols: raise ValueError
                    c_block[ao_lines_read, :] = coeffs
                    ao_lines_read += 1
                except (ValueError, IndexError):
                    raise ValueError(f"Error parsing coefficient line {i}: '{coeff_line}'")
            
            if ao_lines_read != n_ao_total:
                raise RuntimeError(f"Expected {n_ao_total} AO lines, found {ao_lines_read}")

            all_c_cols.append(c_block)
        else:
            i += 1
            
    if not all_c_cols:
        raise RuntimeError("Could not parse any valid MO coefficient blocks.")

    C = np.hstack(all_c_cols)
    print(f"Successfully parsed MOs. Final C matrix shape: {C.shape}")
    return C, np.asarray(all_eps), np.asarray(all_occ)

def parse_basis(fname, wanted):
    """
    A robust parser for MOLOPT basis set files.
    """
    basis = collections.defaultdict(list)
    with open(fname) as f:
        lines = f.readlines()

    line_iter = iter(lines)
    for line in line_iter:
        ln = line.strip()
        if not ln or ln.startswith('#'): continue

        parts = ln.split()
        if len(parts) < 2: continue
        elem, bname = parts[0], parts[1]

        if bname != wanted:
            try:
                nset = int(next(line_iter).strip().split()[0])
                for _ in range(nset):
                    nexp = int(next(line_iter).split()[3])
                    for _ in range(nexp): next(line_iter)
            except (StopIteration, IndexError, ValueError):
                continue
            continue

        try:
            nset = int(next(line_iter).strip().split()[0])
            for _ in range(nset):
                hdr = next(line_iter).split()
                lmin, nexp = int(hdr[1]), int(hdr[3])
                counts = list(map(int, hdr[4:]))
                
                prim_lines = [next(line_iter).split() for _ in range(nexp)]
                exps_full = [float(p[0]) for p in prim_lines]
                coef_rows = [[float(c) for c in p[1:]] for p in prim_lines]
                coef_cols = np.array(coef_rows).T

                if coef_cols.shape[0] != sum(counts):
                    raise ValueError(f"Contraction mismatch for {elem}")
                
                coef_idx = 0
                for j, num_shells in enumerate(counts):
                    l = lmin + j
                    for _ in range(num_shells):
                        current_coeffs = coef_cols[coef_idx]
                        mask = np.abs(current_coeffs) > 1e-12
                        basis[elem].append((l, np.asarray(exps_full)[mask], current_coeffs[mask]))
                        coef_idx += 1
        except (StopIteration, IndexError, ValueError) as e:
            raise IOError(f"FATAL: Could not parse block for '{elem}' basis '{bname}'. Details: {e}")

    return basis

def read_xyz(path):
    """Parses a standard XYZ file."""
    with open(path) as f:
        lines=f.readlines()
    nat=int(lines[0].strip())
    syms=[]; coords=[]
    for l in lines[2:2+nat]:
        p=l.split()
        syms.append(p[0])
        coords.append(tuple(map(float,p[1:4])))
    return syms, np.asarray(coords)

def parse_gth_soc_potentials(path, elements_to_parse):
    """
    An efficient parser for GTH potential files that stops reading
    once all required element blocks have been found.
    """
    ecp_dict = collections.defaultdict(lambda: {'so': []})
    needed_elements = set(elements_to_parse.keys())

    try:
        with open(path, "r") as f:
            line_iter = iter(f.readlines())
    except FileNotFoundError:
        print(f"Warning: Potential file not found at {path}. SOC will be zero.")
        return ecp_dict

    for line in line_iter:
        if not needed_elements: break
        parts = line.strip().split()
        if not parts or parts[0] not in needed_elements: continue
        
        sym, q = parts[0], elements_to_parse.get(parts[0])
        if q is None or not any(f"q{q}" in p for p in parts): continue
        
        print(f"[DEBUG] Found potential block for '{sym}-q{q}'.")
        try:
            next(line_iter); next(line_iter) # Skip header
            n_soc_sets = int(next(line_iter).strip().split()[0])
            for l in range(n_soc_sets):
                proj_line = next(line_iter)
                while not proj_line.strip() or proj_line.strip().startswith('#'):
                    proj_line = next(line_iter)
                proj_parts = proj_line.split()
                r, nprj = float(proj_parts[0]), int(proj_parts[1])
                n_coeffs = nprj * (nprj + 1) // 2
                h = _collect_coeffs_from_iterator(line_iter, n_coeffs, proj_parts[2:])
                k = _collect_coeffs_from_iterator(line_iter, n_coeffs, []) if l > 0 else []
                ecp_dict[sym]['so'].append({'l': l, 'r': r, 'nprj': nprj, 'h_coeffs': h, 'k_coeffs': k})
            needed_elements.remove(sym)
        except (StopIteration, IndexError, ValueError) as e:
            print(f"Warning: Error parsing SOC block for {sym}: {e}")
            continue
            
    return ecp_dict

def read_cp2k_overlap_matrix(filename, n_ao):
    """
    Reads a CP2K-style sparse overlap matrix (1-based indices).
    """
    S = np.zeros((n_ao, n_ao), dtype=float)
    with open(filename) as f:
        for line in f:
            if not line.strip() or line.strip().startswith('#'): continue
            try:
                i, j, val = int(line.split()[0])-1, int(line.split()[1])-1, float(line.split()[2])
                S[i, j] = S[j, i] = val
            except (ValueError, IndexError):
                continue
    return S

