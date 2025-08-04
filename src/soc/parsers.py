# parsers.py
import collections
import os, re
import time
import numpy as np
from scipy.sparse import issparse, csr_matrix, save_npz, load_npz

import re
import numpy as np

_NUM_RE = re.compile(r"""[\+\-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][\+\-]?\d+)?""", re.VERBOSE)

def save_mo_csr_singlefile(C, eps, occ, outpath):
    """
    Store C (CSR or dense), eps, occ in a single .npz file.
    Converts dense C to CSR automatically.
    """
    if not issparse(C):
        C = csr_matrix(C)
    np.savez_compressed(
        outpath,
        data=C.data,
        indices=C.indices,
        indptr=C.indptr,
        shape=C.shape,
        eps=eps,
        occ=occ
    )
    print(f"[MOs] Wrote C, eps, occ to: {outpath}")

def read_mos_auto(path, n_ao_total, verbose=False):
    from scipy import sparse
    import numpy as np
    import os

    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npz":
        if verbose:
            print(f"[MOs] Detected .npz: {path}")
        d = np.load(path)
        C = sparse.csr_matrix((d['data'], d['indices'], d['indptr']), shape=d['shape'])
        eps = d['eps']
        occ = d['occ']
        if verbose:
            print(f"[MOs] Loaded C shape: {C.shape}, eps: {eps.shape}, occ: {occ.shape}")
        return C, eps, occ
    else:
        # Otherwise parse text, then save .npz for next time
        C, eps, occ = read_mos_txt_streaming(path, n_ao_total, verbose=verbose)
        # Write for future use
        outdir = os.path.dirname(os.path.abspath(path))
        base = os.path.splitext(os.path.basename(path))[0]
        outpath = os.path.join(outdir, base + "_csr.npz")
        save_mo_csr_singlefile(C, eps, occ, outpath)
        return C, eps, occ

def _extract_numbers(s: str):
    # Robust fallback (handles labels and D exponents)
    toks = _NUM_RE.findall(s)
    return [float(t.replace('D','E').replace('d','E')) for t in toks]

def _fast_tail_tokens(line: str, n_cols: int):
    """
    Fast path: grab the last n_cols whitespace-separated tokens from the line.
    Assumes those tokens are numeric (possibly with D exponents).
    Returns a list[str] of length n_cols or fewer (if the line is wrapped).
    """
    # Normalize D/d exponents for speed
    s = line.replace('D','E').replace('d','E').strip()
    if not s:
        return []
    parts = s.rsplit(None, n_cols)  # split from right, at most n_cols+1 pieces
    if len(parts) == 1:
        # entire line is <= n_cols tokens; we don't know how many -> tokenise fully
        toks = parts[0].split()
        return toks[-n_cols:]
    # parts[-n_cols:] are the last n_cols tokens (but they are separate strings).
    # If len(parts) == n_cols+1, last n_cols tokens are already separated.
    # If len(parts) < n_cols+1, the first element may contain multiple tokens.
    tail_tokens = []
    # The rightmost piece is the last token; walk from right to left until we have n_cols tokens.
    for i in range(len(parts)-1, -1, -1):
        seg = parts[i].split()
        # append in reverse so overall order remains left->right when we reverse later
        tail_tokens.extend(reversed(seg))
        if len(tail_tokens) >= n_cols:
            break
    tail_tokens = list(reversed(tail_tokens))[:n_cols]
    return tail_tokens

def _parse_tail_floats(line: str, n_cols: int):
    """
    Try the fast tail token path; if count < n_cols, return partial (caller may wrap).
    If a token fails to float(), fall back to regex on the whole line.
    """
    toks = _fast_tail_tokens(line, n_cols)
    if not toks:
        return []
    try:
        arr = np.array(toks, dtype=float)
        return arr
    except ValueError:
        # Fallback: robust but slower
        nums = _extract_numbers(line)
        # take last n_cols numbers
        if len(nums) >= n_cols:
            return np.asarray(nums[-n_cols:], dtype=float)
        return np.asarray(nums, dtype=float)

def read_mos_txt_streaming(path, n_ao_total, *,
                           dtype=np.float32,
                           mmap_path=None,
                           return_memmap=True,
                           verbose=True,
                           debug=False,
                           log_every=200):
    """
    Fast streaming CP2K MO parser with progress and timing info.
    """
    if mmap_path is None:
        mmap_path = os.path.join(os.path.dirname(os.path.abspath(path)), "C_memmap.dat")

    def _is_int_line(s: str) -> bool:
        toks = s.split()
        if not toks: return False
        for t in toks:
            if t.startswith('+'): t = t[1:]
            if not t.isdigit(): return False
        return True

    def _next_nonempty(f):
        for line in f:
            s = line.strip()
            if s:
                return s
        return None

    file_size = os.path.getsize(path) if os.path.exists(path) else 0
    t0 = time.perf_counter()

    # PASS 1: count blocks and widths
    n_mo_total = 0; blocks = []
    with open(path, 'r', buffering=1024*1024) as f:
        while True:
            line = f.readline()
            if not line: break
            s = line.strip()
            if not _is_int_line(s): continue
            n_cols = len(s.split())
            if _next_nonempty(f) is None:
                raise RuntimeError("Unexpected EOF while reading energies line (pass1).")
            if _next_nonempty(f) is None:
                raise RuntimeError("Unexpected EOF while reading occupations line (pass1).")
            ao_count = 0
            while ao_count < n_ao_total:
                coeff_line = f.readline()
                if not coeff_line: raise RuntimeError("Unexpected EOF in coefficients block (pass1).")
                if coeff_line.strip(): ao_count += 1
            blocks.append((n_mo_total, n_cols))
            n_mo_total += n_cols

    if n_mo_total == 0:
        raise RuntimeError("No MO blocks detected in file.")

    order = 'F'
    if return_memmap:
        C = np.memmap(mmap_path, mode='w+', dtype=dtype, shape=(n_ao_total, n_mo_total), order=order)
    else:
        C = np.empty((n_ao_total, n_mo_total), dtype=dtype, order=order)
    eps = np.empty(n_mo_total, dtype=np.float64)
    occ = np.empty(n_mo_total, dtype=np.float64)

    # PASS 2: parse & fill, now with progress reporting
    with open(path, 'r', buffering=1024*1024) as f:
        blk_idx = 0
        n_blocks = len(blocks)
        print(f"[MOs] Starting parse of {n_mo_total} MOs from {os.path.basename(path)}...")
        t1 = time.perf_counter()
        while True:
            line = f.readline()
            if not line: break
            s = line.strip()
            if not _is_int_line(s): continue

            offset, n_cols = blocks[blk_idx]; blk_idx += 1
            col_slice = slice(offset, offset + n_cols)

            # energies (fast path; allow wrapping)
            vals = []
            while len(vals) < n_cols:
                e_line = _next_nonempty(f)
                if e_line is None:
                    raise RuntimeError(f"EOF reading energies at block {blk_idx}.")
                arr = _parse_tail_floats(e_line, n_cols - len(vals))
                if arr.size == 0:
                    nums = _extract_numbers(e_line)
                    arr = np.asarray(nums, dtype=float) if nums else np.array([], float)
                vals.extend(arr.tolist())
            eps[col_slice] = np.asarray(vals[:n_cols], dtype=np.float64)

            # occupations
            vals = []
            while len(vals) < n_cols:
                o_line = _next_nonempty(f)
                if o_line is None:
                    raise RuntimeError(f"EOF reading occupations at block {blk_idx}.")
                arr = _parse_tail_floats(o_line, n_cols - len(vals))
                if arr.size == 0:
                    nums = _extract_numbers(o_line)
                    arr = np.asarray(nums, dtype=float) if nums else np.array([], float)
                vals.extend(arr.tolist())
            occ[col_slice] = np.asarray(vals[:n_cols], dtype=np.float64)

            # coefficients: n_ao_total lines
            ao_row = 0
            while ao_row < n_ao_total:
                coeff_line = f.readline()
                if not coeff_line:
                    raise RuntimeError(f"EOF in coefficients block at block {blk_idx}, ao_row {ao_row}.")
                sline = coeff_line.strip()
                if not sline: continue

                # fast tail (n_cols tokens); if short, we try to wrap
                tail = _parse_tail_floats(sline, n_cols)
                if tail.size < n_cols:
                    acc = tail.tolist()
                    while len(acc) < n_cols:
                        extra = _next_nonempty(f)
                        if extra is None:
                            raise RuntimeError(f"EOF while wrapping coeff line at block {blk_idx}, ao_row {ao_row}.")
                        more = _parse_tail_floats(extra, n_cols - len(acc))
                        if more.size == 0:
                            more = np.asarray(_extract_numbers(extra), dtype=float)
                        acc.extend(more.tolist())
                    tail = np.asarray(acc[-n_cols:], dtype=float)

                C[ao_row, col_slice] = tail.astype(dtype, copy=False)
                ao_row += 1

            # Progress report
            if (blk_idx % log_every == 0) or (blk_idx == 1) or (blk_idx == n_blocks):
                now = time.perf_counter()
                n_done = offset + n_cols
                mb = file_size / (1024 ** 2) if file_size else 0.0
                elapsed = now - t1
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (n_mo_total - n_done) / rate if rate > 0 else 0
                print(f"[MOs] Block {blk_idx}/{n_blocks} | {n_done}/{n_mo_total} MOs "
                      f"({100*n_done/n_mo_total:.1f}%) | "
                      f"Elapsed: {elapsed:6.1f}s | "
                      f"ETA: {eta:6.1f}s")

    if isinstance(C, np.memmap):
        C.flush()

    t2 = time.perf_counter()
    if verbose:
        mb = file_size / (1024**2) if file_size else 0.0
        print(f"[MOs] Finished parse in {t2-t0:.1f} seconds.")
        print(f"[MOs] Parsed: C shape={C.shape} (dtype={dtype.__name__}, order={order}), eps={eps.shape}, occ={occ.shape})")
        if isinstance(C, np.memmap):
            print(f"[MOs] C memmap: {mmap_path}")
        print(f"[MOs] Input size ~ {mb:,.1f} MB")

    return C, eps, occ


def write_c_to_csr(C, *, threshold=0.0, outpath="C_csr.npz",
                   coeff_dtype=np.float32, verbose=True, chunk_rows=2048):
    """
    Convert dense/memmap C (n_ao x n_mo) to CSR on disk with absolute threshold |c|>=threshold.
    Two-pass over rows (count -> fill) to preallocate exactly.
    """
    t0 = time.perf_counter()
    n_ao, n_mo = C.shape

    # Pass 1: count nnz per row
    indptr = np.empty(n_ao + 1, dtype=np.int64)
    indptr[0] = 0
    nnz = 0
    for r0 in range(0, n_ao, chunk_rows):
        r1 = min(n_ao, r0 + chunk_rows)
        block = C[r0:r1, :]
        if threshold > 0.0:
            nz_per_row = np.count_nonzero(np.abs(block) >= threshold, axis=1)
        else:
            nz_per_row = np.full(r1 - r0, n_mo, dtype=np.int64)
        for i, k in enumerate(nz_per_row, start=r0):
            nnz += int(k)
            indptr[i + 1] = nnz

    # Preallocate
    indices = np.empty(nnz, dtype=np.int32)
    data    = np.empty(nnz, dtype=coeff_dtype)

    # Pass 2: fill
    pos = 0
    for r0 in range(0, n_ao, chunk_rows):
        r1 = min(n_ao, r0 + chunk_rows)
        block = C[r0:r1, :]
        if threshold > 0.0:
            mask = np.abs(block) >= threshold
            for i in range(r1 - r0):
                row_vals = block[i, :]
                row_mask = mask[i, :]
                nn = int(row_mask.sum())
                if nn:
                    cols = np.nonzero(row_mask)[0]
                    indices[pos:pos+nn] = cols.astype(np.int32, copy=False)
                    data[pos:pos+nn]    = row_vals[row_mask].astype(coeff_dtype, copy=False)
                    pos += nn
        else:
            for i in range(r1 - r0):
                indices[pos:pos+n_mo] = np.arange(n_mo, dtype=np.int32)
                data[pos:pos+n_mo]    = block[i, :].astype(coeff_dtype, copy=False)
                pos += n_mo

    C_csr = csr_matrix((data, indices, indptr), shape=(n_ao, n_mo))
    save_npz(outpath, C_csr)
    secs = time.perf_counter() - t0

    if verbose:
        dens = nnz / (n_ao * n_mo)
        print(f"[CSR] Wrote {outpath} | shape={n_ao}x{n_mo}, nnz={nnz:,}, density={dens:.4e}, time={secs:.2f}s")

    return outpath, {'nnz': int(nnz), 'density': nnz/(n_ao*n_mo), 'seconds': secs}



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

