# SOC

**Spin–Orbit Coupling and Quantum Chemistry Tools**  
_A hybrid Python/C++ package for advanced spin–orbit coupling (SOC) calculations, leveraging Libint2, Eigen, pybind11, and the scientific Python stack._

---

## ✨ Features

- High‑performance C++ backend (Libint2, Eigen, OpenMP)
- Seamless Python interface via pybind11
- SOC driver to reconstruct KS/Fock in the AO basis and compute spinors
- Optional export of spinors to `.npz` for downstream analysis (e.g., fuzzy bands)
- Easy integration with Python workflows (NumPy, SciPy, Matplotlib, Pymatgen)
- Modern build system — install everything with one command

---

## 📦 Requirements

- Linux, macOS, or WSL (Windows Subsystem for Linux)
- Python ≥ 3.9
- C++17 compiler (provided by conda/mamba)
- [Libint2](https://github.com/evaleev/libint) ≥ 2.6
- [Eigen3](https://eigen.tuxfamily.org/) ≥ 3.4
- [pybind11](https://pybind11.readthedocs.io/) ≥ 2.10  
_(All dependencies are handled via `environment.yml`.)_

---

## ⚙️ Installation

### 1) Create and activate the environment

```bash
mamba env create -f environment.yml
mamba activate soc
```

### 2) Build and install the package

```bash
pip install .
```

This will compile the C++ extension using your environment’s compilers and libraries. After this, you can import the Python modules and the `libint_cpp` extension directly.

---

## 📖 Tutorial: Preparing MOs from CP2K (for SOC)

To reconstruct the KS/Fock matrix in the **AO** basis and compute SOC spinors, you must print **all** relevant MOs (occupied + enough virtuals to cover the unoccupied spectrum you care about). The most robust approach is to **print all MOs** (no `MO_INDEX_RANGE`).

### 1) Single‑point calculation in CP2K

After geometry optimization, perform a **single‑point** calculation with this SCF block:

```fortran
&SCF
  MAX_SCF 1
  EPS_SCF 1.0E-3
  ADDED_MOS 100000
  SCF_GUESS RESTART
#  &OT
#    MINIMIZER DIIS
#    N_DIIS 7
#    PRECONDITIONER FULL_SINGLE_INVERSE
#  &END OT
&END SCF
```

⚠️ **Important**  
- **Do not** use the OT optimizer here (keep it commented).  
- `ADDED_MOS` must be large enough so that **all unoccupied MOs of interest are printed**. When in doubt, overshoot — missing LUMOs cannot be reconstructed later.

### 2) Print MOs to file

Add this to your input (note the **absence** of `MO_INDEX_RANGE` so **all** MOs are printed):

```fortran
&PRINT
  &MO
    &EACH
      QS_SCF 100
    &END
    COEFFICIENTS
    NDIGITS 16
    ADD_LAST NUMERIC
    FILENAME MOs
  &END
&END PRINT
```

- Omitting `MO_INDEX_RANGE` ensures the full MO set is printed, which is required to reliably reconstruct the KS/Fock matrix in the AO basis.

### 3) Clean the `MOs.txt` file

Remove headers and non‑coefficient lines before ingesting into SOC.

**RKS (closed‑shell):**

```bash
#!/bin/bash
# clean_mos_rks.sh
INPUT="MOs.txt"
OUTPUT="MOs_cleaned.txt"

awk '
  BEGIN { skip=0; count=0 }
  /EIGENVALUES/ {
    count++
    if (count == 1) { skip=1; next }
    else if (count == 2) { skip=0; next }
  }
  skip == 0 { print }
' "$INPUT" | \
sed 's/MO|/ /g' | \
grep -v -E '^[[:space:]]*$' | \
grep -v 'E(Fermi)' | \
grep -v 'Band gap' > "$OUTPUT"

echo "✅ Cleaned file written to: $OUTPUT"
```

**UKS (open‑shell):**

```bash
#!/bin/bash
# clean_mos_uks.sh
set -euo pipefail

INPUT="${1:-MOs.txt}"
ALPHA_RAW="MOs_alpha_raw.txt"
BETA_RAW="MOs_beta_raw.txt"
ALPHA_OUT="MOs_alpha.txt"
BETA_OUT="MOs_beta.txt"

awk '
  BEGIN { section=0 }
  /^[[:space:]]*MO\|[[:space:]]*[Aa][Ll][Pp][Hh][Aa]/ { section=1; print > a; next }
  /^[[:space:]]*MO\|[[:space:]]*[Bb][Ee][Tt][Aa]/ { section=2; print > b; next }
  section==1 { print > a }
  section==2 { print > b }
' a="$ALPHA_RAW" b="$BETA_RAW" "$INPUT"

clean_mos() {
  local IN="$1"
  local OUT="$2"
  sed 's/MO|/ /g' "$IN" \
    | grep -viE "alpha|beta" \
    | grep -v -E "^[[:space:]]*$" \
    | grep -v "E(Fermi)" \
    | grep -v "Band gap" > "$OUT"
}

if [ -s "$ALPHA_RAW" ]; then
  clean_mos "$ALPHA_RAW" "$ALPHA_OUT"
else
  echo "Warning: no ALPHA section found in $INPUT" >&2
  : > "$ALPHA_OUT"
fi

if [ -s "$BETA_RAW" ]; then
  clean_mos "$BETA_RAW" "$BETA_OUT"
else
  echo "Note: no BETA section found in $INPUT" >&2
  : > "$BETA_OUT"
fi

rm -f "$ALPHA_RAW" "$BETA_RAW"

echo "Alpha MOs written to: $ALPHA_OUT"
echo "Beta  MOs written to: $BETA_OUT"
```

### 4) First‑time use and caching

Parsing large MO text files can be slow. On first run, SOC will create a cached `.npz` (CSR) file for **fast reuse** in subsequent calculations.

---

## 🚀 Running the SOC driver

Minimal periodic example:

```bash
soc \
  --xyz geom.xyz \
  --mo MOs.txt \            # or MOs_csr.npz (faster on re‑runs)
  --basis-txt BASIS_MOLOPT \
  --basis-name DZVP-MOLOPT-SR-GTH \
  --periodic \
  --A1  2.5128664117413759E+001  0.0000000000000000E+000  0.0000000000000000E+000 \
  --A2 -1.2564332058706880E+001  2.1762061488846783E+001  0.0000000000000000E+000 \
  --A3  0.0000000000000000E+000  0.0000000000000000E+000  3.0619090536956048E+001 \
  --write-spinors \
  --mo-window 100 100
```

### Flag glossary

- `--xyz` : Geometry used in the DFT run (Cartesian, matches the printed MOs).  
- `--mo` : Cleaned MO coefficient file, or its cached `*_csr.npz`.  
- `--basis-txt` : Basis set text file (e.g. `BASIS_MOLOPT`) available in the run folder.  
- `--basis-name` : Basis name used in CP2K (e.g. `DZVP-MOLOPT-SR-GTH`).  
- `--periodic` : Enable for periodic systems.  
- `--A1/--A2/--A3` : Relaxed lattice vectors (only if `--periodic` is set).  
- `--write-spinors` : Store SOC spinors in an `.npz` for downstream use (e.g. fuzzy bands).  
- `--mo-window NOCC NVIRT` : Truncate the MO space to a subspace around the frontier orbitals. See guidance below.

### Choosing a scientifically sound `--mo-window`

`--mo-window nocc nvirt` selects a **subspace** of the full MO space to diagonalize the SOC Hamiltonian:
- `nocc`  = number of **occupied** orbitals kept **below the HOMO**  
- `nvirt` = number of **virtual** orbitals kept **above the LUMO**

This is a standard **subspace (perturbative) treatment** of SOC. Because SOC is **local** and short‑range in typical semiconductors/insulators and molecular systems, a truncated subspace around the frontier often suffices for accurate gaps and near‑edge states. Practical guidance:

- **Target property**:  
  - **HOMO/LUMO gap or near‑edge band splittings** → start with `--mo-window 100 100`.  
  - **Deeper states or core–valence effects** → increase both (e.g. `200 200` or more).
- **Convergence check**: Increase `nocc/nvirt` until observables (gap, SOC splitting, key level ordering) change by **< 1–2 meV** (or your tolerance).  
- **Where truncation can fail**:  
  - **Metals/very small gaps**, strong **near‑degeneracies**, heavy elements with **very strong SOC** coupling far‑from‑edge states → use a **larger** window or consider the **full space** if feasible.
- **Performance**: Dense diagonalization scales ~O(N³) with subspace size. On a laptop, sizes up to ~**1000×1000** are typically fine; for many QD use‑cases, **100–300** per side is already adequate.

---

## 📤 Outputs

- `soc_spinors.npz` (when `--write-spinors`): spinor coefficients and relevant metadata for post‑processing.  
- Console log with timing, convergence, and subspace details.  
- Optional intermediate caches (CSR) for faster MO reads on subsequent runs.

---

## 🧠 Project Structure (typical)

```
SOC/
├── libint/           # C++ extension sources (pybind11, Libint2)
│   ├── CMakeLists.txt
│   ├── bindings.cpp
│   ├── integrals_core.cpp
│   └── integrals_core.hpp
├── src/              # Python modules
│   ├── analysis.py
│   ├── config.py
│   ├── hamiltonian.py
│   ├── parsers.py
│   ├── soc_driver.py
│   └── utils.py
├── environment.yml
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠️ Development & Contributing

- Source code: <https://github.com/nlesc-nano/SOC>
- To contribute, fork the repo and submit a pull request.
- Please open an issue for bug reports, feature requests, or questions.

_Advanced/dev build (normally unnecessary):_

```bash
cd libint
mkdir -p build && cd build
cmake ..
make
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE).

---

## 🧾 Citing

If you use **SOC** in your research, please cite:

> Ivan Infante, SOC: Spin–Orbit Coupling Python/C++ Package (2025)  
> GitHub: <https://github.com/nlesc-nano/SOC>

---

## 🆘 Support

- For help, open a GitHub Issue: <https://github.com/nlesc-nano/SOC/issues>
- For scientific consulting or collaboration, contact: ivan.infante@bcmaterials.net

---

**Quick recap:** CP2K SP (print **all** MOs, no OT) → clean MOs (RKS/UKS scripts) → `soc` with `--mo-window` subspace (converge by increasing NOCC/NVIRT) → optional `--write-spinors` for fuzzy‑band post‑processing.
*Happy computing!*
