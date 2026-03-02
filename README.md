# Emirp Searcher

A GPU-accelerated search engine that was used to find a **world-record 10,069-digit emirp** in a single day. Written entirely by Claude Opus 4.6 in one session.

## What is an Emirp?

An **emirp** (*prime* spelled backwards) is a prime number whose digit-reversal is a different prime. For example, 13 is an emirp because 31 is also prime. This tool by default searches for a 10,069-digit emirp, which is larger than the currently conclusively proven world record.

## Architecture

The search uses a **Hybrid GPU+CPU Twin Boolean Sieve** pipeline:

```
GPU Kernel 1 (mod)       – Compute N mod p and sieve start indices (~203M primes in parallel)
GPU Kernel 2 (sieve)     – Boolean sieve marking composites in forward/reverse arrays
GPU Kernel 3 (inverse)   – Modular inverses via Fermat's little theorem (replaces slow Python pow())
GPU intersection         – Combine forward and reverse sieves; extract survivor indices
Async pipeline           – GPU produces next batch while CPU tests the current one
CPU BPSW pipeline        – Miller-Rabin (base 2 & 3) → full Lucas test on survivors
```

The sieve covers all primes up to 2³² (~203 million primes). An optional extended sieve can reach up to 2³⁴ at the cost of ~4.5 GB of VRAM and much heavier GPU load.

A **ratcheting schedule** progressively searches at increasing digit counts (1,001 → 2,001 → … → 10,069), with configurable time budgets at each level.

## Requirements

- **Python** 3.11+
- **CUDA GPU** with sufficient VRAM (≥ 8 GB recommended; 16 GB+ for extended sieve)
- [**CuPy**](https://cupy.dev/) – GPU array library for CUDA
- [**gmpy2**](https://gmpy2.readthedocs.io/) – GMP-backed arbitrary-precision arithmetic
- [**NumPy**](https://numpy.org/)

Install dependencies:

```bash
pip install cupy-cuda12x gmpy2 numpy
```

> Adjust the `cupy-cuda12x` package name to match your CUDA toolkit version (e.g. `cupy-cuda11x`).

## Usage

```bash
python emirp_search.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--time-limit SECONDS` | 604800 (7 days) | Total wall-clock time budget |
| `--no-resume` | — | Ignore any existing checkpoint and start fresh |
| `--digits N` | — | Jump directly to the schedule level at or above N digits |
| `--extended-sieve N` | 34 | Extended sieve upper limit as a power of 2 (e.g. `33` → 2³³). Pass `0` to disable |

### Examples

```bash
# Full 7-day run with default settings
python emirp_search.py

# Quick 1-hour test run, jumping straight to 5,001 digits
python emirp_search.py --time-limit 3600 --digits 5001

# Restart from scratch without resuming a checkpoint
python emirp_search.py --no-resume

# Disable the extended sieve to save VRAM
python emirp_search.py --extended-sieve 0
```

## Output Files

| File | Description |
|------|-------------|
| `emirp_results.txt` | All emirps found, with full digit representation |
| `checkpoint.json` | Search state — allows resuming interrupted runs |

Both files are written to the same directory as the script.

## Performance Notes

- **GPU warm-up** on first launch compiles CUDA kernels via CuPy's JIT; this takes 10–30 seconds but is only done once per session.
- **VRAM usage** scales with the extended sieve limit: ~4.5 GB at 2³⁴, ~9 GB at 2³⁵, ~19 GB at 2³⁶.
- **RAM usage** is dominated by the prime array: ~800 MB for primes up to 2³².
- The async GPU/CPU pipeline keeps both the GPU and all CPU cores busy simultaneously.
- The BPSW test (Miller-Rabin + Lucas) is implemented via `gmpy2` and parallelised across all CPU cores using `multiprocessing`.

The tool was developed and tested on:
- **CPU:** AMD Ryzen 7 7800X3D
- **GPU:** NVIDIA RTX 4090 (24 GB VRAM)
- **RAM:** 64 GB DDR5

## Search Schedule

The ratcheting digit schedule, with per-level time budgets:

| Digit count | Time budget |
|-------------|-------------|
| 1,001 | 5 minutes |
| 2,001 | 10 minutes |
| 3,001 | 30 minutes |
| 4,001 | 1 hour |
| 5,001 | 2 hours |
| 6,001 | 4 hours |
| **10,069** | **5 days** |

The script advances to the next level only after finding an emirp at the current level, or exhausting that level's budget.

## Primality Verification

Candidates passing BPSW are strong probable primes. For formal certification of a world-record find, use a deterministic prover such as [PRIMO](http://www.ellipsa.eu/) (ECPP) or [PARI/GP](https://pari.math.u-bordeaux.fr/). BPSW has no known composite counterexamples and is sufficient for candidate identification during the search phase.

## Licence

MIT
