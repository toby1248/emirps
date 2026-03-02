#!/usr/bin/env python3
"""
EMIRP SEARCH ENGINE (Hybrid GPU+CPU Twin Boolean Sieve)
=======================================================
Target: World Record 10,069 Digit Emirp

Architecture:
  1. GPU Kernel 1 (mod): Computes N mod p and sieve start indices for all
     primes in parallel. One thread per prime, ~203M threads.
  2. GPU Kernel 2 (sieve): Boolean sieve marks composites via arithmetic
     progressions. One thread per prime marks alive_k and alive_r arrays.
  3. GPU intersection + extraction: Combines forward/reverse sieves and
     extracts survivor indices. Only survivors transfer to CPU.
  4. GPU Kernel 3 (inverse): Computes modular inverses for all primes via
     Fermat's little theorem on GPU, replacing slow Python pow() loop.
  5. Async pipeline: GPU produces next batch while CPU tests current batch.
  6. CPU Primorial GCD: Products of primes beyond sieve range filter
     survivors before expensive PRP tests.
  7. CPU BPSW Pipeline: MR base-2, MR base-3, then full Lucas test.

Requires: Python 3.11+, gmpy2, numpy, cupy (CUDA GPU)
"""

import sys
sys.set_int_max_str_digits(1_000_000)

import os
import json
import math
import time
import random
import argparse
import multiprocessing as mp
from datetime import datetime

import numpy as np
import gmpy2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SIEVE_PRIMES_MAX = 4_294_000_000  # Sieve with primes up to ~2^32 (~203M primes)
EXTENDED_SIEVE_LIMIT = 2**34           # Extended GPU sieve up to 2^34 (~564M extra primes, 4.5 GB)
                                       # Set to 0 to disable. Accumulator overflow limit: 2^47.
                                       # VRAM limits: 2^34=4.5GB, 2^35=9GB, 2^36=19GB
W = 7                                  # Block size: 10^7 = 10M candidates per block
M_OFFSET = 10                          # Zero-block position from right end
K = 10**W                              # Candidates per block
NUM_WORKERS = max(1, os.cpu_count() or 8)
DEFAULT_TIME_LIMIT = 604_800           # 7 days

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
RESULTS_FILE = os.path.join(SCRIPT_DIR, "emirp_results.txt")
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "checkpoint.json")

# Primorial GCD sieve: disabled when sieving to 2^32 (negligible benefit, high CPU cost)
# With 203M primes in the boolean sieve, extra primes beyond 4.29B catch <1% of composites.
# MR base-2 is far more cost-effective as first CPU filter.
PRIMORIAL_UPPER = 0                   # 0 = disabled
PRIMORIAL_CHUNK_SIZE = 1_000          # Primes per chunk (unused when disabled)

# Ratcheting schedule: (digit_count, time_budget_seconds)
SCHEDULE = [
    (1001,      300),
    (2001,      600),
    (3001,     1800),
    (4001,     3600),
    (5001,     7200),
    (6001,    14400),
    (10069,  432000),
]

TOTAL_TIME_LIMIT = DEFAULT_TIME_LIMIT  # overridden by --time-limit

# ---------------------------------------------------------------------------
# Prime generation (segmented sieve for large limits)
# ---------------------------------------------------------------------------
def generate_primes_up_to(limit):
    """
    Segmented sieve of Eratosthenes up to limit.
    Uses segmented approach to avoid a single huge boolean array.
    Returns numpy uint32 array (excludes 2 and 5).
    """
    if limit < 2:
        return np.array([], dtype=np.uint32)

    # Step 1: Generate small primes up to sqrt(limit) with simple sieve
    sqrt_limit = int(limit**0.5) + 1
    small_sieve = np.ones(sqrt_limit + 1, dtype=bool)
    small_sieve[:2] = False
    for i in range(2, int(sqrt_limit**0.5) + 1):
        if small_sieve[i]:
            small_sieve[i*i::i] = False
    small_primes = np.nonzero(small_sieve)[0].astype(np.int64)

    # Segmented sieve: process in 16M-number chunks to keep memory bounded
    SEGMENT_SIZE = 1 << 24  # 16M per segment (~16MB boolean array)

    result_chunks = []
    for lo in range(0, limit + 1, SEGMENT_SIZE):
        hi = min(lo + SEGMENT_SIZE, limit + 1)
        seg = np.ones(hi - lo, dtype=bool)
        if lo == 0:
            seg[:2] = False

        for p in small_primes:
            p_int = int(p)
            if p_int * p_int >= hi:
                break
            # First multiple of p in [lo, hi)
            start = max(p_int * p_int, ((lo + p_int - 1) // p_int) * p_int)
            start -= lo
            seg[start::p_int] = False

        primes_in_seg = np.nonzero(seg)[0].astype(np.int64) + lo
        result_chunks.append(primes_in_seg)

    all_primes = np.concatenate(result_chunks).astype(np.uint32)

    # Exclude 2 and 5 (handled by endpoint selection)
    return all_primes[(all_primes != 2) & (all_primes != 5)]


# ---------------------------------------------------------------------------
# GPU kernels
# ---------------------------------------------------------------------------
GPU_MOD_KERNEL = r'''
extern "C" __global__
void compute_sieve_starts(
    const unsigned char* __restrict__ fwd_digits,
    const unsigned char* __restrict__ rev_digits,
    const unsigned int*  __restrict__ primes,
    const unsigned int*  __restrict__ inv_k,
    const unsigned int*  __restrict__ inv_r,
    unsigned int*        __restrict__ start_k_out,
    unsigned int*        __restrict__ start_r_out,
    int n_digits, int n_primes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primes) return;

    unsigned int p = primes[idx];
    unsigned long long fwd = 0, rev = 0;
    unsigned int pw = 1;

    // Compute N mod p and rev(N) mod p using on-the-fly powers of 10.
    // No intermediate modulo on accumulators needed:
    // max value = n_digits * 9 * (p-1) < 10069 * 9 * 4.29e9 = 3.89e14 < 2^64.
    for (int d = 0; d < n_digits; d++) {
        fwd += (unsigned long long)fwd_digits[d] * pw;
        rev += (unsigned long long)rev_digits[d] * pw;
        pw = (unsigned long long)pw * 10 % p;
    }

    unsigned int fwd_mod = (unsigned int)(fwd % p);
    unsigned int rev_mod = (unsigned int)(rev % p);

    unsigned int ik = inv_k[idx];
    unsigned int ir = inv_r[idx];

    start_k_out[idx] = fwd_mod == 0 ? 0 :
        (unsigned int)((unsigned long long)(p - fwd_mod) * ik % p);
    start_r_out[idx] = rev_mod == 0 ? 0 :
        (unsigned int)((unsigned long long)(p - rev_mod) * ir % p);
}
'''

GPU_SIEVE_KERNEL = r'''
extern "C" __global__
void boolean_sieve(
    unsigned char* __restrict__ alive_k,
    unsigned char* __restrict__ alive_r,
    const unsigned int* __restrict__ start_k,
    const unsigned int* __restrict__ start_r,
    const unsigned int* __restrict__ primes,
    int n_primes, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primes) return;

    unsigned int p = primes[idx];
    unsigned int sk = start_k[idx];
    unsigned int sr = start_r[idx];

    for (unsigned int j = sk; j < (unsigned int)K; j += p)
        alive_k[j] = 0;

    for (unsigned int j = sr; j < (unsigned int)K; j += p)
        alive_r[j] = 0;
}
'''

GPU_INVERSE_KERNEL = r'''
extern "C" __global__
void compute_inverses(
    const unsigned int* __restrict__ primes,
    unsigned int* __restrict__ inv_k,
    unsigned int* __restrict__ inv_r,
    int n_primes,
    int exp_k,   // M_OFFSET
    int exp_r    // num_digits - M_OFFSET - W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primes) return;

    unsigned int p = primes[idx];
    if (p <= 1) return;

    // Modular exponentiation: base^exp mod p
    // Step 1: Compute 10^exp_k mod p, then invert via Fermat
    // Step 2: Compute 10^exp_r mod p, then invert via Fermat

    // Helper: modpow(base, exp, mod) inlined for each output
    // inv_k = pow(10, -exp_k, p) = pow(pow(10, exp_k, p), p-2, p)
    // inv_r = pow(10, -exp_r, p) = pow(pow(10, exp_r, p), p-2, p)

    // --- Compute 10^exp_k mod p ---
    unsigned long long base, result;
    unsigned int e;

    base = 10ULL % p;
    result = 1;
    e = (unsigned int)exp_k;
    while (e > 0) {
        if (e & 1) result = result * base % p;
        base = base * base % p;
        e >>= 1;
    }
    // result = 10^exp_k mod p. Now invert: result^(p-2) mod p
    base = result;
    result = 1;
    e = p - 2;
    while (e > 0) {
        if (e & 1) result = result * base % p;
        base = base * base % p;
        e >>= 1;
    }
    inv_k[idx] = (unsigned int)result;

    // --- Compute 10^exp_r mod p ---
    base = 10ULL % p;
    result = 1;
    e = (unsigned int)exp_r;
    while (e > 0) {
        if (e & 1) result = result * base % p;
        base = base * base % p;
        e >>= 1;
    }
    // Invert: result^(p-2) mod p
    base = result;
    result = 1;
    e = p - 2;
    while (e > 0) {
        if (e & 1) result = result * base % p;
        base = base * base % p;
        e >>= 1;
    }
    inv_r[idx] = (unsigned int)result;
}
'''

# Extended sieve kernel for primes > 2^32 stored as uint64.
# Combined: compute N mod p, modular inverses, sieve start, and mark alive arrays.
# Since p > K = 10^7, each prime marks at most one position per array.
#
# Overflow analysis for p < 2^47 (accumulator limit at 10069 digits):
#   pw = pw * 10 % p: pw < p, pw*10 < 10p, fits uint64 for p < 2^60.
#   Accumulator: n_digits * 9 * (p-1) < 10069 * 9 * p. For p < 2^47: < 2^64 ✓
#   Practical VRAM limits: 2^34 (4.5 GB), 2^35 (9 GB), 2^36 (19 GB)
#   Start index: (p - mod) * inv. Product may overflow uint64.
#     -> Use mulmod33 via Russian peasant (add+compare, no overflow for p < 2^63).
#     -> Only used in powmod + 2 start index multiplies — NOT in the hot digit chain loop.
GPU_EXTENDED_SIEVE_KERNEL = r'''
// Modular multiply for operands < p < 2^33 via Russian peasant.
// Uses add-and-compare (no division in loop). 33 iterations max.
// Only called in powmod (rare per-thread) and start index (2x per-thread).
// The hot digit chain loop does NOT use this.
__device__ unsigned long long mulmod33(
    unsigned long long a, unsigned long long b, unsigned long long m
) {
    unsigned long long r = 0;
    a %= m;
    while (b > 0) {
        if (b & 1) {
            r += a;
            if (r >= m) r -= m;
        }
        a += a;
        if (a >= m) a -= m;
        b >>= 1;
    }
    return r;
}

// Modular exponentiation using mulmod33
__device__ unsigned long long powmod33(
    unsigned long long base, unsigned long long exp, unsigned long long m
) {
    unsigned long long result = 1;
    base %= m;
    while (exp > 0) {
        if (exp & 1) result = mulmod33(result, base, m);
        base = mulmod33(base, base, m);
        exp >>= 1;
    }
    return result;
}

extern "C" __global__
void extended_sieve(
    const unsigned char* __restrict__ fwd_digits,
    const unsigned char* __restrict__ rev_digits,
    const unsigned long long* __restrict__ primes,
    unsigned char* __restrict__ alive_k,
    unsigned char* __restrict__ alive_r,
    int n_digits, int n_primes, int K, int exp_k, int exp_r
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primes) return;

    unsigned long long p = primes[idx];
    if (p <= 1) return;

    // --- Compute N_fwd mod p and N_rev mod p via digit chain ---
    // No intermediate modulo needed: accumulator < n_digits * 9 * p < 2^53.
    unsigned long long fwd = 0, rev = 0;
    unsigned long long pw = 1;
    for (int d = 0; d < n_digits; d++) {
        fwd += (unsigned long long)fwd_digits[d] * pw;
        rev += (unsigned long long)rev_digits[d] * pw;
        pw = pw * 10 % p;   // pw < p, pw*10 < 10*p < 2^40, fits uint64
    }
    unsigned long long fwd_mod = fwd % p;
    unsigned long long rev_mod = rev % p;

    // --- Compute modular inverses of 10^exp_k and 10^exp_r ---
    unsigned long long ik = powmod33(
        powmod33(10, (unsigned long long)exp_k, p), p - 2, p);
    unsigned long long ir = powmod33(
        powmod33(10, (unsigned long long)exp_r, p), p - 2, p);

    // --- Compute sieve start indices ---
    unsigned long long sk = fwd_mod == 0 ? 0 :
        mulmod33(p - fwd_mod, ik, p);
    unsigned long long sr = rev_mod == 0 ? 0 :
        mulmod33(p - rev_mod, ir, p);

    // --- Mark alive arrays (p > K so at most one mark each) ---
    if (sk < (unsigned long long)K) alive_k[(int)sk] = 0;
    if (sr < (unsigned long long)K) alive_r[(int)sr] = 0;
}
'''

_compiled_mod_kernel = None
_compiled_sieve_kernel = None
_compiled_inverse_kernel = None
_compiled_extended_sieve_kernel = None

def get_mod_kernel():
    global _compiled_mod_kernel
    if _compiled_mod_kernel is None:
        import cupy as cp
        _compiled_mod_kernel = cp.RawKernel(GPU_MOD_KERNEL, 'compute_sieve_starts')
    return _compiled_mod_kernel

def get_sieve_kernel():
    global _compiled_sieve_kernel
    if _compiled_sieve_kernel is None:
        import cupy as cp
        _compiled_sieve_kernel = cp.RawKernel(GPU_SIEVE_KERNEL, 'boolean_sieve')
    return _compiled_sieve_kernel

def get_inverse_kernel():
    global _compiled_inverse_kernel
    if _compiled_inverse_kernel is None:
        import cupy as cp
        _compiled_inverse_kernel = cp.RawKernel(GPU_INVERSE_KERNEL, 'compute_inverses')
    return _compiled_inverse_kernel

def get_extended_sieve_kernel():
    global _compiled_extended_sieve_kernel
    if _compiled_extended_sieve_kernel is None:
        import cupy as cp
        _compiled_extended_sieve_kernel = cp.RawKernel(GPU_EXTENDED_SIEVE_KERNEL, 'extended_sieve')
    return _compiled_extended_sieve_kernel


# ---------------------------------------------------------------------------
# Precomputation (once per digit level)
# ---------------------------------------------------------------------------
def precompute_sieve_data(num_digits, gpu_primes):
    """Precompute modular inverses on GPU and reverse mapping for Twin Boolean Sieve."""
    import cupy as cp

    shift_r = num_digits - M_OFFSET - W
    n_primes = len(gpu_primes)

    print(f"  Precomputing modular inverses for {n_primes:,} primes on GPU...", end="", flush=True)
    t0 = time.perf_counter()

    gpu_inv_k = cp.zeros(n_primes, dtype=cp.uint32)
    gpu_inv_r = cp.zeros(n_primes, dtype=cp.uint32)

    inverse_kernel = get_inverse_kernel()
    threads = 256
    grid = (n_primes + threads - 1) // threads
    inverse_kernel((grid,), (threads,),
                   (gpu_primes, gpu_inv_k, gpu_inv_r,
                    np.int32(n_primes), np.int32(M_OFFSET), np.int32(shift_r)))
    cp.cuda.Stream.null.synchronize()

    print(f" done ({time.perf_counter()-t0:.1f}s)", flush=True)

    print(f"  Building reverse mapping array (W={W})...", end="", flush=True)
    t0 = time.perf_counter()
    a = np.arange(K, dtype=np.int32)
    r_arr = np.zeros_like(a)
    for _ in range(W):
        r_arr = r_arr * 10 + (a % 10)
        a //= 10
    print(f" done ({time.perf_counter()-t0:.1f}s)", flush=True)

    return gpu_inv_k, gpu_inv_r, r_arr


# ---------------------------------------------------------------------------
# Primorial GCD sieve builder
# ---------------------------------------------------------------------------
def generate_primes_in_range(lo, hi, dtype=np.uint32):
    """Generate primes in [lo, hi) using segmented sieve. Returns numpy array of given dtype."""
    if lo >= hi:
        return np.array([], dtype=dtype)

    sqrt_hi = int(hi**0.5) + 1
    # Small primes for sieving
    small_sieve = np.ones(sqrt_hi + 1, dtype=bool)
    small_sieve[:2] = False
    for i in range(2, int(sqrt_hi**0.5) + 1):
        if small_sieve[i]:
            small_sieve[i*i::i] = False
    small_primes = np.nonzero(small_sieve)[0].astype(np.int64)

    SEGMENT_SIZE = 1 << 24  # 16M
    result_chunks = []

    for seg_lo in range(lo, hi, SEGMENT_SIZE):
        seg_hi = min(seg_lo + SEGMENT_SIZE, hi)
        seg = np.ones(seg_hi - seg_lo, dtype=bool)

        for p in small_primes:
            p_int = int(p)
            if p_int * p_int >= seg_hi:
                break
            start = max(p_int * p_int, ((seg_lo + p_int - 1) // p_int) * p_int)
            start -= seg_lo
            seg[start::p_int] = False

        primes_in_seg = np.nonzero(seg)[0].astype(np.int64) + seg_lo
        result_chunks.append(primes_in_seg)

    if not result_chunks:
        return np.array([], dtype=dtype)
    return np.concatenate(result_chunks).astype(dtype)


def build_primorial_chunks(sieve_limit, primorial_upper, chunk_size):
    """Build GCD-sieve chunks from primes beyond the boolean sieve range."""
    print(f"  Generating primes in range [{sieve_limit:,}, {primorial_upper:,})...", end="", flush=True)
    t0 = time.perf_counter()
    extra_primes = generate_primes_in_range(sieve_limit, primorial_upper)
    # Exclude 2 and 5 from extra primes (shouldn't be in range, but be safe)
    extra_primes = extra_primes[(extra_primes != 2) & (extra_primes != 5)]
    print(f" {len(extra_primes):,} primes ({time.perf_counter()-t0:.1f}s)", flush=True)

    if len(extra_primes) == 0:
        return []

    print(f"  Building {(len(extra_primes) + chunk_size - 1) // chunk_size} primorial chunks...", end="", flush=True)
    t0 = time.perf_counter()
    chunks = []
    for i in range(0, len(extra_primes), chunk_size):
        batch = extra_primes[i:i + chunk_size]
        product = gmpy2.mpz(1)
        for p in batch:
            product *= int(p)
        chunks.append(str(product))
    print(f" done ({time.perf_counter()-t0:.1f}s)", flush=True)

    return chunks


# ---------------------------------------------------------------------------
# CPU primality worker
# ---------------------------------------------------------------------------
_primorial_chunks = None

def _worker_init(chunk_strings):
    global _primorial_chunks
    sys.set_int_max_str_digits(1_000_000)
    if chunk_strings:
        _primorial_chunks = [gmpy2.mpz(s) for s in chunk_strings]
    else:
        _primorial_chunks = []


def test_batch(args):
    """
    Test a batch of survivors from one block.

    args: (base_n_str, base_rev_str, scale_k_str, scale_rev_k_str,
           k_values, rev_k_values, num_digits)

    Returns: (results_list, timing_dict)
      results_list: [(is_emirp, n_str, rev_str), ...]
      timing_dict:  {stage: seconds} for conversion, mr2, mr3, bpsw
    """
    global _primorial_chunks

    (base_n_str, base_rev_str, scale_k_str, scale_rev_k_str,
     k_values, rev_k_values, num_digits) = args

    t0 = time.perf_counter()

    # One-time per-batch conversion (shared across all candidates in batch)
    base_n = gmpy2.mpz(base_n_str)
    base_rev = gmpy2.mpz(base_rev_str)
    scale_k = gmpy2.mpz(scale_k_str)
    scale_rev_k = gmpy2.mpz(scale_rev_k_str)

    t_conv = time.perf_counter() - t0
    t_mr2 = 0.0
    t_mr3 = 0.0
    t_bpsw = 0.0
    n_mr2 = 0
    n_mr3 = 0
    n_bpsw = 0
    n_mr2_pass = 0
    n_mr3_pass = 0
    n_bpsw_pass = 0

    results = []
    for k_val, rev_k_val in zip(k_values, rev_k_values):
        n = base_n + int(k_val) * scale_k
        rev_n = base_rev + int(rev_k_val) * scale_rev_k

        # Palindrome check: n == rev_n means it's a palindromic prime, not an emirp
        if n == rev_n:
            results.append((False, None, None))
            continue

        # Primorial GCD sieve (if enabled)
        fail = False
        if _primorial_chunks:
            for chunk in _primorial_chunks:
                if gmpy2.gcd(n, chunk) != 1:
                    fail = True
                    break
                if gmpy2.gcd(rev_n, chunk) != 1:
                    fail = True
                    break
        if fail:
            results.append((False, None, None))
            continue

        # MR base-2
        ts = time.perf_counter()
        pass_mr2 = gmpy2.is_strong_prp(n, 2)
        if pass_mr2:
            pass_mr2 = gmpy2.is_strong_prp(rev_n, 2)
        t_mr2 += time.perf_counter() - ts
        n_mr2 += 1
        if not pass_mr2:
            results.append((False, None, None))
            continue
        n_mr2_pass += 1

        # MR base-3
        ts = time.perf_counter()
        pass_mr3 = gmpy2.is_strong_prp(n, 3)
        if pass_mr3:
            pass_mr3 = gmpy2.is_strong_prp(rev_n, 3)
        t_mr3 += time.perf_counter() - ts
        n_mr3 += 1
        if not pass_mr3:
            results.append((False, None, None))
            continue
        n_mr3_pass += 1

        # Full BPSW (expensive Lucas test)
        ts = time.perf_counter()
        pass_bpsw = gmpy2.is_prime(n)
        if pass_bpsw:
            pass_bpsw = gmpy2.is_prime(rev_n)
        t_bpsw += time.perf_counter() - ts
        n_bpsw += 1
        if not pass_bpsw:
            results.append((False, None, None))
            continue

        # Convert back to string for result (only for actual emirps — very rare)
        n_str = gmpy2.digits(n, 10)
        rev_str = gmpy2.digits(rev_n, 10)
        results.append((True, n_str, rev_str))

    timing = {
        'conv': t_conv, 'mr2': t_mr2, 'mr3': t_mr3, 'bpsw': t_bpsw,
        'n_tested': len(k_values), 'n_mr2': n_mr2, 'n_mr3': n_mr3, 'n_bpsw': n_bpsw,
        'n_mr2_pass': n_mr2_pass, 'n_mr3_pass': n_mr3_pass, 'n_bpsw_pass': n_bpsw_pass,
    }
    return (results, timing)


# ---------------------------------------------------------------------------
# GPU-accelerated multi-block processing
# ---------------------------------------------------------------------------
def generate_block_data(num_digits):
    """Generate random base number for one block. Returns (A, B, fwd_digits, rev_digits)."""
    endpoints = ['1', '3', '7', '9']
    first = random.choice(endpoints)
    last = random.choice(endpoints)

    A_len = num_digits - M_OFFSET - W
    A = first + "".join(random.choices("0123456789", k=A_len - 1))
    B = "".join(random.choices("0123456789", k=M_OFFSET - 1)) + last
    N_str = A + ("0" * W) + B

    # LSD-first order for GPU kernel
    rev_str = N_str[::-1]
    fwd_digits = np.array([ord(c) - 48 for c in rev_str], dtype=np.uint8)
    rev_digits = np.array([ord(c) - 48 for c in N_str], dtype=np.uint8)

    return A, B, fwd_digits, rev_digits


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_count(n):
    """Format large numbers in human-readable form: 1234567 -> '1.23M'."""
    if n >= 1e12:   return f"{n/1e12:.2f}T"
    if n >= 1e9:    return f"{n/1e9:.2f}B"
    if n >= 1e6:    return f"{n/1e6:.2f}M"
    if n >= 1e3:    return f"{n/1e3:.1f}K"
    return str(int(n))


def fmt_time(seconds):
    """Format seconds as HH:MM:SS."""
    hrs, remainder = divmod(int(seconds), 3600)
    mins, secs = divmod(remainder, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def fmt_pct(num, denom):
    """Format a ratio as percentage, handles zero denom."""
    if denom == 0:
        return "- %"
    return f"{num / denom * 100:.1f}%"


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------
class SearchStats:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.elapsed_offset = 0
        self.total_generated = 0
        self.total_sieved = 0
        self.total_tested = 0
        self.emirps_found = 0
        self.best_emirp_digits = 0
        self.best_emirp_n = ""
        self.best_emirp_rev = ""
        self.current_level_digits = 0

    def elapsed(self):
        return self.elapsed_offset + (time.perf_counter() - self.start_time)

    def remaining(self):
        return max(0, TOTAL_TIME_LIMIT - self.elapsed())

    def print_status(self):
        e = self.elapsed()
        rem = self.remaining()
        test_rate = self.total_tested / max(e, 0.001)
        sieve_pct = fmt_pct(self.total_sieved, self.total_generated)
        print(
            f"\r[{fmt_time(e)} elapsed, {fmt_time(rem)} left] "
            f"{self.current_level_digits}d | "
            f"{fmt_count(self.total_generated)} gen -> {fmt_count(self.total_sieved)} sieved ({sieve_pct}) "
            f"-> {fmt_count(self.total_tested)} tested ({test_rate:.0f}/s) | "
            f"best={self.best_emirp_digits}d    ",
            end="", flush=True
        )


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_result(n_str, rev_str, stats):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{'='*72}\n")
        f.write(f"EMIRP FOUND: {len(n_str)} digits\n")
        f.write(f"Timestamp: {ts}\n")
        f.write(f"Search time: {stats.elapsed():.1f}s\n")
        f.write(f"Candidates generated: {stats.total_generated:,}\n")
        f.write(f"Sieve survivors tested: {stats.total_tested:,}\n")
        f.write(f"\nPrime (n):\n{n_str}\n")
        f.write(f"\nReverse prime (rev_n):\n{rev_str}\n")
        f.write(f"{'='*72}\n\n")


# ---------------------------------------------------------------------------
# Checkpoint system
# ---------------------------------------------------------------------------
def save_checkpoint(stats, schedule_index):
    data = {
        "version": 3,
        "schedule_index": schedule_index,
        "best_emirp_digits": stats.best_emirp_digits,
        "best_emirp_n": stats.best_emirp_n,
        "best_emirp_rev": stats.best_emirp_rev,
        "total_generated": stats.total_generated,
        "total_sieved": stats.total_sieved,
        "total_tested": stats.total_tested,
        "emirps_found": stats.emirps_found,
        "elapsed_at_checkpoint": stats.elapsed(),
        "timestamp": datetime.now().isoformat(),
    }
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, CHECKPOINT_FILE)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 3:
            print("  Checkpoint version mismatch, starting fresh.", flush=True)
            return None
        return data
    except (json.JSONDecodeError, KeyError, OSError) as e:
        print(f"  Checkpoint corrupt ({e}), starting fresh.", flush=True)
        return None


def restore_from_checkpoint(stats, checkpoint):
    stats.best_emirp_digits = checkpoint["best_emirp_digits"]
    stats.best_emirp_n = checkpoint["best_emirp_n"]
    stats.best_emirp_rev = checkpoint["best_emirp_rev"]
    stats.total_generated = checkpoint["total_generated"]
    stats.total_sieved = checkpoint["total_sieved"]
    stats.total_tested = checkpoint["total_tested"]
    stats.emirps_found = checkpoint["emirps_found"]
    stats.elapsed_offset = checkpoint.get("elapsed_at_checkpoint", 0)
    return checkpoint["schedule_index"]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def sanity_check():
    known = [(13, 31), (17, 71), (37, 73), (79, 97), (107, 701), (1009, 9001)]
    for n, rev in known:
        assert gmpy2.is_prime(gmpy2.mpz(n))
        assert gmpy2.is_prime(gmpy2.mpz(rev))
        assert str(n)[::-1] == str(rev)
    print("Sanity check passed: known small emirps verified.", flush=True)


# ---------------------------------------------------------------------------
# Search at one digit level (double-buffered GPU/CPU pipeline)
# ---------------------------------------------------------------------------
def _process_batch_result(batch_result, stats, schedule_index):
    """Process one test_batch return value. Returns (tested, found, timing)."""
    results, timing = batch_result
    tested = 0
    found = False
    for is_emirp, n_str, rev_str in results:
        tested += 1
        if is_emirp:
            stats.emirps_found += 1
            nd = len(n_str)
            print(f"\n\n  *** EMIRP FOUND: {nd} digits! ***", flush=True)
            save_result(n_str, rev_str, stats)
            if nd > stats.best_emirp_digits:
                stats.best_emirp_digits = nd
                stats.best_emirp_n = n_str
                stats.best_emirp_rev = rev_str
            save_checkpoint(stats, schedule_index)
            found = True
    return tested, found, timing


def search_at_level(num_digits, time_budget, stats, pool, schedule_index,
                    gpu_primes, gpu_inv_k, gpu_inv_r, gpu_r_arr,
                    gpu_ext_primes=None):
    """
    Search for emirps with exactly num_digits digits.

    Pipeline (streaming with apply_async per batch):
      1. GPU: mod kernel -> sieve kernel -> extended sieve -> intersection -> extract
      2. CPU: precompute binary base ONCE per block, split k-values into chunks
      3. Workers: base_n + k*scale (O(n) add) -> MR-2 -> MR-3 -> BPSW
      4. Results stream back via per-batch AsyncResult for live tested counter
      5. Per-stage timing: conv, mr2, mr3, bpsw accumulated across all workers
    """
    import cupy as cp

    stats.current_level_digits = num_digits
    level_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"SEARCHING: {num_digits}-digit emirps (budget: {time_budget:.0f}s)")
    print(f"{'='*60}", flush=True)

    mod_kernel = get_mod_kernel()
    sieve_kernel = get_sieve_kernel()
    n_primes = len(gpu_primes)
    threads = 256
    grid = (n_primes + threads - 1) // threads

    # Pre-allocate reusable GPU arrays
    gpu_start_k = cp.zeros(n_primes, dtype=cp.uint32)
    gpu_start_r = cp.zeros(n_primes, dtype=cp.uint32)
    alive_k = cp.empty(K, dtype=cp.uint8)
    alive_r = cp.empty(K, dtype=cp.uint8)

    # Precompute scale factors (constant for this digit level)
    scale_k = gmpy2.mpz(10) ** M_OFFSET
    scale_rev_k = gmpy2.mpz(10) ** (num_digits - M_OFFSET - W)
    scale_k_str = str(scale_k)
    scale_rev_k_str = str(scale_rev_k)

    # Reverse mapping for W-digit k values (numpy, reusable)
    rev_map = np.zeros(K, dtype=np.int32)
    a = np.arange(K, dtype=np.int32)
    for _ in range(W):
        rev_map = rev_map * 10 + (a % 10)
        a //= 10

    found = False
    last_status_time = time.perf_counter()
    last_detail_time = time.perf_counter()
    last_checkpoint_time = time.perf_counter()
    CHECKPOINT_INTERVAL = 300
    block_count = 0

    # Timing accumulators (GPU sub-stages)
    t_mod_total = 0.0     # mod kernel: compute N mod p for all primes
    t_sieve_total = 0.0   # sieve kernel: boolean marking
    t_ext_total = 0.0     # extended sieve: primes > 2^32
    t_isect_total = 0.0   # intersection + extract survivors
    t_gpu_total = 0.0     # total GPU wall time per block (includes all above + transfers)
    t_prep_total = 0.0    # CPU prep: base_n computation + batch splitting
    # Worker stage timing accumulators (summed across all workers)
    tw_conv = 0.0
    tw_mr2 = 0.0
    tw_mr3 = 0.0
    tw_bpsw = 0.0
    tw_n_mr2 = 0
    tw_n_mr3 = 0
    tw_n_bpsw = 0
    tw_n_mr2_pass = 0
    tw_n_mr3_pass = 0
    tw_n_bpsw_pass = 0

    # Scale chunk size so each worker completes in ~3-5 seconds.
    # MR base-2 cost per candidate (including n + rev_n) scales as O(d^2).
    # Calibrated: 2001d -> ~72ms/candidate total (2x MR-2 + mpz arithmetic).
    # Model: ms_per_cand ≈ (d / 236)^2
    # Target ~5s per chunk: chunk_size ≈ 5000 / ms_per_cand
    ms_per_cand = max(1.0, (num_digits / 236) ** 2)
    WORKER_CHUNK_SIZE = max(16, min(4096, int(5000 / ms_per_cand)))
    print(f"  Worker chunk size: {WORKER_CHUNK_SIZE} "
          f"(est. {ms_per_cand:.1f}ms/cand, ~{WORKER_CHUNK_SIZE * ms_per_cand / 1000:.1f}s/chunk)",
          flush=True)

    # --- Streaming pipeline using apply_async per batch ---
    # Each batch gets its own AsyncResult, polled with .ready() for non-blocking
    # result retrieval. This gives true per-batch streaming (tested counter
    # updates as each 4096-candidate chunk completes, not after all 16 chunks).
    active_results = []  # list of AsyncResult objects
    MAX_PENDING = 10 * NUM_WORKERS  # max outstanding async results

    first_result_logged = False

    def print_detail_block():
        """Print GPU sub-stage timing + CPU pipeline funnel."""
        bc = max(block_count, 1)
        avg_mod = t_mod_total / bc
        avg_sieve = t_sieve_total / bc
        avg_ext = t_ext_total / bc
        avg_isect = t_isect_total / bc
        avg_gpu = t_gpu_total / bc
        avg_prep = t_prep_total / bc

        level_wall = time.perf_counter() - level_start
        gpu_wall_pct = fmt_pct(t_gpu_total, level_wall)

        ext_str = f" ext={avg_ext:.2f}s" if gpu_ext_primes is not None else ""
        print(f"\n  GPU/blk: mod={avg_mod:.2f}s sieve={avg_sieve:.2f}s{ext_str} "
              f"isect={avg_isect:.2f}s "
              f"(={avg_gpu:.2f}s, {gpu_wall_pct} of wall) | "
              f"prep={avg_prep:.2f}s", flush=True)

        avg_mr2 = tw_mr2 / max(tw_n_mr2, 1) * 1000
        avg_mr3 = tw_mr3 / max(tw_n_mr3, 1) * 1000
        avg_bpsw = tw_bpsw / max(tw_n_bpsw, 1) * 1000
        mr2_pass_pct = fmt_pct(tw_n_mr2_pass, tw_n_mr2)
        mr3_pass_pct = fmt_pct(tw_n_mr3_pass, tw_n_mr3)
        bpsw_pass_pct = fmt_pct(tw_n_bpsw_pass, tw_n_bpsw)

        print(f"  CPU: {fmt_count(tw_n_mr2)} tested "
              f"-> MR2: {avg_mr2:.1f}ms, {fmt_count(tw_n_mr2_pass)} pass ({mr2_pass_pct}) "
              f"-> MR3: {avg_mr3:.1f}ms, {fmt_count(tw_n_mr3_pass)} pass ({mr3_pass_pct}) "
              f"-> BPSW: {avg_bpsw:.1f}ms, {fmt_count(tw_n_bpsw_pass)} pass ({bpsw_pass_pct}) "
              f"| {len(active_results)} pending",
              end="", flush=True)

    def drain_ready_results():
        """Non-blocking drain of completed async results."""
        nonlocal found, tw_conv, tw_mr2, tw_mr3, tw_bpsw
        nonlocal tw_n_mr2, tw_n_mr3, tw_n_bpsw, first_result_logged
        nonlocal tw_n_mr2_pass, tw_n_mr3_pass, tw_n_bpsw_pass

        still_active = []
        n_drained = 0
        for ar in active_results:
            if ar.ready():
                try:
                    batch_result = ar.get(timeout=0)
                    tested, batch_found, timing = _process_batch_result(
                        batch_result, stats, schedule_index)
                    stats.total_tested += tested
                    n_drained += 1
                    # Accumulate worker timing
                    tw_conv += timing['conv']
                    tw_mr2 += timing['mr2']
                    tw_mr3 += timing['mr3']
                    tw_bpsw += timing['bpsw']
                    tw_n_mr2 += timing['n_mr2']
                    tw_n_mr3 += timing['n_mr3']
                    tw_n_bpsw += timing['n_bpsw']
                    tw_n_mr2_pass += timing['n_mr2_pass']
                    tw_n_mr3_pass += timing['n_mr3_pass']
                    tw_n_bpsw_pass += timing['n_bpsw_pass']
                    if not first_result_logged and timing['n_mr2'] > 0:
                        actual_ms = timing['mr2'] / timing['n_mr2'] * 1000
                        print(f"\n  First batch: {tested} candidates, "
                              f"MR2={actual_ms:.1f}ms/test, "
                              f"conv={timing['conv']*1000:.1f}ms, "
                              f"{timing['n_mr2_pass']} passed MR2",
                              flush=True)
                        first_result_logged = True
                    if batch_found:
                        found = True
                        return
                except Exception as e:
                    print(f"\n  [DRAIN ERROR: {e}]", flush=True)
            else:
                still_active.append(ar)
        active_results.clear()
        active_results.extend(still_active)

    try:
        while True:
            level_elapsed = time.perf_counter() - level_start
            if level_elapsed >= time_budget or stats.remaining() <= 0:
                print(f"\n  Time budget exhausted for {num_digits}-digit search.", flush=True)
                break

            # --- GPU: produce one block's survivors ---
            t_gpu_start = time.perf_counter()

            A, B, fwd_digits, rev_digits = generate_block_data(num_digits)
            gpu_fwd = cp.asarray(fwd_digits)
            gpu_rev = cp.asarray(rev_digits)

            t_mod_start = time.perf_counter()
            mod_kernel((grid,), (threads,),
                       (gpu_fwd, gpu_rev, gpu_primes, gpu_inv_k, gpu_inv_r,
                        gpu_start_k, gpu_start_r,
                        np.int32(num_digits), np.int32(n_primes)))
            cp.cuda.Stream.null.synchronize()
            t_mod_total += time.perf_counter() - t_mod_start

            alive_k.fill(1)
            alive_r.fill(1)
            t_sieve_start = time.perf_counter()
            sieve_kernel((grid,), (threads,),
                         (alive_k, alive_r, gpu_start_k, gpu_start_r, gpu_primes,
                          np.int32(n_primes), np.int32(K)))
            cp.cuda.Stream.null.synchronize()
            t_sieve_total += time.perf_counter() - t_sieve_start

            # Extended sieve: primes > 2^32 (uint64), combined mod+sieve kernel
            if gpu_ext_primes is not None:
                t_ext_start = time.perf_counter()
                ext_kernel = get_extended_sieve_kernel()
                n_ext = len(gpu_ext_primes)
                ext_grid = (n_ext + threads - 1) // threads
                shift_r = num_digits - M_OFFSET - W
                ext_kernel((ext_grid,), (threads,),
                           (gpu_fwd, gpu_rev, gpu_ext_primes,
                            alive_k, alive_r,
                            np.int32(num_digits), np.int32(n_ext), np.int32(K),
                            np.int32(M_OFFSET), np.int32(shift_r)))
                cp.cuda.Stream.null.synchronize()
                t_ext_total += time.perf_counter() - t_ext_start

            t_isect_start = time.perf_counter()
            combined = alive_k & alive_r[gpu_r_arr]
            survivors_gpu = cp.nonzero(combined)[0]
            survivors_idx = survivors_gpu.get()
            del gpu_fwd, gpu_rev, combined, survivors_gpu
            cp.cuda.Stream.null.synchronize()
            t_isect_total += time.perf_counter() - t_isect_start

            t_gpu_end = time.perf_counter()
            t_gpu_total += t_gpu_end - t_gpu_start

            stats.total_generated += K
            stats.total_sieved += len(survivors_idx)
            block_count += 1

            # --- Precompute binary bases for this block (ONCE) ---
            t_prep_start = time.perf_counter()

            base_n_str = str(gmpy2.mpz(A) * gmpy2.mpz(10) ** (W + M_OFFSET) + gmpy2.mpz(B))

            rev_A = A[::-1]
            rev_B = B[::-1]
            base_rev_str = str(gmpy2.mpz(rev_B) * gmpy2.mpz(10) ** (num_digits - M_OFFSET)
                               + gmpy2.mpz(rev_A))

            k_arr = survivors_idx.astype(np.int64)
            rev_k_arr = rev_map[survivors_idx].astype(np.int64)

            batches = []
            for i in range(0, len(k_arr), WORKER_CHUNK_SIZE):
                chunk_k = k_arr[i:i + WORKER_CHUNK_SIZE].tolist()
                chunk_rev_k = rev_k_arr[i:i + WORKER_CHUNK_SIZE].tolist()
                batches.append((base_n_str, base_rev_str, scale_k_str, scale_rev_k_str,
                                chunk_k, chunk_rev_k, num_digits))

            t_prep_end = time.perf_counter()
            t_prep_total += t_prep_end - t_prep_start

            # Submit each batch individually for true per-batch streaming
            for batch in batches:
                ar = pool.apply_async(test_batch, (batch,))
                active_results.append(ar)

            # --- Drain completed CPU results (non-blocking) ---
            drain_ready_results()
            if found:
                break

            # --- Status update after every block ---
            now = time.perf_counter()
            stats.print_status()

            # Detail block: GPU sub-stage timing + CPU funnel (every 10s or first block)
            if now - last_detail_time >= 10.0 or block_count == 1:
                print_detail_block()
                last_detail_time = now

            last_status_time = now

            # Backpressure: if too many batches pending, block-drain until below threshold
            while len(active_results) > MAX_PENDING:
                time.sleep(0.1)
                drain_ready_results()
                if found:
                    break
                # Update status during backpressure waits (every 2s, detail every 30s)
                bp_now = time.perf_counter()
                if bp_now - last_status_time >= 2.0:
                    stats.print_status()
                    if bp_now - last_detail_time >= 30.0:
                        print_detail_block()
                        last_detail_time = bp_now
                    last_status_time = bp_now
            if found:
                break
            if now - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                save_checkpoint(stats, schedule_index)
                last_checkpoint_time = now

    finally:
        n_pending = len(active_results)
        if found and n_pending > 0:
            # Emirp found — skip expensive drain, discard pending work
            print(f"\n  Emirp found, discarding {n_pending} pending batches.", flush=True)
        elif n_pending > 0:
            # No emirp yet — drain remaining for accurate tested count
            print(f"\n  Draining {n_pending} pending batches...", end="", flush=True)
            t_drain_start = time.perf_counter()
            DRAIN_TIMEOUT = 120
            drained = 0
            for ar in active_results:
                remaining_time = DRAIN_TIMEOUT - (time.perf_counter() - t_drain_start)
                if remaining_time <= 0:
                    print(f" timeout ({drained}/{n_pending})", end="", flush=True)
                    break
                try:
                    batch_result = ar.get(timeout=remaining_time)
                    tested, batch_found, timing = _process_batch_result(
                        batch_result, stats, schedule_index)
                    stats.total_tested += tested
                    drained += 1
                    if batch_found and not found:
                        found = True
                except Exception:
                    drained += 1
            t_drain = time.perf_counter() - t_drain_start
            print(f" done ({t_drain:.1f}s)", flush=True)
        active_results.clear()

        del gpu_start_k, gpu_start_r, alive_k, alive_r
        save_checkpoint(stats, schedule_index)

    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global TOTAL_TIME_LIMIT, EXTENDED_SIEVE_LIMIT

    parser = argparse.ArgumentParser(description="EMIRP Search Engine - World Record Edition")
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT,
                        help=f"Total time limit in seconds (default: {DEFAULT_TIME_LIMIT})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore checkpoint and start fresh")
    parser.add_argument("--digits", type=int, default=None,
                        help="Jump directly to a specific digit count")
    parser.add_argument("--extended-sieve", type=int, default=None,
                        help="Extended sieve upper limit as power of 2 (e.g. 33 for 2^33). 0 to disable.")
    args = parser.parse_args()
    TOTAL_TIME_LIMIT = args.time_limit
    if args.extended_sieve is not None:
        EXTENDED_SIEVE_LIMIT = 0 if args.extended_sieve == 0 else 2 ** args.extended_sieve

    print("EMIRP SEARCH ENGINE -- Hybrid GPU+CPU Twin Boolean Sieve")
    print("=" * 60)
    print(f"Workers:    {NUM_WORKERS}")
    print(f"Block size: {K:,} candidates (W={W})")
    print(f"Max prime:  {NUM_SIEVE_PRIMES_MAX:,}")
    if EXTENDED_SIEVE_LIMIT > NUM_SIEVE_PRIMES_MAX:
        ext_bits = int(math.log2(EXTENDED_SIEVE_LIMIT))
        print(f"Ext sieve:  up to 2^{ext_bits} ({EXTENDED_SIEVE_LIMIT:,})")
    else:
        print(f"Ext sieve:  disabled")
    print(f"Time limit: {TOTAL_TIME_LIMIT:,}s ({TOTAL_TIME_LIMIT/3600:.1f} hours)")
    print(f"Target:     {SCHEDULE[-1][0]:,} digits")
    print(f"Results:    {RESULTS_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print("=" * 60, flush=True)

    sanity_check()

    # GPU check
    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        free, total = dev.mem_info
        print(f"GPU: {dev.attributes['MultiProcessorCount']} SMs, "
              f"{total/1e9:.1f} GB total, {free/1e9:.1f} GB free", flush=True)
    except Exception as e:
        print(f"GPU error: {e}")
        sys.exit(1)

    # Generate sieve primes (once)
    print(f"\nGenerating primes up to {NUM_SIEVE_PRIMES_MAX:,}...", end="", flush=True)
    t0 = time.perf_counter()
    all_primes = generate_primes_up_to(NUM_SIEVE_PRIMES_MAX)
    print(f" {len(all_primes):,} primes ({time.perf_counter()-t0:.1f}s)", flush=True)

    # Generate extended sieve primes (uint64, primes beyond 2^32)
    if EXTENDED_SIEVE_LIMIT > NUM_SIEVE_PRIMES_MAX:
        print(f"\nGenerating extended sieve primes [{NUM_SIEVE_PRIMES_MAX:,}, {EXTENDED_SIEVE_LIMIT:,})...",
              end="", flush=True)
        t0 = time.perf_counter()
        ext_primes = generate_primes_in_range(NUM_SIEVE_PRIMES_MAX, EXTENDED_SIEVE_LIMIT, dtype=np.uint64)
        ext_primes = ext_primes[(ext_primes != 2) & (ext_primes != 5)]
        t_ext = time.perf_counter() - t0
        # Estimate survivor reduction using Mertens: 1 - [ln(lo)/ln(hi)]^2
        reduction_pct = (1 - (math.log(NUM_SIEVE_PRIMES_MAX) / math.log(EXTENDED_SIEVE_LIMIT)) ** 2) * 100
        ext_mem_gb = len(ext_primes) * 8 / 1e9
        print(f" {len(ext_primes):,} primes ({t_ext:.1f}s, {ext_mem_gb:.2f} GB, "
              f"est. {reduction_pct:.1f}% fewer survivors)", flush=True)
    else:
        ext_primes = np.array([], dtype=np.uint64)
        print(f"\nExtended sieve: disabled", flush=True)

    # Build primorial GCD chunks (if enabled)
    if PRIMORIAL_UPPER > NUM_SIEVE_PRIMES_MAX:
        print(f"\nBuilding primorial GCD chunks ({NUM_SIEVE_PRIMES_MAX:,} to {PRIMORIAL_UPPER:,})...", flush=True)
        t0 = time.perf_counter()
        chunk_strings = build_primorial_chunks(NUM_SIEVE_PRIMES_MAX, PRIMORIAL_UPPER, PRIMORIAL_CHUNK_SIZE)
        print(f"  Total: {len(chunk_strings)} chunks ({time.perf_counter()-t0:.1f}s)", flush=True)
    else:
        chunk_strings = []
        print(f"\nPrimorial GCD: disabled (sieve covers primes to {NUM_SIEVE_PRIMES_MAX:,})", flush=True)

    # Checkpoint / Resume
    stats = SearchStats()
    start_schedule_index = 0

    if not args.no_resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            start_schedule_index = restore_from_checkpoint(stats, checkpoint)
            print(f"\nRESUMING from checkpoint:")
            print(f"  Schedule index: {start_schedule_index} "
                  f"({SCHEDULE[min(start_schedule_index, len(SCHEDULE)-1)][0]} digits)")
            print(f"  Best emirp: {stats.best_emirp_digits} digits")
            print(f"  Previous elapsed: {stats.elapsed_offset:.0f}s")
            print(f"  Emirps found so far: {stats.emirps_found}")
        else:
            print("\nNo checkpoint found, starting fresh.")
    else:
        print("\n--no-resume: starting fresh.")

    # Handle --digits flag
    if args.digits:
        for i, (d, _) in enumerate(SCHEDULE):
            if d >= args.digits:
                start_schedule_index = i
                break
        print(f"  Jumping to schedule index {start_schedule_index} "
              f"({SCHEDULE[start_schedule_index][0]} digits)")

    # Create worker pool
    pool = mp.Pool(NUM_WORKERS, initializer=_worker_init, initargs=(chunk_strings,))

    # Upload primes to GPU (reused across all digit levels)
    gpu_primes = cp.asarray(all_primes)
    gpu_ext_primes = cp.asarray(ext_primes) if len(ext_primes) > 0 else None

    # JIT-warmup all GPU kernels to avoid compile latency during search
    print("\nWarming up GPU kernels...", end="", flush=True)
    t0 = time.perf_counter()
    _ = get_mod_kernel()
    _ = get_sieve_kernel()
    _ = get_inverse_kernel()
    if gpu_ext_primes is not None:
        # Force compilation of extended sieve kernel with a tiny launch
        ext_k = get_extended_sieve_kernel()
        dummy_digits = cp.zeros(10, dtype=cp.uint8)
        dummy_primes = cp.array([7], dtype=cp.uint64)
        dummy_alive = cp.ones(K, dtype=cp.uint8)
        ext_k((1,), (1,), (dummy_digits, dummy_digits, dummy_primes,
                           dummy_alive, dummy_alive,
                           np.int32(10), np.int32(1), np.int32(K),
                           np.int32(M_OFFSET), np.int32(1)))
        cp.cuda.Stream.null.synchronize()
        del dummy_digits, dummy_primes, dummy_alive
    print(f" done ({time.perf_counter()-t0:.1f}s)", flush=True)

    try:
        for idx in range(start_schedule_index, len(SCHEDULE)):
            level_digits, level_budget = SCHEDULE[idx]
            remaining = stats.remaining()
            if remaining < 30:
                print(f"\nLess than 30s remaining, stopping.", flush=True)
                break

            # Precompute modular inverses (GPU) and reverse mapping for this digit level
            gpu_inv_k, gpu_inv_r, r_arr = precompute_sieve_data(level_digits, gpu_primes)
            gpu_r_arr = cp.asarray(r_arr)

            effective_budget = min(level_budget, remaining)
            found = search_at_level(
                level_digits, effective_budget, stats, pool, idx,
                gpu_primes, gpu_inv_k, gpu_inv_r, gpu_r_arr,
                gpu_ext_primes
            )

            del gpu_inv_k, gpu_inv_r, gpu_r_arr
            cp.get_default_memory_pool().free_all_blocks()

            save_checkpoint(stats, idx + 1)

            if found:
                print(f"\n  Ratcheting up from {level_digits} digits...", flush=True)
            else:
                print(f"\n  No emirp at {level_digits} digits within budget.", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", flush=True)
        save_checkpoint(stats, start_schedule_index)
    finally:
        pool.terminate()
        pool.join()
        del gpu_primes
        if gpu_ext_primes is not None:
            del gpu_ext_primes
        cp.get_default_memory_pool().free_all_blocks()

    # Final summary
    elapsed = stats.elapsed()
    sieve_pct = (stats.total_sieved / stats.total_generated * 100
                 ) if stats.total_generated > 0 else 0
    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print(f"Total time:    {fmt_time(elapsed)} ({elapsed:.1f}s)")
    print(f"Generated:     {stats.total_generated:>14,}")
    print(f"Sieve pass:    {stats.total_sieved:>14,}  ({sieve_pct:.4f}%)")
    print(f"CPU tested:    {stats.total_tested:>14,}")
    print(f"Emirps found:  {stats.emirps_found:>14}")
    if stats.best_emirp_digits > 0:
        print(f"\nLARGEST EMIRP: {stats.best_emirp_digits} digits")
        print(f"First 80 chars: {stats.best_emirp_n[:80]}...")
        print(f"Last  80 chars: ...{stats.best_emirp_n[-80:]}")
        print(f"Results saved to: {RESULTS_FILE}")
    else:
        print("\nNo emirps found.")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
