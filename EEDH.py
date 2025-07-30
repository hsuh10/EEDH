import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from scipy.integrate import quad
import scipy.special
from functools import lru_cache
from scipy.special import gammaln
from scipy.linalg import eigh
import math
from scipy.interpolate import interp1d
from collections import Counter
from multiprocessing import Pool

# ====================== Input ======================
# Enters the file name (default extension.txt)
filename = input("Please enter the file name containing the input parameters (without the extension) :").strip() + ".txt"

# Read variable definitions from the specified file
try:
    with open(filename, "r", encoding="utf-8") as f:
        exec(f.read())
    print(f"The input parameter has been successfully read from {filename}.")
except FileNotFoundError:
    print(f"Error: File {filename} was not found. Please confirm that the file exists in the current directory.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit(1)

# ====================== Physical constant ======================
h_ev_s = 4.135667696e-15          # Planck constant eV·s
cm1_to_ev = 1.23981e-4
amu_to_kg = 1.66053906660e-27
angstrom_to_m = 1e-10
hbar = 1.054571817e-34            # J·s
eV_to_J = 1.60218e-19             # The conversion factor from eV to J
hbar_eV = hbar / eV_to_J          # eV·s
hartree_to_joule = 4.3597447222071e-18
invcm_to_hartree = cm1_to_ev / 27.2114
log_min = -300.0                 # log(1e-50)
h = 4.135667696e-15              # Planck constant (eV·s)
c_cm = 2.99792458e10  # velocity of light (cm/s)
filtered_ts_freqs = [f for i, f in enumerate(ts_freqs_cm1) if i not in ts_hind]
filtered_react_freqs = [f for i, f in enumerate(reactant_freqs_cm1) if i not in react_hind]
factor=1.25
DE = max(round(min(min(filtered_ts_freqs), min(filtered_react_freqs))/ factor), 20.0)

# ====================== Global cache ======================
harmonic_density_tables = {}
hindered_rotor_levels_cache = {}
hindered_rotor_density_cache = {}

# ====================== Function definition ======================
def exact_enumeration_density_hybrid(freqs_cm1, hindered_indices, E_max_cm1, dE):
    n_bins = int(np.round(E_max_cm1 / dE)) + 1
    cumulative = np.zeros(n_bins, dtype=np.float64)
    cumulative[0] = 1.0  # The number of ground states is 1

    harmonic_freqs = [f for i, f in enumerate(freqs_cm1) if i not in hindered_indices]

    state_counter = Counter()
    state_counter[0.0] = 1.0

    for freq in harmonic_freqs:
        if freq < 1e-5:
            continue

        new_counter = Counter()
        max_n = int(E_max_cm1 / freq) + 1

        for E_old, count in state_counter.items():
            for n in range(1, max_n + 1):
                E_new = E_old + n * freq
                if E_new > E_max_cm1 + 1e-6:
                    break
                E_new_quant = round(E_new / dE) * dE
                new_counter[E_new_quant] += count

        state_counter.update(new_counter)

    for E, count in state_counter.items():
        bin_index = int(round(E / dE))
        if bin_index < n_bins:
            cumulative[bin_index] += count

    for i in range(1, n_bins):
        cumulative[i] += cumulative[i - 1]

    density = np.diff(cumulative, prepend=0) / dE
    return density

@lru_cache(maxsize=None)
def log_rho_harmonic(E_cm1, freqs_cm1, hindered_indices):
    harmonic_freqs = list(freqs_cm1)
    hindered_indices = list(hindered_indices)
    s = len(harmonic_freqs)
    
    if s == 0:
        return 0.0 if E_cm1 < 1e-5 else log_min

    if E_cm1 < 0:
        return log_min

    filtered_freqs = [f for i, f in enumerate(harmonic_freqs) 
                     if (f > 1.0) and (i not in hindered_indices)]
    
    if E_cm1 < max(DE*factor, 80.0): 
        s_nonzero = len(filtered_freqs)
        
        if s_nonzero == 0:
            return 0.0 if E_cm1 < 1e-5 else log_min

        sum_log_freq = sum(np.log(f) for f in filtered_freqs)

        if E_cm1 < 1e-5:
            if s_nonzero == 1:
                return -sum_log_freq  # log(1/ν)
            else:
                return log_min
        
        log_density = (s_nonzero) * np.log(E_cm1) - sum_log_freq - gammaln(s_nonzero+1)
        return log_density
     
    if not filtered_freqs:
        return 0.0 if E_cm1 < 1e-5 else log_min

    E_max_global_cm1 = E_max_global_eV / cm1_to_ev
    E_max_table = min(E_cm1 * 1.1, E_max_global_cm1)
    E_max_table = max(E_max_table, 200.0)

    freq_key = tuple(sorted(filtered_freqs))
    cache_key = (freq_key, round(E_max_table, 2), round(DE, 4))

    if cache_key in harmonic_density_tables:
        density_table = harmonic_density_tables[cache_key]
    else:
        density_table = exact_enumeration_density_hybrid(
            harmonic_freqs, hindered_indices, E_max_table, DE
        )
        harmonic_density_tables[cache_key] = density_table

    index = min(int(round(E_cm1 / DE)), len(density_table) - 1)
    density_val = density_table[index]
    return np.log(max(density_val, 1e-100))

def log_rotational_density(E_cm1, rot_consts, sigma):

    if not rot_consts:
        return 0.0

    if E_cm1 < 1e-5:
        return log_min

    hc = c_cm * h_ev_s

    if len(rot_consts) == 1:
        # Linear molecule
        B = rot_consts[0]
        rho_eV = 1.0 / (sigma * B * hc)
    else:
        # Nonlinear molecule
        A, B, C = sorted(rot_consts, reverse=True)
        E_eV = E_cm1 * cm1_to_ev
        I_factor = np.sqrt(A * B * C)
        denom = I_factor * (hc)**3
        rho_eV = (np.sqrt(np.pi) / (2 * sigma)) * np.sqrt(E_eV) / denom

    return np.log(max(rho_eV, 1e-100))


# Solve for the energy series of the blocked rotor
def solve_hindered_rotor_levels(Vb_cm1, n_sym, I_amuA2, E_max_cm1):
    cache_key = (round(Vb_cm1, 2), n_sym, round(I_amuA2, 4), round(E_max_cm1, 2))
    
    if cache_key in hindered_rotor_levels_cache:
        return hindered_rotor_levels_cache[cache_key]
    
    max_levels = 101
    half = max_levels // 2
    I_SI = I_amuA2 * amu_to_kg * angstrom_to_m**2

    m_vals = np.arange(-half, half + 1)
    T_diag = (hbar**2 / (2 * I_SI)) * m_vals**2
    Vb_J = Vb_cm1 * invcm_to_hartree * hartree_to_joule

    H = np.diag(T_diag) + np.zeros((max_levels, max_levels))
    for i, mi in enumerate(m_vals):
        for j, mj in enumerate(m_vals):
            if mi == mj:
                H[i, j] += Vb_J / 2
            elif abs(mi - mj) == n_sym:
                H[i, j] += -Vb_J / 4

    energies_J = eigh(H, eigvals_only=True)
    conv = 1 / (hartree_to_joule * invcm_to_hartree)
    energies_cm1 = energies_J * conv
    
    n_levels = np.sum(energies_cm1 <= E_max_cm1)
    hindered_rotor_levels_cache[cache_key] = n_levels
    
    return n_levels

# Logarithmic hindered rotor density of states (using cache)
def log_hindered_rotor_density(E_cm1, Vb_cm1, I_amuA2, n_sym):
    if E_cm1 < 1e-5:
        return log_min

    cache_key = (round(Vb_cm1, 2), n_sym, round(I_amuA2, 4), round(E_cm1, 2))
    
    if cache_key in hindered_rotor_density_cache:
        return hindered_rotor_density_cache[cache_key]
    
    dE = 0.1
    E1 = max(0.0, E_cm1 - dE / 2)
    E2 = E_cm1 + dE / 2
    n1 = solve_hindered_rotor_levels(Vb_cm1, n_sym, I_amuA2, E1)
    n2 = solve_hindered_rotor_levels(Vb_cm1, n_sym, I_amuA2, E2)
    dn = max(n2 - n1, 1)
    if dn <= 0:
        return log_min
    result = math.log(dn) - math.log(dE)
    hindered_rotor_density_cache[cache_key] = result
    
    return result

# Logarithmic total density of states
def log_density_of_states(E_eV, freqs_cm1, hindered_indices, barriers, moments, symmetries, rot_consts, sigma):
    E_cm1 = E_eV / cm1_to_ev

    if E_cm1 < 1e-3:
        return log_min + 50
   
    log_rho = log_rho_harmonic(
        E_cm1, 
        tuple(freqs_cm1),
        tuple(hindered_indices)
    )
    log_vib=log_rho_harmonic(
        E_cm1, 
        tuple(freqs_cm1),
        tuple(hindered_indices)
    )
    
    for Vb, I, n_sym in zip(barriers, moments, symmetries):
        hr_contrib = log_hindered_rotor_density(E_cm1, Vb, I, n_sym)
        log_rho += hr_contrib

    log_rho += log_rotational_density(E_cm1, rot_consts, sigma)

    if log_rho < log_min:
        return log_min + 10
    
    return log_rho

# Eckart Tunnel Probability Function
def eckart_tunnel(E1, V0, V1, hw_b):
    if E1 <= -V0:
        return 0.0
    
    denom_inv = 1.0 / (1.0/np.sqrt(V0) + 1.0/np.sqrt(V1))
    a = (4 * np.pi / hw_b) * np.sqrt(max(0, E1 + V0)) * denom_inv
    b = (4 * np.pi / hw_b) * np.sqrt(max(0, E1 + V1)) * denom_inv
    
    if a > 700 or b > 700:  # Hyperbolic function overflow threshold
        return 1.0
    
    term_val = (V0 * V1) / (hw_b**2) - 1/16
    
    if term_val < 0:
        epsilon = 2 * np.pi * max(0, E1) / hw_b
        if epsilon > 100:
            return 1.0
        elif epsilon < -100:
            return 0.0
        return 1.0 / (1.0 + np.exp(-2 * epsilon))
    else:
        c_val = 2 * np.pi * np.sqrt(term_val)
        if a == 0 or b == 0:
            return 0.0
        
        try:
            log_num = np.log(np.sinh(a)) + np.log(np.sinh(b))
            log_den = np.log(np.sinh((a + b)/2)**2 + np.cosh(c_val)**2)
            return np.exp(log_num - log_den)
        except:
            # Approximations are used when numerical calculations fail
            return 1.0 if (E1 > max(V0, V1)) else 0.0

# Predict the density of states function
def precompute_density(E_min, E_max, n_points, freqs_cm1, hindered_indices, barriers, moments, symmetries, rot_consts, sigma):
    E_min = max(0.0, E0 - 2.0)
    E_max = max(E_max_global_eV, E0 + 6.0)
    
    # Create a multi-segment energy grid
    E_lowest = np.linspace(max(E_min, 1e-5), 0.1, int(n_points * 0.3))
    E_low = np.linspace(0.1, E0, int(n_points * 0.3))
    E_high = np.linspace(E0, E_max, int(n_points * 0.4))
    
    E_grid = np.unique(np.concatenate((E_lowest, E_low, E_high)))
    
    density_grid = np.zeros(len(E_grid))
    freqs_tuple = tuple(freqs_cm1)
    hind_tuple = tuple(hindered_indices)
    for i, E in enumerate(E_grid):
        if E < 1e-5: 
            density_grid[i] = 1e-100
        else:
            density_val = np.exp(log_density_of_states(
                E, freqs_cm1, hindered_indices, barriers, moments, symmetries, rot_consts, sigma
            ))
            density_grid[i] = max(density_val, 1e-100)
    
    log_density = np.log(density_grid)
    
    return interp1d(E_grid, log_density, kind='linear', bounds_error=False, fill_value=log_min)

# RRKM with tunneling  (using pre-calculated density of states)
def rrkm_rate_with_tunneling_fast(E, E0, rho_react_interp, rho_ts_interp, imag_freq_eV, V1):
    log_rho_react = rho_react_interp(E)
    rho_react = np.exp(log_rho_react)
    
    if rho_react <= 0:
        return 0.0
    
    lower_bound = -E0
    upper_bound = E - E0
    
    if upper_bound <= lower_bound:
        return 0.0
    
    def integrand(E1):
        P_val = eckart_tunnel(E1, E0, V1, imag_freq_eV)
        E_avail = E - E0 - E1
        
        if E_avail <= 0:
            return 0.0
        
        log_rho_ts = rho_ts_interp(E_avail)
        rho_ts = np.exp(log_rho_ts)
        
        return P_val * rho_ts

    try:
        integral, _ = quad(integrand, lower_bound, upper_bound, epsabs=1e-20, epsrel=1e-9, limit=5000)
    except:
        integral = 0.0
    
    k_QM = integral / (h * rho_react)
    return k_QM

# RRKM without tunneling
def rrkm_rate_fast(E, E0, rho_react_interp, rho_ts_interp):
    if E <= E0:
        return 0.0

    log_rho_react = rho_react_interp(E)
    rho_react = np.exp(log_rho_react)
    
    if rho_react <= 0:
        return 0.0

    delta_E = E - E0

    def integrand(E1):
        E_avail = delta_E - E1
        if E_avail <= 0:
            return 0.0
        log_rho_ts = rho_ts_interp(E_avail)
        return np.exp(log_rho_ts)

    try:
        N_TS, _ = quad(integrand, 0.0, delta_E, epsabs=1e-20, epsrel=1e-9, limit=5000)
    except:
        N_TS = 0.0

    k_E = N_TS / (h * rho_react)
    return k_E

# ====================== mian ======================
if __name__ == "__main__":
    # Determine the maximum energy for the density of states calculation
    global E_max_global_eV
    E_max_global_eV = max(E_total, E0 + 10.0)  # Increase the margin by 10eV
    
    E_min_plot = 0.01                  # Minimum drawing/data energy
    E_max_plot = min(E0+4.0, E_max_global_eV)  # Maximum plotting/data energy (not exceeding E_max_global_eV)
    n_points_data = 500                # Calculation points
    E_values = np.linspace(E_min_plot, E_max_plot, n_points_data)

    # Pre-computed energy range
    E_min_interp = 0.000001
    E_max_interp = E0 + 8.0
    n_points_interp = 1024
    
    print("Pre-calculate the reaction-state density...")
    rho_react_interp = precompute_density(
        E_min_interp, E_max_interp, n_points_interp, 
        reactant_freqs_cm1, react_hind, 
        react_bar, react_mom, react_sym,
        react_rotational, react_sigma
    )
    print("Pre-calculate the transition-state density...")
    rho_ts_interp = precompute_density(
        E_min_interp, E_max_interp, n_points_interp, 
        ts_freqs_cm1, ts_hind, 
        ts_bar, ts_mom, ts_sym,
        ts_rotational, ts_sigma
    )
    
    # ---------------- Calculate and output the result ---------------- #
    print("Calculate the rate constant without tunneling correction:")
    kE_original = rrkm_rate_fast(E_total, E0, rho_react_interp, rho_ts_interp)
    print(f"k(E) = {kE_original:.3e} s⁻¹\n")

    print("Calculate the rate constant with tunneling correction:")
    kE_tunneling = rrkm_rate_with_tunneling_fast(
        E_total, E0, rho_react_interp, rho_ts_interp, 
        imag_freq_cm1 * cm1_to_ev, V1
    )
    print(f"k_QM(E) = {kE_tunneling:.3e} s⁻¹")

    print("Calculate the RRKM rate curve without tunneling...")
    k_values_original = [rrkm_rate_fast(E, E0, rho_react_interp, rho_ts_interp) for E in E_values]
    
    print("Calculate the RRKM rate curve with tunneling...")
    k_values_tunneling = [rrkm_rate_with_tunneling_fast(
        E, E0, rho_react_interp, rho_ts_interp, 
        imag_freq_cm1 * cm1_to_ev, V1
    ) for E in E_values]

    # ---------------- Figure ---------------- #
    plt.figure(figsize=(12, 8))
    plt.plot(E_values, k_values_original, 'b-', linewidth=2, label="Original RRKM (No Tunneling)")
    plt.plot(E_values, k_values_tunneling, 'r-', linewidth=2, label="RRKM with Tunneling")
    plt.axvline(E_total, color='g', linestyle='--', label=f"Current E = {E_total:.3f} eV")
    plt.axvline(E0, color='k', linestyle=':', label=f"Barrier E₀ = {E0} eV")
    plt.ylim(bottom=1e-5, top=max(k_values_tunneling)*10)
    plt.xlim(left=E_min_plot, right=E_max_plot)
    plt.axvspan(E_min_plot, E0, color='gray', alpha=0.2, label="Below Barrier")
    plt.yscale("log")
    plt.xlabel("Total Energy E (eV)", fontsize=12)
    plt.ylabel("Rate Constant [s⁻¹]", fontsize=12)
    plt.title("RRKM Rate vs Energy (With and Without Tunneling)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("rrkm_tunneling_corrected.png", dpi=300)
    plt.show()

    # ---------------- Save data ---------------- #
    with open("k_E_rrkm.txt", "w") as f:
        f.write("E(eV)\tk_original [1/s]\tk_tunneling [1/s]\n")
        for E, k_orig, k_tun in zip(E_values, k_values_original, k_values_tunneling):
            f.write(f"{E:.6f}\t{k_orig:.5e}\t{k_tun:.5e}\n")