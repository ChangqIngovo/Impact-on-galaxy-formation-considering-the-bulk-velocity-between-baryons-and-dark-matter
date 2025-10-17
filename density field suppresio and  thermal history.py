import os, glob, re, json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, PchipInterpolator
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter
from matplotlib.lines import Line2D

'File path and basic parameter, generated from vcb list and CAMB+Pysco'
VCB_NPZ_PATH      = r"./vcb_simulation_npz/all_fields_50Mpc.npz"
COLLISIONLESS_DIR = r"./run_2lpt_demo"
OUTDIR            = r"./analysis_results"
os.makedirs(OUTDIR, exist_ok=True)

H0_km_s_Mpc = 70.2
h    = H0_km_s_Mpc / 100.0
Om   = 0.275
Ob   = 0.0458
Ol   = 1.0 - Om
fb   = Ob / Om
fc   = 1.0 - fb

MPC_M = 3.0856775814913673e22
H0_SI = (1000.0 * H0_km_s_Mpc) / MPC_M
kb    = 1.380649e-23
mp    = 1.67262192369e-27
T_CMB0 = 2.725
def mu_of_z(z): return 0.59

'Read in vcb file'
def _parse_redshift_from_key(key: str):
    m = re.search(r'(?:^|_)z(\d+(?:\.\d+)?)', key, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None

def load_vcb_rms_from_npz(npz_path: str):
    """
    Priority: vmag, which should exist if u used my code to generate,
    if only vx,vy,vz. we combine them to get rms.
    Return：vcb_rms_kms(z)(km/s)。
    """
    npz = np.load(npz_path)
    z2rms = {}
    for k in npz.files:
        if ("vmag" in k.lower()) and ("kms" in k.lower()):
            z = _parse_redshift_from_key(k)
            if z is None: continue
            vmag = np.asarray(npz[k], dtype=float)
            z2rms[z] = float(np.sqrt(np.mean(vmag**2)))
    if not z2rms:
        z_to_comp = {}
        for k in npz.files:
            kl = k.lower()
            if (("vx" in kl) or ("vy" in kl) or ("vz" in kl)) and ("kms" in kl):
                z = _parse_redshift_from_key(k)
                if z is None: continue
                z_to_comp.setdefault(z, {})[kl[:2]] = np.asarray(npz[k], dtype=float)
        for z, comp in z_to_comp.items():
            if all(ax in comp for ax in ("vx", "vy", "vz")):
                vmag = np.sqrt(comp["vx"]**2 + comp["vy"]**2 + comp["vz"]**2)
                z2rms[z] = float(np.sqrt(np.mean(vmag**2)))

    if not z2rms:
        raise RuntimeError(f" NPZ({npz_path}) did not found vcb velocity data")

    zs = np.array(sorted(z2rms.keys()), dtype=float)
    vr = np.array([z2rms[z] for z in zs], dtype=float)  # km/s
    kind = 'cubic' if zs.size >= 4 else 'linear'
    base = interp1d(zs, vr, kind=kind, fill_value="extrapolate", assume_sorted=True)

    zmin, zmax = float(zs.min()), float(zs.max())
    vmin, vmax = float(base(zmin)), float(base(zmax))

    def vcb_rms_kms(z):
        z = float(z)
        if z < zmin:  
            return vmin * (1.0 + z) / (1.0 + zmin)
        if z > zmax: 
            return vmax * (1.0 + z) / (1.0 + zmax)
        return float(base(z))

    print(f"[VCB] NPZ loaded. available z: {zs.tolist()}")
    return vcb_rms_kms, zs

vcb_rms_kms, _ = load_vcb_rms_from_npz(VCB_NPZ_PATH)

'Growth factor, thermal history'
z_col = 4.0
a_col = 1.0/(1.0+z_col)
z_i   = 999.0
a_i   = 1.0/(1.0+z_i)
delta_c = (3.0/20.0) * (12.0*np.pi)**(2.0/3.0)

def E_of_a(a):
    a = np.asarray(a, float)
    return np.sqrt(Om*np.power(a, -3) + Ol)

def growth_D(a_min=9.0e-4, a_max=1.0, n=4000):
    a = np.linspace(a_min, a_max, n)
    Ea = E_of_a(a)
    kern = 1.0 / (np.maximum(a,1e-300)**3 * np.maximum(Ea,1e-300)**3)
    integ = np.cumsum(0.5*(kern[1:]+kern[:-1]) * np.diff(a))
    integ = np.concatenate([[0.0], integ])
    D = Ea * integ
    D /= max(np.interp(1.0, a, D), 1e-300)
    sp = PchipInterpolator(a, D)
    return sp, sp.derivative()

D_spline, dDda_spline = growth_D()
def D_of_a(a): return D_spline(np.asarray(a,float))
def D_of_z(z): return D_of_a(1.0/(1.0+np.asarray(z,float)))
def dDda_of_a(a): return dDda_spline(np.asarray(a,float))
def dD_dz_of_z(z):
    z = np.asarray(z, float)
    a = 1.0/(1.0 + z)
    return dDda_of_a(a) * (-1.0/np.maximum((1.0+z)**2, 1e-300))

def T_PL08(z, z_dec=200.0, z_ad_end=30.0, z_reio=7.0):
    if z > z_dec:
        return T_CMB0*(1.0+z)
    T_ad = T_CMB0*(1.0+z_dec)*((1.0+z)/(1.0+z_dec))**2
    if z > z_ad_end:
        return T_ad
    T_reio = 1.0e4
    A = T_reio*(1.0+z_reio)**4.9
    T_heat = A*(1.0+z)**(-4.9)
    if z >= z_reio:
        return T_heat
    return T_reio

Z_SPLIT = 56.23
Z_ADIAB5_HI = 62.4295903066611
ADIAB_COEF_LOGZ_REV5 = np.array([2.59846436, -5.13599843, 4.48499029, -1.70665061, -0.36291838, 1.68437398])

def T_ADIAB(z):
    z = np.asarray(z, float)
    out = np.empty_like(z)
    mask_hi = z > Z_SPLIT
    if np.any(mask_hi):
        out[mask_hi] = np.vectorize(T_PL08)(z[mask_hi])
    if np.any(~mask_hi):
        x = np.log10(Z_ADIAB5_HI) - np.log10(np.maximum(z[~mask_hi], 1e-12))
        logT = np.polyval(ADIAB_COEF_LOGZ_REV5, x)
        out[~mask_hi] = 10.0**logT
    return float(out) if out.shape == () else out

def z_of_a(a): return 1.0/np.maximum(a,1e-300) - 1.0
def T_gas_from_model(a, which):
    z = z_of_a(a)
    return T_PL08(z) if (which == "PL08" or which == "PL08_VCB") else T_ADIAB(z)

'Mass to k'
def k_of_M_Msun(M):
    M = np.asarray(M, float)
    coeff = 3.6e10 * Om * h**2
    return 10.0 * np.power(np.maximum(coeff/np.maximum(M,1e-300), 0.0), 1.0/3.0)

def k_si_from_M(M):
    return k_of_M_Msun(M) / MPC_M

'ODEs solving'
def rhs(a, y, k_si, model_name):
    d_dm, dd_dm, d_b, dd_b, d_p, dd_p = y
    a  = float(max(a, 1e-8))
    E2 = Om*a**(-3) + Ol
    E2 = max(E2, 1e-300)
    E  = np.sqrt(E2)
    A = (3.0 - 1.5*Om*a**(-3)/E2) / a
    z   = 1.0/a - 1.0

    T   = T_gas_from_model(a, model_name)
    mu  = mu_of_z(z)
    c_thermal_sq = kb * T / (mu * mp)  # m^2/s^2

    if model_name == "PL08_VCB":
        vcb_rms_mps = vcb_rms_kms(z) * 1000.0  # km/s to m/s
        c_used_sq = c_thermal_sq + (vcb_rms_mps**2)/3.0
    else:
        c_used_sq = c_thermal_sq

    press = c_used_sq * (k_si**2) / (a**4 * H0_SI**2 * E**2)
    press = np.nan_to_num(press, nan=0.0, posinf=1e300, neginf=0.0)

    fac_m = (3.0*Om) / (2.0 * a**5 * E2)
    delta_tot = fc*d_dm + fb*d_b
    d2_dm = fac_m*delta_tot - A*dd_dm
    d2_b  = fac_m*delta_tot - A*dd_b - press*d_b
    d2_p  = fac_m*d_p       - A*dd_p
    return [dd_dm, d2_dm, dd_b, d2_b, dd_p, d2_p]

D_i   = D_of_z(z_i)
D_col = D_of_z(z_col)
delta_p_i = delta_c * D_i / D_col
dD_dz_i   = dD_dz_of_z(z_i)
delta_p_i_prime = - delta_p_i * (1.0+z_i)**2 * dD_dz_i / max(D_i, 1e-300)

def init_twofluid_for_k(k_mpc):
    dc_i  = delta_p_i
    db_i  = delta_p_i
    dc_ai = delta_p_i_prime
    db_ai = delta_p_i_prime
    return dc_i, dc_ai, db_i, db_ai, delta_p_i, delta_p_i_prime

'Numerical integration and plots'
M_list = 10.0**np.array([8.0, 8.5, 9.0, 9.5, 10.0])
z_grid_to_plot = np.linspace(999.0, z_col, 600)
a_eval_sorted  = np.sort(1.0/(1.0+z_grid_to_plot))

def integrate_for_mass(M, model_name):
    k_mpc = k_of_M_Msun(M)
    k_si  = k_si_from_M(M)
    dc_i, dc_ai, db_i, db_ai, dp_i, dp_ai = init_twofluid_for_k(k_mpc)
    y0 = [dc_i, dc_ai, db_i, db_ai, dp_i, dp_ai]
    t0 = min(a_i, a_eval_sorted[0])
    t1 = max(a_i, a_eval_sorted[-1])
    sol = solve_ivp(rhs, (t0, t1), y0,
                    args=(k_si, model_name),
                    method="BDF",
                    t_eval=a_eval_sorted,
                    rtol=1e-7, atol=1e-9,
                    max_step=0.02, first_step=1e-4)
    d_dm, _, d_b, _, d_p, _ = sol.y
    d_t = fb*d_b + fc*d_dm
    eps = 1e-300
    den = np.where(np.isfinite(d_p) & (np.abs(d_p) > eps), d_p, eps)
    return d_dm/den, d_b/den, d_t/den

models = ["PL08", "PL08_VCB", "ADIAB"]
curves_dm = {m: [] for m in models}
curves_b  = {m: [] for m in models}
curves_t  = {m: [] for m in models}

print("Start integrate ODE")
for m in models:
    print(f"  model: {m}")
    for i, M in enumerate(M_list):
        print(f"    mass {i+1}/{len(M_list)}: M = {M:.1e} M_sun")
        rdm, rb, rt = integrate_for_mass(M, m)
        curves_dm[m].append(rdm)
        curves_b[m].append(rb)
        curves_t[m].append(rt)

plt.rcParams.update({
    "font.size": 11,
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

M_colors = dict(zip(M_list, ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]))
def _mass_label(M):
    p = np.log10(M)
    return rf"$10^{{{p:.1f}}}M_\odot$"

def axes_common(ax, x_low=999.0, x_high=4.0):
    ax.set_xscale("log")
    ax.set_xlim(x_low, x_high)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2,10)*0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Z")

fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.6))

# Temp
ax = axs[0,0]
zT = np.linspace(4.0, 1000.0, 600)
Tvals_PL = np.array([T_PL08(z)  for z in zT])
Tvals_AD = np.array([T_ADIAB(z) for z in zT])
ax.plot(zT, Tvals_PL, color="k", lw=2, label="PL08")
ax.plot(zT, Tvals_AD, color="k", lw=2, ls=(0,(1.2,1.2)), label="ADIAB")
ax.invert_xaxis()
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Z"); ax.set_ylabel(r"$T/K$")
ax.legend(title="models:", frameon=False, loc="upper left", handlelength=3.0,
          labelspacing=0.3, borderpad=0.2)
ax.margins(x=0.02)

# DM 
ax = axs[0,1]
for M, y in zip(M_list, curves_dm["PL08_VCB"]):
    ax.plot(z_grid_to_plot, y, lw=2.2, color=M_colors[M])
for M, y in zip(M_list, curves_dm["ADIAB"]):
    ax.plot(z_grid_to_plot, y, lw=2.0, color=M_colors[M], ls=(0,(1.2,1.2)))
for M, y in zip(M_list, curves_dm["PL08"]):
    ax.plot(z_grid_to_plot, y, lw=1.2, color=M_colors[M], alpha=0.6)
axes_common(ax)
ax.set_ylim(0.95, 1.000)
ax.set_ylabel(r"$\delta_{\rm dm}/\delta'_t$")
mass_handles = [Line2D([0],[0], color=M_colors[M], lw=2.2) for M in M_list]
ax.legend(mass_handles, [_mass_label(M) for M in M_list],
          frameon=False, loc="lower left", ncol=1, fontsize=9)
ax.text(0.03, 0.92, "dark matter", color="crimson", transform=ax.transAxes, fontsize=11)

# baryon 
ax = axs[1,0]
for M, y in zip(M_list, curves_b["PL08_VCB"]):
    ax.plot(z_grid_to_plot, y, lw=2.2, color=M_colors[M])
for M, y in zip(M_list, curves_b["ADIAB"]):
    ax.plot(z_grid_to_plot, y, lw=2.0, color=M_colors[M], ls=(0,(1.2,1.2)))
for M, y in zip(M_list, curves_b["PL08"]):
    ax.plot(z_grid_to_plot, y, lw=1.2, color=M_colors[M], alpha=0.6)
axes_common(ax)
ax.set_ylim(0.85, 1.000)
ax.set_ylabel(r"$\delta_{b}/\delta'_t$")
ax.text(0.03, 0.92, "baryons", color="crimson", transform=ax.transAxes, fontsize=11)

# total 
ax = axs[1,1]
for M, y in zip(M_list, curves_t["PL08_VCB"]):
    ax.plot(z_grid_to_plot, y, lw=2.2, color=M_colors[M])
for M, y in zip(M_list, curves_t["ADIAB"]):
    ax.plot(z_grid_to_plot, y, lw=2.0, color=M_colors[M], ls=(0,(1.2,1.2)))
for M, y in zip(M_list, curves_t["PL08"]):
    ax.plot(z_grid_to_plot, y, lw=1.2, color=M_colors[M], alpha=0.6)
axes_common(ax)
ax.set_ylim(0.90, 1.000)
ax.set_ylabel(r"$\delta_{t}/\delta'_t$")
ax.text(0.03, 0.92, "total", color="crimson", transform=ax.transAxes, fontsize=11)
ax.text(0.02, 0.05,
        rf"$\delta_t=\left(\frac{{\Omega_b}}{{\Omega_m}}\right)\delta_b + \left(1-\frac{{\Omega_b}}{{\Omega_m}}\right)\delta_{{\rm dm}}$",
        transform=ax.transAxes, fontsize=9)

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
outpng = f"{OUTDIR}/overdensity evolution.png"
plt.savefig(outpng, dpi=2400, bbox_inches="tight")
print(f"[OK] saved figure: {outpng}")
plt.show()
