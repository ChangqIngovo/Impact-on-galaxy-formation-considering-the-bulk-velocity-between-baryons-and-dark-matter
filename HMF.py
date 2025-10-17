import os, re, glob
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import interp1d, PchipInterpolator

'Parameters and file location, name are subject to change'
VCB_NPZ_PATH      = r"C:\Users\dylan_hxw\Desktop\vcb_simulation_HMF_npz\all_fields_50Mpch.npz"
COLLISIONLESS_DIR = r"C:\Users\dylan_hxw\Desktop\run_2lpt_HMF"
BOXLEN_COLL_MPC_H = 30.0


H0=70.2; h=H0/100.0
Om=0.275; Ob=0.0458; Ol=1-Om
fb=Ob/Om; fc=1-fb
delta_c=1.686
TCMB=2.7255

# SI constants
MPC_M=3.0856775814913673e22
H0_SI=H0*1000.0/MPC_M
k_B=1.380649e-23; m_p=1.67262192369e-27

def rho_m0_Msun_Mpc3():
    return Om*2.775e11*h**2

'Growth factor'
def E_of_a(a):
    return np.sqrt(Om*a**(-3)+Ol)
def _growth_spline():
    a=np.linspace(1e-4,1.0,4000)
    Ea=E_of_a(a)
    kern=1.0/((a**3)*(Ea**3))
    integ=np.cumsum(0.5*(kern[1:]+kern[:-1])*np.diff(a)); integ=np.r_[0.0,integ]
    D=Ea*integ; D/=np.interp(1.0,a,D)
    sp=PchipInterpolator(a,D)
    return sp, sp.derivative()
D_of_a, dDda_of_a=_growth_spline()

'Thermal history'
def T_PL08(z, z_dec=200.0, z_ad_end=30.0, z_reio=7.0):
    if z>z_dec:
        return TCMB*(1.0+z)
    T_ad=TCMB*(1.0+z_dec)*((1.0+z)/(1.0+z_dec))**2
    if z>z_ad_end:
        return T_ad
    T_reio=1.0e4
    A=T_reio*(1.0+z_reio)**4.9
    T_heat=A*(1.0+z)**(-4.9)
    return T_heat if z>=z_reio else T_reio

def gas_sound_speed_kms(z, mu=0.59):
    return np.sqrt(k_B*T_PL08(z)/(mu*m_p))/1000.0


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

'Thermal contribution to effective sound speed'
def gas_cs_kms_from_T(TK, mu=0.59):
    return np.sqrt(k_B*np.asarray(TK)/(mu*m_p))/1000.0

def gas_sound_speed_kms_adiab(z, mu=0.59):
    return gas_cs_kms_from_T(T_ADIAB(z), mu)

'File opening'
def _parse_redshift_from_key(key:str):
    m=re.search(r'(?:^|[_\-])z(\d+(?:\.\d+)?)', key, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None

def _load_vcb_npz(p):
    if os.path.isdir(p):
        c=sorted(glob.glob(os.path.join(p,"*.npz")))
        if not c: raise FileNotFoundError("No .npz found under: "+p)
        for f in c:
            if "all_fields" in os.path.basename(f).lower(): return np.load(f)
        return np.load(c[0])
    if not os.path.exists(p): raise FileNotFoundError(p)
    return np.load(p)

def discover_vcb_rms_interp(npz):
    z2mag={}
    for key in npz.files:
        z=_parse_redshift_from_key(key)
        if z is None: continue
        kl=key.lower()
        if ("vmag" in kl) or (("mag" in kl) and ("v" in kl)) or ("speed" in kl):
            arr=np.asarray(npz[key], float)
            z2mag[z]=float(np.sqrt(np.mean(arr**2)))
    if not z2mag: raise RuntimeError("No v_cb magnitude found in NPZ.")
    zs=np.array(sorted(z2mag), float)
    vs=np.array([z2mag[z] for z in zs], float)
    base=interp1d(zs, vs, kind='cubic' if zs.size>=4 else 'linear', fill_value="extrapolate")
    zmin, zmax = zs.min(), zs.max()
    vmin, vmax = float(base(zmin)), float(base(zmax))
    def vcb_rms_kms(z):
        z=float(z)
        if z<zmin: return vmin*(1+z)/(1+zmin)
        if z>zmax: return vmax*(1+z)/(1+zmax)
        return float(base(z))
    return vcb_rms_kms

'2 ODE solving'
def rhs_twofluid(a,y,k_si,include_thermal,vcb_func_or_none,z_tgt, cs_kms_func=None):
    dc,dcd,db,dbd=y
    a=max(float(a), 1e-8)
    z=1.0/a - 1.0
    E2=max(Om*a**(-3)+Ol, 1e-300)
    A=(3.0 - 1.5*Om*a**(-3)/E2)/a

    cs2=0.0
    if include_thermal:
        cs_kms = cs_kms_func(z) if (cs_kms_func is not None) else gas_sound_speed_kms(z)
        cs2 += (cs_kms*1000.0)**2
    if vcb_func_or_none is not None:
        vloc_kms = vcb_func_or_none(z)
        cs2 += (vloc_kms*1000.0)**2/3.0  # isotropic effective term

    press = cs2 * (k_si**2) / (a**4 * H0_SI**2 * E2)
    if not np.isfinite(press): press = 0.0

    delta_tot = fc*dc + fb*db
    d2c = (3.0*Om)/(2.0*a**4*E2)*delta_tot - A*dcd
    d2b = (3.0*Om)/(2.0*a**4*E2)*delta_tot - A*dbd - press*db
    return [dcd, d2c, dbd, d2b]

def S_of_k(k_Mpc, z_tgt, include_thermal, vcb_func_or_none, cs_kms_func=None):
    a_i=1.0/(1.0+200.0); a_t=1.0/(1.0+z_tgt)
    Di=D_of_a(a_i); Dt=D_of_a(a_t); dDi=dDda_of_a(a_i)
    y0=[Di/Dt, dDi/Dt, Di/Dt, dDi/Dt]
    S=np.ones_like(k_Mpc, float)
    for j,kM in enumerate(np.atleast_1d(k_Mpc)):
        k_si = kM/MPC_M
        try:
            solC=solve_ivp(rhs_twofluid, (a_i,a_t), y0,
                           args=(k_si, False, None, z_tgt, None),
                           method="BDF", t_eval=[a_t], rtol=5e-5, atol=1e-7)
            dcC,dbC=solC.y[0,-1], solC.y[2,-1]
            Tcoll = fc*dcC + fb*dbC

            sol=solve_ivp(rhs_twofluid, (a_i,a_t), y0,
                          args=(k_si, include_thermal, vcb_func_or_none, z_tgt, cs_kms_func),
                          method="BDF", t_eval=[a_t], rtol=5e-5, atol=1e-7)
            dc,db=sol.y[0,-1], sol.y[2,-1]
            Tm = fc*dc + fb*db
            S[j] = min((abs(Tm)/abs(Tcoll)) if abs(Tcoll)>1e-30 else 1.0, 1.0)
        except Exception:
            S[j] = 1.0
    return S

'Window functions, and necessary derivatives'
def W_tophat(x):
    x=np.asarray(x, float)
    out=np.empty_like(x)
    m=(np.abs(x)<1e-5)
    out[m]  = 1 - x[m]**2/10 + x[m]**4/280
    xm=x[~m]
    out[~m] = 3*(np.sin(xm)-xm*np.cos(xm))/(xm**3)
    return out

def dWdx_tophat(x):
    x=np.asarray(x, float)
    out=np.empty_like(x)
    m=(np.abs(x)<1e-5)
    out[m] = -x[m]/5 + x[m]**3/70
    xm=x[~m]
    out[~m]=3*((xm**2-3)*np.sin(xm)+3*xm*np.cos(xm))/(xm**4)
    return out

def sigma2_from_pk(k_h, P_h, R_Mpc, hval=h):
    k = np.asarray(k_h)*hval            # [1/Mpc]
    P = np.asarray(P_h)/(hval**3)       # [Mpc^3]
    ok=(k>0)&np.isfinite(k)&np.isfinite(P)&(P>=0)
    k, P = k[ok], P[ok]
    lnk = np.log(k)
    Delta2 = (k**3)*P/(2*np.pi**2)
    R = np.atleast_1d(R_Mpc)
    KR = R[:,None]*k[None,:]
    W2 = W_tophat(KR)**2
    sig2 = simpson(Delta2[None,:]*W2, x=lnk, axis=1)
    return np.maximum(sig2, 0.0)

def dlnsigma_dlnM_from_pk(k_h, P_h, R_Mpc, M, sigma, hval=h):
    k = np.asarray(k_h, float) * hval
    P = np.asarray(P_h, float) / (hval**3)
    ok=(k>0)&np.isfinite(k)&np.isfinite(P)&(P>=0)
    k,P = k[ok],P[ok]
    R = np.atleast_1d(R_Mpc).astype(float)
    M = np.atleast_1d(M).astype(float)

    KR = R[:,None]*k[None,:]
    W  = W_tophat(KR)
    dWdx = dWdx_tophat(KR)
    dR_dM = (R/(3.0*M))[:,None]
    dW2_dM = 2.0 * W * dWdx * (k[None,:] * dR_dM)

    integrand = (k[None,:]**2) * P[None,:] * dW2_dM
    dsigma2_dM = (1.0/(2.0*np.pi**2)) * simpson(integrand, x=k, axis=1)
    dlns_dlnM = (M / (2.0 * np.maximum(sigma, 1e-300)**2)) * dsigma2_dM
    return dlns_dlnM

'HMF'
def f_PS_of_sigma(sigma):
    nu = delta_c/np.maximum(sigma,1e-300)
    return np.sqrt(2/np.pi)*nu*np.exp(-0.5*nu**2)

def hmf_from_pk(k_h, P_h, M_grid, use_st=False):
    R_grid_Mpc=(3.0*M_grid/(4.0*np.pi*rho_m0_Msun_Mpc3()))**(1.0/3.0)
    sig2  = sigma2_from_pk(k_h, P_h, R_grid_Mpc, hval=h)
    sigma = np.sqrt(np.maximum(sig2, 0.0))
    dlns  = dlnsigma_dlnM_from_pk(k_h, P_h, R_grid_Mpc, M_grid, sigma, hval=h)
    f     = f_PS_of_sigma(sigma)
    dn    = (rho_m0_Msun_Mpc3()/M_grid) * f * np.abs(dlns)   # [Mpc^-3]
    return dn*(h**3), sigma  # [h^3 Mpc^-3], sigma

'Extract Power spectrum'
def _find_delta_npy_for_z(base_dir, z_target):
    pats=glob.glob(os.path.join(base_dir,"collisionless_delta_z*.npy"))
    if not pats: raise FileNotFoundError("No collisionless_delta_z*.npy in "+base_dir)
    pairs=[]
    for p in pats:
        m=re.search(r"z(\d+(?:\.\d+)?)", os.path.basename(p))
        if not m: continue
        z=float(m.group(1)); pairs.append((abs(z-z_target), z, p))
    pairs.sort(key=lambda t:t[0])
    return pairs[0][2], pairs[0][1]

def _measure_pk_isotropic_h(delta3d, L_Mpch):
    N = int(delta3d.shape[0])
    nbins = max(N//6, 24)
    dk  = np.fft.fftn(delta3d.astype(np.float64))
    dk2 = np.abs(dk)**2
    P_cube = (L_Mpch**3) * dk2 / (N**6)              # [(Mpc/h)^3]
    k1d = np.fft.fftfreq(N, d=L_Mpch/N) * 2*np.pi    # [h/Mpc]
    kx,ky,kz = np.meshgrid(k1d,k1d,k1d, indexing='ij')
    kk = np.sqrt(kx*kx + ky*ky + kz*kz)
    k_flat = kk.ravel(); p_flat = P_cube.ravel()
    m = k_flat>0
    k_flat, p_flat = k_flat[m], p_flat[m]
    kfund = 2*np.pi/L_Mpch; knyq = np.pi*N/L_Mpch
    bins = np.logspace(np.log10(kfund), np.log10(knyq), nbins+1)
    k_bin, p_bin = [], []
    which = np.digitize(k_flat, bins)
    for i in range(1, len(bins)):
        mi=(which==i)
        if not np.any(mi): continue
        k_bin.append(np.exp(np.mean(np.log(k_flat[mi]))))
        p_bin.append(np.mean(p_flat[mi]))
    k_bin = np.array(k_bin); p_bin = np.array(p_bin)
    s = np.argsort(k_bin)
    return k_bin[s], p_bin[s]

def build_P0h_from_collisionless(z_target):
    npy_path, z_used = _find_delta_npy_for_z(COLLISIONLESS_DIR, z_target)
    delta = np.load(npy_path)
    k_h, P_h = _measure_pk_isotropic_h(delta, BOXLEN_COLL_MPC_H)
    return k_h, P_h, z_used

'Main loop'
def main():
    OUTDIR_PLOTS = "./hmf_plots_ps"
    os.makedirs(OUTDIR_PLOTS, exist_ok=True)

    # Masses & redshifts
    M_grid = np.logspace(4, 10, 240)
    z_list = [30.0, 20.0, 15.0, 10.0, 5.0]

    # v_cb global RMS (legacy curve) + NPZ
    npz = _load_vcb_npz(VCB_NPZ_PATH)
    vcb_rms_kms = discover_vcb_rms_interp(npz)

    # Styles
    color_map = {'COLL': '#000000', 'TH': '#1f77b4', 'TH_AD':'#9467bd', 'TV_RMS': '#d62728'}
    ls_map    = {'COLL':'-', 'TH':'-.', 'TH_AD':':', 'TV_RMS':'--'}
    label_map = {'COLL':'collisionless', 'TH':'thermal (PL08)', 'TH_AD':'thermal (adiabatic)', 'TV_RMS':'thermal+vcb (RMS)'}

    # k-grid for integrations
    k_wide_h = np.logspace(-4, 2.5, 1000)
    k_wide_Mpc = h * k_wide_h

    for z in z_list:
        print(f"--- Processing redshift z = {z:.1f} ---")

        fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.6, 8.0),
                                       gridspec_kw={'height_ratios':[3,1.4]}, sharex=True)

        # Base P0(k) from delta
        k_meas_h, P0_meas_h, z_used = build_P0h_from_collisionless(z)
        Pk_of_k = make_pk_extrapolator(k_meas_h, P0_meas_h)
        P_coll_h = Pk_of_k(k_wide_h)

        # Thermal-only S(k): PL08
        S_th = S_of_k(k_wide_Mpc, z, include_thermal=True, vcb_func_or_none=None, cs_kms_func=None)

        # Thermal-only S(k): Adiabatic 
        S_th_ad = S_of_k(k_wide_Mpc, z, include_thermal=True, vcb_func_or_none=None, cs_kms_func=gas_sound_speed_kms_adiab)

        # thermal+vcb (global RMS)
        S_tv_rms = S_of_k(k_wide_Mpc, z, include_thermal=True, vcb_func_or_none=vcb_rms_kms, cs_kms_func=None)

        # Spectra
        spectra = {
            'COLL'   : P_coll_h,
            'TH'     : P_coll_h * (S_th**2),
            'TH_AD'  : P_coll_h * (S_th_ad**2),     
            'TV_RMS' : P_coll_h * (S_tv_rms**2),
        }

        dn_map = {}
        sigma_map = {}
        for key in ['COLL','TH','TH_AD','TV_RMS']:
            dn, sigma = hmf_from_pk(k_wide_h, spectra[key], M_grid)
            dn_map[key] = dn
            sigma_map[key] = sigma

        # Absolute HMF (top)
        ax.loglog(M_grid, dn_map['COLL'],   linestyle=ls_map['COLL'],   color=color_map['COLL'],   lw=2.1, label=label_map['COLL'])
        ax.loglog(M_grid, dn_map['TH'],     linestyle=ls_map['TH'],     color=color_map['TH'],     lw=2.1, label=label_map['TH'])
        ax.loglog(M_grid, dn_map['TH_AD'],  linestyle=ls_map['TH_AD'],  color=color_map['TH_AD'],  lw=2.1, label=label_map['TH_AD'])
        ax.loglog(M_grid, dn_map['TV_RMS'], linestyle=ls_map['TV_RMS'], color=color_map['TV_RMS'], lw=2.1, label=label_map['TV_RMS'])

        ax.set_ylabel(r"$\mathrm{d}n/\mathrm{d}\ln M\ \ [h^{3}\,\mathrm{Mpc}^{-3}]$")
        ax.set_title(f"Halo Mass Function @ z={int(z)}")
        ax.grid(alpha=0.35, which="both")
        ax.legend(frameon=False, fontsize=10)

        # Ratio panel (bottom)
        eps = 1e-300
        axr.semilogx(M_grid, dn_map['TH']/np.maximum(dn_map['COLL'],eps),     color=color_map['TH'],     lw=1.8, label='TH/COLL')
        axr.semilogx(M_grid, dn_map['TH_AD']/np.maximum(dn_map['COLL'],eps),  color=color_map['TH_AD'],  lw=1.8, label='TH_AD/COLL')
        axr.semilogx(M_grid, dn_map['TV_RMS']/np.maximum(dn_map['COLL'],eps), color=color_map['TV_RMS'], lw=1.8, label='TV_RMS/COLL')

        axr.set_xlabel(r"$M\ [M_\odot]$")
        axr.set_ylabel("ratio")
        axr.grid(alpha=0.25, which='both')
        axr.legend(frameon=False, fontsize=9, loc='lower left')

        fig.tight_layout()
        out_path = os.path.join(OUTDIR_PLOTS, f"hmf_ps_z{int(z)}_adiab.png")
        plt.savefig(out_path, dpi=260, bbox_inches="tight")
        plt.close(fig)
        print(f"Figure saved to: {out_path}")

'Interpolator and extrapolator'
def make_pk_extrapolator(k_h, P_h):
    k = np.asarray(k_h, float); P = np.asarray(P_h, float)
    m = (k>0)&np.isfinite(k)&np.isfinite(P)&(P>0)
    k, P = k[m], P[m]
    lk, lP = np.log(k), np.log(P)
    def local_slope(x, y, i):
        if i==0: return (y[1]-y[0])/(x[1]-x[0])
        if i==len(x)-1: return (y[-1]-y[-2])/(x[-1]-x[-2])
        return (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
    s_lo = local_slope(lk,lP,0)
    s_hi = local_slope(lk,lP,len(lk)-1)
    lkmin, lkmax = lk[0], lk[-1]
    lPmin, lPmax = lP[0], lP[-1]
    base = PchipInterpolator(lk, lP, extrapolate=False)
    def Pk_of_k(k_query_h):
        kq = np.atleast_1d(k_query_h).astype(float)
        lkq = np.log(np.maximum(kq, 1e-300))
        out = np.empty_like(lkq)
        inside = (lkq>=lkmin) & (lkq<=lkmax)
        out[inside] = base(lkq[inside])
        out[(lkq<lkmin)] = lPmin + s_lo*(lkq[(lkq<lkmin)] - lkmin)
        s_hi_clamped = min(s_hi, -2.5)   # suppress unphysical upturn
        out[(lkq>lkmax)] = lPmax + s_hi_clamped*(lkq[(lkq>lkmax)] - lkmax)
        return np.exp(out)
    return Pk_of_k

if __name__ == "__main__":
    main()
