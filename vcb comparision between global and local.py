import os, re, glob, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.fft import rfftn, irfftn, rfftfreq, fftfreq
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import erfc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import camb

'Basic parameter'
VCB_NPZ_PATH      = r"./vcb_simulation_npz/all_fields_50Mpc.npz"
COLLISIONLESS_DIR = r"./run_2lpt_demo"
OUTDIR            = "./analysis_results_with_diff"

ANALYSIS_REDSHIFTS = [20]
R_LIST_Mpc         = [3, 20, 100]

COOLING_MODE    = "molecular"
V0_kms          = 4.0
ALPHA_vcb       = 4.0
SMOOTH_VCB_BY_R = True

H0 = 70.2
h  = H0/100.0
Om   = 0.275
Ob   = 0.0458
Ol   = 0.725
TCMB = 2.7255
T_CMB0 = 2.7255
ns   = 0.968
As   = 2.1e-9
mnu  = 0.0

fc = (Om - Ob) / Om
fb = Ob / Om

# Blocks, change for better resolution if required
S_BLOCK = (32, 32, 32)
S_HOP   = (16, 16, 16)
S_NK    = 24
S_SSTEP = 0.05

# plotting
FCOLL_CMAP     = "RdBu_r"
DELTA_VCB_CMAP = "RdBu_r"

'Box length in Mpc'
FORCE_BOXLEN_VALUE_Mpc = 500.0 / h

k_B   = 1.380649e-23
m_p   = 1.67262192369e-27
mu_mean = 0.58
MPC_M = 3.085677581e22
H0_SI = H0 * 1000.0 / MPC_M

'Thermal history and others'
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

def gas_sound_speed_kms(z, mu=1.22):
    T = float(T_PL08(z))
    cs_mps = np.sqrt(k_B * T / (mu * m_p))
    return cs_mps / 1000.0

def kmag_grid_Mpc(N, L_Mpc):
    kx=2*np.pi*fftfreq(N,d=L_Mpc/N)
    ky=2*np.pi*fftfreq(N,d=L_Mpc/N)
    kz=2*np.pi*rfftfreq(N,d=L_Mpc/N)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.sqrt(KX**2 + KY**2 + KZ**2)

def kgrid_aniso_Mpc(Nx, Ny, Nz, Lx, Ly, Lz):
    kx=2*np.pi*fftfreq(Nx,d=Lx/Nx)
    ky=2*np.pi*fftfreq(Ny,d=Ly/Ny)
    kz=2*np.pi*rfftfreq(Nz,d=Lz/Nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    return np.sqrt(KX**2 + KY**2 + KZ**2)

def W_tophat(kR):
    x=np.asarray(kR,float)
    out=np.empty_like(x)
    m=(np.abs(x)<1e-3)
    out[m]=1.0-(x[m]**2)/10.0+(x[m]**4)/280.0
    xm=x[~m]
    out[~m]=3.0*(np.sin(xm)-xm*np.cos(xm))/(xm**3+1e-300)
    return out

def smooth_top_hat_3d(field, R_Mpc, L_Mpc):
    fk=rfftn(field)
    k=kmag_grid_Mpc(field.shape[0], L_Mpc)
    W=W_tophat(k*R_Mpc)
    return irfftn(fk*W, s=field.shape).real

def sigma_from_P(k_Mpc, Pz, R_Mpc_list):
    R_Mpc_arr=np.asarray(R_Mpc_list,float).ravel()
    lnk=np.log(k_Mpc+1e-300)
    Delta2=(k_Mpc[None,:]**3*Pz)/(2.0*np.pi**2)
    Nz=Pz.shape[0]
    sig=np.empty((Nz, R_Mpc_arr.size), float)
    for j, R in enumerate(R_Mpc_arr):
        W2=W_tophat(k_Mpc*R)**2
        sig2=simpson(Delta2*W2[None,:],x=lnk,axis=1)
        sig[:,j]=np.sqrt(np.maximum(sig2,0.0))
    return sig

def sigma_func_from_P_table(k_Mpc, Pk_table):
    return lambda Rlist_Mpc: sigma_from_P(k_Mpc, Pk_table[None,:], np.asarray(Rlist_Mpc, float))[0]

def camb_pk_interpolator(kmax=100.0, zmax=30.0):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Ob*h**2, omch2=(Om-Ob)*h**2, mnu=mnu, TCMB=TCMB)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.NonLinear = camb.model.NonLinear_none
    return camb.get_matter_power_interpolator(pars, hubble_units=False, k_hunit=False, kmax=kmax, zmax=zmax, nonlinear=False)

def E_of_a(a): return np.sqrt(Om * a**(-3) + Ol)

def growth_D_interp():
    a=np.linspace(1e-4,1.0,4000)
    Ea=E_of_a(a)
    kern=1.0/(np.maximum(a,1e-300)**3*np.maximum(Ea,1e-300)**3)
    integ=np.cumsum(0.5*(kern[1:]+kern[:-1])*np.diff(a))
    integ=np.concatenate([[0.0], integ])
    D=Ea*integ
    D/=max(np.interp(1.0,a,D),1e-300)
    Dsp=PchipInterpolator(a, D)
    return Dsp, Dsp.derivative()
D_of_a, dDda_of_a = growth_D_interp()

def rhs_twofluid_generic(a, y, k_si, include_thermal, include_vcb_rms, vcb_rms_kms):
    dc,dcd,db,dbd=y
    a=float(max(a,1e-8))
    E2=Om*a**(-3)+Ol
    E2=max(E2,1e-300)
    A=(3.0-1.5*Om*a**(-3)/E2)/a
    z=1.0/a-1.0
    cs2 = 0.0
    if include_thermal:
        cs2 += (gas_sound_speed_kms(z)*1000.0)**2
    if include_vcb_rms and (vcb_rms_kms is not None):
        cs2 += (vcb_rms_kms(z)*1000.0)**2/3.0
    press=cs2*(k_si**2)/(a**4*H0_SI**2*E2)
    if not np.isfinite(press): press=0.0
    delta_tot=fc*dc+fb*db
    d2c=(3.0*Om)/(2.0*a**5*E2)*delta_tot-A*dcd
    d2b=(3.0*Om)/(2.0*a**5*E2)*delta_tot-A*dbd-press*db
    return [dcd,d2c,dbd,d2b]

'Suppresion strength'
def solve_S_generic(k_Mpc_arr, z_i, z_tgt, include_thermal=False, include_vcb_rms=False, vcb_rms_kms=None):

    a_i=1.0/(1.0+z_i); a_tgt=1.0/(1.0+z_tgt)
    Di,Dt=D_of_a(a_i),D_of_a(a_tgt); dDi=dDda_of_a(a_i)
    delta_p_i=Di/max(Dt,1e-300); delta_p_i_prime=dDi/max(Dt,1e-300)
    # T_coll
    Tcoll=np.empty_like(k_Mpc_arr,dtype=float)
    for j,k_Mpc in enumerate(k_Mpc_arr):
        k_si=k_Mpc/MPC_M
        y0=[delta_p_i,delta_p_i_prime,delta_p_i,delta_p_i_prime]
        solC=solve_ivp(rhs_twofluid_generic,(a_i,a_tgt),y0,
                       args=(k_si, False, False, None),
                       method="BDF", rtol=1e-7, atol=1e-9, t_eval=[a_tgt])
        dcC,dbC=solC.y[0,-1],solC.y[2,-1]
        Tcoll[j]=fc*dcC+fb*dbC
    # target
    S=np.empty_like(k_Mpc_arr,dtype=float)
    for j,k_Mpc in enumerate(k_Mpc_arr):
        k_si=k_Mpc/MPC_M
        y0=[delta_p_i,delta_p_i_prime,delta_p_i,delta_p_i_prime]
        sol=solve_ivp(rhs_twofluid_generic,(a_i,a_tgt),y0,
                      args=(k_si, include_thermal, include_vcb_rms, vcb_rms_kms),
                      method="BDF", rtol=1e-7, atol=1e-9, t_eval=[a_tgt])
        dc,db=sol.y[0,-1],sol.y[2,-1]
        Tm=fc*dc+fb*db
        S[j]=abs(Tm)/max(abs(Tcoll[j]),1e-30)
    return S

def apply_S_of_k(delta_3d, L_Mpc, k_samp, S_samp):
    N = delta_3d.shape[0]
    k_mag = kmag_grid_Mpc(N, L_Mpc)
    F = rfftn(delta_3d)
    S_interp = interp1d(k_samp, S_samp, kind='linear', bounds_error=False, fill_value=(S_samp[0], S_samp[-1]))
    S_k = np.ones_like(k_mag)
    m = k_mag > 0
    S_k[m] = S_interp(k_mag[m])
    return irfftn(F * S_k, s=delta_3d.shape).real

'Applying suppresion locally, Hann window applied to better satisfy boundry'
def _hann3d(sx,sy,sz):
    wx=np.hanning(sx); wy=np.hanning(sy); wz=np.hanning(sz)
    return (wx[:,None,None]*wy[None,:,None]*wz[None,None,:]).astype(np.float32,copy=False)

def rhs_twofluid_scaled(a, y, k_si, s_scale, vcb_rms_kms):
    dc,dcd,db,dbd,dp,dpd=y
    a=float(max(a,1e-8))
    E2=Om*a**(-3)+Ol; E2=max(E2,1e-300)
    A=(3.0-1.5*Om*a**(-3)/E2)/a
    z=1.0/a-1.0
    cs2=(gas_sound_speed_kms(z)*1000.0)**2+(s_scale*vcb_rms_kms(z)*1000.0)**2/3.0
    press=cs2*(k_si**2)/(a**4*H0_SI**2*E2)
    if not np.isfinite(press): press=0.0
    delta_tot=fc*dc+fb*db
    d2c=(3.0*Om)/(2.0*a**5*E2)*delta_tot-A*dcd
    d2b=(3.0*Om)/(2.0*a**5*E2)*delta_tot-A*dbd-press*db
    d2p=(3.0*Om)/(2.0*a**5*E2)*dp-A*dpd
    return [dcd, d2c, dbd, d2b, dpd, d2p]

def solve_S_bank_for_s(k_Mpc_arr, s_grid, vcb_rms_kms, z_i, z_tgt):
    a_i=1.0/(1.0+z_i); a_tgt=1.0/(1.0+z_tgt)
    Di,Dt=D_of_a(a_i),D_of_a(a_tgt); dDi=dDda_of_a(a_i)
    delta_p_i=Di/max(Dt,1e-300); delta_p_i_prime=dDi/max(Dt,1e-300)
    y0_coll=[delta_p_i,delta_p_i_prime,delta_p_i,delta_p_i_prime]
    Tcoll=np.empty_like(k_Mpc_arr,dtype=float)
    for j,k_Mpc in enumerate(k_Mpc_arr):
        k_si=k_Mpc/MPC_M
        solC=solve_ivp(rhs_twofluid_generic,(a_i,a_tgt),y0_coll,
                       args=(k_si, False, False, None),
                       method="BDF",rtol=1e-7,atol=1e-9,t_eval=[a_tgt])
        dcC,dbC=solC.y[0,-1],solC.y[2,-1]
        Tcoll[j]=fc*dcC+fb*dbC
    bank=[]
    for s in s_grid:
        S_s=np.empty_like(k_Mpc_arr,dtype=float)
        for j,k_Mpc in enumerate(k_Mpc_arr):
            k_si=k_Mpc/MPC_M
            y0=[delta_p_i,delta_p_i_prime,delta_p_i,delta_p_i_prime,delta_p_i,delta_p_i_prime]
            solV=solve_ivp(rhs_twofluid_scaled,(a_i,a_tgt),y0,
                           args=(k_si,float(s),vcb_rms_kms),
                           method="BDF",rtol=1e-7,atol=1e-9,t_eval=[a_tgt])
            dcV,dbV=solV.y[0,-1],solV.y[2,-1]
            Ttv=fc*dcV+fb*dbV
            S_s[j]=np.abs(Ttv)/max(np.abs(Tcoll[j]),1e-30)
        bank.append(S_s)
    return np.asarray(bank)

def apply_with_S_local_ola_vcb(delta_3d,vmag_kms_cube,L_delta_Mpc,z_target,vcb_rms_func,block=(32,32,32),hop=(16,16,16),n_k=24,s_step=0.05):
    N=delta_3d.shape[0]
    sx, sy, sz = block
    hx, hy, hz = hop
    Lx, Ly, Lz = L_delta_Mpc*sx/N, L_delta_Mpc*sy/N, L_delta_Mpc*sz/N
    k_mag_blk = kgrid_aniso_Mpc(sx, sy, sz, Lx, Ly, Lz)
    pos = k_mag_blk[k_mag_blk > 0]
    k_arr = np.logspace(np.log10(float(pos.min())), np.log10(float(pos.max())), n_k)
    vglob = float(max(vcb_rms_func(z_target), 1e-12))
    probe_step = max(1, N // 8)
    s_vals = []
    for x in range(0, N - sx + 1, max(hx, probe_step)):
        for y in range(0, N - sy + 1, max(hy, probe_step)):
            for z in range(0, N - sz + 1, max(hz, probe_step)):
                vblk = vmag_kms_cube[x:x+sx, y:y+sy, z:z+sz]
                s_vals.append(float(np.sqrt(np.mean(vblk**2))/vglob))
    s_vals=np.asarray(s_vals)
    s_lo=max(0.05,float(np.percentile(s_vals,1.0)))
    s_hi=float(np.percentile(s_vals,99.0))
    s_grid = np.arange(s_lo, s_hi + 1e-9, s_step, dtype=float)
    S_bank = solve_S_bank_for_s(k_arr, s_grid, vcb_rms_func, z_i=999.0, z_tgt=z_target)
    W = _hann3d(sx, sy, sz)
    out = np.zeros_like(delta_3d, dtype=np.float32)
    accum = np.zeros_like(delta_3d, dtype=np.float32)
    for x in range(0, N - sx + 1, hx):
        for y in range(0, N - sy + 1, hy):
            for z in range(0, N - sz + 1, hz):
                blk=delta_3d[x:x+sx,y:y+sy,z:z+sz].astype(np.float32,copy=False)
                vblk=vmag_kms_cube[x:x+sx,y:y+sy,z:z+sz].astype(np.float32,copy=False)
                s_loc=float(np.sqrt(np.mean(vblk**2))/vglob)
                if s_loc<=s_grid[0]:   S_tab=S_bank[0]
                elif s_loc>=s_grid[-1]:S_tab=S_bank[-1]
                else:
                    i=int(np.searchsorted(s_grid,s_loc)-1)
                    t=(s_loc-s_grid[i])/(s_grid[i+1]-s_grid[i])
                    S_tab=(1.0-t)*S_bank[i]+t*S_bank[i+1]
                S_interp=interp1d(k_arr,S_tab,kind='cubic',bounds_error=False,fill_value=(S_tab[0],S_tab[-1]))
                S_k=np.ones_like(k_mag_blk)
                m=k_mag_blk>0
                S_k[m]=S_interp(k_mag_blk[m])
                blk_w=blk*W
                out_blk=irfftn(rfftn(blk_w)*S_k,s=blk.shape).real
                out[x:x+sx,y:y+sy,z:z+sz]+=out_blk*W
                accum[x:x+sx,y:y+sy,z:z+sz]+=W*W
    out/=(accum+1e-12)
    return out

'Load vcb'
def _parse_redshift_from_key(key:str):
    m=re.search(r'(?:^|[_\-])z(\d+(?:\.\d+)?)',key,flags=re.IGNORECASE)
    if m:
        try:return float(m.group(1))
        except Exception:return None

def _is_mag_key(k:str):
    kl=k.lower()
    return('vmag'in kl)or(('mag'in kl and'v'in kl))or('speed'in kl)

def _is_comp_key(k:str,comp:str):
    kl=k.lower()
    patterns=[rf'(?:^|[_\-])v(?:cb)?[_\-]?{comp}(?:[_\-]|$)',rf'(?:^|[_\-])vel[_\-]?{comp}(?:[_\-]|$)',]
    return any(re.search(p,kl)for p in patterns)

def _discover_vcb_magnitude_map(npz):
    mag_map={}
    for key in npz.files:
        z=_parse_redshift_from_key(key)
        if z is None:continue
        if _is_mag_key(key):
            try:
                vmag=np.asarray(npz[key],float)
                mag_map[z]=vmag
            except Exception: pass
    comps={}
    for key in npz.files:
        z=_parse_redshift_from_key(key)
        if z is None:continue
        got=None
        if _is_comp_key(key,'x'):got='x'
        elif _is_comp_key(key,'y'):got='y'
        elif _is_comp_key(key,'z'):got='z'
        if got is not None:
            try:
                arr=np.asarray(npz[key],float)
                comps.setdefault(z,{})[got]=arr
            except Exception: pass
    for z,d in comps.items():
        if z in mag_map:continue
        if all(c in d for c in('x','y','z')):
            try:
                vmag=np.sqrt(d['x']**2+d['y']**2+d['z']**2)
                mag_map[z]=vmag
            except Exception: pass
    if not mag_map: raise RuntimeError("No v_cb magnitude or components found in NPZ.")
    out={float(z):mag_map[z] for z in mag_map}
    return dict(sorted(out.items(),key=lambda kv:kv[0]))

def load_vcb_rms_from_npz(npz):
    z2mag=_discover_vcb_magnitude_map(npz)
    zs=np.array(list(z2mag.keys()),dtype=float)
    vrms=np.array([float(np.sqrt(np.mean(np.asarray(z2mag[z],float)**2)))for z in zs],dtype=float)
    kind='cubic'if zs.size>=4 else'linear'
    base=interp1d(zs,vrms,kind=kind,fill_value="extrapolate",assume_sorted=True)
    zmin,zmax=float(zs.min()),float(zs.max())
    vmin,vmax=float(base(zmin)),float(base(zmax))
    def vcb_rms_kms(z):
        z=float(z)
        if z<zmin:return float(vmin*(1.0+z)/(1.0+zmin))
        if z>zmax:return float(vmax*(1.0+z)/(1.0+zmax))
        return float(base(z))
    print(f"[VCB] available z for RMS: {zs.tolist()}")
    return vcb_rms_kms,zs

def load_vmag_cube_kms_closest(npz,z_target):
    z2mag=_discover_vcb_magnitude_map(npz)
    zs=np.array(list(z2mag.keys()),dtype=float)
    idx=int(np.argmin(np.abs(zs-float(z_target))))
    z_used=float(zs[idx])
    cube=np.asarray(z2mag[z_used],float)
    print(f"[VCB] using vmag cube at z={z_used} (closest to {z_target})")
    return cube,z_used

'Colla[sed fraction'
def rho_m0_Msun_Mpc3():
    return Om * 2.775e11

def R_of_M(M):
    rho=rho_m0_Msun_Mpc3()
    return(3.0*np.asarray(M,float)/(4.0*np.pi*rho))**(1.0/3.0)

def Mmin_from_Vc(Vc_kms,z,mode="molecular"):
    base=1.0e8 if mode=="atomic" else 3.0e7
    return base*(Vc_kms/16.9)**3*((1.0+z)/10.0)**(-1.5)*(Om/0.3)**(-0.5)*(h/0.7)**(-1.0)

def eps_fcoll_map(delta3d, R_Mpc, sigma_func, Mmin_of_Vc, Vc_local, box_delta_Mpc, z_target):
    delta_R=smooth_top_hat_3d(delta3d, R_Mpc, box_delta_Mpc)
    S_R=float(sigma_func([R_Mpc])[0]**2)
    if np.isscalar(Vc_local):
        Mmin=Mmin_of_Vc(Vc_local, z_target, COOLING_MODE)*np.ones_like(delta3d)
    else:
        Mmin=Mmin_of_Vc(Vc_local, z_target, COOLING_MODE)
    Rmin_Mpc=R_of_M(Mmin)
    Rmin_min=np.clip(Rmin_Mpc.min(),1e-4,None)
    Rmin_max=np.clip(Rmin_Mpc.max(),Rmin_min*1.0001,None)
    R_samp=np.logspace(np.log10(Rmin_min*0.7),np.log10(Rmin_max*1.3),400)
    S_samp=(sigma_func(R_samp)**2)
    fS=interp1d(np.log(R_samp),np.log(np.maximum(S_samp,1e-300)),kind="linear",fill_value="extrapolate",assume_sorted=True)
    S_min=np.exp(fS(np.log(np.clip(Rmin_Mpc,R_samp.min(),R_samp.max()))))
    delta_c=1.686
    denom=np.maximum(S_min-S_R,1.0e-12)
    arg=(delta_c-delta_R)/np.sqrt(2.0*denom)
    fcoll=erfc(arg)
    fcoll=np.clip(fcoll,1.0e-12,1.0)
    N=delta3d.shape[0]
    return fcoll[N//2]

def find_delta_files(directory):
    pattern=os.path.join(directory,"collisionless_delta_z*.npy")
    files=glob.glob(pattern)
    redshifts,delta_files=[],[]
    for file in files:
        try:
            z=float(os.path.basename(file).replace("collisionless_delta_z","").replace(".npy",""))
            redshifts.append(z); delta_files.append(file)
        except ValueError: pass
    order=np.argsort(redshifts)[::-1]
    return [redshifts[i] for i in order], [delta_files[i] for i in order]

def load_collisionless_delta(path):
    arr=np.load(path)
    return np.asarray(arr,float)

def assert_R_vs_box(R, L, tag):
    if R > 0.4*L: print(f"[WARN] {tag}: smoothing R={R:.1f} Mpc is large vs box L={L:.1f} Mpc; map may be near-constant.")

def _linear_positive_bounds(arr):
    a=np.asarray(arr,float)
    vmin=np.percentile(a,1); vmax=np.percentile(a,99)
    if vmax<=vmin: vmax=vmin*1.001+1e-6
    return float(vmin),float(vmax)

def _get_log_bounds(list_of_arrays):
    all_data=np.concatenate([arr.ravel() for arr in list_of_arrays])
    log_data=np.log10(all_data[all_data>1e-12])
    if len(log_data)==0: return -10,-9
    vmin=np.percentile(log_data,5); vmax=np.percentile(log_data,99)
    if vmax<=vmin: vmax=vmin+1
    return vmin,vmax

def _strip_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    for s in ax.spines.values(): s.set_visible(False)

'Main function'
def process_redshift(z_target, npz, vcb_rms_kms, box_vcb_Mpc, delta_file):
    print(f"\n[Z] processing z={z_target}")
    delta3d = load_collisionless_delta(delta_file)
    N = delta3d.shape[0]
    L_delta_Mpc = FORCE_BOXLEN_VALUE_Mpc
    print(f"[INFO] N={N}, L_delta={L_delta_Mpc:.3f} Mpc, L_vcb={box_vcb_Mpc:.3f} Mpc")

    slice_idx = N // 2
    vmag_3d, z_used = load_vmag_cube_kms_closest(npz, z_target)
    vcb_2d = vmag_3d[slice_idx].astype(float)

    print("[INFO] building S(k) and P(k)")
    pkint = camb_pk_interpolator(kmax=100.0, zmax=z_target + 1.0)
    k_Mpc = np.logspace(-3.5, np.log10(100.0), 420)
    P_no  = pkint.P(z_target, k_Mpc)

    S_vcbrms_only = solve_S_generic(
        k_Mpc, z_i=999.0, z_tgt=z_target,
        include_thermal=False, include_vcb_rms=True,
        vcb_rms_kms=vcb_rms_kms
    )
    S_th_vcb = solve_S_generic(
        k_Mpc, z_i=999.0, z_tgt=z_target,
        include_thermal=True, include_vcb_rms=True,
        vcb_rms_kms=vcb_rms_kms
    )

    P_coll_vcbrms = (S_vcbrms_only**2) * P_no
    P_th_vcb      = (S_th_vcb**2)      * P_no
    P_coll        = P_no

    sigma_vcbrms  = sigma_func_from_P_table(k_Mpc, P_coll_vcbrms)
    sigma_th_vcb  = sigma_func_from_P_table(k_Mpc, P_th_vcb)
    sigma_coll    = sigma_func_from_P_table(k_Mpc, P_coll)

    delta_vcbrms_3d = apply_S_of_k(delta3d, L_delta_Mpc, k_Mpc, S_vcbrms_only)
    print("[INFO] applying local v_cb suppression via OLA (col-2)")
    delta_th_vcb_local_3d = apply_with_S_local_ola_vcb(
        delta3d, vmag_3d, L_delta_Mpc, z_target,
        vcb_rms_func=vcb_rms_kms, block=S_BLOCK, hop=S_HOP, n_k=S_NK, s_step=S_SSTEP
    )

    delta_top_2d = delta3d[slice_idx].astype(float)

    print("[INFO] computing f_coll maps (+vcb, +th⊕vcb, and collisionless baseline)")
    c_s_kms    = gas_sound_speed_kms(z_target, mu=mu_mean)

    fcoll_plus_by_R   = []
    fcoll_thvcb_by_R  = []
    fcoll_coll_by_R   = []

    for R in R_LIST_Mpc:
        if SMOOTH_VCB_BY_R:
            vcb_used = smooth_top_hat_3d(vmag_3d, R, box_vcb_Mpc)
        else:
            vcb_used = vmag_3d

        Vc_local_vcb = np.sqrt(V0_kms**2 + (ALPHA_vcb * vcb_used)**2)
        fcoll_plus_by_R.append(
            eps_fcoll_map(
                delta_vcbrms_3d, R, sigma_vcbrms, Mmin_from_Vc, Vc_local=Vc_local_vcb,
                box_delta_Mpc=L_delta_Mpc, z_target=z_target
            )
        )

        Vc_local_th_vcb = np.sqrt(V0_kms**2 + c_s_kms**2 + (ALPHA_vcb * vcb_used)**2)
        fcoll_thvcb_by_R.append(
            eps_fcoll_map(
                delta_th_vcb_local_3d, R, sigma_th_vcb, Mmin_from_Vc, Vc_local=Vc_local_th_vcb,
                box_delta_Mpc=L_delta_Mpc, z_target=z_target
            )
        )

        fcoll_coll_by_R.append(
            eps_fcoll_map(
                delta3d, R, sigma_coll, Mmin_from_Vc, Vc_local=V0_kms,
                box_delta_Mpc=L_delta_Mpc, z_target=z_target
            )
        )

    print("[INFO] rendering main figure (merged +vcb & +th⊕vcb)")
    os.makedirs(OUTDIR, exist_ok=True)

    nrows_total = 1 + len(R_LIST_Mpc)
    height_ratios = [0.95] + [1.08] * len(R_LIST_Mpc)

    fig = plt.figure(figsize=(14.0, 8.8), dpi=300)
    gs  = fig.add_gridspec(nrows=nrows_total, ncols=1, height_ratios=height_ratios, hspace=0.16)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(nrows_total)]

    def plot_row_merged_top(ax, left_arr, right_arr, left_title, right_title):
        extL = [0, L_delta_Mpc,     0, L_delta_Mpc]
        extR = [L_delta_Mpc, 2*L_delta_Mpc, 0, L_delta_Mpc]

        vminL, vmaxL = np.percentile(left_arr, 2), np.percentile(left_arr, 98)
        imL = ax.imshow(left_arr, origin="lower", extent=extL, cmap=DELTA_VCB_CMAP,
                        vmin=vminL, vmax=vmaxL, interpolation="nearest")
        dL   = make_axes_locatable(ax)
        caxL = dL.append_axes("left", size="2.2%", pad=0.02)
        cbL  = plt.colorbar(imL, cax=caxL, orientation="vertical")
        cbL.ax.yaxis.set_label_position('left'); cbL.ax.yaxis.tick_left()
        cbL.set_label(r'$\delta_{\rm coll}$')

        vminR, vmaxR = _linear_positive_bounds(right_arr)
        imR = ax.imshow(right_arr, origin="lower", extent=extR, cmap=DELTA_VCB_CMAP,
                        vmin=vminR, vmax=vmaxR, interpolation="nearest")
        dR   = make_axes_locatable(ax)
        caxR = dR.append_axes("right", size="2.2%", pad=0.02)
        cbR  = plt.colorbar(imR, cax=caxR, orientation="vertical")
        cbR.set_label(r'$|v_{cb}|$  [km s$^{-1}$]')

        ax.axvline(L_delta_Mpc, color='k', lw=0.8, alpha=0.7)
        ax.text(0.02*L_delta_Mpc, 0.96*L_delta_Mpc, left_title,  fontsize=10, va='top', ha='left',  transform=ax.transData)
        ax.text(1.02*L_delta_Mpc, 0.96*L_delta_Mpc, right_title, fontsize=10, va='top', ha='left',  transform=ax.transData)
        ax.set_xlim(0, 2*L_delta_Mpc); ax.set_ylim(0, L_delta_Mpc)
        _strip_axes(ax)
        ax.text(L_delta_Mpc, -0.06*L_delta_Mpc, "x, y in comoving Mpc", ha="center", va="top",
                transform=ax.transData, fontsize=9)

    plot_row_merged_top(
        axes[0], delta_top_2d, vcb_2d,
        r"$\delta_{\rm coll}$ (slice)", r"$|v_{cb}|$ (km s$^{-1}$)"
    )

    col_left_title  = r"+ $v_{cb}$ (local)"
    col_right_title = r"+ Thermal $\oplus v_{cb}$ (local)"

    for i, R in enumerate(R_LIST_Mpc, start=1):
        ax = axes[i]
        extL = [0,               L_delta_Mpc, 0, L_delta_Mpc]
        extR = [L_delta_Mpc, 2 * L_delta_Mpc, 0, L_delta_Mpc]

        fL = fcoll_plus_by_R[i-1]
        fR = fcoll_thvcb_by_R[i-1]

        log_vmin, log_vmax = _get_log_bounds([fL, fR])

        imL = ax.imshow(np.log10(np.maximum(fL, 1e-12)), origin="lower", extent=extL, cmap=FCOLL_CMAP,
                        vmin=log_vmin, vmax=log_vmax, interpolation="nearest")
        imR = ax.imshow(np.log10(np.maximum(fR, 1e-12)), origin="lower", extent=extR, cmap=FCOLL_CMAP,
                        vmin=log_vmin, vmax=log_vmax, interpolation="nearest")

        div  = make_axes_locatable(ax)
        caxR = div.append_axes("right", size="2.2%", pad=0.02)
        cbR  = plt.colorbar(imR, cax=caxR, orientation="vertical")
        cbR.ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        cbR.set_label(r'$\log_{10}(f_{\rm coll})$')

        ax.axvline(L_delta_Mpc, color='k', lw=0.8, alpha=0.7)
        ax.set_xlim(0, 2*L_delta_Mpc); ax.set_ylim(0, L_delta_Mpc)
        _strip_axes(ax)

        ax.text(0.02*L_delta_Mpc, 0.02*L_delta_Mpc, rf"$R={R:.1f}\ \mathrm{{Mpc}}$",
                ha='left', va='bottom', fontsize=11, color='k')

        if i == 1:
            ax.text(0.50*L_delta_Mpc, 1.03*L_delta_Mpc, col_left_title,
                    ha='center', va='bottom', fontsize=10, color='k', transform=ax.transData)
            ax.text(1.50*L_delta_Mpc, 1.03*L_delta_Mpc, col_right_title,
                    ha='center', va='bottom', fontsize=10, color='k', transform=ax.transData)

    top_note = f"z = {z_target} | L_{{box}} = {L_delta_Mpc:.1f} Mpc"
    fig.suptitle(top_note, fontsize=14, y=0.99)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.04, right=0.96)

    outpng_main = os.path.join(OUTDIR, f"fcoll_merged_z{z_target}.png")
    plt.savefig(outpng_main, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] saved main figure: {outpng_main}")
    plt.show()

'Main function'
def main():
    print("Starting VCB two-models (units in Mpc).")
    os.makedirs(OUTDIR, exist_ok=True)
    npz = np.load(VCB_NPZ_PATH)
    # —— 始终使用强制盒长 —— 
    L_vcb_Mpc = FORCE_BOXLEN_VALUE_Mpc
    vcb_rms_kms, _ = load_vcb_rms_from_npz(npz)

    reds, delta_files = find_delta_files(COLLISIONLESS_DIR)
    if not reds:
        print(f"[ERR] no density cubes found in directory: {COLLISIONLESS_DIR}")
        return
    print(f"[INFO] found {len(reds)} density cubes: {reds}")

    z_to_run = [z for z in ANALYSIS_REDSHIFTS if z in reds]
    if not z_to_run:
        print("[ERR] none of ANALYSIS_REDSHIFTS are available in density cubes")
        return
    print(f"[INFO] will analyze redshifts: {z_to_run}")

    for z in z_to_run:
        idx = reds.index(z)
        process_redshift(z, npz, vcb_rms_kms, L_vcb_Mpc, delta_files[idx])

    print("\n[DONE] Results:", os.path.abspath(OUTDIR))

if __name__ == "__main__":
    main()
