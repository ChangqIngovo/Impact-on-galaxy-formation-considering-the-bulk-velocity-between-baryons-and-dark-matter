import os, glob, json, re
import numpy as np
import camb
from numpy.fft import rfftn, irfftn

'Parameters and file location, name are subject to change'
COLLISIONLESS_DIR = "./run_2lpt_demo"     
OUTDIR            = "./vcb_simulation_npz"
OUT_NPZ_PATH      = os.path.join(OUTDIR, "all_fields_50Mpc.npz")  

os.makedirs(OUTDIR, exist_ok=True)
BOXLEN_MPC_H = 500.0
H0   = 70.2
h    = H0/100.0
Om   = 0.275
Ob   = 0.0458
Ol   = 1.0 - Om
Tcmb = 2.7255
N_eff= 3.044
mnu  = 0.0
ns   = 0.968
As   = 2.43e-9            # CAMB pivot k=.05 1/Mpc
C_LIGHT_KMS = 299_792.458

'CAMB kmax, should be greater than Nyquist'
KMAX_CAMB = 500.0

'Do you wish to store velocity vector? or only need rms velocity'
SAVE_COMPONENTS = True

'CAMB'
def _base_camb_params():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=Ob*h*h, omch2=(Om-Ob)*h*h, mnu=mnu, nnu=N_eff, TCMB=Tcmb)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_accuracy(AccuracyBoost=2.0, lAccuracyBoost=2.0, lSampleBoost=2.0)
    return pars

def build_interps(kmax_hmpc: float):
    pars = _base_camb_params()
    Pv = camb.get_matter_power_interpolator(
        pars, var1='v_baryon_cdm', var2='v_baryon_cdm',
        zmin=0.0, zmax=1000.0, nz_step=256, kmax=kmax_hmpc, k_hunit=True, nonlinear=False
    )
    Pd = camb.get_matter_power_interpolator(
        pars, var1='delta_nonu', var2='delta_nonu',
        zmin=0.0, zmax=1000.0, nz_step=256, kmax=kmax_hmpc, k_hunit=True, nonlinear=False
    )
    return Pv, Pd

'K grid'
def kgrid_rfftn(N: int, L_mpc_h: float):
    'Return kx,ky,kz'
    dx = L_mpc_h / N
    kx_1d = 2*np.pi*np.fft.fftfreq(N, d=dx)
    ky_1d = 2*np.pi*np.fft.fftfreq(N, d=dx)
    kz_1d = 2*np.pi*np.fft.rfftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    kmag = np.sqrt(kx*kx + ky*ky + kz*kz)
    mask = kmag > 0.0
    k_hat_x = np.zeros_like(kx); k_hat_y = np.zeros_like(ky); k_hat_z = np.zeros_like(kz)
    k_hat_x[mask] = kx[mask]/kmag[mask]
    k_hat_y[mask] = ky[mask]/kmag[mask]
    k_hat_z[mask] = kz[mask]/kmag[mask]
    return kx, ky, kz, kmag, k_hat_x, k_hat_y, k_hat_z, mask

def k_nyquist_hmpc(N: int, L_mpc_h: float) -> float:
    return np.pi * N / L_mpc_h

'Load overdensity field'
def _parse_z_from_path(path: str):
    m = re.search(r"collisionless_delta_z(\d+(\.\d+)?)\.npy$", os.path.basename(path))
    if not m: return None
    return float(m.group(1))

def load_delta_snapshots(dirpath: str):
    files = sorted(glob.glob(os.path.join(dirpath, "collisionless_delta_z*.npy")))
    pairs = []
    for f in files:
        z = _parse_z_from_path(f)
        if z is None: 
            continue
        arr = np.load(f)
        pairs.append((z, arr, f))
    pairs.sort(key=lambda t: -t[0])
    if not pairs:
        raise FileNotFoundError(f"did not found collisionless_delta_z*.npy in {dirpath}")
    shapes = {p[1].shape for p in pairs}
    if len(shapes) != 1:
        raise RuntimeError(f"Input snap shot grid size mismatch：{shapes}")
    return pairs
'Assign velocity at each pixcel'
def vcb_from_delta_at_z(delta_x: np.ndarray, z: float, L_mpc_h: float, Pv_interp, Pd_interp):
    """
    input
      δ(x,z) 
      z
      L_mpc_h [Mpc/h]
    output：
      vmag_kms  (km/s)；or vx,vy,vz（km/s） if needed
    """
    N = int(delta_x.shape[0])
    kx, ky, kz, kmag, k_hat_x, k_hat_y, k_hat_z, mask = kgrid_rfftn(N, L_mpc_h)

    # FFT δ(x,z) → δ(k,z)
    delta_k = rfftn(delta_x)
    delta_k[0,0,0] = 0.0

    # Linear theory Pv/Pd, k using h/Mpc
    Pd = np.zeros_like(kmag); Pd[mask] = Pd_interp.P(z, kmag[mask])
    Pv = np.zeros_like(kmag); Pv[mask] = Pv_interp.P(z, kmag[mask])

    conv = np.zeros_like(kmag)
    ok = mask & (Pd > 0)
    'note they used c=1 unit here...'
    conv[ok] = np.sqrt(Pv[ok] / Pd[ok])  

    # v_cb(k) = i k_hat * conv * δ(k)
    v_kx = 1j * k_hat_x * conv * delta_k
    v_ky = 1j * k_hat_y * conv * delta_k
    v_kz = 1j * k_hat_z * conv * delta_k

    'IFFT back to each box, comvert c unit to km/s'
    vx = irfftn(v_kx, s=(N,N,N)).real * C_LIGHT_KMS
    vy = irfftn(v_ky, s=(N,N,N)).real * C_LIGHT_KMS
    vz = irfftn(v_kz, s=(N,N,N)).real * C_LIGHT_KMS
    vmag = np.sqrt(vx*vx + vy*vy + vz*vz)

    return (vmag.astype(np.float32),
            vx.astype(np.float32) if SAVE_COMPONENTS else None,
            vy.astype(np.float32) if SAVE_COMPONENTS else None,
            vz.astype(np.float32) if SAVE_COMPONENTS else None)

'Main loop'
def main():
    # read δ(x,z)
    snaps = load_delta_snapshots(COLLISIONLESS_DIR)
    N = snaps[0][1].shape[0]
    L_h = float(BOXLEN_MPC_H)
    kNy = k_nyquist_hmpc(N, L_h)
    if KMAX_CAMB < kNy:
        print(f"[WARN] KMAX_CAMB={KMAX_CAMB:.1f} < k_Nyquist={kNy:.1f} h/Mpc，建议调大 KMAX_CAMB。")

    # CAMB interpolate
    Pv, Pd = build_interps(kmax_hmpc=max(KMAX_CAMB, kNy*1.05))

    arrays = {}
    summary = {}

    print(f"[INFO] input {len(snaps)} frame，grid N={N}，L={L_h:.3f} Mpc/h")
    for i, (z, delta_x, fpath) in enumerate(snaps, 1):
        vmag, vx, vy, vz = vcb_from_delta_at_z(delta_x, z, L_h, Pv, Pd)

        key_z = f"{z:g}" 
        arrays[f"vmag_kms_z{key_z}"] = vmag
        if SAVE_COMPONENTS:
            arrays[f"vx_kms_z{key_z}"] = vx
            arrays[f"vy_kms_z{key_z}"] = vy
            arrays[f"vz_kms_z{key_z}"] = vz

        vrms = float(np.sqrt(np.mean(vmag**2)))
        summary[key_z] = dict(
            vrms_kms=vrms,
            vmean_kms=float(np.mean(vmag)),
            vmin_kms=float(np.min(vmag)),
            vmax_kms=float(np.max(vmag)),
            delta_std=float(np.std(delta_x))
        )
        if (i % 5 == 0) or (i == len(snaps)):
            print(f"  [{i:>2}/{len(snaps)}] z={z:>6.1f}  v_cb,RMS={vrms:7.4f} km/s")

    'Saving file'
    meta = dict(
        grid_size=int(N),
        box_size_Mpc_h=float(L_h),
        box_size_Mpc=float(L_h / h),
        cosmology=dict(H0=H0, h=h, Om=Om, Ob=Ob, Ol=Ol, Tcmb=Tcmb,
                       N_eff=N_eff, mnu=mnu, ns=ns, As=As),
        kmax_camb_hMpc=float(KMAX_CAMB),
        summary=summary
    )
    arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)

    np.savez_compressed(OUT_NPZ_PATH, **arrays)
    print(f"[OK] saved NPZ → {OUT_NPZ_PATH}")

if __name__ == "__main__":
    main()
