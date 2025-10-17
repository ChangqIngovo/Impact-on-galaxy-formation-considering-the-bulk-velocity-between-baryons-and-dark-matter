from pathlib import Path
import os, sys, glob, csv, re
import numpy as np
import camb

try:
    import pysco
    HAS_PYSCO = True
except Exception:
    HAS_PYSCO = False
'Basic parameter, remember to install CAMB and PySco first'
OUTDIR     = "./run_2lpt_demo"
Z_START    = 999.0
KMAX_CAMB  = 500.0

H0   = 70.2
h    = H0/100.0
Om   = 0.275
Ob   = 0.0458
Tcmb = 2.7255
N_eff= 3.044
mnu  = 0.0
ns   = 0.968
As   = 2.1e-9

TARGET_SIGMA8 = 0.816

BOXLEN     = 500.0
NPART_1D   = 256
SEED       = 42
NTHREADS   = 4

'Redshift data'

z_high = np.logspace(np.log10(999), np.log10(100), 30)  # 30 points above 100
z_low = np.logspace(np.log10(100), np.log10(4), 70)     # 70 below
DESIRED_ZS = np.unique(np.concatenate([z_high, z_low]))[::-1].tolist()

print(f"Genrated {len(DESIRED_ZS)} redshift points")
print(f"z range: {DESIRED_ZS[0]:.1f} to {DESIRED_ZS[-1]:.1f}")

'CAMB'
def _base_camb_params():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0,
                       ombh2=Ob*h*h,
                       omch2=(Om-Ob)*h*h,
                       mnu=mnu, nnu=N_eff, TCMB=Tcmb)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_accuracy(AccuracyBoost=2.0, lAccuracyBoost=2.0, lSampleBoost=2.0)
    return pars

def build_pk_with_camb(kmax: float, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pars = _base_camb_params()
    pars.set_matter_power(redshifts=[0.0], kmax=kmax)
    PK = camb.get_matter_power_interpolator(
        pars, hubble_units=True, k_hunit=True, nonlinear=False,
        var1='delta_tot', var2='delta_tot'
    )
    k = np.logspace(-4, np.log10(kmax), 4096)
    P = PK.P(0.0, k)
    pk_path = outdir / "pk_z0_linear_tot.dat"
    header = ("k[h/Mpc]  P[(Mpc/h)^3]  z=0  (delta_tot, mnu=0.0 -> equals cb)\n"
              f"# H0={H0}, Om={Om}, Ob={Ob}, ns={ns}, As={As:.6e}, Neff={N_eff}")
    np.savetxt(pk_path, np.column_stack([k, P]), header=header)
    print(f"[PK] wrote: {pk_path}  (rows={len(k)})")
    return pk_path.resolve()

def build_species_power_at_z(z_ic: float, kmax: float, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pars = _base_camb_params()
    pars.set_matter_power(redshifts=[float(z_ic)], kmax=kmax)
    PK_cc = camb.get_matter_power_interpolator(
        pars, hubble_units=True, k_hunit=True, nonlinear=False,
        var1='delta_cdm', var2='delta_cdm'
    )
    PK_bb = camb.get_matter_power_interpolator(
        pars, hubble_units=True, k_hunit=True, nonlinear=False,
        var1='delta_baryon', var2='delta_baryon'
    )
    PK_cb = camb.get_matter_power_interpolator(
        pars, hubble_units=True, k_hunit=True, nonlinear=False,
        var1='delta_cdm', var2='delta_baryon'
    )
    k = np.logspace(-4, np.log10(kmax), 4096)
    Pcc = PK_cc.P(z_ic, k)
    Pbb = PK_bb.P(z_ic, k)
    Pcb = PK_cb.P(z_ic, k)
    rtb = np.sqrt(np.maximum(Pbb, 1e-300)/np.maximum(Pcc, 1e-300))
    out = np.column_stack([k, Pcc, Pbb, Pcb, rtb])
    out_path = outdir / f"pk_z{int(round(z_ic))}_cc_bb_cb_rtb.dat"
    header = (f"k[h/Mpc]  P_cc  P_bb  P_cb  r_tb=sqrt(Pbb/Pcc)   (z={z_ic})\n"
              f"# H0={H0}, Om={Om}, Ob={Ob}, ns={ns}, As={As:.6e}, Neff={N_eff}, mnu={mnu}\n"
              f"# Units: P in (Mpc/h)^3")
    np.savetxt(out_path, out, header=header)
    print(f"[PK-species] wrote: {out_path}  (rows={len(k)})")
    return out_path.resolve()

'Pysco parameter'
def make_pysco_param(pk_abs: Path, base_dir: Path, z_list) -> dict:
    base_dir.mkdir(parents=True, exist_ok=True)
    z_out_str = "[" + ",".join(f"{int(round(z))}" for z in z_list) + "]"
    return {
        "nthreads": int(NTHREADS),
        "theory": "newton",
        "H0": float(H0),
        "Om_m": float(Om),
        "T_cmb": float(Tcmb),
        "N_eff": float(N_eff),
        "w0": -1.0,
        "wa": 0.0,
        "boxlen": float(BOXLEN),
        "ncoarse": 7,
        "npart": int(NPART_1D)**3,
        "z_start": int(Z_START),
        "seed": int(SEED),
        "position_ICS": "center",
        "fixed_ICS": False,
        "paired_ICS": False,
        "dealiased_ICS": True,
        "power_spectrum_file": str(pk_abs),
        "initial_conditions": "2LPT",
        "base": str(base_dir) + "/",
        "z_out": z_out_str,           
        "output_snapshot_format": "HDF5",
        "save_power_spectrum": "yes",
        "integrator": "leapfrog",
        "n_reorder": 50,
        "mass_scheme": "TSC",
        "Courant_factor": 1.0,
        "max_aexp_stepping": 10,
        "linear_newton_solver": "multigrid",
        "gradient_stencil_order": 5,
        "Npre": 2,
        "Npost": 1,
        "epsrel": 1e-2,
        "verbose": 1,
    }

def print_test_summary(pk_abs: Path, param: dict, pk_species_abs: Path, z_list):
    print("\n========== SUMMARY ==========")
    print(f"CAMB version      : {getattr(camb, '__version__', 'unknown')}")
    print(f"z_start (IC)      : {Z_START}")
    print(f"k_max (h/Mpc)     : {KMAX_CAMB}")
    print(f"Saved P(k)        : {pk_abs}")
    print(f"Saved species P   : {pk_species_abs}")
    print(f"PySCo base dir    : {param['base']}")
    print(f"PySCo npart (3D)  : {param['npart']}  "
          f"({int(np.round(param['npart']**(1/3)))}^3)")
    print(f"Initial conditions: {param['initial_conditions']}")
    print(f"Boxlen (Mpc/h)    : {param['boxlen']}")
    print(f"z_out (len={len(z_list)}): {z_list}")
    print("=============================\n")

'CIC, making discreat points into a box'
def _read_particles_pos(h5path):
    import h5py
    with h5py.File(h5path, "r") as h:
        candidate = [
            "pos","position","positions",
            "/pos","/position","/positions",
            "PartType1/Coordinates","/PartType1/Coordinates",
            "particles/position","/particles/position",
            "coords","/coords",
        ]
        for key in candidate:
            if key in h:
                return np.asarray(h[key][...], dtype=float)
        for k in h.keys():
            try:
                arr = np.asarray(h[k][...])
                if arr.ndim == 2 and arr.shape[1] == 3:
                    return arr.astype(float)
            except Exception:
                pass
    raise RuntimeError("did not found data, check names")

def _read_z_from_h5(h5path) -> float | None:
    import h5py
    try:
        with h5py.File(h5path, "r") as h:
            for k in ["redshift", "z", "/redshift", "/Header/redshift"]:
                if k in h.attrs: return float(h.attrs[k])
                if k in h: 
                    try: return float(np.array(h[k])[()])
                    except: pass
            for k in ["aexp", "/aexp", "/Header/aexp"]:
                if k in h.attrs:
                    a = float(h.attrs[k]);  return 1.0/a - 1.0
                if k in h:
                    try:
                        a = float(np.array(h[k])[()]);  return 1.0/a - 1.0
                    except: pass
    except Exception:
        pass
    m = re.search(r"_z(\d+(\.\d+)?)", os.path.basename(h5path))
    if m:
        return float(m.group(1))
    m = re.search(r"_a(0?\.\d+)", os.path.basename(h5path))
    if m:
        a = float(m.group(1));  return 1.0/a - 1.0
    return None

def find_snapshot_for_z(base_dir: Path, target_z: float, tol=0.05) -> str | None:
    cands = []
    for pat in [str(base_dir / "output_*/*.h5"),
                str(base_dir / "*.h5"),
                str(base_dir / "output_*/particles*.h5")]:
        cands.extend(glob.glob(pat))
    if not cands:
        return None
    # find data cloeset to designated redshift
    best = None; best_dz = 1e9
    for p in cands:
        z = _read_z_from_h5(p)
        if z is None: continue
        dz = abs(z - target_z)
        if dz < best_dz:
            best_dz = dz; best = p
    if best is None: 
        return None
    if best_dz > tol:
        print(f"[warn] nearest snapshot to z={target_z} is z≈{_read_z_from_h5(best):.3f}, |Δz|={best_dz:.3f} > tol={tol}")
    return best

def deposit_density_CIC(positions, Ngrid, Lbox):
    Nx = Ny = Nz = int(Ngrid)
    dx = Lbox / Nx
    inv_dx = 1.0 / dx
    maxpos = float(np.max(positions))
    pos = positions.copy()
    if maxpos <= 1.5:
        pos *= Lbox
    pos %= Lbox
    gx, gy, gz = pos[:,0]*inv_dx, pos[:,1]*inv_dx, pos[:,2]*inv_dx
    i  = np.floor(gx).astype(int);  j  = np.floor(gy).astype(int);  k  = np.floor(gz).astype(int)
    tx = gx - i;                    ty = gy - j;                    tz = gz - k
    i0 = i % Nx; i1 = (i+1) % Nx
    j0 = j % Ny; j1 = (j+1) % Ny
    k0 = k % Nz; k1 = (k+1) % Nz
    wx0 = 1.0 - tx; wx1 = tx
    wy0 = 1.0 - ty; wy1 = ty
    wz0 = 1.0 - tz; wz1 = tz
    rho = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    np.add.at(rho, (i0, j0, k0), wx0*wy0*wz0)
    np.add.at(rho, (i1, j0, k0), wx1*wy0*wz0)
    np.add.at(rho, (i0, j1, k0), wx0*wy1*wz0)
    np.add.at(rho, (i1, j1, k0), wx1*wy1*wz0)
    np.add.at(rho, (i0, j0, k1), wx0*wy0*wz1)
    np.add.at(rho, (i1, j0, k1), wx1*wy0*wz1)
    np.add.at(rho, (i0, j1, k1), wx0*wy1*wz1)
    np.add.at(rho, (i1, j1, k1), wx1*wy1*wz1)
    rho_mean = positions.shape[0] / (Nx*Ny*Nz)
    delta = rho / rho_mean - 1.0
    return delta

def grid_snapshot_to_delta_for_z(base_dir: Path,
                                 target_z: float,
                                 out_npy: str,
                                 Ngrid: int,
                                 Lbox_Mpch: float) -> Path:
    h5path = find_snapshot_for_z(base_dir, target_z, tol=0.05)
    if h5path is None:
        raise FileNotFoundError(f"did not found z≈{target_z} snap shot")
    print(f"[grid] z={target_z:.2f} using: {h5path}")
    pos = _read_particles_pos(h5path)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise RuntimeError(f"false data shape：{pos.shape}, expecting (Npart,3)")
    delta = deposit_density_CIC(pos, Ngrid=Ngrid, Lbox=Lbox_Mpch)
    np.save(out_npy, delta.astype(np.float32))
    print(f"[grid] wrote δ_coll(z={target_z:.2f}) to: {out_npy}  shape={delta.shape}")
    return Path(out_npy).resolve()

def write_delta_summary(arr: np.ndarray, z: float, boxlen_mpc_h: float, out_csv: str):
    s = dict(
        z=float(z),
        Ngrid=int(arr.shape[0]),
        boxlen_Mpc_h=float(boxlen_mpc_h),
        delta_min=float(np.min(arr)),
        delta_max=float(np.max(arr)),
        delta_mean=float(np.mean(arr)),
        delta_std=float(np.std(arr)),
    )
    first = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(s.keys()))
        if first: w.writeheader()
        w.writerow(s)

'Main loop'
def main():
    outdir = Path(OUTDIR).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) CAMB P(k)
    pk_abs = build_pk_with_camb(KMAX_CAMB, outdir)
    pk_species_abs = build_species_power_at_z(Z_START, KMAX_CAMB, outdir)

    # 2) PySCo 跑多个红移
    base_dir = outdir / "pysco_output_00000"
    param = make_pysco_param(pk_abs, base_dir, DESIRED_ZS)
    print_test_summary(pk_abs, param, pk_species_abs, DESIRED_ZS)

    if HAS_PYSCO:
        print("[PySCo] running via API …")
        pysco.run(param)
        print("[PySCo] done. Outputs at:", param["base"])
    else:
        print("[WARN] PySCo not installed in this environment.")

    fixed_clim = None
    summary_csv = str(outdir / "collisionless_summary.csv")

    for z in DESIRED_ZS:
        out_npy = str(outdir / f"collisionless_delta_z{int(round(z))}.npy")
        out_png = str(outdir / f"collisionless_delta_z{int(round(z))}.png")
        try:
            delta_path = grid_snapshot_to_delta_for_z(
                base_dir=base_dir,
                target_z=float(z),
                out_npy=out_npy,
                Ngrid=NPART_1D,
                Lbox_Mpch=BOXLEN
            )
            arr = np.load(delta_path)

            if fixed_clim is None:
                v = float(np.percentile(np.abs(arr[arr.shape[0]//2]), 98))
                v = max(v, 1e-9)
                fixed_clim = (-v, +v)
            write_delta_summary(arr, z, BOXLEN, summary_csv)
        except Exception as e:
            print(f"[grid ERROR] z={z}: {e}")

if __name__ == "__main__":
    main()
