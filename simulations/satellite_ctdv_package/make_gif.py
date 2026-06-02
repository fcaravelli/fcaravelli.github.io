"""
Animated GIF: Memristive Reservoir Satellite Control
Shows all three trajectories simultaneously with live thruster indicators
and reservoir memory state bar.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from pathlib import Path
from scipy.integrate import solve_ivp
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'axes.edgecolor':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black','text.color':'black',
    'font.size':9,'lines.linewidth':1.3,
})

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ─── Constants (same as before) ──────────────────────────────────────────────
GM=3.986004418e14; R_E=6.371e6; ALT=400e3; R0=R_E+ALT
VC=np.sqrt(GM/R0); T_ORB=2*np.pi*R0/VC; n_mot=VC/R0
M=8.0; F_MAX=5e-3; P_MAX=20.
RHO=2e-12; CD=2.2; AREA=0.03

def ref_orbit(t):
    return np.array([R0*np.cos(n_mot*t), R0*np.sin(n_mot*t),
                    -VC*np.sin(n_mot*t), VC*np.cos(n_mot*t)])

def eom(t,s,u):
    x,y,vx,vy=s; r=np.sqrt(x*x+y*y); v=np.sqrt(vx*vx+vy*vy)+1e-12
    ax=-GM*x/r**3 - 0.5*RHO*CD*AREA*v*vx/M
    ay=-GM*y/r**3 - 0.5*RHO*CD*AREA*v*vy/M
    ax+=(u[0]-u[1])*F_MAX/M; ay+=(u[2]-u[3])*F_MAX/M
    return [vx,vy,ax,ay]

def istep(t0,t1,s,u):
    return solve_ivp(lambda t,y: eom(t,y,u),[t0,t1],s,method='RK45',
                     rtol=1e-9,atol=1e-11).y[:,-1]

def f2u(Fx,Fy):
    u=np.zeros(4)
    if Fx>0: u[0]=min(Fx/F_MAX,1.)
    else:    u[1]=min(-Fx/F_MAX,1.)
    if Fy>0: u[2]=min(Fy/F_MAX,1.)
    else:    u[3]=min(-Fy/F_MAX,1.)
    if u.sum()*F_MAX*VC/1e3>P_MAX: u*=P_MAX/(u.sum()*F_MAX*VC/1e3)
    return u

def cw_ctrl(t,s):
    ref=ref_orbit(t); d=s-ref
    th=np.arctan2(ref[1],ref[0])
    er=np.array([np.cos(th),np.sin(th)]); et=np.array([-np.sin(th),np.cos(th)])
    dr=d[:2]@er; dt_=d[:2]@et; dvr=d[2:]@er; dvt=d[2:]@et
    Kp_r=8*n_mot**2; Kd_r=4*n_mot; Kp_t=4*n_mot**2; Kd_t=4*n_mot
    ar=-(Kp_r*dr+Kd_r*dvr)+2*n_mot*dvt
    at=-(Kp_t*dt_+Kd_t*dvt)-2*n_mot*dvr
    Fx=(ar*np.cos(th)-at*np.sin(th))*M
    Fy=(ar*np.sin(th)+at*np.cos(th))*M
    return f2u(Fx,Fy)

class MemristiveReservoir:
    def __init__(self,N=40,seed=7):
        rng=np.random.default_rng(seed)
        self.N=N; self.xi=(16000.-100.)/100.
        self.beta=100.; self.alpha=1.5e-3; self.V0=1.
        n_nd=N//2+4; B=np.zeros((n_nd,N))
        for e in range(N):
            i=rng.integers(0,n_nd); j=rng.integers(0,n_nd)
            while j==i: j=rng.integers(0,n_nd)
            B[i,e]=1; B[j,e]=-1
        self.Omega=np.eye(N)-B.T@np.linalg.pinv(B@B.T)@B
        self.W_in=rng.normal(0,.7,(N,4))
        self.w=np.full(N,.5); self.hist=[]
    def reset(self): self.w=np.full(self.N,.5); self.hist.clear()
    def step(self,dt,xn):
        E=self.W_in@xn; W=np.diag(self.w)
        A=np.eye(self.N)+self.xi*(W@self.Omega)
        dw=-self.w/self.beta+(self.alpha/self.V0)*(self.Omega@np.linalg.solve(A,E))
        self.w=np.clip(self.w+dt*dw,0.,1.); self.hist.append(self.w.copy())
        return self.w.copy()

SCALE=np.array([1/200.,1/200.,1/.3,1/.3])
def norm(d): return d*SCALE

def train_multi(res,DT=10.,N_TRAJ=25,lam=5e-5):
    rng=np.random.default_rng(99); R_all=[]; F_all=[]
    for trial in range(N_TRAJ):
        pert=rng.uniform(-1,1,4)*np.array([250.,250.,.08,.04])
        s0=ref_orbit(0.)+pert
        times=np.arange(0,T_ORB+DT,DT); S=np.zeros((len(times),4)); S[0]=s0
        for i in range(len(times)-1):
            u=cw_ctrl(times[i],S[i]); S[i+1]=istep(times[i],times[i+1],S[i],u)
        ref_a=np.array([ref_orbit(ti) for ti in times])
        res.reset()
        for i in range(len(times)-1):
            delta=S[i]-ref_a[i]; w=res.step(times[i+1]-times[i],norm(delta))
            u=cw_ctrl(times[i],S[i])
            R_all.append(w.copy()); F_all.append([(u[0]-u[1])*F_MAX,(u[2]-u[3])*F_MAX])
    R=np.array(R_all).T; F=np.array(F_all).T
    Wo=F@R.T@np.linalg.inv(R@R.T+lam*np.eye(res.N))
    res.reset(); return Wo

# ─── Run full simulations ─────────────────────────────────────────────────────
print("Running simulations...")
DT=10.; NORB=2   # 2 orbits for the GIF
res=MemristiveReservoir(N=40,seed=7)
Wo=train_multi(res,DT=DT,N_TRAJ=25)

def simulate_full(ctrl_fn):
    times=np.arange(0,NORB*T_ORB+DT,DT)
    s0=ref_orbit(0.)+np.array([180.,120.,.04,-.025])
    S=np.zeros((len(times),4)); U=np.zeros((len(times),4)); S[0]=s0
    WH=[]
    for i in range(len(times)-1):
        u=np.asarray(ctrl_fn(times[i],S[i],DT))
        U[i]=u; S[i+1]=istep(times[i],times[i+1],S[i],u)
    return times,S,U

times,S_unc,U_unc = simulate_full(lambda t,s,dt: np.zeros(4))
times,S_cw, U_cw  = simulate_full(lambda t,s,dt: cw_ctrl(t,s))
rc=lambda t,s,dt: f2u(*(Wo@res.step(dt,norm(s-ref_orbit(t)))))
times,S_rc, U_rc  = simulate_full(rc)
W_hist=np.array(res.hist)

ref_arr=np.array([ref_orbit(ti) for ti in times])
dev_unc=np.sqrt(((S_unc[:,:2]-ref_arr[:,:2])**2).sum(1))
dev_cw =np.sqrt(((S_cw[:,:2]-ref_arr[:,:2])**2).sum(1))
dev_rc =np.sqrt(((S_rc[:,:2]-ref_arr[:,:2])**2).sum(1))
print(f"  Uncontrolled final dev: {dev_unc[-1]/1e3:.2f} km")
print(f"  CW final dev:           {dev_cw[-1]:.1f} m")
print(f"  Memristive RC final dev:{dev_rc[-1]:.1f} m")

# ─── Build GIF ───────────────────────────────────────────────────────────────
print("Building GIF frames...")

# Subsample: ~120 frames total over 2 orbits
N_total = len(times)
frame_stride = max(1, N_total // 120)
frame_indices = list(range(0, N_total, frame_stride))
lim = (R0 + 400e3)/1e6
res_N = W_hist.shape[1] if len(W_hist)>0 else 40

frames = []
for fi, idx in enumerate(frame_indices):
    fig = plt.figure(figsize=(10, 7), dpi=80)
    fig.patch.set_facecolor('white')
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.07, right=0.97, top=0.90, bottom=0.08)

    # ── Main orbit panel (spans 2 rows, 2 cols) ─────────────────────────
    ax_orb = fig.add_subplot(gs[:2, :2])
    ax_orb.set_aspect('equal')
    ax_orb.set_xlim(-lim, lim); ax_orb.set_ylim(-lim, lim)
    ax_orb.set_facecolor('white')

    # Earth
    earth = Circle((0,0), R_E/1e6, color='#cccccc', zorder=0)
    ax_orb.add_patch(earth)
    ax_orb.text(0, 0, 'Earth', ha='center', va='center',
                fontsize=8, color='#333333', zorder=1)

    # Reference orbit ring
    theta_r = np.linspace(0,2*np.pi,200)
    ax_orb.plot(R0/1e6*np.cos(theta_r), R0/1e6*np.sin(theta_r),
               '--', color='#cccccc', lw=0.7, zorder=1)

    # Tail trails (last 60 steps)
    tail = max(0, idx-60)
    ax_orb.plot(S_unc[tail:idx+1,0]/1e6, S_unc[tail:idx+1,1]/1e6,
               ':', color='#aaaaaa', lw=1.2, label='Uncontrolled', zorder=2)
    ax_orb.plot(S_cw[tail:idx+1,0]/1e6, S_cw[tail:idx+1,1]/1e6,
               '--', color='#555555', lw=1.3, label='CW baseline', zorder=3)
    ax_orb.plot(S_rc[tail:idx+1,0]/1e6, S_rc[tail:idx+1,1]/1e6,
               '-', color='black', lw=1.5, label='Memristive RC', zorder=4)

    # Satellite dots
    ax_orb.plot(S_unc[idx,0]/1e6, S_unc[idx,1]/1e6,
               'o', ms=6, color='#999999', zorder=5)
    ax_orb.plot(S_cw[idx,0]/1e6, S_cw[idx,1]/1e6,
               's', ms=7, color='#444444', zorder=6)
    ax_orb.plot(S_rc[idx,0]/1e6, S_rc[idx,1]/1e6,
               '*', ms=11, color='black', zorder=7)

    # Thruster arrows for the reservoir controller
    u = U_rc[idx]
    x0,y0 = S_rc[idx,0]/1e6, S_rc[idx,1]/1e6
    arr_scale = 0.25
    dirs = [np.array([1,0]), np.array([-1,0]),
            np.array([0,1]), np.array([0,-1])]
    for k,(d,ui) in enumerate(zip(dirs,u)):
        if ui>0.05:
            ax_orb.annotate('', xy=(x0+d[0]*arr_scale*ui, y0+d[1]*arr_scale*ui),
                           xytext=(x0,y0),
                           arrowprops=dict(arrowstyle='->', color='black',
                                          lw=1.2+ui*1.5))

    ax_orb.set_xlabel('x [Mm]', fontsize=8)
    ax_orb.set_ylabel('y [Mm]', fontsize=8)
    t_cur = times[idx]
    ax_orb.set_title(f'Orbit  —  t = {t_cur/60:.1f} min  ({t_cur/T_ORB:.2f} orbits)',
                    fontsize=10)
    ax_orb.legend(loc='upper right', fontsize=7, framealpha=0.8)
    ax_orb.grid(True, alpha=0.4)

    # ── Deviation time series (top-right) ────────────────────────────────
    ax_dev = fig.add_subplot(gs[0, 2])
    ax_dev.plot(times[:idx+1]/3600., dev_unc[:idx+1]/1e3,
               ':', color='#999999', lw=1.2, label='Uncontrolled')
    ax_dev.plot(times[:idx+1]/3600., dev_cw[:idx+1]/1e3,
               '--', color='#555555', lw=1.2, label='CW')
    ax_dev.plot(times[:idx+1]/3600., dev_rc[:idx+1]/1e3,
               '-', color='black', lw=1.5, label='Memristive RC')
    ax_dev.set_xlim(0, NORB*T_ORB/3600.)
    ax_dev.set_ylim(bottom=0)
    ax_dev.set_xlabel('Time [h]', fontsize=8)
    ax_dev.set_ylabel('Dev [km]', fontsize=8)
    ax_dev.set_title('Deviation', fontsize=9)
    ax_dev.legend(fontsize=6); ax_dev.grid(True, alpha=0.4)
    ax_dev.tick_params(labelsize=7)

    # ── Thruster bars (middle-right) ─────────────────────────────────────
    ax_thr = fig.add_subplot(gs[1, 2])
    bar_labels = ['+x','-x','+y','-y']
    bar_colors = ['#333333','#333333','#333333','#333333']
    cw_u  = U_cw[idx]
    rc_u  = U_rc[idx]
    x_pos = np.arange(4)
    ax_thr.bar(x_pos-0.18, cw_u, width=0.32, color='#888888', label='CW')
    ax_thr.bar(x_pos+0.18, rc_u, width=0.32, color='black', label='RC')
    ax_thr.set_xticks(x_pos); ax_thr.set_xticklabels(bar_labels, fontsize=8)
    ax_thr.set_ylim(0,1.05)
    ax_thr.set_title('Thrusters', fontsize=9)
    ax_thr.legend(fontsize=7, loc='upper right')
    ax_thr.set_ylabel('Fraction', fontsize=8)
    ax_thr.tick_params(labelsize=7); ax_thr.grid(True, alpha=0.4, axis='y')

    # ── Reservoir memory bar (bottom row, full width) ────────────────────
    ax_mem = fig.add_subplot(gs[2, :])
    if len(W_hist)>0:
        # Map frame index to reservoir history index
        r_idx = min(int(idx * len(W_hist)/len(times)), len(W_hist)-1)
        w_cur = W_hist[r_idx]
        bar_x = np.arange(res_N)
        ax_mem.bar(bar_x, w_cur, color='black', alpha=0.8, width=0.9)
        ax_mem.set_xlim(-0.5, res_N-0.5)
        ax_mem.set_ylim(0,1)
        ax_mem.set_xlabel('Memristor index', fontsize=8)
        ax_mem.set_ylabel('w', fontsize=8)
        ax_mem.set_title(f'Memristive Reservoir Memory  —  N={res_N} memristors', fontsize=9)
        ax_mem.tick_params(labelsize=7)
        ax_mem.grid(True, alpha=0.3, axis='y')

    # Progress bar overlay
    progress = idx / (len(times)-1)
    fig.text(0.5, 0.96,
             'Memristive Reservoir Control  —  6U CubeSat @ LEO 400 km',
             ha='center', va='top', fontsize=11, fontweight='bold')

    # Save frame to buffer
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)

    if fi % 20 == 0:
        print(f"  Frame {fi+1}/{len(frame_indices)}  (t={times[idx]/60:.1f} min)")

print(f"Saving GIF ({len(frames)} frames)...")
frames[0].save(
    OUT / 'satellite_ctdv.gif',
    save_all=True,
    append_images=frames[1:],
    duration=80,        # ms per frame  → ~12.5 fps
    loop=0,
    optimize=False,
)
print("✓ Saved: satellite_ctdv.gif")
