"""
Microsatellite Orbital Control via Memristive Reservoir Computing
======================================================================
 1. 2D LEO orbital mechanics  (6U CubeSat, 400 km, drag)
 2. Memristive-network reservoir  (Phys. Rev. E 95, 022140, 2017)
 3. Ridge-regression readout trained over many random trajectories
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from pathlib import Path
from scipy.integrate import solve_ivp
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white',
    'axes.edgecolor':'black','axes.labelcolor':'black',
    'xtick.color':'black','ytick.color':'black','text.color':'black',
    'grid.color':'#bbbbbb','grid.linestyle':'--','grid.linewidth':0.5,
    'font.size':10,'axes.titlesize':11,'legend.fontsize':8,'lines.linewidth':1.5,
})

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────
GM=3.986004418e14; R_E=6.371e6; ALT=400e3; R0=R_E+ALT
VC=np.sqrt(GM/R0); T_ORB=2*np.pi*R0/VC; n_mot=VC/R0
M=8.0; F_MAX=5e-3; P_MAX=20.
RHO=2e-12; CD=2.2; AREA=0.03

# ─── Part 1: Orbital mechanics ───────────────────────────────────────────────
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
    """CW PD controller with Coriolis feedforward."""
    ref=ref_orbit(t); d=s-ref
    th=np.arctan2(ref[1],ref[0])
    er=np.array([np.cos(th),np.sin(th)]); et=np.array([-np.sin(th),np.cos(th)])
    dr=d[:2]@er; dt_=d[:2]@et; dvr=d[2:]@er; dvt=d[2:]@et
    Kp_r=8*n_mot**2; Kd_r=4*n_mot; Kp_t=4*n_mot**2; Kd_t=4*n_mot
    ar=-(Kp_r*dr +Kd_r*dvr)+2*n_mot*dvt
    at=-(Kp_t*dt_+Kd_t*dvt)-2*n_mot*dvr
    Fx=(ar*np.cos(th)-at*np.sin(th))*M
    Fy=(ar*np.sin(th)+at*np.cos(th))*M
    return f2u(Fx,Fy)

# ─── Part 2: Memristive Reservoir ────────────────────────────────────────────
class MemristiveReservoir:
    """
    dw/dt = −w/β + (α/V₀) Ω (I + ξ W Ω)⁻¹ E(t)

    E = W_in · x_norm  (input: normalised satellite state deviation)
    Ω = I − Bᵀ(BBᵀ)⁺B  (cycle-space projector)
    W = diag(w),  w ∈ [0,1]ᴺ
    ξ = (R_off − R_on)/R_on
    """
    def __init__(self, N=40, seed=0):
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

# ─── Part 3: Multi-trajectory RC training ────────────────────────────────────
def train_multi(res, DT=10., N_TRAJ=25, lam=1e-4):
    """
    Train readout on N_TRAJ random CW-controlled trajectories.
    Each covers 1 orbit starting from a random perturbation up to ±300 m.
    This gives W_out that generalises across the full perturbation space.
    """
    rng=np.random.default_rng(99)
    R_all=[]; F_all=[]
    print(f"  Training on {N_TRAJ} random trajectories...")
    for trial in range(N_TRAJ):
        pert=rng.uniform(-1,1,4)*np.array([250.,250.,.08,.04])
        s0=ref_orbit(0.)+pert
        times=np.arange(0,T_ORB+DT,DT); S=np.zeros((len(times),4)); S[0]=s0
        for i in range(len(times)-1):
            u=cw_ctrl(times[i],S[i])
            S[i+1]=istep(times[i],times[i+1],S[i],u)
        ref_a=np.array([ref_orbit(ti) for ti in times])
        res.reset()
        for i in range(len(times)-1):
            delta=S[i]-ref_a[i]; w=res.step(times[i+1]-times[i],norm(delta))
            u=cw_ctrl(times[i],S[i])
            R_all.append(w.copy())
            F_all.append([(u[0]-u[1])*F_MAX,(u[2]-u[3])*F_MAX])
    R=np.array(R_all).T; F=np.array(F_all).T
    A=R@R.T+lam*np.eye(res.N)
    W_out=F@R.T@np.linalg.inv(A)
    res.reset()
    print(f"  Training data: {R.shape[1]} samples, W_out norm={np.linalg.norm(W_out):.4f}")
    return W_out

class RCCtrl:
    def __init__(self,res,Wo): self.res=res; self.Wo=Wo
    def compute(self,t,s,dt):
        delta=s-ref_orbit(t); w=self.res.step(dt,norm(delta))
        Fx,Fy=self.Wo@w; return f2u(Fx,Fy)

# ─── Simulator ────────────────────────────────────────────────────────────────
def simulate(n_orb, DT, ctrl_fn, label):
    times=np.arange(0,n_orb*T_ORB+DT,DT)
    s0=ref_orbit(0.)+np.array([180.,120.,.04,-.025])
    S=np.zeros((len(times),4)); U=np.zeros((len(times),4)); S[0]=s0
    for i in range(len(times)-1):
        u=np.asarray(ctrl_fn(times[i],S[i],DT))
        U[i]=u; S[i+1]=istep(times[i],times[i+1],S[i],u)
    ref_a=np.array([ref_orbit(ti) for ti in times])
    dev=np.sqrt(((S[:,:2]-ref_a[:,:2])**2).sum(1))
    print(f"  {label}: final={dev[-1]:.1f}m, max={dev.max():.1f}m")
    return times,S,U,dev,ref_a

# ─── Main ─────────────────────────────────────────────────────────────────────
print("="*62)
print("  Memristive Reservoir Control  —  6U CubeSat, LEO 400 km")
print("="*62)
print(f"  M={M}kg  F_max={F_MAX*1e3:.0f}mN×4  P_budget={P_MAX}W")
print(f"  T_orb={T_ORB/60:.1f}min  V={VC/1e3:.3f}km/s\n")

DT=10.; NORB=4
res=MemristiveReservoir(N=40,seed=7)

print("[1/4] Uncontrolled...")
t,su,_,   dev_u,ref=simulate(NORB,DT,lambda t,s,dt:np.zeros(4),"Uncontrolled")

print("[2/4] CW baseline...")
t,scw,ucw,dev_cw,ref=simulate(NORB,DT,lambda t,s,dt:cw_ctrl(t,s),"CW baseline")

print("[3/4] Multi-trajectory RC training...")
Wo=train_multi(res,DT=DT,N_TRAJ=25,lam=5e-5)

print("[4/4] Memristive RC deployment...")
rc=RCCtrl(res,Wo)
t,src,urc,dev_rc,ref=simulate(NORB,DT,lambda t,s,dt:rc.compute(t,s,dt),"Memristive RC")
res_hist=np.array(res.hist)

# ─── Plots ───────────────────────────────────────────────────────────────────
print("\nGenerating plots...")
t_h=t/3600.; lim=(R0+200e3)/1e6

# Fig 1 — Trajectories
fig,ax=plt.subplots(figsize=(7,7)); ax.set_aspect('equal')
ax.add_patch(Circle((0,0),R_E/1e6,color='#d0d0d0',zorder=0))
ax.text(0,0,'Earth',ha='center',va='center',fontsize=9)
ax.plot(ref[:,0]/1e6,ref[:,1]/1e6,'--',color='#bbbbbb',lw=.8,label='Reference orbit')
ax.plot(su[:,0]/1e6, su[:,1]/1e6, ':',color='#888888',lw=1.8,label='Uncontrolled')
ax.plot(scw[:,0]/1e6,scw[:,1]/1e6,'--',color='#555555',lw=1.4,label='CW baseline')
ax.plot(src[:,0]/1e6,src[:,1]/1e6,'-', color='black',lw=2.0,label='Memristive RC')
ax.plot(ref[0,0]/1e6,ref[0,1]/1e6,'s',ms=8,color='black',zorder=5,label='Start')
ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
ax.set_xlabel('x [Mm]'); ax.set_ylabel('y [Mm]')
ax.set_title(f'2D LEO Trajectories  ({NORB} orbits, 400 km)')
ax.legend(); ax.grid(True); plt.tight_layout()
fig.savefig(OUT / 'fig1_orbit.png',dpi=150,bbox_inches='tight')
print("  fig1_orbit.png"); plt.close()

# Fig 2 — Deviation
fig,ax=plt.subplots(figsize=(9,4))
ax.plot(t_h,dev_u/1e3, ':',color='#888888',lw=1.8,label=f'Uncontrolled  (final {dev_u[-1]/1e3:.1f} km)')
ax.plot(t_h,dev_cw/1e3,'--',color='#555555',lw=1.8,label=f'CW baseline  (final {dev_cw[-1]:.1f} m)')
ax.plot(t_h,dev_rc/1e3,'-', color='black',lw=2.2,label=f'Memristive RC  (final {dev_rc[-1]:.1f} m)')
ax.set_xlabel('Time [hours]'); ax.set_ylabel('Orbital deviation [km]')
ax.set_title('Deviation from Reference Orbit  (initial perturbation ≈ 216 m)')
ax.legend(); ax.grid(True); plt.tight_layout()
fig.savefig(OUT / 'fig2_deviation.png',dpi=150,bbox_inches='tight')
print("  fig2_deviation.png"); plt.close()

# Fig 3 — Thrusters
fig,axes=plt.subplots(2,1,figsize=(9,5.5),sharex=True)
for k,(ax,lab) in enumerate(zip(axes,['+x / −x (radial)','+y / −y (along-track)'])):
    i,j=2*k,2*k+1
    ax.fill_between(t_h,urc[:,i]-urc[:,j],0,alpha=.18,color='black')
    ax.plot(t_h,urc[:,i],'-', color='black',lw=1.,label='+ thrust (Memristive RC)')
    ax.plot(t_h,urc[:,j],'--',color='black',lw=1.,label='− thrust (Memristive RC)')
    ax.plot(t_h,ucw[:,i],':',color='#666666',lw=.8,label='+ thrust (CW)')
    ax.set_ylabel('Fraction'); ax.set_title(lab); ax.legend(ncol=3); ax.grid(True)
axes[-1].set_xlabel('Time [hours]')
plt.suptitle(f'Thruster Activation  (F_max={F_MAX*1e3:.0f}mN, budget={P_MAX}W)',y=1.01)
plt.tight_layout()
fig.savefig(OUT / 'fig3_thrusters.png',dpi=150,bbox_inches='tight')
print("  fig3_thrusters.png"); plt.close()

# Fig 4 — Reservoir memory
if len(res_hist)>0:
    nr,Nm=res_hist.shape; st=np.linspace(0,t_h[-1],nr)
    fig,axes=plt.subplots(2,1,figsize=(9,5.5))
    im=axes[0].imshow(res_hist.T,aspect='auto',cmap='Greys',origin='lower',
                      extent=[st[0],st[-1],0,Nm],vmin=0,vmax=1)
    axes[0].set_ylabel('Memristor index')
    axes[0].set_title(f'Memristive Reservoir: Memory States  w_i(t)  (N={Nm} memristors)')
    plt.colorbar(im,ax=axes[0],label='w ∈ [0,1]')
    mu=res_hist.mean(1); sg=res_hist.std(1)
    axes[1].fill_between(st,mu-sg,mu+sg,alpha=.25,color='black',label='±1σ')
    axes[1].plot(st,mu,'-',color='black',lw=1.6,label='Mean w̄(t)')
    axes[1].set_ylim(0,1); axes[1].set_xlabel('Time [hours]')
    axes[1].set_ylabel('w̄(t)'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout()
    fig.savefig(OUT / 'fig4_reservoir.png',dpi=150,bbox_inches='tight')
    print("  fig4_reservoir.png"); plt.close()

# Fig 5 — Dashboard
fig=plt.figure(figsize=(13,9))
gs=gridspec.GridSpec(3,2,figure=fig,hspace=.46,wspace=.33)

a0=fig.add_subplot(gs[0,0]); a0.set_aspect('equal')
a0.add_patch(Circle((0,0),R_E/1e6,color='#d0d0d0'))
a0.plot(ref[:,0]/1e6,ref[:,1]/1e6,'--',color='#cccccc',lw=.7)
a0.plot(su[:,0]/1e6, su[:,1]/1e6, ':',color='#999999',lw=1.2,label='Uncontrolled')
a0.plot(scw[:,0]/1e6,scw[:,1]/1e6,'--',color='#555555',lw=1.2,label='CW')
a0.plot(src[:,0]/1e6,src[:,1]/1e6,'-', color='black',lw=1.7,label='Memristive RC')
a0.set_xlim(-lim,lim); a0.set_ylim(-lim,lim); a0.grid(True)
a0.set_title('Trajectories'); a0.legend(fontsize=7)
a0.set_xlabel('x [Mm]'); a0.set_ylabel('y [Mm]')

a1=fig.add_subplot(gs[0,1])
a1.plot(t_h,dev_u/1e3,':',color='#888888',lw=1.4,label='Uncontrolled')
a1.plot(t_h,dev_cw/1e3,'--',color='#555555',lw=1.4,label='CW')
a1.plot(t_h,dev_rc/1e3,'-',color='black',lw=1.8,label='Memristive RC')
a1.set_xlabel('Time [h]'); a1.set_ylabel('Dev [km]')
a1.set_title('Orbital Deviation'); a1.legend(); a1.grid(True)

a2=fig.add_subplot(gs[1,0])
a2.plot(t_h,urc[:,0]-urc[:,1],'-', color='black',lw=1.,label='x-net RC')
a2.plot(t_h,urc[:,2]-urc[:,3],'--',color='black',lw=1.,label='y-net RC')
a2.plot(t_h,ucw[:,0]-ucw[:,1],':',color='#777777',lw=.9,label='x-net CW')
a2.set_xlabel('Time [h]'); a2.set_ylabel('Net thrust [−1,1]')
a2.set_title('Net Thrust Commands'); a2.legend(); a2.grid(True)

a3=fig.add_subplot(gs[1,1])
Prc=urc.sum(1)*F_MAX*VC/1e3; Pcw=ucw.sum(1)*F_MAX*VC/1e3
a3.plot(t_h,Prc,'-',color='black',lw=1.,label='Memristive RC')
a3.plot(t_h,Pcw,'--',color='#666666',lw=.9,label='CW ref')
a3.axhline(P_MAX,color='#333333',ls=':',lw=1.2,label=f'{P_MAX}W budget')
a3.set_xlabel('Time [h]'); a3.set_ylabel('Power [W]')
a3.set_title('Thruster Power'); a3.legend(); a3.grid(True)

if len(res_hist)>0:
    nr,Nm=res_hist.shape; st=np.linspace(0,t_h[-1],nr)
    mu=res_hist.mean(1); sg=res_hist.std(1)
    a4=fig.add_subplot(gs[2,0])
    a4.fill_between(st,mu-sg,mu+sg,alpha=.25,color='black')
    a4.plot(st,mu,'-',color='black',lw=1.5,label='w̄(t)')
    a4.set_ylim(0,1); a4.set_xlabel('Time [h]')
    a4.set_title(f'Mean Reservoir Memory (N={Nm})'); a4.legend(); a4.grid(True)

    a5=fig.add_subplot(gs[2,1])
    im5=a5.imshow(res_hist.T,aspect='auto',cmap='Greys',origin='lower',
                  extent=[st[0],st[-1],0,Nm],vmin=0,vmax=1)
    a5.set_xlabel('Time [h]'); a5.set_ylabel('Memristor index')
    a5.set_title('Reservoir Memory Heatmap')
    plt.colorbar(im5,ax=a5,label='w',shrink=.85)

fig.suptitle('Memristive Reservoir Control  —  6U CubeSat @ LEO 400 km',
             fontsize=12,fontweight='bold',y=1.01)
plt.tight_layout()
fig.savefig(OUT / 'fig5_dashboard.png',dpi=150,bbox_inches='tight')
print("  fig5_dashboard.png"); plt.close()

print("\n✓ Done.")
print(f"  Uncontrolled : final={dev_u[-1]/1e3:.2f}km  max={dev_u.max()/1e3:.2f}km")
print(f"  CW baseline  : final={dev_cw[-1]:.1f}m    max={dev_cw.max():.1f}m")
print(f"  Memristive RC: final={dev_rc[-1]:.1f}m    max={dev_rc.max():.1f}m")
