(function (root) {
  "use strict";

  const GM = 3.986004418e14;
  const R_E = 6.371e6;
  const ALT = 400e3;
  const R0 = R_E + ALT;
  const VC = Math.sqrt(GM / R0);
  const T_ORB = 2 * Math.PI * R0 / VC;
  const N_MOT = VC / R0;
  const M = 8.0;
  const F_MAX = 5e-3;
  const P_MAX = 20.0;
  const RHO = 2e-12;
  const CD = 2.2;
  const AREA = 0.03;
  const SCALE = [1 / 200, 1 / 200, 1 / 0.3, 1 / 0.3];
  const CONTROL_DT = 10.0;
  const CRASH_ALTITUDE = 120e3;
  const ESCAPE_ALTITUDE = 1200e3;
  const ESCAPE_DEVIATION = 800e3;

  function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
  }

  function hypot2(x, y) {
    return Math.sqrt(x * x + y * y);
  }

  function refOrbit(t) {
    const c = Math.cos(N_MOT * t);
    const s = Math.sin(N_MOT * t);
    return [R0 * c, R0 * s, -VC * s, VC * c];
  }

  function eom(state, u) {
    const x = state[0];
    const y = state[1];
    const vx = state[2];
    const vy = state[3];
    const r = Math.sqrt(x * x + y * y);
    const v = Math.sqrt(vx * vx + vy * vy) + 1e-12;
    let ax = -GM * x / (r * r * r) - 0.5 * RHO * CD * AREA * v * vx / M;
    let ay = -GM * y / (r * r * r) - 0.5 * RHO * CD * AREA * v * vy / M;
    ax += (u[0] - u[1]) * F_MAX / M;
    ay += (u[2] - u[3]) * F_MAX / M;
    return [vx, vy, ax, ay];
  }

  function addScaled(a, k, h) {
    return [
      a[0] + h * k[0],
      a[1] + h * k[1],
      a[2] + h * k[2],
      a[3] + h * k[3]
    ];
  }

  function rk4Step(state, u, dt) {
    const k1 = eom(state, u);
    const k2 = eom(addScaled(state, k1, dt * 0.5), u);
    const k3 = eom(addScaled(state, k2, dt * 0.5), u);
    const k4 = eom(addScaled(state, k3, dt), u);
    return [
      state[0] + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
      state[1] + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
      state[2] + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6,
      state[3] + dt * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) / 6
    ];
  }

  function integrate(state, u, dt) {
    let next = state;
    let remaining = dt;
    while (remaining > 1e-9) {
      const h = Math.min(2.5, remaining);
      next = rk4Step(next, u, h);
      remaining -= h;
    }
    return next;
  }

  function clampPower(u) {
    const out = u.slice();
    const power = out.reduce((a, b) => a + b, 0) * F_MAX * VC / 1e3;
    if (power > P_MAX && power > 0) {
      const scale = P_MAX / power;
      for (let i = 0; i < out.length; i += 1) out[i] *= scale;
    }
    return out;
  }

  function f2u(Fx, Fy) {
    const u = [0, 0, 0, 0];
    if (Fx > 0) u[0] = Math.min(Fx / F_MAX, 1);
    else u[1] = Math.min(-Fx / F_MAX, 1);
    if (Fy > 0) u[2] = Math.min(Fy / F_MAX, 1);
    else u[3] = Math.min(-Fy / F_MAX, 1);
    return clampPower(u);
  }

  function cwCtrl(t, state) {
    const ref = refOrbit(t);
    const d = [
      state[0] - ref[0],
      state[1] - ref[1],
      state[2] - ref[2],
      state[3] - ref[3]
    ];
    const th = Math.atan2(ref[1], ref[0]);
    const er = [Math.cos(th), Math.sin(th)];
    const et = [-Math.sin(th), Math.cos(th)];
    const dr = d[0] * er[0] + d[1] * er[1];
    const dtan = d[0] * et[0] + d[1] * et[1];
    const dvr = d[2] * er[0] + d[3] * er[1];
    const dvt = d[2] * et[0] + d[3] * et[1];
    const kpR = 8 * N_MOT * N_MOT;
    const kdR = 4 * N_MOT;
    const kpT = 4 * N_MOT * N_MOT;
    const kdT = 4 * N_MOT;
    const ar = -(kpR * dr + kdR * dvr) + 2 * N_MOT * dvt;
    const at = -(kpT * dtan + kdT * dvt) - 2 * N_MOT * dvr;
    const Fx = (ar * Math.cos(th) - at * Math.sin(th)) * M;
    const Fy = (ar * Math.sin(th) + at * Math.cos(th)) * M;
    return f2u(Fx, Fy);
  }

  function normDelta(delta) {
    return [
      delta[0] * SCALE[0],
      delta[1] * SCALE[1],
      delta[2] * SCALE[2],
      delta[3] * SCALE[3]
    ];
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a += 0x6D2B79F5;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randn(rng) {
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  class MemristiveReservoir {
    constructor(N = 40, seed = 7) {
      this.N = N;
      this.xi = (16000 - 100) / 100;
      this.beta = 100;
      this.alpha = 1.5e-3;
      this.V0 = 1;
      this.cycleSize = 2;
      const rng = mulberry32(seed);
      this.WIn = new Array(N);
      for (let i = 0; i < N; i += 1) {
        this.WIn[i] = new Array(4);
        for (let j = 0; j < 4; j += 1) {
          this.WIn[i][j] = randn(rng) * 0.7;
        }
      }
      this.reset();
    }

    reset() {
      this.w = new Array(this.N).fill(0.5);
      this.hist = [];
    }

    projectedReservoirDrive(E) {
      const out = new Array(this.N);
      for (let start = 0; start < this.N; start += this.cycleSize) {
        const end = Math.min(this.N, start + this.cycleSize);
        const size = end - start;
        let meanE = 0;
        let meanXiW = 0;
        for (let i = start; i < end; i += 1) {
          meanE += E[i];
          meanXiW += this.xi * this.w[i];
        }
        meanE /= size;
        meanXiW /= size;
        const projected = meanE / (1 + meanXiW);
        for (let i = start; i < end; i += 1) out[i] = projected;
      }
      return out;
    }

    step(dt, xn) {
      const EVec = new Array(this.N);
      for (let i = 0; i < this.N; i += 1) {
        const row = this.WIn[i];
        EVec[i] = row[0] * xn[0] + row[1] * xn[1] + row[2] * xn[2] + row[3] * xn[3];
      }
      const omegaDriven = this.projectedReservoirDrive(EVec);
      for (let i = 0; i < this.N; i += 1) {
        const dw = -this.w[i] / this.beta + (this.alpha / this.V0) * omegaDriven[i];
        this.w[i] = clamp(this.w[i] + dt * dw, 0, 1);
      }
      const copy = this.w.slice();
      this.hist.push(copy);
      if (this.hist.length > 700) this.hist.shift();
      return copy;
    }
  }

  function solveLinear(A, b) {
    const n = A.length;
    const m = new Array(n);
    for (let i = 0; i < n; i += 1) {
      m[i] = A[i].slice();
      m[i].push(b[i]);
    }
    for (let col = 0; col < n; col += 1) {
      let pivot = col;
      let best = Math.abs(m[col][col]);
      for (let row = col + 1; row < n; row += 1) {
        const value = Math.abs(m[row][col]);
        if (value > best) {
          best = value;
          pivot = row;
        }
      }
      if (best < 1e-12) m[col][col] += 1e-8;
      if (pivot !== col) {
        const tmp = m[col];
        m[col] = m[pivot];
        m[pivot] = tmp;
      }
      const divisor = m[col][col] || 1e-8;
      for (let c = col; c <= n; c += 1) m[col][c] /= divisor;
      for (let row = 0; row < n; row += 1) {
        if (row === col) continue;
        const factor = m[row][col];
        if (Math.abs(factor) < 1e-14) continue;
        for (let c = col; c <= n; c += 1) m[row][c] -= factor * m[col][c];
      }
    }
    const x = new Array(n);
    for (let i = 0; i < n; i += 1) x[i] = m[i][n];
    return x;
  }

  function zeroMatrix(n) {
    const out = new Array(n);
    for (let i = 0; i < n; i += 1) out[i] = new Array(n).fill(0);
    return out;
  }

  function trainMulti(res, dt = CONTROL_DT, nTraj = 25, lam = 5e-5) {
    const rng = mulberry32(99);
    const A = zeroMatrix(res.N);
    const b0 = new Array(res.N).fill(0);
    const b1 = new Array(res.N).fill(0);
    let samples = 0;

    for (let trial = 0; trial < nTraj; trial += 1) {
      const ref0 = refOrbit(0);
      let state = [
        ref0[0] + (rng() * 2 - 1) * 250,
        ref0[1] + (rng() * 2 - 1) * 250,
        ref0[2] + (rng() * 2 - 1) * 0.08,
        ref0[3] + (rng() * 2 - 1) * 0.04
      ];
      res.reset();
      for (let t = 0; t <= T_ORB; t += dt) {
        const ref = refOrbit(t);
        const delta = [
          state[0] - ref[0],
          state[1] - ref[1],
          state[2] - ref[2],
          state[3] - ref[3]
        ];
        const w = res.step(dt, normDelta(delta));
        const u = cwCtrl(t, state);
        const Fx = (u[0] - u[1]) * F_MAX;
        const Fy = (u[2] - u[3]) * F_MAX;
        for (let i = 0; i < res.N; i += 1) {
          const wi = w[i];
          b0[i] += Fx * wi;
          b1[i] += Fy * wi;
          for (let j = 0; j < res.N; j += 1) {
            A[i][j] += wi * w[j];
          }
        }
        samples += 1;
        state = integrate(state, u, dt);
      }
    }

    for (let i = 0; i < res.N; i += 1) A[i][i] += lam;
    const row0 = solveLinear(A, b0);
    const row1 = solveLinear(A, b1);
    res.reset();
    return { WOut: [row0, row1], samples };
  }

  class RCCtrl {
    constructor(res, WOut) {
      this.res = res;
      this.WOut = WOut;
    }

    reset() {
      this.res.reset();
    }

    compute(t, state, dt) {
      const ref = refOrbit(t);
      const delta = [
        state[0] - ref[0],
        state[1] - ref[1],
        state[2] - ref[2],
        state[3] - ref[3]
      ];
      const w = this.res.step(dt, normDelta(delta));
      let Fx = 0;
      let Fy = 0;
      for (let i = 0; i < w.length; i += 1) {
        Fx += this.WOut[0][i] * w[i];
        Fy += this.WOut[1][i] * w[i];
      }
      return f2u(Fx, Fy);
    }
  }

  function relativeState(state, t) {
    const ref = refOrbit(t);
    const dx = state[0] - ref[0];
    const dy = state[1] - ref[1];
    const dvx = state[2] - ref[2];
    const dvy = state[3] - ref[3];
    const th = Math.atan2(ref[1], ref[0]);
    const er = [Math.cos(th), Math.sin(th)];
    const et = [-Math.sin(th), Math.cos(th)];
    const radial = dx * er[0] + dy * er[1];
    const along = dx * et[0] + dy * et[1];
    const radialV = dvx * er[0] + dvy * er[1];
    const alongV = dvx * et[0] + dvy * et[1];
    const r = hypot2(state[0], state[1]);
    return {
      ref,
      dx,
      dy,
      dvx,
      dvy,
      radial,
      along,
      radialV,
      alongV,
      deviation: hypot2(dx, dy),
      deltaV: hypot2(dvx, dvy),
      radius: r,
      altitude: r - R_E
    };
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function formatMeters(x) {
    const ax = Math.abs(x);
    if (ax >= 1000) return `${(x / 1000).toFixed(ax >= 10000 ? 0 : 1)} km`;
    return `${x.toFixed(ax >= 100 ? 0 : 1)} m`;
  }

  function formatPower(u) {
    return (u.reduce((a, b) => a + b, 0) * F_MAX * VC / 1e3).toFixed(1);
  }

  class OrbitGame {
    constructor(document) {
      this.document = document;
      this.canvas = document.getElementById("gameCanvas");
      this.ctx = this.canvas.getContext("2d");
      this.banner = document.getElementById("banner");
      this.modeRecover = document.getElementById("modeRecover");
      this.modeKick = document.getElementById("modeKick");
      this.recoverControls = document.getElementById("recoverControls");
      this.kickControls = document.getElementById("kickControls");
      this.controllerMode = document.getElementById("controllerMode");
      this.controllerState = document.getElementById("controllerState");
      this.throttle = document.getElementById("throttle");
      this.collisionStrength = document.getElementById("collisionStrength");
      this.collisionStrengthLabel = document.getElementById("collisionStrengthLabel");
      this.kickStrength = document.getElementById("kickStrength");
      this.kickStrengthLabel = document.getElementById("kickStrengthLabel");
      this.simRate = document.getElementById("simRate");
      this.pauseButton = document.getElementById("pauseButton");
      this.resetButton = document.getElementById("resetButton");
      this.newCollision = document.getElementById("newCollision");
      this.kickEnergyBar = document.getElementById("kickEnergyBar");
      this.readouts = {
        missionTime: document.getElementById("missionTime"),
        altitude: document.getElementById("altitude"),
        deviation: document.getElementById("deviation"),
        deltaV: document.getElementById("deltaV"),
        power: document.getElementById("power"),
        score: document.getElementById("score")
      };
      this.jetElements = Array.from(document.querySelectorAll("[data-jet]"));
      this.jetRows = Array.from(document.querySelectorAll("[data-jet-row]"));
      this.keys = new Set();
      this.buttonThrusters = new Set();
      this.mode = "recover";
      this.paused = false;
      this.ready = false;
      this.lost = false;
      this.messageTimer = 0;
      this.trail = [];
      this.refTrail = [];
      this.kickEnergy = 100;
      this.kicks = 0;
      this.maxDeviation = 0;
      this.stableTime = 0;
      this.controlAccum = CONTROL_DT;
      this.currentU = [0, 0, 0, 0];
      this.pointerDrag = null;
      this.t = 0;
      this.state = refOrbit(0);
      this.resize();
      this.bind();
      this.reset();
    }

    bind() {
      root.addEventListener("resize", () => this.resize());
      this.modeRecover.addEventListener("click", () => this.setMode("recover"));
      this.modeKick.addEventListener("click", () => this.setMode("kick"));
      this.pauseButton.addEventListener("click", () => this.togglePause());
      this.resetButton.addEventListener("click", () => this.reset());
      this.newCollision.addEventListener("click", () => {
        this.applyCollision(parseFloat(this.collisionStrength.value));
        this.say("Hard collision applied");
      });
      this.collisionStrength.addEventListener("input", () => this.updateCollisionLabel());
      this.kickStrength.addEventListener("input", () => this.updateKickLabel());

      this.document.querySelectorAll("[data-thruster]").forEach((button) => {
        const key = button.getAttribute("data-thruster");
        const down = (event) => {
          event.preventDefault();
          this.buttonThrusters.add(key);
          button.classList.add("pressed");
        };
        const up = () => {
          this.buttonThrusters.delete(key);
          button.classList.remove("pressed");
        };
        button.addEventListener("pointerdown", down);
        button.addEventListener("pointerup", up);
        button.addEventListener("pointercancel", up);
        button.addEventListener("pointerleave", up);
      });

      this.document.querySelectorAll("[data-kick]").forEach((button) => {
        button.addEventListener("click", () => {
          const key = button.getAttribute("data-kick");
          const dirs = {
            "+x": [1, 0],
            "-x": [-1, 0],
            "+y": [0, 1],
            "-y": [0, -1]
          };
          this.applyKick(dirs[key], parseFloat(this.kickStrength.value));
        });
      });

      root.addEventListener("keydown", (event) => {
        if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " ", "KeyW", "KeyA", "KeyS", "KeyD"].includes(event.code)) {
          event.preventDefault();
        }
        if (event.code === "Space" && this.mode === "kick") this.applyKick([Math.cos(this.t), Math.sin(this.t)], parseFloat(this.kickStrength.value));
        if (event.code === "KeyP") this.togglePause();
        this.keys.add(event.code);
      });
      root.addEventListener("keyup", (event) => this.keys.delete(event.code));

      this.canvas.addEventListener("pointerdown", (event) => {
        if (this.mode !== "kick" || !this.ready || this.lost) return;
        const p = this.canvasPoint(event);
        this.pointerDrag = { start: p, end: p };
        this.canvas.setPointerCapture(event.pointerId);
      });
      this.canvas.addEventListener("pointermove", (event) => {
        if (!this.pointerDrag) return;
        this.pointerDrag.end = this.canvasPoint(event);
      });
      this.canvas.addEventListener("pointerup", (event) => {
        if (!this.pointerDrag) return;
        const end = this.canvasPoint(event);
        const dx = end.x - this.pointerDrag.start.x;
        const dy = end.y - this.pointerDrag.start.y;
        const len = hypot2(dx, dy);
        if (len > 16) this.applyKick([dx / len, -dy / len], parseFloat(this.kickStrength.value));
        this.pointerDrag = null;
      });
      this.updateCollisionLabel();
      this.updateKickLabel();
    }

    canvasPoint(event) {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
      };
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const ratio = root.devicePixelRatio || 1;
      this.canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      this.canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.viewW = rect.width;
      this.viewH = rect.height;
      this.draw();
    }

    trainController() {
      const start = root.performance ? root.performance.now() : Date.now();
      const trainRes = new MemristiveReservoir(40, 7);
      const trained = trainMulti(trainRes, CONTROL_DT, 25, 5e-5);
      const deployRes = new MemristiveReservoir(40, 7);
      this.rc = new RCCtrl(deployRes, trained.WOut);
      this.ready = true;
      const elapsed = ((root.performance ? root.performance.now() : Date.now()) - start) / 1000;
      this.controllerState.textContent = `Reservoir ready - ${trained.samples} samples`;
      this.say(`Reservoir trained in ${elapsed.toFixed(1)} s`);
      this.reset();
    }

    setMode(mode) {
      if (this.mode === mode) return;
      this.mode = mode;
      this.modeRecover.classList.toggle("active", mode === "recover");
      this.modeKick.classList.toggle("active", mode === "kick");
      this.recoverControls.classList.toggle("hidden", mode !== "recover");
      this.kickControls.classList.toggle("hidden", mode !== "kick");
      this.reset();
    }

    togglePause() {
      this.paused = !this.paused;
      this.pauseButton.textContent = this.paused ? "Resume" : "Pause";
      this.say(this.paused ? "Paused" : "Running");
    }

    reset() {
      const ref = refOrbit(0);
      this.t = 0;
      this.state = ref.slice();
      this.currentU = [0, 0, 0, 0];
      this.trail = [];
      this.refTrail = [];
      this.lost = false;
      this.kickEnergy = 100;
      this.kicks = 0;
      this.maxDeviation = 0;
      this.stableTime = 0;
      this.controlAccum = CONTROL_DT;
      if (this.rc) this.rc.reset();
      if (this.mode === "recover") this.applyCollision(parseFloat(this.collisionStrength.value));
      this.say(this.mode === "recover" ? "Recover the orbit" : "Controller online");
      this.updateReadouts();
      this.draw();
    }

    applyCollision(strength) {
      const ref = refOrbit(this.t);
      const th = Math.atan2(ref[1], ref[0]);
      const er = [Math.cos(th), Math.sin(th)];
      const et = [-Math.sin(th), Math.cos(th)];
      const posAngle = Math.random() * 2 * Math.PI;
      const pos = (900 + Math.random() * 2600) + strength * 320;
      const vel = strength;
      const tangentialHit = Math.random() < 0.72;
      let velAngle = Math.random() * 2 * Math.PI;
      if (tangentialHit) {
        const prograde = Math.random() < 0.45 ? 0 : Math.PI;
        velAngle = prograde + (Math.random() - 0.5) * 0.65;
      }
      const px = er[0] * Math.cos(posAngle) + et[0] * Math.sin(posAngle);
      const py = er[1] * Math.cos(posAngle) + et[1] * Math.sin(posAngle);
      const vx = et[0] * Math.cos(velAngle) + er[0] * Math.sin(velAngle);
      const vy = et[1] * Math.cos(velAngle) + er[1] * Math.sin(velAngle);
      this.state[0] += px * pos;
      this.state[1] += py * pos;
      this.state[2] += vx * vel;
      this.state[3] += vy * vel;
      this.stableTime = 0;
      this.lost = false;
    }

    applyKick(dir, strength) {
      if (this.mode !== "kick" || !this.ready || this.lost) return;
      const cost = clamp(strength / 180 * 90, 8, 95);
      if (this.kickEnergy < cost) {
        this.say("Kick energy low");
        return;
      }
      this.state[2] += dir[0] * strength;
      this.state[3] += dir[1] * strength;
      this.state[0] += dir[0] * strength * 300;
      this.state[1] += dir[1] * strength * 300;
      this.kickEnergy -= cost;
      this.kicks += 1;
      this.stableTime = 0;
      this.controlAccum = CONTROL_DT;
      this.say(`Kick ${strength.toFixed(0)} m/s`);
    }

    updateKickLabel() {
      if (!this.kickStrengthLabel) return;
      const value = parseFloat(this.kickStrength.value);
      this.kickStrengthLabel.textContent = `Kick ${value.toFixed(0)} m/s`;
    }

    updateCollisionLabel() {
      if (!this.collisionStrengthLabel) return;
      const value = parseFloat(this.collisionStrength.value);
      this.collisionStrengthLabel.textContent = `Collision ${value.toFixed(0)} m/s`;
    }

    say(text) {
      this.banner.textContent = text;
      this.banner.style.opacity = "1";
      this.messageTimer = 2.2;
    }

    manualU() {
      const throttle = parseFloat(this.throttle.value);
      const active = (name) => this.buttonThrusters.has(name);
      const u = [0, 0, 0, 0];
      if (this.keys.has("ArrowRight") || this.keys.has("KeyD") || active("+x")) u[0] = throttle;
      if (this.keys.has("ArrowLeft") || this.keys.has("KeyA") || active("-x")) u[1] = throttle;
      if (this.keys.has("ArrowUp") || this.keys.has("KeyW") || active("+y")) u[2] = throttle;
      if (this.keys.has("ArrowDown") || this.keys.has("KeyS") || active("-y")) u[3] = throttle;
      return clampPower(u);
    }

    update(dtReal) {
      if (!this.ready || this.paused || this.lost) {
        if (this.messageTimer > 0) this.messageTimer -= dtReal;
        this.fadeBanner();
        return;
      }

      let remaining = Math.min(0.07, dtReal) * parseFloat(this.simRate.value);
      while (remaining > 1e-9) {
        const h = Math.min(1.5, remaining);
        if (this.mode === "recover") {
          this.currentU = this.manualU();
        } else {
          this.controlAccum += h;
          if (this.controlAccum >= CONTROL_DT) {
            this.currentU = this.rc.compute(this.t, this.state, CONTROL_DT);
            this.controlAccum = 0;
          }
        }
        this.state = integrate(this.state, this.currentU, h);
        this.t += h;
        remaining -= h;
      }

      const rel = relativeState(this.state, this.t);
      this.maxDeviation = Math.max(this.maxDeviation, rel.deviation);
      const stable = rel.deviation < 90 && rel.deltaV < 0.08;
      this.stableTime = stable ? this.stableTime + dtReal * parseFloat(this.simRate.value) : 0;
      this.kickEnergy = Math.min(100, this.kickEnergy + dtReal * 4.5);

      if (this.trail.length === 0 || this.t - this.trail[this.trail.length - 1].t > 12) {
        this.trail.push({ t: this.t, x: this.state[0], y: this.state[1] });
        const ref = refOrbit(this.t);
        this.refTrail.push({ t: this.t, x: ref[0], y: ref[1] });
        if (this.trail.length > 720) this.trail.shift();
        if (this.refTrail.length > 720) this.refTrail.shift();
      }

      if (rel.altitude <= CRASH_ALTITUDE) this.end("Crash");
      if (rel.altitude > ESCAPE_ALTITUDE || rel.deviation > ESCAPE_DEVIATION) this.end("Escape");
      if (this.messageTimer > 0) this.messageTimer -= dtReal;
      this.fadeBanner();
      this.updateReadouts(rel);
    }

    fadeBanner() {
      if (this.messageTimer <= 0 && this.ready && !this.lost) {
        this.banner.style.opacity = "0";
      }
    }

    end(reason) {
      this.lost = true;
      const result = this.mode === "kick" ? `${reason}: controller defeated` : `${reason}: mission lost`;
      this.say(result);
      this.banner.style.opacity = "1";
    }

    scoreValue(rel) {
      if (this.mode === "recover") {
        const base = Math.floor(this.t / 12);
        const recovery = Math.floor(this.stableTime * 2);
        const precision = Math.max(0, 250 - rel.deviation) / 10;
        return Math.floor(base + recovery + precision);
      }
      return Math.floor(this.maxDeviation / 20 + this.kicks * 30 + (this.lost ? 600 : 0));
    }

    updateReadouts(rel = relativeState(this.state, this.t)) {
      this.readouts.missionTime.textContent = formatTime(this.t);
      this.readouts.altitude.textContent = formatMeters(rel.altitude);
      this.readouts.deviation.textContent = formatMeters(rel.deviation);
      this.readouts.deltaV.textContent = `${rel.deltaV.toFixed(3)} m/s`;
      this.readouts.power.textContent = `${formatPower(this.currentU)} W`;
      this.readouts.score.textContent = `${this.scoreValue(rel)}`;
      this.kickEnergyBar.style.transform = `scaleX(${this.kickEnergy / 100})`;
      this.updateActuators();
    }

    updateActuators() {
      const labels = ["+x", "-x", "+y", "-y"];
      const values = {};
      for (let i = 0; i < labels.length; i += 1) values[labels[i]] = this.currentU[i] || 0;

      if (this.controllerMode) {
        if (this.lost && this.mode === "kick") this.controllerMode.textContent = "Controller defeated";
        else if (this.lost) this.controllerMode.textContent = "Mission lost";
        else this.controllerMode.textContent = this.mode === "recover" ? "Human manual" : "Reservoir controller";
      }

      this.jetElements.forEach((jet) => {
        const value = values[jet.dataset.jet] || 0;
        jet.classList.toggle("active", value > 0.04);
        jet.style.opacity = `${0.14 + value * 0.86}`;
        jet.style.setProperty("--jet-scale", `${0.72 + value * 0.72}`);
      });

      this.jetRows.forEach((row) => {
        const value = values[row.dataset.jetRow] || 0;
        row.style.setProperty("--level", `${Math.round(value * 100)}%`);
        const out = row.querySelector("strong");
        if (out) out.textContent = `${Math.round(value * 100)}%`;
      });
    }

    drawOrbitPanel(ctx, x, y, w, h) {
      const cx = x + w * 0.5;
      const cy = y + h * 0.5;
      const scale = Math.min(w, h) * 0.46 / (R0 + 850e3);
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, y, w, h);
      ctx.clip();
      ctx.fillStyle = "#101316";
      ctx.fillRect(x, y, w, h);

      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      for (let i = -4; i <= 4; i += 1) {
        ctx.beginPath();
        ctx.moveTo(cx + i * w / 8, y);
        ctx.lineTo(cx + i * w / 8, y + h);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, cy + i * h / 8);
        ctx.lineTo(x + w, cy + i * h / 8);
        ctx.stroke();
      }

      ctx.fillStyle = "#63717a";
      ctx.beginPath();
      ctx.arc(cx, cy, R_E * scale, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#c9d2d4";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.arc(cx, cy, R0 * scale, 0, 2 * Math.PI);
      ctx.stroke();

      if (this.trail.length > 1) {
        ctx.strokeStyle = "#f0c36d";
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        this.trail.forEach((p, i) => {
          const px = cx + p.x * scale;
          const py = cy - p.y * scale;
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }

      const ref = refOrbit(this.t);
      const sx = cx + this.state[0] * scale;
      const sy = cy - this.state[1] * scale;
      const rx = cx + ref[0] * scale;
      const ry = cy - ref[1] * scale;
      ctx.fillStyle = "#f6f7f2";
      ctx.beginPath();
      ctx.arc(rx, ry, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = this.mode === "kick" ? "#df6d38" : "#33b7a8";
      this.drawSatelliteGlyph(ctx, sx, sy, Math.atan2(-this.state[3], this.state[2]), 1.05, ctx.fillStyle);
      this.drawThrusterArrows(ctx, sx, sy, 24);
      ctx.fillStyle = "rgba(255,255,255,0.9)";
      ctx.font = "13px ui-sans-serif, system-ui";
      ctx.fillText(`Orbit ${(this.t / T_ORB).toFixed(2)}`, x + 16, y + 24);
      ctx.restore();
    }

    drawSatelliteGlyph(ctx, sx, sy, angle, size, accent) {
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(angle);
      ctx.scale(size, size);

      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1.2;
      ctx.fillStyle = "#146a8a";
      ctx.fillRect(-24, -5, 15, 10);
      ctx.fillRect(9, -5, 15, 10);
      ctx.strokeRect(-24, -5, 15, 10);
      ctx.strokeRect(9, -5, 15, 10);

      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.beginPath();
      ctx.moveTo(-9, 0);
      ctx.lineTo(-5, 0);
      ctx.moveTo(5, 0);
      ctx.lineTo(9, 0);
      ctx.stroke();

      ctx.fillStyle = "#f6f7f2";
      ctx.strokeStyle = "#101316";
      ctx.lineWidth = 1.4;
      this.roundRect(ctx, -7, -9, 14, 18, 3);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = accent;
      ctx.fillRect(-4, 3, 8, 3);
      ctx.fillStyle = "#101316";
      ctx.beginPath();
      ctx.arc(0, -3, 2.1, 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();
    }

    roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r);
      ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h);
      ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r);
      ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }

    drawThrusterArrows(ctx, sx, sy, length) {
      const dirs = [[1, 0], [-1, 0], [0, -1], [0, 1]];
      ctx.save();
      ctx.strokeStyle = "#33b7a8";
      ctx.lineWidth = 2;
      for (let i = 0; i < 4; i += 1) {
        const u = this.currentU[i];
        if (u <= 0.04) continue;
        const dx = dirs[i][0] * length * u;
        const dy = dirs[i][1] * length * u;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + dx, sy + dy);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(sx + dx, sy + dy, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = "#33b7a8";
        ctx.fill();
      }
      ctx.restore();
    }

    drawLocalPanel(ctx, x, y, w, h) {
      const rel = relativeState(this.state, this.t);
      const maxAxis = Math.max(300, Math.min(400000, rel.deviation * 2.2 + 400));
      const scale = Math.min(w, h) * 0.42 / maxAxis;
      const cx = x + w * 0.5;
      const cy = y + h * 0.53;
      ctx.save();
      ctx.fillStyle = "#f6f7f2";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = "#d5d9d2";
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);

      ctx.strokeStyle = "#d5d9d2";
      ctx.lineWidth = 1;
      for (let i = -4; i <= 4; i += 1) {
        ctx.beginPath();
        ctx.moveTo(cx + i * w / 8, y + 34);
        ctx.lineTo(cx + i * w / 8, y + h - 12);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x + 12, cy + i * h / 8);
        ctx.lineTo(x + w - 12, cy + i * h / 8);
        ctx.stroke();
      }

      ctx.strokeStyle = "#147d7e";
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.arc(cx, cy, 90 * scale, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.fillStyle = "#1b1d1e";
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.fill();

      const sx = cx + rel.along * scale;
      const sy = cy - rel.radial * scale;
      ctx.strokeStyle = "#b77216";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(sx, sy);
      ctx.stroke();
      this.drawSatelliteGlyph(ctx, sx, sy, Math.atan2(-rel.radialV, rel.alongV), 1.15, this.lost ? "#b33434" : "#df6d38");

      ctx.fillStyle = "#1b1d1e";
      ctx.font = "13px ui-sans-serif, system-ui";
      ctx.fillText("Relative orbit", x + 14, y + 23);
      ctx.fillStyle = "#687076";
      ctx.fillText(`window +/- ${formatMeters(maxAxis)}`, x + 14, y + h - 14);

      if (this.pointerDrag) {
        ctx.strokeStyle = "#b33434";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.pointerDrag.start.x, this.pointerDrag.start.y);
        ctx.lineTo(this.pointerDrag.end.x, this.pointerDrag.end.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    drawMemoryPanel(ctx, x, y, w, h) {
      ctx.save();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = "#d5d9d2";
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
      ctx.fillStyle = "#1b1d1e";
      ctx.font = "13px ui-sans-serif, system-ui";
      ctx.fillText("Memory state", x + 14, y + 23);
      const hist = this.rc ? this.rc.res.hist : [];
      const latest = hist.length ? hist[hist.length - 1] : new Array(40).fill(0.5);
      const barW = (w - 28) / latest.length;
      for (let i = 0; i < latest.length; i += 1) {
        const value = latest[i];
        const bh = (h - 44) * value;
        ctx.fillStyle = `rgba(20, 125, 126, ${0.32 + value * 0.58})`;
        ctx.fillRect(x + 14 + i * barW, y + h - 14 - bh, Math.max(1, barW - 1), bh);
      }
      ctx.restore();
    }

    draw() {
      if (!this.ctx || !this.viewW || !this.viewH) return;
      const ctx = this.ctx;
      const w = this.viewW;
      const h = this.viewH;
      ctx.clearRect(0, 0, w, h);
      const narrow = w < 720;
      if (narrow) {
        const orbitH = Math.max(250, h * 0.58);
        this.drawOrbitPanel(ctx, 0, 0, w, orbitH);
        this.drawLocalPanel(ctx, 0, orbitH, w, h - orbitH);
      } else {
        const sideW = Math.max(270, Math.min(380, w * 0.34));
        this.drawOrbitPanel(ctx, 0, 0, w - sideW, h);
        this.drawLocalPanel(ctx, w - sideW, 0, sideW, h * 0.66);
        this.drawMemoryPanel(ctx, w - sideW, h * 0.66, sideW, h * 0.34);
      }
    }

    frame(now) {
      if (!this.last) this.last = now;
      const dt = (now - this.last) / 1000;
      this.last = now;
      this.update(dt);
      this.draw();
      root.requestAnimationFrame((t) => this.frame(t));
    }
  }

  function boot() {
    const game = new OrbitGame(root.document);
    root.MemristiveOrbitGameInstance = game;
    root.setTimeout(() => {
      game.trainController();
      root.requestAnimationFrame((t) => game.frame(t));
    }, 40);
  }

  const API = {
    constants: {
      GM,
      R_E,
      ALT,
      R0,
      VC,
      T_ORB,
      N_MOT,
      M,
      F_MAX,
      P_MAX,
      RHO,
      CD,
      AREA,
      CRASH_ALTITUDE,
      ESCAPE_ALTITUDE,
      ESCAPE_DEVIATION
    },
    refOrbit,
    eom,
    integrate,
    f2u,
    cwCtrl,
    MemristiveReservoir,
    RCCtrl,
    trainMulti,
    relativeState
  };

  if (typeof module !== "undefined" && module.exports) module.exports = API;
  if (typeof root.window !== "undefined") {
    root.MemristiveOrbitGame = API;
    root.window.addEventListener("DOMContentLoaded", boot);
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
