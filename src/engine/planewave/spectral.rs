//! The exact discrete incident field of a stratified (and possibly
//! dispersive) background, for [`PlaneWave`](super::PlaneWave) (see the
//! [parent docs](super#layered-and-dispersive-backgrounds)).
//!
//! The incident wave is written as
//!
//! ```text
//! x(r, t) = g(s, t − p·(r − r_ref))
//! ```
//!
//! with `s` the stratification axis and `p` a fixed transverse slowness
//! (`p_s = 0`). Per frequency, `e^{−jω p·r}` is an exact eigenfunction of
//! the transverse Yee differences on a uniform transverse mesh, so the 3-D
//! update equations reduce to a 1-D system along one column of edges
//! (six unknowns per level: `u_t1, u_t2, i_s` on node levels, `u_s, i_t1,
//! i_t2` on dual levels), built from the engine's own coefficients
//! (semi-implicit losses and the discrete dispersive susceptibilities
//! included). The column ends either on the domain's PEC/PMC planes or in
//! a uniform extension of the end medium with a stretched-coordinate PML;
//! the incident wave is radiated by a current sheet in the extension on
//! the side the wave comes from, and normalised with the same sheet in the
//! same line filled with the incidence medium (the pure incident wave).
//! The spectra of the column fields times the excitation spectrum are
//! inverse-FFT'd (radix-2 Cooley–Tukey) into time series sampled at
//! `dt/2`, read with cubic interpolation at each edge's transverse delay.
//!
//! Literature: plane waves in stratified media reduce to 1-D problems at a
//! fixed transverse wavenumber (W. C. Chew, *Waves and Fields in
//! Inhomogeneous Media*, ch. 2); TF/SF sources for layered FDTD media are
//! treated e.g. by I. R. Capoglu and G. S. Smith, IEEE Trans. Antennas
//! Propag. 56(1), 2008. Here the 1-D problem is the engine's discrete one,
//! solved per frequency; the extension PML is the stretched-coordinate
//! PML of W. C. Chew and W. H. Weedon, Microwave Opt. Technol. Lett.
//! 7(13), 1994, in the frequency domain.

use crate::axis::Axis;
use crate::constants::{EPS0, MU0};
use crate::engine::dispersive::{Pole, Scheme};
use crate::engine::grid::Grid;
use crate::post::Complex64;
use crate::{Error, Result};

/// Time samples per engine step of the stored series.
pub(super) const UP: usize = 2;
/// Uniform cells of each extension between the column and its PML.
const EXT_CELLS: usize = 8;
/// Cells of the extension PML.
const PML_CELLS: usize = 48;
/// Design reflection of the extension PML at the incidence angle.
const PML_R: f64 = 1e-10;
/// Relative level below which an excitation bin is not solved.
const BIN_THRESHOLD: f64 = 1e-13;
/// Relative level the series must have decayed to in the last quarter of
/// the FFT window.
const TAIL: f64 = 1e-6;
/// Largest incident field before `t = 0`, relative to the peak.
const ACAUSAL: f64 = 1e-5;
/// Largest FFT window (samples).
const MAX_FFT: usize = 1 << 18;

/// How the stratified column ends at a domain face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum LayerEnd {
    /// The end medium continues to infinity (an absorbing boundary).
    #[default]
    Open,
    /// The domain's PEC face (tangential E zero on the last node plane).
    Pec,
    /// The domain's PMC face (no H beyond the last node plane).
    Pmc,
}

/// Material of one column edge: the engine coefficients `ca`, `cb` (or
/// `da`, `db`), its geometry factor (dual area over length, or area over
/// dual length) and its dispersive poles.
#[derive(Debug, Clone)]
pub(super) struct EdgeMed {
    pub e: bool,
    pub ca: f64,
    pub cb: f64,
    pub geo: f64,
    pub poles: Vec<Pole>,
}

impl EdgeMed {
    /// Admittance per unit geometry, `Y_e ≈ jωε` (E) or `Z_h ≈ jωμ` (H),
    /// of the discrete update at angular frequency `w`; `None` for PEC.
    pub fn y_at(&self, w: f64, dt: f64, scheme: Scheme) -> Option<Complex64> {
        if self.cb == 0.0 {
            return None;
        }
        let th = 0.5 * w * dt;
        let zh = Complex64::cis(th);
        let a = (zh - zh.conj() * self.ca) / (self.cb * self.geo);
        if self.poles.is_empty() {
            return Some(a);
        }
        let f = w / (2.0 * core::f64::consts::PI);
        let chi: Complex64 = self
            .poles
            .iter()
            .map(|p| match scheme {
                Scheme::Trapezoidal => p.discrete_susceptibility(f, dt),
                Scheme::Leapfrog => p.leapfrog_susceptibility(f, dt),
            })
            .sum();
        let om = 2.0 / dt * th.sin();
        let c = if self.e { EPS0 } else { MU0 };
        Some(a + Complex64::new(0.0, om * c) * chi)
    }
}

/// Kind of one of the six column unknowns of a level pair.
#[derive(Debug, Clone, Copy)]
struct Slot {
    e: bool,
    axis: Axis,
}

/// The slots in order: `u_t1, u_t2, i_s` on node levels, `u_s, i_t1, i_t2`
/// on dual levels.
fn slots(s: Axis) -> [Slot; 6] {
    let (t1, t2) = s.others();
    let sl = |e, axis| Slot { e, axis };
    [
        sl(true, t1),
        sl(true, t2),
        sl(false, s),
        sl(true, s),
        sl(false, t1),
        sl(false, t2),
    ]
}

pub(super) fn slot_of(s: Axis, e: bool, axis: Axis) -> usize {
    let (t1, _) = s.others();
    match (e, axis == s, axis == t1) {
        (true, false, true) => 0,
        (true, false, false) => 1,
        (false, true, _) => 2,
        (true, true, _) => 3,
        (false, false, true) => 4,
        (false, false, false) => 5,
    }
}

/// One circulation term of a slot's update equation.
#[derive(Debug, Clone, Copy)]
struct Term {
    slot: usize,
    dl: isize,
    sign: f64,
    /// Transverse displacement \[m\] of the referenced edge from the
    /// equation's own edge (3-D, zero along `s`).
    dr: [f64; 3],
}

/// Transverse coordinate of a node (`dual = false`) or dual node of
/// index `i` (extrapolated with the end cell beyond the mesh).
fn coord(grid: &Grid, t: Axis, i: isize, dual: bool) -> f64 {
    let l = grid.lines(t);
    let n = l.len() as isize;
    let node = |j: isize| -> f64 {
        if j < 0 {
            l[0] + j as f64 * (l[1] - l[0])
        } else if j >= n {
            l[l.len() - 1] + (j - n + 1) as f64 * (l[l.len() - 1] - l[l.len() - 2])
        } else {
            l[j as usize]
        }
    };
    if dual {
        0.5 * (node(i) + node(i + 1))
    } else {
        node(i)
    }
}

/// The circulation terms of the six slots (the engine kernels' signs:
/// `u_a` reads `i_c(p) − i_c(p − e_b) − i_b(p) + i_b(p − e_c)`, `i_a` reads
/// `u_c(q + e_b) − u_c(q) − u_b(q + e_c) + u_b(q)`, `(b, c) = a.others()`).
fn terms(grid: &Grid, s: Axis, col: [usize; 3]) -> [Vec<Term>; 6] {
    let sl = slots(s);
    // Transverse position of an edge (E: centre along its axis).
    let pos = |e: bool, axis: Axis, off: [isize; 3]| -> [f64; 3] {
        let mut r = [0.0; 3];
        for t in Axis::ALL {
            if t == s {
                continue;
            }
            let i = col[t.index()] as isize + off[t.index()];
            r[t.index()] = coord(grid, t, i, (t == axis) == e);
        }
        r
    };
    sl.map(|me| {
        let (b, c) = me.axis.others();
        let unit = |a: Axis, k: isize| {
            let mut o = [0isize; 3];
            o[a.index()] = k;
            o
        };
        let list: [(Axis, [isize; 3], f64); 4] = if me.e {
            [
                (c, [0; 3], 1.0),
                (c, unit(b, -1), -1.0),
                (b, [0; 3], -1.0),
                (b, unit(c, -1), 1.0),
            ]
        } else {
            [
                (c, unit(b, 1), 1.0),
                (c, [0; 3], -1.0),
                (b, unit(c, 1), -1.0),
                (b, [0; 3], 1.0),
            ]
        };
        list.iter()
            .map(|&(ax, off, sign)| {
                let e = !me.e;
                let p = pos(e, ax, off);
                // Relative to the equation's own edge: every slot's
                // amplitude has its phase origin at r_ref.
                let p0 = pos(me.e, me.axis, [0; 3]);
                Term {
                    slot: slot_of(s, e, ax),
                    dl: off[s.index()],
                    sign,
                    dr: [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]],
                }
            })
            .collect()
    })
}

/// One level of the 1-D line: position along `s`, the length along `s`
/// of its edges (dual length on node levels, cell on dual levels), PML
/// loss, and the material of each of its three slots.
#[derive(Debug, Clone)]
struct Level {
    pos: f64,
    len: f64,
    sigma: f64,
    med: [Option<usize>; 3],
    engine: Option<usize>,
}

/// The 1-D line: node levels and the dual levels between them.
#[derive(Debug, Clone)]
struct Line {
    nodes: Vec<Level>,
    duals: Vec<Level>,
    /// Node level of the source sheet.
    src: usize,
}

/// Transverse geometry of the column (regular cells around it).
#[derive(Debug, Clone, Copy)]
struct Transverse {
    /// `Δt1, Δt2`.
    d: [f64; 2],
    /// `Δt1*, Δt2*`.
    dd: [f64; 2],
}

impl Transverse {
    /// Geometry factor of slot `k` at a level of length `len` along `s`.
    fn geo(&self, k: usize, len: Complex64) -> Complex64 {
        let [d1, d2] = self.d;
        let [e1, e2] = self.dd;
        match k {
            0 => len * (e2 / d1),
            1 => len * (e1 / d2),
            2 => (d1 * d2) / len,
            3 => (e1 * e2) / len,
            4 => len * (d2 / e1),
            _ => len * (d1 / e2),
        }
    }

    /// Edge length of slot `k` (for field values).
    fn length(&self, k: usize, len: f64) -> f64 {
        match k {
            0 => self.d[0],
            1 => self.d[1],
            4 => self.dd[0],
            5 => self.dd[1],
            _ => len,
        }
    }
}

/// Everything needed to synthesise the incident field.
pub(super) struct Setup<'a> {
    pub grid: &'a Grid,
    pub dt: f64,
    /// Stratification axis.
    pub s: Axis,
    /// Transverse node indices of the column (the `s` entry is unused).
    pub col: [usize; 3],
    /// Layered: the column spans the domain along `s`, ending as `ends`
    /// says. Homogeneous: it spans the box (± 2 nodes) with open ends.
    pub layered: bool,
    pub ends: [LayerEnd; 2],
    /// The total-field box.
    pub lo: [usize; 3],
    pub hi: [usize; 3],
    /// Materials of the column edges.
    pub meds: Vec<EdgeMed>,
    /// Index into `meds` of slot `k` at engine level `l` (`None`: PEC).
    pub med_index: Box<dyn Fn(usize, usize) -> Option<usize> + 'a>,
    /// Time integration of the poles.
    pub scheme: Scheme,
    pub k_hat: [f64; 3],
    pub e0: [f64; 3],
    /// Frequency \[Hz\] of the incidence medium's phase velocity (sets
    /// the transverse slowness).
    pub match_frequency: f64,
    /// Time \[s\] after which the amplitude function is zero.
    pub duration: f64,
}

/// The synthesised incident field.
#[derive(Debug, Clone)]
pub(super) struct Incident {
    /// Series per (engine level − `k0`)·6 + slot, field values (V/m, A/m)
    /// at `t = (m − pre)·dt/UP`.
    pub series: Vec<Vec<f32>>,
    pub pre: usize,
    pub k0: usize,
    pub k1: usize,
    /// Transverse slowness \[s/m\].
    pub p: [f64; 3],
    /// Last time \[s\] any E series exceeds 1e-6 of its peak.
    pub t_last: f64,
}

impl Incident {
    /// Series index of `(e, axis)` at engine level `k`.
    pub fn id(&self, s: Axis, e: bool, axis: Axis, k: usize) -> Option<usize> {
        (self.k0..=self.k1)
            .contains(&k)
            .then(|| (k - self.k0) * 6 + slot_of(s, e, axis))
    }

    /// Value of series `id` at time `t` \[s\] for step `dt`.
    pub fn value(&self, id: usize, t: f64, dt: f64) -> f64 {
        let x = t / (dt / UP as f64) + self.pre as f64;
        if x < 1.0 {
            return 0.0;
        }
        let (m, w) = super::stencil(x);
        let v = &self.series[id];
        if m + 3 >= v.len() {
            return 0.0;
        }
        (0..4).map(|q| w[q] * v[m + q] as f64).sum()
    }
}

/// Banded LU with partial pivoting (`kl` sub-, `ku` super-diagonals).
struct Band {
    n: usize,
    kl: usize,
    /// Upper bandwidth after fill-in (`kl + ku`).
    ku: usize,
    /// Row-major band: row `i`, column `j` at `i·w + (j + kl − i)`.
    a: Vec<Complex64>,
    piv: Vec<usize>,
}

impl Band {
    fn new(n: usize, kl: usize, ku: usize) -> Self {
        let ku = kl + ku;
        Self {
            n,
            kl,
            ku,
            a: vec![Complex64::ZERO; n * (kl + ku + 1)],
            piv: vec![0; n],
        }
    }

    fn w(&self) -> usize {
        self.kl + self.ku + 1
    }

    fn at(&mut self, i: usize, j: usize) -> &mut Complex64 {
        let w = self.w();
        &mut self.a[i * w + (j + self.kl - i)]
    }

    fn get(&self, i: usize, j: usize) -> Complex64 {
        if j + self.kl < i || j > i + self.ku {
            return Complex64::ZERO;
        }
        self.a[i * self.w() + (j + self.kl - i)]
    }

    fn factor(&mut self) -> Result<()> {
        let n = self.n;
        for k in 0..n {
            let last = (k + self.kl).min(n - 1);
            let (mut p, mut best) = (k, self.get(k, k).norm());
            for i in k + 1..=last {
                let v = self.get(i, k).norm();
                if v > best {
                    (p, best) = (i, v);
                }
            }
            if !(best > 0.0 && best.is_finite()) {
                return Err(Error::Invalid(
                    "plane wave: singular layered-medium system".into(),
                ));
            }
            self.piv[k] = p;
            let jmax = (k + self.ku).min(n - 1);
            if p != k {
                for j in k..=jmax {
                    let (a, b) = (self.get(k, j), self.get(p, j));
                    *self.at(k, j) = b;
                    *self.at(p, j) = a;
                }
            }
            let inv = self.get(k, k).inv();
            for i in k + 1..=last {
                let l = self.get(i, k) * inv;
                if l == Complex64::ZERO {
                    continue;
                }
                *self.at(i, k) = l;
                for j in k + 1..=jmax {
                    let u = self.get(k, j);
                    if u != Complex64::ZERO {
                        *self.at(i, j) -= l * u;
                    }
                }
            }
        }
        Ok(())
    }

    fn solve(&self, b: &mut [Complex64]) {
        let n = self.n;
        for k in 0..n {
            let p = self.piv[k];
            if p != k {
                b.swap(k, p);
            }
            let last = (k + self.kl).min(n - 1);
            let bk = b[k];
            for (i, bi) in b.iter_mut().enumerate().take(last + 1).skip(k + 1) {
                *bi -= self.get(i, k) * bk;
            }
        }
        for k in (0..n).rev() {
            let jmax = (k + self.ku).min(n - 1);
            let s = (k + 1..=jmax).fold(b[k], |s, j| s - self.get(k, j) * b[j]);
            b[k] = s / self.get(k, k);
        }
    }
}

/// In-place radix-2 complex FFT (`inverse`: `e^{+j…}`, no scaling).
fn fft(x: &mut [Complex64], inverse: bool) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * core::f64::consts::PI / len as f64;
        let half = len / 2;
        // Twiddles from exact angles (no recurrence drift).
        let tw: Vec<Complex64> = (0..half).map(|k| Complex64::cis(ang * k as f64)).collect();
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let a = x[start + k];
                let b = x[start + k + half] * tw[k];
                x[start + k] = a + b;
                x[start + k + half] = a - b;
            }
        }
        len <<= 1;
    }
}

/// Solution of one line at one frequency for the two sources (`u_t1` and
/// `u_t2` sheets): per source the six slot values per level, as
/// `[node level][slot]` (dual slots of the last node level unused).
type Solution = [Vec<[Complex64; 6]>; 2];

struct Solver<'a> {
    terms: &'a [Vec<Term>; 6],
    tr: Transverse,
    dt: f64,
    scheme: Scheme,
    meds: &'a [EdgeMed],
    p: [f64; 3],
}

impl Solver<'_> {
    fn solve(&self, line: &Line, w: f64) -> Result<Solution> {
        let ln = line.nodes.len();
        let y: Vec<Option<Complex64>> = self
            .meds
            .iter()
            .map(|m| m.y_at(w, self.dt, self.scheme))
            .collect();
        let phase = |t: &Term| {
            let x = t.dr[0] * self.p[0] + t.dr[1] * self.p[1] + t.dr[2] * self.p[2];
            Complex64::cis(-w * x) * t.sign
        };
        let ph: [Vec<Complex64>; 6] =
            core::array::from_fn(|k| self.terms[k].iter().map(phase).collect());
        // Admittance of slot k at level l (None: forced zero).
        let adm = |l: usize, k: usize| -> Option<Complex64> {
            let lev = if k < 3 {
                &line.nodes[l]
            } else {
                line.duals.get(l)?
            };
            let y = y[lev.med[k % 3]?]?;
            let kappa = if lev.sigma > 0.0 {
                Complex64::new(1.0, -lev.sigma / w)
            } else {
                Complex64::ONE
            };
            Some(y * self.tr.geo(k, kappa * lev.len))
        };
        // Rows over 6·ln unknowns, as sparse (col, coef) lists.
        let idx = |l: usize, k: usize| 6 * l + k;
        let valid =
            |l: isize, k: usize| l >= 0 && (l as usize) < ln && (k < 3 || (l as usize) + 1 < ln);
        let mut rows: Vec<Vec<(usize, Complex64)>> = Vec::with_capacity(6 * ln);
        for l in 0..ln {
            for (k, phk) in ph.iter().enumerate() {
                let mut r = Vec::with_capacity(5);
                match adm(l, k) {
                    Some(a) if valid(l as isize, k) => {
                        r.push((idx(l, k), a));
                        let e = k == 0 || k == 1 || k == 3;
                        for (t, &c) in self.terms[k].iter().zip(phk) {
                            let ll = l as isize + t.dl;
                            if !valid(ll, t.slot) {
                                continue;
                            }
                            // Two terms may read copies of the same
                            // unknown (transverse neighbours): merge.
                            let (j, c) = (idx(ll as usize, t.slot), if e { -c } else { c });
                            match r.iter_mut().find(|x: &&mut (usize, Complex64)| x.0 == j) {
                                Some(x) => x.1 += c,
                                None => r.push((j, c)),
                            }
                        }
                    }
                    _ => r.push((idx(l, k), Complex64::ONE)),
                }
                rows.push(r);
            }
        }
        // Eliminate i_s (slot 2) and u_s (slot 3): their rows only involve
        // the same level's slots 0, 1 (resp. 4, 5).
        let mut elim: Vec<(usize, Vec<(usize, Complex64)>)> = Vec::with_capacity(2 * ln);
        for l in 0..ln {
            for k in [2, 3] {
                let v = idx(l, k);
                let pivot = rows[v].clone();
                let d = pivot
                    .iter()
                    .find(|x| x.0 == v)
                    .map_or(Complex64::ONE, |x| x.1);
                // x_v = −Σ c_j x_j / d.
                let sub: Vec<(usize, Complex64)> = pivot
                    .iter()
                    .filter(|x| x.0 != v)
                    .map(|&(j, c)| (j, -c / d))
                    .collect();
                for (r, row) in rows.iter_mut().enumerate().skip(idx(l, 0)).take(6) {
                    if r == v {
                        continue;
                    }
                    if let Some(pos) = row.iter().position(|x| x.0 == v) {
                        let c = row.remove(pos).1;
                        for &(j, s) in &sub {
                            match row.iter_mut().find(|x| x.0 == j) {
                                Some(x) => x.1 += c * s,
                                None => row.push((j, c * s)),
                            }
                        }
                    }
                }
                elim.push((v, sub));
            }
        }
        // Reduced unknowns: slots 0, 1, 4, 5 → 4·l + [0, 1, 2, 3].
        let red = |j: usize| -> usize {
            let (l, k) = (j / 6, j % 6);
            4 * l + [0, 1, 9, 9, 2, 3][k]
        };
        let n = 4 * ln;
        let (mut kl, mut ku) = (0usize, 0usize);
        for (i, r) in rows.iter().enumerate() {
            if i % 6 == 2 || i % 6 == 3 {
                continue;
            }
            let ri = red(i);
            for &(j, _) in r {
                let cj = red(j);
                if cj < ri {
                    kl = kl.max(ri - cj);
                } else {
                    ku = ku.max(cj - ri);
                }
            }
        }
        let mut band = Band::new(n, kl, ku);
        for (i, r) in rows.iter().enumerate() {
            if i % 6 == 2 || i % 6 == 3 {
                continue;
            }
            let ri = red(i);
            for &(j, c) in r {
                *band.at(ri, red(j)) += c;
            }
        }
        band.factor()?;
        let mut out: Solution = [Vec::new(), Vec::new()];
        for (q, o) in out.iter_mut().enumerate() {
            let mut b = vec![Complex64::ZERO; n];
            b[red(idx(line.src, q))] = Complex64::ONE;
            band.solve(&mut b);
            let mut full = vec![Complex64::ZERO; 6 * ln];
            for (j, x) in full.iter_mut().enumerate() {
                let k = j % 6;
                if k != 2 && k != 3 {
                    *x = b[red(j)];
                }
            }
            for (v, sub) in &elim {
                full[*v] = sub.iter().map(|&(j, c)| c * full[j]).sum();
            }
            *o = (0..ln)
                .map(|l| core::array::from_fn(|k| full[idx(l, k)]))
                .collect();
        }
        Ok(out)
    }
}

/// Extension levels beyond an open end: `count` node levels starting at
/// `start` (the junction node, outward) with spacing `delta` in direction
/// `dir` (±1); returns (nodes, duals between them), outermost last.
fn extension(
    start: f64,
    delta: f64,
    dir: f64,
    med_n: [Option<usize>; 3],
    med_d: [Option<usize>; 3],
    sigma_max: f64,
) -> (Vec<Level>, Vec<Level>) {
    let count = EXT_CELLS + PML_CELLS;
    let depth = PML_CELLS as f64 * delta;
    // `sigma_max` is given as ∫σ ds; the cubic profile integrates to
    // σ_max·depth/4.
    let sigma_max = 4.0 * sigma_max / depth;
    let sig = |m: f64| {
        let d = (m - EXT_CELLS as f64) * delta;
        if d > 0.0 {
            sigma_max * (d / depth).powi(3)
        } else {
            0.0
        }
    };
    let nodes = (0..count)
        .map(|m| Level {
            pos: start + dir * m as f64 * delta,
            len: delta,
            sigma: sig(m as f64),
            med: med_n,
            engine: None,
        })
        .collect();
    let duals = (0..count - 1)
        .map(|m| Level {
            pos: start + dir * (m as f64 + 0.5) * delta,
            len: delta,
            sigma: sig(m as f64 + 0.5),
            med: med_d,
            engine: None,
        })
        .collect();
    (nodes, duals)
}

/// Engine ranges of the column: first and last node level, first and last
/// dual level, and whether each end is open.
fn ranges(st: &Setup<'_>) -> (usize, usize, usize, usize, [bool; 2]) {
    let si = st.s.index();
    let ns = st.grid.lines(st.s).len();
    if st.layered {
        let lo_open = st.ends[0] == LayerEnd::Open;
        let hi_open = st.ends[1] == LayerEnd::Open;
        (
            usize::from(lo_open),
            if hi_open { ns - 2 } else { ns - 1 },
            0,
            ns - 2,
            [lo_open, hi_open],
        )
    } else {
        let da = st.lo[si].saturating_sub(2);
        let db = (st.hi[si] + 1).min(ns - 2);
        (da + 1, db, da, db, [true, true])
    }
}

/// Builds the line of the column (and the uniform reference line).
fn build_lines(st: &Setup<'_>, from_low: bool, sigma_max: f64) -> Result<(Line, Line)> {
    let g = st.grid;
    let s = st.s;
    let l = g.lines(s);
    let mi = |k: usize, lev: usize| (st.med_index)(k, lev);
    let (n_lo, n_hi, d_lo, d_hi, open) = ranges(st);
    let node_med = |k: usize| [mi(0, k), mi(1, k), mi(2, k)];
    let dual_med = |k: usize| [mi(3, k), mi(4, k), mi(5, k)];
    let node = |k: usize| Level {
        pos: l[k],
        len: g.dual_delta(s, k),
        sigma: 0.0,
        med: node_med(k),
        engine: Some(k),
    };
    let dual = |k: usize| Level {
        pos: g.dual_line(s, k),
        len: g.delta(s, k),
        sigma: 0.0,
        med: dual_med(k),
        engine: Some(k),
    };
    let mut nodes = Vec::new();
    let mut duals = Vec::new();
    if open[0] {
        let (mut en, mut ed) = extension(
            l[d_lo],
            g.delta(s, d_lo),
            -1.0,
            node_med(n_lo),
            dual_med(d_lo),
            sigma_max,
        );
        en[0].engine = Some(d_lo);
        en.reverse();
        ed.reverse();
        nodes.extend(en);
        duals.extend(ed);
        for k in n_lo..=n_hi {
            duals.push(dual(k - 1));
            nodes.push(node(k));
        }
    } else {
        nodes.push(node(n_lo));
        for k in n_lo + 1..=n_hi {
            duals.push(dual(k - 1));
            nodes.push(node(k));
        }
    }
    if open[1] {
        duals.push(dual(d_hi));
        let (mut en, ed) = extension(
            l[d_hi + 1],
            g.delta(s, d_hi),
            1.0,
            node_med(n_hi),
            dual_med(d_hi),
            sigma_max,
        );
        en[0].engine = Some(d_hi + 1);
        nodes.extend(en);
        duals.extend(ed);
    }
    debug_assert_eq!(duals.len() + 1, nodes.len());
    debug_assert!(nodes.windows(2).all(|w| w[0].pos < w[1].pos));
    let ln = nodes.len();
    let src = if from_low {
        PML_CELLS + 2
    } else {
        ln - 1 - (PML_CELLS + 2)
    };
    let line = Line { nodes, duals, src };
    // Reference: the same line (mesh, PMLs, source) filled with the
    // incidence medium: the pure incident wave.
    let (jn, jd) = if from_low { (n_lo, d_lo) } else { (n_hi, d_hi) };
    let mut reference = line.clone();
    for n in &mut reference.nodes {
        n.med = node_med(jn);
    }
    for d in &mut reference.duals {
        d.med = dual_med(jd);
    }
    // A closed far end would reflect the incident wave: open it.
    if !open[0] {
        let d = g.delta(s, 0);
        let (en, mut ed) = extension(l[0], d, -1.0, node_med(jn), dual_med(jd), sigma_max);
        let mut en: Vec<Level> = en.into_iter().skip(1).collect();
        en.reverse();
        ed.reverse();
        let k = en.len();
        reference.nodes[0].len = d;
        reference.nodes.splice(0..0, en);
        reference.duals.splice(0..0, ed);
        reference.src += k;
    }
    if !open[1] {
        let ns = l.len();
        let d = g.delta(s, ns - 2);
        let (en, ed) = extension(l[ns - 1], d, 1.0, node_med(jn), dual_med(jd), sigma_max);
        reference.nodes.last_mut().expect("non-empty").len = d;
        reference.nodes.extend(en.into_iter().skip(1));
        reference.duals.extend(ed);
    }
    debug_assert_eq!(reference.duals.len() + 1, reference.nodes.len());
    Ok((line, reference))
}

/// Synthesises the incident field (see the [module docs](self)).
pub(super) fn synthesise(st: &Setup<'_>, amplitude: &dyn Fn(f64) -> f64) -> Result<Incident> {
    let g = st.grid;
    let s = st.s;
    let si = s.index();
    let (t1, t2) = s.others();
    let ks = st.k_hat[si];
    if ks.abs() < 1e-3 {
        return Err(Error::Invalid(format!(
            "plane wave: the propagation direction {:?} is (nearly) parallel to the layers \
             (normal {s})",
            st.k_hat
        )));
    }
    if !(st.duration.is_finite() && st.duration >= 0.0) {
        return Err(Error::Invalid(
            "plane wave: a layered or dispersive background needs the duration of the \
             excitation (PlaneWaveSpec::with_duration)"
                .into(),
        ));
    }
    let from_low = ks > 0.0;
    let dt = st.dt;
    let tr = {
        let reg = |t: Axis| {
            let c = st.col[t.index()];
            let n = g.lines(t).len();
            let d = g.delta(t, c.min(n - 2));
            let dd = if c > 0 && c + 1 < n {
                g.dual_delta(t, c)
            } else {
                d
            };
            (d, dd)
        };
        let (a, b) = (reg(t1), reg(t2));
        Transverse {
            d: [a.0, b.0],
            dd: [a.1, b.1],
        }
    };
    let terms = terms(g, s, st.col);
    // Incidence medium at the matching frequency: transverse slowness.
    let ns = g.lines(s).len();
    // The media of the incidence-side extension.
    let (n_lo, n_hi, d_lo, d_hi, open) = ranges(st);
    if !open[usize::from(!from_low)] {
        return Err(Error::Invalid(format!(
            "plane wave: the wave enters through the {} end of the layered medium along {s}, \
             which is a closed (PEC/PMC) domain face",
            if from_low { "lower" } else { "upper" }
        )));
    }
    let (kn, kd) = if from_low { (n_lo, d_lo) } else { (n_hi, d_hi) };
    let wm = 2.0 * core::f64::consts::PI * st.match_frequency;
    let (Some(inc_e), Some(inc_h)) = ((st.med_index)(0, kn), (st.med_index)(4, kd)) else {
        return Err(Error::Invalid(
            "plane wave: PEC at the incidence end of the layered medium".into(),
        ));
    };
    let (ye, zh) = (
        st.meds[inc_e].y_at(wm, dt, st.scheme),
        st.meds[inc_h].y_at(wm, dt, st.scheme),
    );
    let (Some(ye), Some(zh)) = (ye, zh) else {
        return Err(Error::Invalid(
            "plane wave: PEC at the incidence end of the layered medium".into(),
        ));
    };
    let om = 2.0 / dt * (0.5 * wm * dt).sin();
    // n = √(ε_r μ_r) of the (discrete) incidence medium.
    let n_inc = (-(ye * zh)).sqrt().re / (om / crate::constants::C0);
    if !(n_inc.is_finite() && n_inc > 0.0) {
        return Err(Error::Invalid(format!(
            "plane wave: no propagating wave in the incidence medium at {} Hz",
            st.match_frequency
        )));
    }
    let v_inc = crate::constants::C0 / n_inc;
    let mut p = st.k_hat.map(|k| k / v_inc);
    p[si] = 0.0;
    // ∫σ ds of the extension PMLs for a round trip e^{−2∫σ ds·|k̂_s|/v} = R.
    let sigma_max = -0.5 * PML_R.ln() * v_inc / ks.abs();
    let (line, reference) = build_lines(st, from_low, sigma_max)?;
    let solver = Solver {
        terms: &terms,
        tr,
        dt,
        scheme: st.scheme,
        meds: &st.meds,
        p,
    };
    // Stored levels.
    let k0 = st.lo[si].saturating_sub(1);
    let k1 = (st.hi[si] + 1).min(ns - 1);
    let nl = k1 - k0 + 1;
    let mut node_of = vec![None; nl];
    let mut dual_of = vec![None; nl];
    for (i, lev) in line.nodes.iter().enumerate() {
        if let Some(k) = lev.engine.filter(|k| (k0..=k1).contains(k)) {
            node_of[k - k0] = Some(i);
        }
    }
    for (i, lev) in line.duals.iter().enumerate() {
        if let Some(k) = lev.engine.filter(|k| (k0..=k1).contains(k)) {
            dual_of[k - k0] = Some(i);
        }
    }
    // The node level of r_ref, where the incident E_t is normalised.
    let k_ref = if from_low { st.lo[si] } else { st.hi[si] };
    let l_ref = reference
        .nodes
        .iter()
        .position(|n| n.engine == Some(k_ref))
        .ok_or_else(|| Error::Invalid("plane wave: reference level outside the column".into()))?;
    let e0t = [st.e0[t1.index()], st.e0[t2.index()]];
    // Time axis.
    let delta_t = dt / UP as f64;
    let length_s = g.lines(s)[ns - 1] - g.lines(s)[0];
    let box_t = Axis::ALL
        .iter()
        .map(|&a| {
            let l = g.lines(a);
            (l[st.hi[a.index()]] - l[st.lo[a.index()]]) * p[a.index()].abs()
        })
        .sum::<f64>();
    let est = st.duration + 4.0 * length_s / v_inc + box_t + 200.0 * dt;
    let mut nfft = ((est / delta_t) as usize).next_power_of_two().max(1024) * 2;
    // Pre-time samples (transverse positions slightly before r_ref).
    // Pre-time samples: edges half a cell before r_ref (transversely, and
    // the levels upstream along s).
    let max_cell = |t: Axis| g.deltas(t).into_iter().fold(0.0f64, f64::max);
    let ahead = Axis::ALL
        .iter()
        .map(|&t| {
            if t == s {
                max_cell(t) / v_inc
            } else {
                p[t.index()].abs() * max_cell(t)
            }
        })
        .sum::<f64>();
    let pre = (2.0 * ahead / delta_t).ceil() as usize + 4;
    loop {
        if nfft > MAX_FFT {
            return Err(Error::Invalid(
                "plane wave: the incident field of the layered medium does not decay within \
                 the longest FFT window (resonant layers, or an excitation that does not end)"
                    .into(),
            ));
        }
        let mut a: Vec<Complex64> = (0..nfft)
            .map(|m| Complex64::new(amplitude(m as f64 * delta_t), 0.0))
            .collect();
        fft(&mut a, false);
        let amax = a
            .iter()
            .take(nfft / 2 + 1)
            .fold(0.0f64, |m, x| m.max(x.norm()));
        let nseries = nl * 6;
        let mut spec: Vec<Vec<Complex64>> = vec![vec![Complex64::ZERO; nfft / 2 + 1]; nseries];
        if amax > 0.0 {
            for k in 1..=nfft / 2 {
                if a[k].norm() <= BIN_THRESHOLD * amax {
                    continue;
                }
                let w = 2.0 * core::f64::consts::PI * k as f64 / (nfft as f64 * delta_t);
                // Beyond the engine's Nyquist frequency there is nothing
                // to inject (aliases of lower frequencies).
                if w * dt >= core::f64::consts::PI {
                    break;
                }
                // Only frequencies at which the incident wave propagates
                // in the incidence medium (beyond, the "plane wave" is
                // evanescent along s, and the layers may trap it).
                let (Some(ye), Some(zh)) = (
                    st.meds[inc_e].y_at(w, dt, st.scheme),
                    st.meds[inc_h].y_at(w, dt, st.scheme),
                ) else {
                    continue;
                };
                let kt2: f64 = (0..2)
                    .map(|i| {
                        let t = [t1, t2][i].index();
                        let x = 2.0 / tr.d[i] * (0.5 * w * p[t] * tr.d[i]).sin();
                        x * x
                    })
                    .sum();
                if (-(ye * zh) - kt2).re <= 0.0 {
                    continue;
                }
                let sol = solver.solve(&line, w)?;
                let rs = solver.solve(&reference, w)?;
                // Incident E_t at the junction per source, and the
                // propagation factor per reference cell.
                let et = |x: &[Complex64; 6]| [x[0] / tr.d[0], x[1] / tr.d[1]];
                let v = [et(&rs[0][l_ref]), et(&rs[1][l_ref])];
                let target = [a[k] * e0t[0], a[k] * e0t[1]];
                let det = v[0][0] * v[1][1] - v[1][0] * v[0][1];
                let scale = (v[0][0].norm() + v[0][1].norm()) * (v[1][0].norm() + v[1][1].norm());
                if det.norm() <= 1e-12 * scale || !det.norm().is_finite() {
                    return Err(Error::Invalid(
                        "plane wave: cannot set the polarisation in the layered medium".into(),
                    ));
                }
                let al = [
                    (target[0] * v[1][1] - target[1] * v[1][0]) / det,
                    (v[0][0] * target[1] - v[0][1] * target[0]) / det,
                ];
                for i in 0..nl {
                    for slot in 0..6 {
                        let li = if slot < 3 { node_of[i] } else { dual_of[i] };
                        let Some(li) = li else { continue };
                        let lev = if slot < 3 {
                            &line.nodes[li]
                        } else {
                            &line.duals[li]
                        };
                        let len = tr.length(slot, lev.len);
                        let x = (sol[0][li][slot] * al[0] + sol[1][li][slot] * al[1]) / len;
                        spec[i * 6 + slot][k] = x;
                    }
                }
            }
        }
        // Inverse FFTs. The window is periodic: samples [0, 3N/4) are
        // t ≥ 0, the last quarter is t < 0.
        let (q2, q3) = (nfft / 2, nfft - nfft / 4);
        let mut series: Vec<Vec<f64>> = Vec::with_capacity(nseries);
        let mut buf = vec![Complex64::ZERO; nfft];
        for sp in &spec {
            for (k, b) in buf.iter_mut().enumerate() {
                *b = if k <= nfft / 2 {
                    sp[k]
                } else {
                    sp[nfft - k].conj()
                };
            }
            fft(&mut buf, true);
            let mut v: Vec<f64> = buf.iter().map(|x| x.re / nfft as f64).collect();
            // The DC bin is not solved (the static limit is singular):
            // it only adds a constant, fixed by causality — the field is
            // zero once the pulse has passed (the third quarter).
            let c = v[q2..q3].iter().sum::<f64>() / (q3 - q2) as f64;
            v.iter_mut().for_each(|x| *x -= c);
            series.push(v);
        }
        let is_e = |i: usize| matches!(i % 6, 0 | 1 | 3);
        let peak = |e: bool| {
            series
                .iter()
                .enumerate()
                .filter(|(i, _)| is_e(*i) == e)
                .flat_map(|(_, v)| v.iter())
                .fold(0.0f64, |m, x| m.max(x.abs()))
        };
        let (pe, ph) = (peak(true), peak(false));
        let worst = |r: core::ops::Range<usize>| {
            series
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let pk = if is_e(i) { pe } else { ph };
                    v[r.clone()].iter().fold(0.0f64, |m, x| m.max(x.abs())) / pk
                })
                .fold(0.0f64, f64::max)
        };
        if worst(q2..q3) > TAIL {
            nfft *= 2;
            continue;
        }
        // Before t = 0 (beyond the few samples kept for the edges just
        // upstream of r_ref) the incident field must be negligible: the
        // run starts at t = 0. Strong dispersion near a cutoff (slow
        // components arriving long before the pulse) violates this.
        let early = worst(q3..nfft - pre);
        if early > ACAUSAL {
            return Err(Error::Invalid(format!(
                "plane wave: the incident field reaches the total-field box {early:.1e} \
                 (relative) before t = 0 (slow, strongly dispersed components); delay the \
                 excitation or narrow its band"
            )));
        }
        // Trim after the last significant sample of any series.
        let mut last = 0;
        for (i, v) in series.iter().enumerate() {
            let pk = if is_e(i) { pe } else { ph };
            if let Some(m) = v[..q3].iter().rposition(|x| x.abs() > 0.1 * TAIL * pk) {
                last = last.max(m);
            }
        }
        let len = (last + 8).min(q3);
        let mut e_last = 0;
        for (i, v) in series.iter().enumerate() {
            if is_e(i)
                && let Some(m) = v[..len].iter().rposition(|x| x.abs() > 1e-6 * pe)
            {
                e_last = e_last.max(m);
            }
        }
        let out: Vec<Vec<f32>> = series
            .iter()
            .map(|v| {
                v[nfft - pre..]
                    .iter()
                    .chain(&v[..len])
                    .map(|&x| x as f32)
                    .collect()
            })
            .collect();
        return Ok(Incident {
            series: out,
            pre,
            k0,
            k1,
            p,
            t_last: e_last as f64 * delta_t,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_round_trip_and_tone() {
        let n = 64;
        let mut x: Vec<Complex64> = (0..n)
            .map(|m| {
                Complex64::new(
                    (2.0 * core::f64::consts::PI * 5.0 * m as f64 / n as f64).cos(),
                    0.0,
                )
            })
            .collect();
        let orig = x.clone();
        fft(&mut x, false);
        assert!((x[5].re - 32.0).abs() < 1e-9 && (x[59].re - 32.0).abs() < 1e-9);
        assert!(x[4].norm() < 1e-9);
        fft(&mut x, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((*a / n as f64 - *b).norm() < 1e-12);
        }
    }

    #[test]
    fn band_lu_solves_a_pivoting_system() {
        // Tridiagonal with a zero diagonal entry (needs pivoting).
        let n = 6;
        let mut b = Band::new(n, 1, 1);
        let mut dense = vec![vec![Complex64::ZERO; n]; n];
        for i in 0..n {
            let d = if i == 2 { 0.0 } else { 2.0 + i as f64 };
            dense[i][i] = Complex64::new(d, 0.5);
            if i + 1 < n {
                dense[i][i + 1] = Complex64::new(1.0, -1.0);
                dense[i + 1][i] = Complex64::new(-1.5, 0.2);
            }
        }
        for (i, row) in dense.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != Complex64::ZERO {
                    *b.at(i, j) = v;
                }
            }
        }
        b.factor().unwrap();
        let x0: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(i as f64, 1.0 - i as f64))
            .collect();
        let mut rhs: Vec<Complex64> = (0..n)
            .map(|i| (0..n).map(|j| dense[i][j] * x0[j]).sum())
            .collect();
        b.solve(&mut rhs);
        for (a, e) in rhs.iter().zip(&x0) {
            assert!((*a - *e).norm() < 1e-12, "{a} vs {e}");
        }
    }
}
