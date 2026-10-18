//! Running DFT of the raw E-edge voltages of a box of cells.
//!
//! [`EdgeDft`] accumulates `U_e(f) = Σ_m u_e(m·dt)·e^{-j2πf·m·dt}·K·dt` for
//! every primary (E) edge of a box of primary cells, without any
//! interpolation. It is the input of the power-loss and SAR evaluation in
//! [`post::sar`](crate::post::sar), which needs `|E|²` per edge rather
//! than an interpolated field (the square of an average is not the average
//! of the squares).
//!
//! # Edges of a cell box
//!
//! A [`CellBox`] covers the cells `lo[a] ≤ c[a] < hi[a]` (node indices, so
//! the box spans the nodes `lo..=hi`). Its edges along axis `a` are the
//! edges starting at the nodes `n` with `lo[a] ≤ n[a] < hi[a]` and
//! `lo[b] ≤ n[b] ≤ hi[b]` on the two other axes, i.e. `cells[a]` along
//! `a` and `cells[b] + 1` across. They are stored per component in
//! [`EdgeDft::edge_dims`] order, z fastest.
//!
//! # Time stamps
//!
//! Like the frequency-domain field dumps
//! ([`FreqDump`](super::FreqDump)): the state after `m` completed steps,
//! with `u` at `m·dt`, is sampled when `m` is a multiple of the interval
//! `K`, in the `post_e` hook of the step (and the initial state in the
//! first `pre_h`). Register it after sources, boundaries and lumped
//! elements.
//!
//! Sampling runs on the stepping thread in a fixed order, so the result is
//! bit-identical for any engine thread count.

use core::f64::consts::TAU;

use super::fields::Fields;
use super::grid::Grid;
use super::operator::{Operator, StepContext};
use crate::axis::Axis;
use crate::post::Complex64;
use crate::post::dft::phase_cycles;
use crate::{Error, Result};

/// A box of primary cells: `lo[a] ≤ cell[a] < hi[a]` (node indices).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellBox {
    lo: [usize; 3],
    hi: [usize; 3],
}

impl CellBox {
    /// The cells between the nodes `a` and `b` (corners in any order).
    /// Along an axis where `a` and `b` coincide the box is extended to the
    /// cell above the node (or below it at the last node), so that every
    /// box holds at least one cell per axis.
    pub fn new(a: [usize; 3], b: [usize; 3]) -> Self {
        Self {
            lo: [0, 1, 2].map(|i| a[i].min(b[i])),
            hi: [0, 1, 2].map(|i| a[i].max(b[i])),
        }
    }

    /// Every cell of `grid`.
    pub fn whole(grid: &Grid) -> Self {
        Self::new([0; 3], grid.nodes().map(|n| n - 1))
    }

    /// Clamps degenerate axes to one cell and checks that the box fits a
    /// grid of `nodes` nodes (at least 2 per axis).
    pub fn fitted(mut self, nodes: [usize; 3]) -> Result<Self> {
        for a in 0..3 {
            if nodes[a] < 2 || self.hi[a] >= nodes[a] {
                return Err(Error::Invalid(format!(
                    "cell box {:?}..{:?} does not fit a grid of {nodes:?} nodes",
                    self.lo, self.hi
                )));
            }
            if self.lo[a] == self.hi[a] {
                if self.hi[a] + 1 < nodes[a] {
                    self.hi[a] += 1;
                } else {
                    self.lo[a] -= 1;
                }
            }
        }
        Ok(self)
    }

    /// Lowest node (the lower corner of the first cell).
    pub fn lo(&self) -> [usize; 3] {
        self.lo
    }

    /// Highest node (exclusive cell bound).
    pub fn hi(&self) -> [usize; 3] {
        self.hi
    }

    /// Number of cells per axis.
    pub fn cells(&self) -> [usize; 3] {
        [0, 1, 2].map(|a| self.hi[a] - self.lo[a])
    }

    /// Total number of cells.
    pub fn num_cells(&self) -> usize {
        self.cells().iter().product()
    }

    /// The node lines of the box along `axis` \[m\] (`cells + 1` values).
    pub fn lines<'g>(&self, grid: &'g Grid, axis: Axis) -> &'g [f64] {
        let a = axis.index();
        &grid.lines(axis)[self.lo[a]..=self.hi[a]]
    }
}

/// Running DFT of the E-edge voltages of a [`CellBox`]; see the
/// [module docs](self).
///
/// Memory: 16 bytes per edge per frequency (about 48 bytes per cell and
/// frequency).
#[derive(Debug, Clone)]
pub struct EdgeDft {
    cells: CellBox,
    freqs: Vec<f64>,
    interval: u64,
    acc: [Vec<Complex64>; 3],
    samples: u64,
    dt: f64,
    started: bool,
    phasors: Vec<Complex64>,
}

impl EdgeDft {
    /// DFT of the edges of `cells` at `freqs` \[Hz\], sampling every step.
    pub fn new(cells: CellBox, freqs: impl Into<Vec<f64>>) -> Self {
        let freqs = freqs.into();
        let acc = Axis::ALL.map(|a| {
            vec![Complex64::ZERO; Self::dims_of(&cells, a).iter().product::<usize>() * freqs.len()]
        });
        Self {
            cells,
            phasors: vec![Complex64::ZERO; freqs.len()],
            freqs,
            interval: 1,
            acc,
            samples: 0,
            dt: 0.0,
            started: false,
        }
    }

    /// Samples only the states with `m % k == 0` (`k ≥ 1`; 0 is treated as
    /// 1). Default 1.
    pub fn interval(mut self, k: u64) -> Self {
        self.interval = k.max(1);
        self
    }

    fn dims_of(cells: &CellBox, axis: Axis) -> [usize; 3] {
        let c = cells.cells();
        [0, 1, 2].map(|b| if b == axis.index() { c[b] } else { c[b] + 1 })
    }

    /// The cell box.
    pub fn cell_box(&self) -> &CellBox {
        &self.cells
    }

    /// Number of edges along `axis` per axis: `cells[a]` along `a`,
    /// `cells[b] + 1` across.
    pub fn edge_dims(&self, axis: Axis) -> [usize; 3] {
        Self::dims_of(&self.cells, axis)
    }

    /// The analysis frequencies \[Hz\].
    pub fn freqs(&self) -> &[f64] {
        &self.freqs
    }

    /// Number of states accumulated so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// The engine time step \[s\] (0 before the first step).
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Sampling interval `K·dt` \[s\] (0 before the first step).
    pub fn sample_interval(&self) -> f64 {
        self.interval as f64 * self.dt
    }

    /// The transform of the edge voltages along `axis` at frequency index
    /// `k` \[V·s\], one per edge in [`edge_dims`](Self::edge_dims) order (z
    /// fastest).
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn voltages(&self, axis: Axis, k: usize) -> Vec<Complex64> {
        assert!(k < self.freqs.len(), "frequency index {k} out of range");
        let n: usize = self.edge_dims(axis).iter().product();
        let s = self.sample_interval();
        self.acc[axis.index()][k * n..(k + 1) * n]
            .iter()
            .map(|&c| c * s)
            .collect()
    }

    /// Clears the accumulators.
    pub fn reset(&mut self) {
        for a in &mut self.acc {
            a.fill(Complex64::ZERO);
        }
        self.samples = 0;
    }

    fn sample(&mut self, fields: &Fields<'_>, m: u64, dt: f64) {
        self.dt = dt;
        if !m.is_multiple_of(self.interval) {
            return;
        }
        for (p, &f) in self.phasors.iter_mut().zip(&self.freqs) {
            *p = Complex64::cis(-TAU * phase_cycles(f, 0.0, dt, m as f64));
        }
        let lo = self.cells.lo;
        for axis in Axis::ALL {
            let d = Self::dims_of(&self.cells, axis);
            let n: usize = d.iter().product();
            let acc = &mut self.acc[axis.index()];
            let mut idx = 0;
            for i in 0..d[0] {
                for j in 0..d[1] {
                    for k in 0..d[2] {
                        let u = f64::from(fields.u.get(axis, [lo[0] + i, lo[1] + j, lo[2] + k]));
                        if u != 0.0 {
                            for (f, &p) in self.phasors.iter().enumerate() {
                                acc[f * n + idx] += p * u;
                            }
                        }
                        idx += 1;
                    }
                }
            }
        }
        self.samples += 1;
    }
}

impl Operator for EdgeDft {
    fn pre_h(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        if !core::mem::replace(&mut self.started, true) {
            self.sample(fields, ctx.step, ctx.dt);
        }
    }

    fn post_e(&mut self, fields: &mut Fields<'_>, ctx: &StepContext<'_>) {
        self.started = true;
        self.sample(fields, ctx.step + 1, ctx.dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    #[test]
    fn cell_box_fitting() {
        let b = CellBox::new([3, 1, 4], [1, 1, 4])
            .fitted([5, 5, 5])
            .unwrap();
        assert_eq!(b.lo(), [1, 1, 3]);
        assert_eq!(b.hi(), [3, 2, 4]);
        assert_eq!(b.cells(), [2, 1, 1]);
        assert!(CellBox::new([0; 3], [5; 3]).fitted([5, 5, 5]).is_err());
    }

    #[test]
    fn static_field_transform() {
        // A constant edge voltage transforms to u·N·dt at f = 0.
        let grid = Grid::uniform([4, 4, 4], [1e-3; 3]).unwrap();
        let mut e = Engine::builder(grid).threads(1).build().unwrap();
        let h = e.add_operator(EdgeDft::new(CellBox::new([1, 1, 1], [2, 3, 3]), [0.0]));
        let dft = e.operator(&h);
        assert_eq!(dft.edge_dims(Axis::X), [1, 3, 3]);
        assert_eq!(dft.edge_dims(Axis::Y), [2, 2, 3]);
        e.run(5);
        let dft = e.operator(&h);
        assert_eq!(dft.samples(), 6);
        assert!(
            dft.voltages(Axis::Z, 0)
                .iter()
                .all(|v| *v == Complex64::ZERO)
        );
    }
}
