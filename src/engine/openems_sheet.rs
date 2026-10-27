//! openEMS's update of the conducting-sheet branches (files only,
//! Cartesian grids; design §5.16.13).
//!
//! Each emulated sheet edge (model: [`crate::compile`]'s openEMS sheet
//! emulation) carries two series R–L branches, integrated with the
//! explicit leapfrog scheme: the branch current lives at half steps and
//! is driven by the edge voltage before the update,
//!
//! ```text
//! j_b^{n+½} = k_a·j_b^{n−½} + k_b·u^n,    k_a = (L/dt − R/2)/(L/dt + R/2),  k_b = 1/(L/dt + R/2)
//! u^{n+1}   = u* − cb·j_1^{n+½} − cb·j_2^{n+½}
//! ```
//!
//! (`u*` the kernel result, `cb` the edge's coefficient). This is what the
//! general dispersive operator ([`crate::engine::dispersive`],
//! `Scheme::Leapfrog`) computes in `f64`; openEMS keeps the branch state in
//! single precision, in volts. Measured bit for bit on per-step dumps
//! (black box): the state is `x_b = cb·j_b` with
//!
//! ```text
//! x_b ← f32(f32(k_a·x_b) + f32(f32(k_b·cb)·u^n))     (k_a = f32(k_a))
//! u   ← f32(f32(u* − x_1) − x_2)
//! ```
//!
//! every operation rounded to `f32` (and flushed to zero below the
//! smallest normal number, like the kernels).

use super::fields::Fields;
use super::kernel::flush;
use super::operator::{Hooks, Operator, StepContext};
use crate::axis::Axis;

/// One sheet edge with its two branches.
#[derive(Debug, Clone)]
struct Edge {
    axis: Axis,
    node: [usize; 3],
    /// `f32(k_a)` per branch.
    ka: [f32; 2],
    /// `f32(k_b·cb)` per branch.
    kb: [f32; 2],
    /// Branch state `cb·j_b` \[V\].
    x: [f32; 2],
}

/// The branch update of openEMS's conducting sheets (see the
/// [module docs](self)).
#[derive(Debug, Clone, Default)]
pub(crate) struct OpenEmsSheetBranches {
    edges: Vec<Edge>,
}

impl OpenEmsSheetBranches {
    /// Adds an edge with branches `(L, R)` \[H, Ω\], time step `dt` and
    /// the edge's E coefficient `cb`. An edge with `cb = 0` (PEC) is
    /// skipped.
    pub(crate) fn push(
        &mut self,
        axis: Axis,
        node: [usize; 3],
        branches: [(f64, f64); 2],
        dt: f64,
        cb: f32,
    ) {
        if cb == 0.0 {
            return;
        }
        let mut ka = [0.0f32; 2];
        let mut kb = [0.0f32; 2];
        for (b, &(l, r)) in branches.iter().enumerate() {
            let d = l / dt + 0.5 * r;
            ka[b] = ((l / dt - 0.5 * r) / d) as f32;
            kb[b] = (cb as f64 / d) as f32;
        }
        self.edges.push(Edge {
            axis,
            node,
            ka,
            kb,
            x: [0.0; 2],
        });
    }

    /// Whether there is no edge.
    pub(crate) fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// One branch step: `x ← f32(f32(k_a·x) + f32(k_b·u))`.
#[inline]
fn branch(ka: f32, kb: f32, x: f32, u: f32) -> f32 {
    flush(flush(ka * x) + flush(kb * u))
}

/// The correction of the updated voltage: `u ← f32(f32(u − x_1) − x_2)`.
#[inline]
fn apply(u: f32, x: [f32; 2]) -> f32 {
    flush(flush(u - x[0]) - x[1])
}

impl Operator for OpenEmsSheetBranches {
    fn hooks(&self) -> Hooks {
        if self.edges.is_empty() {
            Hooks::NONE
        } else {
            Hooks::PRE_E | Hooks::POST_E
        }
    }

    /// `pre_e` reads `u^n` only.
    fn movable_hooks(&self) -> Hooks {
        Hooks::PRE_E
    }

    fn pre_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for e in &mut self.edges {
            let u = f.u.get(e.axis, e.node);
            for b in 0..2 {
                e.x[b] = branch(e.ka[b], e.kb[b], e.x[b], u);
            }
        }
    }

    fn post_e(&mut self, f: &mut Fields<'_>, _ctx: &StepContext<'_>) {
        for e in &self.edges {
            let u = f.u.get_mut(e.axis, e.node);
            *u = apply(*u, e.x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// openEMS's sheet edge, black box (1 m cells, bin 0, σ = 5.92e5 S/m,
    /// t = 48.6 µm, `dt` = 1.756 ns, z edge (6, 5, 5) of a sheet at
    /// x = 6, from per-step `DumpMode="0"` dumps): per step `u^n`, the
    /// kernel result `u*` (from the dumped currents, openEMS's curl order
    /// and coefficients) and openEMS's `u^{n+1}`, as `f32` bits. With the
    /// two branches' `f32` coefficients (`k_a` of the formula, `k_b·cb`
    /// within 3 ulps of it: openEMS's sheet constants are not known to
    /// the last bit, design §5.16.13) the update reproduces every step.
    #[test]
    fn reproduces_openems_steps() {
        #[rustfmt::skip]
        const STEPS: [[u32; 3]; 80] = [
        [0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0x00000000, 0x00000000],
        [0x00000000, 0xad11a3e2, 0xad11a3e2],
        [0xad11a3e2, 0xae2489ac, 0xae235756],
        [0xae235756, 0xaeb48017, 0xaeb23ff6],
        [0xaeb23ff6, 0xaf046d0b, 0xaf024f57],
        [0xaf024f57, 0xaf1c6858, 0xaf19a6cf],
        [0xaf19a6cf, 0xaf2cceb2, 0xaf29be81],
        [0xaf29be81, 0xaf416a20, 0xaf3e0b25],
        [0xaf3e0b25, 0xaf58fbba, 0xaf552acb],
        [0xaf552acb, 0xaf6d106e, 0xaf68cf23],
        [0xaf68cf23, 0xaf7f3958, 0xaf7aa3aa],
        [0xaf7aa3aa, 0xaf8abebd, 0xaf884747],
        [0xaf884747, 0xaf966fd1, 0xaf93bc55],
        [0xaf93bc55, 0xaf9e2807, 0xaf9b3f6f],
        [0xaf9b3f6f, 0xafa1839b, 0xaf9e8202],
        [0xaf9e8202, 0xafa41259, 0xafa10748],
        [0xafa10748, 0xafa7977b, 0xafa47e05],
        [0xafa47e05, 0xafaaf628, 0xafa7c9f5],
        [0xafa7c9f5, 0xafad5efc, 0xafaa2489],
        [0xafaa2489, 0xafaf42f9, 0xafabfefa],
        [0xafabfefa, 0xafb09b18, 0xafad4e60],
        [0xafad4e60, 0xafb0577b, 0xafad0608],
        [0xafad0608, 0xafad7280, 0xafaa26cf],
        [0xafaa26cf, 0xafa7944c, 0xafa45c99],
        [0xafa45c99, 0xaf9f487d, 0xaf9c330f],
        [0xaf9c330f, 0xaf95a60e, 0xaf92bc96],
        [0xaf92bc96, 0xaf8a7521, 0xaf87bb95],
        [0xaf87bb95, 0xaf771bbb, 0xaf721cbe],
        [0xaf721cbe, 0xaf4ec643, 0xaf4a6a7e],
        [0xaf4a6a7e, 0xaf1c13f0, 0xaf18906f],
        [0xaf18906f, 0xaebb57e8, 0xaeb65fe0],
        [0xaeb65fe0, 0xad7a01e0, 0xad66a8e5],
        [0xad66a8e5, 0x2ea14d81, 0x2ea08398],
        [0x2ea08398, 0x2f43a244, 0x2f4148c0],
        [0x2f4148c0, 0x2fa467f7, 0x2fa2142a],
        [0x2fa2142a, 0x2ff02226, 0x2fec7c7a],
        [0x2fec7c7a, 0x30235781, 0x3020c56c],
        [0x3020c56c, 0x3054e394, 0x30517589],
        [0x30517589, 0x30862748, 0x3083f2b8],
        [0x3083f2b8, 0x30a48cce, 0x30a1cdb9],
        [0x30a1cdb9, 0x30c5d0ab, 0x30c27a3b],
        [0x30c27a3b, 0x30e9e2d8, 0x30e5e697],
        [0x30e5e697, 0x31084719, 0x3105efb5],
        [0x3105efb5, 0x311d3661, 0x311a7f94],
        [0x311a7f94, 0x31346e49, 0x31314f3f],
        [0x31314f3f, 0x314e97c9, 0x314b03f4],
        [0x314b03f4, 0x316c075e, 0x3167ef71],
        [0x3167ef71, 0x31865a53, 0x3184044f],
        [0x3184044f, 0x31983a5d, 0x319592b1],
        [0x319592b1, 0x31ab80a9, 0x31a88021],
        [0x31a88021, 0x31bff9ff, 0x31bc9a42],
        [0x31bc9a42, 0x31d5a1e9, 0x31d1dda8],
        [0x31d1dda8, 0x31ecc427, 0x31e8956f],
        [0x31e8956f, 0x3202dcd2, 0x32008c42],
        [0x32008c42, 0x321055af, 0x320dc73b],
        [0x320dc73b, 0x321ec6c8, 0x321bf59d],
        [0x321bf59d, 0x322e20a9, 0x322b0842],
        [0x322b0842, 0x323e61ac, 0x323afdcb],
        [0x323afdcb, 0x324f8918, 0x324bd556],
        [0x324bd556, 0x32618189, 0x325d79a5],
        [0x325d79a5, 0x327448bb, 0x326fe909],
        [0x326fe909, 0x3284084e, 0x3281aa8b],
        [0x3281aa8b, 0x328e8454, 0x328bf5f3],
        [0x328bf5f3, 0x3299923e, 0x3296d048],
        [0x3296d048, 0x32a5186c, 0x32a22053],
        [0x32a22053, 0x32b102db, 0x32add297],
        [0x32add297, 0x32bd4cd2, 0x32b9e2a0],
        [0x32b9e2a0, 0x32c9f950, 0x32c6535f],
        [0x32c6535f, 0x32d705fd, 0x32d3226e],
        [0x32d3226e, 0x32e46805, 0x32e04518],
        [0x32e04518, 0x32f21595, 0x32edb1ce],
        [0x32edb1ce, 0x33000478, 0x32fb62f1],
        [0x32fb62f1, 0x330717c3, 0x3304a306],
        [0x3304a306, 0x330e3145, 0x330b9a6d],
        [0x330b9a6d, 0x331543f3, 0x33128b11],
        [0x33128b11, 0x331c561e, 0x33197b5e],
        ];
        let ka = [f32::from_bits(0xbf30d381), f32::from_bits(0xbf76a27c)];
        let kb = [f32::from_bits(0x3cebaad5), f32::from_bits(0x3b8641bb)];
        let mut x = [0.0f32; 2];
        for (n, s) in STEPS.iter().enumerate() {
            let [u0, us, u1] = s.map(f32::from_bits);
            for b in 0..2 {
                x[b] = branch(ka[b], kb[b], x[b], u0);
            }
            assert_eq!(apply(us, x).to_bits(), u1.to_bits(), "step {n}");
        }
    }

    #[test]
    fn coefficients() {
        let mut b = OpenEmsSheetBranches::default();
        b.push(Axis::Z, [1, 1, 1], [(1e-11, 0.04), (2e-11, 0.3)], 1e-9, 0.5);
        b.push(Axis::Z, [1, 1, 2], [(1e-11, 0.04), (2e-11, 0.3)], 1e-9, 0.0);
        assert_eq!(b.edges.len(), 1);
        let d = 1e-11 / 1e-9 + 0.02;
        assert_eq!(b.edges[0].ka[0], ((1e-11 / 1e-9 - 0.02) / d) as f32);
        assert_eq!(b.edges[0].kb[0], (0.5 / d) as f32);
    }
}
