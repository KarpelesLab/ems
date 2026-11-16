//! Row access to the update coefficients for the kernels.
//!
//! On large grids the kernels are limited by memory bandwidth, and the
//! twelve coefficient arrays are 48 of the ~120 bytes moved per cell and
//! step. Many z-rows of a coefficient array are bitwise identical: `ca` and
//! `da` are exactly `1.0` everywhere in lossless media, and `cb`, `db` only
//! depend on the material and the local cell sizes, which repeat across a
//! uniformly meshed region. [`Coefficients`] holds an array with few
//! distinct rows as those rows and an index per row
//! ([storage](Coefficients#storage)); the pool stays in cache, so the
//! kernels then stream only the fields.
//!
//! The values the kernels read are exactly those of [`Coefficients`], so
//! results are unchanged bit for bit.

use super::coeffs::Coefficients;

/// Row access to the coefficients for the kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CoeffRows<'a> {
    co: &'a Coefficients,
    ny: usize,
    oe_curl: bool,
}

impl<'a> CoeffRows<'a> {
    /// The rows of `co`; `oe_curl` selects openEMS's summation order of
    /// the circulations
    /// ([`CurlOrder::OpenEms`](super::CurlOrder::OpenEms)).
    pub(crate) fn new(co: &'a Coefficients, oe_curl: bool) -> Self {
        CoeffRows {
            co,
            ny: co.nodes()[1],
            oe_curl,
        }
    }

    /// Whether the kernels sum the circulations in openEMS's order
    /// ([`CurlOrder::OpenEms`](super::CurlOrder::OpenEms)).
    #[inline(always)]
    pub(crate) fn oe_curl(&self) -> bool {
        self.oe_curl
    }

    /// Row `(gi, j)` (`nz` values, k = 0..nz) of array `a` (ca 0–2,
    /// cb 3–5, da 6–8, db 9–11).
    #[inline(always)]
    pub(crate) fn row(&self, a: usize, gi: usize, j: usize) -> &'a [f32] {
        self.co.row(a, gi * self.ny + j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EdgeMaterials, FaceConditions, Grid};

    #[test]
    fn rows_match_and_pool() {
        let g = Grid::new([
            vec![0.0, 1.0, 2.0, 3.5, 5.0],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![0.0, 0.5, 1.0, 2.0, 3.0],
        ])
        .unwrap();
        let co = Coefficients::from_edges(
            &g,
            &EdgeMaterials::vacuum(&g),
            &FaceConditions::default(),
            1e-12,
        )
        .unwrap();
        // ca, da are 0/1 patterns: pooled.
        assert!(co.pooled() >= 6, "{}", co.pooled());
        let v = CoeffRows::new(&co, false);
        let [nx, ny, nz] = co.nodes();
        // The same values as an array per node, and after writes to it.
        let mut dense = co.clone();
        for axis in crate::Axis::ALL {
            dense.e_mut(axis);
            dense.h_mut(axis);
        }
        assert_eq!(dense.pooled(), 0);
        assert!(dense == co);
        let mut written = co.clone();
        for a in 0..12 {
            for (q, idx) in [3, 17, 18, 40, 41, 77].into_iter().enumerate() {
                let value = 0.25 + a as f32 + q as f32;
                written.set(a, idx, value);
                let (ca, da) = (a / 3 % 2 == 0, a >= 6);
                let axis = crate::Axis::ALL[a % 3];
                let pair = if da {
                    dense.h_mut(axis)
                } else {
                    dense.e_mut(axis)
                };
                (if ca { pair.0 } else { pair.1 })[idx] = value;
            }
        }
        assert!(written == dense && written != co);
        written.compact();
        assert!(written == dense);
        for (co, dense) in [(&co, None), (&written, Some(&dense))] {
            let v = if dense.is_some() {
                CoeffRows::new(co, false)
            } else {
                v
            };
            for a in 0..12 {
                let all = dense.unwrap_or(co).values(a).to_vec();
                assert_eq!(all.len(), nx * ny * nz);
                for gi in 0..nx {
                    for j in 0..ny {
                        let o = (gi * ny + j) * nz;
                        let want = &all[o..o + nz];
                        let got = v.row(a, gi, j);
                        assert!(
                            want.iter()
                                .zip(got)
                                .all(|(x, y)| x.to_bits() == y.to_bits())
                        );
                        for (k, w) in want.iter().enumerate() {
                            assert_eq!(co.values(a)[o + k].to_bits(), w.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn distinct_rows_stay_whole() {
        let g = Grid::new([
            (0..9).map(|i| f64::from(i * i)).collect(),
            (0..9).map(|i| f64::from(i * i * i)).collect(),
            vec![0.0, 0.5, 1.0, 2.0, 3.0],
        ])
        .unwrap();
        let co = Coefficients::from_edges(
            &g,
            &EdgeMaterials::vacuum(&g),
            &FaceConditions::default(),
            1e-13,
        )
        .unwrap();
        // Every row of cb and db has its own cell sizes.
        assert!(co.pooled() >= 6 && co.pooled() < 12, "{}", co.pooled());
    }
}
