//! Cell-constant materials (openEMS `CellConstantMaterial`).
//!
//! With [`Fdtd::cell_constant_material`](crate::Fdtd::cell_constant_material),
//! every grid cell takes the material found at its centre (the
//! highest-priority material property there, with its weighting
//! functions evaluated at the centre; vacuum if none), instead of csx's
//! sub-cell sampling of the quarter cells around each edge. The edges then
//! average the cells around them as usual ([`EdgeMaterials::from_cells`]:
//! area-weighted ε and κ over the four cells of a dual face, harmonic μ
//! over the two half cells of a dual edge).
//!
//! This reproduces openEMS: with `CellConstantMaterial="1"`, a dielectric
//! box whose faces lie inside cells behaves exactly (to ~1e-6) like the
//! box grown or shrunk to the faces of the cells whose centres it contains
//! (black-box comparison, `tests/openems_core.rs`). Metal is unaffected
//! (csx's PEC mask is kept).
//!
//! On cylindrical grids the centre is the parameter-space midpoint
//! `(ρ_{i+½}, α_{j+½}, z_{k+½})` (not the cell's centroid), the
//! weighting functions are evaluated at its Cartesian position, and
//! [`EdgeMaterials::from_cells`] weighs the cells with their annular
//! areas. openEMS does the same: with `CellConstantMaterial="1"` a
//! dielectric box with faces inside cells gives probes bit-identical to
//! the box snapped to the faces of the cells whose midpoints it contains,
//! with a radial face placed between a cell's midpoint and its centroid
//! (`cyl_cellconst`, `tests/openems_cyl.rs`).

use csx::properties::{MaterialQuantity, TypeMask};
use csx::{CoordSystem, Vec3};

use crate::Result;
use crate::engine::{CellMaterials, EdgeMaterials, Grid, Material};

/// Replaces the ε, κ, μ, σ arrays of `materials` by the cell-centre
/// values of `model` (see the module docs).
pub(crate) fn apply(
    model: &csx::ContinuousStructure,
    grid: &Grid,
    materials: &mut EdgeMaterials,
) -> Result<()> {
    let lines: [&[f64]; 3] = core::array::from_fn(|a| model.grid.lines(a));
    let cells = grid.cells();
    let cyl = grid.is_cylindrical();
    let cs = if cyl {
        CoordSystem::Cylindrical
    } else {
        CoordSystem::Cartesian
    };
    let mut cm = CellMaterials::vacuum(grid);
    for i in 0..cells[0] {
        for j in 0..cells[1] {
            for k in 0..cells[2] {
                let c = [i, j, k];
                let p = Vec3(core::array::from_fn(|a| {
                    0.5 * (lines[a][c[a]] + lines[a][c[a] + 1])
                }));
                let Some(hit) = model.property_at(p, cs, TypeMask::MATERIALS) else {
                    continue;
                };
                let Some(m) = hit.property.kind.material() else {
                    continue;
                };
                // Weighting functions take Cartesian points.
                let p = if cyl {
                    let [r, a, z] = p.0;
                    Vec3::new(r * a.cos(), r * a.sin(), z)
                } else {
                    p
                };
                cm.set(
                    c,
                    Material {
                        eps_r: m.values_at(MaterialQuantity::Epsilon, p),
                        mu_r: m.values_at(MaterialQuantity::Mue, p),
                        kappa: m.values_at(MaterialQuantity::Kappa, p),
                        sigma_m: m.values_at(MaterialQuantity::Sigma, p),
                    },
                );
            }
        }
    }
    let em = EdgeMaterials::from_cells(grid, &cm)?;
    for axis in crate::Axis::ALL {
        materials.eps_r_mut(axis).copy_from_slice(em.eps_r(axis));
        materials.kappa_mut(axis).copy_from_slice(em.kappa(axis));
        materials.mu_r_mut(axis).copy_from_slice(em.mu_r(axis));
        materials
            .sigma_m_mut(axis)
            .copy_from_slice(em.sigma_m(axis));
    }
    Ok(())
}
