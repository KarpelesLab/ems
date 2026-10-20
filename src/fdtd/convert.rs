//! Conversion between [`Fdtd`] and csx's file-level
//! [`FdtdSettings`](csx::fdtd::FdtdSettings) (the openEMS `<FDTD>` element).

use super::{Boundaries, Boundary, Cpml, EndCriteria, Face, Fdtd, Mur};
use crate::error::{Error, Result};
use crate::waveform::Waveform;
use csx::fdtd::{
    Boundary as CsxBoundary, BoundaryConditions, ExcitationSignal, FdtdSettings, TimeStepMethod,
};

impl Fdtd {
    /// Number of timesteps used for a document without `<FDTD>` element
    /// (`Fdtd::from_document`, feature `xml`). (A `<FDTD>` element without
    /// `NumberOfTimesteps` runs no step, like openEMS.)
    pub const OPENEMS_DEFAULT_TIMESTEPS: u64 = 1_000_000_000;
    /// End criterion used when an openEMS file does not set `endCriteria`
    /// (1e-6 = −60 dB, what openEMS v0.37 uses: its console reports
    /// "end-criteria of -60dB").
    pub const OPENEMS_DEFAULT_END_CRITERIA: f64 = 1e-6;
    /// `endCriteria` written by [`Fdtd::to_csx`] for settings without end
    /// criteria: openEMS has no "off" value (0 means its default), so a
    /// ratio that is never reached (-3000 dB) is written.
    pub const NO_END_CRITERIA: f64 = 1e-300;

    /// Converts csx / openEMS `<FDTD>` settings into solver settings.
    ///
    /// Returns the settings and a list of human-readable warnings about
    /// settings that are ignored or only approximated. Values that cannot
    /// be run at all (a cylindrical coordinate system, an invalid waveform,
    /// out-of-range numbers) are errors. The result is validated.
    ///
    /// Mapping:
    ///
    /// | csx field (XML attribute) | `Fdtd` |
    /// |---|---|
    /// | `number_of_timesteps` (`NumberOfTimesteps`) | [`max_timesteps`](Fdtd::max_timesteps); absent → 0 with a warning (openEMS runs nothing) |
    /// | `end_criteria` (`endCriteria`) | [`EndCriteria::energy_ratio`]: a linear energy ratio (1e-6 = −60 dB); absent or `0` → [`OPENEMS_DEFAULT_END_CRITERIA`](Fdtd::OPENEMS_DEFAULT_END_CRITERIA) (as openEMS); checked like openEMS ([`openems_run`](Fdtd::openems_run) is set) |
    /// | `f_max` (`f_max`) | [`max_frequency`](Fdtd::max_frequency); else the `f_max` of a Dirac / step signal (if > 0). Written back as [`Fdtd::get_max_frequency`] |
    /// | `over_sampling` (`OverSampling`) | [`oversampling`](Fdtd::oversampling) (dump sampling interval) |
    /// | `time_step_method` (`TimeStepMethod`) | [`timestep_method`](Fdtd::timestep_method): absent or 3 → [`TimestepMethod::Rennings`](crate::engine::TimestepMethod::Rennings), 1 → [`Cfl`](crate::engine::TimestepMethod::Cfl), other ids → `Cfl` with a warning (what openEMS does) |
    /// | `max_time` (`MaxTime`) | [`max_time`](Fdtd::max_time) |
    /// | `time_step` (`TimeStep`) | [`timestep`](Fdtd::timestep) |
    /// | `time_step_factor` (`TimeStepFactor`) | [`timestep_factor`](Fdtd::timestep_factor); absent → 1 (openEMS default) |
    /// | `cell_constant_material` (`CellConstantMaterial`) | [`cell_constant_material`](Fdtd::cell_constant_material) |
    /// | `coord_system` | `0` (Cartesian) or `1` (cylindrical) accepted: the coordinate system is that of the model's grid; anything else → [`Error::Unsupported`] |
    /// | extra attribute `CylinderCoords` | `0` / `1` accepted silently (openEMS's switch for its cylindrical solver; ems follows the grid) |
    /// | `multi_grid` (`MultiGrid`) | [`multigrid`](Fdtd::multigrid): comma-separated radii in drawing units; unparsable → [`Error::Invalid`] |
    /// | `excitation` (`<Excitation>`) | [`waveform`](Fdtd::waveform) via [`Waveform::from_csx`] |
    /// | `boundary.faces` (`<BoundaryCond>`) | PEC / PMC / Mur / `PML_n` → [`Boundary`] with [`Cpml::cells`]`(n)` and default grading; absent → all PEC |
    /// | `boundary.mur_phase_velocity` | [`Mur::phase_velocity`] of every Mur face |
    /// | `boundary.pml_grading` | warning: ignored (openEMS v0.37 ignores the attribute as well), the default polynomial grading is used |
    /// | unknown attributes | warning each |
    pub fn from_csx(s: &FdtdSettings) -> Result<(Fdtd, Vec<String>)> {
        let mut warn = Vec::new();
        let mut f = Fdtd::new();

        if let Some(cs) = s.coord_system
            && cs != 0
            && cs != 1
        {
            return Err(Error::Unsupported(format!(
                "FDTD CoordSystem={cs}: only Cartesian (0) and cylindrical (1) are supported"
            )));
        }
        if let Some(m) = &s.multi_grid {
            f.multigrid = m
                .split(',')
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(|v| {
                    v.parse::<f64>().map_err(|_| {
                        Error::Invalid(format!("FDTD MultiGrid={m:?}: {v:?} is not a number"))
                    })
                })
                .collect::<Result<_>>()?;
        }

        f.openems_run = true;
        f.max_timesteps = match s.number_of_timesteps {
            Some(n) => n,
            None => {
                warn.push("FDTD NumberOfTimesteps is missing: openEMS runs no timestep (0)".into());
                0
            }
        };

        // openEMS replaces a missing or non-positive value by its default
        // (`endCriteria="0"` stops at -60 dB too).
        let e = s
            .end_criteria
            .filter(|&e| e != 0.0)
            .unwrap_or(Self::OPENEMS_DEFAULT_END_CRITERIA);
        f.end_criteria = if e == Self::NO_END_CRITERIA {
            None
        } else if e > 0.0 && e <= 1.0 {
            Some(EndCriteria::energy_ratio(e))
        } else {
            return Err(Error::Invalid(format!(
                "FDTD endCriteria {e} must be a linear energy ratio in [0, 1]"
            )));
        };

        let signal_fmax = match &s.excitation {
            Some(ExcitationSignal::Dirac { f_max }) | Some(ExcitationSignal::Step { f_max }) => {
                Some(*f_max).filter(|&v| v > 0.0)
            }
            _ => None,
        };
        f.max_frequency = s.f_max.or(signal_fmax);
        f.max_time = s.max_time;
        // Used by field dumps to choose their sampling interval; probes still
        // record every timestep.
        f.oversampling = s.over_sampling;

        // openEMS: Rennings (3) by default; every other id selects its
        // CFL criterion (1), as observed with openEMS v0.37.
        f.timestep_method = Some(match s.time_step_method {
            None | Some(TimeStepMethod::Rennings) => crate::engine::TimestepMethod::Rennings,
            Some(TimeStepMethod::Cfl) => crate::engine::TimestepMethod::Cfl,
            Some(m) => {
                warn.push(format!(
                    "FDTD TimeStepMethod={} is not a known method; using the CFL criterion \
                     (method 1) like openEMS",
                    m.id()
                ));
                crate::engine::TimestepMethod::Cfl
            }
        });
        f.timestep = s.time_step;
        // openEMS's default factor is 1.
        f.timestep_factor = s.time_step_factor.unwrap_or(1.0);
        f.cell_constant_material = s.cell_constant_material == Some(true);

        if let Some(sig) = &s.excitation {
            f.waveform = Some(Waveform::from_csx(sig)?);
        }
        for (k, v) in &s.excitation_extra_attributes {
            warn.push(format!("Excitation attribute {k}={v:?} ignored"));
        }

        if let Some(b) = &s.boundary {
            f.boundaries = boundaries_from_csx(b, &mut warn);
        }
        for (k, v) in &s.extra_attributes {
            // openEMS selects its cylindrical solver with `CylinderCoords`
            // (black-box: `CoordSystem` on `<FDTD>` is ignored by openEMS);
            // ems takes the coordinate system from the model's grid, so
            // both are only checked.
            if k == "CylinderCoords" && matches!(v.trim(), "0" | "1") {
                continue;
            }
            warn.push(format!("FDTD attribute {k}={v:?} ignored"));
        }

        f.validate()?;
        Ok((f, warn))
    }

    /// Converts the solver settings into csx / openEMS `<FDTD>` settings.
    ///
    /// Every setting the file format can express is written explicitly
    /// (including [`NO_END_CRITERIA`](Self::NO_END_CRITERIA) when there are no end criteria), so that
    /// [`Fdtd::from_csx`] of the result gives the same settings back.
    ///
    /// Lossy: settings without a [`timestep_method`](Fdtd::timestep_method)
    /// (the ems Courant time step) are written without `TimeStepMethod`,
    /// which openEMS (and [`Fdtd::from_csx`]) read as Rennings; the CPML
    /// grading parameters other than the cell count, the
    /// end-criteria check interval, the thread count and
    /// [`Waveform::Custom`] closures (written without `<Excitation>`) have
    /// no file representation. The file format has a single Mur phase
    /// velocity; the first Mur face's is written. It has no periodic
    /// boundary either: [`Boundary::Periodic`] faces are written as PEC
    /// (use [`to_csx_checked`](Self::to_csx_checked) to get an error
    /// instead), and the Bloch settings are not written.
    pub fn to_csx(&self) -> FdtdSettings {
        let excitation = self
            .waveform
            .as_ref()
            .and_then(Waveform::to_csx)
            .map(|sig| {
                let fmax = self.get_max_frequency().unwrap_or(0.0);
                match sig {
                    ExcitationSignal::Dirac { .. } => ExcitationSignal::Dirac { f_max: fmax },
                    ExcitationSignal::Step { .. } => ExcitationSignal::Step { f_max: fmax },
                    other => other,
                }
            });
        let mut faces = [CsxBoundary::Pec; 6];
        let mut mur_phase_velocity = None;
        for (face, b) in self.boundaries.iter() {
            faces[face.index()] = match b {
                Boundary::Pec => CsxBoundary::Pec,
                Boundary::Pmc => CsxBoundary::Pmc,
                Boundary::Mur(m) => {
                    if mur_phase_velocity.is_none() {
                        mur_phase_velocity = m.phase_velocity;
                    }
                    CsxBoundary::Mur
                }
                Boundary::Pml(p) => CsxBoundary::Pml(u32::try_from(p.cells).unwrap_or(u32::MAX)),
                Boundary::Periodic => CsxBoundary::Pec,
            };
        }
        FdtdSettings {
            number_of_timesteps: Some(self.max_timesteps),
            end_criteria: Some(
                self.end_criteria
                    .map_or(Self::NO_END_CRITERIA, |e| e.energy_ratio),
            ),
            f_max: self.get_max_frequency(),
            over_sampling: self.oversampling,
            time_step_method: self.timestep_method.map(|m| match m {
                crate::engine::TimestepMethod::Cfl => TimeStepMethod::Cfl,
                crate::engine::TimestepMethod::Rennings => TimeStepMethod::Rennings,
            }),
            max_time: self.max_time,
            cell_constant_material: self.cell_constant_material.then_some(true),
            time_step: self.timestep,
            time_step_factor: Some(self.timestep_factor),
            multi_grid: (!self.multigrid.is_empty()).then(|| {
                self.multigrid
                    .iter()
                    .map(f64::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            }),
            excitation,
            boundary: Some(BoundaryConditions {
                faces,
                mur_phase_velocity,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// [`to_csx`](Self::to_csx), failing instead of writing a boundary the
    /// file format cannot express.
    ///
    /// # Errors
    /// [`Error::Unsupported`] if a face is [`Boundary::Periodic`] or Bloch
    /// settings are present.
    pub fn to_csx_checked(&self) -> Result<FdtdSettings> {
        if let Some((face, _)) = self
            .boundaries
            .iter()
            .find(|(_, b)| **b == Boundary::Periodic)
        {
            return Err(Error::Unsupported(format!(
                "periodic boundary on {face}: the openEMS <FDTD> format has no periodic boundary"
            )));
        }
        if self.get_bloch_wavevector().is_some() || self.get_bloch_frequency().is_some() {
            return Err(Error::Unsupported(
                "Bloch settings: the openEMS <FDTD> format has no Bloch boundary".into(),
            ));
        }
        Ok(self.to_csx())
    }
}

fn boundaries_from_csx(b: &BoundaryConditions, warn: &mut Vec<String>) -> Boundaries {
    let mut out = Boundaries::default();
    let mut has_mur = false;
    let mut has_pml = false;
    for (face, cb) in Face::ALL.into_iter().zip(&b.faces) {
        out[face] = match *cb {
            CsxBoundary::Pec => Boundary::Pec,
            CsxBoundary::Pmc => Boundary::Pmc,
            CsxBoundary::Mur => {
                has_mur = true;
                Boundary::Mur(Mur {
                    phase_velocity: b.mur_phase_velocity,
                })
            }
            CsxBoundary::Pml(n) => {
                has_pml = true;
                Boundary::Pml(Cpml::cells(n as usize))
            }
            ref other => {
                warn.push(format!(
                    "boundary condition {other:?} on {face:?} not supported; using PEC"
                ));
                Boundary::Pec
            }
        };
    }
    if !has_mur && b.mur_phase_velocity.is_some() {
        warn.push("BoundaryCond MUR_PhaseVelocity ignored: no Mur face".into());
    }
    if let Some(g) = &b.pml_grading {
        if has_pml {
            // openEMS v0.37 ignores the attribute too: its PML operator is
            // the same with any expression, even an invalid one (black
            // box, design §5.16.1).
            warn.push(format!(
                "BoundaryCond PML_Grading={g:?} ignored (as by openEMS); using the default \
                 polynomial grading"
            ));
        } else {
            warn.push("BoundaryCond PML_Grading ignored: no PML face".into());
        }
    }
    for (k, v) in &b.extra_attributes {
        warn.push(format!("BoundaryCond attribute {k}={v:?} ignored"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_same(a: &Fdtd, b: &Fdtd) {
        assert_eq!(a.boundaries, b.boundaries);
        assert_eq!(a.end_criteria, b.end_criteria);
        assert_eq!(a.max_timesteps, b.max_timesteps);
        assert_eq!(a.timestep_factor, b.timestep_factor);
        assert_eq!(a.timestep, b.timestep);
        assert_eq!(a.get_max_frequency(), b.get_max_frequency());
        assert_eq!(a.max_time, b.max_time);
        assert_eq!(a.oversampling, b.oversampling);
        assert_eq!(
            a.waveform.as_ref().and_then(Waveform::to_csx),
            b.waveform.as_ref().and_then(Waveform::to_csx)
        );
    }

    #[test]
    fn csx_defaults() {
        let s = FdtdSettings::gaussian(2e9, 1e9);
        let (f, warn) = Fdtd::from_csx(&s).unwrap();
        assert!(warn.is_empty(), "{warn:?}");
        assert_eq!(f.get_max_timesteps(), 1_000_000_000);
        assert!(f.get_openems_run());
        // csx's helper sets endCriteria = 1e-5.
        let e = f.get_end_criteria().unwrap();
        assert_eq!(e.energy_ratio, 1e-5);
        assert!((e.decay_db() + 50.0).abs() < 1e-12);
        assert!(matches!(
            f.get_waveform(),
            Some(Waveform::OpenEmsGaussian { f0: 2e9, fc: 1e9 })
        ));
        assert_eq!(f.get_max_frequency(), Some(3e9));
        assert_eq!(*f.get_boundaries(), Boundaries::all(Boundary::pml(8)));
        // openEMS's time step: Rennings, factor 1.
        assert_eq!(f.get_timestep_factor(), 1.0);
        assert_eq!(
            f.get_timestep_method(),
            Some(crate::engine::TimestepMethod::Rennings)
        );
        assert_eq!(f.get_timestep(), None);

        // An empty element: openEMS defaults, PEC walls; no
        // NumberOfTimesteps runs no step.
        let (f, warn) = Fdtd::from_csx(&FdtdSettings::new()).unwrap();
        assert_eq!(warn.len(), 1);
        assert!(warn[0].contains("NumberOfTimesteps"));
        assert_eq!(f.get_max_timesteps(), 0);
        assert_eq!(f.get_end_criteria().unwrap().energy_ratio, 1e-6);
        assert!(f.get_waveform().is_none());
        assert_eq!(*f.get_boundaries(), Boundaries::default());
    }

    #[test]
    fn periodic_has_no_file_form() {
        let f = Fdtd::new()
            .boundary(Face::YMin, Boundary::Periodic)
            .boundary(Face::YMax, Boundary::Periodic);
        let e = f.to_csx_checked().unwrap_err().to_string();
        assert!(e.contains("ymin"), "{e}");
        let b = f.to_csx().boundary.unwrap();
        assert_eq!(b.faces[2], CsxBoundary::Pec);
        assert!(Fdtd::new().to_csx_checked().is_ok());
    }

    #[test]
    fn boundary_kinds() {
        let mut s = FdtdSettings::new();
        s.number_of_timesteps = Some(100);
        s.boundary = Some(BoundaryConditions {
            faces: [
                CsxBoundary::Pec,
                CsxBoundary::Pmc,
                CsxBoundary::Mur,
                CsxBoundary::Pml(8),
                CsxBoundary::Pml(16),
                CsxBoundary::Mur,
            ],
            mur_phase_velocity: Some(2e8),
            ..Default::default()
        });
        let (f, warn) = Fdtd::from_csx(&s).unwrap();
        assert!(warn.is_empty(), "{warn:?}");
        let b = f.get_boundaries();
        assert_eq!(b[Face::XMin], Boundary::Pec);
        assert_eq!(b[Face::XMax], Boundary::Pmc);
        assert_eq!(b[Face::YMin], Boundary::Mur(Mur::with_phase_velocity(2e8)));
        assert_eq!(b[Face::YMax], Boundary::Pml(Cpml::cells(8)));
        assert_eq!(b[Face::ZMin], Boundary::Pml(Cpml::cells(16)));
        assert_eq!(b[Face::ZMax], Boundary::Mur(Mur::with_phase_velocity(2e8)));
        // Back to csx.
        let back = f.to_csx().boundary.unwrap();
        assert_eq!(back.faces, s.boundary.as_ref().unwrap().faces);
        assert_eq!(back.mur_phase_velocity, Some(2e8));

        // Mur without a phase velocity: c0.
        s.boundary = Some(BoundaryConditions::all(CsxBoundary::Mur));
        let (f, _) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(f.get_boundaries()[Face::XMin], Boundary::mur());
        assert_eq!(f.to_csx().boundary.unwrap().mur_phase_velocity, None);

        // PML_0 cannot be run.
        s.boundary = Some(BoundaryConditions::all(CsxBoundary::Pml(0)));
        assert!(Fdtd::from_csx(&s).is_err());

        // Grading expressions and stray attributes are reported.
        let mut b = BoundaryConditions::all(CsxBoundary::Pml(8));
        b.pml_grading =
            Some("-log(1e-6)*log(2.5)/(2*dl*Z*(pow(2.5,W/dl)-1)) * pow(2.5, D/dl)".into());
        b.mur_phase_velocity = Some(1e8);
        b.extra_attributes = vec![("foo".into(), "1".into())];
        s.boundary = Some(b);
        let (f, warn) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(f.get_boundaries()[Face::ZMax], Boundary::pml(8));
        assert_eq!(warn.len(), 3, "{warn:?}");
        assert!(warn.iter().any(|w| w.contains("PML_Grading")));
        assert!(warn.iter().any(|w| w.contains("MUR_PhaseVelocity")));
        assert!(warn.iter().any(|w| w.contains("foo")));
    }

    #[test]
    fn end_criteria_units() {
        let mut s = FdtdSettings::new();
        s.number_of_timesteps = Some(100);
        for (ratio, db) in [(1e-5, -50.0), (1e-6, -60.0), (1e-4, -40.0), (1.0, 0.0)] {
            s.end_criteria = Some(ratio);
            let (f, _) = Fdtd::from_csx(&s).unwrap();
            let e = f.get_end_criteria().unwrap();
            assert_eq!(e.energy_ratio, ratio);
            assert!((e.decay_db() - db).abs() < 1e-9, "{ratio}");
            assert_eq!(e.check_interval, EndCriteria::DEFAULT_CHECK_INTERVAL);
            assert_eq!(f.to_csx().end_criteria, Some(ratio));
        }
        // 0 means openEMS's default (-60 dB); "none" is written as a ratio
        // that is never reached, and read back as none.
        s.end_criteria = Some(0.0);
        let (f, _) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(f.get_end_criteria().unwrap().energy_ratio, 1e-6);
        let none = f.no_end_criteria();
        assert_eq!(none.to_csx().end_criteria, Some(Fdtd::NO_END_CRITERIA));
        let (back, _) = Fdtd::from_csx(&none.to_csx()).unwrap();
        assert!(back.get_end_criteria().is_none());
        // dB values or ratios above 1 are rejected.
        for bad in [-50.0, 2.0, f64::NAN] {
            s.end_criteria = Some(bad);
            assert!(Fdtd::from_csx(&s).is_err(), "{bad}");
        }
    }

    #[test]
    fn timestep_settings_and_warnings() {
        let mut s = FdtdSettings::gaussian(1e9, 5e8);
        s.number_of_timesteps = Some(30_000);
        s.time_step_factor = Some(0.5);
        s.time_step = Some(1e-13);
        s.max_time = Some(2e-9);
        s.over_sampling = Some(4);
        s.time_step_method = Some(TimeStepMethod::Cfl);
        let (f, warn) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(f.get_max_timesteps(), 30_000);
        assert_eq!(f.get_timestep_factor(), 0.5);
        assert_eq!(f.get_timestep(), Some(1e-13));
        assert_eq!(f.get_max_time(), Some(2e-9));
        assert_eq!(f.get_oversampling(), Some(4));
        assert!(warn.is_empty(), "{warn:?}");

        s.over_sampling = None;
        s.time_step_method = Some(TimeStepMethod::Rennings);
        s.cell_constant_material = Some(true);
        s.multi_grid = Some("10,20".into());
        s.coord_system = Some(0);
        s.extra_attributes = vec![("Verbose".into(), "3".into())];
        s.excitation_extra_attributes = vec![("Delay".into(), "1e-9".into())];
        let (f, warn) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(f.get_multigrid(), &[10.0, 20.0]);
        assert_eq!(f.to_csx().multi_grid.as_deref(), Some("10,20"));
        assert_eq!(warn.len(), 2, "{warn:?}");
        assert!(f.get_cell_constant_material());
        assert_eq!(
            f.get_timestep_method(),
            Some(crate::engine::TimestepMethod::Rennings)
        );
        s.time_step_method = Some(TimeStepMethod::Other(2));
        let (f2, w2) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(
            f2.get_timestep_method(),
            Some(crate::engine::TimestepMethod::Cfl)
        );
        assert!(w2.iter().any(|w| w.contains("TimeStepMethod=2")), "{w2:?}");
        for key in ["Verbose", "Delay"] {
            assert!(warn.iter().any(|w| w.contains(key)), "{key}: {warn:?}");
        }

        // Impossible settings are errors.
        let mut cyl = FdtdSettings::new();
        cyl.coord_system = Some(1);
        assert!(Fdtd::from_csx(&cyl).is_ok());
        let mut bad = FdtdSettings::new();
        bad.coord_system = Some(2);
        assert!(matches!(Fdtd::from_csx(&bad), Err(Error::Unsupported(_))));
        let mut bad = FdtdSettings::new();
        bad.multi_grid = Some("1,x".into());
        assert!(Fdtd::from_csx(&bad).is_err());
        bad.multi_grid = Some("-1".into());
        assert!(Fdtd::from_csx(&bad).is_err());
        // Zero steps is valid (openEMS runs nothing).
        let mut zero = FdtdSettings::new();
        zero.number_of_timesteps = Some(0);
        assert_eq!(Fdtd::from_csx(&zero).unwrap().0.get_max_timesteps(), 0);
        let mut bad = FdtdSettings::new();
        bad.time_step_factor = Some(1.5);
        assert!(Fdtd::from_csx(&bad).is_err());
        let mut bad = FdtdSettings::new();
        bad.excitation = Some(ExcitationSignal::Gaussian { f0: 1e9, fc: 0.0 });
        assert!(Fdtd::from_csx(&bad).is_err());
    }

    #[test]
    fn max_frequency_from_signal() {
        let mut s = FdtdSettings::new();
        s.excitation = Some(ExcitationSignal::Dirac { f_max: 4e9 });
        let (f, _) = Fdtd::from_csx(&s).unwrap();
        assert!(matches!(f.get_waveform(), Some(Waveform::Dirac)));
        assert_eq!(f.get_max_frequency(), Some(4e9));
        assert_eq!(
            f.to_csx().excitation,
            Some(ExcitationSignal::Dirac { f_max: 4e9 })
        );
        // The FDTD f_max attribute wins.
        s.f_max = Some(5e9);
        s.excitation = Some(ExcitationSignal::Step { f_max: 4e9 });
        let (f, _) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(f.get_max_frequency(), Some(5e9));
        assert_eq!(
            f.to_csx().excitation,
            Some(ExcitationSignal::Step { f_max: 5e9 })
        );
    }

    #[test]
    fn round_trip_fdtd_csx_fdtd() {
        let cases = [
            Fdtd::new(),
            Fdtd::new()
                .waveform(Waveform::gaussian(2e9, 1e9))
                .boundaries(Boundaries::all(Boundary::pml(8)))
                .boundary(Face::ZMin, Boundary::Pec)
                .boundary(Face::XMax, Boundary::Mur(Mur::with_phase_velocity(1.5e8)))
                .boundary(Face::YMax, Boundary::Pmc)
                .end_criteria(EndCriteria::energy_decay_db(-60.0))
                .max_timesteps(12_345)
                .timestep_factor(0.7)
                .timestep(2e-13)
                .max_time(3e-9)
                .max_frequency(4e9)
                .oversampling(6),
            Fdtd::new()
                .waveform(Waveform::sinusoid(1e9))
                .no_end_criteria(),
            Fdtd::new().waveform(Waveform::step()).max_frequency(1e9),
            Fdtd::new().waveform(
                Waveform::from_csx(&ExcitationSignal::Custom {
                    function: "sin(2*pi*f0*t)".into(),
                    f0: 1e9,
                    f_max: 2e9,
                })
                .unwrap(),
            ),
        ];
        for f in cases {
            let s = f.to_csx();
            assert!(s.boundary.is_some());
            let (back, _) = Fdtd::from_csx(&s).unwrap();
            assert_same(&f, &back);
            // And csx → Fdtd → csx is stable (a missing TimeStepMethod is
            // read as openEMS's default, Rennings).
            let mut s = s;
            s.time_step_method.get_or_insert(TimeStepMethod::Rennings);
            assert_eq!(back.to_csx(), s);
        }

        // Lossy parts: grading parameters and closures.
        let f = Fdtd::new()
            .boundaries(Boundaries::all(Boundary::Pml(
                Cpml::cells(10).kappa_max(3.0),
            )))
            .waveform(Waveform::custom(|t| t));
        let s = f.to_csx();
        assert!(s.excitation.is_none());
        let (back, _) = Fdtd::from_csx(&s).unwrap();
        assert_eq!(back.get_boundaries()[Face::XMin], Boundary::pml(10));
        assert!(back.get_waveform().is_none());
    }

    #[cfg(feature = "xml")]
    #[test]
    fn openems_xml_snippet() {
        use csx::io::xml::OpenEmsDocument;
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<openEMS>
  <FDTD NumberOfTimesteps="20000" endCriteria="1e-4" f_max="3e9" TimeStepFactor="0.9">
    <Excitation Type="0" f0="2e9" fc="1e9"/>
    <BoundaryCond xmin="PML_8" xmax="PML_8" ymin="MUR" ymax="PMC" zmin="PEC" zmax="3"/>
  </FDTD>
  <ContinuousStructure CoordSystem="0">
    <Properties/>
    <RectilinearGrid DeltaUnit="0.001" CoordSystem="0">
      <XLines>0,1,2</XLines>
      <YLines>0,1,2</YLines>
      <ZLines>0,1,2</ZLines>
    </RectilinearGrid>
  </ContinuousStructure>
</openEMS>"#;
        let doc = OpenEmsDocument::from_xml_str(xml).unwrap();
        let s = doc.fdtd_settings().unwrap().unwrap();
        let (f, warn) = Fdtd::from_csx(&s).unwrap();
        assert!(warn.is_empty(), "{warn:?}");
        assert_eq!(f.get_max_timesteps(), 20_000);
        assert_eq!(f.get_end_criteria().unwrap().energy_ratio, 1e-4);
        assert_eq!(f.get_timestep_factor(), 0.9);
        assert_eq!(f.get_max_frequency(), Some(3e9));
        assert!(matches!(
            f.get_waveform(),
            Some(Waveform::OpenEmsGaussian { f0: 2e9, fc: 1e9 })
        ));
        let b = f.get_boundaries();
        assert_eq!(b[Face::XMin], Boundary::pml(8));
        assert_eq!(b[Face::XMax], Boundary::pml(8));
        assert_eq!(b[Face::YMin], Boundary::mur());
        assert_eq!(b[Face::YMax], Boundary::Pmc);
        assert_eq!(b[Face::ZMin], Boundary::Pec);
        assert_eq!(b[Face::ZMax], Boundary::pml(8));

        // Write back through csx and read again.
        let mut doc = doc;
        doc.set_fdtd_settings(&f.to_csx());
        let again = OpenEmsDocument::from_xml_str(&doc.to_xml_string()).unwrap();
        let (f2, _) = Fdtd::from_csx(&again.fdtd_settings().unwrap().unwrap()).unwrap();
        assert_same(&f, &f2);
    }
}
