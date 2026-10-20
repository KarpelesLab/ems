//! Simulation results: model probes and dumps by csx property.

use std::path::Path;

use crate::compile::sar::SarKind;
use crate::engine::{FieldSampler, FreqDump, RunOutcome, Snapshot, VtrSeries};
use crate::post::sar::{SarField, SarVolume};
use crate::post::{Complex64, dft};
use crate::{Error, Result};

/// What a model probe measures (csx `ProbeBox` type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeKind {
    /// Line integral of E \[V\], sampled at the E times `(n + 1)·dt`.
    Voltage,
    /// Loop integral of H \[A\] (or, for
    /// [`Results::lumped_current`], a lumped element's terminal current),
    /// sampled at the H times `(n + ½)·dt`.
    Current,
    /// E field vector at a node \[V/m\], at the E times.
    EField,
    /// H field vector at a node \[A/m\], at the H times.
    HField,
}

/// Where a model probe sits on the mesh, as openEMS reports it in the
/// header of its probe files (`start-coordinates: (x,y,z) m -> [i,j,k]`).
///
/// The indices are primary node indices for voltage and E-field probes
/// and dual indices for current and H-field probes (index `i` is the
/// dual line between nodes `i` and `i + 1`); the coordinates are those of
/// the primary node with the same indices, in metres (openEMS prints
/// them the same way).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeLocation {
    /// Indices of the first corner.
    pub start: [usize; 3],
    /// Indices of the second corner.
    pub stop: [usize; 3],
    /// Coordinates of the first corner's node \[m\].
    pub start_coords: [f64; 3],
    /// Coordinates of the second corner's node \[m\].
    pub stop_coords: [f64; 3],
}

/// Recorded values of a probe.
#[derive(Debug, Clone, PartialEq)]
pub enum ProbeValues {
    /// Voltage or current samples.
    Scalar(Vec<f64>),
    /// Field vectors `[x, y, z]`.
    Vector(Vec<[f64; 3]>),
}

/// The recorded time series of one csx `ProbeBox`, restricted to its
/// `StartTime`/`StopTime` window (a stop time `≤ 0` means "no limit") and
/// multiplied by its weight.
///
/// The window follows openEMS (black-box comparison): E-type probes
/// (voltage, E field) keep the samples with `start − dt ≤ t ≤ stop + dt`,
/// H-type probes (current, H field) those with `start ≤ t ≤ stop + dt`
/// (`t` the sample time, `dt` the time step).
///
/// In a Bloch run ([`Simulation`](crate::Simulation), Bloch mode) the
/// probe value is complex: [`values`](Self::values) is the real part (the
/// real engine), [`imag`](Self::imag) the imaginary part, and the DFTs
/// ([`dft`](Self::dft), [`dft_vector`](Self::dft_vector),
/// [`spectrum`](Self::spectrum)) return the spectrum of the complex series,
/// `X(f) = DFT[x_re](f) + j·DFT[x_im](f)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeResult {
    pub(crate) name: String,
    pub(crate) id: u32,
    pub(crate) number: u32,
    pub(crate) kind: ProbeKind,
    pub(crate) frequencies: Vec<f64>,
    pub(crate) dt: f64,
    pub(crate) times: Vec<f64>,
    pub(crate) values: ProbeValues,
    pub(crate) imag: Option<ProbeValues>,
    pub(crate) location: Option<ProbeLocation>,
}

impl ProbeResult {
    /// csx property name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// csx property id ([`csx::Property::id`], equal to
    /// [`csx::PropertyId::get`]).
    pub fn id(&self) -> u32 {
        self.id
    }

    /// The probe number of the property.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// What is measured.
    pub fn kind(&self) -> ProbeKind {
        self.kind
    }

    /// The snapped position of a model probe (`None` for lumped-element
    /// currents and waveguide probes).
    pub fn location(&self) -> Option<&ProbeLocation> {
        self.location.as_ref()
    }

    /// The property's frequencies for frequency-domain output \[Hz\]
    /// (`FD_Samples`; may be empty).
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Sample interval \[s\].
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Sample times \[s\] (uniform, `dt` apart).
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// The recorded values.
    pub fn values(&self) -> &ProbeValues {
        &self.values
    }

    /// Voltage / current samples (`None` for field probes).
    pub fn scalar(&self) -> Option<&[f64]> {
        match &self.values {
            ProbeValues::Scalar(v) => Some(v),
            ProbeValues::Vector(_) => None,
        }
    }

    /// Field samples (`None` for voltage / current probes).
    pub fn vector(&self) -> Option<&[[f64; 3]]> {
        match &self.values {
            ProbeValues::Vector(v) => Some(v),
            ProbeValues::Scalar(_) => None,
        }
    }

    /// The imaginary part of the recorded values in a Bloch run (`None`
    /// otherwise); same kind and length as [`values`](Self::values).
    pub fn imag(&self) -> Option<&ProbeValues> {
        self.imag.as_ref()
    }

    fn t0(&self) -> f64 {
        self.times.first().copied().unwrap_or(0.0)
    }

    /// `re + j·im` of two spectra.
    fn combine(re: Vec<Complex64>, im: Vec<Complex64>) -> Vec<Complex64> {
        re.into_iter()
            .zip(im)
            .map(|(r, i)| r + Complex64::I * i)
            .collect()
    }

    /// DFT `X(f) = Σ x(t_n)·e^{−j2πf·t_n}·dt` of a voltage / current probe at
    /// `freqs` \[Hz\], with the samples' own absolute times (so the half-step
    /// offset of currents is exact). The result equals that of a running
    /// DFT ([`RunningDft`](crate::post::RunningDft)) over the same samples.
    ///
    /// # Errors
    /// [`Error::Invalid`] for a field probe (use
    /// [`dft_vector`](Self::dft_vector)) or a probe without samples.
    pub fn dft(&self, freqs: &[f64]) -> Result<Vec<Complex64>> {
        let v = self.scalar().ok_or_else(|| {
            Error::Invalid(format!("probe {:?} records a field vector", self.name))
        })?;
        if v.is_empty() {
            return Err(Error::Invalid(format!(
                "probe {:?} has no samples",
                self.name
            )));
        }
        let re = dft(self.t0(), self.dt, v, freqs);
        Ok(match &self.imag {
            Some(ProbeValues::Scalar(im)) => Self::combine(re, dft(self.t0(), self.dt, im, freqs)),
            _ => re,
        })
    }

    /// DFT of each component of a field probe at `freqs` \[Hz\] (see
    /// [`dft`](Self::dft)).
    ///
    /// # Errors
    /// [`Error::Invalid`] for a voltage / current probe or a probe without
    /// samples.
    pub fn dft_vector(&self, freqs: &[f64]) -> Result<Vec<[Complex64; 3]>> {
        let v = self
            .vector()
            .ok_or_else(|| Error::Invalid(format!("probe {:?} records a scalar", self.name)))?;
        if v.is_empty() {
            return Err(Error::Invalid(format!(
                "probe {:?} has no samples",
                self.name
            )));
        }
        let series = |v: &[[f64; 3]], c: usize| {
            let x: Vec<f64> = v.iter().map(|s| s[c]).collect();
            dft(self.t0(), self.dt, &x, freqs)
        };
        let comp: Vec<Vec<Complex64>> = (0..3)
            .map(|c| match &self.imag {
                Some(ProbeValues::Vector(im)) => Self::combine(series(v, c), series(im, c)),
                _ => series(v, c),
            })
            .collect();
        Ok((0..freqs.len())
            .map(|k| [comp[0][k], comp[1][k], comp[2][k]])
            .collect())
    }

    /// [`dft`](Self::dft) at the probe's own [`frequencies`](Self::frequencies).
    ///
    /// # Errors
    /// See [`dft`](Self::dft).
    pub fn spectrum(&self) -> Result<Vec<Complex64>> {
        self.dft(&self.frequencies)
    }
}

/// The recorded data of one csx `DumpBox`.
#[derive(Debug, Clone)]
pub enum DumpData {
    /// Time-domain snapshots (`EField` / `HField` dumps).
    Time {
        /// The sampler (points, coordinates, quantity).
        sampler: FieldSampler,
        /// Snapshots in recording order.
        snapshots: Vec<Snapshot>,
    },
    /// Frequency-domain accumulators (`EFieldFD` / `HFieldFD` dumps).
    Freq(FreqDump),
}

/// One dump result, named after its csx property.
///
/// In a Bloch run a frequency-domain dump holds the complex Bloch field
/// `F_re + j·F_im` (the two engines' DFTs combined), and a time-domain dump
/// the real part, with the imaginary part's snapshots in
/// [`imag_snapshots`](Self::imag_snapshots).
#[derive(Debug, Clone)]
pub struct DumpResult {
    pub(crate) name: String,
    pub(crate) id: u32,
    pub(crate) file_type: csx::properties::DumpFileType,
    pub(crate) dump_type: csx::properties::DumpType,
    pub(crate) data: DumpData,
    pub(crate) imag: Option<Vec<Snapshot>>,
    pub(crate) delta_unit: f64,
    pub(crate) cylindrical: bool,
    pub(crate) dt: f64,
}

impl DumpResult {
    /// csx property name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// csx property id ([`csx::Property::id`], equal to
    /// [`csx::PropertyId::get`]).
    pub fn id(&self) -> u32 {
        self.id
    }

    /// The file type the DumpBox asks for (`FileType`: VTK or HDF5). The
    /// writers here do not look at it; it tells a runner which one to use
    /// ([`write_vtr`](Self::write_vtr) or, with feature `hdf5`,
    /// `write_h5`).
    pub fn file_type(&self) -> csx::properties::DumpFileType {
        self.file_type
    }

    /// The `DumpType` of the DumpBox.
    pub fn dump_type(&self) -> csx::properties::DumpType {
        self.dump_type
    }

    /// The model's drawing unit \[m\] (the openEMS `mesh_scaling`).
    pub fn delta_unit(&self) -> f64 {
        self.delta_unit
    }

    /// Whether the dump is on a cylindrical grid (sample coordinates
    /// `(ρ, α, z)`, components `(ρ, α, z)`).
    pub fn is_cylindrical(&self) -> bool {
        self.cylindrical
    }

    /// The engine time step \[s\].
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Writes the dump the way openEMS does for its `FileType` (see
    /// [`io::openems::dump`](crate::io::openems::dump)) into `dir`;
    /// returns the written paths.
    ///
    /// # Errors
    /// See [`write_dump`](crate::io::openems::dump::write_dump).
    pub fn write_openems(
        &self,
        dir: impl AsRef<Path>,
        opts: &crate::io::openems::dump::DumpFileOptions,
    ) -> Result<Vec<std::path::PathBuf>> {
        crate::io::openems::dump::write_dump(dir, self, opts)
    }

    /// The data.
    pub fn data(&self) -> &DumpData {
        &self.data
    }

    /// The sampler (sample points and quantity).
    pub fn sampler(&self) -> &FieldSampler {
        match &self.data {
            DumpData::Time { sampler, .. } => sampler,
            DumpData::Freq(f) => f.sampler(),
        }
    }

    /// Time-domain snapshots (`None` for a frequency-domain dump).
    pub fn snapshots(&self) -> Option<&[Snapshot]> {
        match &self.data {
            DumpData::Time { snapshots, .. } => Some(snapshots),
            DumpData::Freq(_) => None,
        }
    }

    /// The imaginary part's snapshots of a time-domain dump in a Bloch run
    /// (`None` otherwise), in the order of [`snapshots`](Self::snapshots).
    pub fn imag_snapshots(&self) -> Option<&[Snapshot]> {
        self.imag.as_deref()
    }

    /// The frequency-domain dump (`None` for a time-domain dump).
    pub fn freq(&self) -> Option<&FreqDump> {
        match &self.data {
            DumpData::Freq(f) => Some(f),
            DumpData::Time { .. } => None,
        }
    }

    /// Writes the dump as `.vtr` files plus a `.pvd` index into `dir`
    /// (created if needed), named `<basename>_…`: one file per snapshot
    /// ([`VtrSeries`]) or per frequency ([`FreqDump::write_vtr`]).
    ///
    /// # Errors
    /// I/O errors.
    pub fn write_vtr(&self, dir: impl AsRef<Path>, basename: &str) -> Result<()> {
        match &self.data {
            DumpData::Time { sampler, snapshots } => {
                use crate::engine::DumpSink;
                let mut series = VtrSeries::new(dir.as_ref(), basename);
                for s in snapshots {
                    series.snapshot(sampler, s.clone())?;
                }
                Ok(())
            }
            DumpData::Freq(f) => f.write_vtr(dir, basename),
        }
    }

    /// Writes the dump as one HDF5 file at `path` in the ems layout of
    /// [`io::h5`](crate::io::h5), uncompressed (use
    /// [`io::h5::write_dump`](crate::io::h5::write_dump) for options).
    ///
    /// # Errors
    /// I/O and encoding errors.
    #[cfg(feature = "hdf5")]
    pub fn write_h5(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::io::h5::write_dump(path, self, &crate::io::h5::H5Options::default())
    }
}

/// One SAR dump result (csx `DumpBox` of type `SARLocal`, `SAR1g`,
/// `SAR10g` or `SARRaw`), named after its csx property: the absorbed power
/// and mass density per cell at each of its frequencies (see
/// [`post::sar`](crate::post::sar)).
///
/// The power comes from DFTs of the transient run, so it is a spectral
/// density (W/m³·s²); normalise it to the accepted port power at the same
/// frequency ([`SarVolume::scaled`]) for SAR per watt.
#[derive(Debug, Clone)]
pub struct SarResult {
    pub(crate) name: String,
    pub(crate) id: u32,
    pub(crate) kind: SarKind,
    pub(crate) file_type: csx::properties::DumpFileType,
    pub(crate) frequencies: Vec<f64>,
    pub(crate) volumes: Vec<SarVolume>,
    pub(crate) openems: Option<crate::io::openems::dump::OpenEmsSar>,
    pub(crate) delta_unit: f64,
}

impl SarResult {
    /// csx property name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// csx property id.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// What the dump asks for.
    pub fn kind(&self) -> SarKind {
        self.kind
    }

    /// The DumpBox `FileType` (VTK or HDF5).
    pub fn file_type(&self) -> csx::properties::DumpFileType {
        self.file_type
    }

    /// The analysis frequencies \[Hz\].
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// The power and density per cell at frequency index `k`.
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn volume(&self, k: usize) -> &SarVolume {
        &self.volumes[k]
    }

    /// The model's drawing unit \[m\].
    pub fn delta_unit(&self) -> f64 {
        self.delta_unit
    }

    /// The openEMS-compatible SAR data (cell-interpolated E DFT and cell
    /// materials, see [`io::openems::dump`](crate::io::openems::dump#sar-dumps));
    /// `None` on cylindrical grids and in Bloch runs.
    pub fn openems(&self) -> Option<&crate::io::openems::dump::OpenEmsSar> {
        self.openems.as_ref()
    }

    /// Writes the dump like openEMS into `dir`; see
    /// [`write_sar`](crate::io::openems::dump::write_sar).
    ///
    /// # Errors
    /// See [`write_sar`](crate::io::openems::dump::write_sar).
    pub fn write_openems(&self, dir: impl AsRef<Path>) -> Result<Vec<std::path::PathBuf>> {
        crate::io::openems::dump::write_sar(dir, self)
    }

    /// All volumes, one per frequency.
    pub fn volumes(&self) -> &[SarVolume] {
        &self.volumes
    }

    /// The SAR the dump asks for at frequency index `k`: local
    /// (`SARLocal`, `SARRaw`) or averaged over 1 g / 10 g.
    ///
    /// # Errors
    /// See [`SarVolume::averaged`].
    ///
    /// # Panics
    /// If `k` is out of range.
    pub fn field(&self, k: usize) -> Result<SarField> {
        let v = &self.volumes[k];
        match self.kind {
            SarKind::Local | SarKind::Raw => Ok(v.local()),
            SarKind::Averaged(m) => v.averaged(m),
        }
    }
}

/// Results of a [`Simulation`](crate::Simulation): every model probe and
/// dump, looked up by csx property name or id, plus the run outcome.
///
/// This is an owned copy of the recorded data at the time it was taken
/// ([`Simulation::results`](crate::Simulation::results)); time-domain dumps
/// are copied too (use [`Simulation::time_dump`](crate::Simulation::time_dump)
/// to avoid that for large dumps).
#[derive(Debug, Clone)]
pub struct Results {
    pub(crate) outcome: Option<RunOutcome>,
    pub(crate) dt: f64,
    /// openEMS sampling intervals `(time domain, Nyquist)` in steps.
    pub(crate) openems_intervals: (u64, u64),
    pub(crate) probes: Vec<ProbeResult>,
    pub(crate) lumped: Vec<ProbeResult>,
    pub(crate) dumps: Vec<DumpResult>,
    pub(crate) sar: Vec<SarResult>,
    pub(crate) incident: Vec<IncidentWave>,
}

/// The incident wave of a plane-wave excitation in a Bloch run: the
/// complex incident E on its entry plane at the phase reference (the
/// first mesh line of every axis), per step at the E times `(n + 1)·dt`
/// (see [`PeriodicPlaneWave::incident`](crate::engine::PeriodicPlaneWave::incident)).
/// It is the reference for reflection and transmission: a field
/// `F(r)` measured elsewhere relates to it through the Bloch phase
/// `e^{−j k_t·(r − r_0)}` and the propagation along the normal.
#[derive(Debug, Clone, PartialEq)]
pub struct IncidentWave {
    pub(crate) name: String,
    pub(crate) dt: f64,
    pub(crate) k_t: [f64; 3],
    pub(crate) e: Vec<[Complex64; 3]>,
}

impl IncidentWave {
    /// csx property name of the excitation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The transverse (Bloch) wavevector \[rad/m\].
    pub fn wavevector(&self) -> [f64; 3] {
        self.k_t
    }

    /// Sample times \[s\]: `(n + 1)·dt` for sample `n`.
    pub fn times(&self) -> Vec<f64> {
        (1..=self.e.len()).map(|n| n as f64 * self.dt).collect()
    }

    /// The complex incident E \[V/m\] per step.
    pub fn values(&self) -> &[[Complex64; 3]] {
        &self.e
    }

    /// Spectrum of each component of the complex series at `freqs` \[Hz\]
    /// (the DFT of [`ProbeResult::dft`]).
    pub fn dft(&self, freqs: &[f64]) -> Vec<[Complex64; 3]> {
        let comp: Vec<Vec<Complex64>> = (0..3)
            .map(|c| {
                let re: Vec<f64> = self.e.iter().map(|v| v[c].re).collect();
                let im: Vec<f64> = self.e.iter().map(|v| v[c].im).collect();
                ProbeResult::combine(
                    dft(self.dt, self.dt, &re, freqs),
                    dft(self.dt, self.dt, &im, freqs),
                )
            })
            .collect();
        (0..freqs.len())
            .map(|k| [comp[0][k], comp[1][k], comp[2][k]])
            .collect()
    }
}

impl Results {
    /// The steps between two samples of openEMS's time-domain probe files
    /// for this simulation
    /// ([`Fdtd::openems_probe_interval`](crate::Fdtd::openems_probe_interval)).
    /// The probes here record every step; [`crate::io::openems`] writes
    /// every this many.
    pub fn openems_probe_interval(&self) -> u64 {
        self.openems_intervals.0
    }

    /// openEMS's Nyquist interval in steps
    /// ([`Fdtd::openems_nyquist_interval`](crate::Fdtd::openems_nyquist_interval)),
    /// the sampling of its frequency-domain probe values.
    pub fn openems_nyquist_interval(&self) -> u64 {
        self.openems_intervals.1
    }

    /// How the run stopped (`None` for results taken without
    /// [`Simulation::run`](crate::Simulation::run)).
    pub fn outcome(&self) -> Option<&RunOutcome> {
        self.outcome.as_ref()
    }

    /// The engine time step \[s\].
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// All probes, in model order.
    pub fn probes(&self) -> &[ProbeResult] {
        &self.probes
    }

    /// The probe of the csx property named `name`.
    pub fn probe(&self, name: &str) -> Option<&ProbeResult> {
        self.probes.iter().find(|p| p.name == name)
    }

    /// The probe of the csx property `id`.
    pub fn probe_by_id(&self, id: csx::PropertyId) -> Option<&ProbeResult> {
        self.probes.iter().find(|p| p.id == id.get())
    }

    /// The terminal current of the csx `LumpedElement` named `name`: the
    /// current through the element along `+axis` \[A\], i.e. the current
    /// of its resistor (with any Thevenin source), folded capacitor and
    /// inductor / series branch, without the background capacitance of its
    /// edges (see [`LumpedElement`](crate::engine::LumpedElement), "Terminal
    /// current"). Sampled at the H times `(n + ½)·dt`, kind
    /// [`ProbeKind::Current`], number 0, no frequencies; its
    /// [`id`](ProbeResult::id) is the element's property id.
    ///
    /// This is the port current used by
    /// [`port::LumpedPort::analyze`](crate::port::LumpedPort::analyze).
    pub fn lumped_current(&self, name: &str) -> Option<&ProbeResult> {
        self.lumped.iter().find(|p| p.name == name)
    }

    /// The terminal currents of all mapped lumped elements (see
    /// [`lumped_current`](Self::lumped_current)), in model order.
    pub fn lumped_currents(&self) -> &[ProbeResult] {
        &self.lumped
    }

    /// All dumps, in model order.
    pub fn dumps(&self) -> &[DumpResult] {
        &self.dumps
    }

    /// The dump of the csx property named `name`.
    pub fn dump(&self, name: &str) -> Option<&DumpResult> {
        self.dumps.iter().find(|d| d.name == name)
    }

    /// The dump of the csx property `id`.
    pub fn dump_by_id(&self, id: csx::PropertyId) -> Option<&DumpResult> {
        self.dumps.iter().find(|d| d.id == id.get())
    }

    /// All SAR dumps, in model order (they are not in
    /// [`dumps`](Self::dumps)).
    pub fn sar_dumps(&self) -> &[SarResult] {
        &self.sar
    }

    /// The SAR dump of the csx property named `name`.
    pub fn sar(&self, name: &str) -> Option<&SarResult> {
        self.sar.iter().find(|d| d.name == name)
    }

    /// Whether these are the results of a Bloch run (complex probes).
    pub fn is_bloch(&self) -> bool {
        self.probes
            .iter()
            .chain(&self.lumped)
            .any(|p| p.imag.is_some())
            || !self.incident.is_empty()
    }

    /// The incident waves of the plane-wave excitations of a Bloch run,
    /// in model order (empty otherwise).
    pub fn incident_waves(&self) -> &[IncidentWave] {
        &self.incident
    }

    /// The incident wave of the plane-wave excitation named `name` in a
    /// Bloch run.
    pub fn incident(&self, name: &str) -> Option<&IncidentWave> {
        self.incident.iter().find(|w| w.name == name)
    }
}
