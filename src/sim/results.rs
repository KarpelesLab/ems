//! Simulation results: model probes and dumps by csx property.

use std::path::Path;

use crate::engine::{FieldSampler, FreqDump, RunOutcome, Snapshot, VtrSeries};
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
}

impl ProbeResult {
    /// csx property name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// csx property id.
    pub fn id(&self) -> csx::PropertyId {
        csx::PropertyId(self.id)
    }

    /// The probe number of the property.
    pub fn number(&self) -> u32 {
        self.number
    }

    /// What is measured.
    pub fn kind(&self) -> ProbeKind {
        self.kind
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

    fn t0(&self) -> f64 {
        self.times.first().copied().unwrap_or(0.0)
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
        Ok(dft(self.t0(), self.dt, v, freqs))
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
        let comp: Vec<Vec<Complex64>> = (0..3)
            .map(|c| {
                let x: Vec<f64> = v.iter().map(|s| s[c]).collect();
                dft(self.t0(), self.dt, &x, freqs)
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
#[derive(Debug, Clone)]
pub struct DumpResult {
    pub(crate) name: String,
    pub(crate) id: u32,
    pub(crate) data: DumpData,
}

impl DumpResult {
    /// csx property name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// csx property id.
    pub fn id(&self) -> csx::PropertyId {
        csx::PropertyId(self.id)
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
    pub(crate) probes: Vec<ProbeResult>,
    pub(crate) lumped: Vec<ProbeResult>,
    pub(crate) dumps: Vec<DumpResult>,
}

impl Results {
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
        self.probes.iter().find(|p| p.id == id.0)
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
        self.dumps.iter().find(|d| d.id == id.0)
    }
}
