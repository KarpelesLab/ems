//! Running openEMS XML documents (feature `xml`, design §3 M1, §10 step 8).

use std::path::Path;

use csx::OpenEmsDocument;

use super::Simulation;
use crate::Result;
use crate::compile::CompileOptions;
use crate::fdtd::Fdtd;

impl Fdtd {
    /// The solver settings of an openEMS document: its `<FDTD>` element
    /// converted with [`Fdtd::from_csx`] (settings and warnings).
    ///
    /// A document without `<FDTD>` (e.g. a bare `<ContinuousStructure>`
    /// file) gets the settings of an empty `<FDTD/>` element (openEMS
    /// defaults: no excitation, PEC boundaries), but
    /// [`OPENEMS_DEFAULT_TIMESTEPS`](Fdtd::OPENEMS_DEFAULT_TIMESTEPS)
    /// steps, and a warning.
    ///
    /// # Errors
    /// A malformed `<FDTD>` element ([`Error::Csx`](crate::Error::Csx)) and
    /// the errors of [`Fdtd::from_csx`].
    pub fn from_document(doc: &OpenEmsDocument) -> Result<(Fdtd, Vec<String>)> {
        match doc.fdtd_settings()? {
            Some(s) => Fdtd::from_csx(&s),
            None => {
                let (f, mut w) = Fdtd::from_csx(&Default::default())?;
                let f = f.max_timesteps(Fdtd::OPENEMS_DEFAULT_TIMESTEPS);
                w.retain(|w| !w.contains("NumberOfTimesteps"));
                w.insert(
                    0,
                    "the document has no <FDTD> settings; openEMS defaults are used".into(),
                );
                Ok((f, w))
            }
        }
    }
}

impl Simulation {
    /// Reads the openEMS XML file at `path` (via
    /// [`OpenEmsDocument::read_file`]) and compiles it; see
    /// [`from_document`](Self::from_document).
    ///
    /// ```no_run
    /// let mut sim = ems::Simulation::from_xml("patch.xml")?;
    /// for w in sim.warnings() {
    ///     eprintln!("warning: {w}");
    /// }
    /// let res = sim.run()?;
    /// # Ok::<(), ems::Error>(())
    /// ```
    ///
    /// # Errors
    /// I/O and XML parse errors ([`Error::Csx`](crate::Error::Csx)) and
    /// those of [`from_document`](Self::from_document).
    pub fn from_xml(path: impl AsRef<Path>) -> Result<Self> {
        let doc = OpenEmsDocument::read_file(path)?;
        Self::from_document(&doc)
    }

    /// Parses an openEMS XML document from a string and compiles it; see
    /// [`from_document`](Self::from_document).
    ///
    /// # Errors
    /// XML parse errors ([`Error::Csx`](crate::Error::Csx)) and those of
    /// [`from_document`](Self::from_document).
    pub fn from_xml_str(xml: &str) -> Result<Self> {
        let doc = OpenEmsDocument::from_xml_str(xml)?;
        Self::from_document(&doc)
    }

    /// Compiles a parsed openEMS document with the default
    /// [`CompileOptions`]: the `<FDTD>` element becomes the [`Fdtd`]
    /// settings ([`Fdtd::from_document`]) and the structure is compiled
    /// and mapped like a Rust-built model ([`Simulation::new`]). The
    /// warnings of the settings conversion come first in
    /// [`warnings`](Self::warnings).
    ///
    /// Running a model built in Rust directly and after a round trip
    /// through csx's XML writer gives bit-identical results (the settings
    /// that have no file representation, see [`Fdtd::to_csx`], aside).
    ///
    /// # Errors
    /// See [`Fdtd::from_document`] and [`Simulation::new`].
    pub fn from_document(doc: &OpenEmsDocument) -> Result<Self> {
        Self::from_document_with(doc, &CompileOptions::default(), |f| f)
    }

    /// [`from_document`](Self::from_document) with explicit compile
    /// options, and `adjust` applied to the file's settings before
    /// compiling (e.g. `|f| f.threads(4)` for settings the file format
    /// cannot express).
    ///
    /// # Errors
    /// See [`from_document`](Self::from_document); the adjusted settings
    /// must still be valid.
    pub fn from_document_with(
        doc: &OpenEmsDocument,
        opts: &CompileOptions,
        adjust: impl FnOnce(Fdtd) -> Fdtd,
    ) -> Result<Self> {
        let (fdtd, mut warnings) = Fdtd::from_document(doc)?;
        let fdtd = adjust(fdtd);
        fdtd.validate()?;
        let mut sim = Self::with_options(&doc.structure, &fdtd, opts)?;
        warnings.append(&mut sim.warnings);
        sim.warnings = warnings;
        Ok(sim)
    }
}
