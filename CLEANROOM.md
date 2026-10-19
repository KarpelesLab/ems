# Clean-room policy

`ems` is an independent implementation of an FDTD solver. It interoperates
with openEMS (it reads openEMS models and aims to behave compatibly), but it
must not be derived from openEMS or any other GPL-licensed code. This is what
allows it to be MIT licensed.

These rules apply to everyone who contributes: humans and AI agents alike.

## Not allowed

* Reading, copying, translating, or paraphrasing the source code of openEMS,
  CSXCAD, AppCSXCAD, or their bindings. This includes their Python and Matlab
  helper scripts. It also includes reading them "just to check how they did
  it" or "only to compare results".
* Copying identifiers, file layouts, comments, or numeric constants that only
  exist in that code.
* Pasting code from forks, ports, or derivatives of those projects, or from
  answers that quote them.

If you have studied that code in depth, say so in your PR, and do not work on
the corresponding area of `ems`.

## Allowed

* Published literature. For example: Yee (1966); Taflove & Hagness,
  *Computational Electrodynamics: The FDTD Method*; Gedney, *Introduction to
  the FDTD Method*; Roden & Gedney (2000) on CPML; Berenger (1994); Mur (1981);
  Luebbers and Piket-May on lumped elements; Pozar, *Microwave Engineering*.
* openEMS's public **user documentation**: the wiki, tutorials, published
  papers, and API reference pages. Use these to learn which features exist and
  what they mean to a user.
* The **observable openEMS XML file format**, for interoperability. Element and
  attribute names are an interface. Where the documentation does not specify
  a behaviour, choose one deliberately and document it in rustdoc. Do not go
  and look it up in the source.
* Comparing *outputs* of openEMS and `ems` on the same model, as a black box.
  The primary reference stays analytic results, though (see the validation
  plan in [docs/DESIGN.md](docs/DESIGN.md)).
  The files openEMS writes (probe text files, HDF5 and VTK dumps) are
  observable output and may be matched for compatibility. Use the oracle in
  [scripts/openems-oracle](scripts/openems-oracle/README.md), which keeps
  only the compiled tools.
* Permissively licensed crates as dependencies.

## In practice

* Every non-trivial algorithm has a doc comment that names its literature
  source.
* Each pull request affirms this policy (see the PR template).
