---
name: astropy
description: Work with astronomical data using astropy - coordinate system transforms, FITS file handling, cosmological calculations, unit-aware quantities for astronomical scales, and catalog cross-matching. Use for any astronomy-adjacent task, including astronomy-flavored ML/data-mining work over star/object catalogs.
category: physics
---

# Astropy for Astronomical Computation

## Coordinate systems and transforms
Astronomical positions are expressed in multiple coordinate frames (ICRS, Galactic,
AltAz, and others) — `astropy.coordinates.SkyCoord` handles transforms between them
correctly, including the subtleties (precession, nutation, aberration, proper motion over
time) that a manual spherical-trigonometry implementation is likely to get subtly wrong.
Always transform via `SkyCoord`, not by hand, unless there's a specific reason to
re-derive it.

## Units and quantities
`astropy.units` provides unit-aware quantities specifically calibrated for astronomical
scales (parsecs, solar masses, magnitudes, etc.) that compose correctly through
arithmetic — carry units through a calculation with `astropy.units.Quantity` rather than
bare floats, the same discipline as `pint` for general physics, but with astronomy-specific
unit definitions already correct (e.g. magnitude systems, which don't behave like linear
units and need `astropy`'s explicit handling rather than naive arithmetic).

## FITS files
FITS is the standard astronomical data format (images, spectra, tables, with rich header
metadata). `astropy.io.fits` reads/writes it; always inspect the header (`hdul[0].header`)
before assuming what a FITS file contains — WCS (World Coordinate System) information for
mapping pixel coordinates to sky coordinates lives in the header and is essential for any
spatial analysis of the image data.

## Cosmological calculations
`astropy.cosmology` provides standard cosmological models (e.g. Planck18) with methods for
computing distances (luminosity distance, comoving distance, angular diameter distance),
lookback time, and age of the universe at a given redshift — use the built-in models rather
than re-deriving cosmological distance-redshift relations by hand, since the numerical
integration of the Friedmann equation has known subtleties (numerical precision near z=0,
choice of integration method) already handled correctly in the library.

## Catalog work (relevant for astronomy + ML/retrieval projects)
For cross-matching or spatial-query work over star/object catalogs:
- Use `astropy.coordinates.SkyCoord`'s built-in `match_to_catalog_sky` for nearest-neighbor
  matching in angular coordinates rather than naive Euclidean nearest-neighbor on RA/Dec,
  which is wrong near the poles and at the RA wraparound (0°/360°) — a very common silent
  bug in astronomy-adjacent data pipelines that don't use the library's spherical-aware
  matching.
- When building retrieval/search systems over astronomical catalogs (e.g. sky-region
  queries, nearest-star lookups), validate retrieval metrics (precision@k, MRR) against
  ground truth computed via proper spherical distance, not planar approximations, unless
  the region is small enough that the flat-sky approximation is explicitly justified and
  stated.

## Common pitfalls
- Comparing coordinates across frames without transforming first (ICRS vs. Galactic
  coordinates are not directly comparable).
- RA wraparound bugs in any custom nearest-neighbor or clustering code that doesn't use
  `SkyCoord`'s spherical-aware methods.
- Treating magnitudes as linear when computing averages/differences — magnitude is a
  logarithmic scale; combining fluxes (linear) and converting to magnitude at the end is
  usually correct, averaging magnitudes directly usually is not.

## Related skills
`physics-databases` for querying external astronomical databases/catalogs.
`physics-visualization` for sky-plot conventions (RA/Dec projections, all-sky maps).
