# CAD similarity search — Stage 1

**Stage 2 is now implemented:** see [spherical descriptors and geometric ranking](docs/stage2.md)
for offline preprocessing, APIs, CLI commands, debug images, and benchmarks.
**Stage 3 is now implemented:** see [detailed geometry reranking](docs/stage3.md)
for cached meshes/face properties, alignment, surface comparison, CLI usage, and
limitations. This baseline requires no trained neural checkpoint.
This page documents the existing Stage 1 foundation.

Stage 1 reads historical STEP files once, stores numeric geometry and a
deterministic category in SQLite, and filters candidates using SQL. It does not
compute spherical descriptors, embeddings, or pairwise B-rep similarity.

```text
Historical STEP files
        │ first ingestion / changed files only
        ▼
OpenCascade load → solid validation → geometry extraction → family rules
        │
        ▼
SQLite metadata and indexes
        │ family + sorted size ranges + optional volume range
        ▼
Small candidate set, with partId + original filePath + sourceHash
        │
        └─► Stage 2 descriptors → future detailed B-rep comparison
```

## Setup

Use Conda or micromamba. The supplied environment pins Python 3.12 and
`pythonocc-core` 7.9.3 with the `novtk` variant; the CLI needs no graphical viewer.
SQLite, the CLI, hashing, and the test runner use Python's standard library.
Install pythonOCC through Conda, not `pip install pythonocc-core`. See the
[upstream installation instructions](https://github.com/tpaviot/pythonocc-core#install-with-conda)
and [conda-forge package recipe](https://github.com/conda-forge/pythonocc-core-feedstock/blob/main/recipe/meta.yaml).

From this project directory:

```bash
conda env create -f environment.yml
conda activate cad-stage1
python main.py --help
```

Alternatively:

```bash
micromamba create -f environment.yml
micromamba run -n cad-stage1 python main.py --help
```

Metadata queries and classification work without pythonOCC. Geometry ingestion
and inspection report an actionable dependency error if pythonOCC is missing.
Use Python 3.11+ with SQLite 3.37+ for the standard-library-only functionality;
the supplied environment satisfies these requirements.

## CLI

```bash
# Optional: create six real STEP examples if you have no historical files yet.
python examples/createSampleParts.py ./parts

# Scan recursively; .step and .stp extensions are case insensitive.
python main.py index ./parts/
python main.py index ./parts/ --db ./data/parts.sqlite3 --verbose

# Print all extracted properties and the category as JSON; no database is opened.
python main.py inspect ./parts/block.step

# Filter SQLite records without loading STEP files.
python main.py find --db ./data/parts.sqlite3 \
  --family block-like --dimensions 30 10 20 --tolerance 0.10 \
  --volume-range 5000 7000

# Repeat runs skip unchanged files. Retry failures explicitly when appropriate.
python main.py index ./parts/ --db ./data/parts.sqlite3 --retry-failed

# Rehash every source, while still reusing geometry when the bytes are unchanged.
python main.py index ./parts/ --db ./data/parts.sqlite3 --verify-content
```

The default index is `parts.sqlite3` in the current directory. `index` also accepts
a single STEP file. Directory symlinks are not followed. File symlinks resolve to
one canonical path and are deduplicated during a scan. `find` opens the database
read-only and fails if it does not exist.

`index` prints counts for `discovered`, `indexed`, `skipped`, `reclassified`,
`failed`, and `skippedFailed`. Diagnostics go to stderr; stdout remains JSON,
including when the native STEP parser reports an error. Exit codes are 0 for
success, 1 for a batch containing new or cached file failures, 2 for command/setup
errors, and 130 for interruption. A corrupt file does not stop the rest of a batch.

For the generated 10 × 20 × 30 mm block, `inspect` reports volume 6000 mm³,
surface area 2200 mm², centroid `(5, 10, 15)`, 6 faces, 12 edges, 8 vertices,
6 planar faces, and `block-like`.

## Python API

```python
from src.indexing.database import findCandidates, PartDatabase
from src.indexing.partIndexer import inspectPart, indexPath

metadata = inspectPart("example.step")   # Does not write a database.
print(metadata.toDict())

with PartDatabase("parts.sqlite3") as database:
    summary = indexPath("./parts", database)
    candidates = database.findCandidates(
        shapeFamily="plate-like",
        dimensions=(100, 80, 5),
        dimensionTolerance=0.10,
        volumeRange=(30000, 45000),
    )

# Same API as a standalone function; additional options are keyword-only.
candidates = findCandidates(
    shapeFamily="plate-like",
    dimensions=(100, 80, 5),
    dimensionTolerance=0.10,
    volumeRange=None,
    databasePath="parts.sqlite3",
)
for candidate in candidates:
    print(candidate.partId, candidate.filePath, candidate.sourceHash)
```

The query sorts the three supplied dimensions. For each sorted query dimension
`d`, an indexed part must fall within `[d × (1 − tolerance), d × (1 + tolerance)]`.
`0.10` means **10%**, not 10 mm; the supported range is 0 to 1. Bounds are
inclusive, with one floating-point step of outward rounding for nonzero
tolerances. Zero tolerance means exact stored-value equality. `volumeRange` is
an optional inclusive `(minimum, maximum)` in mm³.

Pass `shapeFamily=None` to search all families. This can improve recall because
the initial rules are coarse. All matches are returned by default, ordered by
`partId`. Optional `limit` explicitly truncates that order; it is not a similarity
ranking. Non-finite values, zero/negative dimensions, invalid ranges, and unknown
categories are rejected. Queries read neither STEP contents nor filesystem stats.

## Project structure

```text
main.py                         Root CLI launcher
environment.yml                 Reproducible Python/pythonOCC versions
src/
    main.py                     Argument parsing and JSON output
    errors.py                   Expected application errors
    cad/
        stepLoader.py           STEP transfer and unit conversion
        geometryExtractor.py    Solid validation and geometric measurements
        surfaceClassifier.py    Analytic surface counts and ratios
        topology.py             Unique topology maps
    indexing/
        partClassifier.py       Pure numeric family rules
        partIndexer.py          Fingerprints and ingestion orchestration
        database.py             SQLite storage and candidate API
        schema.sql              Tables, constraints, and indexes
    models/
        partMetadata.py         GeometryProperties and PartMetadata records
examples/
    createSampleParts.py        Generates six real STEP solids
    benchmarkFiltering.py       Temporary 100,000-row metadata demonstration
tests/
    testDatabase.py             SQL filtering and persistence tests
    testGeometry.py             Real geometry, STEP, ingestion, and CLI tests
```

Modules, functions, variables, record fields, and SQL columns use lower camelCase.
Python classes use conventional PascalCase; Python special methods and external
OpenCascade APIs retain their required names.

## Stored properties and semantics

The full schema is [src/indexing/schema.sql](src/indexing/schema.sql).

| Fields | Meaning |
| --- | --- |
| `partId`, `filePath` | Stable UUID derived from the canonical absolute STEP path; exact source path retained |
| `sourceHash` | SHA-256 of the STEP bytes used for this metadata |
| `sizeX`, `sizeY`, `sizeZ` | World-coordinate axis-aligned bounding box lengths, mm |
| `sizeMin`, `sizeMid`, `sizeMax` | Those lengths sorted in ascending order, mm |
| `volume`, `surfaceArea` | Unit-density solid volume, mm³; unique face area, mm² |
| `centerOfMassX/Y/Z` | Volume-weighted centroid in the imported world coordinates, mm |
| `aspectXY`, `aspectXZ`, `aspectYZ` | `sizeX / sizeY`, `sizeX / sizeZ`, `sizeY / sizeZ` |
| `faceCount`, `edgeCount`, `vertexCount` | Unique topological entities, rather than repeated traversal occurrences |
| `solidCount` | Distinct solid occurrences, preserving assembly locations |
| `planarFaces`, `cylindricalFaces`, `conicalFaces`, `otherFaces` | Counts by underlying surface type |
| `planarRatio`, `cylindricalRatio`, `conicalRatio`, `otherRatio` | Corresponding counts divided by `faceCount`, not area fractions |
| `shapeFamily` | Deterministic coarse category |
| `units`, `extractorVersion`, `classifierVersion`, `extractedAt` | Unit convention and processing provenance |

All measurements refer to validated solid bodies. Loose faces and wires alongside
solids are excluded from every measurement. Surface-only or wire-only models,
empty shapes, invalid/open/unbounded solids, incorrect solid orientation, and
non-finite or nonpositive measurements are rejected. Every solid must pass;
invalid bodies are not silently omitted.

Multiple solids are treated as one historical file. Volume and centroid combine
the solid properties; the bounding box covers all solids. Shared topological
faces count once for surface area. Independent overlapping solids double-count
overlap volume and can include internal surfaces: this stage does not perform a
Boolean union or detect collisions. Multi-body records are `general/unknown`.

**Sorting dimensions handles axis permutations, including quarter-turn rotations.
It does not make an axis-aligned box invariant under arbitrary rotation.** A plate
exported at 45° can have different extents and receive a different category. This
is an explicit limitation of the cheap baseline; align exports consistently or
widen/disable the relevant filters. A future oriented-bounds strategy can be
introduced with an extractor version change.

## Initial category rules

Let `s <= m <= l` be the sorted dimensions and `r` the cylindrical plus conical
face ratio. The first matching rule wins:

| Priority | Category | Rule |
| --- | --- | --- |
| 1 | `general/unknown` | More than one solid |
| 2 | `shaft/turned-like` | `r >= 0.25`, `m/s <= 1.25`, and `l/m >= 3` |
| 3 | `cylindrical/ring-like` | Cylindrical ratio `>= 0.25`, and either `m/s <= 1.25` or `l/m <= 1.25` |
| 4 | `plate-like` | Planar ratio `>= 0.60` and `s/m <= 0.20` |
| 5 | `block-like` | Planar ratio `>= 0.75` and `l/s <= 3` |
| 6 | `general/unknown` | Everything else |

The cylindrical rule precedes the plate rule so a thin washer is treated as
cylindrical/ring-like. These rules are initial screening heuristics, not proof of
how a part was manufactured or whether it contains a hole. Face splitting,
fillets, many drilled holes, and spline-based exports can change the ratios.
Only [partClassifier.py](src/indexing/partClassifier.py) needs to change to revise
the rules; it imports neither SQLite nor OpenCascade.

## Incremental ingestion and failure recovery

1. Resolve the source path and check file size, modification time, and change time.
   If these and the extractor version match, reuse the stored record without
   opening the STEP file. Unchanged known failures are skipped too.
2. For a new/changed file, mark its ingestion status `processing`, hash it, and
   reuse previous geometry if the bytes and extractor version match.
3. Otherwise load and measure it. Check file stats again before publication so
   a file modified during processing is not accepted.
4. Save metadata and the `indexed` file state in one transaction. On failure,
   retain an error in `ingestionFiles`. Existing metadata remains available for
   diagnosis, but queries exclude failed or in-progress sources.

Successful files are committed individually. WAL mode lets readers keep using
the index during serial ingestion. Run one indexer per database; this version
does not coordinate multiple simultaneous ingestion workers. A terminated native
process leaves a `processing` record that will be retried on the next run. Native
process crashes are not recoverable as ordinary Python exceptions within the same
batch; process isolation can be added later if an archive requires it.

The fast skip path trusts filesystem metadata. `--verify-content` checks hashes
even if timestamps and sizes appear unchanged. Source files should remain stable
while being indexed. An extraction algorithm change requires incrementing
`extractorVersion`; a rule change requires incrementing `classifierVersion`.
The next indexing run reclassifies unchanged geometry from SQLite when only the
classifier version changes.

Identity is per canonical path. Moving a file creates a new part identity;
identical files at different paths are separate records. Files absent from a
later directory scan are not automatically removed. Keep the archive stable or
add an explicit removal policy before using it as a live synchronized catalog.
Before Stage 3, verify that the retained path still has the recorded `sourceHash`
so the compared B-rep is the version represented by the metadata.

## Filtering at scale

`partsByFamilyAndSize` indexes family and sorted dimensions, `partsBySize` supports
queries across families, and `partsByFamilyAndVolume` supports volume filtering.
SQL applies every supplied bound and joins file status to exclude invalidated
records. SQLite generally uses equality columns and the first range of a B-tree
index, then checks the remaining bounds; this is not a multidimensional spatial
index. Narrowness and speed depend on the data distribution and chosen tolerance.

Run a reproducible demonstration without pythonOCC:

```bash
python -m examples.benchmarkFiltering --parts 100000
```

It seeds a temporary SQLite database with synthetic box metadata, reports the
candidate count and query time, prints `EXPLAIN QUERY PLAN`, and removes the
database afterwards. It illustrates the 100,000-record filtering step; it is not
a performance guarantee for a manufacturer's archive. No STEP files are opened
in this demonstration.

The validation run on this workspace narrowed 100,000 synthetic records to 15
candidates in 0.421 ms, using `partsByFamilyAndSize`. Setup and seeding time are
excluded from that query timing.

## OpenCascade APIs used

| API | Role in this project |
| --- | --- |
| `STEPControl_Reader.ReadFile`, `IFSelect_RetDone` | Parse STEP and check the read status |
| `SetSystemLengthUnit(1.0)` | Request millimeter output after reading, before transferring; uses the units declared in STEP |
| `NbRootsForTransfer`, `TransferRoots`, `OneShape` | Transfer all roots into one `TopoDS_Shape`; reject incomplete or empty transfer |
| `BRepCheck_Analyzer(..., True).IsValid()` | Check solid topology and associated geometry |
| `BRepCheck_Shell.Closed()` | Check shell closure, rather than trusting a shape's cached `Closed` flag |
| `BRepClass3d_SolidClassifier.PerformInfinitePoint()` | Require infinity to be outside each finite, outward-oriented body |
| `topexp.MapShapes`, `TopTools_IndexedMapOfShape` | Deduplicate topology by identity and location; `FindKey` is 1-based and `Size()` gives the count in the pinned wrapper |
| `topods.Face/Solid/Shell` | Cast generic shape handles to the specialized topology type an API requires |
| `BRep_Builder`, `TopoDS_Compound` | Group validated bodies without modifying/fusing their geometry |
| `Bnd_Box`, `brepbndlib.AddOptimal(..., False, False)` | Compute a precise axis-aligned bounding box from the B-rep without triangulation or shape-tolerance enlargement |
| `brepgprop.VolumeProperties`, `GProp_GProps` | Integrate solid properties; with unit density, `Mass()` is volume and `CentreOfMass()` is the volume centroid |
| `GProp_GProps.Add` | Combine mass properties of validated bodies |
| `brepgprop.SurfaceProperties` | Integrate face area, using underlying surfaces and counting shared faces once |
| `BRepAdaptor_Surface.GetType`, `GeomAbs_*` | Read plane/cylinder/cone types; everything else remains other |

These are per-shape measurements, not shape-to-shape comparisons. The pipeline
does not generate a mesh. Bounding and property calculations are numerical OCCT
operations, not symbolic exact arithmetic. It relies on correctly declared STEP
units; it cannot infer or repair a file authored with incorrect unit metadata.

Primary references: [pythonOCC STEP wrapper](https://github.com/tpaviot/pythonocc-core/blob/7.9.3/src/SWIG_files/wrapper/STEPControl.i),
[bounding API](https://dev.opencascade.org/doc/refman/html/class_b_rep_bnd_lib.html),
[property integration API](https://github.com/tpaviot/pythonocc-core/blob/7.9.3/src/SWIG_files/wrapper/BRepGProp.i),
[BRep validation](https://dev.opencascade.org/doc/refman/html/class_b_rep_check___analyzer.html),
and [solid classification](https://dev.opencascade.org/doc/refman/html/class_b_rep_class3d___solid_classifier.html).

## Extending later stages

Stage 2 now adds `sphereDescriptors` referencing `parts.partId`; see the
[Stage 2 storage contract](docs/stage2.md#storage-versions-and-recovery).
For further stages, add independent tables with `sourceHash` and an artifact
version. Suggested future records are
`partEmbeddings(partId, sourceHash, modelVersion, ...)`,
`manufacturingMetadata(partId, ...)`, and
`brepSimilarityResults(leftPartId, rightPartId, leftSourceHash, rightSourceHash, algorithmVersion, ...)`.
These tables and algorithms are intentionally not implemented yet.

Ingestion uses an upsert, not SQLite `REPLACE`, so updating source metadata keeps
the part identity and future foreign keys intact. Future stages must reject cached
artifacts with mismatched hashes. Geometry extraction, family rules, and database
storage have independent entry points. `PRAGMA user_version` tracks the SQLite
schema; unsupported schema versions fail explicitly instead of being overwritten.

## Tests

```bash
python -m unittest discover -s tests -p 'test*.py' -v
```

The suite checks known volumes/areas/centroids, unique topology, cylinders/rings/
cones/spheres, multi-body behavior, invalid bodies, STEP unit conversion, repeat
ingestion, changed/corrupt files, category upgrades, source mutation, CLI JSON,
candidate boundaries, volume ranges, schema handling, and index selection.
With NumPy installed but without pythonOCC, the CAD integration tests are explicitly
skipped. The complete suite now includes Stage 2 tests; use the Conda environment
for complete validation. The original SQLite-only tests can still run with the
standard library alone: `python -m unittest tests.testDatabase -v`.
