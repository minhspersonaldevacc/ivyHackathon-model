# Stage 3: deterministic detailed geometry reranking

Stage 3 adds alignment, surface comparison, and local B-rep face matching to the
existing project. It requires no trained checkpoint. It is an initial geometric
baseline that can later be evaluated against a learned encoder such as UV-Net.

```text
Historical STEP, offline:
Stage 1 metadata → Stage 2 sphere cache → Stage 3 mesh/samples/face cache

New query, online:
Stage 1 SQL filtering → Stage 2 sphere ranking → top K (default 50)
    → Stage 3 cached geometry comparisons → top N (default 5)
```

Stage 1/2 algorithms and schemas are reused. Stage 3 consumes their shortlist;
it does not retrieve additional parts from the library or combine their scores.
The original STEP remains available for later exact verification.

## Setup and CLI

SciPy is the only added dependency, for `cKDTree` and linear assignment. Update
the existing environment, or use `conda env create` for a fresh installation:

```bash
conda env update -f environment.yml
conda activate cad-stage1

python main.py index ./parts --db ./data/parts.sqlite3
python main.py build-descriptors --db ./data/parts.sqlite3
python main.py build-detailed-geometry --db ./data/parts.sqlite3 --verbose

python main.py search-detailed query.step --db ./data/parts.sqlite3 \
  --stage2-top-k 50 --top-k 5

# Development commands do not open or write a database.
python main.py detailed-geometry example.step
python main.py compare-detailed example.step modified.step
```

Commands use the project's JSON stdout convention. Native OCCT diagnostics and
per-part progress are sent to stderr. Exit codes retain the existing meanings:
0 success, 1 bulk failures, 2 setup/query errors, 130 interruption.

Search reports Stage 1/2/3 counts, candidate paths and IDs, the original Stage 2
`distance`, `axisFlip`, and `cosineSimilarity`, and a new `detailedDistance`.
`sphereDistance` is an alias of `distance`. Every match includes component
measurements, an alignment transform, and timings. Lower detailed distance ranks
first; ties use `partId`. Scores are raw measurements, with no percentage claim.

For broader recall, explicitly select `--all-families` and a suitable dimension
tolerance. All Stage 2 sampling/meshing arguments are available on
`search-detailed` and must match the offline sphere cache configuration.

## Offline artifacts and recovery

Each `DetailedGeometry` contains:

- a copied, triangulated, normalized mesh with body IDs;
- 1,024 area samples by default, plus up to eight exact surface samples per face;
- face surface type, area, area centroid, normal/axis, radius/angle where supported,
  and face adjacency lists;
- the original-to-canonical transform and radius in millimeters;
- source hash, complete preprocessing configuration, version, coverage and timings.

Meshing and centroid/PCA/unit-radius normalization call the existing Stage 2
functions. Stage 3 uses denser default meshing (linear deflection ratio 0.001 and
angular deflection 0.2 radians) and never attaches tessellation to the original
`TopoDS_Shape`. Area samples use deterministic stratification by triangle area
and a low-discrepancy barycentric sequence. Additional trimmed-face samples
provide coverage for small holes/pockets that area sampling may miss. They are
intentionally included in the final sample metrics, so those metrics are not a
pure surface-area integral. Face coverage is reported in `diagnostics`.

The additive `detailedGeometries` table stores compressed NumPy arrays and JSON
face/provenance metadata. Arrays are loaded with `allow_pickle=False` and checked
against a SHA-256 checksum. Compressed/uncompressed size is limited to 256 MiB per
artifact. `detailedSchemaInfo` versions this table independently; Stage 1's
`PRAGMA user_version` and the Stage 2 schema version remain unchanged.

The cache lives in the existing `--db` SQLite file. Its key is `(partId,
configKey)`; the config includes sampling, meshing and algorithm versions. A
search also requires the cached `sourceHash` to match active Stage 1 metadata.
Different preprocessing configurations coexist. Change `geometryVersion` when
changing extraction/sampling semantics and `normalizationVersion` when changing
normalization; update the supported versions together with that implementation.

Repeat builds skip complete, unchanged artifacts without statting/reopening their
STEP files. Failed entries are remembered; `--retry-failed` retries them,
`--rebuild` explicitly regenerates this configuration, and `--limit` caps new
attempts. An interrupted `processing` entry is retried. Successful entries commit
individually; a corrupt/missing/unsupported source is recorded and the batch
continues. As in Stage 1/2, a native OCCT process crash cannot be recovered as a
Python exception; process isolation is a possible future addition.

Reindex changed historical sources before rebuilding caches. Search trusts the
active index hashes and reads no historical STEP files or filesystem stats. A
file changed on disk without being reindexed can therefore leave an old cached
version searchable, as in Stage 2. Missing/stale caches cause an actionable error;
`--skip-missing` explicitly omits and reports IDs. Corrupt cache blobs always
raise; use `--rebuild` to recover them. Run one serial preprocessing worker per
database, following the existing ingestion convention.

## Alignment, comparison, and score

1. Try all 24 proper signed axis permutations of the PCA basis. Reflections are
   excluded. Select the four best sampled-surface starts by default.
2. Refine each selected start using bidirectional point-to-point ICP, with up to
   20 iterations. The best sampled-surface alignment is retained. A 90% trimmed
   correspondence set reduces the effect of modified features during alignment.
   **Final comparison uses all samples without trimming.**
3. Measure query samples against candidate triangles and candidate samples against
   query triangles. Closest points use a branch-and-bound traversal of the existing
   triangle BVH; a point is projected onto triangle interiors or clamped edges.
   This avoids a full triangle scan for each sample.
4. Match faces by a minimum-cost one-to-one assignment, with explicit unmatched
   penalties. Compare surface type, position, area, direction, radius, cone angle
   and adjacency degree. Type mismatch/unmatched cost is 1. Matching face costs
   use weights 0.35, 0.20, 0.15, 0.15, 0.05, 0.10 for those numeric components.

The default ranking distance is:

```text
(0.5 × bidirectionalMeanSurfaceDistance
 + 0.3 × bidirectionalP95SurfaceDistance
 + 0.2 × faceDistance) / (0.5 + 0.3 + 0.2)
```

The surface mean and P95 each average the corresponding statistic from the two
directions. Surface errors use the query radius as the unit. Face costs are
bounded in `[0, 1]`. These are initial heuristic weights, not learned or
calibrated values; configure `--detailed-weights MEAN P95 FACE` and validate on
your parts. Stage 2 scores are retained only for diagnostics.

`--size-mode physical` is the default: meshes retain their relative real sizes,
and `surfaceMeanMm`/`surfaceP95Mm` report physical surface errors. Alignment fits
rotation and translation, never scale. `--size-mode shape` compares unit-radius
geometry and returns `null` for physical millimeter metrics. This option only
affects Stage 3: Stage 1 still applies its dimension/volume filters, so proportional
parts can be excluded before shape-mode reranking.

The alignment convention is:

```text
q = (queryWorld - queryCentroid) @ queryPcaAxes / queryRadius
c = (candidateWorld - candidateCentroid) @ candidatePcaAxes / queryRadius
c ≈ q @ rotation + translation                 # physical mode
```

In shape mode, each part instead uses its own radius. The stored `candidateScale`
reports the radius ratio applied to the candidate's cached unit-radius geometry.
Configure alignment with `--alignment-starts`, `--icp-iterations`, and
`--trim-fraction`. ICP is a local optimizer; the 24 starts are not a guarantee of
correct alignment under arbitrary symmetry or feature changes. See the
[ICP reference](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
for the role of initialization. This project implements rigid point-to-point ICP
with NumPy/SciPy and does not require Open3D.

## Python integration

```python
from src.detailedSimilarity.detailedSearch import (
    findDetailedMatchesForStep, rerankDetailedForStep,
)
from src.models.detailedGeometry import DetailedSearchConfig

report = findDetailedMatchesForStep(
    "query.step", databasePath="parts.sqlite3",
    stage2TopK=50, topK=5, allFamilies=True,
    searchConfig=DetailedSearchConfig(sizeMode="physical"),
)
for match in report.matches:
    print(match.partId, match.sphereDistance, match.detailedDistance,
          match.measurements["surfaceMeanMm"])

# Or consume an existing Stage 2 shortlist directly:
report = rerankDetailedForStep(
    "query.step", stage2Results, topK=5, databasePath="parts.sqlite3",
)
```

`DetailedGeometricMatch` subclasses the existing `GeometricMatch` and preserves
all its fields. `rankDetailedCandidates(queryGeometry, stage2Results, store, ...)`
provides a STEP-independent entry point for reusable preprocessed queries.
Historical artifacts are read one at a time to bound memory; the query BVH is
reused across comparisons. Empty shortlists return empty reports without loading
the query or opening the Stage 3 cache. Nonempty searches generate the query's
Stage 3 artifact exactly once. The current composition loads the query once for
Stage 1/2 and once for Stage 3; it verifies that source hashes agree across both
passes. There is no historical re-encoding during searches.

## Debugging and benchmarks

```bash
python -m examples.createDetailedParts ./demo-parts
python main.py index ./demo-parts --db demo.sqlite3
python main.py build-descriptors --db demo.sqlite3
python main.py build-detailed-geometry --db demo.sqlite3
python main.py search-detailed ./demo-parts/block.step --db demo.sqlite3 \
  --all-families --tolerance 1 --stage2-top-k 6 --top-k 5
python -m examples.benchmarkDetailed ./demo-parts/block.step --db demo.sqlite3 \
  --stage2-top-k 6 --library-size 100000
```

The six fixtures include a block, a rotated/translated copy, an extra hole, two
pocket depths, and a cylinder. A wide tolerance in this demonstration keeps the
unrelated cylinder available for comparison; it is not a recommended universal
production tolerance. Preprocessing reports STEP load, triangulation,
normalization, sampling, and face extraction. Search reports cache lookup,
alignment, BVH construction, surface comparison, face comparison and total times.
The benchmark extrapolates measured comparison cost to 100,000 parts without
performing that exhaustive run. Estimates exclude exhaustive cache reads and
assume the same geometry complexity; they are not performance guarantees.

The workspace validation indexed and cached all six fixtures. Stage 3 offline
preprocessing took 0.254 seconds total; the repeat build skipped all six in
0.00050 seconds. A default-resolution search with all six shortlisted took
2.57 seconds in Stage 3, including 0.0057 seconds of cached lookup and 2.53 seconds
of comparison. It returned the identical block first, its transformed copy second,
then the hole and pocket variants, with the unrelated cylinder outside the top
five. The transformed copy had mean sampled surface error 0.030 mm; this records
observed alignment behavior rather than promising exact rotation invariance.

The benchmark's narrower dimension gate retained five fixtures and measured
0.343 seconds per comparison. Holding that small-fixture cost constant gives
rough estimates of 17 seconds for 50 comparisons versus 9.5 hours for 100,000.
Only five comparisons were actually measured; the large-library figure is an
extrapolation. Curved/complex models, cold caches and larger sample counts can
change these costs substantially.

## What the baseline can establish

Tests check known point-to-triangle distances, repeatability, proper rotations,
physical/shape scale policies, face penalties, cache invalidation/recovery,
preservation of Stage 2 fields, and an end-to-end search with historical STEP
files removed. Real fixtures check near-zero identical block distance, alignment
of a rotated/translated block, and an extra-hole block ranking ahead of an
unrelated cylinder. They use broad numerical tolerances and ranking assertions.

This is **sampled mesh geometry plus analytic face measurements**, not exact
B-rep equivalence or full topology matching. Curved-face anchors are sampled on
the exact B-rep but compared against tessellated surfaces, creating a meshing
error floor even for identical curved models. Thin features can still be missed;
P95 and the reported maximum cover sampled locations only and are not exact
Hausdorff distance. Narrow trimmed faces can exhaust sampling attempts; inspect
`faceSampleCounts` and `facesWithoutSamples`.

Face splitting/merging across exports can change face assignment and adjacency
degree despite equal geometry. Cylinders/cones/spheres use basic analytic
attributes; torus matching currently uses its major radius, and general spline
faces rely on area/position and sampled surface geometry. Adjacency lists are
cached, but only degree contributes to this first comparator. Independent
overlapping bodies retain the Stage 1/2 overlap semantics; no Boolean union is
performed. Assignment has quadratic memory growth with face count, so large
assemblies need a different matching strategy.

Stage 1's strict family/dimension gates and Stage 2's PCA ambiguity can still
exclude relevant parts. Stage 3 cannot recover them. Measure Stage 2 recall at K
and final ranking quality separately on labeled archive examples before claiming
general retrieval accuracy. No manufacturing, tolerance, material, or CAM scores
are inferred.

## OpenCascade APIs

The loader, solid validation, topology maps and meshing are reused from Stages
1/2. New face preprocessing uses:

| API | Purpose |
| --- | --- |
| `BRepAdaptor_Surface(face, True)` | Evaluate the underlying surface in the face's placement and read analytic surface types |
| `Plane/Cylinder/Cone/Sphere/Torus`, `Axis`, `Radius`, `SemiAngle` | Extract supported local analytic attributes |
| `brepgprop.SurfaceProperties`, `GProp_GProps.CentreOfMass` | Compute exact-surface area and area-weighted face centroid |
| `breptools.UVBounds` | Bound a face's UV parameter region |
| `BRepClass_FaceClassifier` with `gp_Pnt2d` | Reject UV samples outside the trimmed face, including holes |
| `BRepAdaptor_Surface.Value` | Evaluate accepted surface samples in world coordinates |
| `TopAbs_REVERSED` | Orient planar normals consistently with outward face orientation |

References: [surface adaptor](https://dev.opencascade.org/doc/refman/html/class_b_rep_adaptor___surface.html)
and [face classifier](https://dev.opencascade.org/doc/refman/html/class_b_rep_class___face_classifier.html).

## Files added and changed

Added:

```text
src/models/detailedGeometry.py
src/detailedCli.py
src/detailedSimilarity/
    __init__.py
    surfaceSampling.py
    faceExtractor.py
    geometryPipeline.py
    surfaceDistance.py
    alignment.py
    faceSimilarity.py
    geometrySimilarity.py
    geometryStore.py
    geometryBuilder.py
    detailedSearch.py
    schema.sql
tests/testDetailedSimilarity.py
examples/createDetailedParts.py
examples/benchmarkDetailed.py
docs/stage3.md
```

Updated `src/main.py` (CLI registration/dispatch), `src/errors.py` (Stage 3 error
type), `environment.yml` (SciPy), and README/Stage 2 documentation links. Existing
Stage 1/2 CAD extraction, classification, storage and ranking modules are unchanged.
