# Stage 2: spherical mesh descriptors

Stage 3 now consumes this stage's shortlist for [detailed geometric reranking](stage3.md).

Stage 2 extends the existing Stage 1 index with a geometric fingerprint and cached
candidate ranking. Stage 1 loading, metadata extraction, family rules, tables, and
coarse filtering are reused. Descriptor storage is additive; manufacturing
dimensions and the original STEP files are unchanged.

```text
Offline, once per source hash and configuration:
STEP → copied B-rep → triangle mesh → centroid/PCA/unit-radius normalization
     → equal-area rays → BVH intersections → 32 × 64 × 4 tensor → SQLite BLOB

Search:
new query → Stage 1 SQL filter → stored candidate descriptors → ranked top K
                                                           → future Stage 3
```

## Setup and quick start

The existing `environment.yml` now explicitly includes NumPy. The environment
name remains `cad-stage1` so existing setup commands continue to work.

```bash
conda env update -f environment.yml
conda activate cad-stage1

# Keep using Stage 1 to register new/changed historical sources.
python main.py index ./parts/ --db parts.sqlite3

# Offline preprocessing; repeat runs reuse complete descriptors.
python main.py build-descriptors --db parts.sqlite3 --verbose

# Debug a new part without writing the database.
python main.py descriptor example.step \
  --save-images ./descriptor-debug --save-tensor ./descriptor-debug/tensor.npy

# Load only this query STEP; all historical comparisons use stored tensors.
python main.py match example.step --db parts.sqlite3 --top-k 50 --tolerance 0.10
```

For a fresh installation, use `conda env create -f environment.yml` instead of
`env update`. The working interpreter used during development is also available
at `/tmp/cad-stage1-env/bin/python` for this session.

NumPy is the only new dependency. Mesh normalization, descriptor comparison,
SQLite storage, and the descriptor benchmark work without pythonOCC. For those
operations in a separate Python 3.11+ environment:

```bash
python -m pip install 'numpy>=1.26,<3'
```

Stage 1 metadata-only commands still work without NumPy or pythonOCC.

## Commands and configuration

All three new commands accept these generation settings. Use the **same settings
for historical descriptors and queries**:

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--theta-resolution` | 32 | Equal-width bins in `mu = cos(theta)` |
| `--phi-resolution` | 64 | Azimuth bins; must be even and at least 4 |
| `--count-scale` | 16 | Clip intersection count at this value and divide by it |
| `--linear-deflection-ratio` | 0.002 | Mesh chord tolerance divided by cube root of solid volume |
| `--angular-deflection` | 0.3 | OpenCascade angular mesh tolerance, radians |
| `--leaf-size` | 32 | Maximum triangles in a BVH leaf |

Smaller deflections create finer meshes; larger sampling grids capture more
angular detail. Both increase preprocessing cost. Choose settings using parts
representative of the archive and keep the full configuration fixed for a search.
Python's `DescriptorConfig` also exposes seam-welding, ray-hit, and PCA-degeneracy
tolerances. Every generation setting contributes to the configuration key.

`build-descriptors` accepts `--retry-failed`, `--rebuild`, and `--limit N`. A limit
caps new/retried generation attempts; cached descriptors are skipped. `--rebuild`
explicitly regenerates the selected configuration, including its completed rows.
`--verbose` prints per-part timing records to stderr. The JSON summary on stdout
reports discovered, built, skipped, failed, and previously failed counts.

`descriptor` reports vertex/triangle counts, the normalization transform, tensor
shape, vector length, per-channel min/max, intersection diagnostics, and timings.
It does not open a database. Optional PNGs are `firstDistance.png`,
`lastDistance.png`, `intersectionCount.png`, and `thickness.png`. They use a fixed
black=0/white=1 scale, with equal-mu rows and azimuth columns. Images are enlarged
4× for visibility. `--save-tensor` writes the normalized tensor as a plain `.npy`
file with pickling disabled; database storage includes the full version metadata.

`match` accepts Stage 1's relative `--tolerance` and optional `--volume-range` in
mm³. `--all-families` disables the family restriction. Its comparison options are
`--top-k`, `--weights FIRST LAST COUNT THICKNESS`, and `--metric euclidean|cosine`.
If any coarse candidate has a missing/stale/incompatible descriptor, the command
reports an error. `--skip-missing` explicitly allows omission and reports the IDs.
Descriptor generation never runs automatically for historical candidates during
search. Corrupt stored tensors always produce an error; `--rebuild` can repair them.

The original CLI exit conventions remain: 0 success, 1 batch with file failures,
2 setup/command errors, and 130 interruption. Native CAD diagnostics go to stderr
so stdout remains JSON.

## Python APIs

Generate a descriptor from STEP, a `TopoDS_Shape`, or an internal mesh:

```python
from src.models.sphereDescriptor import DescriptorConfig
from src.descriptors.descriptorPipeline import buildDescriptorFromStep, generateShapeDescriptor
from src.descriptors.sphereDescriptor import generateSphereDescriptor

config = DescriptorConfig(thetaResolution=32, phiResolution=64)
queryDescriptor = buildDescriptorFromStep("example.step", config)
# queryDescriptor = generateShapeDescriptor(existingShape, config)
# queryDescriptor = generateSphereDescriptor(existingMesh, config)

tensor = queryDescriptor.tensor          # float32, (32, 64, 4), read-only
vector = queryDescriptor.toVector()      # (8192,), interleaved channels, C order
print(queryDescriptor.summary())
```

The vector is an explicit extension point for a later PCA compressor or
autoencoder. Neither compression model is implemented. SQLite's lossless byte
compression does not change descriptor values.

Combine existing Stage 1 candidates with Stage 2:

```python
from src.indexing.database import PartDatabase
from src.indexing.candidateSearch import findGeometricMatches

with PartDatabase("parts.sqlite3", readOnly=True) as database:
    candidates = database.findCandidates(
        shapeFamily="block-like", dimensions=(10, 20, 30),
        dimensionTolerance=0.10, volumeRange=(5000, 7000),
    )

matches = findGeometricMatches(
    queryDescriptor, [part.partId for part in candidates], topK=50,
    databasePath="parts.sqlite3",
    weights=(1.0, 1.0, 0.5, 1.5),
)
for match in matches:
    print(match.partId, match.distance, match.axisFlip)
```

`findGeometricMatches` returns a list sorted by ascending distance, then `partId`
for deterministic ties. Duplicate candidate IDs are removed. An empty candidate
list returns no matches when a descriptor store exists. `topK` must be positive.
`skipMissing=True` permits missing descriptors explicitly. To inspect omitted IDs
and timing details, call `rankCandidates(queryDescriptor, ids, store, ...)`, which
returns a `MatchReport`; `store` is a `DescriptorStore` wrapping `PartDatabase`.

For an integrated query that loads the new STEP only once:

```python
from src.indexing.candidateSearch import findMatchesForStep

report = findMatchesForStep(
    "example.step", databasePath="parts.sqlite3", config=config,
    dimensionTolerance=0.10, topK=50,
)
print(report.toDict())
```

The result contains candidate IDs and raw distances. Stage 3 can retrieve
`filePath` and `sourceHash` from Stage 1 using the IDs; it need not depend on mesh
normalization or descriptor generation.

## Mesh extraction and normalization

`Mesh` contains `vertices: (N, 3)`, `triangles: (M, 3)`, and optional
`triangleBodyIds: (M,)`. Indices are zero based. Arrays are copied on construction
and exposed read-only. Body IDs distinguish multiple solids without Boolean
operations.

The adapter reuses Stage 1 solid validation, excludes loose surface/wire geometry,
and uses `BRepBuilderAPI_Copy(shape, True, False)` before meshing. Copying geometry
and omitting old triangulation avoids changing or reusing the caller's B-rep mesh.
`BRepMesh_IncrementalMesh` runs serially. Its absolute deflection is
`linearDeflectionRatio × cbrt(volume)`, providing a characteristic length that
scales uniformly and is independent of the part's world orientation.

For each face, `BRep_Tool.Triangulation` provides nodes and triangles. Nodes are
transformed with the returned `TopLoc_Location`; faces with `TopAbs_REVERSED`
have their triangle winding reversed. Duplicate face-seam vertices are welded
using a relative coordinate grid. Collapsed triangles and unreferenced vertices
are removed. Normalization rejects a mesh unless each body's edges have two
oppositely oriented incident triangles and its volume is positive.

Normalization is entirely in the mesh layer:

1. Integrate signed tetrahedron volumes to compute the **mesh volume centroid**.
   This approximates the physical solid centroid; it is not the mean of all
   surface vertices. Integration uses translated/scaled coordinates for stability.
2. Compute equal-weight vertex covariance with `numpy.linalg.eigh`, order
   eigenvectors by descending eigenvalue, and choose signs from odd projection
   moments with a deterministic farthest-vertex fallback. Force a right-handed
   basis, then rotate the centered coordinates into it.
3. Divide by the maximum vertex radius, putting the mesh inside the unit sphere.
   Save the centroid, basis, eigenvalues, and original radius scale. The inverse is
   `world = (normalized * radiusScale) @ pcaAxes.T + centroid`.

All Stage 1 dimensions, units, volume, and centroid records remain intact.
Different tessellation densities can change vertex PCA and the approximate mesh
centroid. Nearly repeated eigenvalues are flagged in `ambiguousAxisPairs`.
Four proper axis-flip variants are checked during comparison. They resolve sign
ambiguity; **they do not guarantee rotation invariance within a repeated or nearly
repeated eigenspace**. Arbitrary axis permutations or continuous alignment are
not performed. Symmetric parts, similar PCA eigenvalues, and retessellated models
need accuracy evaluation before aggressively reducing top K.

Primary API references: [shape copying](https://dev.opencascade.org/doc/refman/html/class_b_rep_builder_a_p_i___copy.html),
[pythonOCC meshing wrapper](https://github.com/tpaviot/pythonocc-core/blob/7.9.3/src/SWIG_files/wrapper/BRepMesh.i),
and [triangulation access](https://github.com/tpaviot/pythonocc-core/blob/7.9.3/src/SWIG_files/wrapper/BRep.i).

## Sampling, crossings, and channels

For row `i` and column `j`, sample the centers of equal-area cells:

```text
mu[i]  = 1 - 2 * (i + 0.5) / thetaResolution
phi[j] = 2*pi * (j + 0.5) / phiResolution
d      = (sqrt(1-mu²)*cos(phi), sqrt(1-mu²)*sin(phi), mu)
```

Every direction has equal spherical area weight. There are no duplicated poles.
Rays begin at the normalized origin and use `r(t) = t*d`, `t >= 0`.

A median-split BVH encloses triangles in axis-aligned boxes. Ray/box slab tests
visit only intersected branches; Möller–Trumbore tests are vectorized over the
triangles in visited leaves. Triangle pairs are never compared with one another.
The implementation retains all forward intersections rather than stopping at the
nearest one.

Coincident hits along triangle edges/vertices are grouped by ray distance within
`rayTolerance` (default `1e-7` in normalized units). Each body contributes one
crossing at a grouped distance. Opposite normal directions on the same body are
treated as a tangent contact and excluded. Coplanar ray/triangle hits are ignored.
Distances at the origin are retained as zero; a hit there is distinguished from a
miss by its nonzero count. Features closer than the tolerance can be merged.

| Channel index | Name | Stored value |
| --- | --- | --- |
| 0 | `firstDistance` | Nearest distinct forward crossing, normalized radius |
| 1 | `lastDistance` | Farthest distinct forward crossing, normalized radius |
| 2 | `intersectionCount` | `min(number of distinct crossing distances, countScale) / countScale` |
| 3 | `thickness` | Total material length along the positive half-ray, normalized radius |

**A ray with no crossings has exactly `[0, 0, 0, 0]`.** All stored channels are
float32 in `[0, 1]`. Counts are clipped, with `maxRawIntersectionCount` and
`countClippedRayCount` reported for diagnosis. Increase `countScale` and rebuild
if saturation removes useful cavity detail.

Thickness depends on whether the origin is inside material. For an outside origin,
alternating crossings give `(t2 - t1) + (t4 - t3) + ...`. For an inside origin,
the initial segment contributes: `t1 + (t3 - t2) + ...`. A centered solid box thus
has one exit and positive thickness; a ring's origin can lie in its hole.
Outward triangle normals and the known outside state at infinity determine
occupancy. Invalid occupancy sequences produce an error instead of a fabricated
thickness. For overlapping bodies, thickness measures the union of occupied ray
intervals; crossing count still describes the component surface crossings.

The four proper sign flips preserve handedness. Odd sign flips are excluded so
asymmetric mirror images are not automatically treated as equivalent. Like any
finite radial fingerprint, this representation can miss small or unsampled
features; Stage 3 remains responsible for detailed verification.

## Similarity metrics

The default weighted Euclidean metric is:

```text
distance = sqrt(mean_over_rays(sum_c(w[c] * (query[c] - candidate[c])²) / sum(w)))
weights  = (1.0, 1.0, 0.5, 1.5)  # first, last, count, thickness
```

Weights must be finite and nonnegative, with at least one positive value. The
score is a raw weighted RMS distance, not a calibrated similarity percentage.
The minimum distance over four proper query-axis flips is returned.

`cosineSimilarity(first, second)` compares the flattened vectors and returns raw
cosine similarity, maximizing over the same flips. The ranking API's cosine mode
returns `distance = 1 - cosineSimilarity` and includes the cosine value separately,
so ascending distance always means a better match. Cosine is unweighted. Two
all-zero vectors have cosine 1; one zero vector and one nonzero vector have 0.
Metric and weights can change without rebuilding historical descriptors.

Comparison is vectorized in batches of at most 128 descriptors by default, reduced
for dense grids to keep input tensors near 16 MiB per batch. No CAD modules or
source file operations are needed by descriptor lookup/ranking.

## Storage, versions, and recovery

`sphereDescriptors` and `descriptorSchemaInfo` are independent additive tables in
the existing SQLite file. Stage 1's schema and `PRAGMA user_version = 1` remain
unchanged. Each descriptor is keyed by `(partId, configKey)` and stores:

- The source SHA-256, descriptor and normalization versions, grid resolutions,
  channel count, and complete JSON generation configuration.
- A losslessly zlib-compressed, little-endian float32 tensor and its checksum.
  The default uncompressed tensor occupies 32 KiB. No pickle is used.
- Mesh counts, normalization transform, diagnostics, per-stage timing metadata,
  status, error text, and update time.

The configuration key hashes every generation setting. Different resolutions
and settings coexist. Algorithm changes require a descriptor/normalization version
bump, including semantic changes caused by changing the meshing implementation.
The dependency environment pins pythonOCC; recorded runtime version strings aid
diagnosis. Reproducibility is numerical within a fixed dependency environment,
not a promise of identical floating-point bits across all CAD kernels/platforms.

Offline preprocessing only visits currently indexed Stage 1 parts. Complete
records with matching source hashes/configuration are skipped without reopening
or stat-ing STEP files. New or changed records are marked `processing`, checked
against the Stage 1 source hash, generated, then atomically marked `complete`.
File-stat changes during generation prevent publication. If source bytes no
longer match Stage 1, run `index` before retrying the descriptor build.

Failures are remembered for the specific indexed source/configuration and skipped
on later runs unless `--retry-failed` is supplied. Interrupted `processing` rows
are retried. Search joins descriptor source hashes to current Stage 1 records and
checks ingestion status, so outdated or failed sources are excluded. Keep the
same single-ingestion-worker convention as Stage 1. Native process crashes still
require restarting the batch; committed parts are retained.

Stage 1 determines archive freshness. A changed file that has not been re-indexed
cannot be detected by a metadata-only search. Before Stage 3 loads an exact B-rep,
check its retained source hash against the file.

## Timing and benchmarks

Each successful descriptor stores `stepLoadSeconds`, `triangulationSeconds`,
`normalizationSeconds`, `bvhBuildSeconds`, and `descriptorGenerationSeconds`.
Hashing and total duration are reported separately. Triangulation timing includes
solid validation, copying, meshing, and array extraction. The first load can
include native module import costs. Debug output also reports BVH node count and
the tested triangle count versus a hypothetical full ray/triangle scan.

At the default 32 × 64 grid, six sample STEP solids in this workspace took about
0.25–1.36 seconds each to generate, including loading. A repeat build skipped all
six in under a millisecond inside the build loop. These simple solids are not a
substitute for timing complex production models.

```bash
python -m examples.benchmarkDescriptors
python -m examples.benchmarkDescriptors --counts 100 1000 10000 --metric cosine
```

The benchmark creates a temporary database of synthetic analytic box fingerprints
and performs real stored-descriptor queries. It measures storage/decompression,
numeric comparisons, and total ranking time separately; seeding and descriptor
fixture generation are excluded. Four proper axis variants and all 8,192 values
are compared. It does not measure real-archive retrieval accuracy.

One local validation run with Python 3.12, NumPy 2.5.3, default resolution, and
weighted Euclidean distance produced:

| Stored candidates | Storage/decode | Numeric comparison | Total ranking | Total per candidate |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 9.9 ms | 11.2 ms | 21.1 ms | 211 µs |
| 1,000 | 94.8 ms | 102.4 ms | 198.3 ms | 198 µs |
| 10,000 | 962.2 ms | 995.4 ms | 1,966.9 ms | 197 µs |

These are single local observations, affected by cache state and hardware, not
latency guarantees. The [recorded JSON](stage2-benchmark.json) preserves full
precision and configuration. The command can be rerun with different grids to
compare accuracy and speed on the same fixtures.

## Added modules and tests

```text
src/cad/meshExtractor.py             Copy and triangulate solids
src/models/mesh.py                   Mesh record and closure checks
src/models/sphereDescriptor.py       Config, tensor, metadata, vector interface
src/geometry/normalizer.py           Mesh centroid, PCA, radius normalization
src/geometry/sphericalSampler.py     Equal-area rays and exact flip mappings
src/geometry/bvh.py                  Triangle acceleration structure
src/geometry/rayCaster.py            All crossings and occupied intervals
src/descriptors/sphereDescriptor.py  Tensor generation from a mesh
src/descriptors/descriptorPipeline.py STEP/shape adapters and timings
src/descriptors/descriptorSimilarity.py Metrics and vectorized comparison
src/descriptors/descriptorStore.py   Additive SQLite persistence and validation
src/descriptors/descriptorBuilder.py Offline preprocessing and recovery
src/descriptors/descriptorDebug.py   Optional channel PNGs
src/descriptors/schema.sql          Stage 2 tables
src/indexing/candidateSearch.py      Candidate ranking and Stage 1 integration
src/descriptorCli.py                New CLI commands
examples/benchmarkDescriptors.py    Stored-descriptor timing demonstration
tests/testSphericalDescriptor.py    Analytic geometry and metric regressions
tests/testDescriptorStore.py        Persistence and real STEP integration
```

```bash
python -m unittest discover -s tests -p 'test*.py' -v
```

All 52 tests passed in the supplied pythonOCC environment, including the 24
existing Stage 1 tests. New cases cover normalization transforms, distinct-axis
rotation/scale invariance, PCA ambiguity reporting, equal-area sampling, axis
flips, inside/outside/cavity origins, duplicate hits, tangencies, overlapping
bodies, missing rays, count scaling, BVH pruning, mesh-copy preservation, hash
invalidation, incremental builds, configuration isolation, storage corruption,
ranking, and debug exports.

Detailed B-rep verification, manufacturing compatibility, neural training, and
neural descriptor compression remain future stages.
