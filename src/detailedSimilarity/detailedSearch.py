"""Rerank Stage 2 matches using cached detailed geometry, never archive STEP files."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from src.detailedSimilarity.geometryPipeline import buildDetailedGeometryFromStep
from src.detailedSimilarity.geometrySimilarity import compareDetailedGeometry
from src.detailedSimilarity.geometryStore import GeometryStore
from src.detailedSimilarity.surfaceDistance import SurfaceDistanceIndex
from src.errors import DetailedSimilarityError
from src.indexing.candidateSearch import GeometricMatch, findMatchesForStep
from src.indexing.database import PartDatabase
from src.indexing.partIndexer import fileFingerprint, hashFile
from src.models.detailedGeometry import DetailedGeometryConfig, DetailedSearchConfig


@dataclass(frozen=True)
class DetailedGeometricMatch(GeometricMatch):
    filePath: str | None = None
    detailedDistance: float = 0.0
    measurements: dict = field(default_factory=dict)

    @property
    def sphereDistance(self):
        return self.distance


@dataclass
class DetailedMatchReport:
    matches: list[DetailedGeometricMatch]
    requestedCandidates: int
    comparedCandidates: int
    unavailablePartIds: list[str]
    timings: dict
    searchConfig: dict
    stage1Candidates: int | None = None
    stage2Candidates: int = 0
    stage3Candidates: int = 0
    stage2Report: dict = field(default_factory=dict)
    queryMetadata: dict = field(default_factory=dict)

    def toDict(self):
        result = asdict(self)
        for match in result["matches"]:
            match["sphereDistance"] = match["distance"]
        return result


def rankDetailedCandidates(queryGeometry, stage2Results, store, topK=5, *, config=None, skipMissing=False):
    """Stage 3 consumes an explicit Stage 2 shortlist; no library-wide retrieval.

    Scores contain no Stage 2 contribution. Missing/stale artifacts raise unless
    skipMissing is explicit; corrupt blobs always raise. Work is bounded to one
    historical artifact in memory at a time.
    """
    if type(topK) is not int or topK <= 0:
        raise ValueError("topK must be a positive integer")
    config = config or DetailedSearchConfig()
    unique = {}
    for match in stage2Results:
        if not isinstance(match, GeometricMatch) or not match.partId:
            raise ValueError("Stage 3 requires Stage 2 GeometricMatch records")
        unique.setdefault(match.partId, match)
    shortlist = list(unique.values())
    startedAt = perf_counter()
    matches, unavailable = [], []
    cacheSeconds = comparisonSeconds = 0.0
    queryIndex = SurfaceDistanceIndex(queryGeometry.mesh) if shortlist else None
    indexSeconds = perf_counter() - startedAt
    for match in shortlist:
        readAt = perf_counter()
        loaded = store.loadGeometries([match.partId], queryGeometry.config)
        cacheSeconds += perf_counter() - readAt
        if match.partId not in loaded:
            unavailable.append(match.partId)
            continue
        geometry, filePath = loaded[match.partId]
        compareAt = perf_counter()
        measured = compareDetailedGeometry(queryGeometry, geometry, config, queryIndex=queryIndex)
        comparisonSeconds += perf_counter() - compareAt
        matches.append(DetailedGeometricMatch(match.partId, match.distance, match.axisFlip,
                                              match.cosineSimilarity, filePath=filePath,
                                              detailedDistance=measured["detailedDistance"], measurements=measured))
    if unavailable and not skipMissing:
        raise DetailedSimilarityError(
            f"{len(unavailable)} shortlisted part(s) lack current compatible Stage 3 geometry; "
            "run build-detailed-geometry with matching settings or explicitly use skipMissing=True. "
            f"First IDs: {', '.join(unavailable[:5])}"
        )
    compared = len(matches)
    matches.sort(key=lambda match: (match.detailedDistance, match.partId))
    elapsed = perf_counter() - startedAt
    return DetailedMatchReport(matches[:topK], len(shortlist), compared, unavailable,
                               {"cachedGeometryLookupSeconds": cacheSeconds,
                                "queryDistanceIndexSeconds": indexSeconds,
                                "candidateComparisonSeconds": comparisonSeconds, "stage3TotalSeconds": elapsed,
                                "secondsPerCandidate": comparisonSeconds / compared if compared else 0.0},
                               config.toDict(), stage2Candidates=len(shortlist), stage3Candidates=min(topK, compared))


def rerankDetailedForStep(queryStepPath, stage2Results, topK=5, *, databasePath="parts.sqlite3",
                         geometryConfig=None, searchConfig=None, skipMissing=False, expectedSourceHash=None):
    if type(topK) is not int or topK <= 0:
        raise ValueError("topK must be a positive integer")
    shortlist = list(stage2Results)
    config = searchConfig or DetailedSearchConfig()
    if not shortlist:
        return DetailedMatchReport([], 0, 0, [], {"stage3TotalSeconds": 0.0}, config.toDict())
    startedAt = perf_counter()
    query = buildDetailedGeometryFromStep(queryStepPath, geometryConfig, expectedSourceHash=expectedSourceHash)
    with PartDatabase(databasePath, readOnly=True) as database:
        store = GeometryStore(database, initialize=False)
        report = rankDetailedCandidates(query, shortlist, store, topK, config=config, skipMissing=skipMissing)
    report.timings["queryPreprocessing"] = query.timings
    report.timings["stage3TotalSeconds"] = perf_counter() - startedAt
    report.queryMetadata = query.summary()
    return report


def findDetailedMatchesForStep(queryStepPath, *, databasePath="parts.sqlite3", descriptorConfig=None,
                               geometryConfig=None, searchConfig=None, stage2TopK=50, topK=5,
                               dimensionTolerance=0.1, volumeRange=None, allFamilies=False,
                               sphereMetric="euclidean", skipMissing=False):
    if type(topK) is not int or topK <= 0:
        raise ValueError("topK must be a positive integer")
    startedAt = perf_counter()
    queryPath = Path(queryStepPath).expanduser().resolve()
    before = fileFingerprint(queryPath)
    stage2 = findMatchesForStep(queryPath, databasePath=databasePath, config=descriptorConfig,
                               dimensionTolerance=dimensionTolerance, volumeRange=volumeRange,
                               allFamilies=allFamilies, topK=stage2TopK, metric=sphereMetric,
                               skipMissing=skipMissing)
    sourceHash = stage2.queryMetadata["descriptor"]["sourceHash"]
    report = rerankDetailedForStep(queryPath, stage2.matches, topK, databasePath=databasePath,
                                  geometryConfig=geometryConfig, searchConfig=searchConfig,
                                  skipMissing=skipMissing, expectedSourceHash=sourceHash)
    if fileFingerprint(queryPath) != before or hashFile(queryPath) != sourceHash:
        raise DetailedSimilarityError("Query STEP changed between retrieval and detailed reranking")
    report.stage1Candidates = stage2.requestedCandidates
    report.stage2Report = stage2.toDict()
    report.timings["pipelineTotalSeconds"] = perf_counter() - startedAt
    return report
