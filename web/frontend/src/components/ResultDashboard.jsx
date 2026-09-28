import React, { useState, useEffect, useMemo, useRef } from 'react';
import axios from 'axios';
import ModelViewer from './ModelViewer';
import MetricHelp from './MetricHelp';
import ExtendedSearchPanel from './ExtendedSearchPanel';
import DefectReviewPanel from './DefectReviewPanel';
import PreformRecoveryPanel from './PreformRecoveryPanel';
import useDefectReview, { draftGeometry } from '../hooks/useDefectReview';
import usePreformRecovery from '../hooks/usePreformRecovery';
import useExpertReview from '../hooks/useExpertReview';
import ExpertReviewPanel from './ExpertReviewPanel';
import DefectAwareOptimizationPanel from './DefectAwareOptimizationPanel';
import useDefectAwareOptimization from '../hooks/useDefectAwareOptimization';
import useStonePreservation from '../hooks/useStonePreservation';
import StonePreservationPanel from './StonePreservationPanel';
import {
  AWAITING_REVIEW_GUIDANCE, AWAITING_REVIEW_HEADLINE, CUT_SEQUENCE_LABEL,
  confirmedDefectFingerprint, defectAwareViewerPlan, gemMeshUrl,
} from '../utils/defectAwareOptimization';
import { expertReviewEligibility, pieceState } from '../utils/expertReview';
import { downloadPdf, resolveBackendUrl } from '../utils/pdfDownload';
import { validateGeometry } from '../utils/defectReview';
import { resolveBackendResource } from '../utils/backendUrl';
import { overlayOffset, readCoordinateFrame, viewerMeshFrame } from '../utils/coordinates';
import { OPTIMIZER_MODES, OPTIMIZER_MODE_LABELS, normalizeResult } from '../utils/preformRecovery';
import {
  Download, Layers, Box, Scale, Edit2, Check, X,
  Loader2, Sparkles, Copy, Hash, FileText, SunDim,
  Gem, Maximize2, AlertTriangle, ListOrdered, ShieldCheck, CheckCircle2
} from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const withCacheBust = (url) => {
  if (!url) return url;
  return `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}`;
};

const settingValue = (value, fallback) => {
  if (value === undefined || value === null || value === "") return fallback;
  return String(value);
};

// Canned fallback text per extended-search evaluation state, used only if
// the backend's own remaining_space_metadata.diagnostic_message is ever
// missing for a recognized state — the backend already sends this exact
// wording today, so this map is a defensive backstop, not the primary
// source. Deliberately avoids any "impossible" / "exhaustive" claim the
// bounded optimizer can't back up.
const REMAINING_SPACE_FALLBACK_MESSAGES = {
  extended_search_running:
    'Extended search is currently re-evaluating this remaining geometric space.',
  extended_search_completed_no_candidate:
    'Extended search completed. No higher-yield verified gemstone placement was found in this remaining geometric space.',
  extended_search_completed_candidate_found:
    'Extended search found an improved verified result. This remaining-space diagnostic is superseded by the updated plan.',
  extended_search_resource_stopped:
    'Extended search stopped at an implemented resource boundary before proving any higher-yield verified placement for this remaining geometric space.',
  extended_search_cancelled:
    'Extended search was stopped before final evaluation of this remaining geometric space.',
  extended_search_failed:
    'Extended search failed before final evaluation of this remaining geometric space.',
};

// Merges the (possibly stale) analysis_report rejection_reason with the
// live extended-search remaining_space_metadata, per priority: extended
// search's own current evaluation state wins whenever one has actually
// been recorded; the original report text is the fallback shown only
// before any extended search has ever run for this job.
const buildRemainingSpaceDiagnostic = (reportRejectionReason, extendedSearchStatus) => {
  const metadata = extendedSearchStatus?.remaining_space_metadata || null;
  const evaluationStatus = metadata?.search_evaluation_status || null;

  if (!evaluationStatus || evaluationStatus === 'initial_bounded_search') {
    return {
      heading: 'Initial bounded search diagnostic',
      message: reportRejectionReason || metadata?.diagnostic_message
        || 'No verified saleable placement remained.',
      superseded: false,
    };
  }

  return {
    heading: 'Remaining Geometric Space',
    message: metadata?.diagnostic_message
      || REMAINING_SPACE_FALLBACK_MESSAGES[evaluationStatus]
      || reportRejectionReason
      || 'No verified saleable placement remained.',
    superseded: evaluationStatus === 'extended_search_completed_candidate_found',
  };
};

const readOptimizerSettings = (report, option) => {
  const diag = option?.optimizer_diagnostics ?? report?.optimizer_diagnostics ?? {};
  const settings = diag?.optimizer_settings ?? {};
  return {
    bladeKerfMm: settingValue(
      settings.blade_kerf_mm ?? diag?.blade_clearance?.target_gap_mm ?? diag?.blade_gap?.target_gap_mm,
      "0.5"
    ),
    roughInsetMm: settingValue(
      settings.rough_clearance_mm ?? diag?.rough_clearance?.target_clearance_mm ?? diag?.fit?.rough_clearance_mm,
      "0.8"
    ),
    preformMarginMm: settingValue(
      settings.preform_margin_mm ?? option?.manufacturing_plan?.settings?.preform_margin_mm,
      ""
    ),
    maxCutDepthMm: settingValue(
      settings.max_cut_depth_mm ?? option?.manufacturing_plan?.settings?.max_cut_depth_mm,
      ""
    ),
    maxGems: settingValue(settings.max_gems, "12"),
    minGemCarat: settingValue(settings.min_secondary_carat, "0.5"),
    extraGemPolicy: settingValue(settings.extra_gem_policy, "saleable"),
  };
};

export default function ResultDashboard({
  modelUrl, reportUrl, cutUrl, defectsUrl,
  resultAssetBaseUrl, effectiveResultId, effectiveReportHash,
  initialShape, initialCutMode, jobId: jobIdProp,
  pdfReportAvailable, pdfReportUrl, pdfReportFilename, pdfReportError,
  // Reconstruction passed its quality gate and optimization was deferred
  // until Defect Review (job status `awaiting_defect_review`).
  awaitingDefectReview = false,
  initialOptimizerMode = OPTIMIZER_MODES.PRESERVATION,
}) {
  const [data,           setData]           = useState(null);
  const [selectedOption, setSelectedOption] = useState(null);

  // Weight editing
  const [isEditing,     setIsEditing]     = useState(false);
  const [tempWeight,    setTempWeight]     = useState("");
  const [isUpdating,    setIsUpdating]     = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  // Shape & mode (user can change and recalculate)
  const [shapes,         setShapes]         = useState([]);
  const [selectedShape,  setSelectedShape]  = useState(initialShape || "");
  const [cutMode,        setCutMode]        = useState(initialCutMode || "multi");
  const [bladeKerfMm,    setBladeKerfMm]    = useState("0.5");
  const [roughInsetMm,   setRoughInsetMm]   = useState("0.8");
  const [preformMarginMm, setPreformMarginMm] = useState("");
  const [maxCutDepthMm, setMaxCutDepthMm] = useState("");
  const [maxGems,        setMaxGems]        = useState("12");
  const [minGemCarat,    setMinGemCarat]    = useState("0.5");
  const [extraGemPolicy, setExtraGemPolicy] = useState("saleable");

  // 3D viewer URLs
  const [activeCutUrl,   setActiveCutUrl]   = useState(cutUrl);
  const [activeAssetBaseUrl, setActiveAssetBaseUrl] = useState(
    resolveBackendUrl(resultAssetBaseUrl, API_URL)
  );
  const activeResultIdRef = useRef(effectiveResultId || "result_v1");
  const activeReportHashRef = useRef(effectiveReportHash || null);
  const [showFractures,  setShowFractures]  = useState(true);
  const [showConfirmedRegions, setShowConfirmedRegions] = useState(true);
  const [selectedGemIndex, setSelectedGemIndex] = useState(null);
  // Latest raw status payload from ExtendedSearchPanel's own polling
  // (includes remaining_space_metadata) — null until that panel has
  // fetched something, or if extended search isn't available at all.
  const [extendedSearchStatus, setExtendedSearchStatus] = useState(null);
  // Viewer mode (Inspect / Defect Review / Cut Sequence) is lifted here so
  // the sidebar can show the matching Defect Review tools.
  const [viewerMode, setViewerMode] = useState('inspect');
  const [optimizerMode, setOptimizerMode] = useState(initialOptimizerMode);
  const [selectedRegionId, setSelectedRegionId] = useState(null);
  const [selectedPieceId, setSelectedPieceId] = useState(null);

  const [copyFeedback, setCopyFeedback] = useState(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [pdfDownloadError, setPdfDownloadError] = useState(
    pdfReportError || ""
  );
  const [pdfState, setPdfState] = useState({
    available: Boolean(pdfReportAvailable),
    url: pdfReportUrl || null,
    filename: pdfReportFilename || null,
  });
  const pollTimer = useRef(null);

  const reportJobMatch = reportUrl?.match(/\/files\/([^/]+)\//);
  const jobId = jobIdProp || reportJobMatch?.[1] || null;
  const effectiveAssetBaseUrl = (
    activeAssetBaseUrl
    || resolveBackendUrl(resultAssetBaseUrl, API_URL)
    || (jobId ? `${API_URL}/files/${jobId}/dense` : null)
  );
  const resultAssetUrl = (filename) => {
    if (!effectiveAssetBaseUrl || !filename) return null;
    return `${effectiveAssetBaseUrl.replace(/\/+$/, '')}/${filename}`;
  };

  const applyOptimizerSettings = (report, option = null) => {
    const settings = readOptimizerSettings(report, option);
    setBladeKerfMm(settings.bladeKerfMm);
    setRoughInsetMm(settings.roughInsetMm);
    setPreformMarginMm(settings.preformMarginMm);
    setMaxCutDepthMm(settings.maxCutDepthMm);
    setMaxGems(settings.maxGems);
    setMinGemCarat(settings.minGemCarat);
    setExtraGemPolicy(settings.extraGemPolicy);
  };

  // Load shape list for recalculate dropdown
  useEffect(() => {
    axios.get(`${API_URL}/shapes`)
      .then(res => setShapes(res.data.shapes || []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    setActiveCutUrl(resolveBackendUrl(cutUrl, API_URL));
    setActiveAssetBaseUrl(resolveBackendUrl(resultAssetBaseUrl, API_URL));
    activeResultIdRef.current = effectiveResultId || "result_v1";
    activeReportHashRef.current = effectiveReportHash || null;
  }, [cutUrl, resultAssetBaseUrl, effectiveResultId, effectiveReportHash]);

  useEffect(() => {
    setPdfState({
      available: Boolean(pdfReportAvailable),
      url: pdfReportUrl || null,
      filename: pdfReportFilename || null,
    });
    setPdfDownloadError(pdfReportError || "");
  }, [
    pdfReportAvailable,
    pdfReportUrl,
    pdfReportFilename,
    pdfReportError,
  ]);

  useEffect(() => {
    if (!reportUrl) return;
    const resolvedReportUrl = resolveBackendUrl(reportUrl, API_URL);
    const resolvedAssetBase = (
      resolveBackendUrl(resultAssetBaseUrl, API_URL)
      || (jobId ? `${API_URL}/files/${jobId}/dense` : null)
    );
    axios.get(withCacheBust(resolvedReportUrl))
      .then(res => {
        if (res.data && !res.data.error) {
          setActiveAssetBaseUrl(resolvedAssetBase);
          activeResultIdRef.current = effectiveResultId || "result_v1";
          activeReportHashRef.current = effectiveReportHash || null;
          setData(res.data);
          if (res.data.options?.length) {
            const defaultOption = res.data.options[0];
            setSelectedOption(defaultOption);
            applyOptimizerSettings(res.data, defaultOption);
            // Keep the 3D viewer in sync with the sidebar's default
            // selection on first paint — without this, the viewer kept
            // showing the raw job cut_url (best_cut.ply) while the
            // sidebar already described options[0], which could be a
            // different strategy.
            if (defaultOption.file && resolvedAssetBase) {
              setActiveCutUrl(withCacheBust(
                `${resolvedAssetBase.replace(/\/+$/, '')}/${defaultOption.file}`
              ));
            }
          } else {
            applyOptimizerSettings(res.data);
          }
        }
      })
      .catch(err => console.error("Could not load stats", err));
  }, [
    reportUrl,
    jobId,
    resultAssetBaseUrl,
    effectiveResultId,
    effectiveReportHash,
  ]);

  useEffect(() => () => { if (pollTimer.current) clearInterval(pollTimer.current); }, []);

  const handleSelectOption = (opt) => {
    setSelectedOption(opt);
    applyOptimizerSettings(data, opt);
    setSelectedGemIndex(null);
    if (opt.file) {
      setActiveCutUrl(withCacheBust(resultAssetUrl(opt.file)));
    }
  };

  const handleExtendedStatusChange = (status) => {
    setExtendedSearchStatus(status);
    if (!status) return;

    if (Object.prototype.hasOwnProperty.call(status, 'pdf_report_available')) {
      setPdfState({
        available: Boolean(status.pdf_report_available),
        url: status.pdf_report_url || null,
        filename: status.pdf_report_filename || null,
      });
      setPdfDownloadError(status.pdf_report_error || "");
    }

    const nextResultId = status.effective_result_id || null;
    const nextReportHash = status.effective_report_hash || null;
    const nextReportUrl = resolveBackendUrl(status.report_url, API_URL);
    const nextAssetBase = resolveBackendUrl(
      status.result_asset_base_url,
      API_URL
    );
    const selectionChanged = (
      nextResultId
      && (
        nextResultId !== activeResultIdRef.current
        || nextReportHash !== activeReportHashRef.current
      )
    );
    if (!selectionChanged || !nextReportUrl || !nextAssetBase) return;

    activeResultIdRef.current = nextResultId;
    activeReportHashRef.current = nextReportHash;
    setActiveAssetBaseUrl(nextAssetBase);
    axios.get(withCacheBust(nextReportUrl))
      .then(res => {
        if (!res.data || res.data.error) return;
        setData(res.data);
        setSelectedGemIndex(null);
        if (res.data.options?.length) {
          const defaultOption = res.data.options[0];
          setSelectedOption(defaultOption);
          applyOptimizerSettings(res.data, defaultOption);
          if (defaultOption.file) {
            setActiveCutUrl(withCacheBust(
              `${nextAssetBase.replace(/\/+$/, '')}/${defaultOption.file}`
            ));
          }
        } else {
          setSelectedOption(null);
          applyOptimizerSettings(res.data);
          setActiveCutUrl(resolveBackendUrl(status.cut_url, API_URL));
        }
      })
      .catch(err => console.error("Could not load effective result", err));
  };

  const handleCopyId = () => {
    navigator.clipboard.writeText(jobId);
    setCopyFeedback(true);
    setTimeout(() => setCopyFeedback(false), 2000);
  };

  const handleDownloadReport = async () => {
    if (
      isDownloadingPdf ||
      isUpdating ||
      !jobId ||
      !pdfState.available ||
      !pdfState.url
    ) {
      return;
    }
    setIsDownloadingPdf(true);
    setPdfDownloadError("");
    try {
      await downloadPdf({
        url: resolveBackendUrl(pdfState.url, API_URL),
        filename: pdfState.filename || `quartz-analysis-${jobId}.pdf`,
      });
    } catch (error) {
      setPdfDownloadError(error?.message || "PDF download failed.");
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const handleSaveWeight = async () => {
    if (!jobId) return;
    const roughWeight = Number(tempWeight);
    if (!tempWeight || !Number.isFinite(roughWeight) || roughWeight <= 0) {
      setStatusMessage("Enter a positive rough weight in carats.");
      return;
    }
    if (cutMode === "multi") {
      const margin = Number(preformMarginMm);
      const depth = Number(maxCutDepthMm);
      const inset = Number(roughInsetMm);
      if (!preformMarginMm || !Number.isFinite(margin) || margin <= 0 || margin > 5) {
        setStatusMessage("Enter a preform allowance between 0 and 5 mm.");
        return;
      }
      if (!maxCutDepthMm || !Number.isFinite(depth) || depth <= 0 || depth > 500) {
        setStatusMessage("Enter the saw's maximum usable depth between 0 and 500 mm.");
        return;
      }
      if (!Number.isFinite(inset) || inset < margin) {
        setStatusMessage("Rough inset must be at least the preform allowance.");
        return;
      }
    }
    if (pollTimer.current) clearInterval(pollTimer.current);

    setIsUpdating(true);
    setStatusMessage("Initializing...");

    const formData = new FormData();
    formData.append("known_weight",    tempWeight);
    if (selectedShape) formData.append("preferred_shape", selectedShape);
    formData.append("cut_mode",        cutMode);
    formData.append("blade_kerf_mm", bladeKerfMm);
    formData.append("rough_clearance_mm", roughInsetMm);
    if (cutMode === "multi") {
      formData.append("preform_margin_mm", preformMarginMm);
      formData.append("max_cut_depth_mm", maxCutDepthMm);
    }
    formData.append("max_gems", maxGems);
    formData.append("min_secondary_carat", minGemCarat);
    formData.append("extra_gem_policy", extraGemPolicy);

    try {
      await axios.post(`${API_URL}/jobs/${jobId}/recalculate`, formData);

      pollTimer.current = setInterval(async () => {
        try {
          const ts  = Date.now();
          const res = await axios.get(`${API_URL}/jobs/${jobId}/status?t=${ts}`);
          const s   = res.data;

          if (s.message) setStatusMessage(s.message);

          if (s.status === "Completed") {
            clearInterval(pollTimer.current);
            pollTimer.current = null;
            setPdfState({
              available: Boolean(s.pdf_report_available),
              url: s.pdf_report_url || null,
              filename: s.pdf_report_filename || null,
            });
            setPdfDownloadError(s.pdf_report_error || "");
            const nextAssetBase = (
              resolveBackendUrl(s.result_asset_base_url, API_URL)
              || `${API_URL}/files/${jobId}/dense`
            );
            setActiveAssetBaseUrl(nextAssetBase);
            activeResultIdRef.current = s.effective_result_id || "result_v1";
            activeReportHashRef.current = s.effective_report_hash || null;
            const rr = await axios.get(withCacheBust(
              resolveBackendUrl(s.report_url, API_URL)
            ));
            setData(rr.data);
            if (rr.data.options?.length) {
              const defaultOption = rr.data.options[0];
              setSelectedOption(defaultOption);
              setSelectedGemIndex(null);
              applyOptimizerSettings(rr.data, defaultOption);
              if (defaultOption.file) {
                setActiveCutUrl(withCacheBust(
                  `${nextAssetBase.replace(/\/+$/, '')}/${defaultOption.file}`
                ));
              }
            } else {
              setSelectedOption(null);
              applyOptimizerSettings(rr.data);
              if (s.cut_url) {
                setActiveCutUrl(resolveBackendUrl(s.cut_url, API_URL));
              }
            }
            setIsUpdating(false);
            setIsEditing(false);
          } else if (s.status === "Failed") {
            clearInterval(pollTimer.current);
            pollTimer.current = null;
            setStatusMessage(`Update failed: ${s.message || "Please try again."}`);
            setIsUpdating(false);
          } else if (s.status === "Cancelled") {
            clearInterval(pollTimer.current);
            pollTimer.current = null;
            setStatusMessage("Update cancelled.");
            setIsUpdating(false);
          }
        } catch (e) { console.error(e); }
      }, 1000);
    } catch {
      setStatusMessage("Network error while starting recalculation.");
      setIsUpdating(false);
    }
  };

  // --- Derived display values ---
  const currentYield      = selectedOption?.yield       ?? data?.yield_percent ?? 0;
  const currentWeight     = selectedOption?.weight      ?? data?.estimated_cut_carats ?? 0;
  const currentShape      = selectedOption?.name        ?? "—";
  const currentLightScore = selectedOption?.light_score ?? data?.light_analysis?.score ?? 0;
  const currentLightGrade = selectedOption?.light_grade ?? data?.light_analysis?.grade ?? "N/A";
  const currentDims       = selectedOption?.cut_dims    ?? [0, 0, 0];
  const roughDims         = data?.rough_dimensions_mm   ?? [0, 0, 0];
  const gemDetails        = selectedOption?.gem_details ?? data?.gem_details ?? [];
  // Single source of truth for "how many gems": the actual exported
  // per-gem list, not the separately-reported gem_count field — keeps
  // the badge, the strategy list, and the Gem Details card from ever
  // showing three different numbers.
  const gemCount          = gemDetails.length || 1;
  const isOptimized       = data && data.yield_percent !== 35;
  const barColor          = isOptimized ? "bg-purple-500" : "bg-cyan-400";
  const hasFractures      = !!defectsUrl;
  const selectedUtil      = selectedOption?.space_utilization ?? data?.space_utilization ?? {};
  const utilizationPct    = typeof selectedUtil === 'number'
    ? selectedUtil
    : selectedUtil?.occupied_percent ?? 0;
  const axisUtil          = Array.isArray(selectedUtil?.axis_percent)
    ? selectedUtil.axis_percent
    : [];
  const wasteReduction    = data?.waste_reduction ?? {};
  const wasteDelta        = wasteReduction?.reduction_vs_baseline_percent ?? 0;
  const defectSummary     = data?.defect_summary ?? {};
  const facetPlan         = selectedOption?.facet_recommendation ?? data?.facet_recommendation ?? null;
  const optimizerDiag     = selectedOption?.optimizer_diagnostics ?? data?.optimizer_diagnostics ?? {};
  const pocketFill        = optimizerDiag?.pocket_fill ?? {};
  const recoverySearch    = optimizerDiag?.preserve_fill ?? optimizerDiag?.repacked_search ?? {};
  const bladeClearance    = optimizerDiag?.blade_clearance ?? optimizerDiag?.blade_gap ?? {};
  const clearanceModel    = optimizerDiag?.clearance_model ?? {};
  const optimizerSettings = optimizerDiag?.optimizer_settings ?? {};
  const remainingSpace    = optimizerDiag?.remaining_free_space ?? {};
  const freeComponents    = remainingSpace?.components?.length
    ? { count: remainingSpace.component_count, top_components: remainingSpace.components }
    : optimizerDiag?.free_space_components ?? pocketFill?.free_space_components ?? {};
  const research          = data?.research_completion ?? {};
  const softwareComplete  = research?.proposal_software_completion_percent ?? null;
  const jobValidation     = research?.current_job_validation_percent ?? null;
  const externalValidation = research?.external_research_validation_percent ?? null;
  const externalPending   = Array.isArray(research?.external_validation_required)
    ? research.external_validation_required.length
    : 0;
  const actualGapMm       = clearanceModel?.actual_exported_gap_mm
    ?? bladeClearance?.actual_min_gap_mm ?? bladeClearance?.estimated_min_gap_mm ?? null;
  const bladeKerfDisplay  = clearanceModel?.blade_kerf_mm ?? optimizerSettings?.blade_kerf_mm ?? 0.5;
  const preformDisplay    = clearanceModel?.preform_margin_mm ?? optimizerSettings?.preform_margin_mm ?? null;
  const protectedGapMm    = clearanceModel?.protected_corridor_mm
    ?? optimizerSettings?.protected_corridor_mm ?? bladeClearance?.target_gap_mm ?? 0.5;
  const voxelGapMm        = clearanceModel?.voxelized_protected_corridor_mm ?? null;
  const displayedRoughInsetMm = optimizerDiag?.rough_clearance?.target_clearance_mm
    ?? optimizerDiag?.fit?.rough_clearance_mm
    ?? null;
  const pocketAdded       = recoverySearch?.added_to_best ?? pocketFill?.added ?? 0;
  const reportRejectionReason = remainingSpace?.components?.[0]?.rejection_reason
    ?? pocketFill?.unused_space_reason ?? null;
  // Reflects extended search's current findings for this space when one has
  // run, instead of always showing the original (possibly stale) report
  // text — see buildRemainingSpaceDiagnostic.
  const remainingSpaceDiagnostic = buildRemainingSpaceDiagnostic(
    reportRejectionReason, extendedSearchStatus
  );
  const unusedReason      = remainingSpaceDiagnostic.message || '—';
  const displayedMaxGems  = optimizerSettings?.max_gems ?? maxGems;
  const displayedMinCarat = optimizerSettings?.min_secondary_carat ?? minGemCarat;
  const displayedPolicy   = optimizerSettings?.extra_gem_policy ?? extraGemPolicy;
  const manufacturingPlan = selectedOption?.manufacturing_plan ?? data?.manufacturing_plan ?? {};
  const cutSequence = Array.isArray(manufacturingPlan?.sequence)
    ? manufacturingPlan.sequence
    : [];
  const gemUrls = gemDetails
    .filter((gem) => gem?.file)
    .map((gem) => resultAssetUrl(gem.file));
  // Same gem_details the sidebar renders, augmented with a resolvable PLY
  // URL per gem so ModelViewer can render/select each one individually in
  // Inspect mode without needing to know about API_URL/jobId itself.
  const gemDetailsWithUrl = gemDetails.map((gem) => ({
    ...gem,
    url: resultAssetUrl(gem?.file),
  }));

  // --- Defect Review / Preform Recovery ---
  // Spatial work uses each endpoint's own coordinate_frame, never the
  // selected legacy option's scale.
  const isPreformMode = optimizerMode === OPTIMIZER_MODES.PREFORM;
  const isPreservationMode = optimizerMode === OPTIMIZER_MODES.PRESERVATION;
  const defectReview = useDefectReview({ apiUrl: API_URL, jobId });
  const canPlaceDefects = Boolean(overlayOffset(viewerMeshFrame(modelUrl), defectReview.frame));
  const isExpertMode = viewerMode === 'expert';
  // The Expert Review tab needs the preform result to decide eligibility.
  const preform = usePreformRecovery({ apiUrl: API_URL, jobId, enabled: isPreformMode || isExpertMode });
  const expertEligibility = expertReviewEligibility({ phase: preform.phase, resultView: preform.result });
  const expert = useExpertReview({
    apiUrl: API_URL, jobId, runId: expertEligibility.runId, enabled: isExpertMode && expertEligibility.eligible,
  });
  // Backend piece list + safely resolved mesh URLs (null mesh_file → list only).
  const viewerExpertPieces = useMemo(() => (expert.review?.pieces || []).map((piece) => ({
    piece_id: piece.piece_id,
    state: pieceState(piece),
    resolvedUrl: resolveBackendResource(piece.mesh_file, API_URL),
  })), [expert.review]);
  const viewerExpert = isExpertMode ? {
    pieces: viewerExpertPieces,
    selectedPieceId,
    onSelectPiece: setSelectedPieceId,
    frame: preform.result?.frame ?? null,
  } : null;
  const draftShape = draftGeometry(defectReview.draft);
  const draftPreview = draftShape && (
    draftShape.kind === 'ellipsoid'
      ? validateGeometry(draftShape) === null
      : draftShape.points_mm.length >= 2 && Number(draftShape.radius_mm) > 0
  ) ? draftShape : null;
  // Memoized: the viewer resets its cut-sequence step whenever this
  // result's identity changes.
  // mesh_file is backend-root-relative: resolved against the backend base,
  // never the legacy result asset directory.
  const viewerPreformResult = useMemo(() => {
    if (!preform.result) return null;
    return {
      ...preform.result,
      regions: preform.result.regions.map((region) => ({
        ...region,
        resolvedUrl: resolveBackendResource(region?.mesh_file, API_URL),
      })),
    };
  }, [preform.result]);
  const viewerDefectReview = defectReview.availability === 'available' ? {
    frame: defectReview.frame,
    candidates: defectReview.review.candidates,
    annotations: defectReview.review.annotations,
    showRejected: defectReview.showRejected,
    selectedId: defectReview.selectedId,
    onSelect: (id) => defectReview.setSelectedId(
      (current) => (current === id ? null : id)
    ),
    draft: defectReview.draft,
    draftPreview,
    onPick: defectReview.handlePick,
  } : null;
  const viewerPreform = isPreformMode ? {
    active: true,
    result: viewerPreformResult,
    selectedRegionId,
    onSelectRegion: setSelectedRegionId,
  } : null;

  // --- Defect-aware faceted optimization ---
  // Re-packs gems around confirmed defects on the EXISTING reconstruction.
  // The viewer and Cut Sequence switch to its gems + sequence together only
  // while the run is completed and the backend says it is not stale;
  // otherwise both fall back to the original optimization as a pair.
  const confirmedFingerprint = useMemo(
    () => confirmedDefectFingerprint(defectReview.review),
    [defectReview.review],
  );
  const defectAware = useDefectAwareOptimization({ apiUrl: API_URL, jobId, confirmedFingerprint });
  const preservation = useStonePreservation({ apiUrl: API_URL, jobId, confirmedFingerprint, enabled: isPreservationMode });
  const preservationView = useMemo(() => {
    if (!preservation.result || preservation.result.stale) return null;
    const view = normalizeResult(preservation.result);
    return { ...view, regions: view.regions.map(region => ({ ...region,
      resolvedUrl: resolveBackendResource(region.mesh_file, API_URL) })) };
  }, [preservation.result]);
  const viewerPreservation = isPreservationMode ? { active: true, label: 'Stone Preservation', result: preservationView,
    selectedRegionId, onSelectRegion: setSelectedRegionId } : null;
  const defectAwareResult = defectAware.result;
  const defectAwareActive = Boolean(
    !isPreservationMode && !isPreformMode && defectAware.phase === 'completed' && defectAwareResult && !defectAwareResult.stale
  );
  const defectAwareGems = useMemo(() => (defectAwareResult?.gems || []).map((gem, index) => ({
    ...gem,
    index: gem?.index ?? index + 1,
    url: gemMeshUrl(
      gem,
      (path) => resolveBackendResource(path, API_URL),
      (file) => (effectiveAssetBaseUrl && file ? `${effectiveAssetBaseUrl.replace(/\/+$/, '')}/${file}` : null),
    ),
  })), [defectAwareResult, effectiveAssetBaseUrl]);
  const legacyPlanSettings = manufacturingPlan?.settings;
  const defectAwarePlan = useMemo(
    () => defectAwareViewerPlan(
      defectAwareResult, { settings: legacyPlanSettings }, defectAware.lastRequest?.preform_mm ?? null,
    ),
    // Identity must only change with the result, so the viewer's cut step
    // is not reset on unrelated re-renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [defectAwareResult, defectAware.lastRequest],
  );
  const defectAwareFrame = useMemo(
    () => readCoordinateFrame(defectAwareResult?.coordinate_frame),
    [defectAwareResult],
  );
  useEffect(() => { setSelectedGemIndex(null); }, [defectAwareActive]);
  const viewerGemDetails = defectAwareActive ? defectAwareGems : gemDetailsWithUrl;
  const viewerGemUrls = defectAwareActive
    ? defectAwareGems.map((gem) => gem.url).filter(Boolean)
    : gemUrls;
  const reconstructionReady = Boolean(modelUrl);
  const effectiveOption = data?.options?.[0] ?? null;
  const originalSummary = data ? {
    gemCount: (effectiveOption?.gem_details ?? data?.gem_details ?? []).length || effectiveOption?.gem_count || null,
    weightCt: effectiveOption?.weight ?? data?.estimated_cut_carats ?? null,
    yieldPercent: effectiveOption?.yield ?? data?.yield_percent ?? null,
  } : null;
  const defectAwareSettings = {
    bladeKerfMm, preformMm: preformMarginMm, roughInsetMm, maxCutDepthMm, maxGems, minGemCarat,
  };
  const defectAwareSetters = {
    bladeKerfMm: setBladeKerfMm, preformMm: setPreformMarginMm, roughInsetMm: setRoughInsetMm,
    maxCutDepthMm: setMaxCutDepthMm, maxGems: setMaxGems, minGemCarat: setMinGemCarat,
  };
  const showLegacyMetrics = !isPreservationMode && !isPreformMode && !(awaitingDefectReview && !data);

  return (
    <div className="min-h-screen w-full flex flex-col overflow-auto lg:h-screen lg:flex-row lg:overflow-hidden">

      {/* LEFT: 3D Viewer */}
      <div className="relative h-[56vh] min-h-[420px] w-full bg-black lg:h-full lg:min-h-0 lg:flex-1">
        <ModelViewer
          modelUrl={modelUrl}
          cutUrl={isPreservationMode || defectAwareActive ? null : activeCutUrl}
          defectsUrl={!isPreservationMode && showFractures ? defectsUrl : null}
          gemUrls={isPreservationMode ? [] : viewerGemUrls}
          gemDetails={isPreservationMode ? [] : viewerGemDetails}
          manufacturingPlan={isPreservationMode ? preservation.result?.manufacturing_plan : defectAwareActive ? defectAwarePlan : manufacturingPlan}
          remainingSpace={isPreservationMode || defectAwareActive ? null : remainingSpace}
          remainingSpaceDiagnostic={remainingSpaceDiagnostic}
          activeStrategyName={isPreservationMode ? 'Stone Preservation' : defectAwareActive ? 'Defect-aware placement' : (data ? currentShape : null)}
          activeGemCount={isPreservationMode ? null : defectAwareActive ? defectAwareResult.gem_count : (data ? gemCount : null)}
          sequenceLabel={isPreservationMode ? 'Preservation cut sequence' : defectAwareActive ? CUT_SEQUENCE_LABEL : null}
          showConfirmedDefects={isPreservationMode || (defectAwareActive && showConfirmedRegions)}
          gemFrame={defectAwareActive ? defectAwareFrame : null}
          selectedGemIndex={selectedGemIndex}
          onSelectGem={setSelectedGemIndex}
          viewerMode={viewerMode}
          onViewerModeChange={(mode) => {
            if (mode === 'expert' && isPreservationMode) setOptimizerMode(OPTIMIZER_MODES.PREFORM);
            setViewerMode(mode);
          }}
          defectReview={viewerDefectReview}
          preform={viewerPreservation || viewerPreform}
          expertReview={viewerExpert}
        />
      </div>

      {/* RIGHT: Sidebar */}
      <div className="z-20 flex w-full flex-col gap-5 overflow-y-visible border-t border-slate-800 bg-slate-900 p-4 shadow-2xl sm:p-6 lg:w-96 lg:shrink-0 lg:overflow-y-auto lg:border-l lg:border-t-0">

        {/* HEADER */}
        <div className="border-b border-slate-800 pb-4">
          <h2 className="text-2xl font-bold text-white mb-1">Analysis Report</h2>
          <p className="text-slate-400 text-sm">
            {isPreservationMode ? 'Clean Material Recovery' : isPreformMode
              ? 'Preform Recovery Planning'
              : (awaitingDefectReview && !data ? 'Defect Review before gemstone placement' : 'Automated Yield Estimation')}
          </p>
          <button
            onClick={handleCopyId}
            className="flex items-center gap-1.5 px-2 py-1 rounded mt-3 bg-slate-800 hover:bg-slate-700 border border-slate-700 transition-all group"
            title="Copy Job ID"
          >
            <Hash className="w-3 h-3 text-slate-500 group-hover:text-cyan-400" />
            <span className="text-[10px] font-mono text-slate-400 group-hover:text-white">
              {copyFeedback ? "Copied!" : jobId}
            </span>
            {!copyFeedback && <Copy className="w-3 h-3 text-slate-500 group-hover:text-cyan-400" />}
          </button>
        </div>

        {/* RECONSTRUCTION COMPLETE — optimization deferred until Defect Review */}
        {awaitingDefectReview && !defectAwareActive && (
          <div className="rounded-xl border border-emerald-500/40 bg-emerald-500/10 p-4" role="status" aria-label="Reconstruction complete">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-400" />
              <h3 className="font-semibold text-white">{AWAITING_REVIEW_HEADLINE}</h3>
            </div>
            <p className="mt-1.5 text-xs leading-5 text-emerald-100/90">{isPreservationMode ? 'Review confirmed defects, then calculate Stone Preservation using this reconstruction.' : AWAITING_REVIEW_GUIDANCE}</p>
            {viewerMode !== 'defects' && (
              <button
                type="button"
                onClick={() => setViewerMode('defects')}
                className="mt-3 flex min-h-10 w-full items-center justify-center gap-2 rounded-lg bg-red-600 px-3 py-2 text-sm font-semibold text-white hover:bg-red-500"
              >
                <AlertTriangle className="h-4 w-4" /> Review Defects
              </button>
            )}
          </div>
        )}

        {/* FRACTURE TOGGLE (only shown if fractures were detected) */}
        {hasFractures && (
          <button
            onClick={() => setShowFractures(v => !v)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm font-medium transition-all ${
              showFractures
                ? 'bg-red-500/10 border-red-500/40 text-red-400'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-white'
            }`}
          >
            <AlertTriangle className="w-4 h-4" />
            {showFractures ? 'Hide Fractures / Clouds' : 'Show Fractures / Clouds'}
          </button>
        )}

        {/* DEFECT REVIEW — compact outside the Defect Review viewer mode */}
        {jobId && (
          <DefectReviewPanel
            review={defectReview}
            active={viewerMode === 'defects'}
            onOpen={() => setViewerMode('defects')}
            canPlace={canPlaceDefects}
            apiUrl={API_URL}
          />
        )}

        {/* DEFECT-AWARE GEM OPTIMIZATION — re-runs only placement + cuts */}
        {isPreservationMode && jobId && <StonePreservationPanel preservation={preservation}
          ready={reconstructionReady && defectReview.availability === 'available'} onEditDefects={() => setViewerMode('defects')} />}
        {jobId && !isPreformMode && !isPreservationMode && (
          <DefectAwareOptimizationPanel
            optimization={defectAware}
            confirmedCount={defectReview.review.summary.confirmed}
            provisionalCount={defectReview.review.summary.provisional}
            reviewAvailable={defectReview.availability === 'available'}
            reconstructionReady={reconstructionReady}
            awaitingReview={awaitingDefectReview}
            settings={defectAwareSettings}
            onSettingChange={(key, value) => defectAwareSetters[key]?.(value)}
            onCalculate={() => defectAware.start(defectAwareSettings)}
            onEditDefects={() => setViewerMode('defects')}
            original={originalSummary}
          />
        )}
        {defectAwareActive && (
          <label className="-mt-2 flex items-center gap-2 text-[11px] text-slate-400">
            <input
              type="checkbox" checked={showConfirmedRegions}
              onChange={(event) => setShowConfirmedRegions(event.target.checked)}
              className="accent-red-500"
            />
            Show confirmed defect safety regions with gems
          </label>
        )}

        {/* EXPERT REVIEW — shown in the Expert Review viewer mode */}
        {jobId && isExpertMode && (
          <ExpertReviewPanel
            eligibility={
              preform.availability === 'unavailable'
                ? { eligible: false, reason: 'Expert Review is available for Preform Recovery V2 results.' }
                : preform.availability === 'unknown'
                  ? { eligible: false, reason: 'Checking for a completed Preform Recovery result…' }
                  : expertEligibility
            }
            expert={expert}
            selectedPieceId={selectedPieceId}
            onSelectPiece={setSelectedPieceId}
          />
        )}

        {/* WEIGHT CARD */}
        {!isPreservationMode && <div className="bg-gradient-to-br from-emerald-900/50 to-slate-900 p-5 rounded-xl border border-emerald-500/30 relative group">
          <div className="flex items-center gap-3 mb-2">
            <Box className="w-5 h-5 text-emerald-400" />
            <span className="text-emerald-100 font-semibold">Rough Weight</span>
          </div>

          {isEditing ? (
            <div className="mt-2 space-y-3">
              {/* Weight field */}
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="0.000001"
                  step="any"
                  className="w-24 bg-black/40 border border-emerald-500/50 rounded px-2 py-1 text-xl text-white font-bold outline-none focus:border-emerald-400"
                  autoFocus
                  defaultValue={data?.raw_carats ?? ""}
                  onChange={(e) => setTempWeight(e.target.value)}
                />
                <span className="text-sm text-slate-400">cts</span>
                <div className="ml-auto flex gap-2">
                  <button
                    onClick={handleSaveWeight}
                    disabled={isUpdating}
                    className={`flex items-center justify-center w-8 h-8 rounded-full transition-all ${
                      isUpdating ? 'bg-emerald-700 cursor-wait' : 'bg-emerald-600 hover:bg-emerald-500'
                    }`}
                  >
                    {isUpdating
                      ? <Loader2 className="w-4 h-4 text-white animate-spin" />
                      : <Check className="w-4 h-4 text-white" />}
                  </button>
                  <button
                    onClick={() => setIsEditing(false)}
                    disabled={isUpdating}
                    className="flex items-center justify-center w-8 h-8 bg-slate-700 hover:bg-slate-600 rounded-full text-slate-300 transition-all"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Shape selector inside edit panel */}
              <div className="flex items-center gap-2">
                <Gem className="w-4 h-4 text-purple-400 shrink-0" />
                <select
                  className="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white outline-none"
                  value={selectedShape}
                  onChange={(e) => setSelectedShape(e.target.value)}
                >
                  <option value="">Auto (Best Yield)</option>
                  {shapes.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>

              {/* Cut mode inside edit panel */}
              <div className="flex gap-2">
                <button
                  onClick={() => setCutMode('single')}
                  className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                    cutMode === 'single'
                      ? 'bg-purple-600 border-purple-500 text-white'
                      : 'bg-slate-800 border-slate-700 text-slate-400'
                  }`}
                >
                  <Maximize2 className="w-3 h-3" /> One Large
                </button>
                <button
                  onClick={() => setCutMode('multi')}
                  className={`flex-1 flex items-center justify-center gap-1 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                    cutMode === 'multi'
                      ? 'bg-purple-600 border-purple-500 text-white'
                      : 'bg-slate-800 border-slate-700 text-slate-400'
                  }`}
                >
                  <Layers className="w-3 h-3" /> Multi-Gem
                </button>
              </div>

              {/* Optimizer settings inside edit panel */}
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-emerald-500/20">
                <label className="block">
                  <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Blade Gap</span>
                  <input
                    type="number"
                    min="0.2"
                    max="2"
                    step="0.1"
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white font-mono outline-none focus:border-emerald-400"
                    value={bladeKerfMm}
                    onChange={(e) => setBladeKerfMm(e.target.value)}
                  />
                </label>
                <label className="block">
                  <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Preform mm</span>
                  <input
                    type="number"
                    min="0.01"
                    max="5"
                    step="0.1"
                    disabled={cutMode !== "multi"}
                    placeholder={cutMode === "multi" ? "Required" : "N/A"}
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white font-mono outline-none focus:border-emerald-400 disabled:opacity-40"
                    value={preformMarginMm}
                    onChange={(e) => setPreformMarginMm(e.target.value)}
                  />
                </label>
                <label className="block">
                  <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Max Depth</span>
                  <input
                    type="number"
                    min="0.01"
                    max="500"
                    step="0.1"
                    disabled={cutMode !== "multi"}
                    placeholder={cutMode === "multi" ? "Required" : "N/A"}
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white font-mono outline-none focus:border-emerald-400 disabled:opacity-40"
                    value={maxCutDepthMm}
                    onChange={(e) => setMaxCutDepthMm(e.target.value)}
                  />
                </label>
                <label className="block">
                  <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Inset</span>
                  <input
                    type="number"
                    min="0.2"
                    max="5"
                    step="0.1"
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white font-mono outline-none focus:border-emerald-400"
                    value={roughInsetMm}
                    onChange={(e) => setRoughInsetMm(e.target.value)}
                  />
                </label>
                <label className="block">
                  <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Max Gems</span>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    step="1"
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white font-mono outline-none focus:border-emerald-400"
                    value={maxGems}
                    onChange={(e) => setMaxGems(e.target.value)}
                  />
                </label>
                <label className="block">
                  <span className="block text-[10px] text-slate-500 uppercase font-bold mb-1">Min Ct</span>
                  <input
                    type="number"
                    min="0.1"
                    max="10"
                    step="0.1"
                    className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white font-mono outline-none focus:border-emerald-400"
                    value={minGemCarat}
                    onChange={(e) => setMinGemCarat(e.target.value)}
                  />
                </label>
              </div>
              <select
                className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm text-white outline-none"
                value={extraGemPolicy}
                onChange={(e) => setExtraGemPolicy(e.target.value)}
              >
                <option value="saleable">Saleable pockets</option>
                <option value="maximum_count">Maximum count</option>
              </select>

              {isUpdating && (
                <p className="text-[10px] font-mono text-emerald-400 animate-pulse text-right">
                  {statusMessage || "Processing..."}
                </p>
              )}
              {!isUpdating && statusMessage && (
                <p className="text-[10px] text-red-400" role="alert">{statusMessage}</p>
              )}
            </div>
          ) : (
            <div className="flex items-end justify-between">
              <div>
                <span className="text-4xl font-bold text-white">{data?.raw_carats ?? defectAwareResult?.rough_weight_ct ?? "..."}</span>
                <span className="text-lg text-slate-400 ml-2">cts</span>
              </div>
              {!isPreformMode && data && (
                <button
                  onClick={() => {
                    setTempWeight(data?.raw_carats);
                    applyOptimizerSettings(data, selectedOption);
                    setIsEditing(true);
                  }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-emerald-400"
                >
                  <Edit2 className="w-4 h-4" />
                </button>
              )}
            </div>
          )}

          <p className="text-xs text-emerald-300/70 mt-2 border-t border-emerald-500/20 pt-2 flex justify-between">
            <span>Vol: {data?.volume_cm3 ?? 0} cm³</span>
            <span>L: {roughDims[0]} mm</span>
          </p>
        </div>

        }
        {/* OPTIMIZER MODE */}
        <div>
          <h3 className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Optimizer mode
          </h3>
          <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-label="Optimizer mode">
            {[OPTIMIZER_MODES.PRESERVATION, OPTIMIZER_MODES.LEGACY, OPTIMIZER_MODES.PREFORM].map((mode) => (
              <button
                key={mode} type="button" role="radio"
                aria-checked={optimizerMode === mode}
                disabled={isEditing}
                onClick={() => {
                  setOptimizerMode(mode);
                  setSelectedRegionId(null);
                  setSelectedGemIndex(null);
                }}
                className={`min-h-10 rounded-lg border px-2 py-1.5 text-xs font-semibold transition-all disabled:opacity-50 ${
                  optimizerMode === mode
                    ? (mode === OPTIMIZER_MODES.PREFORM
                      ? 'border-cyan-500 bg-cyan-600/20 text-cyan-200'
                      : 'border-purple-500 bg-purple-600/20 text-purple-200')
                    : 'border-slate-700 bg-slate-800 text-slate-400 hover:text-white'
                }`}
              >{OPTIMIZER_MODE_LABELS[mode]}</button>
            ))}
          </div>
        </div>

        {isPreformMode && jobId && (
          <PreformRecoveryPanel
            preform={preform}
            review={defectReview}
            selectedRegionId={selectedRegionId}
            onSelectRegion={setSelectedRegionId}
          />
        )}

        {showLegacyMetrics && (<>
        {defectAwareActive && (
          <h3 className="border-t border-slate-800 pt-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">
            Original Optimization (before confirmed defects)
          </h3>
        )}
        {/* OPTIONS LIST */}
        {data?.options?.length > 0 && (
          <div className="space-y-2">
            <h3 className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
              Cut Strategies
            </h3>
            <div className="flex flex-col gap-2 max-h-44 overflow-y-auto pr-1">
              {data.options.map((opt, idx) => {
                const optGemCount = opt.gem_details?.length || opt.gem_count || 1;
                return (
                <button
                  key={idx}
                  onClick={() => handleSelectOption(opt)}
                  className={`flex items-center justify-between p-3 rounded-lg border transition-all text-left ${
                    selectedOption === opt
                      ? 'bg-purple-500/10 border-purple-500/50 shadow-lg'
                      : 'bg-slate-800 border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <div>
                    <div className={`font-semibold text-sm ${selectedOption === opt ? 'text-white' : 'text-slate-300'}`}>
                      {opt.name}
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {opt.type} · {optGemCount} gem{optGemCount > 1 ? 's' : ''} · {opt.space_utilization?.occupied_percent ?? '—'}%
                    </div>
                    {opt.optimizer_diagnostics?.baseline_comparison && (
                      <div className="text-[10px] text-cyan-400 mt-0.5">
                        {opt.optimizer_diagnostics.baseline_comparison.added_gems >= 0 ? '+' : ''}{opt.optimizer_diagnostics.baseline_comparison.added_gems} gems · {opt.optimizer_diagnostics.baseline_comparison.yield_delta_percentage_points >= 0 ? '+' : ''}{opt.optimizer_diagnostics.baseline_comparison.yield_delta_percentage_points} yield pts
                      </div>
                    )}
                    <div className={`text-[10px] mt-0.5 ${
                      opt.manufacturing_plan?.status === 'complete'
                        ? 'text-emerald-400'
                        : opt.manufacturing_plan?.status === 'geometric_comparison_only'
                          ? 'text-amber-400'
                          : 'text-slate-500'
                    }`}>
                      {opt.manufacturing_plan?.status?.replaceAll('_', ' ') || 'legacy plan'}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono font-bold text-emerald-400">{opt.weight} ct</div>
                    <div className="text-[10px] text-slate-500">
                      {opt.light_score > 0 ? `${opt.light_score} Brilliance` : `${opt.yield}% Yield`}
                    </div>
                  </div>
                </button>
              );})}
            </div>
          </div>
        )}

        {/* METRICS */}
        <div className="space-y-3">

          {/* Cut weight + yield bar */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {isOptimized
                  ? <Sparkles className="w-4 h-4 text-purple-400" />
                  : <Scale    className="w-4 h-4 text-cyan-400" />}
                <span className="text-slate-300">Cut Weight</span>
                {gemCount > 1 && (
                  <span className="text-[10px] bg-purple-500/20 text-purple-300 px-1.5 py-0.5 rounded-full">
                    {gemCount} gems
                  </span>
                )}
              </div>
              <span className={`font-mono font-bold text-lg ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>
                {currentWeight} cts
              </span>
            </div>
            <div className="w-full bg-slate-700 h-1.5 rounded-full overflow-hidden">
              <div className={`${barColor} h-full transition-all duration-700`}
                   style={{ width: `${Math.min(currentYield, 100)}%` }} />
            </div>
            <div className="flex justify-between items-center mt-2">
              <p className="text-[10px] text-slate-500">{currentShape}</p>
              <p className={`text-xs font-bold flex items-center ${isOptimized ? "text-purple-400" : "text-cyan-400"}`}>
                {currentYield}% Yield
                <MetricHelp metricKey="yield" align="right" />
              </p>
            </div>
          </div>

          {/* Space utilization */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Scale className="w-4 h-4 text-emerald-400" />
                <span className="text-slate-300">Space Utilization</span>
                <MetricHelp metricKey="spaceUtilization" />
              </div>
              <span className="font-mono font-bold text-emerald-400">{utilizationPct}%</span>
            </div>
            <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
              <div className="bg-emerald-400 h-full transition-all duration-700"
                   style={{ width: `${Math.min(utilizationPct, 100)}%` }} />
            </div>
            <div className="flex justify-between mt-2 text-[10px] text-slate-500">
              <span>Axis: {axisUtil.length ? axisUtil.join(' / ') : '—'}%</span>
              <span className="flex items-center">
                Waste: {wasteReduction?.projected_waste_percent ?? (100 - currentYield).toFixed(1)}%
                <MetricHelp metricKey="waste" align="right" />
              </span>
            </div>
            {wasteReduction?.traditional_waste_baseline_percent !== undefined && (
              <>
                <p className={`text-[10px] mt-2 ${wasteDelta >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {wasteDelta >= 0 ? '+' : ''}{wasteDelta} pts vs {wasteReduction.traditional_waste_baseline_percent}% assumed baseline
                </p>
                <p className="text-[9px] text-slate-600 mt-0.5">
                  Internal reference figure, not a validated traditional-cutting comparison
                </p>
              </>
            )}
          </div>

          {/* Manufacturing clearance and pocket fill */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span className="text-slate-300">Manufacturing Clearance</span>
                <MetricHelp metricKey="manufacturingClearance" />
              </div>
              <span className={`font-mono font-bold text-sm ${
                bladeClearance?.meets_target === false ? 'text-orange-400' : 'text-cyan-400'
              }`}>
                 {actualGapMm !== null ? `${actualGapMm} mm` : '—'}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Blade kerf</div>
                <div className="font-mono text-white">{bladeKerfDisplay} mm</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Preform</div>
                <div className="font-mono text-white">{preformDisplay ?? '—'} mm</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Protected</div>
                <div className="font-mono text-white">{protectedGapMm} mm</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Voxelized</div>
                <div className="font-mono text-white">{voxelGapMm ?? '—'} mm</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Pocket gems</div>
                <div className="font-mono text-white">+{pocketAdded}</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Free regions</div>
                <div className="font-mono text-white">{freeComponents?.count ?? 0}</div>
              </div>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">
              {displayedRoughInsetMm !== null ? `Inset: ${displayedRoughInsetMm} mm · ` : ''}{unusedReason}
            </p>
            <p className="text-[10px] text-slate-600 mt-1">
              Max: {displayedMaxGems} · Min: {displayedMinCarat} ct · {displayedPolicy}
            </p>
          </div>

          {/* Individual gem details */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {manufacturingPlan?.status === 'complete'
                  ? <ShieldCheck className="w-4 h-4 text-emerald-400" />
                  : <ListOrdered className="w-4 h-4 text-amber-400" />}
                <span className="text-slate-300">Saw Sequence</span>
                <MetricHelp metricKey="sawSequence" />
              </div>
              <span className={`font-mono text-xs ${
                manufacturingPlan?.status === 'complete' ? 'text-emerald-400' : 'text-amber-400'
              }`}>
                {manufacturingPlan?.status?.replaceAll('_', ' ') || 'unavailable'}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Cuts</div>
                <div className="font-mono text-white">{cutSequence.length}</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Preform</div>
                <div className="font-mono text-white">{manufacturingPlan?.settings?.preform_margin_mm ?? '—'} mm</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="text-xs text-slate-500">Depth</div>
                <div className="font-mono text-white">{manufacturingPlan?.maximum_required_depth_mm ?? '—'} mm</div>
              </div>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">
              {manufacturingPlan?.operator_guidance_only
                ? 'Operator guidance only · not CNC or G-code'
                : 'Select a verified cuttable strategy to inspect its sequence.'}
            </p>
          </div>

          {/* Individual gem details */}
          {gemDetails.length > 0 && (
            <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Gem className="w-4 h-4 text-pink-400" />
                  <span className="text-slate-300">Gem Details</span>
                  <MetricHelp metricKey="planShare" />
                </div>
                <span className="font-mono text-xs text-pink-300">
                  {gemDetails.length}
                </span>
              </div>
              <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                {gemDetails.map((gem) => (
                  <button
                    key={gem.index}
                    type="button"
                    onClick={() => setSelectedGemIndex(
                      (current) => (current === gem.index ? null : gem.index)
                    )}
                    className={`w-full text-left bg-slate-900 rounded p-2 border transition-colors ${
                      selectedGemIndex === gem.index
                        ? 'border-cyan-400 ring-1 ring-cyan-400/40'
                        : 'border-slate-800 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex justify-between gap-2">
                      <div className="min-w-0">
                        <div className="text-xs font-semibold text-white truncate">
                          #{gem.index} {gem.shape}
                        </div>
                        <div className="text-[10px] text-slate-500">
                          {Array.isArray(gem.dimensions_mm)
                            ? gem.dimensions_mm.join(' x ')
                            : '0 x 0 x 0'} mm
                        </div>
                      </div>
                      <div className="text-right shrink-0">
                        <div className="font-mono text-sm text-emerald-400">
                          {gem.weight_ct} ct
                        </div>
                        <div className="text-[10px] text-slate-500">
                          {gem.plan_share_percent}% share
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Research completion — internal QA metadata, not a cutting
              decision input, so it's tucked behind a collapsed toggle
              instead of competing for attention with the primary cards. */}
          {softwareComplete !== null && (
            <details className="group rounded-lg border border-slate-700 bg-slate-800/50">
              <summary className="flex cursor-pointer list-none items-center justify-between p-4 text-slate-400 hover:text-slate-300">
                <div className="flex items-center gap-2">
                  <Check className="w-4 h-4 text-emerald-400" />
                  <span>Implementation Alignment</span>
                </div>
                <span className="font-mono text-xs text-slate-500 group-open:hidden">
                  {softwareComplete}% implementation alignment
                </span>
              </summary>
              <div className="px-4 pb-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-slate-300">Implementation Alignment</span>
                  <span className="font-mono font-bold text-emerald-400">
                    {softwareComplete}%
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-center">
                  <div className="bg-slate-900 rounded p-2">
                    <div className="text-xs text-slate-500">Job Valid</div>
                    <div className="font-mono text-white">{jobValidation ?? 0}%</div>
                  </div>
                  <div className="bg-slate-900 rounded p-2">
                    <div className="text-xs text-slate-500">External</div>
                    <div className="font-mono text-white">{externalValidation ?? 0}%</div>
                  </div>
                </div>
                <p className="text-[10px] text-slate-500 mt-2">
                  {research?.overall_status ?? 'software_prototype_complete'} · {externalPending} validation item{externalPending === 1 ? '' : 's'}
                </p>
              </div>
            </details>
          )}

          {/* Light performance */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <SunDim className="w-4 h-4 text-yellow-400" />
              <span className="text-slate-300">Light Performance</span>
              <MetricHelp metricKey="lightPerformance" />
            </div>
            <div className="flex justify-between items-end">
              <div>
                <div className="text-3xl font-bold text-white">{currentLightScore}</div>
                <div className="text-[10px] text-slate-500">Brilliance Score</div>
              </div>
              <div className="px-3 py-1 rounded-full bg-slate-900 border border-slate-600 text-xs font-mono text-cyan-400">
                {currentLightGrade}
              </div>
            </div>
            <div className="w-full bg-slate-900 h-1 mt-3 rounded-full overflow-hidden">
              <div className="bg-yellow-400 h-full transition-all duration-1000"
                   style={{ width: `${currentLightScore}%` }} />
            </div>
          </div>

          {/* Defect and facet recommendation */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="w-4 h-4 text-orange-400" />
              <span className="text-slate-300">Defect-Aware Facet Plan</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="bg-slate-900 rounded p-2">
                <div className="flex items-center justify-center text-xs text-slate-500">
                  Defects
                  <MetricHelp metricKey="defectCount" align="right" />
                </div>
                <div className="font-mono text-white">{defectSummary?.point_count ?? 0}</div>
              </div>
              <div className="bg-slate-900 rounded p-2">
                <div className="flex items-center justify-center text-xs text-slate-500">
                  Facet Score
                  <MetricHelp metricKey="facetScore" align="right" />
                </div>
                <div className="font-mono text-white">{facetPlan?.score ?? 0}</div>
              </div>
            </div>
            <p className="text-[10px] text-slate-500 mt-2">
              Normal: {facetPlan?.normal ? facetPlan.normal.join(', ') : '0, 0, 1'}
            </p>
          </div>

          {/* Dimensions */}
          <div className="p-4 bg-slate-800/50 rounded-lg border border-slate-700">
            <div className="flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4 text-purple-400" />
              <span className="text-slate-300">Cut Dimensions</span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-center">
              {['L', 'W', 'H'].map((label, i) => (
                <div key={label} className="bg-slate-900 rounded p-2">
                  <div className="text-xs text-slate-500">{label}</div>
                  <div className="font-mono text-white">{currentDims[i] ?? 0}</div>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-slate-500 mt-2 text-center">Measurements in mm</p>
          </div>

        </div>
        </>)}

        {/* Extended search — renders nothing if the backend doesn't
            support it yet, so this is safe to always mount. Kept mounted
            (hidden) in Preform Recovery mode so a running legacy search
            keeps reporting status. */}
        {jobId && (
          <div className={isPreformMode || isPreservationMode ? 'hidden' : ''}>
            <ExtendedSearchPanel
              apiUrl={API_URL} jobId={jobId}
              onStatusChange={handleExtendedStatusChange}
            />
          </div>
        )}

        {/* ACTIONS */}
        <div className="mt-auto pt-4 border-t border-slate-800 space-y-3">
          <button
            onClick={handleDownloadReport}
            disabled={
              isDownloadingPdf ||
              isUpdating ||
              !jobId ||
              !pdfState.available ||
              !pdfState.url
            }
            title={
              pdfState.available
                ? "Download PDF report"
                : pdfDownloadError || "PDF report is not available yet"
            }
            className="flex items-center justify-center gap-2 w-full bg-slate-800 hover:bg-slate-700 disabled:bg-slate-900 disabled:text-slate-600 disabled:cursor-not-allowed text-cyan-400 border border-slate-700 py-3 rounded-lg font-bold transition-colors"
          >
            {isDownloadingPdf
              ? <Loader2 className="w-4 h-4 animate-spin" />
              : <FileText className="w-4 h-4" />}
            {isDownloadingPdf ? "Downloading PDF..." : "Download PDF Report"}
          </button>
          {pdfDownloadError && (
            <p className="text-xs text-red-400" role="alert">
              {pdfDownloadError}
            </p>
          )}
          <a
            href={modelUrl}
            download
            className="flex items-center justify-center gap-2 w-full bg-slate-100 hover:bg-white text-slate-900 py-3 rounded-lg font-bold transition-colors"
          >
            <Download className="w-4 h-4" /> Download .PLY Model
          </a>
        </div>
      </div>
    </div>
  );
}
