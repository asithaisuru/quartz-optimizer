import React, { Suspense, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { Bounds, ContactShadows, Line, TrackballControls } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import {
  Box, Crosshair, Eye, EyeOff, Grid, Pause, Play, RotateCcw,
  ScanSearch, Scissors, SkipBack, SkipForward, Sun, X,
} from 'lucide-react';
import * as THREE from 'three';
import MetricHelp from './MetricHelp';
import PreformRegionDetails from './PreformRegionDetails';
import { VISUAL_STATES, itemId, readGeometry, readSparsePoints, visualStateOf } from '../utils/defectReview';
import { cutPlaneCentered, regionColor } from '../utils/preformRecovery';
import { PIECE_STATES } from '../utils/expertReview';

const EXPERT_LEGEND = ['auto_usable', 'pending', 'usable_preform', 'needs_further_separation', 'waste_unusable'];
import {
  BACKEND_TO_VIEWER_ROTATION, overlayOffset, pickedLocalToMm, viewerMeshFrame,
} from '../utils/coordinates';

const BACKEND_TO_VIEWER = BACKEND_TO_VIEWER_ROTATION;
const NO_RAYCAST = () => null;
// Pointer travel (px) above which a click is treated as a rotate-drag.
const CLICK_DRAG_TOLERANCE = 4;

// Reports a click on the rough surface in the backend mesh frame, with the
// face normal flipped to point toward the camera (outward on the visible
// side) — the only depth cue a surface click can honestly provide.
function surfacePick(event) {
  const pointMesh = event.object.worldToLocal(event.point.clone()).toArray();
  let normalMesh = null;
  if (event.face?.normal) {
    const local = event.face.normal.clone();
    const world = local.clone().transformDirection(event.object.matrixWorld);
    if (world.dot(event.ray.direction) > 0) local.negate();
    normalMesh = local.toArray();
  }
  return { pointMesh, normalMesh };
}

function RoughStone({ url, isXRay, showWireframe, sequenceMode, onPick, onBounds }) {
  const geometry = useLoader(PLYLoader, url);
  useLayoutEffect(() => {
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    if (onBounds) onBounds(geometry.boundingSphere?.radius || 0);
  }, [geometry, onBounds]);
  const pickHandlers = onPick ? {
    onClick: (event) => {
      if (event.delta > CLICK_DRAG_TOLERANCE) return;
      event.stopPropagation();
      onPick(surfacePick(event));
    },
    onPointerOver: () => { document.body.style.cursor = 'crosshair'; },
    onPointerOut: () => { document.body.style.cursor = 'auto'; },
  } : {};
  return (
    <group>
      <mesh geometry={geometry} {...pickHandlers}>
        {isXRay || sequenceMode ? (
          <meshPhysicalMaterial
            color="#dbe7ef" transmission={0.2}
            opacity={sequenceMode ? 0.2 : 0.38} transparent
            roughness={0.24} metalness={0.03} depthWrite={false}
            side={THREE.DoubleSide}
          />
        ) : (
          <meshStandardMaterial
            color="#ffffff" vertexColors side={THREE.DoubleSide}
            roughness={0.55} metalness={0.02}
          />
        )}
      </mesh>
      {showWireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial color="#168fe5" wireframe transparent opacity={0.5} />
        </mesh>
      )}
    </group>
  );
}

function CutGem({ url, showWireframe }) {
  const geometry = useLoader(PLYLoader, url);
  useLayoutEffect(() => { geometry.computeVertexNormals(); }, [geometry]);
  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial color="#ff175f" roughness={0.12} metalness={0.68} />
      </mesh>
      {showWireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial color="#ffe45e" wireframe transparent opacity={0.82} />
        </mesh>
      )}
    </group>
  );
}

// Individual per-gem PLY files are exported directly from the same
// placement used to build the combined cut mesh, so — unlike ProtectedGem
// below — they already sit in the rough stone's coordinate frame and need
// no re-centering to line up with it.
function InspectableGem({ gem, isSelected, isMuted, showWireframe, onSelect }) {
  const geometry = useLoader(PLYLoader, gem.url);
  useLayoutEffect(() => { geometry.computeVertexNormals(); }, [geometry]);
  const color = isSelected ? '#22d3ee' : isMuted ? '#475569' : '#ff175f';
  return (
    <group>
      <mesh
        geometry={geometry}
        onClick={(event) => { event.stopPropagation(); onSelect(gem.index); }}
        onPointerOver={(event) => { event.stopPropagation(); document.body.style.cursor = 'pointer'; }}
        onPointerOut={() => { document.body.style.cursor = 'auto'; }}
      >
        <meshStandardMaterial
          color={color} roughness={0.12} metalness={0.68}
          transparent opacity={isMuted ? 0.16 : 1} depthWrite={!isMuted}
        />
      </mesh>
      {showWireframe && (
        <mesh geometry={geometry}>
          <meshBasicMaterial
            color="#ffe45e" wireframe transparent
            opacity={isMuted ? 0.12 : 0.82}
          />
        </mesh>
      )}
      {isSelected && (
        <mesh geometry={geometry} scale={1.012}>
          <meshBasicMaterial color="#67e8f9" wireframe transparent opacity={0.9} depthWrite={false} />
        </mesh>
      )}
    </group>
  );
}

function GemStat({ label, help, value, mono }) {
  return (
    <div className="bg-slate-900 rounded px-2 py-1.5 min-w-0">
      <div className="flex items-center text-slate-500">
        <span className="truncate">{label}</span>
        {help}
      </div>
      <div
        className={`text-white break-words leading-snug ${mono ? 'font-mono text-[9px]' : ''}`}
        title={typeof value === 'string' || typeof value === 'number' ? String(value) : undefined}
      >
        {value}
      </div>
    </div>
  );
}

function SelectedGemPanel({ gem, mmPerMesh, onClear }) {
  if (!gem) return null;
  const dims = Array.isArray(gem.dimensions_mm) ? `${gem.dimensions_mm.join(' × ')} mm` : '—';
  const center = Array.isArray(gem.center_mm) ? `${gem.center_mm.join(', ')} mm` : '—';
  const clearanceMesh = gem.surface_clearance_mesh_units;
  const clearanceMm = typeof clearanceMesh === 'number' && mmPerMesh > 0
    ? (clearanceMesh * mmPerMesh).toFixed(3)
    : null;
  const clearanceValue = clearanceMm !== null
    ? `${clearanceMm} mm`
    : (typeof clearanceMesh === 'number' ? `${clearanceMesh} mesh u.` : '—');

  return (
    <div className="absolute left-4 bottom-4 z-20 w-80 max-w-[88vw] border border-cyan-500/40 bg-slate-950/95 p-3 rounded-lg shadow-xl">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-semibold text-cyan-300 break-words pr-2">
          Gem #{gem.index}{gem.shape ? ` · ${gem.shape}` : ''}
        </div>
        <button
          onClick={onClear}
          title="Clear selection"
          className="shrink-0 text-slate-500 hover:text-white"
        ><X className="w-3.5 h-3.5" /></button>
      </div>
      <div className="grid grid-cols-2 gap-1.5 text-[10px] leading-snug">
        <GemStat label="Weight" value={gem.weight_ct != null ? `${gem.weight_ct} ct` : '—'} />
        <GemStat label="Dimensions" value={dims} />
        <GemStat
          label="Rough yield"
          help={<MetricHelp metricKey="roughYield" align="right" />}
          value={gem.yield_percent != null ? `${gem.yield_percent}%` : '—'}
        />
        <GemStat
          label="Plan share"
          help={<MetricHelp metricKey="planShare" align="right" />}
          value={gem.plan_share_percent != null ? `${gem.plan_share_percent}%` : '—'}
        />
        <GemStat label="Scale" value={gem.scale != null ? Number(gem.scale).toFixed(4) : '—'} />
        <GemStat label="Center" value={center} />
        <GemStat
          label="Surface clearance"
          help={<MetricHelp metricKey="surfaceClearance" align="right" />}
          value={clearanceValue}
        />
        <GemStat label="Reference" value={gem.file || '—'} mono />
      </div>
    </div>
  );
}

function ProtectedGem({ url, index, activeStep, marginMesh }) {
  const loaded = useLoader(PLYLoader, url);
  const prepared = useMemo(() => {
    const geometry = loaded.clone();
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const center = geometry.boundingBox.getCenter(new THREE.Vector3());
    const radius = Math.max(geometry.boundingSphere?.radius || 0, 1e-6);
    geometry.translate(-center.x, -center.y, -center.z);
    return { geometry, center, radius };
  }, [loaded]);

  const id = `gem_${index + 1}`;
  const negative = activeStep?.negative_side_gems?.includes(id);
  const positive = activeStep?.positive_side_gems?.includes(id);
  const retained = activeStep?.retained_gems?.includes(id);
  const color = negative ? '#19d3ae' : positive ? '#ffb020' : '#64748b';
  const scale = 1 + Math.max(0, marginMesh) / prepared.radius;
  return (
    <group position={prepared.center}>
      <mesh geometry={prepared.geometry}>
        <meshStandardMaterial
          color={color} roughness={0.15} metalness={0.58} transparent
          opacity={retained ? 0.92 : 0.22} depthWrite={retained}
        />
      </mesh>
      <mesh geometry={prepared.geometry} scale={scale}>
        <meshBasicMaterial
          color={retained ? '#ffffff' : '#64748b'} wireframe transparent
          opacity={retained ? 0.42 : 0.12} depthWrite={false}
        />
      </mesh>
    </group>
  );
}

function DefectCloud({ url }) {
  const geometry = useLoader(PLYLoader, url);
  useLayoutEffect(() => { geometry.computeBoundingSphere(); }, [geometry]);
  return (
    <points geometry={geometry}>
      <pointsMaterial
        color="#ff7a00" size={0.018} sizeAttenuation transparent
        opacity={0.95} depthWrite={false}
      />
    </points>
  );
}

function RemainingSpaceEnvelope({ component }) {
  const center = component?.centre_mesh_units;
  const extent = component?.bbox_extent_mesh_units;
  const peak = component?.clearance_peak_mesh_units;
  const radius = Number(component?.max_inscribed_radius_mesh_units) || 0;
  if (!Array.isArray(center) || !Array.isArray(extent)) return null;
  return (
    <group>
      <mesh position={center}>
        <boxGeometry args={extent.map((value) => Math.max(Number(value) || 0, 0.002))} />
        <meshBasicMaterial
          color="#22d3ee" transparent opacity={0.12} depthWrite={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh position={center}>
        <boxGeometry args={extent.map((value) => Math.max(Number(value) || 0, 0.002))} />
        <meshBasicMaterial color="#67e8f9" wireframe transparent opacity={0.72} />
      </mesh>
      {Array.isArray(peak) && radius > 0 && (
        <mesh position={peak}>
          <sphereGeometry args={[radius, 24, 16]} />
          <meshBasicMaterial color="#facc15" wireframe transparent opacity={0.9} />
        </mesh>
      )}
    </group>
  );
}

function DirectionArrow({ origin, direction, length, color }) {
  const arrow = useMemo(() => {
    const start = new THREE.Vector3(...origin);
    const vector = new THREE.Vector3(...direction).normalize();
    return new THREE.ArrowHelper(
      vector, start, length, color,
      Math.max(length * 0.18, 0.05), Math.max(length * 0.08, 0.025),
    );
  }, [origin, direction, length, color]);
  return <primitive object={arrow} />;
}

function CutOverlay({ step }) {
  const plane = step?.plane;
  const contour = step?.section_contour || [];
  const overlay = useMemo(() => {
    if (!plane?.normal || !plane?.origin) return null;
    const normal = new THREE.Vector3(...plane.normal).normalize();
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1), normal,
    );
    const points = contour.map((point) => new THREE.Vector3(...point));
    const bounds = new THREE.Box3().setFromPoints(points);
    const size = bounds.isEmpty()
      ? new THREE.Vector3(1, 1, 1)
      : bounds.getSize(new THREE.Vector3());
    return {
      quaternion,
      points,
      span: Math.max(size.x, size.y, size.z, 0.4) * 1.15,
    };
  }, [plane, contour]);
  if (!overlay) return null;

  const origin = plane.origin;
  const thickness = Math.max(step?.kerf_slab?.thickness_mesh_units || 0.01, 0.006);
  return (
    <group>
      <mesh position={origin} quaternion={overlay.quaternion}>
        <boxGeometry args={[overlay.span, overlay.span, thickness]} />
        <meshBasicMaterial
          color="#ffca28" transparent opacity={0.24}
          side={THREE.DoubleSide} depthWrite={false}
        />
      </mesh>
      {overlay.points.length > 2 && (
        <Line
          points={[...overlay.points, overlay.points[0]]}
          color="#fff36a" lineWidth={2.5}
        />
      )}
      <DirectionArrow
        origin={origin} direction={plane.normal}
        length={overlay.span * 0.45} color="#fff36a"
      />
      {!step.hide_feed_direction && (
        <DirectionArrow
          origin={origin} direction={step.feed_direction || [1, 0, 0]}
          length={overlay.span * 0.55} color="#32d6ff"
        />
      )}
    </group>
  );
}

// Keeps one missing/broken optional asset (e.g. a region PLY) from taking
// down the whole canvas.
class SafeLoad extends React.Component {
  constructor(props) {
    super(props);
    this.state = { failed: false };
  }

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    return this.state.failed ? null : this.props.children;
  }
}

// Material treatment per defect state — distinct in form as well as hue:
// candidates are wire-only, manual drafts faint fill + wire, confirmed a
// denser fill, rejected a faint wire.
const DEFECT_MATERIALS = {
  candidate: { fill: 0.08, wire: 0.9 },
  manual: { fill: 0.22, wire: 0.65 },
  confirmed: { fill: 0.4, wire: 0.8 },
  rejected: { fill: 0, wire: 0.3 },
  draft: { fill: 0.16, wire: 1 },
};

function DefectMaterialPair({ color, look, children, selected }) {
  return (
    <>
      {look.fill > 0 && (
        <mesh raycast={NO_RAYCAST}>
          {children}
          <meshBasicMaterial color={color} transparent opacity={look.fill} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      )}
      <mesh raycast={NO_RAYCAST}>
        {children}
        <meshBasicMaterial color={selected ? '#67e8f9' : color} wireframe transparent opacity={look.wire} depthWrite={false} />
      </mesh>
    </>
  );
}

function DefectShape({ geometry, mmPerMesh, color, look, selected, onSelect }) {
  const hit = onSelect ? {
    onClick: (event) => {
      if (event.delta > CLICK_DRAG_TOLERANCE) return;
      event.stopPropagation();
      onSelect();
    },
  } : { raycast: NO_RAYCAST };

  if (geometry.kind === 'ellipsoid') {
    const position = geometry.center_mm.map((v) => v / mmPerMesh);
    const scale = geometry.radii_mm.map((v) => Math.max(v / mmPerMesh, 1e-4));
    return (
      <group position={position} scale={scale}>
        <DefectMaterialPair color={color} look={look} selected={selected}>
          <sphereGeometry args={[1, 24, 16]} />
        </DefectMaterialPair>
        <mesh {...hit}><sphereGeometry args={[1, 12, 8]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} /></mesh>
      </group>
    );
  }

  const radius = Math.max(geometry.radius_mm / mmPerMesh, 1e-4);
  const points = geometry.points_mm.map((p) => new THREE.Vector3(...p.map((v) => v / mmPerMesh)));
  return (
    <group>
      {points.slice(1).map((end, index) => {
        const start = points[index];
        const direction = end.clone().sub(start);
        const length = direction.length();
        if (length < 1e-6) return null;
        const quaternion = new THREE.Quaternion().setFromUnitVectors(
          new THREE.Vector3(0, 1, 0), direction.clone().normalize(),
        );
        const mid = start.clone().add(end).multiplyScalar(0.5);
        return (
          <group key={`s${index}`} position={mid.toArray()} quaternion={quaternion}>
            <DefectMaterialPair color={color} look={look} selected={selected}>
              <cylinderGeometry args={[radius, radius, length, 14, 1, true]} />
            </DefectMaterialPair>
            <mesh {...hit}><cylinderGeometry args={[radius, radius, length, 8]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} /></mesh>
          </group>
        );
      })}
      {points.map((point, index) => (
        <group key={`j${index}`} position={point.toArray()}>
          <DefectMaterialPair color={color} look={look} selected={selected}>
            <sphereGeometry args={[radius, 14, 10]} />
          </DefectMaterialPair>
        </group>
      ))}
    </group>
  );
}

// Points placed so far while tracing a fracture (before it has 2 points
// there is no corridor to draw yet).
function DraftPoints({ points, mmPerMesh }) {
  const size = Math.max(0.6 / mmPerMesh, 1e-4);
  return points.map((point, index) => (
    <mesh key={index} position={point.map((v) => v / mmPerMesh)} raycast={NO_RAYCAST}>
      <sphereGeometry args={[size, 12, 8]} />
      <meshBasicMaterial color="#facc15" />
    </mesh>
  ));
}

function SparsePoints({ points, color, size, selected }) {
  return points.map((point, index) => (
    <mesh key={index} position={point} raycast={NO_RAYCAST}>
      <octahedronGeometry args={[size, 0]} />
      <meshBasicMaterial color={selected ? '#67e8f9' : color} wireframe />
    </mesh>
  ));
}

function DefectOverlays({ items, draftGeometry: draftShape, draftPoints, mmPerMesh, selectedId, onSelect, interactive }) {
  if (!(mmPerMesh > 0)) return null;
  return (
    <group>
      {items.map(({ id, item, state }) => {
        const geometry = readGeometry(item);
        if (!geometry) {
          // Raw detector evidence: aligned sparse points (centered mesh
          // units), drawn as dots — never inflated into a volume.
          const sparse = readSparsePoints(item);
          return sparse.length ? (
            <SparsePoints
              key={id} points={sparse} color={VISUAL_STATES[state].color}
              size={Math.max(0.5 / mmPerMesh, 1e-4)}
              selected={selectedId !== null && selectedId === id}
            />
          ) : null;
        }
        return (
          <DefectShape
            key={id} geometry={geometry} mmPerMesh={mmPerMesh}
            color={VISUAL_STATES[state].color} look={DEFECT_MATERIALS[state]}
            selected={selectedId !== null && selectedId === id}
            onSelect={interactive ? () => onSelect(id) : null}
          />
        );
      })}
      {draftShape && (
        <DefectShape geometry={draftShape} mmPerMesh={mmPerMesh} color="#facc15" look={DEFECT_MATERIALS.draft} />
      )}
      {draftPoints?.length > 0 && <DraftPoints points={draftPoints} mmPerMesh={mmPerMesh} />}
    </group>
  );
}

function RegionMesh({ url, color, selected, muted, onSelect }) {
  const geometry = useLoader(PLYLoader, url);
  useLayoutEffect(() => { geometry.computeVertexNormals(); }, [geometry]);
  return (
    <group>
      <mesh
        geometry={geometry}
        onClick={onSelect ? (event) => {
          if (event.delta > CLICK_DRAG_TOLERANCE) return;
          event.stopPropagation();
          onSelect();
        } : undefined}
        onPointerOver={onSelect ? () => { document.body.style.cursor = 'pointer'; } : undefined}
        onPointerOut={onSelect ? () => { document.body.style.cursor = 'auto'; } : undefined}
      >
        <meshStandardMaterial
          color={color} roughness={0.25} metalness={0.2} transparent
          opacity={muted ? 0.18 : 0.72} depthWrite={!muted} side={THREE.DoubleSide}
        />
      </mesh>
      <mesh geometry={geometry} scale={selected ? 1.01 : 1} raycast={NO_RAYCAST}>
        <meshBasicMaterial
          color={selected ? '#67e8f9' : color} wireframe transparent
          opacity={selected ? 0.9 : muted ? 0.1 : 0.35} depthWrite={false}
        />
      </mesh>
    </group>
  );
}

// Region PLY vertices are already in the centered frame: rendered as-is
// (no per-region centering or scaling). Regions without a mesh are listed
// in the sidebar only — no centroid is promised by the contract.
function PreformRegions({ regions, selectedRegionId, onSelectRegion, muted }) {
  return regions.map((region, index) => {
    if (!region.resolvedUrl) return null;
    const color = regionColor(index);
    const selected = selectedRegionId === region.region_id;
    const select = onSelectRegion ? () => onSelectRegion(selected ? null : region.region_id) : null;
    return (
      <SafeLoad key={region.region_id ?? index}>
        <Suspense fallback={null}>
          <RegionMesh
            url={region.resolvedUrl} color={color} selected={selected}
            muted={muted || (selectedRegionId !== null && !selected)} onSelect={select}
          />
        </Suspense>
      </SafeLoad>
    );
  });
}

// Preform cut planes: selected verified cuts as solid amber planes,
// geometric candidates as faint grey wireframes, so the two can't be
// mistaken for each other.
function PreformCutPlane({ origin, normal, span, verified }) {
  const quaternion = useMemo(() => new THREE.Quaternion().setFromUnitVectors(
    new THREE.Vector3(0, 0, 1), new THREE.Vector3(...normal).normalize(),
  ), [normal]);
  return (
    <mesh position={origin} quaternion={quaternion} raycast={NO_RAYCAST}>
      <planeGeometry args={[span, span]} />
      {verified ? (
        <meshBasicMaterial color="#ffca28" transparent opacity={0.28} side={THREE.DoubleSide} depthWrite={false} />
      ) : (
        <meshBasicMaterial color="#94a3b8" wireframe transparent opacity={0.45} depthWrite={false} />
      )}
    </mesh>
  );
}

function AutoSpinner({ isSpinning }) {
  useFrame((state) => {
    if (!isSpinning) return;
    const speed = 0.004;
    const x = state.camera.position.x;
    const z = state.camera.position.z;
    state.camera.position.x = x * Math.cos(speed) - z * Math.sin(speed);
    state.camera.position.z = x * Math.sin(speed) + z * Math.cos(speed);
    state.camera.lookAt(0, 0, 0);
  });
  return null;
}

// [state, full label, compact label for phone-width viewers]
const DEFECT_LEGEND = [
  ['candidate', 'AI/OpenCV candidate · provisional', 'AI candidate'],
  ['manual', 'Manual · not yet confirmed', 'Manual'],
  ['confirmed', 'Confirmed defect safety region', 'Confirmed'],
  ['rejected', 'Rejected (when shown)', 'Rejected'],
];

export default function ModelViewer({
  modelUrl, cutUrl, defectsUrl, gemUrls = [], gemDetails = [], manufacturingPlan = {},
  remainingSpace = null, remainingSpaceDiagnostic = null,
  activeStrategyName = null, activeGemCount = null,
  selectedGemIndex = null, onSelectGem = () => {},
  // Optional: parent-controlled viewer mode ('inspect' | 'defects' | 'sequence').
  viewerMode: controlledMode, onViewerModeChange,
  // Optional Defect Review overlay state (see ResultDashboard). Carries the
  // review's own coordinate frame; the legacy report scale is never used.
  defectReview = null,
  // Optional Preform Recovery overlay state (see ResultDashboard).
  preform = null,
  // Optional Expert Review state: backend physical leaf pieces (with a
  // safely resolved mesh URL and review state) plus the run's frame.
  expertReview = null,
  // Optional: label shown on the Cut Sequence panel (e.g. when the plan is
  // the defect-aware placement rather than the original optimization).
  sequenceLabel = null,
  // Optional: keep confirmed defect safety regions visible in Inspect and
  // Cut Sequence alongside the gems (defect-aware placement).
  showConfirmedDefects = false,
  // Optional: coordinate frame (readCoordinateFrame) of centered-frame gem
  // meshes and cut planes, e.g. a defect-aware result. They are offset once
  // into the rough's frame like the other overlays. Null = legacy behaviour.
  gemFrame = null,
}) {
  const [internalMode, setInternalMode] = useState('inspect');
  const requestedMode = controlledMode ?? internalMode;
  const setViewerMode = (mode) => {
    setInternalMode(mode);
    if (onViewerModeChange) onViewerModeChange(mode);
  };

  // Legacy manufacturing-plan scale — used only by legacy gem/sequence UI.
  const mmPerMesh = Number(manufacturingPlan?.settings?.mm_per_mesh_unit) || 0;
  const preformActive = Boolean(preform?.active);
  const preformResult = preformActive ? preform.result : null;

  // Which frame the loaded rough is in, and where centered-frame overlays
  // must sit inside it (applied exactly once, never to the rough itself).
  const meshFrame = viewerMeshFrame(modelUrl);
  const reviewFrame = defectReview?.frame || null;
  const resultFrame = preformResult?.frame || null;
  const defectOffset = overlayOffset(meshFrame, reviewFrame);
  const preformOffset = overlayOffset(meshFrame, resultFrame);
  const expertOffset = overlayOffset(meshFrame, expertReview?.frame || null);
  const gemOffset = gemFrame ? overlayOffset(meshFrame, gemFrame) : [0, 0, 0];

  // In Preform Recovery mode the Cut Sequence steps through the result's
  // selected_verified cuts only (already ordered by `sequence`).
  const preformSequence = useMemo(() => {
    if (!preformResult || !resultFrame) return [];
    return preformResult.selectedCuts.map((cut) => {
      const plane = cutPlaneCentered(cut, resultFrame);
      if (!plane) return null;
      const kerf = Number(cut.kerf_mm);
      return {
        step: cut.sequence,
        parent_piece_id: cut.cut_id,
        plane,
        kerf_slab: {
          thickness_mm: cut.kerf_mm,
          thickness_mesh_units: Number.isFinite(kerf) && kerf > 0 ? kerf / resultFrame.mmPerMesh : undefined,
        },
        required_depth_mm: cut.required_depth_mm,
        hide_feed_direction: true,
        section_contour: [],
      };
    }).filter(Boolean);
  }, [preformResult, resultFrame]);

  const legacySequence = Array.isArray(manufacturingPlan?.sequence)
    ? manufacturingPlan.sequence : [];
  const sequence = preformActive ? preformSequence : legacySequence;
  const canInspectSequence = preformActive
    ? sequence.length > 0
    : manufacturingPlan?.status === 'complete' && sequence.length > 0;
  const viewerMode = requestedMode === 'sequence' && !canInspectSequence ? 'inspect' : requestedMode;

  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAutoRotating, setIsAutoRotating] = useState(true);
  const [brightness, setBrightness] = useState(1.0);
  const [showCut, setShowCut] = useState(true);
  const [showWireframe, setShowWireframe] = useState(false);
  const [isXRay, setIsXRay] = useState(false);
  const [showRemainingSpace, setShowRemainingSpace] = useState(false);
  const [showCandidateCuts, setShowCandidateCuts] = useState(false);
  const [roughRadius, setRoughRadius] = useState(0);

  const defectMode = viewerMode === 'defects';
  const expertMode = viewerMode === 'expert';
  const sequenceMode = viewerMode === 'sequence' && canInspectSequence;
  const inspectMode = !defectMode && !sequenceMode && !expertMode;
  const expertPieces = expertMode && expertOffset && Array.isArray(expertReview?.pieces)
    ? expertReview.pieces.filter((piece) => piece.resolvedUrl)
    : [];
  const legacyInspect = inspectMode && !preformActive;
  const activeStep = sequence[stepIndex] || null;
  const preformMargin = Number(manufacturingPlan?.settings?.preform_margin_mm) || 0;
  const marginMesh = mmPerMesh > 0 ? preformMargin / mmPerMesh : 0;
  const remainingComponent = remainingSpace?.components?.[0] || null;

  // Individual gem PLYs exist for this strategy — render each gem
  // separately (selectable) instead of the single combined cut mesh.
  // Falls back to the combined CutGem below when none are available.
  const inspectableGems = (Array.isArray(gemDetails) ? gemDetails : [])
    .filter((gem) => gem?.url);
  const hasIndividualGems = inspectableGems.length > 0;
  const selectedGem = (Array.isArray(gemDetails) ? gemDetails : [])
    .find((gem) => gem?.index === selectedGemIndex) || null;

  const regions = preformResult?.regions || [];
  const selectedRegionIndex = regions.findIndex((region) => region.region_id === preform?.selectedRegionId);
  const selectedRegion = selectedRegionIndex >= 0 ? regions[selectedRegionIndex] : null;

  // Defect overlays: everything under review in Defect Review mode; only
  // confirmed safety regions alongside preform regions in Inspect.
  const draft = defectReview?.draft || null;
  const defectItems = defectReview
    ? [...(defectReview.candidates || []), ...(defectReview.annotations || [])]
      .map((item, index) => ({ id: itemId(item) ?? `idx-${index}`, item, state: visualStateOf(item) }))
      .filter(({ state }) => (defectMode
        ? (defectReview.showRejected || state !== 'rejected')
        : state === 'confirmed'))
    : [];
  const showDefectOverlays = Boolean(defectOffset) && (
    defectMode || (inspectMode && preformActive)
    || (showConfirmedDefects && !preformActive && (inspectMode || sequenceMode))
  );
  const canPlace = Boolean(reviewFrame && defectOffset);
  const pickingActive = Boolean(
    defectMode && draft?.placing && canPlace
    && !(draft.kind === 'tube_polyline' && draft.finished),
  );
  // Picked point → rough-mesh local (worldToLocal already undid the group
  // rotation) → centered → mm, via the review's own frame.
  const handleRoughPick = ({ pointMesh, normalMesh }) => {
    const pointMm = pickedLocalToMm(pointMesh, meshFrame, reviewFrame);
    if (pointMm) defectReview.onPick({ pointMm, normal: normalMesh });
  };
  const draftPoints = defectMode && draft?.kind === 'tube_polyline'
    ? draft.points.map((point) => point.map(Number)).filter((point) => point.every(Number.isFinite))
    : null;

  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
  }, [manufacturingPlan, preformResult, preformActive, canInspectSequence]);

  useEffect(() => {
    if (!isPlaying || !sequenceMode) return undefined;
    const timer = window.setInterval(() => {
      setStepIndex((current) => {
        if (current >= sequence.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1800);
    return () => window.clearInterval(timer);
  }, [isPlaying, sequenceMode, sequence.length]);

  const modeButton = (mode, label, activeClass, enabled = true) => (
    <button
      type="button"
      onClick={() => enabled && setViewerMode(mode)}
      disabled={!enabled}
      aria-pressed={viewerMode === mode}
      className={`px-2.5 sm:px-3 py-1.5 text-[11px] sm:text-xs rounded-md whitespace-nowrap disabled:text-slate-700 ${viewerMode === mode ? activeClass : 'text-slate-400'}`}
    >{label}</button>
  );

  const showingLabel = preformActive
    ? `Preform Recovery${preformResult ? ` · ${regions.length} region${regions.length === 1 ? '' : 's'}` : ' · no result yet'}`
    : activeStrategyName;
  // Stray reconstruction fragments inflate the bounding sphere, so keep the
  // plane indicator modest rather than spanning the full sphere.
  const cutPlaneSpan = roughRadius > 0 ? roughRadius * 1.3 : 1.2;

  return (
    <div className="w-full h-full min-h-[420px] bg-gradient-to-b from-slate-900 to-black relative group">
      <div className="absolute top-4 left-4 z-20 flex max-w-[calc(100%-5.5rem)] overflow-x-auto bg-slate-900/90 p-1 border border-slate-700 rounded-lg">
        {modeButton('inspect', 'Inspect', 'bg-cyan-600 text-white')}
        {modeButton('defects', 'Defect Review', 'bg-red-600 text-white')}
        {modeButton('expert', 'Expert Review', 'bg-emerald-600 text-white')}
        {modeButton('sequence', 'Cut Sequence', 'bg-amber-500 text-slate-950', canInspectSequence)}
      </div>

      {expertMode && (
        <div className="absolute top-16 left-4 z-20 max-w-[calc(100%-5.5rem)] rounded-lg border border-slate-700 bg-slate-950/90 p-2 text-[10px] leading-4 text-slate-300 sm:w-64 sm:p-2.5">
          <div className="mb-1 hidden font-semibold text-white sm:block">Expert Review · physical pieces</div>
          <ul className="flex flex-wrap gap-x-2.5 gap-y-0.5 sm:block sm:space-y-0.5">
            {EXPERT_LEGEND.map((state) => (
              <li key={state} className="flex items-center gap-1.5">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: PIECE_STATES[state].color }} aria-hidden="true" />
                <span className="min-w-0 break-words">{PIECE_STATES[state].label}</span>
              </li>
            ))}
          </ul>
          {!expertReview?.pieces?.length ? (
            <p className="mt-1 text-slate-500">No reviewable physical pieces loaded.</p>
          ) : !expertOffset ? (
            <p className="mt-1 text-amber-300">Piece geometry cannot be placed: the run's coordinate frame does not match this viewer mesh.</p>
          ) : null}
        </div>
      )}

      {!defectMode && !expertMode && showingLabel && (
        <div
          className="absolute top-16 left-4 z-20 max-w-[calc(100%-5.5rem)] break-words bg-slate-900/90 border border-slate-700 rounded-lg px-3 py-1.5 text-[11px] text-slate-300"
          title="The plan currently rendered here always matches the sidebar's selection."
        >
          Showing: <span className="text-white font-semibold">{showingLabel}</span>
          {!preformActive && activeGemCount ? (
            <span className="text-slate-400"> · {activeGemCount} gem{activeGemCount > 1 ? 's' : ''}</span>
          ) : null}
        </div>
      )}

      {defectMode && (
        <div className="absolute top-16 left-4 z-20 max-w-[calc(100%-5.5rem)] rounded-lg border border-slate-700 bg-slate-950/90 p-2 text-[10px] leading-4 text-slate-300 sm:w-64 sm:p-2.5">
          <div className="mb-1 hidden font-semibold text-white sm:block">Defect Review</div>
          <ul className="flex flex-wrap gap-x-2.5 gap-y-0.5 sm:block sm:space-y-0.5">
            {DEFECT_LEGEND.map(([state, label, compact]) => (
              <li key={state} className="flex items-center gap-1.5">
                <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: VISUAL_STATES[state].color }} aria-hidden="true" />
                <span className="min-w-0 break-words sm:hidden">{compact}</span>
                <span className="hidden min-w-0 break-words sm:inline">{label}</span>
              </li>
            ))}
          </ul>
          <p className="mt-1 text-slate-500">
            <span className="sm:hidden">Approximate safety regions.</span>
            <span className="hidden sm:inline">Shapes are approximate safety regions, not exact internal reconstructions.</span>
          </p>
          {pickingActive && (
            <p className="mt-1.5 flex items-start gap-1 font-semibold text-yellow-200" role="status">
              <Crosshair className="mt-0.5 h-3 w-3 shrink-0" />
              {draft.kind === 'tube_polyline'
                ? `Click the model to add fracture points · ${draft.points.length} placed`
                : 'Click the reconstructed model to place the centre'}
            </p>
          )}
          {defectReview && !canPlace && (
            <p className="mt-1.5 text-amber-300">
              3D placement and overlays unavailable: no calibrated defect-review coordinate frame for this viewer mesh. Enter coordinates in the sidebar.
            </p>
          )}
        </div>
      )}

      <div className="absolute top-4 right-4 z-20 flex flex-col gap-2">
        {cutUrl && legacyInspect && (
          <button
            onClick={() => setShowCut(!showCut)}
            title={showCut
              ? (hasIndividualGems ? 'Hide gems' : 'Hide cut plan')
              : (hasIndividualGems ? 'Show gems' : 'Show cut plan')}
            className="w-11 h-11 grid place-items-center bg-slate-800/90 border border-slate-600 rounded-lg text-white hover:bg-cyan-600"
          ><Box className="w-4 h-4" /></button>
        )}
        <button
          onClick={() => setShowWireframe(!showWireframe)}
          title={showWireframe ? 'Hide wireframe' : 'Show wireframe'}
          className={`w-11 h-11 grid place-items-center border rounded-lg ${showWireframe ? 'bg-blue-600 border-blue-400 text-white' : 'bg-slate-800/90 border-slate-600 text-slate-300'}`}
        ><Grid className="w-4 h-4" /></button>
        {cutUrl && legacyInspect && (
          <button
            onClick={() => setIsXRay(!isXRay)}
            title={isXRay ? 'Use solid shell' : 'Use X-ray shell'}
            className={`w-11 h-11 grid place-items-center border rounded-lg ${isXRay ? 'bg-cyan-600 border-cyan-400 text-white' : 'bg-slate-800/90 border-slate-600 text-slate-300'}`}
          >{isXRay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}</button>
        )}
        {remainingComponent && legacyInspect && (
          <button
            onClick={() => setShowRemainingSpace((value) => !value)}
            title={showRemainingSpace ? 'Hide remaining geometric space (diagnostic)' : 'Show remaining geometric space (diagnostic)'}
            className={`w-11 h-11 grid place-items-center border rounded-lg ${showRemainingSpace ? 'bg-cyan-600 border-cyan-400 text-white' : 'bg-slate-800/90 border-slate-600 text-slate-300'}`}
          ><ScanSearch className="w-4 h-4" /></button>
        )}
        {preformActive && inspectMode && preformResult?.comparisonCuts.length > 0 && (
          <button
            onClick={() => setShowCandidateCuts((value) => !value)}
            title={showCandidateCuts ? 'Hide geometric comparison cuts' : 'Show geometric comparison cuts (not in the recommended plan)'}
            aria-pressed={showCandidateCuts}
            className={`w-11 h-11 grid place-items-center border rounded-lg ${showCandidateCuts ? 'bg-slate-500 border-slate-300 text-white' : 'bg-slate-800/90 border-slate-600 text-slate-300'}`}
          ><Scissors className="w-4 h-4" /></button>
        )}
      </div>

      {showRemainingSpace && remainingComponent && legacyInspect && (
        <div className="absolute top-16 left-4 z-20 max-w-xs border border-cyan-500/40 bg-slate-950/90 p-3 rounded-lg">
          <div className="text-xs font-semibold text-cyan-300">
            {remainingSpaceDiagnostic?.heading || 'Remaining Geometric Space'}
          </div>
          <div className="mt-1 inline-flex items-center gap-1 rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-amber-300">
            Diagnostic only · not a gemstone candidate
          </div>
          {remainingSpaceDiagnostic?.superseded && (
            <div className="mt-1 inline-flex items-center gap-1 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-emerald-300">
              Superseded by improved verified result
            </div>
          )}
          <p className="mt-2 text-[10px] leading-4 text-slate-400">
            This area represents available geometric space. Additional gems
            require candidate fitting and manufacturing verification.
          </p>
          <div className="mt-2 text-[10px] leading-4 text-slate-400">
            {remainingSpaceDiagnostic?.message
              || remainingComponent.rejection_reason
              || 'No verified saleable placement remained.'}
          </div>
          <div className="mt-1 text-[10px] text-slate-500">
            Inscribed estimate: {remainingComponent.estimated_inscribed_sphere_carat ?? '—'} ct
          </div>
        </div>
      )}

      {legacyInspect && (
        <SelectedGemPanel
          gem={selectedGem} mmPerMesh={mmPerMesh}
          onClear={() => onSelectGem(null)}
        />
      )}

      {inspectMode && preformActive && selectedRegion && (
        <PreformRegionDetails
          region={selectedRegion} color={regionColor(selectedRegionIndex)}
          onClear={() => preform.onSelectRegion(null)}
        />
      )}

      {inspectMode && preformActive && preformResult
        && (preformResult.selectedCuts.length + preformResult.comparisonCuts.length) > 0 && !selectedRegion && (
        <div className="absolute bottom-4 left-4 z-20 max-w-[calc(100%-2rem)] rounded-lg border border-slate-700 bg-slate-950/90 px-2.5 py-1.5 text-[10px] leading-4 text-slate-300">
          <span className="mr-2 inline-flex items-center gap-1"><span className="h-2 w-3 bg-amber-300/70" aria-hidden="true" /> Selected verified cut</span>
          <span className="inline-flex items-center gap-1"><span className="h-2 w-3 border border-slate-400" aria-hidden="true" /> Geometric comparison{showCandidateCuts ? '' : ' (hidden)'}</span>
        </div>
      )}
      {gemFrame && !gemOffset && (
        <div className="absolute bottom-16 left-4 z-20 max-w-[calc(100%-2rem)] rounded-lg border border-amber-500/40 bg-slate-950/90 px-2.5 py-1.5 text-[10px] leading-4 text-amber-300">
          Gem and cut geometry cannot be placed: the result's coordinate frame does not match this viewer mesh.
        </div>
      )}
      {preformActive && preformResult && !preformOffset && (
        <div className="absolute bottom-16 left-4 z-20 max-w-[calc(100%-2rem)] rounded-lg border border-amber-500/40 bg-slate-950/90 px-2.5 py-1.5 text-[10px] leading-4 text-amber-300">
          Region and cut geometry cannot be placed: the result's coordinate frame does not match this viewer mesh.
        </div>
      )}

      {sequenceMode && activeStep && (
        <div className="absolute left-4 right-4 bottom-4 z-20 bg-slate-950/92 border border-slate-700 rounded-lg p-3">
          {sequenceLabel && !preformActive && (
            <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">{sequenceLabel}</div>
          )}
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setIsPlaying(false); setStepIndex(0); }}
              title="Reset sequence"
              className="w-11 h-11 shrink-0 grid place-items-center bg-slate-800 border border-slate-700 rounded text-slate-300"
            ><RotateCcw className="w-4 h-4" /></button>
            <button
              onClick={() => setStepIndex((value) => Math.max(0, value - 1))}
              disabled={stepIndex === 0} title="Previous cut"
              className="w-11 h-11 shrink-0 grid place-items-center bg-slate-800 border border-slate-700 rounded text-slate-300 disabled:opacity-30"
            ><SkipBack className="w-4 h-4" /></button>
            <button
              onClick={() => setIsPlaying((value) => !value)}
              title={isPlaying ? 'Pause sequence' : 'Play sequence'}
              className="w-11 h-11 shrink-0 grid place-items-center bg-amber-500 rounded text-slate-950"
            >{isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}</button>
            <button
              onClick={() => setStepIndex((value) => Math.min(sequence.length - 1, value + 1))}
              disabled={stepIndex === sequence.length - 1} title="Next cut"
              className="w-11 h-11 shrink-0 grid place-items-center bg-slate-800 border border-slate-700 rounded text-slate-300 disabled:opacity-30"
            ><SkipForward className="w-4 h-4" /></button>
            <div className="min-w-0 ml-2">
              <div className="text-xs font-semibold text-white break-words">
                Cut {stepIndex + 1} of {sequence.length} · {activeStep.parent_piece_id}
              </div>
              <div className="text-[10px] text-slate-400 truncate">
                {preformActive
                  ? `Selected verified cut · Kerf ${activeStep.kerf_slab?.thickness_mm ?? '—'} mm · Depth ${activeStep.required_depth_mm ?? '—'} mm`
                  : `Kerf ${activeStep.kerf_slab?.thickness_mm ?? '—'} mm · Preform ${preformMargin || '—'} mm · Depth ${activeStep.required_depth_mm ?? '—'} mm`}
              </div>
            </div>
          </div>
          <div className="flex gap-1 mt-2" aria-label="Cut sequence timeline">
            {sequence.map((step, index) => (
              <button
                key={step.step}
                onClick={() => { setIsPlaying(false); setStepIndex(index); }}
                title={`Cut ${index + 1}: ${step.parent_piece_id}`}
                className={`h-1.5 flex-1 rounded-sm ${index === stepIndex ? 'bg-amber-400' : index < stepIndex ? 'bg-emerald-500' : 'bg-slate-700'}`}
              />
            ))}
          </div>
        </div>
      )}

      {!sequenceMode && (
        <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-20 flex items-center gap-3 bg-slate-900/85 px-4 py-2 rounded-lg border border-slate-700 opacity-0 group-hover:opacity-100 transition-opacity">
          <Sun className="w-4 h-4 text-yellow-400" />
          <input
            aria-label="Scene brightness" type="range" min="0.2" max="3.0"
            step="0.1" value={brightness}
            onChange={(event) => setBrightness(parseFloat(event.target.value))}
            className="w-36 accent-cyan-400"
          />
        </div>
      )}

      <Canvas
        camera={{ position: [0, 0, 5], fov: 45 }} dpr={[1, 2]}
        gl={{ localClippingEnabled: true, preserveDrawingBuffer: true }}
        onPointerMissed={() => {
          if (legacyInspect) onSelectGem(null);
          if (inspectMode && preformActive) preform.onSelectRegion(null);
          if (expertMode && expertReview) expertReview.onSelectPiece(null);
        }}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={2.8 * brightness} />
          <directionalLight position={[0, 10, 10]} intensity={2 * brightness} />
          <directionalLight position={[0, -10, -10]} intensity={1.6 * brightness} />
          <Bounds fit clip observe margin={1.25}>
            <group rotation={BACKEND_TO_VIEWER}>
              <RoughStone
                url={modelUrl}
                isXRay={defectMode || expertMode || (inspectMode && preformActive) || (showCut && cutUrl && isXRay)}
                showWireframe={showWireframe} sequenceMode={sequenceMode}
                onPick={pickingActive ? handleRoughPick : undefined}
                onBounds={setRoughRadius}
              />
              {/* Expert Review: backend physical leaf meshes (already centered), offset once. */}
              {expertPieces.length > 0 && (
                <group position={expertOffset}>
                  {expertPieces.map((piece) => {
                    const selected = expertReview.selectedPieceId === piece.piece_id;
                    return (
                      <SafeLoad key={piece.piece_id}>
                        <Suspense fallback={null}>
                          <RegionMesh
                            url={piece.resolvedUrl}
                            color={PIECE_STATES[piece.state]?.color ?? '#64748b'}
                            selected={selected}
                            muted={piece.state === 'waste_unusable' || piece.state === 'not_review_required'
                              || (expertReview.selectedPieceId != null && !selected)}
                            onSelect={() => expertReview.onSelectPiece(selected ? null : piece.piece_id)}
                          />
                        </Suspense>
                      </SafeLoad>
                    );
                  })}
                </group>
              )}
              {gemOffset && (<group position={gemOffset}>
              {legacyInspect && showCut && (
                hasIndividualGems ? (
                  inspectableGems.map((gem) => (
                    <InspectableGem
                      key={gem.file || gem.index}
                      gem={gem}
                      isSelected={selectedGemIndex === gem.index}
                      isMuted={selectedGemIndex !== null && selectedGemIndex !== gem.index}
                      showWireframe={showWireframe}
                      onSelect={onSelectGem}
                    />
                  ))
                ) : (
                  cutUrl && <CutGem url={cutUrl} showWireframe={showWireframe} />
                )
              )}
              {sequenceMode && !preformActive && gemUrls.map((url, index) => (
                <ProtectedGem
                  key={url} url={url} index={index}
                  activeStep={activeStep} marginMesh={marginMesh}
                />
              ))}
              {sequenceMode && activeStep && !preformActive && <CutOverlay step={activeStep} />}
              </group>)}
              {/* Preform regions/cuts: centered frame, offset once to the rough's frame. */}
              {preformResult && preformOffset && (inspectMode || sequenceMode) && (
                <group position={preformOffset}>
                  <PreformRegions
                    regions={regions}
                    selectedRegionId={inspectMode ? preform.selectedRegionId : null}
                    onSelectRegion={inspectMode ? preform.onSelectRegion : null}
                    muted={sequenceMode}
                  />
                  {inspectMode && [
                    ...preformResult.selectedCuts.map((cut) => [cut, true]),
                    ...(showCandidateCuts ? preformResult.comparisonCuts.map((cut) => [cut, false]) : []),
                  ].map(([cut, selected]) => {
                    const plane = cutPlaneCentered(cut, resultFrame);
                    if (!plane) return null;
                    return (
                      <PreformCutPlane
                        key={`${selected ? 's' : 'c'}-${cut.cut_id}`}
                        origin={plane.origin} normal={plane.normal}
                        span={cutPlaneSpan} verified={selected}
                      />
                    );
                  })}
                </group>
              )}
              {defectsUrl && <DefectCloud url={defectsUrl} />}
              {showDefectOverlays && (
                <group position={defectOffset}>
                  <DefectOverlays
                    items={defectItems}
                    draftGeometry={defectMode ? defectReview?.draftPreview : null}
                    draftPoints={draftPoints}
                    mmPerMesh={reviewFrame?.mmPerMesh}
                    selectedId={defectMode ? defectReview.selectedId : null}
                    onSelect={(id) => defectReview.onSelect(id)}
                    interactive={defectMode && !pickingActive}
                  />
                </group>
              )}
              {legacyInspect && showRemainingSpace && remainingComponent && (
                <RemainingSpaceEnvelope component={remainingComponent} />
              )}
              {sequenceMode && activeStep && preformActive && preformOffset && (
                <group position={preformOffset}><CutOverlay step={activeStep} /></group>
              )}
              {sequenceMode && <axesHelper args={[0.42]} />}
            </group>
          </Bounds>
          <ContactShadows
            resolution={512} scale={20} blur={2} opacity={0.45}
            far={10} color="#000000"
          />
          <AutoSpinner isSpinning={isAutoRotating && inspectMode} />
        </Suspense>
        <TrackballControls
          makeDefault noPan={false} rotateSpeed={3.5} zoomSpeed={1.2}
          onStart={() => setIsAutoRotating(false)}
        />
      </Canvas>
    </div>
  );
}
