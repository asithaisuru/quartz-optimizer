import React, { Suspense, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { Bounds, ContactShadows, Line, TrackballControls } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import {
  Box, Eye, EyeOff, Grid, Pause, Play, RotateCcw,
  ScanSearch, SkipBack, SkipForward, Sun, X,
} from 'lucide-react';
import * as THREE from 'three';
import MetricHelp from './MetricHelp';

const BACKEND_TO_VIEWER = [-Math.PI / 2, 0, 0];

function RoughStone({ url, isXRay, showWireframe, sequenceMode }) {
  const geometry = useLoader(PLYLoader, url);
  useLayoutEffect(() => { geometry.computeVertexNormals(); }, [geometry]);
  return (
    <group>
      <mesh geometry={geometry}>
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
      <DirectionArrow
        origin={origin} direction={step.feed_direction || [1, 0, 0]}
        length={overlay.span * 0.55} color="#32d6ff"
      />
    </group>
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

export default function ModelViewer({
  modelUrl, cutUrl, defectsUrl, gemUrls = [], gemDetails = [], manufacturingPlan = {},
  remainingSpace = null, remainingSpaceDiagnostic = null,
  activeStrategyName = null, activeGemCount = null,
  selectedGemIndex = null, onSelectGem = () => {},
}) {
  const sequence = Array.isArray(manufacturingPlan?.sequence)
    ? manufacturingPlan.sequence : [];
  const canInspectSequence = manufacturingPlan?.status === 'complete' && sequence.length > 0;
  const [viewerMode, setViewerMode] = useState('inspect');
  const [stepIndex, setStepIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isAutoRotating, setIsAutoRotating] = useState(true);
  const [brightness, setBrightness] = useState(1.0);
  const [showCut, setShowCut] = useState(true);
  const [showWireframe, setShowWireframe] = useState(false);
  const [isXRay, setIsXRay] = useState(false);
  const [showRemainingSpace, setShowRemainingSpace] = useState(false);

  const sequenceMode = viewerMode === 'sequence' && canInspectSequence;
  const activeStep = sequence[stepIndex] || null;
  const mmPerMesh = Number(manufacturingPlan?.settings?.mm_per_mesh_unit) || 0;
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

  useEffect(() => {
    setStepIndex(0);
    setIsPlaying(false);
    if (!canInspectSequence) setViewerMode('inspect');
  }, [manufacturingPlan, canInspectSequence]);

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

  return (
    <div className="w-full h-full min-h-[420px] bg-gradient-to-b from-slate-900 to-black relative group">
      <div className="absolute top-4 left-4 z-20 flex bg-slate-900/90 p-1 border border-slate-700 rounded-lg">
        <button
          onClick={() => setViewerMode('inspect')}
          className={`px-3 py-1.5 text-xs rounded-md ${viewerMode === 'inspect' ? 'bg-cyan-600 text-white' : 'text-slate-400'}`}
        >Inspect</button>
        <button
          onClick={() => canInspectSequence && setViewerMode('sequence')}
          disabled={!canInspectSequence}
          className={`px-3 py-1.5 text-xs rounded-md disabled:text-slate-700 ${viewerMode === 'sequence' ? 'bg-amber-500 text-slate-950' : 'text-slate-400'}`}
        >Cut Sequence</button>
      </div>

      {activeStrategyName && (
        <div
          className="absolute top-16 left-4 z-20 bg-slate-900/90 border border-slate-700 rounded-lg px-3 py-1.5 text-[11px] text-slate-300"
          title="The cut strategy currently rendered here always matches the sidebar's selection."
        >
          Showing: <span className="text-white font-semibold">{activeStrategyName}</span>
          {activeGemCount ? (
            <span className="text-slate-400"> · {activeGemCount} gem{activeGemCount > 1 ? 's' : ''}</span>
          ) : null}
        </div>
      )}

      <div className="absolute top-4 right-4 z-20 flex flex-col gap-2">
        {cutUrl && !sequenceMode && (
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
        {cutUrl && (
          <button
            onClick={() => setIsXRay(!isXRay)}
            title={isXRay ? 'Use solid shell' : 'Use X-ray shell'}
            className={`w-11 h-11 grid place-items-center border rounded-lg ${isXRay ? 'bg-cyan-600 border-cyan-400 text-white' : 'bg-slate-800/90 border-slate-600 text-slate-300'}`}
          >{isXRay ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}</button>
        )}
        {remainingComponent && !sequenceMode && (
          <button
            onClick={() => setShowRemainingSpace((value) => !value)}
            title={showRemainingSpace ? 'Hide remaining geometric space (diagnostic)' : 'Show remaining geometric space (diagnostic)'}
            className={`w-11 h-11 grid place-items-center border rounded-lg ${showRemainingSpace ? 'bg-cyan-600 border-cyan-400 text-white' : 'bg-slate-800/90 border-slate-600 text-slate-300'}`}
          ><ScanSearch className="w-4 h-4" /></button>
        )}
      </div>

      {showRemainingSpace && remainingComponent && !sequenceMode && (
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

      {!sequenceMode && (
        <SelectedGemPanel
          gem={selectedGem} mmPerMesh={mmPerMesh}
          onClear={() => onSelectGem(null)}
        />
      )}

      {sequenceMode && activeStep && (
        <div className="absolute left-4 right-4 bottom-4 z-20 bg-slate-950/92 border border-slate-700 rounded-lg p-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setIsPlaying(false); setStepIndex(0); }}
              title="Reset sequence"
              className="w-11 h-11 grid place-items-center bg-slate-800 border border-slate-700 rounded text-slate-300"
            ><RotateCcw className="w-4 h-4" /></button>
            <button
              onClick={() => setStepIndex((value) => Math.max(0, value - 1))}
              disabled={stepIndex === 0} title="Previous cut"
              className="w-11 h-11 grid place-items-center bg-slate-800 border border-slate-700 rounded text-slate-300 disabled:opacity-30"
            ><SkipBack className="w-4 h-4" /></button>
            <button
              onClick={() => setIsPlaying((value) => !value)}
              title={isPlaying ? 'Pause sequence' : 'Play sequence'}
              className="w-11 h-11 grid place-items-center bg-amber-500 rounded text-slate-950"
            >{isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}</button>
            <button
              onClick={() => setStepIndex((value) => Math.min(sequence.length - 1, value + 1))}
              disabled={stepIndex === sequence.length - 1} title="Next cut"
              className="w-11 h-11 grid place-items-center bg-slate-800 border border-slate-700 rounded text-slate-300 disabled:opacity-30"
            ><SkipForward className="w-4 h-4" /></button>
            <div className="min-w-0 ml-2">
              <div className="text-xs font-semibold text-white">
                Cut {stepIndex + 1} of {sequence.length} · {activeStep.parent_piece_id}
              </div>
              <div className="text-[10px] text-slate-400 truncate">
                Kerf {activeStep.kerf_slab?.thickness_mm ?? '—'} mm · Preform {preformMargin || '—'} mm · Depth {activeStep.required_depth_mm ?? '—'} mm
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
        onPointerMissed={() => { if (!sequenceMode) onSelectGem(null); }}
      >
        <Suspense fallback={null}>
          <ambientLight intensity={2.8 * brightness} />
          <directionalLight position={[0, 10, 10]} intensity={2 * brightness} />
          <directionalLight position={[0, -10, -10]} intensity={1.6 * brightness} />
          <Bounds fit clip observe margin={1.25}>
            <group rotation={BACKEND_TO_VIEWER}>
              <RoughStone
                url={modelUrl} isXRay={showCut && cutUrl && isXRay}
                showWireframe={showWireframe} sequenceMode={sequenceMode}
              />
              {!sequenceMode && showCut && (
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
              {sequenceMode && gemUrls.map((url, index) => (
                <ProtectedGem
                  key={url} url={url} index={index}
                  activeStep={activeStep} marginMesh={marginMesh}
                />
              ))}
              {defectsUrl && <DefectCloud url={defectsUrl} />}
              {!sequenceMode && showRemainingSpace && remainingComponent && (
                <RemainingSpaceEnvelope component={remainingComponent} />
              )}
              {sequenceMode && activeStep && <CutOverlay step={activeStep} />}
              {sequenceMode && <axesHelper args={[0.42]} />}
            </group>
          </Bounds>
          <ContactShadows
            resolution={512} scale={20} blur={2} opacity={0.45}
            far={10} color="#000000"
          />
          <AutoSpinner isSpinning={isAutoRotating && !sequenceMode} />
        </Suspense>
        <TrackballControls
          makeDefault noPan={false} rotateSpeed={3.5} zoomSpeed={1.2}
          onStart={() => setIsAutoRotating(false)}
        />
      </Canvas>
    </div>
  );
}
