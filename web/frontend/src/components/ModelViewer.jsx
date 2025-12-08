import React, { Suspense, useState } from 'react';
import { Canvas, useLoader, useFrame } from '@react-three/fiber';
import { TrackballControls, Center, Environment, ContactShadows } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import { Sun, RotateCw, Box } from 'lucide-react';
import * as THREE from 'three';

function RoughStone({ url, isXRay }) {
  const geometry = useLoader(PLYLoader, url);
  geometry.computeVertexNormals();

  if (isXRay) {
    return (
      <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
        <meshPhysicalMaterial
          color="#ffffff" transmission={0.6} opacity={0.3} transparent={true}
          roughness={0.1} metalness={0.1} side={THREE.DoubleSide}
        />
      </mesh>
    );
  }
  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial vertexColors={true} side={THREE.DoubleSide} roughness={0.4} />
    </mesh>
  );
}

function CutGem({ url }) {
  const geometry = useLoader(PLYLoader, url);
  geometry.computeVertexNormals();
  return (
    <mesh geometry={geometry}>
      <meshStandardMaterial color="#ff0055" roughness={0.1} metalness={0.8} />
    </mesh>
  );
}

function AutoSpinner({ isSpinning }) {
  useFrame((state) => {
    if (isSpinning) {
      const speed = 0.005;
      const camera = state.camera;
      const x = camera.position.x;
      const z = camera.position.z;
      camera.position.x = x * Math.cos(speed) - z * Math.sin(speed);
      camera.position.z = x * Math.sin(speed) + z * Math.cos(speed);
      camera.lookAt(0, 0, 0);
    }
  });
  return null;
}

export default function ModelViewer({ modelUrl, cutUrl }) {
  const [isAutoRotating, setIsAutoRotating] = useState(true);
  const [brightness, setBrightness] = useState(1.0);
  const [showCut, setShowCut] = useState(true);

  return (
    <div className="w-full h-full bg-gradient-to-b from-slate-900 to-black relative group">

      {cutUrl && (
        <button
          onClick={() => setShowCut(!showCut)}
          className="absolute top-6 right-6 z-20 flex items-center gap-2 px-4 py-2 bg-slate-800/80 backdrop-blur border border-slate-600 rounded-lg text-white hover:bg-cyan-600 transition-colors"
        >
          <Box className="w-4 h-4" />
          {showCut ? "Hide Cut Plan" : "Show Cut Plan"}
        </button>
      )}

      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20 flex items-center gap-4 bg-slate-900/80 backdrop-blur-md px-6 py-3 rounded-2xl border border-slate-700 shadow-2xl transition-opacity duration-300 opacity-0 group-hover:opacity-100">
        <Sun className="w-5 h-5 text-yellow-400" />
        <input
          type="range" min="0.2" max="3.0" step="0.1"
          value={brightness} onChange={(e) => setBrightness(parseFloat(e.target.value))}
          className="w-48 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
        />
      </div>

      <Canvas camera={{ position: [0, 0, 5], fov: 45 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          <ambientLight intensity={3.0 * brightness} />
          <directionalLight position={[0, 10, 10]} intensity={2.0 * brightness} />
          <directionalLight position={[0, -10, -10]} intensity={2.0 * brightness} />
          <Environment preset="studio" />

          {/* 
             ALIGNMENT FIX:
             We put BOTH items in ONE <Center> tag.
             This ensures they stay locked together relative to each other.
          */}
          <Center top>
            <group>
              <RoughStone url={modelUrl} isXRay={showCut && cutUrl} />
              {showCut && cutUrl && <CutGem url={cutUrl} />}
            </group>
          </Center>

          <ContactShadows resolution={512} scale={20} blur={2} opacity={0.5} far={10} color="#000000" />
          <AutoSpinner isSpinning={isAutoRotating} />
        </Suspense>

        <TrackballControls makeDefault noPan={true} rotateSpeed={4.0} zoomSpeed={1.2} onStart={() => setIsAutoRotating(false)} />
      </Canvas>
    </div>
  );
}