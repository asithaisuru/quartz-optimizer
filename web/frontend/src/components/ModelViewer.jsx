import React, { Suspense, useState, useLayoutEffect } from 'react';
import { Canvas, useLoader, useFrame } from '@react-three/fiber';
import { TrackballControls, ContactShadows, Environment } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import { Sun, Box, Grid } from 'lucide-react';
import * as THREE from 'three';

// Standard Glass Stone
function RoughStone({ url, isXRay, showWireframe }) {
  const geometry = useLoader(PLYLoader, url);
  useLayoutEffect(() => { geometry.computeVertexNormals(); }, [geometry]);

  return (
    <group>
      <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
        {isXRay ? (
          <meshPhysicalMaterial color="#ffffff" transmission={0.6} opacity={0.3} transparent={true} roughness={0.1} metalness={0.1} side={THREE.DoubleSide} />
        ) : (
          <meshStandardMaterial vertexColors={true} side={THREE.DoubleSide} roughness={0.4} />
        )}
      </mesh>
      {showWireframe && <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}><meshBasicMaterial color="#0088ff" wireframe={true} transparent={true} opacity={0.5} /></mesh>}
    </group>
  );
}

// Standard Red Gem
function CutGem({ url, showWireframe }) {
  const geometry = useLoader(PLYLoader, url);
  useLayoutEffect(() => { geometry.computeVertexNormals(); }, [geometry]);

  return (
    <group>
      {/* 
         FIXED: Added the same rotation as the RoughStone so it fits perfectly inside!
      */}
      <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
        <meshStandardMaterial color="#ff0055" roughness={0.1} metalness={0.8} />
      </mesh>
      {showWireframe && (
        <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
          <meshBasicMaterial color="#ffff00" wireframe={true} transparent={true} opacity={0.8} />
        </mesh>
      )}
    </group>
  );
}

function AutoSpinner({ isSpinning }) {
  useFrame((state) => {
    if (isSpinning) {
      const speed = 0.005;
      state.camera.position.x = state.camera.position.x * Math.cos(speed) - state.camera.position.z * Math.sin(speed);
      state.camera.position.z = state.camera.position.x * Math.sin(speed) + state.camera.position.z * Math.cos(speed);
      state.camera.lookAt(0, 0, 0);
    }
  });
  return null;
}

export default function ModelViewer({ modelUrl, cutUrl }) {
  const [isAutoRotating, setIsAutoRotating] = useState(true);
  const [brightness, setBrightness] = useState(1.0);
  const[showCut, setShowCut] = useState(true);
  const [showWireframe, setShowWireframe] = useState(false);

  return (
    <div className="w-full h-full bg-gradient-to-b from-slate-900 to-black relative group">
      
      {/* Controls UI */}
      <div className="absolute top-6 right-6 z-20 flex flex-col gap-2">
        {cutUrl && (
            <button onClick={() => setShowCut(!showCut)} className="flex items-center gap-2 px-4 py-2 bg-slate-800/80 backdrop-blur border border-slate-600 rounded-lg text-white hover:bg-cyan-600 transition-colors shadow-lg">
                <Box className="w-4 h-4" /> {showCut ? "Hide Cut Plan" : "Show Cut Plan"}
            </button>
        )}
        <button onClick={() => setShowWireframe(!showWireframe)} className={`flex items-center gap-2 px-4 py-2 backdrop-blur border rounded-lg transition-colors shadow-lg ${showWireframe ? 'bg-blue-600/80 border-blue-400 text-white' : 'bg-slate-800/80 border-slate-600 text-slate-300 hover:text-white'}`}>
            <Grid className="w-4 h-4" /> {showWireframe ? "Hide Blueprints" : "Show Blueprints"}
        </button>
      </div>
      
      {/* Brightness UI */}
      <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-20 flex items-center gap-4 bg-slate-900/80 backdrop-blur-md px-6 py-3 rounded-2xl border border-slate-700 shadow-2xl transition-opacity duration-300 opacity-0 group-hover:opacity-100">
        <Sun className="w-5 h-5 text-yellow-400" />
        <input type="range" min="0.2" max="3.0" step="0.1" value={brightness} onChange={(e) => setBrightness(parseFloat(e.target.value))} className="w-48 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400" />
      </div>

      <Canvas camera={{ position: [0, 0, 5], fov: 45 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          <ambientLight intensity={3.0 * brightness} />
          <directionalLight position={[0, 10, 10]} intensity={2.0 * brightness} />
          <directionalLight position={[0, -10, -10]} intensity={2.0 * brightness} />
          <Environment preset="studio" />

          {/* Group at (0,0,0) so files determine alignment */}
          <group position={[0,0,0]}>
             <RoughStone url={modelUrl} isXRay={showCut && cutUrl} showWireframe={showWireframe} />
             {showCut && cutUrl && <CutGem url={cutUrl} showWireframe={showWireframe} />}
          </group>

          <ContactShadows resolution={512} scale={20} blur={2} opacity={0.5} far={10} color="#000000" />
          <AutoSpinner isSpinning={isAutoRotating} />
        </Suspense>
        
        <TrackballControls makeDefault noPan={true} rotateSpeed={4.0} zoomSpeed={1.2} onStart={() => setIsAutoRotating(false)} />
      </Canvas>
    </div>
  );
}