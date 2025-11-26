import React, { Suspense, useState, useRef } from 'react';
import { Canvas, useLoader, useFrame } from '@react-three/fiber';
import { TrackballControls, Center, Environment, ContactShadows } from '@react-three/drei';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import * as THREE from 'three';

function Model({ url }) {
  const geometry = useLoader(PLYLoader, url);
  geometry.computeVertexNormals();

  return (
    <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
      <meshStandardMaterial 
        vertexColors={true} 
        side={THREE.DoubleSide} 
        flatShading={false}
        roughness={0.4} 
        metalness={0.1} 
      />
    </mesh>
  );
}

// --- CUSTOM AUTO-ROTATOR ---
// TrackballControls doesn't have 'autoRotate', so we spin the camera manually
function AutoSpinner({ isSpinning }) {
  useFrame((state) => {
    if (isSpinning) {
      // Rotate the camera around the center (0,0,0)
      const speed = 0.005;
      const camera = state.camera;
      
      // Simple circular rotation math
      const x = camera.position.x;
      const z = camera.position.z;
      
      camera.position.x = x * Math.cos(speed) - z * Math.sin(speed);
      camera.position.z = x * Math.sin(speed) + z * Math.cos(speed);
      camera.lookAt(0, 0, 0);
    }
  });
  return null;
}

export default function ModelViewer({ modelUrl }) {
  const [isAutoRotating, setIsAutoRotating] = useState(true);

  return (
    <div className="w-full h-full bg-gradient-to-b from-slate-900 to-black relative">
      
      {isAutoRotating && (
        <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 z-10 pointer-events-none text-white/30 text-xs font-mono bg-black/20 px-3 py-1 rounded-full backdrop-blur-sm">
          Click & Drag to Inspect (Free Rotation)
        </div>
      )}

      <Canvas camera={{ position: [0, 0, 5], fov: 45 }} dpr={[1, 2]}>
        <Suspense fallback={null}>
          
          <ambientLight intensity={3.0} />
          <directionalLight position={[0, 0, 10]} intensity={2.0} />
          <directionalLight position={[0, 0, -10]} intensity={2.0} />
          <directionalLight position={[10, 0, 0]} intensity={2.0} />
          <directionalLight position={[-10, 0, 0]} intensity={2.0} />
          <directionalLight position={[0, 10, 0]} intensity={2.0} />
          <directionalLight position={[0, -10, 0]} intensity={1.0} />

          <Environment preset="studio" />

          <Center top>
            <Model url={modelUrl} />
          </Center>

          <ContactShadows resolution={512} scale={20} blur={2} opacity={0.5} far={10} color="#000000" />

          {/* Inject the AutoSpinner logic */}
          <AutoSpinner isSpinning={isAutoRotating} />

        </Suspense>
        
        {/* 
           TRACKBALL CONTROLS:
           - noPan={true}: Keeps the object in the center (prevents dragging it off-screen).
           - rotateSpeed: Controls sensitivity.
           - dynamicDampingFactor: Makes it stop smoothly.
           - onStart: Disables our auto-spinner the moment you touch it.
        */}
        <TrackballControls 
          makeDefault 
          noPan={true} 
          rotateSpeed={4.0} 
          zoomSpeed={1.2}
          onStart={() => setIsAutoRotating(false)} 
        />
        
      </Canvas>
    </div>
  );
}