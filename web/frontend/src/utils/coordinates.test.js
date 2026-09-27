import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  BACKEND_TO_VIEWER_ROTATION, canonicalToCentered, centeredToCanonical, centeredToMm,
  centeredToViewerLocal, mmToCentered, mmToRoughLocal, overlayOffset, pickedLocalToMm,
  readCoordinateFrame, viewerLocalToCentered, viewerMeshFrame,
} from './coordinates';
import { resolveBackendResource } from './backendUrl';

// Values from the PREFORM_RECOVERY.md §A example frame.
const FRAME = readCoordinateFrame({
  mm_per_mesh_unit: 20.0,
  canonical_to_centered_translation_mesh_units: [-8.0, 3.0, -12.0],
});

const close = (actual, expected) => actual.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 9));

describe('canonical → centered → mm', () => {
  it('reads the frame and rejects a missing scale', () => {
    expect(FRAME).toEqual({ mmPerMesh: 20, translation: [-8, 3, -12] });
    expect(readCoordinateFrame({ canonical_to_centered_translation_mesh_units: [0, 0, 0] })).toBeNull();
    expect(readCoordinateFrame(null)).toBeNull();
  });

  it('applies q = p + t then mm = s·q', () => {
    const p = [8.5, -2.0, 12.25];
    const q = canonicalToCentered(p, FRAME.translation);
    close(q, [0.5, 1.0, 0.25]);
    close(centeredToMm(q, FRAME.mmPerMesh), [10, 20, 5]);
  });

  it('inverts exactly: q = mm / s, p = q - t', () => {
    const mm = [10, 20, 5];
    const q = mmToCentered(mm, FRAME.mmPerMesh);
    close(q, [0.5, 1.0, 0.25]);
    close(centeredToCanonical(q, FRAME.translation), [8.5, -2.0, 12.25]);
  });
});

describe('viewer mesh frames', () => {
  it('identifies the centered derivative and the canonical fallback', () => {
    expect(viewerMeshFrame('http://h/files/j/dense/visual_aligned_stone.ply?t=1')).toBe('centered');
    expect(viewerMeshFrame('/files/j/dense/final_textured_model.ply')).toBe('canonical');
    expect(viewerMeshFrame('/files/j/dense/other.ply')).toBeNull();
  });

  it('offsets centered overlays by -t only for the canonical rough (once)', () => {
    expect(overlayOffset('centered', FRAME)).toEqual([0, 0, 0]);
    expect(overlayOffset('canonical', FRAME)).toEqual([8, -3, 12]);
    expect(overlayOffset('canonical', { mmPerMesh: 20, translation: null })).toBeNull();
    expect(overlayOffset(null, FRAME)).toBeNull();
  });

  it('converts a picked rough-local point to mm for both frames', () => {
    close(pickedLocalToMm([0.5, 1.0, 0.25], 'centered', FRAME), [10, 20, 5]);
    close(pickedLocalToMm([8.5, -2.0, 12.25], 'canonical', FRAME), [10, 20, 5]);
    close(mmToRoughLocal([10, 20, 5], 'canonical', FRAME), [8.5, -2.0, 12.25]);
    close(mmToRoughLocal([10, 20, 5], 'centered', FRAME), [0.5, 1.0, 0.25]);
  });
});

describe('inverse viewer → backend conversion', () => {
  it('documents the -π/2 X rotation used by the viewer group', () => {
    const q = [0.5, 1.0, 0.25];
    const rotated = new THREE.Vector3(...q).applyEuler(new THREE.Euler(...BACKEND_TO_VIEWER_ROTATION)).toArray();
    close(rotated, centeredToViewerLocal(q));
    close(viewerLocalToCentered(centeredToViewerLocal(q)), q);
  });

  it.each(['centered', 'canonical'])('recovers backend mm from a world-space hit (%s rough)', (meshFrame) => {
    // Same hierarchy as ModelViewer: rotated group → rough mesh (untransformed).
    const group = new THREE.Group();
    group.rotation.set(...BACKEND_TO_VIEWER_ROTATION);
    group.position.set(0.3, -0.2, 0.1); // any extra parent transform
    const rough = new THREE.Mesh();
    group.add(rough);
    group.updateMatrixWorld(true);

    const expectedMm = [10, 20, 5];
    const roughLocal = mmToRoughLocal(expectedMm, meshFrame, FRAME);
    const world = rough.localToWorld(new THREE.Vector3(...roughLocal));

    // What surfacePick does: worldToLocal undoes the group transform once.
    const picked = rough.worldToLocal(world.clone()).toArray();
    close(pickedLocalToMm(picked, meshFrame, FRAME), expectedMm);
  });

  it('places centered-frame overlays on the same world point as the rough', () => {
    const group = new THREE.Group();
    group.rotation.set(...BACKEND_TO_VIEWER_ROTATION);
    const rough = new THREE.Mesh();
    const overlay = new THREE.Group();
    overlay.position.set(...overlayOffset('canonical', FRAME));
    const marker = new THREE.Object3D();
    marker.position.set(...mmToCentered([10, 20, 5], FRAME.mmPerMesh));
    overlay.add(marker);
    group.add(rough, overlay);
    group.updateMatrixWorld(true);

    const onRough = rough.localToWorld(new THREE.Vector3(...mmToRoughLocal([10, 20, 5], 'canonical', FRAME)));
    const onOverlay = marker.getWorldPosition(new THREE.Vector3());
    close(onOverlay.toArray(), onRough.toArray());
  });
});

describe('resolveBackendResource', () => {
  it('resolves mesh_file against the backend base, not a legacy asset dir', () => {
    expect(resolveBackendResource('/files/JOB/preform_recovery/RUN/R1.ply', 'http://localhost:8000'))
      .toBe('http://localhost:8000/files/JOB/preform_recovery/RUN/R1.ply');
    expect(resolveBackendResource('/files/JOB/preform_recovery/RUN/R1.ply', 'http://localhost:8000/'))
      .toBe('http://localhost:8000/files/JOB/preform_recovery/RUN/R1.ply');
  });

  it('keeps a proxied base path prefix', () => {
    expect(resolveBackendResource('/files/J/images/f.jpg', '/api'))
      .toBe(`${window.location.origin}/api/files/J/images/f.jpg`);
  });

  it('rejects non-URL paths instead of requesting them', () => {
    expect(resolveBackendResource(null, 'http://b')).toBeNull();
    expect(resolveBackendResource('C:\\jobs\\R1.ply', 'http://b')).toBeNull();
    expect(resolveBackendResource('//evil.example/x', 'http://b')).toBeNull();
    expect(resolveBackendResource('R1.ply', 'http://b')).toBeNull();
    expect(resolveBackendResource('/../x', 'http://b/api')).toBeNull();
    expect(resolveBackendResource('/files/../../secret.ply', 'http://b')).toBeNull();
    expect(resolveBackendResource('/files/%2e%2e/secret.ply', 'http://b')).toBeNull();
    expect(resolveBackendResource('/files/./R1.ply', 'http://b')).toBeNull();
    expect(resolveBackendResource('/files/J/preform_recovery/RUN/R1..v2.ply', 'http://b'))
      .toBe('http://b/files/J/preform_recovery/RUN/R1..v2.ply');
  });
});
