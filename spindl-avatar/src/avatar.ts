import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRM, VRMUtils } from '@pixiv/three-vrm';

let vrm: VRM | null = null;

export function getVRM(): VRM | null {
  return vrm;
}

export async function loadAvatar(scene: THREE.Scene, url: string): Promise<VRM> {
  if (vrm) {
    scene.remove(vrm.scene);
    VRMUtils.deepDispose(vrm.scene);
    vrm = null;
  }

  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  return new Promise((resolve, reject) => {
    loader.load(
      url,
      (gltf) => {
        const model = gltf.userData.vrm as VRM;
        if (!model) {
          reject(new Error('No VRM data found in GLTF'));
          return;
        }

        VRMUtils.removeUnnecessaryVertices(model.scene);
        VRMUtils.combineSkeletons(model.scene);
        VRMUtils.rotateVRM0(model);

        model.scene.traverse((obj) => {
          if ((obj as THREE.Mesh).isMesh) {
            obj.frustumCulled = false;
          }
        });

        scene.add(model.scene);
        vrm = model;
        setupRestPose(model);
        enhanceMToonMaterials(model);

        if (import.meta.env.DEV) {
          const sbm = model.springBoneManager;
          const joints = sbm?.joints?.size ?? 0;
          console.log(`[SpindL] Spring bones: ${joints} joints`);
          const exprs = model.expressionManager?.expressions?.map(e => e.expressionName) ?? [];
          console.log(`[SpindL] Expressions: ${exprs.join(', ')}`);
          const humanoid = model.humanoid;
          if (humanoid) {
            const boneNames = [
              'hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
              'leftEye', 'rightEye', 'jaw',
              'leftShoulder', 'rightShoulder',
              'leftUpperArm', 'rightUpperArm', 'leftLowerArm', 'rightLowerArm',
              'leftHand', 'rightHand',
            ];
            const available = boneNames.filter(b => humanoid.getNormalizedBoneNode(b as any));
            console.log(`[SpindL] Bones available: ${available.join(', ')}`);
          }
          console.log('[SpindL] VRM loaded:', url);
        }
        resolve(model);
      },
      (progress) => {
        if (import.meta.env.DEV) {
          const pct = progress.total > 0 ? (progress.loaded / progress.total * 100).toFixed(0) : '?';
          console.log(`[SpindL] Loading VRM: ${pct}%`);
        }
      },
      (error) => reject(error)
    );
  });
}

// Rest pose values — detected from model's bind pose on load
export let REST_ARM_Z = 1.1;
export let REST_FOREARM_Z = 0.2;

function setupRestPose(model: VRM): void {
  const humanoid = model.humanoid;
  if (!humanoid) return;

  const leftUpperArm = humanoid.getNormalizedBoneNode('leftUpperArm');
  const rightUpperArm = humanoid.getNormalizedBoneNode('rightUpperArm');
  const leftLowerArm = humanoid.getNormalizedBoneNode('leftLowerArm');
  const rightLowerArm = humanoid.getNormalizedBoneNode('rightLowerArm');

  const TARGET_Z = 1.1;
  if (leftUpperArm) {
    const bindZ = leftUpperArm.rotation.z;
    REST_ARM_Z = bindZ === 0 ? TARGET_Z : TARGET_Z;
    if (Math.abs(bindZ) > 0.3) {
      REST_ARM_Z = bindZ + (TARGET_Z - Math.abs(bindZ));
    }
    if (import.meta.env.DEV) console.log(`[SpindL] Arm bind pose Z: ${bindZ.toFixed(3)}, rest target: ${REST_ARM_Z.toFixed(3)}`);
  }

  if (leftUpperArm) leftUpperArm.rotation.z = REST_ARM_Z;
  if (rightUpperArm) rightUpperArm.rotation.z = -REST_ARM_Z;

  const FOREARM_TARGET = 0.2;
  if (leftLowerArm) {
    const bindZ = leftLowerArm.rotation.z;
    REST_FOREARM_Z = Math.abs(bindZ) > 0.1 ? bindZ + (FOREARM_TARGET - Math.abs(bindZ)) : FOREARM_TARGET;
  }
  if (leftLowerArm) leftLowerArm.rotation.z = REST_FOREARM_Z;
  if (rightLowerArm) rightLowerArm.rotation.z = -REST_FOREARM_Z;
}

function enhanceMToonMaterials(model: VRM): void {
  let matCount = 0;
  model.scene.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.material) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const mat of materials) {
      if (!(mat as any).isMToonMaterial) continue;
      const mtoon = mat as any;
      matCount++;

      if (mtoon.parametricRimColorFactor) {
        mtoon.parametricRimColorFactor.setRGB(0.35, 0.33, 0.4);
      }
      if ('parametricRimFresnelPowerFactor' in mtoon) {
        mtoon.parametricRimFresnelPowerFactor = 3.5;
      }
      if ('parametricRimLiftFactor' in mtoon) {
        mtoon.parametricRimLiftFactor = 0.05;
      }
      if ('rimLightingMixFactor' in mtoon) {
        mtoon.rimLightingMixFactor = 0.5;
      }

      if (mtoon.shadeColorFactor) {
        const shade = mtoon.shadeColorFactor;
        shade.r = Math.min(1.0, shade.r * 1.15 + 0.03);
        shade.g = Math.min(1.0, shade.g * 1.05);
        shade.b = Math.max(0.0, shade.b * 0.92);
      }

      if ('shadingToonyFactor' in mtoon && mtoon.shadingToonyFactor > 0.85) {
        mtoon.shadingToonyFactor = 0.82;
      }

      mtoon.needsUpdate = true;
    }
  });
  if (import.meta.env.DEV) console.log(`[SpindL] Enhanced ${matCount} MToon materials`);
}

export function setExpression(name: string, value: number): void {
  if (!vrm?.expressionManager) return;
  vrm.expressionManager.setValue(name, value);
}

// -- Force-opaque for transparent window mode --
// MToon blend-mode materials (hair, accessories) write alpha < 1.0 which
// looks correct against a solid background but becomes see-through against
// a transparent desktop window. This swaps them to alphaTest (hard cutoff)
// so every model pixel is fully opaque.

interface SavedAlphaState {
  transparent: boolean;
  alphaTest: number;
  blending: THREE.Blending;
  depthWrite: boolean;
}

const savedAlphaStates = new WeakMap<THREE.Material, SavedAlphaState>();

export function setForceOpaqueMaterials(force: boolean): void {
  if (!vrm) return;
  vrm.scene.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.material) return;
    const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const mat of materials) {
      if (force) {
        // Save original state if not already saved
        if (!savedAlphaStates.has(mat)) {
          savedAlphaStates.set(mat, {
            transparent: mat.transparent,
            alphaTest: mat.alphaTest,
            blending: mat.blending,
            depthWrite: mat.depthWrite,
          });
        }
        // Force hard alpha cutoff — anything with alpha > 0.1 is fully opaque
        mat.transparent = false;
        mat.alphaTest = 0.1;
        mat.blending = THREE.NormalBlending;
        mat.depthWrite = true;
        mat.needsUpdate = true;
      } else {
        // Restore original state
        const saved = savedAlphaStates.get(mat);
        if (saved) {
          mat.transparent = saved.transparent;
          mat.alphaTest = saved.alphaTest;
          mat.blending = saved.blending;
          mat.depthWrite = saved.depthWrite;
          mat.needsUpdate = true;
        }
      }
    }
  });
}

export function updateAvatar(delta: number): void {
  if (!vrm) return;
  vrm.update(delta);
}
