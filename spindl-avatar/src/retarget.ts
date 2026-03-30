import * as THREE from 'three';
import type { VRM } from '@pixiv/three-vrm';
import type { VRMHumanBoneName } from '@pixiv/three-vrm';

/**
 * Mixamo rig name → VRM Humanoid bone name.
 * 56 entries covering full skeleton + fingers + toes.
 */
export const mixamoVRMRigMap: Record<string, VRMHumanBoneName> = {
  mixamorigHips: 'hips',
  mixamorigSpine: 'spine',
  mixamorigSpine1: 'chest',
  mixamorigSpine2: 'upperChest',
  mixamorigNeck: 'neck',
  mixamorigHead: 'head',
  mixamorigLeftShoulder: 'leftShoulder',
  mixamorigLeftArm: 'leftUpperArm',
  mixamorigLeftForeArm: 'leftLowerArm',
  mixamorigLeftHand: 'leftHand',
  mixamorigLeftHandThumb1: 'leftThumbMetacarpal',
  mixamorigLeftHandThumb2: 'leftThumbProximal',
  mixamorigLeftHandThumb3: 'leftThumbDistal',
  mixamorigLeftHandIndex1: 'leftIndexProximal',
  mixamorigLeftHandIndex2: 'leftIndexIntermediate',
  mixamorigLeftHandIndex3: 'leftIndexDistal',
  mixamorigLeftHandMiddle1: 'leftMiddleProximal',
  mixamorigLeftHandMiddle2: 'leftMiddleIntermediate',
  mixamorigLeftHandMiddle3: 'leftMiddleDistal',
  mixamorigLeftHandRing1: 'leftRingProximal',
  mixamorigLeftHandRing2: 'leftRingIntermediate',
  mixamorigLeftHandRing3: 'leftRingDistal',
  mixamorigLeftHandPinky1: 'leftLittleProximal',
  mixamorigLeftHandPinky2: 'leftLittleIntermediate',
  mixamorigLeftHandPinky3: 'leftLittleDistal',
  mixamorigRightShoulder: 'rightShoulder',
  mixamorigRightArm: 'rightUpperArm',
  mixamorigRightForeArm: 'rightLowerArm',
  mixamorigRightHand: 'rightHand',
  mixamorigRightHandPinky1: 'rightLittleProximal',
  mixamorigRightHandPinky2: 'rightLittleIntermediate',
  mixamorigRightHandPinky3: 'rightLittleDistal',
  mixamorigRightHandRing1: 'rightRingProximal',
  mixamorigRightHandRing2: 'rightRingIntermediate',
  mixamorigRightHandRing3: 'rightRingDistal',
  mixamorigRightHandMiddle1: 'rightMiddleProximal',
  mixamorigRightHandMiddle2: 'rightMiddleIntermediate',
  mixamorigRightHandMiddle3: 'rightMiddleDistal',
  mixamorigRightHandIndex1: 'rightIndexProximal',
  mixamorigRightHandIndex2: 'rightIndexIntermediate',
  mixamorigRightHandIndex3: 'rightIndexDistal',
  mixamorigRightHandThumb1: 'rightThumbMetacarpal',
  mixamorigRightHandThumb2: 'rightThumbProximal',
  mixamorigRightHandThumb3: 'rightThumbDistal',
  mixamorigLeftUpLeg: 'leftUpperLeg',
  mixamorigLeftLeg: 'leftLowerLeg',
  mixamorigLeftFoot: 'leftFoot',
  mixamorigLeftToeBase: 'leftToes',
  mixamorigRightUpLeg: 'rightUpperLeg',
  mixamorigRightLeg: 'rightLowerLeg',
  mixamorigRightFoot: 'rightFoot',
  mixamorigRightToeBase: 'rightToes',
};

/**
 * Retarget a Mixamo FBX animation clip onto a VRM skeleton.
 *
 * Quaternion tracks get rest-pose-corrected so the motion transfers
 * correctly regardless of bind pose differences. Position tracks
 * (hips translation) are scaled by the VRM-to-Mixamo hips height ratio.
 *
 * @param vrm  The loaded VRM model
 * @param asset  The loaded FBX asset (THREE.Group from FBXLoader)
 * @returns A new AnimationClip named "vrmAnimation" targeting VRM bone nodes
 */
export function remapMixamoAnimationToVrm(
  vrm: VRM,
  asset: THREE.Group,
): THREE.AnimationClip {
  const sourceClip = THREE.AnimationClip.findByName(asset.animations, 'mixamo.com');
  if (!sourceClip) {
    throw new Error('No "mixamo.com" clip found in FBX — is this a Mixamo animation?');
  }
  const clip = sourceClip.clone();

  const tracks: THREE.KeyframeTrack[] = [];

  const restRotationInverse = new THREE.Quaternion();
  const parentRestWorldRotation = new THREE.Quaternion();
  const _quatA = new THREE.Quaternion();
  const _vec3 = new THREE.Vector3();

  // Hips height scaling
  const motionHipsNode = asset.getObjectByName('mixamorigHips');
  if (!motionHipsNode) {
    throw new Error('Mixamo FBX missing mixamorigHips — is this a Mixamo animation?');
  }
  const motionHipsHeight = motionHipsNode.position.y;
  const vrmHipsY = vrm.humanoid
    ?.getNormalizedBoneNode('hips')
    ?.getWorldPosition(_vec3).y ?? 0;
  const vrmRootY = vrm.scene.getWorldPosition(_vec3).y;
  const vrmHipsHeight = Math.abs(vrmHipsY - vrmRootY);
  const hipsPositionScale = vrmHipsHeight / motionHipsHeight;

  clip.tracks.forEach((track) => {
    const trackSplitted = track.name.split('.');
    const mixamoRigName = trackSplitted[0];
    const vrmBoneName = mixamoVRMRigMap[mixamoRigName];
    const vrmNodeName = vrm.humanoid?.getNormalizedBoneNode(vrmBoneName)?.name;
    const mixamoRigNode = asset.getObjectByName(mixamoRigName);

    if (vrmNodeName != null && mixamoRigNode != null && mixamoRigNode.parent != null) {
      const propertyName = trackSplitted[1];

      // Store rotations of rest-pose
      mixamoRigNode.getWorldQuaternion(restRotationInverse).invert();
      mixamoRigNode.parent.getWorldQuaternion(parentRestWorldRotation);

      if (track instanceof THREE.QuaternionKeyframeTrack) {
        // Retarget rotation: parentRestWorld * trackRot * restInverse
        for (let i = 0; i < track.values.length; i += 4) {
          const flatQuaternion = track.values.slice(i, i + 4);

          _quatA.fromArray(flatQuaternion);
          _quatA
            .premultiply(parentRestWorldRotation)
            .multiply(restRotationInverse);
          _quatA.toArray(flatQuaternion);

          flatQuaternion.forEach((v, index) => {
            track.values[index + i] = v;
          });
        }

        tracks.push(
          new THREE.QuaternionKeyframeTrack(
            `${vrmNodeName}.${propertyName}`,
            track.times,
            track.values.map((v, i) =>
              vrm.meta?.metaVersion === '0' && i % 2 === 0 ? -v : v,
            ),
          ),
        );
      } else if (track instanceof THREE.VectorKeyframeTrack) {
        const value = track.values.map(
          (v, i) =>
            (vrm.meta?.metaVersion === '0' && i % 3 !== 1 ? -v : v) *
            hipsPositionScale,
        );
        tracks.push(
          new THREE.VectorKeyframeTrack(
            `${vrmNodeName}.${propertyName}`,
            track.times,
            value,
          ),
        );
      }
    }
  });

  return new THREE.AnimationClip('vrmAnimation', clip.duration, tracks);
}
