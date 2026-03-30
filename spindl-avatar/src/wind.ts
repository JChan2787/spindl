import * as THREE from 'three';
import { getVRM } from './avatar';
import { fbm } from './noise';

const BASE_WIND_STRENGTH = 0.3;
const GUST_STRENGTH = 0.8;
const GUST_FREQUENCY = 0.15;
const WIND_TURBULENCE = 0.4;
const WIND_DIRECTION = new THREE.Vector3(1, 0, 0.3).normalize();

const BASE_GRAVITY_DIR = new THREE.Vector3(0, -1, 0);
const BASE_GRAVITY_POWER = 0.05;

let elapsed = 0;
let enabled = true;

const origSettings = new Map<number, { gravityPower: number; gravityDir: THREE.Vector3 }>();
let initialized = false;

export function initWind(): void {
  const vrm = getVRM();
  if (!vrm?.springBoneManager) return;

  origSettings.clear();
  let i = 0;
  for (const joint of vrm.springBoneManager.joints) {
    origSettings.set(i, {
      gravityPower: joint.settings.gravityPower,
      gravityDir: joint.settings.gravityDir.clone(),
    });
    i++;
  }
  initialized = true;
  if (import.meta.env.DEV) console.log(`[SpindL] Wind system initialized: ${origSettings.size} spring bone joints`);
}

export function updateWind(deltaMs: number): void {
  if (!enabled || !initialized) return;

  const vrm = getVRM();
  if (!vrm?.springBoneManager) return;

  elapsed += deltaMs / 1000;

  const gustAmount = Math.max(0, fbm(elapsed * GUST_FREQUENCY, 3));
  const windStrength = BASE_WIND_STRENGTH + gustAmount * GUST_STRENGTH;

  const turbX = fbm(elapsed * 0.3 + 100, 2) * WIND_TURBULENCE;
  const turbY = fbm(elapsed * 0.2 + 200, 2) * WIND_TURBULENCE * 0.3;
  const turbZ = fbm(elapsed * 0.25 + 300, 2) * WIND_TURBULENCE;

  const windX = WIND_DIRECTION.x + turbX;
  const windY = WIND_DIRECTION.y + turbY;
  const windZ = WIND_DIRECTION.z + turbZ;

  const combinedDir = new THREE.Vector3(
    BASE_GRAVITY_DIR.x + windX * windStrength,
    BASE_GRAVITY_DIR.y + windY * windStrength,
    BASE_GRAVITY_DIR.z + windZ * windStrength,
  ).normalize();

  const combinedPower = BASE_GRAVITY_POWER + windStrength * 0.05;

  for (const joint of vrm.springBoneManager.joints) {
    joint.settings.gravityDir.copy(combinedDir);
    joint.settings.gravityPower = combinedPower;
  }
}

export function setWindEnabled(value: boolean): void {
  enabled = value;
  if (!value) restoreOriginalSettings();
}

function restoreOriginalSettings(): void {
  const vrm = getVRM();
  if (!vrm?.springBoneManager) return;

  let i = 0;
  for (const joint of vrm.springBoneManager.joints) {
    const orig = origSettings.get(i);
    if (orig) {
      joint.settings.gravityPower = orig.gravityPower;
      joint.settings.gravityDir.copy(orig.gravityDir);
    }
    i++;
  }
}
