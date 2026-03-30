/**
 * Critically damped spring — the gold standard for procedural animation.
 *
 * Parameterized by half-life: how many seconds until the spring is halfway to target.
 *
 * Typical values:
 *   0.05 = very snappy (head nod reaction)
 *   0.1  = responsive (eye tracking)
 *   0.2  = natural (head following mouse)
 *   0.4  = floaty (body sway)
 *   0.8  = very slow (weight shifting)
 *
 * Based on: https://theorangeduck.com/page/spring-roll-call
 */

function dampingFromHalfLife(halfLife: number): number {
  return (4.0 * 0.693147) / Math.max(halfLife, 0.001);
}

export interface SpringState {
  pos: number;
  vel: number;
}

export function springDamped(
  current: SpringState,
  target: number,
  halfLife: number,
  dt: number,
): SpringState {
  const d = dampingFromHalfLife(halfLife);
  const d_dt = d * dt;

  const decay = 1.0 + d_dt + 0.48 * d_dt * d_dt + 0.235 * d_dt * d_dt * d_dt;
  const invDecay = 1.0 / decay;

  const err = current.pos - target;
  const errDot = current.vel + err * d;

  const newPos = target + (err + errDot * dt) * invDecay;
  const newVel = (current.vel - errDot * d * dt) * invDecay;

  return { pos: newPos, vel: newVel };
}

/**
 * Underdamped spring — allows bounce/overshoot.
 * damping_ratio < 1.0 = bouncy, = 1.0 = critically damped, > 1.0 = overdamped
 */
export function springUnderdamped(
  current: SpringState,
  target: number,
  halfLife: number,
  dampingRatio: number,
  dt: number,
): SpringState {
  const d = dampingFromHalfLife(halfLife);
  const stiffness = (d / (2.0 * dampingRatio)) ** 2;
  const damping = d / dampingRatio;

  const force = -stiffness * (current.pos - target) - damping * current.vel;
  const newVel = current.vel + force * dt;
  const newPos = current.pos + newVel * dt;

  return { pos: newPos, vel: newVel };
}

export function createSpring(pos: number = 0): SpringState {
  return { pos, vel: 0 };
}

export interface SpringVec3 {
  x: SpringState;
  y: SpringState;
  z: SpringState;
}

export function createSpringVec3(x = 0, y = 0, z = 0): SpringVec3 {
  return {
    x: createSpring(x),
    y: createSpring(y),
    z: createSpring(z),
  };
}
