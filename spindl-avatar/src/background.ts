import * as THREE from 'three';

export interface BackgroundConfig {
  type: 'gradient' | 'color' | 'image';
  colorCenter?: string;
  colorEdge?: string;
  color?: string;
  imagePath?: string;
  blur?: number;
  vignette?: boolean;
  tint?: string;
  tintOpacity?: number;
}

const DEFAULT_CONFIG: BackgroundConfig = {
  type: 'gradient',
  colorCenter: '#2d1854',
  colorEdge: '#0c0618',
};

function createGradientTexture(centerColor: string, edgeColor: string): THREE.CanvasTexture {
  const size = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const gradient = ctx.createRadialGradient(
    size / 2, size * 0.4, 0,
    size / 2, size / 2, size * 0.7
  );
  gradient.addColorStop(0, centerColor);
  gradient.addColorStop(0.5, mixColors(centerColor, edgeColor, 0.5));
  gradient.addColorStop(1, edgeColor);

  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, size, size);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function mixColors(a: string, b: string, t: number): string {
  const parse = (hex: string) => {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  };
  const ca = parse(a);
  const cb = parse(b);
  const mix = ca.map((v, i) => Math.round(v + (cb[i] - v) * t));
  return `#${mix.map(v => v.toString(16).padStart(2, '0')).join('')}`;
}

function loadImage(path: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (err) => reject(err);
    img.src = path;
  });
}

function createImageTexture(
  img: HTMLImageElement,
  blur: number = 0,
  vignette: boolean = false,
  tint?: string,
  tintOpacity: number = 0.3,
): THREE.CanvasTexture {
  const size = 1024;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;

  const viewportAspect = window.innerWidth / window.innerHeight;
  const imgAspect = img.width / img.height;

  let drawW: number, drawH: number, drawX: number, drawY: number;

  if (imgAspect > viewportAspect) {
    drawH = size;
    drawW = size * (imgAspect / viewportAspect);
    drawX = -(drawW - size) / 2;
    drawY = 0;
  } else {
    drawW = size;
    drawH = size * (viewportAspect / imgAspect);
    drawX = 0;
    drawY = -(drawH - size) / 2;
  }

  ctx.drawImage(img, drawX, drawY, drawW, drawH);

  if (blur > 0) {
    const scale = Math.max(0.02, 1 / (1 + blur * 0.15));
    const tmpCanvas = document.createElement('canvas');
    const tmpCtx = tmpCanvas.getContext('2d')!;

    tmpCanvas.width = Math.max(1, Math.round(size * scale));
    tmpCanvas.height = Math.max(1, Math.round(size * scale));
    tmpCtx.drawImage(canvas, 0, 0, tmpCanvas.width, tmpCanvas.height);

    for (let i = 1; i < 3; i++) {
      const w2 = Math.max(1, Math.round(tmpCanvas.width * 0.5));
      const h2 = Math.max(1, Math.round(tmpCanvas.height * 0.5));
      tmpCtx.drawImage(tmpCanvas, 0, 0, w2, h2);
      tmpCtx.drawImage(tmpCanvas, 0, 0, w2, h2, 0, 0, tmpCanvas.width, tmpCanvas.height);
    }

    ctx.clearRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(tmpCanvas, 0, 0, size, size);
  }

  ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
  ctx.fillRect(0, 0, size, size);

  if (tint) {
    ctx.fillStyle = tint;
    ctx.globalAlpha = tintOpacity;
    ctx.fillRect(0, 0, size, size);
    ctx.globalAlpha = 1;
  }

  if (vignette) {
    const vigGrad = ctx.createRadialGradient(
      size / 2, size * 0.4, Math.min(size, size) * 0.2,
      size / 2, size / 2, Math.max(size, size) * 0.75,
    );
    vigGrad.addColorStop(0, 'rgba(0,0,0,0)');
    vigGrad.addColorStop(0.7, 'rgba(0,0,0,0.3)');
    vigGrad.addColorStop(1, 'rgba(0,0,0,0.7)');
    ctx.fillStyle = vigGrad;
    ctx.fillRect(0, 0, size, size);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export const BACKGROUND_PRESETS: Record<string, BackgroundConfig> = {
  darkPurple: {
    type: 'gradient',
    colorCenter: '#2d1854',
    colorEdge: '#0c0618',
  },
  midnight: {
    type: 'gradient',
    colorCenter: '#162d4a',
    colorEdge: '#040c18',
  },
  warmDark: {
    type: 'gradient',
    colorCenter: '#3a2420',
    colorEdge: '#120808',
  },
  purple: {
    type: 'gradient',
    colorCenter: '#2e1250',
    colorEdge: '#140028',
  },
  ocean: {
    type: 'gradient',
    colorCenter: '#142840',
    colorEdge: '#061018',
  },
  solid: {
    type: 'color',
    color: '#0a0a0f',
  },
  chromaGreen: {
    type: 'color',
    color: '#00FF00',
  },
  chromaMagenta: {
    type: 'color',
    color: '#FF00FF',
  },
};

const IMAGE_CACHE_MAX = 10;
const imageCache = new Map<string, HTMLImageElement>();

function imageCacheSet(key: string, value: HTMLImageElement): void {
  imageCache.set(key, value);
  if (imageCache.size > IMAGE_CACHE_MAX) {
    const oldest = imageCache.keys().next().value;
    if (oldest !== undefined) imageCache.delete(oldest);
  }
}

function disposePreviousBackground(scene: THREE.Scene): void {
  if (scene.background && scene.background instanceof THREE.Texture) {
    scene.background.dispose();
  }
}

export async function applyBackground(
  scene: THREE.Scene,
  config: BackgroundConfig = DEFAULT_CONFIG,
): Promise<void> {
  disposePreviousBackground(scene);

  switch (config.type) {
    case 'gradient': {
      const center = config.colorCenter ?? '#1a0e2e';
      const edge = config.colorEdge ?? '#060210';
      scene.background = createGradientTexture(center, edge);
      break;
    }
    case 'color': {
      scene.background = new THREE.Color(config.color ?? '#0a0a0f');
      break;
    }
    case 'image': {
      if (config.imagePath) {
        try {
          let img = imageCache.get(config.imagePath);
          if (!img) {
            img = await loadImage(config.imagePath);
            imageCacheSet(config.imagePath, img);
          }
          scene.background = createImageTexture(
            img,
            config.blur ?? 15,
            config.vignette ?? true,
            config.tint,
            config.tintOpacity,
          );
        } catch (e) {
          if (import.meta.env.DEV) console.warn('[SpindL] Failed to load background image, falling back to gradient', e);
          scene.background = createGradientTexture('#2d1854', '#0c0618');
        }
      }
      break;
    }
  }
}

export function loadBackgroundConfig(): BackgroundConfig {
  try {
    const saved = localStorage.getItem('spindl-avatar-background');
    if (saved) return JSON.parse(saved) as BackgroundConfig;
  } catch { /* use default */ }
  return DEFAULT_CONFIG;
}

export function saveBackgroundConfig(config: BackgroundConfig): void {
  localStorage.setItem('spindl-avatar-background', JSON.stringify(config));
}
