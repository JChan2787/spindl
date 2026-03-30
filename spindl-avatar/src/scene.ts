import * as THREE from 'three';
import { setForceOpaqueMaterials } from './avatar';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export interface SceneContext {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  composer: EffectComposer;
  clock: THREE.Clock;
  transparent: boolean;
  setTransparent(on: boolean): void;
  dispose(): void;
}

// -- Color Grade + Vignette + Film Grain + Alpha Restore --
const ColorGradeShader = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    tAlphaStash: { value: null as THREE.Texture | null },
    useAlphaStash: { value: 0 },
    brightness: { value: 0.03 },
    contrast: { value: 0.12 },
    saturation: { value: 0.15 },
    vignetteOffset: { value: 1.1 },
    vignetteDarkness: { value: 1.0 },
    grainIntensity: { value: 0.04 },
    time: { value: 0.0 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform sampler2D tAlphaStash;
    uniform float useAlphaStash;
    uniform float brightness;
    uniform float contrast;
    uniform float saturation;
    uniform float vignetteOffset;
    uniform float vignetteDarkness;
    uniform float grainIntensity;
    uniform float time;
    varying vec2 vUv;

    float hash(vec2 p) {
      vec3 p3 = fract(vec3(p.xyx) * 0.1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
    }

    void main() {
      vec4 color = texture2D(tDiffuse, vUv);
      float sceneAlpha = texture2D(tAlphaStash, vUv).r;
      bool isTransparent = useAlphaStash > 0.5;

      // In transparent mode, only apply color grading to pixels with content
      float gradeMix = isTransparent ? sceneAlpha : 1.0;

      // Brightness
      color.rgb += brightness * gradeMix;

      // Contrast (centered on 0.5)
      vec3 contrasted = (color.rgb - 0.5) * (1.0 + contrast) + 0.5;
      color.rgb = mix(color.rgb, contrasted, gradeMix);

      // Saturation (luminance-preserving)
      float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
      vec3 saturated = mix(vec3(luma), color.rgb, 1.0 + saturation);
      color.rgb = mix(color.rgb, saturated, gradeMix);

      // Vignette — skip in transparent mode (no background to darken)
      if (!isTransparent) {
        vec2 uv = (vUv - 0.5) * 2.0;
        float dist = dot(uv, uv);
        float vig = 1.0 - smoothstep(vignetteOffset, vignetteOffset + 0.7, dist * vignetteDarkness);
        color.rgb *= mix(1.0, vig, 0.6);
      }

      // Film grain — only on opaque content
      float grain = hash(vUv * 1000.0 + time) - 0.5;
      float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
      float grainWeight = 1.0 - smoothstep(0.0, 0.6, luminance);
      color.rgb += grain * grainIntensity * grainWeight * gradeMix;

      // Alpha restoration in transparent mode
      if (isTransparent) {
        // Model pixels: any scene alpha > tiny threshold = fully opaque.
        // Background pixels: bloom glow gets soft alpha for a floating halo effect.
        float finalAlpha;
        if (sceneAlpha > 0.01) {
          finalAlpha = 1.0;
        } else {
          // Pure background — only bright bloom glow gets alpha
          float bloomAlpha = smoothstep(0.05, 0.3, luminance);
          finalAlpha = bloomAlpha;
        }
        color.rgb *= finalAlpha;
        color.a = finalAlpha;
      }

      gl_FragColor = color;
    }
  `
};

// Saved background color for restoring from transparent mode
const DEFAULT_BG = new THREE.Color(0x0a0a0f);

export function createScene(canvas: HTMLCanvasElement): SceneContext {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0f);

  const camera = new THREE.PerspectiveCamera(
    30,
    window.innerWidth / window.innerHeight,
    0.1,
    20
  );
  camera.position.set(0, 1.42, 1.0);
  camera.lookAt(0, 1.44, 0);

  // Key light — warm white, front-right above
  const keyLight = new THREE.DirectionalLight(0xfff0e0, 1.3);
  keyLight.position.set(1, 3, 2);
  scene.add(keyLight);

  // Fill light — cool lavender-blue, front-left
  const fillLight = new THREE.DirectionalLight(0x8899cc, 0.5);
  fillLight.position.set(-1, 2, 1.5);
  scene.add(fillLight);

  // Rim light — cool blue-white, behind
  const rimLight = new THREE.DirectionalLight(0xa0a0ee, 0.9);
  rimLight.position.set(0, 2, -1.5);
  scene.add(rimLight);

  // Hair/kicker light — top-down accent
  const kickerLight = new THREE.DirectionalLight(0xddccee, 0.4);
  kickerLight.position.set(0.3, 4, 0);
  scene.add(kickerLight);

  // Hemisphere light — main ambient fill (MToon responds well to this)
  const hemiLight = new THREE.HemisphereLight(0xd0d0dd, 0x141820, 0.45);
  scene.add(hemiLight);

  // Ambient — raise the floor
  const ambient = new THREE.AmbientLight(0x484855, 0.35);
  scene.add(ambient);

  // Eye catch light — bright point near camera for eye sparkle
  const catchLight = new THREE.PointLight(0xffffff, 0.3, 3.0);
  catchLight.position.set(0.05, 1.44, 0.85);
  scene.add(catchLight);

  // -- Alpha stash render target --
  // Captures the scene's alpha BEFORE bloom runs (bloom stomps alpha to 1.0).
  // This is rendered manually in the composer.render intercept, NOT as a pass
  // in the chain (ShaderPass would overwrite the color buffer).
  const pixelRatio = renderer.getPixelRatio();
  let alphaStashRT = new THREE.WebGLRenderTarget(
    window.innerWidth * pixelRatio,
    window.innerHeight * pixelRatio,
    { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }
  );

  // -- Post-processing chain: Render → Bloom → ColorGrade → Output --
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.25,  // strength
    0.4,   // radius
    0.78   // threshold
  );
  composer.addPass(bloomPass);

  const colorGradePass = new ShaderPass(ColorGradeShader);
  composer.addPass(colorGradePass);

  composer.addPass(new OutputPass());

  const clock = new THREE.Clock();

  let transparentMode = false;

  // Override material for alpha stash: renders every mesh pixel as solid white.
  // This ensures the stash has alpha=1 for ALL model pixels regardless of
  // the material's actual alpha/blend mode (MToon hair, accessories, etc.)
  const stashOverrideMaterial = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    side: THREE.DoubleSide,
  });

  // Intercept composer.render to update time + capture alpha stash
  const _origRender = composer.render.bind(composer);
  composer.render = function (...args: Parameters<typeof _origRender>) {
    colorGradePass.uniforms['time'].value = performance.now() * 0.001;

    if (transparentMode) {
      // Render scene with override material to the stash RT.
      // Every mesh pixel = white (alpha 1), background = black (alpha 0).
      const prevOverride = scene.overrideMaterial;
      const prevBg = scene.background;
      scene.overrideMaterial = stashOverrideMaterial;
      scene.background = null;
      renderer.setRenderTarget(alphaStashRT);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      renderer.render(scene, camera);
      renderer.setRenderTarget(null);
      scene.overrideMaterial = prevOverride;
      scene.background = prevBg;

      // Feed stash to ColorGradeShader for alpha restoration
      colorGradePass.uniforms['tAlphaStash'].value = alphaStashRT.texture;
      colorGradePass.uniforms['useAlphaStash'].value = 1;
    } else {
      colorGradePass.uniforms['useAlphaStash'].value = 0;
    }

    return _origRender(...args);
  };

  let savedBackground: THREE.Color | THREE.Texture | null = scene.background;

  function setTransparent(on: boolean): void {
    transparentMode = on;
    ctx.transparent = on;

    if (on) {
      // Save current background before clearing
      if (scene.background !== null) {
        if (scene.background instanceof THREE.Color) {
          savedBackground = scene.background.clone();
        } else {
          savedBackground = scene.background;
        }
      }
      scene.background = null;
      renderer.setClearColor(0x000000, 0);
      document.documentElement.style.background = 'transparent';
      document.body.style.background = 'transparent';
      // Force all materials to hard alpha cutoff so model is fully opaque
      setForceOpaqueMaterials(true);
    } else {
      scene.background = savedBackground ?? DEFAULT_BG.clone();
      renderer.setClearColor(0x000000, 1);
      document.documentElement.style.background = '#050208';
      document.body.style.background = '#050208';
      // Restore original material alpha modes
      setForceOpaqueMaterials(false);
    }
  }

  const onResize = () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
    composer.setSize(w, h);
    // Resize alpha stash RT to match
    const pr = renderer.getPixelRatio();
    alphaStashRT.dispose();
    alphaStashRT = new THREE.WebGLRenderTarget(
      w * pr, h * pr,
      { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }
    );
  };
  window.addEventListener('resize', onResize);

  const ctx: SceneContext = {
    scene, camera, renderer, composer, clock, transparent: false,
    setTransparent,
    dispose() { window.removeEventListener('resize', onResize); alphaStashRT.dispose(); },
  };

  return ctx;
}

/** Access the rim light to change its color for mood */
export function setRimLightColor(scene: THREE.Scene, color: THREE.Color): void {
  const lights = scene.children.filter(
    (c): c is THREE.DirectionalLight => c instanceof THREE.DirectionalLight
  );
  const rim = lights[2]; // key=0, fill=1, rim=2
  if (rim) rim.color.copy(color);

  const fill = lights[1];
  if (fill) {
    const baseFill = new THREE.Color(0x8899cc);
    fill.color.copy(baseFill).lerp(color, 0.25);
  }

  const hemi = scene.children.find(
    (c): c is THREE.HemisphereLight => c instanceof THREE.HemisphereLight
  );
  if (hemi) {
    const baseHemiSky = new THREE.Color(0xccccdd);
    hemi.color.copy(baseHemiSky).lerp(color, 0.15);
  }
}
