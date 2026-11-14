/**
 * AnnolidShaders — Reusable GLSL shader enhancements for Three.js
 *
 * A standalone ES module that adds 5 photorealistic shader effects to any
 * Three.js MeshStandardMaterial / MeshPhysicalMaterial via onBeforeCompile.
 *
 * Effects:
 *   1. Fur/Hair Anisotropic  — Kajiya-Kay highlight + root→tip darkening
 *   2. Subsurface Scatter     — Wrap-diffuse + back-light transmission
 *   3. Micro-Detail Normal    — Procedural hash-based normal perturbation
 *   4. Contact AO             — Ground-proximity ambient occlusion
 *   5. Post-Processing Quad   — Vignette + film grain + color grading
 *
 * Usage:
 *   import { AnnolidShaders } from './annolid_shaders.js';
 *   const shaders = new AnnolidShaders(THREE);
 *   shaders.applyFurShader(material, opts);
 *   shaders.applySSSShader(material, opts);
 *   ...
 *
 * @version 1.0.0
 * @license MIT
 */

// ─────────────────────────────────────────────────────────────
// GLSL CHUNKS (string constants)
// ─────────────────────────────────────────────────────────────

// ── 1. Fur / Hair Anisotropic ────────────────────────────────
const FUR_UNIFORMS_GLSL = /* glsl */ `
  uniform vec3  uFurRootColor;
  uniform vec3  uFurTipColor;
  uniform float uFurAnisotropy;
  uniform float uFurSpecIntensity;
  uniform vec3  uFurFlowDir;
`;

const FUR_VERTEX_GLSL = /* glsl */ `
  // Pass world-space normal, flow tangent & position for anisotropic calc
  varying vec3 vFurWorldNormal;
  varying vec3 vFurWorldFlow;
  varying vec3 vFurWorldPos;
`;

const FUR_VERTEX_MAIN_GLSL = /* glsl */ `
  vFurWorldNormal = (modelMatrix * vec4(objectNormal, 0.0)).xyz;
  vFurWorldFlow   = (modelMatrix * vec4(uFurFlowDir, 0.0)).xyz;
  vFurWorldPos    = (modelMatrix * vec4(position, 1.0)).xyz;
`;

const FUR_FRAGMENT_GLSL = /* glsl */ `
  varying vec3 vFurWorldNormal;
  varying vec3 vFurWorldFlow;
  varying vec3 vFurWorldPos;
`;

const FUR_FRAGMENT_MAIN_GLSL = /* glsl */ `
  {
    // Marschner-inspired double anisotropic highlight for hair/fur (NaN-safe)
    vec3 nrm = vFurWorldNormal * inversesqrt(max(dot(vFurWorldNormal, vFurWorldNormal), 1e-8));
    vec3 worldFlow = vFurWorldFlow * inversesqrt(max(dot(vFurWorldFlow, vFurWorldFlow), 1e-8));
    vec3 rawTangent = worldFlow - nrm * dot(worldFlow, nrm);
    float rawLenSq = dot(rawTangent, rawTangent);
    vec3 fallbackAxis = abs(nrm.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 fallbackCross = cross(nrm, fallbackAxis);
    vec3 tangent = rawLenSq > 1e-6
      ? rawTangent * inversesqrt(rawLenSq)
      : fallbackCross * inversesqrt(max(dot(fallbackCross, fallbackCross), 1e-8));

    vec3 toCam    = cameraPosition - vFurWorldPos;
    vec3 viewDir  = toCam * inversesqrt(max(dot(toCam, toCam), 1e-8));
    vec3 keyDir   = normalize(vec3(-0.26, 0.92, 0.28));
    vec3 rimDir   = normalize(vec3(0.42, 0.62, -0.66));
    vec3 sumKey   = viewDir + keyDir;
    vec3 sumRim   = viewDir + rimDir;
    vec3 halfKey  = sumKey * inversesqrt(max(dot(sumKey, sumKey), 1e-8));
    vec3 halfRim  = sumRim * inversesqrt(max(dot(sumRim, sumRim), 1e-8));

    // Primary R cuticle lobe (shifted toward root) and secondary TRT cortex lobe (shifted toward tip)
    vec3 rawT1 = tangent - nrm * 0.08;
    vec3 rawT2 = tangent + nrm * 0.14;
    vec3 t1 = rawT1 * inversesqrt(max(dot(rawT1, rawT1), 1e-8));
    vec3 t2 = rawT2 * inversesqrt(max(dot(rawT2, rawT2), 1e-8));

    float TdotH1 = clamp(dot(t1, halfKey), -1.0, 1.0);
    float TdotH2 = clamp(dot(t2, halfKey), -1.0, 1.0);
    float TdotHR = clamp(dot(t1, halfRim), -1.0, 1.0);

    float sinTH1 = sqrt(max(0.0, 1.0 - TdotH1 * TdotH1));
    float sinTH2 = sqrt(max(0.0, 1.0 - TdotH2 * TdotH2));
    float sinTHR = sqrt(max(0.0, 1.0 - TdotHR * TdotHR));

    float keyFacing = smoothstep(-0.25, 0.65, dot(nrm, keyDir));
    float rimFacing = smoothstep(-0.15, 0.75, dot(nrm, rimDir));
    float graze = pow(1.0 - abs(dot(viewDir, nrm)), 2.4);

    float specExp = max(18.0, uFurAnisotropy);
    float specScale = clamp(uFurSpecIntensity, 0.0, 0.14);
    float anisoPrimary   = pow(sinTH1, specExp) * specScale * keyFacing;
    float anisoSecondary = pow(sinTH2, max(10.0, specExp * 0.45)) * specScale * 0.45 * keyFacing;
    float anisoRim       = pow(sinTHR, max(14.0, specExp * 0.60)) * specScale * 0.35 * rimFacing;

    // Subtle dorsal-to-ventral / root-to-tip tonal richness without crushing or bleaching base albedo
    float dorsalFactor = clamp(nrm.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 fiberTint = mix(uFurRootColor, uFurTipColor, dorsalFactor);

    vec3 pigmentSafeCuticle = mix(fiberTint * 1.4 + vec3(0.03), vec3(0.75, 0.74, 0.72), 0.35);
    vec3 sheenCuticle = pigmentSafeCuticle * (anisoPrimary + anisoRim * 0.55);
    vec3 sheenCortex  = fiberTint * (anisoSecondary + graze * specScale * 0.22);

    gl_FragColor.rgb = gl_FragColor.rgb * mix(vec3(1.0), fiberTint, 0.10) + sheenCuticle + sheenCortex;
  }
`;

// ── 2. Subsurface Scatter ────────────────────────────────────
const SSS_UNIFORMS_GLSL = /* glsl */ `
  uniform vec3  uSSSColor;
  uniform float uSSSThickness;
  uniform float uSSSDistortion;
  uniform float uSSSWrap;
  uniform float uSSSBacklight;
`;

const SSS_VERTEX_GLSL = /* glsl */ `
  varying vec3 vSSSWorldNormal;
  varying vec3 vSSSWorldPos;
`;

const SSS_VERTEX_MAIN_GLSL = /* glsl */ `
  vSSSWorldNormal = (modelMatrix * vec4(objectNormal, 0.0)).xyz;
  vSSSWorldPos    = (modelMatrix * vec4(position, 1.0)).xyz;
`;

const SSS_FRAGMENT_GLSL = /* glsl */ `
  varying vec3 vSSSWorldNormal;
  varying vec3 vSSSWorldPos;
`;

const SSS_FRAGMENT_MAIN_GLSL = /* glsl */ `
  {
    vec3 toCam    = cameraPosition - vSSSWorldPos;
    vec3 viewDir  = toCam * inversesqrt(max(dot(toCam, toCam), 1e-8));
    vec3 lightDir = normalize(vec3(-0.26, 0.92, 0.28));
    vec3 nrm      = vSSSWorldNormal * inversesqrt(max(dot(vSSSWorldNormal, vSSSWorldNormal), 1e-8));

    // Terminator-focused wrap scatter (warms shadow transitions without blowing out highlights)
    float NdotL = dot(nrm, lightDir);
    float wrapDenom = max(0.001, 1.0 + uSSSWrap);
    float wrapDiffuse = max(0.0, (NdotL + uSSSWrap) / wrapDenom);
    float terminatorBand = max(0.0, wrapDiffuse - max(0.0, NdotL)) * 1.35;

    // Thin silhouettes transmit more light than thick centers
    float viewDotNrm = abs(dot(viewDir, nrm));
    float thinness = 1.0 - smoothstep(0.08, 0.88, viewDotNrm);

    vec3 scatter = uSSSColor * terminatorBand * uSSSThickness * 0.55;

    // Back-light transmission with melanin-scaled vascular spectral shift
    vec3 rawScatterDir = lightDir + nrm * uSSSDistortion;
    vec3 scatterDir = rawScatterDir * inversesqrt(max(dot(rawScatterDir, rawScatterDir), 1e-8));
    float backDot   = max(0.0, dot(viewDir, -scatterDir));
    float scatterPow = pow(backDot, 3.8) * (0.35 + 0.65 * thinness) * uSSSThickness;

    float skinLuma = clamp(dot(uSSSColor, vec3(0.299, 0.587, 0.114)), 0.04, 1.0);
    vec3 transmitColor = mix(uSSSColor, vec3(0.86, 0.24, 0.16) * (0.25 + 0.75 * skinLuma), 0.38);
    scatter += transmitColor * scatterPow * uSSSBacklight * 0.68;

    gl_FragColor.rgb += scatter;
  }
`;

// ── 3. Micro-Detail Normal ────────────────────────────────────
const MICRO_UNIFORMS_GLSL = /* glsl */ `
  uniform float uMicroScale;
  uniform float uMicroStrength;
`;

const MICRO_VERTEX_GLSL = /* glsl */ `
  varying vec3 vMicroWorldPos;
`;

const MICRO_VERTEX_MAIN_GLSL = /* glsl */ `
  vMicroWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
`;

const MICRO_FRAGMENT_GLSL = /* glsl */ `
  varying vec3 vMicroWorldPos;
`;

// Injected BEFORE lighting so it perturbs the normal used for shading
const MICRO_FRAGMENT_PRENORMAL_GLSL = /* glsl */ `
  {
    vec3 wp = vMicroWorldPos * uMicroScale;

    // 3D hash-based noise for procedural micro-detail
    float n1 = fract(sin(dot(wp.xy, vec2(127.1, 311.7))) * 43758.5453);
    float n2 = fract(sin(dot(wp.yz, vec2(269.5, 183.3))) * 43758.5453);
    float n3 = fract(sin(dot(wp.xz, vec2(419.2, 371.9))) * 43758.5453);

    // Smoothstep the noise to create cell-like pores/scales instead of uniform static
    vec3 noise = vec3(n1, n2, n3);
    noise = smoothstep(0.2, 0.8, noise);

    vec3 microPerturb = (noise - 0.5) * uMicroStrength;
    vec3 perturbedNormal = normal + microPerturb;
    normal = perturbedNormal * inversesqrt(max(dot(perturbedNormal, perturbedNormal), 1e-8));
  }
`;

// ── 4. Contact AO ────────────────────────────────────────────
const AO_UNIFORMS_GLSL = /* glsl */ `
  uniform float uAOGroundY;
  uniform float uAORadius;
  uniform float uAOIntensity;
  uniform vec3  uAOColor;
`;

const AO_VERTEX_GLSL = /* glsl */ `
  varying float vAOWorldY;
`;

const AO_VERTEX_MAIN_GLSL = /* glsl */ `
  vAOWorldY = (modelMatrix * vec4(position, 1.0)).y;
`;

const AO_FRAGMENT_GLSL = /* glsl */ `
  varying float vAOWorldY;
`;

const AO_FRAGMENT_MAIN_GLSL = /* glsl */ `
  {
    // Height-based ambient occlusion near ground plane
    float groundDist = max(0.0, vAOWorldY - uAOGroundY);
    // Non-linear falloff for deeper shadows exactly at contact points
    float ao = smoothstep(0.0, max(1e-4, uAORadius), groundDist);
    ao = mix(1.0 - clamp(uAOIntensity, 0.0, 1.0), 1.0, pow(ao, 0.6));
    gl_FragColor.rgb *= mix(uAOColor, vec3(1.0), ao);
  }
`;

// ── 5. Post-Processing Quad ──────────────────────────────────
const POST_VERTEX_GLSL = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
  }
`;

const POST_FRAGMENT_GLSL = /* glsl */ `
  uniform sampler2D tDiffuse;
  uniform sampler2D tDepth;
  uniform float uTime;
  uniform float uVigIntensity;
  uniform float uVigSoftness;
  uniform float uGrainStrength;
  uniform float uSaturation;
  uniform float uBrightness;

  uniform float uCameraNear;
  uniform float uCameraFar;
  uniform float uFocusDistance;
  uniform float uAperture;
  uniform float uOutputSRGB;
  uniform vec2 uResolution;

  varying vec2 vUv;

  #include <packing>

  vec3 annolidLinearToSRGB(vec3 value) {
    vec3 clamped = max(value, vec3(0.0));
    return mix(
      pow(clamped, vec3(0.41666)) * 1.055 - vec3(0.055),
      clamped * 12.92,
      vec3(lessThanEqual(clamped, vec3(0.0031308)))
    );
  }

  float getLinearDepth(vec2 uv) {
    float fragCoordZ = texture2D(tDepth, uv).x;
    float viewZ = perspectiveDepthToViewZ(fragCoordZ, uCameraNear, uCameraFar);
    return viewZToOrthographicDepth(viewZ, uCameraNear, uCameraFar);
  }

  void main() {
    float d = getLinearDepth(vUv);

    // CoC (Circle of Confusion) around the active focus plane
    float coc = clamp(abs(d - uFocusDistance) * uAperture, 0.0, 1.0);

    vec2 dir = vUv - 0.5;
    float dist = length(dir);
    vec2 offset = dir * (dist * 0.0022); // Subtle lateral chromatic aberration

    vec3 colorSum = vec3(0.0);
    float weightSum = 0.0;

    // Center sample in linear HDR space
    float r = texture2D(tDiffuse, clamp(vUv - offset, 0.0, 1.0)).r;
    float g = texture2D(tDiffuse, vUv).g;
    float b = texture2D(tDiffuse, clamp(vUv + offset, 0.0, 1.0)).b;
    colorSum += vec3(r, g, b);
    weightSum += 1.0;

    // DoF Blur Taps (only if CoC is meaningful)
    if (coc > 0.015) {
      vec2 safeRes = max(uResolution, vec2(1.0));
      vec2 texel = 1.0 / safeRes;
      float radius = coc * max(6.0, min(safeRes.x, safeRes.y) * 0.012);
      vec2 tRad = texel * radius;

      const float TWO_PI = 6.28318530718;
      for (float i = 0.0; i < 8.0; i++) {
        float angle = i * (TWO_PI / 8.0);
        vec2 suv = clamp(vUv + vec2(cos(angle), sin(angle)) * tRad, 0.0, 1.0);

        float sd = getLinearDepth(suv);
        float w = 1.0;
        if (sd < d) {
          float sampleCoc = abs(sd - uFocusDistance) * uAperture;
          w = clamp(sampleCoc * 10.0, 0.0, 1.0);
        }

        float sr = texture2D(tDiffuse, clamp(suv - offset, 0.0, 1.0)).r;
        float sg = texture2D(tDiffuse, suv).g;
        float sb = texture2D(tDiffuse, clamp(suv + offset, 0.0, 1.0)).b;

        colorSum += vec3(sr, sg, sb) * w;
        weightSum += w;
      }
    }

    // Apply renderer tone mapping (e.g. ACESFilmic) + IEC 61966-2-1 sRGB OETF
    // (Three.js forces LinearSRGB output on ShaderMaterial, so OutputPass-style
    // sRGB conversion is required when renderer.outputColorSpace is SRGBColorSpace).
    gl_FragColor = vec4(max(colorSum / max(weightSum, 1e-5), vec3(0.0)), 1.0);
    #include <tonemapping_fragment>
    if (uOutputSRGB > 0.5) {
      gl_FragColor.rgb = annolidLinearToSRGB(gl_FragColor.rgb);
    }

    // ── Vignette (in display space) ──
    float vig = 1.0 - smoothstep(0.32, 0.32 + max(1e-4, uVigSoftness), dist) * uVigIntensity;
    gl_FragColor.rgb *= vig;

    // ── Saturation & Brightness adjustment ──
    float luma = dot(gl_FragColor.rgb, vec3(0.2126, 0.7152, 0.0722));
    gl_FragColor.rgb = mix(vec3(luma), gl_FragColor.rgb, uSaturation) * uBrightness;

    // ── Subtle Film Grain ──
    float grainPhase = mod(uTime, 60.0) * 37.0 + 1.0;
    float grain = fract(sin(dot(vUv * grainPhase, vec2(12.9898, 78.233))) * 43758.5453);
    gl_FragColor.rgb = clamp(gl_FragColor.rgb + (grain - 0.5) * uGrainStrength, 0.0, 1.0);
  }
`;


// ─────────────────────────────────────────────────────────────
// AnnolidShaders class
// ─────────────────────────────────────────────────────────────

export class AnnolidShaders {
    /**
     * @param {object} THREE — The Three.js module (import * as THREE from ...)
     */
    constructor(THREE) {
        this.THREE = THREE;
        this._applied = new WeakMap(); // Track per-material shader keys & live uniform refs
    }

    _ensureRecord(material) {
        if (!this._applied.has(material)) {
            this._applied.set(material, {
                keys: [],
                applied: Object.create(null),
                uniforms: Object.create(null),
                baseOnBeforeCompile: material.onBeforeCompile,
                baseCacheKey: material.customProgramCacheKey,
            });
        }
        return this._applied.get(material);
    }

    // ─── Internal: chain onBeforeCompile with unique program cache key ───
    _chainCompile(material, key, compileFn) {
        const record = this._ensureRecord(material);
        if (record.applied[key]) return; // Already applied

        record.applied[key] = true;
        record.keys.push(key);

        const prevCompile = material.onBeforeCompile;
        material.onBeforeCompile = (shader, renderer) => {
            if (prevCompile) prevCompile(shader, renderer);
            compileFn(shader, record.uniforms);
        };

        const baseCacheKey = record.baseCacheKey;
        const signature = record.keys.join('+');
        material.customProgramCacheKey = function () {
            const prefix = typeof baseCacheKey === 'function' ? baseCacheKey.call(this) : '';
            return `${prefix}|annolid:${signature}`;
        };

        // Force recompilation
        material.needsUpdate = true;
    }

    /**
     * Return tracked uniform objects for a material, or null if none applied.
     */
    getUniforms(material) {
        return this._applied.get(material)?.uniforms ?? null;
    }

    /**
     * Update live uniform values on an already-configured material without recompiling.
     */
    updateUniforms(material, values = {}) {
        const uniforms = this.getUniforms(material);
        if (!uniforms || !values) return false;
        let updated = false;
        for (const [name, nextVal] of Object.entries(values)) {
            const target = uniforms[name];
            if (!target || nextVal === undefined) continue;
            if (target.value?.isColor && Array.isArray(nextVal)) {
                target.value.setRGB(nextVal[0], nextVal[1], nextVal[2]);
                updated = true;
            } else if (target.value?.isVector3 && Array.isArray(nextVal)) {
                target.value.set(nextVal[0], nextVal[1], nextVal[2]);
                updated = true;
            } else {
                target.value = nextVal;
                updated = true;
            }
        }
        return updated;
    }

    // ─── Internal: inject GLSL into shader source ─────────────
    static _inject(source, anchor, chunk, position = 'after') {
        if (!source.includes(anchor)) {
            console.warn(`[AnnolidShaders] anchor not found: "${anchor}"`);
            return source;
        }
        if (position === 'before') {
            return source.replace(anchor, chunk + '\n' + anchor);
        }
        return source.replace(anchor, anchor + '\n' + chunk);
    }

    // ═══════════════════════════════════════════════════════════
    // 1. FUR / HAIR ANISOTROPIC SHADER
    // ═══════════════════════════════════════════════════════════
    /**
     * @param {MeshStandardMaterial|MeshPhysicalMaterial} material
     * @param {object} opts
     * @param {number[]} opts.rootColor  — RGB [0-1] for strand root
     * @param {number[]} opts.tipColor   — RGB [0-1] for strand tip
     * @param {number}   opts.anisotropy — Specular exponent (20-200)
     * @param {number}   opts.specIntensity — Highlight brightness (0-2)
     * @param {number[]} [opts.flowDirection] — Tangent flow direction [x,y,z] (alias: opts.flowDir)
     */
    applyFurShader(material, opts = {}) {
        const rootColor = opts.rootColor ?? [0.15, 0.12, 0.10];
        const tipColor = opts.tipColor ?? [0.35, 0.30, 0.28];
        const aniso = opts.anisotropy ?? 80.0;
        const specI = opts.specIntensity ?? 0.4;
        const flow = opts.flowDirection ?? opts.flowDir ?? [0.0, 0.0, -1.0];
        const flowVec = new this.THREE.Vector3(flow[0] ?? 0, flow[1] ?? 0, flow[2] ?? -1);
        if (flowVec.lengthSq() < 1e-8) flowVec.set(0, 0, -1);
        else flowVec.normalize();

        const record = this._ensureRecord(material);
        if (!record.applied.fur) {
            record.uniforms.uFurRootColor = { value: new this.THREE.Color(...rootColor) };
            record.uniforms.uFurTipColor = { value: new this.THREE.Color(...tipColor) };
            record.uniforms.uFurAnisotropy = { value: aniso };
            record.uniforms.uFurSpecIntensity = { value: specI };
            record.uniforms.uFurFlowDir = { value: flowVec };
        }

        this._chainCompile(material, 'fur', (shader, uniforms) => {
            // Uniforms
            shader.uniforms.uFurRootColor = uniforms.uFurRootColor;
            shader.uniforms.uFurTipColor = uniforms.uFurTipColor;
            shader.uniforms.uFurAnisotropy = uniforms.uFurAnisotropy;
            shader.uniforms.uFurSpecIntensity = uniforms.uFurSpecIntensity;
            shader.uniforms.uFurFlowDir = uniforms.uFurFlowDir;

            // Vertex shader
            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader,
                'void main() {',
                FUR_UNIFORMS_GLSL + FUR_VERTEX_GLSL,
                'before'
            );
            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader,
                '#include <worldpos_vertex>',
                FUR_VERTEX_MAIN_GLSL
            );

            // Fragment shader
            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader,
                'void main() {',
                FUR_UNIFORMS_GLSL + FUR_FRAGMENT_GLSL,
                'before'
            );
            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader,
                '#include <dithering_fragment>',
                FUR_FRAGMENT_MAIN_GLSL
            );
        });

        return material;
    }

    // ═══════════════════════════════════════════════════════════
    // 2. SUBSURFACE SCATTER SHADER
    // ═══════════════════════════════════════════════════════════
    /**
     * @param {MeshStandardMaterial|MeshPhysicalMaterial} material
     * @param {object} opts
     * @param {number[]} opts.scatterColor — RGB scatter color [0-1]
     * @param {number}   opts.thickness    — Material thickness factor
     * @param {number}   opts.distortion   — Normal distortion for backlight
     * @param {number}   opts.wrap         — Wrap-diffuse amount (0-1)
     * @param {number}   opts.backlight    — Back-light transmission strength
     */
    applySSSShader(material, opts = {}) {
        const scatterColor = opts.scatterColor ?? [0.8, 0.3, 0.25];
        const thickness = opts.thickness ?? 0.5;
        const distortion = opts.distortion ?? 0.2;
        const wrap = opts.wrap ?? 0.5;
        const backlight = opts.backlight ?? 0.6;

        const record = this._ensureRecord(material);
        if (!record.applied.sss) {
            record.uniforms.uSSSColor = { value: new this.THREE.Color(...scatterColor) };
            record.uniforms.uSSSThickness = { value: thickness };
            record.uniforms.uSSSDistortion = { value: distortion };
            record.uniforms.uSSSWrap = { value: wrap };
            record.uniforms.uSSSBacklight = { value: backlight };
        }

        this._chainCompile(material, 'sss', (shader, uniforms) => {
            shader.uniforms.uSSSColor = uniforms.uSSSColor;
            shader.uniforms.uSSSThickness = uniforms.uSSSThickness;
            shader.uniforms.uSSSDistortion = uniforms.uSSSDistortion;
            shader.uniforms.uSSSWrap = uniforms.uSSSWrap;
            shader.uniforms.uSSSBacklight = uniforms.uSSSBacklight;

            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader, 'void main() {',
                SSS_UNIFORMS_GLSL + SSS_VERTEX_GLSL, 'before'
            );
            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader, '#include <worldpos_vertex>',
                SSS_VERTEX_MAIN_GLSL
            );

            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader, 'void main() {',
                SSS_UNIFORMS_GLSL + SSS_FRAGMENT_GLSL, 'before'
            );
            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader, '#include <dithering_fragment>',
                SSS_FRAGMENT_MAIN_GLSL
            );
        });

        return material;
    }

    // ═══════════════════════════════════════════════════════════
    // 3. MICRO-DETAIL NORMAL SHADER
    // ═══════════════════════════════════════════════════════════
    /**
     * @param {MeshStandardMaterial|MeshPhysicalMaterial} material
     * @param {object} opts
     * @param {number} opts.scale    — World-space frequency of detail (5-200)
     * @param {number} opts.strength — Perturbation magnitude (0-0.5)
     */
    applyMicroDetail(material, opts = {}) {
        const scale = opts.scale ?? 40.0;
        const strength = opts.strength ?? 0.08;

        const record = this._ensureRecord(material);
        if (!record.applied.micro) {
            record.uniforms.uMicroScale = { value: scale };
            record.uniforms.uMicroStrength = { value: strength };
        }

        this._chainCompile(material, 'micro', (shader, uniforms) => {
            shader.uniforms.uMicroScale = uniforms.uMicroScale;
            shader.uniforms.uMicroStrength = uniforms.uMicroStrength;

            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader, 'void main() {',
                MICRO_UNIFORMS_GLSL + MICRO_VERTEX_GLSL, 'before'
            );
            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader, '#include <worldpos_vertex>',
                MICRO_VERTEX_MAIN_GLSL
            );

            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader, 'void main() {',
                MICRO_UNIFORMS_GLSL + MICRO_FRAGMENT_GLSL, 'before'
            );
            // Inject BEFORE normal_fragment_maps so our perturbation affects lighting
            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader, '#include <normal_fragment_maps>',
                MICRO_FRAGMENT_PRENORMAL_GLSL
            );
        });

        return material;
    }

    // ═══════════════════════════════════════════════════════════
    // 4. CONTACT AO SHADER
    // ═══════════════════════════════════════════════════════════
    /**
     * @param {MeshStandardMaterial|MeshPhysicalMaterial} material
     * @param {object} opts
     * @param {number} opts.groundY   — World Y of ground plane (default 0)
     * @param {number} opts.radius    — AO falloff radius (default 1.5)
     * @param {number} opts.intensity — Darkening strength (0-1)
     * @param {number[]} opts.color   — AO tint color RGB [0-1]
     */
    applyContactAO(material, opts = {}) {
        const groundY = opts.groundY ?? 0.0;
        const radius = opts.radius ?? 1.5;
        const intensity = opts.intensity ?? 0.4;
        const color = opts.color ?? [0.02, 0.01, 0.0];

        const record = this._ensureRecord(material);
        if (!record.applied.ao) {
            record.uniforms.uAOGroundY = { value: groundY };
            record.uniforms.uAORadius = { value: radius };
            record.uniforms.uAOIntensity = { value: intensity };
            record.uniforms.uAOColor = { value: new this.THREE.Color(...color) };
        }

        this._chainCompile(material, 'ao', (shader, uniforms) => {
            shader.uniforms.uAOGroundY = uniforms.uAOGroundY;
            shader.uniforms.uAORadius = uniforms.uAORadius;
            shader.uniforms.uAOIntensity = uniforms.uAOIntensity;
            shader.uniforms.uAOColor = uniforms.uAOColor;

            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader, 'void main() {',
                AO_UNIFORMS_GLSL + AO_VERTEX_GLSL, 'before'
            );
            shader.vertexShader = AnnolidShaders._inject(
                shader.vertexShader, '#include <worldpos_vertex>',
                AO_VERTEX_MAIN_GLSL
            );

            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader, 'void main() {',
                AO_UNIFORMS_GLSL + AO_FRAGMENT_GLSL, 'before'
            );
            shader.fragmentShader = AnnolidShaders._inject(
                shader.fragmentShader, '#include <dithering_fragment>',
                AO_FRAGMENT_MAIN_GLSL
            );
        });

        return material;
    }

    // ═══════════════════════════════════════════════════════════
    // 5. POST-PROCESSING FULLSCREEN QUAD
    // ═══════════════════════════════════════════════════════════
    /**
     * Creates a post-processing setup: renders scene to a half-float HDR target,
     * applies renderer tone mapping + sRGB encoding, and adds subtle DoF, vignette & grain.
     *
     * @param {WebGLRenderer} renderer
     * @param {object} opts
     * @param {number} opts.vigIntensity  — Vignette darkness (0-1)
     * @param {number} opts.vigSoftness   — Vignette edge softness (0-1)
     * @param {number} opts.grainStrength — Film grain amount (0-0.1)
     * @param {number} opts.saturation    — Color saturation (0.5-1.5)
     * @param {number} opts.brightness    — Overall brightness multiplier
     * @returns {{ render, resize, uniforms, enabled, dispose }}
     */
    createPostProcessing(renderer, opts = {}) {
        const THREE = this.THREE;
        const vigI = opts.vigIntensity ?? 0.16;
        const vigS = opts.vigSoftness ?? 0.54;
        const grainS = opts.grainStrength ?? 0.004;
        const sat = opts.saturation ?? 1.0;
        const bright = opts.brightness ?? 1.0;
        const aperture = opts.aperture ?? 1.4;

        const getPixelSize = () => {
            const sz = renderer.getSize(new THREE.Vector2());
            const pr = renderer.getPixelRatio();
            return {
                width: Math.max(1, Math.round(sz.x * pr)),
                height: Math.max(1, Math.round(sz.y * pr)),
            };
        };

        const initial = getPixelSize();
        const renderTarget = new THREE.WebGLRenderTarget(
            initial.width,
            initial.height,
            { format: THREE.RGBAFormat, type: THREE.HalfFloatType }
        );
        renderTarget.depthTexture = new THREE.DepthTexture(initial.width, initial.height);
        renderTarget.depthTexture.type = THREE.UnsignedIntType;

        const postUniforms = {
            tDiffuse: { value: renderTarget.texture },
            tDepth: { value: renderTarget.depthTexture },
            uTime: { value: 0.0 },
            uVigIntensity: { value: vigI },
            uVigSoftness: { value: vigS },
            uGrainStrength: { value: grainS },
            uSaturation: { value: sat },
            uBrightness: { value: bright },
            uCameraNear: { value: 0.1 },
            uCameraFar: { value: 50.0 },
            uFocusDistance: { value: 0.05 },
            uAperture: { value: aperture },
            uOutputSRGB: { value: renderer.outputColorSpace === THREE.SRGBColorSpace ? 1.0 : 0.0 },
            uResolution: { value: new THREE.Vector2(initial.width, initial.height) }
        };

        const postMaterial = new THREE.ShaderMaterial({
            vertexShader: POST_VERTEX_GLSL,
            fragmentShader: POST_FRAGMENT_GLSL,
            uniforms: postUniforms,
            toneMapped: true,
            depthTest: false,
            depthWrite: false,
        });

        const postGeometry = new THREE.PlaneGeometry(2, 2);
        const postMesh = new THREE.Mesh(postGeometry, postMaterial);
        const postScene = new THREE.Scene();
        postScene.add(postMesh);

        const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

        let enabled = false;

        return {
            /** Toggle post-processing on/off */
            get enabled() { return enabled; },
            set enabled(val) { enabled = Boolean(val); },

            /** Uniforms for live GUI tweaking */
            uniforms: postUniforms,

            /** Call this instead of renderer.render(scene, camera) */
            render(scene, camera, time = 0, focusTarget = null) {
                if (!enabled) {
                    renderer.render(scene, camera);
                    return;
                }
                postUniforms.uTime.value = time;
                postUniforms.uCameraNear.value = camera.near;
                postUniforms.uCameraFar.value = camera.far;
                postUniforms.uOutputSRGB.value = renderer.outputColorSpace === THREE.SRGBColorSpace ? 1.0 : 0.0;
                if (focusTarget && camera?.position) {
                    const span = Math.max(1e-4, camera.far - camera.near);
                    const focusWorldDist = camera.position.distanceTo(focusTarget);
                    postUniforms.uFocusDistance.value = THREE.MathUtils.clamp(
                        (focusWorldDist - camera.near) / span,
                        0.0,
                        1.0
                    );
                }

                // Render scene to offscreen linear HDR target
                renderer.setRenderTarget(renderTarget);
                renderer.render(scene, camera);
                renderer.setRenderTarget(null);
                // Render tone-mapped + sRGB-encoded post quad to screen
                renderer.render(postScene, postCamera);
            },

            /** Call on window or pixel-ratio resize */
            resize() {
                const next = getPixelSize();
                renderTarget.setSize(next.width, next.height);
                postUniforms.uResolution.value.set(next.width, next.height);
            },

            /** Cleanup */
            dispose() {
                renderTarget.dispose();
                postMaterial.dispose();
                postGeometry.dispose();
            }
        };
    }

    // ═══════════════════════════════════════════════════════════
    // UTILITY: Remove all shader effects from a material
    // ═══════════════════════════════════════════════════════════
    removeAllShaders(material) {
        const record = this._applied.get(material);
        if (record) {
            material.onBeforeCompile = record.baseOnBeforeCompile || (() => { });
            material.customProgramCacheKey = typeof record.baseCacheKey === 'function'
                ? record.baseCacheKey
                : function () { return ''; };
            this._applied.delete(material);
        } else {
            material.onBeforeCompile = () => { };
        }
        material.needsUpdate = true;
    }
}
