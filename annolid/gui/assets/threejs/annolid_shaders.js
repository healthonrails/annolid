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
  // Pass world-space normal & position for anisotropic calc
  varying vec3 vFurWorldNormal;
  varying vec3 vFurWorldPos;
`;

const FUR_VERTEX_MAIN_GLSL = /* glsl */ `
  vFurWorldNormal = normalize((modelMatrix * vec4(objectNormal, 0.0)).xyz);
  vFurWorldPos    = (modelMatrix * vec4(position, 1.0)).xyz;
`;

const FUR_FRAGMENT_GLSL = /* glsl */ `
  varying vec3 vFurWorldNormal;
  varying vec3 vFurWorldPos;
`;

const FUR_FRAGMENT_MAIN_GLSL = /* glsl */ `
  {
    // Marschner-inspired double anisotropic highlight for hair/fur
    vec3 tangent = normalize(cross(vFurWorldNormal, uFurFlowDir));
    vec3 viewDir = normalize(cameraPosition - vFurWorldPos);
    vec3 lightDir = normalize(vec3(0.3, 1.0, 0.2));
    vec3 halfDir  = normalize(viewDir + lightDir);

    // Shift tangents for primary (reflection) and secondary (transmission) highlights
    vec3 t1 = normalize(tangent + vFurWorldNormal * -0.05); // Shift down towards root
    vec3 t2 = normalize(tangent + vFurWorldNormal * 0.1);   // Shift up towards tip

    float TdotH1 = dot(t1, halfDir);
    float TdotH2 = dot(t2, halfDir);

    float sinTH1 = sqrt(max(0.0, 1.0 - TdotH1 * TdotH1));
    float sinTH2 = sqrt(max(0.0, 1.0 - TdotH2 * TdotH2));

    float aniso1 = pow(sinTH1, uFurAnisotropy) * uFurSpecIntensity;
    float aniso2 = pow(sinTH2, uFurAnisotropy * 0.5) * uFurSpecIntensity * 0.5;

    // Root-to-tip color gradient based on object-space Y
    float strandProgress = clamp(
      (vFurWorldPos.y - uFurRootColor.x * 0.01) * 0.5 + 0.5, 0.0, 1.0
    );
    vec3 furTint = mix(uFurRootColor, uFurTipColor, strandProgress);

    // Primary highlight is white/light, secondary is tinted by hair color
    vec3 totalAniso = vec3(aniso1) + furTint * aniso2;

    // Blend highlight and root-tip color
    gl_FragColor.rgb = gl_FragColor.rgb * furTint + totalAniso;
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
  vSSSWorldNormal = normalize((modelMatrix * vec4(objectNormal, 0.0)).xyz);
  vSSSWorldPos    = (modelMatrix * vec4(position, 1.0)).xyz;
`;

const SSS_FRAGMENT_GLSL = /* glsl */ `
  varying vec3 vSSSWorldNormal;
  varying vec3 vSSSWorldPos;
`;

const SSS_FRAGMENT_MAIN_GLSL = /* glsl */ `
  {
    vec3 viewDir  = normalize(cameraPosition - vSSSWorldPos);
    vec3 lightDir = normalize(vec3(0.3, 1.0, 0.2));
    vec3 nrm      = normalize(vSSSWorldNormal);

    // Wrap diffuse (soft terminator)
    float wrapDiffuse = max(0.0,
      (dot(nrm, lightDir) + uSSSWrap) / (1.0 + uSSSWrap)
    );

    // Pseudo-thickness based on viewing angle (edges appear thinner)
    float viewDotNrm = abs(dot(viewDir, nrm));
    float thicknessMap = mix(uSSSThickness * 0.3, uSSSThickness, viewDotNrm);

    vec3 scatter = uSSSColor * wrapDiffuse * thicknessMap;

    // Back-light transmission with spectral shift (light becomes warmer as it passes through flesh)
    vec3 scatterDir = lightDir + nrm * uSSSDistortion;
    float backDot   = max(0.0, dot(viewDir, -normalize(scatterDir)));

    float scatterPow = pow(backDot, 5.0) * smoothstep(0.0, 1.0, 1.0 - thicknessMap);

    // Shift color towards red/orange for deep scattering
    vec3 transmitColor = max(uSSSColor, vec3(0.8, 0.2, 0.05));
    scatter += transmitColor * scatterPow * uSSSBacklight;

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
    normal = normalize(normal + microPerturb);
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
    float ao = smoothstep(0.0, uAORadius, groundDist);
    ao = mix(1.0 - uAOIntensity, 1.0, pow(ao, 0.6));
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
  uniform float uTime;
  uniform float uVigIntensity;
  uniform float uVigSoftness;
  uniform float uGrainStrength;
  uniform float uSaturation;
  uniform float uBrightness;
  varying vec2 vUv;

  void main() {
    // ── Chromatic Aberration ──
    vec2 dir = vUv - 0.5;
    float dist = length(dir);
    vec2 offset = dir * (dist * 0.008); // Intensity scales with distance from center

    float r = texture2D(tDiffuse, vUv - offset).r;
    float g = texture2D(tDiffuse, vUv).g;
    float b = texture2D(tDiffuse, vUv + offset).b;
    vec4 color = vec4(r, g, b, 1.0);

    // ── Vignette ──
    float vig = 1.0 - smoothstep(0.3, 0.3 + uVigSoftness, dist) * uVigIntensity;
    color.rgb *= vig;

    // ── Film Grain ──
    float grain = fract(
      sin(dot(vUv * (uTime * 100.0 + 1.0), vec2(12.9898, 78.233))) * 43758.5453
    );
    color.rgb += (grain - 0.5) * uGrainStrength;

    // ── Saturation adjustment ──
    float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    color.rgb = mix(vec3(luma), color.rgb, uSaturation);

    // ── Brightness / Minor Contrast curve ──
    color.rgb *= uBrightness;
    // Mild S-curve for contrast
    color.rgb = smoothstep(0.0, 1.0, color.rgb);

    gl_FragColor = color;
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
        this._applied = new WeakMap(); // Track which effects are on which materials
    }

    // ─── Internal: chain onBeforeCompile ───────────────────────
    _chainCompile(material, key, compileFn) {
        if (!this._applied.has(material)) {
            this._applied.set(material, {});
        }
        const record = this._applied.get(material);
        if (record[key]) return; // Already applied

        const prevCompile = material.onBeforeCompile;
        material.onBeforeCompile = (shader, renderer) => {
            if (prevCompile) prevCompile(shader, renderer);
            compileFn(shader);
        };
        // Force recompilation
        material.needsUpdate = true;
        record[key] = true;
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
     * @param {number[]} opts.flowDirection — Tangent flow direction [x,y,z]
     */
    applyFurShader(material, opts = {}) {
        const rootColor = opts.rootColor || [0.15, 0.12, 0.10];
        const tipColor = opts.tipColor || [0.35, 0.30, 0.28];
        const aniso = opts.anisotropy || 80.0;
        const specI = opts.specIntensity || 0.4;
        const flow = opts.flowDirection || [0.0, 0.0, -1.0];

        this._chainCompile(material, 'fur', (shader) => {
            // Uniforms
            shader.uniforms.uFurRootColor = { value: new this.THREE.Color(...rootColor) };
            shader.uniforms.uFurTipColor = { value: new this.THREE.Color(...tipColor) };
            shader.uniforms.uFurAnisotropy = { value: aniso };
            shader.uniforms.uFurSpecIntensity = { value: specI };
            shader.uniforms.uFurFlowDir = { value: new this.THREE.Vector3(...flow) };

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
        const scatterColor = opts.scatterColor || [0.8, 0.3, 0.25];
        const thickness = opts.thickness || 0.5;
        const distortion = opts.distortion || 0.2;
        const wrap = opts.wrap || 0.5;
        const backlight = opts.backlight || 0.6;

        this._chainCompile(material, 'sss', (shader) => {
            shader.uniforms.uSSSColor = { value: new this.THREE.Color(...scatterColor) };
            shader.uniforms.uSSSThickness = { value: thickness };
            shader.uniforms.uSSSDistortion = { value: distortion };
            shader.uniforms.uSSSWrap = { value: wrap };
            shader.uniforms.uSSSBacklight = { value: backlight };

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
        const scale = opts.scale || 40.0;
        const strength = opts.strength || 0.08;

        this._chainCompile(material, 'micro', (shader) => {
            shader.uniforms.uMicroScale = { value: scale };
            shader.uniforms.uMicroStrength = { value: strength };

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
        const radius = opts.radius || 1.5;
        const intensity = opts.intensity || 0.4;
        const color = opts.color || [0.02, 0.01, 0.0];

        this._chainCompile(material, 'ao', (shader) => {
            shader.uniforms.uAOGroundY = { value: groundY };
            shader.uniforms.uAORadius = { value: radius };
            shader.uniforms.uAOIntensity = { value: intensity };
            shader.uniforms.uAOColor = { value: new this.THREE.Color(...color) };

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
     * Creates a post-processing setup: renders scene to a render target,
     * then displays it through a fullscreen quad with vignette + grain.
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
        const vigI = opts.vigIntensity || 0.35;
        const vigS = opts.vigSoftness || 0.45;
        const grainS = opts.grainStrength || 0.03;
        const sat = opts.saturation || 1.05;
        const bright = opts.brightness || 1.0;

        const size = renderer.getSize(new THREE.Vector2());
        const renderTarget = new THREE.WebGLRenderTarget(
            size.x * renderer.getPixelRatio(),
            size.y * renderer.getPixelRatio(),
            { format: THREE.RGBAFormat, type: THREE.UnsignedByteType }
        );

        const postUniforms = {
            tDiffuse: { value: renderTarget.texture },
            uTime: { value: 0.0 },
            uVigIntensity: { value: vigI },
            uVigSoftness: { value: vigS },
            uGrainStrength: { value: grainS },
            uSaturation: { value: sat },
            uBrightness: { value: bright },
        };

        const postMaterial = new THREE.ShaderMaterial({
            vertexShader: POST_VERTEX_GLSL,
            fragmentShader: POST_FRAGMENT_GLSL,
            uniforms: postUniforms,
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
            set enabled(val) { enabled = val; },

            /** Uniforms for live GUI tweaking */
            uniforms: postUniforms,

            /** Call this instead of renderer.render(scene, camera) */
            render(scene, camera, time = 0) {
                if (!enabled) {
                    renderer.render(scene, camera);
                    return;
                }
                postUniforms.uTime.value = time;
                // Render scene to offscreen target
                renderer.setRenderTarget(renderTarget);
                renderer.render(scene, camera);
                renderer.setRenderTarget(null);
                // Render post quad to screen
                renderer.render(postScene, postCamera);
            },

            /** Call on window resize */
            resize() {
                const sz = renderer.getSize(new THREE.Vector2());
                const pr = renderer.getPixelRatio();
                renderTarget.setSize(sz.x * pr, sz.y * pr);
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
        material.onBeforeCompile = () => { };
        material.needsUpdate = true;
        if (this._applied.has(material)) {
            this._applied.delete(material);
        }
    }
}
