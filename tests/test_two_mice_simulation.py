"""Run the dependency-free simulation core in Node; no browser or CDN needed."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SIMULATION = REPO_ROOT / "annolid/gui/assets/threejs/two_mice_simulation.js"


def _find_node_binary() -> str | None:
    node = shutil.which("node")
    if node is not None:
        return node
    try:
        from playwright._impl._driver import compute_driver_executable

        driver = compute_driver_executable()
        candidate = Path(driver[0] if isinstance(driver, tuple) else driver)
        if candidate.is_file():
            return str(candidate)
    except Exception:
        return None
    return None


def run_javascript(body: str, include_simulation: bool = True):
    node = _find_node_binary()
    if node is None:
        pytest.skip("Node.js is required to validate the JavaScript simulation")
    prefix = SIMULATION.read_text(encoding="utf-8") + "\n" if include_simulation else ""
    result = subprocess.run(
        [node, "--input-type=module", "-e", prefix + body],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("fps", [30, 60, 144])
def test_fixed_physics_matches_across_display_rates(fps):
    result = run_javascript(
        f"""
        const clock = new FixedStepClock();
        const random = createSeededRandom(123);
        let position = 0, velocity = 1;
        for (let i = 0; i < {fps} * 10; i++) {{
          clock.advance(1 / {fps}, (time, dt) => {{
            velocity += (random() - 0.5 - velocity * 0.3) * dt;
            position += velocity * dt;
          }});
        }}
        console.log(JSON.stringify({{steps: clock.steps, position, velocity}}));
        """
    )
    assert result["steps"] == 300
    reference = run_javascript(
        """
        const random = createSeededRandom(123);
        let position = 0, velocity = 1;
        for (let i = 0; i < 300; i++) {
          velocity += (random() - 0.5 - velocity * 0.3) / 30;
          position += velocity / 30;
        }
        console.log(JSON.stringify({position, velocity}));
        """
    )
    assert result["position"] == pytest.approx(reference["position"], abs=1e-12)
    assert result["velocity"] == pytest.approx(reference["velocity"], abs=1e-12)


def test_pause_invalid_deltas_and_long_resume_are_bounded():
    result = run_javascript(
        """
        const clock = new FixedStepClock();
        let calls = 0;
        const step = () => calls++;
        clock.advance(1 / 60, step);
        clock.advance(90, step, true);
        const paused = [clock.steps, calls, clock.remainder];
        for (const dt of [NaN, Infinity, -1, 0]) clock.advance(dt, step);
        clock.advance(1 / 60, step);
        const beforeResume = calls;
        clock.advance(90, step);
        console.log(JSON.stringify({paused, beforeResume, calls, remainder: clock.remainder}));
        """
    )
    assert result["paused"] == [0, 0, 0]
    assert result["beforeResume"] == 0
    assert result["calls"] == 6
    assert 0 <= result["remainder"] < 1 / 30


def test_wall_normals_repel_and_attract_on_all_four_sides():
    result = run_javascript(
        """
        const normals = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        const forces = normals.map(([x, z]) => {
          const near = wallNormalForce(0.4, 5.8, 1.8);
          const far = wallNormalForce(5, 5.8, 1.8);
          return {nearDot: x*x*near + z*z*near, farDot: x*x*far + z*z*far};
        });
        console.log(JSON.stringify({forces,
          boundary: wallNormalForce(2, 5.8, 1.8),
          otherWall: wallNormalForce(5, 5.8, 1.8, false),
          disabled: wallNormalForce(5, 5.8, 0)}));
        """
    )
    assert all(force["nearDot"] > 0 for force in result["forces"])
    assert all(force["farDot"] < 0 for force in result["forces"])
    assert result["boundary"] == result["otherWall"] == result["disabled"] == 0


def test_spine_flex_arc_kinematics_and_oriented_wall_margin():
    result = run_javascript(
        """
        const rest = evaluateSpinePoint(0.45, 0.25, 2.1, null, 2.68);
        const bent = evaluateSpinePoint(0.0, 0.20, 2.4, { bend: 0.55, sCurve: 0.0, arch: 0.08, stretch: 0.0 }, 2.68);
        const stretched = evaluateSpinePoint(0.50, 0.30, 2.4, { bend: 0.0, sCurve: 0.0, arch: 0.0, stretch: 0.20 }, 2.68);
        const bunched = evaluateSpinePoint(0.50, 0.30, 2.4, { bend: 0.0, sCurve: 0.0, arch: 0.0, stretch: -0.18 }, 2.68);

        // Parallel to North wall (normal = [0, 1], heading = [1, 0]) vs nose-first / rump-first into wall
        const parallelMargin = computeOrientedWallMargin(1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0);
        const headOnMargin = computeOrientedWallMargin(0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0);
        const rumpFirstMargin = computeOrientedWallMargin(0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0);
        const stretchedHeadOn = computeOrientedWallMargin(0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.20);

        console.log(JSON.stringify({
          restExact:
            Math.abs(rest.x - 0.45) < 1e-9
            && Math.abs(rest.y - 0.25) < 1e-9
            && Math.abs(rest.z - 2.1) < 1e-9
            && Math.abs(rest.yaw) < 1e-9,
          bentLateralPositive: bent.x > 0.35,
          bentForeshortenedZ: bent.z < 2.4,
          bentYawPositive: bent.yaw > 0.35,
          stretchedLongerZ: stretched.z > 2.65,
          stretchedSlimmerX: stretched.x < 0.50,
          bunchedShorterZ: bunched.z < 2.15,
          bunchedWiderX: bunched.x > 0.50,
          parallelCloserThanHeadOn: parallelMargin < headOnMargin - 0.75,
          parallelCloseThigmotaxis: parallelMargin < 1.75 && parallelMargin > 1.25,
          rumpKeepsSacralRootInsideWall: rumpFirstMargin >= 2.75,
          stretchedIncreasesHeadOn: stretchedHeadOn > headOnMargin,
        }));
        """
    )
    assert all(result.values()), f"Failed spinal/wall kinematic checks: {result}"


def test_seeded_streams_repeat_without_modifying_global_random():
    result = run_javascript(
        """
        const original = Math.random;
        const a = createSeededRandom(123), b = createSeededRandom(123);
        const appearance = createSeededRandom(999);
        let equal = true, bounded = true;
        for (let i = 0; i < 10000; i++) {
          for (let j = 0; j < i % 11; j++) appearance();
          const value = a();
          equal &&= value === b();
          bounded &&= value >= 0 && value < 1;
        }
        console.log(JSON.stringify({equal, bounded, unchanged: original === Math.random}));
        """
    )
    assert result == {"equal": True, "bounded": True, "unchanged": True}


def test_annolid_shaders_program_keys_zero_options_and_live_uniforms():
    result = run_javascript(
        """
        import { AnnolidShaders } from './annolid/gui/assets/threejs/annolid_shaders.js';

        class Color {
          constructor(r = 0, g = 0, b = 0) {
            this.isColor = true;
            this.r = r; this.g = g; this.b = b;
          }
          setRGB(r, g, b) { this.r = r; this.g = g; this.b = b; return this; }
        }
        class Vector3 {
          constructor(x = 0, y = 0, z = 0) {
            this.isVector3 = true;
            this.x = x; this.y = y; this.z = z;
          }
          set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
          lengthSq() { return this.x * this.x + this.y * this.y + this.z * this.z; }
          normalize() {
            const len = Math.sqrt(this.lengthSq()) || 1;
            this.x /= len; this.y /= len; this.z /= len;
            return this;
          }
        }
        const shaders = new AnnolidShaders({ Color, Vector3 });

        const makeMat = () => ({
          needsUpdate: false,
          onBeforeCompile: () => {},
          customProgramCacheKey() { return 'base'; },
        });
        const makeShaderSource = () => ({
          uniforms: {},
          vertexShader: 'void main() {\\n#include <worldpos_vertex>\\n}',
          fragmentShader: 'void main() {\\n#include <normal_fragment_maps>\\n#include <dithering_fragment>\\n}',
        });

        const furAndAo = makeMat();
        shaders.applyFurShader(furAndAo, {
          anisotropy: 0,
          specIntensity: 0,
          flowDir: [0, -0.6, -0.8],
        });
        shaders.applyContactAO(furAndAo, { radius: 0, intensity: 0 });

        const sssOnly = makeMat();
        shaders.applySSSShader(sssOnly, { thickness: 0, wrap: 0, backlight: 0 });

        const bodySpine = makeMat();
        shaders.applyFurShader(bodySpine, { anisotropy: 0.4, specIntensity: 0.12 });
        shaders.applySpinalFlexShader(bodySpine, { halfLen: 2.68 });

        const compiledFur = makeShaderSource();
        furAndAo.onBeforeCompile(compiledFur, null);
        shaders.updateUniforms(furAndAo, { uAOIntensity: 0.25, uFurSpecIntensity: 0.08 });

        const compiledSpine = {
          uniforms: {},
          vertexShader: 'void main() {\\n#include <beginnormal_vertex>\\n#include <project_vertex>\\n#include <worldpos_vertex>\\n}',
          fragmentShader: 'void main() {\\n#include <normal_fragment_maps>\\n#include <dithering_fragment>\\n}',
        };
        bodySpine.onBeforeCompile(compiledSpine, null);
        shaders.updateUniforms(bodySpine, { uSpineBend: 0.42, uAxialStretch: 0.14 });

        const compiledSss = makeShaderSource();
        sssOnly.onBeforeCompile(compiledSss, null);

        const keyFurAo = furAndAo.customProgramCacheKey();
        const keySss = sssOnly.customProgramCacheKey();
        const keyBodySpine = bodySpine.customProgramCacheKey();

        shaders.removeAllShaders(sssOnly);
        const restoredKey = sssOnly.customProgramCacheKey();

        console.log(JSON.stringify({
          keyFurAo,
          keySss,
          keyBodySpine,
          restoredKey,
          distinctKeys: keyFurAo !== keySss && keyBodySpine !== keyFurAo,
          zeroAnisotropyPreserved: compiledFur.uniforms.uFurAnisotropy.value === 0,
          zeroRadiusPreserved: compiledFur.uniforms.uAORadius.value === 0,
          zeroSssPreserved:
            compiledSss.uniforms.uSSSThickness.value === 0
            && compiledSss.uniforms.uSSSWrap.value === 0
            && compiledSss.uniforms.uSSSBacklight.value === 0,
          flowDirAliasY: Number(compiledFur.uniforms.uFurFlowDir.value.y.toFixed(4)),
          flowDirAliasZ: Number(compiledFur.uniforms.uFurFlowDir.value.z.toFixed(4)),
          liveAoUpdated: compiledFur.uniforms.uAOIntensity.value === 0.25,
          liveFurUpdated: compiledFur.uniforms.uFurSpecIntensity.value === 0.08,
          worldFlowInjected: compiledFur.vertexShader.includes('vFurWorldFlow'),
          spineShaderInjected: compiledSpine.vertexShader.includes('annolidDeformSpinePos'),
          liveSpineBendUpdated: compiledSpine.uniforms.uSpineBend.value === 0.42,
          liveSpineStretchUpdated: compiledSpine.uniforms.uAxialStretch.value === 0.14,
        }));
        """,
        include_simulation=False,
    )
    assert result["distinctKeys"] is True
    assert result["keyFurAo"] == "base|annolid:fur+ao"
    assert result["keySss"] == "base|annolid:sss"
    assert result["keyBodySpine"] == "base|annolid:fur+spine"
    assert result["restoredKey"] == "base"
    assert result["zeroAnisotropyPreserved"] is True
    assert result["zeroRadiusPreserved"] is True
    assert result["zeroSssPreserved"] is True
    assert result["flowDirAliasY"] == pytest.approx(-0.6, abs=1e-4)
    assert result["flowDirAliasZ"] == pytest.approx(-0.8, abs=1e-4)
    assert result["liveAoUpdated"] is True
    assert result["liveFurUpdated"] is True
    assert result["worldFlowInjected"] is True
    assert result["spineShaderInjected"] is True
    assert result["liveSpineBendUpdated"] is True
    assert result["liveSpineStretchUpdated"] is True


def test_threejs_runtime_and_two_mice_assets_have_no_external_cdn_imports():
    assets = REPO_ROOT / "annolid/gui/assets/threejs"
    offline_files = (
        "annolid_threejs_runtime.js",
        "annolid_threejs_css2d.js",
        "annolid_shaders.js",
        "two_mice.html",
        "fly_brain_mouse.html",
        "two_mice_scene.js",
        "two_mice_model.js",
        "two_mice_arenas.js",
        "two_mice_simulation.js",
        "two_mice_control.js",
        "two_mice_fly_brain.js",
        "swarm_visualizer.html",
    )
    for name in offline_files:
        text = (assets / name).read_text(encoding="utf-8")
        assert "esm.sh" not in text, f"{name} still references esm.sh"
        assert "https://" not in text, f"{name} still references external https URL"

    result = run_javascript(
        """
        import {
          THREE,
          OrbitControls,
          RoomEnvironment,
          GUI,
        } from './annolid/gui/assets/threejs/annolid_threejs_runtime.js';
        import {
          CSS2DRenderer,
          CSS2DObject,
        } from './annolid/gui/assets/threejs/annolid_threejs_css2d.js';

        console.log(JSON.stringify({
          revision: THREE.REVISION,
          hasOrbitControls: typeof OrbitControls === 'function',
          hasRoomEnvironment: typeof RoomEnvironment === 'function',
          hasGui: typeof GUI === 'function',
          hasCss2d: typeof CSS2DRenderer === 'function' && typeof CSS2DObject === 'function',
        }));
        """,
        include_simulation=False,
    )
    assert result == {
        "revision": "182",
        "hasOrbitControls": True,
        "hasRoomEnvironment": True,
        "hasGui": True,
        "hasCss2d": True,
    }


_CANVAS_DOCUMENT_STUB_JS = """
globalThis.document = {
  createElement() {
    return {
      width: 0,
      height: 0,
      getContext() {
        return new Proxy({}, {
          get(_t, key) {
            if (key === 'createRadialGradient' || key === 'createLinearGradient') {
              return () => ({ addColorStop() {} });
            }
            if (key === 'getImageData') {
              return (_x, _y, w, h) => ({ data: new Uint8ClampedArray(w * h * 4) });
            }
            return () => {};
          },
        });
      },
    };
  },
};
"""


def test_mouse_model_fur_options_and_in_place_anatomy_rebuild():
    result = run_javascript(
        """
        import { THREE } from './annolid/gui/assets/threejs/annolid_threejs_runtime.js';
        import { AnnolidShaders } from './annolid/gui/assets/threejs/annolid_shaders.js';
        import { createMouseModel } from './annolid/gui/assets/threejs/two_mice_model.js';
        import { createSeededRandom } from './annolid/gui/assets/threejs/two_mice_simulation.js';
        """
        + _CANVAS_DOCUMENT_STUB_JS
        + """
        const model = createMouseModel(THREE, new AnnolidShaders(THREE), createSeededRandom(77));
        const posAttrRef = model.bodyGeo.attributes.position;
        model.bodyGeo.computeBoundingBox();
        const initialSpanZ = model.bodyGeo.boundingBox.max.z - model.bodyGeo.boundingBox.min.z;
        model.PROP.bodyLength = 3.4;
        model.rebuildSharedAnatomyGeometries();
        const rebuiltSpanZ = model.bodyGeo.boundingBox.max.z - model.bodyGeo.boundingBox.min.z;
        const sameBufferRef = model.bodyGeo.attributes.position === posAttrRef;

        const defaultCoat = model.generateMultiLayerFur(model.deformBody, 0.1, '#ffffff', false);
        const explicitDefaultCoat = model.generateMultiLayerFur(
          model.deformBody,
          0.1,
          '#ffffff',
          false,
          { furGuardRatio: 0.5, furLength: 1.0, furClumpiness: 0.3 }
        );
        const underOnly = model.generateMultiLayerFur(
          model.deformBody,
          0.1,
          '#ffffff',
          false,
          { furGuardRatio: 0.0, furLength: 1.5, furClumpiness: 0.8 }
        );
        const defaultCounts = defaultCoat.children.map(layer => layer.count);
        const explicitCounts = explicitDefaultCoat.children.map(layer => layer.count);
        const underOnlyCounts = underOnly.children.map(layer => layer.count);
        const hairlessCoat = model.generateMultiLayerFur(
          model.deformBody,
          0.0,
          '#f0c4bc',
          false
        );
        const hairlessMat = model.createCoatMaterial('#f0c4bc', '#efc0b8', 0.0, {});
        const hasAlbinoEyeMap = Boolean(model.albinoEyeIrisMap?.isTexture);

        model.disposeSubject(defaultCoat);
        model.disposeSubject(explicitDefaultCoat);
        model.disposeSubject(underOnly);
        model.disposeSubject(hairlessCoat);
        hairlessMat.dispose();
        model.disposeSharedResources();

        console.log(JSON.stringify({
          initialSpanZ: Number(initialSpanZ.toFixed(3)),
          rebuiltSpanZ: Number(rebuiltSpanZ.toFixed(3)),
          sameBufferRef,
          defaultCounts,
          explicitCounts,
          underOnlyCounts,
          hairlessLayerCount: hairlessCoat.children.length,
          hairlessHasRoughnessMap: Boolean(hairlessMat.roughnessMap),
          hasAlbinoEyeMap,
        }));
        """,
        include_simulation=False,
    )
    assert result["initialSpanZ"] == pytest.approx(5.36, abs=0.02)
    assert result["rebuiltSpanZ"] == pytest.approx(6.80, abs=0.02)
    assert result["sameBufferRef"] is True
    assert result["defaultCounts"] == result["explicitCounts"] == [3800, 2200]
    assert result["underOnlyCounts"] == [7600]
    assert result["hairlessLayerCount"] == 0
    assert result["hairlessHasRoughnessMap"] is False
    assert result["hasAlbinoEyeMap"] is True

    scene_js = (
        REPO_ROOT / "annolid" / "gui" / "assets" / "threejs" / "two_mice_scene.js"
    ).read_text(encoding="utf-8")
    assert "params.furDensity ?? 0.85" in scene_js
    assert "params.furDensity || 0.85" not in scene_js


def test_mouse_model_dorsal_skull_and_anterior_neck_geometry():
    result = run_javascript(
        """
        import { THREE } from './annolid/gui/assets/threejs/annolid_threejs_runtime.js';
        import { AnnolidShaders } from './annolid/gui/assets/threejs/annolid_shaders.js';
        import { createMouseModel } from './annolid/gui/assets/threejs/two_mice_model.js';
        import { createSeededRandom } from './annolid/gui/assets/threejs/two_mice_simulation.js';
        """
        + _CANVAS_DOCUMENT_STUB_JS
        + """
        const model = createMouseModel(THREE, new AnnolidShaders(THREE), createSeededRandom(77));

        const isFiniteVec = (v) =>
          Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
        const symmetricDeform = (fn, x, y, z) => {
          const left = new THREE.Vector3(x, y, z);
          const right = new THREE.Vector3(-x, y, z);
          fn(left);
          fn(right);
          return (
            isFiniteVec(left)
            && isFiniteVec(right)
            && Math.abs(left.x + right.x) < 1e-9
            && Math.abs(left.y - right.y) < 1e-9
            && Math.abs(left.z - right.z) < 1e-9
          );
        };

        // Dorsal skull midline contour from occiput (zNorm=0.05) to muzzle tip (zNorm=0.98).
        const dorsal = [];
        for (let i = 0; i <= 40; i++) {
          const zNorm = 0.05 + (0.98 - 0.05) * (i / 40);
          const z = zNorm * 2 - 1;
          const y = Math.sqrt(Math.max(0, 1 - z * z));
          const v = new THREE.Vector3(0, y, z);
          model.deformHead(v);
          dorsal.push({ zNorm, x: v.x, y: v.y, z: v.z, finite: isFiniteVec(v) });
        }
        let dorsalMaxSecondDiff = 0;
        let dorsalZMonotonic = true;
        for (let i = 1; i < dorsal.length; i++) {
          if (dorsal[i].z <= dorsal[i - 1].z) dorsalZMonotonic = false;
        }
        for (let i = 1; i < dorsal.length - 1; i++) {
          const d2 = Math.abs(dorsal[i + 1].y - 2 * dorsal[i].y + dorsal[i - 1].y);
          dorsalMaxSecondDiff = Math.max(dorsalMaxSecondDiff, d2);
        }
        const dorsalPeak = dorsal.reduce((a, b) => (b.y > a.y ? b : a));
        const dorsalEnd = dorsal[dorsal.length - 1];

        // Lateral head taper along equator (cranio-rostral wedge).
        const headWidths = [0.30, 0.50, 0.70, 0.90, 0.96].map((zNorm) => {
          const z = zNorm * 2 - 1;
          const x = Math.sqrt(Math.max(0, 1 - z * z));
          const v = new THREE.Vector3(x, 0, z);
          model.deformHead(v);
          return v.x;
        });

        // Anterior torso-to-neck transition across dense zNorm samples (0.50..0.96).
        const anterior = [];
        for (let i = 0; i <= 30; i++) {
          const zNorm = 0.50 + (0.96 - 0.50) * (i / 30);
          const z = zNorm * 2 - 1;
          const r = Math.sqrt(Math.max(0, 1 - z * z));
          const lat = new THREE.Vector3(r, 0, z);
          const dor = new THREE.Vector3(0, r, z);
          const ven = new THREE.Vector3(0, -r, z);
          model.deformBody(lat);
          model.deformBody(dor);
          model.deformBody(ven);
          anterior.push({
            zNorm,
            x: lat.x,
            dorY: dor.y,
            venY: ven.y,
            height: dor.y - ven.y,
            z: lat.z,
            finite: isFiniteVec(lat) && isFiniteVec(dor) && isFiniteVec(ven),
          });
        }
        let bodyZMonotonic = true;
        let maxD2X = 0;
        let maxD2Dor = 0;
        let maxD2Ven = 0;
        for (let i = 1; i < anterior.length; i++) {
          if (anterior[i].z <= anterior[i - 1].z) bodyZMonotonic = false;
        }
        for (let i = 1; i < anterior.length - 1; i++) {
          maxD2X = Math.max(maxD2X, Math.abs(anterior[i + 1].x - 2 * anterior[i].x + anterior[i - 1].x));
          maxD2Dor = Math.max(maxD2Dor, Math.abs(anterior[i + 1].dorY - 2 * anterior[i].dorY + anterior[i - 1].dorY));
          maxD2Ven = Math.max(maxD2Ven, Math.abs(anterior[i + 1].venY - 2 * anterior[i].venY + anterior[i - 1].venY));
        }
        const sampleBodySection = (zNorm) => {
          const z = zNorm * 2 - 1;
          const r = Math.sqrt(Math.max(0, 1 - z * z));
          const lat = new THREE.Vector3(r, 0, z);
          const dor = new THREE.Vector3(0, r, z);
          const ven = new THREE.Vector3(0, -r, z);
          model.deformBody(lat);
          model.deformBody(dor);
          model.deformBody(ven);
          return { halfWidth: lat.x, height: dor.y - ven.y };
        };
        const bodySections = [0.50, 0.65, 0.75, 0.85, 0.92, 0.96].map(sampleBodySection);

        // Cervical mantle and core inset continuity.
        const neckProfiles = [0, 0.25, 0.5, 0.75, 1.0].map((t) => model.getCervicalMantleProfile(t));
        const coreInsetRatios = [-0.9, -0.5, 0.0, 0.5, 0.9].map((z) => {
          const rawT = (z + 1) * 0.5;
          const mantle = model.getCervicalMantleProfile(rawT * 0.96);
          const r = Math.sqrt(Math.max(0, 1 - z * z));
          const v = new THREE.Vector3(r, 0, z);
          model.deformCervicalCore(v);
          return v.x / mantle.radiusX;
        });

        const headSymmetric = [0.25, 0.60, 0.85].every((zNorm) => {
          const z = zNorm * 2 - 1;
          const r = Math.sqrt(Math.max(0, 1 - z * z));
          return symmetricDeform(model.deformHead, r * 0.7, r * 0.7, z);
        });
        const bodySymmetric = [0.30, 0.65, 0.88].every((zNorm) => {
          const z = zNorm * 2 - 1;
          const r = Math.sqrt(Math.max(0, 1 - z * z));
          return symmetricDeform(model.deformBody, r * 0.7, r * 0.7, z);
        });

        model.disposeSharedResources();

        console.log(JSON.stringify({
          dorsalFiniteAndMidline: dorsal.every((p) => p.finite && Math.abs(p.x) < 1e-9),
          headSymmetric,
          bodySymmetric,
          dorsalZMonotonic,
          dorsalMaxSecondDiff,
          dorsalPeakZNorm: dorsalPeak.zNorm,
          dorsalCrownToTipDrop: dorsalPeak.y - dorsalEnd.y,
          headWidths,
          anteriorFinite: anterior.every((p) => p.finite),
          bodyZMonotonic,
          maxD2X,
          maxD2Dor,
          maxD2Ven,
          bodySections,
          neckProfiles,
          coreInsetRatios,
        }));
        """,
        include_simulation=False,
    )

    assert result["dorsalFiniteAndMidline"] is True
    assert result["headSymmetric"] is True
    assert result["bodySymmetric"] is True
    assert result["dorsalZMonotonic"] is True
    assert result["dorsalMaxSecondDiff"] < 0.08
    assert 0.18 <= result["dorsalPeakZNorm"] <= 0.55
    assert result["dorsalCrownToTipDrop"] > 0.35

    head_widths = result["headWidths"]
    assert head_widths[1] > head_widths[2] > head_widths[3] > head_widths[4] > 0
    assert head_widths[3] < head_widths[0] * 0.5

    assert result["anteriorFinite"] is True
    assert result["bodyZMonotonic"] is True
    assert result["maxD2X"] < 0.05
    assert result["maxD2Dor"] < 0.05
    assert result["maxD2Ven"] < 0.05

    sections = result["bodySections"]
    half_widths = [s["halfWidth"] for s in sections]
    heights = [s["height"] for s in sections]
    assert all(a > b > 0 for a, b in zip(half_widths, half_widths[1:]))
    assert all(a > b > 0 for a, b in zip(heights, heights[1:]))
    assert half_widths[4] < half_widths[1] * 0.65
    assert heights[4] < heights[1] * 0.75

    neck_profiles = result["neckProfiles"]
    assert all(
        a["z"] < b["z"]
        and a["centerY"] < b["centerY"]
        and a["radiusX"] > b["radiusX"] > 0
        and a["radiusY"] > b["radiusY"] > 0
        for a, b in zip(neck_profiles, neck_profiles[1:])
    )

    core_ratios = result["coreInsetRatios"]
    assert all(0.0 < ratio < 1.0 for ratio in core_ratios)
    assert core_ratios[-1] < core_ratios[2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "frame_count": 1,
            "train_frames": 1,
            "fps": 5,
            "seed": 1,
            "width": 640,
            "height": 480,
        },
        {
            "frame_count": 6,
            "train_frames": 0,
            "fps": 5,
            "seed": 1,
            "width": 640,
            "height": 480,
        },
        {
            "frame_count": 6,
            "train_frames": 6,
            "fps": 5,
            "seed": 1,
            "width": 640,
            "height": 480,
        },
        {
            "frame_count": 6,
            "train_frames": 4,
            "fps": 0,
            "seed": 1,
            "width": 640,
            "height": 480,
        },
        {
            "frame_count": 6,
            "train_frames": 4,
            "fps": 5,
            "seed": 1,
            "width": 64,
            "height": 480,
        },
    ],
)
def test_generate_dataset_rejects_invalid_options_before_browser_launch(
    tmp_path: Path, kwargs: dict[str, int]
):
    from scripts.generate_two_mice_pose_dataset import generate_dataset

    with pytest.raises(ValueError):
        generate_dataset(tmp_path / "invalid_out", **kwargs)
    assert not (tmp_path / "invalid_out").exists()
