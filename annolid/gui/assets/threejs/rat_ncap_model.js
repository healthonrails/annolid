/**
 * Authentic 41-bone skinned 3D Rat v2 model loader & MuJoCo-faithful FK rig.
 *
 * Loads `rat_v2_skin.glb` (14,754 vertices, 19,608 faces, 41 bones, 55 hinge joints)
 * and the three Long-Evans coat textures (`rat_blackhooded.jpg`, `rat_cinnamon.jpg`,
 * `rat_gray.jpg`) derived from `rat-ncap/rat-ncap` (MIT License; Bhattasali et al. 2026).
 */

import {
  RAT_COAT_PROFILES,
  RAT_KEYPOINTS,
  RAT_LIMB_NAMES,
  computeBumpyTerrainHeight,
} from './rat_ncap_circuit.js';

export const RAT_V2_GLB_FILE = 'rat_v2_skin.glb';

export const RAT_V2_COAT_TEXTURES = Object.freeze({
  black_hooded: 'rat_blackhooded.jpg',
  cinnamon: 'rat_cinnamon.jpg',
  gray: 'rat_gray.jpg',
});

const COMPONENT_TYPED_ARRAY = Object.freeze({
  5120: Int8Array,
  5121: Uint8Array,
  5122: Int16Array,
  5123: Uint16Array,
  5125: Uint32Array,
  5126: Float32Array,
});

const TYPE_NUM_COMPONENTS = Object.freeze({
  SCALAR: 1,
  VEC2: 2,
  VEC3: 3,
  VEC4: 4,
  MAT4: 16,
});

/**
 * Pure zero-dependency glTF 2.0 Binary (.glb) parser for `rat_v2_skin.glb`.
 * Works identically in browser ES modules and Node.js unit tests.
 */
export function parseRatGlbBuffer(arrayBuffer) {
  const bytes = arrayBuffer instanceof Uint8Array
    ? arrayBuffer
    : new Uint8Array(arrayBuffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  if (bytes.byteLength < 20) {
    throw new Error('GLB buffer too short.');
  }
  const magic = view.getUint32(0, true);
  if (magic !== 0x46546C67) {
    throw new Error(`Invalid GLB magic: 0x${magic.toString(16)}`);
  }
  const version = view.getUint32(4, true);
  if (version !== 2) {
    throw new Error(`Unsupported glTF version: ${version}`);
  }

  let offset = 12;
  let gltf = null;
  let binBytes = null;
  const decoder = new TextDecoder('utf-8');

  while (offset + 8 <= bytes.byteLength) {
    const chunkLen = view.getUint32(offset, true);
    const chunkType = view.getUint32(offset + 4, true);
    offset += 8;
    const chunkEnd = offset + chunkLen;
    if (chunkType === 0x4E4F534A) {
      gltf = JSON.parse(decoder.decode(bytes.subarray(offset, chunkEnd)));
    } else if (chunkType === 0x004E4942) {
      binBytes = bytes.subarray(offset, chunkEnd);
    }
    offset = chunkEnd;
  }

  if (!gltf || !binBytes) {
    throw new Error('GLB missing JSON or BIN chunk.');
  }

  const readAccessor = accessorIndex => {
    const acc = gltf.accessors[accessorIndex];
    const bv = gltf.bufferViews[acc.bufferView];
    const Ctor = COMPONENT_TYPED_ARRAY[acc.componentType];
    const comps = TYPE_NUM_COMPONENTS[acc.type];
    if (!Ctor || !comps) {
      throw new Error(`Unsupported accessor componentType=${acc.componentType} type=${acc.type}`);
    }
    const byteOffset = (bv.byteOffset || 0) + (acc.byteOffset || 0);
    const totalCount = acc.count * comps;
    const slice = binBytes.slice(byteOffset, byteOffset + totalCount * Ctor.BYTES_PER_ELEMENT);
    return new Ctor(slice.buffer, slice.byteOffset, totalCount);
  };

  const prim = gltf.meshes[0].primitives[0];
  const attrs = prim.attributes;
  const skin = gltf.skins[0];
  const rig = gltf.extras?.ratV2Rig;
  if (!rig || !Array.isArray(rig.bones) || !Array.isArray(rig.joints)) {
    throw new Error('GLB missing extras.ratV2Rig metadata.');
  }

  return {
    gltf,
    rig,
    indices: readAccessor(prim.indices),
    positions: readAccessor(attrs.POSITION),
    normals: readAccessor(attrs.NORMAL),
    uvs: readAccessor(attrs.TEXCOORD_0),
    skinIndices: readAccessor(attrs.JOINTS_0),
    skinWeights: readAccessor(attrs.WEIGHTS_0),
    partIds: attrs._PART_ID != null ? readAccessor(attrs._PART_ID) : null,
    inverseBindMatrices: readAccessor(skin.inverseBindMatrices),
  };
}

function clamp(v, lo, hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

function createFurMicroNormalMap(THREE, size = 256) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#8080ff';
  ctx.fillRect(0, 0, size, size);
  let seed = 20261125;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < 4200; i++) {
    const x = rand() * size;
    const y = rand() * size;
    const len = 3 + rand() * 7;
    const nx = 114 + Math.floor(rand() * 28);
    const ny = 118 + Math.floor(rand() * 20);
    ctx.strokeStyle = `rgba(${nx},${ny},255,0.45)`;
    ctx.lineWidth = 0.8 + rand() * 0.9;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + (rand() - 0.5) * 1.4, y + len);
    ctx.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(3.5, 3.5);
  return tex;
}

function createSoftShadowTexture(THREE, size = 256) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(
    size * 0.5, size * 0.5, size * 0.06,
    size * 0.5, size * 0.5, size * 0.50
  );
  g.addColorStop(0.0, 'rgba(18, 20, 26, 0.34)');
  g.addColorStop(0.55, 'rgba(18, 20, 26, 0.14)');
  g.addColorStop(1.0, 'rgba(18, 20, 26, 0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  return tex;
}

/**
 * Full 41-bone skinned Long-Evans Rat v2 rig with exact MuJoCo hinge-joint FK.
 */
export class RatV2SkinnedModel {
  constructor(THREE, annolidShaders, parsedGlb, texturesByCoat, { coatId = 'black_hooded', worldScale = 10.0 } = {}) {
    this.THREE = THREE;
    this.worldScale = worldScale;
    this.rig = parsedGlb.rig;
    this.texturesByCoat = texturesByCoat;
    this.coatId = RAT_V2_COAT_TEXTURES[coatId] ? coatId : 'black_hooded';

    this.root = new THREE.Group();
    this.root.name = 'rat_v2_skinned_root';

    // Scale all vertex positions, bone translations, joint pivots, keypoints, and
    // inverseBindMatrix translations into Three.js world units ONCE so SkinnedMesh
    // never suffers scale mismatch between bindMatrix and boneInverses.
    const s = worldScale;
    const scaledPositions = new Float32Array(parsedGlb.positions.length);
    for (let i = 0; i < parsedGlb.positions.length; i++) {
      scaledPositions[i] = parsedGlb.positions[i] * s;
    }

    const geo = new THREE.BufferGeometry();
    geo.setIndex(new THREE.BufferAttribute(parsedGlb.indices, 1));
    geo.setAttribute('position', new THREE.BufferAttribute(scaledPositions, 3));
    geo.setAttribute('normal', new THREE.BufferAttribute(parsedGlb.normals, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(parsedGlb.uvs, 2));
    geo.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(Uint16Array.from(parsedGlb.skinIndices), 4));
    geo.setAttribute('skinWeight', new THREE.BufferAttribute(parsedGlb.skinWeights, 4));
    if (parsedGlb.partIds) {
      geo.setAttribute('aPartId', new THREE.BufferAttribute(Float32Array.from(parsedGlb.partIds), 2));
    }

    // Build 41-bone Three.js Skeleton with exact scaled inverseBindMatrices
    this.bones = [];
    this.boneByName = {};
    this.boneRestPos = [];
    this.boneRestQuat = [];
    const boneInverses = [];
    const ibm = parsedGlb.inverseBindMatrices;

    for (let i = 0; i < this.rig.bones.length; i++) {
      const bm = this.rig.bones[i];
      const bone = new THREE.Bone();
      bone.name = bm.name;
      const restPos = new THREE.Vector3(bm.pos[0] * s, bm.pos[1] * s, bm.pos[2] * s);
      const restQuat = new THREE.Quaternion(bm.quat[0], bm.quat[1], bm.quat[2], bm.quat[3]).normalize();
      bone.position.copy(restPos);
      bone.quaternion.copy(restQuat);
      this.bones.push(bone);
      this.boneByName[bm.name] = bone;
      this.boneRestPos.push(restPos);
      this.boneRestQuat.push(restQuat);

      const invMat = new THREE.Matrix4().fromArray(ibm, i * 16);
      invMat.elements[12] *= s;
      invMat.elements[13] *= s;
      invMat.elements[14] *= s;
      boneInverses.push(invMat);
    }

    // Attach bone tree hierarchy directly under this.root (rigid world pose only, scale = 1)
    for (let i = 0; i < this.rig.bones.length; i++) {
      const parentIdx = this.rig.bones[i].parent;
      if (parentIdx >= 0) {
        this.bones[parentIdx].add(this.bones[i]);
      } else {
        this.root.add(this.bones[i]);
      }
    }

    // Index all 55 hinge joints per bone (scaled pivots) for exact MuJoCo FK
    this.jointsByBone = new Array(this.bones.length).fill(null).map(() => []);
    this.jointByName = {};
    this.jointAngles = {};
    for (const j of this.rig.joints) {
      const entry = {
        name: j.name,
        bone: j.bone,
        pos: new THREE.Vector3(j.pos[0] * s, j.pos[1] * s, j.pos[2] * s),
        axis: new THREE.Vector3(j.axis[0], j.axis[1], j.axis[2]).normalize(),
        min: j.range[0],
        max: j.range[1],
        init: j.init,
      };
      this.jointsByBone[j.bone].push(entry);
      this.jointByName[j.name] = entry;
      this.jointAngles[j.name] = j.init;
    }

    this.rootPosInit = new THREE.Vector3(
      this.rig.rootPosInit[0] * s,
      this.rig.rootPosInit[1] * s,
      this.rig.rootPosInit[2] * s
    );
    this.rootQuatInit = new THREE.Quaternion(...this.rig.rootQuatInit).normalize();

    // Studio-grade PBR velvety fur coat material (fully GPU-skinning-aware in Three.js r182)
    const initialTex = this.texturesByCoat[this.coatId] || Object.values(this.texturesByCoat)[0] || null;
    const coatProfile = RAT_COAT_PROFILES[this.coatId] || RAT_COAT_PROFILES.black_hooded;
    const furNormalMap = createFurMicroNormalMap(THREE);
    const sheenHex = this.coatId === 'black_hooded' ? 0x3b3d44 : coatProfile.bodyColor;
    this.skinMaterial = new THREE.MeshPhysicalMaterial({
      map: initialTex,
      normalMap: furNormalMap,
      normalScale: new THREE.Vector2(0.18, 0.18),
      roughness: 0.86,
      metalness: 0.0,
      sheen: 0.16,
      sheenRoughness: 0.88,
      sheenColor: new THREE.Color(sheenHex),
      envMapIntensity: 0.20,
    });
    // Skinning-aware ground Contact AO injected AFTER #include <skinning_vertex>
    this.skinMaterial.onBeforeCompile = shader => {
      shader.vertexShader = shader.vertexShader
        .replace(
          '#include <common>',
          '#include <common>\nvarying float vRatSkinnedWorldY;'
        )
        .replace(
          '#include <project_vertex>',
          `vRatSkinnedWorldY = (modelMatrix * vec4(transformed, 1.0)).y;\n#include <project_vertex>`
        );
      shader.fragmentShader = shader.fragmentShader
        .replace(
          '#include <common>',
          '#include <common>\nvarying float vRatSkinnedWorldY;'
        )
        .replace(
          '#include <dithering_fragment>',
          `{
            float ratGroundAo = smoothstep(0.0, 0.48, max(0.0, vRatSkinnedWorldY));
            gl_FragColor.rgb *= mix(0.84, 1.0, pow(ratGroundAo, 0.65));
          }
          #include <dithering_fragment>`
        );
    };

    this.skinnedMesh = new THREE.SkinnedMesh(geo, this.skinMaterial);
    this.skinnedMesh.name = 'rat_v2_skinned_mesh';
    this.skinnedMesh.castShadow = true;
    this.skinnedMesh.receiveShadow = true;
    this.skinnedMesh.frustumCulled = false;

    // Use DetachedBindMode with identity bindMatrix and keep skinnedMesh matrixWorld
    // at identity: bone.matrixWorld already contains this.root's full world transform,
    // so skinned vertices land directly in exact world coordinates with zero double-transform.
    this.skinnedMesh.bindMode = THREE.DetachedBindMode;
    this.skinnedMesh.matrixAutoUpdate = false;
    this.skinnedMesh.matrix.identity();
    this.skinnedMesh.matrixWorld.identity();
    this.skeleton = new THREE.Skeleton(this.bones, boneInverses);
    this.skinnedMesh.bind(this.skeleton, new THREE.Matrix4());
    // sceneMeshGroup holds the world-space SkinnedMesh + root rig together cleanly
    this.sceneGroup = new THREE.Group();
    this.sceneGroup.name = 'rat_v2_scene_group';
    this.sceneGroup.add(this.root);
    this.sceneGroup.add(this.skinnedMesh);

    // GPU-skinned segmentation material for Sim-to-Real captureSegmentationMasks()
    this.segmentationMaterial = new THREE.ShaderMaterial({
      vertexShader: `
        attribute vec2 aPartId;
        varying vec2 vPartId;
        #include <common>
        #include <skinning_pars_vertex>
        void main() {
          vPartId = aPartId;
          #include <skinbase_vertex>
          #include <begin_vertex>
          #include <skinning_vertex>
          gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vPartId;
        void main() {
          gl_FragColor = vec4(1.0 / 255.0, vPartId.x / 255.0, vPartId.y / 255.0, 1.0);
        }
      `,
      transparent: false,
      depthTest: true,
      depthWrite: true,
      toneMapped: false,
    });

    // Attach exact 16 anatomical keypoint tracking nodes to their parent bones
    this.trackingNodes = {};
    this.keypointMarkers = new THREE.Group();
    this.keypointMarkers.visible = false;
    this.root.add(this.keypointMarkers);

    for (const kpName of RAT_KEYPOINTS) {
      const spec = this.rig.keypoints[kpName];
      if (!spec) continue;
      const node = new THREE.Object3D();
      node.name = `kp_${kpName}`;
      node.position.set(spec.pos[0] * s, spec.pos[1] * s, spec.pos[2] * s);
      this.bones[spec.bone].add(node);
      this.trackingNodes[kpName] = node;

      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.036, 10, 10),
        new THREE.MeshBasicMaterial({ color: kpName.includes('paw') ? 0x38bdf8 : 0xfacc15 })
      );
      dot.userData.kpName = kpName;
      this.keypointMarkers.add(dot);
    }

    // Optional biomechanical muscle-effort spheres on major actuated limb & lumbar joints
    this.muscleOverlay = new THREE.Group();
    this.muscleOverlay.visible = false;
    this.muscleMarkers = {};
    const effortBones = [
      ['FL', 'upper_arm_L', 0.011 * s],
      ['FL_elbow', 'lower_arm_L', 0.008 * s],
      ['FR', 'upper_arm_R', 0.011 * s],
      ['FR_elbow', 'lower_arm_R', 0.008 * s],
      ['HL', 'upper_leg_L', 0.013 * s],
      ['HL_knee', 'lower_leg_L', 0.009 * s],
      ['HR', 'upper_leg_R', 0.013 * s],
      ['HR_knee', 'lower_leg_R', 0.009 * s],
      ['lumbar', 'vertebra_3', 0.014 * s],
    ];
    for (const [key, bName, radius] of effortBones) {
      const bone = this.boneByName[bName];
      if (!bone) continue;
      const mat = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.72,
        depthTest: false,
      });
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(radius, 12, 10), mat);
      sphere.renderOrder = 5;
      bone.add(sphere);
      sphere.visible = false;
      this.muscleMarkers[key] = sphere;
    }

    // Soft radial ground contact shadow beneath the rat torso
    const shadowGeo = new THREE.PlaneGeometry(2.75, 1.35);
    shadowGeo.rotateX(-Math.PI / 2);
    this.contactShadow = new THREE.Mesh(
      shadowGeo,
      new THREE.MeshBasicMaterial({
        map: createSoftShadowTexture(THREE),
        transparent: true,
        depthWrite: false,
      })
    );
    this.contactShadow.position.set(-0.18, 0.008, 0);
    this.root.add(this.contactShadow);

    // Scratch math objects to avoid per-frame allocations
    this._tmpMat = new THREE.Matrix4();
    this._tmpJointMat = new THREE.Matrix4();
    this._tmpRotMat = new THREE.Matrix4();
    this._tmpTransA = new THREE.Matrix4();
    this._tmpTransB = new THREE.Matrix4();
    this._tmpScale = new THREE.Vector3(1, 1, 1);
    this._tmpPos = new THREE.Vector3();
    this._tmpQuat = new THREE.Quaternion();
    this._tmpEuler = new THREE.Euler();
    this._pawWorld = new THREE.Vector3();

    // Initialize into exact standing pose
    this.applyJointAnglesToSkeleton(this.jointAngles, 0, 0, 0);
  }

  setCoat(coatId) {
    const key = RAT_V2_COAT_TEXTURES[coatId] ? coatId : 'black_hooded';
    this.coatId = key;
    const tex = this.texturesByCoat[key];
    const profile = RAT_COAT_PROFILES[key] || RAT_COAT_PROFILES.black_hooded;
    if (tex) {
      this.skinMaterial.map = tex;
      const sheenHex = key === 'black_hooded' ? 0x3b3d44 : profile.bodyColor;
      this.skinMaterial.sheenColor?.set(sheenHex);
      this.skinMaterial.needsUpdate = true;
    }
  }

  /**
   * Exact MuJoCo forward kinematics for all 41 bones and 55 hinge joints.
   */
  applyJointAnglesToSkeleton(anglesByName, rootPitchDelta = 0, rootRollDelta = 0, rootBounceM = 0) {
    const THREE = this.THREE;
    for (let i = 0; i < this.bones.length; i++) {
      const bone = this.bones[i];
      if (i === 0) {
        // Root torso bone: standing init pos/quat + dynamic gait pitch/roll/bounce
        this._tmpEuler.set(rootRollDelta, 0, rootPitchDelta, 'XYZ');
        this._tmpQuat.setFromEuler(this._tmpEuler).multiply(this.rootQuatInit);
        this._tmpPos.set(
          this.rootPosInit.x,
          this.rootPosInit.y + rootBounceM * this.worldScale,
          this.rootPosInit.z
        );
        this._tmpMat.compose(this._tmpPos, this._tmpQuat, this._tmpScale);
      } else {
        this._tmpMat.compose(this.boneRestPos[i], this.boneRestQuat[i], this._tmpScale);
      }

      const joints = this.jointsByBone[i];
      for (let k = 0; k < joints.length; k++) {
        const j = joints[k];
        const rawAng = anglesByName[j.name] != null ? anglesByName[j.name] : j.init;
        const ang = clamp(rawAng, j.min, j.max);
        this._tmpTransA.makeTranslation(j.pos.x, j.pos.y, j.pos.z);
        this._tmpRotMat.makeRotationAxis(j.axis, ang);
        this._tmpTransB.makeTranslation(-j.pos.x, -j.pos.y, -j.pos.z);
        this._tmpJointMat.copy(this._tmpTransA).multiply(this._tmpRotMat).multiply(this._tmpTransB);
        this._tmpMat.multiply(this._tmpJointMat);
      }

      this._tmpMat.decompose(bone.position, bone.quaternion, this._tmpScale);
    }
  }

  /**
   * Maps RatNCAPController telemetry onto all 55 Rat v2 anatomical joints.
   */
  applyControllerState(state, { arenaId = 'treadmill', bumpCm = 0, showMuscleEffort = false, showKeypoints = false } = {}) {
    const angles = this.jointAngles;
    // Reset to exact standing pose before applying gait deltas
    for (const j of this.rig.joints) {
      angles[j.name] = j.init;
    }

    const cmd = state.command || 0;
    const amp = cmd < 0.03 ? 0 : (0.30 + 0.70 * cmd);
    const t = state.timeSec || 0;

    // 1. Lumbar spine flexion/extension, lateral bend, and twist (exact rat.xml tendon ratios)
    const lumExt = (state.lumbarExtend - 0.1) * 0.85 * amp;
    const lumBend = (state.lumbarBend || 0) * 0.90 * amp;
    const lumTwist = (state.lumbarTwist || 0) * 0.70 * amp;
    angles.vertebra_1_extend = this.jointByName.vertebra_1_extend.init + 0.605 * lumExt;
    angles.vertebra_4_extend = this.jointByName.vertebra_4_extend.init + 0.395 * lumExt;
    angles.vertebra_2_bend = this.jointByName.vertebra_2_bend.init + 0.658 * lumBend;
    angles.vertebra_5_bend = this.jointByName.vertebra_5_bend.init + 0.342 * lumBend;
    angles.vertebra_3_twist = this.jointByName.vertebra_3_twist.init + 0.571 * lumTwist;
    angles.vertebra_6_twist = this.jointByName.vertebra_6_twist.init + 0.429 * lumTwist;

    // 2. Forelimbs (FL, FR) and Hindlimbs (HL, HR) from controller phase & swing/stance synergies
    const baseFreq = Math.max(state.stepFrequencyHz || 0, 0.05);
    for (const limb of RAT_LIMB_NAMES) {
      const side = limb.endsWith('L') ? 'L' : 'R';
      const swing = state.limbSwing[limb] || 0;
      const stance = state.limbStance[limb] || 0;
      const contact = state.footContacts[limb] ?? 1;
      const phase = ((state.phaseOffsetsRelFR[limb] || 0) + (t * baseFreq)) % 1.0;
      const stride = Math.sin(2.0 * Math.PI * phase);
      const lift = Math.max(0, Math.sin(2.0 * Math.PI * phase + 0.45)) * (1.0 - contact * 0.65);

      if (limb === 'FL' || limb === 'FR') {
        // Forelimb: scapula glide + shoulder swing + elbow lift + wrist/finger plantar compliance
        angles[`scapula_${side}_abduct`] = this.jointByName[`scapula_${side}_abduct`].init
          + (0.06 * swing - 0.03 * stance) * amp;
        angles[`scapula_${side}_extend`] = this.jointByName[`scapula_${side}_extend`].init
          + 0.022 * stride * amp;
        angles[`shoulder_${side}_supinate`] = this.jointByName[`shoulder_${side}_supinate`].init
          + (0.10 * lift - 0.05 * contact) * amp;
        angles[`shoulder_${side}_extend`] = this.jointByName[`shoulder_${side}_extend`].init
          + (-0.42 * swing + 0.34 * stance - 0.20 * stride) * amp;
        angles[`elbow_${side}`] = this.jointByName[`elbow_${side}`].init
          + (-0.48 * swing + 0.22 * contact - 0.14 * lift) * amp;
        angles[`wrist_${side}`] = this.jointByName[`wrist_${side}`].init
          + (-0.34 * lift + 0.16 * contact) * amp;
        angles[`finger_${side}`] = this.jointByName[`finger_${side}`].init
          + (-0.28 * lift + 0.22 * contact) * amp;
      } else {
        // Hindlimb (verified world-space Jacobian signs):
        // hip_extend(-): protract fwd, knee(-): flex/lift paw up, ankle(-): dorsiflex clearance
        angles[`hip_${side}_supinate`] = this.jointByName[`hip_${side}_supinate`].init
          + (0.07 * lift - 0.04 * stance) * amp;
        angles[`hip_${side}_abduct`] = this.jointByName[`hip_${side}_abduct`].init
          + (0.08 * lift - 0.04 * contact) * amp;
        angles[`hip_${side}_extend`] = this.jointByName[`hip_${side}_extend`].init
          + (-0.42 * swing + 0.34 * stance - 0.18 * stride) * amp;
        angles[`knee_${side}`] = this.jointByName[`knee_${side}`].init
          + (-0.40 * lift + 0.18 * stance + 0.14 * stride) * amp;
        angles[`ankle_${side}`] = this.jointByName[`ankle_${side}`].init
          + (-0.17 * lift + 0.16 * stance) * amp;
        angles[`toe_${side}`] = this.jointByName[`toe_${side}`].init
          + (-0.20 * lift + 0.28 * stance) * amp;
      }
    }

    // 3. Cervical neck & skull counter-stabilization + subtle sniffing micro-motion
    const headPitchStab = -state.pose.pitch * 0.45 + 0.03 * Math.sin(t * 7.5);
    angles.vertebra_cervical_5_extend = this.jointByName.vertebra_cervical_5_extend.init + 0.34 * headPitchStab;
    angles.vertebra_cervical_2_extend = this.jointByName.vertebra_cervical_2_extend.init + 0.33 * headPitchStab;
    angles.vertebra_atlant_extend = this.jointByName.vertebra_atlant_extend.init + 0.33 * headPitchStab;
    angles.vertebra_cervical_4_bend = this.jointByName.vertebra_cervical_4_bend.init - 0.25 * lumBend;
    angles.vertebra_cervical_1_bend = this.jointByName.vertebra_cervical_1_bend.init - 0.25 * lumBend;
    angles.atlas = this.jointByName.atlas.init + 0.04 * Math.sin(t * 9.0);
    angles.mandible = this.jointByName.mandible.init + 0.03 * Math.sin(t * 11.0);

    // 4. 6-segment traveling tail wave
    const tailSegs = [
      'vertebrae_C1C3', 'vertebrae_C4C9', 'vertebrae_C10C15',
      'vertebrae_C16C20', 'vertebrae_C21C25', 'vertebrae_C26C30',
    ];
    for (let i = 0; i < tailSegs.length; i++) {
      const seg = tailSegs[i];
      const lag = i * 0.55;
      const sway = (0.04 + 0.035 * i) * Math.sin(t * (3.8 + 2.2 * cmd) - lag) * (0.25 + 0.75 * amp);
      const lift = 0.025 * Math.cos(t * (4.0 + 2.0 * cmd) - lag) * amp;
      angles[`${seg}_bend`] = this.jointByName[`${seg}_bend`].init + sway - 0.14 * lumBend;
      angles[`${seg}_extend`] = this.jointByName[`${seg}_extend`].init + lift - 0.025 * cmd;
    }

    // Apply exact 41-bone FK in physical meters
    const pitchDelta = state.pose.pitch * 0.65;
    const rollDelta = state.pose.roll * 0.55;
    const bounceM = state.pose.bounce * 0.75;
    this.applyJointAnglesToSkeleton(angles, pitchDelta, rollDelta, bounceM);

    // Position root in arena coordinates + snap lowest paw gently onto ground / bumpy heightfield
    const isTreadmill = arenaId === 'treadmill';
    const wx = isTreadmill ? 0.0 : clamp(state.pose.x * this.worldScale, -5.0, 5.0);
    const wz = isTreadmill ? 0.0 : clamp(state.pose.z * this.worldScale, -5.0, 5.0);
    const groundY = computeBumpyTerrainHeight(
      isTreadmill ? state.pose.x : wx / this.worldScale,
      isTreadmill ? state.pose.z : wz / this.worldScale,
      bumpCm * 0.01
    ) * this.worldScale;

    this.root.position.set(wx, groundY, wz);
    this.root.rotation.y = isTreadmill ? 0.0 : -state.pose.yaw;
    this.root.updateMatrixWorld(true);

    // Ground-contact correction: evaluate local heightfield under each paw so paws
    // never penetrate flat floors or bumpy terrain ridges.
    const bumpM = Math.max(0, bumpCm) * 0.01;
    let maxRequiredLift = -Infinity;
    let minGap = Infinity;
    for (const kp of ['left_forepaw', 'right_forepaw', 'left_hindpaw', 'right_hindpaw']) {
      const node = this.trackingNodes[kp];
      if (!node) continue;
      node.getWorldPosition(this._pawWorld);
      const localTerrainY = bumpM > 0.0005
        ? computeBumpyTerrainHeight(
            this._pawWorld.x / this.worldScale,
            this._pawWorld.z / this.worldScale,
            bumpM
          ) * this.worldScale
        : groundY;
      const desiredPawY = localTerrainY + 0.022;
      const gap = this._pawWorld.y - desiredPawY;
      if (gap < minGap) minGap = gap;
      if (-gap > maxRequiredLift) maxRequiredLift = -gap;
    }
    if (Number.isFinite(minGap)) {
      if (maxRequiredLift > 0) {
        this.root.position.y += maxRequiredLift;
      } else if (state.gait !== 'bound') {
        this.root.position.y -= minGap * 0.65;
      }
    }
    if (this.contactShadow) {
      // Flat contact shadow works cleanly on planar floors; on curved bumpy heightfields
      // rely on the 2048px soft directional shadow map so no quad clips into slopes.
      this.contactShadow.visible = bumpM <= 0.002;
      this.contactShadow.position.y = Math.max(0.008, groundY - this.root.position.y + 0.008);
    }

    // Update optional biomechanical muscle-effort spheres
    for (const [key, sphere] of Object.entries(this.muscleMarkers)) {
      sphere.visible = Boolean(showMuscleEffort);
      if (!showMuscleEffort) continue;
      if (key === 'lumbar') {
        const eff = clamp(Math.abs(state.lumbarExtend) + Math.abs(state.lumbarBend), 0.1, 1.0);
        sphere.material.color.setHex(state.lumbarExtend >= 0 ? 0xf59e0b : 0xa78bfa);
        sphere.scale.setScalar(0.75 + 0.85 * eff);
      } else {
        const limbId = key.slice(0, 2);
        const sw = state.limbSwing[limbId] || 0;
        const st = state.limbStance[limbId] || 0;
        sphere.material.color.setHex(sw > st ? 0x22d3ee : 0xf97316);
        sphere.scale.setScalar(0.65 + 0.95 * Math.max(sw, st));
      }
    }

    // Sync optional 3D keypoint markers
    this.keypointMarkers.visible = Boolean(showKeypoints);
    if (showKeypoints) {
      this.root.updateMatrixWorld(true);
      const wp = new this.THREE.Vector3();
      for (const dot of this.keypointMarkers.children) {
        const node = this.trackingNodes[dot.userData.kpName];
        if (node) {
          node.getWorldPosition(wp);
          this.root.worldToLocal(wp);
          dot.position.copy(wp);
        }
      }
    }
  }
}

function loadTextureAsync(THREE, url, maxAnisotropy = 8) {
  return new Promise((resolve, reject) => {
    const loader = new THREE.TextureLoader();
    loader.load(
      url,
      tex => {
        tex.flipY = false; // Matches glTF 2.0 UV convention verified on rat_v2_skin.glb
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = maxAnisotropy;
        tex.wrapS = THREE.ClampToEdgeWrapping;
        tex.wrapT = THREE.ClampToEdgeWrapping;
        resolve(tex);
      },
      undefined,
      err => reject(err)
    );
  });
}

/**
 * Fetch and instantiate the authentic 41-bone skinned Long-Evans Rat v2 model.
 */
export async function loadRatV2SkinnedModel(
  THREE,
  annolidShaders,
  { coatId = 'black_hooded', worldScale = 10.0, maxAnisotropy = 8 } = {}
) {
  const glbUrl = new URL(`./${RAT_V2_GLB_FILE}`, import.meta.url).href;
  const resp = await fetch(glbUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${RAT_V2_GLB_FILE}: HTTP ${resp.status}`);
  }
  const buf = await resp.arrayBuffer();
  const parsed = parseRatGlbBuffer(buf);

  const texturesByCoat = {};
  await Promise.all(
    Object.entries(RAT_V2_COAT_TEXTURES).map(async ([id, filename]) => {
      const texUrl = new URL(`./${filename}`, import.meta.url).href;
      try {
        texturesByCoat[id] = await loadTextureAsync(THREE, texUrl, maxAnisotropy);
      } catch (err) {
        console.warn(`Could not load rat coat texture ${filename}:`, err);
      }
    })
  );

  return new RatV2SkinnedModel(THREE, annolidShaders, parsed, texturesByCoat, {
    coatId,
    worldScale,
  });
}
