/** Procedural mouse anatomy, coat, and materials. No simulation or UI state. */

/** Differentiate the actual deformed surface, including its nonuniform scale. */
export function createDeformedSurfaceSampler(THREE, deformFn) {
  const axis = new THREE.Vector3();
  const tangent = new THREE.Vector3();
  const bitangent = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const epsilon = 0.0002;
  return (source, position, normal) => {
    axis.set(0, Math.abs(source.y) < 0.9 ? 1 : 0, Math.abs(source.y) < 0.9 ? 0 : 1);
    tangent.crossVectors(source, axis).normalize();
    bitangent.crossVectors(source, tangent).normalize();
    a.copy(source).addScaledVector(tangent, epsilon).normalize();
    b.copy(source).addScaledVector(bitangent, epsilon).normalize();
    position.copy(source);
    deformFn(position);
    deformFn(a);
    deformFn(b);
    normal.crossVectors(a.sub(position), b.sub(position)).normalize();
    if (normal.lengthSq() < 0.5) normal.copy(source);
  };
}

export function createMouseModel(THREE, annolidShaders, random) {
  function defaultProportions() {
    return {
      bodyLength: 2.68, bodyGirthX: 1.13, bodyGirthY: 1.14,
      ribcageDepth: 1.04, waistTaper: 0.88, haunchWidth: 1.02, shoulderWidth: 0.88,
      neckLength: 1.36, neckGirth: 0.70,
      headSize: 0.96, snoutLength: 1.04, snoutWidth: 0.90, cheekFullness: 0.38,
      earSize: 0.80, earSetBack: -0.38, earYaw: 0.68, earThickness: 0.08,
      eyeSize: 0.108, eyeSpacing: 0.415, eyeProtrusion: 0.88, eyeYaw: 0.44,
      foreLimbLength: 0.94, hindLimbLength: 0.80,
      forePawSize: 1.10, hindPawSize: 1.14,
      tailLength: 1.0, tailThickness: 1.0, tailTaper: 0.88, tailSag: 0.018,
    };
  }
  const PROP = defaultProportions();
  const FORELIMB_NEUTRAL = Object.freeze({
    shoulderX: 0.50,
    shoulderZ: 0.0,
    elbowX: -0.94,
    wristX: 0.30,
    pawX: 0.14,
  });

  // ═══════════════════════════════════════════════════════════════
  // 3. PROCEDURAL CANVAS TEXTURES (micro-normal + roughness)
  // ═══════════════════════════════════════════════════════════════
  function createPadNormalMap(size = 128) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    // Micro bumps
    for (let i = 0; i < 800; i++) {
      const x = random() * size, y = random() * size;
      const r = 1 + random() * 2;
      const v = 120 + Math.floor(random() * 16);
      ctx.fillStyle = `rgb(${v},${v},255)`;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }
  function createLabFloorAlbedoMap(size = 1024) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    // Seamlessly tileable warm-putty anti-glare behavioral arena PVC so both
    // snow-white BALB/c and jet-black C57BL/6 coats separate cleanly with zero grid seams.
    ctx.fillStyle = '#bab7af';
    ctx.fillRect(0, 0, size, size);

    const drawWrapped = (x, y, margin, drawFn) => {
      const oxList = [0];
      const oyList = [0];
      if (x < margin) oxList.push(size);
      else if (x > size - margin) oxList.push(-size);
      if (y < margin) oyList.push(size);
      else if (y > size - margin) oyList.push(-size);
      for (const ox of oxList) {
        for (const oy of oyList) {
          drawFn(x + ox, y + oy);
        }
      }
    };

    // Soft large-scale mottled polymer tonal clouds (wrapped seamlessly).
    for (let i = 0; i < 36; i++) {
      const x = random() * size;
      const y = random() * size;
      const r = size * (0.06 + random() * 0.12);
      const warm = random() > 0.5;
      drawWrapped(x, y, r, (px, py) => {
        const g = ctx.createRadialGradient(px, py, 0, px, py, r);
        g.addColorStop(0.0, warm ? 'rgba(202,199,190,0.08)' : 'rgba(154,151,144,0.07)');
        g.addColorStop(1.0, 'rgba(0,0,0,0.0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Fine matte polymer micro-pebble variation.
    for (let i = 0; i < 16000; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 162 + Math.floor(random() * 54);
      const a = 0.016 + random() * 0.030;
      ctx.fillStyle = random() > 0.52
        ? `rgba(238,236,230,${a})`
        : `rgba(${v},${Math.max(0, v - 6)},${Math.max(0, v - 14)},${a})`;
      ctx.fillRect(x, y, 1 + random() * 2.0, 1 + random() * 0.8);
    }

    // Subtle lab wipe marks and faint organic arena wear (wrapped seamlessly).
    for (let i = 0; i < 40; i++) {
      const x = random() * size;
      const y = random() * size;
      const r = size * (0.018 + random() * 0.055);
      const rx = r * (0.7 + random() * 1.4);
      const ry = r * (0.24 + random() * 0.55);
      const ang = random() * Math.PI;
      const warm = random() > 0.48;
      drawWrapped(x, y, Math.max(rx, ry), (px, py) => {
        const g = ctx.createRadialGradient(px, py, 0, px, py, r);
        g.addColorStop(0.0, warm ? 'rgba(142,116,74,0.042)' : 'rgba(68,64,58,0.036)');
        g.addColorStop(0.68, warm ? 'rgba(142,116,74,0.014)' : 'rgba(68,64,58,0.012)');
        g.addColorStop(1.0, 'rgba(0,0,0,0.0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.ellipse(px, py, rx, ry, ang, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    ctx.lineCap = 'round';
    for (let i = 0; i < 150; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 18 + random() * 88;
      const a = random() * Math.PI * 2;
      const dx = Math.cos(a) * len;
      const dy = Math.sin(a) * len;
      ctx.strokeStyle = random() > 0.5 ? 'rgba(236,234,228,0.042)' : 'rgba(84,76,64,0.038)';
      ctx.lineWidth = 0.45 + random() * 1.0;
      drawWrapped(x, y, len, (px, py) => {
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + dx, py + dy);
        ctx.stroke();
      });
    }

    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.0, 2.1);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    return tex;
  }

  function createLabFloorRoughnessMap(size = 512) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#d2d2d2';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 9000; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 165 + Math.floor(random() * 78);
      ctx.fillStyle = `rgba(${v},${v},${v},${0.20 + random() * 0.25})`;
      ctx.fillRect(x, y, 1 + random() * 2.0, 1);
    }
    for (let i = 0; i < 28; i++) {
      const x = random() * size;
      const y = random() * size;
      const r = size * (0.02 + random() * 0.06);
      const ang = random() * Math.PI;
      const oxList = [0];
      const oyList = [0];
      if (x < r * 1.5) oxList.push(size);
      else if (x > size - r * 1.5) oxList.push(-size);
      if (y < r * 1.5) oyList.push(size);
      else if (y > size - r * 1.5) oyList.push(-size);
      for (const ox of oxList) {
        for (const oy of oyList) {
          const px = x + ox;
          const py = y + oy;
          const g = ctx.createRadialGradient(px, py, 0, px, py, r);
          g.addColorStop(0, 'rgba(128,128,128,0.20)');
          g.addColorStop(1, 'rgba(230,230,230,0.0)');
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.ellipse(px, py, r * 1.45, r * 0.45, ang, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.0, 2.1);
    return tex;
  }

  function createLabFloorNormalMap(size = 512) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 10000; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 118 + Math.floor(random() * 25);
      ctx.fillStyle = `rgba(${v},${v + 2},255,${0.12 + random() * 0.20})`;
      ctx.fillRect(x, y, 1 + random() * 1.5, 1);
    }
    ctx.strokeStyle = 'rgba(116,118,255,0.06)';
    for (let i = 0; i < 120; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 12 + random() * 48;
      const a = random() * Math.PI * 2;
      ctx.lineWidth = 0.7 + random() * 0.8;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len);
      ctx.stroke();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.0, 2.1);
    return tex;
  }

  function createSoftOvalTexture(size = 256, color = '0,0,0', alpha = 0.18) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(size * 0.5, size * 0.5, size * 0.02, size * 0.5, size * 0.5, size * 0.50);
    g.addColorStop(0.0, `rgba(${color},${alpha})`);
    g.addColorStop(0.45, `rgba(${color},${alpha * 0.45})`);
    g.addColorStop(1.0, `rgba(${color},0.0)`);
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }

  function createTailAlbedoMap(size = 512) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const rng = makeIsolatedRng(0x5A17C4E1);
    // Warm neutral multiplicative dermal base so each strain's tailColor drives hue cleanly.
    ctx.fillStyle = '#efe9e5';
    ctx.fillRect(0, 0, size, size);

    const ringRows = 26;
    const cols = 20;
    const rowH = size / ringRows;
    const colW = size / cols;

    // 1. Subtle longitudinal dermal tonal variation (wrapped seamlessly in U).
    for (let i = 0; i < 42; i++) {
      const x = rng() * size;
      const w = size * (0.04 + rng() * 0.08);
      const warm = rng() > 0.45;
      const alpha = 0.025 + rng() * 0.028;
      const offsets = [0];
      if (x - w < 0) offsets.push(size);
      if (x + w > size) offsets.push(-size);
      for (const ox of offsets) {
        const g = ctx.createLinearGradient(x + ox - w, 0, x + ox + w, 0);
        g.addColorStop(0.0, 'rgba(0,0,0,0)');
        g.addColorStop(0.5, warm ? `rgba(218,176,168,${alpha})` : `rgba(156,144,138,${alpha})`);
        g.addColorStop(1.0, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.fillRect(x + ox - w, 0, w * 2, size);
      }
    }

    // 2. Staggered imbricate annular epidermal scales (~182 delicate rings at repeat.y = 7).
    // Note: CanvasTexture flipY=true maps +V (root->tip) to -Y on canvas.
    for (let r = 0; r < ringRows; r++) {
      const y0 = r * rowH;
      const stagger = (r % 2) * colW * 0.5;

      // Delicate imbricate scale plate luminance ramp (luminous crown -> soft sulcus crease).
      const bandGrad = ctx.createLinearGradient(0, y0, 0, y0 + rowH);
      bandGrad.addColorStop(0.00, 'rgba(112,92,86,0.18)');
      bandGrad.addColorStop(0.15, 'rgba(134,114,106,0.08)');
      bandGrad.addColorStop(0.34, 'rgba(255,250,246,0.09)');
      bandGrad.addColorStop(0.76, 'rgba(248,242,238,0.04)');
      bandGrad.addColorStop(0.92, 'rgba(146,126,118,0.08)');
      bandGrad.addColorStop(1.00, 'rgba(108,88,82,0.16)');
      ctx.fillStyle = bandGrad;
      ctx.fillRect(0, y0, size, rowH);

      // Individual staggered epidermal scale tiles (wrapped cleanly at x=0/size).
      for (let col = 0; col < cols; col++) {
        const x0 = (col * colW + stagger) % size;
        const jitter = (rng() - 0.5) * 14;
        const fill = jitter >= 0
          ? `rgba(255,248,244,${(jitter / 255 * 0.52).toFixed(3)})`
          : `rgba(122,104,98,${(-jitter / 255 * 0.55).toFixed(3)})`;
        const xOffsets = x0 + colW > size ? [0, -size] : [0];
        for (const ox of xOffsets) {
          ctx.fillStyle = fill;
          ctx.fillRect(x0 + ox + 0.8, y0 + rowH * 0.16, colW - 1.6, rowH * 0.72);
          ctx.strokeStyle = 'rgba(122,102,96,0.11)';
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          ctx.moveTo(x0 + ox, y0 + rowH * 0.12);
          ctx.lineTo(x0 + ox + (rng() - 0.5) * 1.0, y0 + rowH * 0.90);
          ctx.stroke();
        }
      }

      // Fine inter-ring epidermal sulcus groove.
      ctx.fillStyle = 'rgba(96,78,72,0.14)';
      ctx.fillRect(0, y0, size, Math.max(1.0, rowH * 0.07));
    }

    // 3. Murine scale-margin caudal hair triplets (3 short fine hairs emerging caudally per scale).
    ctx.lineCap = 'round';
    for (let r = 0; r < ringRows; r++) {
      const yBase = r * rowH + rowH * 0.14;
      const stagger = (r % 2) * colW * 0.5;
      for (let col = 0; col < cols; col++) {
        if (rng() > 0.58) continue;
        const xCenter = (col * colW + stagger + colW * (0.28 + rng() * 0.44)) % size;
        const hairCount = rng() > 0.35 ? 3 : 2;
        for (let h = 0; h < hairCount; h++) {
          const hx = (xCenter + (h - (hairCount - 1) * 0.5) * 2.0 + size) % size;
          const hLen = rowH * (0.45 + rng() * 0.44);
          const drift = (rng() - 0.5) * 1.6;
          const alpha = 0.11 + rng() * 0.10;
          ctx.strokeStyle = `rgba(98,84,78,${alpha.toFixed(3)})`;
          ctx.lineWidth = 0.5 + rng() * 0.3;
          const yOffsets = yBase - hLen < 0 ? [0, size] : [0];
          const xOffsets = hx < 4 ? [0, size] : (hx > size - 4 ? [0, -size] : [0]);
          for (const ox of xOffsets) {
            for (const oy of yOffsets) {
              ctx.beginPath();
              ctx.moveTo(hx + ox, yBase + oy);
              ctx.lineTo(hx + ox + drift, yBase + oy - hLen);
              ctx.stroke();
            }
          }
        }
      }
    }

    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1.0, 7.0);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  function createTailRingRoughnessMap(size = 512) {
    const c = document.createElement('canvas'); c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    const rng = makeIsolatedRng(0x3B72D19A);
    const ringRows = 26;
    const rowH = size / ringRows;
    for (let y = 0; y < size; y++) {
      const rowFrac = (y % rowH) / rowH;
      // Imbricate keratinized ring profile: satin scale plate crown, matte inter-ring sulcus.
      const sulcusTop = Math.exp(-Math.pow(rowFrac / 0.12, 2.0));
      const sulcusBot = Math.exp(-Math.pow((1.0 - rowFrac) / 0.10, 2.0));
      const sulcus = Math.max(sulcusTop, sulcusBot);
      const crown = Math.exp(-Math.pow((rowFrac - 0.46) / 0.26, 2.0));
      const grain = (rng() - 0.5) * 5.0;
      const v = Math.floor(THREE.MathUtils.clamp(166 + sulcus * 54 - crown * 16 + grain, 138, 232));
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(0, y, size, 1);
    }
    ctx.globalAlpha = 0.06;
    for (let i = 0; i < 120; i++) {
      const y0 = rng() * size;
      const x0 = rng() * size;
      const shade = 144 + Math.floor(rng() * 42);
      ctx.strokeStyle = `rgb(${shade},${shade},${shade})`;
      ctx.lineWidth = 0.6 + rng() * 0.7;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x0 + (rng() - 0.5) * 3.0, y0 - (8 + rng() * 13));
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1.0, 7.0);
    return tex;
  }

  function createTailNormalMap(size = 512) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const rng = makeIsolatedRng(0x7E29B4C3);
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    const ringRows = 26;
    const cols = 20;
    const rowH = size / ringRows;
    const colW = size / cols;
    // Staggered annular epidermal scales aligned 1:1 with tailAlbedoMap & tailRoughnessMap.
    for (let r = 0; r < ringRows; r++) {
      const y0 = r * rowH;
      const stagger = (r % 2) * colW * 0.5;
      // Crisp yet delicate imbricate ring bevel along V across sulcus & crown.
      ctx.fillStyle = 'rgba(128,92,250,0.56)';
      ctx.fillRect(0, y0, size, Math.max(1.2, rowH * 0.13));
      ctx.fillStyle = 'rgba(128,154,252,0.36)';
      ctx.fillRect(0, y0 + rowH * 0.13, size, Math.max(1.2, rowH * 0.17));
      ctx.fillStyle = 'rgba(128,118,254,0.18)';
      ctx.fillRect(0, y0 + rowH * 0.78, size, Math.max(1.2, rowH * 0.22));
      for (let col = 0; col < cols; col++) {
        const x0 = (col * colW + stagger) % size;
        const xOffsets = x0 < 3 ? [0, size] : (x0 > size - 3 ? [0, -size] : [0]);
        for (const ox of xOffsets) {
          ctx.strokeStyle = 'rgba(108,128,248,0.28)';
          ctx.lineWidth = 0.9;
          ctx.beginPath();
          ctx.moveTo(x0 + ox, y0 + rowH * 0.12);
          ctx.lineTo(x0 + ox + (rng() - 0.5) * 0.9, y0 + rowH * 0.90);
          ctx.stroke();
          ctx.strokeStyle = 'rgba(146,128,250,0.20)';
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          ctx.moveTo(x0 + ox + 1.1, y0 + rowH * 0.12);
          ctx.lineTo(x0 + ox + 1.1 + (rng() - 0.5) * 0.9, y0 + rowH * 0.90);
          ctx.stroke();
        }
      }
    }
    // Fine caudal bristle impressions emerging along +V (-Y on canvas).
    ctx.lineCap = 'round';
    for (let i = 0; i < 360; i++) {
      const x = rng() * size;
      const y = rng() * size;
      const len = rowH * (0.48 + rng() * 0.50);
      ctx.strokeStyle = 'rgba(140,146,255,0.24)';
      ctx.lineWidth = 0.7;
      const yOffsets = y - len < 0 ? [0, size] : [0];
      for (const oy of yOffsets) {
        ctx.beginPath();
        ctx.moveTo(x, y + oy);
        ctx.lineTo(x + (rng() - 0.5) * 1.6, y + oy - len);
        ctx.stroke();
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1.0, 7.0);
    return tex;
  }

  function createEyeIrisMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    // Organic murine eye map: large deep dark pupil core surrounded by fine
    // radial iris stroma and a soft limbal border so both glossy jet-black
    // C57BL/6 beads and luminous ruby BALB/c eyes show living depth.
    ctx.fillStyle = '#c4b6b4';
    ctx.fillRect(0, 0, size, size);
    const cx = size * 0.5;
    const cy = size * 0.5;
    const g = ctx.createRadialGradient(cx, cy, size * 0.04, cx, cy, size * 0.48);
    g.addColorStop(0.0, '#180f11');
    g.addColorStop(0.32, '#2d1c1f');
    g.addColorStop(0.50, '#9c7e80');
    g.addColorStop(0.78, '#d4bcbd');
    g.addColorStop(0.92, '#785c5e');
    g.addColorStop(1.0, '#4a3638');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    ctx.lineCap = 'round';
    for (let i = 0; i < 200; i++) {
      const a = (i / 200) * Math.PI * 2 + (random() - 0.5) * 0.04;
      const r0 = size * (0.24 + random() * 0.06);
      const r1 = size * (0.40 + random() * 0.07);
      const shade = 170 + Math.floor(random() * 68);
      ctx.strokeStyle = `rgba(${shade},${shade - 12},${shade - 14},0.16)`;
      ctx.lineWidth = 0.60 + random() * 0.60;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.stroke();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }
  function createAlbinoEyeIrisMap(size = 256) {
    const rng = makeIsolatedRng(0x7A1B0E42);
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    // Authentic BALB/c & Nude albino eye: unpigmented translucent iris & retina give
    // a warm glowing ruby-rose central pupil ringed by deeper vascular garnet-claret.
    // SphereGeometry maps u across 360 deg and v across 180 deg, so scale X by 0.5
    // in equirectangular UV space for a true spherical iris with zero side-view banding.
    ctx.fillStyle = '#6c2432';
    ctx.fillRect(0, 0, size, size);
    const cx = size * 0.5;
    const cy = size * 0.5;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(0.5, 1.0);
    const g = ctx.createRadialGradient(0, 0, size * 0.02, 0, 0, size * 0.44);
    g.addColorStop(0.0, '#d86e7c');
    g.addColorStop(0.30, '#bf4d5f');
    g.addColorStop(0.60, '#962e40');
    g.addColorStop(0.84, '#732030');
    g.addColorStop(1.0, '#6c2432');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(0, 0, size * 0.46, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineCap = 'round';
    for (let i = 0; i < 96; i++) {
      const a = (i / 96) * Math.PI * 2 + (rng() - 0.5) * 0.05;
      const r0 = size * (0.16 + rng() * 0.05);
      const r1 = size * (0.31 + rng() * 0.05);
      ctx.strokeStyle = `rgba(232,148,162,${0.055 + rng() * 0.045})`;
      ctx.lineWidth = 0.70 + rng() * 0.55;
      ctx.beginPath();
      ctx.moveTo(Math.cos(a) * r0, Math.sin(a) * r0);
      ctx.lineTo(Math.cos(a) * r1, Math.sin(a) * r1);
      ctx.stroke();
    }
    ctx.restore();
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }
  function createEarNormalMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    const cx = size * 0.5;
    const cy = size * 0.58;
    const maxR = size * 0.48;
    // Fine cartilage grain
    for (let i = 0; i < 220; i++) {
      const a = random() * Math.PI * 2;
      const r = Math.pow(random(), 0.55) * maxR;
      const x = cx + Math.cos(a) * r;
      const y = cy + Math.sin(a) * r * 0.92;
      const d = Math.hypot(x - cx, y - cy);
      if (d > maxR) continue;
      const h = Math.floor(120 + random() * 25);
      ctx.fillStyle = `rgba(${h},${h},255,0.55)`;
      ctx.beginPath();
      ctx.ellipse(x, y, 1.0 + random() * 2.2, 0.7 + random() * 1.6, a, 0, Math.PI * 2);
      ctx.fill();
    }
    // Subtle radial striations (magnified ear look)
    ctx.lineCap = 'round';
    for (let i = 0; i < 120; i++) {
      const a = -Math.PI * 0.75 + random() * Math.PI * 1.5;
      const r0 = maxR * (0.08 + random() * 0.10);
      const r1 = maxR * (0.55 + random() * 0.40);
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0 * 0.90;
      const x1 = cx + Math.cos(a) * r1;
      const y1 = cy + Math.sin(a) * r1 * 0.92;
      const dx = Math.cos(a) * (8 + random() * 10);
      const dy = Math.sin(a) * (5 + random() * 9);
      const r = Math.max(98, Math.min(158, 128 + dx));
      const g = Math.max(98, Math.min(158, 128 + dy));
      ctx.strokeStyle = `rgba(${Math.floor(r)},${Math.floor(g)},255,${0.06 + random() * 0.07})`;
      ctx.lineWidth = 0.6 + random() * 1.2;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      const mx = (x0 + x1) * 0.5 + (random() - 0.5) * 6;
      const my = (y0 + y1) * 0.5 + (random() - 0.5) * 6;
      ctx.quadraticCurveTo(mx, my, x1, y1);
      ctx.stroke();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }
  function createEarRoughnessMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(size * 0.5, size * 0.55, size * 0.08, size * 0.5, size * 0.55, size * 0.52);
    g.addColorStop(0.0, '#9a9a9a');
    g.addColorStop(0.55, '#b8b8b8');
    g.addColorStop(1.0, '#d0d0d0');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 900; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 140 + Math.floor(random() * 80);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(x, y, 1, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }
  function createEarAlbedoMap(size = 512) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const cx = size * 0.5;
    const cy = size * 0.6;
    const maxR = size * 0.48;
    // Neutral skin response lets each strain's material color remain visible.
    const base = ctx.createRadialGradient(cx, cy, size * 0.04, cx, cy, maxR);
    base.addColorStop(0.0, '#f3ece9');
    base.addColorStop(0.35, '#ece3df');
    base.addColorStop(0.65, '#e4d9d4');
    base.addColorStop(0.85, '#d8cbc5');
    base.addColorStop(1.0, '#cbbcb5');
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, size, size);
    // Root warmth remains subtle so it does not tint the whole pinna peach.
    const root = ctx.createRadialGradient(cx, cy + maxR * 0.25, size * 0.04, cx, cy + maxR * 0.30, maxR * 0.60);
    root.addColorStop(0.0, 'rgba(188,112,108,0.12)');
    root.addColorStop(0.5, 'rgba(176,112,108,0.07)');
    root.addColorStop(1.0, 'rgba(170,118,98,0.0)');
    ctx.fillStyle = root;
    ctx.fillRect(0, 0, size, size);
    // Organic branching auricular microvasculature (curved, soft contrast so
    // the pinna reads as delicate translucent living tissue rather than spokes).
    const drawVein = (x0, y0, x1, y1, width, opacity, branches) => {
      ctx.strokeStyle = `rgba(176,88,90,${opacity})`;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.beginPath();
      const dx = x1 - x0;
      const dy = y1 - y0;
      const nx = -dy * (0.12 + (random() - 0.5) * 0.22);
      const ny = dx * (0.12 + (random() - 0.5) * 0.22);
      const c1x = x0 + dx * 0.35 + nx;
      const c1y = y0 + dy * 0.35 + ny;
      const c2x = x0 + dx * 0.72 - nx * 0.55;
      const c2y = y0 + dy * 0.72 - ny * 0.55;
      ctx.moveTo(x0, y0);
      ctx.bezierCurveTo(c1x, c1y, c2x, c2y, x1, y1);
      ctx.stroke();
      if (branches > 0 && width > 0.45) {
        for (let b = 0; b < 2; b++) {
          const bt = 0.45 + random() * 0.38;
          const bx = x0 + dx * bt + (random() - 0.5) * 14;
          const by = y0 + dy * bt + (random() - 0.5) * 14;
          const bx2 = bx + dx * 0.32 + (random() - 0.5) * size * 0.16;
          const by2 = by + dy * 0.32 - random() * size * 0.10;
          drawVein(bx, by, bx2, by2, width * 0.58, opacity * 0.72, branches - 1);
        }
      }
    };
    // Subtle organic auricular vessels curving from the basal concha toward the rim
    for (let i = 0; i < 9; i++) {
      const a = -Math.PI * 0.58 + (i / 8) * Math.PI * 1.16 + (random() - 0.5) * 0.10;
      const r0 = size * 0.07;
      const r1 = size * 0.38 + random() * size * 0.08;
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0;
      const x1 = cx + Math.cos(a) * r1;
      const y1 = cy + Math.sin(a) * r1;
      drawVein(x0, y0, x1, y1, 1.15 + random() * 0.85, 0.15 + random() * 0.07, 2);
    }
    // Fine capillary arcade near the outer third of the pinna
    for (let i = 0; i < 26; i++) {
      const a = -Math.PI * 0.72 + random() * Math.PI * 1.44;
      const r0 = size * (0.14 + random() * 0.15);
      const r1 = size * (0.31 + random() * 0.15);
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0 * 0.94;
      const x1 = cx + Math.cos(a + (random() - 0.5) * 0.25) * r1;
      const y1 = cy + Math.sin(a + (random() - 0.5) * 0.25) * r1 * 0.96;
      ctx.strokeStyle = `rgba(182,98,98,${0.095 + random() * 0.06})`;
      ctx.lineWidth = 0.50 + random() * 0.55;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      const mx = (x0 + x1) * 0.5 + (random() - 0.5) * 16;
      const my = (y0 + y1) * 0.5 + (random() - 0.5) * 12;
      ctx.quadraticCurveTo(mx, my, x1, y1);
      ctx.stroke();
    }
    // Translucent rim glow (cooler at edges)
    const rim = ctx.createRadialGradient(cx, cy, maxR * 0.75, cx, cy, maxR * 1.05);
    rim.addColorStop(0.0, 'rgba(255,255,255,0.0)');
    rim.addColorStop(0.40, 'rgba(255,245,235,0.0)');
    rim.addColorStop(0.70, 'rgba(240,234,230,0.18)');
    rim.addColorStop(1.0, 'rgba(214,202,196,0.28)');
    ctx.fillStyle = rim;
    ctx.fillRect(0, 0, size, size);
    // Micro mottling for skin texture
    for (let i = 0; i < 1500; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 200 + Math.floor(random() * 40);
      ctx.fillStyle = `rgba(${v},${v - 5},${v - 8},0.04)`;
      ctx.fillRect(x, y, 1, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }
  function createNoseAlbedoMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const cx = size * 0.5;
    const cy = size * 0.55;
    // Soft pale rosy-ivory base so strain noseColor modulates cleanly from
    // delicate shell-pink (BALB/c) to muted dusky rose-taupe (C57BL/6).
    const g = ctx.createRadialGradient(cx, cy, size * 0.03, cx, cy, size * 0.50);
    g.addColorStop(0.0, '#f6dcd8');
    g.addColorStop(0.48, '#ebd0cc');
    g.addColorStop(1.0, '#cfb5b0');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    // Gentle capillary warmth around nostril apertures without over-saturating.
    const ring = ctx.createRadialGradient(cx, cy, size * 0.05, cx, cy, size * 0.22);
    ring.addColorStop(0.0, 'rgba(155,72,74,0.06)');
    ring.addColorStop(0.55, 'rgba(170,84,86,0.10)');
    ring.addColorStop(1.0, 'rgba(120,40,40,0.0)');
    ctx.fillStyle = ring;
    ctx.fillRect(0, 0, size, size);
    // Median philtrum sulcus shading
    ctx.strokeStyle = 'rgba(118,68,68,0.28)';
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(cx, cy - size * 0.05);
    ctx.quadraticCurveTo(cx, cy + size * 0.08, cx, cy + size * 0.22);
    ctx.stroke();
    // Fine pavimentous rhinarium papillae
    for (let i = 0; i < 650; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 188 + Math.floor(random() * 44);
      ctx.fillStyle = `rgba(${v},${v - 14},${v - 16},0.10)`;
      ctx.beginPath();
      ctx.arc(x, y, 0.7 + random() * 1.1, 0, Math.PI * 2);
      ctx.fill();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }
  function createNoseNormalMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    // Pavimentous dermal pebble bumps characteristic of rodent rhinarium
    for (let i = 0; i < 920; i++) {
      const x = random() * size;
      const y = random() * size;
      const r = 0.8 + random() * 1.5;
      ctx.fillStyle = 'rgba(146,146,255,0.45)';
      ctx.beginPath();
      ctx.arc(x - r * 0.22, y - r * 0.22, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'rgba(110,112,250,0.32)';
      ctx.beginPath();
      ctx.arc(x + r * 0.25, y + r * 0.25, r * 0.85, 0, Math.PI * 2);
      ctx.fill();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }
  function createNoseRoughnessMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const g = ctx.createRadialGradient(size * 0.5, size * 0.54, size * 0.03, size * 0.5, size * 0.54, size * 0.52);
    g.addColorStop(0.0, '#585858');
    g.addColorStop(1.0, '#bebebe');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 900; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 112 + Math.floor(random() * 92);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(x, y, 1, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }
  function createCoatAlbedoMap(colorHex, size = 512) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const base = new THREE.Color(colorHex);
    const hsl = {}; base.getHSL(hsl);
    const isPaleCoat = hsl.l > 0.68;
    const isDarkCoat = hsl.l < 0.18;
    const isAgouti = hsl.s > 0.16 && hsl.l > 0.12 && hsl.l < 0.62;
    const root = isPaleCoat
      ? base.clone().offsetHSL(0.008, 0.04, -0.072)
      : (isDarkCoat
        ? base.clone().offsetHSL(0.012, 0.04, -0.028)
        : base.clone().offsetHSL(0.0, -0.02, -0.075));
    const subapical = isAgouti
      ? base.clone().offsetHSL(0.025, 0.15, 0.12)
      : (isDarkCoat
        ? base.clone().offsetHSL(0.015, 0.03, 0.032)
        : base.clone().offsetHSL(0.008, 0.01, isPaleCoat ? 0.014 : 0.045));
    const tip = isAgouti
      ? base.clone().offsetHSL(-0.01, -0.05, -0.10)
      : (isDarkCoat
        ? base.clone().offsetHSL(0.012, 0.01, 0.052)
        : base.clone().offsetHSL(0.006, 0.0, isPaleCoat ? 0.026 : 0.065));
    const baseSRGB = base.clone().convertLinearToSRGB();
    ctx.fillStyle = `rgb(${Math.round(baseSRGB.r * 255)},${Math.round(baseSRGB.g * 255)},${Math.round(baseSRGB.b * 255)})`;
    ctx.fillRect(0, 0, size, size);

    // Subtle wrapped undercoat shadow tufts give plush macro-depth without hard seams.
    const drawWrappedStroke = (x0, y0, dx, dy) => {
      const x1 = x0 + dx;
      const y1 = y0 + dy;
      const xOffsets = x1 < 0 ? [0, size] : (x1 > size ? [0, -size] : [0]);
      const yOffsets = y1 < 0 ? [0, size] : (y1 > size ? [0, -size] : [0]);
      for (const ox of xOffsets) {
        for (const oy of yOffsets) {
          ctx.beginPath();
          ctx.moveTo(x0 + ox, y0 + oy);
          ctx.lineTo(x1 + ox, y1 + oy);
          ctx.stroke();
        }
      }
    };

    ctx.lineCap = 'round';
    const rootSRGB = root.clone().convertLinearToSRGB();
    const rootRgb = `${Math.round(rootSRGB.r * 255)},${Math.round(rootSRGB.g * 255)},${Math.round(rootSRGB.b * 255)}`;
    for (let i = 0; i < 650; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 14 + random() * 26;
      const alpha = isPaleCoat ? (0.045 + random() * 0.045) : (0.07 + random() * 0.06);
      ctx.strokeStyle = `rgba(${rootRgb},${alpha})`;
      ctx.lineWidth = 2.4 + random() * 3.0;
      drawWrappedStroke(x, y, (random() - 0.5) * 4.5, len);
    }

    // Dense combed multi-scale pelage fibers (wrapped seamlessly across tile edges).
    for (let i = 0; i < 14000; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 6 + random() * 20;
      const roll = random();
      const tone = roll < 0.36 ? root : (roll < 0.74 ? subapical : tip);
      const fiber = tone.clone().convertLinearToSRGB();
      const alpha = isPaleCoat
        ? (0.09 + random() * 0.13)
        : (isDarkCoat ? (0.14 + random() * 0.18) : (0.13 + random() * 0.18));
      ctx.strokeStyle = `rgba(${Math.round(fiber.r * 255)},${Math.round(fiber.g * 255)},${Math.round(fiber.b * 255)},${alpha})`;
      ctx.lineWidth = 0.42 + random() * 0.85;
      drawWrappedStroke(x, y, (random() - 0.5) * 2.2, len);
    }

    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    // Exact integer U repeat eliminates the sphere seam down the cheek and flank.
    tex.repeat.set(4.0, 2.0);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    return tex;
  }
  function createCoatNormalMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    ctx.lineCap = 'round';
    for (let i = 0; i < 6400; i++) {
      const x = random() * size, y = random() * size;
      const span = 4 + random() * 14;
      const bend = (random() - 0.5) * 1.4;
      const yOffsets = y + span > size ? [0, -size] : [0];
      for (const oy of yOffsets) {
        for (const [offset, red] of [[-0.50, 106], [0.50, 150]]) {
          ctx.strokeStyle = `rgba(${red},128,251,0.34)`;
          ctx.lineWidth = 0.72;
          ctx.beginPath();
          ctx.moveTo(x + offset, y + oy);
          ctx.quadraticCurveTo(x + offset + bend, y + oy + span * 0.5, x + offset + bend * 0.4, y + oy + span);
          ctx.stroke();
        }
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(4.0, 2.0);
    return tex;
  }
  function createCoatRoughnessMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#c8c8c8';
    ctx.fillRect(0, 0, size, size);
    ctx.lineCap = 'round';
    for (let i = 0; i < 2800; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 4 + random() * 11;
      const v = 152 + Math.floor(random() * 72);
      ctx.strokeStyle = `rgba(${v},${v},${v},0.34)`;
      ctx.lineWidth = 0.9 + random() * 1.2;
      const yOffsets = y + len > size ? [0, -size] : [0];
      for (const oy of yOffsets) {
        ctx.beginPath();
        ctx.moveTo(x, y + oy);
        ctx.lineTo(x + (random() - 0.5) * 1.4, y + oy + len);
        ctx.stroke();
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.0, 2.0);
    return tex;
  }
  const padNormalMap = createPadNormalMap();
  const tailAlbedoMap = createTailAlbedoMap();
  const tailRoughnessMap = createTailRingRoughnessMap();
  const tailNormalMap = createTailNormalMap();
  const eyeIrisMap = createEyeIrisMap();
  const albinoEyeIrisMap = createAlbinoEyeIrisMap();
  const earNormalMap = createEarNormalMap();
  const earRoughnessMap = createEarRoughnessMap();
  const earAlbedoMap = createEarAlbedoMap();
  const noseAlbedoMap = createNoseAlbedoMap();
  const noseNormalMap = createNoseNormalMap();
  const noseRoughnessMap = createNoseRoughnessMap();
  const labFloorAlbedoMap = createLabFloorAlbedoMap();
  const labFloorRoughnessMap = createLabFloorRoughnessMap();
  const labFloorNormalMap = createLabFloorNormalMap();
  const urineSpotTexture = createSoftOvalTexture(256, '147,116,62', 0.085);
  const shadowSmudgeTexture = createSoftOvalTexture(256, '46,39,31', 0.13);

  function makeIsolatedRng(seed) {
    let st = seed >>> 0;
    return () => {
      st = (st + 0x6D2B79F5) >>> 0;
      let v = st;
      v = Math.imul(v ^ (v >>> 15), v | 1);
      v ^= v + Math.imul(v ^ (v >>> 7), v | 61);
      return ((v ^ (v >>> 14)) >>> 0) / 4294967296;
    };
  }

  function createHomeCageBeddingMaps(size = 512) {
    const rng = makeIsolatedRng(0x51C39E4D);
    const cAlb = document.createElement('canvas'); cAlb.width = cAlb.height = size;
    const cNrm = document.createElement('canvas'); cNrm.width = cNrm.height = size;
    const cRgh = document.createElement('canvas'); cRgh.width = cRgh.height = size;
    const ctxA = cAlb.getContext('2d');
    const ctxN = cNrm.getContext('2d');
    const ctxR = cRgh.getContext('2d');

    // Deep inter-granule shadow substrate
    ctxA.fillStyle = '#736858';
    ctxA.fillRect(0, 0, size, size);
    ctxN.fillStyle = '#8080ff';
    ctxN.fillRect(0, 0, size, size);
    ctxR.fillStyle = '#dcdcdc';
    ctxR.fillRect(0, 0, size, size);

    // Dense overlapping corncob & aspen wood-chip granules (tileable across seams)
    const granuleCount = 5400;
    for (let i = 0; i < granuleCount; i++) {
      const x = rng() * size;
      const y = rng() * size;
      const isShaving = rng() < 0.28;
      const w = isShaving ? (6 + rng() * 14) : (3.5 + rng() * 7.5);
      const h = isShaving ? (2.2 + rng() * 4.8) : (2.8 + rng() * 6.2);
      const ang = rng() * Math.PI;
      const toneRoll = rng();
      let rC;
      let gC;
      let bC;
      if (toneRoll < 0.16) {
        // Dark bark / toasted hull fleck
        const d = 76 + Math.floor(rng() * 38);
        rC = d + 10; gC = d + 4; bC = d - 6;
      } else if (toneRoll < 0.72) {
        // Natural oat / corncob medium granule
        const m = 148 + Math.floor(rng() * 46);
        rC = m + 12; gC = m + 6; bC = m - 10;
      } else {
        // Bright aspen / birch chip highlight
        const l = 198 + Math.floor(rng() * 42);
        rC = Math.min(246, l + 8); gC = Math.min(242, l + 4); bC = Math.max(170, l - 12);
      }

      const nx = Math.floor(128 + Math.cos(ang) * (16 + rng() * 18));
      const ny = Math.floor(128 + Math.sin(ang) * (16 + rng() * 18));
      const rv = 176 + Math.floor(rng() * 62);

      // Wrap tile draws near borders for seamless RepeatWrapping
      const oxList = [0];
      const oyList = [0];
      if (x < 16) oxList.push(size);
      else if (x > size - 16) oxList.push(-size);
      if (y < 16) oyList.push(size);
      else if (y > size - 16) oyList.push(-size);

      for (const ox of oxList) {
        for (const oy of oyList) {
          const px = x + ox;
          const py = y + oy;
          // Subtle crevice shadow under each granule
          ctxA.save();
          ctxA.translate(px + 0.9, py + 1.0);
          ctxA.rotate(ang);
          ctxA.fillStyle = 'rgba(38,33,27,0.34)';
          ctxA.fillRect(-w * 0.52, -h * 0.52, w * 1.04, h * 1.04);
          ctxA.restore();

          ctxA.save();
          ctxA.translate(px, py);
          ctxA.rotate(ang);
          ctxA.fillStyle = `rgb(${rC},${gC},${bC})`;
          ctxA.fillRect(-w * 0.5, -h * 0.5, w, h);
          ctxA.fillStyle = 'rgba(255,250,240,0.14)';
          ctxA.fillRect(-w * 0.45, -h * 0.45, w * 0.9, h * 0.32);
          ctxA.restore();

          ctxN.save();
          ctxN.translate(px, py);
          ctxN.rotate(ang);
          ctxN.fillStyle = `rgb(${nx},${ny},248)`;
          ctxN.fillRect(-w * 0.5, -h * 0.5, w, h);
          ctxN.restore();

          ctxR.save();
          ctxR.translate(px, py);
          ctxR.rotate(ang);
          ctxR.fillStyle = `rgb(${rv},${rv},${rv})`;
          ctxR.fillRect(-w * 0.5, -h * 0.5, w, h);
          ctxR.restore();
        }
      }
    }

    const albedo = new THREE.CanvasTexture(cAlb);
    albedo.wrapS = albedo.wrapT = THREE.RepeatWrapping;
    albedo.repeat.set(3.8, 2.7);
    albedo.colorSpace = THREE.SRGBColorSpace;
    albedo.anisotropy = 8;

    const normal = new THREE.CanvasTexture(cNrm);
    normal.wrapS = normal.wrapT = THREE.RepeatWrapping;
    normal.repeat.set(3.8, 2.7);

    const roughness = new THREE.CanvasTexture(cRgh);
    roughness.wrapS = roughness.wrapT = THREE.RepeatWrapping;
    roughness.repeat.set(3.8, 2.7);

    return { albedo, normal, roughness };
  }
  function createPawSkinMaps(size = 128) {
    const rng = makeIsolatedRng(0x4C81D92B);
    // 1. Digital skin normal map: fine transverse epidermal annular creases
    //    characteristic of murine phalanges plus subtle dermal micro-grain.
    const cDig = document.createElement('canvas'); cDig.width = cDig.height = size;
    const dCtx = cDig.getContext('2d');
    dCtx.fillStyle = '#8080ff';
    dCtx.fillRect(0, 0, size, size);
    const rings = 18;
    const bandH = size / rings;
    for (let r = 0; r < rings; r++) {
      const y0 = r * bandH;
      const g = dCtx.createLinearGradient(0, y0, 0, y0 + bandH);
      g.addColorStop(0.0, 'rgb(128,154,250)');
      g.addColorStop(0.28, 'rgb(128,128,255)');
      g.addColorStop(0.76, 'rgb(128,108,248)');
      g.addColorStop(1.0, 'rgb(128,150,250)');
      dCtx.fillStyle = g;
      dCtx.fillRect(0, y0, size, bandH + 0.5);
    }
    for (let i = 0; i < 420; i++) {
      const x = rng() * size;
      const y = rng() * size;
      const nx = 121 + Math.floor(rng() * 14);
      const ny = 121 + Math.floor(rng() * 14);
      dCtx.fillStyle = `rgba(${nx},${ny},255,0.32)`;
      dCtx.beginPath();
      dCtx.arc(x, y, 0.8 + rng() * 1.4, 0, Math.PI * 2);
      dCtx.fill();
    }
    const digitNormal = new THREE.CanvasTexture(cDig);
    digitNormal.wrapS = digitNormal.wrapT = THREE.RepeatWrapping;

    // 2. Dorsal paw albedo & normal maps: subtle longitudinal fine pelage
    //    streaks and soft metacarpal/metatarsal vascular tonal warmth.
    const cAlb = document.createElement('canvas'); cAlb.width = cAlb.height = size;
    const aCtx = cAlb.getContext('2d');
    const cNrm = document.createElement('canvas'); cNrm.width = cNrm.height = size;
    const nCtx = cNrm.getContext('2d');
    aCtx.fillStyle = '#f3ece9';
    aCtx.fillRect(0, 0, size, size);
    nCtx.fillStyle = '#8080ff';
    nCtx.fillRect(0, 0, size, size);
    // Soft vascular warmth distally, slightly brighter silky sheen dorsally.
    const grad = aCtx.createLinearGradient(0, 0, 0, size);
    grad.addColorStop(0.0, 'rgba(232,226,222,0.22)');
    grad.addColorStop(0.55, 'rgba(248,244,242,0.10)');
    grad.addColorStop(1.0, 'rgba(238,205,200,0.20)');
    aCtx.fillStyle = grad;
    aCtx.fillRect(0, 0, size, size);
    for (let i = 0; i < 520; i++) {
      const x = rng() * size;
      const y = rng() * size;
      const len = 5 + rng() * 12;
      const drift = (rng() - 0.5) * 2.2;
      const light = rng() > 0.35;
      aCtx.strokeStyle = light
        ? `rgba(255,252,248,${(0.10 + rng() * 0.14).toFixed(3)})`
        : `rgba(196,178,172,${(0.06 + rng() * 0.08).toFixed(3)})`;
      aCtx.lineWidth = 0.7 + rng() * 0.9;
      aCtx.beginPath();
      aCtx.moveTo(x, y);
      aCtx.lineTo(x + drift, y + len);
      aCtx.stroke();

      nCtx.strokeStyle = light ? 'rgba(140,132,255,0.38)' : 'rgba(116,124,255,0.34)';
      nCtx.lineWidth = 0.9;
      nCtx.beginPath();
      nCtx.moveTo(x, y);
      nCtx.lineTo(x + drift, y + len);
      nCtx.stroke();
    }
    const dorsalAlbedo = new THREE.CanvasTexture(cAlb);
    dorsalAlbedo.wrapS = dorsalAlbedo.wrapT = THREE.RepeatWrapping;
    dorsalAlbedo.colorSpace = THREE.SRGBColorSpace;
    const dorsalNormal = new THREE.CanvasTexture(cNrm);
    dorsalNormal.wrapS = dorsalNormal.wrapT = THREE.RepeatWrapping;
    return { digitNormal, dorsalAlbedo, dorsalNormal };
  }
  const pawSkinMaps = createPawSkinMaps();
  const homeCageBeddingMaps = createHomeCageBeddingMaps();

  // ═══════════════════════════════════════════════════════════════
  // 4. PBR MATERIALS
  // ═══════════════════════════════════════════════════════════════
  const fleshMatBase = new THREE.MeshPhysicalMaterial({
    roughness: 0.56, transmission: 0.12, thickness: 1.1,
    attenuationDistance: 0.9, clearcoat: 0.08, clearcoatRoughness: 0.28,
    sheen: 0.22, sheenColor: new THREE.Color(0xead2cc), sheenRoughness: 0.65,
  });
  const clawMat = new THREE.MeshPhysicalMaterial({
    color: 0xeee5da, roughness: 0.24, metalness: 0.0,
    clearcoat: 0.72, clearcoatRoughness: 0.18,
    transmission: 0.18, thickness: 0.16,
    ior: 1.54, // translucent keratin IOR
  });
  // Two-layer eye with subtle iris stroma depth + ultra-clear wet tear-film cornea
  const eyeMat = new THREE.MeshPhysicalMaterial({
    color: 0x050404, map: eyeIrisMap, roughness: 0.08, metalness: 0.0,
    clearcoat: 1.0, clearcoatRoughness: 0.03, envMapIntensity: 1.25,
    ior: 1.38,
  });
  const corneaMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, transparent: true, opacity: 0.045,
    depthWrite: false,
    roughness: 0.01, metalness: 0.0, ior: 1.376,
    clearcoat: 1.0, clearcoatRoughness: 0.015, envMapIntensity: 1.85,
  });

  function createHairlessSkinAlbedoMap(colorHex, size = 256) {
    const rng = makeIsolatedRng(0x6E8D3A11);
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const base = new THREE.Color(colorHex).convertLinearToSRGB();
    const r0 = Math.round(base.r * 255);
    const g0 = Math.round(base.g * 255);
    const b0 = Math.round(base.b * 255);
    ctx.fillStyle = `rgb(${r0},${g0},${b0})`;
    ctx.fillRect(0, 0, size, size);
    // Soft subdermal vascular mottling (wrapped seamlessly across tile edges)
    for (let i = 0; i < 42; i++) {
      const x = rng() * size;
      const y = rng() * size;
      const rad = size * (0.06 + rng() * 0.14);
      const warm = rng() > 0.45;
      const oxList = [0];
      const oyList = [0];
      if (x < rad) oxList.push(size);
      else if (x > size - rad) oxList.push(-size);
      if (y < rad) oyList.push(size);
      else if (y > size - rad) oyList.push(-size);
      for (const ox of oxList) {
        for (const oy of oyList) {
          const px = x + ox;
          const py = y + oy;
          const g = ctx.createRadialGradient(px, py, 0, px, py, rad);
          g.addColorStop(0.0, warm ? 'rgba(218,142,138,0.085)' : 'rgba(255,236,232,0.075)');
          g.addColorStop(1.0, 'rgba(0,0,0,0.0)');
          ctx.fillStyle = g;
          ctx.beginPath();
          ctx.arc(px, py, rad, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
    // Fine dermal pore grain & subtle transverse skin creases
    for (let i = 0; i < 2400; i++) {
      const x = rng() * size;
      const y = rng() * size;
      ctx.fillStyle = rng() > 0.5
        ? 'rgba(204,142,138,0.055)'
        : 'rgba(255,244,240,0.050)';
      ctx.fillRect(x, y, 1.4, 1.0);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.0, 2.0);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  function createCoatMaterial(furColor, skinColor, density, params) {
    const fur = new THREE.Color(furColor);
    const skin = new THREE.Color(skinColor);
    const hsl = {}; fur.getHSL(hsl);
    const isPale = hsl.l > 0.65;
    const isDark = hsl.l < 0.18;
    const isHairless = density <= 0;
    const material = new THREE.MeshPhysicalMaterial({
      // The albedo map already contains the strain pigment. Keeping the
      // material tint neutral avoids multiplying dark coats into black.
      color: 0xffffff,
      roughness: isHairless ? 0.62 : (isPale ? 0.89 : (isDark ? 0.82 : 0.84)),
      envMapIntensity: isHairless ? 0.34 : (isPale ? 0.34 : (isDark ? 0.42 : 0.38)),
      metalness: 0.0,
      clearcoat: isHairless ? 0.05 : (isDark ? 0.02 : 0.0),
      clearcoatRoughness: isHairless ? 0.52 : 0.68,
      sheen: isHairless ? 0.20 : (isPale ? 0.68 : (isDark ? 0.54 : 0.50)),
      sheenColor: isDark
        ? new THREE.Color(0x3a3438)
        : (isPale
          ? new THREE.Color(0xfcf8f2)
          : (isHairless ? skin : fur).clone().offsetHSL(0.005, -0.01, 0.09)),
      sheenRoughness: isHairless ? 0.64 : (isDark ? 0.56 : 0.60),
      map: isHairless
        ? createHairlessSkinAlbedoMap(skinColor)
        : createCoatAlbedoMap(furColor),
      normalMap: isHairless ? padNormalMap : createCoatNormalMap(),
      normalScale: isHairless
        ? new THREE.Vector2(0.08, 0.08)
        : (isPale
          ? new THREE.Vector2(0.10 + density * 0.06, 0.11 + density * 0.06)
          : new THREE.Vector2(0.14 + density * 0.09, 0.15 + density * 0.10)),
      roughnessMap: isHairless ? null : createCoatRoughnessMap(),
    });
    if (!isHairless && annolidShaders?.applyFurShader && params?.enableFurShader !== false) {
      const rootCol = fur.clone().offsetHSL(0.005, 0.01, isPale ? -0.032 : -0.025);
      const tipCol = isDark
        ? fur.clone().offsetHSL(0.012, 0.01, 0.072)
        : fur.clone().offsetHSL(0.008, -0.01, isPale ? 0.026 : 0.070);
      annolidShaders.applyFurShader(material, {
        anisotropy: isPale ? 26.0 : (isDark ? 32.0 : 30.0),
        specIntensity: isPale ? 0.042 : (isDark ? 0.066 : 0.054),
        rootColor: [rootCol.r, rootCol.g, rootCol.b],
        tipColor: [tipCol.r, tipCol.g, tipCol.b],
        flowDir: [0.0, -0.08, -0.996],
      });
    } else if (isHairless && annolidShaders?.applySSSShader && params?.enableSSS !== false) {
      annolidShaders.applySSSShader(material, {
        scatterColor: [skin.r * 0.88, skin.g * 0.76, skin.b * 0.74],
        thickness: (params.sssThickness ?? 0.45) * 0.35,
        distortion: 0.2,
        wrap: params.sssWrap ?? 0.5,
        backlight: (params.sssBacklight ?? 0.6) * 0.4,
      });
    }
    if (params.enableMicroDetail !== false) {
      annolidShaders.applyMicroDetail(material, {
        scale: Math.max(70.0, params.microScale ?? 80.0),
        strength: Math.min(isPale ? 0.003 : 0.015, (params.microStrength ?? 0.06) * (isPale ? 0.05 : 0.24))
      });
    }
    return material;
  }

  function enhanceFurMaterial(material, furColor, isUndercoat = false) {
    const base = new THREE.Color(furColor);
    const hsl = {}; base.getHSL(hsl);
    const isPale = hsl.l > 0.65;
    const isDark = hsl.l < 0.18;
    material.color.copy(base).offsetHSL(
      isDark ? 0.008 : 0.004,
      isPale ? 0.003 : 0.0,
      isUndercoat
        ? (isPale ? -0.010 : (isDark ? -0.010 : -0.016))
        : (isPale ? 0.012 : (isDark ? 0.012 : 0.016))
    );
    material.transparent = false;
    material.depthWrite = true;
    material.envMapIntensity = isUndercoat ? (isPale ? 0.26 : 0.16) : (isDark ? 0.36 : 0.38);
    if (material.isMeshPhysicalMaterial) {
      material.roughness = isDark ? 0.80 : (isPale ? 0.86 : 0.80);
      material.sheen = isUndercoat ? 0.26 : (isDark ? 0.50 : 0.68);
      material.sheenColor = isDark
        ? new THREE.Color(0x3a3438)
        : (isPale
          ? new THREE.Color(0xfdfaf5)
          : base.clone().offsetHSL(0.008, -0.02, 0.12));
      material.sheenRoughness = isUndercoat ? 0.84 : (isDark ? 0.58 : 0.62);
    }
    return material;
  }

  // ═══════════════════════════════════════════════════════════════
  // 5. DEFORMATION FUNCTIONS
  // ═══════════════════════════════════════════════════════════════
  function weldParametricSeamNormals(geometry, radialSegments, rings) {
    const nrm = geometry.attributes.normal;
    if (!nrm || !radialSegments || !rings) return;
    const row = radialSegments + 1;
    if (nrm.count < (rings + 1) * row) return;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    for (let r = 0; r <= rings; r++) {
      const i0 = r * row;
      const i1 = i0 + radialSegments;
      a.fromBufferAttribute(nrm, i0);
      b.fromBufferAttribute(nrm, i1);
      a.add(b);
      if (a.lengthSq() > 1e-10) {
        a.normalize();
        nrm.setXYZ(i0, a.x, a.y, a.z);
        nrm.setXYZ(i1, a.x, a.y, a.z);
      }
    }
    nrm.needsUpdate = true;
  }

  function deform(geometry, deformFn) {
    const pos = geometry.attributes.position;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      deformFn(v);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geometry.computeVertexNormals();
    const p = geometry.parameters;
    if (p && p.widthSegments && p.heightSegments) {
      weldParametricSeamNormals(geometry, p.widthSegments, p.heightSegments);
    }
    return geometry;
  }

  const deformBody = (v) => {
    const zNorm = (v.z + 1) / 2;
    // Authentic C2-continuous murine pear silhouette matching C57BL/6 & BALB/c dorsal
    // and lateral lab recordings: broad U-rounded hindquarters & muscular pelvic haunches
    // tapering gently through a supple mid-trunk into sleeker shoulders and continuous nape.
    const rumpRoundness = Math.exp(-Math.pow((zNorm - 0.11) / 0.13, 2.0));
    const rearMass = Math.exp(-Math.pow((zNorm - 0.27) / 0.27, 2.0));
    const shoulderSlope = THREE.MathUtils.smoothstep(zNorm, 0.50, 0.96);
    const waistNarrow = Math.exp(-Math.pow((zNorm - 0.58) / 0.18, 2.0));
    const lateral = (
      0.95
      + rumpRoundness * 0.20
      + rearMass * 0.21 * (PROP.haunchWidth / 1.02)
      - shoulderSlope * (0.14 - (PROP.shoulderWidth - 0.88) * 0.22)
      - waistNarrow * (1.0 - PROP.waistTaper) * 0.20
    );
    v.x *= lateral * PROP.bodyGirthX;
    // Sleeker vertical profile (less balloon-egg height) with slightly fuller pelvic depth
    // so the torso sits low and grounded over the crouched limbs.
    v.y *= (0.87 + rearMass * 0.11 + rumpRoundness * 0.06 - zNorm * 0.09) * PROP.bodyGirthY;
    v.z *= PROP.bodyLength;
    // Smooth elliptical flank curvature so top view shows zero hard shelf.
    const sideBlend = THREE.MathUtils.clamp((Math.abs(v.y) - 0.05) / 0.72, 0, 1);
    v.x *= 0.96 + sideBlend * 0.04;

    // Dorsal thoracolumbar arch: natural gentle murine kyphotic curve peaking over lumbar
    // spine, rounding softly toward the sacrum, and flowing smoothly over the withers into
    // the cervical nape (eliminating the sunken notch behind the skull).
    const hump = Math.exp(-Math.pow((zNorm - 0.34) / 0.26, 2.0));
    const rump = Math.exp(-Math.pow((zNorm - 0.10) / 0.13, 2.0));
    const withers = Math.exp(-Math.pow((zNorm - 0.76) / 0.18, 2.0));
    const napeBridge = THREE.MathUtils.smoothstep(zNorm, 0.74, 0.96) * Math.max(0, v.y);
    v.y += hump * 0.185 + rump * 0.045 + withers * 0.135 + napeBridge * 0.065 - shoulderSlope * 0.014;

    // Low-slung grounded ventrum: fuller lower haunch/abdomen settling organically over
    // the flexed hindlimbs rather than curving high like a floating boat hull.
    const yRef = v.y;
    const bellyWidth = Math.exp(-Math.pow((zNorm - 0.40) / 0.30, 2.0));
    const bellyMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.68, 0.16);
    const bellyInfluence = bellyWidth * bellyMask;
    v.x *= 1.0 + bellyInfluence * 0.10;
    const dropMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.26, 0.20);
    const rearDrop = Math.exp(-Math.pow((zNorm - 0.26) / 0.22, 2.0));
    v.y -= (bellyWidth * 0.105 + rearDrop * 0.095) * dropMask;

    const lowerBlend = THREE.MathUtils.smoothstep(0.08 - yRef, 0.0, 0.48);
    // Keep caudal tuck gentle and localized right near the perineum/tail root.
    const groinTuck = Math.exp(-Math.pow((zNorm - 0.06) / 0.12, 2.0)) * 0.075;
    let chestDrop = THREE.MathUtils.clamp((zNorm - 0.68) / 0.32, 0, 1);
    chestDrop = chestDrop * chestDrop * (3.0 - 2.0 * chestDrop) * 0.065;
    const targetY = v.y * 0.87 + groinTuck - chestDrop;
    v.y = THREE.MathUtils.lerp(v.y, targetY, lowerBlend);

    // Muscular gluteal-haunch and scapular relief so flanks read as living anatomy.
    if (Math.abs(v.x) > 0.20) {
      const scapBell = Math.exp(-Math.pow((zNorm - 0.74) / 0.10, 2.0));
      v.x *= 1.0 + scapBell * 0.032;
      if (v.y > 0) v.y += scapBell * 0.024;
      const haunchBell = Math.exp(-Math.pow((zNorm - 0.24) / 0.17, 2.0));
      v.x *= 1.0 + haunchBell * 0.052;
      if (v.y < 0.05) v.y -= haunchBell * 0.042;
    }

    // Keep rump broadly U-rounded and curve gently downward over the sacrum into
    // the low pelvic base (real crouched mouse rump rather than a floating cone).
    const sacralDrop = Math.pow(1.0 - THREE.MathUtils.smoothstep(zNorm, 0.0, 0.24), 1.55) * 0.145;
    v.y -= sacralDrop;
    if (zNorm < 0.09) {
      const tailBlend = THREE.MathUtils.smoothstep(zNorm, 0.0, 0.09);
      v.x *= 0.96 + tailBlend * 0.04;
      v.y *= 0.94 + tailBlend * 0.06;
    }

    // Smooth C2 anterior taper merging tangentially into the broad cervical nape.
    if (zNorm > 0.60) {
      const neckT = THREE.MathUtils.smoothstep(zNorm, 0.60, 0.99);
      v.x *= 1.0 - neckT * 0.28;
      v.y *= 1.0 - neckT * (v.y >= 0 ? 0.14 : 0.12);
    }
  };

  const deformHead = (v) => {
    const zNorm = (v.z + 1) / 2;
    const smooth = THREE.MathUtils.smoothstep;
    // Blunt the rostral pole gently into a small rounded murine muzzle tip that
    // carries the rhinarium without creating a bulbous knob or needle spike.
    const tipZone = smooth(zNorm, 0.82, 1.0);
    const ringRadius = Math.sqrt(Math.max(0, 1 - v.z * v.z));
    const blunt = 1.0 + tipZone * 0.42 * (1.0 - ringRadius);
    v.x *= blunt;
    v.y *= blunt;
    v.z -= tipZone * tipZone * 0.048;

    // Sleek murine wedge (micro-CT proportions): gently domed cranium transitioning
    // into an elongated, delicate tapered rostrum with a noticeably lean, tucked mandible
    // so front & side profiles read like a real mouse rather than a round cheek ball.
    const rostral = smooth(zNorm, 0.22, 1.0);
    v.x *= (1.0 - 0.65 * rostral) * PROP.snoutWidth * 0.80;
    const dorsalMask = smooth(v.y, -0.14, 0.24);
    const vertBase = THREE.MathUtils.lerp(0.51, 0.82, dorsalMask);
    v.y *= (1.0 - 0.48 * rostral) * vertBase;
    v.z *= PROP.snoutLength * 1.08;
    // Slight natural forward-downward rostral slope characteristic of sniffing rodents.
    v.y -= zNorm * 0.115;

    // Seamless occipital sweep into the cervical mantle (no proud circular cheek step),
    // with restrained zygomatic fullness below the orbits.
    const occipitalBlend = THREE.MathUtils.smoothstep(zNorm, 0.02, 0.56);
    v.x *= 0.88 + occipitalBlend * 0.12;
    v.y *= 0.90 + occipitalBlend * 0.10;
    const cheek = Math.exp(-Math.pow((zNorm - 0.46) / 0.18, 2.0));
    const lowerMask = 1.0 - smooth(v.y, -0.12, 0.20);
    v.x *= 1.0 + cheek * lowerMask * PROP.cheekFullness * 0.15;

    // Retrognathic tucked rodent lower jaw: sweeps cleanly upward from throat to a small
    // recessed chin tucked back underneath the overhanging upper muzzle.
    const ventral = 1.0 - smooth(v.y, -0.20, 0.04);
    const jawTuck = smooth(zNorm, 0.36, 0.96);
    v.x *= 1.0 - ventral * jawTuck * 0.30;
    v.y += ventral * smooth(zNorm, 0.46, 0.98) * 0.082;
    const chin = Math.exp(-Math.pow((zNorm - 0.82) / 0.08, 2.0));
    v.y -= ventral * chin * 0.010;
    const overhang = smooth(zNorm, 0.74, 1.0);
    v.z -= ventral * overhang * 0.068;

    // Sculpted orbital fossa + soft supraorbital brow ridge so eyes nestle organically
    // inside the cranium instead of sitting proud on a smooth egg surface.
    const orbitZ = Math.exp(-Math.pow((zNorm - 0.64) / 0.11, 2.0));
    const orbitY = Math.exp(-Math.pow((v.y - 0.12) / 0.14, 2.0));
    const latMask = smooth(Math.abs(v.x), 0.12, 0.32);
    v.x *= 1.0 - orbitZ * orbitY * latMask * 0.095;
    const browZ = Math.exp(-Math.pow((zNorm - 0.62) / 0.11, 2.0));
    const browY = Math.exp(-Math.pow((v.y - 0.27) / 0.09, 2.0));
    v.x *= 1.0 + browZ * browY * latMask * 0.042;
    v.y += browZ * browY * latMask * 0.022;

    // Mystacial whisker pads: soft plush bilateral cushions framing the sub-nasal philtrum.
    const padZ = Math.exp(-Math.pow((zNorm - 0.85) / 0.085, 2.0));
    const padY = Math.exp(-Math.pow((v.y + 0.095) / 0.082, 2.0));
    const padLat = smooth(Math.abs(v.x), 0.022, 0.11);
    v.x *= 1.0 + padZ * padY * padLat * 0.19;
    v.y -= padZ * padY * padLat * 0.008;
    const philtrumCenter = Math.exp(-Math.pow(v.x / 0.030, 2.0));
    const philtrumZone = smooth(zNorm, 0.86, 0.99) * Math.exp(-Math.pow((v.y + 0.12) / 0.06, 2.0));
    v.z -= philtrumCenter * philtrumZone * 0.018;
    // Recessed sub-terminal oral fossa tucked underneath the overhanging rhinarium.
    const oralX = Math.exp(-Math.pow(v.x / 0.054, 2.0));
    const oralY = Math.exp(-Math.pow((v.y + 0.175) / 0.052, 2.0));
    const oralZ = smooth(zNorm, 0.80, 0.95);
    v.z -= oralX * oralY * oralZ * 0.034;

    // Dorsal skull silhouette: smooth cranial vault -> subtle sleek nasofrontal stop
    // ahead of the eyes -> delicately arched elongated nasal bridge to the tip.
    const dorsal = smooth(v.y, -0.02, 0.22);
    const cranium = Math.exp(-Math.pow((zNorm - 0.32) / 0.25, 2.0));
    v.y += dorsal * cranium * 0.095;
    const nasalStop = Math.exp(-Math.pow((zNorm - 0.62) / 0.11, 2.0));
    v.y -= dorsal * nasalStop * 0.024;
    const nasal = Math.exp(-Math.pow((zNorm - 0.81) / 0.13, 2.0));
    v.y += dorsal * nasal * 0.044;

    // Temporal fullness supports the pinna roots cleanly.
    const temporal = Math.exp(-Math.pow((zNorm - 0.26) / 0.14, 2.0));
    const temporalHeight = smooth(v.y, -0.10, 0.44);
    v.x *= 1.0 + temporal * temporalHeight * 0.040;
    // Subtle zygomatic arch ridge below/behind the eye.
    const zygoZ = Math.exp(-Math.pow((zNorm - 0.48) / 0.10, 2.0));
    const zygoY = Math.exp(-Math.pow((v.y + 0.01) / 0.10, 2.0));
    v.x *= 1.0 + zygoZ * zygoY * 0.026;
  };

  // Precomputed source directions used to locate features (eyes, nose, mouth,
  // whisker pads, ears) on the deformed head surface. Rebuilt with the head.
  let headSurfaceGrid = null;
  function buildHeadSurfaceGrid() {
    const grid = [];
    const p = new THREE.Vector3();
    for (let i = 0; i <= 72; i++) {
      const theta = (i / 72) * Math.PI;
      for (let j = 0; j < 144; j++) {
        const phi = (j / 144) * Math.PI * 2;
        const source = new THREE.Vector3(
          Math.sin(theta) * Math.cos(phi),
          Math.sin(theta) * Math.sin(phi),
          Math.cos(theta)
        );
        p.copy(source);
        deformHead(p);
        grid.push({ source, x: p.x, y: p.y, z: p.z });
      }
    }
    return grid;
  }
  const headSurfaceSampler = createDeformedSurfaceSampler(THREE, deformHead);

  /**
   * Closest point on the deformed head surface to the given partial target
   * (any of x, y, z), returned with its outward normal in head-local space.
   */
  function findHeadSurfacePoint(target = {}) {
    if (!headSurfaceGrid) headSurfaceGrid = buildHeadSurfaceGrid();
    let best = null;
    let bestCost = Infinity;
    for (const sample of headSurfaceGrid) {
      let cost = 0;
      if (target.x !== undefined) cost += (sample.x - target.x) ** 2;
      if (target.y !== undefined) cost += (sample.y - target.y) ** 2;
      if (target.z !== undefined) cost += (sample.z - target.z) ** 2;
      if (cost < bestCost) {
        bestCost = cost;
        best = sample;
      }
    }
    const position = new THREE.Vector3();
    const normal = new THREE.Vector3();
    headSurfaceSampler(best.source.clone(), position, normal);
    return { position, normal };
  }

  /**
   * Facial landmarks on the current head surface (head-local, before the
   * per-subject head scale). Shared by the scene (eyes, nose, mouth, whisker
   * pads, keypoints) and by head-fur generation so fur never covers them.
   */
  let headFeatureAnchors = null;
  function getHeadFeatureAnchors() {
    if (headFeatureAnchors) return headFeatureAnchors;
    const nose = findHeadSurfacePoint({ x: 0, y: -0.085, z: 3.0 });
    const tipZ = nose.position.z;
    const eye = side => findHeadSurfacePoint({ x: side * 0.64, y: 0.14, z: 0.34 * PROP.snoutLength });
    const pad = side => findHeadSurfacePoint({ x: side * 0.48, y: nose.position.y - 0.035, z: tipZ - 0.22 });
    const mouth = findHeadSurfacePoint({ x: 0, y: nose.position.y - 0.140, z: tipZ - 0.11 });
    headFeatureAnchors = Object.freeze({
      nose,
      mouth,
      eyeL: eye(1),
      eyeR: eye(-1),
      padL: pad(1),
      padR: pad(-1),
    });
    return headFeatureAnchors;
  }

  function getCervicalMantleProfile(t) {
    const eased = t * t * (3.0 - 2.0 * t);
    const arch = Math.sin(t * Math.PI);
    return {
      z: (t - 0.5) * PROP.neckLength,
      centerY: THREE.MathUtils.lerp(0.05, 0.31, eased) + arch * 0.032,
      radiusX: THREE.MathUtils.lerp(
        PROP.neckGirth * 0.93,
        PROP.neckGirth * 0.78,
        t
      ),
      radiusY: THREE.MathUtils.lerp(
        PROP.neckGirth * 0.96,
        PROP.neckGirth * 0.82,
        t
      ),
    };
  }

  function createCervicalMantleGeometry() {
    const rings = 18;
    const radialSegments = 36;
    const row = radialSegments + 1;
    const positions = [];
    const uvs = [];
    const indices = [];
    for (let ring = 0; ring <= rings; ring++) {
      const t = ring / rings;
      const profile = getCervicalMantleProfile(t);
      for (let segment = 0; segment <= radialSegments; segment++) {
        const angle = segment / radialSegments * Math.PI * 2;
        positions.push(
          Math.cos(angle) * profile.radiusX,
          profile.centerY + Math.sin(angle) * profile.radiusY,
          profile.z
        );
        uvs.push(segment / radialSegments, t);
      }
    }
    for (let ring = 0; ring < rings; ring++) {
      for (let segment = 0; segment < radialSegments; segment++) {
        const a = ring * row + segment;
        const b = a + row;
        indices.push(a, a + 1, b, b, a + 1, b + 1);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    return geometry;
  }

  // Radius fraction of the cervical mantle occupied by the solid core: matches the
  // anterior shoulder tangent (rawT -> 0) and tapers cleanly inside the occipital skull
  // (rawT -> 1) so no barrel step or vertical seam appears across furred or hairless coats.
  function cervicalCoreInset(rawT) {
    const backBlend = 1.0 - (1.0 - THREE.MathUtils.smoothstep(rawT, 0.0, 0.22)) * 0.14;
    const frontBlend = 1.0 - THREE.MathUtils.smoothstep(rawT, 0.58, 1.0) * 0.44;
    return THREE.MathUtils.lerp(0.94, 0.84, rawT) * backBlend * frontBlend;
  }

  /**
   * Unit-sphere -> cervical core surface map (sphere z runs along the neck).
   * Lets the body fur generator coat the neck with identical materials and
   * flow so no shading collar appears between head, neck, and body.
   */
  const deformCervicalCore = (v) => {
    const rawT = THREE.MathUtils.clamp((v.z + 1) * 0.5, 0, 1);
    const profile = getCervicalMantleProfile(rawT * 0.96);
    const angle = Math.atan2(v.y, v.x);
    const inset = cervicalCoreInset(rawT);
    v.set(
      Math.cos(angle) * profile.radiusX * inset,
      profile.centerY + Math.sin(angle) * profile.radiusY * inset,
      profile.z
    );
  };

  function createCervicalCoreGeometry() {
    const rings = 18;
    const radialSegments = 32;
    const row = radialSegments + 1;
    const positions = [];
    const uvs = [];
    const indices = [];
    for (let ring = 0; ring <= rings; ring++) {
      const rawT = ring / rings;
      const t = rawT * 0.96;
      const profile = getCervicalMantleProfile(t);
      const inset = cervicalCoreInset(rawT);
      for (let segment = 0; segment <= radialSegments; segment++) {
        const angle = segment / radialSegments * Math.PI * 2;
        positions.push(
          Math.cos(angle) * profile.radiusX * inset,
          profile.centerY + Math.sin(angle) * profile.radiusY * inset,
          profile.z
        );
        uvs.push(segment / radialSegments, rawT * 0.42);
      }
    }
    for (let ring = 0; ring < rings; ring++) {
      for (let segment = 0; segment < radialSegments; segment++) {
        const a = ring * row + segment;
        const b = a + row;
        indices.push(a, a + 1, b, b, a + 1, b + 1);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    weldParametricSeamNormals(geometry, radialSegments, rings);
    return geometry;
  }

  const deformThigh = (v) => {
    const yN = (v.y + 1) / 2; // 0=bottom(knee), 1=top(hip)
    // Muscular quadriceps/biceps femoris haunch mass tapering cleanly into stifle (knee)
    const kneeTaper = 0.74 + yN * 0.36;
    v.x *= 0.64 * kneeTaper;
    v.y *= 1.24;
    v.z *= 1.24 * (0.82 + yN * 0.26);
    v.z -= v.y * 0.26;
    // Tuck caudal-distal hamstring so thigh flows smoothly into the flexed knee joint
    if (v.z < 0 && yN < 0.42) {
      v.z *= 0.80 + (yN / 0.42) * 0.20;
    }
    const gluteal = Math.exp(-Math.pow((yN - 0.82) / 0.18, 2.0));
    v.x *= 1.0 + gluteal * 0.16;
    if (v.z > 0) v.z += gluteal * 0.05;
  };

  function shapePinnaPoint(x, y, depthMul = 1.0, zLift = 0.0, sideSign = 1) {
    const radius = Math.min(1.0, Math.sqrt(x * x + y * y));
    // Anatomical lateral coordinate: xm < 0 is medial/anterior helix margin,
    // xm > 0 is lateral/posterior outer margin (mirrored cleanly per ear).
    const xm = x * sideSign;
    const lowerOpen = THREE.MathUtils.smoothstep(y, -0.98, 0.04);
    const funnelTaper = 0.16 + 0.84 * Math.pow(lowerOpen, 0.64);
    // Natural tulip-petal murine dome: slightly ovoid/swept rather than a flat circle.
    const crownTaper = 1.0 - THREE.MathUtils.smoothstep(y, 0.42, 1.0) * 0.075;
    const midFullness = Math.exp(-Math.pow((y - 0.08) / 0.66, 2.0));
    const rim = THREE.MathUtils.smoothstep(radius, 0.76, 1.0);

    // Smooth C-infinity lateral/medial shaping without any corner kinks:
    // gentle posterolateral lobe fullness + subtle lower antitragal curve.
    const lateralMask = 0.5 * (1.0 + Math.tanh(xm * 2.4));
    const lateralFull = lateralMask * Math.abs(xm) * (
      midFullness * 0.056
      - Math.exp(-Math.pow((y + 0.46) / 0.26, 2.0)) * 0.046
    );
    const apexSweep = sideSign * THREE.MathUtils.smoothstep(y, -0.25, 0.96) * 0.036;
    const px = x * (0.83 + midFullness * 0.055) * funnelTaper * crownTaper
      + sideSign * lateralFull + apexSweep;

    const py = y * 0.96 + 0.040 * THREE.MathUtils.smoothstep(y, -0.25, 0.95);

    // Deeper acoustic conchal bowl + sculpted helical curl + prominent murine
    // antihelix/plica cartilaginous folds so pinnae read as living organic petals.
    const conchaBowl = Math.exp(-(
      Math.pow(xm + 0.03, 2.0) * 2.1 + Math.pow(y + 0.22, 2.0) * 1.85
    ));
    const upperScaphaCup = Math.max(0.0, 1.0 - radius * radius)
      * THREE.MathUtils.smoothstep(y, -0.35, 0.55);
    const upperCupMask = THREE.MathUtils.smoothstep(y, -0.45, 0.28);
    const sideCurl = (x * x) * (0.168 * upperCupMask + 0.055);
    // Mid-upper medial helix rolls gently forward; lower-anterior base tucks
    // smoothly backward into the temporal fur collar.
    const medialCurl = THREE.MathUtils.smoothstep(-xm, 0.22, 0.98)
      * THREE.MathUtils.smoothstep(y, -0.10, 0.48)
      * (1.0 - THREE.MathUtils.smoothstep(y, 0.58, 0.98) * 0.55)
      * 0.056;
    const anteriorBaseTuck = THREE.MathUtils.smoothstep(-xm, 0.05, 0.92)
      * THREE.MathUtils.smoothstep(-y, 0.08, 0.86)
      * 0.155;
    const crownBackFlex = THREE.MathUtils.smoothstep(y, 0.34, 1.0) * 0.048;
    const antihelixDist = Math.hypot((xm - 0.05) * 1.15, (y + 0.10) * 0.95) - 0.54;
    const antihelixArc = Math.exp(-(antihelixDist * antihelixDist) * 32.0)
      * THREE.MathUtils.smoothstep(y, -0.52, 0.38);
    const plicaRidge = Math.exp(-(
      Math.pow(xm + 0.04, 2.0) * 4.8
      + Math.pow(y + 0.26 - xm * 0.16, 2.0) * 36.0
    ));
    const basalWall = Math.exp(-(
      Math.pow(xm + 0.20, 2.0) * 10.0 + Math.pow(y + 0.54, 2.0) * 16.0
    ));
    const stemRecess = THREE.MathUtils.smoothstep(-y, 0.48, 1.0) * 0.125;

    const pz = (
      -conchaBowl * 0.34
      - upperScaphaCup * 0.20
      + sideCurl
      + medialCurl
      - anteriorBaseTuck
      - crownBackFlex
      + rim * 0.068
      + antihelixArc * 0.052
      + plicaRidge * 0.048
      + basalWall * 0.040
      - stemRecess
    ) * depthMul + zLift;
    return new THREE.Vector3(px, py, pz);
  }

  function createPinnaShellGeometry(sideSign = 1) {
    const rings = 20;
    const segments = 72;
    const row = segments + 1;
    const surfaceVertexCount = (rings + 1) * row;
    const positions = [];
    const uvs = [];
    const indices = [];

    const thicknessAt = (x, y, normalizedRadius) => {
      const rootMass = Math.exp(-(x * x * 4.5 + Math.pow(y + 0.78, 2.0) * 10.0));
      return 0.011 + (1.0 - normalizedRadius) * 0.010 + rootMass * 0.018;
    };

    for (let side = 0; side < 2; side++) {
      for (let ring = 0; ring <= rings; ring++) {
        const normalizedRadius = ring / rings;
        for (let seg = 0; seg <= segments; seg++) {
          const angle = (seg / segments) * Math.PI * 2;
          const x = Math.cos(angle) * normalizedRadius;
          const y = Math.sin(angle) * normalizedRadius;
          const point = shapePinnaPoint(x, y, 1.0, 0.0, sideSign);
          if (side === 1) {
            point.z -= thicknessAt(x, y, normalizedRadius);
          }
          positions.push(point.x, point.y, point.z);
          uvs.push(
            0.5 + Math.cos(angle) * normalizedRadius * 0.5,
            0.5 + Math.sin(angle) * normalizedRadius * 0.5
          );
        }
      }
    }

    // Rounded mid-margin seam ring so the outer helix curls smoothly from the
    // inner cup onto the posterior back with no harsh paper-cut sidewall.
    const edgeMidStart = surfaceVertexCount * 2;
    for (let seg = 0; seg <= segments; seg++) {
      const angle = (seg / segments) * Math.PI * 2;
      const x = Math.cos(angle);
      const y = Math.sin(angle);
      const basePt = shapePinnaPoint(x, y, 1.0, 0.0, sideSign);
      const outPt = shapePinnaPoint(x * 0.97, y * 0.97, 1.0, 0.0, sideSign);
      const outDirX = basePt.x - outPt.x;
      const outDirY = basePt.y - outPt.y;
      const outLen = Math.hypot(outDirX, outDirY) || 1.0;
      const thick = thicknessAt(x, y, 1.0);
      const roundOut = thick * 0.42;
      positions.push(
        basePt.x + (outDirX / outLen) * roundOut,
        basePt.y + (outDirY / outLen) * roundOut,
        basePt.z - thick * 0.50
      );
      uvs.push(0.5 + x * 0.5, 0.5 + y * 0.5);
    }

    // Group 0: concave inner (anterior) surface. Group 1: convex outer
    // (posterior) surface plus the rounded rolled margin.
    const innerIndices = [];
    const outerIndices = [];
    for (let ring = 0; ring < rings; ring++) {
      for (let seg = 0; seg < segments; seg++) {
        const frontA = ring * row + seg;
        const frontB = frontA + row;
        innerIndices.push(frontA, frontB, frontA + 1, frontB, frontB + 1, frontA + 1);
        const backA = surfaceVertexCount + frontA;
        const backB = surfaceVertexCount + frontB;
        outerIndices.push(backA, backA + 1, backB, backB, backA + 1, backB + 1);
      }
    }

    const outerFront = rings * row;
    const outerBack = surfaceVertexCount + outerFront;
    for (let seg = 0; seg < segments; seg++) {
      const frontA = outerFront + seg;
      const frontB = frontA + 1;
      const midA = edgeMidStart + seg;
      const midB = midA + 1;
      const backA = outerBack + seg;
      const backB = backA + 1;
      outerIndices.push(frontA, midA, frontB, frontB, midA, midB);
      outerIndices.push(midA, backA, midB, midB, backA, backB);
    }
    indices.push(...innerIndices, ...outerIndices);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.addGroup(0, innerIndices.length, 0);
    geometry.addGroup(innerIndices.length, outerIndices.length, 1);
    geometry.computeVertexNormals();
    return geometry;
  }

  function createPinnaSurfaceGeometry(radiusScale = 0.78, depthMul = 1.0, zLift = 0.002, sideSign = 1) {
    const rings = 16;
    const segments = 72;
    const row = segments + 1;
    const positions = [];
    const uvs = [];
    const indices = [];
    for (let ring = 0; ring <= rings; ring++) {
      const normalizedRadius = ring / rings;
      const radius = normalizedRadius * radiusScale;
      for (let seg = 0; seg <= segments; seg++) {
        const angle = (seg / segments) * Math.PI * 2;
        const point = shapePinnaPoint(
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          depthMul,
          zLift,
          sideSign
        );
        positions.push(point.x, point.y, point.z);
        uvs.push(0.5 + Math.cos(angle) * normalizedRadius * 0.5, 0.5 + Math.sin(angle) * normalizedRadius * 0.5);
      }
    }
    for (let ring = 0; ring < rings; ring++) {
      for (let seg = 0; seg < segments; seg++) {
        const a = ring * row + seg;
        const b = a + row;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    return geometry;
  }

  function createPinnaRimGeometry(sideSign = 1) {
    const segments = 76;
    const positions = [];
    const uvs = [];
    const indices = [];
    for (let band = 0; band < 2; band++) {
      for (let seg = 0; seg <= segments; seg++) {
        const t = seg / segments;
        // Delicate flush rolled helix lip fading cleanly into the basal funnel.
        const angle = -Math.PI * 0.22 + t * Math.PI * 1.44;
        const endFade = THREE.MathUtils.smoothstep(t, 0.0, 0.18)
          * THREE.MathUtils.smoothstep(1.0 - t, 0.0, 0.18);
        const radius = band === 0 ? 1.0 - 0.015 * endFade : 0.999;
        const point = shapePinnaPoint(
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          1.0,
          0.0003 + endFade * 0.0011,
          sideSign
        );
        positions.push(point.x, point.y, point.z + (band === 0 ? 0.0004 * endFade : 0.0));
        uvs.push(seg / segments, band);
      }
    }
    const row = segments + 1;
    for (let seg = 0; seg < segments; seg++) {
      const a = seg;
      const b = row + seg;
      indices.push(a, b, a + 1, b, b + 1, a + 1);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    return geometry;
  }

  // Upper arm / forearm deform
  const deformUpperArm = (v) => {
    // Slight taper from shoulder to elbow
    const t = (v.y + 1) / 2;
    v.x *= 0.7 - t * 0.15;
    v.z *= 0.7 - t * 0.15;
    v.y *= 0.45;
  };

  const bodyGeo = deform(new THREE.SphereGeometry(1, 64, 64), deformBody);
  const headGeo = deform(new THREE.SphereGeometry(1, 48, 48), deformHead);
  const neckGeo = createCervicalMantleGeometry();
  const cervicalCoreGeo = createCervicalCoreGeometry();
  const thighGeo = deform(new THREE.SphereGeometry(1, 32, 32), deformThigh);
  const earGeo = createPinnaShellGeometry(1);
  const earGeoR = createPinnaShellGeometry(-1);
  const earInnerGeo = createPinnaSurfaceGeometry(0.44, 1.0, 0.002, 1);
  const earInnerGeoR = createPinnaSurfaceGeometry(0.44, 1.0, 0.002, -1);
  const earRimGeo = createPinnaRimGeometry(1);
  const earRimGeoR = createPinnaRimGeometry(-1);

  function copyGeometryPositionsInPlace(targetGeo, freshGeo) {
    const dst = targetGeo.attributes.position;
    const src = freshGeo.attributes.position;
    if (dst && src && dst.array.length === src.array.length) {
      dst.array.set(src.array);
      dst.needsUpdate = true;
    } else {
      targetGeo.setAttribute('position', src);
    }
    targetGeo.computeVertexNormals();
    if (targetGeo.attributes.normal) {
      targetGeo.attributes.normal.needsUpdate = true;
    }
    targetGeo.computeBoundingBox();
    targetGeo.computeBoundingSphere();
  }

  function rebuildSharedAnatomyGeometries() {
    copyGeometryPositionsInPlace(
      bodyGeo,
      deform(new THREE.SphereGeometry(1, 64, 64), deformBody)
    );
    copyGeometryPositionsInPlace(
      headGeo,
      deform(new THREE.SphereGeometry(1, 48, 48), deformHead)
    );
    headSurfaceGrid = null;
    headFeatureAnchors = null;
    copyGeometryPositionsInPlace(neckGeo, createCervicalMantleGeometry());
    copyGeometryPositionsInPlace(cervicalCoreGeo, createCervicalCoreGeometry());
  }


  // ═══════════════════════════════════════════════════════════════
  // 6. PAW & LIMB PELAGE BUILDERS (handed murine manus & pes)
  // ═══════════════════════════════════════════════════════════════
  function computePawDorsalColor(strain, furColor) {
    const col = new THREE.Color(strain.pawColor ?? '#e6b0a7');
    if (furColor) {
      const coat = new THREE.Color(furColor);
      const coatHsl = {}; coat.getHSL(coatHsl);
      const silkyLift = coatHsl.l < 0.25 ? 0.16 : (coatHsl.l > 0.70 ? 0.36 : 0.22);
      const dorsalCoatBlend = coatHsl.l < 0.25 ? 0.12 : (coatHsl.l > 0.70 ? 0.22 : 0.15);
      col.lerp(new THREE.Color(0xf6efec), silkyLift);
      col.lerp(coat, dorsalCoatBlend);
    }
    return col;
  }

  function applyCuffVertexGradient(geometry, span, distalDirY, proxColor, distColor) {
    const pos = geometry.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    const tmp = new THREE.Color();
    const halfSpan = Math.max(span * 0.5, 0.01);
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      const yN = THREE.MathUtils.clamp((y + halfSpan) / (halfSpan * 2), 0, 1);
      const tDist = distalDirY < 0 ? (1.0 - yN) : yN;
      const blend = THREE.MathUtils.smoothstep(tDist, 0.12, 0.78);
      tmp.copy(proxColor).lerp(distColor, blend);
      colors[i * 3] = tmp.r;
      colors[i * 3 + 1] = tmp.g;
      colors[i * 3 + 2] = tmp.b;
    }
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  }

  function createLimbFurSleeve({
    profileFn,
    span = 0.0,
    count = 800,
    furColor = '#2a2a2e',
    pawSkinColor = '#e6b0a7',
    hairLength = 0.044,
    hairRadius = null,
    distalDirY = -1,
    featherDistal = false,
    coverCaps = false,
    seed = 0x51A01923,
    isUndercoat = false,
  }) {
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(seed);
    const furCol = new THREE.Color(furColor);
    const furHsl = {}; furCol.getHSL(furHsl);
    const isPaleCoat = furHsl.l > 0.65;
    // Match real murine limb-to-paw transition: pure silky ivory-white on pale coats,
    // soft silvery-warm transition on dark coats (zero harsh contrasting speckles).
    const distalTipCol = isPaleCoat
      ? furCol.clone().lerp(new THREE.Color(0xfbf7f4), 0.50)
      : new THREE.Color(pawSkinColor).lerp(new THREE.Color(0xdad2cc), 0.42);
    const geo = new THREE.ConeGeometry(
      hairRadius ?? (hairLength * (isUndercoat ? 0.145 : 0.125)),
      hairLength,
      3
    );
    geo.translate(0, hairLength * 0.5, 0);
    const mat = enhanceFurMaterial(
      new THREE.MeshStandardMaterial({
        color: featherDistal ? 0xffffff : furCol,
        roughness: isUndercoat ? 0.90 : 0.84,
      }),
      furColor,
      isUndercoat
    );
    // Keep feathered cuffs neutral white so instanceColor alone drives the
    // smooth coat-to-paw gradient without double-multiplying dark/muddy bands.
    if (featherDistal) mat.color.setHex(0xffffff);
    const inst = new THREE.InstancedMesh(geo, mat, count);
    inst.castShadow = false;
    inst.receiveShadow = !isUndercoat && !isPaleCoat;
    inst.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const nrm = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const strandCol = new THREE.Color();
    for (let i = 0; i < count; i++) {
      // Sample distal progress tDist in [0, 1] (0 = proximal, 1 = distal).
      // When feathering onto a paw cuff, concentrate fur densely over the proximal
      // coat-colored cuff (0..0.42) and taper smoothly out by 0.72 so no isolated
      // dark speckles scatter onto the pale dorsal foot.
      const tDist = featherDistal ? Math.pow(rng(), 2.10) * 0.72 : rng();
      const yN = distalDirY < 0 ? (1.0 - tDist) : tDist;
      const ang = rng() * Math.PI * 2;
      const ca = Math.cos(ang);
      const sa = Math.sin(ang);
      const { rx, rz, offsetZ = 0 } = profileFn(yN);
      let px;
      let py;
      let pz;
      if (coverCaps && rng() < 0.34) {
        // Wrap the rounded hemispherical end-caps (e.g. calcaneal heel bulb)
        // so the rear of the hock has zero bare smooth sphere peeking out.
        const capSign = rng() < 0.72 ? -distalDirY : distalDirY;
        const elev = Math.asin(rng());
        const ringR = Math.cos(elev);
        px = ca * rx * ringR;
        pz = sa * rz * ringR + offsetZ;
        py = capSign * (span * 0.5 + Math.sin(elev) * Math.min(rx, rz));
        nrm.set(ca * ringR, capSign * Math.sin(elev), sa * ringR).normalize();
      } else {
        px = ca * rx;
        py = (yN - 0.5) * span;
        pz = sa * rz + offsetZ;
        nrm.set(ca / Math.max(rx, 1e-4), 0, sa / Math.max(rz, 1e-4)).normalize();
      }
      dummy.position.set(px + nrm.x * 0.0014, py + nrm.y * 0.0014, pz + nrm.z * 0.0014);
      // Sleek proximal-to-distal lay down the limb axis with subtle radial lift.
      const lift = featherDistal
        ? THREE.MathUtils.lerp(0.24, 0.09, tDist / 0.72)
        : (isUndercoat ? 0.26 : 0.18);
      dir.set(
        nrm.x * lift + (rng() - 0.5) * 0.06,
        distalDirY * 0.94 + nrm.y * 0.18,
        nrm.z * lift + (rng() - 0.5) * 0.06
      ).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const featherLen = featherDistal
        ? THREE.MathUtils.lerp(1.06, 0.24, Math.pow(tDist / 0.72, 0.85))
        : (0.82 + rng() * 0.34);
      const sLen = featherLen * (0.82 + rng() * 0.32);
      const sWid = (featherDistal ? THREE.MathUtils.lerp(1.0, 0.56, tDist / 0.72) : 1.0)
        * (0.82 + rng() * 0.28);
      dummy.scale.set(sWid, sLen, sWid);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);

      const shade = (isPaleCoat ? 0.96 : 0.90) + rng() * (isPaleCoat ? 0.04 : 0.10);
      if (featherDistal) {
        const blend = THREE.MathUtils.smoothstep(tDist, 0.20, 0.68);
        strandCol.copy(furCol).lerp(distalTipCol, blend).multiplyScalar(shade);
      } else {
        strandCol.setRGB(shade, shade, shade);
      }
      inst.instanceColor.setXYZ(i, strandCol.r, strandCol.g, strandCol.b);
    }
    return inst;
  }

  function createPawDorsalMicroFur(isFore, pawScale, strain, furColor, density, sideSign) {
    if (density <= 0) return new THREE.Group();
    const count = Math.floor((isFore ? 150 : 230) * Math.max(0.45, density));
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(
      (isFore ? 0x6A31F100 : 0x6A31F200) + (sideSign > 0 ? 1 : 2)
    );
    const hairLen = (isFore ? 0.022 : 0.026) * pawScale;
    const geo = new THREE.ConeGeometry(hairLen * 0.10, hairLen, 3);
    geo.translate(0, hairLen * 0.5, 0);
    // Real murine dorsal paw micro-hairs are fine, pale silky-ivory to warm silver,
    // softly tinted toward the coat only at the very proximal carpus/tarsus.
    const coat = furColor ? new THREE.Color(furColor) : new THREE.Color('#c8c0b8');
    const coatHsl = {}; coat.getHSL(coatHsl);
    const paleSilk = new THREE.Color(strain.pawColor ?? '#e6b0a7').lerp(
      new THREE.Color(0xf9f5f2),
      coatHsl.l > 0.65 ? 0.78 : 0.62
    );
    const proxTint = coatHsl.l > 0.65
      ? paleSilk.clone().lerp(coat, 0.55)
      : paleSilk.clone().lerp(coat, coatHsl.l < 0.25 ? 0.24 : 0.18);
    const mat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 0.78,
      metalness: 0.0,
    });
    const inst = new THREE.InstancedMesh(geo, mat, count);
    inst.name = isFore ? 'forepaw-dorsal-microfur' : 'hindpaw-dorsal-microfur';
    inst.castShadow = false;
    inst.receiveShadow = false;
    inst.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const dir = new THREE.Vector3();
    const col = new THREE.Color();
    for (let i = 0; i < count; i++) {
      const zFrac = Math.pow(rng(), 0.88); // 0 = proximal wrist/tarsus, 1 = knuckle line
      const u = (rng() - 0.5) * 2.0; // [-1, 1] across width
      const lateralEllip = Math.sqrt(Math.max(0.08, 1.0 - u * u * 0.78));
      let x;
      let y;
      let z;
      if (isFore) {
        const halfW = THREE.MathUtils.lerp(0.044, 0.068, zFrac) * pawScale;
        x = u * halfW;
        z = THREE.MathUtils.lerp(-0.042, 0.104, zFrac) * pawScale;
        const archY = 0.011 * lateralEllip * (1.0 - zFrac * 0.42) - zFrac * zFrac * 0.014;
        y = -0.018 + archY;
      } else {
        const halfW = THREE.MathUtils.lerp(0.046, 0.082, zFrac) * pawScale;
        x = u * halfW;
        z = THREE.MathUtils.lerp(-0.185, 0.175, zFrac) * pawScale;
        const archY = 0.015 * lateralEllip * (1.0 - zFrac * 0.36)
          + Math.sin(zFrac * Math.PI) * 0.006 - zFrac * 0.014;
        y = -0.019 + archY;
      }
      dummy.position.set(x, y, z);
      dir.set(u * 0.18 + (rng() - 0.5) * 0.08, 0.10 - zFrac * 0.14, 0.96).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const lenScale = THREE.MathUtils.lerp(1.12, 0.52, zFrac) * (0.76 + rng() * 0.42);
      dummy.scale.set(0.85, lenScale, 0.85);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
      col.copy(proxTint).lerp(paleSilk, THREE.MathUtils.smoothstep(zFrac, 0.08, 0.65));
      const shade = 0.92 + rng() * 0.12;
      inst.instanceColor.setXYZ(i, col.r * shade, col.g * shade, col.b * shade);
    }
    return inst;
  }

  function createPaw(mat, isFore = false, strain = {}, furColor = null, side = 'L', density = 1.0) {
    const paw = new THREE.Group();
    const sideSign = side === 'R' ? -1 : 1; // +1 = Left (+X lateral, -X medial)
    paw.name = isFore ? 'connected-forepaw' : 'connected-hindpaw';
    const pawBaseColor = new THREE.Color(strain.pawColor ?? '#e6b0a7');
    const pawFleshMat = mat.clone();
    pawFleshMat.color.copy(pawBaseColor);
    pawFleshMat.transmission = 0.11;
    pawFleshMat.thickness = 0.14;
    pawFleshMat.roughness = 0.54;
    pawFleshMat.clearcoat = 0.05;
    pawFleshMat.clearcoatRoughness = 0.44;
    pawFleshMat.sheen = 0.28;
    pawFleshMat.sheenColor = pawBaseColor.clone().offsetHSL(0, -0.04, 0.10);
    pawFleshMat.normalMap = pawSkinMaps.digitNormal;
    pawFleshMat.normalScale = new THREE.Vector2(0.22, 0.22);

    // Delicate silky dorsal pelage tint over the metacarpus/metatarsus:
    // uses computePawDorsalColor so wrist/tarsus cuffs match the exact dorsal tone.
    const pawDorsalMat = pawFleshMat.clone();
    pawDorsalMat.map = pawSkinMaps.dorsalAlbedo;
    pawDorsalMat.normalMap = pawSkinMaps.dorsalNormal;
    pawDorsalMat.normalScale = new THREE.Vector2(0.18, 0.18);
    pawDorsalMat.color.copy(computePawDorsalColor(strain, furColor));
    if (furColor) {
      const coat = new THREE.Color(furColor);
      pawDorsalMat.roughness = 0.76;
      pawDorsalMat.transmission = 0.05;
      pawDorsalMat.sheen = 0.56;
      pawDorsalMat.sheenColor = coat.clone().lerp(new THREE.Color(0xffffff), 0.55);
    }
    const padMat = mat.clone();
    padMat.color.set(strain.pawPadColor ?? strain.pawColor ?? '#e9b8ae');
    padMat.color.multiplyScalar(strain.padDarken ?? 0.92);
    padMat.transmission = 0.08; padMat.thickness = 0.12;
    padMat.roughness = 0.50; padMat.clearcoat = 0.08; padMat.clearcoatRoughness = 0.40;
    padMat.normalMap = padNormalMap;
    padMat.normalScale = new THREE.Vector2(0.22, 0.22);

    const pawScale = isFore ? (PROP.forePawSize) : (PROP.hindPawSize);
    // Sculpted carpus/tarsus root bridge seamlessly joining wrist/low-hock into the
    // dorsal paw without an abrupt step or protruding heel balloon.
    const rootLength = isFore ? 0.082 : 0.24;
    const rootBridgeGeo = new THREE.CapsuleGeometry(isFore ? 0.038 : 0.048, rootLength, 10, 14);
    deform(rootBridgeGeo, (rv) => {
      const yN = THREE.MathUtils.clamp((rv.y + rootLength * 0.5) / Math.max(0.01, rootLength), 0, 1);
      // Taper gently into the wrist/calcaneus and flare smoothly toward the dorsal arch.
      const widthTaper = isFore ? (0.86 + yN * 0.26) : (0.76 + yN * 0.38);
      rv.x *= widthTaper;
      rv.z *= isFore ? (0.70 - yN * 0.08) : (0.66 - yN * 0.06);
    });
    const rootBridge = new THREE.Mesh(rootBridgeGeo, pawDorsalMat);
    rootBridge.name = isFore ? 'forepaw-root-bridge' : 'hindpaw-root-bridge';
    rootBridge.rotation.x = Math.PI / 2 - (isFore ? 0.06 : 0.03);
    rootBridge.position.set(0, isFore ? -0.014 : -0.022, isFore ? -0.026 : -0.118);
    rootBridge.castShadow = true;
    paw.add(rootBridge);

    if (isFore) {
      // Sculpted murine dorsal manus with integrated metacarpal ray contours
      // and soft interdigital slope down toward the knuckle web.
      const dorsalHandGeo = new THREE.SphereGeometry(0.098, 22, 20);
      deform(dorsalHandGeo, (hv) => {
        const zN = THREE.MathUtils.clamp((hv.z / 0.098 + 1) * 0.5, 0, 1);
        const xNorm = THREE.MathUtils.clamp(hv.x / 0.098, -1, 1);
        const isDorsal = hv.y > 0;
        // Subtle 4-ray metacarpal relief on the dorsum
        const rayRidge = isDorsal
          ? Math.pow(Math.max(0, Math.cos(xNorm * Math.PI * 3.5)), 2.0) * 0.0045 * zN
          : 0;
        hv.x *= (0.70 + zN * 0.20) * pawScale;
        hv.y *= (isDorsal ? 0.34 : 0.30) * (1.0 - zN * 0.16);
        hv.z *= 1.12 * pawScale;
        hv.y += rayRidge - zN * zN * 0.020;
      });
      const dorsalHand = new THREE.Mesh(dorsalHandGeo, pawDorsalMat);
      dorsalHand.name = 'forepaw-dorsal-hand';
      dorsalHand.position.set(0, -0.024, 0.048);
      dorsalHand.castShadow = true;
      paw.add(dorsalHand);

      // True medial vestigial pollex (D1 thumb nub) mirrored cleanly per side
      const pollex = new THREE.Mesh(new THREE.SphereGeometry(0.0155, 10, 10), pawFleshMat);
      pollex.name = 'forepaw-medial-pollex';
      pollex.scale.set(0.84, 0.66, 1.14);
      pollex.position.set(-sideSign * 0.062 * pawScale, -0.052, 0.028);
      paw.add(pollex);

      // Central palmar cushion + 5 true murine palmar pads (thenar, hypothenar, 3 interdigital)
      const palm = new THREE.Mesh(new THREE.SphereGeometry(0.068, 16, 16), padMat);
      palm.scale.set(0.78 * pawScale, 0.26, 0.96 * pawScale);
      palm.position.set(0, -0.064, 0.054);
      paw.add(palm);

      const carpalPad = new THREE.Mesh(new THREE.SphereGeometry(0.028, 12, 12), padMat);
      carpalPad.scale.set(0.84, 0.36, 0.96);
      carpalPad.position.set(sideSign * 0.024 * pawScale, -0.062, -0.014);
      paw.add(carpalPad);

      const thenarPad = new THREE.Mesh(new THREE.SphereGeometry(0.025, 12, 12), padMat);
      thenarPad.scale.set(0.82, 0.36, 0.94);
      thenarPad.position.set(-sideSign * 0.026 * pawScale, -0.062, -0.008);
      paw.add(thenarPad);

      for (let p = 0; p < 3; p++) {
        const iPad = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 10), padMat);
        iPad.scale.set(0.84 * pawScale, 0.40, 0.92 * pawScale);
        iPad.position.set((p - 1) * 0.032 * pawScale, -0.070, 0.102 * pawScale);
        paw.add(iPad);
      }
    } else {
      // Sleek elongated plantigrade murine pes: slender calcaneal heel tapering
      // smoothly across an arched metatarsus with integrated 5-ray dorsal relief.
      const dorsalFootGeo = new THREE.SphereGeometry(0.118, 24, 22);
      deform(dorsalFootGeo, (fv) => {
        const zN = THREE.MathUtils.clamp((fv.z / 0.118 + 1) * 0.5, 0, 1);
        const xNorm = THREE.MathUtils.clamp(fv.x / 0.118, -1, 1);
        const isDorsal = fv.y > 0;
        const widthProfile = 0.52 + Math.pow(zN, 0.85) * 0.30;
        const rayRidge = isDorsal
          ? Math.pow(Math.max(0, Math.cos(xNorm * Math.PI * 4.5)), 2.0) * 0.0052
            * THREE.MathUtils.smoothstep(zN, 0.20, 0.90)
          : 0;
        fv.x *= widthProfile * pawScale;
        fv.y *= (isDorsal ? 0.30 : 0.28) + (1.0 - zN) * 0.07;
        fv.z *= 1.66 * pawScale;
        // Natural plantar arch + gentle downward taper toward the distal metatarsal heads
        fv.y += rayRidge + Math.sin(zN * Math.PI) * 0.008 - zN * 0.020;
      });
      const dorsalFoot = new THREE.Mesh(dorsalFootGeo, pawDorsalMat);
      dorsalFoot.name = 'hindpaw-dorsal-foot';
      dorsalFoot.position.set(0, -0.024, 0.064);
      dorsalFoot.castShadow = true;
      paw.add(dorsalFoot);

      // Elongated plantar sole + 6 true murine plantar pads (calcaneal heel,
      // 2 asymmetric metatarsal pads, and 4 interdigital cushions) kissing the floor.
      const sole = new THREE.Mesh(new THREE.SphereGeometry(0.098, 18, 18), padMat);
      sole.scale.set(0.66 * pawScale, 0.22, 1.54 * pawScale);
      sole.position.set(0, -0.066, 0.064);
      paw.add(sole);

      const heelPad = new THREE.Mesh(new THREE.SphereGeometry(0.042, 12, 12), padMat);
      heelPad.scale.set(0.76, 0.34, 1.04);
      heelPad.position.set(0, -0.064, -0.140 * pawScale);
      paw.add(heelPad);

      const midPad = new THREE.Mesh(new THREE.SphereGeometry(0.036, 12, 12), padMat);
      midPad.scale.set(0.74, 0.36, 1.05);
      midPad.position.set(-sideSign * 0.024 * pawScale, -0.070, 0.056 * pawScale);
      paw.add(midPad);

      const latMetPad = new THREE.Mesh(new THREE.SphereGeometry(0.034, 12, 12), padMat);
      latMetPad.scale.set(0.74, 0.36, 1.00);
      latMetPad.position.set(sideSign * 0.026 * pawScale, -0.070, 0.098 * pawScale);
      paw.add(latMetPad);

      for (let p = 0; p < 4; p++) {
        const u = p - 1.5;
        const iPad = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 10), padMat);
        iPad.scale.set(0.80 * pawScale, 0.40, 0.92 * pawScale);
        iPad.position.set(
          sideSign * u * 0.028 * pawScale,
          -0.072,
          (0.168 - Math.abs(u) * 0.012) * pawScale
        );
        paw.add(iPad);
      }
    }

    // Fine silky pale dorsal paw micro-hairs tapering cleanly before the knuckles.
    paw.add(createPawDorsalMicroFur(isFore, pawScale, strain, furColor, density, sideSign));

    const pawSpreadMul = strain.pawSpreadMul ?? 1.0;
    const baseSpread = isFore ? 0.034 : 0.031;
    const spread = baseSpread * pawSpreadMul * pawScale;

    const clawLocalMat = clawMat.clone();
    clawLocalMat.color.set(strain.clawColor ?? 0xf0e7dc);

    // Anatomically handed murine digit formulas (indexed medial -> lateral):
    // Forepaw (manus): D2..D5 (D3 longest, D5 shortest & splayed) + medial pollex above.
    // Hindpaw (pes): D1 (proximal medial hallux), D2..D5 (D3 longest, D5 offset).
    const digitSpecs = isFore
      ? [
        { xMul: -1.42, zBack: 0.018, lenMul: 0.86, yaw: -0.22, radMul: 0.94 },
        { xMul: -0.45, zBack: 0.000, lenMul: 1.04, yaw: -0.05, radMul: 1.00 },
        { xMul: 0.52, zBack: 0.004, lenMul: 0.98, yaw: 0.08, radMul: 0.98 },
        { xMul: 1.44, zBack: 0.028, lenMul: 0.74, yaw: 0.26, radMul: 0.88 },
      ]
      : [
        { xMul: -1.88, zBack: 0.068, lenMul: 0.60, yaw: -0.34, radMul: 0.90 },
        { xMul: -0.94, zBack: 0.014, lenMul: 0.94, yaw: -0.11, radMul: 0.96 },
        { xMul: 0.00, zBack: 0.000, lenMul: 1.05, yaw: 0.00, radMul: 1.00 },
        { xMul: 0.94, zBack: 0.012, lenMul: 0.96, yaw: 0.11, radMul: 0.96 },
        { xMul: 1.84, zBack: 0.038, lenMul: 0.72, yaw: 0.28, radMul: 0.88 },
      ];

    const toes = [];
    for (let i = 0; i < digitSpecs.length; i++) {
      const spec = digitSpecs[i];
      const toe = new THREE.Group();
      toe.name = `${isFore ? 'fore' : 'hind'}-digit-${i + 1}`;
      const toeLen = (isFore ? 0.140 : 0.194)
        * (strain.toeLenMul ?? 1.0) * spec.lenMul * pawScale;
      const toeRadius = (isFore ? 0.0118 : 0.0132) * spec.radMul;

      // Multi-phalanx sculpted digit: slender proximal/middle phalanges with subtle
      // PIP/DIP interphalangeal joint swells and a graceful downward arch onto the pad.
      const phalanxGeo = new THREE.CapsuleGeometry(toeRadius, toeLen, 10, 14);
      deform(phalanxGeo, (pv) => {
        const yN = THREE.MathUtils.clamp((pv.y + toeLen * 0.5) / Math.max(0.01, toeLen), 0, 1);
        const pipSwell = Math.exp(-Math.pow((yN - 0.40) / 0.11, 2.0)) * 0.09;
        const dipSwell = Math.exp(-Math.pow((yN - 0.74) / 0.10, 2.0)) * 0.07;
        const taper = (1.06 - yN * 0.34) * (1.0 + pipSwell + dipSwell);
        pv.x *= taper;
        pv.z *= taper * 0.90;
        // After rotation.x = +PI/2, local +Z maps to toe -Y (toward the floor):
        // gently arch over the proximal/middle phalanx and curve the distal tip onto the floor.
        pv.z += -Math.sin(yN * Math.PI * 0.88) * 0.008 + Math.pow(yN, 2.1) * 0.024;
      });
      const phalanx = new THREE.Mesh(phalanxGeo, pawFleshMat);
      phalanx.position.z = toeLen * 0.46;
      phalanx.rotation.x = Math.PI / 2;
      phalanx.castShadow = true;
      toe.add(phalanx);

      // Subtle dorsal metapodial-phalangeal knuckle web blending toe into dorsal paw
      const knuckle = new THREE.Mesh(
        new THREE.SphereGeometry(toeRadius * 1.18, 10, 10),
        pawDorsalMat
      );
      knuckle.scale.set(1.06, 0.70, 1.15);
      knuckle.position.set(0, -0.002, 0.006);
      toe.add(knuckle);

      // Rounded fleshy apical digital bulb curving smoothly to kiss the floor
      const toeTip = new THREE.Mesh(new THREE.SphereGeometry(toeRadius * 0.96, 10, 10), pawFleshMat);
      toeTip.scale.set(1.02, 0.82, 1.08);
      toeTip.position.set(0, -0.022, toeLen + toeRadius * 0.24);
      toe.add(toeTip);

      // Ventral apical digital friction pad beneath the fingertip
      const toePad = new THREE.Mesh(new THREE.SphereGeometry(toeRadius * 1.10, 10, 10), padMat);
      toePad.scale.set(0.94, 0.50, 1.06);
      toePad.position.set(0, -0.028, toeLen * 0.84);
      toe.add(toePad);

      // Tapered, laterally compressed, downward-curved keratin falcula claw
      // emerging cleanly from the dorsal unguicular sheath.
      const clawLength = (isFore ? 0.032 : 0.042) * (0.88 + spec.lenMul * 0.14);
      const clawRadius = (isFore ? 0.0044 : 0.0052) * spec.radMul;
      const clawGeo = new THREE.CapsuleGeometry(clawRadius, clawLength, 8, 10);
      clawGeo.translate(0, clawLength * 0.5, 0);
      deform(clawGeo, (cv) => {
        const yN = THREE.MathUtils.clamp(
          (cv.y + clawRadius) / (clawLength + clawRadius * 2),
          0,
          1
        );
        // Laterally compressed keratin blade tapering cleanly to a fine crescent point
        cv.x *= (0.76 - yN * 0.62);
        cv.z *= (1.06 - yN * 0.72);
        // Downward crescent falcula curve (+Z maps to -Y after rotation.x = +PI/2)
        cv.z += Math.pow(yN, 1.85) * (isFore ? 0.015 : 0.018);
      });
      const claw = new THREE.Mesh(clawGeo, clawLocalMat);
      claw.rotation.x = Math.PI / 2 + (isFore ? 0.22 : 0.19);
      claw.position.set(0, -0.020, toeLen + toeRadius * 0.16);
      claw.castShadow = true;
      toe.add(claw);

      const xPos = sideSign * spec.xMul * spread;
      const zPos = ((isFore ? 0.106 : 0.178) - spec.zBack) * pawScale;
      const yPos = (isFore ? -0.036 : -0.038) + spec.zBack * 0.06;
      const baseRotY = sideSign * spec.yaw;
      const baseRotX = isFore ? 0.11 + spec.zBack * 0.35 : 0.09 + spec.zBack * 0.25;
      toe.position.set(xPos, yPos, zPos);
      toe.rotation.y = baseRotY;
      toe.rotation.x = baseRotX;
      paw.add(toe);
      toes.push({
        group: toe,
        baseRotX,
        baseRotY,
        spreadSign: sideSign * Math.sign(spec.xMul || (i - (digitSpecs.length - 1) * 0.5)),
        isOuter: i === 0 || i === digitSpecs.length - 1,
      });
    }
    paw._toes = toes;

    // Invisible sole landmark used by the stance-phase IK solver. Keeping it
    // inside the paw hierarchy lets the solver preserve every visible joint.
    const groundContact = new THREE.Object3D();
    groundContact.name = isFore ? 'forepaw-ground-contact' : 'hindpaw-ground-contact';
    groundContact.position.set(0, isFore ? -0.064 : -0.068, isFore ? 0.24 : 0.36);
    paw.add(groundContact);
    paw._groundContact = groundContact;
    if (isFore) {
      const groomingContact = new THREE.Object3D();
      groomingContact.name = 'forepaw-palmar-contact';
      groomingContact.position.set(0, -0.056, 0.15);
      paw.add(groomingContact);
      paw._groomingContact = groomingContact;
    }
    return paw;
  }

  // ═══════════════════════════════════════════════════════════════
  // 7. LIMB CHAIN BUILDERS
  // ═══════════════════════════════════════════════════════════════
  function createForelimbChain(surfaceMat, skinMat, strain, density, furColor, side, legLengthScale = 1.0, legThicknessScale = 1.0) {
    const sign = side === 'L' ? 1 : -1;
    const chain = new THREE.Group();
    const fl = PROP.foreLimbLength * legLengthScale;
    const thickness = THREE.MathUtils.clamp(legThicknessScale, 0.65, 1.5);
    // Tucked shoulder anchor: proximal joints remain smoothly inside the chest coat.
    chain.position.set(sign * 0.38 * PROP.shoulderWidth / 0.90, -0.11, PROP.bodyLength * 0.70);

    // Plush scapular + pectoral-brachial soft-tissue mantle enveloping the proximal
    // humerus & elbow smoothly into the anterior chest coat (zero L-bracket notch).
    const shoulderMantle = new THREE.Mesh(new THREE.SphereGeometry(0.168, 16, 16), surfaceMat);
    shoulderMantle.position.set(-sign * 0.10, 0.02, -0.10);
    shoulderMantle.scale.set(1.08, 0.98, 1.28);
    shoulderMantle.castShadow = false;
    chain.add(shoulderMantle);
    const pectoralFold = new THREE.Mesh(new THREE.SphereGeometry(0.152 * thickness, 16, 16), surfaceMat);
    pectoralFold.name = `${side}-pectoral-brachial-fold`;
    pectoralFold.position.set(-sign * 0.058, -0.11, -0.045);
    pectoralFold.scale.set(1.08, 1.32, 1.24);
    pectoralFold.castShadow = false;
    chain.add(pectoralFold);
    const scapularBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.104 * thickness, 0.24, 10, 12),
      surfaceMat
    );
    scapularBridge.name = `${side}-scapular-soft-tissue-bridge`;
    scapularBridge.position.set(-sign * 0.068, 0.03, -0.11);
    scapularBridge.rotation.x = Math.PI / 2;
    scapularBridge.castShadow = false;
    chain.add(scapularBridge);

    // Adult murine radius/ulna is slightly longer than the humerus. The
    // remaining scale belongs to the carpus and paw, represented below.
    const upperArm = new THREE.Group();
    upperArm.name = `${side}-upper-arm`;
    const uaLen = 0.42 * fl;
    const upperArmProfile = yN => {
      const topTuck = 1.0 - THREE.MathUtils.smoothstep(yN, 0.76, 1.0) * 0.14;
      const triceps = Math.exp(-Math.pow((yN - 0.56) / 0.28, 2.0)) * 0.18;
      const t = (0.84 + yN * 0.28 + triceps) * topTuck;
      return {
        rx: 0.128 * thickness * t * 0.92,
        rz: 0.128 * thickness * t * 1.08,
        offsetZ: -triceps * 0.035,
      };
    };
    const upperArmGeo = new THREE.CapsuleGeometry(0.128 * thickness, uaLen, 10, 14);
    deform(upperArmGeo, (uv) => {
      const yN = THREE.MathUtils.clamp((uv.y + uaLen * 0.5) / Math.max(0.01, uaLen), 0, 1);
      const prof = upperArmProfile(yN);
      uv.x *= prof.rx / (0.128 * thickness);
      uv.z *= prof.rz / (0.128 * thickness);
      uv.z += prof.offsetZ;
    });
    const upperArmMesh = new THREE.Mesh(upperArmGeo, surfaceMat);
    upperArmMesh.position.y = -uaLen * 0.5;
    upperArmMesh.castShadow = true;
    upperArm.add(upperArmMesh);
    upperArm.rotation.x = FORELIMB_NEUTRAL.shoulderX;
    const shoulderJoint = new THREE.Mesh(new THREE.SphereGeometry(0.092 * thickness, 12, 12), surfaceMat);
    shoulderJoint.scale.set(0.88, 0.86, 1.02);
    shoulderJoint.castShadow = false;
    upperArm.add(shoulderJoint);
    chain.add(upperArm);

    // Forearm (radius/ulna) — muscular proximal flexor/brachioradialis belly
    // tapering organically into a delicate slender carpus (never a straight peg).
    const forearm = new THREE.Group();
    forearm.name = `${side}-forearm`;
    const faLen = 0.45 * fl;
    forearm.position.set(0, -uaLen - 0.01, -0.01);
    forearm.rotation.x = FORELIMB_NEUTRAL.elbowX;
    const elbowBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.088 * thickness, 0.11, 10, 12),
      surfaceMat
    );
    elbowBridge.position.set(-sign * 0.010, -uaLen * 0.90, 0.012);
    elbowBridge.rotation.x = 0.18;
    elbowBridge.castShadow = false;
    upperArm.add(elbowBridge);
    const elbowJoint = new THREE.Mesh(new THREE.SphereGeometry(0.074 * thickness, 12, 12), surfaceMat);
    elbowJoint.scale.set(0.92, 0.96, 1.02);
    elbowJoint.castShadow = false;
    forearm.add(elbowJoint);

    const forearmProfile = yN => {
      const flexorBelly = Math.exp(-Math.pow((yN - 0.70) / 0.30, 2.0)) * 0.28;
      const t = 0.52 + Math.pow(yN, 0.85) * 0.56 + flexorBelly;
      return {
        rx: 0.092 * thickness * t * 0.94,
        rz: 0.092 * thickness * t * 1.08,
        offsetZ: -flexorBelly * 0.024 + Math.sin(yN * Math.PI) * 0.010,
      };
    };
    const forearmGeo = new THREE.CapsuleGeometry(0.092 * thickness, faLen, 10, 14);
    deform(forearmGeo, (fv) => {
      const yN = THREE.MathUtils.clamp((fv.y + faLen * 0.5) / Math.max(0.01, faLen), 0, 1);
      const prof = forearmProfile(yN);
      fv.x *= prof.rx / (0.092 * thickness);
      fv.z *= prof.rz / (0.092 * thickness);
      fv.z += prof.offsetZ;
    });
    const forearmMesh = new THREE.Mesh(forearmGeo, surfaceMat);
    forearmMesh.position.y = -faLen * 0.5;
    forearmMesh.castShadow = true;
    forearm.add(forearmMesh);
    upperArm.add(forearm);

    // Wrist joint — smoothly contoured, coat-to-skin feathered carpus cuff
    // aligned collinearly from the distal forearm right onto the dorsal manus.
    const wrist = new THREE.Group();
    wrist.name = `${side}-wrist`;
    wrist.position.set(0, -faLen - 0.018, 0.03);
    wrist.rotation.x = FORELIMB_NEUTRAL.wristX;
    const wristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.044 * thickness, 10, 10), surfaceMat);
    wristJoint.scale.set(0.94, 0.90, 0.96);
    wristJoint.castShadow = false;
    wrist.add(wristJoint);

    const wristCuffMat = skinMat.clone();
    const furColFore = new THREE.Color(furColor);
    const dorsalColFore = computePawDorsalColor(strain, furColor);
    wristCuffMat.color.setHex(0xffffff);
    wristCuffMat.vertexColors = true;
    wristCuffMat.map = pawSkinMaps.dorsalAlbedo;
    wristCuffMat.roughness = 0.80;
    wristCuffMat.transmission = 0.03;
    wristCuffMat.sheen = 0.54;
    wristCuffMat.sheenColor = furColFore.clone().lerp(new THREE.Color(0xffffff), 0.45);

    const wristCuffLen = 0.132;
    const wristCuffProfile = yN => {
      // Proximal (yN=1) hugs the slender distal forearm; distal (yN=0) flattens
      // smoothly right onto forepaw-root-bridge so there is zero ledge or step.
      const rx = (0.044 + (1.0 - yN) * 0.010) * thickness;
      const rz = (0.042 - (1.0 - yN) * 0.010) * thickness;
      return { rx, rz, offsetZ: 0 };
    };
    const wristCuffGeo = new THREE.CapsuleGeometry(0.044 * thickness, wristCuffLen, 10, 14);
    deform(wristCuffGeo, (wv) => {
      const yN = THREE.MathUtils.clamp((wv.y + wristCuffLen * 0.5) / wristCuffLen, 0, 1);
      const prof = wristCuffProfile(yN);
      wv.x *= prof.rx / (0.044 * thickness);
      wv.z *= prof.rz / (0.044 * thickness);
    });
    applyCuffVertexGradient(wristCuffGeo, wristCuffLen, -1, furColFore, dorsalColFore);
    const wristCuff = new THREE.Mesh(wristCuffGeo, density <= 0 ? surfaceMat : wristCuffMat);
    wristCuff.position.set(sign * 0.019, -0.074, 0.028);
    wristCuff.rotation.x = -0.35;
    wristCuff.rotation.z = sign * 0.25;
    wristCuff.castShadow = false;
    wrist.add(wristCuff);
    forearm.add(wrist);

    // Handed forepaw — sits level at bottom of wrist with subtle natural toe-out yaw
    const paw = createPaw(skinMat, true, strain, furColor, side, density);
    paw.position.set(sign * 0.035, -0.13, 0.07);
    paw.scale.setScalar(0.96 * Math.sqrt(thickness));
    paw.rotation.x = FORELIMB_NEUTRAL.pawX;
    paw.rotation.y = sign * 0.14;
    wrist.add(paw);

    // Proximal-to-distal combed limb pelage sleeves hugging the exact muscle contours
    // from shoulder & pectoral fold down into a feathered carpal cuff.
    const coatSphere = (mesh, radius, count, hairLength) => {
      const coat = generateFurLayer(v => {
        v.multiplyScalar(radius);
      }, Math.floor(count * density), furColor, hairLength, false, true);
      mesh.add(coat);
    };
    const seedOffset = sign > 0 ? 11 : 29;
    upperArmMesh.add(createLimbFurSleeve({
      profileFn: upperArmProfile,
      span: uaLen,
      count: Math.floor(2600 * density),
      furColor,
      pawSkinColor: strain.pawColor ?? '#e6b0a7',
      hairLength: 0.064,
      distalDirY: -1,
      seed: 0x31F00100 + seedOffset,
    }));
    forearmMesh.add(createLimbFurSleeve({
      profileFn: forearmProfile,
      span: faLen,
      count: Math.floor(2400 * density),
      furColor,
      pawSkinColor: strain.pawColor ?? '#e6b0a7',
      hairLength: 0.046,
      distalDirY: -1,
      seed: 0x31F00200 + seedOffset,
    }));
    wristCuff.add(createLimbFurSleeve({
      profileFn: wristCuffProfile,
      span: wristCuffLen,
      count: Math.floor(960 * density),
      furColor,
      pawSkinColor: strain.pawColor ?? '#e6b0a7',
      hairLength: 0.030,
      distalDirY: -1,
      featherDistal: true,
      seed: 0x31F00300 + seedOffset,
      isUndercoat: true,
    }));
    coatSphere(shoulderJoint, 0.092 * thickness, 360, 0.064);
    coatSphere(elbowBridge, 0.088 * thickness, 520, 0.056);
    coatSphere(elbowJoint, 0.074 * thickness, 340, 0.050);
    coatSphere(wristJoint, 0.044 * thickness, 240, 0.034);
    coatSphere(shoulderMantle, 0.168, 1500, 0.068);
    coatSphere(pectoralFold, 0.152 * thickness, 1600, 0.068);

    chain._upperArm = upperArm;
    chain._forearm = forearm;
    chain._wrist = wrist;
    chain._paw = paw;
    const shoulderToElbow = forearm.position.length();
    const elbowToWrist = wrist.position.length();
    const wristToPaw = paw.position.length();
    const pawToContact = paw._groundContact.position.length() * paw.scale.x;
    chain._segmentLengths = Object.freeze({
      humerus: shoulderToElbow,
      radiusUlna: elbowToWrist,
      carpus: wristToPaw,
      manus: pawToContact,
    });
    chain._maximumReach = shoulderToElbow + elbowToWrist
      + wristToPaw + pawToContact;
    return chain;
  }

  function createHindlimbChain(surfaceMat, skinMat, strain, density, furColor, side, legLengthScale = 1.0, legThicknessScale = 1.0) {
    const sign = side === 'L' ? 1 : -1;
    const chain = new THREE.Group();
    const hl = PROP.hindLimbLength * legLengthScale;
    const thickness = THREE.MathUtils.clamp(legThicknessScale, 0.65, 1.5);
    const seedOffset = sign > 0 ? 17 : 37;
    // Hip anchor — at widest haunch point
    chain.position.set(sign * 0.62 * PROP.haunchWidth / 1.04, -0.04, -PROP.bodyLength * 0.36);

    // Hip joint (internal blend mass)
    const hipJoint = new THREE.Mesh(new THREE.SphereGeometry(0.18, 14, 14), surfaceMat);
    hipJoint.scale.set(1.08, 0.88, 1.18);
    hipJoint.castShadow = false;
    chain.add(hipJoint);
    // Soft blend into lower back (wider for smoother pelvis-hip transition)
    const hipBlend = new THREE.Mesh(new THREE.CapsuleGeometry(0.13, 0.26, 10, 10), surfaceMat);
    hipBlend.position.set(0, 0.07, -0.03);
    hipBlend.rotation.x = 0.34;
    hipBlend.scale.set(1.14, 0.92, 1.28);
    hipBlend.castShadow = false;
    chain.add(hipBlend);
    // Inboard pelvis bridge to blend hip into torso and avoid detached look.
    const pelvisBridge = new THREE.Mesh(new THREE.CapsuleGeometry(0.11, 0.30, 10, 10), surfaceMat);
    pelvisBridge.position.set(-sign * 0.14, 0.08, -0.02);
    pelvisBridge.rotation.z = Math.PI / 2;
    pelvisBridge.rotation.x = 0.16;
    pelvisBridge.scale.set(1.0, 0.88, 1.15);
    pelvisBridge.castShadow = false;
    chain.add(pelvisBridge);
    const hipSkirt = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 12), surfaceMat);
    hipSkirt.position.set(-sign * 0.10, -0.01, 0.02);
    hipSkirt.scale.set(1.20, 0.82, 1.08);
    hipSkirt.castShadow = false;
    chain.add(hipSkirt);
    // Extra local fur on hip blend to hide seam against back.
    const hipFurCount = Math.floor(5200 * density);
    if (hipFurCount > 0) {
      const hfGeo = new THREE.ConeGeometry(0.006, 0.09, 3);
      hfGeo.translate(0, 0.045, 0);
      const hfMat = enhanceFurMaterial(
        new THREE.MeshStandardMaterial({ color: new THREE.Color(furColor), roughness: 0.86 }),
        furColor,
        true
      );
      const hfInst = new THREE.InstancedMesh(hfGeo, hfMat, hipFurCount);
      const hDummy = new THREE.Object3D();
      const hNrm = new THREE.Vector3();
      for (let i = 0; i < hipFurCount; i++) {
        const theta = random() * Math.PI * 2;
        const y = (random() - 0.5) * 1.6;
        hNrm.set(Math.cos(theta) * 0.95, y * 0.48 - 0.12, Math.sin(theta) * 1.1).normalize();
        hDummy.position.set(hNrm.x * 0.16, hNrm.y * 0.18, hNrm.z * 0.17);
        hDummy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), hNrm);
        const s = 0.72 + random() * 0.55;
        hDummy.scale.set(s, s, s);
        hDummy.updateMatrix();
        hfInst.setMatrixAt(i, hDummy.matrix);
      }
      hipBlend.add(hfInst);
    }

    // Thigh (femur) — sculpted muscular haunch tucked smoothly into the lower flank
    const thigh = new THREE.Group();
    thigh.name = `${side}-thigh`;
    thigh.position.set(-sign * 0.035, -0.10, 0.02);
    chain.add(thigh);
    const thighMuscle = new THREE.Mesh(thighGeo, surfaceMat);
    thighMuscle.scale.set(0.76 * thickness, 0.68 * hl, 0.88 * thickness);
    thighMuscle.castShadow = false;
    thigh.add(thighMuscle);
    thighMuscle.add(generateFurLayer(deformThigh, Math.floor(8800 * density), furColor, 0.115, false));

    // Shank (crus / tibia-fibula) — muscular gastrocnemius calf belly tapering
    // smoothly down into a sleek furred Achilles/tarsal hock.
    const shank = new THREE.Group();
    shank.name = `${side}-shank`;
    shank.position.set(0.02 * sign, -0.32 * hl, 0.18);
    thigh.add(shank);

    const kneeJoint = new THREE.Mesh(new THREE.SphereGeometry(0.112 * thickness, 12, 12), surfaceMat);
    kneeJoint.position.set(0, 0.045, 0.005);
    kneeJoint.scale.set(0.96, 1.04, 1.12);
    kneeJoint.castShadow = false;
    shank.add(kneeJoint);
    kneeJoint.add(generateFurLayer(v => {
      v.multiplyScalar(0.110 * thickness);
    }, Math.floor(520 * density), furColor, 0.068, false, true));

    const shankLen = 0.46 * hl;
    const shankProfile = yN => {
      // Prominent proximal gastrocnemius calf belly tapering into sturdy furred Achilles
      const gastroc = Math.exp(-Math.pow((yN - 0.66) / 0.28, 2.0)) * 0.38;
      const t = 0.68 + yN * 0.44 + gastroc;
      return {
        rx: 0.092 * thickness * t * 0.90,
        rz: 0.092 * thickness * t * 1.12,
        offsetZ: -gastroc * 0.038,
      };
    };
    const shankGeo = new THREE.CapsuleGeometry(0.092 * thickness, shankLen, 10, 14);
    deform(shankGeo, (sv) => {
      const yN = THREE.MathUtils.clamp((sv.y + shankLen * 0.5) / Math.max(0.01, shankLen), 0, 1);
      const prof = shankProfile(yN);
      sv.x *= prof.rx / (0.092 * thickness);
      sv.z *= prof.rz / (0.092 * thickness);
      sv.z += prof.offsetZ;
    });
    const shankMesh = new THREE.Mesh(shankGeo, surfaceMat);
    shankMesh.rotation.x = -0.16;
    shankMesh.position.set(0, -shankLen * 0.5, 0.09);
    shankMesh.castShadow = true;
    shank.add(shankMesh);
    shankMesh.add(createLimbFurSleeve({
      profileFn: shankProfile,
      span: shankLen,
      count: Math.floor(2400 * density),
      furColor,
      pawSkinColor: strain.pawColor ?? '#e6b0a7',
      hairLength: 0.054,
      distalDirY: -1,
      seed: 0x31B00100 + seedOffset,
    }));

    // Ankle / low furred calcaneal hock (tarsocrural joint descending smoothly
    // to a grounded heel right behind the horizontal semi-plantigrade pes).
    const ankle = new THREE.Group();
    ankle.name = `${side}-ankle`;
    ankle.position.set(0, -shankLen - 0.04, 0.18);
    shank.add(ankle);

    const ankleJoint = new THREE.Mesh(new THREE.SphereGeometry(0.056 * thickness, 10, 10), surfaceMat);
    ankleJoint.castShadow = false;
    ankle.add(ankleJoint);
    ankleJoint.add(generateFurLayer(v => {
      v.multiplyScalar(0.056 * thickness);
    }, Math.floor(480 * density), furColor, 0.044, false, true));

    // Furred distal crus / Achilles bridge descending from the lower shank down
    // to the low grounded calcaneal hock (replaces the exposed 45-deg bare stick).
    const crusLen = 0.168;
    const crusProfile = yN => {
      const t = 0.88 + yN * 0.18;
      return {
        rx: 0.054 * thickness * t * 0.92,
        rz: 0.054 * thickness * t * 1.06,
        offsetZ: -(1.0 - yN) * 0.008,
      };
    };
    const distalCrusGeo = new THREE.CapsuleGeometry(0.054 * thickness, crusLen, 10, 12);
    deform(distalCrusGeo, (cv) => {
      const yN = THREE.MathUtils.clamp((cv.y + crusLen * 0.5) / crusLen, 0, 1);
      const prof = crusProfile(yN);
      cv.x *= prof.rx / (0.054 * thickness);
      cv.z *= prof.rz / (0.054 * thickness);
      cv.z += prof.offsetZ;
    });
    const distalCrus = new THREE.Mesh(distalCrusGeo, surfaceMat);
    distalCrus.name = `${side}-distal-crus-achilles`;
    distalCrus.position.set(sign * 0.018, -0.094, 0.028);
    distalCrus.rotation.x = -0.29;
    distalCrus.rotation.z = sign * 0.19;
    distalCrus.castShadow = true;
    ankle.add(distalCrus);
    distalCrus.add(createLimbFurSleeve({
      profileFn: crusProfile,
      span: crusLen,
      count: Math.floor(1250 * density),
      furColor,
      pawSkinColor: strain.pawColor ?? '#e6b0a7',
      hairLength: 0.040,
      distalDirY: -1,
      coverCaps: true,
      seed: 0x31B00200 + seedOffset,
    }));

    // Low, tucked, fully furred calcaneus (hock/heel) right behind the horizontal hindfoot
    const heel = new THREE.Mesh(new THREE.CapsuleGeometry(0.050 * thickness, 0.078, 8, 10), surfaceMat);
    heel.rotation.x = Math.PI / 2 - 0.18;
    heel.position.set(sign * 0.036, -0.192, 0.038);
    heel.scale.set(0.92, 1.0, 0.84);
    heel.castShadow = false;
    ankle.add(heel);
    heel.add(createLimbFurSleeve({
      profileFn: () => ({ rx: 0.048 * thickness, rz: 0.044 * thickness, offsetZ: 0 }),
      span: 0.072,
      count: Math.floor(880 * density),
      furColor,
      pawSkinColor: strain.pawColor ?? '#e6b0a7',
      hairLength: 0.036,
      distalDirY: 1,
      coverCaps: true,
      seed: 0x31B00300 + seedOffset,
      isUndercoat: true,
    }));

    // Sleek, nearly horizontal tarsometatarsal bridge running cleanly from the low
    // calcaneal hock forward onto the dorsal pes with seamless vertex-color & fur feathering.
    const metaMat = skinMat.clone();
    const furColObj = new THREE.Color(furColor);
    const dorsalColHind = computePawDorsalColor(strain, furColor);
    metaMat.color.setHex(0xffffff);
    metaMat.vertexColors = true;
    metaMat.map = pawSkinMaps.dorsalAlbedo;
    metaMat.normalMap = pawSkinMaps.dorsalNormal;
    metaMat.normalScale = new THREE.Vector2(0.18, 0.18);
    metaMat.roughness = 0.78;
    metaMat.transmission = 0.04;
    metaMat.sheen = 0.56;
    metaMat.sheenColor = furColObj.clone().lerp(new THREE.Color(0xffffff), 0.45);
    const metaLen = 0.21 * PROP.hindPawSize;
    const metaProfile = yN => {
      // Proximal heel (yN=0) -> distal metatarsus (yN=1): low dorsoventrally compressed arch
      return {
        rx: (0.046 + yN * 0.015) * thickness,
        rz: (0.038 - yN * 0.007) * thickness,
        offsetZ: 0,
      };
    };
    const metaGeo = new THREE.CapsuleGeometry(0.048 * thickness, metaLen, 10, 14);
    deform(metaGeo, (mv) => {
      const yN = THREE.MathUtils.clamp((mv.y + metaLen * 0.5) / Math.max(0.01, metaLen), 0, 1);
      const prof = metaProfile(yN);
      mv.x *= prof.rx / (0.048 * thickness);
      mv.z *= prof.rz / (0.048 * thickness);
    });
    applyCuffVertexGradient(metaGeo, metaLen, 1, furColObj, dorsalColHind);
    const metaMesh = new THREE.Mesh(metaGeo, density <= 0 ? surfaceMat : metaMat);
    metaMesh.rotation.order = 'YXZ';
    metaMesh.rotation.x = Math.PI / 2 + 0.085;
    metaMesh.rotation.y = sign * 0.05;
    metaMesh.position.set(sign * 0.041, -0.206, 0.158);
    metaMesh.castShadow = true;
    ankle.add(metaMesh);
    // Feathered proximal-to-distal tarsal fur combed sleekly forward from the
    // calcaneal hock onto the dorsal foot (zero harsh speckles or bare stick seam).
    metaMesh.add(createLimbFurSleeve({
      profileFn: metaProfile,
      span: metaLen,
      count: Math.floor(1150 * density),
      furColor,
      pawSkinColor: strain.pawColor ?? '#e6b0a7',
      hairLength: 0.032,
      distalDirY: 1,
      featherDistal: true,
      seed: 0x31B00400 + seedOffset,
      isUndercoat: true,
    }));

    // Handed hindpaw — planted level on ground with subtle natural outward splay
    const paw = createPaw(skinMat, false, strain, furColor, side, density);
    paw.position.set(sign * 0.045, -0.20, 0.31);
    paw.rotation.x = 0.02;
    paw.rotation.y = sign * 0.18;
    paw.scale.setScalar(0.92 * Math.sqrt(thickness));
    ankle.add(paw);

    // Baseline crouched posture so underside and side views read organic.
    chain.rotation.x = -0.24;
    shank.rotation.x = 0.56;
    ankle.rotation.x = -0.34;

    chain._thigh = thigh;
    chain._shank = shank;
    chain._ankle = ankle;
    chain._paw = paw;
    return chain;
  }

  // ═══════════════════════════════════════════════════════════════
  // 8. ANATOMICAL VIBRISSAE + FUR
  // ═══════════════════════════════════════════════════════════════
  function createTaperedWhiskerGeometry(
    curve,
    tubularSegments = 14,
    rootRadius = 0.004,
    tipRadius = 0.0003,
    radialSegments = 5
  ) {
    const frames = curve.computeFrenetFrames(tubularSegments, false);
    const positions = [];
    const colors = [];
    const uvs = [];
    const indices = [];
    const point = new THREE.Vector3();
    const offset = new THREE.Vector3();

    for (let segment = 0; segment <= tubularSegments; segment++) {
      const t = segment / tubularSegments;
      curve.getPointAt(t, point);
      const radius = THREE.MathUtils.lerp(rootRadius, tipRadius, Math.pow(t, 0.72));
      const shade = THREE.MathUtils.lerp(0.68, 1.0, Math.pow(t, 0.55));
      for (let radial = 0; radial < radialSegments; radial++) {
        const angle = (radial / radialSegments) * Math.PI * 2;
        offset.copy(frames.normals[segment]).multiplyScalar(Math.cos(angle));
        offset.addScaledVector(frames.binormals[segment], Math.sin(angle));
        offset.multiplyScalar(radius);
        positions.push(point.x + offset.x, point.y + offset.y, point.z + offset.z);
        colors.push(shade, shade, shade);
        uvs.push(t, radial / radialSegments);
      }
    }

    for (let segment = 0; segment < tubularSegments; segment++) {
      for (let radial = 0; radial < radialSegments; radial++) {
        const nextRadial = (radial + 1) % radialSegments;
        const a = segment * radialSegments + radial;
        const b = segment * radialSegments + nextRadial;
        const c = (segment + 1) * radialSegments + radial;
        const d = (segment + 1) * radialSegments + nextRadial;
        indices.push(a, c, b, b, c, d);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    return geometry;
  }

  function createAnatomicalWhiskers(colorHex, isLeft, opacity = 0.52) {
    const group = new THREE.Group();
    const sign = isLeft ? 1 : -1;
    const material = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
      alphaToCoverage: true,
      vertexColors: true,
    });
    // Five mystacial macrovibrissae rows (A-E) + rostral downward microvibrissae.
    const rows = [
      { y: 0.102, z: -0.032, count: 4, minLength: 0.66, maxLength: 1.04, forward: 0.22, lift: 0.285, droop: -0.015 },
      { y: 0.058, z: -0.015, count: 5, minLength: 0.78, maxLength: 1.22, forward: 0.30, lift: 0.155, droop: -0.030 },
      { y: 0.010, z: 0.000, count: 5, minLength: 0.88, maxLength: 1.38, forward: 0.38, lift: 0.020, droop: -0.055 },
      { y: -0.042, z: 0.016, count: 5, minLength: 0.74, maxLength: 1.16, forward: 0.34, lift: -0.115, droop: -0.075 },
      { y: -0.092, z: 0.030, count: 4, minLength: 0.56, maxLength: 0.92, forward: 0.26, lift: -0.225, droop: -0.095 },
    ];
    rows.forEach((row, rowIndex) => {
      for (let i = 0; i < row.count; i++) {
        // i=0 is rostral (shorter, more forward), higher i is caudal (longer, wider sweep)
        const caudalFrac = (i + 1) / (row.count + 1);
        const rostralFrac = 1.0 - caudalFrac;
        const length = THREE.MathUtils.lerp(row.minLength, row.maxLength, caudalFrac)
          * (0.94 + random() * 0.12);
        const rootJitter = random() - 0.5;
        const rootY = row.y + (rostralFrac - 0.5) * 0.016 + rootJitter * 0.012;
        const rootZ = row.z + (rostralFrac - 0.5) * 0.11 + (random() - 0.5) * 0.010;
        const forward = row.forward * (0.72 + rostralFrac * 0.48)
          + (random() - 0.5) * 0.030;
        const droop = row.droop * (0.75 + caudalFrac * 0.35)
          + (random() - 0.5) * 0.016;
        const start = new THREE.Vector3(
          sign * (0.010 + caudalFrac * 0.022 + rowIndex * 0.002),
          rootY,
          rootZ
        );
        // Natural murine vibrissa arc: emerges slightly forward/outward from the
        // mystacial follicle, then sweeps gracefully backward along the cheek.
        const backSweep = length * (0.24 + caudalFrac * 0.28);
        const curve = new THREE.CubicBezierCurve3(
          start,
          new THREE.Vector3(
            sign * length * 0.32,
            rootY + row.lift * 0.18,
            rootZ + forward * 0.42
          ),
          new THREE.Vector3(
            sign * length * 0.72,
            rootY + row.lift * 0.60 + droop * 0.34 + rootJitter * 0.022,
            rootZ + forward * 0.48 - backSweep * 0.48
          ),
          new THREE.Vector3(
            sign * length * 0.96,
            rootY + row.lift * 0.86 + droop * 1.15,
            rootZ + forward * 0.25 - backSweep
          )
        );
        const rootRadius = 0.0034 + caudalFrac * 0.0012;
        const whisker = new THREE.Mesh(
          createTaperedWhiskerGeometry(curve, 14, rootRadius, 0.00038, 5),
          material
        );
        whisker.name = `vibrissa-${rowIndex}-${i}`;
        whisker.castShadow = false;
        whisker.frustumCulled = false;
        group.add(whisker);
      }
    });
    // Short rostral downward microvibrissae framing upper lip for tactile floor sensing
    for (let m = 0; m < 5; m++) {
      const mf = m / 4;
      const mLen = 0.18 + mf * 0.14 + random() * 0.04;
      const start = new THREE.Vector3(sign * (0.006 + mf * 0.014), -0.095 - mf * 0.015, 0.065 + (1 - mf) * 0.045);
      const curve = new THREE.CubicBezierCurve3(
        start,
        new THREE.Vector3(start.x + sign * mLen * 0.25, start.y - mLen * 0.28, start.z + mLen * 0.35),
        new THREE.Vector3(start.x + sign * mLen * 0.58, start.y - mLen * 0.68, start.z + mLen * 0.52),
        new THREE.Vector3(start.x + sign * mLen * 0.78, start.y - mLen * 0.95, start.z + mLen * 0.58)
      );
      const micro = new THREE.Mesh(createTaperedWhiskerGeometry(curve, 8, 0.0019, 0.00028, 4), material);
      micro.name = `microvibrissa-${m}`;
      micro.frustumCulled = false;
      group.add(micro);
    }
    return group;
  }

  function createSupraorbitalWhiskers(colorHex, isLeft, opacity = 0.44) {
    const group = new THREE.Group();
    const sign = isLeft ? 1 : -1;
    const material = new THREE.MeshBasicMaterial({
      color: colorHex,
      transparent: true,
      opacity,
      depthWrite: false,
      side: THREE.DoubleSide,
      alphaToCoverage: true,
      vertexColors: true,
    });
    // Two supraorbital brow tactile hairs + one postorbital genal hair
    const specs = [
      { root: [sign * 0.38, 0.31, 0.46], tip: [sign * 0.68, 0.66, 0.36], r: 0.0024 },
      { root: [sign * 0.40, 0.29, 0.40], tip: [sign * 0.76, 0.58, 0.22], r: 0.0022 },
      { root: [sign * 0.47, 0.02, 0.24], tip: [sign * 0.88, 0.06, 0.02], r: 0.0023 },
    ];
    specs.forEach((s, idx) => {
      const p0 = new THREE.Vector3(...s.root);
      const p3 = new THREE.Vector3(...s.tip);
      const p1 = p0.clone().lerp(p3, 0.34).add(new THREE.Vector3(sign * 0.04, 0.05, 0.02));
      const p2 = p0.clone().lerp(p3, 0.72).add(new THREE.Vector3(sign * 0.03, 0.03, -0.01));
      const mesh = new THREE.Mesh(
        createTaperedWhiskerGeometry(new THREE.CubicBezierCurve3(p0, p1, p2, p3), 10, s.r, 0.0003, 4),
        material
      );
      mesh.name = `supraorbital-vibrissa-${idx}`;
      mesh.frustumCulled = false;
      group.add(mesh);
    });
    return group;
  }

  function generateFurLayer(deformFn, count, colorHex, length, isHead, isUndercoat = false, clumpiness = 0.3) {
    if (count <= 0) return new THREE.Group();
    const baseFurColor = new THREE.Color(colorHex);
    const furHsl = {};
    baseFurColor.getHSL(furHsl);
    const isPaleCoat = furHsl.l > 0.65;
    const isDarkCoat = furHsl.l < 0.18;
    const clumpScale = THREE.MathUtils.clamp((clumpiness ?? 0.3) / 0.3, 0.0, 3.5);
    // Fuller, softly tapered hair blades reduce sub-pixel speckle aliasing and
    // give a plush, cohesive murine velvet pelt across both dark and pale coats.
    const radius = isUndercoat
      ? (isPaleCoat ? 0.0040 : 0.0036)
      : (isPaleCoat ? 0.0029 : 0.0026);
    // Only the visible guard layer needs the extra bend segment. Undercoat
    // remains a single tapered segment with lightweight matte shading.
    const geo = new THREE.ConeGeometry(radius, length, 3, isUndercoat ? 1 : 2, true);
    geo.translate(0, length / 2, 0);
    geo.rotateX(Math.PI / 2);
    const vertices = geo.attributes.position;
    const colors = new Float32Array(vertices.count * 3);
    for (let i = 0; i < vertices.count; i++) {
      const progress = THREE.MathUtils.clamp(vertices.getZ(i) / length, 0, 1);
      vertices.setX(i, vertices.getX(i) + length * 0.08 * progress * progress);
      const shade = isPaleCoat
        ? (0.94 + progress * 0.09)
        : (isDarkCoat ? (0.80 + progress * 0.22) : (0.80 + progress * 0.25));
      colors.set([shade, shade, shade], i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const mat = isUndercoat
      ? new THREE.MeshStandardMaterial({
        vertexColors: true,
        color: 0xffffff,
        roughness: isPaleCoat ? 0.92 : 0.94,
      })
      : new THREE.MeshPhysicalMaterial({
        vertexColors: true,
        color: 0xffffff, roughness: isPaleCoat ? 0.84 : 0.80,
        clearcoat: 0.0,
        sheen: isPaleCoat ? 0.70 : 0.52,
        sheenColor: isDarkCoat
          ? new THREE.Color(0x3a3438)
          : (isPaleCoat ? new THREE.Color(0xfdfaf5) : new THREE.Color(colorHex)),
        sheenRoughness: isPaleCoat ? 0.68 : 0.58,
      });
    enhanceFurMaterial(mat, colorHex, isUndercoat);
    const inst = new THREE.InstancedMesh(geo, mat, count);
    inst.castShadow = false;
    // Pale fur cones receiving hard directional shadows create salt-and-pepper stipple;
    // the underlying pale skin mesh carries the clean soft macro-shadows instead.
    inst.receiveShadow = !isUndercoat && !isPaleCoat;
    inst.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
    const strandColor = new THREE.Color();
    const dummy = new THREE.Object3D();
    const pos = new THREE.Vector3(), normal = new THREE.Vector3();
    const source = new THREE.Vector3(), lookTarget = new THREE.Vector3();
    const sampleSurface = createDeformedSurfaceSampler(THREE, deformFn);
    for (let i = 0; i < count; i++) {
      const u = random(), vv = random();
      const theta = u * 2 * Math.PI, phi = Math.acos(2 * vv - 1);
      source.set(Math.sin(phi) * Math.cos(theta), Math.sin(phi) * Math.sin(theta), Math.cos(phi));
      const preZ = source.z;
      const preY = source.y;
      sampleSurface(source, pos, normal);
      pos.addScaledVector(normal, isUndercoat ? 0.0018 : 0.0032);
      let flowDir;
      if (isHead) {
        // Face: continuous smooth caudal sweep from nasal bridge over forehead
        // and cheeks (smoothstep avoids any vertical combing seam down the cheek).
        const snoutFlow = THREE.MathUtils.smoothstep(preZ, 0.02, 0.58);
        flowDir = new THREE.Vector3(
          normal.x * THREE.MathUtils.lerp(0.28, 0.22, snoutFlow),
          THREE.MathUtils.lerp(normal.y * 0.16, -0.06, snoutFlow),
          THREE.MathUtils.lerp(-1.0, -0.82, snoutFlow)
        ).normalize();
      } else {
        // Body & neck: smooth region-blended caudal fur flow with subtle clumping.
        const clump = Math.sin(source.x * 19.0 + preZ * 15.0) * (0.08 * clumpScale);
        const tailW = 1.0 - THREE.MathUtils.smoothstep(preZ, -0.82, -0.55);
        const haunchW = (1.0 - THREE.MathUtils.smoothstep(preZ, -0.45, -0.10)) * (1.0 - tailW);
        const shoulderW = THREE.MathUtils.smoothstep(preZ, 0.25, 0.72);
        const dorsalW = THREE.MathUtils.smoothstep(preY, 0.08, 0.52);
        const ventralW = 1.0 - THREE.MathUtils.smoothstep(preY, -0.52, -0.08);
        const midX = THREE.MathUtils.lerp(0.45, 0.18, dorsalW) * (1.0 - ventralW) + 0.62 * ventralW;
        const midY = THREE.MathUtils.lerp(normal.y * 0.24 - 0.12, 0.04, dorsalW) * (1.0 - ventralW)
          + (normal.y * 0.40 - 0.16) * ventralW;
        const fx = normal.x * (
          midX * (1.0 - tailW - haunchW - shoulderW)
          + 0.15 * tailW + 0.52 * haunchW + 0.30 * shoulderW
        ) + clump * (1.0 - tailW - shoulderW * 0.7) * (1.0 - dorsalW * 0.5);
        const fy = midY * (1.0 - tailW - haunchW - shoulderW)
          + (-0.08) * tailW + (normal.y * 0.18 - 0.12) * haunchW + (normal.y * 0.14 - 0.10) * shoulderW;
        const fz = -0.94 * (1.0 - ventralW * 0.22);
        flowDir = new THREE.Vector3(fx, fy, fz).normalize();
      }
      if (isUndercoat) { flowDir.lerp(new THREE.Vector3(0, 0, -1), 0.3); flowDir.normalize(); }
      // Project combing onto the skin before adding lift. A radial sphere
      // normal buries strands at the shoulder and points them into the rump.
      flowDir.addScaledVector(normal, -flowDir.dot(normal));
      if (flowDir.lengthSq() < 1e-8) flowDir.set(1, 0, 0).addScaledVector(normal, -normal.x);
      flowDir.normalize().addScaledVector(normal, isUndercoat ? 0.22 : 0.10).normalize();
      dummy.position.copy(pos);
      dummy.lookAt(lookTarget.copy(pos).add(flowDir));
      let scale = 0.70 + random() * 0.40;
      if (isHead) {
        // Continuous rostral fur-length gradient across cheek and muzzle (zero hard seam)
        scale *= 1.0 - THREE.MathUtils.smoothstep(preZ, -0.05, 0.88) * 0.56;
        // Periorbital & perinasal vellus zone: prevent fur cones from poking
        // through the glossy eyeballs or moist rhinarium pad.
        const anchors = getHeadFeatureAnchors();
        const eyeRadius = PROP.eyeSize * 1.65;
        const eyeDist = Math.min(
          pos.distanceTo(anchors.eyeL.position),
          pos.distanceTo(anchors.eyeR.position)
        ) / eyeRadius;
        if (eyeDist < 1.18) {
          scale *= THREE.MathUtils.smoothstep(eyeDist, 0.80, 1.18);
        }
        const noseDist = pos.distanceTo(anchors.nose.position) / 0.13;
        if (noseDist < 1.15) {
          scale *= THREE.MathUtils.smoothstep(noseDist, 0.78, 1.15);
        }
        // Perioral bare cleft zone so fur cones never poke through the mouth or incisors.
        const mouthDist = pos.distanceTo(anchors.mouth.position) / 0.088;
        if (mouthDist < 1.15) {
          scale *= THREE.MathUtils.smoothstep(mouthDist, 0.62, 1.15);
        }
      } else if (preZ < -0.6) {
        scale *= 1.06;
      }
      if (isUndercoat) scale = Math.min(scale, 0.74 + random() * 0.20);
      else random(); // preserve identical RNG consumption cadence across layers
      const jitter = isUndercoat ? 0.065 : 0.038;
      dummy.rotation.x += (random() - 0.5) * jitter;
      dummy.rotation.y += (random() - 0.5) * jitter;
      dummy.scale.set(scale, scale, scale); dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
      // Subtle murine countershading: slightly softer/brighter ventral pelage
      const ventralLift = !isHead && preY < -0.25 ? (-preY - 0.25) * 0.08 : 0.0;
      const baseShade = isPaleCoat
        ? (isUndercoat ? 0.97 : 0.98)
        : (isUndercoat ? 0.90 : 0.93);
      const shadeSpread = isPaleCoat ? 0.03 : (isUndercoat ? 0.07 : 0.08);
      const shade = baseShade + ventralLift + random() * shadeSpread;
      strandColor.setRGB(shade, shade, shade);
      inst.instanceColor.setXYZ(i, strandColor.r, strandColor.g, strandColor.b);
    }
    return inst;
  }

  // Fur detail-level multipliers
  const FUR_DETAIL = { High: 1.0, Med: 0.5, Low: 0.15, Off: 0.0 };

  function generateMultiLayerFur(deformFn, density, furColor, isHead, furOptions = {}) {
    const group = new THREE.Group();
    if (density <= 0) return group;
    const guardRatio = THREE.MathUtils.clamp(furOptions.furGuardRatio ?? 0.5, 0.0, 1.0);
    const lengthScale = THREE.MathUtils.clamp(furOptions.furLength ?? 1.0, 0.3, 2.5);
    const clumpiness = THREE.MathUtils.clamp(furOptions.furClumpiness ?? 0.3, 0.0, 1.0);
    const underCount = Math.floor((isHead ? 16000 : 38000) * density * (1.0 - guardRatio) * 2.0);
    const guardCount = Math.floor((isHead ? 9500 : 22000) * density * guardRatio * 2.0);
    if (underCount > 0) {
      group.add(
        generateFurLayer(
          deformFn,
          underCount,
          furColor,
          (isHead ? 0.033 : 0.052) * lengthScale,
          isHead,
          true,
          clumpiness
        )
      );
    }
    if (guardCount > 0) {
      group.add(
        generateFurLayer(
          deformFn,
          guardCount,
          furColor,
          (isHead ? 0.058 : 0.112) * lengthScale,
          isHead,
          false,
          clumpiness
        )
      );
    }
    return group;
  }

  function createEarBaseFur(furColor, density, sideSign) {
    const count = Math.floor(760 * density);
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(sideSign > 0 ? 0x3B9E17A5 : 0x3B9E17A6);
    const geo = new THREE.ConeGeometry(0.0048, 0.082, 3);
    geo.translate(0, 0.041, 0);
    const mat = enhanceFurMaterial(
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(furColor),
        roughness: 0.88,
      }),
      furColor,
      true
    );
    const inst = new THREE.InstancedMesh(geo, mat, count);
    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const nrm = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      // Wrap smoothly around the entire narrow conchal stem (medial, anterior,
      // posterior, and lateral) so cranial pelage flows seamlessly onto the pinna base.
      const ang = rng() * Math.PI * 2;
      const frac = Math.pow(rng(), 0.78);
      const isBack = Math.sin(ang) < 0.25;
      const rad = THREE.MathUtils.lerp(0.24, 0.14, frac) * (0.86 + rng() * 0.26);
      const x = Math.cos(ang) * rad * 1.15;
      const z = Math.sin(ang) * rad * 0.78 - (isBack ? 0.055 : 0.015);
      const y = -0.10 + frac * (isBack ? 0.34 : 0.22) + (rng() - 0.5) * 0.04;
      dummy.position.set(x, y, z);
      nrm.set(
        Math.cos(ang) * 0.42 + sideSign * 0.06,
        0.56 + frac * 0.32,
        Math.sin(ang) * 0.38 - (isBack ? 0.18 : -0.12)
      ).normalize();
      dummy.quaternion.setFromUnitVectors(up, nrm);
      const s = (0.62 + rng() * 0.48) * (1.0 - frac * 0.28);
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    return inst;
  }

  function createEarRimFuzz(furColor, skinColor, density, sideSign = 1) {
    if (density <= 0) return new THREE.Group();
    const count = Math.floor(280 * Math.max(0.35, density));
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(sideSign > 0 ? 0x61D42C11 : 0x61D42C12);
    const geo = new THREE.ConeGeometry(0.0022, 0.028, 3);
    geo.translate(0, 0.014, 0);
    const furHslRim = {};
    new THREE.Color(furColor).getHSL(furHslRim);
    const fuzzColor = furHslRim.l < 0.2
      ? new THREE.Color(furColor).lerp(new THREE.Color(skinColor), 0.24)
      : new THREE.Color(skinColor).lerp(new THREE.Color(furColor), 0.42).offsetHSL(0, -0.04, 0.10);
    const mat = new THREE.MeshBasicMaterial({
      color: fuzzColor,
      transparent: true,
      opacity: furHslRim.l < 0.2 ? 0.24 : 0.30,
      depthWrite: false,
    });
    const inst = new THREE.InstancedMesh(geo, mat, count);
    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const dir = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      const t = i / count + (rng() - 0.5) * 0.008;
      const angle = -Math.PI * 0.16 + THREE.MathUtils.clamp(t, 0, 1) * Math.PI * 1.32;
      const pt = shapePinnaPoint(Math.cos(angle) * 0.99, Math.sin(angle) * 0.99, 1.0, -0.002, sideSign);
      dummy.position.copy(pt);
      dir.set(Math.cos(angle) * 0.85, Math.sin(angle) * 0.85, -0.18 + (rng() - 0.5) * 0.18).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const s = 0.60 + rng() * 0.50;
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    return inst;
  }

  function createEarBackFuzz(furColor, density, sideSign = 1) {
    const count = Math.floor(400 * Math.max(0.0, density));
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(sideSign > 0 ? 0x51A7C23D : 0x51A7C23E);
    const geo = new THREE.ConeGeometry(0.0031, 0.042, 3);
    geo.translate(0, 0.021, 0);
    const mat = enhanceFurMaterial(
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(furColor),
        roughness: 0.88,
      }),
      furColor,
      true
    );
    const inst = new THREE.InstancedMesh(geo, mat, count);
    inst.name = 'posterior-pinna-vellus';
    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const dir = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      // Bias short velvety posterior pelage more densely toward the lower/mid pinna.
      const r = Math.sqrt(rng()) * 0.90;
      const a = rng() * Math.PI * 2;
      const x = Math.cos(a) * r;
      const y = Math.sin(a) * r * 0.92 - 0.08 * (1.0 - r);
      const pt = shapePinnaPoint(x, y, 1.0, -0.018, sideSign);
      dummy.position.copy(pt);
      dir.set(x * 0.34, 0.74 + y * 0.22, -0.46).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const upperThin = 1.0 - THREE.MathUtils.smoothstep(y, 0.10, 0.92) * 0.42;
      const s = (0.50 + rng() * 0.46) * upperThin;
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    return inst;
  }

  function buildMouseEar({
    sideSign,
    eSize,
    strain,
    skinColor,
    furColor,
    density,
    outerMat,
    innerMat,
    asymmetryTilt,
    asymmetryYaw,
    heightOffset,
  }) {
    const ear = new THREE.Group();
    ear.name = sideSign > 0 ? 'left-ear-root-pivot' : 'right-ear-root-pivot';
    const pinna = new THREE.Group();
    pinna.name = sideSign > 0 ? 'left-pinna' : 'right-pinna';
    pinna.position.set(sideSign * 0.012, 0.76, 0.0);
    ear.add(pinna);

    const outerHsl = {};
    new THREE.Color(furColor).getHSL(outerHsl);
    // Posterior pinna: velvety short pelage over thin translucent cartilage.
    // Keep dark C57BL/6 pinna backs deep matte charcoal-sable (prevents bright grey rim glare).
    const backMat = outerMat.clone();
    backMat.color = new THREE.Color(furColor).lerp(
      outerMat.color,
      outerHsl.l < 0.2 ? 0.14 : 0.74
    );
    if (outerHsl.l >= 0.2) {
      backMat.color.lerp(innerMat.color, 0.22);
    }
    backMat.map = null;
    backMat.normalMap = null;
    backMat.transmission = outerHsl.l < 0.2
      ? 0.04
      : Math.min(0.28, outerMat.transmission ?? 0.22);
    backMat.roughness = outerHsl.l < 0.2 ? 0.92 : 0.76;
    backMat.envMapIntensity = outerHsl.l < 0.2 ? 0.18 : 0.42;
    backMat.clearcoat = 0.0;
    backMat.sheen = outerHsl.l < 0.2 ? 0.14 : 0.54;
    backMat.sheenColor = outerHsl.l < 0.2
      ? new THREE.Color(0x221d1f)
      : new THREE.Color(furColor).lerp(new THREE.Color(0xffffff), 0.18);
    backMat.sheenRoughness = outerHsl.l < 0.2 ? 0.82 : 0.62;

    const pinnaShellGeo = sideSign > 0 ? earGeo : earGeoR;
    const pinnaInnerGeo = sideSign > 0 ? earInnerGeo : earInnerGeoR;
    const pinnaRimGeo = sideSign > 0 ? earRimGeo : earRimGeoR;

    // Group 0 (concave inner surface) carries innerMat directly; Group 1
    // (convex back + rounded rolled margin) carries backMat with zero gap.
    const outer = new THREE.Mesh(pinnaShellGeo, [innerMat, backMat]);
    outer.name = 'pinna-shell';
    outer.castShadow = true;
    outer.receiveShadow = true;
    pinna.add(outer);

    const rimMat = backMat.clone();
    rimMat.color = rimMat.color.clone().lerp(innerMat.color, outerHsl.l < 0.2 ? 0.28 : 0.48);
    rimMat.transmission = 0.0;
    rimMat.envMapIntensity = outerHsl.l < 0.2 ? 0.18 : 0.40;
    rimMat.roughness = Math.min(0.86, (outerMat.roughness ?? 0.48) + 0.18);
    rimMat.polygonOffset = true;
    rimMat.polygonOffsetFactor = -1;
    rimMat.polygonOffsetUnits = -1;
    const rim = new THREE.Mesh(pinnaRimGeo, rimMat);
    rim.name = 'open-auricular-rim';
    rim.castShadow = false;
    pinna.add(rim);

    const rimFuzz = createEarRimFuzz(furColor, skinColor, density, sideSign);
    rimFuzz.name = 'auricular-marginal-vellus';
    pinna.add(rimFuzz);

    const backFuzz = createEarBackFuzz(furColor, density, sideSign);
    pinna.add(backFuzz);

    // Subtle non-transmissive basal conchal blush over the unified pinna shell.
    const conchalBlushMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(innerMat.color).offsetHSL(0.0, 0.04, -0.05),
      roughness: 0.72,
      transparent: true,
      opacity: 0.18,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const inner = new THREE.Mesh(pinnaInnerGeo, conchalBlushMat);
    inner.name = 'concave-pinna-surface';
    inner.position.set(0, -0.10, 0.001);
    inner.castShadow = false;
    pinna.add(inner);

    // Compact buried pedicle seated entirely inside the temporal skull & base fur collar.
    const pedicle = new THREE.Mesh(
      new THREE.SphereGeometry(0.15, 16, 12),
      backMat
    );
    pedicle.name = 'buried-pinna-pedicle';
    pedicle.position.set(sideSign * 0.008, -0.03, -0.045);
    pedicle.scale.set(0.68, 0.44, 0.46);
    pedicle.castShadow = false;
    ear.add(pedicle);

    const foldMat = innerMat.clone();
    foldMat.color = foldMat.color.clone().offsetHSL(0.0, 0.012, -0.055);
    foldMat.transmission = Math.min(0.16, innerMat.transmission ?? 0.20);
    foldMat.roughness = 0.70;
    // Concha, tragus, and auditory meatus shadow sit deeply inside the mirrored
    // basal funnel so they read as organic cartilage contours.
    const cupDepthAt = (px, py) => shapePinnaPoint(px, py - 0.76, 1.0, 0.002, sideSign).z;
    const concha = new THREE.Mesh(
      new THREE.SphereGeometry(0.12, 16, 12),
      foldMat
    );
    concha.name = 'basal-conchal-fold';
    concha.position.set(sideSign * 0.015, 0.27, cupDepthAt(sideSign * 0.015, 0.27) - 0.006);
    concha.scale.set(0.52, 0.34, 0.09);
    concha.rotation.z = sideSign * 0.08;
    concha.castShadow = false;
    ear.add(concha);

    const tragus = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.015, 0.055, 8, 10),
      foldMat
    );
    tragus.name = 'auricular-tragus-fold';
    tragus.position.set(-sideSign * 0.052, 0.22, cupDepthAt(-sideSign * 0.052, 0.22) + 0.004);
    tragus.rotation.z = sideSign * 0.38;
    tragus.scale.set(0.86, 0.86, 0.38);
    tragus.castShadow = false;
    ear.add(tragus);

    const canalMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(innerMat.color).multiplyScalar(0.36),
      roughness: 0.92,
      transparent: true,
      opacity: 0.48,
      depthWrite: false,
    });
    const canal = new THREE.Mesh(
      new THREE.SphereGeometry(0.050, 14, 12),
      canalMat
    );
    canal.name = 'ear-canal-shadow';
    canal.position.set(-sideSign * 0.012, 0.18, cupDepthAt(-sideSign * 0.012, 0.18) + 0.003);
    canal.scale.set(0.76, 0.48, 0.12);
    ear.add(canal);

    const baseFur = createEarBaseFur(furColor, density, sideSign);
    baseFur.name = 'pinna-root-fur-collar';
    baseFur.position.set(0, -0.02, -0.012);
    ear.add(baseFur);

    const strainScale = strain.earScaleMul ?? 1.0;
    const s = eSize * 0.69 * strainScale;
    // Pinnae seat on the posterolateral temporal skull behind the eyes, cupping
    // gently anterolaterally while sweeping organically back along the head contour.
    const openingYaw = 0.34 + (strain.earYaw ?? PROP.earYaw) * 0.24;
    const root = findHeadSurfacePoint({
      x: sideSign * 0.46,
      y: 0.35,
      z: PROP.earSetBack + 0.02,
    });
    ear.position.copy(root.position).addScaledVector(root.normal, -0.095);
    ear.position.y += heightOffset;
    ear.rotation.set(
      -0.14 + asymmetryTilt,
      sideSign * openingYaw + asymmetryYaw,
      sideSign * -0.40
    );
    ear.scale.set(s * 0.90, s * 1.00, s * 1.06);
    ear._pinna = pinna;
    ear._pedicle = pedicle;
    ear._canal = canal;
    return ear;
  }

  function createTailBaseFur(furColor, density) {
    const count = Math.floor(640 * Math.max(0.0, density));
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(0x2F84A91B);
    const geo = new THREE.ConeGeometry(0.0042, 0.095, 3);
    geo.translate(0, 0.0475, 0);
    const mat = enhanceFurMaterial(
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(furColor),
        roughness: 0.86,
      }),
      furColor,
      false
    );
    const inst = new THREE.InstancedMesh(geo, mat, count);
    inst.name = 'sacral-tail-root-fur';
    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const dir = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      // Exponential caudal falloff: plush sacral collar emerging right at the rump
      // feathering organically into sparse sleek hairs down the visible proximal shaft.
      const frac = Math.pow(rng(), 1.35);
      const ang = rng() * Math.PI * 2;
      const ovalX = THREE.MathUtils.lerp(1.05, 1.0, frac);
      const ovalY = THREE.MathUtils.lerp(0.95, 1.0, frac);
      // Match fuller emerging tailSurfaceRadiusAt(0.04..0.14) so sacral fur collar
      // blends seamlessly from rump pelage down the muscular proximal tail root.
      const baseRad = THREE.MathUtils.lerp(0.148, 0.102, frac) * (0.98 + rng() * 0.05);
      const x = Math.cos(ang) * baseRad * ovalX;
      const y = Math.sin(ang) * baseRad * ovalY;
      const z = -0.10 + frac * 0.84;
      dummy.position.set(x, y, z);
      // Sleek caudal lay hugging the emerging shaft (less flare distally).
      const flare = THREE.MathUtils.lerp(0.28, 0.12, frac);
      dir.set(Math.cos(ang) * flare, Math.sin(ang) * flare, 0.95 + rng() * 0.12).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const scLen = (0.76 + rng() * 0.44) * THREE.MathUtils.lerp(1.18, 0.36, frac);
      const scWid = (0.80 + rng() * 0.32) * THREE.MathUtils.lerp(1.05, 0.56, frac);
      dummy.scale.set(scWid, scLen, scWid);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    return inst;
  }


  // ═══════════════════════════════════════════════════════════════
  // Shared assets survive subject disposal and can be used by later spawns.
  // ═══════════════════════════════════════════════════════════════
  const sharedResources = new Set([
    bodyGeo, headGeo, neckGeo, cervicalCoreGeo, thighGeo,
    earGeo, earGeoR, earInnerGeo, earInnerGeoR, earRimGeo, earRimGeoR,
    fleshMatBase, clawMat, eyeMat, corneaMat, padNormalMap, tailAlbedoMap,
    tailRoughnessMap, tailNormalMap,
    eyeIrisMap, albinoEyeIrisMap, earNormalMap, earRoughnessMap, earAlbedoMap, noseAlbedoMap, noseNormalMap,
    noseRoughnessMap, labFloorAlbedoMap, labFloorRoughnessMap, labFloorNormalMap,
    homeCageBeddingMaps.albedo, homeCageBeddingMaps.normal, homeCageBeddingMaps.roughness,
    pawSkinMaps.digitNormal, pawSkinMaps.dorsalAlbedo, pawSkinMaps.dorsalNormal,
    urineSpotTexture, shadowSmudgeTexture,
  ]);

  function disposeSubject(root) {
    const resources = new Set();
    const registerMaterial = material => {
      if (!material) return;
      resources.add(material);
      for (const value of Object.values(material)) {
        if (value?.isTexture) resources.add(value);
      }
    };
    const ownedMaterials = root?.userData?.ownedMaterials;
    if (Array.isArray(ownedMaterials)) {
      for (const material of ownedMaterials) registerMaterial(material);
    }
    root.traverse(node => {
      if (node.isInstancedMesh) node.dispose();
      if (node.geometry) resources.add(node.geometry);
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        registerMaterial(material);
      }
    });
    for (const resource of resources) {
      if (!sharedResources.has(resource)) resource.dispose();
    }
  }

  function disposeSharedResources() {
    for (const resource of sharedResources) {
      resource?.dispose?.();
    }
  }

  return {
    PROP,
    FORELIMB_NEUTRAL,
    tailAlbedoMap,
    tailRoughnessMap,
    tailNormalMap,
    eyeIrisMap,
    albinoEyeIrisMap,
    earNormalMap,
    earRoughnessMap,
    earAlbedoMap,
    noseAlbedoMap,
    noseNormalMap,
    noseRoughnessMap,
    labFloorAlbedoMap,
    labFloorRoughnessMap,
    labFloorNormalMap,
    homeCageBeddingMaps,
    urineSpotTexture,
    shadowSmudgeTexture,
    fleshMatBase,
    eyeMat,
    corneaMat,
    createCoatMaterial,
    enhanceFurMaterial,
    deform,
    deformBody,
    deformHead,
    findHeadSurfacePoint,
    getHeadFeatureAnchors,
    getCervicalMantleProfile,
    deformCervicalCore,
    bodyGeo,
    headGeo,
    neckGeo,
    cervicalCoreGeo,
    rebuildSharedAnatomyGeometries,
    createForelimbChain,
    createHindlimbChain,
    createAnatomicalWhiskers,
    createSupraorbitalWhiskers,
    FUR_DETAIL,
    generateMultiLayerFur,
    buildMouseEar,
    createTailBaseFur,
    disposeSubject,
    disposeSharedResources,
  };
}
