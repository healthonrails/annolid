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
    shoulderX: 0.38,
    shoulderZ: 0.0,
    elbowX: -0.74,
    wristX: 0.24,
    pawX: 0.12,
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

  function createTailRingRoughnessMap(size = 256) {
    const c = document.createElement('canvas'); c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    const ringRows = 32;
    const rowH = size / ringRows;
    for (let y = 0; y < size; y++) {
      const rowFrac = (y % rowH) / rowH;
      // Imbricate keratinized ring profile: smoother distal scale plate, matte sulcus.
      const sulcus = Math.exp(-Math.pow((rowFrac - 0.90) / 0.12, 2.0));
      const grain = (random() - 0.5) * 5.0;
      const v = Math.floor(THREE.MathUtils.clamp(162 + sulcus * 46 - rowFrac * 10 + grain, 128, 228));
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(0, y, size, 1);
    }
    ctx.globalAlpha = 0.05;
    for (let i = 0; i < 64; i++) {
      const y0 = random() * size;
      const x0 = random() * size;
      const shade = 148 + Math.floor(random() * 34);
      ctx.strokeStyle = `rgb(${shade},${shade},${shade})`;
      ctx.lineWidth = 0.45 + random() * 0.65;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x0 + (random() - 0.5) * 3.0, y0 + 5 + random() * 9);
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1.0, 6.0);
    return tex;
  }

  function createTailNormalMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    const ringRows = 32;
    const cols = 18;
    const rowH = size / ringRows;
    const colW = size / cols;
    // Staggered annular epidermal scales (~192 rings across repeat.y = 6).
    for (let r = 0; r < ringRows; r++) {
      const y0 = r * rowH;
      const stagger = (r % 2) * colW * 0.5;
      // Transverse ring crease (normal tilted along V)
      ctx.fillStyle = 'rgba(128,104,252,0.55)';
      ctx.fillRect(0, y0, size, Math.max(1.0, rowH * 0.18));
      ctx.fillStyle = 'rgba(128,150,252,0.42)';
      ctx.fillRect(0, y0 + rowH * 0.76, size, Math.max(1.0, rowH * 0.20));
      for (let col = -1; col <= cols; col++) {
        const x0 = col * colW + stagger;
        ctx.strokeStyle = 'rgba(114,128,250,0.28)';
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        ctx.moveTo(x0, y0 + rowH * 0.10);
        ctx.lineTo(x0 + (random() - 0.5) * 0.8, y0 + rowH * 0.90);
        ctx.stroke();
      }
    }
    // Sparse short caudal bristle impressions emerging from scale margins
    ctx.lineCap = 'round';
    for (let i = 0; i < 220; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = rowH * (0.55 + random() * 0.55);
      ctx.strokeStyle = 'rgba(138,142,255,0.24)';
      ctx.lineWidth = 0.55;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (random() - 0.5) * 1.4, y + len);
      ctx.stroke();
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1.0, 6.0);
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
    for (let i = 0; i < 7; i++) {
      const a = -Math.PI * 0.54 + (i / 6) * Math.PI * 1.08 + (random() - 0.5) * 0.10;
      const r0 = size * 0.08;
      const r1 = size * 0.36 + random() * size * 0.08;
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0;
      const x1 = cx + Math.cos(a) * r1;
      const y1 = cy + Math.sin(a) * r1;
      drawVein(x0, y0, x1, y1, 1.05 + random() * 0.75, 0.11 + random() * 0.06, 2);
    }
    // Fine capillary arcade near the outer third of the pinna
    for (let i = 0; i < 20; i++) {
      const a = -Math.PI * 0.70 + random() * Math.PI * 1.40;
      const r0 = size * (0.14 + random() * 0.15);
      const r1 = size * (0.30 + random() * 0.15);
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0 * 0.94;
      const x1 = cx + Math.cos(a + (random() - 0.5) * 0.25) * r1;
      const y1 = cy + Math.sin(a + (random() - 0.5) * 0.25) * r1 * 0.96;
      ctx.strokeStyle = `rgba(186,112,110,${0.07 + random() * 0.05})`;
      ctx.lineWidth = 0.45 + random() * 0.55;
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
    const isPaleCoat = hsl.l > 0.70;
    const isAgouti = hsl.s > 0.16 && hsl.l > 0.12 && hsl.l < 0.62;
    const root = isPaleCoat
      ? base.clone().offsetHSL(0.01, 0.03, -0.045)
      : base.clone().offsetHSL(0.0, -0.02, -0.075);
    const subapical = isAgouti
      ? base.clone().offsetHSL(0.025, 0.14, 0.11)
      : base.clone().offsetHSL(0.012, 0.01, isPaleCoat ? 0.018 : 0.045);
    const tip = isAgouti
      ? base.clone().offsetHSL(-0.01, -0.05, -0.10)
      : base.clone().offsetHSL(0.01, 0.0, isPaleCoat ? 0.024 : 0.065);
    // Keep the base tileable across torso UV seams.
    const baseSRGB = base.clone().convertLinearToSRGB();
    ctx.fillStyle = `rgb(${Math.round(baseSRGB.r * 255)},${Math.round(baseSRGB.g * 255)},${Math.round(baseSRGB.b * 255)})`;
    ctx.fillRect(0, 0, size, size);
    ctx.lineCap = 'round';
    for (let i = 0; i < 8600; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 5 + random() * 20;
      const roll = random();
      const tone = roll < 0.34 ? root : (roll < 0.72 ? subapical : tip);
      const fiber = tone.clone().convertLinearToSRGB();
      const alpha = isPaleCoat ? (0.07 + random() * 0.11) : (0.12 + random() * 0.18);
      ctx.strokeStyle = `rgba(${Math.round(fiber.r * 255)},${Math.round(fiber.g * 255)},${Math.round(fiber.b * 255)},${alpha})`;
      ctx.lineWidth = 0.32 + random() * 0.72;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (random() - 0.5) * 2.0, y + len);
      ctx.stroke();
    }
    for (let i = 0; i < 5000; i++) {
      const x = random() * size;
      const y = random() * size;
      const alpha = isPaleCoat
        ? 0.005 + random() * 0.014
        : 0.014 + random() * 0.038;
      const light = random() > 0.48;
      ctx.fillStyle = light
        ? `rgba(255,252,246,${alpha})`
        : `rgba(${isPaleCoat ? 155 : 0},${isPaleCoat ? 148 : 0},${isPaleCoat ? 138 : 0},${alpha})`;
      ctx.fillRect(x, y, 1 + random() * 1.5, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.2, 2.2);
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
    for (let i = 0; i < 5800; i++) {
      const x = random() * size, y = random() * size;
      const span = 3 + random() * 14;
      for (const [offset, red] of [[-0.45, 102], [0.45, 154]]) {
        ctx.strokeStyle = `rgba(${red},128,251,0.30)`;
        ctx.lineWidth = 0.65;
        ctx.beginPath();
        ctx.moveTo(x + offset, y);
        ctx.quadraticCurveTo(x + offset + 0.6, y + span * 0.5, x + offset + 0.2, y + span);
        ctx.stroke();
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.2, 1.8);
    return tex;
  }
  function createCoatRoughnessMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#c4c4c4';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 2400; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 148 + Math.floor(random() * 72);
      ctx.fillStyle = `rgba(${v},${v},${v},0.36)`;
      ctx.fillRect(x, y, 1 + random() * 2, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(2.6, 1.5);
    return tex;
  }
  const padNormalMap = createPadNormalMap();
  const tailRoughnessMap = createTailRingRoughnessMap();
  const tailNormalMap = createTailNormalMap();
  const eyeIrisMap = createEyeIrisMap();
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
    color: 0xe6ddd2, roughness: 0.16, metalness: 0.02,
    clearcoat: 1.0, clearcoatRoughness: 0.08,
    transmission: 0.14, thickness: 0.22,
    ior: 1.54, // keratin IOR
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
      roughness: isHairless ? 0.64 : (isPale ? 0.84 : (isDark ? 0.74 : 0.82)),
      envMapIntensity: isHairless ? 0.36 : (isPale ? 0.42 : (isDark ? 0.62 : 0.40)),
      metalness: 0.0,
      clearcoat: isHairless ? 0.04 : (isDark ? 0.06 : 0.0),
      clearcoatRoughness: isHairless ? 0.56 : 0.62,
      sheen: isHairless ? 0.16 : (isPale ? 0.62 : (isDark ? 0.72 : 0.48)),
      sheenColor: isDark
        ? new THREE.Color(0x5c5760)
        : (isHairless ? skin : fur).clone().offsetHSL(0.0, -0.02, isPale ? 0.06 : 0.10),
      sheenRoughness: isHairless ? 0.62 : (isDark ? 0.46 : 0.58),
      map: createCoatAlbedoMap(isHairless ? skinColor : furColor),
      normalMap: createCoatNormalMap(),
      normalScale: isHairless
        ? new THREE.Vector2(0.05, 0.06)
        : new THREE.Vector2(0.13 + density * 0.09, 0.14 + density * 0.10),
      roughnessMap: createCoatRoughnessMap(),
    });
    if (!isHairless && annolidShaders?.applyFurShader && params?.enableFurShader !== false) {
      const rootCol = fur.clone().offsetHSL(0.0, 0.0, isPale ? -0.025 : -0.025);
      const tipCol = isDark
        ? fur.clone().offsetHSL(0.004, -0.01, 0.14)
        : fur.clone().offsetHSL(0.008, -0.01, isPale ? 0.035 : 0.075);
      annolidShaders.applyFurShader(material, {
        anisotropy: isPale ? 30.0 : (isDark ? 34.0 : 30.0),
        specIntensity: isPale ? 0.048 : (isDark ? 0.095 : 0.058),
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
        strength: Math.min(isPale ? 0.006 : 0.020, (params.microStrength ?? 0.06) * (isPale ? 0.10 : 0.30))
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
      0.0,
      isPale ? -0.005 : -0.01,
      isUndercoat
        ? (isPale ? -0.004 : -0.014)
        : (isPale ? 0.018 : (isDark ? 0.024 : 0.018))
    );
    material.transparent = false;
    material.depthWrite = true;
    material.envMapIntensity = isUndercoat ? (isPale ? 0.34 : 0.22) : (isDark ? 0.64 : 0.48);
    if (material.isMeshPhysicalMaterial) {
      material.roughness = isDark ? 0.68 : (isPale ? 0.82 : 0.78);
      material.sheen = isUndercoat ? 0.32 : (isDark ? 0.85 : 0.72);
      material.sheenColor = isDark
        ? new THREE.Color(0x645f68)
        : base.clone().offsetHSL(0.01, -0.03, isPale ? 0.08 : 0.12);
      material.sheenRoughness = isUndercoat ? 0.82 : (isDark ? 0.46 : 0.56);
    }
    return material;
  }

  // ═══════════════════════════════════════════════════════════════
  // 5. DEFORMATION FUNCTIONS
  // ═══════════════════════════════════════════════════════════════
  function deform(geometry, deformFn) {
    const pos = geometry.attributes.position;
    const v = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      deformFn(v);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
    geometry.computeVertexNormals();
    return geometry;
  }

  const deformBody = (v) => {
    const zNorm = (v.z + 1) / 2;
    // Sleek C2-continuous murine tubular-pear silhouette: full hemispherical U-rounded rump
    // (compensating for polar narrowing of the stretched sphere), smooth mid-trunk, and soft shoulder taper.
    const rumpRoundness = Math.exp(-Math.pow((zNorm - 0.11) / 0.12, 2.0));
    const rearMass = Math.exp(-Math.pow((zNorm - 0.30) / 0.28, 2.0));
    const shoulderSlope = THREE.MathUtils.smoothstep(zNorm, 0.50, 0.96);
    const waistNarrow = Math.exp(-Math.pow((zNorm - 0.56) / 0.20, 2.0));
    const lateral = (
      0.96
      + rumpRoundness * 0.15
      + rearMass * 0.15 * (PROP.haunchWidth / 1.02)
      - shoulderSlope * (0.12 - (PROP.shoulderWidth - 0.88) * 0.22)
      - waistNarrow * (1.0 - PROP.waistTaper) * 0.18
    );
    v.x *= lateral * PROP.bodyGirthX;
    v.y *= (0.96 + rearMass * 0.08 + rumpRoundness * 0.05 - zNorm * 0.10) * PROP.bodyGirthY;
    v.z *= PROP.bodyLength;
    // Smooth elliptical flank curvature so top view shows zero hard shelf.
    const sideBlend = THREE.MathUtils.clamp((Math.abs(v.y) - 0.05) / 0.72, 0, 1);
    v.x *= 0.96 + sideBlend * 0.04;

    // Dorsal thoracolumbar arch: natural murine kyphotic curve peaking over lumbar spine.
    const hump = Math.exp(-Math.pow((zNorm - 0.36) / 0.26, 2.0));
    const rump = Math.exp(-Math.pow((zNorm - 0.12) / 0.13, 2.0));
    const withers = Math.exp(-Math.pow((zNorm - 0.74) / 0.17, 2.0));
    v.y += hump * 0.23 + rump * 0.075 + withers * 0.108 - shoulderSlope * 0.020;

    // Ventrum: tucked inguinal groin and streamlined ribcage underside.
    const yRef = v.y;
    const bellyWidth = Math.exp(-Math.pow((zNorm - 0.46) / 0.28, 2.0));
    const bellyMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.65, 0.16);
    const bellyInfluence = bellyWidth * bellyMask;
    v.x *= 1.0 + bellyInfluence * 0.08;
    const dropMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.22, 0.22);
    v.y -= bellyWidth * dropMask * 0.10;

    const lowerBlend = THREE.MathUtils.smoothstep(0.10 - yRef, 0.0, 0.46);
    const groinTuck = Math.exp(-Math.pow((zNorm - 0.14) / 0.20, 2.0)) * 0.095;
    let chestDrop = THREE.MathUtils.clamp((zNorm - 0.72) / 0.28, 0, 1);
    chestDrop = chestDrop * chestDrop * (3.0 - 2.0 * chestDrop) * 0.045;
    const targetY = v.y * 0.82 + groinTuck - chestDrop;
    v.y = THREE.MathUtils.lerp(v.y, targetY, lowerBlend);

    // Subtle gluteal-haunch and scapular muscular relief (avoids angular diamond bulge).
    if (Math.abs(v.x) > 0.22) {
      const scapBell = Math.exp(-Math.pow((zNorm - 0.75) / 0.09, 2.0));
      v.x *= 1.0 + scapBell * 0.024;
      if (v.y > 0) v.y += scapBell * 0.020;
      const haunchBell = Math.exp(-Math.pow((zNorm - 0.22) / 0.15, 2.0));
      v.x *= 1.0 + haunchBell * 0.034;
      if (v.y < 0) v.y *= 1.0 + haunchBell * 0.028;
    }

    // Keep rump hemispherical right up to the sacral pole (no pointed cone pinch or lower shelf).
    if (zNorm < 0.08) {
      const tailBlend = THREE.MathUtils.smoothstep(zNorm, 0.0, 0.08);
      v.x *= 0.92 + tailBlend * 0.08;
      v.y *= 0.93 + tailBlend * 0.07;
    }

    // Smooth C2 anterior taper inside the broad cervical mantle (eliminates shoulder step).
    if (zNorm > 0.58) {
      const neckT = THREE.MathUtils.smoothstep(zNorm, 0.58, 0.99);
      v.x *= 1.0 - neckT * 0.34;
      v.y *= 1.0 - neckT * (v.y >= 0 ? 0.22 : 0.10);
    }
  };

  const deformHead = (v) => {
    const zNorm = (v.z + 1) / 2;
    const smooth = THREE.MathUtils.smoothstep;
    // Blunt the rostral pole gently into a small rounded muzzle dome that
    // carries the rhinarium without creating a bulbous nose knob or sharp cone.
    const tipZone = smooth(zNorm, 0.82, 1.0);
    const ringRadius = Math.sqrt(Math.max(0, 1 - v.z * v.z));
    const blunt = 1.0 + tipZone * 0.48 * (1.0 - ringRadius);
    v.x *= blunt;
    v.y *= blunt;
    v.z -= tipZone * tipZone * 0.055;

    // Sleek murine wedge: tapered cranium flowing into a delicate pointed muzzle.
    // Ventral depth is kept noticeably leaner than dorsal cranial height so the
    // lower jaw sweeps cleanly from throat to chin instead of ballooning down.
    const rostral = smooth(zNorm, 0.24, 1.0);
    v.x *= (1.0 - 0.62 * rostral) * PROP.snoutWidth * 0.82;
    const dorsalMask = smooth(v.y, -0.15, 0.25);
    const vertBase = THREE.MathUtils.lerp(0.64, 0.84, dorsalMask);
    v.y *= (1.0 - 0.46 * rostral) * vertBase;
    v.z *= PROP.snoutLength * 1.04;
    v.y -= zNorm * 0.09;

    // Smooth occiput taper into the cervical core without pinching a collar notch,
    // plus subtle zygomatic/cheek mass below the eyes.
    const occipitalBlend = THREE.MathUtils.smoothstep(zNorm, 0.04, 0.62);
    v.x *= 0.84 + occipitalBlend * 0.16;
    v.y *= 0.88 + occipitalBlend * 0.12;
    const cheek = Math.exp(-Math.pow((zNorm - 0.48) / 0.19, 2.0));
    const lowerMask = 1.0 - smooth(v.y, -0.10, 0.22);
    v.x *= 1.0 + cheek * lowerMask * PROP.cheekFullness * 0.24;

    // Lower jaw & tucked chin: smooth upward ventral sweep behind the nose,
    // avoiding any deep concave pocket under the rhinarium in front view.
    const ventral = 1.0 - smooth(v.y, -0.22, 0.02);
    const jawTuck = smooth(zNorm, 0.44, 0.96);
    v.x *= 1.0 - ventral * jawTuck * 0.24;
    v.y += ventral * smooth(zNorm, 0.56, 0.98) * 0.055;
    const chin = Math.exp(-Math.pow((zNorm - 0.84) / 0.08, 2.0));
    v.y -= ventral * chin * 0.014;
    const overhang = smooth(zNorm, 0.78, 1.0);
    v.z -= ventral * overhang * 0.042;

    // Shallow orbital fossa so the eyes nestle organically into the cranium.
    const orbitZ = Math.exp(-Math.pow((zNorm - 0.66) / 0.12, 2.0));
    const orbitY = Math.exp(-Math.pow((v.y - 0.14) / 0.15, 2.0));
    const latMask = smooth(Math.abs(v.x), 0.14, 0.34);
    v.x *= 1.0 - orbitZ * orbitY * latMask * 0.055;

    // Mystacial whisker pads: soft bilateral cushions framing a subtle median
    // philtrum dimple directly below the rhinarium.
    const padZ = Math.exp(-Math.pow((zNorm - 0.86) / 0.085, 2.0));
    const padY = Math.exp(-Math.pow((v.y + 0.09) / 0.085, 2.0));
    const padLat = smooth(Math.abs(v.x), 0.025, 0.12);
    v.x *= 1.0 + padZ * padY * padLat * 0.14;
    v.y -= padZ * padY * padLat * 0.006;
    const philtrumCenter = Math.exp(-Math.pow(v.x / 0.032, 2.0));
    const philtrumZone = smooth(zNorm, 0.86, 0.99) * Math.exp(-Math.pow((v.y + 0.11) / 0.06, 2.0));
    v.z -= philtrumCenter * philtrumZone * 0.014;
    // Localized sub-philtral oral recess so the split upper lip, incisors, and
    // delicate lower lip sit organically inside the muzzle contour.
    const oralX = Math.exp(-Math.pow(v.x / 0.052, 2.0));
    const oralY = Math.exp(-Math.pow((v.y + 0.165) / 0.048, 2.0));
    const oralZ = smooth(zNorm, 0.85, 0.97);
    v.z -= oralX * oralY * oralZ * 0.024;

    // One continuous dorsal skull slope from the occiput into the snout.
    // A gentle cranial crown and low convex nasal bridge lead cleanly into the nose.
    const dorsal = smooth(v.y, -0.02, 0.22);
    const cranium = Math.exp(-Math.pow((zNorm - 0.32) / 0.26, 2.0));
    v.y += dorsal * cranium * 0.095;
    const nasal = Math.exp(-Math.pow((zNorm - 0.78) / 0.14, 2.0));
    v.y += dorsal * nasal * 0.032;

    // Temporal fullness supports the pinna roots without rounding the forehead.
    const temporal = Math.exp(-Math.pow((zNorm - 0.28) / 0.14, 2.0));
    const temporalHeight = smooth(v.y, -0.14, 0.46);
    v.x *= 1.0 + temporal * temporalHeight * 0.035;
    // Zygomatic arch: smooth lateral ridge below/behind the eye (no hard step).
    const zygoZ = Math.exp(-Math.pow((zNorm - 0.48) / 0.10, 2.0));
    const zygoY = Math.exp(-Math.pow((v.y + 0.01) / 0.11, 2.0));
    v.x *= 1.0 + zygoZ * zygoY * 0.024;
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
    const nose = findHeadSurfacePoint({ x: 0, y: -0.06, z: 3.0 });
    const tipZ = nose.position.z;
    const eye = side => findHeadSurfacePoint({ x: side * 0.66, y: 0.16, z: 0.36 * PROP.snoutLength });
    const pad = side => findHeadSurfacePoint({ x: side * 0.50, y: nose.position.y - 0.03, z: tipZ - 0.22 });
    const mouth = findHeadSurfacePoint({ x: 0, y: nose.position.y - 0.125, z: tipZ - 0.05 });
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
      centerY: THREE.MathUtils.lerp(0.04, 0.22, eased) + arch * 0.016,
      radiusX: THREE.MathUtils.lerp(
        PROP.neckGirth * 0.92,
        PROP.neckGirth * 0.74,
        t
      ),
      radiusY: THREE.MathUtils.lerp(
        PROP.neckGirth * 0.92,
        PROP.neckGirth * 0.74,
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

  // Radius fraction of the cervical mantle occupied by the solid core: tapers
  // monotonically from inside the thorax through the neck waist into the skull
  // so it emerges tangentially with no outward flare or vertical shading kink.
  function cervicalCoreInset(rawT) {
    const frontBlend = 1.0 - THREE.MathUtils.smoothstep(rawT, 0.58, 1.0) * 0.46;
    return THREE.MathUtils.lerp(0.95, 0.85, rawT) * frontBlend;
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
        uvs.push(segment / radialSegments, rawT);
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

  const deformThigh = (v) => {
    const yN = (v.y + 1) / 2; // 0=bottom(knee), 1=top(hip)
    // Muscular quadriceps/biceps femoris mass tapering cleanly into stifle (knee)
    const kneeTaper = 0.72 + yN * 0.34;
    v.x *= 0.58 * kneeTaper;
    v.y *= 1.22;
    v.z *= 1.18 * (0.80 + yN * 0.25);
    v.z -= v.y * 0.28;
    // Tuck caudal-distal hamstring so thigh does not hang below knee joint
    if (v.z < 0 && yN < 0.42) {
      v.z *= 0.78 + (yN / 0.42) * 0.22;
    }
    const gluteal = Math.exp(-Math.pow((yN - 0.84) / 0.16, 2.0));
    v.x *= 1.0 + gluteal * 0.13;
    if (v.z > 0) v.z += gluteal * 0.04;
  };

  function shapePinnaPoint(x, y, depthMul = 1.0, zLift = 0.0) {
    const radius = Math.min(1.0, Math.sqrt(x * x + y * y));
    const lowerOpen = THREE.MathUtils.smoothstep(y, -0.90, -0.06);
    const crownTaper = 1.0 - THREE.MathUtils.smoothstep(y, 0.30, 0.96) * 0.14;
    const rootPad = Math.exp(-(x * x * 5.0 + Math.pow(y + 0.76, 2.0) * 15.0));
    const shoulder = Math.exp(-Math.pow((y - 0.06) / 0.70, 2.0));
    const rim = THREE.MathUtils.smoothstep(radius, 0.78, 1.0);
    const centerCup = Math.max(0.0, 1.0 - radius * radius);
    // Gentle forward-cupped side curl + soft backward crown flex so top and
    // side views show an organic cupped murine ear rather than a flat plate.
    const lateralCup = (x * x) * 0.105 * (0.45 + 0.55 * lowerOpen);
    const crownBackFlex = THREE.MathUtils.smoothstep(y, 0.18, 0.98) * 0.085;
    const upperFold = Math.exp(-(
      Math.pow(x + 0.10, 2.0) * 5.5 + Math.pow(y - 0.22, 2.0) * 8.5
    ));
    const basalFold = Math.exp(-(
      Math.pow(x + 0.16, 2.0) * 9.0 + Math.pow(y + 0.50, 2.0) * 14.0
    ));
    const attachmentBlend = THREE.MathUtils.smoothstep(-y, 0.58, 0.96);

    const px = x * (0.92 + shoulder * 0.065)
      * (0.54 + lowerOpen * 0.46) * crownTaper;
    const attachmentY = -0.70 + Math.abs(x) * 0.035;
    const py = THREE.MathUtils.lerp(y * 0.94, attachmentY, attachmentBlend)
      + rootPad * 0.018;
    const pz = (
      -centerCup * 0.285 + lateralCup - crownBackFlex + rim * 0.090
      + upperFold * 0.028 + basalFold * 0.036 + rootPad * 0.020
    ) * depthMul + zLift;
    return new THREE.Vector3(px, py, pz);
  }

  function createPinnaShellGeometry() {
    const rings = 18;
    const segments = 72;
    const row = segments + 1;
    const surfaceVertexCount = (rings + 1) * row;
    const positions = [];
    const uvs = [];
    const indices = [];

    for (let side = 0; side < 2; side++) {
      for (let ring = 0; ring <= rings; ring++) {
        const normalizedRadius = ring / rings;
        for (let seg = 0; seg <= segments; seg++) {
          const angle = (seg / segments) * Math.PI * 2;
          const x = Math.cos(angle) * normalizedRadius;
          const y = Math.sin(angle) * normalizedRadius;
          const point = shapePinnaPoint(x, y);
          if (side === 1) {
            const rootThickness = Math.exp(-(
              x * x * 5.0 + Math.pow(y + 0.72, 2.0) * 12.0
            ));
            const shellThickness = 0.019
              + (1.0 - normalizedRadius) * 0.008
              + rootThickness * 0.015;
            point.z -= shellThickness;
          }
          positions.push(point.x, point.y, point.z);
          uvs.push(0.5 + Math.cos(angle) * normalizedRadius * 0.5, 0.5 + Math.sin(angle) * normalizedRadius * 0.5);
        }
      }
    }

    // Group 0: concave inner (anterior) surface. Group 1: convex outer
    // (posterior) surface plus the margin, which carries short coat pigment.
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
      const backA = outerBack + seg;
      const backB = backA + 1;
      outerIndices.push(frontA, backA, frontB, frontB, backA, backB);
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

  function createPinnaSurfaceGeometry(radiusScale = 0.78, depthMul = 1.0, zLift = 0.002) {
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
          zLift
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

  function createPinnaRimGeometry() {
    const segments = 76;
    const positions = [];
    const uvs = [];
    const indices = [];
    for (let band = 0; band < 2; band++) {
      for (let seg = 0; seg <= segments; seg++) {
        const t = seg / segments;
        // Flush rolled helix edge tapering smoothly to zero into both sides of
        // the skull attachment so grazing/top views never show a notched step.
        const angle = -Math.PI * 0.24 + t * Math.PI * 1.48;
        const endFade = THREE.MathUtils.smoothstep(t, 0.0, 0.16)
          * THREE.MathUtils.smoothstep(1.0 - t, 0.0, 0.16);
        const radius = band === 0 ? 1.0 - 0.018 * endFade : 1.0;
        const point = shapePinnaPoint(
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          1.0,
          0.0005 + endFade * 0.0015
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
  const earGeo = createPinnaShellGeometry();
  const earInnerGeo = createPinnaSurfaceGeometry(0.42, 1.0, 0.002);
  const earRimGeo = createPinnaRimGeometry();

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
  // 6. PAW BUILDER (anatomically correct forepaw vs hindpaw)
  // ═══════════════════════════════════════════════════════════════
  function createPaw(mat, isFore = false, strain = {}, furColor = null) {
    const paw = new THREE.Group();
    paw.name = isFore ? 'connected-forepaw' : 'connected-hindpaw';
    const pawFleshMat = mat.clone();
    pawFleshMat.color.set(strain.pawColor ?? '#e6b0a7');
    pawFleshMat.transmission = 0.09;
    pawFleshMat.thickness = 0.10;
    pawFleshMat.roughness = 0.56;
    pawFleshMat.clearcoat = 0.04;
    pawFleshMat.clearcoatRoughness = 0.46;
    pawFleshMat.sheen = 0.24;
    pawFleshMat.sheenColor = new THREE.Color(strain.pawColor ?? '#e6b0a7').offsetHSL(0, -0.04, 0.08);
    // Thin dorsal pelage tint over the metacarpus/metatarsus: blends subtly
    // toward the coat tone on dark strains and soft ivory-pink on pale strains.
    const pawDorsalMat = pawFleshMat.clone();
    if (furColor) {
      const coat = new THREE.Color(furColor);
      const coatHsl = {}; coat.getHSL(coatHsl);
      const dorsalBlend = coatHsl.l < 0.25 ? 0.34 : (coatHsl.l > 0.70 ? 0.24 : 0.26);
      pawDorsalMat.color.lerp(coat, dorsalBlend);
      pawDorsalMat.roughness = 0.72;
      pawDorsalMat.transmission = 0.04;
      pawDorsalMat.sheen = 0.48;
      pawDorsalMat.sheenColor = coat.clone().lerp(new THREE.Color(0xffffff), 0.30);
    }
    const padMat = mat.clone();
    padMat.color.set(strain.pawPadColor ?? strain.pawColor ?? '#e9b8ae');
    padMat.color.multiplyScalar(strain.padDarken ?? 0.9);
    padMat.transmission = 0.07; padMat.thickness = 0.10;
    padMat.roughness = 0.54; padMat.clearcoat = 0.05; padMat.clearcoatRoughness = 0.44;
    padMat.normalMap = padNormalMap;
    padMat.normalScale = new THREE.Vector2(0.20, 0.20);

    const pawScale = isFore ? (PROP.forePawSize) : (PROP.hindPawSize);
    // Compact carpus/tarsus root bridge seamlessly joining wrist/ankle into the
    // dorsal paw without a protruding heel balloon behind the foreleg.
    const rootLength = isFore ? 0.075 : 0.22;
    const rootBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(isFore ? 0.036 : 0.050, rootLength, 10, 12),
      pawDorsalMat
    );
    rootBridge.name = isFore ? 'forepaw-root-bridge' : 'hindpaw-root-bridge';
    rootBridge.rotation.x = Math.PI / 2 - (isFore ? 0.08 : 0.06);
    rootBridge.position.set(0, isFore ? 0.004 : -0.004, isFore ? -0.026 : -0.108);
    rootBridge.scale.set(isFore ? 0.94 : 0.96, isFore ? 0.68 : 0.64, 1.0);
    rootBridge.castShadow = true;
    paw.add(rootBridge);

    if (isFore) {
      const dorsalHandGeo = new THREE.SphereGeometry(0.096, 18, 18);
      deform(dorsalHandGeo, (hv) => {
        const zN = THREE.MathUtils.clamp((hv.z / 0.096 + 1) * 0.5, 0, 1);
        hv.x *= (0.72 + zN * 0.14) * pawScale;
        hv.y *= 0.34 * (1.0 - zN * 0.14);
        hv.z *= 1.06 * pawScale;
        hv.y -= zN * zN * 0.012;
      });
      const dorsalHand = new THREE.Mesh(dorsalHandGeo, pawDorsalMat);
      dorsalHand.name = 'forepaw-dorsal-hand';
      dorsalHand.position.set(0, -0.008, 0.048);
      dorsalHand.castShadow = true;
      paw.add(dorsalHand);

      // Vestigial medial pollex (thumb) nub characteristic of murine manus
      const pollex = new THREE.Mesh(new THREE.SphereGeometry(0.016, 8, 8), pawFleshMat);
      pollex.scale.set(0.82, 0.64, 1.08);
      pollex.position.set(-0.064 * pawScale, -0.026, 0.032);
      paw.add(pollex);

      const palm = new THREE.Mesh(new THREE.SphereGeometry(0.068, 16, 16), padMat);
      palm.scale.set(0.76 * pawScale, 0.24, 0.96 * pawScale);
      palm.position.set(0, -0.044, 0.056);
      paw.add(palm);

      const carpalPad = new THREE.Mesh(new THREE.SphereGeometry(0.030, 12, 12), padMat);
      carpalPad.scale.set(0.84, 0.32, 0.92);
      carpalPad.position.set(0, -0.036, -0.018);
      paw.add(carpalPad);

      for (let p = 0; p < 3; p++) {
        const iPad = new THREE.Mesh(new THREE.SphereGeometry(0.022, 10, 10), padMat);
        iPad.scale.set(0.82 * pawScale, 0.34, 0.88 * pawScale);
        iPad.position.set((p - 1) * 0.030 * pawScale, -0.046, 0.106);
        paw.add(iPad);
      }
    } else {
      // Sleek elongated murine pes: slender tapered arch from heel to metatarsals
      const dorsalFootGeo = new THREE.SphereGeometry(0.118, 20, 20);
      deform(dorsalFootGeo, (fv) => {
        const zN = THREE.MathUtils.clamp((fv.z / 0.118 + 1) * 0.5, 0, 1);
        const widthProfile = 0.56 + zN * 0.24;
        fv.x *= widthProfile * pawScale;
        fv.y *= (0.30 + (1.0 - zN) * 0.08);
        fv.z *= 1.62 * pawScale;
        // Subtle natural plantar arch & gentle downward taper to the toe line
        fv.y += Math.sin(zN * Math.PI) * 0.008 - zN * 0.014;
      });
      const dorsalFoot = new THREE.Mesh(dorsalFootGeo, pawDorsalMat);
      dorsalFoot.name = 'hindpaw-dorsal-foot';
      dorsalFoot.position.set(0, -0.010, 0.062);
      dorsalFoot.castShadow = true;
      paw.add(dorsalFoot);

      // Subtle low-relief dorsal metatarsal tendons flush with the foot arch
      for (let r = -1; r <= 1; r++) {
        const tendon = new THREE.Mesh(
          new THREE.CapsuleGeometry(0.007, 0.20 * pawScale, 6, 8),
          pawDorsalMat
        );
        tendon.rotation.x = Math.PI / 2 - 0.02;
        tendon.rotation.y = r * 0.036;
        tendon.scale.set(0.76, 1.0, 0.42);
        tendon.position.set(r * 0.022 * pawScale, 0.016, 0.064);
        paw.add(tendon);
      }

      const sole = new THREE.Mesh(new THREE.SphereGeometry(0.098, 18, 18), padMat);
      sole.scale.set(0.70 * pawScale, 0.20, 1.48 * pawScale);
      sole.position.set(0, -0.048, 0.068);
      paw.add(sole);

      const heelPad = new THREE.Mesh(new THREE.SphereGeometry(0.042, 12, 12), padMat);
      heelPad.scale.set(0.78, 0.28, 0.94);
      heelPad.position.set(0, -0.040, -0.118 * pawScale);
      paw.add(heelPad);

      const midPad = new THREE.Mesh(new THREE.SphereGeometry(0.044, 12, 12), padMat);
      midPad.scale.set(0.76, 0.28, 1.06);
      midPad.position.set(0, -0.050, 0.132 * pawScale);
      paw.add(midPad);
    }

    const pawSpreadMul = strain.pawSpreadMul ?? 1.0;
    const toeCount = isFore ? 4 : 5;
    const baseSpread = isFore ? 0.038 : 0.035;
    const spread = baseSpread * pawSpreadMul * pawScale;

    const clawLocalMat = clawMat.clone();
    clawLocalMat.color.set(strain.clawColor ?? 0xf2e9e0);

    for (let i = 0; i < toeCount; i++) {
      const centeredIndex = i - (toeCount - 1) * 0.5;
      const t = centeredIndex * spread;
      const arc = Math.abs(centeredIndex) / (toeCount * 0.5 + 0.01);
      const toe = new THREE.Group();
      toe.name = `${isFore ? 'fore' : 'hind'}-digit-${i + 1}`;
      let lenMul = 1.0;
      if (isFore && (i === 0 || i === 3)) lenMul = 0.74;
      if (!isFore && (i === 0 || i === 4)) lenMul = 0.68;
      const toeLen = (isFore ? 0.138 : 0.192)
        * (strain.toeLenMul ?? 1.0) * lenMul * pawScale;
      const toeRadius = isFore ? 0.0138 : 0.0154;
      const phalanxGeo = new THREE.CapsuleGeometry(toeRadius, toeLen, 10, 10);
      deform(phalanxGeo, (pv) => {
        const yN = THREE.MathUtils.clamp((pv.y + toeLen * 0.5) / Math.max(0.01, toeLen), 0, 1);
        const taper = 1.0 - yN * 0.28;
        pv.x *= taper;
        pv.z *= taper;
        // After rotation.x = +PI/2, local +Z maps to toe -Y (toward the floor):
        // gently arch the mid-phalanx upward and curve the tip down to the floor.
        pv.z += -Math.sin(yN * Math.PI) * 0.005 + yN * yN * 0.010;
      });
      const phalanx = new THREE.Mesh(phalanxGeo, pawFleshMat);
      phalanx.position.z = toeLen * 0.46;
      phalanx.rotation.x = Math.PI / 2;
      phalanx.scale.set(0.92, 1.0, 0.84);
      phalanx.castShadow = true;
      toe.add(phalanx);

      const knuckle = new THREE.Mesh(
        new THREE.SphereGeometry(isFore ? 0.0145 : 0.0165, 10, 10),
        pawDorsalMat
      );
      knuckle.scale.set(0.98, 0.68, 1.05);
      knuckle.position.set(0, -0.001, 0.008);
      toe.add(knuckle);

      const toeTip = new THREE.Mesh(new THREE.SphereGeometry(isFore ? 0.0115 : 0.013, 10, 10), pawFleshMat);
      toeTip.scale.set(1.00, 0.78, 1.00);
      toeTip.position.set(0, -0.009, toeLen + toeRadius * 0.28);
      toe.add(toeTip);

      const toePad = new THREE.Mesh(new THREE.SphereGeometry(isFore ? 0.0145 : 0.017, 10, 10), padMat);
      toePad.scale.set(isFore ? 0.90 : 0.86, isFore ? 0.44 : 0.40, isFore ? 0.92 : 0.98);
      toePad.position.set(0, -0.011, toeLen * 0.74);
      toe.add(toePad);

      const clawLength = isFore ? 0.030 : 0.040;
      const clawRadius = isFore ? 0.0046 : 0.0054;
      const clawGeo = new THREE.CapsuleGeometry(clawRadius, clawLength, 6, 6);
      clawGeo.translate(0, clawLength / 2, 0);
      deform(clawGeo, (cv) => {
        const yN = THREE.MathUtils.clamp(
          (cv.y + clawRadius) / (clawLength + clawRadius * 2),
          0,
          1
        );
        cv.x *= 1.0 - yN * 0.56;
        cv.z *= 1.0 - yN * 0.46;
        // Downward falcula curvature (+Z maps to -Y after rotation.x = +PI/2)
        cv.z += yN * yN * 0.010;
      });
      const claw = new THREE.Mesh(clawGeo, clawLocalMat);
      claw.rotation.x = Math.PI / 2 + (isFore ? 0.16 : 0.14);
      claw.position.set(0, -0.009, toeLen + toeRadius * 0.38);
      claw.castShadow = true;
      toe.add(claw);

      const zPos = (isFore ? 0.114 - arc * 0.018 : 0.184 - arc * 0.026) * pawScale;
      toe.position.set(t, isFore ? -0.018 + arc * 0.003 : -0.020 + arc * 0.003, zPos);
      toe.rotation.y = centeredIndex * (isFore ? 0.075 : 0.055);
      // Slight downward toe-resting pitch (positive X pitches +Z toward floor)
      toe.rotation.x = isFore ? 0.04 + arc * 0.02 : 0.02 + arc * 0.015;
      paw.add(toe);
    }

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
      groomingContact.position.set(0, -0.044, 0.15);
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

    const shoulderMantle = new THREE.Mesh(new THREE.SphereGeometry(0.155, 16, 16), surfaceMat);
    shoulderMantle.position.set(-sign * 0.11, 0.02, -0.10);
    shoulderMantle.scale.set(1.04, 0.96, 1.24);
    shoulderMantle.castShadow = false;
    chain.add(shoulderMantle);
    // Pectoral-brachial skin fold enveloping the proximal arm into the sternum
    const pectoralFold = new THREE.Mesh(new THREE.SphereGeometry(0.135 * thickness, 14, 14), surfaceMat);
    pectoralFold.name = `${side}-pectoral-brachial-fold`;
    pectoralFold.position.set(-sign * 0.075, -0.08, -0.03);
    pectoralFold.scale.set(1.04, 1.24, 1.18);
    pectoralFold.castShadow = false;
    chain.add(pectoralFold);
    const scapularBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.095 * thickness, 0.22, 10, 10),
      surfaceMat
    );
    scapularBridge.name = `${side}-scapular-soft-tissue-bridge`;
    scapularBridge.position.set(-sign * 0.07, 0.04, -0.10);
    scapularBridge.rotation.x = Math.PI / 2;
    scapularBridge.castShadow = false;
    chain.add(scapularBridge);

    // Adult murine radius/ulna is slightly longer than the humerus. The
    // remaining scale belongs to the carpus and paw, represented below.
    const upperArm = new THREE.Group();
    upperArm.name = `${side}-upper-arm`;
    const uaLen = 0.42 * fl;
    const upperArmGeo = new THREE.CapsuleGeometry(0.108 * thickness, uaLen, 10, 12);
    deform(upperArmGeo, (uv) => {
      // Taper muscular brachium cleanly from tucked shoulder cap into elbow
      const yN = THREE.MathUtils.clamp((uv.y + uaLen * 0.5) / Math.max(0.01, uaLen), 0, 1);
      const topTuck = 1.0 - THREE.MathUtils.smoothstep(yN, 0.74, 1.0) * 0.20;
      const t = (0.72 + yN * 0.30) * topTuck;
      uv.x *= t * 0.88;
      uv.z *= t * 1.04;
    });
    const upperArmMesh = new THREE.Mesh(upperArmGeo, surfaceMat);
    upperArmMesh.position.y = -uaLen * 0.5;
    upperArmMesh.castShadow = true;
    upperArm.add(upperArmMesh);
    upperArm.rotation.x = FORELIMB_NEUTRAL.shoulderX;
    // Shoulder joint sphere tucked smoothly inside chest coat
    const shoulderJoint = new THREE.Mesh(new THREE.SphereGeometry(0.084 * thickness, 12, 12), surfaceMat);
    shoulderJoint.scale.set(0.84, 0.82, 0.98);
    shoulderJoint.castShadow = false;
    upperArm.add(shoulderJoint);
    chain.add(upperArm);

    // Forearm (radius/ulna) — attached at elbow
    const forearm = new THREE.Group();
    forearm.name = `${side}-forearm`;
    const faLen = 0.45 * fl;
    forearm.position.set(0, -uaLen - 0.01, -0.01);
    forearm.rotation.x = FORELIMB_NEUTRAL.elbowX;
    // Soft-tissue bridge to avoid segmented look between upper arm and forearm.
    const elbowBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.074 * thickness, 0.088, 10, 10),
      surfaceMat
    );
    elbowBridge.position.set(0, -uaLen * 0.92, 0.020);
    elbowBridge.rotation.x = 0.15;
    elbowBridge.castShadow = false;
    upperArm.add(elbowBridge);
    // Elbow joint (sleek furred contour flush with forearm)
    const elbowJoint = new THREE.Mesh(new THREE.SphereGeometry(0.062 * thickness, 10, 10), surfaceMat);
    elbowJoint.scale.set(0.90, 0.94, 0.96);
    elbowJoint.castShadow = false;
    forearm.add(elbowJoint);
    const forearmGeo = new THREE.CapsuleGeometry(0.074 * thickness, faLen, 8, 12);
    deform(forearmGeo, (fv) => {
      // Natural proximal-to-distal taper from forearm flexors down into slender carpus
      const yN = THREE.MathUtils.clamp((fv.y + faLen * 0.5) / Math.max(0.01, faLen), 0, 1);
      const t = 0.56 + yN * 0.48;
      fv.x *= t;
      fv.z *= t * 1.04;
    });
    const forearmMesh = new THREE.Mesh(forearmGeo, surfaceMat);
    forearmMesh.position.y = -faLen * 0.5;
    forearmMesh.castShadow = true;
    forearm.add(forearmMesh);
    upperArm.add(forearm);

    // Wrist joint — coaxial furred carpus tapering seamlessly onto dorsal manus
    const wrist = new THREE.Group();
    wrist.name = `${side}-wrist`;
    wrist.position.set(0, -faLen - 0.018, 0.03);
    wrist.rotation.x = FORELIMB_NEUTRAL.wristX;
    const wristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.038 * thickness, 8, 8), surfaceMat);
    wristJoint.scale.set(0.92, 0.88, 0.94);
    wristJoint.castShadow = false;
    wrist.add(wristJoint);
    const wristCuffGeo = new THREE.CapsuleGeometry(0.040 * thickness, 0.074, 8, 10);
    deform(wristCuffGeo, (wv) => {
      const yN = THREE.MathUtils.clamp((wv.y + 0.037) / 0.074, 0, 1);
      const t = 0.76 + yN * 0.26;
      wv.x *= t;
      wv.z *= t;
    });
    const wristCuff = new THREE.Mesh(wristCuffGeo, surfaceMat);
    wristCuff.position.set(sign * 0.017, -0.064, 0.035);
    wristCuff.rotation.x = -0.49;
    wristCuff.rotation.z = sign * 0.22;
    wristCuff.castShadow = false;
    wrist.add(wristCuff);
    forearm.add(wrist);

    // Paw — sits level at bottom of wrist
    const paw = createPaw(skinMat, true, strain, furColor);
    paw.position.set(sign * 0.035, -0.13, 0.07);
    paw.scale.setScalar(0.96 * Math.sqrt(thickness));
    paw.rotation.x = FORELIMB_NEUTRAL.pawX;
    wrist.add(paw);

    // Continuous coat hugging the actual tapered limb contours & joint caps
    // from shoulder down over the carpal cuff (zero bare shiny joint knobs).
    const coatCapsule = (
      mesh,
      radius,
      span,
      count,
      hairLength,
      taperMin = 1.0,
      taperSpan = 0.0,
      isUndercoat = false
    ) => {
      const coat = generateFurLayer(v => {
        const y = v.y;
        const yN = THREE.MathUtils.clamp((y + 1) * 0.5, 0, 1);
        const t = taperMin + yN * taperSpan;
        v.multiplyScalar(radius * t);
        v.y += y * span * 0.5;
      }, Math.floor(count * density), furColor, hairLength, false, isUndercoat);
      mesh.add(coat);
    };
    coatCapsule(upperArmMesh, 0.106 * thickness, uaLen, 3400, 0.066, 0.72, 0.24);
    coatCapsule(shoulderJoint, 0.084 * thickness, 0, 340, 0.064, 1.0, 0.0, true);
    coatCapsule(elbowBridge, 0.074 * thickness, 0.088, 340, 0.052, 1.0, 0.0, true);
    coatCapsule(elbowJoint, 0.062 * thickness, 0, 280, 0.048, 1.0, 0.0, true);
    coatCapsule(forearmMesh, 0.074 * thickness, faLen, 2800, 0.046, 0.56, 0.48);
    coatCapsule(wristJoint, 0.038 * thickness, 0, 200, 0.034, 1.0, 0.0, true);
    coatCapsule(wristCuff, 0.040 * thickness, 0.074, 850, 0.030, 0.76, 0.26, true);
    coatCapsule(shoulderMantle, 0.156, 0, 1400, 0.066, 1.0, 0.0, true);
    coatCapsule(pectoralFold, 0.136 * thickness, 0.04, 1300, 0.066, 1.0, 0.0, true);

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

    // Thigh (femur) — sculpted muscular haunch
    const thigh = new THREE.Group();
    thigh.name = `${side}-thigh`;
    thigh.position.set(0, -0.10, 0.02);
    chain.add(thigh);
    const thighMuscle = new THREE.Mesh(thighGeo, surfaceMat);
    thighMuscle.scale.set(0.76 * thickness, 0.68 * hl, 0.86 * thickness);
    thighMuscle.castShadow = true;
    thigh.add(thighMuscle);
    thighMuscle.add(generateFurLayer(deformThigh, Math.floor(8200 * density), furColor, 0.115, false));

    // Shank (crus / tibia-fibula) — furred distal leg tapering toward hock
    const shank = new THREE.Group();
    shank.name = `${side}-shank`;
    shank.position.set(0.02 * sign, -0.32 * hl, 0.18);
    thigh.add(shank);

    const kneeJoint = new THREE.Mesh(new THREE.SphereGeometry(0.096 * thickness, 10, 10), surfaceMat);
    kneeJoint.position.set(0, 0.045, 0.01);
    kneeJoint.castShadow = false;
    shank.add(kneeJoint);

    const shankLen = 0.46 * hl;
    const shankGeo = new THREE.CapsuleGeometry(0.068 * thickness, shankLen, 8, 12);
    deform(shankGeo, (sv) => {
      // Gastrocnemius muscle taper into slender Achilles/tarsal ankle
      const yN = THREE.MathUtils.clamp((sv.y + shankLen * 0.5) / Math.max(0.01, shankLen), 0, 1);
      const t = 0.66 + yN * 0.44;
      sv.x *= t * 0.92;
      sv.z *= t * 1.06;
    });
    const shankMesh = new THREE.Mesh(shankGeo, surfaceMat);
    shankMesh.rotation.x = -0.16;
    shankMesh.position.set(0, -shankLen * 0.5, 0.09);
    shankMesh.castShadow = true;
    shank.add(shankMesh);
    shankMesh.add(generateFurLayer(v => {
      const y = v.y;
      const yN = THREE.MathUtils.clamp((y + 1) * 0.5, 0, 1);
      const t = 0.66 + yN * 0.44;
      v.multiplyScalar(0.068 * thickness * t);
      v.y += y * shankLen * 0.5;
    }, Math.floor(2400 * density), furColor, 0.052, false));

    // Ankle/hock (tarsocrural joint + sleek calcaneal tuber)
    const ankle = new THREE.Group();
    ankle.name = `${side}-ankle`;
    ankle.position.set(0, -shankLen - 0.04, 0.18);
    shank.add(ankle);

    const ankleJoint = new THREE.Mesh(new THREE.SphereGeometry(0.052 * thickness, 10, 10), surfaceMat);
    ankleJoint.castShadow = false;
    ankle.add(ankleJoint);
    ankleJoint.add(generateFurLayer(v => {
      v.multiplyScalar(0.052 * thickness);
    }, Math.floor(240 * density), furColor, 0.036, false, true));

    // Calcaneus (hock): compact tucked tuber rather than a protruding sphere
    const heel = new THREE.Mesh(new THREE.CapsuleGeometry(0.048 * thickness, 0.075, 8, 8), surfaceMat);
    heel.rotation.x = -0.42;
    heel.position.set(0, 0.018, -0.032);
    heel.castShadow = false;
    ankle.add(heel);
    heel.add(generateFurLayer(v => {
      const y = v.y;
      v.multiplyScalar(0.049 * thickness);
      v.y += y * 0.036;
    }, Math.floor(600 * density), furColor, 0.038, false));

    // Tarsometatarsal bridge routed collinearly from hock directly into the
    // proximal dorsal foot, coated with a smooth fur-to-skin transition.
    const metaMat = skinMat.clone();
    const furColObj = new THREE.Color(furColor);
    const furHslMeta = {}; furColObj.getHSL(furHslMeta);
    metaMat.color.set(strain.pawColor ?? '#e6b0a7').lerp(
      furColObj,
      furHslMeta.l < 0.25 ? 0.36 : 0.26
    );
    metaMat.roughness = 0.68;
    metaMat.transmission = 0.05;
    metaMat.sheen = 0.42;
    metaMat.sheenColor = new THREE.Color(strain.pawColor ?? '#e6b0a7').lerp(furColObj, 0.35);
    const metaLen = 0.25 * PROP.hindPawSize;
    const metaGeo = new THREE.CapsuleGeometry(0.046 * thickness, metaLen, 8, 10);
    deform(metaGeo, (mv) => {
      const yN = THREE.MathUtils.clamp((mv.y + metaLen * 0.5) / Math.max(0.01, metaLen), 0, 1);
      const t = 0.80 + yN * 0.30;
      mv.x *= t;
      mv.z *= t * 0.82;
    });
    const metaMesh = new THREE.Mesh(metaGeo, metaMat);
    metaMesh.rotation.x = Math.PI / 2 + 0.62;
    metaMesh.rotation.z = -sign * 0.13;
    metaMesh.position.set(sign * 0.022, -0.104, 0.112);
    metaMesh.castShadow = true;
    ankle.add(metaMesh);
    // Fine tapered dorsal tarsal fur blending cleanly from furred hock onto
    // the upper half of the metatarsus (zero sharp black-to-pink cuff step).
    metaMesh.add(generateFurLayer(v => {
      const y = v.y;
      const yN = THREE.MathUtils.clamp((y + 1) * 0.5, 0, 1);
      const t = 0.82 + yN * 0.26;
      v.x *= 0.045 * thickness * t;
      v.z *= 0.037 * thickness * t;
      v.y = (y * 0.36 - 0.16) * metaLen;
    }, Math.floor(820 * density), furColor, 0.026, false, true));

    // Hindpaw — planted level on ground
    const paw = createPaw(skinMat, false, strain, furColor);
    paw.position.set(sign * 0.045, -0.20, 0.31);
    paw.rotation.x = 0.02;
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
        const curve = new THREE.CubicBezierCurve3(
          start,
          new THREE.Vector3(
            sign * length * 0.30,
            rootY + row.lift * 0.12,
            rootZ + forward * 0.34
          ),
          new THREE.Vector3(
            sign * length * 0.72,
            rootY + row.lift * 0.62 + droop * 0.25 + rootJitter * 0.022,
            rootZ + forward * 0.78
          ),
          new THREE.Vector3(
            sign * length,
            rootY + row.lift + droop,
            rootZ + forward
          )
        );
        const rootRadius = 0.0030 + caudalFrac * 0.0011;
        const whisker = new THREE.Mesh(
          createTaperedWhiskerGeometry(curve, 14, rootRadius, 0.00034, 5),
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
    const clumpScale = THREE.MathUtils.clamp((clumpiness ?? 0.3) / 0.3, 0.0, 3.5);
    const radius = isUndercoat ? 0.0023 : 0.0017;
    // Only the visible guard layer needs the extra bend segment. Undercoat
    // remains a single tapered segment with lightweight matte shading.
    const geo = new THREE.ConeGeometry(radius, length, 3, isUndercoat ? 1 : 2, true);
    geo.translate(0, length / 2, 0);
    geo.rotateX(Math.PI / 2);
    const vertices = geo.attributes.position;
    const colors = new Float32Array(vertices.count * 3);
    for (let i = 0; i < vertices.count; i++) {
      const progress = THREE.MathUtils.clamp(vertices.getZ(i) / length, 0, 1);
      vertices.setX(i, vertices.getX(i) + length * 0.09 * progress * progress);
      const shade = isPaleCoat
        ? (0.93 + progress * 0.12)
        : (0.80 + progress * 0.30);
      colors.set([shade, shade, shade], i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const mat = isUndercoat
      ? new THREE.MeshStandardMaterial({
        vertexColors: true,
        color: 0xffffff,
        roughness: isPaleCoat ? 0.88 : 0.93,
      })
      : new THREE.MeshPhysicalMaterial({
        vertexColors: true,
        color: 0xffffff, roughness: isPaleCoat ? 0.78 : 0.72,
        clearcoat: 0.0,
        sheen: isPaleCoat ? 0.70 : 0.82,
        sheenColor: new THREE.Color(colorHex),
        sheenRoughness: isPaleCoat ? 0.66 : 0.48,
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
      flowDir.normalize().addScaledVector(normal, isUndercoat ? 0.26 : 0.13).normalize();
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
        ? (isUndercoat ? 0.95 : 0.96)
        : (isUndercoat ? 0.88 : 0.92);
      const shadeSpread = isPaleCoat ? 0.05 : (isUndercoat ? 0.08 : 0.10);
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
    const count = Math.floor(620 * density);
    if (count <= 0) return new THREE.Group();
    const geo = new THREE.ConeGeometry(0.0046, 0.074, 3);
    geo.translate(0, 0.037, 0);
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(furColor),
      roughness: 0.88,
    });
    const inst = new THREE.InstancedMesh(geo, mat, count);
    const dummy = new THREE.Object3D();
    const nrm = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      const t = random() * 2.0 - 1.0;
      const arch = 1.0 - t * t;
      const x = t * (0.28 + random() * 0.07);
      const y = -0.035 + arch * 0.11 + random() * 0.055;
      const z = -0.09 + random() * 0.075;
      dummy.position.set(x, y, z);
      nrm.set(
        t * 0.36 + sideSign * 0.04,
        0.50 + arch * 0.28,
        0.38 + random() * 0.22
      ).normalize();
      dummy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), nrm);
      const s = 0.58 + random() * 0.48;
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    return inst;
  }

  function createEarRimFuzz(furColor, skinColor, density) {
    if (density <= 0) return new THREE.Group();
    const count = Math.floor(260 * Math.max(0.35, density));
    if (count <= 0) return new THREE.Group();
    const geo = new THREE.ConeGeometry(0.0024, 0.032, 3);
    geo.translate(0, 0.016, 0);
    const fuzzColor = new THREE.Color(skinColor).lerp(new THREE.Color(furColor), 0.35).offsetHSL(0, -0.05, 0.18);
    const mat = new THREE.MeshBasicMaterial({
      color: fuzzColor,
      transparent: true,
      opacity: 0.34,
      depthWrite: false,
    });
    const inst = new THREE.InstancedMesh(geo, mat, count);
    const dummy = new THREE.Object3D();
    const up = new THREE.Vector3(0, 1, 0);
    const dir = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      const t = i / count + (random() - 0.5) * 0.008;
      const angle = -Math.PI * 0.18 + THREE.MathUtils.clamp(t, 0, 1) * Math.PI * 1.36;
      const pt = shapePinnaPoint(Math.cos(angle) * 0.99, Math.sin(angle) * 0.99, 1.0, 0.004);
      dummy.position.copy(pt);
      dir.set(Math.cos(angle), Math.sin(angle), 0.15 + (random() - 0.5) * 0.2).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const s = 0.65 + random() * 0.55;
      dummy.scale.set(s, s, s);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    return inst;
  }

  function createEarBackFuzz(furColor, density) {
    const count = Math.floor(260 * Math.max(0.0, density));
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(0x51A7C23D);
    const geo = new THREE.ConeGeometry(0.0032, 0.040, 3);
    geo.translate(0, 0.020, 0);
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
      const r = Math.sqrt(rng()) * 0.88;
      const a = rng() * Math.PI * 2;
      const x = Math.cos(a) * r;
      const y = Math.sin(a) * r;
      const pt = shapePinnaPoint(x, y, 1.0, -0.024);
      dummy.position.copy(pt);
      dir.set(x * 0.35, 0.72 + y * 0.25, -0.52).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const s = 0.52 + rng() * 0.45;
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
    // Posterior pinna: short, fine pelage over thin skin. Dark coats stay
    // deeply pigmented charcoal-taupe; pale coats stay translucent rose-ivory.
    const backMat = outerMat.clone();
    backMat.color = new THREE.Color(furColor).lerp(
      outerMat.color,
      outerHsl.l < 0.2 ? 0.18 : 0.56
    );
    backMat.map = null;
    backMat.normalMap = null;
    backMat.transmission = outerHsl.l < 0.2
      ? 0.05
      : Math.min(0.20, outerMat.transmission ?? 0.16);
    backMat.roughness = outerHsl.l < 0.2 ? 0.86 : 0.76;
    backMat.clearcoat = 0.0;
    backMat.sheen = 0.62;
    backMat.sheenColor = new THREE.Color(furColor).lerp(new THREE.Color(0xffffff), 0.22);
    backMat.sheenRoughness = 0.58;

    // Group 0 (concave inner surface) carries innerMat directly; Group 1
    // (convex back + rolled margin) carries backMat so there is zero double-layer gap.
    const outer = new THREE.Mesh(earGeo, [innerMat, backMat]);
    outer.name = 'pinna-shell';
    outer.castShadow = true;
    outer.receiveShadow = true;
    pinna.add(outer);

    const rimMat = backMat.clone();
    rimMat.color = rimMat.color.clone().lerp(outerMat.color, 0.35);
    rimMat.transmission = 0.0;
    rimMat.roughness = Math.min(0.72, (outerMat.roughness ?? 0.48) + 0.12);
    rimMat.polygonOffset = true;
    rimMat.polygonOffsetFactor = -1;
    rimMat.polygonOffsetUnits = -1;
    const rim = new THREE.Mesh(earRimGeo, rimMat);
    rim.name = 'open-auricular-rim';
    rim.castShadow = false;
    pinna.add(rim);

    const rimFuzz = createEarRimFuzz(furColor, skinColor, density);
    rimFuzz.name = 'auricular-marginal-vellus';
    pinna.add(rimFuzz);

    const backFuzz = createEarBackFuzz(furColor, density);
    pinna.add(backFuzz);

    // Subtle non-transmissive basal conchal blush over the unified pinna shell
    // (avoids overlapping two transmissive surfaces which creates a ring cutout).
    const conchalBlushMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(innerMat.color).offsetHSL(0.0, 0.04, -0.04),
      roughness: 0.72,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const inner = new THREE.Mesh(earInnerGeo, conchalBlushMat);
    inner.name = 'concave-pinna-surface';
    inner.position.set(0, -0.08, 0.001);
    inner.castShadow = false;
    pinna.add(inner);

    // Compact buried pedicle seated entirely inside the temporal skull & base fur
    // collar so no bare lobe ever pokes out behind the eye.
    const pedicle = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 16, 12),
      backMat
    );
    pedicle.name = 'buried-pinna-pedicle';
    pedicle.position.set(sideSign * 0.008, -0.02, -0.042);
    pedicle.scale.set(0.72, 0.42, 0.46);
    pedicle.castShadow = false;
    ear.add(pedicle);

    const foldMat = innerMat.clone();
    foldMat.color = foldMat.color.clone().offsetHSL(0.0, 0.010, -0.045);
    foldMat.transmission = Math.min(0.18, innerMat.transmission ?? 0.22);
    foldMat.roughness = 0.68;
    // Concha, tragus, and auditory meatus shadow sit deeply inside the basal
    // cup so they read as organic cartilage contours rather than floating pegs.
    const cupDepthAt = (px, py) => shapePinnaPoint(px, py - 0.76, 1.0, 0.002).z;
    const concha = new THREE.Mesh(
      new THREE.SphereGeometry(0.14, 16, 12),
      foldMat
    );
    concha.name = 'basal-conchal-fold';
    concha.position.set(sideSign * 0.010, 0.26, cupDepthAt(0.0, 0.26) - 0.008);
    concha.scale.set(0.56, 0.36, 0.08);
    concha.rotation.z = sideSign * 0.06;
    concha.castShadow = false;
    ear.add(concha);

    const tragus = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.014, 0.056, 8, 10),
      foldMat
    );
    tragus.name = 'auricular-tragus-fold';
    tragus.position.set(-sideSign * 0.046, 0.22, cupDepthAt(-0.046, 0.22) - 0.002);
    tragus.rotation.z = sideSign * 0.42;
    tragus.scale.set(0.85, 0.85, 0.34);
    tragus.castShadow = false;
    ear.add(tragus);

    const canalMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(innerMat.color).multiplyScalar(0.42),
      roughness: 0.90,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
    });
    const canal = new THREE.Mesh(
      new THREE.SphereGeometry(0.048, 14, 12),
      canalMat
    );
    canal.name = 'ear-canal-shadow';
    canal.position.set(-sideSign * 0.012, 0.19, cupDepthAt(-0.012, 0.19) + 0.002);
    canal.scale.set(0.78, 0.46, 0.10);
    ear.add(canal);

    const baseFur = createEarBaseFur(furColor, density, sideSign);
    baseFur.name = 'pinna-root-fur-collar';
    baseFur.position.set(0, -0.02, -0.015);
    ear.add(baseFur);

    const strainScale = strain.earScaleMul ?? 1.0;
    const s = eSize * 0.78 * strainScale;
    // Pinnae rise from the dorsolateral cranium behind the eyes, cupping
    // forward-laterally with a natural subtle backward sweep and outward splay.
    const openingYaw = 0.48 + (strain.earYaw ?? PROP.earYaw) * 0.30;
    const root = findHeadSurfacePoint({
      x: sideSign * 0.37,
      y: 0.48,
      z: PROP.earSetBack + 0.18,
    });
    ear.position.copy(root.position).addScaledVector(root.normal, -0.055);
    ear.position.y += heightOffset;
    ear.rotation.set(
      -0.20 + asymmetryTilt,
      sideSign * openingYaw + asymmetryYaw,
      sideSign * -0.25
    );
    ear.scale.set(s * 0.94, s, s);
    ear._pinna = pinna;
    ear._pedicle = pedicle;
    ear._canal = canal;
    return ear;
  }

  function createTailBaseFur(furColor, density) {
    const count = Math.floor(420 * Math.max(0.0, density));
    if (count <= 0) return new THREE.Group();
    const rng = makeIsolatedRng(0x2F84A91B);
    const geo = new THREE.ConeGeometry(0.0048, 0.088, 3);
    geo.translate(0, 0.044, 0);
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
      const frac = rng() * 0.85;
      const ang = rng() * Math.PI * 2;
      const rad = THREE.MathUtils.lerp(0.132, 0.096, frac) * (0.94 + rng() * 0.10);
      const x = Math.cos(ang) * rad;
      const y = Math.sin(ang) * rad;
      const z = frac * 0.24;
      dummy.position.set(x, y, z);
      dir.set(x * 0.42, y * 0.42, 0.92 + rng() * 0.18).normalize();
      dummy.quaternion.setFromUnitVectors(up, dir);
      const sc = (0.68 + rng() * 0.46) * (1.0 - frac * 0.42);
      dummy.scale.set(sc, sc, sc);
      dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
    }
    return inst;
  }


  // ═══════════════════════════════════════════════════════════════
  // Shared assets survive subject disposal and can be used by later spawns.
  // ═══════════════════════════════════════════════════════════════
  const sharedResources = new Set([
    bodyGeo, headGeo, neckGeo, cervicalCoreGeo, thighGeo, earGeo, earInnerGeo, earRimGeo,
    fleshMatBase, clawMat, eyeMat, corneaMat, padNormalMap, tailRoughnessMap, tailNormalMap,
    eyeIrisMap, earNormalMap, earRoughnessMap, earAlbedoMap, noseAlbedoMap, noseNormalMap,
    noseRoughnessMap, labFloorAlbedoMap, labFloorRoughnessMap, labFloorNormalMap,
    homeCageBeddingMaps.albedo, homeCageBeddingMaps.normal, homeCageBeddingMaps.roughness,
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
    tailRoughnessMap,
    tailNormalMap,
    eyeIrisMap,
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
