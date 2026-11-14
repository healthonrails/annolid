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
      bodyLength: 2.68, bodyGirthX: 1.27, bodyGirthY: 1.18,
      ribcageDepth: 1.04, waistTaper: 0.82, haunchWidth: 1.04, shoulderWidth: 0.90,
      neckLength: 1.34, neckGirth: 0.64,
      headSize: 0.98, snoutLength: 1.02, snoutWidth: 0.92, cheekFullness: 0.34,
      earSize: 0.78, earSetBack: -0.40, earYaw: 0.72, earThickness: 0.08,
      eyeSize: 0.128, eyeSpacing: 0.45, eyeProtrusion: 1.08, eyeYaw: 0.43,
      foreLimbLength: 0.94, hindLimbLength: 0.80,
      forePawSize: 1.12, hindPawSize: 1.16,
      tailLength: 1.0, tailThickness: 1.0, tailTaper: 0.88, tailSag: 0.02,
    };
  }
  const PROP = defaultProportions();
  const FORELIMB_NEUTRAL = Object.freeze({
    shoulderX: 0.20,
    shoulderZ: 0.0,
    elbowX: -0.62,
    wristX: -0.12,
    pawX: -0.12,
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
    // Neutral warm-putty anti-glare behavioral arena PVC so both snow-white
    // BALB/c and jet-black C57BL/6 coats separate cleanly from the floor.
    const base = ctx.createLinearGradient(0, 0, size, size);
    base.addColorStop(0.0, '#b5b2aa');
    base.addColorStop(0.52, '#c1beb6');
    base.addColorStop(1.0, '#b0ada5');
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, size, size);

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

    // Subtle lab wipe marks and faint organic arena wear.
    for (let i = 0; i < 40; i++) {
      const x = random() * size;
      const y = random() * size;
      const r = size * (0.018 + random() * 0.060);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      const warm = random() > 0.48;
      g.addColorStop(0.0, warm ? 'rgba(142,116,74,0.048)' : 'rgba(68,64,58,0.042)');
      g.addColorStop(0.68, warm ? 'rgba(142,116,74,0.016)' : 'rgba(68,64,58,0.014)');
      g.addColorStop(1.0, 'rgba(0,0,0,0.0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(x, y, r * (0.7 + random() * 1.4), r * (0.24 + random() * 0.55), random() * Math.PI, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.lineCap = 'round';
    for (let i = 0; i < 150; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 18 + random() * 88;
      const a = random() * Math.PI * 2;
      ctx.strokeStyle = random() > 0.5 ? 'rgba(236,234,228,0.045)' : 'rgba(84,76,64,0.042)';
      ctx.lineWidth = 0.45 + random() * 1.0;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(a) * len, y + Math.sin(a) * len);
      ctx.stroke();
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
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, 'rgba(115,115,115,0.24)');
      g.addColorStop(1, 'rgba(230,230,230,0.0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(x, y, r * 1.45, r * 0.45, random() * Math.PI, 0, Math.PI * 2);
      ctx.fill();
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
    // Equirectangular sphere UV: front/lateral pole sits around v = 0.5, u = 0.25/0.75,
    // while SphereGeometry(r) aligned along +Z puts the apex at u=0.25..0.75, v=0.5.
    // Use radial shading around UV center plus subtle stroma fibers so any eye orientation
    // shows organic depth instead of flat plastic.
    ctx.fillStyle = '#d8d2ce';
    ctx.fillRect(0, 0, size, size);
    const cx = size * 0.5;
    const cy = size * 0.5;
    const g = ctx.createRadialGradient(cx, cy, size * 0.05, cx, cy, size * 0.48);
    g.addColorStop(0.0, '#5a5250');
    g.addColorStop(0.38, '#7c7270');
    g.addColorStop(0.72, '#c8beb8');
    g.addColorStop(1.0, '#9e928e');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    ctx.lineCap = 'round';
    for (let i = 0; i < 180; i++) {
      const a = (i / 180) * Math.PI * 2 + (random() - 0.5) * 0.04;
      const r0 = size * (0.14 + random() * 0.05);
      const r1 = size * (0.36 + random() * 0.10);
      const shade = 150 + Math.floor(random() * 75);
      ctx.strokeStyle = `rgba(${shade},${shade - 6},${shade - 10},0.16)`;
      ctx.lineWidth = 0.7 + random() * 0.7;
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
    // PROMINENT VASCULAR NETWORK — main arteries branching from base
    const drawVein = (x0, y0, x1, y1, width, opacity, branches) => {
      ctx.strokeStyle = `rgba(182,78,82,${opacity})`;
      ctx.lineWidth = width;
      ctx.beginPath();
      const mx = (x0 + x1) * 0.5 + (random() - 0.5) * 12;
      const my = (y0 + y1) * 0.5 + (random() - 0.5) * 8;
      ctx.moveTo(x0, y0);
      ctx.quadraticCurveTo(mx, my, x1, y1);
      ctx.stroke();
      // Recursive branching
      if (branches > 0 && width > 0.5) {
        for (let b = 0; b < 2; b++) {
          const bt = 0.5 + random() * 0.4;
          const bx = x0 + (x1 - x0) * bt + (random() - 0.5) * 15;
          const by = y0 + (y1 - y0) * bt + (random() - 0.5) * 15;
          const bx2 = bx + (random() - 0.5) * size * 0.18;
          const by2 = by - random() * size * 0.12;
          drawVein(bx, by, bx2, by2, width * 0.6, opacity * 0.78, branches - 1);
        }
      }
    };
    // Main marginal & central auricular arteries radiating from ear base
    for (let i = 0; i < 9; i++) {
      const a = -Math.PI * 0.58 + (i / 8) * Math.PI * 1.16;
      const r0 = size * 0.07;
      const r1 = size * 0.40 + random() * size * 0.07;
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0;
      const x1 = cx + Math.cos(a) * r1;
      const y1 = cy + Math.sin(a) * r1;
      drawVein(x0, y0, x1, y1, 1.45 + random() * 1.15, 0.18 + random() * 0.10, 2);
    }
    // Secondary capillary network — finer veins
    for (let i = 0; i < 24; i++) {
      const a = -Math.PI * 0.72 + random() * Math.PI * 1.44;
      const r0 = size * (0.12 + random() * 0.15);
      const r1 = size * (0.30 + random() * 0.15);
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0 * 0.94;
      const x1 = cx + Math.cos(a) * r1;
      const y1 = cy + Math.sin(a) * r1 * 0.96;
      ctx.strokeStyle = `rgba(192,108,104,${0.11 + random() * 0.08})`;
      ctx.lineWidth = 0.55 + random() * 0.75;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      const mx = (x0 + x1) * 0.5 + (random() - 0.5) * 14;
      const my = (y0 + y1) * 0.5 + (random() - 0.5) * 10;
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
    // Neutral rosy-ivory base so strain noseColor modulates cleanly from pink to slate-rose.
    const g = ctx.createRadialGradient(cx, cy, size * 0.03, cx, cy, size * 0.50);
    g.addColorStop(0.0, '#f0beb8');
    g.addColorStop(0.46, '#e0a6a0');
    g.addColorStop(1.0, '#b88680');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    // Warm capillary flush around nostril apertures.
    const ring = ctx.createRadialGradient(cx, cy, size * 0.05, cx, cy, size * 0.22);
    ring.addColorStop(0.0, 'rgba(155,48,52,0.12)');
    ring.addColorStop(0.55, 'rgba(176,62,64,0.22)');
    ring.addColorStop(1.0, 'rgba(120,30,30,0.0)');
    ctx.fillStyle = ring;
    ctx.fillRect(0, 0, size, size);
    // Median philtrum sulcus shading
    ctx.strokeStyle = 'rgba(118,58,58,0.38)';
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
  // Two-layer eye with subtle iris stroma depth + tear-film cornea
  const eyeMat = new THREE.MeshPhysicalMaterial({
    color: 0x050404, map: eyeIrisMap, roughness: 0.14, metalness: 0.0,
    clearcoat: 0.92, clearcoatRoughness: 0.05, envMapIntensity: 0.95,
  });
  const corneaMat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff, transparent: true, opacity: 0.11,
    roughness: 0.01, metalness: 0.0, ior: 1.376,
    clearcoat: 1.0, clearcoatRoughness: 0.02, envMapIntensity: 1.35,
  });

  function createCoatMaterial(furColor, skinColor, density, params) {
    const fur = new THREE.Color(furColor);
    const skin = new THREE.Color(skinColor);
    const hsl = {}; fur.getHSL(hsl);
    const isPale = hsl.l > 0.65;
    const isHairless = density <= 0;
    const material = new THREE.MeshPhysicalMaterial({
      // The albedo map already contains the strain pigment. Keeping the
      // material tint neutral avoids multiplying dark coats into black.
      color: 0xffffff,
      roughness: isHairless ? 0.66 : (isPale ? 0.88 : 0.92),
      envMapIntensity: isHairless ? 0.36 : (isPale ? 0.34 : 0.32),
      metalness: 0.0,
      clearcoat: isHairless ? 0.04 : 0.0,
      clearcoatRoughness: isHairless ? 0.56 : 0.78,
      sheen: isHairless ? 0.16 : (isPale ? 0.32 : 0.38),
      sheenColor: (isHairless ? skin : fur).clone().offsetHSL(0.0, -0.02, isPale ? 0.03 : 0.07),
      sheenRoughness: isHairless ? 0.62 : 0.68,
      map: createCoatAlbedoMap(isHairless ? skinColor : furColor),
      normalMap: createCoatNormalMap(),
      normalScale: isHairless
        ? new THREE.Vector2(0.05, 0.06)
        : new THREE.Vector2(0.09 + density * 0.06, 0.10 + density * 0.07),
      roughnessMap: createCoatRoughnessMap(),
    });
    if (!isHairless && annolidShaders?.applyFurShader && params?.enableFurShader !== false) {
      const rootCol = fur.clone().offsetHSL(0.0, 0.0, isPale ? -0.04 : -0.03);
      const tipCol = fur.clone().offsetHSL(0.006, -0.02, isPale ? 0.03 : 0.06);
      annolidShaders.applyFurShader(material, {
        anisotropy: isPale ? 32.0 : 28.0,
        specIntensity: isPale ? 0.035 : 0.045,
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
        strength: Math.min(0.024, (params.microStrength ?? 0.06) * 0.32)
      });
    }
    return material;
  }

  function enhanceFurMaterial(material, furColor, isUndercoat = false) {
    const base = new THREE.Color(furColor);
    const hsl = {}; base.getHSL(hsl);
    const isPale = hsl.l > 0.65;
    material.color.copy(base).offsetHSL(
      0.0,
      isPale ? -0.005 : -0.01,
      isUndercoat ? (isPale ? -0.012 : -0.018) : (isPale ? 0.020 : 0.016)
    );
    material.transparent = false;
    material.depthWrite = true;
    material.envMapIntensity = isUndercoat ? 0.18 : 0.42;
    if (material.isMeshPhysicalMaterial) {
      material.sheen = isUndercoat ? 0.26 : 0.72;
      material.sheenColor = base.clone().offsetHSL(0.01, -0.03, isPale ? 0.08 : 0.12);
      material.sheenRoughness = isUndercoat ? 0.86 : 0.56;
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
    // Sleek murine pear silhouette: muscular haunches, supple lumbar waist, narrower chest.
    const rearMass = Math.exp(-Math.pow((zNorm - 0.26) / 0.24, 2.0));
    let shoulderDrop = THREE.MathUtils.clamp((zNorm - 0.66) / 0.34, 0, 1);
    shoulderDrop = shoulderDrop * shoulderDrop * (3.0 - 2.0 * shoulderDrop);
    const waistNarrow = Math.exp(-Math.pow((zNorm - 0.56) / 0.16, 2.0));
    const frontNarrow = THREE.MathUtils.smoothstep(zNorm, 0.64, 0.94) * 0.055;
    const lateral = (
      0.98 + rearMass * 0.20 - shoulderDrop * 0.065
      - waistNarrow * (1.0 - PROP.waistTaper) * 0.20 - frontNarrow
    );
    v.x *= lateral * PROP.bodyGirthX;
    v.y *= (0.97 + rearMass * 0.08 - zNorm * 0.11) * PROP.bodyGirthY;
    v.z *= PROP.bodyLength;
    // Smooth elliptical flank curvature so top view shows zero hard shelf.
    const sideBlend = THREE.MathUtils.clamp((Math.abs(v.y) - 0.06) / 0.72, 0, 1);
    v.x *= 0.95 + sideBlend * 0.05;

    // Dorsal thoracolumbar arch: natural murine kyphotic curve peaking over lumbar spine.
    const hump = Math.exp(-Math.pow((zNorm - 0.35) / 0.25, 2.0));
    const rump = Math.exp(-Math.pow((zNorm - 0.11) / 0.11, 2.0));
    const withers = Math.exp(-Math.pow((zNorm - 0.74) / 0.16, 2.0));
    v.y += hump * 0.25 + rump * 0.075 + withers * 0.115 - shoulderDrop * 0.022;

    // Ventrum: tucked inguinal groin and streamlined ribcage underside.
    const yRef = v.y;
    const bellyWidth = Math.exp(-Math.pow((zNorm - 0.48) / 0.28, 2.0));
    const bellyMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.65, 0.16);
    const bellyInfluence = bellyWidth * bellyMask;
    v.x *= 1.0 + bellyInfluence * 0.13;
    const dropMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.22, 0.22);
    v.y -= bellyWidth * dropMask * 0.11;

    const lowerBlend = THREE.MathUtils.clamp((0.10 - yRef) / 0.46, 0, 1);
    const groinTuck = Math.exp(-Math.pow((zNorm - 0.18) / 0.15, 2.0)) * 0.12;
    let chestDrop = THREE.MathUtils.clamp((zNorm - 0.72) / 0.28, 0, 1);
    chestDrop = chestDrop * chestDrop * (3.0 - 2.0 * chestDrop) * 0.045;
    const targetY = v.y * 0.78 + groinTuck - chestDrop;
    v.y = THREE.MathUtils.lerp(v.y, targetY, lowerBlend);

    // Subtle scapular and gluteal-haunch muscular relief.
    if (zNorm > 0.68 && zNorm < 0.86 && Math.abs(v.x) > 0.26) {
      const scapBulge = 1.0 - Math.abs(zNorm - 0.76) * 10.0;
      if (scapBulge > 0) { v.x *= 1.0 + scapBulge * 0.042; if (v.y > 0) v.y += scapBulge * 0.032; }
    }
    if (zNorm > 0.14 && zNorm < 0.36 && Math.abs(v.x) > 0.26) {
      const hB = 1.0 - Math.abs(zNorm - 0.25) * 6.0;
      if (hB > 0) { v.x *= 1.0 + hB * 0.125; if (v.y < 0) v.y *= 1.0 + hB * 0.055; }
    }

    // Tail base integration: gradual sacral taper into caudal root
    if (zNorm < 0.10) {
      const tailBlend = zNorm / 0.10;
      v.x *= 0.68 + tailBlend * 0.32;
      v.y *= 0.78 + tailBlend * 0.22;
    }

    // Taper the anterior thorax inside the cervical mantle. Without this
    // transition the body sphere wraps around the skull as a visible ring.
    if (zNorm > 0.62) {
      const neckT = THREE.MathUtils.smoothstep(zNorm, 0.62, 0.98);
      v.x *= 1.0 - neckT * 0.40;
      v.y *= 1.0 - neckT * (v.y >= 0 ? 0.25 : 0.10);
    }
  };

  const deformHead = (v) => {
    const zNorm = (v.z + 1) / 2;
    const snoutTaper = Math.pow(1.0 - zNorm * 0.58, 1.04);
    v.x *= snoutTaper * PROP.snoutWidth;
    v.y *= snoutTaper * 1.00;
    v.z *= PROP.snoutLength;
    v.y -= zNorm * 0.145;
    // Narrow the occiput into the neck, then add cheek mass around the
    // zygomatic region instead of inflating the entire rear hemisphere.
    const occipitalBlend = THREE.MathUtils.smoothstep(zNorm, 0.04, 0.68);
    v.x *= 0.90 + occipitalBlend * 0.10;
    if (v.y > 0) v.y *= 0.93 + occipitalBlend * 0.07;
    const cheek = Math.exp(-Math.pow((zNorm - 0.54) / 0.17, 2.0));
    v.x *= 1.0 + cheek * PROP.cheekFullness * 0.32;
    if (v.y < 0) {
      // Taper lower mandible toward chin like a true rodent jaw
      const jawTuck = THREE.MathUtils.smoothstep(zNorm, 0.52, 0.95);
      v.x *= (1.0 + cheek * 0.03) * (1.0 - jawTuck * 0.16);
      v.y *= 1.0 - jawTuck * 0.10;
    }
    // Gentle orbital fossa concavity + soft brow contour so eyes sit naturally without harsh cheek creases
    const orbitZ = Math.exp(-Math.pow((zNorm - 0.69) / 0.13, 2.0));
    const orbitY = Math.exp(-Math.pow((v.y - 0.15) / 0.20, 2.0));
    const latMask = THREE.MathUtils.smoothstep(Math.abs(v.x), 0.20, 0.48);
    v.x *= 1.0 - orbitZ * orbitY * latMask * 0.032;
    const browY = Math.exp(-Math.pow((v.y - 0.31) / 0.12, 2.0));
    v.x *= 1.0 + orbitZ * browY * latMask * 0.024;
    v.y += orbitZ * browY * latMask * 0.012;

    // Mystacial vibrissal pad cushions — smooth bilateral fleshy whisker pads on rostrum
    if (zNorm > 0.60 && zNorm < 0.92 && Math.abs(v.x) > 0.08) {
      const padBulge = Math.exp(-Math.pow((zNorm - 0.76) / 0.11, 2.0));
      const latBulge = Math.exp(-Math.pow((Math.abs(v.x) - 0.24) / 0.16, 2.0));
      const yMask = Math.exp(-Math.pow((v.y + 0.06) / 0.18, 2.0));
      v.x *= 1.0 + padBulge * latBulge * yMask * 0.068;
    }
    // Refined pointed murine rostrum wedge
    if (zNorm > 0.64) {
      const p = (zNorm - 0.64) / 0.36;
      v.x *= 1.0 - p * 0.22;
      v.y *= 1.0 - p * 0.13;
      v.y -= p * 0.032;
    }
    if (zNorm > 0.84) {
      const tip = (zNorm - 0.84) / 0.16;
      v.x *= 1.0 + tip * 0.04;
      v.y *= 1.0 + tip * 0.025;
    }
    // One continuous dorsal skull slope from the occiput into the snout.
    // A smooth bell avoids a raised rear dome that self-occludes the forehead.
    if (zNorm > 0.04 && zNorm < 0.78 && v.y > 0) {
      const cranium = Math.exp(-Math.pow((zNorm - 0.28) / 0.27, 2.0));
      v.y += cranium * 0.135;
    }
    // Temporal fullness supports the pinna root without making the forehead round.
    if (zNorm > 0.10 && zNorm < 0.46 && v.y > -0.16) {
      const temporal = Math.max(0, 1.0 - Math.abs(zNorm - 0.28) * 5.0);
      const dorsal = THREE.MathUtils.smoothstep(v.y, -0.16, 0.58);
      v.x *= 1.0 + temporal * dorsal * 0.052;
      if (v.y > 0.18) v.y += temporal * 0.014;
    }
    // Subtle zygomatic arch / orbital rim definition
    if (zNorm > 0.34 && zNorm < 0.56 && Math.abs(v.x) > 0.26) {
      v.y += 0.032;
      v.x *= 1.0 + 0.025;
    }
  };

  function getCervicalMantleProfile(t) {
    const eased = t * t * (3.0 - 2.0 * t);
    const arch = Math.sin(t * Math.PI);
    return {
      z: (t - 0.5) * PROP.neckLength,
      centerY: THREE.MathUtils.lerp(-0.02, 0.18, eased) + arch * 0.018,
      radiusX: THREE.MathUtils.lerp(
        PROP.neckGirth * 1.10,
        PROP.neckGirth * 1.04,
        eased
      ) + arch * 0.035,
      radiusY: THREE.MathUtils.lerp(
        PROP.neckGirth * 1.10,
        PROP.neckGirth * 1.04,
        eased
      ) + arch * 0.035,
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

  function createCervicalCoreGeometry() {
    const rings = 14;
    const radialSegments = 28;
    const row = radialSegments + 1;
    const positions = [];
    const uvs = [];
    const indices = [];
    for (let ring = 0; ring <= rings; ring++) {
      const rawT = ring / rings;
      const t = rawT * 0.86;
      const profile = getCervicalMantleProfile(t);
      // Recessed well inside the cervical fur collar so it blocks background light
      // bleed between fur strands while never protruding past occipital/cheek fur.
      const endPinch = rawT > 0.88 ? (1.0 - (rawT - 0.88) / 0.12 * 0.55) : 1.0;
      const inset = THREE.MathUtils.lerp(0.84, 0.58, rawT) * endPinch;
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
    const lowerOpen = THREE.MathUtils.smoothstep(y, -0.90, -0.10);
    const crownTaper = 1.0 - THREE.MathUtils.smoothstep(y, 0.34, 0.96) * 0.12;
    const rootPad = Math.exp(-(x * x * 5.0 + Math.pow(y + 0.76, 2.0) * 15.0));
    const shoulder = Math.exp(-Math.pow((y - 0.04) / 0.72, 2.0));
    const rim = THREE.MathUtils.smoothstep(radius, 0.82, 1.0);
    const centerCup = Math.max(0.0, 1.0 - radius * radius);
    const upperFold = Math.exp(-(
      Math.pow(x + 0.12, 2.0) * 5.0 + Math.pow(y - 0.24, 2.0) * 8.0
    ));
    const basalFold = Math.exp(-(
      Math.pow(x + 0.20, 2.0) * 9.0 + Math.pow(y + 0.48, 2.0) * 14.0
    ));
    const attachmentBlend = THREE.MathUtils.smoothstep(-y, 0.58, 0.96);

    const px = x * (0.90 + shoulder * 0.045)
      * (0.58 + lowerOpen * 0.42) * crownTaper;
    const attachmentY = -0.70 + Math.abs(x) * 0.035;
    const py = THREE.MathUtils.lerp(y * 0.94, attachmentY, attachmentBlend)
      + rootPad * 0.020;
    const pz = (
      -centerCup * 0.112 + rim * 0.046 + upperFold * 0.028
      + basalFold * 0.038 + rootPad * 0.020
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
            const shellThickness = 0.026
              + (1.0 - normalizedRadius) * 0.010
              + rootThickness * 0.018;
            point.z -= shellThickness;
          }
          positions.push(point.x, point.y, point.z);
          uvs.push(0.5 + Math.cos(angle) * normalizedRadius * 0.5, 0.5 + Math.sin(angle) * normalizedRadius * 0.5);
        }
      }
    }

    for (let ring = 0; ring < rings; ring++) {
      for (let seg = 0; seg < segments; seg++) {
        const frontA = ring * row + seg;
        const frontB = frontA + row;
        indices.push(frontA, frontB, frontA + 1, frontB, frontB + 1, frontA + 1);
        const backA = surfaceVertexCount + frontA;
        const backB = surfaceVertexCount + frontB;
        indices.push(backA, backA + 1, backB, backB, backA + 1, backB + 1);
      }
    }

    const outerFront = rings * row;
    const outerBack = surfaceVertexCount + outerFront;
    for (let seg = 0; seg < segments; seg++) {
      const frontA = outerFront + seg;
      const frontB = frontA + 1;
      const backA = outerBack + seg;
      const backB = backA + 1;
      indices.push(frontA, backA, frontB, frontB, backA, backB);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    return geometry;
  }

  function createPinnaSurfaceGeometry(radiusScale = 0.76, depthMul = 1.0, zLift = 0.012) {
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
        // The auricular rim terminates into the skull instead of closing
        // across the lower attachment like a manufactured ring.
        const angle = -Math.PI * 0.22 + t * Math.PI * 1.45;
        const endFade = THREE.MathUtils.smoothstep(t, 0.0, 0.10)
          * THREE.MathUtils.smoothstep(1.0 - t, 0.0, 0.10);
        const radius = band === 0 ? 1.0 - 0.027 * endFade : 1.0;
        const point = shapePinnaPoint(
          Math.cos(angle) * radius,
          Math.sin(angle) * radius,
          1.0,
          0.006 + endFade * 0.004
        );
        positions.push(point.x, point.y, point.z + (band === 0 ? 0.001 : 0.0));
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
  const earInnerGeo = createPinnaSurfaceGeometry(0.945, 1.08, 0.018);
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
    copyGeometryPositionsInPlace(neckGeo, createCervicalMantleGeometry());
    copyGeometryPositionsInPlace(cervicalCoreGeo, createCervicalCoreGeometry());
  }


  // ═══════════════════════════════════════════════════════════════
  // 6. PAW BUILDER (anatomically correct forepaw vs hindpaw)
  // ═══════════════════════════════════════════════════════════════
  function createPaw(mat, isFore = false, strain = {}) {
    const paw = new THREE.Group();
    paw.name = isFore ? 'connected-forepaw' : 'connected-hindpaw';
    const pawFleshMat = mat.clone();
    pawFleshMat.color.set(strain.pawColor ?? '#e6b0a7');
    pawFleshMat.transmission = 0.10;
    pawFleshMat.thickness = 0.10;
    pawFleshMat.roughness = 0.54;
    pawFleshMat.clearcoat = 0.06;
    pawFleshMat.clearcoatRoughness = 0.42;
    pawFleshMat.sheen = 0.25;
    pawFleshMat.sheenColor = new THREE.Color(strain.pawColor ?? '#e6b0a7').offsetHSL(0, -0.04, 0.10);
    const padMat = mat.clone();
    padMat.color.set(strain.pawPadColor ?? strain.pawColor ?? '#e9b8ae');
    padMat.color.multiplyScalar(strain.padDarken ?? 0.9);
    padMat.transmission = 0.08; padMat.thickness = 0.10;
    padMat.roughness = 0.52; padMat.clearcoat = 0.06; padMat.clearcoatRoughness = 0.44;
    padMat.normalMap = padNormalMap;
    padMat.normalScale = new THREE.Vector2(0.22, 0.22);

    const pawScale = isFore ? (PROP.forePawSize) : (PROP.hindPawSize);
    const rootLength = isFore ? 0.16 : 0.30;
    const rootBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(isFore ? 0.056 : 0.068, rootLength, 10, 12),
      pawFleshMat
    );
    rootBridge.name = isFore ? 'forepaw-root-bridge' : 'hindpaw-root-bridge';
    rootBridge.rotation.x = Math.PI / 2 - (isFore ? 0.04 : 0.10);
    rootBridge.position.set(0, 0.010, isFore ? -0.072 : -0.140);
    rootBridge.scale.set(1.04, 0.72, 1.0);
    rootBridge.castShadow = true;
    paw.add(rootBridge);

    if (isFore) {
      const dorsalHand = new THREE.Mesh(new THREE.SphereGeometry(0.122, 18, 18), pawFleshMat);
      dorsalHand.name = 'forepaw-dorsal-hand';
      dorsalHand.scale.set(0.90 * pawScale, 0.36, 1.16 * pawScale);
      dorsalHand.position.set(0, -0.004, 0.055);
      dorsalHand.castShadow = true;
      paw.add(dorsalHand);

      // Vestigial medial pollex (thumb) nub characteristic of murine manus
      const pollex = new THREE.Mesh(new THREE.SphereGeometry(0.024, 8, 8), pawFleshMat);
      pollex.scale.set(0.85, 0.68, 1.15);
      pollex.position.set(-0.088 * pawScale, -0.028, 0.032);
      paw.add(pollex);

      const palm = new THREE.Mesh(new THREE.SphereGeometry(0.086, 16, 16), padMat);
      palm.scale.set(0.88 * pawScale, 0.28, 1.02 * pawScale);
      palm.position.set(0, -0.056, 0.065);
      paw.add(palm);

      const carpalPad = new THREE.Mesh(new THREE.SphereGeometry(0.046, 12, 12), padMat);
      carpalPad.scale.set(0.92, 0.38, 1.14);
      carpalPad.position.set(0, -0.048, -0.045);
      paw.add(carpalPad);

      for (let p = 0; p < 3; p++) {
        const iPad = new THREE.Mesh(new THREE.SphereGeometry(0.032, 10, 10), padMat);
        iPad.scale.set(0.90 * pawScale, 0.40, 0.98 * pawScale);
        iPad.position.set((p - 1) * 0.040 * pawScale, -0.059, 0.124);
        paw.add(iPad);
      }
    } else {
      const dorsalFoot = new THREE.Mesh(new THREE.SphereGeometry(0.148, 20, 20), pawFleshMat);
      dorsalFoot.name = 'hindpaw-dorsal-foot';
      dorsalFoot.scale.set(0.82 * pawScale, 0.32, 1.68 * pawScale);
      dorsalFoot.position.set(0, -0.006, 0.068);
      dorsalFoot.castShadow = true;
      paw.add(dorsalFoot);

      // Subtle dorsal metatarsal extensor tendon ridges along elongated pes
      for (let r = -1; r <= 1; r++) {
        const tendon = new THREE.Mesh(
          new THREE.CapsuleGeometry(0.012, 0.24 * pawScale, 6, 8),
          pawFleshMat
        );
        tendon.rotation.x = Math.PI / 2 - 0.04;
        tendon.rotation.y = r * 0.045;
        tendon.scale.set(0.85, 1.0, 0.55);
        tendon.position.set(r * 0.032 * pawScale, 0.030, 0.065);
        paw.add(tendon);
      }

      const sole = new THREE.Mesh(new THREE.SphereGeometry(0.125, 18, 18), padMat);
      sole.scale.set(0.82 * pawScale, 0.24, 1.58 * pawScale);
      sole.position.set(0, -0.062, 0.075);
      paw.add(sole);

      const heelPad = new THREE.Mesh(new THREE.SphereGeometry(0.064, 12, 12), padMat);
      heelPad.scale.set(0.86, 0.34, 1.02);
      heelPad.position.set(0, -0.052, -0.145 * pawScale);
      paw.add(heelPad);

      const midPad = new THREE.Mesh(new THREE.SphereGeometry(0.060, 12, 12), padMat);
      midPad.scale.set(0.84, 0.32, 1.20);
      midPad.position.set(0, -0.066, 0.145 * pawScale);
      paw.add(midPad);
    }

    const pawSpreadMul = strain.pawSpreadMul ?? 1.0;
    const toeCount = isFore ? 4 : 5;
    const baseSpread = isFore ? 0.056 : 0.049;
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
      if (isFore && (i === 0 || i === 3)) lenMul = 0.78;
      if (!isFore && (i === 0 || i === 4)) lenMul = 0.72;
      const toeLen = (isFore ? 0.172 : 0.236)
        * (strain.toeLenMul ?? 1.0) * lenMul * pawScale;
      const toeRadius = isFore ? 0.021 : 0.023;
      const phalanx = new THREE.Mesh(
        new THREE.CapsuleGeometry(toeRadius, toeLen, 10, 10),
        pawFleshMat
      );
      phalanx.position.z = toeLen * 0.48;
      phalanx.rotation.x = Math.PI / 2 - (isFore ? 0.08 : 0.11);
      phalanx.scale.set(0.94, 1.0, 0.88);
      phalanx.castShadow = true;
      toe.add(phalanx);

      const knuckle = new THREE.Mesh(
        new THREE.SphereGeometry(isFore ? 0.025 : 0.027, 10, 10),
        pawFleshMat
      );
      knuckle.scale.set(1.02, 0.78, 1.12);
      knuckle.position.set(0, 0.003, 0.016);
      toe.add(knuckle);

      const toeTip = new THREE.Mesh(new THREE.SphereGeometry(isFore ? 0.018 : 0.020, 10, 10), pawFleshMat);
      toeTip.scale.set(1.12, 0.82, 1.05);
      toeTip.position.set(0, isFore ? -0.011 : -0.016, toeLen + toeRadius * 0.50);
      toe.add(toeTip);

      const toePad = new THREE.Mesh(new THREE.SphereGeometry(isFore ? 0.028 : 0.034, 10, 10), padMat);
      toePad.scale.set(isFore ? 1.05 : 0.96, isFore ? 0.44 : 0.36, isFore ? 0.96 : 1.05);
      toePad.position.set(0, isFore ? -0.036 : -0.044, toeLen * 0.68);
      toe.add(toePad);

      const clawLength = isFore ? 0.044 : 0.054;
      const clawRadius = isFore ? 0.0072 : 0.0082;
      const clawGeo = new THREE.CapsuleGeometry(clawRadius, clawLength, 6, 6);
      clawGeo.translate(0, clawLength / 2, 0);
      deform(clawGeo, (cv) => {
        const yN = THREE.MathUtils.clamp(
          (cv.y + clawRadius) / (clawLength + clawRadius * 2),
          0,
          1
        );
        cv.x *= 1.0 - yN * 0.52;
        cv.z *= 1.0 - yN * 0.42;
        // Natural downward falcula curvature toward claw tip
        cv.z -= yN * yN * 0.012;
      });
      const claw = new THREE.Mesh(clawGeo, clawLocalMat);
      claw.rotation.x = Math.PI / 2 + (isFore ? 0.16 : 0.14);
      claw.position.set(0, isFore ? -0.012 : -0.016, toeLen + toeRadius * 0.58);
      claw.castShadow = true;
      toe.add(claw);

      const zPos = (isFore ? 0.132 - arc * 0.020 : 0.202 - arc * 0.028) * pawScale;
      toe.position.set(t, isFore ? -0.018 + arc * 0.005 : -0.024 + arc * 0.005, zPos);
      toe.rotation.y = centeredIndex * (isFore ? 0.052 : 0.040);
      toe.rotation.x = isFore ? -0.050 - arc * 0.030 : -0.095 - arc * 0.035;
      paw.add(toe);
    }

    // Invisible sole landmark used by the stance-phase IK solver. Keeping it
    // inside the paw hierarchy lets the solver preserve every visible joint.
    const groundContact = new THREE.Object3D();
    groundContact.name = isFore ? 'forepaw-ground-contact' : 'hindpaw-ground-contact';
    groundContact.position.set(0, isFore ? -0.090 : -0.100, isFore ? 0.29 : 0.42);
    paw.add(groundContact);
    paw._groundContact = groundContact;
    if (isFore) {
      const groomingContact = new THREE.Object3D();
      groomingContact.name = 'forepaw-palmar-contact';
      groomingContact.position.set(0, -0.052, 0.17);
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
    // Tucked shoulder anchor: proximal joints remain inside the chest coat.
    chain.position.set(sign * 0.52 * PROP.shoulderWidth / 0.90, -0.08, PROP.bodyLength * 0.70);

    const shoulderMantle = new THREE.Mesh(new THREE.SphereGeometry(0.195, 16, 16), surfaceMat);
    shoulderMantle.position.set(-sign * 0.11, 0.03, -0.10);
    shoulderMantle.scale.set(1.14, 0.98, 1.32);
    shoulderMantle.castShadow = false;
    chain.add(shoulderMantle);
    // Pectoral-brachial skin fold enveloping the proximal arm into the sternum
    const pectoralFold = new THREE.Mesh(new THREE.SphereGeometry(0.155 * thickness, 14, 14), surfaceMat);
    pectoralFold.name = `${side}-pectoral-brachial-fold`;
    pectoralFold.position.set(-sign * 0.075, -0.10, -0.03);
    pectoralFold.scale.set(1.10, 1.32, 1.24);
    pectoralFold.castShadow = false;
    chain.add(pectoralFold);
    const scapularBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.105 * thickness, 0.22, 10, 10),
      surfaceMat
    );
    scapularBridge.name = `${side}-scapular-soft-tissue-bridge`;
    scapularBridge.position.set(-sign * 0.06, 0.05, -0.10);
    scapularBridge.rotation.x = Math.PI / 2;
    scapularBridge.castShadow = false;
    chain.add(scapularBridge);

    // Adult murine radius/ulna is slightly longer than the humerus. The
    // remaining scale belongs to the carpus and paw, represented below.
    const upperArm = new THREE.Group();
    upperArm.name = `${side}-upper-arm`;
    const uaLen = 0.42 * fl;
    const upperArmMesh = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.118 * thickness, uaLen, 10, 10), surfaceMat
    );
    upperArmMesh.position.y = -uaLen * 0.5;
    upperArmMesh.castShadow = true;
    upperArm.add(upperArmMesh);
    upperArm.rotation.x = FORELIMB_NEUTRAL.shoulderX;
    // Shoulder joint sphere (internal smoothing without harsh self-shadow ring)
    const shoulderJoint = new THREE.Mesh(new THREE.SphereGeometry(0.11 * thickness, 12, 12), surfaceMat);
    shoulderJoint.scale.set(1.0, 0.84, 1.05);
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
      new THREE.CapsuleGeometry(0.092 * thickness, 0.085, 10, 10),
      surfaceMat
    );
    elbowBridge.position.set(0, -uaLen * 0.92, 0.03);
    elbowBridge.rotation.x = 0.15;
    elbowBridge.castShadow = false;
    upperArm.add(elbowBridge);
    // Elbow joint (flush radius matching forearm taper so front view shows sleek continuous arm)
    const elbowJoint = new THREE.Mesh(new THREE.SphereGeometry(0.073 * thickness, 10, 10), surfaceMat);
    elbowJoint.scale.set(0.98, 0.94, 0.98);
    elbowJoint.castShadow = false;
    forearm.add(elbowJoint);
    const forearmMesh = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.068 * thickness, faLen, 8, 8), surfaceMat
    );
    forearmMesh.position.y = -faLen * 0.5;
    forearmMesh.castShadow = true;
    forearm.add(forearmMesh);
    upperArm.add(forearm);

    // Wrist joint — slender furred carpus tapering smoothly into pink dorsal hand
    const wrist = new THREE.Group();
    wrist.name = `${side}-wrist`;
    wrist.position.set(0, -faLen - 0.018, 0.03);
    const wristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.048 * thickness, 8, 8), surfaceMat);
    wristJoint.scale.set(0.98, 0.90, 0.98);
    wristJoint.castShadow = false;
    wrist.add(wristJoint);
    const wristCuff = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.047 * thickness, 0.072, 8, 8),
      surfaceMat
    );
    wristCuff.position.set(sign * 0.012, -0.042, 0.028);
    wristCuff.rotation.x = -0.32;
    wristCuff.castShadow = false;
    wrist.add(wristCuff);
    forearm.add(wrist);

    // Paw — sits at bottom of wrist
    const paw = createPaw(skinMat, true, strain);
    paw.position.set(sign * 0.035, -0.13, 0.07);
    paw.scale.setScalar(0.96 * Math.sqrt(thickness));
    paw.rotation.x = -0.12;
    wrist.add(paw);

    // Continuous coat along the limb from pectoral fold down to carpal cuff.
    const coatCapsule = (mesh, radius, span, count, hairLength) => {
      const coat = generateFurLayer(v => {
        const y = v.y;
        v.multiplyScalar(radius);
        v.y += y * span * 0.5;
      }, Math.floor(count * density), furColor, hairLength, false);
      mesh.add(coat);
    };
    coatCapsule(upperArmMesh, 0.119 * thickness, uaLen, 3400, 0.068);
    coatCapsule(forearmMesh, 0.073 * thickness, faLen, 2400, 0.046);
    coatCapsule(wristCuff, 0.055 * thickness, 0.07, 650, 0.032);
    coatCapsule(shoulderMantle, 0.196, 0, 1600, 0.064);
    coatCapsule(pectoralFold, 0.156 * thickness, 0.04, 1400, 0.068);

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
    const shankMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.066 * thickness, shankLen, 8, 8), surfaceMat);
    shankMesh.rotation.x = -0.16;
    shankMesh.position.set(0, -shankLen * 0.5, 0.09);
    shankMesh.castShadow = true;
    shank.add(shankMesh);
    shankMesh.add(generateFurLayer(v => {
      const y = v.y;
      v.multiplyScalar(0.067 * thickness);
      v.y += y * shankLen * 0.5;
    }, Math.floor(2400 * density), furColor, 0.052, false));

    // Ankle/hock (tarsocrural joint + sleek calcaneal tuber)
    const ankle = new THREE.Group();
    ankle.name = `${side}-ankle`;
    ankle.position.set(0, -shankLen - 0.04, 0.18);
    shank.add(ankle);

    const ankleJoint = new THREE.Mesh(new THREE.SphereGeometry(0.066 * thickness, 10, 10), surfaceMat);
    ankleJoint.castShadow = false;
    ankle.add(ankleJoint);

    // Calcaneus (hock): compact tucked tuber rather than a protruding sphere
    const heel = new THREE.Mesh(new THREE.CapsuleGeometry(0.052 * thickness, 0.075, 8, 8), surfaceMat);
    heel.rotation.x = -0.42;
    heel.position.set(0, 0.018, -0.032);
    heel.castShadow = false;
    ankle.add(heel);
    heel.add(generateFurLayer(v => {
      const y = v.y;
      v.multiplyScalar(0.053 * thickness);
      v.y += y * 0.036;
    }, Math.floor(600 * density), furColor, 0.038, false));

    // Tarsometatarsal bridge routed collinearly from hock (0,-0.01,0) directly
    // into the proximal dorsal foot (sign*0.038, -0.185, 0.20) with paw-matched pigment.
    const metaMat = skinMat.clone();
    metaMat.color.set(strain.pawColor ?? '#e6b0a7').lerp(new THREE.Color(furColor), 0.12);
    metaMat.roughness = 0.56;
    metaMat.transmission = 0.08;
    metaMat.sheen = 0.24;
    metaMat.sheenColor = new THREE.Color(strain.pawColor ?? '#e6b0a7');
    const metaLen = 0.24 * PROP.hindPawSize;
    const metaMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.046 * thickness, metaLen, 8, 8), metaMat);
    metaMesh.rotation.x = Math.PI / 2 + 0.64;
    metaMesh.rotation.z = -sign * 0.14;
    metaMesh.position.set(sign * 0.022, -0.102, 0.105);
    metaMesh.scale.set(0.96, 1.0, 0.78);
    metaMesh.castShadow = true;
    ankle.add(metaMesh);

    // Hindpaw — planted on ground
    const paw = createPaw(skinMat, false, strain);
    paw.position.set(sign * 0.045, -0.20, 0.31);
    paw.rotation.x = -0.10;
    paw.scale.setScalar(0.92 * Math.sqrt(thickness));
    ankle.add(paw);

    // Baseline crouched posture so underside view reads organic, not columnar.
    chain.rotation.x = -0.18;
    shank.rotation.x = 0.50;
    ankle.rotation.x = -0.28;

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
      const shade = 0.80 + progress * 0.30;
      colors.set([shade, shade, shade], i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const mat = isUndercoat
      ? new THREE.MeshStandardMaterial({
        vertexColors: true,
        color: 0xffffff,
        roughness: 0.93,
      })
      : new THREE.MeshPhysicalMaterial({
        vertexColors: true,
        color: 0xffffff, roughness: 0.80,
        clearcoat: 0.0,
        sheen: 0.64,
        sheenColor: new THREE.Color(colorHex),
        sheenRoughness: 0.62,
      });
    enhanceFurMaterial(mat, colorHex, isUndercoat);
    const inst = new THREE.InstancedMesh(geo, mat, count);
    inst.castShadow = false;
    inst.receiveShadow = !isUndercoat;
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
        // Face: sleek caudal sweep from nasal bridge over forehead and cheeks
        if (preZ > 0.3) flowDir = new THREE.Vector3(normal.x * 0.24, -0.08, -0.78).normalize();
        else flowDir = new THREE.Vector3(normal.x * 0.28, normal.y * 0.18, -1).normalize();
      } else {
        // Body: region-specific fur flow with subtle natural lock clumping
        const clump = Math.sin(source.x * 19.0 + preZ * 15.0) * (0.08 * clumpScale);
        const isHaunch = preZ < -0.3;
        const isShoulder = preZ > 0.5;
        const isDorsal = preY > 0.3;
        const isVentral = preY < -0.3;
        const isTailBase = preZ < -0.7;
        if (isTailBase) {
          flowDir = new THREE.Vector3(normal.x * 0.15, -0.08, -0.92).normalize();
        } else if (isHaunch) {
          flowDir = new THREE.Vector3(normal.x * 0.55 + clump, normal.y * 0.18 - 0.12, -0.84).normalize();
        } else if (isShoulder) {
          flowDir = new THREE.Vector3(normal.x * 0.32, normal.y * 0.14 - 0.12, -0.92).normalize();
        } else if (isDorsal) {
          flowDir = new THREE.Vector3(normal.x * 0.18 + clump * 0.5, 0.04, -0.96).normalize();
        } else if (isVentral) {
          flowDir = new THREE.Vector3(normal.x * 0.62, normal.y * 0.42 - 0.18, -0.68).normalize();
        } else {
          flowDir = new THREE.Vector3(normal.x * 0.45 + clump, normal.y * 0.26 - 0.14, -1).normalize();
        }
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
        const eyeDist = Math.hypot(
          (Math.abs(pos.x) - PROP.eyeSpacing) / 0.16,
          (pos.y - 0.15) / 0.15,
          (pos.z - PROP.eyeYaw) / 0.17
        );
        if (eyeDist < 1.18) {
          scale *= THREE.MathUtils.smoothstep(eyeDist, 0.80, 1.18);
        }
        const noseDist = Math.hypot(
          pos.x / 0.15,
          (pos.y + 0.15) / 0.13,
          (pos.z - PROP.snoutLength * 0.99) / 0.15
        );
        if (noseDist < 1.15) {
          scale *= THREE.MathUtils.smoothstep(noseDist, 0.78, 1.15);
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
      const shade = (isUndercoat ? 0.88 : 0.92) + ventralLift + random() * (isUndercoat ? 0.08 : 0.10);
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

    const outer = new THREE.Mesh(earGeo, outerMat);
    outer.name = 'pinna-shell';
    outer.castShadow = true;
    outer.receiveShadow = true;
    pinna.add(outer);

    const rimMat = outerMat.clone();
    rimMat.color = rimMat.color.clone().offsetHSL(0.0, -0.01, -0.025);
    rimMat.transmission = Math.min(0.28, outerMat.transmission ?? 0.3);
    rimMat.thickness = 0.036;
    rimMat.roughness = Math.min(0.60, (outerMat.roughness ?? 0.42) + 0.08);
    const rim = new THREE.Mesh(earRimGeo, rimMat);
    rim.name = 'open-auricular-rim';
    rim.castShadow = true;
    pinna.add(rim);

    const rimFuzz = createEarRimFuzz(furColor, skinColor, density);
    rimFuzz.name = 'auricular-marginal-vellus';
    pinna.add(rimFuzz);

    const inner = new THREE.Mesh(earInnerGeo, innerMat);
    inner.name = 'concave-pinna-surface';
    inner.position.set(sideSign * -0.014, -0.018, 0.006);
    inner.rotation.z = sideSign * 0.025;
    inner.castShadow = false;
    pinna.add(inner);

    const rootMat = outerMat.clone();
    rootMat.color = rootMat.color.clone().lerp(new THREE.Color(furColor), 0.35);
    rootMat.transmission = Math.min(0.14, outerMat.transmission ?? 0.2);
    rootMat.thickness = 0.08;
    rootMat.roughness = 0.76;
    rootMat.clearcoat = 0.0;
    const pedicle = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 20, 16),
      rootMat
    );
    pedicle.name = 'buried-pinna-pedicle';
    pedicle.position.set(sideSign * 0.015, 0.01, -0.055);
    pedicle.scale.set(0.88, 0.44, 0.56);
    pedicle.castShadow = false;
    ear.add(pedicle);

    const foldMat = innerMat.clone();
    foldMat.color = foldMat.color.clone().offsetHSL(0.0, 0.015, -0.030);
    foldMat.transmission = Math.min(0.22, innerMat.transmission ?? 0.3);
    foldMat.roughness = 0.64;
    // Concha and tragus sit nested inside the concave lower pinna bowl so they
    // never protrude inboard/forward through the temporal cranium fur.
    const concha = new THREE.Mesh(
      new THREE.SphereGeometry(0.22, 18, 14),
      foldMat
    );
    concha.name = 'basal-conchal-fold';
    concha.position.set(sideSign * 0.018, 0.39, 0.008);
    concha.scale.set(0.68, 0.56, 0.16);
    concha.rotation.z = sideSign * 0.08;
    concha.castShadow = false;
    ear.add(concha);

    const tragus = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.026, 0.095, 8, 10),
      foldMat
    );
    tragus.name = 'auricular-tragus-fold';
    tragus.position.set(-sideSign * 0.042, 0.31, 0.026);
    tragus.rotation.z = sideSign * 0.38;
    tragus.scale.z = 0.65;
    tragus.castShadow = false;
    ear.add(tragus);

    const canalMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(innerMat.color).multiplyScalar(0.42),
      roughness: 0.80,
      transparent: true,
      opacity: 0.50,
      depthWrite: true,
    });
    const canal = new THREE.Mesh(
      new THREE.SphereGeometry(0.068, 14, 12),
      canalMat
    );
    canal.name = 'ear-canal-shadow';
    canal.position.set(-sideSign * 0.018, 0.27, 0.030);
    canal.scale.set(0.86, 0.54, 0.18);
    ear.add(canal);

    const baseFur = createEarBaseFur(furColor, density, sideSign);
    baseFur.name = 'pinna-root-fur-collar';
    baseFur.position.set(0, -0.02, -0.015);
    ear.add(baseFur);

    const strainScale = strain.earScaleMul ?? 1.0;
    const s = eSize * 0.88 * strainScale;
    const openingYaw = 0.20 + (strain.earYaw ?? PROP.earYaw) * 0.22;
    ear.position.set(
      sideSign * 0.62,
      0.02 + heightOffset,
      PROP.earSetBack + 0.10
    );
    ear.rotation.set(
      -0.10 + asymmetryTilt,
      sideSign * openingYaw + asymmetryYaw,
      sideSign * -0.11
    );
    ear.scale.set(s * 0.94, s, s);
    ear._pinna = pinna;
    ear._pedicle = pedicle;
    ear._canal = canal;
    return ear;
  }


  // ═══════════════════════════════════════════════════════════════
  // Shared assets survive subject disposal and can be used by later spawns.
  // ═══════════════════════════════════════════════════════════════
  const sharedResources = new Set([
    bodyGeo, headGeo, neckGeo, cervicalCoreGeo, thighGeo, earGeo, earInnerGeo, earRimGeo,
    fleshMatBase, clawMat, eyeMat, corneaMat, padNormalMap, tailRoughnessMap, tailNormalMap,
    eyeIrisMap, earNormalMap, earRoughnessMap, earAlbedoMap, noseAlbedoMap, noseNormalMap,
    noseRoughnessMap, labFloorAlbedoMap, labFloorRoughnessMap, labFloorNormalMap,
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
    getCervicalMantleProfile,
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
    disposeSubject,
    disposeSharedResources,
  };
}
