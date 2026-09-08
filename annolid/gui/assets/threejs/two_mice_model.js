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
    const base = ctx.createLinearGradient(0, 0, size, size);
    base.addColorStop(0.0, '#bfc3c1');
    base.addColorStop(0.52, '#d2d4d1');
    base.addColorStop(1.0, '#b7bcba');
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, size, size);

    // Fine plastic/fiber variation.
    for (let i = 0; i < 14000; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 180 + Math.floor(random() * 58);
      const a = 0.014 + random() * 0.032;
      ctx.fillStyle = random() > 0.55
        ? `rgba(255,255,245,${a})`
        : `rgba(${v},${Math.max(0, v - 10)},${Math.max(0, v - 22)},${a})`;
      ctx.fillRect(x, y, 1 + random() * 2.2, 1);
    }

    // Scuffs, wipe marks, and faint urine-like stains kept subtle.
    for (let i = 0; i < 44; i++) {
      const x = random() * size;
      const y = random() * size;
      const r = size * (0.018 + random() * 0.065);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      const warm = random() > 0.45;
      g.addColorStop(0.0, warm ? 'rgba(153,120,65,0.060)' : 'rgba(70,62,52,0.050)');
      g.addColorStop(0.68, warm ? 'rgba(153,120,65,0.020)' : 'rgba(70,62,52,0.016)');
      g.addColorStop(1.0, 'rgba(0,0,0,0.0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.ellipse(x, y, r * (0.7 + random() * 1.4), r * (0.24 + random() * 0.55), random() * Math.PI, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.lineCap = 'round';
    for (let i = 0; i < 170; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 18 + random() * 92;
      const a = random() * Math.PI * 2;
      ctx.strokeStyle = random() > 0.5 ? 'rgba(255,255,245,0.055)' : 'rgba(90,76,58,0.050)';
      ctx.lineWidth = 0.45 + random() * 1.1;
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
    for (let y = 0; y < size; y++) {
      // Mouse-tail scales read as shallow skin folds, not separate bands.
      const fold = Math.sin(y * 0.34) * 0.55 + Math.sin(y * 0.79 + 0.8) * 0.18;
      const grain = (random() - 0.5) * 4.0;
      const v = Math.floor(174 + fold * 10 + grain);
      ctx.fillStyle = `rgb(${v},${v},${v})`;
      ctx.fillRect(0, y, size, 1);
    }
    // Sparse, low-contrast keratin texture keeps the surface organic.
    ctx.globalAlpha = 0.035;
    for (let i = 0; i < 42; i++) {
      const y0 = random() * size;
      const x0 = random() * size;
      const shade = 158 + Math.floor(random() * 26);
      ctx.strokeStyle = `rgb(${shade},${shade},${shade})`;
      ctx.lineWidth = 0.4 + random() * 0.6;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x0 + 6 + random() * 10, y0 + (random() - 0.5) * 3);
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
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
      ctx.strokeStyle = `rgba(195,110,100,${opacity})`;
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
          drawVein(bx, by, bx2, by2, width * 0.6, opacity * 0.75, branches - 1);
        }
      }
    };
    // Main arteries radiating from ear base
    for (let i = 0; i < 8; i++) {
      const a = -Math.PI * 0.55 + (i / 7) * Math.PI * 1.1;
      const r0 = size * 0.08;
      const r1 = size * 0.38 + random() * size * 0.08;
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0;
      const x1 = cx + Math.cos(a) * r1;
      const y1 = cy + Math.sin(a) * r1;
      drawVein(x0, y0, x1, y1, 1.4 + random() * 1.2, 0.15 + random() * 0.09, 2);
    }
    // Secondary capillary network — finer veins
    for (let i = 0; i < 20; i++) {
      const a = -Math.PI * 0.70 + random() * Math.PI * 1.4;
      const r0 = size * (0.12 + random() * 0.15);
      const r1 = size * (0.30 + random() * 0.14);
      const x0 = cx + Math.cos(a) * r0;
      const y0 = cy + Math.sin(a) * r0 * 0.94;
      const x1 = cx + Math.cos(a) * r1;
      const y1 = cy + Math.sin(a) * r1 * 0.96;
      ctx.strokeStyle = `rgba(200,140,125,${0.10 + random() * 0.08})`;
      ctx.lineWidth = 0.6 + random() * 0.8;
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
    return tex;
  }
  function createNoseAlbedoMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    const cx = size * 0.5;
    const cy = size * 0.55;
    const g = ctx.createRadialGradient(cx, cy, size * 0.03, cx, cy, size * 0.50);
    g.addColorStop(0.0, '#e38f87');
    g.addColorStop(0.42, '#d07e76');
    g.addColorStop(1.0, '#a16861');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    // Warm red wet ring around nostril region.
    const ring = ctx.createRadialGradient(cx, cy, size * 0.06, cx, cy, size * 0.18);
    ring.addColorStop(0.0, 'rgba(140,35,35,0.10)');
    ring.addColorStop(0.55, 'rgba(170,50,50,0.28)');
    ring.addColorStop(1.0, 'rgba(120,30,30,0.0)');
    ctx.fillStyle = ring;
    ctx.fillRect(0, 0, size, size);
    // subtle philtrum tone
    ctx.strokeStyle = 'rgba(165,100,96,0.30)';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.moveTo(cx, cy - size * 0.03);
    ctx.quadraticCurveTo(cx + size * 0.01, cy + size * 0.05, cx, cy + size * 0.14);
    ctx.stroke();
    for (let i = 0; i < 350; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 170 + Math.floor(random() * 40);
      ctx.fillStyle = `rgba(${v},${v - 15},${v - 20},0.12)`;
      ctx.fillRect(x, y, 1, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
    return tex;
  }
  function createNoseNormalMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#8080ff';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 480; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 120 + Math.floor(random() * 22);
      ctx.fillStyle = `rgb(${v},${v},255)`;
      ctx.beginPath();
      ctx.arc(x, y, 0.4 + random() * 1.0, 0, Math.PI * 2);
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
    g.addColorStop(0.0, '#666666');
    g.addColorStop(1.0, '#c2c2c2');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 900; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 120 + Math.floor(random() * 90);
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
    const root = base.clone().offsetHSL(0.0, -0.02, -0.08);
    const tip = base.clone().offsetHSL(0.015, 0.01, hsl.l > 0.65 ? -0.015 : 0.06);
    // Keep the base tileable. A full-canvas gradient creates visible bands
    // when the map repeats around the torso's UV seam.
    ctx.fillStyle = `#${base.getHexString()}`;
    ctx.fillRect(0, 0, size, size);
    ctx.lineCap = 'round';
    for (let i = 0; i < 7200; i++) {
      const x = random() * size;
      const y = random() * size;
      const len = 4 + random() * 18;
      const useTip = random() > 0.52;
      const fiber = (useTip ? tip : root).clone().convertLinearToSRGB();
      ctx.strokeStyle = `rgba(${Math.round(fiber.r * 255)},${Math.round(fiber.g * 255)},${Math.round(fiber.b * 255)},${0.10 + random() * 0.16})`;
      ctx.lineWidth = 0.35 + random() * 0.75;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + (random() - 0.5) * 1.8, y + len);
      ctx.stroke();
    }
    const isPaleCoat = hsl.l > 0.70;
    for (let i = 0; i < 5000; i++) {
      const x = random() * size;
      const y = random() * size;
      const alpha = isPaleCoat
        ? 0.004 + random() * 0.016
        : 0.014 + random() * 0.040;
      const light = random() > 0.48;
      ctx.fillStyle = light
        ? `rgba(255,255,255,${alpha})`
        : `rgba(${isPaleCoat ? 110 : 0},${isPaleCoat ? 110 : 0},${isPaleCoat ? 105 : 0},${alpha})`;
      ctx.fillRect(x, y, 1 + random() * 1.5, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.0, 2.0);
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
    for (let i = 0; i < 5200; i++) {
      const x = random() * size, y = random() * size;
      const span = 3 + random() * 14;
      for (const [offset, red] of [[-0.45, 100], [0.45, 156]]) {
        ctx.strokeStyle = `rgba(${red},128,251,0.32)`;
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        ctx.moveTo(x + offset, y);
        ctx.quadraticCurveTo(x + offset + 0.7, y + span * 0.5, x + offset + 0.2, y + span);
        ctx.stroke();
      }
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3.0, 1.6);
    return tex;
  }
  function createCoatRoughnessMap(size = 256) {
    const c = document.createElement('canvas'); c.width = c.height = size;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#c8c8c8';
    ctx.fillRect(0, 0, size, size);
    for (let i = 0; i < 2400; i++) {
      const x = random() * size;
      const y = random() * size;
      const v = 150 + Math.floor(random() * 70);
      ctx.fillStyle = `rgba(${v},${v},${v},0.38)`;
      ctx.fillRect(x, y, 1 + random() * 2, 1);
    }
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(2.5, 1.4);
    return tex;
  }
  const padNormalMap = createPadNormalMap();
  const tailRoughnessMap = createTailRingRoughnessMap();
  const earNormalMap = createEarNormalMap();
  const earRoughnessMap = createEarRoughnessMap();
  const earAlbedoMap = createEarAlbedoMap();
  const noseAlbedoMap = createNoseAlbedoMap();
  const noseNormalMap = createNoseNormalMap();
  const noseRoughnessMap = createNoseRoughnessMap();
  const labFloorAlbedoMap = createLabFloorAlbedoMap();
  const labFloorRoughnessMap = createLabFloorRoughnessMap();
  const labFloorNormalMap = createLabFloorNormalMap();
  const urineSpotTexture = createSoftOvalTexture(256, '147,116,62', 0.105);
  const shadowSmudgeTexture = createSoftOvalTexture(256, '46,39,31', 0.16);

  // ═══════════════════════════════════════════════════════════════
  // 4. PBR MATERIALS
  // ═══════════════════════════════════════════════════════════════
  const fleshMatBase = new THREE.MeshPhysicalMaterial({
    roughness: 0.55, transmission: 0.15, thickness: 1.5,
    attenuationDistance: 1.0, clearcoat: 0.1, clearcoatRoughness: 0.2,
    sheen: 1.0, sheenColor: new THREE.Color(0xffeebb),
  });
  const clawMat = new THREE.MeshPhysicalMaterial({
    color: 0xd7cdc2, roughness: 0.15, metalness: 0.05,
    clearcoat: 1.0, clearcoatRoughness: 0.08,
    transmission: 0.08, thickness: 0.3,
    ior: 1.54, // keratin IOR
  });
  // Two-layer eye
  const eyeMat = new THREE.MeshPhysicalMaterial({
    color: 0x020202, roughness: 0.13, metalness: 0.0,
    clearcoat: 0.85, clearcoatRoughness: 0.08, envMapIntensity: 0.85,
  });
  const corneaMat = new THREE.MeshPhysicalMaterial({
    color: 0x000000, transparent: true, opacity: 0.12,
    roughness: 0.0, metalness: 0.0, ior: 1.38,
    clearcoat: 1.0, clearcoatRoughness: 0.04, envMapIntensity: 1.0,
  });

  function createCoatMaterial(furColor, skinColor, density, params) {
    const fur = new THREE.Color(furColor);
    const hsl = {}; fur.getHSL(hsl);
    const material = new THREE.MeshPhysicalMaterial({
      // The albedo map already contains the strain pigment. Keeping the
      // material tint neutral avoids multiplying dark coats into black.
      color: 0xffffff,
      roughness: hsl.l > 0.65 ? 0.90 : 0.96,
      envMapIntensity: 0.45,
      metalness: 0.0,
      clearcoat: 0.0,
      clearcoatRoughness: 0.72,
      sheen: 0.34,
      sheenColor: fur.clone().offsetHSL(0.0, -0.04, hsl.l > 0.55 ? -0.06 : 0.08),
      sheenRoughness: 0.70,
      map: createCoatAlbedoMap(furColor),
      normalMap: createCoatNormalMap(),
      normalScale: new THREE.Vector2(0.07 + density * 0.05, 0.08 + density * 0.06),
      roughnessMap: createCoatRoughnessMap(),
    });
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
    material.color.copy(base).offsetHSL(0.0, -0.01, isUndercoat ? -0.018 : 0.012);
    material.transparent = false;
    material.depthWrite = true;
    material.envMapIntensity = isUndercoat ? 0.12 : 0.30;
    material.sheen = isUndercoat ? 0.30 : 0.68;
    material.sheenColor = base.clone().offsetHSL(0.01, -0.03, 0.045);
    material.sheenRoughness = isUndercoat ? 0.88 : 0.68;
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
    // Compact murine silhouette: broad haunches, arched dorsum, narrower chest.
    const rearMass = Math.exp(-Math.pow((zNorm - 0.27) / 0.26, 2.0));
    let shoulderDrop = THREE.MathUtils.clamp((zNorm - 0.68) / 0.32, 0, 1);
    shoulderDrop = shoulderDrop * shoulderDrop * (3.0 - 2.0 * shoulderDrop);
    const waistNarrow = Math.exp(-Math.pow((zNorm - 0.58) / 0.16, 2.0));
    const frontNarrow = THREE.MathUtils.smoothstep(zNorm, 0.66, 0.94) * 0.035;
    const lateral = (
      1.0 + rearMass * 0.18 - shoulderDrop * 0.045
      - waistNarrow * (1.0 - PROP.waistTaper) * 0.14 - frontNarrow
    );
    v.x *= lateral * PROP.bodyGirthX;
    v.y *= (0.99 + rearMass * 0.07 - zNorm * 0.10) * PROP.bodyGirthY;
    v.z *= PROP.bodyLength;
    // Rounder cross-section so top view does not show a hard flank shelf.
    const sideBlend = THREE.MathUtils.clamp((Math.abs(v.y) - 0.06) / 0.72, 0, 1);
    v.x *= 0.95 + sideBlend * 0.05;

    // Dorsal line: peak around rear-mid, shoulder and tail-base slightly lower.
    const hump = Math.exp(-Math.pow((zNorm - 0.34) / 0.26, 2.0));
    const rump = Math.exp(-Math.pow((zNorm - 0.11) / 0.11, 2.0));
    const withers = Math.exp(-Math.pow((zNorm - 0.73) / 0.17, 2.0));
    v.y += hump * 0.24 + rump * 0.08 + withers * 0.13 - shoulderDrop * 0.015;

    // Ventrum: continuous shaping (avoid crease/cut line in top view).
    const yRef = v.y;
    const bellyWidth = Math.exp(-Math.pow((zNorm - 0.52) / 0.30, 2.0));
    const bellyMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.65, 0.18);
    const bellyInfluence = bellyWidth * bellyMask;
    v.x *= 1.0 + bellyInfluence * 0.18;
    const dropMask = 1 - THREE.MathUtils.smoothstep(yRef, -0.22, 0.24);
    v.y -= bellyWidth * dropMask * 0.15;

    const lowerBlend = THREE.MathUtils.clamp((0.10 - yRef) / 0.46, 0, 1);
    const groinTuck = Math.exp(-Math.pow((zNorm - 0.18) / 0.15, 2.0)) * 0.10;
    let chestDrop = THREE.MathUtils.clamp((zNorm - 0.72) / 0.28, 0, 1);
    chestDrop = chestDrop * chestDrop * (3.0 - 2.0 * chestDrop) * 0.05;
    const targetY = v.y * 0.76 + groinTuck - chestDrop;
    v.y = THREE.MathUtils.lerp(v.y, targetY, lowerBlend);

    // Subtle scapular and haunch emphasis from side.
    if (zNorm > 0.68 && zNorm < 0.86 && Math.abs(v.x) > 0.26) {
      const scapBulge = 1.0 - Math.abs(zNorm - 0.76) * 10.0;
      if (scapBulge > 0) { v.x *= 1.0 + scapBulge * 0.045; if (v.y > 0) v.y += scapBulge * 0.035; }
    }
    if (zNorm > 0.14 && zNorm < 0.36 && Math.abs(v.x) > 0.26) {
      const hB = 1.0 - Math.abs(zNorm - 0.25) * 6.0;
      if (hB > 0) { v.x *= 1.0 + hB * 0.12; if (v.y < 0) v.y *= 1.0 + hB * 0.07; }
    }

    // Tail base integration: gradual taper into pelvis
    if (zNorm < 0.10) {
      const tailBlend = zNorm / 0.10;
      v.x *= 0.7 + tailBlend * 0.3;
      v.y *= 0.8 + tailBlend * 0.2;
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
    const snoutTaper = Math.pow(1.0 - zNorm * 0.55, 1.0);
    v.x *= snoutTaper * PROP.snoutWidth;
    v.y *= snoutTaper * 1.02;
    v.z *= PROP.snoutLength;
    v.y -= zNorm * 0.14;
    // Narrow the occiput into the neck, then add cheek mass around the
    // zygomatic region instead of inflating the entire rear hemisphere.
    const occipitalBlend = THREE.MathUtils.smoothstep(zNorm, 0.04, 0.68);
    v.x *= 0.91 + occipitalBlend * 0.09;
    if (v.y > 0) v.y *= 0.93 + occipitalBlend * 0.07;
    const cheek = Math.exp(-Math.pow((zNorm - 0.57) / 0.18, 2.0));
    v.x *= 1.0 + cheek * PROP.cheekFullness * 0.30;
    if (v.y < 0) v.x *= 1.0 + cheek * 0.035;
    // Vibrissal pad bulge — fleshy muzzle mass at whisker roots
    if (zNorm > 0.55 && zNorm < 0.78 && Math.abs(v.x) > 0.18) {
      const padBulge = Math.exp(-Math.pow((zNorm - 0.66) / 0.08, 2.0));
      const latBulge = Math.exp(-Math.pow((Math.abs(v.x) - 0.38) / 0.12, 2.0));
      v.x *= 1.0 + padBulge * latBulge * 0.12;
      if (v.y < 0.05) v.y -= padBulge * latBulge * 0.03;
    }
    // Snout wedge with softer, rounder tip.
    if (zNorm > 0.64) {
      const p = (zNorm - 0.64) / 0.36;
      v.x *= 1.0 - p * 0.18;
      v.y *= 1.0 - p * 0.12;
      v.y -= p * 0.035;
      if (v.y < 0) v.x *= 1.0 + p * 0.05;
    }
    if (zNorm > 0.82) {
      const tip = (zNorm - 0.82) / 0.18;
      v.x *= 1.0 + tip * 0.06;
      v.y *= 1.0 + tip * 0.03;
    }
    // One continuous dorsal skull slope from the occiput into the snout.
    // A smooth bell avoids a raised rear dome that self-occludes the forehead.
    if (zNorm > 0.04 && zNorm < 0.78 && v.y > 0) {
      const cranium = Math.exp(-Math.pow((zNorm - 0.27) / 0.27, 2.0));
      v.y += cranium * 0.14;
    }
    // Temporal fullness supports the pinna root without making the forehead round.
    if (zNorm > 0.10 && zNorm < 0.46 && v.y > -0.16) {
      const temporal = Math.max(0, 1.0 - Math.abs(zNorm - 0.28) * 5.0);
      const dorsal = THREE.MathUtils.smoothstep(v.y, -0.16, 0.58);
      v.x *= 1.0 + temporal * dorsal * 0.050;
      if (v.y > 0.18) v.y += temporal * 0.012;
    }
    // Orbital ridges — slightly more prominent for side-view realism
    if (zNorm > 0.32 && zNorm < 0.54 && Math.abs(v.x) > 0.26) {
      v.y += 0.04;
      v.x *= 1.0 + 0.03; // slight lateral orbital bulge
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

  const deformThigh = (v) => {
    const yN = (v.y + 1) / 2; // 0=bottom(knee), 1=top(hip)
    // Stronger lateral mass with taper toward knee
    v.x *= 0.6 * (0.85 + yN * 0.2);
    v.y *= 1.25;
    v.z *= 1.25 * (0.9 + yN * 0.15);
    v.z -= v.y * 0.3;
    // Gluteal bulge near hip joint for rear-view mass
    const gluteal = Math.exp(-Math.pow((yN - 0.85) / 0.15, 2.0));
    v.x *= 1.0 + gluteal * 0.12;
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
  const thighGeo = deform(new THREE.SphereGeometry(1, 32, 32), deformThigh);
  const earGeo = createPinnaShellGeometry();
  const earInnerGeo = createPinnaSurfaceGeometry(0.945, 1.08, 0.018);
  const earRimGeo = createPinnaRimGeometry();


  // ═══════════════════════════════════════════════════════════════
  // 6. PAW BUILDER (anatomically correct forepaw vs hindpaw)
  // ═══════════════════════════════════════════════════════════════
  function createPaw(mat, isFore = false, strain = {}) {
    const paw = new THREE.Group();
    paw.name = isFore ? 'connected-forepaw' : 'connected-hindpaw';
    const pawFleshMat = mat.clone();
    pawFleshMat.color.set(strain.pawColor ?? '#e6b0a7');
    pawFleshMat.transmission = 0.08;
    pawFleshMat.thickness = 0.12;
    pawFleshMat.roughness = 0.60;
    pawFleshMat.clearcoat = 0.05;
    pawFleshMat.clearcoatRoughness = 0.48;
    const padMat = mat.clone();
    padMat.color.set(strain.pawPadColor ?? strain.pawColor ?? '#e9b8ae');
    padMat.color.multiplyScalar(strain.padDarken ?? 0.9);
    padMat.transmission = 0.06; padMat.thickness = 0.12;
    padMat.roughness = 0.58; padMat.clearcoat = 0.04; padMat.clearcoatRoughness = 0.52;
    padMat.normalMap = padNormalMap;
    padMat.normalScale = new THREE.Vector2(0.22, 0.22);

    const pawScale = isFore ? (PROP.forePawSize) : (PROP.hindPawSize);
    const rootLength = isFore ? 0.17 : 0.32;
    const rootBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(isFore ? 0.066 : 0.078, rootLength, 10, 12),
      pawFleshMat
    );
    rootBridge.name = isFore ? 'forepaw-root-bridge' : 'hindpaw-root-bridge';
    rootBridge.rotation.x = Math.PI / 2 - (isFore ? 0.04 : 0.10);
    rootBridge.position.set(0, 0.014, isFore ? -0.075 : -0.145);
    rootBridge.castShadow = true;
    paw.add(rootBridge);

    if (isFore) {
      const dorsalHand = new THREE.Mesh(new THREE.SphereGeometry(0.13, 18, 18), pawFleshMat);
      dorsalHand.name = 'forepaw-dorsal-hand';
      dorsalHand.scale.set(0.95 * pawScale, 0.42, 1.18 * pawScale);
      dorsalHand.position.set(0, 0.0, 0.055);
      dorsalHand.castShadow = true;
      paw.add(dorsalHand);

      const palm = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 16), padMat);
      palm.scale.set(0.90 * pawScale, 0.30, 1.04 * pawScale);
      palm.position.set(0, -0.058, 0.065);
      palm.castShadow = true;
      paw.add(palm);

      const carpalPad = new THREE.Mesh(new THREE.SphereGeometry(0.050, 12, 12), padMat);
      carpalPad.scale.set(0.95, 0.42, 1.18);
      carpalPad.position.set(0, -0.050, -0.045);
      carpalPad.castShadow = true;
      paw.add(carpalPad);

      for (let p = 0; p < 3; p++) {
        const iPad = new THREE.Mesh(new THREE.SphereGeometry(0.034, 12, 12), padMat);
        iPad.scale.set(0.92 * pawScale, 0.42, 1.0 * pawScale);
        iPad.position.set((p - 1) * 0.042 * pawScale, -0.061, 0.125);
        iPad.castShadow = true;
        paw.add(iPad);
      }
    } else {
      const dorsalFoot = new THREE.Mesh(new THREE.SphereGeometry(0.16, 20, 20), pawFleshMat);
      dorsalFoot.name = 'hindpaw-dorsal-foot';
      dorsalFoot.scale.set(0.88 * pawScale, 0.38, 1.65 * pawScale);
      dorsalFoot.position.set(0, -0.002, 0.070);
      dorsalFoot.castShadow = true;
      paw.add(dorsalFoot);

      const sole = new THREE.Mesh(new THREE.SphereGeometry(0.13, 18, 18), padMat);
      sole.scale.set(0.86 * pawScale, 0.25, 1.58 * pawScale);
      sole.position.set(0, -0.064, 0.075);
      sole.castShadow = true;
      paw.add(sole);

      const heelPad = new THREE.Mesh(new THREE.SphereGeometry(0.07, 12, 12), padMat);
      heelPad.scale.set(0.90, 0.36, 1.02);
      heelPad.position.set(0, -0.054, -0.145 * pawScale);
      heelPad.castShadow = true;
      paw.add(heelPad);

      const midPad = new THREE.Mesh(new THREE.SphereGeometry(0.065, 12, 12), padMat);
      midPad.scale.set(0.86, 0.34, 1.22);
      midPad.position.set(0, -0.069, 0.145 * pawScale);
      midPad.castShadow = true;
      paw.add(midPad);
    }

    const pawSpreadMul = strain.pawSpreadMul ?? 1.0;
    const toeCount = isFore ? 4 : 5;
    const baseSpread = isFore ? 0.062 : 0.055;
    const spread = baseSpread * pawSpreadMul * pawScale;

    const clawLocalMat = new THREE.MeshPhysicalMaterial({
      color: strain.clawColor ?? 0xf0e7df, roughness: 0.18, clearcoat: 1.0, clearcoatRoughness: 0.12
    });

    for (let i = 0; i < toeCount; i++) {
      const centeredIndex = i - (toeCount - 1) * 0.5;
      const t = centeredIndex * spread;
      const arc = Math.abs(centeredIndex) / (toeCount * 0.5 + 0.01);
      const toe = new THREE.Group();
      toe.name = `${isFore ? 'fore' : 'hind'}-digit-${i + 1}`;
      let lenMul = 1.0;
      if (isFore && (i === 0 || i === 3)) lenMul = 0.8;
      if (!isFore && (i === 0 || i === 4)) lenMul = 0.75;
      const toeLen = (isFore ? 0.18 : 0.245)
        * (strain.toeLenMul ?? 1.0) * lenMul * pawScale;
      const toeRadius = isFore ? 0.026 : 0.029;
      const phalanx = new THREE.Mesh(
        new THREE.CapsuleGeometry(toeRadius, toeLen, 10, 10),
        pawFleshMat
      );
      phalanx.position.z = toeLen * 0.48;
      phalanx.rotation.x = Math.PI / 2 - (isFore ? 0.08 : 0.12);
      phalanx.castShadow = true;
      toe.add(phalanx);

      const knuckle = new THREE.Mesh(
        new THREE.SphereGeometry(isFore ? 0.032 : 0.035, 10, 10),
        pawFleshMat
      );
      knuckle.scale.set(1.08, 0.82, 1.16);
      knuckle.position.set(0, 0.005, 0.018);
      knuckle.castShadow = true;
      toe.add(knuckle);

      const toeTip = new THREE.Mesh(new THREE.SphereGeometry(isFore ? 0.020 : 0.022, 10, 10), pawFleshMat);
      toeTip.scale.set(1.25, 0.82, 1.05);
      toeTip.position.set(0, isFore ? -0.012 : -0.018, toeLen + toeRadius * 0.52);
      toeTip.castShadow = true;
      toe.add(toeTip);

      const toePad = new THREE.Mesh(new THREE.SphereGeometry(isFore ? 0.035 : 0.045, 10, 10), padMat);
      toePad.scale.set(isFore ? 1.15 : 1.0, isFore ? 0.48 : 0.38, isFore ? 1.0 : 1.1);
      toePad.position.set(0, isFore ? -0.044 : -0.052, toeLen * 0.68);
      toePad.castShadow = true;
      toe.add(toePad);

      const clawLength = isFore ? 0.045 : 0.052;
      const clawRadius = isFore ? 0.008 : 0.009;
      const clawGeo = new THREE.CapsuleGeometry(clawRadius, clawLength, 6, 6);
      clawGeo.translate(0, clawLength / 2, 0);
      deform(clawGeo, (cv) => {
        const yN = THREE.MathUtils.clamp(
          (cv.y + clawRadius) / (clawLength + clawRadius * 2),
          0,
          1
        );
        cv.x *= 1.0 - yN * 0.35;
        cv.z *= 1.0 - yN * 0.25;
        cv.z -= yN * 0.005;
      });
      const claw = new THREE.Mesh(clawGeo, clawLocalMat);
      claw.rotation.x = Math.PI / 2 + (isFore ? 0.12 : 0.10);
      claw.position.set(0, isFore ? -0.014 : -0.019, toeLen + toeRadius * 0.62);
      claw.castShadow = true;
      toe.add(claw);

      const zPos = (isFore ? 0.135 - arc * 0.018 : 0.205 - arc * 0.025) * pawScale;
      toe.position.set(t, isFore ? -0.020 + arc * 0.006 : -0.028 + arc * 0.005, zPos);
      toe.rotation.y = centeredIndex * (isFore ? 0.045 : 0.035);
      toe.rotation.x = isFore ? -0.055 - arc * 0.035 : -0.10 - arc * 0.040;
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

    const shoulderMantle = new THREE.Mesh(new THREE.SphereGeometry(0.19, 16, 16), surfaceMat);
    shoulderMantle.position.set(-sign * 0.11, 0.03, -0.10);
    shoulderMantle.scale.set(1.12, 0.94, 1.28);
    shoulderMantle.castShadow = true;
    chain.add(shoulderMantle);
    const scapularBridge = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.105 * thickness, 0.22, 10, 10),
      surfaceMat
    );
    scapularBridge.name = `${side}-scapular-soft-tissue-bridge`;
    scapularBridge.position.set(-sign * 0.06, 0.05, -0.10);
    scapularBridge.rotation.x = Math.PI / 2;
    scapularBridge.castShadow = true;
    chain.add(scapularBridge);

    // Adult murine radius/ulna is slightly longer than the humerus. The
    // remaining scale belongs to the carpus and paw, represented below.
    const upperArm = new THREE.Group();
    upperArm.name = `${side}-upper-arm`;
    const uaLen = 0.42 * fl;
    const upperArmMesh = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.115 * thickness, uaLen, 10, 10), surfaceMat
    );
    upperArmMesh.position.y = -uaLen * 0.5;
    upperArmMesh.castShadow = true;
    upperArm.add(upperArmMesh);
    upperArm.rotation.x = FORELIMB_NEUTRAL.shoulderX;
    // Shoulder joint sphere (visible scapula bump)
    const shoulderJoint = new THREE.Mesh(new THREE.SphereGeometry(0.11 * thickness, 12, 12), surfaceMat);
    shoulderJoint.scale.set(1.0, 0.82, 1.05);
    shoulderJoint.castShadow = true;
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
    elbowBridge.castShadow = true;
    upperArm.add(elbowBridge);
    // Elbow joint
    const elbowJoint = new THREE.Mesh(new THREE.SphereGeometry(0.085 * thickness, 10, 10), surfaceMat);
    elbowJoint.scale.set(1.08, 0.95, 1.0);
    elbowJoint.castShadow = true;
    forearm.add(elbowJoint);
    const forearmMesh = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.074 * thickness, faLen, 8, 8), surfaceMat
    );
    forearmMesh.position.y = -faLen * 0.5;
    forearmMesh.castShadow = true;
    forearm.add(forearmMesh);
    upperArm.add(forearm);

    // Wrist joint
    const wrist = new THREE.Group();
    wrist.name = `${side}-wrist`;
    wrist.position.set(0, -faLen - 0.018, 0.03);
    const wristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.058 * thickness, 8, 8), surfaceMat);
    wristJoint.scale.set(1.1, 0.95, 1.05);
    wristJoint.castShadow = true;
    wrist.add(wristJoint);
    const wristCuff = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.062 * thickness, 0.07, 8, 8),
      surfaceMat
    );
    wristCuff.position.set(0, -0.03, 0.02);
    wristCuff.rotation.x = 0.18;
    wristCuff.castShadow = true;
    wrist.add(wristCuff);
    forearm.add(wrist);

    // Paw — sits at bottom of wrist
    const paw = createPaw(skinMat, true, strain);
    paw.position.set(sign * 0.035, -0.13, 0.07);
    paw.scale.setScalar(0.96 * Math.sqrt(thickness));
    paw.rotation.x = -0.12;
    wrist.add(paw);

    // Continuous coat along the limb, including the previously bare forearm.
    // Each coat uses the capsule surface, so the wrist remains a skin transition.
    const coatCapsule = (mesh, radius, span, count, hairLength) => {
      const coat = generateFurLayer(v => {
        const y = v.y;
        v.multiplyScalar(radius);
        v.y += y * span * 0.5;
      }, Math.floor(count * density), furColor, hairLength, false);
      mesh.add(coat);
    };
    coatCapsule(upperArmMesh, 0.116 * thickness, uaLen, 3200, 0.065);
    coatCapsule(forearmMesh, 0.075 * thickness, faLen, 2200, 0.045);
    coatCapsule(shoulderMantle, 0.192, 0, 1500, 0.060);

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

    // Hip joint (visible mass)
    const hipJoint = new THREE.Mesh(new THREE.SphereGeometry(0.18, 14, 14), surfaceMat);
    hipJoint.scale.set(1.08, 0.88, 1.18);
    hipJoint.castShadow = true;
    chain.add(hipJoint);
    // Soft blend into lower back (wider for smoother pelvis-hip transition)
    const hipBlend = new THREE.Mesh(new THREE.CapsuleGeometry(0.13, 0.26, 10, 10), surfaceMat);
    hipBlend.position.set(0, 0.07, -0.03);
    hipBlend.rotation.x = 0.34;
    hipBlend.scale.set(1.14, 0.92, 1.28);
    hipBlend.castShadow = true;
    chain.add(hipBlend);
    // Inboard pelvis bridge to blend hip into torso and avoid detached look.
    const pelvisBridge = new THREE.Mesh(new THREE.CapsuleGeometry(0.11, 0.30, 10, 10), surfaceMat);
    pelvisBridge.position.set(-sign * 0.14, 0.08, -0.02);
    pelvisBridge.rotation.z = Math.PI / 2;
    pelvisBridge.rotation.x = 0.16;
    pelvisBridge.scale.set(1.0, 0.88, 1.15);
    pelvisBridge.castShadow = true;
    chain.add(pelvisBridge);
    const hipSkirt = new THREE.Mesh(new THREE.SphereGeometry(0.13, 12, 12), surfaceMat);
    hipSkirt.position.set(-sign * 0.10, -0.01, 0.02);
    hipSkirt.scale.set(1.20, 0.82, 1.08);
    hipSkirt.castShadow = true;
    chain.add(hipSkirt);
    // Extra local fur on hip blend to hide seam against back.
    const hipFurCount = Math.floor(5200 * density);
    if (hipFurCount > 0) {
      const hfGeo = new THREE.ConeGeometry(0.006, 0.09, 3);
      hfGeo.translate(0, 0.045, 0);
      const hfMat = enhanceFurMaterial(
        new THREE.MeshPhysicalMaterial({ color: new THREE.Color(furColor), roughness: 0.82, transparent: true, opacity: 0.82 }),
        furColor
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

    // Thigh (femur) — enlarged deformed sphere
    const thigh = new THREE.Group();
    thigh.name = `${side}-thigh`;
    thigh.position.set(0, -0.10, 0.02);
    chain.add(thigh);
    const thighMuscle = new THREE.Mesh(thighGeo, surfaceMat);
    thighMuscle.scale.set(0.76 * thickness, 0.68 * hl, 0.86 * thickness);
    thighMuscle.castShadow = true;
    thigh.add(thighMuscle);
    thighMuscle.add(generateFurLayer(deformThigh, Math.floor(8200 * density), furColor, 0.12, false));

    // Shank (tibia/fibula) — substantial capsule
    const shank = new THREE.Group();
    shank.name = `${side}-shank`;
    shank.position.set(0.02 * sign, -0.32 * hl, 0.18);
    thigh.add(shank);

    // Knee joint (visible bump)
    const kneeJoint = new THREE.Mesh(new THREE.SphereGeometry(0.105 * thickness, 10, 10), surfaceMat);
    kneeJoint.position.set(0, 0.05, 0);
    kneeJoint.castShadow = true;
    shank.add(kneeJoint);

    const shankLen = 0.46 * hl;
    const shankMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.064 * thickness, shankLen, 8, 8), surfaceMat);
    shankMesh.rotation.x = -0.16;
    shankMesh.position.set(0, -shankLen * 0.5, 0.09);
    shankMesh.castShadow = true;
    shank.add(shankMesh);

    // Ankle/heel (digitigrade visible heel rise)
    const ankle = new THREE.Group();
    ankle.name = `${side}-ankle`;
    ankle.position.set(0, -shankLen - 0.04, 0.18);
    shank.add(ankle);

    const ankleJoint = new THREE.Mesh(new THREE.SphereGeometry(0.078 * thickness, 10, 10), surfaceMat);
    ankleJoint.castShadow = true;
    ankle.add(ankleJoint);

    // Heel bone (calcaneus) - visible heel rise
    const heel = new THREE.Mesh(new THREE.SphereGeometry(0.11, 14, 14), surfaceMat);
    heel.scale.set(0.82, 1.18, 1.45);
    heel.position.set(0, 0.02, -0.09);
    heel.castShadow = true;
    ankle.add(heel);

    // Metatarsals (connects heel to toes) — angled forward
    const metaLen = 0.36 * PROP.hindPawSize;
    const metaMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.055, metaLen, 6, 6), skinMat);
    metaMesh.rotation.x = Math.PI / 2 - 0.22;
    metaMesh.position.set(0, -0.125, 0.18);
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
    // Five mystacial rows with caudal whiskers longest and lower rows drooping.
    const rows = [
      { y: 0.105, z: -0.030, count: 4, minLength: 0.64, maxLength: 1.00, forward: 0.16, lift: 0.300, droop: 0.006 },
      { y: 0.060, z: -0.014, count: 4, minLength: 0.76, maxLength: 1.14, forward: 0.24, lift: 0.170, droop: -0.006 },
      { y: 0.010, z: 0.000, count: 5, minLength: 0.86, maxLength: 1.30, forward: 0.34, lift: 0.025, droop: -0.035 },
      { y: -0.045, z: 0.015, count: 4, minLength: 0.74, maxLength: 1.12, forward: 0.31, lift: -0.130, droop: -0.055 },
      { y: -0.098, z: 0.026, count: 3, minLength: 0.58, maxLength: 0.90, forward: 0.22, lift: -0.260, droop: -0.085 },
    ];
    rows.forEach((row, rowIndex) => {
      for (let i = 0; i < row.count; i++) {
        const fraction = (i + 1) / (row.count + 1);
        const length = THREE.MathUtils.lerp(row.minLength, row.maxLength, fraction)
          * (0.94 + random() * 0.12);
        const rootJitter = random() - 0.5;
        const rootY = row.y + (fraction - 0.5) * (0.020 + rowIndex * 0.004)
          + rootJitter * 0.014;
        const rootZ = row.z + (fraction - 0.5) * 0.085 + (random() - 0.5) * 0.012;
        const forward = row.forward * (0.62 + fraction * 0.38)
          + (random() - 0.5) * 0.035;
        const droop = row.droop * (0.75 + fraction * 0.35)
          + (random() - 0.5) * 0.018;
        const start = new THREE.Vector3(
          sign * (0.008 + fraction * 0.020 + rowIndex * 0.002),
          rootY,
          rootZ
        );
        const curve = new THREE.CubicBezierCurve3(
          start,
          new THREE.Vector3(
            sign * length * 0.28,
            rootY + row.lift * 0.05,
            rootZ + forward * 0.16
          ),
          new THREE.Vector3(
            sign * length * 0.72,
            rootY + row.lift * 0.55 + droop * 0.12 + rootJitter * 0.025,
            rootZ + forward * 0.68
          ),
          new THREE.Vector3(
            sign * length,
            rootY + row.lift + droop,
            rootZ + forward
          )
        );
        const rootRadius = 0.0035 + fraction * 0.0008;
        const whisker = new THREE.Mesh(
          createTaperedWhiskerGeometry(curve, 14, rootRadius, 0.00042, 5),
          material
        );
        whisker.name = `vibrissa-${rowIndex}-${i}`;
        whisker.castShadow = true;
        whisker.frustumCulled = false;
        group.add(whisker);
      }
    });
    return group;
  }

  function generateFurLayer(deformFn, count, colorHex, length, isHead, isUndercoat = false) {
    if (count <= 0) return new THREE.Group();
    const radius = isUndercoat ? 0.0024 : 0.0018;
    // Only the visible guard layer needs the extra bend segment. Undercoat
    // remains a single tapered segment to keep the default scene interactive.
    const geo = new THREE.ConeGeometry(radius, length, 3, isUndercoat ? 1 : 2, true);
    geo.translate(0, length / 2, 0);
    geo.rotateX(Math.PI / 2);
    const vertices = geo.attributes.position;
    const colors = new Float32Array(vertices.count * 3);
    for (let i = 0; i < vertices.count; i++) {
      const progress = THREE.MathUtils.clamp(vertices.getZ(i) / length, 0, 1);
      vertices.setX(i, vertices.getX(i) + length * 0.09 * progress * progress);
      const shade = 0.82 + progress * 0.28;
      colors.set([shade, shade, shade], i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const mat = new THREE.MeshPhysicalMaterial({
      vertexColors: true,
      color: 0xffffff, roughness: isUndercoat ? 0.94 : 0.82,
      clearcoat: 0.0,
      sheen: isUndercoat ? 0.24 : 0.58,
      sheenColor: new THREE.Color(colorHex),
      sheenRoughness: isUndercoat ? 0.88 : 0.68,
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
      pos.addScaledVector(normal, isUndercoat ? 0.002 : 0.004);
      let flowDir;
      if (isHead) {
        // Face: fur sweeps toward nose on snout, backward on cranium
        if (preZ > 0.3) flowDir = new THREE.Vector3(normal.x * 0.22, -0.10, -0.72).normalize();
        else flowDir = new THREE.Vector3(normal.x * 0.3, normal.y * 0.2, -1).normalize();
      } else {
        // Body: region-specific fur flow
        const isHaunch = preZ < -0.3;
        const isShoulder = preZ > 0.5;
        const isDorsal = preY > 0.3;
        const isVentral = preY < -0.3;
        const isTailBase = preZ < -0.7;
        if (isTailBase) {
          // Tail base: converge toward tail
          flowDir = new THREE.Vector3(normal.x * 0.15, -0.08, -0.9).normalize();
        } else if (isHaunch) {
          // Haunches: posterior sweep with slight outward fan
          flowDir = new THREE.Vector3(normal.x * 0.6, normal.y * 0.2 - 0.1, -0.8).normalize();
        } else if (isShoulder) {
          // Shoulders: caudal flow with tight body-hugging direction
          flowDir = new THREE.Vector3(normal.x * 0.35, normal.y * 0.15 - 0.12, -0.9).normalize();
        } else if (isDorsal) {
          // Dorsal: follows spine line caudally
          flowDir = new THREE.Vector3(normal.x * 0.2, 0.05, -0.95).normalize();
        } else if (isVentral) {
          // Ventral: fans outward and slightly caudal
          flowDir = new THREE.Vector3(normal.x * 0.7, normal.y * 0.5 - 0.2, -0.6).normalize();
        } else {
          // Flanks: general caudal flow
          flowDir = new THREE.Vector3(normal.x * 0.5, normal.y * 0.3 - 0.15, -1).normalize();
        }
      }
      if (isUndercoat) { flowDir.lerp(new THREE.Vector3(0, 0, -1), 0.3); flowDir.normalize(); }
      // Project combing onto the skin before adding lift. A radial sphere
      // normal buries strands at the shoulder and points them into the rump.
      flowDir.addScaledVector(normal, -flowDir.dot(normal));
      if (flowDir.lengthSq() < 1e-8) flowDir.set(1, 0, 0).addScaledVector(normal, -normal.x);
      flowDir.normalize().addScaledVector(normal, isUndercoat ? 0.32 : 0.16).normalize();
      dummy.position.copy(pos);
      dummy.lookAt(lookTarget.copy(pos).add(flowDir));
      let scale = 0.72 + random() * 0.42;
      if (isHead && preZ > 0.4) scale *= 0.42;
      if (isHead && preZ > 0.2 && preZ <= 0.4) scale *= 0.6;
      if (!isHead && preZ < -0.6) scale *= 1.08;
      if (isUndercoat) scale = 0.76 + random() * 0.22;
      const jitter = isUndercoat ? 0.08 : 0.045;
      dummy.rotation.x += (random() - 0.5) * jitter;
      dummy.rotation.y += (random() - 0.5) * jitter;
      dummy.scale.set(scale, scale, scale); dummy.updateMatrix();
      inst.setMatrixAt(i, dummy.matrix);
      const shade = (isUndercoat ? 0.88 : 0.92) + random() * (isUndercoat ? 0.08 : 0.10);
      strandColor.setRGB(shade, shade, shade);
      inst.instanceColor.setXYZ(i, strandColor.r, strandColor.g, strandColor.b);
    }
    return inst;
  }

  // Fur detail-level multipliers
  const FUR_DETAIL = { High: 1.0, Med: 0.5, Low: 0.15, Off: 0.0 };

  function generateMultiLayerFur(deformFn, density, furColor, isHead) {
    const group = new THREE.Group();
    if (density <= 0) return group;
    const underCount = Math.floor((isHead ? 16000 : 38000) * density);
    const guardCount = Math.floor((isHead ? 9500 : 22000) * density);
    group.add(generateFurLayer(deformFn, underCount, furColor, isHead ? 0.035 : 0.055, isHead, true));
    group.add(generateFurLayer(deformFn, guardCount, furColor, isHead ? 0.062 : 0.12, isHead, false));
    return group;
  }

  function createEarBaseFur(furColor, density, sideSign) {
    const count = Math.floor(620 * density);
    if (count <= 0) return new THREE.Group();
    const geo = new THREE.ConeGeometry(0.0048, 0.076, 3);
    geo.translate(0, 0.038, 0);
    const mat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(furColor),
      roughness: 0.90,
      transparent: true,
      opacity: 0.86,
      sheen: 0.52,
      sheenColor: new THREE.Color(furColor).offsetHSL(0.0, -0.04, 0.12),
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
    rimMat.color = rimMat.color.clone().offsetHSL(0.0, -0.01, -0.030);
    rimMat.transmission = Math.min(0.24, outerMat.transmission ?? 0.3);
    rimMat.thickness = 0.040;
    rimMat.roughness = Math.min(0.62, (outerMat.roughness ?? 0.42) + 0.10);
    const rim = new THREE.Mesh(earRimGeo, rimMat);
    rim.name = 'open-auricular-rim';
    rim.castShadow = true;
    pinna.add(rim);

    const inner = new THREE.Mesh(earInnerGeo, innerMat);
    inner.name = 'concave-pinna-surface';
    inner.position.set(sideSign * -0.014, -0.018, 0.006);
    inner.rotation.z = sideSign * 0.025;
    inner.castShadow = true;
    pinna.add(inner);

    const rootMat = outerMat.clone();
    rootMat.color = rootMat.color.clone().lerp(new THREE.Color(furColor), 0.30);
    rootMat.transmission = Math.min(0.16, outerMat.transmission ?? 0.2);
    rootMat.thickness = 0.08;
    rootMat.roughness = 0.74;
    rootMat.clearcoat = 0.0;
    const pedicle = new THREE.Mesh(
      new THREE.SphereGeometry(0.32, 24, 18),
      rootMat
    );
    pedicle.name = 'buried-pinna-pedicle';
    pedicle.position.set(0, -0.005, -0.055);
    pedicle.scale.set(0.96, 0.46, 0.60);
    pedicle.castShadow = true;
    ear.add(pedicle);

    const foldMat = innerMat.clone();
    foldMat.color = foldMat.color.clone().offsetHSL(0.0, 0.015, -0.035);
    foldMat.transmission = Math.min(0.20, innerMat.transmission ?? 0.3);
    foldMat.roughness = 0.68;
    const concha = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 20, 16),
      foldMat
    );
    concha.name = 'basal-conchal-fold';
    concha.position.set(-sideSign * 0.055, 0.29, 0.045);
    concha.scale.set(0.86, 0.64, 0.18);
    concha.rotation.z = sideSign * 0.10;
    concha.castShadow = true;
    ear.add(concha);

    const tragus = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.038, 0.15, 8, 10),
      foldMat
    );
    tragus.name = 'auricular-tragus-fold';
    tragus.position.set(-sideSign * 0.15, 0.20, 0.083);
    tragus.rotation.z = sideSign * 0.48;
    tragus.scale.z = 0.70;
    tragus.castShadow = true;
    ear.add(tragus);

    const canalMat = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(innerMat.color).multiplyScalar(0.45),
      roughness: 0.78,
      transparent: true,
      opacity: 0.52,
      depthWrite: true,
    });
    const canal = new THREE.Mesh(
      new THREE.SphereGeometry(0.078, 16, 12),
      canalMat
    );
    canal.name = 'ear-canal-shadow';
    canal.position.set(-sideSign * 0.075, 0.135, 0.105);
    canal.scale.set(0.90, 0.55, 0.20);
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
    bodyGeo, headGeo, neckGeo, thighGeo, earGeo, earInnerGeo, earRimGeo,
    fleshMatBase, clawMat, eyeMat, corneaMat, padNormalMap, tailRoughnessMap,
    earNormalMap, earRoughnessMap, earAlbedoMap, noseAlbedoMap, noseNormalMap,
    noseRoughnessMap, labFloorAlbedoMap, labFloorRoughnessMap, labFloorNormalMap,
    urineSpotTexture, shadowSmudgeTexture,
  ]);

  function disposeSubject(root) {
    const resources = new Set();
    root.traverse(node => {
      if (node.isInstancedMesh) node.dispose();
      if (node.geometry) resources.add(node.geometry);
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const material of materials) {
        if (!material) continue;
        resources.add(material);
        for (const value of Object.values(material)) {
          if (value?.isTexture) resources.add(value);
        }
      }
    });
    for (const resource of resources) {
      if (!sharedResources.has(resource)) resource.dispose();
    }
  }

  return {
    PROP,
    FORELIMB_NEUTRAL,
    tailRoughnessMap,
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
    createForelimbChain,
    createHindlimbChain,
    createAnatomicalWhiskers,
    FUR_DETAIL,
    generateMultiLayerFur,
    buildMouseEar,
    disposeSubject,
  };
}
