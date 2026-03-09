async function boot() {
  const statusEl = document.getElementById("annolidThreeStatus");
  const canvas = document.getElementById("annolidThreeCanvas");
  const timelineEl = document.getElementById("annolidThreeTimeline");
  const timelineSlider = document.getElementById("simulationFrameSlider");
  const timelineLabel = document.getElementById("simulationFrameLabel");
  const timelinePlayBtn = document.getElementById("btnSimulationPlay");
  const metaEl = document.getElementById("annolidThreeMeta");
  const flybodyControlsEl = document.getElementById("annolidThreeFlybodyControls");
  const toolbarEl = document.getElementById("annolidThreeToolbar");
  const legendEl = document.getElementById("annolidThreeLegend");
  const categoryPanelEl = document.getElementById("annolidThreeCategoryPanel");
  const modelUrl = window.__annolidThreeModelUrl || "";
  const modelExtHint = (window.__annolidThreeModelExt || "").toLowerCase();
  const title = window.__annolidThreeTitle || "3D";

  const setStatus = (msg, level = "info") => {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.setAttribute("data-level", level);
  };

  if (!canvas) {
    setStatus("Missing canvas element.", "error");
    document.body.setAttribute("data-threejs-error", "missing-canvas");
    return;
  }

  try {
    const THREE = await import("https://esm.sh/three@0.182.0");
    const { OrbitControls } = await import(
      "https://esm.sh/three@0.182.0/examples/jsm/controls/OrbitControls.js"
    );
    const { STLLoader } = await import(
      "https://esm.sh/three@0.182.0/examples/jsm/loaders/STLLoader.js"
    );
    const { PLYLoader } = await import(
      "https://esm.sh/three@0.182.0/examples/jsm/loaders/PLYLoader.js"
    );
    const { OBJLoader } = await import(
      "https://esm.sh/three@0.182.0/examples/jsm/loaders/OBJLoader.js"
    );
    const { MTLLoader } = await import(
      "https://esm.sh/three@0.182.0/examples/jsm/loaders/MTLLoader.js"
    );
    const { GLTFLoader } = await import(
      "https://esm.sh/three@0.182.0/examples/jsm/loaders/GLTFLoader.js"
    );

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(Math.max(1, window.devicePixelRatio || 1), 2));
    const getCanvasSize = () => {
      const w = Math.max(1, canvas.clientWidth || window.innerWidth || 800);
      const h = Math.max(1, canvas.clientHeight || window.innerHeight || 600);
      return { w, h };
    };
    {
      const { w, h } = getCanvasSize();
      renderer.setSize(w, h, false);
    }
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121824);

    const camera = new THREE.PerspectiveCamera(
      50,
      getCanvasSize().w / Math.max(1, getCanvasSize().h),
      0.01,
      10000
    );
    camera.position.set(0, 0, 3);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controls.update();

    let realtimeEnabled = true;

    // --- Toolbar Interaction ---
    const btnHome = document.getElementById("btnHome");
    const btnToggleRealtime = document.getElementById("btnToggleRealtime");

    const resetCamera = () => {
      if (!root) return;
      const box = new THREE.Box3().setFromObject(root);
      if (box.isEmpty()) return;

      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);

      const maxDim = Math.max(size.x, size.y, size.z, 0.1);
      const distance = maxDim * 1.8;

      camera.position.set(distance, distance * 0.65, distance);
      controls.target.set(0, 0, 0);
      camera.lookAt(0, 0, 0);
      controls.update();
    };

    if (btnHome) btnHome.onclick = resetCamera;

    // --- Advanced Views ---
    const btnViewX = document.getElementById("btnViewX");
    const btnViewY = document.getElementById("btnViewY");
    const btnViewZ = document.getElementById("btnViewZ");

    const setView = (axis) => {
      const box = new THREE.Box3().setFromObject(root);
      const size = new THREE.Vector3();
      box.getSize(size);
      const maxDim = Math.max(size.x, size.y, size.z, 0.1);
      const dist = maxDim * 2.2;

      if (axis === "x") camera.position.set(dist, 0, 0);
      else if (axis === "y") camera.position.set(0, dist, 0);
      else if (axis === "z") camera.position.set(0, 0, dist);

      controls.target.set(0, 0, 0);
      controls.update();
    };

    if (btnViewX) btnViewX.onclick = () => setView("x");
    if (btnViewY) btnViewY.onclick = () => setView("y");
    if (btnViewZ) btnViewZ.onclick = () => setView("z");

    // --- Orientation Controls ---
    const btnFlipX = document.getElementById("btnFlipX");
    const btnFlipY = document.getElementById("btnFlipY");
    const btnFlipZ = document.getElementById("btnFlipZ");
    const btnRotateX = document.getElementById("btnRotateX");
    const btnRotateY = document.getElementById("btnRotateY");
    const btnRotateZ = document.getElementById("btnRotateZ");

    if (btnFlipX) btnFlipX.onclick = () => { root.scale.x *= -1; };
    if (btnFlipY) btnFlipY.onclick = () => { root.scale.y *= -1; };
    if (btnFlipZ) btnFlipZ.onclick = () => { root.scale.z *= -1; };

    const rotate90 = (axis) => {
      const angle = Math.PI / 2;
      if (axis === "x") root.rotateX(angle);
      else if (axis === "y") root.rotateY(angle);
      else if (axis === "z") root.rotateZ(angle);
    };

    if (btnRotateX) btnRotateX.onclick = () => rotate90("x");
    if (btnRotateY) btnRotateY.onclick = () => rotate90("y");
    if (btnRotateZ) btnRotateZ.onclick = () => rotate90("z");

    // --- Helpers and Themes ---
    const btnToggleAxes = document.getElementById("btnToggleAxes");
    const btnToggleTheme = document.getElementById("btnToggleTheme");
    let axesHelper = null;

    if (btnToggleAxes) {
      btnToggleAxes.onclick = () => {
        if (!axesHelper) {
          const box = new THREE.Box3().setFromObject(root);
          const size = new THREE.Vector3();
          box.getSize(size);
          const maxDim = Math.max(size.x, size.y, size.z, 0.1);
          axesHelper = new THREE.AxesHelper(maxDim * 0.5);
          root.add(axesHelper);
        } else {
          axesHelper.visible = !axesHelper.visible;
        }
        btnToggleAxes.style.color = axesHelper.visible ? "#fff" : "#888";
      };
    }

    if (btnToggleTheme) {
      btnToggleTheme.onclick = () => {
        const isWhite = document.body.classList.toggle("white-theme");
        scene.background = new THREE.Color(isWhite ? 0xd9e2ec : 0x121824);
        btnToggleTheme.style.color = isWhite ? "#333" : "#fff";
      };
    }

    const btnToggleAutoRotate = document.getElementById("btnToggleAutoRotate");
    if (btnToggleAutoRotate) {
      btnToggleAutoRotate.onclick = () => {
        controls.autoRotate = !controls.autoRotate;
        btnToggleAutoRotate.style.color = controls.autoRotate ? "#fff" : "#888";
      };
    }
    if (btnToggleRealtime) {
      btnToggleRealtime.onclick = () => {
        realtimeEnabled = !realtimeEnabled;
        btnToggleRealtime.style.color = realtimeEnabled ? "#fff" : "#ff3366";
        btnToggleRealtime.style.background = realtimeEnabled ? "rgba(255, 255, 255, 0.08)" : "rgba(255, 51, 102, 0.2)";
        setStatus(realtimeEnabled ? "Real-time updates resumed" : "Real-time updates paused");
      };
    }

    // --- Gesture Indicators ---
    const indRotate = document.getElementById("indRotate");
    const indPan = document.getElementById("indPan");
    const indZoom = document.getElementById("indZoom");

    const updateIndicator = (el, active) => {
      if (!el) return;
      if (active) el.classList.add("active");
      else el.classList.remove("active");
    };

    scene.add(new THREE.AmbientLight(0x90a8ff, 0.55));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.1);
    keyLight.position.set(3, 5, 8);
    scene.add(keyLight);
    const rimLight = new THREE.DirectionalLight(0xb1d5ff, 0.45);
    rimLight.position.set(-4, 2, -6);
    scene.add(rimLight);

    const ext = (modelExtHint || "").replace(/^\./, "");
    const root = new THREE.Group();
    scene.add(root);

    const fitCameraToObject = (obj, options = {}) => {
      const box = new THREE.Box3().setFromObject(obj || root);
      if (box.isEmpty()) {
        setStatus(`Loaded ${title} but geometry bounds are empty.`, "error");
        return;
      }
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      const centerControls = options.centerControls !== false;
      if (centerControls) {
        controls.target.copy(center);
      }
      controls.update();

      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      const distance = maxDim * 1.8;
      camera.position.set(center.x + distance, center.y + distance * 0.65, center.z + distance);
      camera.near = Math.max(0.001, maxDim / 1000);
      camera.far = Math.max(1000, maxDim * 20);
      camera.updateProjectionMatrix();
    };

    const addLoadedObject = (obj) => {
      root.add(obj);
      const box = new THREE.Box3().setFromObject(root);
      if (box.isEmpty()) {
        setStatus(`Loaded ${title} but geometry bounds are empty.`, "error");
        return;
      }
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);
      root.position.sub(center);
      controls.target.set(0, 0, 0);
      fitCameraToObject(root, { centerControls: false });

      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      const axes = new THREE.AxesHelper(maxDim * 0.35);
      axes.visible = false;
      root.add(axes);

      setStatus(`Loaded ${title} (${ext.toUpperCase()}).`);
      document.body.setAttribute("data-threejs-ready", "1");
    };

    // Gaussian Splats PLY detection and loading functions
    const detectGaussianSplatsPLY = (buffer) => {
      const headerText = new TextDecoder().decode(buffer.slice(0, 4000));
      return headerText.includes('f_dc_0') &&
        headerText.includes('opacity') &&
        headerText.includes('scale_0');
    };

    const loadGaussianSplatsPLY = (buffer, addLoadedObject, setStatus) => {
      try {
        const headerText = new TextDecoder().decode(buffer.slice(0, 10000));
        const headerEnd = 'end_header\n';
        const headerEndIndex = headerText.indexOf(headerEnd);
        if (headerEndIndex === -1) throw new Error("Invalid PLY header");

        const headerPart = headerText.slice(0, headerEndIndex);
        const vertexCountMatch = headerPart.match(/element vertex (\d+)/);
        const vertexCount = vertexCountMatch ? parseInt(vertexCountMatch[1]) : 0;

        // Find property definitions to calculate stride
        const properties = headerPart.match(/property float \w+/g) || [];
        const stride = properties.length * 4;
        const offset = headerEndIndex + headerEnd.length;

        console.log(`Loading Gaussian Splats: ${vertexCount} points, stride ${stride} bytes.`);

        const limit = 3500000;
        const count = Math.min(vertexCount, limit);

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);

        const dataView = new DataView(buffer, offset);

        // Map common property names to their indices in the stride
        const propIndices = {};
        properties.forEach((prop, i) => {
          const name = prop.split(' ').pop();
          propIndices[name] = i;
        });

        for (let i = 0; i < count; i++) {
          const base = i * stride;

          // Position
          positions[i * 3 + 0] = dataView.getFloat32(base + (propIndices['x'] || 0) * 4, true);
          positions[i * 3 + 1] = dataView.getFloat32(base + (propIndices['y'] || 1) * 4, true);
          positions[i * 3 + 2] = dataView.getFloat32(base + (propIndices['z'] || 2) * 4, true);

          // SH DC to RGB (approximate)
          const r_dc = dataView.getFloat32(base + (propIndices['f_dc_0'] || 6) * 4, true);
          const g_dc = dataView.getFloat32(base + (propIndices['f_dc_1'] || 7) * 4, true);
          const b_dc = dataView.getFloat32(base + (propIndices['f_dc_2'] || 8) * 4, true);

          // SH coefficient to sRGB: 0.5 + 0.28209 * DC
          // Simpler approx: 0.5 + 0.5 * dc
          colors[i * 3 + 0] = Math.max(0, Math.min(1, 0.5 + 0.28209 * r_dc));
          colors[i * 3 + 1] = Math.max(0, Math.min(1, 0.5 + 0.28209 * g_dc));
          colors[i * 3 + 2] = Math.max(0, Math.min(1, 0.5 + 0.28209 * b_dc));
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
          vertexColors: true,
          size: 0.015,
          sizeAttenuation: true,
          transparent: true,
          opacity: 0.95,
          alphaTest: 0.05
        });

        const points = new THREE.Points(geometry, material);
        addLoadedObject(points);
        setStatus(`Loaded Gaussian Splats with ${count.toLocaleString()} points.`);
      } catch (err) {
        console.error(err);
        setStatus(`Failed to parse GS-PLY: ${err.message}`, "error");
      }
    };

    const buildPointCloudFromRows = (rows) => {
      const points = [];
      for (const row of rows) {
        if (!row || row.length < 3) continue;
        const x = Number(row[0]);
        const y = Number(row[1]);
        const z = Number(row[2]);
        if (
          Number.isFinite(x) &&
          Number.isFinite(y) &&
          Number.isFinite(z)
        ) {
          points.push(x, y, z);
        }
      }
      if (!points.length) return null;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
      const mat = new THREE.PointsMaterial({
        color: 0x7bc6ff,
        size: 0.02,
        sizeAttenuation: true,
      });
      return new THREE.Points(geo, mat);
    };

    const parseDelimitedPointCloud = async () => {
      const resp = await fetch(modelUrl, { cache: "no-store" });
      if (!resp.ok) {
        throw new Error(`Unable to fetch model: HTTP ${resp.status}`);
      }
      const text = await resp.text();
      const lines = text.split(/\r?\n/);
      const rows = [];
      for (const line of lines) {
        const clean = line.trim();
        if (!clean || clean.startsWith("#")) continue;
        const cols = clean
          .split(/[,\s]+/)
          .map((s) => s.trim())
          .filter(Boolean);
        rows.push(cols);
      }
      return buildPointCloudFromRows(rows);
    };

    const parseJsonPayload = async () => {
      const resp = await fetch(modelUrl, { cache: "no-store" });
      if (!resp.ok) {
        throw new Error(`Unable to fetch model: HTTP ${resp.status}`);
      }
      return await resp.json();
    };

    const simulationRoot = new THREE.Group();
    const simulationEnvironmentRoot = new THREE.Group();
    const simulationModelRoot = new THREE.Group();
    const simulationBodyPartsRoot = new THREE.Group();
    const simulationPoints = new THREE.Group();
    const simulationEdges = new THREE.Group();
    const simulationTrails = new THREE.Group();
    const simulationLabels = new THREE.Group();
    simulationRoot.add(simulationEnvironmentRoot);
    simulationRoot.add(simulationModelRoot);
    simulationRoot.add(simulationBodyPartsRoot);
    simulationRoot.add(simulationPoints);
    simulationRoot.add(simulationEdges);
    simulationRoot.add(simulationTrails);
    simulationRoot.add(simulationLabels);
    let simulationPayload = null;
    let simulationBodyPartMap = new Map();
    let simulationCategoryState = new Map();
    let simulationFrameIndex = 0;
    let simulationPlaying = false;
    let simulationTimer = null;
    let simulationLoopEnabled = true;
    let simulationMeshKey = "";
    let simulationEnvironmentKey = "";
    let simulationActiveBehavior = "";
    let flybodyControlsMoved = false;
    let flybodyDragState = null;

    const positionFlybodyControls = () => {
      if (!flybodyControlsEl || !toolbarEl || flybodyControlsMoved) return;
      const toolbarRect = toolbarEl.getBoundingClientRect();
      flybodyControlsEl.style.width = `${Math.max(140, Math.ceil(toolbarRect.width))}px`;
      flybodyControlsEl.style.left = "auto";
      flybodyControlsEl.style.right = "12px";
      flybodyControlsEl.style.top = `${Math.round(toolbarRect.bottom + 8)}px`;
    };

    const stopFlybodyDrag = () => {
      flybodyDragState = null;
      if (flybodyControlsEl) {
        flybodyControlsEl.classList.remove("dragging");
      }
      window.removeEventListener("pointermove", onFlybodyDragMove);
      window.removeEventListener("pointerup", stopFlybodyDrag);
      window.removeEventListener("pointercancel", stopFlybodyDrag);
    };

    const onFlybodyDragMove = (event) => {
      if (!flybodyDragState || !flybodyControlsEl) return;
      flybodyControlsMoved = true;
      const nextLeft = Math.max(8, event.clientX - flybodyDragState.offsetX);
      const nextTop = Math.max(8, event.clientY - flybodyDragState.offsetY);
      flybodyControlsEl.style.left = `${nextLeft}px`;
      flybodyControlsEl.style.top = `${nextTop}px`;
      flybodyControlsEl.style.right = "auto";
    };

    const issueFlybodyCommand = (action, behavior) => {
      const url = new URL(`annolid://flybody-live/${action}`);
      if (behavior) {
        url.searchParams.set("behavior", behavior);
      }
      window.location.href = url.toString();
    };

    const rebuildFlybodyControls = () => {
      if (!flybodyControlsEl) return;
      flybodyControlsEl.innerHTML = "";
      const adapter = String((simulationPayload && simulationPayload.adapter) || "");
      if (!(adapter === "flybody" || adapter === "flybody-live")) {
        flybodyControlsEl.hidden = true;
        return;
      }
      const titleEl = document.createElement("div");
      titleEl.className = "three-toolbar-subtitle";
      titleEl.textContent = "FlyBody";
      titleEl.title = "Drag to move FlyBody controls";
      titleEl.addEventListener("pointerdown", (event) => {
        if (!flybodyControlsEl) return;
        const rect = flybodyControlsEl.getBoundingClientRect();
        flybodyDragState = {
          offsetX: event.clientX - rect.left,
          offsetY: event.clientY - rect.top,
        };
        flybodyControlsEl.classList.add("dragging");
        window.addEventListener("pointermove", onFlybodyDragMove);
        window.addEventListener("pointerup", stopFlybodyDrag);
        window.addEventListener("pointercancel", stopFlybodyDrag);
      });
      flybodyControlsEl.appendChild(titleEl);
      const controls = [
        ["walk_imitation", "walk", "Run FlyBody walk imitation"],
        ["walk_on_ball", "ball", "Run FlyBody walk-on-ball"],
        ["flight_imitation", "flight", "Run FlyBody flight imitation"],
        ["vision_guided_flight", "vision", "Run FlyBody vision-guided flight"],
        ["template_task", "template", "Run FlyBody template task"],
      ];
      const iconSvg = (kind) => {
        if (kind === "walk") {
          return '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M13 4a2 2 0 1 1 0 4 2 2 0 0 1 0-4Zm-2.2 5.2 1.8-.2 1.6 1.5c.4.4 1 .7 1.6.7H19v2h-3.2c-1.1 0-2.1-.4-2.9-1.1l-.8-.7-.7 3 1.8 1.8V20H11v-3.2l-2.2-2.3-.6 2.9H6l1.1-5.1 1.7-3.1c.4-.7 1.1-1.1 2-1.1Zm-2 1.6L7.1 14H4v-2h2l1.4-2.4 1.4 1.2Z"/></svg>';
        }
        if (kind === "ball") {
          return '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Zm6.7 6h-3a15 15 0 0 0-1.3-3.1A8.1 8.1 0 0 1 18.7 8ZM12 4.1A13 13 0 0 1 13.8 8h-3.6A13 13 0 0 1 12 4.1ZM5.3 14a8.4 8.4 0 0 1 0-4h3.4a17.5 17.5 0 0 0 0 4Zm.9 2h3a15 15 0 0 0 1.3 3.1A8.1 8.1 0 0 1 6.2 16Zm3-8h-3a8.1 8.1 0 0 1 4.3-3.1A15 15 0 0 0 9.2 8Zm2.8 11.9A13 13 0 0 1 10.2 16h3.6A13 13 0 0 1 12 19.9ZM14.2 14h-4.4a15.7 15.7 0 0 1 0-4h4.4a15.7 15.7 0 0 1 0 4Zm.3 5.1a15 15 0 0 0 1.3-3.1h3a8.1 8.1 0 0 1-4.3 3.1Zm1.8-5.1a17.5 17.5 0 0 0 0-4h3.4a8.4 8.4 0 0 1 0 4Z"/></svg>';
        }
        if (kind === "flight") {
          return '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M21 16v-2l-8-5V3.5A1.5 1.5 0 0 0 11.5 2h-1A1.5 1.5 0 0 0 9 3.5V9l-8 5v2l8-2.5V19l-2 1.5V22l4-1 4 1v-1.5L13 19v-5.5Z"/></svg>';
        }
        if (kind === "vision") {
          return '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M12 5c5 0 9.3 3.1 11 7-1.7 3.9-6 7-11 7S2.7 15.9 1 12c1.7-3.9 6-7 11-7Zm0 2.2A4.8 4.8 0 1 0 16.8 12 4.8 4.8 0 0 0 12 7.2Zm0 2A2.8 2.8 0 1 1 9.2 12 2.8 2.8 0 0 1 12 9.2Z"/></svg>';
        }
        return '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M5 4h10l4 4v12H5V4Zm9 1.5V9h3.5L14 5.5ZM8 12h8v2H8v-2Zm0 4h8v2H8v-2Zm0-8h4v2H8V8Z"/></svg>';
      };
      controls.forEach(([behavior, iconKind, titleText]) => {
        const btn = document.createElement("button");
        btn.className = "tool-btn flybody-tool-btn";
        if (behavior === simulationActiveBehavior) {
          btn.classList.add("active");
        }
        btn.innerHTML = iconSvg(iconKind);
        btn.title = titleText;
        btn.setAttribute("aria-label", titleText);
        btn.setAttribute("data-behavior", behavior);
        btn.addEventListener("click", () => issueFlybodyCommand("start", behavior));
        flybodyControlsEl.appendChild(btn);
      });
      const stopBtn = document.createElement("button");
      stopBtn.className = "tool-btn flybody-tool-btn flybody-stop-btn";
      stopBtn.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M6 6h12v12H6z"/></svg>';
      stopBtn.title = "Stop live FlyBody simulation";
      stopBtn.setAttribute("aria-label", "Stop live FlyBody simulation");
      stopBtn.addEventListener("click", () => {
        stopSimulationPlayback();
        issueFlybodyCommand("stop", "");
      });
      flybodyControlsEl.appendChild(stopBtn);
      flybodyControlsEl.hidden = false;
      positionFlybodyControls();
    };

    const _proceduralTextureCache = new Map();
    const buildProceduralTexture = (category, baseColor) => {
      const key = `${category}:${baseColor}`;
      if (_proceduralTextureCache.has(key)) return _proceduralTextureCache.get(key);
      const size = 128;
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      const c = new THREE.Color(baseColor);

      // Fill with base color
      ctx.fillStyle = `rgb(${Math.round(c.r * 255)},${Math.round(c.g * 255)},${Math.round(c.b * 255)})`;
      ctx.fillRect(0, 0, size, size);

      if (category === "wing") {
        // Vein pattern: thin radiating lines from base
        ctx.strokeStyle = `rgba(${Math.round(c.r * 255 * 0.6)},${Math.round(c.g * 255 * 0.6)},${Math.round(c.b * 255 * 0.6)},0.55)`;
        ctx.lineWidth = 0.7;
        for (let i = 0; i < 18; i++) {
          const angle = (i / 18) * Math.PI;
          ctx.beginPath();
          ctx.moveTo(size * 0.1, size * 0.85);
          ctx.lineTo(
            size * 0.1 + Math.cos(angle) * size * 1.25,
            size * 0.85 - Math.sin(angle) * size * 1.1
          );
          ctx.stroke();
        }
        // Cross veins
        ctx.lineWidth = 0.5;
        for (let row = 1; row < 5; row++) {
          ctx.beginPath();
          ctx.moveTo(0, size * row / 5);
          ctx.lineTo(size, size * row / 5 + (row % 2 === 0 ? -8 : 8));
          ctx.stroke();
        }
      } else if (category === "thorax" || category === "head" || category === "abdomen") {
        // Chitin grain: scattered dark micro-dots and faint grid
        ctx.fillStyle = `rgba(0,0,0,0.07)`;
        const seeded = category === "abdomen" ? 17 : category === "head" ? 31 : 53;
        for (let i = 0; i < 260; i++) {
          // Deterministic pseudo-random using simple LCG
          const px = ((i * 1664525 + 1013904223 + seeded) >>> 0) % size;
          const py = ((i * 22695477 + 1 + seeded * 3) >>> 0) % size;
          const r = 0.6 + ((i * 6364136 + seeded) >>> 0) % 100 / 200;
          ctx.beginPath();
          ctx.arc(px, py, r, 0, Math.PI * 2);
          ctx.fill();
        }
        // Faint segmentation lines for abdomen
        if (category === "abdomen") {
          ctx.strokeStyle = "rgba(0,0,0,0.12)";
          ctx.lineWidth = 1;
          for (let row = 1; row < 5; row++) {
            ctx.beginPath();
            ctx.moveTo(0, size * row / 5);
            ctx.lineTo(size, size * row / 5);
            ctx.stroke();
          }
        }
      } else {
        // Legs, antenna, etc: fine dot noise
        ctx.fillStyle = "rgba(0,0,0,0.06)";
        for (let i = 0; i < 80; i++) {
          const px = ((i * 1664525 + 101) >>> 0) % size;
          const py = ((i * 22695477 + 303) >>> 0) % size;
          ctx.beginPath();
          ctx.arc(px, py, 0.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      const tex = new THREE.CanvasTexture(canvas);
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      tex.repeat.set(3, 3);
      _proceduralTextureCache.set(key, tex);
      return tex;
    };

    const buildFlybodyPartMaterial = (part) => {
      const color = new THREE.Color(String(part && part.color ? part.color : "#c8ab72"));
      const roughness = Number(part && part.roughness);
      const metalness = Number(part && part.metalness);
      const category = String(part && part.category ? part.category : "body");
      let opacity = 1.0;
      let transparent = false;
      if (category === "wing") {
        opacity = 0.68;
        transparent = true;
      } else if (category === "antenna") {
        opacity = 0.92;
      }
      const colorHex = String(part && part.color ? part.color : "#c8ab72");
      const map = buildProceduralTexture(category, colorHex);
      return new THREE.MeshStandardMaterial({
        color,
        map,
        roughness: Number.isFinite(roughness) ? roughness : (category === "wing" ? 0.22 : 0.58),
        metalness: Number.isFinite(metalness) ? metalness : (category === "wing" ? 0.01 : 0.05),
        transparent,
        opacity,
        side: category === "wing" ? THREE.DoubleSide : THREE.FrontSide,
      });
    };


    const formatCategoryLabel = (category) =>
      String(category || "body")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (match) => match.toUpperCase());

    const updateBodyPartVisibility = () => {
      simulationBodyPartMap.forEach((entry) => {
        const enabled = simulationCategoryState.get(entry.category) !== false;
        entry.group.visible = enabled && entry.hasPose !== false;
      });
    };

    const rebuildSimulationCategoryUI = () => {
      if (!legendEl || !categoryPanelEl) return;
      legendEl.innerHTML = "";
      categoryPanelEl.innerHTML = "";
      const categories = Array.from(
        new Set(
          Array.from(simulationBodyPartMap.values()).map((entry) => String(entry.category || "body"))
        )
      ).sort();
      if (!categories.length) {
        legendEl.hidden = true;
        categoryPanelEl.hidden = true;
        return;
      }
      legendEl.hidden = false;
      categoryPanelEl.hidden = false;

      const legendTitle = document.createElement("div");
      legendTitle.className = "three-panel-title";
      legendTitle.textContent = "Part Legend";
      legendEl.appendChild(legendTitle);

      const toggleTitle = document.createElement("div");
      toggleTitle.className = "three-panel-title";
      toggleTitle.textContent = "Visible Parts";
      categoryPanelEl.appendChild(toggleTitle);

      categories.forEach((category) => {
        const first = Array.from(simulationBodyPartMap.values()).find(
          (entry) => String(entry.category || "body") === category
        );
        const color = first && first.color ? first.color : "#c8ab72";
        if (!simulationCategoryState.has(category)) {
          simulationCategoryState.set(category, true);
        }

        const legendRow = document.createElement("div");
        legendRow.className = "three-legend-row";
        const swatch = document.createElement("span");
        swatch.className = "three-legend-swatch";
        swatch.style.background = color;
        const text = document.createElement("span");
        text.textContent = formatCategoryLabel(category);
        legendRow.appendChild(swatch);
        legendRow.appendChild(text);
        legendEl.appendChild(legendRow);

        const toggleRow = document.createElement("label");
        toggleRow.className = "three-toggle-row";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = simulationCategoryState.get(category) !== false;
        checkbox.addEventListener("change", () => {
          simulationCategoryState.set(category, checkbox.checked);
          updateBodyPartVisibility();
        });
        const toggleText = document.createElement("span");
        toggleText.textContent = formatCategoryLabel(category);
        toggleRow.appendChild(checkbox);
        toggleRow.appendChild(toggleText);
        categoryPanelEl.appendChild(toggleRow);
      });
      updateBodyPartVisibility();
    };

    const stopSimulationPlayback = () => {
      simulationPlaying = false;
      if (timelinePlayBtn) timelinePlayBtn.textContent = "Play";
      if (simulationTimer !== null) {
        window.clearInterval(simulationTimer);
        simulationTimer = null;
      }
    };

    const updateSimulationMeta = (frame) => {
      if (!metaEl || !simulationPayload || !frame) return;
      const adapter = simulationPayload.adapter || "simulation";
      const qposLen = Array.isArray(frame.qpos) ? frame.qpos.length : 0;
      const dryRun = frame.dry_run ? "yes" : "no";
      const lines = [
        `Adapter: ${adapter}`,
        `Frame: ${frame.frame_index}`,
        `Points: ${Array.isArray(frame.points) ? frame.points.length : 0}`,
        `Qpos: ${qposLen}`,
        `Dry run: ${dryRun}`,
      ];
      if (Number.isFinite(frame.timestamp_sec)) {
        lines.push(`Time: ${Number(frame.timestamp_sec).toFixed(3)} s`);
      }
      metaEl.textContent = lines.join("\n");
    };

    const renderSimulationFrame = (index) => {
      if (!simulationPayload || !Array.isArray(simulationPayload.frames) || !simulationPayload.frames.length) {
        return;
      }
      const frames = simulationPayload.frames;
      const safeIndex = Math.max(0, Math.min(frames.length - 1, Number(index) || 0));
      const frame = frames[safeIndex];
      simulationFrameIndex = safeIndex;
      const display = simulationPayload && typeof simulationPayload.display === "object"
        ? simulationPayload.display
        : {};
      const showPoints = display.show_points !== false;
      const showLabels = display.show_labels !== false;
      const showEdges = display.show_edges !== false;
      const showTrails = display.show_trails !== false;

      clearGroupAndDispose(simulationPoints);
      clearGroupAndDispose(simulationEdges);
      clearGroupAndDispose(simulationTrails);
      clearGroupAndDispose(simulationLabels);

      const pointMap = new Map();
      const orderedPoints = Array.isArray(frame.points) ? frame.points : [];
      orderedPoints.forEach((pt, ptIdx) => {
        if (!pt || typeof pt !== "object") return;
        const x = Number(pt.x);
        const y = Number(pt.y);
        const z = Number(pt.z);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
        const label = String(pt.label || `point_${ptIdx}`);
        const color = new THREE.Color().setHSL((ptIdx * 0.13) % 1, 0.78, 0.56);
        if (showPoints) {
          const sphere = new THREE.Mesh(
            new THREE.SphereGeometry(0.02, 12, 12),
            new THREE.MeshStandardMaterial({
              color,
              emissive: color.clone().multiplyScalar(0.28),
              roughness: 0.35,
              metalness: 0.08,
            })
          );
          sphere.position.set(x, y, z);
          simulationPoints.add(sphere);
        }
        pointMap.set(label, new THREE.Vector3(x, y, z));

        if (showLabels) {
          const labelSprite = createBehaviorLabelSprite(label, `#${color.getHexString()}`);
          if (labelSprite) {
            labelSprite.scale.set(0.6, 0.16, 1);
            labelSprite.position.set(x, y + 0.035, z);
            simulationLabels.add(labelSprite);
          }
        }

        const start = Math.max(0, safeIndex - 24);
        const trailPoints = [];
        for (let i = start; i <= safeIndex; i += 1) {
          const srcFrame = frames[i];
          if (!srcFrame || !Array.isArray(srcFrame.points)) continue;
          const srcPoint = srcFrame.points.find((item) => item && String(item.label || "") === label);
          if (!srcPoint) continue;
          const tx = Number(srcPoint.x);
          const ty = Number(srcPoint.y);
          const tz = Number(srcPoint.z);
          if (!Number.isFinite(tx) || !Number.isFinite(ty) || !Number.isFinite(tz)) continue;
          trailPoints.push(new THREE.Vector3(tx, ty, tz));
        }
        if (showTrails && trailPoints.length > 1) {
          const trail = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(trailPoints),
            new THREE.LineBasicMaterial({
              color,
              transparent: true,
              opacity: 0.42,
            })
          );
          simulationTrails.add(trail);
        }
      });

      if (showEdges) {
        const edges = Array.isArray(simulationPayload.edges) ? simulationPayload.edges : [];
        edges.forEach((edge) => {
          if (!Array.isArray(edge) || edge.length < 2) return;
          const a = pointMap.get(String(edge[0] || ""));
          const b = pointMap.get(String(edge[1] || ""));
          if (!a || !b) return;
          const line = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([a, b]),
            new THREE.LineBasicMaterial({ color: 0xe7edf7, transparent: true, opacity: 0.78 })
          );
          simulationEdges.add(line);
        });
      }

      if (timelineSlider) timelineSlider.value = String(safeIndex);
      if (timelineLabel) timelineLabel.textContent = `Frame ${safeIndex + 1} / ${frames.length}`;
      const modelPose = frame.model_pose || {};
      const bodyPoses = frame.body_poses && typeof frame.body_poses === "object" ? frame.body_poses : {};
      const modelPos = Array.isArray(modelPose.position) ? modelPose.position : [0, 0, 0];
      const modelRot = Array.isArray(modelPose.rotation) ? modelPose.rotation : [0, 0, 0];
      const modelScale = Number(modelPose.scale);
      simulationModelRoot.position.set(
        Number(modelPos[0] || 0) * (Number.isFinite(modelScale) && modelScale > 0 ? modelScale : 1),
        Number(modelPos[2] || 0) * (Number.isFinite(modelScale) && modelScale > 0 ? modelScale : 1),
        Number(modelPos[1] || 0) * (Number.isFinite(modelScale) && modelScale > 0 ? modelScale : 1)
      );
      // Pre-rotate model root so MuJoCo Z-up aligns to Three.js Y-up,
      // then apply the payload's Euler Z-rotation (yaw) on top.
      const yaw = Number(modelRot[2] || 0);
      simulationModelRoot.rotation.set(-Math.PI / 2, 0, yaw);
      if (Number.isFinite(modelScale) && modelScale > 0) {
        simulationModelRoot.scale.setScalar(modelScale);
      }
      simulationBodyPartMap.forEach((entry, label) => {
        const pose = bodyPoses[label];
        const enabled = simulationCategoryState.get(entry.category) !== false;
        if (!pose || !Array.isArray(pose.position) || !Array.isArray(pose.quaternion)) {
          entry.group.visible = false;
          entry.hasPose = false;
          return;
        }
        entry.hasPose = true;
        entry.group.visible = enabled;
        entry.group.position.set(
          Number(pose.position[0] || 0),
          Number(pose.position[1] || 0),
          Number(pose.position[2] || 0)
        );
        const qw = Number(pose.quaternion[0] || 1);
        const qx = Number(pose.quaternion[1] || 0);
        const qy = Number(pose.quaternion[2] || 0);
        const qz = Number(pose.quaternion[3] || 0);
        entry.group.quaternion.set(qx, qy, qz, qw);
      });
      updateSimulationMeta(frame);
      setStatus(`Viewing ${title}: frame ${safeIndex + 1} / ${frames.length}`);
    };

    const loadSimulationPayload = (payload) => {
      if (!payload || payload.kind !== "annolid-simulation-v1") {
        throw new Error("Unsupported simulation payload");
      }
      simulationPayload = payload;
      simulationActiveBehavior = String(
        (((payload.metadata || {}).run_metadata || {}).behavior || "")
      );
      document.body.setAttribute("data-threejs-simulation", "1");
      root.position.set(0, 0, 0);
      root.rotation.set(0, 0, 0);
      root.scale.set(1, 1, 1);
      if (timelineEl) timelineEl.hidden = false;
      if (metaEl) metaEl.hidden = false;
      simulationLoopEnabled = !payload.playback || payload.playback.loop !== false;
      const nextMeshKey = JSON.stringify(payload.mesh || null);
      const nextEnvironmentKey = JSON.stringify(payload.environment || null);
      const shouldReloadMesh = nextMeshKey !== simulationMeshKey;
      const shouldReloadEnvironment = nextEnvironmentKey !== simulationEnvironmentKey;
      if (shouldReloadEnvironment) {
        clearGroupAndDispose(simulationEnvironmentRoot);
        simulationEnvironmentKey = nextEnvironmentKey;
      }
      if (shouldReloadMesh) {
        clearGroupAndDispose(simulationModelRoot);
        clearGroupAndDispose(simulationBodyPartsRoot);
        simulationBodyPartMap = new Map();
        simulationCategoryState = new Map();
        simulationMeshKey = nextMeshKey;
      }
      if (legendEl) {
        legendEl.hidden = true;
        legendEl.innerHTML = "";
      }
      if (categoryPanelEl) {
        categoryPanelEl.hidden = true;
        categoryPanelEl.innerHTML = "";
      }
      if (simulationRoot.parent !== root) {
        root.add(simulationRoot);
      }
      const floor = payload.environment && payload.environment.floor ? payload.environment.floor : null;
      if (shouldReloadEnvironment && floor && floor.type === "plane") {
        const floorSize = Array.isArray(floor.size) ? floor.size : [5, 5];
        const floorPos = Array.isArray(floor.position) ? floor.position : [0, 0, -0.132];
        const width = Number(floorSize[0] || 5) * 2;
        const depth = Number(floorSize[1] || 5) * 2;
        const plane = new THREE.Mesh(
          new THREE.PlaneGeometry(width, depth),
          new THREE.MeshStandardMaterial({
            color: new THREE.Color(String(floor.color || "#314759")),
            roughness: 0.92,
            metalness: 0.02,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5,
          })
        );
        plane.rotation.x = -Math.PI / 2;
        // Floor height is a Y-coordinate in Three.js Y-up world space
        const floorY = Number(
          Array.isArray(floor.position) ? (floor.position[1] != null ? floor.position[1] : 0) : 0
        );
        plane.position.set(0, floorY, 0);
        simulationEnvironmentRoot.add(plane);
        const grid = new THREE.GridHelper(
          Math.max(width, depth),
          20,
          new THREE.Color(String(floor.gridColor || "#3f5f73")),
          new THREE.Color(String(floor.gridColor || "#3f5f73"))
        );
        grid.position.set(0, floorY, 0);
        simulationEnvironmentRoot.add(grid);
      }
      rebuildFlybodyControls();
      if (!shouldReloadMesh) {
        if (!payload.mesh || payload.mesh.type !== "flybody_parts") {
          if (legendEl) legendEl.hidden = true;
          if (categoryPanelEl) categoryPanelEl.hidden = true;
        }
      } else if (payload.mesh && payload.mesh.type === "flybody_parts" && Array.isArray(payload.mesh.parts)) {
        const loader = new OBJLoader();
        payload.mesh.parts.forEach((part) => {
          if (part && (!part.type || part.type === "obj")) {
            const meshUrl = new URL(String(part.path || ""), modelUrl).toString();
            const bodyLabel = String(part.body || "");
            if (!meshUrl || !bodyLabel) return;
            loader.load(
              meshUrl,
              (obj) => {
                const material = buildFlybodyPartMaterial(part);
                obj.traverse((child) => {
                  if (child && child.isMesh) {
                    child.material = material.clone();
                    child.castShadow = false;
                    child.receiveShadow = false;
                  }
                });
                const partRoot = new THREE.Group();
                partRoot.add(obj);
                simulationBodyPartsRoot.add(partRoot);
                simulationBodyPartMap.set(bodyLabel, {
                  group: partRoot,
                  category: String(part.category || "body"),
                  color: String(part.color || "#c8ab72"),
                  hasPose: false,
                });
                rebuildSimulationCategoryUI();
                renderSimulationFrame(simulationFrameIndex);
              },
              undefined,
              (err) => {
                console.warn("Failed to load FlyBody body part mesh:", bodyLabel, err);
              }
            );
          }
        });
      } else if (payload.mesh && payload.mesh.type === "obj" && payload.mesh.path) {
        const meshUrl = new URL(String(payload.mesh.path), modelUrl).toString();
        // Derive MTL URL from OBJ URL (same filename, .mtl extension)
        const mtlUrl = meshUrl.replace(/\.obj(\?.*)?$/i, ".mtl");
        const applyObjToScene = (obj) => {
          obj.traverse((child) => {
            if (child && child.isMesh) {
              // Ensure wing materials are transparent
              if (child.material) {
                const mats = Array.isArray(child.material) ? child.material : [child.material];
                mats.forEach((mat) => {
                  // "membrane" = wing skin in fruitfly.xml (rgba alpha=0.4)
                  if (mat && mat.name && (
                    mat.name === "membrane" ||
                    mat.name.toLowerCase().includes("wing")
                  )) {
                    mat.transparent = true;
                    mat.side = THREE.DoubleSide;
                  }
                  if (mat) mat.needsUpdate = true;
                });
              }
              child.castShadow = false;
              child.receiveShadow = false;
            }
          });
          simulationModelRoot.add(obj);
          renderSimulationFrame(simulationFrameIndex);
        };
        const fallbackLoad = () => {
          const loader = new OBJLoader();
          loader.load(
            meshUrl,
            applyObjToScene,
            undefined,
            (err) => { console.warn("Failed to load FlyBody example mesh:", err); }
          );
        };
        try {
          const mtlLoader = new MTLLoader();
          mtlLoader.setResourcePath(meshUrl.replace(/[^/]*$/, ""));
          mtlLoader.load(
            mtlUrl,
            (materials) => {
              materials.preload();
              const objLoader = new OBJLoader();
              objLoader.setMaterials(materials);
              objLoader.load(meshUrl, applyObjToScene, undefined, fallbackLoad);
            },
            undefined,
            () => fallbackLoad()
          );
        } catch (_e) {
          fallbackLoad();
        }
      }

      if (!payload.mesh || payload.mesh.type !== "flybody_parts") {
        if (legendEl) legendEl.hidden = true;
        if (categoryPanelEl) categoryPanelEl.hidden = true;
      }
      if (timelineSlider) {
        const frameCount = Array.isArray(payload.frames) ? payload.frames.length : 0;
        timelineSlider.min = "0";
        timelineSlider.max = String(Math.max(0, frameCount - 1));
        timelineSlider.step = "1";
        timelineSlider.value = "0";
        timelineSlider.oninput = (event) => {
          const value = Number(event && event.target ? event.target.value : 0);
          renderSimulationFrame(value);
        };
      }
      if (timelinePlayBtn) {
        timelinePlayBtn.onclick = () => {
          if (!simulationPayload || !Array.isArray(simulationPayload.frames) || simulationPayload.frames.length < 2) {
            return;
          }
          if (simulationPlaying) {
            stopSimulationPlayback();
            return;
          }
          simulationPlaying = true;
          timelinePlayBtn.textContent = "Pause";
          const intervalMs = Number(
            (simulationPayload.playback && simulationPayload.playback.interval_ms) || 120
          );
          simulationTimer = window.setInterval(() => {
            const nextIndex = simulationFrameIndex + 1;
            if (nextIndex >= simulationPayload.frames.length) {
              if (!simulationLoopEnabled) {
                stopSimulationPlayback();
                return;
              }
              renderSimulationFrame(0);
              return;
            }
            renderSimulationFrame(nextIndex);
          }, intervalMs);
        };
      }
      renderSimulationFrame(0);
      if (payload.playback && payload.playback.autoplay && timelinePlayBtn) {
        timelinePlayBtn.click();
      }
      fitCameraToObject(simulationRoot);
      setStatus(`Loaded ${title} (${(payload.adapter || "simulation").toUpperCase()}).`);
      document.body.setAttribute("data-threejs-ready", "1");
    };

    window.__annolidLoadSimulationPayloadFromUrl = async (nextUrl, nextTitle) => {
      try {
        const resp = await fetch(String(nextUrl || modelUrl), { cache: "no-store" });
        if (!resp.ok) {
          throw new Error(`Unable to fetch simulation: HTTP ${resp.status}`);
        }
        const payload = await resp.json();
        if (nextTitle) {
          window.__annolidThreeTitle = nextTitle;
        }
        loadSimulationPayload(payload);
      } catch (err) {
        setStatus(`Failed to update simulation payload: ${err.message}`, "error");
      }
    };

    window.addEventListener("resize", positionFlybodyControls);

    if (modelUrl) {
      if (ext === "stl") {
        const loader = new STLLoader();
        loader.load(
          modelUrl,
          (geometry) => {
            geometry.computeVertexNormals();
            const mat = new THREE.MeshStandardMaterial({
              color: 0x8cc6ff,
              roughness: 0.55,
              metalness: 0.1,
            });
            addLoadedObject(new THREE.Mesh(geometry, mat));
          },
          undefined,
          (err) => {
            setStatus(`Failed to load STL: ${err}`, "error");
          }
        );
      } else if (ext === "ply") {
        // Check if this is a Gaussian Splats PLY file
        fetch(modelUrl)
          .then(response => response.arrayBuffer())
          .then(buffer => {
            const isGaussianSplats = detectGaussianSplatsPLY(buffer);
            if (isGaussianSplats) {
              loadGaussianSplatsPLY(buffer, addLoadedObject, setStatus);
            } else {
              // Regular PLY loading - reuse the buffer to avoid redundant fetch
              const loader = new PLYLoader();
              try {
                const geometry = loader.parse(buffer);
                geometry.computeVertexNormals();
                const hasIndex =
                  geometry.getIndex() && geometry.getAttribute("normal") !== undefined;
                if (hasIndex) {
                  const mat = new THREE.MeshStandardMaterial({
                    color: 0x84d6bf,
                    roughness: 0.5,
                    metalness: 0.12,
                  });
                  addLoadedObject(new THREE.Mesh(geometry, mat));
                } else {
                  const mat = new THREE.PointsMaterial({
                    color: 0x84d6bf,
                    size: 0.01,
                    sizeAttenuation: true,
                  });
                  addLoadedObject(new THREE.Points(geometry, mat));
                }
              } catch (err) {
                setStatus(`Failed to parse PLY: ${err}`, "error");
              }
            }
          })
          .catch(err => {
            setStatus(`Failed to load PLY: ${err}`, "error");
          });
      } else if (ext === "obj") {
        // Try to load MTL file first
        const title = window.__annolidThreeTitle || "";
        const baseName = title.replace(/\.obj$/i, '');
        const mtlFilename = baseName + '.mtl';
        const mtlUrl = modelUrl.substring(0, modelUrl.lastIndexOf('/') + 1) + mtlFilename;

        const loadOBJ = (materials = null) => {
          const loader = new OBJLoader();
          if (materials) {
            loader.setMaterials(materials);
          }
          loader.load(
            modelUrl,
            (obj) => {
              if (!materials) {
                obj.traverse((child) => {
                  if (child && child.isMesh && child.material) {
                    // Clear all texture maps to prevent GL errors
                    if (child.material.map) child.material.map = null;
                    if (child.material.normalMap) child.material.normalMap = null;
                    if (child.material.roughnessMap) child.material.roughnessMap = null;
                    if (child.material.metalnessMap) child.material.metalnessMap = null;
                    if (child.material.emissiveMap) child.material.emissiveMap = null;
                    if (child.material.aoMap) child.material.aoMap = null;
                    if (child.material.bumpMap) child.material.bumpMap = null;
                    if (child.material.displacementMap) child.material.displacementMap = null;
                    // Then set a simple material
                    child.material = new THREE.MeshStandardMaterial({
                      color: 0xb7b9ff,
                      roughness: 0.58,
                      metalness: 0.1,
                    });
                  }
                });
              }
              addLoadedObject(obj);
            },
            undefined,
            (err) => {
              setStatus(`Failed to load OBJ: ${err}`, "error");
            }
          );
        };

        const mtlLoader = new MTLLoader();
        const mtlDir = mtlUrl.substring(0, mtlUrl.lastIndexOf('/') + 1);
        const mtlFile = mtlUrl.substring(mtlUrl.lastIndexOf('/') + 1);

        mtlLoader.setPath(mtlDir);
        mtlLoader.load(
          mtlFile,
          (materials) => {
            materials.preload();
            loadOBJ(materials);
          },
          undefined,
          (err) => {
            console.warn("MTL load failed, falling back to simple OBJ:", err);
            loadOBJ();
          }
        );
      } else if (ext === "glb" || ext === "gltf") {
        const loader = new GLTFLoader();
        loader.load(
          modelUrl,
          (gltf) => {
            addLoadedObject(gltf.scene || new THREE.Group());
          },
          undefined,
          (err) => {
            setStatus(`Failed to load ${ext.toUpperCase()}: ${err}`, "error");
          }
        );
      } else if (ext === "csv" || ext === "xyz") {
        parseDelimitedPointCloud()
          .then((cloud) => {
            if (cloud) addLoadedObject(cloud);
            else setStatus("No valid XYZ rows found in file.", "error");
          })
          .catch((err) => {
            setStatus(`Failed to load point cloud: ${err.message}`, "error");
          });
      } else if (ext === "json") {
        parseJsonPayload()
          .then((payload) => {
            loadSimulationPayload(payload);
          })
          .catch((err) => {
            setStatus(`Failed to load simulation payload: ${err.message}`, "error");
          });
      } else {
        setStatus(`Unsupported 3D format: .${ext}`, "error");
        document.body.setAttribute("data-threejs-error", "unsupported-format");
        return;
      }
    } else {
      // No model, just ready for real-time
      setStatus("Real-time mode active.");
      document.body.setAttribute("data-threejs-ready", "1");
    }

    // --- Real-time Video and Pose Integration ---
    let videoPlane, videoTexture, videoCanvas, videoCtx;
    let realtimeFrameSeq = 0;
    const poseGroup = new THREE.Group();
    scene.add(poseGroup);

    const handGroup = new THREE.Group();
    scene.add(handGroup);

    // --- Gaze Integration ---
    const gazeReticle = new THREE.Group();
    const reticleRing = new THREE.Mesh(
      new THREE.RingGeometry(0.1, 0.12, 32),
      new THREE.MeshBasicMaterial({ color: 0xff3366, transparent: true, opacity: 0.8 })
    );
    const reticleDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xff3366 })
    );
    gazeReticle.add(reticleRing);
    gazeReticle.add(reticleDot);
    gazeReticle.visible = false;
    scene.add(gazeReticle);

    const initRealtimeVideo = () => {
      // Use a persistent CanvasTexture to avoid texture-object churn in embedded Chromium.
      videoCanvas = document.createElement("canvas");
      videoCanvas.width = 2;
      videoCanvas.height = 2;
      videoCtx = videoCanvas.getContext("2d", { alpha: false });
      if (videoCtx) {
        videoCtx.fillStyle = "#000";
        videoCtx.fillRect(0, 0, videoCanvas.width, videoCanvas.height);
      }
      videoTexture = new THREE.CanvasTexture(videoCanvas);
      videoTexture.colorSpace = THREE.SRGBColorSpace;
      videoTexture.minFilter = THREE.LinearFilter;
      videoTexture.magFilter = THREE.LinearFilter;
      videoTexture.wrapS = THREE.ClampToEdgeWrapping;
      videoTexture.wrapT = THREE.ClampToEdgeWrapping;
      videoTexture.generateMipmaps = false;
      videoTexture.needsUpdate = true;
      const material = new THREE.MeshBasicMaterial({
        map: videoTexture,
        side: THREE.DoubleSide,
        depthTest: false,
        depthWrite: false
      });
      const geometry = new THREE.PlaneGeometry(1, 1);
      videoPlane = new THREE.Mesh(geometry, material);
      videoPlane.position.z = -5; // Background
      videoPlane.visible = false;
      scene.add(videoPlane);
    };
    initRealtimeVideo();

    const poseKeypoints = new THREE.Group();
    const poseSkeleton = new THREE.Group();
    const behaviorLabels = new THREE.Group();
    const irisGroup = new THREE.Group();
    poseGroup.add(poseKeypoints);
    poseGroup.add(poseSkeleton);
    poseGroup.add(behaviorLabels);
    poseGroup.add(irisGroup);
    // Render on top
    poseGroup.renderOrder = 999;

    const createBehaviorLabelSprite = (text, colorHex) => {
      const labelText = String(text || "").trim();
      if (!labelText) return null;
      const c = document.createElement("canvas");
      c.width = 512;
      c.height = 128;
      const ctx = c.getContext("2d");
      if (!ctx) return null;
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
      ctx.fillRect(8, 12, c.width - 16, c.height - 24);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(8, 12, c.width - 16, c.height - 24);
      ctx.font = "bold 48px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = colorHex || "#ffffff";
      ctx.fillText(labelText, c.width / 2, c.height / 2);
      const texture = new THREE.CanvasTexture(c);
      texture.needsUpdate = true;
      const material = new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthTest: false,
        depthWrite: false
      });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(2.8, 0.7, 1.0);
      sprite.renderOrder = 1001;
      return sprite;
    };

    const clearGroupAndDispose = (group) => {
      if (!group) return;
      const nodes = [...group.children];
      nodes.forEach((node) => {
        if (node.geometry && typeof node.geometry.dispose === "function") {
          node.geometry.dispose();
        }
        const disposeMaterial = (mat) => {
          if (!mat) return;
          if (mat.map && typeof mat.map.dispose === "function") {
            mat.map.dispose();
          }
          if (typeof mat.dispose === "function") {
            mat.dispose();
          }
        };
        if (Array.isArray(node.material)) {
          node.material.forEach(disposeMaterial);
        } else {
          disposeMaterial(node.material);
        }
      });
      group.clear();
    };

    const overlaySphereGeometry = new THREE.SphereGeometry(0.05, 8, 8);
    const handSphereGeometry = new THREE.SphereGeometry(0.04, 8, 8);
    const materialCache = new Map();
    const getCachedBasicMaterial = (hex, opacity = 1.0) => {
      const key = `${hex}:${opacity}`;
      if (materialCache.has(key)) return materialCache.get(key);
      const mat = new THREE.MeshBasicMaterial({
        color: hex,
        transparent: opacity < 1.0,
        opacity
      });
      materialCache.set(key, mat);
      return mat;
    };

    window.updateRealtimeData = (base64Frame, detections, frameWidthArg = 0, frameHeightArg = 0) => {
      if (!realtimeEnabled) return;
      const overlayZ = (videoPlane && Number.isFinite(videoPlane.position.z))
        ? (videoPlane.position.z + 0.02)
        : -4.98;
      const labelZ = overlayZ + 0.02;
      const gazeZ = overlayZ + 0.03;

      // 1. Update Video Frame
      if (base64Frame) {
        const seq = ++realtimeFrameSeq;
        const img = new Image();
        img.onload = () => {
          if (!videoPlane || !videoTexture || !videoCanvas || !videoCtx) return;
          // Drop stale async decodes so older textures do not race newer ones.
          if (seq !== realtimeFrameSeq) return;

          // Keep a fixed texture object; only update canvas pixels.
          if (videoCanvas.width !== img.width || videoCanvas.height !== img.height) {
            videoCanvas.width = img.width;
            videoCanvas.height = img.height;
          }
          videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
          videoCtx.drawImage(img, 0, 0, videoCanvas.width, videoCanvas.height);
          videoTexture.needsUpdate = true;
          if (videoPlane.material.map !== videoTexture) {
            videoPlane.material.map = videoTexture;
            videoPlane.material.needsUpdate = true;
          }

          // Adjust plane size to maintain aspect ratio
          const aspect = img.width / Math.max(1, img.height);
          videoPlane.scale.set(aspect * 10, 10, 1);
          videoPlane.visible = true;
        };
        img.src = `data:image/jpeg;base64,${base64Frame}`;
      }

      // 2. Update Poses and Hands
      clearGroupAndDispose(poseKeypoints);
      clearGroupAndDispose(poseSkeleton);
      clearGroupAndDispose(behaviorLabels);
      clearGroupAndDispose(irisGroup);
      clearGroupAndDispose(handGroup);

      if (Array.isArray(detections) && detections.length > 0) {
        detections.forEach((det, detIdx) => {
          // Prefer normalized keypoints because scene mapping assumes [0..1].
          const kps = det.keypoints || det.keypoints_pixels;
          if (!kps) return;

          const color = new THREE.Color().setHSL((detIdx * 0.1) % 1, 0.8, 0.5);
          const sphereMat = getCachedBasicMaterial(color.getHex(), 1.0);

          const points = [];
          const aspect = videoPlane.scale.x / videoPlane.scale.y;

          const frameW = Number(frameWidthArg || 0) > 0
            ? Number(frameWidthArg)
            : (videoTexture && videoTexture.image ? Number(videoTexture.image.width || 0) : 0);
          const frameH = Number(frameHeightArg || 0) > 0
            ? Number(frameHeightArg)
            : (videoTexture && videoTexture.image ? Number(videoTexture.image.height || 0) : 0);

          const projected = new Array(kps.length);
          kps.forEach((kp, kpIdx) => {
            if (!Array.isArray(kp) || kp.length < 2) return;
            let nx = Number(kp[0]);
            let ny = Number(kp[1]);
            if (!Number.isFinite(nx) || !Number.isFinite(ny)) return;
            const looksNormalized = nx >= 0 && nx <= 1.2 && ny >= 0 && ny <= 1.2;
            if (!looksNormalized && frameW > 1 && frameH > 1) {
              nx = nx / frameW;
              ny = ny / frameH;
            }
            nx = Math.max(0, Math.min(1, nx));
            ny = Math.max(0, Math.min(1, ny));

            // Map normalized [0..1] to scene space [-aspect*5, aspect*5] and [5, -5]
            const x = (nx - 0.5) * (aspect * 10);
            const y = -(ny - 0.5) * 10;
            projected[kpIdx] = { x, y };

            const kpMesh = new THREE.Mesh(overlaySphereGeometry, sphereMat);
            kpMesh.position.set(x, y, overlayZ); // Slightly in front of video plane
            poseKeypoints.add(kpMesh);
            points.push(new THREE.Vector3(x, y, overlayZ));
          });

          // Draw skeleton if we have multiple points
          if (points.length > 1) {
            const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
            const line = new THREE.Line(lineGeo, lineMat);
            poseSkeleton.add(line);
          }
          const behaviorText = String(det.behavior || "").trim();
          if (behaviorText && points.length > 0) {
            const labelSprite = createBehaviorLabelSprite(
              behaviorText,
              `#${color.getHexString()}`
            );
            if (labelSprite) {
              // Show behavior labels in a fixed top-left stack on the frame.
              const labelMarginX = 1.4;
              const labelMarginY = 0.5;
              const lineSpacing = 0.78;
              const labelX = -(aspect * 5) + labelMarginX;
              const labelY = 5 - labelMarginY - (detIdx * lineSpacing);
              labelSprite.position.set(labelX, labelY, labelZ);
              behaviorLabels.add(labelSprite);
            }
          }

          // Iris visualization for MediaPipe face landmarks.
          if (projected.length >= 478) {
            const LEFT_IRIS = [468, 469, 470, 471, 472];
            const RIGHT_IRIS = [473, 474, 475, 476, 477];
            const renderIris = (indices, colorHex) => {
              const center = projected[indices[0]];
              if (!center) return;
              const ringPts = indices
                .slice(1)
                .map((idx) => projected[idx])
                .filter((p) => !!p);
              if (ringPts.length < 2) return;

              // Estimate iris radius from boundary landmarks.
              let sum = 0.0;
              ringPts.forEach((p) => {
                const dx = p.x - center.x;
                const dy = p.y - center.y;
                sum += Math.sqrt(dx * dx + dy * dy);
              });
              const radius = Math.max(0.03, Math.min(0.25, sum / ringPts.length));

              const ringGeo = new THREE.RingGeometry(radius * 0.72, radius, 24);
              const ringMat = new THREE.MeshBasicMaterial({
                color: colorHex,
                transparent: true,
                opacity: 0.9,
                side: THREE.DoubleSide,
                depthTest: false,
                depthWrite: false
              });
              const ring = new THREE.Mesh(ringGeo, ringMat);
              ring.position.set(center.x, center.y, overlayZ + 0.015);
              irisGroup.add(ring);

              const pupilGeo = new THREE.SphereGeometry(radius * 0.26, 12, 12);
              const pupilMat = new THREE.MeshBasicMaterial({
                color: colorHex,
                transparent: true,
                opacity: 0.95,
                depthTest: false,
                depthWrite: false
              });
              const pupil = new THREE.Mesh(pupilGeo, pupilMat);
              pupil.position.set(center.x, center.y, overlayZ + 0.02);
              irisGroup.add(pupil);
            };
            renderIris(LEFT_IRIS, 0x33ffcc);
            renderIris(RIGHT_IRIS, 0x66ccff);
          }

          // Keep hand pinch state available for downstream gaze/interaction logic.
          let leftPinch = null;
          let rightPinch = null;

          // 4. Update Hand Controls
          if (det.metadata && det.metadata.hands) {
            const handsData = det.metadata.hands;
            const aspect = videoPlane.scale.x / videoPlane.scale.y;

            handsData.forEach((hand) => {
              const color = hand.label === "Left" ? 0x33ff66 : 0x3366ff;
              const isPinching = hand.is_pinching;

              // Draw hand landmarks
              const mat = getCachedBasicMaterial(color, 0.7);

              hand.landmarks.forEach((kp, idx) => {
                // Only draw tips for less clutter
                if (idx % 4 !== 0 && idx !== 0) return;

                const x = (kp[0] - 0.5) * (aspect * 10);
                const y = -(kp[1] - 0.5) * 10;
                const mesh = new THREE.Mesh(handSphereGeometry, mat);
                mesh.position.set(x, y, overlayZ + 0.01);
                handGroup.add(mesh);

                // Pinch visual
                if (isPinching && (idx === 4 || idx === 8)) {
                  mesh.scale.set(1.5, 1.5, 1.5);
                  mesh.material.opacity = 1.0;
                }
              });

              // Track pinches for interaction
              const p4 = hand.landmarks[4];
              const p8 = hand.landmarks[8];
              const pinchCenter = [
                (p4[0] + p8[0]) / 2,
                (p4[1] + p8[1]) / 2
              ];

              if (isPinching) {
                if (hand.label === "Left") leftPinch = pinchCenter;
                else rightPinch = pinchCenter;
              }
            });

            // --- Hand Interactions Logic ---
            const lerpFactor = 0.2;

            if (window.__annolidEnableHandControl) {
              // 1. Zoom: Both hands pinching
              if (leftPinch && rightPinch) {
                updateIndicator(indZoom, true);
                updateIndicator(indRotate, false);
                updateIndicator(indPan, false);

                const dist = Math.sqrt(
                  Math.pow(leftPinch[0] - rightPinch[0], 2) +
                  Math.pow(leftPinch[1] - rightPinch[1], 2)
                );

                if (window.__prevHandDist !== undefined) {
                  const delta = dist - window.__prevHandDist;
                  const zoomSpeed = 15;
                  camera.position.multiplyScalar(1 - delta * zoomSpeed);
                }
                window.__prevHandDist = dist;
                window.__prevLeftPinch = null;
                window.__prevRightPinch = null;
              }
              // 2. Rotate: Right hand only pinch (alternative to eye control)
              else if (rightPinch) {
                updateIndicator(indRotate, true);
                updateIndicator(indZoom, false);
                updateIndicator(indPan, false);

                if (window.__prevRightPinch) {
                  const dx = rightPinch[0] - window.__prevRightPinch[0];
                  const dy = rightPinch[1] - window.__prevRightPinch[1];

                  const rotateSpeed = 5;
                  root.rotation.y += dx * rotateSpeed;
                  root.rotation.x += dy * rotateSpeed;
                }
                window.__prevRightPinch = rightPinch;
                window.__prevLeftPinch = null;
                window.__prevHandDist = undefined;
              }
              // 3. Pan: Left hand only pinch
              else if (leftPinch) {
                updateIndicator(indPan, true);
                updateIndicator(indRotate, false);
                updateIndicator(indZoom, false);

                if (window.__prevLeftPinch) {
                  const dx = leftPinch[0] - window.__prevLeftPinch[0];
                  const dy = leftPinch[1] - window.__prevLeftPinch[1];

                  const panSpeed = 10;
                  // Move camera target for panning
                  const offset = new THREE.Vector3(-dx * panSpeed, dy * panSpeed, 0);
                  offset.applyQuaternion(camera.quaternion);
                  controls.target.add(offset);
                  camera.position.add(offset);
                }
                window.__prevLeftPinch = leftPinch;
                window.__prevRightPinch = null;
                window.__prevHandDist = undefined;
              } else {
                updateIndicator(indRotate, false);
                updateIndicator(indPan, false);
                updateIndicator(indZoom, false);

                window.__prevLeftPinch = null;
                window.__prevRightPinch = null;
                window.__prevHandDist = undefined;
              }
            }
          }

          // 5. Update Gaze if present
          if (det.metadata && det.metadata.gaze_avg) {
            const gaze = det.metadata.gaze_avg;

            // Only show indicator if eye control is actually toggled on and no hand gesture is overriding
            const handActive = leftPinch || rightPinch;
            if (window.__annolidEnableEyeControl && !handActive) {
              updateIndicator(indRotate, true);
            }

            // Smoothing
            const lerpFactor = 0.25;
            const gx = (gaze[0] * (aspect * 5)) * lerpFactor + gazeReticle.position.x * (1 - lerpFactor);
            const gy = (-gaze[1] * 5) * lerpFactor + gazeReticle.position.y * (1 - lerpFactor);

            gazeReticle.position.set(gx, gy, gazeZ);
            gazeReticle.visible = true;

            if (window.__annolidEnableEyeControl && root.children.length > 0) {
              // Map gaze [-1..1] to rotation
              const targetRotX = gaze[1] * 1.2;
              const targetRotY = gaze[0] * 1.2;
              root.rotation.x = targetRotX * lerpFactor + root.rotation.x * (1 - lerpFactor);
              root.rotation.y = targetRotY * lerpFactor + root.rotation.y * (1 - lerpFactor);
            }
          } else {
            gazeReticle.visible = false;
          }
        });
      } else {
        gazeReticle.visible = false;
      }
    };

    const onResize = () => {
      const { w, h } = getCanvasSize();
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      if (videoPlane) {
        // Keep background plane centered
        videoPlane.position.set(0, 0, -5);
      }
    };
    window.addEventListener("resize", onResize, { passive: true });

    let animationFrameId = 0;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrameId = window.requestAnimationFrame(tick);
    };
    tick();

    const disposeViewer = () => {
      window.removeEventListener("resize", onResize);
      if (animationFrameId) {
        window.cancelAnimationFrame(animationFrameId);
      }
      controls.enabled = false;
      controls.dispose();
      clearGroupAndDispose(poseKeypoints);
      clearGroupAndDispose(poseSkeleton);
      clearGroupAndDispose(behaviorLabels);
      clearGroupAndDispose(irisGroup);
      clearGroupAndDispose(handGroup);
      clearGroupAndDispose(root);
      if (overlaySphereGeometry) overlaySphereGeometry.dispose();
      if (handSphereGeometry) handSphereGeometry.dispose();
      materialCache.forEach((mat) => {
        if (mat && typeof mat.dispose === "function") mat.dispose();
      });
      materialCache.clear();
      if (videoTexture) videoTexture.dispose();
      if (videoPlane && videoPlane.geometry) videoPlane.geometry.dispose();
      if (videoPlane && videoPlane.material) {
        if (videoPlane.material.map && typeof videoPlane.material.map.dispose === "function") {
          videoPlane.material.map.dispose();
        }
        if (typeof videoPlane.material.dispose === "function") videoPlane.material.dispose();
      }
      // Avoid hard renderer disposal on Qt WebEngine unload. Aggressive context
      // teardown during page switch can produce Chromium shared-image mailbox errors.
      // Let page teardown release renderer resources.
    };
    window.addEventListener("beforeunload", disposeViewer, { once: true });
  } catch (err) {
    const msg = String(err || "Failed to initialize Three.js viewer");
    console.error(err);
    setStatus(msg, "error");
    document.body.setAttribute("data-threejs-error", msg);
  }
}

boot();
