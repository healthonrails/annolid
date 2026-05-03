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
  const volumePanelEl = document.getElementById("annolidThreeVolumePanel");
  const btnToggleVolumePanel = document.getElementById("btnToggleVolumePanel");
  const modelUrl = window.__annolidThreeModelUrl || "";
  const modelExtHint = (window.__annolidThreeModelExt || "").toLowerCase();
  const title = window.__annolidThreeTitle || "3D";
  const pickMode = String(window.__annolidThreePickMode || "");
  const objectRegionMap = (() => {
    const raw = window.__annolidThreeObjectRegionMap;
    if (!raw || typeof raw !== "object") return {};
    return raw;
  })();

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

    let rendererBackend = "webgl";
    const createRenderer = async () => {
      const maxPixelRatio = Math.min(Math.max(1, window.devicePixelRatio || 1), 2);
      if (typeof navigator !== "undefined" && navigator.gpu) {
        try {
          const webgpuModule = await import(
            "https://esm.sh/three@0.182.0/examples/jsm/renderers/webgpu/WebGPURenderer.js"
          );
          const WebGPURenderer = webgpuModule.WebGPURenderer || webgpuModule.default;
          if (WebGPURenderer) {
            const webgpuRenderer = new WebGPURenderer({
              canvas,
              antialias: false,
              alpha: false,
              powerPreference: "high-performance",
            });
            if (typeof webgpuRenderer.init === "function") {
              await webgpuRenderer.init();
            }
            webgpuRenderer.setPixelRatio(maxPixelRatio);
            rendererBackend = "webgpu";
            return webgpuRenderer;
          }
        } catch (_err) {
          rendererBackend = "webgl";
        }
      }
      const webglRenderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
      });
      webglRenderer.setPixelRatio(maxPixelRatio);
      rendererBackend = "webgl";
      return webglRenderer;
    };

    const renderer = await createRenderer();
    const getCanvasSize = () => {
      const w = Math.max(1, canvas.clientWidth || window.innerWidth || 800);
      const h = Math.max(1, canvas.clientHeight || window.innerHeight || 600);
      return { w, h };
    };
    {
      const { w, h } = getCanvasSize();
      renderer.setSize(w, h, false);
    }
    if ("outputColorSpace" in renderer) {
      renderer.outputColorSpace = THREE.SRGBColorSpace;
    }
    document.body.setAttribute("data-threejs-renderer", String(rendererBackend));

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

    window.annolidZoomView = (factor = 1.0) => {
      const zoomFactor = Math.min(10.0, Math.max(0.1, Number(factor) || 1.0));
      const target = controls && controls.target ? controls.target : new THREE.Vector3(0, 0, 0);
      const offset = camera.position.clone().sub(target);
      const currentDistance = Math.max(0.05, offset.length());
      const nextDistance = Math.min(10000, Math.max(0.05, currentDistance / zoomFactor));
      offset.setLength(nextDistance);
      camera.position.copy(target).add(offset);
      camera.updateProjectionMatrix();
      if (controls) controls.update();
    };

    window.annolidResetView = () => {
      camera.position.set(0, 0, 3);
      if (controls) {
        controls.target.set(0, 0, 0);
        controls.update();
      }
      camera.updateProjectionMatrix();
    };

    let realtimeEnabled = true;

    // --- Toolbar Interaction ---
    const btnHome = document.getElementById("btnHome");
    const btnToggleRealtime = document.getElementById("btnToggleRealtime");
    const btnToggleMoveMode = document.getElementById("btnToggleMoveMode");
    const btnCenterObject = document.getElementById("btnCenterObject");
    const btnResetObjectMove = document.getElementById("btnResetObjectMove");
    let moveModeEnabled = false;
    let moveDragState = null;
    let rootBaselinePosition = null;

    const setMoveModeEnabled = (enabled) => {
      moveModeEnabled = Boolean(enabled);
      if (btnToggleMoveMode) {
        btnToggleMoveMode.classList.toggle("active", moveModeEnabled);
      }
    };

    const centerLoadedContent = ({ fit = false } = {}) => {
      if (!root) return;
      const box = new THREE.Box3().setFromObject(root);
      if (box.isEmpty()) return;
      const center = new THREE.Vector3();
      box.getCenter(center);
      root.position.sub(center);
      controls.target.sub(center);
      controls.update();
      if (fit) {
        fitCameraToObject(root, { centerControls: false });
      }
      if (rootBaselinePosition) {
        rootBaselinePosition.copy(root.position);
      } else {
        rootBaselinePosition = root.position.clone();
      }
    };

    const resetLoadedContentTranslation = () => {
      if (!root) return;
      const baseline = rootBaselinePosition || new THREE.Vector3(0, 0, 0);
      root.position.copy(baseline);
      controls.update();
      setStatus("Object translation reset.");
    };

    const getDepthForMove = () => {
      const box = new THREE.Box3().setFromObject(root || scene);
      const center = new THREE.Vector3();
      if (!box.isEmpty()) {
        box.getCenter(center);
      } else {
        center.copy(controls.target);
      }
      return Math.max(0.1, camera.position.distanceTo(center));
    };

    const startMoveDrag = (event) => {
      if (!root || event.button !== 0) return;
      const allowMove = moveModeEnabled || event.shiftKey;
      if (!allowMove) return;
      moveDragState = {
        startX: Number(event.clientX) || 0,
        startY: Number(event.clientY) || 0,
        startPos: root.position.clone(),
        depth: getDepthForMove(),
      };
      controls.enabled = false;
      canvas.style.cursor = "grabbing";
      event.preventDefault();
      event.stopPropagation();
    };

    const onMoveDrag = (event) => {
      if (!moveDragState || !root) return;
      const { w, h } = getCanvasSize();
      const dx = (Number(event.clientX) || 0) - moveDragState.startX;
      const dy = (Number(event.clientY) || 0) - moveDragState.startY;
      const fovRad = THREE.MathUtils.degToRad(Number(camera.fov) || 50);
      const worldPerPixelY = (2 * Math.tan(fovRad / 2) * moveDragState.depth) / Math.max(1, h);
      const worldPerPixelX = worldPerPixelY * (Math.max(1, w) / Math.max(1, h));
      const up = camera.up.clone().normalize();
      const forward = new THREE.Vector3();
      camera.getWorldDirection(forward);
      const right = forward.clone().cross(up).normalize();
      const deltaWorld = right.multiplyScalar(dx * worldPerPixelX)
        .add(up.clone().multiplyScalar(-dy * worldPerPixelY));
      root.position.copy(moveDragState.startPos).add(deltaWorld);
      controls.update();
      event.preventDefault();
    };

    const stopMoveDrag = () => {
      if (!moveDragState) return;
      moveDragState = null;
      controls.enabled = true;
      canvas.style.cursor = moveModeEnabled ? "grab" : "";
    };

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
    if (btnToggleMoveMode) {
      btnToggleMoveMode.onclick = () => {
        setMoveModeEnabled(!moveModeEnabled);
        canvas.style.cursor = moveModeEnabled ? "grab" : "";
        setStatus(moveModeEnabled ? "Move mode enabled. Drag to move object." : "Move mode disabled.");
      };
    }
    if (btnCenterObject) {
      btnCenterObject.onclick = () => {
        centerLoadedContent({ fit: false });
        setStatus("Centered loaded content.");
      };
    }
    if (btnResetObjectMove) {
      btnResetObjectMove.onclick = () => {
        resetLoadedContentTranslation();
      };
    }
    setMoveModeEnabled(false);

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
    if (btnToggleVolumePanel) {
      btnToggleVolumePanel.onclick = () => {
        if (!volumePanelEl || btnToggleVolumePanel.hidden) return;
        volumePanelEl.hidden = !volumePanelEl.hidden;
        btnToggleVolumePanel.style.color = volumePanelEl.hidden ? "#888" : "#fff";
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
    canvas.addEventListener("pointerdown", startMoveDrag, { passive: false });
    window.addEventListener("pointermove", onMoveDrag, { passive: false });
    window.addEventListener("pointerup", stopMoveDrag);
    window.addEventListener("pointercancel", stopMoveDrag);
    canvas.addEventListener("dblclick", () => {
      centerLoadedContent({ fit: false });
      setStatus("Centered loaded content.");
    });

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
    const isPanoramaImageExt = (value) => {
      const lower = String(value || "").toLowerCase();
      return ["jpg", "jpeg", "png", "webp", "bmp", "gif"].includes(lower);
    };

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
      hideVolumePanel();
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
      rootBaselinePosition = root.position.clone();
      controls.target.set(0, 0, 0);
      fitCameraToObject(root, { centerControls: false });

      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      const axes = new THREE.AxesHelper(maxDim * 0.35);
      axes.visible = false;
      root.add(axes);

      setStatus(`Loaded ${title} (${ext.toUpperCase()}).`);
      document.body.setAttribute("data-threejs-ready", "1");
    };

    const resolveRegionIdForObject = (obj) => {
      if (!obj) return "";
      const direct = String((obj.userData && obj.userData.regionId) || "").trim();
      if (direct) return direct;
      const candidateNames = [obj.name, obj.parent && obj.parent.name].filter(Boolean);
      for (const key of candidateNames) {
        const lookup = String(objectRegionMap[String(key)] || "").trim();
        if (lookup) return lookup;
      }
      return "";
    };

    if (pickMode === "brain3d_region") {
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      canvas.addEventListener("click", (event) => {
        if (!root || root.children.length === 0) return;
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, rect.width);
        const height = Math.max(1, rect.height);
        pointer.x = ((event.clientX - rect.left) / width) * 2 - 1;
        pointer.y = -(((event.clientY - rect.top) / height) * 2 - 1);
        raycaster.setFromCamera(pointer, camera);
        const intersections = raycaster.intersectObjects(root.children, true);
        if (!Array.isArray(intersections) || intersections.length === 0) return;
        const regionId = resolveRegionIdForObject(intersections[0].object);
        if (!regionId) return;
        const url = new URL("annolid://brain3d-select");
        url.searchParams.set("region_id", regionId);
        window.location.href = url.toString();
      });
    }

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
    const simulationVolumeRoot = new THREE.Group();
    const simulationSlices = new THREE.Group();
    const simulationPoints = new THREE.Group();
    const simulationEdges = new THREE.Group();
    const simulationTrails = new THREE.Group();
    const simulationLabels = new THREE.Group();
    simulationRoot.add(simulationEnvironmentRoot);
    simulationRoot.add(simulationModelRoot);
    simulationRoot.add(simulationBodyPartsRoot);
    simulationRoot.add(simulationVolumeRoot);
    simulationRoot.add(simulationSlices);
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
    let volumeRenderState = null;
    let volumeRenderDefaults = null;
    let volumeGridCache = null;
    let volumeTextureCache = null;
    let volumeLabelColorTableCache = null;
    let volumeLabelColorTextureCache = null;
    let volumeRenderFrameHandle = 0;
    let volumeSlicePlaybackTimer = null;
    let activeRaymarchMaterial = null;
    let frameTimeEmaMs = 16.6;
    let adaptiveRaymarchFactor = 1.0;
    let flybodyControlsMoved = false;
    let flybodyDragState = null;
    let volumePanelMoved = false;
    let volumePanelDragState = null;

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

    const stopVolumePanelDrag = () => {
      volumePanelDragState = null;
      if (volumePanelEl) {
        volumePanelEl.classList.remove("dragging");
      }
      window.removeEventListener("pointermove", onVolumePanelDragMove);
      window.removeEventListener("pointerup", stopVolumePanelDrag);
      window.removeEventListener("pointercancel", stopVolumePanelDrag);
    };

    const onVolumePanelDragMove = (event) => {
      if (!volumePanelDragState || !volumePanelEl) return;
      volumePanelMoved = true;
      const nextLeft = Math.max(8, event.clientX - volumePanelDragState.offsetX);
      const nextTop = Math.max(8, event.clientY - volumePanelDragState.offsetY);
      volumePanelEl.style.left = `${nextLeft}px`;
      volumePanelEl.style.top = `${nextTop}px`;
      volumePanelEl.style.right = "auto";
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
      const renderMode = String(
        (((simulationPayload || {}).metadata || {}).render_mode || "")
      ).trim();
      const qposLen = Array.isArray(frame.qpos) ? frame.qpos.length : 0;
      const dryRun = frame.dry_run ? "yes" : "no";
      const lines = [
        `Adapter: ${adapter}`,
        `Renderer Backend: ${String(rendererBackend || "webgl")}`,
        `Frame: ${frame.frame_index}`,
        `Points: ${Array.isArray(frame.points) ? frame.points.length : 0}`,
        `Qpos: ${qposLen}`,
        `Dry run: ${dryRun}`,
      ];
      if (renderMode) {
        lines.push(`Render: ${renderMode}`);
      }
      if (Number.isFinite(frame.timestamp_sec)) {
        lines.push(`Time: ${Number(frame.timestamp_sec).toFixed(3)} s`);
      }
      metaEl.textContent = lines.join("\n");
    };

    let _zarrSplatTexture = null;
    const getZarrSplatTexture = () => {
      if (_zarrSplatTexture) return _zarrSplatTexture;
      const texCanvas = document.createElement("canvas");
      texCanvas.width = 64;
      texCanvas.height = 64;
      const ctx = texCanvas.getContext("2d");
      const gradient = ctx.createRadialGradient(32, 32, 2, 32, 32, 30);
      gradient.addColorStop(0.0, "rgba(255,255,255,1)");
      gradient.addColorStop(0.4, "rgba(255,255,255,0.7)");
      gradient.addColorStop(1.0, "rgba(255,255,255,0)");
      ctx.clearRect(0, 0, 64, 64);
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 64, 64);
      _zarrSplatTexture = new THREE.CanvasTexture(texCanvas);
      _zarrSplatTexture.needsUpdate = true;
      return _zarrSplatTexture;
    };

    let _zarrSectionTexture = null;
    const getZarrSectionTexture = () => {
      if (_zarrSectionTexture) return _zarrSectionTexture;
      const texCanvas = document.createElement("canvas");
      texCanvas.width = 48;
      texCanvas.height = 48;
      const ctx = texCanvas.getContext("2d");
      ctx.clearRect(0, 0, 48, 48);
      const gradient = ctx.createLinearGradient(0, 0, 0, 48);
      gradient.addColorStop(0.0, "rgba(255,255,255,0.96)");
      gradient.addColorStop(0.55, "rgba(255,255,255,0.84)");
      gradient.addColorStop(1.0, "rgba(255,255,255,0.08)");
      ctx.fillStyle = gradient;
      ctx.fillRect(8, 8, 32, 32);
      ctx.fillStyle = "rgba(0,0,0,0.06)";
      for (let i = 0; i < 80; i += 1) {
        const x = ((i * 1103515245 + 12345) >>> 0) % 32;
        const y = ((i * 134775813 + 1) >>> 0) % 32;
        ctx.fillRect(8 + x, 8 + y, 1, 1);
      }
      _zarrSectionTexture = new THREE.CanvasTexture(texCanvas);
      _zarrSectionTexture.needsUpdate = true;
      return _zarrSectionTexture;
    };

    const decodeVolumeGrid = (metadata) => {
      const shape = Array.isArray(metadata && metadata.volume_grid_shape)
        ? metadata.volume_grid_shape.map((v) => Math.max(1, Number(v) || 1))
        : [];
      const encoded = String((metadata && metadata.volume_grid_base64) || "");
      if (shape.length !== 3 || !encoded) return null;
      const cacheKey = `${String((metadata && metadata.source_path) || "")}:${shape.join("x")}:${encoded.length}`;
      if (volumeGridCache && volumeGridCache.key === cacheKey) {
        return volumeGridCache.value;
      }
      try {
        const binary = window.atob(encoded);
        const data = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
          data[i] = binary.charCodeAt(i);
        }
        const value = { data, shape };
        volumeGridCache = { key: cacheKey, value };
        return value;
      } catch (_err) {
        return null;
      }
    };

    const getVolumeLabelIdLut = (metadata) => {
      const raw = metadata && Array.isArray(metadata.volume_label_id_lut)
        ? metadata.volume_label_id_lut
        : [];
      return raw
        .map((v) => Number(v))
        .filter((v) => Number.isFinite(v) && v > 0);
    };

    const getVolumeLabelColorSeed = (metadata, state) => {
      const stateSeed = Number(state && state.labelColorSeed);
      if (Number.isFinite(stateSeed)) {
        return Math.max(0, Math.floor(stateSeed));
      }
      const metadataSeed = Number(metadata && metadata.volume_label_color_seed);
      if (Number.isFinite(metadataSeed)) {
        return Math.max(0, Math.floor(metadataSeed));
      }
      return 1337;
    };

    const normalizeLabelColorEntry = (entry) => {
      if (entry == null) return null;
      if (typeof entry === "string") {
        if (entry.trim().toLowerCase() === "transparent") {
          return [0, 0, 0, 0];
        }
        try {
          const color = new THREE.Color();
          color.setStyle(entry);
          return [color.r, color.g, color.b, 1];
        } catch (_err) {
          return null;
        }
      }
      if (Array.isArray(entry)) {
        if (entry.length < 3) return null;
        const channels = entry.map((v) => Number(v));
        if (!channels.slice(0, 3).every((v) => Number.isFinite(v))) {
          return null;
        }
        const scale = channels.slice(0, 3).some((v) => v > 1.0) ? 255.0 : 1.0;
        const alphaRaw = Number.isFinite(channels[3]) ? channels[3] : 1.0;
        const alphaScale = alphaRaw > 1.0 ? 255.0 : 1.0;
        return [
          clamp01(channels[0] / scale),
          clamp01(channels[1] / scale),
          clamp01(channels[2] / scale),
          clamp01(alphaRaw / alphaScale),
        ];
      }
      if (typeof entry === "object") {
        const r = Number(entry.r);
        const g = Number(entry.g);
        const b = Number(entry.b);
        if (![r, g, b].every((v) => Number.isFinite(v))) {
          return null;
        }
        const scale = [r, g, b].some((v) => v > 1.0) ? 255.0 : 1.0;
        const alphaRaw = Number.isFinite(Number(entry.a)) ? Number(entry.a) : 1.0;
        const alphaScale = alphaRaw > 1.0 ? 255.0 : 1.0;
        return [
          clamp01(r / scale),
          clamp01(g / scale),
          clamp01(b / scale),
          clamp01(alphaRaw / alphaScale),
        ];
      }
      return null;
    };

    const getVolumeLabelColorOverrides = (metadata) => {
      const raw = metadata && (
        metadata.volume_label_colors ||
        metadata.volume_label_color_map ||
        metadata.label_color_map
      );
      if (!raw || typeof raw !== "object") {
        return { map: new Map(), signature: "none" };
      }
      const map = new Map();
      const signatureParts = [];
      Object.entries(raw).forEach(([key, value]) => {
        const labelId = Number(key);
        if (!Number.isFinite(labelId) || labelId < 0) return;
        const rgba = normalizeLabelColorEntry(value);
        if (!rgba) return;
        const canonicalLabelId = Math.floor(labelId);
        map.set(canonicalLabelId, rgba);
        signatureParts.push(
          `${canonicalLabelId}:${rgba.map((v) => Math.round(clamp01(v) * 255)).join(",")}`
        );
      });
      signatureParts.sort();
      return { map, signature: signatureParts.join("|") || "none" };
    };

    const getVolumeTexture = (THREE, metadata) => {
      const volumeGrid = decodeVolumeGrid(metadata);
      if (!volumeGrid) return null;
      const isLabelVolume = Boolean(metadata && metadata.label_volume);
      const cacheKey = `${String((metadata && metadata.source_path) || "")}:${volumeGrid.shape.join("x")}:${String((metadata && metadata.volume_grid_base64) || "").length}:${isLabelVolume ? "label" : "intensity"}`;
      if (volumeTextureCache && volumeTextureCache.key === cacheKey) {
        return volumeTextureCache.texture;
      }
      const [zCount, yCount, xCount] = volumeGrid.shape;
      const texture = new THREE.Data3DTexture(volumeGrid.data, xCount, yCount, zCount);
      texture.format = THREE.RedFormat;
      texture.type = THREE.UnsignedByteType;
      texture.minFilter = isLabelVolume ? THREE.NearestFilter : THREE.LinearFilter;
      texture.magFilter = isLabelVolume ? THREE.NearestFilter : THREE.LinearFilter;
      texture.unpackAlignment = 1;
      texture.needsUpdate = true;
      volumeTextureCache = { key: cacheKey, texture };
      return texture;
    };

    const clamp01 = (value) => Math.max(0, Math.min(1, Number(value) || 0));
    const lerp = (a, b, t) => Number(a) + (Number(b) - Number(a)) * clamp01(t);

    const hash01 = (value) => {
      const seed = (Math.imul(Number(value) | 0, 1664525) + 1013904223) >>> 0;
      return seed / 4294967295;
    };

    const hashLabel01 = (value, salt = 0, seed = 0) => {
      const mixed = Math.imul((Number(value) | 0) ^ (salt | 0), 1103515245) + 12345 + ((Number(seed) | 0) * 2654435761);
      return ((mixed >>> 0) % 1000003) / 1000003;
    };

    const sampleLabelIdColor = (labelId, saturationBoost = 1, seed = 0) => {
      const id = Math.max(1, Math.floor(Number(labelId) || 0));
      const h = hashLabel01(id, 97, seed);
      const s = 0.55 + hashLabel01(id, 193, seed) * 0.35;
      const l = 0.42 + hashLabel01(id, 389, seed) * 0.26;
      const color = new THREE.Color();
      color.setHSL(h, clamp01(s * Math.max(0.2, Number(saturationBoost) || 1)), clamp01(l));
      return color;
    };

    const getVolumeLabelColorTable = (metadata, state) => {
      const labelLut = getVolumeLabelIdLut(metadata);
      const saturation = Math.max(0.0, Number(state && state.saturation) || 1.0);
      const seed = getVolumeLabelColorSeed(metadata, state);
      const overrides = getVolumeLabelColorOverrides(metadata);
      const cacheKey = [
        String((metadata && metadata.source_path) || ""),
        String((metadata && metadata.label_volume) ? "label" : "intensity"),
        labelLut.join(","),
        `${seed}`,
        `${Math.round(saturation * 1000)}`,
        overrides.signature,
      ].join("|");
      if (volumeLabelColorTableCache && volumeLabelColorTableCache.key === cacheKey) {
        return volumeLabelColorTableCache.value;
      }
      const rgba8 = new Uint8Array(256 * 4);
      const rgbaFloat = new Float32Array(256 * 4);
      const byId = new Map();
      for (let labelIndex = 0; labelIndex <= 255; labelIndex += 1) {
        let rgba = [0, 0, 0, 0];
        if (labelIndex > 0) {
          const labelId = labelLut[labelIndex - 1] || labelIndex;
          const override = overrides.map.get(labelId);
          if (override) {
            rgba = [override[0], override[1], override[2], override[3]];
          } else {
            const color = sampleLabelIdColor(labelId, saturation, seed);
            rgba = [color.r, color.g, color.b, 1.0];
          }
          byId.set(labelId, rgba);
        }
        const base = labelIndex * 4;
        rgbaFloat[base + 0] = clamp01(rgba[0]);
        rgbaFloat[base + 1] = clamp01(rgba[1]);
        rgbaFloat[base + 2] = clamp01(rgba[2]);
        rgbaFloat[base + 3] = clamp01(rgba[3]);
        rgba8[base + 0] = Math.round(rgbaFloat[base + 0] * 255);
        rgba8[base + 1] = Math.round(rgbaFloat[base + 1] * 255);
        rgba8[base + 2] = Math.round(rgbaFloat[base + 2] * 255);
        rgba8[base + 3] = Math.round(rgbaFloat[base + 3] * 255);
      }
      const value = {
        rgba8,
        rgbaFloat,
        byId,
        seed,
        key: cacheKey,
      };
      volumeLabelColorTableCache = { key: cacheKey, value };
      return value;
    };

    const getVolumeLabelColorTexture = (THREE, metadata, state) => {
      const table = getVolumeLabelColorTable(metadata, state);
      const textureKey = table.key;
      if (volumeLabelColorTextureCache && volumeLabelColorTextureCache.key === textureKey) {
        return volumeLabelColorTextureCache.texture;
      }
      const texture = new THREE.DataTexture(table.rgba8, 256, 1, THREE.RGBAFormat);
      texture.type = THREE.UnsignedByteType;
      texture.minFilter = THREE.NearestFilter;
      texture.magFilter = THREE.NearestFilter;
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.unpackAlignment = 1;
      texture.needsUpdate = true;
      volumeLabelColorTextureCache = { key: textureKey, texture };
      return texture;
    };

    const getLabelColorRgba = (labelId, metadata, state) => {
      const id = Math.max(0, Math.floor(Number(labelId) || 0));
      if (id <= 0) return null;
      const table = getVolumeLabelColorTable(metadata, state);
      const override = table.byId.get(id);
      if (override) return override;
      const color = sampleLabelIdColor(id, Math.max(0.0, Number(state && state.saturation) || 1.0), table.seed);
      return [color.r, color.g, color.b, 1.0];
    };

    const getVolumeStateStorageKey = (metadata) => {
      const sourcePath = String((metadata && metadata.source_path) || "").trim();
      if (!sourcePath) return "";
      return `annolid-three-volume-look:${sourcePath}`;
    };

    const loadPersistedVolumeState = (metadata) => {
      const key = getVolumeStateStorageKey(metadata);
      if (!key) return null;
      try {
        const raw = window.localStorage ? window.localStorage.getItem(key) : null;
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        return parsed && typeof parsed === "object" ? parsed : null;
      } catch (_err) {
        return null;
      }
    };

    const persistVolumeState = (metadata, state) => {
      const key = getVolumeStateStorageKey(metadata);
      if (!key || !state) return;
      try {
        if (window.localStorage) {
          window.localStorage.setItem(key, JSON.stringify(state));
        }
      } catch (_err) {
      }
    };

    const clearPersistedVolumeState = (metadata) => {
      const key = getVolumeStateStorageKey(metadata);
      if (!key) return;
      try {
        if (window.localStorage) {
          window.localStorage.removeItem(key);
        }
      } catch (_err) {
      }
    };

    const ensureRaymarchControlState = (rawState) =>
      Object.assign(
        {
          raymarchSteps: 220,
          raymarchStepScale: 1.0,
          raymarchJitter: 0.45,
          raymarchGradientOpacity: false,
          raymarchGradientFactor: 2.8,
          raymarchShading: true,
          raymarchAmbient: 0.34,
          raymarchDiffuse: 0.86,
          raymarchSpecular: 0.22,
          raymarchSpecularPower: 24.0,
          raymarchLightX: 0.38,
          raymarchLightY: 0.52,
          raymarchLightZ: 0.76,
        },
        rawState || {}
      );

    const getVolumeRenderDefaults = (metadata) => {
      const raw = (metadata && metadata.volume_render_defaults) || {};
      return ensureRaymarchControlState({
        preset: String(raw.preset || "cinematic"),
        intensity: Number.isFinite(Number(raw.intensity)) ? Number(raw.intensity) : 1.1,
        contrast: Number.isFinite(Number(raw.contrast)) ? Number(raw.contrast) : 1.28,
        gamma: Number.isFinite(Number(raw.gamma)) ? Number(raw.gamma) : 0.9,
        opacity: Number.isFinite(Number(raw.opacity)) ? Number(raw.opacity) : 0.42,
        size: Number.isFinite(Number(raw.size)) ? Number(raw.size) : 0.03,
        threshold: Number.isFinite(Number(raw.threshold)) ? Number(raw.threshold) : 0.16,
        density: Number.isFinite(Number(raw.density)) ? Number(raw.density) : 0.9,
        saturation: Number.isFinite(Number(raw.saturation)) ? Number(raw.saturation) : 1.08,
        tfLow: Number.isFinite(Number(raw.tf_low)) ? Number(raw.tf_low) : 0.06,
        tfMid: Number.isFinite(Number(raw.tf_mid)) ? Number(raw.tf_mid) : 0.48,
        tfHigh: Number.isFinite(Number(raw.tf_high)) ? Number(raw.tf_high) : 0.96,
        clipAxis: String(raw.clip_axis || "none"),
        clipCenter: Number.isFinite(Number(raw.clip_center)) ? Number(raw.clip_center) : 0.5,
        clipThickness: Number.isFinite(Number(raw.clip_thickness)) ? Number(raw.clip_thickness) : 1.0,
        clipInvert: Boolean(raw.clip_invert),
        palette: String(raw.palette || (metadata && metadata.label_volume ? "allen_labels" : "ice_fire")),
        blendMode: String(raw.blend_mode || "additive"),
        pointTexture: String(raw.point_texture || "glow"),
        backgroundTheme: String(raw.background_theme || "dark"),
        labelColorSeed: Number.isFinite(Number(raw.label_color_seed))
          ? Number(raw.label_color_seed)
          : Number.isFinite(Number(metadata && metadata.volume_label_color_seed))
            ? Number(metadata.volume_label_color_seed)
            : 1337,
        sliceModeEnabled: Boolean(raw.slice_mode_enabled || raw.slice_mode),
        sliceAxis: String(
          raw.slice_axis ||
          raw.clip_axis ||
          ((metadata && metadata.section_axis) || "z")
        ),
        slicePosition: Number.isFinite(Number(raw.slice_position))
          ? Number(raw.slice_position)
          : Number.isFinite(Number(raw.clip_center))
            ? Number(raw.clip_center)
            : 0.5,
        sliceStep: Number.isFinite(Number(raw.slice_step))
          ? Math.max(1, Math.round(Number(raw.slice_step)))
          : 1,
        slicePlaybackFps: Number.isFinite(Number(raw.slice_playback_fps))
          ? Math.max(1, Math.round(Number(raw.slice_playback_fps)))
          : 8,
        sliceAutoplay: Boolean(raw.slice_autoplay),
        collapsedCards: raw.collapsed_cards && typeof raw.collapsed_cards === "object"
          ? Object.assign({}, raw.collapsed_cards)
          : {},
        renderStyle: String(raw.render_style || "hybrid"),
        sectionEmphasis: String(raw.section_emphasis || "auto"),
        raymarchSteps: Number.isFinite(Number(raw.raymarch_steps))
          ? Number(raw.raymarch_steps)
          : 220,
        raymarchStepScale: Number.isFinite(Number(raw.raymarch_step_scale))
          ? Number(raw.raymarch_step_scale)
          : 1.0,
        raymarchJitter: Number.isFinite(Number(raw.raymarch_jitter))
          ? Number(raw.raymarch_jitter)
          : 0.45,
        raymarchGradientOpacity: Boolean(raw.gradient_opacity),
        raymarchGradientFactor: Number.isFinite(Number(raw.gradient_opacity_factor))
          ? Number(raw.gradient_opacity_factor)
          : 2.8,
        raymarchShading: raw.use_shading !== false,
        raymarchAmbient: Number.isFinite(Number(raw.ambient_strength))
          ? Number(raw.ambient_strength)
          : 0.34,
        raymarchDiffuse: Number.isFinite(Number(raw.diffuse_strength))
          ? Number(raw.diffuse_strength)
          : 0.86,
        raymarchSpecular: Number.isFinite(Number(raw.specular_strength))
          ? Number(raw.specular_strength)
          : 0.22,
        raymarchSpecularPower: Number.isFinite(Number(raw.specular_power))
          ? Number(raw.specular_power)
          : 24.0,
        raymarchLightX: Array.isArray(raw.light_direction) && Number.isFinite(Number(raw.light_direction[0]))
          ? Number(raw.light_direction[0])
          : 0.38,
        raymarchLightY: Array.isArray(raw.light_direction) && Number.isFinite(Number(raw.light_direction[1]))
          ? Number(raw.light_direction[1])
          : 0.52,
        raymarchLightZ: Array.isArray(raw.light_direction) && Number.isFinite(Number(raw.light_direction[2]))
          ? Number(raw.light_direction[2])
          : 0.76,
      });
    };

    const getVolumePreset = (presetName) => {
      const presets = {
        cinematic: {
          intensity: 1.1,
          contrast: 1.28,
          gamma: 0.9,
          opacity: 0.42,
          size: 0.03,
          threshold: 0.16,
          density: 0.9,
          saturation: 1.08,
          tfLow: 0.06,
          tfMid: 0.48,
          tfHigh: 0.96,
          palette: "ice_fire",
          blendMode: "additive",
          pointTexture: "glow",
          backgroundTheme: "dark",
          renderStyle: "points",
          sectionEmphasis: "auto",
        },
        histology_defaults: {
          intensity: 1.18,
          contrast: 1.52,
          gamma: 0.84,
          opacity: 0.74,
          size: 0.05,
          threshold: 0.02,
          density: 1.0,
          saturation: 0.84,
          tfLow: 0.02,
          tfMid: 0.30,
          tfHigh: 0.86,
          palette: "section_ink",
          blendMode: "normal",
          pointTexture: "section",
          backgroundTheme: "light",
          renderStyle: "raymarch",
          sectionEmphasis: "auto",
        },
        section_stack: {
          intensity: 1.22,
          contrast: 1.56,
          gamma: 0.84,
          opacity: 0.68,
          size: 0.05,
          threshold: 0.03,
          density: 1.0,
          saturation: 0.78,
          tfLow: 0.02,
          tfMid: 0.32,
          tfHigh: 0.84,
          palette: "section_ink",
          blendMode: "normal",
          pointTexture: "section",
          backgroundTheme: "light",
          renderStyle: "slab",
          sectionEmphasis: "auto",
        },
        nissl_sections: {
          intensity: 1.26,
          contrast: 1.62,
          gamma: 0.82,
          opacity: 0.66,
          size: 0.048,
          threshold: 0.03,
          density: 1.0,
          saturation: 0.96,
          tfLow: 0.02,
          tfMid: 0.34,
          tfHigh: 0.84,
          palette: "nissl",
          blendMode: "normal",
          pointTexture: "section",
          backgroundTheme: "light",
          renderStyle: "slab",
          sectionEmphasis: "nissl",
        },
        myelin_sections: {
          intensity: 1.14,
          contrast: 1.78,
          gamma: 0.88,
          opacity: 0.72,
          size: 0.05,
          threshold: 0.04,
          density: 1.0,
          saturation: 0.22,
          tfLow: 0.03,
          tfMid: 0.30,
          tfHigh: 0.82,
          palette: "myelin",
          blendMode: "normal",
          pointTexture: "section",
          backgroundTheme: "light",
          renderStyle: "slab",
          sectionEmphasis: "myelin",
        },
        xray: {
          intensity: 1.35,
          contrast: 1.42,
          gamma: 0.72,
          opacity: 0.3,
          size: 0.026,
          threshold: 0.24,
          density: 0.72,
          saturation: 0.35,
          tfLow: 0.14,
          tfMid: 0.52,
          tfHigh: 0.98,
          palette: "grayscale",
          blendMode: "normal",
          pointTexture: "section",
          backgroundTheme: "light",
          renderStyle: "slab",
          sectionEmphasis: "auto",
        },
        neon: {
          intensity: 1.28,
          contrast: 1.5,
          gamma: 0.82,
          opacity: 0.52,
          size: 0.034,
          threshold: 0.14,
          density: 1.0,
          saturation: 1.28,
          tfLow: 0.04,
          tfMid: 0.42,
          tfHigh: 0.88,
          palette: "aurora",
          blendMode: "additive",
          pointTexture: "glow",
          backgroundTheme: "dark",
          renderStyle: "points",
          sectionEmphasis: "auto",
        },
      };
      return ensureRaymarchControlState(
        presets[String(presetName || "").toLowerCase()] || presets.cinematic
      );
    };

    const resolveAutoSectionEmphasis = (metadata, state) => {
      if (!metadata || !metadata.interleaved_detected) return "neutral";
      const sectionAxis = String(metadata.section_axis || "z");
      const bounds = metadata.volume_bounds || {};
      const axisBounds = Array.isArray(bounds[sectionAxis]) ? bounds[sectionAxis] : [0, 1];
      const min = Number(axisBounds[0] || 0);
      const max = Number(axisBounds[1] || 1);
      const centerWorld = lerp(min, max, state.clipAxis === "none" ? 0.5 : state.clipCenter);
      const stepWorld = Math.max(1e-5, Number(metadata.section_step_world) || 1);
      const sliceIndex = Math.max(0, Math.round(Math.abs(centerWorld - min) / stepWorld));
      return sliceIndex % 2 === 0 ? "nissl" : "myelin";
    };

    const buildEffectiveVolumeState = (state, metadata) => {
      const next = Object.assign({}, state || {});
      next.sliceAxis = normalizeVolumeAxis(next.sliceAxis || (metadata && metadata.section_axis) || "z", "z");
      next.slicePosition = clamp01(
        Number.isFinite(Number(next.slicePosition)) ? Number(next.slicePosition) : 0.5
      );
      next.sliceStep = Math.max(1, Math.round(Number(next.sliceStep) || 1));
      next.slicePlaybackFps = Math.max(1, Math.round(Number(next.slicePlaybackFps) || 8));
      if (metadata && metadata.label_volume) {
        next.palette = String(next.palette || "allen_labels");
        next.sectionEmphasis = "neutral";
        next.labelColorSeed = getVolumeLabelColorSeed(metadata, next);
        next.renderStyle = getMetadataResolvedRenderStyle(
          { renderStyle: String(next.renderStyle || "slab") },
          metadata
        );
        next.blendMode = "normal";
        next.pointTexture = "section";
        next.backgroundTheme = String(next.backgroundTheme || "dark");
      }
      if (next.sliceModeEnabled) {
        const axis = normalizeVolumeAxis(next.sliceAxis || (metadata && metadata.section_axis) || "z", "z");
        next.sliceAxis = axis;
        next.clipAxis = axis;
        next.clipCenter = clamp01(next.slicePosition);
        next.clipThickness = getSingleSliceThicknessNorm(metadata, axis);
        if (next.renderStyle === "points") {
          next.renderStyle = "slab";
        }
      }
      const emphasis = String(next.sectionEmphasis || "auto").toLowerCase();
      const resolved = emphasis === "auto"
        ? resolveAutoSectionEmphasis(metadata, next)
        : emphasis;
      next.resolvedSectionEmphasis = resolved;
      if (resolved === "nissl") {
        next.palette = "nissl";
        next.backgroundTheme = "light";
        next.pointTexture = "section";
        next.blendMode = "normal";
        next.renderStyle = next.renderStyle === "points" ? "slab" : next.renderStyle;
      } else if (resolved === "myelin") {
        next.palette = "myelin";
        next.backgroundTheme = "light";
        next.pointTexture = "section";
        next.blendMode = "normal";
        next.renderStyle = next.renderStyle === "points" ? "slab" : next.renderStyle;
      }
      return next;
    };

    const applyVolumeCurve = (value, state) => {
      let v = clamp01(value);
      const tfLow = clamp01(state.tfLow);
      const tfHigh = Math.max(tfLow + 0.01, clamp01(state.tfHigh));
      if (v <= tfLow) return 0;
      if (v >= tfHigh) {
        v = 1;
      } else {
        v = (v - tfLow) / Math.max(1e-5, tfHigh - tfLow);
      }
      const tfMid = clamp01(state.tfMid);
      const midpointGamma = tfMid <= 0
        ? 0.25
        : Math.max(0.2, Math.log(Math.max(1e-5, tfMid)) / Math.log(0.5));
      v = Math.pow(clamp01(v), midpointGamma);
      const threshold = clamp01(state.threshold);
      if (v < threshold) {
        return 0;
      }
      const denom = Math.max(1e-5, 1 - threshold);
      v = (v - threshold) / denom;
      v = Math.pow(clamp01(v), Math.max(0.15, Number(state.gamma) || 1));
      v = (v - 0.5) * Math.max(0.2, Number(state.contrast) || 1) + 0.5;
      v = clamp01(v * Math.max(0.05, Number(state.intensity) || 1));
      return v;
    };

    const sampleVolumePalette = (value, paletteName, saturationBoost = 1, labelSeed = 0) => {
      const v = clamp01(value);
      const palette = String(paletteName || "ice_fire").toLowerCase();
      const color = new THREE.Color();
      if (palette === "allen_labels") {
        return sampleLabelIdColor(value, saturationBoost, labelSeed);
      } else if (palette === "grayscale") {
        color.setRGB(v, v, v);
      } else if (palette === "magma") {
        color.setRGB(
          clamp01(Math.pow(v, 0.7) * 1.15),
          clamp01(Math.pow(v, 1.15) * 0.62),
          clamp01(Math.pow(v, 2.2) * 0.32 + 0.06)
        );
      } else if (palette === "viridis") {
        color.setRGB(
          clamp01(0.18 + v * 0.7),
          clamp01(0.08 + Math.sin(v * Math.PI) * 0.82),
          clamp01(0.36 + (1 - v) * 0.48)
        );
      } else if (palette === "aurora") {
        color.setRGB(
          clamp01(0.22 + Math.pow(v, 0.6) * 0.42),
          clamp01(0.25 + Math.sin(v * Math.PI * 0.95) * 0.72),
          clamp01(0.45 + Math.pow(1 - v, 0.8) * 0.42)
        );
      } else if (palette === "nissl") {
        color.setRGB(
          clamp01(0.72 - Math.pow(v, 0.85) * 0.36),
          clamp01(0.67 - Math.pow(v, 0.9) * 0.42),
          clamp01(0.86 - Math.pow(v, 0.72) * 0.56)
        );
      } else if (palette === "myelin") {
        color.setRGB(
          clamp01(0.82 - Math.pow(v, 0.78) * 0.54),
          clamp01(0.81 - Math.pow(v, 0.82) * 0.56),
          clamp01(0.79 - Math.pow(v, 0.86) * 0.6)
        );
      } else if (palette === "section_ink") {
        color.setRGB(
          clamp01(0.86 - Math.pow(v, 0.78) * 0.44),
          clamp01(0.8 - Math.pow(v, 0.9) * 0.52),
          clamp01(0.72 - Math.pow(v, 0.95) * 0.56)
        );
      } else {
        color.setRGB(
          clamp01(0.18 + Math.pow(v, 0.7) * 0.82),
          clamp01(0.32 + Math.sin(v * Math.PI) * 0.42),
          clamp01(0.92 - Math.pow(v, 0.82) * 0.72)
        );
      }
      const hsl = {};
      color.getHSL(hsl);
      color.setHSL(hsl.h, clamp01(hsl.s * Math.max(0, Number(saturationBoost) || 1)), hsl.l);
      return color;
    };

    const applyVolumeSceneStyle = (state) => {
      const theme = String((state && state.backgroundTheme) || "dark").toLowerCase();
      const isLight = theme === "light";
      document.body.classList.toggle("white-theme", isLight);
      scene.background = new THREE.Color(isLight ? 0xf3eee5 : 0x121824);
    };

    const getMetadataResolvedRenderStyle = (state, metadata) => {
      const style = String((state && state.renderStyle) || "points").toLowerCase();
      if (["points", "slab", "raymarch", "hybrid"].includes(style)) {
        return style;
      }
      return "points";
    };

    const normalizeVolumeAxis = (axis, fallback = "z") => {
      const value = String(axis || fallback).toLowerCase();
      return ["x", "y", "z"].includes(value) ? value : String(fallback || "z").toLowerCase();
    };

    const getVolumeSliceCount = (metadata, axis) => {
      const shape = Array.isArray(metadata && metadata.volume_grid_shape)
        ? metadata.volume_grid_shape
        : [];
      if (shape.length !== 3) return 1;
      const ax = normalizeVolumeAxis(axis, (metadata && metadata.section_axis) || "z");
      const idx = ax === "z" ? 0 : (ax === "y" ? 1 : 2);
      const count = Number(shape[idx] || 1);
      return Math.max(1, Math.round(count));
    };

    const getSingleSliceThicknessNorm = (metadata, axis) => {
      const count = getVolumeSliceCount(metadata, axis);
      if (count <= 1) return 1.0;
      const oneSlice = 1.0 / Math.max(1, count - 1);
      return Math.max(0.002, Math.min(0.2, oneSlice));
    };

    const getSliceIndexFromPosition = (position, metadata, axis) => {
      const count = getVolumeSliceCount(metadata, axis);
      if (count <= 1) return 0;
      return Math.max(0, Math.min(count - 1, Math.round(clamp01(position) * (count - 1))));
    };

    const getSlicePositionFromIndex = (index, metadata, axis) => {
      const count = getVolumeSliceCount(metadata, axis);
      if (count <= 1) return 0.0;
      const clamped = Math.max(0, Math.min(count - 1, Math.round(Number(index) || 0)));
      return clamped / Math.max(1, count - 1);
    };

    const formatSliceIndexLabel = (position, metadata, axis) => {
      const count = getVolumeSliceCount(metadata, axis);
      const index = getSliceIndexFromPosition(position, metadata, axis);
      return `${index + 1}/${count}`;
    };

    const getActiveVolumeSlabConfig = (metadata, state) => {
      const axis = String(
        (state && state.clipAxis && state.clipAxis !== "none")
          ? state.clipAxis
          : ((metadata && metadata.section_axis) || "z")
      );
      const bounds = (metadata && metadata.volume_bounds) || {};
      const axisBounds = Array.isArray(bounds[axis]) ? bounds[axis] : [0, 1];
      const min = Number(axisBounds[0] || 0);
      const max = Number(axisBounds[1] || 1);
      const centerNorm = (state && state.clipAxis && state.clipAxis !== "none")
        ? clamp01(state.clipCenter)
        : 0.5;
      const thicknessNorm = (state && state.clipAxis && state.clipAxis !== "none")
        ? clamp01(state.clipThickness)
        : 0.06;
      return {
        axis,
        min,
        max,
        centerWorld: lerp(min, max, centerNorm),
        halfThickness: Math.max(1e-5, Math.abs(max - min) * Math.max(0.01, thicknessNorm) * 0.5),
      };
    };

    const isVolumePointVisible = (pt, state, metadata) => {
      const axis = String(state.clipAxis || "none");
      if (axis === "none") return true;
      const bounds = (metadata && metadata.volume_bounds) || {};
      const axisBounds = Array.isArray(bounds[axis]) ? bounds[axis] : null;
      if (!axisBounds || axisBounds.length < 2) return true;
      const min = Number(axisBounds[0]);
      const max = Number(axisBounds[1]);
      if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < 1e-6) {
        return true;
      }
      const centerWorld = lerp(min, max, state.clipCenter);
      const halfThickness = Math.max(1e-5, Math.abs(max - min) * clamp01(state.clipThickness) * 0.5);
      const coord = Number(pt && pt[axis]);
      if (!Number.isFinite(coord)) return true;
      const inside = Math.abs(coord - centerWorld) <= halfThickness;
      return state.clipInvert ? !inside : inside;
    };

    const fitCameraToVolumeSlab = (metadata, state) => {
      const slab = getActiveVolumeSlabConfig(metadata, state);
      const bounds = (metadata && metadata.volume_bounds) || {};
      const xBounds = Array.isArray(bounds.x) ? bounds.x : [0, 1];
      const yBounds = Array.isArray(bounds.y) ? bounds.y : [0, 1];
      const zBounds = Array.isArray(bounds.z) ? bounds.z : [0, 1];
      const center = new THREE.Vector3(
        lerp(xBounds[0], xBounds[1], 0.5),
        lerp(yBounds[0], yBounds[1], 0.5),
        lerp(zBounds[0], zBounds[1], 0.5)
      );
      center[slab.axis] = slab.centerWorld;
      const width = Math.abs(Number(xBounds[1]) - Number(xBounds[0]));
      const height = Math.abs(Number(yBounds[1]) - Number(yBounds[0]));
      const depth = Math.abs(Number(zBounds[1]) - Number(zBounds[0]));
      const distance = Math.max(width, height, depth, 1) * 0.95;
      if (slab.axis === "x") {
        camera.position.set(center.x + distance, center.y, center.z);
      } else if (slab.axis === "y") {
        camera.position.set(center.x, center.y + distance, center.z);
      } else {
        camera.position.set(center.x, center.y, center.z + distance);
      }
      controls.target.copy(center);
      camera.lookAt(center);
      controls.update();
    };

    const renderVolumeSlabPlane = (orderedPoints, metadata, state) => {
      const effectiveState = buildEffectiveVolumeState(state, metadata);
      const isLabelVolume = Boolean(metadata && metadata.label_volume);
      const labelLut = isLabelVolume ? getVolumeLabelIdLut(metadata) : [];
      const labelColorTable = isLabelVolume ? getVolumeLabelColorTable(metadata, effectiveState) : null;
      const slab = getActiveVolumeSlabConfig(metadata, effectiveState);
      const bounds = (metadata && metadata.volume_bounds) || {};
      const axisToPlane = {
        x: ["z", "y"],
        y: ["x", "z"],
        z: ["x", "y"],
      };
      const [uAxis, vAxis] = axisToPlane[slab.axis] || ["x", "y"];
      const uBounds = Array.isArray(bounds[uAxis]) ? bounds[uAxis] : [0, 1];
      const vBounds = Array.isArray(bounds[vAxis]) ? bounds[vAxis] : [0, 1];
      const volumeGrid = decodeVolumeGrid(metadata);
      if (!volumeGrid) {
        return;
      }
      const [zCount, yCount, xCount] = volumeGrid.shape;
      const axisIndexMap = { x: 2, y: 1, z: 0 };
      const sliceCount = [zCount, yCount, xCount][axisIndexMap[slab.axis]] || 1;
      const axisBounds = Array.isArray(bounds[slab.axis]) ? bounds[slab.axis] : [0, 1];
      const axisMin = Number(axisBounds[0] || 0);
      const axisMax = Number(axisBounds[1] || 1);
      const sliceNormCenter = clamp01((slab.centerWorld - axisMin) / Math.max(1e-5, axisMax - axisMin));
      const startIndex = Math.max(
        0,
        Math.floor((sliceNormCenter - clamp01(effectiveState.clipThickness) * 0.5) * (sliceCount - 1))
      );
      const endIndex = Math.min(
        sliceCount - 1,
        Math.ceil((sliceNormCenter + clamp01(effectiveState.clipThickness) * 0.5) * (sliceCount - 1))
      );
      const maxPlanes = effectiveState.renderStyle === "hybrid" ? 12 : 28;
      const step = Math.max(1, Math.ceil((endIndex - startIndex + 1) / maxPlanes));
      const makeSliceCanvas = (sliceIndex, emphasis) => {
        const width = slab.axis === "x" ? zCount : xCount;
        const height = slab.axis === "z" ? yCount : (slab.axis === "y" ? zCount : yCount);
        const canvasEl = document.createElement("canvas");
        canvasEl.width = width;
        canvasEl.height = height;
        const ctx = canvasEl.getContext("2d");
        if (!ctx) return null;
        const image = ctx.createImageData(width, height);
        let maxStrength = 0;
        const strengths = new Float32Array(width * height);
        const rgb = new Float32Array(width * height * 3);
        const palette = emphasis === "nissl"
          ? "nissl"
          : emphasis === "myelin"
            ? "myelin"
            : effectiveState.palette;
        for (let row = 0; row < height; row += 1) {
          for (let col = 0; col < width; col += 1) {
            let raw = 0;
            if (slab.axis === "z") {
              raw = volumeGrid.data[sliceIndex * (yCount * xCount) + row * xCount + col];
            } else if (slab.axis === "y") {
              raw = volumeGrid.data[row * (yCount * xCount) + sliceIndex * xCount + col];
            } else {
              raw = volumeGrid.data[row * (yCount * xCount) + col * xCount + sliceIndex];
            }
            const idx = row * width + col;
            let curved = 0;
            let color = null;
            if (isLabelVolume) {
              const labelIndex = Math.round(Number(raw) || 0);
              const labelId = labelIndex > 0
                ? (labelLut[labelIndex - 1] || labelIndex)
                : 0;
              if (labelId > 0) {
                const base = labelIndex * 4;
                const alpha = labelColorTable ? labelColorTable.rgbaFloat[base + 3] : 0;
                if (alpha > 0) {
                  curved = alpha;
                  color = new THREE.Color(
                    labelColorTable.rgbaFloat[base + 0],
                    labelColorTable.rgbaFloat[base + 1],
                    labelColorTable.rgbaFloat[base + 2]
                  );
                }
              }
            } else {
              const value = raw / 255;
              curved = applyVolumeCurve(value, effectiveState);
              if (curved > 0) {
                color = sampleVolumePalette(curved, palette, effectiveState.saturation);
              }
            }
            strengths[idx] = curved;
            if (curved > maxStrength) maxStrength = curved;
            if (!color) continue;
            rgb[idx * 3 + 0] = color.r;
            rgb[idx * 3 + 1] = color.g;
            rgb[idx * 3 + 2] = color.b;
          }
        }
        if (maxStrength <= 0) return null;
        for (let idx = 0; idx < strengths.length; idx += 1) {
          const base = idx * 4;
          const strength = isLabelVolume
            ? strengths[idx]
            : strengths[idx] / maxStrength;
          image.data[base + 0] = Math.round(clamp01(rgb[idx * 3 + 0]) * 255);
          image.data[base + 1] = Math.round(clamp01(rgb[idx * 3 + 1]) * 255);
          image.data[base + 2] = Math.round(clamp01(rgb[idx * 3 + 2]) * 255);
          image.data[base + 3] = Math.round(
            clamp01((isLabelVolume ? strength : Math.pow(strength, 0.8)) * (0.2 + 0.8 * clamp01(effectiveState.opacity))) * 255
          );
        }
        ctx.putImageData(image, 0, 0);
        return canvasEl;
      };

      const planeWidth = Math.abs(Number(uBounds[1]) - Number(uBounds[0]));
      const planeHeight = Math.abs(Number(vBounds[1]) - Number(vBounds[0]));
      let renderedPlanes = 0;
      for (let sliceIndex = startIndex; sliceIndex <= endIndex; sliceIndex += step) {
        const emphasis = effectiveState.sectionEmphasis === "auto"
          ? ((sliceIndex % 2 === 0) ? "nissl" : "myelin")
          : effectiveState.resolvedSectionEmphasis;
        const canvasEl = makeSliceCanvas(sliceIndex, emphasis);
        if (!canvasEl) continue;
        const texture = new THREE.CanvasTexture(canvasEl);
        texture.needsUpdate = true;
        const plane = new THREE.Mesh(
          new THREE.PlaneGeometry(Math.max(planeWidth, 1), Math.max(planeHeight, 1)),
          new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            opacity: Math.max(0.08, effectiveState.opacity / Math.max(1, Math.ceil((endIndex - startIndex + 1) / step)) * 2.2),
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: String(effectiveState.blendMode || "normal") === "additive"
              ? THREE.AdditiveBlending
              : THREE.NormalBlending,
          })
        );
        const t = sliceCount <= 1 ? 0.5 : sliceIndex / Math.max(1, sliceCount - 1);
        const position = new THREE.Vector3(
          lerp((bounds.x || [0, 1])[0], (bounds.x || [0, 1])[1], 0.5),
          lerp((bounds.y || [0, 1])[0], (bounds.y || [0, 1])[1], 0.5),
          lerp((bounds.z || [0, 1])[0], (bounds.z || [0, 1])[1], 0.5)
        );
        position[slab.axis] = lerp(axisMin, axisMax, t);
        plane.position.copy(position);
        if (slab.axis === "x") {
          plane.rotation.y = Math.PI / 2;
        } else if (slab.axis === "y") {
          plane.rotation.x = -Math.PI / 2;
        }
        simulationSlices.add(plane);
        renderedPlanes += 1;
      }

    };

    const renderVolumeRaymarch = (metadata, state) => {
      const effectiveState = buildEffectiveVolumeState(state, metadata);
      const isLabelVolume = Boolean(metadata && metadata.label_volume);
      const texture = getVolumeTexture(THREE, metadata);
      const labelColorTexture = getVolumeLabelColorTexture(THREE, metadata, effectiveState);
      if (!texture) return;
      const volumeGrid = decodeVolumeGrid(metadata);
      if (!volumeGrid || !Array.isArray(volumeGrid.shape) || volumeGrid.shape.length !== 3) {
        return;
      }
      const bounds = (metadata && metadata.volume_bounds) || {};
      const xBounds = Array.isArray(bounds.x) ? bounds.x : [0, 1];
      const yBounds = Array.isArray(bounds.y) ? bounds.y : [0, 1];
      const zBounds = Array.isArray(bounds.z) ? bounds.z : [0, 1];
      const size = new THREE.Vector3(
        Math.max(1e-3, Math.abs(Number(xBounds[1]) - Number(xBounds[0]))),
        Math.max(1e-3, Math.abs(Number(yBounds[1]) - Number(yBounds[0]))),
        Math.max(1e-3, Math.abs(Number(zBounds[1]) - Number(zBounds[0])))
      );
      const center = new THREE.Vector3(
        lerp(xBounds[0], xBounds[1], 0.5),
        lerp(yBounds[0], yBounds[1], 0.5),
        lerp(zBounds[0], zBounds[1], 0.5)
      );
      const slab = getActiveVolumeSlabConfig(metadata, effectiveState);
      const axisBounds = Array.isArray(bounds[slab.axis]) ? bounds[slab.axis] : [0, 1];
      const clipCenterNorm = clamp01(
        (slab.centerWorld - Number(axisBounds[0] || 0)) /
        Math.max(1e-5, Number(axisBounds[1] || 1) - Number(axisBounds[0] || 0))
      );
      const clipHalfNorm = clamp01(
        slab.halfThickness / Math.max(1e-5, Math.abs(Number(axisBounds[1] || 1) - Number(axisBounds[0] || 0)))
      );
      const [zCount, yCount, xCount] = volumeGrid.shape;
      const texel = new THREE.Vector3(
        1.0 / Math.max(1, Number(xCount) || 1),
        1.0 / Math.max(1, Number(yCount) || 1),
        1.0 / Math.max(1, Number(zCount) || 1)
      );
      const baseSteps = Math.max(64, Math.min(640, Math.round(Number(effectiveState.raymarchSteps) || 220)));
      const stepScale = Math.max(0.25, Math.min(3.0, Number(effectiveState.raymarchStepScale) || 1.0));
      const jitterStrength = clamp01(Number(effectiveState.raymarchJitter) || 0.0);
      const lightDir = new THREE.Vector3(
        Number(effectiveState.raymarchLightX) || 0.38,
        Number(effectiveState.raymarchLightY) || 0.52,
        Number(effectiveState.raymarchLightZ) || 0.76
      ).normalize();
      const paletteNames = ["grayscale", "section_ink", "nissl", "myelin", "ice_fire", "aurora", "magma", "viridis", "allen_labels"];
      const paletteIndex = Math.max(0, paletteNames.indexOf(String(effectiveState.palette || "section_ink")));
      const clipAxisIndex = { none: 0, x: 1, y: 2, z: 3 }[String(effectiveState.clipAxis || "none")] || 0;
      const geometry = new THREE.BoxGeometry(1, 1, 1);
      const material = new THREE.ShaderMaterial({
        side: THREE.BackSide,
        transparent: true,
        depthWrite: false,
        uniforms: {
          u_data: { value: texture },
          u_labelColors: { value: labelColorTexture },
          u_labelColorCount: { value: 256.0 },
          u_steps: { value: Number(baseSteps) },
          u_stepScale: { value: Number(stepScale) },
          u_jitter: { value: Number(jitterStrength) },
          u_opacity: { value: Math.max(0.04, Number(effectiveState.opacity) || 0.6) },
          u_density: { value: Math.max(0.05, Number(effectiveState.density) || 1.0) },
          u_tfLow: { value: clamp01(effectiveState.tfLow) },
          u_tfMid: { value: clamp01(effectiveState.tfMid) },
          u_tfHigh: { value: clamp01(effectiveState.tfHigh) },
          u_threshold: { value: clamp01(effectiveState.threshold) },
          u_intensity: { value: Math.max(0.05, Number(effectiveState.intensity) || 1.0) },
          u_contrast: { value: Math.max(0.05, Number(effectiveState.contrast) || 1.0) },
          u_gamma: { value: Math.max(0.15, Number(effectiveState.gamma) || 1.0) },
          u_saturation: { value: Math.max(0.0, Number(effectiveState.saturation) || 1.0) },
          u_paletteIndex: { value: Number(paletteIndex) },
          u_labelVolume: { value: isLabelVolume ? 1.0 : 0.0 },
          u_clipAxis: { value: Number(clipAxisIndex) },
          u_clipCenter: { value: Number(clipCenterNorm) },
          u_clipHalfThickness: { value: Number(clipHalfNorm) },
          u_clipInvert: { value: effectiveState.clipInvert ? 1.0 : 0.0 },
          u_texel: { value: texel },
          u_useGradientOpacity: {
            value: effectiveState.raymarchGradientOpacity ? 1.0 : 0.0,
          },
          u_gradientOpacityFactor: {
            value: Math.max(0.0, Number(effectiveState.raymarchGradientFactor) || 0.0),
          },
          u_useShading: { value: effectiveState.raymarchShading ? 1.0 : 0.0 },
          u_ambientStrength: {
            value: Math.max(0.0, Number(effectiveState.raymarchAmbient) || 0.34),
          },
          u_diffuseStrength: {
            value: Math.max(0.0, Number(effectiveState.raymarchDiffuse) || 0.86),
          },
          u_specularStrength: {
            value: Math.max(0.0, Number(effectiveState.raymarchSpecular) || 0.22),
          },
          u_specularPower: {
            value: Math.max(2.0, Number(effectiveState.raymarchSpecularPower) || 24.0),
          },
          u_lightDir: { value: lightDir },
          u_time: { value: performance.now() / 1000.0 },
        },
        vertexShader: `
          varying vec3 vOrigin;
          varying vec3 vDirection;
          void main() {
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
            vOrigin = vec3(inverse(modelMatrix) * vec4(cameraPosition, 1.0));
            vDirection = position - vOrigin;
            gl_Position = projectionMatrix * mvPosition;
          }
        `,
        fragmentShader: `
          precision highp float;
          precision highp sampler3D;
          varying vec3 vOrigin;
          varying vec3 vDirection;
          uniform sampler3D u_data;
          uniform sampler2D u_labelColors;
          uniform float u_labelColorCount;
          uniform float u_steps;
          uniform float u_stepScale;
          uniform float u_jitter;
          uniform float u_opacity;
          uniform float u_density;
          uniform float u_tfLow;
          uniform float u_tfMid;
          uniform float u_tfHigh;
          uniform float u_threshold;
          uniform float u_intensity;
          uniform float u_contrast;
          uniform float u_gamma;
          uniform float u_saturation;
          uniform float u_paletteIndex;
          uniform float u_labelVolume;
          uniform float u_clipAxis;
          uniform float u_clipCenter;
          uniform float u_clipHalfThickness;
          uniform float u_clipInvert;
          uniform vec3 u_texel;
          uniform float u_useGradientOpacity;
          uniform float u_gradientOpacityFactor;
          uniform float u_useShading;
          uniform float u_ambientStrength;
          uniform float u_diffuseStrength;
          uniform float u_specularStrength;
          uniform float u_specularPower;
          uniform vec3 u_lightDir;
          uniform float u_time;

          float hash12(vec2 p) {
            vec3 p3 = fract(vec3(p.xyx) * 0.1031);
            p3 += dot(p3, p3.yzx + 33.33);
            return fract((p3.x + p3.y) * p3.z);
          }

          vec2 hitBox(vec3 orig, vec3 dir) {
            const vec3 boxMin = vec3(-0.5);
            const vec3 boxMax = vec3(0.5);
            vec3 invDir = 1.0 / dir;
            vec3 tMinTmp = (boxMin - orig) * invDir;
            vec3 tMaxTmp = (boxMax - orig) * invDir;
            vec3 tMin = min(tMinTmp, tMaxTmp);
            vec3 tMax = max(tMinTmp, tMaxTmp);
            float t0 = max(tMin.x, max(tMin.y, tMin.z));
            float t1 = min(tMax.x, min(tMax.y, tMax.z));
            return vec2(t0, t1);
          }

          float applyCurve(float value) {
            float v = clamp(value, 0.0, 1.0);
            float tfHigh = max(u_tfLow + 0.01, u_tfHigh);
            if (v <= u_tfLow) return 0.0;
            if (v >= tfHigh) {
              v = 1.0;
            } else {
              v = (v - u_tfLow) / max(1e-5, tfHigh - u_tfLow);
            }
            float midpointGamma = u_tfMid <= 0.0 ? 0.25 : max(0.2, log(max(1e-5, u_tfMid)) / log(0.5));
            v = pow(clamp(v, 0.0, 1.0), midpointGamma);
            if (v < u_threshold) return 0.0;
            v = (v - u_threshold) / max(1e-5, 1.0 - u_threshold);
            v = pow(clamp(v, 0.0, 1.0), max(0.15, u_gamma));
            v = (v - 0.5) * max(0.05, u_contrast) + 0.5;
            v = clamp(v * max(0.05, u_intensity), 0.0, 1.0);
            return v;
          }

          vec3 paletteColor(float value) {
            float v = clamp(value, 0.0, 1.0);
            vec3 color;
            if (u_paletteIndex > 7.5) {
              float id = max(1.0, floor(v * 255.0 + 0.5));
              float h = fract(sin(id * 12.9898 + 78.233) * 43758.5453);
              float s = 0.55 + fract(sin(id * 17.13 + 19.19) * 15731.743) * 0.35;
              float l = 0.42 + fract(sin(id * 7.77 + 39.91) * 9113.117) * 0.24;
              vec3 p = abs(fract(vec3(h) + vec3(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0);
              color = clamp(vec3(l) + vec3(s) * (clamp(p - 1.0, 0.0, 1.0) - 0.5), 0.0, 1.0);
            } else if (u_paletteIndex < 0.5) {
              color = vec3(v);
            } else if (u_paletteIndex < 1.5) {
              color = vec3(
                clamp(0.86 - pow(v, 0.78) * 0.44, 0.0, 1.0),
                clamp(0.8 - pow(v, 0.9) * 0.52, 0.0, 1.0),
                clamp(0.72 - pow(v, 0.95) * 0.56, 0.0, 1.0)
              );
            } else if (u_paletteIndex < 2.5) {
              color = vec3(
                clamp(0.72 - pow(v, 0.85) * 0.36, 0.0, 1.0),
                clamp(0.67 - pow(v, 0.9) * 0.42, 0.0, 1.0),
                clamp(0.86 - pow(v, 0.72) * 0.56, 0.0, 1.0)
              );
            } else if (u_paletteIndex < 3.5) {
              color = vec3(
                clamp(0.82 - pow(v, 0.78) * 0.54, 0.0, 1.0),
                clamp(0.81 - pow(v, 0.82) * 0.56, 0.0, 1.0),
                clamp(0.79 - pow(v, 0.86) * 0.6, 0.0, 1.0)
              );
            } else if (u_paletteIndex < 4.5) {
              color = vec3(
                clamp(0.18 + pow(v, 0.7) * 0.82, 0.0, 1.0),
                clamp(0.32 + sin(v * 3.14159265) * 0.42, 0.0, 1.0),
                clamp(0.92 - pow(v, 0.82) * 0.72, 0.0, 1.0)
              );
            } else if (u_paletteIndex < 5.5) {
              color = vec3(
                clamp(0.22 + pow(v, 0.6) * 0.42, 0.0, 1.0),
                clamp(0.25 + sin(v * 3.14159265 * 0.95) * 0.72, 0.0, 1.0),
                clamp(0.45 + pow(1.0 - v, 0.8) * 0.42, 0.0, 1.0)
              );
            } else if (u_paletteIndex < 6.5) {
              color = vec3(
                clamp(pow(v, 0.7) * 1.15, 0.0, 1.0),
                clamp(pow(v, 1.15) * 0.62, 0.0, 1.0),
                clamp(pow(v, 2.2) * 0.32 + 0.06, 0.0, 1.0)
              );
            } else {
              color = vec3(
                clamp(0.18 + v * 0.7, 0.0, 1.0),
                clamp(0.08 + sin(v * 3.14159265) * 0.82, 0.0, 1.0),
                clamp(0.36 + (1.0 - v) * 0.48, 0.0, 1.0)
              );
            }
            float l = dot(color, vec3(0.299, 0.587, 0.114));
            return mix(vec3(l), color, clamp(u_saturation, 0.0, 2.0));
          }

          vec4 lookupLabelColor(float rawValue) {
            float labelIndex = floor(rawValue * 255.0 + 0.5);
            if (labelIndex < 0.5) {
              return vec4(0.0);
            }
            float width = max(1.0, u_labelColorCount);
            float u = clamp((labelIndex + 0.5) / width, 0.0, 1.0);
            return texture2D(u_labelColors, vec2(u, 0.5));
          }

          bool insideClip(vec3 texPos) {
            if (u_clipAxis < 0.5) return true;
            float coord = u_clipAxis < 1.5 ? texPos.x : (u_clipAxis < 2.5 ? texPos.y : texPos.z);
            bool inside = abs(coord - u_clipCenter) <= max(0.005, u_clipHalfThickness);
            if (u_clipInvert > 0.5) return !inside;
            return inside;
          }

          vec3 computeGradient(vec3 texPos) {
            vec3 dx = vec3(u_texel.x, 0.0, 0.0);
            vec3 dy = vec3(0.0, u_texel.y, 0.0);
            vec3 dz = vec3(0.0, 0.0, u_texel.z);
            float gx = texture(u_data, clamp(texPos + dx, 0.0, 1.0)).r - texture(u_data, clamp(texPos - dx, 0.0, 1.0)).r;
            float gy = texture(u_data, clamp(texPos + dy, 0.0, 1.0)).r - texture(u_data, clamp(texPos - dy, 0.0, 1.0)).r;
            float gz = texture(u_data, clamp(texPos + dz, 0.0, 1.0)).r - texture(u_data, clamp(texPos - dz, 0.0, 1.0)).r;
            return vec3(gx, gy, gz);
          }

          vec3 shadeColor(vec3 color, vec3 normal, vec3 viewDir) {
            vec3 n = normalize(normal);
            vec3 l = normalize(u_lightDir);
            vec3 v = normalize(viewDir);
            vec3 h = normalize(l + v);
            float ndotl = max(dot(n, l), 0.0);
            float spec = pow(max(dot(n, h), 0.0), max(2.0, u_specularPower));
            float shade = u_ambientStrength + u_diffuseStrength * ndotl;
            return color * shade + vec3(u_specularStrength * spec);
          }

          void main() {
            vec3 rayDir = normalize(vDirection);
            vec2 bounds = hitBox(vOrigin, rayDir);
            if (bounds.x > bounds.y) discard;
            bounds.x = max(bounds.x, 0.0);
            float steps = clamp(u_steps * u_stepScale, 32.0, 720.0);
            float delta = (bounds.y - bounds.x) / max(steps, 8.0);
            float jitter = (hash12(gl_FragCoord.xy + vec2(u_time * 13.1, u_time * 7.7)) - 0.5) * u_jitter;
            float startT = bounds.x + delta * (0.5 + jitter);
            vec3 p = vOrigin + startT * rayDir;
            vec3 stepDir = rayDir * delta;
            vec4 accum = vec4(0.0);
            for (float i = 0.0; i < 720.0; i += 1.0) {
              if (i >= steps) break;
              vec3 texPos = p + 0.5;
              if (all(greaterThanEqual(texPos, vec3(0.0))) && all(lessThanEqual(texPos, vec3(1.0))) && insideClip(texPos)) {
                float raw = texture(u_data, texPos).r;
                if (u_labelVolume > 0.5) {
                  vec4 labelColor = lookupLabelColor(raw);
                  if (labelColor.a > 0.001) {
                    float extinction = max(0.02, u_density) * 1.6;
                    float alpha = (1.0 - exp(-extinction * delta * 16.0)) * clamp(u_opacity, 0.0, 1.0) * labelColor.a;
                    accum.rgb += (1.0 - accum.a) * alpha * labelColor.rgb;
                    accum.a += (1.0 - accum.a) * alpha;
                    if (accum.a >= 0.98) break;
                  }
                } else {
                  float value = applyCurve(raw);
                  if (value <= 0.0) {
                    p += stepDir;
                    continue;
                  }
                  vec3 color = paletteColor(value);
                  vec3 grad = computeGradient(texPos);
                  float gradMag = length(grad);
                  if (u_useShading > 0.5 && gradMag > 1e-6) {
                    color = shadeColor(color, grad, -rayDir);
                  }
                  float gradientBoost = 1.0;
                  if (u_useGradientOpacity > 0.5) {
                    gradientBoost = clamp(gradMag * u_gradientOpacityFactor, 0.2, 2.5);
                  }
                  float extinction = max(0.0, value) * max(0.01, u_density) * gradientBoost * 4.0;
                  float alpha = (1.0 - exp(-extinction * delta * 42.0)) * clamp(u_opacity, 0.0, 1.0);
                  accum.rgb += (1.0 - accum.a) * alpha * color;
                  accum.a += (1.0 - accum.a) * alpha;
                  if (accum.a >= 0.98) break;
                }
              }
              p += stepDir;
            }
            if (accum.a <= 0.001) discard;
            gl_FragColor = accum;
          }
        `,
      });
      material.userData = material.userData || {};
      material.userData.baseRaymarchSteps = Number(baseSteps);
      const mesh = new THREE.Mesh(geometry, material);
      mesh.scale.copy(size);
      mesh.position.copy(center);
      simulationVolumeRoot.add(mesh);
      activeRaymarchMaterial = material;
    };

    const refreshVolumePanelLayout = () => {
      if (!volumePanelEl || volumePanelEl.hidden) return;
      if (volumePanelMoved) return;
      const toolbarRect = toolbarEl ? toolbarEl.getBoundingClientRect() : null;
      const top = toolbarRect ? Math.round(toolbarRect.bottom + 8) : 64;
      volumePanelEl.style.top = `${top}px`;
      volumePanelEl.style.left = "auto";
      volumePanelEl.style.right = "12px";
    };

    const syncVolumeToolbarButton = () => {
      if (!btnToggleVolumePanel) return;
      btnToggleVolumePanel.style.color = !volumePanelEl || volumePanelEl.hidden ? "#888" : "#fff";
    };

    const stopVolumeSlicePlayback = () => {
      if (volumeSlicePlaybackTimer !== null) {
        window.clearInterval(volumeSlicePlaybackTimer);
        volumeSlicePlaybackTimer = null;
      }
    };

    const drawVolumeHistogram = (canvasEl, metadata, state) => {
      if (!canvasEl) return;
      const ctx = canvasEl.getContext("2d");
      if (!ctx) return;
      const histogram = (metadata && metadata.volume_histogram) || {};
      const values = Array.isArray(histogram.normalized_counts) ? histogram.normalized_counts : [];
      const width = canvasEl.width;
      const height = canvasEl.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "rgba(12, 18, 26, 0.86)";
      ctx.fillRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0.0, "#5cc8ff");
      gradient.addColorStop(0.5, "#ff8a5b");
      gradient.addColorStop(1.0, "#f4f7fb");
      if (values.length > 0) {
        const barWidth = width / values.length;
        ctx.fillStyle = "rgba(92, 200, 255, 0.32)";
        values.forEach((value, index) => {
          const barHeight = Math.max(1, clamp01(value) * (height - 16));
          const x = index * barWidth;
          ctx.fillRect(x, height - barHeight - 8, Math.max(1, barWidth - 1), barHeight);
        });
      }
      const lowX = clamp01(state.tfLow) * width;
      const highX = clamp01(state.tfHigh) * width;
      const midX = clamp01(state.tfMid) * width;
      ctx.fillStyle = "rgba(255,255,255,0.08)";
      ctx.fillRect(0, 0, lowX, height);
      ctx.fillRect(highX, 0, width - highX, height);
      ctx.strokeStyle = gradient;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(lowX, height - 4);
      ctx.lineTo(midX, 8);
      ctx.lineTo(highX, height - 4);
      ctx.stroke();
      [lowX, midX, highX].forEach((x, idx) => {
        ctx.beginPath();
        ctx.fillStyle = idx === 1 ? "#ffd36a" : "#ffffff";
        ctx.arc(x, idx === 1 ? 10 : height - 6, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    };

    const renderVolumePanel = () => {
      if (!volumePanelEl) return;
      const adapter = String((simulationPayload && simulationPayload.adapter) || "");
      const isVolumeAdapter = adapter === "zarr-volume" || adapter === "tiff-volume";
      if (!isVolumeAdapter || !volumeRenderState || !volumeRenderDefaults) {
        volumePanelEl.hidden = true;
        if (btnToggleVolumePanel) btnToggleVolumePanel.hidden = true;
        syncVolumeToolbarButton();
        return;
      }
      if (btnToggleVolumePanel) btnToggleVolumePanel.hidden = false;
      volumePanelEl.hidden = false;
      const state = volumeRenderState;
      const metadata = (simulationPayload && simulationPayload.metadata) || {};
      const resolvedRenderStyle = getMetadataResolvedRenderStyle(state, metadata);
      const collapsedCards = state.collapsedCards && typeof state.collapsedCards === "object"
        ? state.collapsedCards
        : {};
      const cardClass = (key) =>
        collapsedCards[key]
          ? "three-volume-card is-collapsed"
          : "three-volume-card";
      const makeSlider = (id, label, min, max, step, value, formatter) => {
        const displayValue = typeof formatter === "function" ? formatter(value) : String(value);
        return `
          <label for="${id}">${label}</label>
          <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}" />
          <output id="${id}Value">${displayValue}</output>
        `;
      };
      const sourceTag = adapter === "tiff-volume" ? "TIFF Volume" : "Zarr Volume";
      const requestVolumeRender = ({ rerenderPanel = false, refreshSceneStyle = false } = {}) => {
        const activeState = volumeRenderState || state;
        if (refreshSceneStyle) {
          applyVolumeSceneStyle(activeState);
        }
        persistVolumeState(metadata, activeState);
        if (volumeRenderFrameHandle) {
          return;
        }
        volumeRenderFrameHandle = window.requestAnimationFrame(() => {
          volumeRenderFrameHandle = 0;
          if (rerenderPanel) {
            renderVolumePanel();
          }
          renderSimulationFrame(simulationFrameIndex);
        });
      };
      const sliceAxisValue = normalizeVolumeAxis(
        state.sliceAxis || (metadata && metadata.section_axis) || "z",
        "z"
      );
      const sliceCount = getVolumeSliceCount(metadata, sliceAxisValue);
      const sliceIndexLabel = formatSliceIndexLabel(state.slicePosition, metadata, sliceAxisValue);
      volumePanelEl.innerHTML = `
        <div class="three-volume-panel-header">
          <div class="three-panel-title">Volume Look</div>
          <div class="three-volume-panel-tag">${sourceTag}</div>
        </div>
        <div class="three-volume-quick-presets">
          <button id="volumeQuickPresetCinematic" type="button">Cinematic</button>
          <button id="volumeQuickPresetSection" type="button">Section</button>
          <button id="volumeQuickPresetNissl" type="button">Nissl</button>
          <button id="volumeQuickPresetMyelin" type="button">Myelin</button>
        </div>
        <section class="${cardClass("transfer")}" data-card-key="transfer">
          <div class="three-volume-section-title">Transfer Function</div>
          <div class="three-volume-card-body">
          <canvas id="volumeHistogramCanvas" class="three-volume-histogram" width="260" height="92"></canvas>
          <div class="three-volume-grid">
          <label for="volumePreset">Preset</label>
          <select id="volumePreset">
            <option value="cinematic">Cinematic</option>
            <option value="histology_defaults">Histology Defaults</option>
            <option value="section_stack">Section Stack</option>
            <option value="nissl_sections">Nissl Sections</option>
            <option value="myelin_sections">Myelin Sections</option>
            <option value="xray">X-Ray</option>
            <option value="neon">Neon</option>
            <option value="custom">Custom</option>
          </select>
          <output id="volumePresetValue">${state.preset}</output>
          <label for="volumeRenderStyle">Renderer</label>
          <select id="volumeRenderStyle">
            <option value="points">Point Cloud</option>
            <option value="slab">Volume Stack</option>
            <option value="raymarch">Raymarch</option>
            <option value="hybrid">Hybrid</option>
          </select>
          <output id="volumeRenderStyleValue">${resolvedRenderStyle}</output>
          <label for="volumeSectionEmphasis">Section</label>
          <select id="volumeSectionEmphasis">
            <option value="auto">Auto</option>
            <option value="neutral">Neutral</option>
            <option value="nissl">Nissl</option>
            <option value="myelin">Myelin</option>
          </select>
          <output id="volumeSectionEmphasisValue">${state.sectionEmphasis}</output>
          <label for="volumePalette">Palette</label>
          <select id="volumePalette">
            <option value="allen_labels">Allen Labels</option>
            <option value="ice_fire">Ice Fire</option>
            <option value="section_ink">Section Ink</option>
            <option value="nissl">Nissl</option>
            <option value="myelin">Myelin</option>
            <option value="aurora">Aurora</option>
            <option value="magma">Magma</option>
            <option value="viridis">Viridis</option>
            <option value="grayscale">Grayscale</option>
          </select>
          <output id="volumePaletteValue">${state.palette}</output>
          ${(metadata && metadata.label_volume)
            ? makeSlider("volumeLabelColorSeed", "Label Seed", 0, 100000, 1, state.labelColorSeed, (v) => `${Math.round(Number(v))}`)
            : ""}
          ${makeSlider("volumeTfLow", "Black Point", 0.0, 0.9, 0.01, state.tfLow, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeTfMid", "Midtone", 0.05, 0.95, 0.01, state.tfMid, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeTfHigh", "White Point", 0.1, 1.0, 0.01, state.tfHigh, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeIntensity", "Intensity", 0.2, 2.5, 0.01, state.intensity, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeContrast", "Contrast", 0.3, 2.5, 0.01, state.contrast, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeGamma", "Gamma", 0.25, 2.2, 0.01, state.gamma, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeThreshold", "Threshold", 0.0, 0.95, 0.01, state.threshold, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeDensity", "Density", 0.08, 1.0, 0.01, state.density, (v) => `${Math.round(Number(v) * 100)}%`)}
          ${makeSlider("volumeSaturation", "Saturation", 0.0, 1.8, 0.01, state.saturation, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeOpacity", "Opacity", 0.05, 1.0, 0.01, state.opacity, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeSize", "Point Size", 0.01, 0.08, 0.001, state.size, (v) => Number(v).toFixed(3))}
          <label for="volumeBlendMode">Blend</label>
          <select id="volumeBlendMode">
            <option value="additive">Additive Glow</option>
            <option value="normal">Normal</option>
          </select>
          <output id="volumeBlendModeValue">${state.blendMode}</output>
          </div>
          </div>
        </section>
        <section class="${cardClass("raymarch")}" data-card-key="raymarch">
          <div class="three-volume-section-title">Raymarch</div>
          <div class="three-volume-card-body">
          <div class="three-volume-grid">
          ${makeSlider("volumeRaymarchSteps", "Ray Steps", 64, 640, 1, state.raymarchSteps, (v) => `${Math.round(Number(v))}`)}
          ${makeSlider("volumeRaymarchStepScale", "Step Scale", 0.4, 2.5, 0.01, state.raymarchStepScale, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeRaymarchJitter", "Jitter", 0.0, 1.0, 0.01, state.raymarchJitter, (v) => Number(v).toFixed(2))}
          <label for="volumeRaymarchGradientOpacity">Gradient Opacity</label>
          <input id="volumeRaymarchGradientOpacity" type="checkbox" ${state.raymarchGradientOpacity ? "checked" : ""} />
          <output id="volumeRaymarchGradientOpacityValue">${state.raymarchGradientOpacity ? "on" : "off"}</output>
          ${makeSlider("volumeRaymarchGradientFactor", "Gradient Gain", 0.0, 8.0, 0.05, state.raymarchGradientFactor, (v) => Number(v).toFixed(2))}
          <label for="volumeRaymarchShading">Shading</label>
          <input id="volumeRaymarchShading" type="checkbox" ${state.raymarchShading ? "checked" : ""} />
          <output id="volumeRaymarchShadingValue">${state.raymarchShading ? "on" : "off"}</output>
          ${makeSlider("volumeRaymarchAmbient", "Ambient", 0.0, 1.0, 0.01, state.raymarchAmbient, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeRaymarchDiffuse", "Diffuse", 0.0, 2.0, 0.01, state.raymarchDiffuse, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeRaymarchSpecular", "Specular", 0.0, 1.2, 0.01, state.raymarchSpecular, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeRaymarchSpecularPower", "Spec Power", 4.0, 96.0, 1.0, state.raymarchSpecularPower, (v) => `${Math.round(Number(v))}`)}
          ${makeSlider("volumeRaymarchLightX", "Light X", -1.0, 1.0, 0.01, state.raymarchLightX, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeRaymarchLightY", "Light Y", -1.0, 1.0, 0.01, state.raymarchLightY, (v) => Number(v).toFixed(2))}
          ${makeSlider("volumeRaymarchLightZ", "Light Z", -1.0, 1.0, 0.01, state.raymarchLightZ, (v) => Number(v).toFixed(2))}
          </div>
          </div>
        </section>
        <section class="${cardClass("slice")}" data-card-key="slice">
          <div class="three-volume-section-title">Slice Navigator</div>
          <div class="three-volume-card-body">
          <div class="three-volume-grid">
          <label for="volumeSliceMode">Single Slice</label>
          <input id="volumeSliceMode" type="checkbox" ${state.sliceModeEnabled ? "checked" : ""} />
          <output id="volumeSliceModeValue">${state.sliceModeEnabled ? "on" : "off"}</output>
          <label for="volumeSliceAxis">Slice Axis</label>
          <select id="volumeSliceAxis">
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
          <output id="volumeSliceAxisValue">${sliceAxisValue}</output>
          ${makeSlider("volumeSlicePosition", "Slice", 0.0, 1.0, 0.001, state.slicePosition, () => sliceIndexLabel)}
          ${makeSlider("volumeSliceStep", "Step", 1, Math.max(1, Math.min(32, sliceCount)), 1, state.sliceStep, (v) => `${Math.round(Number(v))}`)}
          ${makeSlider("volumeSlicePlaybackFps", "FPS", 1, 30, 1, state.slicePlaybackFps, (v) => `${Math.round(Number(v))}`)}
          <label for="volumeSliceAutoplay">Autoplay</label>
          <input id="volumeSliceAutoplay" type="checkbox" ${state.sliceAutoplay ? "checked" : ""} />
          <output id="volumeSliceAutoplayValue">${state.sliceAutoplay ? "on" : "off"}</output>
          </div>
          <div class="three-volume-actions">
            <button id="volumeSlicePrev" type="button">Prev Slice</button>
            <button id="volumeSliceNext" type="button">Next Slice</button>
          </div>
          </div>
        </section>
        <section class="${cardClass("clipping")}" data-card-key="clipping">
          <div class="three-volume-section-title">Clipping</div>
          <div class="three-volume-card-body">
          <div class="three-volume-grid">
          <label for="volumeClipAxis">Axis</label>
          <select id="volumeClipAxis">
            <option value="none">Off</option>
            <option value="x">X</option>
            <option value="y">Y</option>
            <option value="z">Z</option>
          </select>
          <output id="volumeClipAxisValue">${state.clipAxis}</output>
          ${makeSlider("volumeClipCenter", "Slice Center", 0.0, 1.0, 0.01, state.clipCenter, (v) => `${Math.round(Number(v) * 100)}%`)}
          ${makeSlider("volumeClipThickness", "Slab Width", 0.02, 1.0, 0.01, state.clipThickness, (v) => `${Math.round(Number(v) * 100)}%`)}
          <label for="volumeClipInvert">Invert</label>
          <input id="volumeClipInvert" type="checkbox" ${state.clipInvert ? "checked" : ""} />
          <output id="volumeClipInvertValue">${state.clipInvert ? "outside" : "inside"}</output>
          </div>
          </div>
        </section>
        <section class="${cardClass("actions")}" data-card-key="actions">
          <div class="three-volume-section-title">Actions</div>
          <div class="three-volume-card-body">
          <div class="three-volume-actions">
          <button id="volumePresetApply" type="button">Apply Preset</button>
          <button id="volumeHistologyDefaults" type="button">Histology Defaults</button>
          <button id="volumeFocusSlab" type="button">Focus Slab</button>
          <button id="volumeReset" type="button">Reset</button>
          </div>
          </div>
        </section>
      `;
      const presetSelect = document.getElementById("volumePreset");
      const renderStyleSelect = document.getElementById("volumeRenderStyle");
      const sectionEmphasisSelect = document.getElementById("volumeSectionEmphasis");
      const paletteSelect = document.getElementById("volumePalette");
      const blendSelect = document.getElementById("volumeBlendMode");
      const clipAxisSelect = document.getElementById("volumeClipAxis");
      const sliceModeInput = document.getElementById("volumeSliceMode");
      const sliceAxisSelect = document.getElementById("volumeSliceAxis");
      const sliceAutoplayInput = document.getElementById("volumeSliceAutoplay");
      const slicePrevBtn = document.getElementById("volumeSlicePrev");
      const sliceNextBtn = document.getElementById("volumeSliceNext");
      const headerEl = volumePanelEl.querySelector(".three-volume-panel-header");
      if (presetSelect) presetSelect.value = String(state.preset || "custom");
      if (renderStyleSelect) renderStyleSelect.value = String(resolvedRenderStyle || "points");
      if (sectionEmphasisSelect) sectionEmphasisSelect.value = String(state.sectionEmphasis || "auto");
      if (paletteSelect) paletteSelect.value = String(state.palette || (metadata && metadata.label_volume ? "allen_labels" : "ice_fire"));
      if (blendSelect) blendSelect.value = String(state.blendMode || "additive");
      if (clipAxisSelect) clipAxisSelect.value = String(state.clipAxis || "none");
      if (sliceAxisSelect) sliceAxisSelect.value = sliceAxisValue;
      if (headerEl) {
        headerEl.title = "Drag to move volume controls";
        headerEl.addEventListener("pointerdown", (event) => {
          if (!volumePanelEl) return;
          const target = event.target;
          if (target && target.closest && target.closest("button, input, select, canvas")) {
            return;
          }
          const rect = volumePanelEl.getBoundingClientRect();
          volumePanelDragState = {
            offsetX: event.clientX - rect.left,
            offsetY: event.clientY - rect.top,
          };
          volumePanelEl.classList.add("dragging");
          window.addEventListener("pointermove", onVolumePanelDragMove);
          window.addEventListener("pointerup", stopVolumePanelDrag);
          window.addEventListener("pointercancel", stopVolumePanelDrag);
        });
      }
      const cardEls = volumePanelEl.querySelectorAll(".three-volume-card[data-card-key]");
      cardEls.forEach((cardEl) => {
        const key = String(cardEl.getAttribute("data-card-key") || "");
        if (!key) return;
        const titleEl = cardEl.querySelector(".three-volume-section-title");
        if (!titleEl) return;
        titleEl.setAttribute("role", "button");
        titleEl.setAttribute("tabindex", "0");
        titleEl.setAttribute("aria-expanded", cardEl.classList.contains("is-collapsed") ? "false" : "true");
        const toggleCard = () => {
          const nextCollapsed = !cardEl.classList.contains("is-collapsed");
          cardEl.classList.toggle("is-collapsed", nextCollapsed);
          titleEl.setAttribute("aria-expanded", nextCollapsed ? "false" : "true");
          if (!state.collapsedCards || typeof state.collapsedCards !== "object") {
            state.collapsedCards = {};
          }
          state.collapsedCards[key] = nextCollapsed;
          persistVolumeState(metadata, state);
        };
        titleEl.addEventListener("click", () => {
          toggleCard();
        });
        titleEl.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            toggleCard();
          }
        });
      });
      const histogramCanvas = document.getElementById("volumeHistogramCanvas");
      const setPresetCustom = () => {
        state.preset = "custom";
        const presetValue = document.getElementById("volumePresetValue");
        if (presetValue) presetValue.textContent = "custom";
        if (presetSelect) presetSelect.value = "custom";
      };
      const enforceTransferFunctionBounds = () => {
        state.tfLow = clamp01(Number(state.tfLow));
        state.tfHigh = clamp01(Number(state.tfHigh));
        if (state.tfHigh < state.tfLow + 0.02) {
          state.tfHigh = Math.min(1, state.tfLow + 0.02);
          if (state.tfHigh < state.tfLow + 0.02) {
            state.tfLow = Math.max(0, state.tfHigh - 0.02);
          }
        }
        const midMin = Math.min(0.99, state.tfLow + 0.01);
        const midMax = Math.max(0.01, state.tfHigh - 0.01);
        state.tfMid = Math.min(midMax, Math.max(midMin, clamp01(Number(state.tfMid))));
      };
      const syncTransferFunctionControls = () => {
        const tfBindings = [
          ["volumeTfLow", state.tfLow],
          ["volumeTfMid", state.tfMid],
          ["volumeTfHigh", state.tfHigh],
        ];
        tfBindings.forEach(([id, value]) => {
          const input = document.getElementById(id);
          const output = document.getElementById(`${id}Value`);
          if (input) input.value = String(value);
          if (output) output.textContent = Number(value).toFixed(2);
        });
        drawVolumeHistogram(histogramCanvas, metadata, state);
      };
      const pickTransferFunctionHandle = (event) => {
        if (!histogramCanvas) return null;
        const rect = histogramCanvas.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return null;
        const relX = event.clientX - rect.left;
        const relY = event.clientY - rect.top;
        const handleY = {
          tfLow: rect.height - 6,
          tfMid: 10,
          tfHigh: rect.height - 6,
        };
        const handles = [
          { key: "tfLow", x: clamp01(state.tfLow) * rect.width, y: handleY.tfLow },
          { key: "tfMid", x: clamp01(state.tfMid) * rect.width, y: handleY.tfMid },
          { key: "tfHigh", x: clamp01(state.tfHigh) * rect.width, y: handleY.tfHigh },
        ];
        let closest = null;
        handles.forEach((handle) => {
          const dx = relX - handle.x;
          const dy = relY - handle.y;
          const distance = Math.hypot(dx, dy);
          if (!closest || distance < closest.distance) {
            closest = { key: handle.key, distance };
          }
        });
        return closest && closest.distance <= 14 ? closest.key : null;
      };
      syncTransferFunctionControls();
      applyVolumeSceneStyle(state);

      const bindSlider = (id, key, formatter) => {
        const input = document.getElementById(id);
        const output = document.getElementById(`${id}Value`);
        if (!input || !output) return;
        const updateOutput = (value) => {
          output.textContent = typeof formatter === "function" ? formatter(value) : String(value);
        };
        updateOutput(state[key]);
        input.addEventListener("input", () => {
          state[key] = Number(input.value);
          if (key === "tfLow" || key === "tfMid" || key === "tfHigh") {
            enforceTransferFunctionBounds();
            syncTransferFunctionControls();
          } else {
            updateOutput(state[key]);
            drawVolumeHistogram(histogramCanvas, metadata, state);
          }
          setPresetCustom();
          requestVolumeRender();
        });
      };
      bindSlider("volumeTfLow", "tfLow", (v) => Number(v).toFixed(2));
      bindSlider("volumeTfMid", "tfMid", (v) => Number(v).toFixed(2));
      bindSlider("volumeTfHigh", "tfHigh", (v) => Number(v).toFixed(2));
      bindSlider("volumeIntensity", "intensity", (v) => Number(v).toFixed(2));
      bindSlider("volumeContrast", "contrast", (v) => Number(v).toFixed(2));
      bindSlider("volumeGamma", "gamma", (v) => Number(v).toFixed(2));
      bindSlider("volumeThreshold", "threshold", (v) => Number(v).toFixed(2));
      bindSlider("volumeDensity", "density", (v) => `${Math.round(Number(v) * 100)}%`);
      bindSlider("volumeSaturation", "saturation", (v) => Number(v).toFixed(2));
      bindSlider("volumeOpacity", "opacity", (v) => Number(v).toFixed(2));
      bindSlider("volumeSize", "size", (v) => Number(v).toFixed(3));
      bindSlider("volumeLabelColorSeed", "labelColorSeed", (v) => `${Math.round(Number(v))}`);
      bindSlider("volumeRaymarchSteps", "raymarchSteps", (v) => `${Math.round(Number(v))}`);
      bindSlider("volumeRaymarchStepScale", "raymarchStepScale", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchJitter", "raymarchJitter", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchGradientFactor", "raymarchGradientFactor", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchAmbient", "raymarchAmbient", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchDiffuse", "raymarchDiffuse", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchSpecular", "raymarchSpecular", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchSpecularPower", "raymarchSpecularPower", (v) => `${Math.round(Number(v))}`);
      bindSlider("volumeRaymarchLightX", "raymarchLightX", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchLightY", "raymarchLightY", (v) => Number(v).toFixed(2));
      bindSlider("volumeRaymarchLightZ", "raymarchLightZ", (v) => Number(v).toFixed(2));
      bindSlider(
        "volumeSlicePosition",
        "slicePosition",
        (v) => formatSliceIndexLabel(v, metadata, normalizeVolumeAxis(state.sliceAxis || "z", "z"))
      );
      bindSlider("volumeSliceStep", "sliceStep", (v) => `${Math.max(1, Math.round(Number(v)))}`);
      bindSlider("volumeSlicePlaybackFps", "slicePlaybackFps", (v) => `${Math.max(1, Math.round(Number(v)))}`);
      if (histogramCanvas) {
        let activeHandle = null;
        const applyPointerToTransferFunction = (event) => {
          const rect = histogramCanvas.getBoundingClientRect();
          if (rect.width <= 0) return;
          const next = clamp01((event.clientX - rect.left) / rect.width);
          if (activeHandle === "tfLow") {
            state.tfLow = next;
          } else if (activeHandle === "tfMid") {
            state.tfMid = next;
          } else if (activeHandle === "tfHigh") {
            state.tfHigh = next;
          }
          enforceTransferFunctionBounds();
          syncTransferFunctionControls();
          setPresetCustom();
          requestVolumeRender();
        };
        const stopHandleDrag = () => {
          if (!activeHandle) return;
          activeHandle = null;
          histogramCanvas.classList.remove("is-dragging");
          histogramCanvas.style.cursor = "crosshair";
        };
        histogramCanvas.style.cursor = "crosshair";
        histogramCanvas.addEventListener("pointerdown", (event) => {
          const handle = pickTransferFunctionHandle(event);
          if (!handle) return;
          event.preventDefault();
          activeHandle = handle;
          histogramCanvas.classList.add("is-dragging");
          histogramCanvas.style.cursor = "grabbing";
          if (typeof histogramCanvas.setPointerCapture === "function") {
            histogramCanvas.setPointerCapture(event.pointerId);
          }
          applyPointerToTransferFunction(event);
        });
        histogramCanvas.addEventListener("pointermove", (event) => {
          if (activeHandle) {
            applyPointerToTransferFunction(event);
            return;
          }
          const hoverHandle = pickTransferFunctionHandle(event);
          histogramCanvas.style.cursor = hoverHandle ? "grab" : "crosshair";
        });
        histogramCanvas.addEventListener("pointerup", () => {
          stopHandleDrag();
        });
        histogramCanvas.addEventListener("pointercancel", () => {
          stopHandleDrag();
        });
        histogramCanvas.addEventListener("pointerleave", () => {
          if (!activeHandle) {
            histogramCanvas.style.cursor = "crosshair";
          }
        });
      }

      const bindToggle = (id, key, outputId, labels = ["off", "on"]) => {
        const input = document.getElementById(id);
        const output = document.getElementById(outputId);
        if (!input) return;
        const sync = () => {
          if (output) {
            output.textContent = state[key] ? labels[1] : labels[0];
          }
        };
        sync();
        input.addEventListener("change", () => {
          state[key] = Boolean(input.checked);
          state.preset = "custom";
          const presetValue = document.getElementById("volumePresetValue");
          if (presetValue) presetValue.textContent = "custom";
          if (presetSelect) presetSelect.value = "custom";
          sync();
          requestVolumeRender();
        });
      };
      bindToggle(
        "volumeRaymarchGradientOpacity",
        "raymarchGradientOpacity",
        "volumeRaymarchGradientOpacityValue",
        ["off", "on"]
      );
      bindToggle(
        "volumeRaymarchShading",
        "raymarchShading",
        "volumeRaymarchShadingValue",
        ["off", "on"]
      );

      const stepSliceBy = (delta) => {
        const axis = normalizeVolumeAxis(state.sliceAxis || "z", "z");
        const count = getVolumeSliceCount(metadata, axis);
        if (count <= 1) {
          state.slicePosition = 0.0;
          return;
        }
        const step = Math.max(1, Math.round(Number(state.sliceStep) || 1));
        const current = getSliceIndexFromPosition(state.slicePosition, metadata, axis);
        const next = ((current + delta * step) % count + count) % count;
        state.slicePosition = getSlicePositionFromIndex(next, metadata, axis);
      };
      const refreshSliceReadouts = () => {
        const axis = normalizeVolumeAxis(state.sliceAxis || "z", "z");
        const axisOutput = document.getElementById("volumeSliceAxisValue");
        const posOutput = document.getElementById("volumeSlicePositionValue");
        if (axisOutput) axisOutput.textContent = axis;
        if (posOutput) {
          posOutput.textContent = formatSliceIndexLabel(state.slicePosition, metadata, axis);
        }
      };
      const restartSlicePlaybackIfNeeded = () => {
        stopVolumeSlicePlayback();
        if (!state.sliceAutoplay) return;
        const fps = Math.max(1, Math.round(Number(state.slicePlaybackFps) || 8));
        const delayMs = Math.max(33, Math.round(1000 / fps));
        volumeSlicePlaybackTimer = window.setInterval(() => {
          stepSliceBy(1);
          refreshSliceReadouts();
          requestVolumeRender();
        }, delayMs);
      };

      if (sliceModeInput) {
        sliceModeInput.addEventListener("change", () => {
          state.sliceModeEnabled = Boolean(sliceModeInput.checked);
          const sliceModeValue = document.getElementById("volumeSliceModeValue");
          if (sliceModeValue) sliceModeValue.textContent = state.sliceModeEnabled ? "on" : "off";
          if (state.sliceModeEnabled && String(state.renderStyle || "points") === "points") {
            state.renderStyle = "slab";
            if (renderStyleSelect) renderStyleSelect.value = "slab";
            const renderStyleValue = document.getElementById("volumeRenderStyleValue");
            if (renderStyleValue) renderStyleValue.textContent = "slab";
          }
          state.preset = "custom";
          requestVolumeRender();
        });
      }
      if (sliceAxisSelect) {
        sliceAxisSelect.addEventListener("change", () => {
          state.sliceAxis = normalizeVolumeAxis(sliceAxisSelect.value, "z");
          refreshSliceReadouts();
          state.preset = "custom";
          requestVolumeRender({ rerenderPanel: true });
        });
      }
      if (sliceAutoplayInput) {
        sliceAutoplayInput.addEventListener("change", () => {
          state.sliceAutoplay = Boolean(sliceAutoplayInput.checked);
          const sliceAutoplayValue = document.getElementById("volumeSliceAutoplayValue");
          if (sliceAutoplayValue) sliceAutoplayValue.textContent = state.sliceAutoplay ? "on" : "off";
          restartSlicePlaybackIfNeeded();
          state.preset = "custom";
          requestVolumeRender();
        });
      }
      if (slicePrevBtn) {
        slicePrevBtn.addEventListener("click", () => {
          stepSliceBy(-1);
          refreshSliceReadouts();
          state.preset = "custom";
          requestVolumeRender();
        });
      }
      if (sliceNextBtn) {
        sliceNextBtn.addEventListener("click", () => {
          stepSliceBy(1);
          refreshSliceReadouts();
          state.preset = "custom";
          requestVolumeRender();
        });
      }
      const quickPresetMap = {
        volumeQuickPresetCinematic: "cinematic",
        volumeQuickPresetSection: "section_stack",
        volumeQuickPresetNissl: "nissl_sections",
        volumeQuickPresetMyelin: "myelin_sections",
      };
      Object.entries(quickPresetMap).forEach(([elementId, presetName]) => {
        const button = document.getElementById(elementId);
        if (!button) return;
        button.addEventListener("click", () => {
          const presetState = Object.assign({}, volumeRenderDefaults, getVolumePreset(presetName));
          volumeRenderState = Object.assign({}, volumeRenderState || {}, presetState, { preset: presetName });
          if (presetSelect) {
            presetSelect.value = presetName;
          }
          const presetValue = document.getElementById("volumePresetValue");
          if (presetValue) {
            presetValue.textContent = presetName;
          }
          requestVolumeRender({ rerenderPanel: true, refreshSceneStyle: true });
        });
      });

      if (paletteSelect) {
        paletteSelect.addEventListener("change", () => {
          state.palette = paletteSelect.value;
          if (["nissl", "myelin", "section_ink", "grayscale"].includes(state.palette)) {
            state.backgroundTheme = "light";
            state.pointTexture = "section";
          } else if (state.palette === "allen_labels") {
            state.backgroundTheme = "dark";
            state.pointTexture = "section";
          }
          state.preset = "custom";
          const paletteValue = document.getElementById("volumePaletteValue");
          const presetValue = document.getElementById("volumePresetValue");
          if (paletteValue) paletteValue.textContent = state.palette;
          if (presetValue) presetValue.textContent = "custom";
          if (presetSelect) presetSelect.value = "custom";
          requestVolumeRender({ refreshSceneStyle: true });
        });
      }
      if (blendSelect) {
        blendSelect.addEventListener("change", () => {
          state.blendMode = blendSelect.value;
          state.preset = "custom";
          const blendValue = document.getElementById("volumeBlendModeValue");
          const presetValue = document.getElementById("volumePresetValue");
          if (blendValue) blendValue.textContent = state.blendMode;
          if (presetValue) presetValue.textContent = "custom";
          if (presetSelect) presetSelect.value = "custom";
          requestVolumeRender();
        });
      }
      if (renderStyleSelect) {
        renderStyleSelect.addEventListener("change", () => {
          state.renderStyle = renderStyleSelect.value;
          state.preset = "custom";
          const renderStyleValue = document.getElementById("volumeRenderStyleValue");
          const presetValue = document.getElementById("volumePresetValue");
          if (renderStyleValue) {
            renderStyleValue.textContent = getMetadataResolvedRenderStyle(state, metadata);
          }
          if (presetValue) presetValue.textContent = "custom";
          if (presetSelect) presetSelect.value = "custom";
          requestVolumeRender();
        });
      }
      if (sectionEmphasisSelect) {
        sectionEmphasisSelect.addEventListener("change", () => {
          state.sectionEmphasis = sectionEmphasisSelect.value;
          state.preset = "custom";
          const sectionValue = document.getElementById("volumeSectionEmphasisValue");
          const presetValue = document.getElementById("volumePresetValue");
          if (sectionValue) sectionValue.textContent = state.sectionEmphasis;
          if (presetValue) presetValue.textContent = "custom";
          if (presetSelect) presetSelect.value = "custom";
          requestVolumeRender({ rerenderPanel: true });
        });
      }
      if (clipAxisSelect) {
        clipAxisSelect.addEventListener("change", () => {
          state.clipAxis = clipAxisSelect.value;
          const clipAxisValue = document.getElementById("volumeClipAxisValue");
          if (clipAxisValue) clipAxisValue.textContent = state.clipAxis;
          state.preset = "custom";
          requestVolumeRender();
        });
      }
      const focusSlabBtn = document.getElementById("volumeFocusSlab");
      const histologyDefaultsBtn = document.getElementById("volumeHistologyDefaults");
      if (focusSlabBtn) {
        focusSlabBtn.addEventListener("click", () => {
          fitCameraToVolumeSlab(metadata, buildEffectiveVolumeState(state, metadata));
        });
      }
      if (histologyDefaultsBtn) {
        histologyDefaultsBtn.addEventListener("click", () => {
          const next = Object.assign({}, getVolumePreset("section_stack"), {
            preset: "histology_defaults",
            renderStyle: "raymarch",
            sectionEmphasis: "auto",
            clipAxis: "none",
            clipCenter: 0.5,
            clipThickness: 1.0,
            backgroundTheme: "light",
            raymarchGradientOpacity: true,
          });
          volumeRenderState = Object.assign({}, volumeRenderState || {}, next);
          requestVolumeRender({ rerenderPanel: true, refreshSceneStyle: true });
        });
      }
      const clipInvert = document.getElementById("volumeClipInvert");
      if (clipInvert) {
        clipInvert.addEventListener("change", () => {
          state.clipInvert = Boolean(clipInvert.checked);
          const clipInvertValue = document.getElementById("volumeClipInvertValue");
          if (clipInvertValue) clipInvertValue.textContent = state.clipInvert ? "outside" : "inside";
          state.preset = "custom";
          requestVolumeRender();
        });
      }
      bindSlider("volumeClipCenter", "clipCenter", (v) => `${Math.round(Number(v) * 100)}%`);
      bindSlider("volumeClipThickness", "clipThickness", (v) => `${Math.round(Number(v) * 100)}%`);
      if (presetSelect) {
        presetSelect.addEventListener("change", () => {
          const nextPreset = String(presetSelect.value || "cinematic");
          const presetValue = document.getElementById("volumePresetValue");
          if (presetValue) presetValue.textContent = nextPreset;
        });
      }
      const applyBtn = document.getElementById("volumePresetApply");
      if (applyBtn) {
        applyBtn.addEventListener("click", () => {
          const nextPreset = presetSelect ? String(presetSelect.value || "cinematic") : "cinematic";
          const presetState = nextPreset === "custom"
            ? volumeRenderDefaults
            : Object.assign({}, volumeRenderDefaults, getVolumePreset(nextPreset));
          volumeRenderState = Object.assign({}, volumeRenderState || {}, presetState, { preset: nextPreset });
          requestVolumeRender({ rerenderPanel: true, refreshSceneStyle: true });
        });
      }
      const resetBtn = document.getElementById("volumeReset");
      if (resetBtn) {
        resetBtn.addEventListener("click", () => {
          stopVolumeSlicePlayback();
          volumeRenderState = Object.assign({}, volumeRenderDefaults, { preset: volumeRenderDefaults.preset || "cinematic" });
          clearPersistedVolumeState(metadata);
          requestVolumeRender({ rerenderPanel: true, refreshSceneStyle: true });
        });
      }
      restartSlicePlaybackIfNeeded();
      refreshVolumePanelLayout();
      syncVolumeToolbarButton();
    };

    const hideVolumePanel = () => {
      stopVolumeSlicePlayback();
      volumeRenderDefaults = null;
      volumeRenderState = null;
      volumePanelMoved = false;
      activeRaymarchMaterial = null;
      if (volumePanelEl) {
        volumePanelEl.hidden = true;
        volumePanelEl.innerHTML = "";
        volumePanelEl.style.left = "auto";
        volumePanelEl.style.right = "12px";
      }
      if (btnToggleVolumePanel) {
        btnToggleVolumePanel.hidden = true;
      }
      applyVolumeSceneStyle({ backgroundTheme: "dark" });
      syncVolumeToolbarButton();
    };

    const renderZarrGaussianSplatPoints = (orderedPoints, showPoints) => {
      if (!showPoints || !Array.isArray(orderedPoints) || orderedPoints.length <= 0) {
        return;
      }
      const metadata = (simulationPayload && simulationPayload.metadata) || {};
      const isLabelVolume = Boolean(metadata && metadata.label_volume);
      const state = buildEffectiveVolumeState(
        volumeRenderState || getVolumeRenderDefaults(metadata),
        metadata
      );
      const labelColorTable = isLabelVolume ? getVolumeLabelColorTable(metadata, state) : null;
      const splatSize = Number(state.size || metadata.splat_size);
      const splatOpacity = Number(state.opacity || metadata.splat_opacity);
      const density = clamp01(state.density);
      const pointCount = orderedPoints.length;
      const positions = new Float32Array(pointCount * 3);
      const colors = new Float32Array(pointCount * 3);
      let writeIndex = 0;
      orderedPoints.forEach((pt, ptIdx) => {
        if (!pt || typeof pt !== "object") return;
        const x = Number(pt.x);
        const y = Number(pt.y);
        const z = Number(pt.z);
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
        if (!isVolumePointVisible({ x, y, z }, state, metadata)) return;
        let curved = 0;
        let color = null;
        if (isLabelVolume) {
          const labelId = Number(pt.label_id);
          const labelIndex = Number(pt.label_index);
          if ((!Number.isFinite(labelId) || labelId <= 0) && (!Number.isFinite(labelIndex) || labelIndex <= 0)) {
            return;
          }
          let rgba = null;
          if (Number.isFinite(labelIndex) && labelIndex > 0 && labelColorTable && labelIndex <= 255) {
            const base = Math.floor(labelIndex) * 4;
            rgba = [
              labelColorTable.rgbaFloat[base + 0],
              labelColorTable.rgbaFloat[base + 1],
              labelColorTable.rgbaFloat[base + 2],
              labelColorTable.rgbaFloat[base + 3],
            ];
          } else {
            rgba = getLabelColorRgba(labelId, metadata, state);
          }
          if (!rgba || rgba[3] <= 0) return;
          curved = rgba[3];
          color = new THREE.Color(rgba[0], rgba[1], rgba[2]);
        } else {
          const confidence = clamp01(Number(pt.confidence));
          curved = applyVolumeCurve(confidence, state);
          if (curved <= 0) return;
          color = sampleVolumePalette(
            curved,
            state.palette,
            state.saturation
          );
        }
        if (density < 0.999 && hash01(ptIdx) > Math.min(1, density * (0.52 + curved * 0.48))) {
          return;
        }
        const base = writeIndex * 3;
        positions[base + 0] = x;
        positions[base + 1] = y;
        positions[base + 2] = z;
        colors[base + 0] = color.r;
        colors[base + 1] = color.g;
        colors[base + 2] = color.b;
        writeIndex += 1;
      });
      if (writeIndex <= 0) {
        return;
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(positions.subarray(0, writeIndex * 3), 3)
      );
      geometry.setAttribute(
        "color",
        new THREE.Float32BufferAttribute(colors.subarray(0, writeIndex * 3), 3)
      );
      const material = new THREE.PointsMaterial({
        size: Number.isFinite(splatSize) && splatSize > 0 ? splatSize : 0.03,
        sizeAttenuation: true,
        vertexColors: true,
        transparent: true,
        opacity: Number.isFinite(splatOpacity) ? splatOpacity : 0.38,
        depthWrite: false,
        blending: String(state.blendMode || "additive") === "normal"
          ? THREE.NormalBlending
          : THREE.AdditiveBlending,
        map: String(state.pointTexture || "glow") === "section"
          ? getZarrSectionTexture()
          : getZarrSplatTexture(),
        alphaTest: String(state.pointTexture || "glow") === "section" ? 0.005 : 0.02,
      });
      const splatPoints = new THREE.Points(geometry, material);
      simulationPoints.add(splatPoints);
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
      const adapter = String((simulationPayload && simulationPayload.adapter) || "");
      const renderMode = String(
        (((simulationPayload || {}).metadata || {}).render_mode || "")
      ).toLowerCase();
      const metadata = (simulationPayload && simulationPayload.metadata) || {};
      const isVolumeAdapter = adapter === "zarr-volume" || adapter === "tiff-volume";
      const isLabelVolume = Boolean((metadata && metadata.label_volume) || renderMode === "label_ids");
      const useGaussianSplat =
        isVolumeAdapter &&
        (renderMode === "gaussian_splatting" || renderMode === "gaussian_splats" || isLabelVolume);
      const effectiveVolumeState = buildEffectiveVolumeState(
        volumeRenderState || getVolumeRenderDefaults(metadata),
        metadata
      );
      const showPoints = display.show_points !== false;
      const showLabels = display.show_labels !== false;
      const showEdges = display.show_edges !== false;
      const showTrails = display.show_trails !== false;

      clearGroupAndDispose(simulationVolumeRoot);
      clearGroupAndDispose(simulationSlices);
      clearGroupAndDispose(simulationPoints);
      clearGroupAndDispose(simulationEdges);
      clearGroupAndDispose(simulationTrails);
      clearGroupAndDispose(simulationLabels);
      activeRaymarchMaterial = null;

      const pointMap = new Map();
      const orderedPoints = Array.isArray(frame.points) ? frame.points : [];
      if (useGaussianSplat) {
        orderedPoints.forEach((pt, ptIdx) => {
          if (!pt || typeof pt !== "object") return;
          const x = Number(pt.x);
          const y = Number(pt.y);
          const z = Number(pt.z);
          if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;
          const label = String(pt.label || `point_${ptIdx}`);
          pointMap.set(label, new THREE.Vector3(x, y, z));
        });
        const renderStyle = getMetadataResolvedRenderStyle(effectiveVolumeState, metadata);
        if (renderStyle === "raymarch") {
          renderVolumeRaymarch(metadata, effectiveVolumeState);
        }
        if (renderStyle === "slab" || renderStyle === "hybrid") {
          renderVolumeSlabPlane(orderedPoints, metadata, effectiveVolumeState);
        }
        if (renderStyle === "points" || renderStyle === "hybrid") {
          renderZarrGaussianSplatPoints(orderedPoints, showPoints);
        }
      } else {
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
      }

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
      const adapter = String((payload && payload.adapter) || "");
      const metadata = (payload && payload.metadata) || {};
      if (adapter === "zarr-volume" || adapter === "tiff-volume") {
        volumeRenderDefaults = getVolumeRenderDefaults(metadata);
        volumeRenderState = Object.assign(
          {},
          volumeRenderDefaults,
          loadPersistedVolumeState(metadata) || {}
        );
      } else {
        stopVolumeSlicePlayback();
        volumeRenderDefaults = null;
        volumeRenderState = null;
      }
      simulationActiveBehavior = String(
        (((payload.metadata || {}).run_metadata || {}).behavior || "")
      );
      document.body.setAttribute("data-threejs-simulation", "1");
      root.position.set(0, 0, 0);
      rootBaselinePosition = root.position.clone();
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
      renderVolumePanel();
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

    window.addEventListener("resize", () => {
      positionFlybodyControls();
      refreshVolumePanelLayout();
    });

    if (modelUrl) {
      if (isPanoramaImageExt(ext)) {
        const loader = new THREE.TextureLoader();
        loader.load(
          modelUrl,
          (texture) => {
            try {
              texture.colorSpace = THREE.SRGBColorSpace;
            } catch (err) {
              // Older builds may not expose colorSpace.
            }
            texture.mapping = THREE.EquirectangularReflectionMapping;
            const sphere = new THREE.Mesh(
              new THREE.SphereGeometry(500, 64, 40),
              new THREE.MeshBasicMaterial({
                map: texture,
                side: THREE.BackSide,
              })
            );
            scene.add(sphere);
            controls.enablePan = false;
            controls.enableZoom = false;
            controls.target.set(0, 0, 0);
            camera.position.set(0, 0, 0.1);
            controls.update();
            const texImage = texture.image || {};
            const width = Number(texImage.width) || 0;
            const height = Number(texImage.height) || 0;
            const ratio = height > 0 ? width / height : 0;
            const ratioHint =
              ratio > 1.95 && ratio < 2.05
                ? ""
                : " (image is not near 2:1; projection may look stretched)";
            setStatus(`Loaded 360 panorama: ${title}${ratioHint}.`);
            document.body.setAttribute("data-threejs-ready", "1");
          },
          undefined,
          (err) => {
            setStatus(`Failed to load 360 panorama: ${err}`, "error");
          }
        );
      } else if (ext === "stl") {
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
              obj.traverse((child) => {
                if (!child || !child.isMesh) return;
                const meshName = String(child.name || "");
                const parentName = String((child.parent && child.parent.name) || "");
                const regionId = String(
                  objectRegionMap[meshName] || objectRegionMap[parentName] || ""
                ).trim();
                if (regionId) {
                  child.userData = child.userData || {};
                  child.userData.regionId = regionId;
                }
              });
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
    let realtimeVideoAspect = 0;
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

          // Adjust plane size only when the source aspect changes. Reapplying
          // transforms on every streamed frame causes visible jitter in
          // embedded WebEngine builds.
          const aspect = img.width / Math.max(1, img.height);
          if (Math.abs(aspect - realtimeVideoAspect) > 0.0001) {
            videoPlane.scale.set(aspect * 10, 10, 1);
            realtimeVideoAspect = aspect;
          }
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

    const updateAdaptiveRaymarchQuality = () => {
      if (
        !activeRaymarchMaterial ||
        !activeRaymarchMaterial.uniforms ||
        !activeRaymarchMaterial.uniforms.u_steps
      ) {
        adaptiveRaymarchFactor = 1.0;
        return;
      }
      const state = volumeRenderState || {};
      const metadata = (simulationPayload && simulationPayload.metadata) || {};
      const renderStyle = getMetadataResolvedRenderStyle(state, metadata);
      if (!(renderStyle === "raymarch" || renderStyle === "hybrid")) {
        adaptiveRaymarchFactor = 1.0;
        return;
      }
      const targetMs = rendererBackend === "webgpu" ? 16.7 : 18.5;
      if (frameTimeEmaMs > targetMs + 6.0) {
        adaptiveRaymarchFactor *= 0.96;
      } else if (frameTimeEmaMs > targetMs + 2.5) {
        adaptiveRaymarchFactor *= 0.985;
      } else if (frameTimeEmaMs < targetMs - 3.0) {
        adaptiveRaymarchFactor *= 1.015;
      }
      adaptiveRaymarchFactor = Math.max(0.45, Math.min(1.35, adaptiveRaymarchFactor));
      const baseSteps = Number(activeRaymarchMaterial.userData.baseRaymarchSteps) || 220;
      const nextSteps = Math.max(
        48,
        Math.min(720, Math.round(baseSteps * adaptiveRaymarchFactor))
      );
      const currentSteps = Number(activeRaymarchMaterial.uniforms.u_steps.value) || 0;
      if (Math.abs(nextSteps - currentSteps) >= 1.0) {
        activeRaymarchMaterial.uniforms.u_steps.value = nextSteps;
      }
    };

    let animationFrameId = 0;
    let previousTickTime = performance.now();
    const tick = () => {
      const now = performance.now();
      const dtMs = Math.max(1, now - previousTickTime);
      previousTickTime = now;
      frameTimeEmaMs = frameTimeEmaMs * 0.9 + dtMs * 0.1;
      updateAdaptiveRaymarchQuality();
      controls.update();
      renderer.render(scene, camera);
      animationFrameId = window.requestAnimationFrame(tick);
    };
    tick();

    const disposeViewer = () => {
      stopVolumeSlicePlayback();
      stopMoveDrag();
      canvas.removeEventListener("pointerdown", startMoveDrag);
      window.removeEventListener("pointermove", onMoveDrag);
      window.removeEventListener("pointerup", stopMoveDrag);
      window.removeEventListener("pointercancel", stopMoveDrag);
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
