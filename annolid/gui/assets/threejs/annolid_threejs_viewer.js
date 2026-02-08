// Use esm.sh so addon modules resolve their internal "three" dependency
// without requiring browser import maps.
const THREE_ESM_BASE = "https://esm.sh/three@0.160.0";

async function boot() {
  const statusEl = document.getElementById("annolidThreeStatus");
  const canvas = document.getElementById("annolidThreeCanvas");
  const modelUrl = window.__annolidThreeModelUrl || "";
  const modelExtHint = (window.__annolidThreeModelExt || "").toLowerCase();
  const title = window.__annolidThreeTitle || "3D";

  const setStatus = (msg, level = "info") => {
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.setAttribute("data-level", level);
  };

  if (!canvas || !modelUrl) {
    setStatus("Missing canvas or model URL.", "error");
    document.body.setAttribute("data-threejs-error", "missing-canvas-or-url");
    return;
  }

  try {
    const THREE = await import(`${THREE_ESM_BASE}`);
    const { OrbitControls } = await import(
      `${THREE_ESM_BASE}/examples/jsm/controls/OrbitControls.js`
    );
    const { STLLoader } = await import(
      `${THREE_ESM_BASE}/examples/jsm/loaders/STLLoader.js`
    );
    const { PLYLoader } = await import(
      `${THREE_ESM_BASE}/examples/jsm/loaders/PLYLoader.js`
    );
    const { OBJLoader } = await import(
      `${THREE_ESM_BASE}/examples/jsm/loaders/OBJLoader.js`
    );

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.max(1, window.devicePixelRatio || 1));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121824);

    const camera = new THREE.PerspectiveCamera(
      50,
      window.innerWidth / Math.max(1, window.innerHeight),
      0.01,
      10000
    );
    camera.position.set(0, 0, 3);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);
    controls.update();

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
      controls.update();

      const maxDim = Math.max(size.x, size.y, size.z, 0.001);
      const distance = maxDim * 1.8;
      camera.position.set(distance, distance * 0.65, distance);
      camera.near = Math.max(0.001, maxDim / 1000);
      camera.far = Math.max(1000, maxDim * 20);
      camera.updateProjectionMatrix();

      const axes = new THREE.AxesHelper(maxDim * 0.35);
      root.add(axes);

      setStatus(`Loaded ${title} (${ext.toUpperCase()}).`);
      document.body.setAttribute("data-threejs-ready", "1");
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
      const loader = new PLYLoader();
      loader.load(
        modelUrl,
        (geometry) => {
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
        },
        undefined,
        (err) => {
          setStatus(`Failed to load PLY: ${err}`, "error");
        }
      );
    } else if (ext === "obj") {
      const loader = new OBJLoader();
      loader.load(
        modelUrl,
        (obj) => {
          obj.traverse((child) => {
            if (child && child.isMesh) {
              child.material = new THREE.MeshStandardMaterial({
                color: 0xb7b9ff,
                roughness: 0.58,
                metalness: 0.08,
              });
            }
          });
          addLoadedObject(obj);
        },
        undefined,
        (err) => {
          setStatus(`Failed to load OBJ: ${err}`, "error");
        }
      );
    } else if (ext === "csv" || ext === "xyz") {
      const cloud = await parseDelimitedPointCloud();
      if (!cloud) {
        setStatus("No valid XYZ rows found in file.", "error");
      } else {
        addLoadedObject(cloud);
      }
    } else {
      setStatus(`Unsupported 3D format: .${ext}`, "error");
      document.body.setAttribute("data-threejs-error", "unsupported-format");
      return;
    }

    const onResize = () => {
      const w = Math.max(1, window.innerWidth);
      const h = Math.max(1, window.innerHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    };
    window.addEventListener("resize", onResize, { passive: true });

    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      window.requestAnimationFrame(tick);
    };
    tick();
  } catch (err) {
    const msg = String(err || "Failed to initialize Three.js viewer");
    setStatus(msg, "error");
    document.body.setAttribute("data-threejs-error", msg);
  }
}

boot();
