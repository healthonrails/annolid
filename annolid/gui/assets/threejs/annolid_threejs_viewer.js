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

  if (!canvas) {
    setStatus("Missing canvas element.", "error");
    document.body.setAttribute("data-threejs-error", "missing-canvas");
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
    const { MTLLoader } = await import(
      `${THREE_ESM_BASE}/examples/jsm/loaders/MTLLoader.js`
    );
    const { GLTFLoader } = await import(
      `${THREE_ESM_BASE}/examples/jsm/loaders/GLTFLoader.js`
    );

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(Math.max(1, window.devicePixelRatio || 1), 2));
    renderer.setSize(canvas.clientWidth || 800, canvas.clientHeight || 600, false);
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

        const limit = 1000000;
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

      if (detections && detections.length > 0) {
        detections.forEach((det, detIdx) => {
          // Prefer normalized keypoints because scene mapping assumes [0..1].
          const kps = det.keypoints || det.keypoints_pixels;
          if (!kps) return;

          const color = new THREE.Color().setHSL((detIdx * 0.1) % 1, 0.8, 0.5);
          const sphereGeo = new THREE.SphereGeometry(0.05, 8, 8);
          const sphereMat = new THREE.MeshBasicMaterial({ color });

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

            const kpMesh = new THREE.Mesh(sphereGeo, sphereMat);
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
              const geo = new THREE.SphereGeometry(0.04, 8, 8);
              const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.7 });

              hand.landmarks.forEach((kp, idx) => {
                // Only draw tips for less clutter
                if (idx % 4 !== 0 && idx !== 0) return;

                const x = (kp[0] - 0.5) * (aspect * 10);
                const y = -(kp[1] - 0.5) * 10;
                const mesh = new THREE.Mesh(geo, mat);
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
      const w = Math.max(1, window.innerWidth);
      const h = Math.max(1, window.innerHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      if (videoPlane) {
        // Keep background plane centered
        videoPlane.position.set(0, 0, -5);
      }
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
    console.error(err);
    setStatus(msg, "error");
    document.body.setAttribute("data-threejs-error", msg);
  }
}

boot();
