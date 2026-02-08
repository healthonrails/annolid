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

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.max(1, window.devicePixelRatio || 1));
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
    let videoPlane, videoTexture;
    const poseGroup = new THREE.Group();
    scene.add(poseGroup);

    const initRealtimeVideo = () => {
      // Use RGBAFormat for better compatibility with newer Three.js data textures
      videoTexture = new THREE.DataTexture(new Uint8Array([0, 0, 0, 255]), 1, 1, THREE.RGBAFormat);
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
    poseGroup.add(poseKeypoints);
    poseGroup.add(poseSkeleton);
    // Render on top
    poseGroup.renderOrder = 999;

    window.updateRealtimeData = (base64Frame, detections) => {
      // 1. Update Video Frame
      if (base64Frame) {
        const img = new Image();
        img.onload = () => {
          if (!videoPlane) return;

          if (!videoTexture || videoTexture.image.width !== img.width || videoTexture.image.height !== img.height) {
            if (videoTexture) videoTexture.dispose();
            videoTexture = new THREE.Texture(img);
            videoTexture.colorSpace = THREE.SRGBColorSpace;
            videoTexture.needsUpdate = true;
            videoPlane.material.map = videoTexture;

            // Adjust plane size to maintain aspect ratio
            const aspect = img.width / img.height;
            videoPlane.scale.set(aspect * 10, 10, 1);
          } else {
            videoTexture.image = img;
            videoTexture.needsUpdate = true;
          }
          videoPlane.visible = true;
        };
        img.src = `data:image/jpeg;base64,${base64Frame}`;
      }

      // 2. Update Poses
      poseKeypoints.clear();
      poseSkeleton.clear();

      if (detections && detections.length > 0) {
        detections.forEach((det, detIdx) => {
          const kps = det.keypoints_pixels || det.keypoints;
          if (!kps) return;

          const color = new THREE.Color().setHSL((detIdx * 0.1) % 1, 0.8, 0.5);
          const sphereGeo = new THREE.SphereGeometry(0.05, 8, 8);
          const sphereMat = new THREE.MeshBasicMaterial({ color });

          const points = [];
          const aspect = videoPlane.scale.x / videoPlane.scale.y;

          kps.forEach(kp => {
            let nx = kp[0];
            let ny = kp[1];

            // If pixels are provided, normalize them first
            if (det.keypoints_pixels) {
              // We don't have the original image dimensions here easily,
              // but we can assume they match the videoTexture aspect or just use normalized if available.
              // For now, let's stick to normalized if available, or assume a standard width/height if not.
              // Most YOLO pose results include both.
            }

            // Map normalized [0..1] to scene space [-aspect*5, aspect*5] and [5, -5]
            const x = (nx - 0.5) * (aspect * 10);
            const y = -(ny - 0.5) * 10;

            const kpMesh = new THREE.Mesh(sphereGeo, sphereMat);
            kpMesh.position.set(x, y, 0.01); // Slightly in front of plane
            poseKeypoints.add(kpMesh);
            points.push(new THREE.Vector3(x, y, 0.01));
          });

          // Draw skeleton if we have multiple points
          if (points.length > 1) {
            const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
            const lineMat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
            const line = new THREE.Line(lineGeo, lineMat);
            poseSkeleton.add(line);
          }
        });
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
