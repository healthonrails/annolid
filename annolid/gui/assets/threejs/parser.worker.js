// --- Start of parser.worker.js (Batching Version) ---
console.log("Worker script execution started (Batching).");

// Define minimal THREE.Color first
const THREE = {
    Color: class Color {
        constructor(r, g, b) { this.isColor = true; this.r = 1; this.g = 1; this.b = 1; if (g !== undefined && b !== undefined) { this.setRGB(r, g, b); } else if (r !== undefined) { this.set(r); } }
        set(value) { if (value && value.isColor) { this.copy(value); } else if (typeof value === 'number') { this.setHex(value); } else if (typeof value === 'string') { this.setStyle(value); } return this; }
        setHex(hex) { hex = Math.floor(hex); this.r = (hex >> 16 & 255) / 255; this.g = (hex >> 8 & 255) / 255; this.b = (hex & 255) / 255; return this; }
        setRGB(r, g, b) { this.r = r; this.g = g; this.b = b; return this; }
        setStyle(style) { if (/^#([A-Fa-f0-9]{6})$/.test(style)) { const hex = parseInt(style.substring(1), 16); this.setHex(hex); } return this; }
        copy(color) { this.r = color.r; this.g = color.g; this.b = color.b; return this; }
        getHexString() { const r = Math.floor(this.r * 255).toString(16).padStart(2, '0'); const g = Math.floor(this.g * 255).toString(16).padStart(2, '0'); const b = Math.floor(this.b * 255).toString(16).padStart(2, '0'); return r + g + b; }
        setHSL(h, s, l) { h = ((h % 1) + 1) % 1; s = Math.max(0, Math.min(1, s)); l = Math.max(0, Math.min(1, l)); if (s === 0) { this.r = this.g = this.b = l; } else { const hue2rgb = (p, q, t) => { if (t < 0) t += 1; if (t > 1) t -= 1; if (t < 1 / 6) return p + (q - p) * 6 * t; if (t < 1 / 2) return q; if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6; return p; }; const p = l <= 0.5 ? l * (1 + s) : l + s - (l * s); const q = (2 * l) - p; this.r = hue2rgb(q, p, h + 1 / 3); this.g = hue2rgb(q, p, h); this.b = hue2rgb(q, p, h - 1 / 3); } return this; }
    }
};

try {
    console.log("Worker attempting to import PapaParse...");
    self.importScripts('https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js');
    console.log("Worker PapaParse imported successfully. Papa =", typeof Papa);
} catch (e) {
    console.error("Worker failed to import PapaParse!", e);
    self.postMessage({ type: 'error', message: 'Worker failed to load PapaParse library. Details: ' + e });
    throw e;
}

function workerLog(...args) { /* console.log(...args); */ }

console.log("Worker setting up onmessage handler...");

// --- Batch Configuration ---
const BATCH_SIZE = 1 * 1000 * 1000; // Process 1 million points per batch

self.onmessage = function (e) {
    console.log("Worker onmessage received data.");
    const { file, config } = e.data;
    const RESOLUTION = config.RESOLUTION;
    workerLog("Worker (Batching) received file:", file.name, "Size:", file.size);

    if (typeof Papa === 'undefined') {
        console.error("Worker Papa is undefined inside onmessage!");
        self.postMessage({ type: 'error', message: 'PapaParse library not available in worker.' });
        return;
    }

    console.log("Worker starting Papa.parse with step function...");

    // --- Batch Data Structures ---
    let batchPositions = [];
    let batchColorsUint8 = [];
    let batchPointRegionIDsList = [];
    let batchPointCounter = 0;
    let totalPointCounter = 0;

    // --- Shared Data ---
    let regionAcronymToId = {};
    let regionInfoList = [];
    let nextRegionID = 0;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

    let processedRows = 0;
    let startTime = performance.now();
    const estimatedBytesPerRow = 60;
    const estimatedTotalRows = file.size / estimatedBytesPerRow;
    let parseComplete = false; // Flag to track if Papa's complete callback ran

    // --- Function to process and send a batch ---
    function sendBatch() {
        if (batchPointCounter === 0) return;

        console.log(`Worker sending batch, ${batchPointCounter} points, total ${totalPointCounter}`);
        // Create TypedArrays from the batch data
        const positionsTyped = new Float32Array(batchPositions);
        const colorsTyped = new Uint8Array(batchColorsUint8);
        const pointRegionIDsTyped = new Uint32Array(batchPointRegionIDsList);

        // Post the message with transferable objects
        self.postMessage({
            type: 'batch_data',
            payload: {
                positions: positionsTyped,
                colors: colorsTyped,
                pointRegionIDs: pointRegionIDsTyped,
                startIndex: totalPointCounter - batchPointCounter // Start index for this batch
            }
        }, [positionsTyped.buffer, colorsTyped.buffer, pointRegionIDsTyped.buffer]);

        // Reset batch arrays for the next batch
        batchPositions = [];
        batchColorsUint8 = [];
        batchPointRegionIDsList = [];
        batchPointCounter = 0;
    }

    // --- Use PapaParse Step Function ---
    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        dynamicTyping: false,
        worker: false,
        step: function (results, parser) {
            processedRows++;
            const row = results.data;

            // Row Validation
            if (!row || typeof row.x !== 'string' || typeof row.y !== 'string' || typeof row.z !== 'string' || typeof row.region_acronym !== 'string') { return; }
            const regionAcronym = row.region_acronym.trim();
            if (regionAcronym === "") { return; }
            const xStr = row.x.trim(); const yStr = row.y.trim(); const zStr = row.z.trim();
            const x = Number(xStr); const y = Number(yStr); const z = Number(zStr);

            // Coordinate Validation
            if (!isNaN(x) && !isNaN(y) && !isNaN(z) && isFinite(x) && isFinite(y) && isFinite(z)) {
                const scaledX = x * RESOLUTION.x; const scaledY = y * RESOLUTION.y; const scaledZ = z * RESOLUTION.z;
                // Add to current batch array
                batchPositions.push(scaledX, scaledY, scaledZ);

                // Update overall bounds
                if (scaledX < minX) minX = scaledX; if (scaledX > maxX) maxX = scaledX;
                if (scaledY < minY) minY = scaledY; if (scaledY > maxY) maxY = scaledY;
                if (scaledZ < minZ) minZ = scaledZ; if (scaledZ > maxZ) maxZ = scaledZ;

                // Region ID Handling
                let regionID;
                if (regionAcronymToId[regionAcronym] === undefined) {
                    regionID = nextRegionID++;
                    regionAcronymToId[regionAcronym] = regionID;
                    let hash = 0; for (let i = 0; i < regionAcronym.length; i++) { hash = regionAcronym.charCodeAt(i) + ((hash << 5) - hash); hash = hash & hash; }
                    const randHue = (parseInt(hash.toString().slice(-5)) % 360) / 360;
                    const color = new THREE.Color().setHSL(randHue, 0.6 + Math.random() * 0.3, 0.5 + Math.random() * 0.2);
                    const regionFullName = (row.region_name && typeof row.region_name === 'string') ? row.region_name.trim() : regionAcronym;
                    regionInfoList[regionID] = {
                        id: regionID, acronym: regionAcronym, fullName: regionFullName,
                        colorHex: '#' + color.getHexString(), lowerAcronym: regionAcronym.toLowerCase(),
                        lowerFullName: regionFullName.toLowerCase(), r: Math.floor(color.r * 255),
                        g: Math.floor(color.g * 255), b: Math.floor(color.b * 255)
                    };
                } else { regionID = regionAcronymToId[regionAcronym]; }

                const regionInfo = regionInfoList[regionID];
                if (regionInfo) {
                    // Add color and ID to current batch arrays
                    batchColorsUint8.push(regionInfo.r, regionInfo.g, regionInfo.b);
                    batchPointRegionIDsList.push(regionID);
                    batchPointCounter++;
                    totalPointCounter++; // Increment total count

                    // If batch is full, send it
                    if (batchPointCounter >= BATCH_SIZE) {
                        sendBatch();
                    }
                } else { console.warn('Worker missing regionInfo for ID ' + regionID + ', acronym ' + regionAcronym); }
            } // End if valid point

            // Send progress update periodically
            if (processedRows % 100000 === 0) { // e.g., every 100k rows
                if (estimatedTotalRows > 0) {
                    const progress = Math.round((processedRows / estimatedTotalRows) * 90);
                    self.postMessage({ type: 'progress', processedRows: processedRows, progress: Math.min(90, progress > 0 ? progress : 0) });
                }
            }
        }, // End step handler
        complete: function () {
            console.log("Worker Papa.parse complete callback triggered.");
            parseComplete = true; // Mark parsing as officially done

            // Send any remaining points in the last batch
            sendBatch();

            // Calculate final bounds
            const centerX = (minX + maxX) / 2; const centerY = (minY + maxY) / 2; const centerZ = (minZ + maxZ) / 2;
            const sizeX = maxX - minX; const sizeY = maxY - minY; const sizeZ = maxZ - minZ;
            const maxSize = Math.max(sizeX, sizeY, sizeZ);
            const bounds = {
                min: { x: minX === Infinity ? 0 : minX, y: minY === Infinity ? 0 : minY, z: minZ === Infinity ? 0 : minZ },
                max: { x: maxX === -Infinity ? 0 : maxX, y: maxY === -Infinity ? 0 : maxY, z: maxZ === -Infinity ? 0 : maxZ },
                center: { x: isNaN(centerX) ? 0 : centerX, y: isNaN(centerY) ? 0 : centerY, z: isNaN(centerZ) ? 0 : centerZ },
                size: isNaN(maxSize) || maxSize <= 0 ? 1000 : maxSize
            };

            // Send the final message containing metadata
            self.postMessage({
                type: 'processing_complete',
                payload: {
                    regionInfoList: regionInfoList, // Send the lookup table
                    bounds: bounds,                  // Send calculated bounds
                    totalPoints: totalPointCounter   // Send final verified count
                }
            });

            console.log("Worker sent processing_complete message.");
            const endTime = performance.now();
            workerLog('Worker parsing took ' + ((endTime - startTime) / 1000).toFixed(2) + ' seconds.');

        }, // End complete handler
        error: function (error) {
            console.error("Worker Papa.parse error:", error);
            self.postMessage({ type: 'error', message: error.message || 'Unknown parsing error' });
        } // End error handler
    }); // End Papa.parse

    console.log("Worker Papa.parse setup complete (processing rows via step).");
}; // End self.onmessage

console.log("Worker script execution finished (waiting for messages).");
// --- End of parser.worker.js ---
