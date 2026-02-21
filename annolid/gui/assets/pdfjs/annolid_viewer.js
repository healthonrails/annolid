// Annolid PDF.js viewer runtime
// This file is generated from the embedded viewer and can be edited directly.

(() => {
  const pdfUrl = String(window.__annolidPdfUrl || "");
  const pdfBase64 = String(window.__annolidPdfBase64 || "");
  const pdfTitle = String(window.__annolidPdfTitle || "");

  document.addEventListener("DOMContentLoaded", async () => {
    try {
      if (typeof pdfjsLib === "undefined" || !pdfjsLib) {
        document.body.setAttribute("data-pdfjs-error", "pdfjsLib not loaded");
        return;
      }
      try {
        const workerUrl = (typeof URL !== "undefined" && document.baseURI)
          ? (new URL("pdfjs/annolid.worker.js", document.baseURI)).toString()
          : "pdfjs/annolid.worker.js";
        if (pdfjsLib.GlobalWorkerOptions) {
          pdfjsLib.GlobalWorkerOptions.workerSrc = workerUrl;
        }
      } catch (e) { }
      window.__annolidPdfjsReady = true;

      window.__annolidSpans = [];
      window.__annolidSpanCounter = 0;
      window.__annolidSelectionSpans = [];
      window.__annolidSpanMeta = {};
      window.__annolidPages = {};
      window.__annolidTts = { sentenceIndices: [], wordIndex: null, lastPages: [] };
      window.__annolidMarks = { tool: "select", color: "#ffb300", size: 10, undo: [], drawing: null };
      window.__annolidReaderEnabled = window.__annolidReaderEnabled || false;
      window.__annolidParagraphsByPage = {};
      window.__annolidParagraphs = [];
      window.__annolidParagraphOffsets = {};
      window.__annolidParagraphTotal = 0;
      window.__annolidSplitTextIntoSentenceRanges = function (text) {
        const s = String(text || "");
        if (!s) return [];
        const END = new Set([".", "!", "?", "。", "！", "？"]);
        const isCjkLead = (ch) => /[぀-ヿ㐀-䶿一-鿿豈-﫿]/.test(ch || "");
        const isNarrationLead = (ch) => {
          const c = String(ch || "");
          return isCjkLead(c) || c === "（" || c === "【" || c === "“" || c === "‘";
        };
        const QUOTE_CLOSERS = new Set(["”", "’", "\"", "'"]);
        const CLOSERS = new Set([
          "”", "’", "\"", "'", ")", "]", "）", "】", "》", "」", "』", "〉",
        ]);
        const ranges = [];
        let start = 0;
        for (let i = 0; i < s.length; i++) {
          const ch = s[i];
          if (!END.has(ch)) continue;
          let end = i + 1;
          while (end < s.length && END.has(s[end])) end++;
          const punctEnd = end;
          // Attach trailing quotes/brackets, even if PDF extraction inserted whitespace
          // between end punctuation and the closer (e.g. "。”" or "。 ”").
          let probe = end;
          for (let guard = 0; guard < 8; guard++) {
            let ws = probe;
            while (ws < s.length && /\s/.test(s[ws])) ws++;
            if (ws < s.length && CLOSERS.has(s[ws])) {
              probe = ws + 1;
              while (probe < s.length && CLOSERS.has(s[probe])) probe++;
              end = probe;
              continue;
            }
            break;
          }
          // Fix skipped patterns around Chinese quote endings like:
          //   "。”说完..." / "。”小鹊..."
          // When a quoted sentence ends and narration continues immediately, some PDFs place the
          // continuation spans slightly off-baseline; splitting here makes the next part prone
          // to being reordered and "skipped". Treat these as a single sentence chunk.
          try {
            let hasQuoteCloser = false;
            for (let j = punctEnd; j < end; j++) {
              const c = s[j];
              if (/\s/.test(c)) continue;
              if (QUOTE_CLOSERS.has(c)) {
                hasQuoteCloser = true;
                break;
              }
            }
            if (hasQuoteCloser) {
              let k = end;
              // PDFs may insert multiple spaces/newlines between the quote closer and narration.
              // Skip a reasonable amount of whitespace and merge when narration continues.
              while (k < s.length && /\s/.test(s[k]) && (k - end) < 48) k++;
              if (k < s.length && isNarrationLead(s[k])) {
                continue;
              }
            }
          } catch (e) { }
          ranges.push([start, end]);
          start = end;
        }
        if (start < s.length) ranges.push([start, s.length]);
        // Trim whitespace around ranges.
        const out = [];
        for (const pair of ranges) {
          let a = pair[0];
          let b = pair[1];
          while (a < b && /\s/.test(s[a])) a++;
          while (b > a && /\s/.test(s[b - 1])) b--;
          if (b > a) out.push([a, b]);
        }
        return out;
      };

      window.__annolidSplitParagraphIntoSentences = function (para) {
        try {
          const spansRaw = (para && Array.isArray(para.spans)) ? para.spans.filter((n) => Number.isInteger(n)) : [];
          if (!spansRaw.length) return [];
          const spans = (typeof _annolidOrderSpanIndicesForReading === "function")
            ? _annolidOrderSpanIndicesForReading(spansRaw)
            : spansRaw.slice();
          const nodes = window.__annolidSpans || [];
          const isCjkLike = (ch) => /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af，。！？、；：“”‘’（）《》「」『』]/.test(ch || "");
          const parts = [];
          let combined = "";
          spans.forEach((idx) => {
            const node = nodes[idx];
            if (!node) return;
            const raw = String(node.textContent || "");
            const cleaned = raw.replace(/\u00ad/g, "").replace(/\u00a0/g, " ");
            const text = cleaned.replace(/\s+/g, " ").trim();
            if (!text) return;
            let sep = "";
            if (combined) {
              const prev = combined[combined.length - 1] || "";
              const next = text[0] || "";
              if (!/\s/.test(prev) && !(isCjkLike(prev) && isCjkLike(next))) {
                sep = " ";
              }
            }
            const start = combined.length + sep.length;
            combined += sep + text;
            const end = combined.length;
            parts.push({ idx, start, end });
          });
          // Use the exact combined span text for range computation so indices
          // align with span boundaries; normalize only per sentence output.
          const paraText = combined;
          const normalizedText = _annolidNormalizeText(paraText);
          const resolvedPageNum = (parseInt(para.pageNum || para.page || 0, 10) || 0);
          if (!normalizedText) return [];
          const shortCharLimit = 120;
          const shortSpanLimit = 4;
          const shortParagraph = (normalizedText.length <= shortCharLimit || spans.length <= shortSpanLimit);
          if (shortParagraph) {
            return [{
              text: normalizedText,
              spans: spans.slice(),
              pageNum: resolvedPageNum,
            }];
          }
          const sentences = [];
          const ranges = (typeof window.__annolidSplitTextIntoSentenceRanges === "function")
            ? (window.__annolidSplitTextIntoSentenceRanges(paraText) || [])
            : [];
          if (!ranges.length) {
            return [{
              text: normalizedText,
              spans: spans.slice(),
              pageNum: resolvedPageNum,
            }];
          }
          for (const r of ranges) {
            const start = r[0];
            const end = r[1];
            const outText = _annolidNormalizeText(paraText.slice(start, end));
            if (!outText) continue;
            const group = parts
              .filter((p) => p.end > start && p.start < end)
              .map((p) => p.idx);
            sentences.push({
              text: outText,
              spans: group.length ? group : spans.slice(),
              pageNum: resolvedPageNum,
            });
          }
          // Post-merge edge cases where a quoted sentence ends and narration continues immediately,
          // e.g., "。”说完..." / "。”小鹊..." which otherwise may split and reorder.
          if (sentences.length > 1) {
            const merged = [];
            const endQuoteRe = /[.!?。！？]+[”’"'][\)\]）】》」』〉]*$/;
            const cjkLeadRe = /^[぀-ヿ㐀-䶿一-鿿豈-﫿（【“‘]/;
            let i = 0;
            while (i < sentences.length) {
              let cur = sentences[i];
              if (i + 1 < sentences.length && endQuoteRe.test(cur.text || "")) {
                const next = sentences[i + 1];
                if (cjkLeadRe.test((next.text || "").trim())) {
                  const nextText = String(next.text || "");
                  const joiner = cjkLeadRe.test(nextText) ? "" : " ";
                  const mergedText = _annolidNormalizeText((cur.text || "") + joiner + nextText);
                  const mergedSpansRaw = []
                    .concat(Array.isArray(cur.spans) ? cur.spans : [])
                    .concat(Array.isArray(next.spans) ? next.spans : []);
                  const mergedSpans = (typeof _annolidOrderSpanIndicesForReading === "function")
                    ? _annolidOrderSpanIndicesForReading(mergedSpansRaw)
                    : mergedSpansRaw;
                  merged.push({
                    text: mergedText,
                    spans: mergedSpans,
                    pageNum: resolvedPageNum,
                  });
                  i += 2;
                  continue;
                }
              }
              merged.push(cur);
              i += 1;
            }
            sentences.length = 0;
            merged.forEach((m) => sentences.push(m));
          }
          if (sentences.length > 1) {
            const totalLen = sentences.reduce((s, item) => s + (item.text || "").length, 0);
            const avgLen = totalLen / Math.max(1, sentences.length);
            if (avgLen < 45 && normalizedText.length <= 180) {
              return [{
                text: normalizedText,
                spans: spans.slice(),
                pageNum: resolvedPageNum,
              }];
            }
          }
          if (!sentences.length && paraText) {
            sentences.push({
              text: normalizedText,
              spans: spans.slice(),
              pageNum: resolvedPageNum,
            });
          }
          return sentences;
        } catch (e) {
          return [];
        }
      };
      window.__annolidBridge = null;
      const annolidPdfKey = String(window.__annolidPdfKey || "");
      const annolidStorageKey = annolidPdfKey ? ("annolidPdfState:" + annolidPdfKey) : "";
      const annolidNowMs = () => (Date.now ? Date.now() : (new Date()).getTime());
      const annolidSafeJsonParse = (raw) => {
        try {
          const obj = JSON.parse(raw || "");
          return (obj && typeof obj === "object") ? obj : null;
        } catch (e) {
          return null;
        }
      };
      function _annolidGetStoredUserState() {
        if (!annolidStorageKey) return null;
        try {
          return annolidSafeJsonParse(localStorage.getItem(annolidStorageKey));
        } catch (e) {
          return null;
        }
      }
      function _annolidSetStoredUserState(state) {
        if (!annolidStorageKey) return;
        try {
          localStorage.setItem(annolidStorageKey, JSON.stringify(state || {}));
        } catch (e) { }
      }
      function _annolidPickNewestState(a, b) {
        const aa = (a && typeof a.updatedAt === "number") ? a.updatedAt : 0;
        const bb = (b && typeof b.updatedAt === "number") ? b.updatedAt : 0;
        return (bb >= aa) ? b : a;
      }
      function _annolidNormalizeUserState(state) {
        const out = (state && typeof state === "object") ? state : {};
        if (!out.version) out.version = 1;
        if (!out.updatedAt) out.updatedAt = annolidNowMs() / 1000.0;
        if (!out.reading) out.reading = {};
        if (!Array.isArray(out.bookmarks)) out.bookmarks = [];
        if (!Array.isArray(out.notes)) out.notes = [];
        if (!out.marks) out.marks = {};
        if (!out.tool) out.tool = {};
        return out;
      }
      function _annolidMergeUserState(base, incoming) {
        const out = Object.assign({}, base || {});
        if (incoming && typeof incoming === "object") {
          for (const k of Object.keys(incoming)) out[k] = incoming[k];
        }
        return out;
      }
      window.__annolidRenderedPages = 0;
      window.__annolidPdfLoaded = false;
      window.__annolidLinkTargets = {};
      window.__annolidLinkTargetCounter = 0;

      if (window.qt && window.qt.webChannelTransport && typeof QWebChannel !== "undefined") {
        try {
          new QWebChannel(window.qt.webChannelTransport, function (channel) {
            window.__annolidBridge = channel.objects.annolidBridge || null;
          });
        } catch (e) {
          window.__annolidBridge = null;
        }
      }
      function _annolidWaitForBridge(timeoutMs) {
        const limit = Math.max(50, parseInt(timeoutMs || 0, 10) || 900);
        const start = annolidNowMs();
        return new Promise((resolve) => {
          function tick() {
            const bridge = window.__annolidBridge;
            if (bridge && typeof bridge === "object") {
              resolve(bridge);
              return;
            }
            if (annolidNowMs() - start > limit) {
              resolve(null);
              return;
            }
            setTimeout(tick, 50);
          }
          tick();
        });
      }
      function _annolidFetchBridgeUserState() {
        if (!annolidPdfKey) return Promise.resolve(null);
        return _annolidWaitForBridge(1200).then((bridge) => {
          if (!bridge || typeof bridge.getUserState !== "function") return null;
          return new Promise((resolve) => {
            try {
              bridge.getUserState(annolidPdfKey, function (result) {
                resolve((result && typeof result === "object") ? result : null);
              });
            } catch (e) {
              resolve(null);
            }
          });
        });
      }
      function _annolidSendBridgeUserState(state) {
        if (!annolidPdfKey) return;
        const bridge = window.__annolidBridge;
        if (!bridge || typeof bridge.saveUserState !== "function") return;
        try {
          bridge.saveUserState({ pdfKey: annolidPdfKey, state: state || {} });
        } catch (e) { }
      }
      function _annolidClearBridgeUserState() {
        if (!annolidPdfKey) return;
        const bridge = window.__annolidBridge;
        if (!bridge || typeof bridge.clearUserState !== "function") return;
        try {
          bridge.clearUserState(annolidPdfKey);
        } catch (e) { }
      }

      function _annolidHexToRgba(hex, alpha) {
        const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
        if (!m) return `rgba(255, 179, 0, ${alpha})`;
        const r = parseInt(m[1], 16);
        const g = parseInt(m[2], 16);
        const b = parseInt(m[3], 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
      }

      function _annolidSetupHiDpiCanvas(canvas, cssWidth, cssHeight) {
        const dpr = (window.devicePixelRatio || 1);
        const w = Math.max(1, Math.round(cssWidth * dpr));
        const h = Math.max(1, Math.round(cssHeight * dpr));
        canvas.width = w;
        canvas.height = h;
        canvas.style.width = cssWidth + "px";
        canvas.style.height = cssHeight + "px";
        const ctx = canvas.getContext("2d");
        if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { ctx, dpr };
      }

      function _annolidClearCanvas(canvas, ctx, dpr) {
        if (!canvas || !ctx) return;
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }

      function _annolidGetPageState(pageNum) {
        const key = String(pageNum);
        let state = window.__annolidPages[key];
        if (!state) {
          state = {
            pageNum,
            pageDiv: null,
            width: 0,
            height: 0,
            dpr: 1,
            ttsCanvas: null,
            ttsCtx: null,
            markCanvas: null,
            markCtx: null,
            marks: { highlights: [], strokes: [] },
          };
          window.__annolidPages[key] = state;
        }
        if (!state.marks) state.marks = { highlights: [], strokes: [] };
        if (!state.marks.highlights) state.marks.highlights = [];
        if (!state.marks.strokes) state.marks.strokes = [];
        return state;
      }

      function _annolidPointFromEvent(ev, canvas) {
        const rect = canvas.getBoundingClientRect();
        const x = (ev.clientX - rect.left) * (canvas.width / rect.width) / (window.devicePixelRatio || 1);
        const y = (ev.clientY - rect.top) * (canvas.height / rect.height) / (window.devicePixelRatio || 1);
        return { x, y };
      }

      function _annolidStrokeAlpha(tool) {
        if (tool === "highlighter") return 0.28;
        return 0.92;
      }

      function _annolidDrawStrokeSegment(ctx, from, to, tool, color, size) {
        if (!ctx || !from || !to) return;
        ctx.save();
        ctx.globalAlpha = _annolidStrokeAlpha(tool);
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(1, size);
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        if (tool === "highlighter") {
          try { ctx.globalCompositeOperation = "multiply"; } catch (e) { }
        }
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
        ctx.restore();
      }

      function _annolidRenderMarks(pageNum) {
        const state = _annolidGetPageState(pageNum);
        if (!state || !state.markCanvas || !state.markCtx) return;
        _annolidClearCanvas(state.markCanvas, state.markCtx, state.dpr || 1);
        for (const hl of (state.marks.highlights || [])) {
          const fill = _annolidHexToRgba(hl.color || "#ffb300", hl.alpha || 0.28);
          const rects = hl.rects || [];
          state.markCtx.save();
          state.markCtx.fillStyle = fill;
          for (const r of rects) {
            const pad = 1.5;
            state.markCtx.fillRect(
              Math.max(0, r.x - pad),
              Math.max(0, r.y - pad),
              Math.max(0, r.w + pad * 2),
              Math.max(0, r.h + pad * 2),
            );
          }
          state.markCtx.restore();
        }
        for (const stroke of (state.marks.strokes || [])) {
          const points = stroke.points || [];
          for (let i = 1; i < points.length; i++) {
            _annolidDrawStrokeSegment(
              state.markCtx,
              points[i - 1],
              points[i],
              stroke.tool,
              stroke.color,
              stroke.size,
            );
          }
        }
      }

      function _annolidGetSpanMeta(idx) {
        if (window.__annolidSpanMeta && window.__annolidSpanMeta[idx]) {
          const cached = window.__annolidSpanMeta[idx];
          if (cached && isFinite(cached.x) && isFinite(cached.y) && isFinite(cached.w) && isFinite(cached.h) && cached.w > 0 && cached.h > 0) {
            return cached;
          }
        }
        const spans = window.__annolidSpans || [];
        const span = spans[idx];
        if (!span) return null;
        const pageDiv = span.closest ? span.closest(".page") : null;
        if (!pageDiv) return null;
        const pageNumRaw = pageDiv.getAttribute("data-page-number") || "0";
        const pageNum = parseInt(pageNumRaw, 10) || 0;
        const pageRect = pageDiv.getBoundingClientRect();
        let spanRect = null;
        try { spanRect = span.getBoundingClientRect(); } catch (e) { spanRect = null; }
        if (!spanRect || !isFinite(spanRect.width) || !isFinite(spanRect.height) || spanRect.width <= 0 || spanRect.height <= 0) {
          try {
            const rects = span.getClientRects ? span.getClientRects() : [];
            for (let i = 0; i < rects.length; i++) {
              const r = rects[i];
              if (r && isFinite(r.width) && isFinite(r.height) && r.width > 0 && r.height > 0) {
                spanRect = r;
                break;
              }
            }
          } catch (e) {
            spanRect = null;
          }
        }
        if (!spanRect || !isFinite(spanRect.width) || !isFinite(spanRect.height) || spanRect.width <= 0 || spanRect.height <= 0) return null;
        const meta = {
          pageNum,
          x: spanRect.left - pageRect.left,
          y: spanRect.top - pageRect.top,
          w: spanRect.width,
          h: spanRect.height,
        };
        if (window.__annolidSpanMeta && isFinite(meta.w) && isFinite(meta.h) && meta.w > 0 && meta.h > 0) {
          window.__annolidSpanMeta[idx] = meta;
        }
        return meta;
      }

      function _annolidRenderTts() {
        const tts = window.__annolidTts || { sentenceIndices: [], wordIndex: null, lastPages: [] };
        const sentence = tts.sentenceIndices || [];
        const wordIndex = (tts.wordIndex === 0 || tts.wordIndex) ? tts.wordIndex : null;
        const pagesToDraw = {};
        function addMeta(kind, idx) {
          const meta = _annolidGetSpanMeta(idx);
          if (!meta) return;
          const key = String(meta.pageNum);
          if (!pagesToDraw[key]) pagesToDraw[key] = { sentence: [], word: null };
          if (kind === "sentence") pagesToDraw[key].sentence.push(meta);
          if (kind === "word") pagesToDraw[key].word = meta;
        }
        for (const idx of sentence) addMeta("sentence", idx);
        if (wordIndex !== null) addMeta("word", wordIndex);

        const pageKeys = new Set([...(tts.lastPages || []), ...Object.keys(pagesToDraw)]);
        for (const key of pageKeys) {
          const state = window.__annolidPages[key];
          if (!state || !state.ttsCanvas || !state.ttsCtx) continue;
          _annolidClearCanvas(state.ttsCanvas, state.ttsCtx, state.dpr || 1);
        }
        for (const key of Object.keys(pagesToDraw)) {
          const state = window.__annolidPages[key];
          if (!state || !state.ttsCanvas || !state.ttsCtx) continue;
          const entry = pagesToDraw[key];
          // Sentence highlight (soft)
          state.ttsCtx.save();
          state.ttsCtx.fillStyle = "rgba(255, 210, 80, 0.30)";
          for (const meta of (entry.sentence || [])) {
            const pad = 1.0;
            state.ttsCtx.fillRect(
              Math.max(0, meta.x - pad),
              Math.max(0, meta.y - pad),
              Math.max(0, meta.w + pad * 2),
              Math.max(0, meta.h + pad * 2),
            );
          }
          state.ttsCtx.restore();
          // Word highlight (strong)
          if (entry.word) {
            const meta = entry.word;
            state.ttsCtx.save();
            state.ttsCtx.fillStyle = "rgba(255, 140, 0, 0.55)";
            const pad = 1.6;
            state.ttsCtx.fillRect(
              Math.max(0, meta.x - pad),
              Math.max(0, meta.y - pad),
              Math.max(0, meta.w + pad * 2),
              Math.max(0, meta.h + pad * 2),
            );
            state.ttsCtx.restore();
          }
        }
        tts.lastPages = Object.keys(pagesToDraw);
        window.__annolidTts = tts;
      }

      window.__annolidRenderTts = _annolidRenderTts;
      window.__annolidClearSentenceHighlight = function () {
        if (!window.__annolidTts) window.__annolidTts = { sentenceIndices: [], wordIndex: null, lastPages: [] };
        window.__annolidTts.sentenceIndices = [];
        _annolidRenderTts();
      };
      window.__annolidClearWordHighlight = function () {
        if (!window.__annolidTts) window.__annolidTts = { sentenceIndices: [], wordIndex: null, lastPages: [] };
        window.__annolidTts.wordIndex = null;
        _annolidRenderTts();
      };
      window.__annolidClearHighlight = function () {
        if (!window.__annolidTts) window.__annolidTts = { sentenceIndices: [], wordIndex: null, lastPages: [] };
        window.__annolidTts.sentenceIndices = [];
        window.__annolidTts.wordIndex = null;
        _annolidRenderTts();
      };
      window.__annolidHighlightSentenceIndices = function (indices) {
        if (!window.__annolidTts) window.__annolidTts = { sentenceIndices: [], wordIndex: null, lastPages: [] };
        window.__annolidTts.sentenceIndices = (indices && indices.length) ? indices : [];
        window.__annolidTts.wordIndex = null;
        _annolidRenderTts();
      };
      window.__annolidHighlightWordIndex = function (idx) {
        if (!window.__annolidTts) window.__annolidTts = { sentenceIndices: [], wordIndex: null, lastPages: [] };
        window.__annolidTts.wordIndex = idx;
        _annolidRenderTts();
      };
      window.__annolidHighlightSelection = function () {
        const indices = window.__annolidSelectionSpans || [];
        window.__annolidHighlightSentenceIndices(indices);
      };
      window.__annolidHighlightParagraphIndices = function (indices) {
        window.__annolidHighlightSentenceIndices(indices);
      };
      window.__annolidSetReaderEnabled = function (enabled) {
        window.__annolidReaderEnabled = !!enabled;
        try {
          document.body.classList.toggle("annolid-reader-enabled", window.__annolidReaderEnabled);
        } catch (e) { }
      };
      window.__annolidScrollToPage = function (pageNum) {
        try {
          const n = parseInt(pageNum, 10) || 1;
          if (typeof _annolidGoToPage === "function") {
            _annolidGoToPage(n);
            return;
          }
        } catch (e) { }
        const container = document.getElementById("viewerContainer");
        if (!container) return;
        const target = document.querySelector(`.page[data-page-number='${pageNum}']`);
        if (!target) return;
        const offset = Math.max(0, target.offsetTop - 60);
        try {
          container.scrollTo({ top: offset, behavior: "smooth" });
        } catch (e) {
          container.scrollTop = offset;
        }
      };
      window.__annolidScrollToSentence = function (indices, pageNum) {
        try {
          const spans = Array.isArray(indices) ? indices : [];
          if (!spans.length) return;
          const container = document.getElementById("viewerContainer");
          if (!container) return;
          const requestedPage = (parseInt(pageNum || 0, 10) || 0);

          const now = Date.now();
          const key = String(requestedPage) + ":" + spans.slice(0, 10).join(",");
          const last = window.__annolidLastSentenceScroll || null;
          if (last && last.key === key && (now - (last.t || 0)) < 120) {
            return;
          }
          window.__annolidLastSentenceScroll = { key, t: now };

          const scrollBehavior = "smooth";
          const minMovePx = 18;
          const safeHeight = Math.max(1, container.clientHeight || 1);
          const comfortMargin = Math.min(220, Math.max(80, safeHeight * 0.25));

          const clampScroll = (value) => {
            const maxScroll = Math.max(0, (container.scrollHeight || 0) - safeHeight);
            return Math.max(0, Math.min(maxScroll, value));
          };

          const scrollToTop = (target) => {
            const clamped = clampScroll(target);
            if (Math.abs((container.scrollTop || 0) - clamped) < minMovePx) return;
            try {
              container.scrollTo({ top: clamped, behavior: scrollBehavior });
            } catch (e) {
              container.scrollTop = clamped;
            }
          };

          const scrollToPageCenter = (pageNumToUse) => {
            const page = document.querySelector(`.page[data-page-number='${pageNumToUse}']`);
            if (!page) return;
            const center = (page.offsetTop || 0) + (page.clientHeight || 0) / 2;
            scrollToTop(center - safeHeight / 2);
          };

          const groupMetasByPage = (metas) => {
            const groups = {};
            metas.forEach((m) => {
              const p = m && (m.pageNum || 0);
              if (!p) return;
              const k = String(p);
              if (!groups[k]) groups[k] = [];
              groups[k].push(m);
            });
            return groups;
          };

          const pickPageGroup = (groups) => {
            const keys = Object.keys(groups || {});
            if (!keys.length) return { pageNum: 0, metas: [] };
            if (requestedPage > 0 && groups[String(requestedPage)] && groups[String(requestedPage)].length) {
              return { pageNum: requestedPage, metas: groups[String(requestedPage)] };
            }
            let bestKey = keys[0];
            let bestCount = (groups[bestKey] || []).length;
            keys.forEach((k) => {
              const c = (groups[k] || []).length;
              if (c > bestCount) {
                bestCount = c;
                bestKey = k;
              }
            });
            return { pageNum: (parseInt(bestKey, 10) || 0), metas: groups[bestKey] || [] };
          };

          const computeSentenceBox = (metas, pageEl) => {
            if (!metas || !metas.length || !pageEl) return null;
            let minY = Infinity;
            let maxY = -Infinity;
            metas.forEach((m) => {
              const y0 = (m.y || 0);
              const y1 = (m.y || 0) + (m.h || 0);
              if (isFinite(y0)) minY = Math.min(minY, y0);
              if (isFinite(y1)) maxY = Math.max(maxY, y1);
            });
            if (!isFinite(minY) || !isFinite(maxY)) return null;
            const absTop = (pageEl.offsetTop || 0) + minY;
            const absBottom = (pageEl.offsetTop || 0) + maxY;
            const absCenter = (absTop + absBottom) / 2;
            return { absTop, absBottom, absCenter };
          };

          const isComfortablyVisible = (box) => {
            if (!box) return false;
            const topBound = (container.scrollTop || 0) + comfortMargin;
            const bottomBound = (container.scrollTop || 0) + safeHeight - comfortMargin;
            return box.absCenter >= topBound && box.absCenter <= bottomBound;
          };

          const ensureRenderedThroughIfNeeded = (pageNumToUse) => {
            if (pageNumToUse > 0 && typeof _annolidEnsureRenderedThrough === "function") {
              const lastEnsure = window.__annolidLastEnsurePage || 0;
              if (pageNumToUse > lastEnsure) {
                window.__annolidLastEnsurePage = pageNumToUse;
                return _annolidEnsureRenderedThrough(pageNumToUse);
              }
            }
            return Promise.resolve();
          };

          const runScrollAttempt = (attempt) => {
            const metas = spans
              .map((idx) => _annolidGetSpanMeta(idx))
              .filter((m) => m && Number.isFinite(m.y) && Number.isFinite(m.h) && (m.pageNum || 0) > 0);

            if (!metas.length) {
              if (requestedPage > 0) scrollToPageCenter(requestedPage);
              if (attempt < 8) requestAnimationFrame(() => runScrollAttempt(attempt + 1));
              return;
            }

            const groups = groupMetasByPage(metas);
            const picked = pickPageGroup(groups);
            const pageNumToUse = picked.pageNum || requestedPage || 0;
            const pageEl = pageNumToUse
              ? document.querySelector(`.page[data-page-number='${pageNumToUse}']`)
              : null;

            if (!pageEl) {
              if (pageNumToUse > 0 && typeof _annolidGoToPage === "function") {
                _annolidGoToPage(pageNumToUse);
              }
              if (attempt < 8) requestAnimationFrame(() => runScrollAttempt(attempt + 1));
              return;
            }

            const box = computeSentenceBox(picked.metas, pageEl);
            if (!box) {
              if (pageNumToUse > 0) scrollToPageCenter(pageNumToUse);
              return;
            }
            if (isComfortablyVisible(box)) return;
            scrollToTop(box.absCenter - safeHeight / 2);
          };

          const initialPage = requestedPage || 0;
          Promise.resolve(ensureRenderedThroughIfNeeded(initialPage))
            .then(() => requestAnimationFrame(() => runScrollAttempt(0)))
            .catch(() => requestAnimationFrame(() => runScrollAttempt(0)));
        } catch (e) { }
      };

      function _annolidSetTool(tool) {
        window.__annolidMarks.tool = tool;
        const drawing = tool !== "select";
        document.body.classList.toggle("annolid-drawing", drawing);
        if (drawing) {
          _annolidHideCitationPopover();
        }
        const btns = [
          ["select", document.getElementById("annolidToolSelect")],
          ["pen", document.getElementById("annolidToolPen")],
          ["highlighter", document.getElementById("annolidToolHighlighter")],
        ];
        btns.forEach(([name, el]) => {
          if (!el) return;
          if (name === tool) el.classList.add("annolid-active");
          else el.classList.remove("annolid-active");
        });
        const pages = window.__annolidPages || {};
        Object.keys(pages).forEach((key) => {
          const state = pages[key];
          if (!state || !state.markCanvas) return;
          state.markCanvas.style.pointerEvents = drawing ? "auto" : "none";
          state.markCanvas.style.cursor = drawing ? "crosshair" : "default";
        });
      }

      function _annolidBindMarkCanvas(state) {
        const canvas = state && state.markCanvas;
        if (!canvas || canvas.__annolidBound) return;
        canvas.__annolidBound = true;

        canvas.addEventListener("pointerdown", (ev) => {
          if ((window.__annolidMarks.tool || "select") === "select") return;
          if (ev.button !== 0) return;
          const pt = _annolidPointFromEvent(ev, canvas);
          const tool = window.__annolidMarks.tool;
          const stroke = {
            tool,
            color: window.__annolidMarks.color || "#ffb300",
            size: window.__annolidMarks.size || 10,
            points: [pt],
          };
          window.__annolidMarks.drawing = { pageNum: state.pageNum, stroke, pointerId: ev.pointerId };
          try { canvas.setPointerCapture(ev.pointerId); } catch (e) { }
          ev.preventDefault();
        });

        canvas.addEventListener("pointermove", (ev) => {
          const drawing = window.__annolidMarks.drawing;
          if (!drawing || drawing.pageNum !== state.pageNum) return;
          if (!state.markCtx) return;
          const stroke = drawing.stroke;
          const pts = stroke.points || [];
          const pt = _annolidPointFromEvent(ev, canvas);
          if (pts.length) {
            _annolidDrawStrokeSegment(state.markCtx, pts[pts.length - 1], pt, stroke.tool, stroke.color, stroke.size);
          }
          pts.push(pt);
          stroke.points = pts;
          drawing.stroke = stroke;
          window.__annolidMarks.drawing = drawing;
          ev.preventDefault();
        });

        function finishStroke(ev) {
          const drawing = window.__annolidMarks.drawing;
          if (!drawing || drawing.pageNum !== state.pageNum) return;
          const stroke = drawing.stroke;
          window.__annolidMarks.drawing = null;
          if (stroke && stroke.points && stroke.points.length > 1) {
            state.marks.strokes.push(stroke);
            window.__annolidMarks.undo.push({ type: "stroke", pageNum: state.pageNum });
            _annolidRenderMarks(state.pageNum);
            if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(600);
            try {
              const anchor = (typeof _annolidGetScrollAnchor === "function") ? _annolidGetScrollAnchor() : { pageNum: state.pageNum, offsetFrac: 0 };
              const bridge = window.__annolidBridge;
              if (bridge && typeof bridge.logEvent === "function") {
                bridge.logEvent({
                  type: "mark_stroke",
                  label: "Mark (stroke)",
                  pageNum: anchor.pageNum || state.pageNum,
                  offsetFrac: (typeof anchor.offsetFrac === "number" && isFinite(anchor.offsetFrac)) ? anchor.offsetFrac : 0,
                });
              }
            } catch (e) { }
          }
          try { canvas.releasePointerCapture(ev.pointerId); } catch (e) { }
          ev.preventDefault();
        }

        canvas.addEventListener("pointerup", finishStroke);
        canvas.addEventListener("pointercancel", finishStroke);
      }

      function _annolidManualHighlightSelection() {
        const indices = window.__annolidSelectionSpans || [];
        if (!indices.length) return;
        const grouped = {};
        for (const idx of indices) {
          const meta = _annolidGetSpanMeta(idx);
          if (!meta) continue;
          const key = String(meta.pageNum);
          if (!grouped[key]) grouped[key] = [];
          grouped[key].push({ x: meta.x, y: meta.y, w: meta.w, h: meta.h });
        }
        const color = window.__annolidMarks.color || "#ffb300";
        for (const key of Object.keys(grouped)) {
          const pageNum = parseInt(key, 10) || 0;
          const state = _annolidGetPageState(pageNum);
          state.marks.highlights.push({
            rects: grouped[key],
            color,
            alpha: 0.28,
          });
          window.__annolidMarks.undo.push({ type: "highlight", pageNum });
          _annolidRenderMarks(pageNum);
        }
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(650);
        try {
          const anchor = (typeof _annolidGetScrollAnchor === "function") ? _annolidGetScrollAnchor() : null;
          const bridge = window.__annolidBridge;
          if (bridge && typeof bridge.logEvent === "function") {
            bridge.logEvent({
              type: "mark_highlight",
              label: "Mark (highlight)",
              pageNum: anchor ? (anchor.pageNum || 1) : 1,
              offsetFrac: anchor ? (anchor.offsetFrac || 0) : 0,
            });
          }
        } catch (e) { }
      }

      function _annolidUndo() {
        const op = (window.__annolidMarks.undo || []).pop();
        if (!op) return;
        const state = _annolidGetPageState(op.pageNum);
        if (op.type === "stroke") state.marks.strokes.pop();
        if (op.type === "highlight") state.marks.highlights.pop();
        _annolidRenderMarks(op.pageNum);
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(650);
      }

      function _annolidClearMarks() {
        window.__annolidMarks.undo = [];
        const pages = window.__annolidPages || {};
        Object.keys(pages).forEach((key) => {
          const state = pages[key];
          if (!state || !state.marks) return;
          state.marks.strokes = [];
          state.marks.highlights = [];
          _annolidRenderMarks(state.pageNum);
        });
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(650);
      }

      // Toolbar bindings.
      const selectBtn = document.getElementById("annolidToolSelect");
      const penBtn = document.getElementById("annolidToolPen");
      const hiBtn = document.getElementById("annolidToolHighlighter");
      const highlightBtn = document.getElementById("annolidHighlightSelection");
      const undoBtn = document.getElementById("annolidUndo");
      const clearBtn = document.getElementById("annolidClear");
      const bookmarkBtn = document.getElementById("annolidBookmarkBtn");
      const notesBtn = document.getElementById("annolidNotesBtn");
      const notesModal = document.getElementById("annolidNotesModal");
      const notesSearch = document.getElementById("annolidNotesSearch");
      const bookmarksList = document.getElementById("annolidBookmarksList");
      const notesList = document.getElementById("annolidNotesList");
      const noteMeta = document.getElementById("annolidNoteMeta");
      const noteEditor = document.getElementById("annolidNoteEditor");
      const noteAddBtn = document.getElementById("annolidNoteAdd");
      const noteDeleteBtn = document.getElementById("annolidNoteDelete");
      const noteSaveBtn = document.getElementById("annolidNoteSave");
      const noteCloseBtn = document.getElementById("annolidNoteClose");
      const colorInput = document.getElementById("annolidColor");
      const sizeInput = document.getElementById("annolidSize");
      const colorInline = document.getElementById("annolidColorInline");
      const sizeInline = document.getElementById("annolidSizeInline");

      if (selectBtn) selectBtn.addEventListener("click", () => _annolidSetTool("select"));
      if (penBtn) penBtn.addEventListener("click", () => _annolidSetTool("pen"));
      if (hiBtn) hiBtn.addEventListener("click", () => _annolidSetTool("highlighter"));
      if (highlightBtn) highlightBtn.addEventListener("click", _annolidManualHighlightSelection);
      if (undoBtn) undoBtn.addEventListener("click", _annolidUndo);
      if (clearBtn) clearBtn.addEventListener("click", _annolidClearMarks);

      function _annolidSetStrokeColor(value) {
        const color = value || "#ffb300";
        const prev = window.__annolidMarks.color;
        window.__annolidMarks.color = color;
        if (colorInput && colorInput.value !== color) colorInput.value = color;
        if (colorInline && colorInline.value !== color) colorInline.value = color;
        if (prev !== color && window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(700);
      }

      function _annolidSetStrokeSize(value) {
        const v = parseFloat(value);
        const clamped = isFinite(v) ? Math.max(2, Math.min(24, v)) : 10;
        const prev = window.__annolidMarks.size;
        window.__annolidMarks.size = clamped;
        const str = String(clamped);
        if (sizeInput && sizeInput.value !== str) sizeInput.value = str;
        if (sizeInline && sizeInline.value !== str) sizeInline.value = str;
        if (prev !== clamped && window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(700);
      }

      if (colorInput) {
        colorInput.addEventListener("input", (ev) => {
          _annolidSetStrokeColor(ev.target.value);
        });
      }
      if (colorInline) {
        colorInline.addEventListener("input", (ev) => {
          _annolidSetStrokeColor(ev.target.value);
        });
      }
      if (sizeInput) {
        sizeInput.addEventListener("input", (ev) => {
          _annolidSetStrokeSize(ev.target.value);
        });
      }
      if (sizeInline) {
        sizeInline.addEventListener("input", (ev) => {
          _annolidSetStrokeSize(ev.target.value);
        });
      }

      _annolidSetStrokeColor(window.__annolidMarks.color || "#ffb300");
      _annolidSetStrokeSize(window.__annolidMarks.size || 10);

      _annolidSetTool("select");
      window.__annolidSetReaderEnabled(window.__annolidReaderEnabled);

      // ---- Bookmarks & notes ----------------------------------------------
      let _annolidActiveNoteId = null;

      function _annolidEnsureUserState() {
        if (!window.__annolidUserState || typeof window.__annolidUserState !== "object") {
          window.__annolidUserState = _annolidNormalizeUserState({});
        }
        if (!Array.isArray(window.__annolidUserState.bookmarks)) window.__annolidUserState.bookmarks = [];
        if (!Array.isArray(window.__annolidUserState.notes)) window.__annolidUserState.notes = [];
        return window.__annolidUserState;
      }

      function _annolidNowSec() {
        return annolidNowMs() / 1000.0;
      }

      function _annolidRandomId(prefix) {
        try {
          if (window.crypto && typeof window.crypto.randomUUID === "function") {
            return String(prefix || "id") + ":" + window.crypto.randomUUID();
          }
        } catch (e) { }
        return String(prefix || "id") + ":" + Math.random().toString(16).slice(2) + ":" + String(annolidNowMs());
      }

      function _annolidIsBookmarked(pageNum) {
        const state = _annolidEnsureUserState();
        const p = parseInt(pageNum, 10) || 1;
        return (state.bookmarks || []).some((b) => (parseInt(b.pageNum, 10) || 0) === p);
      }

      function _annolidUpdateBookmarkButton() {
        if (!bookmarkBtn) return;
        const current = (typeof _annolidGetCurrentPageNum === "function") ? _annolidGetCurrentPageNum() : 1;
        const active = _annolidIsBookmarked(current);
        bookmarkBtn.textContent = active ? "★" : "☆";
        bookmarkBtn.classList.toggle("annolid-active", active);
      }

      function _annolidToggleBookmark() {
        const state = _annolidEnsureUserState();
        const current = (typeof _annolidGetCurrentPageNum === "function") ? _annolidGetCurrentPageNum() : 1;
        const p = parseInt(current, 10) || 1;
        const existingIdx = (state.bookmarks || []).findIndex((b) => (parseInt(b.pageNum, 10) || 0) === p);
        if (existingIdx >= 0) {
          state.bookmarks.splice(existingIdx, 1);
        } else {
          state.bookmarks.push({
            pageNum: p,
            page: Math.max(0, p - 1),
            title: `Page ${p}`,
            createdAt: _annolidNowSec(),
          });
          state.bookmarks.sort((a, b) => (parseInt(a.pageNum, 10) || 0) - (parseInt(b.pageNum, 10) || 0));
        }
        window.__annolidUserState = state;
        _annolidUpdateBookmarkButton();
        _annolidUpdatePersistenceButtons();
        try {
          if (notesModal && notesModal.classList.contains("annolid-open")) _annolidRenderNotesUi();
        } catch (e) { }
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(500);
        try {
          const bridge = window.__annolidBridge;
          if (bridge && typeof bridge.logEvent === "function") {
            const kind = (existingIdx >= 0) ? "bookmark_remove" : "bookmark_add";
            const anchor = (typeof _annolidGetScrollAnchor === "function") ? _annolidGetScrollAnchor() : { pageNum: p, offsetFrac: 0 };
            bridge.logEvent({
              type: kind,
              label: (kind === "bookmark_add") ? "Bookmark added" : "Bookmark removed",
              pageNum: anchor.pageNum || p,
              offsetFrac: anchor.offsetFrac || 0,
            });
          }
        } catch (e) { }
      }

      function _annolidOpenNotesModal() {
        if (!notesModal) return;
        notesModal.classList.add("annolid-open");
        _annolidRenderNotesUi();
        try {
          if (notesSearch) notesSearch.focus();
        } catch (e) { }
      }

      function _annolidCloseNotesModal() {
        if (!notesModal) return;
        notesModal.classList.remove("annolid-open");
      }

      function _annolidSetActiveNote(noteId) {
        _annolidActiveNoteId = noteId ? String(noteId) : null;
        const state = _annolidEnsureUserState();
        const note = (state.notes || []).find((n) => String(n.id || "") === String(_annolidActiveNoteId || ""));
        if (noteEditor) {
          noteEditor.value = note ? String(note.text || "") : "";
        }
        if (noteMeta) {
          if (!note) {
            noteMeta.textContent = "Select a note to view/edit.";
          } else {
            const pageNum = parseInt(note.pageNum, 10) || 1;
            const snippet = String(note.snippet || "").trim();
            const created = note.createdAt ? new Date((Number(note.createdAt) || 0) * 1000).toLocaleString() : "";
            noteMeta.textContent = `Page ${pageNum}\n${created}\n\n${snippet || "(no selection)"}`;
          }
        }
        _annolidRenderNotesUi();
      }

      function _annolidJumpToPage(pageNum) {
        try {
          const p = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
          _annolidGoToPage(p);
        } catch (e) { }
      }

      function _annolidRenderNotesUi() {
        const state = _annolidEnsureUserState();
        const q = notesSearch ? String(notesSearch.value || "").trim().toLowerCase() : "";

        function matches(text) {
          if (!q) return true;
          return String(text || "").toLowerCase().includes(q);
        }

        if (bookmarksList) {
          bookmarksList.innerHTML = "";
          const list = (state.bookmarks || []).filter((b) => matches(b.title) || matches("page " + b.pageNum));
          if (!list.length) {
            const empty = document.createElement("div");
            empty.className = "annolid-muted";
            empty.textContent = "No bookmarks yet.";
            bookmarksList.appendChild(empty);
          } else {
            list.forEach((b) => {
              const btn = document.createElement("button");
              btn.className = "annolid-notes-item";
              const pageNum = parseInt(b.pageNum, 10) || 1;
              btn.innerHTML = `<div style="min-width:0"><div class="annolid-notes-item-title">★ ${_annolidNormalizeText(b.title || ("Page " + pageNum))}</div></div><div class="annolid-notes-item-meta">p${pageNum}</div>`;
              btn.addEventListener("click", () => _annolidJumpToPage(pageNum));
              bookmarksList.appendChild(btn);
            });
          }
        }

        if (notesList) {
          notesList.innerHTML = "";
          const list = (state.notes || []).filter((n) => matches(n.text) || matches(n.snippet) || matches("page " + n.pageNum));
          if (!list.length) {
            const empty = document.createElement("div");
            empty.className = "annolid-muted";
            empty.textContent = "No notes yet. Select text and click Add.";
            notesList.appendChild(empty);
          } else {
            list.forEach((n) => {
              const btn = document.createElement("button");
              const active = _annolidActiveNoteId && String(n.id || "") === String(_annolidActiveNoteId);
              btn.className = "annolid-notes-item" + (active ? " annolid-active" : "");
              const pageNum = parseInt(n.pageNum, 10) || 1;
              const title = _annolidNormalizeText((n.text || n.snippet || "").toString()).slice(0, 70) || "Note";
              const sub = _annolidNormalizeText((n.snippet || "").toString()).slice(0, 90);
              btn.innerHTML = `<div style="min-width:0"><div class="annolid-notes-item-title">${title}</div><div class="annolid-notes-item-sub">${sub}</div></div><div class="annolid-notes-item-meta">p${pageNum}</div>`;
              btn.addEventListener("click", () => {
                _annolidSetActiveNote(n.id);
                const frac = parseFloat(n.offsetFrac || 0) || 0;
                if (window.__annolidScrollToAnchor) {
                  window.__annolidScrollToAnchor(pageNum, frac);
                } else {
                  _annolidJumpToPage(pageNum);
                }
              });
              notesList.appendChild(btn);
            });
          }
        }

        if (noteDeleteBtn) {
          noteDeleteBtn.disabled = !_annolidActiveNoteId;
        }
        if (noteSaveBtn) {
          noteSaveBtn.disabled = !_annolidActiveNoteId;
        }
      }

      function _annolidAddNoteFromSelection() {
        const state = _annolidEnsureUserState();
        const anchor = (typeof _annolidGetScrollAnchor === "function") ? _annolidGetScrollAnchor() : { pageNum: (typeof _annolidGetCurrentPageNum === "function") ? _annolidGetCurrentPageNum() : 1, offsetFrac: 0 };
        const selectionText = _annolidNormalizeText(window.__annolidSelection || "");
        const id = _annolidRandomId("note");
        const createdAt = _annolidNowSec();
        const note = {
          id,
          pageNum: anchor.pageNum || 1,
          page: Math.max(0, (anchor.pageNum || 1) - 1),
          offsetFrac: (typeof anchor.offsetFrac === "number" && isFinite(anchor.offsetFrac)) ? anchor.offsetFrac : 0,
          snippet: selectionText.slice(0, 400),
          text: "",
          createdAt,
          updatedAt: createdAt,
        };

        // Optionally attach a light highlight to the selected text for context.
        try {
          const indices = window.__annolidSelectionSpans || [];
          if (indices && indices.length) {
            const grouped = {};
            for (const idx of indices) {
              const meta = _annolidGetSpanMeta(idx);
              if (!meta) continue;
              const key = String(meta.pageNum);
              if (!grouped[key]) grouped[key] = [];
              grouped[key].push({ x: meta.x, y: meta.y, w: meta.w, h: meta.h });
            }
            const color = window.__annolidMarks.color || "#ffb300";
            for (const key of Object.keys(grouped)) {
              const pageNum = parseInt(key, 10) || 0;
              if (!pageNum) continue;
              const st = _annolidGetPageState(pageNum);
              st.marks.highlights.push({
                id,
                kind: "note",
                rects: grouped[key],
                color,
                alpha: 0.18,
              });
              _annolidRenderMarks(pageNum);
            }
          }
        } catch (e) { }

        state.notes.unshift(note);
        window.__annolidUserState = state;
        _annolidUpdatePersistenceButtons();
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(450);
        try {
          const bridge = window.__annolidBridge;
          if (bridge && typeof bridge.logEvent === "function") {
            bridge.logEvent({
              type: "note_add",
              label: "Note added",
              noteId: id,
              pageNum: note.pageNum || 1,
              offsetFrac: note.offsetFrac || 0,
              snippet: note.snippet || "",
            });
          }
        } catch (e) { }
        _annolidOpenNotesModal();
        _annolidSetActiveNote(id);
        try {
          if (noteEditor) noteEditor.focus();
        } catch (e) { }
      }

      function _annolidSaveActiveNote() {
        const state = _annolidEnsureUserState();
        if (!_annolidActiveNoteId) return;
        const idx = (state.notes || []).findIndex((n) => String(n.id || "") === String(_annolidActiveNoteId));
        if (idx < 0) return;
        const note = state.notes[idx];
        note.text = noteEditor ? String(noteEditor.value || "") : String(note.text || "");
        note.updatedAt = _annolidNowSec();
        state.notes[idx] = note;
        window.__annolidUserState = state;
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(450);
        _annolidRenderNotesUi();
        try {
          const bridge = window.__annolidBridge;
          if (bridge && typeof bridge.logEvent === "function") {
            bridge.logEvent({
              type: "note_save",
              label: "Note saved",
              noteId: note.id || "",
              pageNum: parseInt(note.pageNum, 10) || 1,
              offsetFrac: parseFloat(note.offsetFrac || 0) || 0,
              snippet: note.snippet || "",
            });
          }
        } catch (e) { }
      }

      function _annolidDeleteActiveNote() {
        const state = _annolidEnsureUserState();
        if (!_annolidActiveNoteId) return;
        const id = String(_annolidActiveNoteId);
        state.notes = (state.notes || []).filter((n) => String(n.id || "") !== id);
        window.__annolidUserState = state;
        // Remove associated highlights.
        try {
          const pages = window.__annolidPages || {};
          Object.keys(pages).forEach((key) => {
            const st = pages[key];
            if (!st || !st.marks || !Array.isArray(st.marks.highlights)) return;
            st.marks.highlights = st.marks.highlights.filter((hl) => String(hl.id || "") !== id);
            _annolidRenderMarks(st.pageNum);
          });
        } catch (e) { }
        _annolidActiveNoteId = null;
        if (noteEditor) noteEditor.value = "";
        if (noteMeta) noteMeta.textContent = "Select a note to view/edit.";
        _annolidUpdatePersistenceButtons();
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(450);
        _annolidRenderNotesUi();
        try {
          const bridge = window.__annolidBridge;
          if (bridge && typeof bridge.logEvent === "function") {
            bridge.logEvent({
              type: "note_delete",
              label: "Note deleted",
              noteId: id,
            });
          }
        } catch (e) { }
      }

      window.__annolidOpenNotesAndSelect = function (noteId) {
        try {
          _annolidOpenNotesModal();
          if (noteId) {
            _annolidSetActiveNote(String(noteId));
          }
        } catch (e) { }
      };

      if (bookmarkBtn) bookmarkBtn.addEventListener("click", _annolidToggleBookmark);
      if (notesBtn) notesBtn.addEventListener("click", _annolidOpenNotesModal);
      if (noteCloseBtn) noteCloseBtn.addEventListener("click", _annolidCloseNotesModal);
      if (noteAddBtn) noteAddBtn.addEventListener("click", _annolidAddNoteFromSelection);
      if (noteSaveBtn) noteSaveBtn.addEventListener("click", _annolidSaveActiveNote);
      if (noteDeleteBtn) noteDeleteBtn.addEventListener("click", _annolidDeleteActiveNote);
      if (notesSearch) notesSearch.addEventListener("input", _annolidRenderNotesUi);
      if (noteEditor) noteEditor.addEventListener("input", () => {
        const state = _annolidEnsureUserState();
        if (!_annolidActiveNoteId) return;
        const idx = (state.notes || []).findIndex((n) => String(n.id || "") === String(_annolidActiveNoteId));
        if (idx < 0) return;
        const note = state.notes[idx];
        note.text = String(noteEditor.value || "");
        note.updatedAt = _annolidNowSec();
        state.notes[idx] = note;
        window.__annolidUserState = state;
        _annolidRenderNotesUi();
        if (window.__annolidRequestSaveUserState) window.__annolidRequestSaveUserState(900);
      });
      if (notesModal) {
        notesModal.addEventListener("click", (ev) => {
          // Only close on backdrop clicks for real modals (not popups).
          if (notesModal.classList && notesModal.classList.contains("annolid-modal")) {
            if (ev.target === notesModal) _annolidCloseNotesModal();
          }
        });
      }
      document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") {
          _annolidCloseNotesModal();
        }
      });
      _annolidUpdateBookmarkButton();

      function _annolidNormalizeText(text) {
        // Normalize whitespace and drop discretionary (soft) hyphens.
        return String(text || "").replace(/\u00ad/g, "").replace(/\s+/g, " ").trim();
      }

      function _annolidMergeHyphenated(prevText, nextText) {
        const prev = String(prevText || "");
        const next = String(nextText || "");
        const nextTrim = next.trimStart();
        if (!nextTrim) return _annolidNormalizeText(prev);
        const m = /([A-Za-z]{1,})[-\u2010\u2011\u2012\u2013]$/.exec(prev.trimEnd());
        if (!m) {
          return _annolidNormalizeText(prev + " " + nextTrim);
        }
        // Only treat as line-wrap continuation when the next token starts with lowercase.
        if (!/^[a-z]/.test(nextTrim)) {
          return _annolidNormalizeText(prev + " " + nextTrim);
        }
        // Prefer removing the hyphen when it is used only for line wrapping:
        // "pri-" + "marily" -> "primarily", "eluci-" + "date" -> "elucidate".
        // This may also turn "non-" + "linear" into "nonlinear", which is better for TTS.
        return _annolidNormalizeText(
          prev.trimEnd().replace(/[-\u2010\u2011\u2012\u2013]$/, "") + nextTrim
        );
      }

      function _annolidMedian(values) {
        if (!values || !values.length) return 0;
        const sorted = values.slice().sort((a, b) => a - b);
        return sorted[Math.floor(sorted.length / 2)];
      }

      function _annolidGroupIntoColumns(entries, pageWidth) {
        if (!entries || !entries.length) return [entries];
        const pw = isFinite(pageWidth) ? Math.max(1, pageWidth) : 1;
        // Use left edges for column clustering; centers can be noisy for long spans.
        const centers = entries.map((e) => e.x).filter((x) => isFinite(x));
        const sorted = centers.slice().sort((a, b) => a - b);
        if (sorted.length < 12) return [entries];

        const quantile = (arr, q) => {
          if (!arr.length) return 0;
          const pos = Math.max(0, Math.min(arr.length - 1, Math.round(q * (arr.length - 1))));
          return arr[pos];
        };
        const mean = (arr) => {
          if (!arr.length) return 0;
          let s = 0;
          for (let i = 0; i < arr.length; i++) s += arr[i];
          return s / arr.length;
        };
        const std = (arr, m) => {
          if (!arr.length) return 0;
          let s = 0;
          for (let i = 0; i < arr.length; i++) {
            const d = arr[i] - m;
            s += d * d;
          }
          return Math.sqrt(s / arr.length);
        };

        // Fast gap-based split (more robust than sepScore when indent noise is high).
        const q10 = quantile(sorted, 0.10);
        const q90 = quantile(sorted, 0.90);
        let bestGap = 0;
        let bestCut = null;
        for (let i = 0; i < sorted.length - 1; i++) {
          const a0 = sorted[i];
          const b0 = sorted[i + 1];
          if (a0 < q10 || b0 > q90) continue;
          const gap = b0 - a0;
          if (gap <= bestGap) continue;
          const cut = (a0 + b0) * 0.5;
          if (cut < pw * 0.30 || cut > pw * 0.70) continue;
          bestGap = gap;
          bestCut = cut;
        }
        const minGap = Math.max(24, pw * 0.06);
        if (bestCut != null && bestGap >= minGap) {
          const left = [];
          const right = [];
          entries.forEach((e) => {
            const x = e.x;
            if (!isFinite(x)) return;
            if (x <= bestCut) left.push(e);
            else right.push(e);
          });
          const minCount = Math.max(6, Math.floor(entries.length * 0.12));
          if (left.length >= minCount && right.length >= minCount) {
            return [left, right];
          }
        }

        // 1D k-means (k=2) to detect a stable two-column split.
        let m1 = quantile(sorted, 0.2);
        let m2 = quantile(sorted, 0.8);
        if (Math.abs(m2 - m1) < 1) return [entries];
        let a = [];
        let b = [];
        for (let iter = 0; iter < 10; iter++) {
          a = [];
          b = [];
          for (let i = 0; i < centers.length; i++) {
            const c = centers[i];
            if (Math.abs(c - m1) <= Math.abs(c - m2)) a.push(c);
            else b.push(c);
          }
          const next1 = mean(a);
          const next2 = mean(b);
          if (Math.abs(next1 - m1) < 0.5 && Math.abs(next2 - m2) < 0.5) break;
          m1 = next1;
          m2 = next2;
        }
        if (!a.length || !b.length) return [entries];
        const leftMean = Math.min(m1, m2);
        const rightMean = Math.max(m1, m2);
        const leftStd = std(a, m1);
        const rightStd = std(b, m2);
        const separation = rightMean - leftMean;

        const minSeparation = Math.max(24, pw * 0.08);
        const denom = Math.max(1e-6, leftStd + rightStd);
        const sepScore = separation / denom;
        if (separation < minSeparation) return [entries];
        if (sepScore < 1.6) return [entries];
        const cut = (leftMean + rightMean) * 0.5;
        if (cut < pw * 0.30 || cut > pw * 0.70) return [entries];

        const left = [];
        const right = [];
        entries.forEach((e) => {
          const c = e.x;
          if (Math.abs(c - leftMean) <= Math.abs(c - rightMean)) left.push(e);
          else right.push(e);
        });
        const minCount = Math.max(6, Math.floor(entries.length * 0.12));
        if (left.length < minCount || right.length < minCount) return [entries];
        return [left, right];
      }

      function _annolidGroupLinesIntoColumns(lines, pageWidth) {
        const list = Array.isArray(lines)
          ? lines.filter((l) => l && isFinite(l.xMin) && (isFinite(l.yCenter) || isFinite(l.yMin)))
          : [];
        if (!list.length) return [lines || []];
        const pw = isFinite(pageWidth) ? Math.max(1, pageWidth) : 1;

        const widthOf = (l) => {
          if (!l || !isFinite(l.xMin) || !isFinite(l.xMax)) return 0;
          return Math.max(0, l.xMax - l.xMin);
        };
        const heightOf = (l) => {
          if (!l) return 0;
          if (isFinite(l.h) && l.h > 0) return l.h;
          if (isFinite(l.yMax) && isFinite(l.yMin)) return Math.max(1, l.yMax - l.yMin);
          return 0;
        };
        const yCenterOf = (l) => {
          if (!l) return 0;
          if (isFinite(l.yCenter)) return l.yCenter;
          if (isFinite(l.yMin) && isFinite(l.yMax)) return (l.yMin + l.yMax) * 0.5;
          return 0;
        };
        const xCenterOf = (l) => {
          if (!l || !isFinite(l.xMin)) return 0;
          const w = widthOf(l);
          return l.xMin + w * 0.5;
        };

        const sortLines = (arr) => {
          const out = (arr || []).slice();
          const hs = out.map((l) => heightOf(l)).filter((h) => isFinite(h) && h > 0);
          const medH = _annolidMedian(hs);
          const sameRowTol = Math.max(2, (medH || 0) * 0.35);
          out.sort((a, b) => {
            const ay = yCenterOf(a);
            const by = yCenterOf(b);
            const dy = ay - by;
            if (Math.abs(dy) <= sameRowTol) {
              const ax = (a && isFinite(a.xMin)) ? a.xMin : 0;
              const bx = (b && isFinite(b.xMin)) ? b.xMin : 0;
              if (Math.abs(ax - bx) > 0.5) return ax - bx;
              return dy;
            }
            return dy;
          });
          return out;
        };

        // Exclude very wide lines (titles, centered headers) from column detection.
        const usable = list.filter((l) => {
          const w = widthOf(l);
          return w > 0 && w <= pw * 0.78;
        });
        const xsSource = (usable.length >= 6) ? usable : ((usable.length >= 4) ? usable : list);
        const xs = xsSource.map((l) => xCenterOf(l)).filter((x) => isFinite(x));
        if (xs.length < 6) return [sortLines(list)];
        xs.sort((a, b) => a - b);

        const q = (arr, t) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(t * (arr.length - 1))))];
        const mean = (arr) => {
          if (!arr.length) return 0;
          let s = 0;
          for (let i = 0; i < arr.length; i++) s += arr[i];
          return s / arr.length;
        };
        const std = (arr, m) => {
          if (!arr.length) return 0;
          let s = 0;
          for (let i = 0; i < arr.length; i++) {
            const d = arr[i] - m;
            s += d * d;
          }
          return Math.sqrt(s / arr.length);
        };

        const minCount = Math.max(3, Math.min(8, Math.floor(xsSource.length * 0.22)));

        function classifyByCut(cut, leftMean, rightMean) {
          const left = [];
          const right = [];
          const wide = [];
          list.forEach((l) => {
            if (!l) return;
            const w = widthOf(l);
            // Keep wide lines with the left column so headings stay before body text.
            if (w > pw * 0.85) {
              wide.push(l);
              return;
            }
            const cx = xCenterOf(l);
            if (!isFinite(cx)) return;
            if (leftMean != null && rightMean != null) {
              if (Math.abs(cx - leftMean) <= Math.abs(cx - rightMean)) left.push(l);
              else right.push(l);
              return;
            }
            if (cx <= cut) left.push(l);
            else right.push(l);
          });
          if (left.length >= minCount && right.length >= minCount) {
            // Guard against false "two columns" caused by centered/indented short lines
            // (common in novels/poetry), which makes reading appear to skip rows.
            const robustBoundsOf = (arr) => {
              const xMins = [];
              const xMaxs = [];
              const yMins = [];
              const yMaxs = [];
              for (const it of (arr || [])) {
                if (!it) continue;
                if (isFinite(it.xMin)) xMins.push(it.xMin);
                if (isFinite(it.xMax)) xMaxs.push(it.xMax);
                if (isFinite(it.yMin)) yMins.push(it.yMin);
                if (isFinite(it.yMax)) yMaxs.push(it.yMax);
              }
              const pick = (xs, t, fallback) => {
                if (!xs.length) return fallback;
                xs.sort((a, b) => a - b);
                return xs[Math.max(0, Math.min(xs.length - 1, Math.round(t * (xs.length - 1))))];
              };
              let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
              if (xMins.length && xMaxs.length) {
                x0 = pick(xMins, 0.10, x0);
                x1 = pick(xMaxs, 0.90, x1);
              }
              if (yMins.length && yMaxs.length) {
                y0 = pick(yMins, 0.10, y0);
                y1 = pick(yMaxs, 0.90, y1);
              }
              return { x0, x1, y0, y1 };
            };
            const allB = robustBoundsOf(list);
            // Ignore full-width lines when estimating column overlap/coverage.
            const lB = robustBoundsOf(left);
            const rB = robustBoundsOf(right);
            const totalY = Math.max(1, (allB.y1 - allB.y0) || 0);
            const coverL = ((lB.y1 - lB.y0) || 0) / totalY;
            const coverR = ((rB.y1 - rB.y0) || 0) / totalY;
            const overlapX = Math.min(lB.x1, rB.x1) - Math.max(lB.x0, rB.x0);
            const wL = Math.max(1, (lB.x1 - lB.x0) || 0);
            const wR = Math.max(1, (rB.x1 - rB.x0) || 0);
            const overlapFrac = (overlapX > 0) ? (overlapX / Math.max(1, Math.min(wL, wR))) : 0;
            // If x ranges overlap substantially, or one side covers only a small vertical slice,
            // treat as single-column.
            if (overlapFrac > 0.30 || Math.min(coverL, coverR) < 0.55) {
              return null;
            }
            // Put full-width headings into the left column so they are read first,
            // but don't let them break column detection.
            wide.forEach((l) => left.push(l));
            return [sortLines(left), sortLines(right)];
          }
          return null;
        }

        // Fast gap-based split on x centers.
        const q10 = q(xs, 0.10);
        const q90 = q(xs, 0.90);
        let bestGap = 0;
        let bestCut = null;
        for (let i = 0; i < xs.length - 1; i++) {
          const a0 = xs[i];
          const b0 = xs[i + 1];
          if (a0 < q10 || b0 > q90) continue;
          const gap = b0 - a0;
          if (gap <= bestGap) continue;
          const cut = (a0 + b0) * 0.5;
          if (cut < pw * 0.35 || cut > pw * 0.65) continue;
          bestGap = gap;
          bestCut = cut;
        }
        const minGap = Math.max(28, pw * 0.10);
        if (bestCut != null && bestGap >= minGap) {
          const res = classifyByCut(bestCut, null, null);
          if (res) return res;
        }

        // 1D k-means (k=2) fallback for indented/variable line starts.
        let m1 = q(xs, 0.25);
        let m2 = q(xs, 0.75);
        if (Math.abs(m2 - m1) < 1) return [sortLines(list)];
        let a = [];
        let b = [];
        for (let iter = 0; iter < 10; iter++) {
          a = [];
          b = [];
          for (let i = 0; i < xs.length; i++) {
            const x = xs[i];
            if (Math.abs(x - m1) <= Math.abs(x - m2)) a.push(x);
            else b.push(x);
          }
          const nm1 = mean(a);
          const nm2 = mean(b);
          if (Math.abs(nm1 - m1) < 0.5 && Math.abs(nm2 - m2) < 0.5) break;
          m1 = nm1;
          m2 = nm2;
        }
        if (!a.length || !b.length) return [sortLines(list)];
        const leftMean = Math.min(m1, m2);
        const rightMean = Math.max(m1, m2);
        const separation = rightMean - leftMean;
        const minSeparation = Math.max(48, pw * 0.18);
        if (separation < minSeparation) return [sortLines(list)];
        const score = separation / Math.max(1e-6, std(a, m1) + std(b, m2));
        if (score < 1.25) return [sortLines(list)];
        const cut = (leftMean + rightMean) * 0.5;
        if (cut < pw * 0.35 || cut > pw * 0.65) return [sortLines(list)];
        const res = classifyByCut(cut, leftMean, rightMean);
        if (res) return res;
        return [sortLines(list)];
      }

      function _annolidBuildLinesFromEntries(entries) {
        const list = Array.isArray(entries) ? entries.filter((e) => e && isFinite(e.x) && isFinite(e.y)) : [];
        if (!list.length) return [];
        const typicalH = _annolidMedian(list.map((e) => e.h).filter((h) => isFinite(h) && h > 0));
        // Use a slightly larger tolerance than the raw bbox height since some PDFs
        // jitter punctuation/quotes vertically (e.g. around "。”), which can split lines.
        const yTol = Math.max(2, (typicalH || 0) * 1.05);
        // Prevent merging different columns into the same "line" when y matches.
        // This is essential for correct two-column reading/highlighting.
        let minX = Infinity;
        let maxX = -Infinity;
        for (const e of list) {
          const x0 = e.x;
          const x1 = e.x + (e.w || 0);
          if (isFinite(x0)) minX = Math.min(minX, x0);
          if (isFinite(x1)) maxX = Math.max(maxX, x1);
        }
        const spanWidth = (isFinite(minX) && isFinite(maxX) && maxX > minX) ? (maxX - minX) : 0;
        // Use a conservative horizontal gap threshold so left/right columns never merge.
        // (~2% of page text width, clamped)
        let joinGapX = Math.max(10, Math.min(28, spanWidth * 0.02));
        // Optional column cut hint from a max-gap in x positions (helps when some spans have
        // oversized bounding boxes that reduce the apparent inter-column gap).
        let columnCut = null;
        try {
          const xs = list.map((e) => e.x).filter((x) => isFinite(x)).sort((a, b) => a - b);
          if (xs.length >= 32 && spanWidth > 0) {
            const q = (arr, t) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(t * (arr.length - 1))))];
            const q10 = q(xs, 0.10);
            const q90 = q(xs, 0.90);
            let bestGap = 0;
            let bestCut = null;
            for (let i = 0; i < xs.length - 1; i++) {
              const a0 = xs[i];
              const b0 = xs[i + 1];
              if (a0 < q10 || b0 > q90) continue;
              const gap = b0 - a0;
              if (gap <= bestGap) continue;
              const cut = (a0 + b0) * 0.5;
              const frac = (cut - minX) / spanWidth;
              if (!isFinite(frac) || frac < 0.30 || frac > 0.70) continue;
              bestGap = gap;
              bestCut = cut;
            }
            const minGap = Math.max(14, spanWidth * 0.025);
            if (bestCut != null && bestGap >= minGap) {
              columnCut = bestCut;
              // Ensure our join threshold is comfortably smaller than the inferred column gap.
              joinGapX = Math.min(joinGapX, Math.max(8, bestGap * 0.45));
            }
          }
        } catch (e) {
          columnCut = null;
        }
        const withCenter = list.map((e) => {
          const h = isFinite(e.h) ? e.h : 0;
          const yCenter = e.y + h * 0.5;
          return Object.assign({}, e, { yCenter });
        });
        withCenter.sort((a, b) => {
          const dy = a.yCenter - b.yCenter;
          if (Math.abs(dy) < Math.max(0.5, yTol * 0.15)) return a.x - b.x;
          return dy;
        });

        const lines = [];
        for (const entry of withCenter) {
          const y = entry.yCenter;
          let best = null;
          let bestOverlap = 0;
          let bestDist = Infinity;
          const scanStart = Math.max(0, lines.length - 6);
          for (let i = lines.length - 1; i >= scanStart; i--) {
            const cand = lines[i];
            if (!cand) continue;
            const dist = Math.abs(y - cand.yCenter);
            const overlap = Math.min(entry.y + entry.h, cand.yMax) - Math.max(entry.y, cand.yMin);
            const candH = Math.max(1, cand.yMax - cand.yMin);
            const minH = Math.max(1, Math.min(entry.h || candH, candH));
            const overlapRatio = (overlap > 0) ? (overlap / minH) : 0;
            const gapX = (() => {
              const eLeft = entry.x;
              const eRight = entry.x + (entry.w || 0);
              const cLeft = cand.xMin;
              const cRight = cand.xMax;
              if (eRight < cLeft) return cLeft - eRight;
              if (eLeft > cRight) return eLeft - cRight;
              return 0;
            })();
            // Hard guard against cross-column merges when a column split is detected.
            if (columnCut != null) {
              const margin = Math.max(4, (typicalH || 0) * 0.35);
              const candIsLeft = (cand.xMax <= (columnCut - margin));
              const candIsRight = (cand.xMin >= (columnCut + margin));
              if (candIsLeft || candIsRight) {
                const entryIsLeft = entry.x <= columnCut;
                if ((candIsLeft && !entryIsLeft) || (candIsRight && entryIsLeft)) {
                  continue;
                }
              }
            }
            if ((dist <= yTol || overlapRatio >= 0.65) && gapX <= joinGapX) {
              if (
                overlapRatio > bestOverlap + 0.05 ||
                (Math.abs(overlapRatio - bestOverlap) <= 0.05 && dist < bestDist)
              ) {
                best = cand;
                bestOverlap = overlapRatio;
                bestDist = dist;
              }
            }
          }
          let line = best;
          if (!line) {
            line = {
              yCenter: y,
              items: [],
              spans: [],
              texts: [],
              yMin: entry.y,
              yMax: entry.y + entry.h,
              xMin: entry.x,
              xMax: entry.x + entry.w,
            };
            lines.push(line);
          }
          line.items.push(entry);
          const n = line.items.length;
          line.yCenter = (line.yCenter * (n - 1) + y) / n;
          line.yMin = Math.min(line.yMin, entry.y);
          line.yMax = Math.max(line.yMax, entry.y + entry.h);
          line.xMin = Math.min(line.xMin, entry.x);
          line.xMax = Math.max(line.xMax, entry.x + entry.w);
        }

        function _annolidLineFromItems(items) {
          const out = {
            yCenter: 0,
            spans: [],
            texts: [],
            yMin: Infinity,
            yMax: -Infinity,
            xMin: Infinity,
            xMax: -Infinity,
            h: 1,
          };
          let n = 0;
          for (const it of items) {
            if (!it) continue;
            out.spans.push(it.idx);
            out.texts.push(it.text);
            out.yMin = Math.min(out.yMin, it.y);
            out.yMax = Math.max(out.yMax, it.y + (it.h || 0));
            out.xMin = Math.min(out.xMin, it.x);
            out.xMax = Math.max(out.xMax, it.x + (it.w || 0));
            out.yCenter += (it.yCenter || (it.y + (it.h || 0) * 0.5));
            n += 1;
          }
          if (!n) {
            out.yCenter = 0;
            out.yMin = 0;
            out.yMax = 1;
            out.xMin = 0;
            out.xMax = 1;
            out.h = 1;
            return out;
          }
          out.yCenter = out.yCenter / n;
          out.h = Math.max(1, out.yMax - out.yMin);
          return out;
        }

        const outLines = [];
        const splitGapX = Math.max(joinGapX * 1.6, Math.max(24, spanWidth * 0.04));
        for (const line of lines) {
          const items = Array.isArray(line.items) ? line.items.slice() : [];
          items.sort((a, b) => a.x - b.x);
          let bestSplit = -1;
          let bestGap = 0;
          for (let i = 0; i < items.length - 1; i++) {
            const a0 = items[i];
            const b0 = items[i + 1];
            const gap = (b0.x - (a0.x + (a0.w || 0)));
            if (gap > bestGap) {
              bestGap = gap;
              bestSplit = i;
            }
          }
          if (bestSplit >= 0 && bestGap >= splitGapX) {
            const leftItems = items.slice(0, bestSplit + 1);
            const rightItems = items.slice(bestSplit + 1);
            outLines.push(_annolidLineFromItems(leftItems));
            outLines.push(_annolidLineFromItems(rightItems));
            continue;
          }
          const out = _annolidLineFromItems(items);
          outLines.push(out);
        }

        const outHeights = outLines
          .map((l) => (l && isFinite(l.h)) ? l.h : 0)
          .filter((h) => isFinite(h) && h > 0);
        const outMedianH = _annolidMedian(outHeights);
        const sameRowTol = Math.max(2, (outMedianH || typicalH || 0) * 0.35);
        outLines.sort((a, b) => {
          const ay = (a && isFinite(a.yCenter)) ? a.yCenter : ((a && isFinite(a.yMin) && isFinite(a.yMax)) ? (a.yMin + a.yMax) * 0.5 : 0);
          const by = (b && isFinite(b.yCenter)) ? b.yCenter : ((b && isFinite(b.yMin) && isFinite(b.yMax)) ? (b.yMin + b.yMax) * 0.5 : 0);
          const dy = ay - by;
          if (Math.abs(dy) <= sameRowTol) {
            const ax = (a && isFinite(a.xMin)) ? a.xMin : 0;
            const bx = (b && isFinite(b.xMin)) ? b.xMin : 0;
            if (Math.abs(ax - bx) > 0.5) return ax - bx;
            return dy;
          }
          return dy;
        });
        return outLines;
      }

      function _annolidLinesToParagraphs(lines, pageNum) {
        const paragraphs = [];
        let current = null;
        lines.forEach((line) => {
          const lineText = _annolidNormalizeText(line.texts.join(" "));
          if (!lineText) return;
          if (!current) {
            current = {
              pageNum,
              text: lineText,
              spans: [].concat(line.spans),
              yMin: line.yMin,
              yMax: line.yMax,
              xMin: line.xMin,
              xMax: line.xMax,
              _lastXMin: line.xMin,
              _lastXMax: line.xMax,
              _lastYMin: line.yMin,
              _lastYMax: line.yMax,
            };
            return;
          }
          const gap = line.yMin - current.yMax;
          const gapLimit = Math.max(8, (current.yMax - current.yMin) * 0.9);
          // If two lines share the same row baseline but have disjoint x ranges,
          // avoid merging them into one paragraph (common when column detection fails).
          try {
            const lastXMin = current._lastXMin;
            const lastXMax = current._lastXMax;
            const lastYMin = current._lastYMin;
            const lastYMax = current._lastYMax;
            if (
              isFinite(lastXMin) && isFinite(lastXMax) &&
              isFinite(lastYMin) && isFinite(lastYMax) &&
              isFinite(line.xMin) && isFinite(line.xMax) &&
              isFinite(line.yMin) && isFinite(line.yMax)
            ) {
              const overlapY = Math.min(line.yMax, lastYMax) - Math.max(line.yMin, lastYMin);
              const h1 = Math.max(1, lastYMax - lastYMin);
              const h2 = Math.max(1, line.yMax - line.yMin);
              const overlapYFrac = (overlapY > 0) ? (overlapY / Math.min(h1, h2)) : 0;
              const overlapX = Math.min(line.xMax, lastXMax) - Math.max(line.xMin, lastXMin);
              if (overlapYFrac >= 0.70 && overlapX <= 0 && gap <= gapLimit) {
                paragraphs.push(current);
                current = {
                  pageNum,
                  text: lineText,
                  spans: [].concat(line.spans),
                  yMin: line.yMin,
                  yMax: line.yMax,
                  xMin: line.xMin,
                  xMax: line.xMax,
                  _lastXMin: line.xMin,
                  _lastXMax: line.xMax,
                  _lastYMin: line.yMin,
                  _lastYMax: line.yMax,
                };
                return;
              }
            }
          } catch (e) { }
          if (gap > gapLimit) {
            paragraphs.push(current);
            current = {
              pageNum,
              text: lineText,
              spans: [].concat(line.spans),
              yMin: line.yMin,
              yMax: line.yMax,
              xMin: line.xMin,
              xMax: line.xMax,
              _lastXMin: line.xMin,
              _lastXMax: line.xMax,
              _lastYMin: line.yMin,
              _lastYMax: line.yMax,
            };
          } else {
            current.text = _annolidMergeHyphenated(current.text, lineText);
            current.spans = current.spans.concat(line.spans);
            current.yMax = Math.max(current.yMax, line.yMax);
            current.xMin = Math.min(current.xMin, line.xMin);
            current.xMax = Math.max(current.xMax, line.xMax);
            current._lastXMin = line.xMin;
            current._lastXMax = line.xMax;
            current._lastYMin = line.yMin;
            current._lastYMax = line.yMax;
          }
        });
        if (current) paragraphs.push(current);
        // Remove internal tracking fields.
        paragraphs.forEach((p) => {
          try {
            delete p._lastXMin;
            delete p._lastXMax;
            delete p._lastYMin;
            delete p._lastYMax;
          } catch (e) { }
        });
        return paragraphs;
      }

      function _annolidBuildParagraphsForPage(pageNum) {
        const state = _annolidGetPageState(pageNum);
        if (!state || !state.pageDiv) return [];
        const pageDiv = state.pageDiv;
        const spans = Array.from(pageDiv.querySelectorAll(".textLayer span"));
        if (!spans.length) return [];
        const pageRect = pageDiv.getBoundingClientRect();
        const entries = [];
        spans.forEach((span) => {
          const text = _annolidNormalizeText(span.textContent || "");
          if (!text) return;
          const rect = span.getBoundingClientRect();
          const idx = parseInt(span.dataset.annolidIndex || "-1", 10);
          entries.push({
            idx,
            text,
            x: rect.left - pageRect.left,
            y: rect.top - pageRect.top,
            w: rect.width,
            h: rect.height,
          });
        });
        if (!entries.length) return [];
        const pageWidth = Math.max(1, pageRect.width || 1);
        // Build lines first, then detect columns using line starts (xMin).
        // Detecting columns from *per-span* x positions can wrongly split within a line,
        // which is a common root cause of "skipping" right after punctuation like "。”.
        const rawLines = _annolidBuildLinesFromEntries(entries);
        const colLines = _annolidGroupLinesIntoColumns(rawLines, pageWidth);
        // Ensure columns are ordered left->right (median xMin).
        colLines.sort((a, b) => {
          const ax = _annolidMedian((a || []).map((l) => l.xMin).filter((x) => isFinite(x)));
          const bx = _annolidMedian((b || []).map((l) => l.xMin).filter((x) => isFinite(x)));
          return ax - bx;
        });

        const paragraphs = [];
        colLines.forEach((lines) => {
          const colParas = _annolidLinesToParagraphs(lines, pageNum);
          colParas.forEach((p) => paragraphs.push(p));
        });

        window.__annolidParagraphsByPage[String(pageNum)] = paragraphs;
        return paragraphs;
      }

      function _annolidRebuildParagraphIndex() {
        const totalPages = window.__annolidTotalPages || 0;
        const paragraphs = [];
        const offsets = {};
        let offset = 0;
        for (let p = 1; p <= totalPages; p++) {
          offsets[p] = offset;
          const pageList = window.__annolidParagraphsByPage[String(p)] || [];
          pageList.forEach((para) => paragraphs.push(para));
          offset += pageList.length;
        }
        window.__annolidParagraphs = paragraphs;
        window.__annolidParagraphOffsets = offsets;
        window.__annolidParagraphTotal = offset;
      }

      function _annolidFindParagraphIndexBySpan(pageNum, spanIdx) {
        const list = window.__annolidParagraphsByPage[String(pageNum)] || [];
        for (let i = 0; i < list.length; i++) {
          const spans = list[i].spans || [];
          if (spans.indexOf(spanIdx) >= 0) return i;
        }
        return -1;
      }

      function _annolidInferColumnCutFromParagraphs(paras, pageWidth) {
        if (!Array.isArray(paras) || paras.length < 10) return null;
        if (!isFinite(pageWidth) || pageWidth <= 1) return null;
        const wideLimit = pageWidth * 0.82;
        const leftXMax = [];
        const rightXMin = [];
        for (const p of paras) {
          if (!p) continue;
          const xMin = (p.xMin == null) ? NaN : p.xMin;
          const xMax = (p.xMax == null) ? NaN : p.xMax;
          if (!isFinite(xMin) || !isFinite(xMax) || xMax <= xMin) continue;
          const w = xMax - xMin;
          // Ignore full-width blocks (titles, figures, centered headers).
          if (w >= wideLimit) continue;
          const c = (xMin + xMax) * 0.5;
          if (!isFinite(c)) continue;
          if (c < pageWidth * 0.5) leftXMax.push(xMax);
          else rightXMin.push(xMin);
        }
        if (leftXMax.length < 3 || rightXMin.length < 3) return null;
        leftXMax.sort((a, b) => a - b);
        rightXMin.sort((a, b) => a - b);
        // Robust quantiles to reduce outlier influence (indentation, footnotes).
        const q = (arr, t) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(t * (arr.length - 1))))];
        const leftEdge = q(leftXMax, 0.70);
        const rightEdge = q(rightXMin, 0.30);
        if (!isFinite(leftEdge) || !isFinite(rightEdge) || rightEdge <= leftEdge) return null;
        const gap = rightEdge - leftEdge;
        const minGap = Math.max(26, pageWidth * 0.06);
        if (gap < minGap) return null;
        const cut = (leftEdge + rightEdge) * 0.5;
        // Two-column cut is usually near the center; keep this conservative.
        if (cut < pageWidth * 0.35 || cut > pageWidth * 0.65) return null;
        return cut;
      }

      function _annolidFindParagraphIndexByPoint(pageNum, x, y) {
        const list = window.__annolidParagraphsByPage[String(pageNum)] || [];
        let pageWidth = null;
        try {
          const state = _annolidGetPageState(pageNum);
          if (state && state.pageDiv && state.pageDiv.getBoundingClientRect) {
            const r = state.pageDiv.getBoundingClientRect();
            if (r && isFinite(r.width) && r.width > 1) pageWidth = r.width;
          }
        } catch (e) { }
        if (!isFinite(pageWidth) || pageWidth <= 1) {
          let maxX = 0;
          for (const p of list) {
            if (p && isFinite(p.xMax)) maxX = Math.max(maxX, p.xMax);
          }
          pageWidth = Math.max(1, maxX || 1);
        }
        const columnCut = _annolidInferColumnCutFromParagraphs(list, pageWidth);
        const clickIsLeft = (columnCut != null) ? (x < columnCut) : null;
        const wideLimit = (isFinite(pageWidth) && pageWidth > 1) ? (pageWidth * 0.82) : Infinity;
        let candidateIdxs = null;
        if (columnCut != null && clickIsLeft != null) {
          candidateIdxs = [];
          for (let i = 0; i < list.length; i++) {
            const para = list[i];
            if (!para) continue;
            const xMin0 = (para.xMin == null) ? NaN : para.xMin;
            const xMax0 = (para.xMax == null) ? NaN : para.xMax;
            if (!isFinite(xMin0) || !isFinite(xMax0) || xMax0 <= xMin0) continue;
            const w = xMax0 - xMin0;
            const c = (xMin0 + xMax0) * 0.5;
            const isWide = (w >= wideLimit);
            const sameSide = isWide || ((clickIsLeft && c < columnCut) || (!clickIsLeft && c >= columnCut));
            if (sameSide) candidateIdxs.push(i);
          }
          if (candidateIdxs.length < 3) candidateIdxs = null;
        }
        let best = -1;
        let bestDist = Infinity;
        const scan = (candidateIdxs && candidateIdxs.length) ? candidateIdxs : null;
        const total = scan ? scan.length : list.length;
        for (let k = 0; k < total; k++) {
          const i = scan ? scan[k] : k;
          const para = list[i];
          if (para.yMin == null || para.yMax == null) continue;
          const xMin = (para.xMin == null) ? -Infinity : para.xMin;
          const xMax = (para.xMax == null) ? Infinity : para.xMax;
          const yMin = para.yMin;
          const yMax = para.yMax;
          const dx = (x < xMin) ? (xMin - x) : ((x > xMax) ? (x - xMax) : 0);
          const dy = (y < yMin) ? (yMin - y) : ((y > yMax) ? (y - yMax) : 0);
          if (dx === 0 && dy === 0) return i;
          const dist = dx + dy;
          if (dist < bestDist) {
            best = i;
            bestDist = dist;
          }
        }
        return best;
      }

      async function _annolidBuildTextParagraphsForPage(pageNum) {
        if (!window.__annolidPdf) return;
        if (window.__annolidParagraphsByPage[String(pageNum)]) return;
        const page = await window.__annolidPdf.getPage(pageNum);
        const textContent = await page.getTextContent();
        const viewport1 = page.getViewport({ scale: 1, rotation: 0 });
        const linesRaw = _annolidExtractLinesFromTextContent(textContent);
        const ordered = _annolidOrderLinesForReading(linesRaw, viewport1.width || 1);
        const paragraphs = [];
        let current = null;
        let lastYMin = null;
        let lastYMax = null;
        for (const line of ordered) {
          const lineText = _annolidNormalizeText(line.text || "");
          if (!lineText) continue;
          if (!current) {
            current = {
              pageNum,
              text: lineText,
              spans: [],
              yMin: line.yMin,
              yMax: line.yMax,
              xMin: line.xMin,
              xMax: line.xMax,
            };
            lastYMin = line.yMin;
            lastYMax = line.yMax;
            continue;
          }
          const gap = (lastYMin != null) ? (lastYMin - line.yMax) : 0;
          const lastH = (lastYMax != null && lastYMin != null) ? Math.abs(lastYMax - lastYMin) : 0;
          const gapLimit = Math.max(4, lastH * 1.15);
          if (gap > gapLimit) {
            paragraphs.push(current);
            current = {
              pageNum,
              text: lineText,
              spans: [],
              yMin: line.yMin,
              yMax: line.yMax,
              xMin: line.xMin,
              xMax: line.xMax,
            };
          } else {
            current.text = _annolidMergeHyphenated(current.text, lineText);
            current.yMin = Math.min(current.yMin, line.yMin);
            current.yMax = Math.max(current.yMax, line.yMax);
            current.xMin = Math.min(current.xMin, line.xMin);
            current.xMax = Math.max(current.xMax, line.xMax);
          }
          lastYMin = line.yMin;
          lastYMax = line.yMax;
        }
        if (current) paragraphs.push(current);
        window.__annolidParagraphsByPage[String(pageNum)] = paragraphs;
      }

      async function _annolidEnsureParagraphsFrom(pageNum) {
        const totalPages = window.__annolidTotalPages || 0;
        for (let p = pageNum; p <= totalPages; p++) {
          if (!window.__annolidParagraphsByPage[String(p)]) {
            await _annolidBuildTextParagraphsForPage(p);
            await new Promise(r => setTimeout(r, 0));
          }
        }
        _annolidRebuildParagraphIndex();
      }

      document.addEventListener("selectionchange", () => {
        try {
          const sel = window.getSelection ? window.getSelection() : null;
          window.__annolidSelection = sel ? sel.toString() : "";
          window.__annolidSelectionSpans = [];
          if (sel && !sel.isCollapsed && window.__annolidSpans.length) {
            const ranges = [];
            for (let i = 0; i < sel.rangeCount; i++) {
              ranges.push(sel.getRangeAt(i));
            }
            window.__annolidSpans.forEach((span, idx) => {
              for (const range of ranges) {
                if (range.intersectsNode(span)) {
                  window.__annolidSelectionSpans.push(idx);
                  break;
                }
              }
            });
          }
        } catch (e) {
          window.__annolidSelection = "";
          window.__annolidSelectionSpans = [];
        }
      });
      let loadingTask = null;
      if (pdfBase64 && pdfBase64.length > 0) {
        const raw = atob(pdfBase64);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) {
          bytes[i] = raw.charCodeAt(i) & 0xff;
        }
        loadingTask = pdfjsLib.getDocument({
          data: bytes,
        });
      } else {
        loadingTask = pdfjsLib.getDocument({
          url: pdfUrl,
        });
      }
      const pdf = await loadingTask.promise;
      window.__annolidPdf = pdf;
      window.__annolidPdfLoaded = true;
      var container = document.getElementById("viewerContainer");
      const MIN_SCALE = 0.25;
      const MAX_SCALE = 4.0;
      const DEFAULT_SCALE = 1.25;
      let scale = DEFAULT_SCALE;
      let rotation = 0;
      let nextPage = 1;
      let renderEpoch = 0;
      let renderChain = Promise.resolve();
      let zoomBusy = false;
      let pendingZoom = null;
      let renderFailureCounts = {};
      const total = pdf.numPages || 1;
      window.__annolidTotalPages = total;
      // Load persisted user state (disk via Qt bridge, plus localStorage).
      const initialStateRaw = (window.__annolidInitialUserState && typeof window.__annolidInitialUserState === "object")
        ? window.__annolidInitialUserState
        : null;
      const localStateRaw = _annolidGetStoredUserState();
      let annolidUserState = _annolidNormalizeUserState(
        _annolidPickNewestState(
          _annolidNormalizeUserState(initialStateRaw || {}),
          _annolidNormalizeUserState(localStateRaw || {}),
        ) || (initialStateRaw || localStateRaw || {})
      );
      try {
        const bridgeStateRaw = await _annolidFetchBridgeUserState();
        if (bridgeStateRaw) {
          const bridgeState = _annolidNormalizeUserState(bridgeStateRaw);
          const chosen = _annolidPickNewestState(annolidUserState, bridgeState) || annolidUserState;
          annolidUserState = _annolidNormalizeUserState(_annolidMergeUserState(chosen, bridgeState));
          annolidUserState.updatedAt = Math.max(
            annolidUserState.updatedAt || 0,
            bridgeState.updatedAt || 0,
          );
        }
      } catch (e) { }
      window.__annolidUserState = annolidUserState;
      // Apply persisted reading view (scale/rotation) early so rendering uses it.
      try {
        const reading = (annolidUserState && annolidUserState.reading) ? annolidUserState.reading : {};
        if (reading && typeof reading.zoom === "number" && isFinite(reading.zoom)) {
          scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, reading.zoom));
        }
        if (reading && typeof reading.rotation === "number" && isFinite(reading.rotation)) {
          rotation = reading.rotation;
        }
      } catch (e) { }
      // Apply persisted marks to page states (they will render as pages appear).
      try {
        const marks = (annolidUserState && annolidUserState.marks) ? annolidUserState.marks : null;
        const pages = marks && marks.pages ? marks.pages : null;
        if (pages && typeof pages === "object") {
          for (const key of Object.keys(pages)) {
            const entry = pages[key];
            const pageNum = parseInt(key, 10) || 0;
            if (!pageNum || !entry) continue;
            const st = _annolidGetPageState(pageNum);
            const strokes = entry.strokes || [];
            const highlights = entry.highlights || [];
            st.marks.strokes = Array.isArray(strokes) ? strokes : [];
            st.marks.highlights = Array.isArray(highlights) ? highlights : [];
          }
        }
      } catch (e) { }
      // Apply persisted tool defaults.
      try {
        const tool = (annolidUserState && annolidUserState.tool) ? annolidUserState.tool : {};
        if (tool && typeof tool.color === "string" && tool.color) window.__annolidMarks.color = tool.color;
        if (tool && typeof tool.size === "number" && isFinite(tool.size)) window.__annolidMarks.size = tool.size;
      } catch (e) { }
      let pdfObjectUrl = null;
      let pdfObjectUrlPromise = null;

      const titleEl = document.getElementById("annolidTitle");
      const prevPageBtn = document.getElementById("annolidPrevPage");
      const nextPageBtn = document.getElementById("annolidNextPage");
      const pageInput = document.getElementById("annolidPageInput");
      const totalPagesEl = document.getElementById("annolidTotalPages");
      const zoomOutBtn = document.getElementById("annolidZoomOut");
      const zoomInBtn = document.getElementById("annolidZoomIn");
      const zoomResetBtn = document.getElementById("annolidZoomReset");
      const zoomFitBtn = document.getElementById("annolidZoomFit");
      const rotateBtn = document.getElementById("annolidRotate");
      const zoomLabel = document.getElementById("annolidZoomLabel");
      const printBtn = document.getElementById("annolidPrint");
      const menuBtn = document.getElementById("annolidMenuBtn");
      const menuPanel = document.getElementById("annolidMenuPanel");
      const resumeBtn = document.getElementById("annolidResumeBtn");
      const clearStateBtn = document.getElementById("annolidClearStateBtn");
      const previewModal = document.getElementById("annolidPreviewModal");
      const previewCloseBtn = document.getElementById("annolidPreviewClose");
      const previewZoomOutBtn = document.getElementById("annolidPreviewZoomOut");
      const previewZoomInBtn = document.getElementById("annolidPreviewZoomIn");
      const previewZoomResetBtn = document.getElementById("annolidPreviewZoomReset");
      const previewTitleEl = document.getElementById("annolidPreviewTitle");
      const previewTextEl = document.getElementById("annolidPreviewText");
      const previewBody = document.getElementById("annolidPreviewBody");
      const previewCanvas = document.getElementById("annolidPreviewCanvas");
      const previewWrap = document.getElementById("annolidPreviewCanvasWrap");
      const previewGrid = document.getElementById("annolidPreviewGrid");
      const previewHighlight = document.getElementById("annolidPreviewHighlight");
      const previewCtx = previewCanvas ? previewCanvas.getContext("2d") : null;
      const citePopover = document.getElementById("annolidCitePopover");
      const citeTitleEl = document.getElementById("annolidCiteTitle");
      const citeBodyEl = document.getElementById("annolidCiteBody");
      const previewState = {
        open: false,
        scale: 2.0,
        pageNum: 1,
        info: null,
        title: "",
        mode: "",
        citation: null,
        autoclose: false,
      };
      const citePopoverState = {
        open: false,
        kind: "",
        key: null,
        number: null,
        anchor: null,
        autoclose: true,
      };
      let citeHoverTimer = null;
      let citeCloseTimer = null;
      let previewCloseTimer = null;

      if (titleEl) titleEl.textContent = pdfTitle || "PDF";
      if (totalPagesEl) totalPagesEl.textContent = String(total);
      if (pageInput) pageInput.setAttribute("max", String(total));
      _annolidUpdatePersistenceButtons();

      function _annolidClampScale(value) {
        const v = parseFloat(value);
        if (!isFinite(v)) return DEFAULT_SCALE;
        return Math.max(MIN_SCALE, Math.min(MAX_SCALE, v));
      }

      function _annolidNormalizeRotation(value) {
        const v = parseInt(value, 10);
        const norm = isFinite(v) ? ((v % 360) + 360) % 360 : 0;
        const steps = [0, 90, 180, 270];
        let closest = 0;
        let bestDiff = Infinity;
        steps.forEach((step) => {
          const diff = Math.abs(step - norm);
          if (diff < bestDiff) {
            bestDiff = diff;
            closest = step;
          }
        });
        return closest;
      }

      function _annolidUpdateZoomLabel() {
        if (!zoomLabel) return;
        const pct = Math.round(_annolidClampScale(scale) * 100);
        zoomLabel.textContent = String(pct) + "%";
      }

      function _annolidCleanupObjectUrl() {
        if (pdfObjectUrl) {
          try { URL.revokeObjectURL(pdfObjectUrl); } catch (e) { }
        }
        pdfObjectUrl = null;
        pdfObjectUrlPromise = null;
      }

      async function _annolidGetPdfObjectUrl() {
        if (pdfObjectUrl) return pdfObjectUrl;
        if (!pdfObjectUrlPromise) {
          if (pdf && typeof pdf.getData === "function") {
            pdfObjectUrlPromise = pdf.getData().then((data) => {
              const blob = new Blob([data], { type: "application/pdf" });
              pdfObjectUrl = URL.createObjectURL(blob);
              return pdfObjectUrl;
            }).catch(() => {
              pdfObjectUrlPromise = null;
              return pdfUrl;
            });
          } else {
            pdfObjectUrlPromise = Promise.resolve(pdfUrl);
          }
        }
        const url = await pdfObjectUrlPromise;
        if (url) pdfObjectUrl = url;
        return url || pdfUrl;
      }

      function _annolidGetSavedAnchor() {
        const state = window.__annolidUserState || {};
        const reading = state.reading || {};
        const pageNum = parseInt(reading.pageNum || ((typeof reading.page === "number") ? (reading.page + 1) : 1), 10) || 1;
        const frac = (typeof reading.offsetFrac === "number" && isFinite(reading.offsetFrac)) ? reading.offsetFrac : 0;
        return {
          pageNum: Math.max(1, Math.min(total, pageNum)),
          offsetFrac: Math.max(0, Math.min(1, frac)),
        };
      }

      function _annolidGetStoppedAnchor() {
        const state = window.__annolidUserState || {};
        const stopped = state.readingStopped || {};
        const pageNum = parseInt(stopped.pageNum || 0, 10) || 0;
        const frac = (typeof stopped.offsetFrac === "number" && isFinite(stopped.offsetFrac)) ? stopped.offsetFrac : 0;
        if (!pageNum) return null;
        return {
          pageNum: Math.max(1, Math.min(total, pageNum)),
          offsetFrac: Math.max(0, Math.min(1, frac)),
        };
      }

      function _annolidGetResumeAnchor() {
        return _annolidGetStoppedAnchor() || _annolidGetSavedAnchor();
      }

      function _annolidHasSavedState() {
        try {
          const state = window.__annolidUserState || {};
          const anchor = _annolidGetSavedAnchor();
          const hasProgress = anchor.pageNum > 1 || anchor.offsetFrac > 0.02;
          const hasBookmarks = Array.isArray(state.bookmarks) && state.bookmarks.length > 0;
          const hasNotes = Array.isArray(state.notes) && state.notes.length > 0;
          const marks = state.marks && state.marks.pages ? state.marks.pages : null;
          const hasMarks = marks && typeof marks === "object" && Object.keys(marks).length > 0;
          return hasProgress || hasBookmarks || hasNotes || hasMarks;
        } catch (e) {
          return false;
        }
      }

      async function _annolidResumeFromSavedState(force) {
        try {
          const anchor = _annolidGetResumeAnchor();
          if (anchor.pageNum <= 1 && anchor.offsetFrac <= 0.02) return;
          if (!force && anchor.pageNum > 12) {
            // Avoid rendering hundreds of pages on startup; user can click Resume.
            return;
          }
          await _annolidEnsureRenderedThrough(anchor.pageNum);
          _annolidScrollToPage(anchor.pageNum, anchor.offsetFrac);
          _annolidUpdateNavState();
          try {
            const bridge = window.__annolidBridge;
            if (bridge && typeof bridge.logEvent === "function") {
              bridge.logEvent({
                type: "resume",
                label: "Resume reading",
                pageNum: anchor.pageNum || 1,
                offsetFrac: anchor.offsetFrac || 0,
              });
            }
          } catch (e) { }
        } catch (e) { }
      }

      function _annolidUpdatePersistenceButtons() {
        if (resumeBtn) {
          const anchor = _annolidGetResumeAnchor();
          const show = anchor.pageNum > 1 || anchor.offsetFrac > 0.02;
          resumeBtn.classList.toggle("annolid-hidden", !show);
          if (show) {
            resumeBtn.textContent = `Resume reading (page ${anchor.pageNum})`;
          }
        }
        if (clearStateBtn) {
          clearStateBtn.classList.toggle("annolid-hidden", !_annolidHasSavedState());
        }
      }

      function _annolidClearSavedState() {
        window.__annolidSuppressAutosave = true;
        try {
          if (_annolidSaveTimer) {
            clearTimeout(_annolidSaveTimer);
            _annolidSaveTimer = null;
          }
        } catch (e) { }
        try {
          if (annolidStorageKey) localStorage.removeItem(annolidStorageKey);
        } catch (e) { }
        try {
          _annolidClearBridgeUserState();
        } catch (e) { }
        try {
          window.__annolidUserState = _annolidNormalizeUserState({});
        } catch (e) { }
        try {
          _annolidClearMarks();
        } catch (e) { }
        try {
          if (typeof _annolidUpdateBookmarkButton === "function") _annolidUpdateBookmarkButton();
        } catch (e) { }
        try {
          _annolidUpdatePersistenceButtons();
        } catch (e) { }
        window.__annolidSuppressAutosave = false;
      }

      function _annolidCloseMenu() {
        if (menuPanel) menuPanel.classList.remove("annolid-open");
      }
      function _annolidToggleMenu() {
        if (!menuPanel) return;
        const open = menuPanel.classList.contains("annolid-open");
        if (open) menuPanel.classList.remove("annolid-open");
        else {
          menuPanel.classList.add("annolid-open");
        }
      }

      if (menuBtn) menuBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        _annolidToggleMenu();
      });
      if (menuPanel) menuPanel.addEventListener("click", (ev) => {
        ev.stopPropagation();
        const target = ev.target;
        const action = target && target.dataset ? target.dataset.action : "";
        if (!action) return;
        _annolidCloseMenu();
        switch (action) {
          case "resume":
            _annolidResumeFromSavedState(true);
            break;
          case "first-page":
            _annolidGoToPage(1);
            break;
          case "fit":
            _annolidZoomFitWidth();
            break;
          case "reset":
            _annolidRerenderAll(1.0);
            break;
          case "print":
            _annolidPrintPdf().catch(() => { });
            break;
          case "clear-state":
            _annolidClearSavedState();
            break;
          default:
            break;
        }
      });
      document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") {
          _annolidCloseMenu();
        }
      });
      document.addEventListener("click", () => {
        _annolidCloseMenu();
      });
      window.addEventListener("beforeunload", _annolidCleanupObjectUrl);

      function _annolidOpenPreviewModal(title) {
        if (!previewModal) return;
        if (previewCloseTimer) {
          clearTimeout(previewCloseTimer);
          previewCloseTimer = null;
        }
        previewState.open = true;
        previewState.title = String(title || "Preview");
        if (previewTitleEl) previewTitleEl.textContent = previewState.title;
        _annolidSetPreviewLayout(previewState.mode || "");
        previewModal.classList.add("annolid-open");
      }

      function _annolidClosePreviewModal() {
        if (!previewModal) return;
        previewState.open = false;
        previewModal.classList.remove("annolid-open");
        previewState.info = null;
        previewState.mode = "";
        previewState.citation = null;
        previewState.autoclose = false;
        if (previewHighlight) previewHighlight.style.display = "none";
        _annolidSetPreviewLayout("");
      }

      function _annolidEscapeHtml(text) {
        return String(text || "")
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;")
          .replace(/"/g, "&quot;")
          .replace(/'/g, "&#39;");
      }

      function _annolidSetPreviewText(text) {
        if (!previewTextEl) return;
        const t = String(text || "").trim();
        if (!t) {
          previewTextEl.innerHTML = '<span class="annolid-muted">No preview available.</span>';
          return;
        }
        previewTextEl.textContent = t;
      }

      function _annolidSetPreviewMessage(message) {
        if (!previewTextEl) return;
        previewTextEl.innerHTML = '<span class="annolid-muted">' + _annolidEscapeHtml(message) + "</span>";
      }

      function _annolidSetPreviewLayout(mode) {
        const isCitation = mode === "citation";
        if (previewWrap) previewWrap.style.display = isCitation ? "none" : "";
        if (previewGrid) {
          previewGrid.classList.toggle("annolid-preview-citation", isCitation);
        }
        if (previewHighlight) previewHighlight.style.display = "none";
      }

      function _annolidShowCitePopover(title, text, anchorEl, kind, key, number) {
        if (!citePopover || !citeBodyEl) return;
        if (citeCloseTimer) {
          clearTimeout(citeCloseTimer);
          citeCloseTimer = null;
        }
        citePopoverState.open = true;
        citePopoverState.kind = kind || "";
        citePopoverState.key = (key != null) ? String(key) : null;
        citePopoverState.number = (number != null) ? number : null;
        citePopoverState.anchor = anchorEl || null;
        if (citeTitleEl) {
          citeTitleEl.textContent = String(title || "Citation");
        }
        citeBodyEl.textContent = String(text || "").trim() || "No preview available.";
        citePopover.classList.add("annolid-open");
        _annolidPositionCitePopover(anchorEl);
      }

      function _annolidShowCitationPopover(number, text, anchorEl) {
        const n = parseInt(number, 10);
        if (!n) return;
        _annolidShowCitePopover(
          "Reference [" + String(n) + "]",
          text,
          anchorEl,
          "numeric",
          "ref:" + String(n),
          n
        );
      }

      function _annolidHideCitationPopover() {
        if (!citePopover) return;
        citePopoverState.open = false;
        citePopoverState.kind = "";
        citePopoverState.key = null;
        citePopoverState.number = null;
        citePopoverState.anchor = null;
        citePopover.classList.remove("annolid-open");
      }

      function _annolidPositionCitePopover(anchorEl) {
        if (!citePopover || !anchorEl) return;
        if (!document.body.contains(anchorEl)) {
          _annolidHideCitationPopover();
          return;
        }
        const rect = anchorEl.getBoundingClientRect();
        citePopover.style.visibility = "hidden";
        citePopover.classList.add("annolid-open");
        const popRect = citePopover.getBoundingClientRect();
        const margin = 8;
        const viewportW = window.innerWidth || document.documentElement.clientWidth || 1;
        const viewportH = window.innerHeight || document.documentElement.clientHeight || 1;
        let left = rect.left + rect.width / 2 - popRect.width / 2;
        left = Math.max(margin, Math.min(left, viewportW - popRect.width - margin));
        let top = rect.bottom + margin;
        if (top + popRect.height > viewportH - margin) {
          top = rect.top - popRect.height - margin;
        }
        if (top < margin) {
          top = Math.min(viewportH - popRect.height - margin, rect.bottom + margin);
        }
        citePopover.style.left = Math.round(left) + "px";
        citePopover.style.top = Math.round(top) + "px";
        citePopover.style.visibility = "visible";
      }

      function _annolidUpdateCitePopoverPosition() {
        if (!citePopoverState.open) return;
        _annolidPositionCitePopover(citePopoverState.anchor);
      }

      const referenceIndex = {
        built: false,
        building: false,
        promise: null,
        byNumber: {},
        byAuthorYear: {},
        authorYearPromises: {},
        pageTextCache: {},
        startPage: null,
      };

      function _annolidSimplifyAlpha(text) {
        return String(text || "").toLowerCase().replace(/[^a-z]/g, "");
      }

      function _annolidIsReferencesHeading(text) {
        const t = _annolidSimplifyAlpha(text);
        return (
          t.startsWith("references") ||
          t.startsWith("bibliography") ||
          t.startsWith("literaturecited")
        );
      }

      function _annolidParseReferenceStart(text) {
        const t = String(text || "").trim();
        if (!t) return null;
        let m = /^\[\s*(\d{1,4})\s*\]/.exec(t);
        if (m) return parseInt(m[1], 10);
        m = /^\(\s*(\d{1,4})\s*\)/.exec(t);
        if (m) return parseInt(m[1], 10);
        m = /^(\d{1,4})$/.exec(t);
        if (m) {
          const n = parseInt(m[1], 10);
          if (n >= 1 && n <= 999) return n;
        }
        m = /^(\d{1,4})\s*[\.)]\s*/.exec(t);
        if (m) {
          const n = parseInt(m[1], 10);
          if (n >= 1 && n <= 999) return n;
        }
        m = /^(\d{1,4})\s+(?=[A-Za-z])/.exec(t);
        if (m) {
          const n = parseInt(m[1], 10);
          if (n >= 1 && n <= 999) return n;
        }
        return null;
      }

      function _annolidParseCitationNumber(text) {
        const raw = String(text || "").replace(/\s+/g, "");
        if (!raw) return null;
        let m = /^\[(\d{1,4})\]$/.exec(raw);
        if (m) return parseInt(m[1], 10);
        m = /^\[(\d{1,4})[,;\]]/.exec(raw);
        if (m) return parseInt(m[1], 10);
        m = /^\((\d{1,4})\)$/.exec(raw);
        if (m) return parseInt(m[1], 10);
        return null;
      }

      function _annolidExtractAuthorYearPairs(text) {
        const out = [];
        const s = String(text || "");
        if (!s) return out;
        // Patterns like: Bordes et al., 2024; Chen et al., 2025b
        const re = /([A-Z][A-Za-z\u00c0-\u017f'\u2019\u2018\u2010\u2011\u2012\u2013\u2014-]{2,})(?:\s+(et\s+al\.)\s*)?(?:,|\()\s*((?:19|20)\d{2})([a-z])?\s*\)?/g;
        let m;
        while ((m = re.exec(s)) !== null) {
          const author = String(m[1] || "").trim();
          const etal = String(m[2] || "").trim();
          const year = parseInt(m[3], 10);
          const suffix = (m[4] || "").trim();
          if (!author || !isFinite(year)) continue;
          const start = (typeof m.index === "number") ? m.index : 0;
          const end = (typeof re.lastIndex === "number") ? re.lastIndex : (start + String(m[0] || "").length);
          out.push({
            author,
            year,
            suffix,
            key: (author.toLowerCase() + ":" + String(year) + suffix.toLowerCase()),
            display: author + (etal ? " et al." : "") + ", " + String(year) + suffix,
            start,
            end,
          });
        }
        return out;
      }

      function _annolidFindCitationFromSpan(span, clientX = null, clientY = null) {
        if (!span) return null;
        const lineSpans = _annolidGetLineSpansForSpan(span);
        const lineInfo = _annolidBuildLineText(lineSpans);
        const range = lineInfo.ranges.get(span);
        if (lineInfo.text && range) {
          let anchorPos = null;
          const off = _annolidTextOffsetInSpanAtPoint(span, clientX, clientY);
          if (off != null) {
            anchorPos = range.start + off;
          }
          const center = (anchorPos != null) ? anchorPos : ((range.start + range.end) * 0.5);

          function pickClosestByCenter(localCenter, entries) {
            if (!entries || !entries.length) return null;
            let best = entries[0];
            let bestDist = Infinity;
            entries.forEach((e) => {
              const s0 = (typeof e.start === "number") ? e.start : 0;
              const e0 = (typeof e.end === "number") ? e.end : s0;
              const c0 = (s0 + e0) * 0.5;
              const dist = Math.abs(c0 - localCenter);
              if (dist < bestDist) {
                bestDist = dist;
                best = e;
              }
            });
            return best;
          }

          function extractNumericTokens(groupText) {
            const tokens = [];
            const str = String(groupText || "");
            const numRe = /\d{1,4}/g;
            let mm;
            while ((mm = numRe.exec(str)) !== null) {
              const n = parseInt(mm[0], 10);
              if (!n) continue;
              const start = mm.index;
              const end = mm.index + mm[0].length;
              tokens.push({ number: n, start, end });
            }
            return tokens;
          }

          const matches = _annolidExtractCitationMatches(lineInfo.text);
          for (const match of matches) {
            if (range.start < match.end && range.end > match.start) {
              const tokens = extractNumericTokens(match.text || "");
              if (!tokens.length) return null;
              const localCenter = center - match.start;
              const picked = pickClosestByCenter(localCenter, tokens);
              const n = picked ? parseInt(picked.number || 0, 10) : 0;
              if (n) return { kind: "numeric", number: n };
            }
          }

          // Author-year citations: [Bordes et al., 2024, Chen et al., 2025b]
          const text = lineInfo.text;
          const groupRe = /\[[^\]]*(?:19|20)\d{2}[a-z]?[^\]]*\]|\([^\)]*(?:19|20)\d{2}[a-z]?[^\)]*\)/g;
          let gm;
          while ((gm = groupRe.exec(text)) !== null) {
            const gText = gm[0] || "";
            const gStart = gm.index;
            const gEnd = gm.index + gText.length;
            if (!(range.start < gEnd && range.end > gStart)) continue;
            // Require at least one letter so we don't treat (2024) as a citation.
            if (!/[A-Za-z\u00c0-\u017f]/.test(gText)) continue;
            const pairs = _annolidExtractAuthorYearPairs(gText);
            if (!pairs.length) continue;
            const localCenter = center - gStart;
            // Prefer matching the span's year if it contains one, otherwise choose closest pair by position.
            let chosen = null;
            try {
              const digits = (String(span.textContent || "").match(/\d{4}/) || [])[0] || "";
              if (digits) {
                const targetYear = parseInt(digits, 10);
                const found = pairs.find((p) => parseInt(p.year, 10) === targetYear);
                if (found) chosen = found;
              }
            } catch (e) { }
            if (!chosen) {
              // If center falls inside a pair, choose it; else closest.
              const inside = pairs.find((p) => {
                const s0 = (typeof p.start === "number") ? p.start : 0;
                const e0 = (typeof p.end === "number") ? p.end : s0;
                return localCenter >= s0 && localCenter <= e0;
              });
              chosen = inside || pickClosestByCenter(localCenter, pairs) || pairs[0];
            }
            return {
              kind: "authorYear",
              raw: gText,
              chosen,
              pairs,
            };
          }
        }

        // Fallback: only accept strict numeric bracket/paren patterns.
        const t0 = String(span.textContent || "").trim();
        const n0 = _annolidParseCitationNumber(t0);
        if (n0) return { kind: "numeric", number: n0 };
        const prev = _annolidGetSpanNeighborText(span, -1);
        const next = _annolidGetSpanNeighborText(span, 1);
        const combo = (prev || "") + (t0 || "") + (next || "");
        const n1 = _annolidParseCitationNumber(combo);
        if (n1) return { kind: "numeric", number: n1 };
        return null;
      }

      function _annolidGetSpanNeighborText(span, dir) {
        try {
          const sib = (dir < 0) ? span.previousElementSibling : span.nextElementSibling;
          if (!sib || sib.tagName !== "SPAN") return "";
          return String(sib.textContent || "").trim();
        } catch (e) {
          return "";
        }
      }

      function _annolidGetLineSpansForSpan(span) {
        if (!span) return [];
        const layer = span.closest ? span.closest(".textLayer") : null;
        if (!layer) return [span];
        let targetRect = null;
        try { targetRect = span.getBoundingClientRect(); } catch (e) { targetRect = null; }
        if (!targetRect) return [span];
        const targetY = targetRect.top + targetRect.height * 0.5;
        const tol = Math.max(2, targetRect.height * 0.6);
        const spans = Array.from(layer.querySelectorAll("span"));
        return spans.filter((s) => {
          try {
            const r = s.getBoundingClientRect();
            const y = r.top + r.height * 0.5;
            return Math.abs(y - targetY) <= tol;
          } catch (e) {
            return false;
          }
        });
      }

      function _annolidBuildLineText(spans) {
        const items = [];
        spans.forEach((s) => {
          try {
            const text = String(s.textContent || "");
            if (!text) return;
            const r = s.getBoundingClientRect();
            items.push({ span: s, rect: r, text });
          } catch (e) { }
        });
        if (!items.length) return { text: "", ranges: new Map() };
        items.sort((a, b) => a.rect.left - b.rect.left);
        let text = "";
        let cursor = 0;
        let prevRight = null;
        const ranges = new Map();
        items.forEach((item) => {
          if (!item.text) return;
          if (text && prevRight != null) {
            const gap = item.rect.left - prevRight;
            if (gap > 3) {
              text += " ";
              cursor += 1;
            }
          }
          const start = cursor;
          text += item.text;
          cursor += item.text.length;
          ranges.set(item.span, { start, end: cursor });
          prevRight = item.rect.right;
        });
        return { text, ranges };
      }

      function _annolidExtractCitationMatches(text) {
        const out = [];
        if (!text) return out;
        const re = /[\[(]\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?(?:\s*[,;]\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?)*\s*[\])]/g;
        let m;
        while ((m = re.exec(text)) !== null) {
          out.push({ text: m[0], start: m.index, end: m.index + m[0].length });
        }
        return out;
      }

      function _annolidFindCitationNumberFromSpan(span, clientX = null, clientY = null) {
        const c = _annolidFindCitationFromSpan(span, clientX, clientY);
        if (!c || c.kind !== "numeric") return null;
        return parseInt(c.number || 0, 10) || null;
      }

      function _annolidTextOffsetInSpanAtPoint(span, clientX, clientY) {
        try {
          if (!span) return null;
          if (clientX == null || clientY == null) return null;
          const spanText = String(span.textContent || "");
          const spanTextLen = spanText.length;
          const approxByRect = () => {
            try {
              if (!spanTextLen) return null;
              const r = span.getBoundingClientRect();
              const w = r && isFinite(r.width) ? r.width : 0;
              if (!(w > 2)) return null;
              const frac = Math.max(0, Math.min(1, (clientX - r.left) / w));
              return Math.max(0, Math.min(spanTextLen, Math.round(frac * spanTextLen)));
            } catch (e) {
              return null;
            }
          };

          const doc = document;
          let node = null;
          let offset = null;
          if (doc.caretPositionFromPoint) {
            const pos = doc.caretPositionFromPoint(clientX, clientY);
            if (pos) {
              node = pos.offsetNode;
              offset = pos.offset;
            }
          } else if (doc.caretRangeFromPoint) {
            const range = doc.caretRangeFromPoint(clientX, clientY);
            if (range) {
              node = range.startContainer;
              offset = range.startOffset;
            }
          }
          if (!node) return approxByRect();
          if (node.nodeType === Node.ELEMENT_NODE) {
            const elementNode = node;
            const idx0 = Math.max(0, Math.min(elementNode.childNodes.length, offset || 0));
            let candidate = elementNode.childNodes[idx0] || elementNode.childNodes[idx0 - 1] || null;
            if (candidate && candidate.nodeType !== Node.TEXT_NODE) {
              const tw = document.createTreeWalker(candidate, NodeFilter.SHOW_TEXT, null);
              candidate = tw.nextNode();
            }
            if (candidate && candidate.nodeType === Node.TEXT_NODE) {
              node = candidate;
              offset = 0;
            } else {
              return approxByRect();
            }
          }

          const el = (node.nodeType === Node.ELEMENT_NODE) ? node : (node.parentElement || null);
          const owner = el && el.closest ? el.closest(".textLayer span") : null;
          if (owner !== span) return approxByRect();

          let idx = 0;
          const walker = document.createTreeWalker(span, NodeFilter.SHOW_TEXT, null);
          let cur;
          while ((cur = walker.nextNode())) {
            const t = cur.nodeValue || "";
            if (cur === node) {
              idx += Math.max(0, Math.min(t.length, offset || 0));
              break;
            }
            idx += t.length;
          }
          if (!isFinite(idx)) return approxByRect();
          return Math.max(0, Math.min(spanTextLen, idx));
        } catch (e) {
          return null;
        }
      }

      function _annolidFindUnderlyingTextSpan(linkEl, clientX = null, clientY = null) {
        try {
          // Prefer the actual pointer position when available (more reliable than link center).
          if (clientX != null && clientY != null && document.elementsFromPoint) {
            const els = document.elementsFromPoint(clientX, clientY) || [];
            for (const el of els) {
              if (!el || !el.tagName) continue;
              if (el.tagName === "SPAN" && el.closest && el.closest(".textLayer")) {
                return el;
              }
            }
          }

          const pageDiv = linkEl && linkEl.closest ? linkEl.closest(".page") : null;
          const layer = pageDiv ? pageDiv.querySelector(".textLayer") : null;
          const spans = layer ? Array.from(layer.querySelectorAll("span")) : [];
          const linkRect = linkEl.getBoundingClientRect();
          const lx = linkRect.left + linkRect.width / 2;
          const ly = linkRect.top + linkRect.height / 2;
          const px = (clientX != null) ? clientX : lx;
          const py = (clientY != null) ? clientY : ly;

          function intersects(a, b) {
            return !(a.right < b.left || a.left > b.right || a.bottom < b.top || a.top > b.bottom);
          }
          function containsPoint(r, x, y) {
            return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
          }
          function intersectionArea(a, b) {
            const w = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
            const h = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
            return w * h;
          }

          let best = null;
          let bestScore = -Infinity;
          for (const s of spans) {
            let r = null;
            try { r = s.getBoundingClientRect(); } catch (e) { r = null; }
            if (!r) continue;
            if (!intersects(r, linkRect)) continue;
            const hit = containsPoint(r, px, py);
            const area = intersectionArea(r, linkRect);
            const cx = r.left + r.width / 2;
            const cy = r.top + r.height / 2;
            const dist = Math.hypot(cx - px, cy - py);
            // Prioritize direct pointer hit, then intersection area, then distance.
            const score = (hit ? 1e9 : 0) + (area * 10.0) - dist;
            if (score > bestScore) {
              bestScore = score;
              best = s;
            }
          }
          if (best) return best;

          // Last resort: link center elementsFromPoint.
          if (document.elementsFromPoint) {
            const els = document.elementsFromPoint(lx, ly) || [];
            for (const el of els) {
              if (!el || !el.tagName) continue;
              if (el.tagName === "SPAN" && el.closest && el.closest(".textLayer")) {
                return el;
              }
            }
          }
        } catch (e) { }
        return null;
      }

      function _annolidExtractLinesFromTextContent(textContent) {
        const items = (textContent && textContent.items) ? textContent.items : [];
        const rows = [];
        for (const it of items) {
          try {
            const str = _annolidNormalizeText(it.str || "");
            if (!str) continue;
            const tr = it.transform || [];
            const x = isFinite(tr[4]) ? tr[4] : 0;
            const y = isFinite(tr[5]) ? tr[5] : 0;
            const w = isFinite(it.width) ? it.width : 0;
            const h = isFinite(it.height) ? Math.abs(it.height) : 0;
            rows.push({ str, x, y, w, h });
          } catch (e) { }
        }
        if (!rows.length) return [];
        const typicalH = _annolidMedian(rows.map((r) => r.h).filter((h) => isFinite(h) && h > 0));
        const yTol = Math.max(1.5, (typicalH || 0) * 0.85);
        rows.forEach((r) => {
          r.yCenter = r.y + (isFinite(r.h) ? r.h * 0.5 : 0);
        });
        rows.sort((a, b) => {
          const dy = b.yCenter - a.yCenter;
          if (Math.abs(dy) > 0.01) return dy;
          return a.x - b.x;
        });

        const lines = [];
        for (const r of rows) {
          const y = r.yCenter;
          let line = lines.length ? lines[lines.length - 1] : null;
          if (line && Math.abs(y - line.yCenter) > yTol) {
            const prev = (lines.length >= 2) ? lines[lines.length - 2] : null;
            if (prev && Math.abs(y - prev.yCenter) <= yTol) {
              line = prev;
            } else {
              line = null;
            }
          }
          if (!line || Math.abs(y - line.yCenter) > yTol) {
            line = {
              yCenter: y,
              y: r.y,
              items: [],
              xMin: r.x,
              xMax: r.x + r.w,
              yMin: r.y,
              yMax: r.y + r.h,
              text: "",
            };
            lines.push(line);
          }
          line.items.push(r);
          const n = line.items.length;
          line.yCenter = (line.yCenter * (n - 1) + y) / n;
          line.xMin = Math.min(line.xMin, r.x);
          line.xMax = Math.max(line.xMax, r.x + r.w);
          line.yMin = Math.min(line.yMin, r.y);
          line.yMax = Math.max(line.yMax, r.y + r.h);
        }

        for (const line of lines) {
          const items = Array.isArray(line.items) ? line.items : [];
          items.sort((a, b) => a.x - b.x);
          const parts = [];
          let lastX = null;
          for (const it of items) {
            if (lastX != null && it.x - lastX > 8) parts.push(" ");
            parts.push(it.str);
            lastX = it.x + (it.w || 0);
          }
          line.text = _annolidNormalizeText(parts.join(" "));
          line.y = line.yCenter;
          delete line.items;
        }
        return lines.filter((l) => l.text && l.text.length);
      }

      function _annolidOrderLinesForReading(lines, pageWidth) {
        const pw = isFinite(pageWidth) ? Math.max(1, pageWidth) : 1;
        const list = Array.isArray(lines)
          ? lines.filter((l) => l && isFinite(l.xMin) && isFinite(l.xMax) && isFinite(l.y))
          : [];
        if (!list.length) return Array.isArray(lines) ? lines.slice() : [];

        const xCenterOf = (l) => (l.xMin + l.xMax) * 0.5;
        const xs = list.map((l) => xCenterOf(l)).filter((x) => isFinite(x));
        if (xs.length < 8) {
          return list.slice().sort((a, b) => b.y - a.y);
        }
        xs.sort((a, b) => a - b);
        const q = (arr, t) => arr[Math.max(0, Math.min(arr.length - 1, Math.round(t * (arr.length - 1))))];
        const mean = (arr) => {
          if (!arr.length) return 0;
          let s = 0;
          for (let i = 0; i < arr.length; i++) s += arr[i];
          return s / arr.length;
        };
        const std = (arr, m) => {
          if (!arr.length) return 0;
          let s = 0;
          for (let i = 0; i < arr.length; i++) {
            const d = arr[i] - m;
            s += d * d;
          }
          return Math.sqrt(s / arr.length);
        };

        let m1 = q(xs, 0.25);
        let m2 = q(xs, 0.75);
        if (Math.abs(m2 - m1) < pw * 0.10) {
          return list.slice().sort((a, b) => b.y - a.y);
        }
        let a = [];
        let b = [];
        for (let iter = 0; iter < 10; iter++) {
          a = [];
          b = [];
          for (let i = 0; i < xs.length; i++) {
            const x = xs[i];
            if (Math.abs(x - m1) <= Math.abs(x - m2)) a.push(x);
            else b.push(x);
          }
          const nm1 = mean(a);
          const nm2 = mean(b);
          if (Math.abs(nm1 - m1) < 0.5 && Math.abs(nm2 - m2) < 0.5) break;
          m1 = nm1;
          m2 = nm2;
        }
        if (!a.length || !b.length) {
          return list.slice().sort((a0, b0) => b0.y - a0.y);
        }
        const leftMean = Math.min(m1, m2);
        const rightMean = Math.max(m1, m2);
        const separation = rightMean - leftMean;
        const minSeparation = Math.max(36, pw * 0.16);
        if (separation < minSeparation) {
          return list.slice().sort((a0, b0) => b0.y - a0.y);
        }
        const score = separation / Math.max(1e-6, std(a, m1) + std(b, m2));
        if (score < 1.15) {
          return list.slice().sort((a0, b0) => b0.y - a0.y);
        }

        const minCount = Math.max(3, Math.floor(xs.length * 0.18));
        const left = [];
        const right = [];
        for (const l of list) {
          const x = xCenterOf(l);
          if (!isFinite(x)) continue;
          if (Math.abs(x - leftMean) <= Math.abs(x - rightMean)) left.push(l);
          else right.push(l);
        }
        if (left.length < minCount || right.length < minCount) {
          return list.slice().sort((a0, b0) => b0.y - a0.y);
        }
        left.sort((a0, b0) => b0.y - a0.y);
        right.sort((a0, b0) => b0.y - a0.y);
        return left.concat(right);
      }

      async function _annolidFindReferencesStartPage() {
        const maxScan = Math.min(total, 120);
        const minP = Math.max(1, total - maxScan + 1);
        let inRefsBlock = false;
        let earliestRefsPage = null;
        let nonRefStreak = 0;
        for (let p = total; p >= minP; p--) {
          try {
            const page = await pdf.getPage(p);
            const tc = await page.getTextContent();
            const lines = _annolidExtractLinesFromTextContent(tc);
            const anyHeading = lines.some((l) => _annolidIsReferencesHeading((l.text || "").trim()));
            let starts = 0;
            for (const line of lines) {
              if (_annolidParseReferenceStart((line.text || "").trim()) != null) {
                starts += 1;
                if (starts >= 2) break;
              }
            }
            const isRefsPage = anyHeading || starts >= 2 || (inRefsBlock && starts >= 1);
            if (isRefsPage) {
              inRefsBlock = true;
              earliestRefsPage = p;
              nonRefStreak = 0;
              continue;
            }
            if (inRefsBlock) {
              nonRefStreak += 1;
              if (nonRefStreak >= 2) break;
            }
          } catch (e) { }
        }
        if (earliestRefsPage != null) return earliestRefsPage;
        return Math.max(1, total - 25);
      }

      async function _annolidBuildReferenceIndex() {
        if (referenceIndex.built || referenceIndex.building) return referenceIndex.promise;
        referenceIndex.building = true;
        referenceIndex.promise = (async () => {
          const start = await _annolidFindReferencesStartPage();
          referenceIndex.startPage = start;
          const byNum = {};
          let started = false;
          let current = null;
          for (let p = start; p <= total; p++) {
            let page = null;
            try {
              page = await pdf.getPage(p);
            } catch (e) {
              continue;
            }
            let tc = null;
            try {
              tc = await page.getTextContent();
            } catch (e) {
              continue;
            }
            const viewport1 = page.getViewport({ scale: 1, rotation: 0 });
            const linesRaw = _annolidExtractLinesFromTextContent(tc);
            const lines = _annolidOrderLinesForReading(linesRaw, viewport1.width || 1);
            for (const line of lines) {
              const text = (line.text || "").trim();
              if (!text) continue;
              if (_annolidIsReferencesHeading(text)) {
                started = true;
                continue;
              }
              if (!started) {
                const firstN = _annolidParseReferenceStart(text);
                if (firstN == null) {
                  continue;
                }
                started = true;
              }
              const n = _annolidParseReferenceStart(text);
              if (n != null && isFinite(n)) {
                if (current && current.num != null) {
                  byNum[String(current.num)] = current;
                }
                const rect = [line.xMin, line.yMin, line.xMax, line.yMax];
                current = {
                  num: n,
                  pageNum: p,
                  text: text,
                  rect,
                };
                continue;
              }
              if (current) {
                current.text = _annolidNormalizeText(current.text + " " + text);
                const r = current.rect || [0, 0, 0, 0];
                current.rect = [
                  Math.min(r[0], line.xMin),
                  Math.min(r[1], line.yMin),
                  Math.max(r[2], line.xMax),
                  Math.max(r[3], line.yMax),
                ];
              }
            }
          }
          if (current && current.num != null) {
            byNum[String(current.num)] = current;
          }
          referenceIndex.byNumber = byNum;
          referenceIndex.built = true;
          referenceIndex.building = false;
          return byNum;
        })().catch(() => {
          referenceIndex.byNumber = {};
          referenceIndex.built = true;
          referenceIndex.building = false;
          return referenceIndex.byNumber;
        });
        return referenceIndex.promise;
      }

      async function _annolidGetReference(number) {
        const key = String(parseInt(number, 10) || "");
        if (!key) return null;
        if (!referenceIndex.built) {
          await _annolidBuildReferenceIndex();
        }
        return referenceIndex.byNumber[key] || null;
      }

      async function _annolidOpenCitationPreview(number, fallbackDest, autoclose = true, anchorEl = null) {
        const n = parseInt(number, 10);
        if (!n) return false;
        if (citePopoverState.open && citePopoverState.kind === "numeric" && citePopoverState.number === n) {
          _annolidPositionCitePopover(anchorEl || citePopoverState.anchor);
          return true;
        }
        citePopoverState.autoclose = !!autoclose;
        _annolidShowCitationPopover(n, `Loading reference [${n}]…`, anchorEl);
        const ref = await _annolidGetReference(n);
        if (!ref) {
          _annolidShowCitationPopover(
            n,
            `Reference [${n}] not found in this PDF.`,
            anchorEl || citePopoverState.anchor
          );
          return false;
        }
        _annolidShowCitationPopover(
          n,
          ref.text || "",
          anchorEl || citePopoverState.anchor
        );
        return true;
      }

      function _annolidStripReferenceNumberPrefix(text) {
        const t = String(text || "").trim();
        if (!t) return "";
        return t
          .replace(/^\[\s*\d{1,4}\s*\]\s*/g, "")
          .replace(/^\(\s*\d{1,4}\s*\)\s*/g, "")
          .replace(/^\d{1,4}\s*[\.)]\s*/g, "")
          .replace(/^\d{1,4}\s+(?=[A-Za-z])/g, "")
          .trim();
      }

      function _annolidBuildScholarUrl(query) {
        const q = String(query || "").trim();
        const base = "https://scholar.google.com/scholar?hl=en&q=";
        return base + encodeURIComponent(q || "");
      }

      function _annolidBuildContextQueryFromLine(lineText, citationMatchText) {
        const text = String(lineText || "");
        const match = String(citationMatchText || "");
        if (!text) return "";
        let idx = -1;
        if (match) idx = text.indexOf(match);
        if (idx < 0) idx = Math.floor(text.length * 0.5);
        const start = Math.max(0, idx - 120);
        const end = Math.min(text.length, idx + 120);
        let snippet = text.slice(start, end);
        if (match) snippet = snippet.replace(match, " ");
        // Remove other bracketed numeric groups to keep the query focused.
        snippet = snippet.replace(/[\[(]\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?(?:\s*[,;]\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?)*\s*[\])]/g, " ");
        snippet = _annolidNormalizeText(snippet);
        // Drop trailing sentence fragments if the snippet is too long.
        if (snippet.length > 180) snippet = snippet.slice(0, 180).trim();
        return snippet;
      }

      function _annolidPickClosestIndex(localCenter, tokens) {
        if (!tokens || !tokens.length) return 0;
        let bestIndex = 0;
        let bestDist = Infinity;
        for (let i = 0; i < tokens.length; i++) {
          const t = tokens[i];
          const s0 = (typeof t.start === "number") ? t.start : 0;
          const e0 = (typeof t.end === "number") ? t.end : s0;
          const c0 = (s0 + e0) * 0.5;
          const dist = Math.abs(c0 - localCenter);
          if (dist < bestDist) {
            bestDist = dist;
            bestIndex = i;
          }
        }
        return bestIndex;
      }

      function _annolidFindNumericCitationGroupFromSpan(span, clientX = null, clientY = null) {
        if (!span) return null;
        const lineSpans = _annolidGetLineSpansForSpan(span);
        const lineInfo = _annolidBuildLineText(lineSpans);
        const range = lineInfo.ranges.get(span);
        if (!lineInfo.text || !range) return null;
        let anchorPos = null;
        const off = _annolidTextOffsetInSpanAtPoint(span, clientX, clientY);
        if (off != null) {
          anchorPos = range.start + off;
        }
        const center = (anchorPos != null) ? anchorPos : ((range.start + range.end) * 0.5);
        const matches = _annolidExtractCitationMatches(lineInfo.text);
        for (const match of matches) {
          if (!(range.start < match.end && range.end > match.start)) continue;
          const str = String(match.text || "");
          const tokens = [];
          const numRe = /\d{1,4}/g;
          let mm;
          while ((mm = numRe.exec(str)) !== null) {
            const n = parseInt(mm[0], 10);
            if (!n) continue;
            const start = mm.index;
            const end = mm.index + mm[0].length;
            tokens.push({ number: n, start, end });
          }
          if (!tokens.length) return null;
          const localCenter = center - match.start;
          const pickedIndex = _annolidPickClosestIndex(localCenter, tokens);
          return {
            raw: str,
            numbers: tokens.map((t) => parseInt(t.number || 0, 10)).filter((n) => n),
            pickedIndex,
            contextQuery: _annolidBuildContextQueryFromLine(lineInfo.text, str),
          };
        }
        return null;
      }

      async function _annolidOpenScholarForCitationGroup(group, anchorEl = null) {
        try {
          if (!group || !group.numbers || !group.numbers.length) return false;
          const numbers = group.numbers.map((n) => parseInt(n, 10)).filter((n) => n);
          if (!numbers.length) return false;
          const activeIndex = Math.max(0, Math.min(parseInt(group.pickedIndex || 0, 10) || 0, numbers.length - 1));
          try { await _annolidBuildReferenceIndex(); } catch (e) { }
          const refs = await Promise.all(numbers.map((n) => _annolidGetReference(n).catch(() => null)));
          const contextQuery = String(group.contextQuery || "").trim();
          const items = [];
          for (let i = 0; i < numbers.length; i++) {
            const n = numbers[i];
            const ref = refs[i];
            const refText = ref && ref.text ? String(ref.text) : "";
            const stripped = _annolidStripReferenceNumberPrefix(refText);
            const query = (stripped || refText || contextQuery || ("Reference " + String(n))).slice(0, 240);
            const title = (stripped || "").slice(0, 120);
            items.push({
              number: String(n),
              title: title,
              query: query,
              url: _annolidBuildScholarUrl(query),
            });
          }
          const payload = {
            groupLabel: String(group.raw || "").trim() || ("[" + numbers.join(", ") + "]"),
            activeIndex: activeIndex,
            items: items,
          };
          if (window.__annolidBridge && typeof window.__annolidBridge.openScholarCitations === "function") {
            window.__annolidBridge.openScholarCitations(payload);
            return true;
          }
          const first = items[activeIndex] || items[0];
          if (first && first.url) window.open(String(first.url), "_blank");
          return true;
        } catch (e) {
          return false;
        }
      }

      async function _annolidOpenScholarForAuthorYear(citation, anchorEl = null) {
        try {
          const chosen = citation && citation.chosen ? citation.chosen : null;
          const raw = citation ? String(citation.raw || "").trim() : "";
          const query = chosen && chosen.display ? String(chosen.display) : (raw || "citation");
          const item = {
            number: "",
            title: "",
            query: query,
            url: _annolidBuildScholarUrl(query),
          };
          const payload = {
            groupLabel: raw || "Citation",
            activeIndex: 0,
            items: [item],
          };
          if (window.__annolidBridge && typeof window.__annolidBridge.openScholarCitations === "function") {
            window.__annolidBridge.openScholarCitations(payload);
            return true;
          }
          window.open(String(item.url), "_blank");
          return true;
        } catch (e) {
          return false;
        }
      }

      async function _annolidResolveDestination(dest) {
        try {
          let resolved = dest;
          if (typeof resolved === "string") {
            resolved = await pdf.getDestination(resolved);
          }
          if (!resolved || !Array.isArray(resolved) || resolved.length < 2) return null;
          const ref = resolved[0];
          const kind = String(resolved[1] || "");
          let pageIndex = 0;
          try {
            pageIndex = await pdf.getPageIndex(ref);
          } catch (e) {
            pageIndex = 0;
          }
          const pageNum = (pageIndex || 0) + 1;
          const out = { pageNum, kind, left: null, top: null, right: null, bottom: null };
          if (kind === "XYZ") {
            out.left = resolved.length > 2 ? resolved[2] : null;
            out.top = resolved.length > 3 ? resolved[3] : null;
          } else if (kind === "FitH" || kind === "FitBH") {
            out.top = resolved.length > 2 ? resolved[2] : null;
          } else if (kind === "FitV" || kind === "FitBV") {
            out.left = resolved.length > 2 ? resolved[2] : null;
          } else if (kind === "FitR") {
            out.left = resolved.length > 2 ? resolved[2] : null;
            out.bottom = resolved.length > 3 ? resolved[3] : null;
            out.right = resolved.length > 4 ? resolved[4] : null;
            out.top = resolved.length > 5 ? resolved[5] : null;
          }
          return out;
        } catch (e) {
          return null;
        }
      }

      async function _annolidRenderPreview() {
        if (!previewState.open || !previewCanvas || !previewWrap || !previewBody || !previewCtx) return;
        if (previewState.mode === "citation") {
          if (previewHighlight) previewHighlight.style.display = "none";
          return;
        }
        const info = previewState.info;
        const pageNum = previewState.pageNum;
        try {
          const page = await pdf.getPage(pageNum);
          const viewport = page.getViewport({ scale: previewState.scale, rotation });
          previewCanvas.width = Math.max(1, Math.round(viewport.width));
          previewCanvas.height = Math.max(1, Math.round(viewport.height));
          previewCanvas.style.width = viewport.width + "px";
          previewCanvas.style.height = viewport.height + "px";
          if (previewCtx) {
            try { previewCtx.setTransform(1, 0, 0, 1, 0, 0); } catch (e) { }
            try { previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height); } catch (e) { }
          }
          await page.render({ canvasContext: previewCtx, viewport }).promise;

          let highlightRect = null;
          if (info && info.kind === "FitR" && info.left != null && info.right != null && info.bottom != null && info.top != null) {
            try {
              const rect = [info.left, info.bottom, info.right, info.top];
              const vrect = viewport.convertToViewportRectangle(rect);
              const x1 = Math.min(vrect[0], vrect[2]);
              const x2 = Math.max(vrect[0], vrect[2]);
              const y1 = Math.min(vrect[1], vrect[3]);
              const y2 = Math.max(vrect[1], vrect[3]);
              highlightRect = { x: x1, y: y1, w: Math.max(2, x2 - x1), h: Math.max(2, y2 - y1) };
            } catch (e) {
              highlightRect = null;
            }
          } else if (info && info.top != null) {
            try {
              const xPdf = (info.left != null) ? info.left : 0;
              const yPdf = info.top;
              const pt = viewport.convertToViewportPoint(xPdf, yPdf);
              const x = pt[0];
              const y = pt[1];
              highlightRect = { x: Math.max(0, x - 20), y: Math.max(0, y - 14), w: 220, h: 36 };
            } catch (e) {
              highlightRect = null;
            }
          }

          if (previewHighlight && highlightRect) {
            previewHighlight.style.display = "block";
            previewHighlight.style.left = Math.max(0, highlightRect.x) + "px";
            previewHighlight.style.top = Math.max(0, highlightRect.y) + "px";
            previewHighlight.style.width = Math.max(2, highlightRect.w) + "px";
            previewHighlight.style.height = Math.max(2, highlightRect.h) + "px";
            const targetY = Math.max(0, highlightRect.y - 90);
            try {
              previewBody.scrollTo({ top: targetY, behavior: "smooth" });
            } catch (e) {
              previewBody.scrollTop = targetY;
            }
          } else if (previewHighlight) {
            previewHighlight.style.display = "none";
          }
        } catch (e) {
          if (previewHighlight) previewHighlight.style.display = "none";
        }
      }

      async function _annolidOpenDestinationPreview(dest, title) {
        const info = await _annolidResolveDestination(dest);
        if (!info) return false;
        previewState.mode = "dest";
        previewState.citation = null;
        previewState.autoclose = false;
        previewState.pageNum = info.pageNum || 1;
        previewState.info = info;
        _annolidSetPreviewMessage("Jump target preview");
        _annolidSetPreviewLayout("dest");
        _annolidOpenPreviewModal(title || ("Page " + String(previewState.pageNum)));
        await _annolidRenderPreview();
        return true;
      }

      if (previewCloseBtn) previewCloseBtn.addEventListener("click", () => _annolidClosePreviewModal());
      if (previewModal) previewModal.addEventListener("click", (ev) => {
        if (ev.target === previewModal) _annolidClosePreviewModal();
      });
      if (previewModal) previewModal.addEventListener("mouseenter", () => {
        if (previewCloseTimer) {
          clearTimeout(previewCloseTimer);
          previewCloseTimer = null;
        }
      });
      if (previewModal) previewModal.addEventListener("mouseleave", () => {
        if (!previewState.open || !previewState.autoclose) return;
        if (previewCloseTimer) clearTimeout(previewCloseTimer);
        previewCloseTimer = setTimeout(() => _annolidClosePreviewModal(), 450);
      });
      document.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape" && previewState.open) _annolidClosePreviewModal();
      });
      if (previewZoomOutBtn) previewZoomOutBtn.addEventListener("click", async () => {
        previewState.scale = Math.max(0.6, previewState.scale / 1.15);
        await _annolidRenderPreview();
      });
      if (previewZoomInBtn) previewZoomInBtn.addEventListener("click", async () => {
        previewState.scale = Math.min(6.0, previewState.scale * 1.15);
        await _annolidRenderPreview();
      });
      if (previewZoomResetBtn) previewZoomResetBtn.addEventListener("click", async () => {
        previewState.scale = 2.0;
        await _annolidRenderPreview();
      });

      if (citePopover) {
        citePopover.addEventListener("mouseenter", () => {
          if (citeCloseTimer) {
            clearTimeout(citeCloseTimer);
            citeCloseTimer = null;
          }
        });
        citePopover.addEventListener("mouseleave", () => {
          _annolidScheduleCitationClose();
        });
      }
      if (container) {
        container.addEventListener("scroll", () => _annolidUpdateCitePopoverPosition());
      }
      window.addEventListener("resize", () => _annolidUpdateCitePopoverPosition());

      function _annolidCancelCitationHover() {
        if (citeHoverTimer) {
          clearTimeout(citeHoverTimer);
          citeHoverTimer = null;
        }
      }

      function _annolidScheduleCitationHover(span, fallbackDest, autoclose, delayMs, clientX = null, clientY = null, citeOverride = null) {
        const cite = citeOverride || _annolidFindCitationFromSpan(span, clientX, clientY);
        if (!cite) return false;
        const citeKey = (cite.kind === "numeric")
          ? ("ref:" + String(cite.number || ""))
          : ("ay:" + String((cite.chosen && cite.chosen.key) ? cite.chosen.key : ""));
        if (citePopoverState.open && citePopoverState.key && citePopoverState.key === citeKey) {
          _annolidPositionCitePopover(span);
          return true;
        }
        _annolidCancelCitationHover();
        citeHoverTimer = setTimeout(() => {
          citeHoverTimer = null;
          if (cite.kind === "numeric") {
            _annolidOpenCitationPreview(
              cite.number,
              fallbackDest,
              autoclose,
              span
            ).catch(() => { });
          } else {
            _annolidOpenAuthorYearPreview(
              cite,
              fallbackDest,
              autoclose,
              span
            ).catch(() => { });
          }
        }, Math.max(80, delayMs || 220));
        return true;
      }

      function _annolidScheduleCitationClose() {
        if (!citePopoverState.open || !citePopoverState.autoclose) return;
        if (citeCloseTimer) clearTimeout(citeCloseTimer);
        citeCloseTimer = setTimeout(() => _annolidHideCitationPopover(), 350);
      }

      let _annolidHoverSpan = null;
      let _annolidHoverCiteKey = null;
      if (container) {
        container.addEventListener("click", (ev) => {
          try {
            if (ev.button !== 0) return;
            if (ev.defaultPrevented) return;
            const tool = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool : "select";
            if (tool !== "select") return;
            const link = ev.target && ev.target.closest ? ev.target.closest(".annotationLayer a") : null;
            if (link) return;
            const span = ev.target && ev.target.closest ? ev.target.closest(".textLayer span") : null;
            if (!span) return;
            const cite = _annolidFindCitationFromSpan(span, ev.clientX, ev.clientY);
            if (!cite) return;
            ev.preventDefault();
            ev.stopPropagation();
            if (cite.kind === "numeric") {
              const group = _annolidFindNumericCitationGroupFromSpan(span, ev.clientX, ev.clientY) || {
                raw: "[" + String(cite.number || "") + "]",
                numbers: [parseInt(cite.number || 0, 10)],
                pickedIndex: 0,
              };
              _annolidOpenScholarForCitationGroup(group, span).catch(() => { });
              return;
            }
            _annolidOpenScholarForAuthorYear(cite, span).catch(() => { });
          } catch (e) { }
        });
        container.addEventListener("mousemove", (ev) => {
          try {
            if (ev.buttons && ev.buttons !== 0) return;
            const tool = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool : "select";
            if (tool !== "select") return;
            const link = ev.target && ev.target.closest ? ev.target.closest(".annotationLayer a") : null;
            if (link) return;
            const span = ev.target && ev.target.closest ? ev.target.closest(".textLayer span") : null;
            let citeKey = null;
            const cite = _annolidFindCitationFromSpan(span, ev.clientX, ev.clientY);
            if (cite) {
              citeKey = (cite.kind === "numeric")
                ? ("ref:" + String(cite.number || ""))
                : ("ay:" + String((cite.chosen && cite.chosen.key) ? cite.chosen.key : ""));
            }
            if (span === _annolidHoverSpan && citeKey === _annolidHoverCiteKey) return;
            _annolidHoverSpan = span;
            _annolidHoverCiteKey = citeKey;
            if (!span) {
              _annolidCancelCitationHover();
              _annolidScheduleCitationClose();
              return;
            }
            _annolidCancelCitationHover();
            const handled = _annolidScheduleCitationHover(span, null, true, 200, ev.clientX, ev.clientY, cite);
            if (!handled) {
              _annolidScheduleCitationClose();
            }
          } catch (e) { }
        });
        container.addEventListener("mouseleave", () => {
          _annolidHoverSpan = null;
          _annolidHoverCiteKey = null;
          _annolidCancelCitationHover();
          _annolidScheduleCitationClose();
        });
      }

      // --- Persistence helpers (progress, marks, notes, bookmarks) ---------
      let _annolidSaveTimer = null;
      function _annolidExportUserState() {
        const base = _annolidNormalizeUserState(window.__annolidUserState || {});
        const anchor = (typeof _annolidGetScrollAnchor === "function")
          ? _annolidGetScrollAnchor()
          : { pageNum: (typeof _annolidGetCurrentPageNum === "function") ? _annolidGetCurrentPageNum() : 1, offsetFrac: 0 };
        base.updatedAt = annolidNowMs() / 1000.0;
        base.reading = base.reading || {};
        base.reading.pageNum = anchor.pageNum || 1;
        base.reading.page = Math.max(0, (anchor.pageNum || 1) - 1);
        base.reading.offsetFrac = (typeof anchor.offsetFrac === "number" && isFinite(anchor.offsetFrac)) ? anchor.offsetFrac : 0;
        base.reading.zoom = scale;
        base.reading.rotation = rotation;
        base.tool = {
          color: window.__annolidMarks && window.__annolidMarks.color ? window.__annolidMarks.color : "#ffb300",
          size: window.__annolidMarks && typeof window.__annolidMarks.size === "number" ? window.__annolidMarks.size : 10,
        };
        const pagesRaw = window.__annolidPages || {};
        const pages = {};
        for (const key of Object.keys(pagesRaw)) {
          const st = pagesRaw[key];
          if (!st || !st.marks) continue;
          const strokes = st.marks.strokes || [];
          const highlights = st.marks.highlights || [];
          if ((strokes && strokes.length) || (highlights && highlights.length)) {
            pages[String(st.pageNum || key)] = {
              strokes: strokes,
              highlights: highlights,
            };
          }
        }
        base.marks = {
          pages,
          view: { zoom: scale, rotation },
        };
        if (!Array.isArray(base.bookmarks)) base.bookmarks = [];
        if (!Array.isArray(base.notes)) base.notes = [];
        return base;
      }
      function _annolidFlushSaveUserState() {
        try {
          if (window.__annolidSuppressAutosave) return;
          const state = _annolidExportUserState();
          window.__annolidUserState = state;
          _annolidSetStoredUserState(state);
          _annolidSendBridgeUserState(state);
        } catch (e) { }
      }
      function _annolidRequestSaveUserState(delayMs) {
        if (window.__annolidSuppressAutosave) return;
        const delay = Math.max(200, parseInt(delayMs || 0, 10) || 900);
        if (_annolidSaveTimer) clearTimeout(_annolidSaveTimer);
        _annolidSaveTimer = setTimeout(() => {
          _annolidSaveTimer = null;
          _annolidFlushSaveUserState();
        }, delay);
      }
      window.__annolidExportUserState = _annolidExportUserState;
      window.__annolidRequestSaveUserState = _annolidRequestSaveUserState;
      window.addEventListener("beforeunload", () => _annolidFlushSaveUserState());

      window.__annolidStartReaderAtAnchor = async function (pageNum, offsetFrac) {
        try {
          if (!window.__annolidBridge || typeof window.__annolidBridge.onParagraphClicked !== "function") return false;
          const target = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
          const frac = isFinite(offsetFrac) ? Math.max(0, Math.min(1, offsetFrac)) : 0;
          if (window.__annolidScrollToAnchor) {
            await window.__annolidScrollToAnchor(target, frac);
          } else {
            await _annolidEnsurePageAvailable(target);
            _annolidScrollToPage(target, frac);
          }

          let pageParas = window.__annolidParagraphsByPage[String(target)];
          if (!pageParas || !pageParas.length) {
            pageParas = _annolidBuildParagraphsForPage(target);
            _annolidRebuildParagraphIndex();
          }
          if (!pageParas || !pageParas.length) {
            await _annolidBuildTextParagraphsForPage(target);
            await _annolidEnsureParagraphsFrom(target);
            pageParas = window.__annolidParagraphsByPage[String(target)] || [];
          }
          if (!pageParas || !pageParas.length) return false;

          const paraIndex = Math.max(
            0,
            Math.min(pageParas.length - 1, Math.floor(frac * pageParas.length))
          );
          await _annolidEnsureParagraphsFrom(target);
          const offset = window.__annolidParagraphOffsets[target] || 0;
          const startIndex = offset + paraIndex;
          const remaining = window.__annolidParagraphs.slice(startIndex).map((p) => ({
            text: p.text || "",
            spans: p.spans || [],
            pageNum: (parseInt(p.pageNum || p.page || 0, 10) || target),
          }));
          if (!remaining.length) return false;

          window.__annolidBridge.onParagraphClicked({
            startIndex,
            total: window.__annolidParagraphTotal || (startIndex + remaining.length),
            paragraphs: remaining,
          });
          return true;
        } catch (e) {
          return false;
        }
      };

      function _annolidQueueRender(fn) {
        const myEpoch = renderEpoch;
        const task = renderChain.then(async () => {
          if (myEpoch !== renderEpoch) return;
          return await fn(myEpoch);
        });
        renderChain = task.catch((e) => {
          console.warn("Annolid render op failed", e);
        });
        return task;
      }

      async function _annolidRenderPageSafely(pageNum, epoch, maxRetries) {
        const retries = Math.max(0, parseInt(maxRetries, 10) || 0);
        let lastErr = null;
        for (let attempt = 0; attempt <= retries; attempt++) {
          try {
            await renderPage(pageNum, epoch);
            return true;
          } catch (e) {
            lastErr = e;
            if (attempt < retries) {
              await new Promise((r) => setTimeout(r, 120 * (attempt + 1)));
            }
          }
        }
        try {
          console.warn("Annolid renderPage failed", { pageNum, retries, error: lastErr });
        } catch (e) { }
        return false;
      }

      async function _annolidTryRenderNextPage(epoch, maxRetries) {
        const pageNum = Math.max(1, parseInt(nextPage, 10) || 1);
        const ok = await _annolidRenderPageSafely(pageNum, epoch, maxRetries);
        if (ok) {
          try { delete renderFailureCounts[pageNum]; } catch (e) { }
          nextPage += 1;
          return true;
        }
        const prev = parseInt(renderFailureCounts[pageNum] || 0, 10) || 0;
        const failures = prev + 1;
        renderFailureCounts[pageNum] = failures;
        // Keep retrying transient failures; only emit a placeholder after repeated
        // failures for the same page so first-open page holes are not created.
        if (failures >= 3) {
          try {
            if (epoch === renderEpoch) {
              _annolidInsertPageErrorPlaceholder(pageNum, {
                message: `render retry limit reached (${failures})`,
              });
            }
          } catch (e) { }
          nextPage += 1;
          return true;
        }
        try {
          console.warn("Annolid renderPage deferred for retry", { pageNum, failures });
        } catch (e) { }
        return false;
      }

      function _annolidInsertPageErrorPlaceholder(pageNum, err) {
        if (!container) return;
        const n = Math.max(1, parseInt(pageNum, 10) || 1);
        if (container.querySelector(`.page[data-page-number="${n}"]`)) return;
        const pageDiv = document.createElement("div");
        pageDiv.className = "page annolid-page-error";
        pageDiv.setAttribute("data-page-number", String(n));
        pageDiv.style.minHeight = "220px";
        pageDiv.style.display = "flex";
        pageDiv.style.alignItems = "center";
        pageDiv.style.justifyContent = "center";
        pageDiv.style.padding = "20px";
        const msg = document.createElement("div");
        msg.style.maxWidth = "720px";
        msg.style.color = "#b71c1c";
        msg.style.background = "rgba(255, 235, 238, 0.92)";
        msg.style.border = "1px solid rgba(183, 28, 28, 0.25)";
        msg.style.borderRadius = "8px";
        msg.style.padding = "12px 14px";
        msg.style.fontSize = "14px";
        msg.style.lineHeight = "1.35";
        const detail =
          (err && (err.message || err.name)) ? ` (${String(err.message || err.name)})` : "";
        msg.textContent = `Page ${n} failed to render${detail}.`;
        pageDiv.appendChild(msg);
        container.appendChild(pageDiv);
      }

      function _annolidNormalizeRefText(text) {
        return _annolidNormalizeText(String(text || "")).toLowerCase();
      }

      async function _annolidGetOrderedLinesForPage(pageNum) {
        const key = String(pageNum);
        if (referenceIndex.pageTextCache && referenceIndex.pageTextCache[key]) {
          return referenceIndex.pageTextCache[key];
        }
        let page = null;
        try { page = await pdf.getPage(pageNum); } catch (e) { page = null; }
        if (!page) return [];
        let tc = null;
        try { tc = await page.getTextContent(); } catch (e) { tc = null; }
        if (!tc) return [];
        const viewport1 = page.getViewport({ scale: 1, rotation: 0 });
        const linesRaw = _annolidExtractLinesFromTextContent(tc);
        const lines = _annolidOrderLinesForReading(linesRaw, viewport1.width || 1);
        if (referenceIndex.pageTextCache) referenceIndex.pageTextCache[key] = lines;
        return lines;
      }

      async function _annolidGetAuthorYearReference(author, year) {
        const a = String(author || "").trim();
        const y = parseInt(year, 10);
        if (!a || !isFinite(y)) return null;
        const key = a.toLowerCase() + ":" + String(y);
        if (referenceIndex.byAuthorYear && referenceIndex.byAuthorYear[key]) {
          return referenceIndex.byAuthorYear[key];
        }
        if (referenceIndex.authorYearPromises && referenceIndex.authorYearPromises[key]) {
          try {
            await referenceIndex.authorYearPromises[key];
          } catch (e) { }
          return referenceIndex.byAuthorYear[key] || null;
        }

        referenceIndex.authorYearPromises[key] = (async () => {
          const start = referenceIndex.startPage || (await _annolidFindReferencesStartPage());
          referenceIndex.startPage = start;
          const authorNorm = a.toLowerCase();
          const yearStr = String(y);
          let best = null;
          let bestScore = 0;

          for (let p = start; p <= total; p++) {
            const lines = await _annolidGetOrderedLinesForPage(p);
            if (!lines || !lines.length) continue;
            for (let i = 0; i < lines.length; i++) {
              const line = lines[i];
              let merged = String(line.text || "");
              // Merge a couple subsequent lines for context.
              for (let j = 1; j <= 3; j++) {
                const nxt = lines[i + j];
                if (!nxt || !nxt.text) break;
                const nt = String(nxt.text || "").trim();
                // Stop if it looks like a new entry (starts with surname + comma).
                if (/^[A-Z][A-Za-z\u00c0-\u017f'\u2019\u2018-]+\s*,/.test(nt)) break;
                merged = _annolidNormalizeText(merged + " " + nt);
              }
              const t = _annolidNormalizeRefText(merged);
              if (!t) continue;
              if (t.indexOf(yearStr) < 0) continue;
              if (t.indexOf(authorNorm) < 0) continue;
              let score = 0;
              if (t.startsWith(authorNorm)) score += 4;
              if (t.indexOf(authorNorm + ",") >= 0) score += 3;
              score += 2;
              score += Math.min(3, Math.floor(merged.length / 120));
              if (score > bestScore) {
                bestScore = score;
                best = {
                  key,
                  pageNum: p,
                  text: merged,
                  rect: [line.xMin, line.yMin, line.xMax, line.yMax],
                };
              }
              if (bestScore >= 8) break;
            }
            if (bestScore >= 8) break;
          }
          if (best) {
            referenceIndex.byAuthorYear[key] = best;
          }
          return referenceIndex.byAuthorYear;
        })().catch(() => {
          return referenceIndex.byAuthorYear;
        });

        try {
          await referenceIndex.authorYearPromises[key];
        } catch (e) { }
        try {
          delete referenceIndex.authorYearPromises[key];
        } catch (e) { }
        return referenceIndex.byAuthorYear[key] || null;
      }

      async function _annolidOpenAuthorYearPreview(citation, fallbackDest, autoclose = true, anchorEl = null) {
        try {
          if (!citation) return false;
          const chosen = citation.chosen || (citation.pairs ? citation.pairs[0] : null);
          if (!chosen) return false;
          const author = chosen.author || "";
          const year = chosen.year || 0;
          const raw = String(citation.raw || "").trim();
          const title = chosen.display ? ("Citation: " + String(chosen.display)) : "Citation";
          citePopoverState.autoclose = !!autoclose;
          _annolidShowCitePopover(
            title,
            raw ? raw : (String(author) + ", " + String(year)),
            anchorEl,
            "authorYear",
            "ay:" + String(chosen.key || (author.toLowerCase() + ":" + String(year))),
            null
          );
          const ref = await _annolidGetAuthorYearReference(author, year);
          if (ref && ref.text) {
            const body = (raw ? (raw + "\n\n") : "") + String(ref.text || "");
            _annolidShowCitePopover(
              title,
              body,
              anchorEl || citePopoverState.anchor,
              "authorYear",
              citePopoverState.key,
              null
            );
            return true;
          }
          // If a destination exists, fall back to showing the destination preview.
          if (fallbackDest) {
            // keep the popover message instead of opening the modal on hover
            _annolidShowCitePopover(
              title,
              (raw ? (raw + "\n\n") : "") + "No matching reference found in this PDF.",
              anchorEl || citePopoverState.anchor,
              "authorYear",
              citePopoverState.key,
              null
            );
            return false;
          }
          _annolidShowCitePopover(
            title,
            (raw ? (raw + "\n\n") : "") + "No matching reference found in this PDF.",
            anchorEl || citePopoverState.anchor,
            "authorYear",
            citePopoverState.key,
            null
          );
          return false;
        } catch (e) {
          return false;
        }
      }

      function _annolidGetCurrentPageNum() {
        if (!container) return 1;
        const pages = Array.from(container.querySelectorAll(".page"));
        if (!pages.length) return 1;
        const scrollTop = container.scrollTop;
        let bestPage = 1;
        let bestDist = Infinity;
        pages.forEach((page) => {
          const top = page.offsetTop || 0;
          const dist = Math.abs(top - scrollTop);
          if (dist < bestDist) {
            bestDist = dist;
            bestPage = parseInt(page.getAttribute("data-page-number") || "1", 10) || bestPage;
          }
        });
        return bestPage;
      }

      function _annolidSetDisabled(el, disabled) {
        if (!el) return;
        el.classList.toggle("annolid-disabled", !!disabled);
        try { el.disabled = !!disabled; } catch (e) { }
      }

      function _annolidUpdateNavState() {
        const current = _annolidGetCurrentPageNum();
        if (pageInput && document.activeElement !== pageInput) {
          pageInput.value = String(current);
        }
        _annolidSetDisabled(prevPageBtn, current <= 1);
        _annolidSetDisabled(nextPageBtn, current >= total);
        _annolidUpdateZoomLabel();
        try {
          if (typeof _annolidUpdateBookmarkButton === "function") _annolidUpdateBookmarkButton();
        } catch (e) { }
        if (window.__annolidRequestSaveUserState) {
          window.__annolidRequestSaveUserState(1200);
        }
      }

      function _annolidScrollToPage(pageNum, offsetFrac) {
        if (!container) return;
        const el = container.querySelector(`.page[data-page-number="${pageNum}"]`);
        if (!el) return;
        const frac = isFinite(offsetFrac) ? Math.max(0, Math.min(1, offsetFrac)) : 0;
        const inner = el.clientHeight || 1;
        container.scrollTop = Math.max(0, (el.offsetTop || 0) + inner * frac - 8);
      }

      function _annolidGetScrollAnchor() {
        if (!container) return { pageNum: 1, offsetFrac: 0 };
        const current = _annolidGetCurrentPageNum();
        const el = container.querySelector(`.page[data-page-number="${current}"]`);
        if (!el) return { pageNum: current, offsetFrac: 0 };
        const top = el.offsetTop || 0;
        const h = el.clientHeight || 1;
        const frac = (container.scrollTop - top) / h;
        return { pageNum: current, offsetFrac: isFinite(frac) ? frac : 0 };
      }

      async function _annolidTransformMarksForView(oldScale, newScale, oldRotation, newRotation) {
        if (!pdf) return;
        const pages = window.__annolidPages || {};
        const keys = Object.keys(pages);
        if (!keys.length) return;
        const fromScale = _annolidClampScale(oldScale);
        const toScale = _annolidClampScale(newScale);
        const fromRotation = _annolidNormalizeRotation(oldRotation);
        const toRotation = _annolidNormalizeRotation(newRotation);
        await Promise.all(keys.map(async (key) => {
          const state = pages[key];
          if (!state || !state.marks) return;
          const pageNum = parseInt(key, 10) || state.pageNum || 0;
          if (!pageNum) return;
          let pageRef = null;
          try {
            pageRef = await pdf.getPage(pageNum);
          } catch (e) {
            return;
          }
          let fromVp = null;
          let toVp = null;
          try {
            fromVp = pageRef.getViewport({ scale: fromScale, rotation: fromRotation });
            toVp = pageRef.getViewport({ scale: toScale, rotation: toRotation });
          } catch (e) {
            return;
          }
          const convertPoint = (pt) => {
            if (!pt || !fromVp || !toVp || typeof fromVp.convertToPdfPoint !== "function") return pt;
            try {
              const pdfPt = fromVp.convertToPdfPoint(pt.x, pt.y);
              const nextPt = toVp.convertToViewportPoint(pdfPt[0], pdfPt[1]);
              return { ...pt, x: nextPt[0], y: nextPt[1] };
            } catch (e) {
              return pt;
            }
          };
          const transformRect = (rect) => {
            if (!rect) return rect;
            const corners = [
              convertPoint({ x: rect.x, y: rect.y }),
              convertPoint({ x: rect.x + rect.w, y: rect.y }),
              convertPoint({ x: rect.x + rect.w, y: rect.y + rect.h }),
              convertPoint({ x: rect.x, y: rect.y + rect.h }),
            ].filter(Boolean);
            if (!corners.length) return rect;
            const xs = corners.map((c) => c.x);
            const ys = corners.map((c) => c.y);
            const minX = Math.min(...xs);
            const maxX = Math.max(...xs);
            const minY = Math.min(...ys);
            const maxY = Math.max(...ys);
            return { ...rect, x: minX, y: minY, w: maxX - minX, h: maxY - minY };
          };
          if (state.marks.strokes) {
            state.marks.strokes = state.marks.strokes.map((stroke) => ({
              ...stroke,
              points: (stroke.points || []).map((p) => convertPoint(p)),
            }));
          }
          if (state.marks.highlights) {
            state.marks.highlights = state.marks.highlights.map((hl) => ({
              ...hl,
              rects: (hl.rects || []).map((r) => transformRect(r)),
            }));
          }
        }));
      }

      async function _annolidEnsureRenderedThrough(pageNum) {
        const target = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
        return _annolidQueueRender(async (epoch) => {
          let stalls = 0;
          while (nextPage <= target && nextPage <= total) {
            const progressed = await _annolidTryRenderNextPage(epoch, 2);
            if (!progressed) {
              stalls += 1;
              if (stalls >= 3) break;
              await new Promise(r => setTimeout(r, 150 * stalls));
              continue;
            }
            stalls = 0;
            await new Promise(r => setTimeout(r, 0));
          }
        });
      }

      function _annolidIsPageRendered(pageNum) {
        if (!container) return false;
        const n = parseInt(pageNum, 10) || 1;
        return !!container.querySelector(`.page[data-page-number="${n}"]`);
      }

      function _annolidResetContainerForJump(startPage) {
        if (!container) return;
        renderEpoch += 1;
        renderChain = Promise.resolve();
        nextPage = Math.max(1, Math.min(total, parseInt(startPage, 10) || 1));
        container.innerHTML = "";
        window.__annolidSpans = [];
        window.__annolidSpanCounter = 0;
        window.__annolidSpanMeta = {};
        window.__annolidLinkTargets = {};
        window.__annolidLinkTargetCounter = 0;
        window.__annolidRenderedPages = 0;
        window.__annolidParagraphsByPage = {};
        window.__annolidParagraphOffsets = {};
        window.__annolidParagraphTotal = 0;
        window.__annolidParagraphs = [];
        renderFailureCounts = {};
        const pages = window.__annolidPages || {};
        Object.keys(pages).forEach((key) => {
          const state = pages[key];
          if (!state) return;
          state.pageDiv = null;
          state.width = 0;
          state.height = 0;
          state.dpr = 1;
          state.ttsCanvas = null;
          state.ttsCtx = null;
          state.markCanvas = null;
          state.markCtx = null;
        });
      }

      async function _annolidEnsurePageAvailable(pageNum) {
        const target = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
        if (_annolidIsPageRendered(target)) return;
        const current = (typeof _annolidGetCurrentPageNum === "function") ? _annolidGetCurrentPageNum() : 1;
        const gap = Math.abs(target - current);
        // Large jumps: render a small window around the target instead of every prior page.
        if (gap >= 12) {
          _annolidResetContainerForJump(Math.max(1, target - 1));
        }
        await _annolidEnsureRenderedThrough(target);
      }

      async function _annolidGoToPage(pageNum) {
        const target = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
        await _annolidEnsurePageAvailable(target);
        _annolidScrollToPage(target, 0);
        _annolidUpdateNavState();
      }

      window.__annolidScrollToAnchor = async function (pageNum, offsetFrac) {
        try {
          const target = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
          const frac = isFinite(offsetFrac) ? Math.max(0, Math.min(1, offsetFrac)) : 0;
          await _annolidEnsurePageAvailable(target);
          _annolidScrollToPage(target, frac);
          _annolidUpdateNavState();
        } catch (e) { }
      };

      // Expose render helpers for the Qt bridge.
      window.__annolidEnsureRenderedThrough = _annolidEnsureRenderedThrough;
      window.__annolidGoToPage = _annolidGoToPage;
      window.__annolidZoomFitWidth = _annolidZoomFitWidth;
      window.__annolidRotate = function (delta) {
        const step = isFinite(delta) ? delta : 90;
        _annolidRerenderAll(scale, rotation + step);
      };
      window.__annolidSetRotation = function (angle) {
        if (!isFinite(angle)) return;
        _annolidRerenderAll(scale, angle);
      };

      window.__annolidHighlightParagraphByText = async function (pageNum, text) {
        try {
          const p = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
          const wanted = _annolidNormalizeText(text || "").toLowerCase();
          if (!wanted) return;
          await _annolidEnsureRenderedThrough(p);
          const paras = _annolidBuildParagraphsForPage(p) || [];
          if (!paras.length) return;

          function scoreCandidate(candidate) {
            const cand = _annolidNormalizeText(candidate || "").toLowerCase();
            if (!cand) return 0;
            if (cand === wanted) return 1.0;
            if (cand.includes(wanted) || wanted.includes(cand)) {
              return Math.min(cand.length, wanted.length) / Math.max(1, Math.max(cand.length, wanted.length));
            }
            const a = wanted.split(" ").filter(Boolean);
            const b = cand.split(" ").filter(Boolean);
            if (!a.length || !b.length) return 0;
            const limit = 60;
            const setA = new Set(a.slice(0, limit));
            const setB = new Set(b.slice(0, limit));
            let inter = 0;
            setA.forEach((t) => { if (setB.has(t)) inter += 1; });
            const denom = Math.max(1, setA.size + setB.size - inter);
            return inter / denom;
          }

          let best = null;
          let bestScore = 0;
          for (const para of paras) {
            const s = scoreCandidate(para.text);
            if (s > bestScore) {
              bestScore = s;
              best = para;
            }
          }
          if (!best || bestScore < 0.10) return;
          const spans = best.spans || [];
          if (!spans.length) return;
          window.__annolidHighlightSentenceIndices(spans);
        } catch (e) {
          // ignore
        }
      };

      window.__annolidHighlightSentenceByText = async function (pageNum, text) {
        try {
          const p = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
          const wanted = _annolidNormalizeText(text || "").toLowerCase();
          const scrollPageFallback = () => {
            try {
              const container = document.getElementById("viewerContainer");
              const page = document.querySelector(`.page[data-page-number='${p}']`);
              if (!container || !page) return;
              const center = (page.offsetTop || 0) + (page.clientHeight || 0) / 2;
              const target = Math.max(0, center - container.clientHeight / 2);
              container.scrollTop = target;
            } catch (e) { }
          };
          const currentPage = (typeof _annolidGetCurrentPageNum === "function") ? _annolidGetCurrentPageNum() : 0;
          if (!wanted) {
            await _annolidEnsureRenderedThrough(p);
            if (currentPage !== p) scrollPageFallback();
            return;
          }

          await _annolidEnsureRenderedThrough(p);
          const paras = _annolidBuildParagraphsForPage(p) || [];
          if (!paras.length) {
            if (currentPage !== p) scrollPageFallback();
            return;
          }

          function scoreCandidate(candidate) {
            const cand = _annolidNormalizeText(candidate || "").toLowerCase();
            if (!cand) return 0;
            if (cand === wanted) return 1.0;
            if (cand.includes(wanted) || wanted.includes(cand)) {
              return Math.min(cand.length, wanted.length) / Math.max(1, Math.max(cand.length, wanted.length));
            }
            const a = wanted.split(" ").filter(Boolean);
            const b = cand.split(" ").filter(Boolean);
            if (!a.length || !b.length) return 0;
            const limit = 80;
            const setA = new Set(a.slice(0, limit));
            const setB = new Set(b.slice(0, limit));
            let inter = 0;
            setA.forEach((t) => { if (setB.has(t)) inter += 1; });
            const denom = Math.max(1, setA.size + setB.size - inter);
            return inter / denom;
          }

          let best = null;
          let bestScore = 0;
          for (const para of paras) {
            const splits = (typeof window.__annolidSplitParagraphIntoSentences === "function")
              ? (window.__annolidSplitParagraphIntoSentences(para) || [])
              : [];
            if (splits.length) {
              for (const s of splits) {
                const sc = scoreCandidate(s.text || "");
                if (sc > bestScore) {
                  bestScore = sc;
                  best = s;
                }
              }
            } else {
              const sc = scoreCandidate(para.text || "");
              if (sc > bestScore) {
                bestScore = sc;
                best = para;
              }
            }
          }

          if (!best || bestScore < 0.05) {
            if (currentPage !== p) scrollPageFallback();
            return;
          }
          const spans = best.spans || [];
          if (!spans.length) {
            if (currentPage !== p) scrollPageFallback();
            return;
          }
          window.__annolidHighlightSentenceIndices && window.__annolidHighlightSentenceIndices(spans);
          window.__annolidScrollToSentence && window.__annolidScrollToSentence(spans, p);
        } catch (e) {
          try {
            const p = Math.max(1, Math.min(total, parseInt(pageNum, 10) || 1));
            if (typeof _annolidEnsureRenderedThrough === "function") {
              await _annolidEnsureRenderedThrough(p);
            }
          } catch (e2) { }
        }
      };

      async function _annolidRerenderAll(newScale, newRotation = rotation) {
        if (!container) return;
        const desiredScale = (newScale == null) ? scale : newScale;
        const clamped = _annolidClampScale(desiredScale);
        const targetRotation = _annolidNormalizeRotation(newRotation);
        const oldScale = scale;
        const oldRotation = rotation;
        if (Math.abs(clamped - oldScale) < 0.001 && targetRotation === oldRotation) return;
        if (zoomBusy) {
          pendingZoom = { scale: clamped, rotation: targetRotation };
          return;
        }
        zoomBusy = true;
        pendingZoom = null;

        try {
          await _annolidTransformMarksForView(oldScale, clamped, oldRotation, targetRotation);
          const anchor = _annolidGetScrollAnchor();
          scale = clamped;
          rotation = targetRotation;

          renderEpoch += 1;
          renderChain = Promise.resolve();
          nextPage = 1;
          container.innerHTML = "";
          window.__annolidSpans = [];
          window.__annolidSpanCounter = 0;
          window.__annolidSpanMeta = {};
          window.__annolidLinkTargets = {};
          window.__annolidLinkTargetCounter = 0;
          window.__annolidRenderedPages = 0;
          const pages = window.__annolidPages || {};
          Object.keys(pages).forEach((key) => {
            const state = pages[key];
            if (!state) return;
            state.pageDiv = null;
            state.width = 0;
            state.height = 0;
            state.dpr = 1;
            state.ttsCanvas = null;
            state.ttsCtx = null;
            state.markCanvas = null;
            state.markCtx = null;
          });
          window.__annolidParagraphsByPage = {};
          window.__annolidParagraphOffsets = {};
          window.__annolidParagraphTotal = 0;
          window.__annolidParagraphs = [];
          renderFailureCounts = {};

          _annolidUpdateNavState();
          await _annolidEnsureRenderedThrough(anchor.pageNum);
          _annolidScrollToPage(anchor.pageNum, anchor.offsetFrac);
          _annolidUpdateNavState();
        } finally {
          zoomBusy = false;
          if (pendingZoom) {
            const next = pendingZoom;
            pendingZoom = null;
            _annolidRerenderAll(next.scale, next.rotation);
          }
        }
      }

      async function _annolidZoomFitWidth() {
        if (!container) return;
        try {
          const page = await pdf.getPage(1);
          const baseViewport = page.getViewport({ scale: 1, rotation });
          const gutter = 32;
          const available = Math.max(100, container.clientWidth - gutter);
          const target = available / Math.max(1, baseViewport.width);
          await _annolidRerenderAll(target, rotation);
        } catch (e) {
          console.warn("Zoom fit failed", e);
        }
      }

      function _annolidZoomBy(factor) {
        const next = _annolidClampScale(scale * factor);
        _annolidRerenderAll(next, rotation);
      }

      async function _annolidPrintPdf() {
        try {
          const url = await _annolidGetPdfObjectUrl();
          const iframe = document.createElement("iframe");
          iframe.style.position = "fixed";
          iframe.style.right = "0";
          iframe.style.bottom = "0";
          iframe.style.width = "1px";
          iframe.style.height = "1px";
          iframe.style.border = "0";
          iframe.src = url || pdfUrl;
          iframe.onload = () => {
            try {
              iframe.contentWindow.focus();
              iframe.contentWindow.print();
            } catch (e) {
              window.print();
            }
            setTimeout(() => {
              try { iframe.remove(); } catch (e) { }
            }, 1500);
          };
          document.body.appendChild(iframe);
        } catch (e) {
          window.print();
        }
      }

      if (prevPageBtn) prevPageBtn.addEventListener("click", () => _annolidGoToPage(_annolidGetCurrentPageNum() - 1));
      if (nextPageBtn) nextPageBtn.addEventListener("click", () => _annolidGoToPage(_annolidGetCurrentPageNum() + 1));
      if (pageInput) {
        pageInput.addEventListener("keydown", (ev) => {
          if (ev.key === "Enter") {
            ev.preventDefault();
            _annolidGoToPage(pageInput.value);
          }
        });
        pageInput.addEventListener("change", () => _annolidGoToPage(pageInput.value));
      }
      if (zoomOutBtn) zoomOutBtn.addEventListener("click", () => _annolidZoomBy(1 / 1.1));
      if (zoomInBtn) zoomInBtn.addEventListener("click", () => _annolidZoomBy(1.1));
      if (zoomResetBtn) zoomResetBtn.addEventListener("click", () => _annolidRerenderAll(1.0));
      if (zoomFitBtn) zoomFitBtn.addEventListener("click", _annolidZoomFitWidth);
      if (rotateBtn) rotateBtn.addEventListener("click", () => _annolidRerenderAll(scale, rotation + 90));
      if (printBtn) printBtn.addEventListener("click", () => _annolidPrintPdf().catch(() => { }));
      _annolidUpdateNavState();
      if (container) {
        container.addEventListener("dblclick", async (ev) => {
          if (!window.__annolidReaderEnabled) return;
          if (!window.__annolidBridge || typeof window.__annolidBridge.onParagraphClicked !== "function") return;
          if (window.__annolidMarks && window.__annolidMarks.tool && window.__annolidMarks.tool !== "select") return;
          const sel = window.getSelection ? window.getSelection() : null;
          // Capture double-click selection to start reading from the exact word position.
          let selectedText = "";
          let selSpanEl = null;
          let selSpanIdx = -1;
          let selCharOffset = 0;
          let selOffsetIsChar = false;
          try {
            if (sel && !sel.isCollapsed && sel.rangeCount) {
              selectedText = String(sel.toString() || "").trim();
              const range = sel.getRangeAt(0);
              let node = range ? range.startContainer : null;
              let el = null;
              selOffsetIsChar = !!(node && node.nodeType === 3);
              if (node && node.nodeType === 3) el = node.parentElement;
              else if (node && node.nodeType === 1) el = node;
              if (el && el.closest) {
                const closestSpan = el.closest(".textLayer span");
                if (closestSpan) {
                  selSpanEl = closestSpan;
                  if (closestSpan.dataset && closestSpan.dataset.annolidIndex) {
                    selSpanIdx = parseInt(closestSpan.dataset.annolidIndex, 10);
                  }
                }
              }
              selCharOffset = (
                selOffsetIsChar && range && typeof range.startOffset === "number"
              ) ? range.startOffset : 0;
            }
          } catch (e) {
            selectedText = "";
            selSpanEl = null;
            selSpanIdx = -1;
            selCharOffset = 0;
            selOffsetIsChar = false;
          }
          // Allow double-click to trigger reading even if the browser selects a word.
          if (ev.type !== "dblclick" && sel && !sel.isCollapsed) return;
          if (ev.type === "dblclick" && sel && !sel.isCollapsed) {
            try { sel.removeAllRanges(); } catch (e) { }
          }
          const pageDiv = ev.target && ev.target.closest ? ev.target.closest(".page") : null;
          if (!pageDiv) return;
          const pageNum = parseInt(pageDiv.getAttribute("data-page-number") || "0", 10);
          if (!pageNum) return;
          let clickOffsetFrac = 0.0;
          try {
            const pageRect = pageDiv.getBoundingClientRect();
            const y = ev.clientY - pageRect.top;
            clickOffsetFrac = (pageRect.height > 0) ? Math.max(0, Math.min(1, y / pageRect.height)) : 0;
          } catch (e) {
            clickOffsetFrac = 0.0;
          }

          let pageParas = window.__annolidParagraphsByPage[String(pageNum)];
          if (!pageParas || !pageParas.length) {
            pageParas = _annolidBuildParagraphsForPage(pageNum);
            _annolidRebuildParagraphIndex();
          }

          let spanIdx = -1;
          let spanEl = ev.target && ev.target.closest ? ev.target.closest(".textLayer span") : null;
          if (selSpanIdx >= 0 && selSpanEl) {
            spanIdx = selSpanIdx;
            spanEl = selSpanEl;
          } else if (spanEl && spanEl.dataset && spanEl.dataset.annolidIndex) {
            spanIdx = parseInt(spanEl.dataset.annolidIndex, 10);
          }
          let paraIndex = -1;
          if (spanIdx >= 0) {
            paraIndex = _annolidFindParagraphIndexBySpan(pageNum, spanIdx);
          }
          if (paraIndex < 0) {
            const pageRect = pageDiv.getBoundingClientRect();
            const x = ev.clientX - pageRect.left;
            const y = ev.clientY - pageRect.top;
            paraIndex = _annolidFindParagraphIndexByPoint(pageNum, x, y);
            if (paraIndex < 0) {
              // If textLayer spans are missing, fall back to PDF text extraction.
              await _annolidBuildTextParagraphsForPage(pageNum);
              await _annolidEnsureParagraphsFrom(pageNum);
              const rebuilt = window.__annolidParagraphsByPage[String(pageNum)] || [];
              if (rebuilt.length) {
                const frac = pageRect.height > 0 ? Math.max(0, Math.min(1, y / pageRect.height)) : 0;
                paraIndex = Math.max(0, Math.min(rebuilt.length - 1, Math.floor(frac * rebuilt.length)));
              }
            }
          }
          if (paraIndex < 0) return;
          await _annolidEnsureParagraphsFrom(pageNum);
          const offset = window.__annolidParagraphOffsets[pageNum] || 0;
          const startIndex = offset + paraIndex;
          const remaining = window.__annolidParagraphs.slice(startIndex).map((p) => ({
            text: p.text || "",
            spans: p.spans || [],
            pageNum: (parseInt(p.pageNum || p.page || 0, 10) || pageNum),
          }));
          if (!remaining.length) return;
          try {
            const bridge = window.__annolidBridge;
            if (bridge && typeof bridge.logEvent === "function") {
              const anchor = (typeof _annolidGetScrollAnchor === "function") ? _annolidGetScrollAnchor() : null;
              bridge.logEvent({
                type: "dblclick_read",
                label: "Read (double click)",
                pageNum,
                offsetFrac: (clickOffsetFrac || (anchor ? anchor.offsetFrac : 0) || 0),
                snippet: selectedText || (remaining[0] && remaining[0].text ? String(remaining[0].text).slice(0, 160) : ""),
              });
            }
          } catch (e) { }
          const sentences = [];
          const targetSpanIdx = (spanIdx >= 0 && Number.isInteger(spanIdx)) ? spanIdx : null;
          let sentenceCursor = 0;
          const splitFallback = (text) => {
            const normalized = _annolidNormalizeText(text || "");
            if (!normalized) return [];
            const out = [];
            const ranges = (typeof window.__annolidSplitTextIntoSentenceRanges === "function")
              ? (window.__annolidSplitTextIntoSentenceRanges(normalized) || [])
              : [];
            for (const r of ranges) {
              const seg = _annolidNormalizeText(normalized.slice(r[0], r[1]));
              if (seg) out.push(seg);
            }
            return out.length ? out : [normalized];
          };
          let startSentenceIdx = 0;
          remaining.forEach((p, idx) => {
            const isTargetParagraph = (idx === 0);
            const paraSentences = [];
            if (typeof window.__annolidSplitParagraphIntoSentences === "function") {
              const splits = window.__annolidSplitParagraphIntoSentences(p) || [];
              if (splits.length) {
                splits.forEach((s) => paraSentences.push(s));
              }
            }
            const pageForPara = (parseInt(p.pageNum || p.page || 0, 10) || pageNum);
            if (!paraSentences.length) {
              const segs = splitFallback(p.text || "");
              if (segs.length) {
                segs.forEach((seg) => {
                  paraSentences.push({
                    text: seg,
                    spans: [],
                    pageNum: pageForPara,
                  });
                });
              }
            }
            if (!paraSentences.length) {
              paraSentences.push({
                text: p.text || "",
                spans: [],
                pageNum: pageForPara,
              });
            }

            if (isTargetParagraph && paraSentences.length) {
              let matched = 0;
              if (targetSpanIdx != null) {
                for (let i = 0; i < paraSentences.length; i++) {
                  const spans = Array.isArray(paraSentences[i].spans) ? paraSentences[i].spans : [];
                  if (spans.indexOf(targetSpanIdx) >= 0) {
                    matched = i;
                    break;
                  }
                }
              }
              startSentenceIdx = sentenceCursor + matched;
            }

            paraSentences.forEach((s) => sentences.push(s));
            sentenceCursor += paraSentences.length;
          });

          const totalSentences = sentences.length;
          if (startSentenceIdx < 0 || startSentenceIdx >= totalSentences) {
            startSentenceIdx = 0;
          }
          // Start from the clicked word within the starting sentence when possible.
          if (totalSentences && targetSpanIdx != null) {
            try {
              const startSentence = sentences[startSentenceIdx];
              const spans = Array.isArray(startSentence.spans) ? startSentence.spans : [];
              const pos = spans.indexOf(targetSpanIdx);
              if (pos >= 0) {
                const nodes = window.__annolidSpans || [];
                const trimmedSpans = spans.slice(pos);
                const parts = [];
                for (let i = 0; i < trimmedSpans.length; i++) {
                  const spanIndex = trimmedSpans[i];
                  const node = nodes[spanIndex];
                  if (!node) continue;
                  let raw = String(node.textContent || "");
                  if (i === 0) {
                    // Prefer the exact char offset from DOM selection, otherwise best-effort locate the selected word.
                    let cut = (selSpanIdx === targetSpanIdx && selOffsetIsChar)
                      ? (parseInt(selCharOffset, 10) || 0)
                      : 0;
                    if ((!selOffsetIsChar || cut >= raw.length) && selectedText) {
                      const idx2 = raw.toLowerCase().indexOf(String(selectedText).toLowerCase());
                      if (idx2 >= 0) cut = idx2;
                    }
                    cut = Math.max(0, Math.min(raw.length, cut));
                    raw = raw.slice(cut);
                  }
                  const t = _annolidNormalizeText(raw);
                  if (t) parts.push(t);
                }
                const rebuilt = _annolidNormalizeText(parts.join(" "));
                if (rebuilt) {
                  startSentence.text = rebuilt;
                  startSentence.spans = trimmedSpans;
                }
              }
            } catch (e) { }
          }

          window.__annolidBridge.onParagraphClicked({
            startIndex,
            total: window.__annolidParagraphTotal || (startIndex + remaining.length),
            paragraphs: remaining,
            sentences,
            sentenceStartIndex: 0,
            sentenceLocalStartIndex: startSentenceIdx,
            sentenceTotal: totalSentences,
          });
        });
      }

      async function renderPage(pageNum, epoch) {
        if (epoch !== renderEpoch) return;
        const page = await pdf.getPage(pageNum);
        const viewport = page.getViewport({ scale, rotation });

        const pageDiv = document.createElement("div");
        pageDiv.className = "page";
        pageDiv.setAttribute("data-page-number", String(pageNum));
        pageDiv.style.width = viewport.width + "px";
        pageDiv.style.height = viewport.height + "px";

        const canvas = document.createElement("canvas");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        pageDiv.appendChild(canvas);

        const textLayerDiv = document.createElement("div");
        textLayerDiv.className = "textLayer";
        pageDiv.appendChild(textLayerDiv);

        const annotationLayerDiv = document.createElement("div");
        annotationLayerDiv.className = "annotationLayer";
        annotationLayerDiv.style.pointerEvents = "none";
        pageDiv.appendChild(annotationLayerDiv);

        const ttsLayer = document.createElement("canvas");
        ttsLayer.className = "annolid-tts-layer";
        pageDiv.appendChild(ttsLayer);

        const markLayer = document.createElement("canvas");
        markLayer.className = "annolid-mark-layer";
        pageDiv.appendChild(markLayer);

        if (epoch !== renderEpoch) return;
        container.appendChild(pageDiv);

        const ctx = canvas.getContext("2d");
        if (epoch !== renderEpoch) return;
        await page.render({ canvasContext: ctx, viewport }).promise;
        window.__annolidRenderedPages = (window.__annolidRenderedPages || 0) + 1;

        try {
          const annots = await page.getAnnotations({ intent: "display" });
          annotationLayerDiv.innerHTML = "";
          (annots || []).forEach((a) => {
            if (!a || a.subtype !== "Link" || !a.rect) return;
            let dest = a.dest || null;
            const url = a.url || a.unsafeUrl || null;
            const action = a.action || null;
            if (!dest && action && typeof action === "object") {
              dest = action.dest || action.destination || null;
            }
            if (!dest && !url && !action) return;
            let rect = null;
            try {
              const vrect = viewport.convertToViewportRectangle(a.rect);
              const x1 = Math.min(vrect[0], vrect[2]);
              const x2 = Math.max(vrect[0], vrect[2]);
              const y1 = Math.min(vrect[1], vrect[3]);
              const y2 = Math.max(vrect[1], vrect[3]);
              rect = { left: x1, top: y1, width: Math.max(1, x2 - x1), height: Math.max(1, y2 - y1) };
            } catch (e) {
              rect = null;
            }
            if (!rect) return;
            const id = String(window.__annolidLinkTargetCounter || 0);
            window.__annolidLinkTargetCounter = (window.__annolidLinkTargetCounter || 0) + 1;
            window.__annolidLinkTargets[id] = { dest, url, action, pageNum };
            const link = document.createElement("a");
            link.setAttribute("data-annolid-link-id", id);
            link.style.pointerEvents = "auto";
            link.style.left = rect.left + "px";
            link.style.top = rect.top + "px";
            link.style.width = rect.width + "px";
            link.style.height = rect.height + "px";
            link.href = url ? String(url) : "#";
            if (url) link.target = "_blank";
            link.addEventListener("click", async (ev) => {
              try {
                const tool = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool : "select";
                if (tool !== "select") return;
                const payload = window.__annolidLinkTargets[id] || null;
                if (!payload || !payload.dest) return; // allow external links
                ev.preventDefault();
                ev.stopPropagation();
                const span = _annolidFindUnderlyingTextSpan(link, ev.clientX, ev.clientY);
                const cite = _annolidFindCitationFromSpan(span, ev.clientX, ev.clientY);
                if (cite) {
                  if (cite.kind === "numeric") {
                    const group = _annolidFindNumericCitationGroupFromSpan(span, ev.clientX, ev.clientY) || {
                      raw: "[" + String(cite.number || "") + "]",
                      numbers: [parseInt(cite.number || 0, 10)],
                      pickedIndex: 0,
                    };
                    await _annolidOpenScholarForCitationGroup(group, span || link);
                    return;
                  }
                  await _annolidOpenScholarForAuthorYear(cite, span || link);
                  return;
                }
                await _annolidOpenDestinationPreview(payload.dest, "Link preview");
              } catch (e) { }
            });
            link.addEventListener("mouseenter", (ev) => {
              try {
                const tool = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool : "select";
                if (tool !== "select") return;
                const payload = window.__annolidLinkTargets[id] || null;
                if (!payload || !payload.dest) return;
                const span = _annolidFindUnderlyingTextSpan(link, ev.clientX, ev.clientY);
                _annolidScheduleCitationHover(span, payload.dest, true, 200, ev.clientX, ev.clientY);
              } catch (e) { }
            });
            link.addEventListener("mouseleave", () => {
              try {
                _annolidCancelCitationHover();
                _annolidScheduleCitationClose();
              } catch (e) { }
            });
            annotationLayerDiv.appendChild(link);
          });
        } catch (e) {
          try { annotationLayerDiv.innerHTML = ""; } catch (e2) { }
        }

        try {
          const textContent = await page.getTextContent();
          if (pdfjsLib.renderTextLayer) {
            const task = pdfjsLib.renderTextLayer({
              textContent,
              container: textLayerDiv,
              viewport,
              textDivs: [],
              enhanceTextSelection: false,
            });
            if (task && task.promise) {
              await task.promise;
            }
          }
        } catch (e) {
          console.warn("PDF.js text layer failed", e);
        }
        const newSpans = Array.from(textLayerDiv.querySelectorAll("span"));
        newSpans.forEach((span) => {
          if (!span.dataset.annolidIndex) {
            const nextIdx = window.__annolidSpanCounter || 0;
            span.dataset.annolidIndex = String(nextIdx);
            window.__annolidSpanCounter = nextIdx + 1;
          }
          const idx = parseInt(span.dataset.annolidIndex, 10);
          if (!Number.isNaN(idx)) {
            window.__annolidSpans[idx] = span;
          }
        });
        _annolidBuildParagraphsForPage(pageNum);
        _annolidRebuildParagraphIndex();
        try {
          const state = _annolidGetPageState(pageNum);
          const ttsSetup = _annolidSetupHiDpiCanvas(ttsLayer, viewport.width, viewport.height);
          const markSetup = _annolidSetupHiDpiCanvas(markLayer, viewport.width, viewport.height);
          state.pageDiv = pageDiv;
          state.width = viewport.width;
          state.height = viewport.height;
          state.dpr = markSetup.dpr || 1;
          state.ttsCanvas = ttsLayer;
          state.ttsCtx = ttsSetup.ctx;
          state.markCanvas = markLayer;
          state.markCtx = markSetup.ctx;
          _annolidBindMarkCanvas(state);
          // Apply current tool mode to this newly created mark layer.
          const drawing = (window.__annolidMarks && window.__annolidMarks.tool) ? window.__annolidMarks.tool !== "select" : false;
          markLayer.style.pointerEvents = drawing ? "auto" : "none";
          markLayer.style.cursor = drawing ? "crosshair" : "default";
          _annolidRenderMarks(pageNum);
          if (window.__annolidRenderTts) window.__annolidRenderTts();
        } catch (e) {
          console.warn("Annolid page layer init failed", e);
        }
      }

      function renderMore(maxCount) {
        return _annolidQueueRender(async (epoch) => {
          let count = 0;
          while (nextPage <= total && count < maxCount) {
            const progressed = await _annolidTryRenderNextPage(epoch, 2);
            if (!progressed) break;
            count += 1;
            await new Promise(r => setTimeout(r, 0));
          }
        });
      }

      function _annolidScheduleBackgroundRender(startDelayMs) {
        const myEpoch = renderEpoch;
        const delay = Math.max(0, parseInt(startDelayMs, 10) || 0);
        const tick = () => {
          if (myEpoch !== renderEpoch) return;
          if (nextPage > total) return;
          renderMore(2).then(() => {
            if (myEpoch !== renderEpoch) return;
            if (nextPage > total) return;
            setTimeout(tick, 120);
          }).catch(() => {
            if (myEpoch !== renderEpoch) return;
            setTimeout(tick, 300);
          });
        };
        setTimeout(tick, delay);
      }

      await renderMore(2);
      await _annolidResumeFromSavedState(false);
      _annolidScheduleBackgroundRender(300);
      _annolidUpdatePersistenceButtons();
      if (container) {
        let scrollScheduled = false;
        container.addEventListener("scroll", () => {
          if (scrollScheduled) return;
          scrollScheduled = true;
          requestAnimationFrame(() => {
            scrollScheduled = false;
            _annolidUpdateNavState();
            const nearBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 600;
            if (nearBottom) {
              renderMore(2);
            }
          });
        });
      }
    } catch (err) {
      console.error("PDF.js render failed", err);
      try {
        const msg = (err && err.message) ? err.message : String(err);
        document.body.setAttribute("data-pdfjs-error", msg);
      } catch (e) {
        document.body.setAttribute("data-pdfjs-error", "PDF.js render failed");
      }
    }
  });
})();
