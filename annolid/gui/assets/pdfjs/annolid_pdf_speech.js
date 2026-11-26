// Annolid PDF.js text layout, sentence parsing, and keyword/word speech highlighting helpers.
// Pure logic module: runs in QtWebEngine/browsers (window.AnnolidPdfSpeech) and standalone JS runtimes.

(function (globalScope) {
  "use strict";

  const SENTENCE_END = new Set([".", "!", "?", "。", "！", "？"]);
  const QUOTE_CLOSERS = new Set(["”", "’", "\"", "'"]);
  const CLOSERS = new Set([
    "”", "’", "\"", "'", ")", "]", "}", "）", "】", "》", "」", "』", "〉",
  ]);
  const NON_BREAKING_ABBREVS = new Set([
    "al", "approx", "art", "ca", "cf", "ch", "chap", "co", "corp", "dept",
    "dr", "ed", "eds", "eg", "eq", "eqs", "etc", "fig", "figs", "ie", "inc",
    "jr", "ltd", "mr", "mrs", "ms", "no", "nos", "p", "pp", "prof", "ref",
    "refs", "rev", "sec", "secs", "sr", "st", "supp", "tab", "vol", "vols", "vs",
  ]);
  const STRONG_ABBREVS = new Set([
    "dr", "mr", "mrs", "ms", "prof", "st", "fig", "figs", "eq", "eqs",
    "sec", "secs", "ch", "chap", "ref", "refs", "no", "nos", "vol", "vols",
    "tab", "art", "supp",
  ]);
  const STOPWORDS = new Set([
    "about", "above", "after", "again", "against", "all", "also", "among",
    "and", "any", "are", "because", "been", "before", "being", "below",
    "between", "both", "but", "can", "could", "did", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "here", "hers", "herself", "him", "himself", "his", "how",
    "into", "its", "itself", "just", "may", "might", "more", "most", "much",
    "must", "nor", "not", "now", "off", "once", "only", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "show", "shown", "shows", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "under", "until", "used", "using", "very", "was",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "within", "without", "would", "your", "yours",
  ]);
  const MAJOR_PAUSE = new Set([".", "!", "?", "。", "！", "？", ";", "；", ":", "："]);
  const MINOR_PAUSE = new Set([",", "，", "、", "—", "–", "-", ")", "]", "）", "】", "》"]);

  const CJK_LEAD_RE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;
  const CJK_LIKE_RE = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af，。！？、；：“”‘’（）《》「」『』]/;
  const WORD_TOKEN_RE = /[A-Za-z0-9\u00c0-\u024f\u0400-\u04ff]+(?:[-'\u2019][A-Za-z0-9\u00c0-\u024f\u0400-\u04ff]+)*|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]/g;

  function isCjkChar(ch) {
    return !!(ch && CJK_LEAD_RE.test(ch));
  }

  function isCjkLike(ch) {
    return !!(ch && CJK_LIKE_RE.test(ch));
  }

  function isNarrationLead(ch) {
    const c = String(ch || "");
    return isCjkChar(c) || c === "（" || c === "【" || c === "“" || c === "‘";
  }

  function isAsciiAlpha(ch) {
    return !!(ch && /^[A-Za-z]$/.test(ch));
  }

  function isAsciiAlnum(ch) {
    return !!(ch && /^[A-Za-z0-9]$/.test(ch));
  }

  function isDigit(ch) {
    return !!(ch && /^[0-9]$/.test(ch));
  }

  function normalizeText(text) {
    return String(text || "")
      .replace(/\u00ad/g, "")
      .replace(/\u00a0/g, " ")
      .replace(/\u2029/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function joinTextSegments(prevText, nextText) {
    const prev = String(prevText || "");
    const next = String(nextText || "");
    if (!prev) return next;
    if (!next) return prev;
    const prevLast = prev[prev.length - 1] || "";
    const nextFirst = next[0] || "";
    if (/\s/.test(prevLast) || /\s/.test(nextFirst)) {
      return prev + next;
    }
    if (isCjkLike(prevLast) && isCjkLike(nextFirst)) {
      return prev + next;
    }
    return prev + " " + next;
  }

  function mergeHyphenated(prevText, nextText) {
    const prev = String(prevText || "").trimEnd();
    const nextTrim = String(nextText || "").trimStart();
    if (!nextTrim) return normalizeText(prev);
    if (!prev) return normalizeText(nextTrim);
    const m = /([A-Za-z]{2,})[-\u2010\u2011\u2012\u2013]$/.exec(prev);
    if (m && /^[a-z]/.test(nextTrim)) {
      return normalizeText(prev.replace(/[-\u2010\u2011\u2012\u2013]$/, "") + nextTrim);
    }
    return normalizeText(joinTextSegments(prev, nextTrim));
  }

  function median(values) {
    if (!values || !values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }

  function quantile(sortedArr, q) {
    if (!sortedArr || !sortedArr.length) return 0;
    const pos = Math.max(0, Math.min(sortedArr.length - 1, Math.round(q * (sortedArr.length - 1))));
    return sortedArr[pos];
  }

  function mean(arr) {
    if (!arr || !arr.length) return 0;
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    return sum / arr.length;
  }

  function std(arr, avg) {
    if (!arr || !arr.length) return 0;
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      const d = arr[i] - avg;
      sum += d * d;
    }
    return Math.sqrt(sum / arr.length);
  }

  function isNonBreakingPeriod(text, dotIndex) {
    if (!text || dotIndex < 0 || dotIndex >= text.length || text[dotIndex] !== ".") {
      return false;
    }
    const prev = dotIndex > 0 ? text[dotIndex - 1] : "";
    const next = dotIndex + 1 < text.length ? text[dotIndex + 1] : "";

    // Decimals & version numbers: 3.14, 0.05, v1.2.3
    if (isDigit(prev) && isDigit(next)) return true;
    if (isDigit(next) && (!prev || !isAsciiAlnum(prev))) return true;

    // Inline dotted identifiers / URLs / filenames: example.org, file.pdf
    if (isAsciiAlnum(prev) && isAsciiAlnum(next)) return true;

    // Dotted acronyms: e.g., i.e., U.S.A., Ph.D.
    if (dotIndex >= 2 && text[dotIndex - 2] === "." && isAsciiAlpha(prev)) return true;
    if (
      dotIndex + 2 < text.length &&
      isAsciiAlpha(next) &&
      text[dotIndex + 2] === "." &&
      isAsciiAlpha(prev)
    ) {
      return true;
    }

    let tokStart = dotIndex;
    while (tokStart > 0 && isAsciiAlpha(text[tokStart - 1])) tokStart--;
    const token = text.slice(tokStart, dotIndex);
    if (!token) return false;

    // Single-letter initials: A. Smith, J. B. Chen
    if (token.length === 1 && /^[A-Z]$/.test(token)) {
      if (tokStart === 0 || !isAsciiAlnum(text[tokStart - 1])) {
        let probe = dotIndex + 1;
        while (probe < text.length && /\s/.test(text[probe])) probe++;
        if (probe < text.length && /^[A-Z]$/.test(text[probe])) {
          return true;
        }
      }
    }

    const lowered = token.toLowerCase();
    if (NON_BREAKING_ABBREVS.has(lowered)) {
      let probe = dotIndex + 1;
      while (probe < text.length && /\s/.test(text[probe])) probe++;
      if (probe >= text.length) return false;
      const after = text[probe];
      if (lowered === "al") {
        if (
          after === "," || after === ";" || after === ":" ||
          after === ")" || after === "]" || after === "(" ||
          after === "[" || isDigit(after) || /^[a-z]$/.test(after)
        ) {
          return true;
        }
        return false;
      }
      if (
        after === "," || after === ";" || after === ":" ||
        after === "(" || after === "[" || isDigit(after) || /^[a-z]$/.test(after)
      ) {
        return true;
      }
      if (STRONG_ABBREVS.has(lowered)) {
        return true;
      }
    }
    return false;
  }

  function splitTextIntoSentenceRanges(text) {
    const s = String(text || "");
    if (!s) return [];
    const ranges = [];
    let start = 0;
    const n = s.length;
    for (let i = 0; i < n; i++) {
      const ch = s[i];
      if (!SENTENCE_END.has(ch)) continue;
      if (ch === "." && isNonBreakingPeriod(s, i)) continue;

      let end = i + 1;
      while (end < n && SENTENCE_END.has(s[end])) end++;
      const punctEnd = end;

      // Ellipsis followed by lowercase continuation stays inside the sentence.
      if (punctEnd - i >= 2 && /^\.+$/.test(s.slice(i, punctEnd))) {
        let probeEll = punctEnd;
        while (probeEll < n && /\s/.test(s[probeEll])) probeEll++;
        if (probeEll < n && /^[a-z]$/.test(s[probeEll])) {
          i = punctEnd - 1;
          continue;
        }
      }

      // Attach trailing closing quotes/brackets even across extracted whitespace.
      let probe = end;
      for (let guard = 0; guard < 8; guard++) {
        let ws = probe;
        while (ws < n && /\s/.test(s[ws])) ws++;
        if (ws < n && CLOSERS.has(s[ws])) {
          probe = ws + 1;
          while (probe < n && CLOSERS.has(s[probe])) probe++;
          end = probe;
          continue;
        }
        break;
      }

      // Glue CJK quoted speech to immediately-following narration ("。”说完...").
      let hasQuoteCloser = false;
      for (let j = punctEnd; j < end; j++) {
        const c = s[j];
        if (!/\s/.test(c) && QUOTE_CLOSERS.has(c)) {
          hasQuoteCloser = true;
          break;
        }
      }
      if (hasQuoteCloser) {
        let k = end;
        while (k < n && /\s/.test(s[k]) && (k - end) < 48) k++;
        if (k < n && isNarrationLead(s[k])) {
          i = end - 1;
          continue;
        }
      }

      ranges.push([start, end]);
      start = end;
      i = end - 1;
    }
    if (start < n) ranges.push([start, n]);

    const out = [];
    for (let r = 0; r < ranges.length; r++) {
      let a = ranges[r][0];
      let b = ranges[r][1];
      while (a < b && /\s/.test(s[a])) a++;
      while (b > a && /\s/.test(s[b - 1])) b--;
      if (b > a) out.push([a, b]);
    }
    return out;
  }

  function orderSpanIndicesForReading(indices, getMeta) {
    const raw = Array.isArray(indices) ? indices : [];
    const unique = [];
    const seen = new Set();
    for (let i = 0; i < raw.length; i++) {
      const idx = raw[i];
      if (!Number.isInteger(idx) || idx < 0 || seen.has(idx)) continue;
      seen.add(idx);
      unique.push(idx);
    }
    if (unique.length <= 1 || typeof getMeta !== "function") return unique;

    const annotated = unique.map((idx, pos) => ({ idx, pos, meta: getMeta(idx) }));
    const heights = annotated
      .map((a) => (a.meta && isFinite(a.meta.h) && a.meta.h > 0) ? a.meta.h : 0)
      .filter((h) => h > 0);
    const medH = median(heights) || 12;
    const sameRowTol = Math.max(2, medH * 0.45);

    annotated.sort((a, b) => {
      const ma = a.meta;
      const mb = b.meta;
      if (!ma || !mb) return a.pos - b.pos;
      if ((ma.pageNum || 0) !== (mb.pageNum || 0)) {
        return (ma.pageNum || 0) - (mb.pageNum || 0);
      }
      const ay = ma.y + (ma.h || 0) * 0.5;
      const by = mb.y + (mb.h || 0) * 0.5;
      if (Math.abs(ay - by) <= sameRowTol) {
        if (Math.abs(ma.x - mb.x) > 0.5) return ma.x - mb.x;
      }
      if (Math.abs(ay - by) > sameRowTol) return ay - by;
      return a.pos - b.pos;
    });
    return annotated.map((a) => a.idx);
  }

  function splitParagraphEntriesIntoSentences(spanEntries, pageNum) {
    const entries = Array.isArray(spanEntries) ? spanEntries : [];
    if (!entries.length) return [];

    // Build exact character map from combined string back to each span's raw text offsets.
    const parts = [];
    let combined = "";
    for (let i = 0; i < entries.length; i++) {
      const item = entries[i];
      if (!item || !Number.isInteger(item.idx)) continue;
      const raw = String(item.rawText != null ? item.rawText : (item.text || ""));
      // Trim only outer whitespace while recording its start offset inside raw.
      const trimmedLeadMatch = /^\s*/.exec(raw);
      const leadTrim = trimmedLeadMatch ? trimmedLeadMatch[0].length : 0;
      const cleanedCore = raw.slice(leadTrim).trimEnd();
      if (!cleanedCore) continue;
      const coreRawEnd = leadTrim + cleanedCore.length;

      // Join across spans, de-hyphenating word wraps cleanly while keeping char mapping.
      let sep = "";
      let dropPrevHyphen = false;
      if (combined) {
        const prevTrim = combined.trimEnd();
        const prevLast = prevTrim[prevTrim.length - 1] || "";
        const nextFirst = cleanedCore[0] || "";
        if (
          /([A-Za-z]{2,})[-\u2010\u2011\u2012\u2013]$/.test(prevTrim) &&
          /^[a-z]/.test(nextFirst)
        ) {
          dropPrevHyphen = true;
          combined = prevTrim.replace(/[-\u2010\u2011\u2012\u2013]$/, "");
          if (parts.length) {
            parts[parts.length - 1].end = combined.length;
          }
        } else if (!/\s/.test(prevLast) && !(isCjkLike(prevLast) && isCjkLike(nextFirst))) {
          sep = " ";
        }
      }

      const displayText = cleanedCore.replace(/\u00ad/g, "").replace(/\u00a0/g, " ");
      const start = combined.length + sep.length;
      combined += sep + displayText;
      const end = combined.length;
      parts.push({
        idx: item.idx,
        start,
        end,
        rawStart: leadTrim,
        rawEnd: coreRawEnd,
        rawLen: raw.length,
        displayLen: displayText.length,
        joinedHyphen: dropPrevHyphen,
      });
    }

    if (!parts.length) return [];
    const resolvedPageNum = parseInt(pageNum || 0, 10) || 0;
    const allSpans = parts.map((p) => p.idx);
    const normalizedFull = normalizeText(combined);
    if (!normalizedFull) return [];

    const ranges = splitTextIntoSentenceRanges(combined);
    if (!ranges.length) {
      return [{
        text: normalizedFull,
        spans: allSpans,
        spanSlices: parts.map((p) => ({
          idx: p.idx,
          startOffset: p.rawStart,
          endOffset: p.rawEnd,
        })),
        pageNum: resolvedPageNum,
      }];
    }

    const sentences = [];
    for (let r = 0; r < ranges.length; r++) {
      const sStart = ranges[r][0];
      const sEnd = ranges[r][1];
      const segText = normalizeText(combined.slice(sStart, sEnd));
      if (!segText) continue;
      const groupSpans = [];
      const spanSlices = [];
      for (let j = 0; j < parts.length; j++) {
        const p = parts[j];
        if (p.end <= sStart || p.start >= sEnd) continue;
        const localStart = Math.max(0, sStart - p.start);
        const localEnd = Math.min(p.displayLen, sEnd - p.start);
        const scale = p.displayLen > 0 ? ((p.rawEnd - p.rawStart) / p.displayLen) : 1;
        const rawOffsetStart = Math.max(0, Math.min(p.rawLen, p.rawStart + Math.floor(localStart * scale)));
        const rawOffsetEnd = Math.max(rawOffsetStart, Math.min(p.rawLen, p.rawStart + Math.ceil(localEnd * scale)));
        groupSpans.push(p.idx);
        spanSlices.push({
          idx: p.idx,
          startOffset: rawOffsetStart,
          endOffset: rawOffsetEnd,
        });
      }
      sentences.push({
        text: segText,
        spans: groupSpans.length ? groupSpans : allSpans.slice(),
        spanSlices: spanSlices.length
          ? spanSlices
          : parts.map((p) => ({ idx: p.idx, startOffset: p.rawStart, endOffset: p.rawEnd })),
        pageNum: resolvedPageNum,
      });
    }

    if (!sentences.length) {
      return [{
        text: normalizedFull,
        spans: allSpans,
        spanSlices: parts.map((p) => ({
          idx: p.idx,
          startOffset: p.rawStart,
          endOffset: p.rawEnd,
        })),
        pageNum: resolvedPageNum,
      }];
    }
    return sentences;
  }

  function escapeRegExp(str) {
    return String(str || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function parseKeywords(raw) {
    if (raw == null) return [];
    const pieces = [];
    if (Array.isArray(raw)) {
      for (let i = 0; i < raw.length; i++) {
        String(raw[i] || "").split(/[,;\n，；]+/).forEach((t) => pieces.push(t));
      }
    } else {
      String(raw).split(/[,;\n，；]+/).forEach((t) => pieces.push(t));
    }
    const out = [];
    const seen = new Set();
    for (let i = 0; i < pieces.length; i++) {
      const term = String(pieces[i] || "").replace(/\s+/g, " ").trim();
      if (!term) continue;
      const key = term.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(term);
    }
    return out;
  }

  function extractAutoKeywords(text, maxKeywords) {
    const source = String(text || "").trim();
    const limit = Math.max(0, parseInt(maxKeywords == null ? 5 : maxKeywords, 10) || 0);
    if (!source || !limit) return [];

    const scores = new Map();
    WORD_TOKEN_RE.lastIndex = 0;
    let m;
    let pos = 0;
    while ((m = WORD_TOKEN_RE.exec(source)) !== null) {
      const token = String(m[0] || "").replace(/^[-'\u2019]+|[-'\u2019]+$/g, "");
      pos += 1;
      if (!token || isCjkChar(token[0]) || /^\d+$/.test(token)) continue;
      const lowered = token.toLowerCase();
      if (STOPWORDS.has(lowered)) continue;
      const upperCount = (token.match(/[A-Z]/g) || []).length;
      const isAcronym = token.length >= 2 && upperCount >= 2;
      if (!isAcronym && token.length < 5) continue;

      let bonus = 0;
      if (isAcronym) bonus += 3.0;
      if (token.indexOf("-") >= 0) bonus += 1.5;
      if (token.length >= 8) bonus += 1.2;
      else if (token.length >= 6) bonus += 0.6;
      if (/^[A-Z]/.test(token) && pos > 1) bonus += 0.8;

      const prev = scores.get(lowered);
      if (prev) {
        prev.score += 1.0 + bonus;
      } else {
        scores.set(lowered, { score: 1.0 + bonus, display: token, firstPos: pos });
      }
    }

    const ranked = Array.from(scores.values()).sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.firstPos - b.firstPos;
    });
    return ranked.slice(0, limit).map((item) => item.display);
  }

  function findKeywordRanges(text, keywords) {
    const source = String(text || "");
    if (!source) return [];
    const terms = parseKeywords(keywords);
    if (!terms.length) return [];

    const ordered = terms.slice().sort((a, b) => b.length - a.length);
    const matches = [];
    for (let i = 0; i < ordered.length; i++) {
      const term = ordered[i];
      const hasCjk = CJK_LEAD_RE.test(term);
      const escaped = escapeRegExp(term).replace(/\s+/g, "\\s+");
      const re = new RegExp(escaped, "gi");
      let m;
      while ((m = re.exec(source)) !== null) {
        const start = m.index;
        const end = start + m[0].length;
        if (end <= start) continue;
        if (!hasCjk) {
          const leftOk = start === 0 || !/[A-Za-z0-9_]/.test(term[0]) || !/[A-Za-z0-9_]/.test(source[start - 1]);
          const rightOk = end >= source.length || !/[A-Za-z0-9_]/.test(term[term.length - 1]) || !/[A-Za-z0-9_]/.test(source[end]);
          if (!leftOk || !rightOk) continue;
        }
        matches.push({ start, end, keyword: term });
      }
    }
    if (!matches.length) return [];

    matches.sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return (b.end - b.start) - (a.end - a.start);
    });
    const resolved = [];
    let cursor = -1;
    for (let i = 0; i < matches.length; i++) {
      const item = matches[i];
      if (item.start < cursor) continue;
      resolved.push(item);
      cursor = item.end;
    }
    return resolved;
  }

  function tokenSpeechWeight(word, trailingText) {
    const text = String(word || "");
    if (!text) return 1.0;
    let cjkCount = 0;
    for (let i = 0; i < text.length; i++) {
      if (isCjkChar(text[i])) cjkCount += 1;
    }
    let weight = cjkCount > 0
      ? (cjkCount * 2.6 + (text.length - cjkCount) * 0.9)
      : Math.max(1.8, text.length * 0.85 + 0.8);

    const trail = String(trailingText || "");
    for (let i = 0; i < trail.length; i++) {
      const ch = trail[i];
      if (MAJOR_PAUSE.has(ch)) {
        weight += 3.2;
        break;
      }
      if (MINOR_PAUSE.has(ch)) {
        weight += 1.6;
        break;
      }
    }
    return Math.max(1.0, weight);
  }

  function trailingPauseStrength(trailingText) {
    const trailing = String(trailingText || "");
    for (let i = 0; i < trailing.length; i++) {
      const ch = trailing[i];
      if (MAJOR_PAUSE.has(ch)) return 2.0;
      if (MINOR_PAUSE.has(ch)) return 1.0;
    }
    return 0.0;
  }

  function tokenizeWords(text, keywords) {
    const source = String(text || "");
    if (!source) return [];
    const kwRanges = (keywords && keywords.length) ? findKeywordRanges(source, keywords) : [];
    const rawMatches = [];
    WORD_TOKEN_RE.lastIndex = 0;
    let m;
    while ((m = WORD_TOKEN_RE.exec(source)) !== null) {
      rawMatches.push({ start: m.index, end: m.index + m[0].length, text: m[0] });
    }
    const tokens = [];
    for (let i = 0; i < rawMatches.length; i++) {
      const cur = rawMatches[i];
      const nextStart = (i + 1 < rawMatches.length) ? rawMatches[i + 1].start : source.length;
      const trailing = source.slice(cur.end, nextStart);
      let isKeyword = false;
      for (let k = 0; k < kwRanges.length; k++) {
        if (kwRanges[k].start < cur.end && kwRanges[k].end > cur.start) {
          isKeyword = true;
          break;
        }
      }
      tokens.push({
        start: cur.start,
        end: cur.end,
        text: cur.text,
        weight: tokenSpeechWeight(cur.text, trailing),
        isKeyword,
        pauseStrength: trailingPauseStrength(trailing),
      });
    }
    return tokens;
  }

  function allocateWeightedDurationsMs(totalMs, weights) {
    const total = Math.round(Number(totalMs) || 0);
    if (total <= 0 || !Array.isArray(weights) || !weights.length) return [];
    const clean = weights.map((w) => Math.max(0.1, Number(w) || 1.0));
    let sumW = 0;
    for (let i = 0; i < clean.length; i++) sumW += clean[i];
    if (sumW <= 0) {
      const eq = Math.max(1, Math.floor(total / clean.length));
      return clean.map(() => eq);
    }
    const raw = clean.map((w) => total * (w / sumW));
    const durations = raw.map((x) => Math.max(1, Math.floor(x)));
    let curSum = 0;
    for (let i = 0; i < durations.length; i++) curSum += durations[i];
    const remainder = total - curSum;
    if (remainder === 0) return durations;

    const order = raw.map((x, idx) => ({ idx, frac: x - Math.floor(x) }));
    order.sort((a, b) => (remainder > 0) ? (b.frac - a.frac) : (a.frac - b.frac));
    let remaining = Math.abs(remainder);
    let step = 0;
    const maxIters = remaining + order.length * 2;
    while (remaining > 0 && step < maxIters && order.length) {
      const idx = order[step % order.length].idx;
      if (remainder > 0) {
        durations[idx] += 1;
        remaining -= 1;
      } else if (durations[idx] > 1) {
        durations[idx] -= 1;
        remaining -= 1;
      }
      step += 1;
    }
    return durations;
  }

  function normalizeSpeechSegments(rawSegments, totalMs) {
    if (!Array.isArray(rawSegments) || !rawSegments.length || totalMs <= 0) return [];
    const cleaned = [];
    for (let i = 0; i < rawSegments.length; i++) {
      const seg = rawSegments[i];
      if (!Array.isArray(seg) || seg.length < 2) continue;
      const s = Math.max(0, Math.min(totalMs, Math.round(Number(seg[0]) || 0)));
      const e = Math.max(0, Math.min(totalMs, Math.round(Number(seg[1]) || 0)));
      if (e > s) cleaned.push([s, e]);
    }
    if (!cleaned.length) return [];
    cleaned.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    const merged = [[cleaned[0][0], cleaned[0][1]]];
    for (let i = 1; i < cleaned.length; i++) {
      const prev = merged[merged.length - 1];
      if (cleaned[i][0] <= prev[1]) {
        prev[1] = Math.max(prev[1], cleaned[i][1]);
      } else {
        merged.push([cleaned[i][0], cleaned[i][1]]);
      }
    }
    return merged;
  }

  function selectClauseSplitIndices(weights, pauseStrengths, segments) {
    const kBreaks = segments.length - 1;
    const n = weights.length;
    if (kBreaks <= 0 || n <= kBreaks || pauseStrengths.length !== n) return null;
    const candidates = [];
    for (let i = 0; i < n - 1; i++) {
      if ((Number(pauseStrengths[i]) || 0) > 0) candidates.push(i);
    }
    if (candidates.length < kBreaks) return null;
    if (candidates.length === kBreaks) return candidates;

    let totalVoiced = 0;
    for (let i = 0; i < segments.length; i++) {
      totalVoiced += Math.max(1, segments[i][1] - segments[i][0]);
    }
    const targetFracs = [];
    let accV = 0;
    for (let i = 0; i < segments.length - 1; i++) {
      accV += Math.max(1, segments[i][1] - segments[i][0]);
      targetFracs.push(accV / totalVoiced);
    }

    let totalW = 0;
    for (let i = 0; i < n; i++) totalW += Math.max(0.1, Number(weights[i]) || 1.0);
    const cumW = [];
    let running = 0;
    for (let i = 0; i < n; i++) {
      running += Math.max(0.1, Number(weights[i]) || 1.0);
      cumW.push(running / totalW);
    }

    const chosen = [];
    let minIdx = 0;
    for (let b = 0; b < targetFracs.length; b++) {
      const remainingNeeded = kBreaks - b - 1;
      const valid = candidates.filter((c) => {
        if (c < minIdx) return false;
        let afterCount = 0;
        for (let j = 0; j < candidates.length; j++) {
          if (candidates[j] > c) afterCount += 1;
        }
        return afterCount >= remainingNeeded;
      });
      if (!valid.length) return null;
      let best = valid[0];
      for (let j = 1; j < valid.length; j++) {
        const cur = valid[j];
        const pCur = Number(pauseStrengths[cur]) || 0;
        const pBest = Number(pauseStrengths[best]) || 0;
        const dCur = Math.abs(cumW[cur] - targetFracs[b]);
        const dBest = Math.abs(cumW[best] - targetFracs[b]);
        if (pCur > pBest || (pCur === pBest && dCur < dBest)) {
          best = cur;
        }
      }
      chosen.push(best);
      minIdx = best + 1;
    }
    return chosen;
  }

  function selectProportionalWordSplits(weights, segments) {
    const kBreaks = segments.length - 1;
    const n = weights.length;
    if (kBreaks <= 0 || n <= kBreaks) return null;

    let totalVoiced = 0;
    for (let i = 0; i < segments.length; i++) {
      totalVoiced += Math.max(1, segments[i][1] - segments[i][0]);
    }
    const targetFracs = [];
    let accV = 0;
    for (let i = 0; i < segments.length - 1; i++) {
      accV += Math.max(1, segments[i][1] - segments[i][0]);
      targetFracs.push(accV / totalVoiced);
    }

    let totalW = 0;
    for (let i = 0; i < n; i++) totalW += Math.max(0.1, Number(weights[i]) || 1.0);
    const cumW = [];
    let running = 0;
    for (let i = 0; i < n; i++) {
      running += Math.max(0.1, Number(weights[i]) || 1.0);
      cumW.push(running / totalW);
    }

    const chosen = [];
    let minIdx = 0;
    for (let b = 0; b < targetFracs.length; b++) {
      const maxIdx = n - (kBreaks - b) - 1;
      if (minIdx > maxIdx) return null;
      let best = minIdx;
      let bestDist = Math.abs(cumW[minIdx] - targetFracs[b]);
      for (let c = minIdx + 1; c <= maxIdx; c++) {
        const d = Math.abs(cumW[c] - targetFracs[b]);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      chosen.push(best);
      minIdx = best + 1;
    }
    return chosen;
  }

  function voicedOffsetToWallMs(voicedMs, segments, isEnd) {
    if (!segments.length) return Math.max(0, Math.round(voicedMs));
    let rem = Math.max(0, Number(voicedMs) || 0);
    for (let i = 0; i < segments.length; i++) {
      const s = segments[i][0];
      const e = segments[i][1];
      const span = Math.max(1, e - s);
      const inSeg = isEnd ? (rem <= span) : (rem < span);
      if (inSeg || i === segments.length - 1) {
        return Math.round(Math.min(e, s + rem));
      }
      rem -= span;
    }
    return segments[segments.length - 1][1];
  }

  function buildWordTimingSchedule(tokens, totalMs, minIntervalMs, speechSegments) {
    const total = Math.round(Number(totalMs) || 0);
    if (total <= 0 || !Array.isArray(tokens) || !tokens.length) return [];
    const segments = normalizeSpeechSegments(speechSegments, total);
    let activeVoicedMs = total;
    if (segments.length) {
      activeVoicedMs = 0;
      for (let i = 0; i < segments.length; i++) {
        activeVoicedMs += Math.max(1, segments[i][1] - segments[i][0]);
      }
    }
    const minStep = Math.max(20, parseInt(minIntervalMs == null ? 45 : minIntervalMs, 10) || 45);
    const maxUnits = Math.max(1, Math.floor(activeVoicedMs / minStep));

    let sampled = tokens;
    let weights = tokens.map((t) => Math.max(0.1, Number(t.weight) || 1.0));
    let pauses = tokens.map((t) => Number(t.pauseStrength) || 0);
    if (tokens.length > maxUnits) {
      sampled = [];
      weights = [];
      pauses = [];
      const n = tokens.length;
      for (let b = 0; b < maxUnits; b++) {
        const bStart = Math.floor((b * n) / maxUnits);
        const bEnd = Math.min(n, Math.max(bStart + 1, Math.floor(((b + 1) * n) / maxUnits)));
        let pick = bStart;
        let bucketWeight = 0;
        let anyKw = false;
        for (let k = bStart; k < bEnd; k++) {
          bucketWeight += Math.max(0.1, Number(tokens[k].weight) || 1.0);
          if (tokens[k].isKeyword && !anyKw) {
            pick = k;
            anyKw = true;
          }
        }
        sampled.push(tokens[pick]);
        weights.push(bucketWeight);
        pauses.push(0);
      }
    }

    // Strategy 1 & 2: Multi-segment alignment (punctuation clause breaks first,
    // then word-boundary partitioning so no single word straddles a silent pause).
    if (segments.length > 1) {
      let splitIndices = selectClauseSplitIndices(weights, pauses, segments);
      if (!splitIndices) {
        splitIndices = selectProportionalWordSplits(weights, segments);
      }
      if (splitIndices) {
        const schedule = [];
        let clauseStart = 0;
        const boundaries = splitIndices.concat([sampled.length - 1]);
        for (let segIdx = 0; segIdx < boundaries.length; segIdx++) {
          const endInc = boundaries[segIdx];
          const segStart = segments[segIdx][0];
          const segEnd = segments[segIdx][1];
          const segDur = Math.max(1, segEnd - segStart);
          const cTokens = sampled.slice(clauseStart, endInc + 1);
          const cWeights = weights.slice(clauseStart, endInc + 1);
          const cDurs = allocateWeightedDurationsMs(segDur, cWeights);
          let cursor = segStart;
          for (let j = 0; j < cTokens.length; j++) {
            const dur = cDurs[j] || 1;
            schedule.push({
              token: cTokens[j],
              startMs: cursor,
              endMs: cursor + dur,
              durationMs: dur,
              isKeyword: !!cTokens[j].isKeyword,
            });
            cursor += dur;
          }
          clauseStart = endInc + 1;
        }
        return schedule;
      }
    }

    // Strategy 3: Voiced-time warping across active speech segments (single segment or fewer words than segments).
    if (segments.length) {
      const voicedDurs = allocateWeightedDurationsMs(activeVoicedMs, weights);
      const schedule = [];
      let vCursor = 0;
      for (let i = 0; i < sampled.length; i++) {
        const sMs = voicedOffsetToWallMs(vCursor, segments, false);
        vCursor += voicedDurs[i] || 1;
        const eMsRaw = (i === sampled.length - 1)
          ? segments[segments.length - 1][1]
          : voicedOffsetToWallMs(vCursor, segments, true);
        const eMs = Math.max(sMs + 1, eMsRaw);
        schedule.push({
          token: sampled[i],
          startMs: sMs,
          endMs: eMs,
          durationMs: eMs - sMs,
          isKeyword: !!sampled[i].isKeyword,
        });
      }
      return schedule;
    }

    // Strategy 3: Weighted distribution across full chunk duration.
    const durations = allocateWeightedDurationsMs(total, weights);
    const schedule = [];
    let cursor = 0;
    for (let i = 0; i < sampled.length; i++) {
      const dur = durations[i] || 1;
      schedule.push({
        token: sampled[i],
        startMs: cursor,
        endMs: cursor + dur,
        durationMs: dur,
        isKeyword: !!sampled[i].isKeyword,
      });
      cursor += dur;
    }
    return schedule;
  }

  function scoreTextSimilarity(wantedRaw, candidateRaw) {
    const wanted = normalizeText(wantedRaw || "").toLowerCase();
    const cand = normalizeText(candidateRaw || "").toLowerCase();
    if (!wanted || !cand) return 0;
    if (cand === wanted) return 1.0;
    if (cand.indexOf(wanted) >= 0 || wanted.indexOf(cand) >= 0) {
      return Math.min(cand.length, wanted.length) / Math.max(1, Math.max(cand.length, wanted.length));
    }

    // Latin/word Jaccard
    const wordsA = wanted.split(/\s+/).filter(Boolean).slice(0, 80);
    const wordsB = cand.split(/\s+/).filter(Boolean).slice(0, 80);
    let wordScore = 0;
    if (wordsA.length > 1 && wordsB.length > 1) {
      const setA = new Set(wordsA);
      const setB = new Set(wordsB);
      let inter = 0;
      setA.forEach((t) => { if (setB.has(t)) inter += 1; });
      wordScore = inter / Math.max(1, setA.size + setB.size - inter);
    }

    // Character bigram Dice (essential for CJK text without spaces & minor OCR/kerning differences).
    const compactA = wanted.replace(/\s+/g, "");
    const compactB = cand.replace(/\s+/g, "");
    let bigramScore = 0;
    if (compactA.length >= 2 && compactB.length >= 2) {
      const gramsA = new Map();
      const limA = Math.min(compactA.length - 1, 160);
      for (let i = 0; i < limA; i++) {
        const g = compactA.slice(i, i + 2);
        gramsA.set(g, (gramsA.get(g) || 0) + 1);
      }
      let inter2 = 0;
      const limB = Math.min(compactB.length - 1, 160);
      for (let i = 0; i < limB; i++) {
        const g = compactB.slice(i, i + 2);
        const count = gramsA.get(g) || 0;
        if (count > 0) {
          inter2 += 1;
          gramsA.set(g, count - 1);
        }
      }
      bigramScore = (2 * inter2) / Math.max(1, limA + limB);
    }

    return Math.max(wordScore, bigramScore);
  }

  // ---- Geometry / Column / Line / Paragraph Parsing ----------------------

  function _lineWidth(l) {
    if (!l || !isFinite(l.xMin) || !isFinite(l.xMax)) return 0;
    return Math.max(0, l.xMax - l.xMin);
  }

  function _lineHeight(l) {
    if (!l) return 0;
    if (isFinite(l.h) && l.h > 0) return l.h;
    if (isFinite(l.yMax) && isFinite(l.yMin)) return Math.max(1, Math.abs(l.yMax - l.yMin));
    return 0;
  }

  function _lineYCenter(l) {
    if (!l) return 0;
    if (isFinite(l.yCenter)) return l.yCenter;
    if (isFinite(l.yMin) && isFinite(l.yMax)) return (l.yMin + l.yMax) * 0.5;
    if (isFinite(l.y)) return l.y;
    return 0;
  }

  function _lineXCenter(l) {
    if (!l || !isFinite(l.xMin)) return 0;
    return l.xMin + _lineWidth(l) * 0.5;
  }

  function _joinLineItemsText(items, gapThreshold) {
    if (!items || !items.length) return "";
    let out = "";
    let prevRight = null;
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const str = normalizeText(it.text != null ? it.text : (it.str || ""));
      if (!str) continue;
      if (out) {
        const prevCh = out[out.length - 1] || "";
        const nextCh = str[0] || "";
        const gap = (prevRight != null && isFinite(it.x)) ? (it.x - prevRight) : gapThreshold + 1;
        if (isCjkLike(prevCh) && isCjkLike(nextCh)) {
          out += str;
        } else if (gap <= gapThreshold && /[A-Za-z0-9]/.test(prevCh) && /[A-Za-z0-9]/.test(nextCh)) {
          // Tight kerning split inside a word (e.g. "T" + "he").
          out += str;
        } else if (gap <= gapThreshold && /^[,.;:!?)\]】》」』]/.test(nextCh)) {
          out += str;
        } else {
          out += " " + str;
        }
      } else {
        out = str;
      }
      prevRight = (isFinite(it.x) ? it.x : 0) + (isFinite(it.w) ? Math.max(0, it.w) : 0);
    }
    return normalizeText(out);
  }

  function _detectColumnCutFromXs(xs, minX, spanWidth) {
    if (!xs || xs.length < 24 || !(spanWidth > 0)) return null;
    const sorted = xs.slice().sort((a, b) => a - b);
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
      const frac = (cut - minX) / spanWidth;
      if (!isFinite(frac) || frac < 0.30 || frac > 0.70) continue;
      bestGap = gap;
      bestCut = cut;
    }
    const minGap = Math.max(14, spanWidth * 0.025);
    if (bestCut != null && bestGap >= minGap) {
      return { cut: bestCut, gap: bestGap };
    }
    return null;
  }

  function buildLinesFromDomEntries(entries) {
    const list = Array.isArray(entries)
      ? entries.filter((e) => e && isFinite(e.x) && isFinite(e.y))
      : [];
    if (!list.length) return [];

    const typicalH = median(list.map((e) => e.h).filter((h) => isFinite(h) && h > 0)) || 12;
    const yTol = Math.max(2, typicalH * 0.95);

    let minX = Infinity;
    let maxX = -Infinity;
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      minX = Math.min(minX, e.x);
      maxX = Math.max(maxX, e.x + (e.w || 0));
    }
    const spanWidth = (isFinite(minX) && isFinite(maxX) && maxX > minX) ? (maxX - minX) : 0;
    let joinGapX = Math.max(10, Math.min(28, spanWidth * 0.025));

    const cutInfo = _detectColumnCutFromXs(
      list.map((e) => e.x).filter((x) => isFinite(x)),
      minX,
      spanWidth,
    );
    const columnCut = cutInfo ? cutInfo.cut : null;
    if (cutInfo) {
      joinGapX = Math.min(joinGapX, Math.max(8, cutInfo.gap * 0.45));
    }

    const withCenter = list.map((e) => {
      const h = isFinite(e.h) ? e.h : 0;
      return Object.assign({}, e, { yCenter: e.y + h * 0.5 });
    });
    withCenter.sort((a, b) => {
      const dy = a.yCenter - b.yCenter;
      if (Math.abs(dy) < Math.max(0.5, yTol * 0.15)) return a.x - b.x;
      return dy;
    });

    const lines = [];
    for (let idx = 0; idx < withCenter.length; idx++) {
      const entry = withCenter[idx];
      const y = entry.yCenter;
      let best = null;
      let bestOverlap = 0;
      let bestDist = Infinity;
      const scanStart = Math.max(0, lines.length - 8);
      for (let i = lines.length - 1; i >= scanStart; i--) {
        const cand = lines[i];
        if (!cand) continue;
        const dist = Math.abs(y - cand.yCenter);
        const overlap = Math.min(entry.y + (entry.h || 0), cand.yMax) - Math.max(entry.y, cand.yMin);
        const candH = Math.max(1, cand.yMax - cand.yMin);
        const minH = Math.max(1, Math.min(entry.h || candH, candH));
        const overlapRatio = (overlap > 0) ? (overlap / minH) : 0;
        const eRight = entry.x + (entry.w || 0);
        const gapX = (eRight < cand.xMin)
          ? (cand.xMin - eRight)
          : ((entry.x > cand.xMax) ? (entry.x - cand.xMax) : 0);

        if (columnCut != null) {
          const margin = Math.max(4, typicalH * 0.35);
          const candIsLeft = cand.xMax <= (columnCut - margin);
          const candIsRight = cand.xMin >= (columnCut + margin);
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
      if (!best) {
        best = {
          yCenter: y,
          items: [],
          yMin: entry.y,
          yMax: entry.y + (entry.h || 0),
          xMin: entry.x,
          xMax: entry.x + (entry.w || 0),
        };
        lines.push(best);
      }
      best.items.push(entry);
      const n = best.items.length;
      best.yCenter = (best.yCenter * (n - 1) + y) / n;
      best.yMin = Math.min(best.yMin, entry.y);
      best.yMax = Math.max(best.yMax, entry.y + (entry.h || 0));
      best.xMin = Math.min(best.xMin, entry.x);
      best.xMax = Math.max(best.xMax, entry.x + (entry.w || 0));
    }

    function finalizeItems(items) {
      let yMin = Infinity, yMax = -Infinity, xMin = Infinity, xMax = -Infinity, ySum = 0;
      const spans = [];
      const texts = [];
      for (let i = 0; i < items.length; i++) {
        const it = items[i];
        spans.push(it.idx);
        texts.push(it.text);
        yMin = Math.min(yMin, it.y);
        yMax = Math.max(yMax, it.y + (it.h || 0));
        xMin = Math.min(xMin, it.x);
        xMax = Math.max(xMax, it.x + (it.w || 0));
        ySum += isFinite(it.yCenter) ? it.yCenter : (it.y + (it.h || 0) * 0.5);
      }
      const lineH = Math.max(1, yMax - yMin);
      const kerningTol = Math.max(1.0, Math.min(3.5, lineH * 0.14));
      return {
        yCenter: items.length ? (ySum / items.length) : 0,
        spans,
        texts,
        text: _joinLineItemsText(items, kerningTol),
        yMin: isFinite(yMin) ? yMin : 0,
        yMax: isFinite(yMax) ? yMax : 1,
        xMin: isFinite(xMin) ? xMin : 0,
        xMax: isFinite(xMax) ? xMax : 1,
        h: isFinite(lineH) ? lineH : 1,
      };
    }

    const outLines = [];
    const splitGapX = Math.max(joinGapX * 1.6, Math.max(24, spanWidth * 0.04));
    for (let li = 0; li < lines.length; li++) {
      const items = (lines[li].items || []).slice().sort((a, b) => a.x - b.x);
      let bestSplit = -1;
      let bestGap = 0;
      for (let i = 0; i < items.length - 1; i++) {
        const gap = items[i + 1].x - (items[i].x + (items[i].w || 0));
        if (gap > bestGap) {
          bestGap = gap;
          bestSplit = i;
        }
      }
      if (bestSplit >= 0 && bestGap >= splitGapX) {
        outLines.push(finalizeItems(items.slice(0, bestSplit + 1)));
        outLines.push(finalizeItems(items.slice(bestSplit + 1)));
      } else {
        outLines.push(finalizeItems(items));
      }
    }

    const medH = median(outLines.map((l) => l.h).filter((h) => isFinite(h) && h > 0)) || typicalH;
    const sameRowTol = Math.max(2, medH * 0.35);
    outLines.sort((a, b) => {
      const dy = a.yCenter - b.yCenter;
      if (Math.abs(dy) <= sameRowTol) {
        if (Math.abs(a.xMin - b.xMin) > 0.5) return a.xMin - b.xMin;
      }
      return dy;
    });
    return outLines;
  }

  function groupLinesIntoColumns(lines, pageWidth, yIncreasesDown) {
    const down = yIncreasesDown !== false;
    const list = Array.isArray(lines)
      ? lines.filter((l) => l && isFinite(l.xMin) && (isFinite(l.yCenter) || isFinite(l.yMin) || isFinite(l.y)))
      : [];
    if (!list.length) return [lines || []];
    const pw = isFinite(pageWidth) ? Math.max(1, pageWidth) : 1;

    const sortLines = (arr) => {
      const out = (arr || []).slice();
      const medH = median(out.map(_lineHeight).filter((h) => h > 0));
      const sameRowTol = Math.max(2, (medH || 0) * 0.35);
      out.sort((a, b) => {
        const ay = _lineYCenter(a);
        const by = _lineYCenter(b);
        const dy = down ? (ay - by) : (by - ay);
        if (Math.abs(dy) <= sameRowTol) {
          if (Math.abs((a.xMin || 0) - (b.xMin || 0)) > 0.5) return (a.xMin || 0) - (b.xMin || 0);
        }
        return dy;
      });
      return out;
    };

    const usable = list.filter((l) => {
      const w = _lineWidth(l);
      return w > 0 && w <= pw * 0.78;
    });
    const xsSource = usable.length >= 4 ? usable : list;
    const xs = xsSource.map(_lineXCenter).filter((x) => isFinite(x)).sort((a, b) => a - b);
    if (xs.length < 6) return [sortLines(list)];

    const minCount = Math.max(3, Math.min(8, Math.floor(xsSource.length * 0.22)));

    function classifyByCut(cut, leftMean, rightMean) {
      const left = [];
      const right = [];
      const wide = [];
      for (let i = 0; i < list.length; i++) {
        const l = list[i];
        const w = _lineWidth(l);
        if (w > pw * 0.80 || (l.xMin < cut - pw * 0.04 && l.xMax > cut + pw * 0.04)) {
          wide.push(l);
          continue;
        }
        const cx = _lineXCenter(l);
        if (leftMean != null && rightMean != null) {
          if (Math.abs(cx - leftMean) <= Math.abs(cx - rightMean)) left.push(l);
          else right.push(l);
        } else if (cx <= cut) {
          left.push(l);
        } else {
          right.push(l);
        }
      }
      if (left.length < minCount || right.length < minCount) return null;

      function boundsOf(arr) {
        const xMins = [], xMaxs = [], yVals = [];
        for (let i = 0; i < arr.length; i++) {
          const it = arr[i];
          if (isFinite(it.xMin)) xMins.push(it.xMin);
          if (isFinite(it.xMax)) xMaxs.push(it.xMax);
          yVals.push(_lineYCenter(it));
        }
        xMins.sort((a, b) => a - b);
        xMaxs.sort((a, b) => a - b);
        yVals.sort((a, b) => a - b);
        return {
          x0: quantile(xMins, 0.10),
          x1: quantile(xMaxs, 0.90),
          y0: quantile(yVals, 0.10),
          y1: quantile(yVals, 0.90),
        };
      }
      const allB = boundsOf(left.concat(right));
      const lB = boundsOf(left);
      const rB = boundsOf(right);
      const totalY = Math.max(1, allB.y1 - allB.y0);
      const coverL = (lB.y1 - lB.y0) / totalY;
      const coverR = (rB.y1 - rB.y0) / totalY;
      const overlapX = Math.min(lB.x1, rB.x1) - Math.max(lB.x0, rB.x0);
      const minW = Math.max(1, Math.min(Math.max(1, lB.x1 - lB.x0), Math.max(1, rB.x1 - rB.x0)));
      const overlapFrac = overlapX > 0 ? (overlapX / minW) : 0;
      if (overlapFrac > 0.30 || Math.min(coverL, coverR) < 0.45) {
        return null;
      }

      // Split full-width lines into top band (before columns) vs bottom band (after columns).
      const colYs = left.concat(right).map(_lineYCenter);
      const colTop = down ? Math.min.apply(null, colYs) : Math.max.apply(null, colYs);
      const colBottom = down ? Math.max.apply(null, colYs) : Math.min.apply(null, colYs);
      const topWide = [];
      const bottomWide = [];
      for (let i = 0; i < wide.length; i++) {
        const wy = _lineYCenter(wide[i]);
        const isBottom = down
          ? (wy > colBottom - 4 || Math.abs(wy - colBottom) < Math.abs(wy - colTop))
          : (wy < colBottom + 4 || Math.abs(wy - colBottom) < Math.abs(wy - colTop));
        if (isBottom && (down ? wy > (colTop + colBottom) * 0.5 : wy < (colTop + colBottom) * 0.5)) {
          bottomWide.push(wide[i]);
        } else {
          topWide.push(wide[i]);
        }
      }

      const groups = [];
      if (topWide.length) groups.push(sortLines(topWide));
      groups.push(sortLines(left));
      groups.push(sortLines(right));
      if (bottomWide.length) groups.push(sortLines(bottomWide));
      return groups;
    }

    // Fast gap-based split
    const q10 = quantile(xs, 0.10);
    const q90 = quantile(xs, 0.90);
    let bestGap = 0;
    let bestCut = null;
    for (let i = 0; i < xs.length - 1; i++) {
      if (xs[i] < q10 || xs[i + 1] > q90) continue;
      const gap = xs[i + 1] - xs[i];
      if (gap <= bestGap) continue;
      const cut = (xs[i] + xs[i + 1]) * 0.5;
      if (cut < pw * 0.35 || cut > pw * 0.65) continue;
      bestGap = gap;
      bestCut = cut;
    }
    if (bestCut != null && bestGap >= Math.max(28, pw * 0.10)) {
      const res = classifyByCut(bestCut, null, null);
      if (res) return res;
    }

    // 1D k-means (k=2) fallback
    let m1 = quantile(xs, 0.25);
    let m2 = quantile(xs, 0.75);
    if (Math.abs(m2 - m1) < 1) return [sortLines(list)];
    let a = [];
    let b = [];
    for (let iter = 0; iter < 10; iter++) {
      a = [];
      b = [];
      for (let i = 0; i < xs.length; i++) {
        if (Math.abs(xs[i] - m1) <= Math.abs(xs[i] - m2)) a.push(xs[i]);
        else b.push(xs[i]);
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
    if (separation < Math.max(42, pw * 0.16)) return [sortLines(list)];
    const score = separation / Math.max(1e-6, std(a, m1) + std(b, m2));
    if (score < 1.2) return [sortLines(list)];
    const cut = (leftMean + rightMean) * 0.5;
    if (cut < pw * 0.35 || cut > pw * 0.65) return [sortLines(list)];
    return classifyByCut(cut, leftMean, rightMean) || [sortLines(list)];
  }

  function linesToParagraphs(lines, pageNum, yIncreasesDown) {
    const down = yIncreasesDown !== false;
    const paragraphs = [];
    let current = null;

    // Estimate typical line height and left margin in this column for robust paragraph breaks.
    const validHeights = (lines || []).map(_lineHeight).filter((h) => h > 0);
    const colMedianH = median(validHeights) || 12;

    for (let i = 0; i < (lines || []).length; i++) {
      const line = lines[i];
      if (!line) continue;
      const lineText = normalizeText(
        line.text != null ? line.text : ((line.texts || []).join(" "))
      );
      if (!lineText) continue;

      const lineH = Math.max(1, _lineHeight(line) || colMedianH);
      const lineSpans = Array.isArray(line.spans) ? line.spans.slice() : [];

      if (!current) {
        current = {
          pageNum,
          text: lineText,
          spans: lineSpans,
          yMin: line.yMin,
          yMax: line.yMax,
          xMin: line.xMin,
          xMax: line.xMax,
          _lastXMin: line.xMin,
          _lastXMax: line.xMax,
          _lastYMin: line.yMin,
          _lastYMax: line.yMax,
          _lastH: lineH,
        };
        continue;
      }

      // Vertical gap from the PREVIOUS LINE (NOT cumulative paragraph height!).
      const gap = down
        ? (line.yMin - current._lastYMax)
        : (current._lastYMin - line.yMax);
      const refLineH = Math.max(1, Math.min(current._lastH || colMedianH, colMedianH * 1.6));
      const gapLimit = Math.max(4, refLineH * 0.95);

      // Disjoint horizontal columns on the same baseline should never merge.
      let disjointSameRow = false;
      if (
        isFinite(current._lastXMin) && isFinite(current._lastXMax) &&
        isFinite(current._lastYMin) && isFinite(current._lastYMax) &&
        isFinite(line.xMin) && isFinite(line.xMax) &&
        isFinite(line.yMin) && isFinite(line.yMax)
      ) {
        const overlapY = Math.min(line.yMax, current._lastYMax) - Math.max(line.yMin, current._lastYMin);
        const minH = Math.max(1, Math.min(current._lastH || 1, lineH));
        const overlapYFrac = overlapY > 0 ? (overlapY / minH) : 0;
        const overlapX = Math.min(line.xMax, current._lastXMax) - Math.max(line.xMin, current._lastXMin);
        if (overlapYFrac >= 0.65 && overlapX <= 0) {
          disjointSameRow = true;
        }
      }

      // First-line indent break when previous line ended with sentence punctuation.
      const indentShift = isFinite(line.xMin) && isFinite(current.xMin)
        ? (line.xMin - current.xMin)
        : 0;
      const prevEndedSentence = /[.!?。！？][”’"'\)\]）】》]*$/.test(current.text || "");
      const looksLikeNewIndentedPara = (
        prevEndedSentence &&
        indentShift >= Math.max(10, refLineH * 0.9) &&
        gap >= -2
      );

      if (disjointSameRow || gap > gapLimit || looksLikeNewIndentedPara) {
        paragraphs.push(current);
        current = {
          pageNum,
          text: lineText,
          spans: lineSpans,
          yMin: line.yMin,
          yMax: line.yMax,
          xMin: line.xMin,
          xMax: line.xMax,
          _lastXMin: line.xMin,
          _lastXMax: line.xMax,
          _lastYMin: line.yMin,
          _lastYMax: line.yMax,
          _lastH: lineH,
        };
      } else {
        current.text = mergeHyphenated(current.text, lineText);
        if (lineSpans.length) current.spans = current.spans.concat(lineSpans);
        current.yMin = Math.min(current.yMin, line.yMin);
        current.yMax = Math.max(current.yMax, line.yMax);
        current.xMin = Math.min(current.xMin, line.xMin);
        current.xMax = Math.max(current.xMax, line.xMax);
        current._lastXMin = line.xMin;
        current._lastXMax = line.xMax;
        current._lastYMin = line.yMin;
        current._lastYMax = line.yMax;
        current._lastH = lineH;
      }
    }

    if (current) paragraphs.push(current);
    for (let i = 0; i < paragraphs.length; i++) {
      delete paragraphs[i]._lastXMin;
      delete paragraphs[i]._lastXMax;
      delete paragraphs[i]._lastYMin;
      delete paragraphs[i]._lastYMax;
      delete paragraphs[i]._lastH;
    }
    return paragraphs;
  }

  function extractLinesFromTextContent(textContent) {
    const items = (textContent && Array.isArray(textContent.items)) ? textContent.items : [];
    const rows = [];
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      if (!it) continue;
      const str = normalizeText(it.str || "");
      if (!str) continue;
      const tr = Array.isArray(it.transform) ? it.transform : [];
      const x = isFinite(tr[4]) ? tr[4] : 0;
      const y = isFinite(tr[5]) ? tr[5] : 0;
      const w = isFinite(it.width) ? Math.abs(it.width) : 0;
      let h = (isFinite(it.height) && Math.abs(it.height) > 0) ? Math.abs(it.height) : 0;
      if (h <= 0 && tr.length >= 4) {
        const mh = Math.hypot(Number(tr[2]) || 0, Number(tr[3]) || 0);
        const mw = Math.hypot(Number(tr[0]) || 0, Number(tr[1]) || 0);
        h = mh > 0 ? mh : mw;
      }
      if (!isFinite(h) || h <= 0) h = 10;
      rows.push({ str, x, y, w, h, yCenter: y + h * 0.5 });
    }
    if (!rows.length) return [];

    const typicalH = median(rows.map((r) => r.h).filter((h) => h > 0)) || 10;
    const yTol = Math.max(1.5, typicalH * 0.75);
    let minX = Infinity, maxX = -Infinity;
    for (let i = 0; i < rows.length; i++) {
      minX = Math.min(minX, rows[i].x);
      maxX = Math.max(maxX, rows[i].x + rows[i].w);
    }
    const spanWidth = (isFinite(minX) && isFinite(maxX) && maxX > minX) ? (maxX - minX) : 0;
    let joinGapX = Math.max(8, Math.min(24, spanWidth * 0.03));
    const cutInfo = _detectColumnCutFromXs(rows.map((r) => r.x), minX, spanWidth);
    const columnCut = cutInfo ? cutInfo.cut : null;
    if (cutInfo) {
      joinGapX = Math.min(joinGapX, Math.max(6, cutInfo.gap * 0.45));
    }

    // PDF y increases upward: sort top-to-bottom (descending yCenter), then left-to-right.
    rows.sort((a, b) => {
      const dy = b.yCenter - a.yCenter;
      if (Math.abs(dy) > Math.max(0.5, yTol * 0.2)) return dy;
      return a.x - b.x;
    });

    const lines = [];
    for (let idx = 0; idx < rows.length; idx++) {
      const r = rows[idx];
      let best = null;
      let bestDist = Infinity;
      const scanStart = Math.max(0, lines.length - 8);
      for (let i = lines.length - 1; i >= scanStart; i--) {
        const cand = lines[i];
        const dist = Math.abs(r.yCenter - cand.yCenter);
        if (dist > yTol) continue;
        const rRight = r.x + r.w;
        const gapX = (rRight < cand.xMin)
          ? (cand.xMin - rRight)
          : ((r.x > cand.xMax) ? (r.x - cand.xMax) : 0);
        if (gapX > joinGapX) continue;
        if (columnCut != null) {
          const margin = Math.max(3, typicalH * 0.35);
          const candIsLeft = cand.xMax <= (columnCut - margin);
          const candIsRight = cand.xMin >= (columnCut + margin);
          if (candIsLeft || candIsRight) {
            const rIsLeft = r.x <= columnCut;
            if ((candIsLeft && !rIsLeft) || (candIsRight && rIsLeft)) continue;
          }
        }
        if (dist < bestDist) {
          bestDist = dist;
          best = cand;
        }
      }
      if (!best) {
        best = {
          yCenter: r.yCenter,
          y: r.yCenter,
          items: [],
          xMin: r.x,
          xMax: r.x + r.w,
          yMin: r.y,
          yMax: r.y + r.h,
          h: r.h,
          text: "",
        };
        lines.push(best);
      }
      best.items.push(r);
      const n = best.items.length;
      best.yCenter = (best.yCenter * (n - 1) + r.yCenter) / n;
      best.y = best.yCenter;
      best.xMin = Math.min(best.xMin, r.x);
      best.xMax = Math.max(best.xMax, r.x + r.w);
      best.yMin = Math.min(best.yMin, r.y);
      best.yMax = Math.max(best.yMax, r.y + r.h);
      best.h = Math.max(1, best.yMax - best.yMin);
    }

    const out = [];
    const splitGapX = Math.max(joinGapX * 1.6, Math.max(18, spanWidth * 0.04));
    function finalizeTextItems(arr) {
      let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity, ySum = 0;
      for (let i = 0; i < arr.length; i++) {
        const it = arr[i];
        xMin = Math.min(xMin, it.x);
        xMax = Math.max(xMax, it.x + it.w);
        yMin = Math.min(yMin, it.y);
        yMax = Math.max(yMax, it.y + it.h);
        ySum += it.yCenter;
      }
      const lineH = Math.max(1, yMax - yMin);
      const kerningTol = Math.max(0.8, Math.min(3.0, lineH * 0.14));
      const text = _joinLineItemsText(arr, kerningTol);
      if (!text) return null;
      return {
        yCenter: ySum / Math.max(1, arr.length),
        y: ySum / Math.max(1, arr.length),
        xMin,
        xMax,
        yMin,
        yMax,
        h: lineH,
        text,
      };
    }

    for (let i = 0; i < lines.length; i++) {
      const sortedItems = lines[i].items.slice().sort((a, b) => a.x - b.x);
      let bestSplit = -1;
      let bestGap = 0;
      for (let j = 0; j < sortedItems.length - 1; j++) {
        const gap = sortedItems[j + 1].x - (sortedItems[j].x + sortedItems[j].w);
        if (gap > bestGap) {
          bestGap = gap;
          bestSplit = j;
        }
      }
      if (bestSplit >= 0 && bestGap >= splitGapX) {
        const leftLine = finalizeTextItems(sortedItems.slice(0, bestSplit + 1));
        const rightLine = finalizeTextItems(sortedItems.slice(bestSplit + 1));
        if (leftLine) out.push(leftLine);
        if (rightLine) out.push(rightLine);
      } else {
        const single = finalizeTextItems(sortedItems);
        if (single) out.push(single);
      }
    }
    return out;
  }

  function orderLinesForReading(lines, pageWidth) {
    const groups = groupLinesIntoColumns(lines, pageWidth, false);
    const ordered = [];
    for (let i = 0; i < groups.length; i++) {
      const g = groups[i] || [];
      for (let j = 0; j < g.length; j++) ordered.push(g[j]);
    }
    return ordered;
  }

  const api = {
    isCjkChar,
    isCjkLike,
    normalizeText,
    joinTextSegments,
    mergeHyphenated,
    median,
    isNonBreakingPeriod,
    splitTextIntoSentenceRanges,
    orderSpanIndicesForReading,
    splitParagraphEntriesIntoSentences,
    parseKeywords,
    extractAutoKeywords,
    findKeywordRanges,
    tokenizeWords,
    allocateWeightedDurationsMs,
    buildWordTimingSchedule,
    scoreTextSimilarity,
    buildLinesFromDomEntries,
    groupLinesIntoColumns,
    linesToParagraphs,
    extractLinesFromTextContent,
    orderLinesForReading,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  if (globalScope) {
    globalScope.AnnolidPdfSpeech = api;
  }
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this));
