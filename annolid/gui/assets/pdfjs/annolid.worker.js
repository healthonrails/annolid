/* Annolid PDF.js worker bootstrap for older Chromium (QtWebEngine 5.15).
 *
 * This script runs inside the WebWorker context. It installs minimal polyfills
 * required by PDF.js (notably `at()`), while ensuring they are NON-enumerable
 * so PDF.js' `for...in` prototype checks keep passing.
 */

(function () {
  "use strict";

  function _defineAt(proto, fn) {
    if (!proto || proto.at) return;
    try {
      Object.defineProperty(proto, "at", {
        value: fn,
        writable: true,
        configurable: true,
        enumerable: false,
      });
    } catch (e) {
      try {
        proto.at = fn;
      } catch (e2) { }
    }
  }

  function _atPolyfill(n) {
    n = Math.trunc(n) || 0;
    if (n < 0) n += this.length;
    if (n < 0 || n >= this.length) return undefined;
    return this[n];
  }

  _defineAt(Array.prototype, _atPolyfill);
  _defineAt(String.prototype, function (n) {
    n = Math.trunc(n) || 0;
    if (n < 0) n += this.length;
    if (n < 0 || n >= this.length) return undefined;
    return this.charAt(n);
  });

  const typed = [
    typeof Int8Array !== "undefined" ? Int8Array : null,
    typeof Uint8Array !== "undefined" ? Uint8Array : null,
    typeof Uint8ClampedArray !== "undefined" ? Uint8ClampedArray : null,
    typeof Int16Array !== "undefined" ? Int16Array : null,
    typeof Uint16Array !== "undefined" ? Uint16Array : null,
    typeof Int32Array !== "undefined" ? Int32Array : null,
    typeof Uint32Array !== "undefined" ? Uint32Array : null,
    typeof Float32Array !== "undefined" ? Float32Array : null,
    typeof Float64Array !== "undefined" ? Float64Array : null,
    typeof BigInt64Array !== "undefined" ? BigInt64Array : null,
    typeof BigUint64Array !== "undefined" ? BigUint64Array : null,
  ];
  for (const T of typed) {
    if (T && T.prototype) _defineAt(T.prototype, _atPolyfill);
  }

  if (typeof structuredClone === "undefined") {
    self.structuredClone = function (obj) {
      try {
        return JSON.parse(JSON.stringify(obj));
      } catch (e) {
        return obj;
      }
    };
  }

  // Load the actual PDF.js worker implementation.
  importScripts("pdf.worker.min.js");
})();
