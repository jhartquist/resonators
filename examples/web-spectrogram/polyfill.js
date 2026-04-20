// AudioWorkletGlobalScope lacks TextDecoder/TextEncoder. Imported BEFORE the
// wasm-bindgen JS so the polyfills install before its module-level init runs.
if (typeof TextDecoder === 'undefined') {
  globalThis.TextDecoder = class {
    decode(buf) {
      if (!buf) return '';
      const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
      let s = '';
      for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
      return s;
    }
  };
}
if (typeof TextEncoder === 'undefined') {
  globalThis.TextEncoder = class {
    encode(str = '') {
      const u8 = new Uint8Array(str.length);
      for (let i = 0; i < str.length; i++) u8[i] = str.charCodeAt(i) & 0xff;
      return u8;
    }
    encodeInto(str, dest) {
      const len = Math.min(str.length, dest.length);
      for (let i = 0; i < len; i++) dest[i] = str.charCodeAt(i) & 0xff;
      return { read: len, written: len };
    }
  };
}
