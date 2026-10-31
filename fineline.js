// Turns a photo into smooth fine-line strokes.
//
// Pipeline: flatten texture -> follow the image's edge flow to find lines
// (flow-based difference of Gaussians) -> thin to one pixel -> trace into long
// strokes -> drop the unimportant ones -> smooth.
//
// traceFineLine must stay self-contained (no outside references): index.html
// runs its source text inside a Web Worker.

function traceFineLine(gray, w, h, opt) {
  const n = w * h;
  const scale = Math.max(w, h) / 1000;
  const detail = opt.detail / 100;     // 0 = only the main lines, 1 = everything
  const simplify = opt.simplify / 100; // 0 = keep texture, 1 = very clean

  function gaussian(src, sigma) {
    const r = Math.max(1, Math.ceil(sigma * 2.5));
    const k = new Float32Array(2 * r + 1);
    let sum = 0;
    for (let i = -r; i <= r; i++) sum += k[i + r] = Math.exp(-i * i / (2 * sigma * sigma));
    for (let i = 0; i < k.length; i++) k[i] /= sum;
    const tmp = new Float32Array(n), out = new Float32Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) {
        const xx = x + i < 0 ? 0 : x + i >= w ? w - 1 : x + i;
        v += k[i + r] * src[y * w + xx];
      }
      tmp[y * w + x] = v;
    }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      let v = 0;
      for (let i = -r; i <= r; i++) {
        const yy = y + i < 0 ? 0 : y + i >= h ? h - 1 : y + i;
        v += k[i + r] * tmp[yy * w + x];
      }
      out[y * w + x] = v;
    }
    return out;
  }

  function boxBlur(src, r) {
    const tmp = new Float32Array(n), out = new Float32Array(n);
    for (let y = 0; y < h; y++) {
      let sum = 0;
      for (let x = -r; x <= r; x++) sum += src[y * w + Math.min(w - 1, Math.max(0, x))];
      for (let x = 0; x < w; x++) {
        tmp[y * w + x] = sum / (2 * r + 1);
        sum += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
      }
    }
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let y = -r; y <= r; y++) sum += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
      for (let y = 0; y < h; y++) {
        out[y * w + x] = sum / (2 * r + 1);
        sum += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
      }
    }
    return out;
  }

  // Edge-preserving smoothing (self-guided filter): flattens skin and fabric
  // texture while keeping real contours sharp.
  function flatten(src, r, eps) {
    const sq = new Float32Array(n);
    for (let i = 0; i < n; i++) sq[i] = src[i] * src[i];
    const mean = boxBlur(src, r), meanSq = boxBlur(sq, r);
    const a = new Float32Array(n), b = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const variance = Math.max(0, meanSq[i] - mean[i] * mean[i]);
      a[i] = variance / (variance + eps);
      b[i] = (1 - a[i]) * mean[i];
    }
    const ma = boxBlur(a, r), mb = boxBlur(b, r);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = ma[i] * src[i] + mb[i];
    return out;
  }

  // Direction each pixel's contour runs in, from the smoothed structure tensor.
  function edgeFlow(img) {
    const gxx = new Float32Array(n), gxy = new Float32Array(n), gyy = new Float32Array(n);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = (img[i - w + 1] + 2 * img[i + 1] + img[i + w + 1] - img[i - w - 1] - 2 * img[i - 1] - img[i + w - 1]) / 8;
      const gy = (img[i + w - 1] + 2 * img[i + w] + img[i + w + 1] - img[i - w - 1] - 2 * img[i - w] - img[i - w + 1]) / 8;
      gxx[i] = gx * gx; gxy[i] = gx * gy; gyy[i] = gy * gy;
    }
    const s = 2.5 * scale;
    const E = gaussian(gxx, s), F = gaussian(gxy, s), G = gaussian(gyy, s);
    const tx = new Float32Array(n), ty = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const lambda = 0.5 * (E[i] + G[i] + Math.sqrt((E[i] - G[i]) ** 2 + 4 * F[i] * F[i]));
      let vx = lambda - E[i], vy = -F[i];
      const len = Math.hypot(vx, vy);
      if (len > 1e-9) { tx[i] = vx / len; ty[i] = vy / len; } else { tx[i] = 0; ty[i] = 1; }
    }
    return { tx, ty };
  }

  function sample(img, x, y) {
    x = x < 0 ? 0 : x > w - 1.001 ? w - 1.001 : x;
    y = y < 0 ? 0 : y > h - 1.001 ? h - 1.001 : y;
    const xi = x | 0, yi = y | 0, fx = x - xi, fy = y - yi, i = yi * w + xi;
    return (img[i] * (1 - fx) + img[i + 1] * fx) * (1 - fy) + (img[i + w] * (1 - fx) + img[i + w + 1] * fx) * fy;
  }

  // Flow-based DoG: measure "darker than its surroundings" across each contour,
  // then average that along the contour so lines come out long and unbroken.
  function lineResponse(img, tx, ty, sigma, sigmaM) {
    const sigmaS = 1.6 * sigma;
    const R = Math.ceil(sigmaS * 2.5);
    const dog = new Float32Array(2 * R + 1);
    let sc = 0, ss = 0;
    for (let i = -R; i <= R; i++) {
      sc += Math.exp(-i * i / (2 * sigma * sigma));
      ss += Math.exp(-i * i / (2 * sigmaS * sigmaS));
    }
    for (let i = -R; i <= R; i++) {
      dog[i + R] = Math.exp(-i * i / (2 * sigma * sigma)) / sc - Math.exp(-i * i / (2 * sigmaS * sigmaS)) / ss;
    }
    const across = new Float32Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, nx = -ty[i], ny = tx[i];
      let v = 0;
      for (let k = -R; k <= R; k++) v += dog[k + R] * sample(img, x + nx * k, y + ny * k);
      across[i] = v;
    }
    const steps = Math.ceil(sigmaM * 2);
    const weights = [];
    for (let k = 0; k <= steps; k++) weights.push(Math.exp(-k * k / (2 * sigmaM * sigmaM)));
    const out = new Float32Array(n);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      let v = across[i] * weights[0], total = weights[0];
      for (const dir of [1, -1]) {
        let px = x, py = y, dx = tx[i] * dir, dy = ty[i] * dir;
        for (let k = 1; k <= steps; k++) {
          px += dx; py += dy;
          if (px < 0 || py < 0 || px >= w - 1 || py >= h - 1) break;
          v += sample(across, px, py) * weights[k];
          total += weights[k];
          const j = (py | 0) * w + (px | 0);
          const sgn = tx[j] * dx + ty[j] * dy < 0 ? -1 : 1;
          dx = tx[j] * sgn; dy = ty[j] * sgn;
        }
      }
      out[i] = v / total;
    }
    return out;
  }

  // Keep strong lines, plus weaker pixels only when connected to a strong one.
  function hysteresis(resp, hi, lo) {
    const mask = new Uint8Array(n), stack = [];
    for (let i = 0; i < n; i++) if (resp[i] >= hi) { mask[i] = 1; stack.push(i); }
    while (stack.length) {
      const i = stack.pop(), x = i % w, y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
        const j = yy * w + xx;
        if (!mask[j] && resp[j] >= lo) { mask[j] = 1; stack.push(j); }
      }
    }
    return mask;
  }

  // Zhang-Suen thinning down to one-pixel-wide lines.
  function thin(mask) {
    for (let x = 0; x < w; x++) mask[x] = mask[(h - 1) * w + x] = 0;
    for (let y = 0; y < h; y++) mask[y * w] = mask[y * w + w - 1] = 0;
    let active = [];
    for (let i = 0; i < n; i++) if (mask[i]) active.push(i);
    let changed = true;
    while (changed) {
      changed = false;
      for (let pass = 0; pass < 2; pass++) {
        const remove = [];
        for (const i of active) {
          const p2 = mask[i - w], p3 = mask[i - w + 1], p4 = mask[i + 1], p5 = mask[i + w + 1];
          const p6 = mask[i + w], p7 = mask[i + w - 1], p8 = mask[i - 1], p9 = mask[i - w - 1];
          const count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
          if (count < 2 || count > 6) continue;
          const flips = (!p2 && p3) + (!p3 && p4) + (!p4 && p5) + (!p5 && p6) +
                        (!p6 && p7) + (!p7 && p8) + (!p8 && p9) + (!p9 && p2);
          if (flips !== 1) continue;
          if (pass === 0 ? (p2 && p4 && p6) || (p4 && p6 && p8) : (p2 && p4 && p8) || (p2 && p6 && p8)) continue;
          remove.push(i);
        }
        for (const i of remove) mask[i] = 0;
        if (remove.length) changed = true;
      }
      active = active.filter(i => mask[i]);
    }
    return mask;
  }

  // Walks the skeleton into pixel chains, going straight through junctions so
  // strokes stay long, and snapping side branches onto the line they meet.
  function traceChains(mask) {
    const offsets = [-w - 1, -w, -w + 1, -1, 1, w - 1, w, w + 1];
    const degree = new Uint8Array(n);
    for (let i = 0; i < n; i++) if (mask[i]) for (const o of offsets) degree[i] += mask[i + o];
    const seen = new Uint8Array(n), chains = [];

    function walk(start) {
      const chain = [start];
      seen[start] = 1;
      let cur = start;
      for (;;) {
        const back = chain[Math.max(0, chain.length - 6)];
        const dirX = cur % w - back % w, dirY = ((cur / w) | 0) - ((back / w) | 0);
        let best = -1, bestScore = -Infinity;
        for (const o of offsets) {
          const j = cur + o;
          if (!mask[j] || seen[j]) continue;
          const stepX = j % w - cur % w, stepY = ((j / w) | 0) - ((cur / w) | 0);
          const score = (dirX * stepX + dirY * stepY) / Math.hypot(stepX, stepY) + (stepX === 0 || stepY === 0 ? 0.1 : 0);
          if (score > bestScore) { bestScore = score; best = j; }
        }
        if (best < 0) break;
        seen[best] = 1;
        chain.push(best);
        cur = best;
      }
      return chain;
    }

    function attach(chain) {
      const end = chain[chain.length - 1], before = chain[chain.length - 2];
      for (const o of offsets) {
        const j = end + o;
        if (mask[j] && j !== before && chain.length > 3 && j !== chain[0]) { chain.push(j); return; }
      }
    }

    for (let i = 0; i < n; i++) if (mask[i] && !seen[i] && degree[i] === 1) {
      const chain = walk(i);
      if (chain.length > 1) { attach(chain); chains.push(chain); }
    }
    for (let i = 0; i < n; i++) if (mask[i] && !seen[i]) {
      const a = walk(i);
      if (a.length > 1) {
        a.reverse();
        const rest = walk(i === a[a.length - 1] ? i : a[a.length - 1]);
        chains.push(a.concat(rest.slice(1)));
      }
    }
    return chains.map(c => {
      const pts = [];
      for (const i of c) pts.push([i % w, (i / w) | 0]);
      return pts;
    });
  }

  // Joins strokes whose ends nearly touch and point at each other.
  function linkChains(chains, gap) {
    const reach = (pts, atEnd) => {
      const m = pts.length, k = Math.min(m - 1, 8);
      const a = atEnd ? pts[m - 1] : pts[0], b = atEnd ? pts[m - 1 - k] : pts[k];
      const len = Math.hypot(a[0] - b[0], a[1] - b[1]) || 1;
      return { x: a[0], y: a[1], dx: (a[0] - b[0]) / len, dy: (a[1] - b[1]) / len };
    };
    const ends = [];
    chains.forEach(c => { ends.push(reach(c, false), reach(c, true)); });
    const grid = new Map(), cell = Math.ceil(gap);
    ends.forEach((e, id) => {
      const key = ((e.x / cell) | 0) + ',' + ((e.y / cell) | 0);
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(id);
    });
    const pairs = [];
    ends.forEach((e, id) => {
      const cx = (e.x / cell) | 0, cy = (e.y / cell) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        for (const other of grid.get((cx + dx) + ',' + (cy + dy)) || []) {
          if (other <= id || (other >> 1) === (id >> 1)) continue;
          const f = ends[other], vx = f.x - e.x, vy = f.y - e.y, dist = Math.hypot(vx, vy);
          if (dist > gap) continue;
          const facing = -(e.dx * f.dx + e.dy * f.dy);
          const aligned = dist < 1 ? 1 : Math.min((e.dx * vx + e.dy * vy) / dist, -(f.dx * vx + f.dy * vy) / dist);
          if (facing < 0.5 || aligned < 0.5) continue;
          pairs.push([dist / gap - facing - aligned, id, other]);
        }
      }
    });
    pairs.sort((a, b) => a[0] - b[0]);
    const link = new Int32Array(ends.length).fill(-1);
    const parent = chains.map((_, i) => i);
    const find = i => { while (parent[i] !== i) i = parent[i] = parent[parent[i]]; return i; };
    for (const [, a, b] of pairs) {
      if (link[a] >= 0 || link[b] >= 0 || find(a >> 1) === find(b >> 1)) continue;
      link[a] = b; link[b] = a;
      parent[find(a >> 1)] = find(b >> 1);
    }
    const used = new Uint8Array(chains.length), merged = [];
    const follow = entry => {
      // entry is the chain end we walk in from; the walk leaves through its twin.
      const out = [];
      let id = entry;
      for (;;) {
        const c = id >> 1;
        used[c] = 1;
        const pts = chains[c];
        for (let i = 0; i < pts.length; i++) out.push(id & 1 ? pts[pts.length - 1 - i] : pts[i]);
        const next = link[id ^ 1];
        if (next < 0 || used[next >> 1]) break;
        id = next;
      }
      return out;
    };
    for (let id = 0; id < ends.length; id++) {
      if (!used[id >> 1] && link[id] < 0) merged.push(follow(id));
    }
    return merged;
  }

  function smoothed(pts, passes) {
    let cur = pts.map(p => p.slice());
    for (let k = 0; k < passes; k++) {
      const next = cur.map(p => p.slice());
      for (let i = 1; i < cur.length - 1; i++) {
        for (let d = 0; d < 2; d++) next[i][d] = 0.25 * cur[i - 1][d] + 0.5 * cur[i][d] + 0.25 * cur[i + 1][d];
      }
      cur = next;
    }
    return cur;
  }

  function resample(pts, step) {
    const out = [pts[0]];
    let carry = 0;
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i], len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      let t = step - carry;
      while (t < len) {
        const f = t / len;
        out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
        t += step;
      }
      carry = (carry + len) % step;
    }
    out.push(pts[pts.length - 1]);
    return out;
  }

  // The line map depends only on the image and Simplify, so moving the
  // Detail slider can reuse it through `cache`.
  const cache = opt.cache || {};
  let resp = cache.resp, crest = cache.crest;
  if (cache.simplify !== opt.simplify) {
    const base = flatten(gray, Math.round((3 + 7 * simplify) * scale), 0.0004 + 0.012 * simplify * simplify);
    const { tx, ty } = edgeFlow(base);
    const flowLength = (2 + 3 * simplify) * scale;

    // Dividing by local brightness treats shadows the same as highlights.
    const fine = lineResponse(base, tx, ty, 1.1 * scale, flowLength);
    const light = gaussian(base, 12 * scale);
    resp = new Float32Array(n);
    for (let i = 0; i < n; i++) resp[i] = fine[i] < 0 ? -fine[i] / (light[i] + 0.12) : 0;

    // Judge each line against its neighbourhood: faint features on a soft face get
    // lifted, while busy areas (foliage, patterned walls) compete among themselves.
    // Lines near the middle get a small bonus because that is usually the subject.
    const busy = gaussian(resp, 35 * scale);
    for (let i = 0; i < n; i++) {
      const cx = (i % w) / w - 0.5, cy = ((i / w) | 0) / h - 0.5;
      const focus = 1.25 - 1.1 * (cx * cx + cy * cy);
      resp[i] *= focus * 0.012 / (Math.sqrt(busy[i] * 0.012) + 0.0035);
    }

    // With an AI line map, only lines the model also drew survive: it knows the
    // subject from the background, while the tracer keeps the curves precise.
    if (opt.lineMap) {
      const guide = gaussian(opt.lineMap, 4 * scale);
      for (let i = 0; i < n; i++) {
        const t = Math.min(1, Math.max(0, (guide[i] - 0.03) / 0.1));
        resp[i] = resp[i] * t * t * (3 - 2 * t) * 1.5;
      }
    }

    // Keep only the crest of each line. Tracing crests gives steadier curves than
    // skeletonizing thick blobs, and two close lines stay two lines.
    crest = new Float32Array(n);
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x, nx = -ty[i], ny = tx[i], v = resp[i];
      if (v > 0 && v >= sample(resp, x + nx, y + ny) && v >= sample(resp, x - nx, y - ny)) crest[i] = v;
    }
    Object.assign(cache, { simplify: opt.simplify, resp, crest });
  }

  // Curved so the top of the slider reaches far fainter lines while the low end
  // stays gentle: each step of `level` lets through lines 10x fainter.
  const level = 2 * Math.pow(detail, 1.85);
  const hi = 0.11 * Math.pow(0.1, level);
  const mask = thin(hysteresis(crest, hi, hi * 0.4));
  const chains = linkChains(traceChains(mask), 10 * scale);

  const strength = gaussian(resp, 1.2 * scale);
  const minLength = (10 / Math.max(1, level) + 90 * simplify * Math.max(0, 1.15 - level)) * scale;
  const strokes = [];
  for (const chain of chains) {
    let sum = 0;
    for (const p of chain) sum += strength[p[1] * w + p[0]];
    const mean = sum / chain.length;
    // Short marks only survive if they are unusually dark (eyelashes, nostrils).
    const needed = minLength * Math.min(1, 0.35 + hi * 0.9 / mean);
    if (chain.length < needed) continue;

    const pts = smoothed(chain, Math.round((4 + 14 * simplify) * Math.sqrt(scale)));
    for (let i = 0; i < pts.length; i++) pts[i][2] = strength[chain[i][1] * w + chain[i][0]];
    const even = resample(smoothed(pts, 6), 2.5);
    strokes.push({
      x: even.map(p => +p[0].toFixed(1)),
      y: even.map(p => +p[1].toFixed(1)),
      s: even.map(p => +Math.min(1, p[2] / 0.1).toFixed(2)),
    });
  }
  return strokes;
}

// Builds one SVG path: each stroke is a thin ribbon that swells where the photo
// line is strong and tapers to a point at its ends, like a pen lifted off paper.
function strokesToPath(strokes, weight) {
  const parts = [];
  for (const { x, y, s } of strokes) {
    const m = x.length;
    if (m < 2) continue;
    const taper = Math.min(40, m * 0.3);
    const left = [], right = [];
    for (let i = 0; i < m; i++) {
      const a = Math.max(0, i - 1), b = Math.min(m - 1, i + 1);
      const dx = x[b] - x[a], dy = y[b] - y[a], len = Math.hypot(dx, dy) || 1;
      const edge = Math.min(1, (Math.min(i, m - 1 - i) + 0.5) / taper);
      const half = 0.5 * weight * (0.55 + 0.75 * Math.sqrt(s[i])) * Math.sin(edge * Math.PI / 2);
      const nx = -dy / len * half, ny = dx / len * half;
      left.push((x[i] + nx).toFixed(1) + ' ' + (y[i] + ny).toFixed(1));
      right.push((x[i] - nx).toFixed(1) + ' ' + (y[i] - ny).toFixed(1));
    }
    parts.push('M' + left.join('L') + 'L' + right.reverse().join('L') + 'Z');
  }
  return parts.join('');
}

if (typeof module !== 'undefined') module.exports = { traceFineLine, strokesToPath };
