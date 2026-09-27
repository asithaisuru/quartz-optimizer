import axios from 'axios';
import { vi } from 'vitest';

// Minimal axios router for component tests. Routes are matched by
// "METHOD path-suffix"; unmatched requests fail like a missing route (404),
// which is exactly what an older backend does.
export function httpError(status, data = {}) {
  const error = new Error(`HTTP ${status}`);
  error.isAxiosError = true;
  error.response = { status, data };
  return error;
}

export function installMockApi(routes) {
  const calls = [];
  const handle = (method) => vi.fn(async (url, body) => {
    const path = String(url).replace(/^https?:\/\/[^/]+/, '').replace(/\?.*$/, '');
    calls.push({ method, path, body });
    const key = Object.keys(routes).find((pattern) => {
      const [m, suffix] = pattern.split(' ');
      return m === method && path.endsWith(suffix);
    });
    if (!key) throw httpError(404, { detail: 'Not Found' });
    const value = typeof routes[key] === 'function' ? await routes[key]({ path, body, calls }) : routes[key];
    return { data: value };
  });
  axios.get = handle('GET');
  axios.post = handle('POST');
  axios.patch = handle('PATCH');
  axios.delete = handle('DELETE');
  return calls;
}
