// Resolves a backend-root-relative resource (e.g. region `mesh_file`
// "/files/JOB/preform_recovery/RUN/R1.ply", or a source-frame `url`) against
// the backend base URL. Keeps any path prefix in the base (so a proxied
// "/api" base still works) and never joins onto a legacy asset directory.
//
// Only absolute http(s) URLs and single-slash root-relative paths are
// accepted; anything else (filesystem paths, "//host", "..") yields null so
// it is never turned into a request.
export function resolveBackendResource(path, apiUrl) {
  if (typeof path !== 'string' || !path) return null;
  if (/^https?:\/\//i.test(path)) return path;
  if (!path.startsWith('/') || path.startsWith('//') || path.includes('\\')) return null;
  // Reject dot segments (raw or percent-encoded) so a path can never be
  // normalized to a different backend resource than the one it names.
  const segments = path.split(/[?#]/)[0].split('/');
  if (segments.some((segment) => /^(?:\.|%2e){1,2}$/i.test(segment))) return null;
  const origin = typeof window !== 'undefined' ? window.location.href : 'http://localhost/';
  let base;
  try {
    base = new URL(`${String(apiUrl || '').replace(/\/+$/, '')}/`, origin);
  } catch {
    return null;
  }
  const resolved = new URL(path.slice(1), base);
  // Guard against "/../" escaping the backend base path.
  return resolved.href.startsWith(base.href) ? resolved.href : null;
}
