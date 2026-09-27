import { useCallback, useEffect, useState } from 'react';
import { expertErrorMessage } from '../utils/expertReview.js';
import { fetchExpertReview, patchExpertPiece, patchExpertReviewer } from '../utils/reviewApi.js';

// Loads and edits the Expert Review of one completed V2 preform run. Every
// PATCH returns the full recomputed review, which replaces local state —
// summary figures are never derived in the frontend.
export default function useExpertReview({ apiUrl, jobId, runId, enabled }) {
  const [loaded, setLoaded] = useState({ key: null, status: 'idle', review: null, error: '' });
  const [busy, setBusy] = useState(null); // 'reviewer' | piece_id | null
  const [actionError, setActionError] = useState('');

  const key = enabled && jobId && runId ? `${jobId}/${runId}` : null;

  const load = useCallback(() => {
    if (!key) return Promise.resolve();
    return fetchExpertReview(apiUrl, jobId, runId)
      .then((review) => setLoaded({ key, status: 'available', review, error: '' }))
      .catch((error) => setLoaded({
        key,
        status: error?.response?.status === 404 || error?.response?.status === 405 ? 'unavailable' : 'error',
        review: null,
        error: expertErrorMessage(error, 'load'),
      }));
  }, [apiUrl, jobId, runId, key]);

  useEffect(() => {
    if (!key) return undefined;
    let cancelled = false;
    fetchExpertReview(apiUrl, jobId, runId)
      .then((review) => { if (!cancelled) setLoaded({ key, status: 'available', review, error: '' }); })
      .catch((error) => {
        if (cancelled) return;
        setLoaded({
          key,
          status: error?.response?.status === 404 || error?.response?.status === 405 ? 'unavailable' : 'error',
          review: null,
          error: expertErrorMessage(error, 'load'),
        });
      });
    return () => { cancelled = true; };
  }, [apiUrl, jobId, runId, key]);

  // A result from a different run is never shown against the current one.
  const current = loaded.key === key ? loaded : { status: key ? 'loading' : 'idle', review: null, error: '' };

  const apply = useCallback(async (busyKey, request, action) => {
    setBusy(busyKey);
    setActionError('');
    try {
      const review = await request();
      setLoaded({ key, status: 'available', review, error: '' });
      return true;
    } catch (error) {
      setActionError(expertErrorMessage(error, action));
      return false;
    } finally {
      setBusy(null);
    }
  }, [key]);

  const saveReviewer = useCallback((reviewer) => apply(
    'reviewer',
    () => patchExpertReviewer(apiUrl, jobId, runId, reviewer),
    'reviewer',
  ), [apiUrl, jobId, runId, apply]);

  const setDecision = useCallback((pieceId, { decision, reason_code = null, notes = '' }) => apply(
    pieceId,
    () => patchExpertPiece(apiUrl, jobId, runId, pieceId, { decision, reason_code, notes }),
    'decision',
  ), [apiUrl, jobId, runId, apply]);

  return {
    status: current.status, // idle | loading | available | unavailable | error
    review: current.review,
    loadError: current.error,
    actionError,
    busy,
    reload: load,
    saveReviewer,
    setDecision,
  };
}
