import React, { useLayoutEffect, useRef, useState } from 'react';
import { HelpCircle } from 'lucide-react';
import { METRIC_EXPLANATIONS } from '../utils/metricExplanations';

const VIEWPORT_MARGIN = 8;   // never let the tooltip touch the screen edge
const ANCHOR_GAP = 6;        // gap between the "?" icon and the tooltip
const MAX_TOOLTIP_WIDTH = 256; // px (~16rem) on screens with room for it

// Anchored to the icon's live position on screen (not the icon's parent),
// so it works the same whether that parent scrolls, clips overflow, or sits
// inside a card near a viewport edge.
function computeTooltipPosition(anchorRect, tooltipWidth, tooltipHeight) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  // Horizontal: open toward the right when there's room, else toward the
  // left, else clamp flush against whichever edge fits.
  let left;
  if (anchorRect.left + tooltipWidth + VIEWPORT_MARGIN <= vw) {
    left = anchorRect.left;
  } else if (anchorRect.right - tooltipWidth - VIEWPORT_MARGIN >= 0) {
    left = anchorRect.right - tooltipWidth;
  } else {
    left = vw - tooltipWidth - VIEWPORT_MARGIN;
  }
  left = Math.max(VIEWPORT_MARGIN, Math.min(left, vw - tooltipWidth - VIEWPORT_MARGIN));

  // Vertical: prefer opening below the icon; flip above when there isn't
  // room below but there is above.
  const spaceBelow = vh - anchorRect.bottom - ANCHOR_GAP;
  const spaceAbove = anchorRect.top - ANCHOR_GAP;
  let top;
  if (tooltipHeight <= spaceBelow || spaceBelow >= spaceAbove) {
    top = anchorRect.bottom + ANCHOR_GAP;
  } else {
    top = anchorRect.top - ANCHOR_GAP - tooltipHeight;
  }
  top = Math.max(VIEWPORT_MARGIN, Math.min(top, vh - tooltipHeight - VIEWPORT_MARGIN));

  return { top, left };
}

/**
 * Small "?" affordance that reveals a metric's formula/meaning on click or
 * hover. Pass `metricKey` to pull from the shared registry, or override any
 * field (title/formula/meaning/boundary) directly.
 *
 * Positioned with `position: fixed` and measured against the real viewport
 * on open (see computeTooltipPosition) so it can never render off-screen or
 * get clipped by a scrolling ancestor — it always opens on whichever side
 * (and above/below) actually has room.
 */
export default function MetricHelp({
  metricKey, title, formula, meaning, boundary, className = '',
}) {
  const [open, setOpen] = useState(false);
  const [coords, setCoords] = useState(null);
  const buttonRef = useRef(null);
  const tooltipRef = useRef(null);

  const entry = METRIC_EXPLANATIONS[metricKey] || {};
  const resolvedTitle = title || entry.title || 'Metric';
  const resolvedFormula = formula || entry.formula;
  const resolvedMeaning = meaning || entry.meaning;
  const resolvedBoundary = boundary || entry.boundary;

  const tooltipWidth = typeof window !== 'undefined'
    ? Math.min(MAX_TOOLTIP_WIDTH, window.innerWidth - VIEWPORT_MARGIN * 2)
    : MAX_TOOLTIP_WIDTH;

  const close = () => { setOpen(false); setCoords(null); };

  // Measure the tooltip (rendered off-screen at its final width) against
  // the icon's current screen position, then place it. Re-run whenever it
  // opens; close instead of re-tracking on scroll/resize so it never ends
  // up pinned in a stale spot after the page moves under it.
  useLayoutEffect(() => {
    if (!open || !buttonRef.current || !tooltipRef.current) return undefined;
    const anchorRect = buttonRef.current.getBoundingClientRect();
    const { offsetWidth, offsetHeight } = tooltipRef.current;
    setCoords(computeTooltipPosition(anchorRect, offsetWidth, offsetHeight));

    window.addEventListener('resize', close);
    window.addEventListener('scroll', close, true);
    return () => {
      window.removeEventListener('resize', close);
      window.removeEventListener('scroll', close, true);
    };
  }, [open]);

  return (
    <span className={`relative inline-flex align-middle ${className}`}>
      <button
        ref={buttonRef}
        type="button"
        onClick={(event) => { event.stopPropagation(); setOpen((value) => !value); }}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={close}
        onBlur={close}
        aria-label={`About ${resolvedTitle}`}
        aria-expanded={open}
        className="ml-1 grid h-3.5 w-3.5 shrink-0 place-items-center rounded-full border border-slate-600 text-slate-500 transition-colors hover:border-cyan-400 hover:text-cyan-400 focus:border-cyan-400 focus:text-cyan-400 focus:outline-none"
      >
        <HelpCircle className="h-2.5 w-2.5" />
      </button>
      {open && (
        <div
          ref={tooltipRef}
          role="tooltip"
          style={{
            position: 'fixed',
            width: tooltipWidth,
            top: coords ? coords.top : -9999,
            left: coords ? coords.left : -9999,
            visibility: coords ? 'visible' : 'hidden',
            overflowWrap: 'anywhere',
            wordBreak: 'break-word',
          }}
          className="z-50 max-w-[calc(100vw-16px)] rounded-lg border border-slate-700 bg-slate-950 p-2.5 text-[10px] leading-4 text-slate-300 shadow-2xl"
        >
          <div className="mb-1 font-semibold text-white">{resolvedTitle}</div>
          {resolvedFormula && (
            <div className="mb-1 rounded bg-slate-900 px-1.5 py-1 font-mono text-[9.5px] text-cyan-300">
              {resolvedFormula}
            </div>
          )}
          {resolvedMeaning && <p className="mb-1">{resolvedMeaning}</p>}
          {resolvedBoundary && <p className="text-amber-400/90">{resolvedBoundary}</p>}
        </div>
      )}
    </span>
  );
}
