// Presentation (demo) mode: the simplified final-gemstone workflow
// (Inspect → Defect Review → Final Gemstones → Cut Sequence). Experimental
// optimizer modes, Expert Review and legacy comparison cards stay in the
// code and are reachable with VITE_PRESENTATION_MODE=false.
//
// Enabled unless explicitly set to "false" (the final demo build default).
export function isPresentationMode(value) {
  return String(value ?? '').trim().toLowerCase() !== 'false';
}

export const PRESENTATION_MODE = isPresentationMode(import.meta.env.VITE_PRESENTATION_MODE);
