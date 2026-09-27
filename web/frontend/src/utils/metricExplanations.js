// Reference copy for the MetricHelp tooltip component. Text here describes
// how each metric is computed/sourced today so the presentation layer never
// overstates what the backend actually measures — update the wording if the
// underlying computation changes, but this file has no effect on any
// calculation itself.
export const METRIC_EXPLANATIONS = {
  yield: {
    title: 'Yield %',
    formula: 'Yield % = (finished cut volume ÷ rough stone volume) × 100',
    meaning: 'Share of the original rough stone that survives as the finished, sellable gem(s) under this strategy.',
  },
  planShare: {
    title: 'Plan share %',
    formula: "Plan share % = (this gem's volume ÷ total planned-cut volume) × 100",
    meaning: "How much of this strategy's total finished weight one gem accounts for. Shares across all gems in a plan add up to 100%.",
  },
  roughYield: {
    title: 'Rough yield %',
    formula: "Rough yield % = (this gem's volume ÷ original rough stone volume) × 100",
    meaning: "This gem's own share of the original rough stone. Summed across every gem in the plan, it approximates the strategy's overall Yield %.",
  },
  spaceUtilization: {
    title: 'Space utilization',
    formula: 'Occupied % = (occupied volume ÷ usable rough volume) × 100 — usable volume excludes fracture / no-cut zones',
    meaning: "How much of the rough stone's defect-free volume is actually filled by placed gems. Distinct from Yield %, which is measured against the whole rough stone, defects included.",
  },
  waste: {
    title: 'Waste %',
    formula: 'Waste % = 100 − Yield %',
    meaning: 'Estimated material not retained in the finished gem(s) under this strategy.',
    boundary: 'A projected planning figure, not a measured cutting loss. Any comparison baseline shown alongside it is an internal reference assumption, not a validated traditional-cutting benchmark.',
  },
  manufacturingClearance: {
    title: 'Manufacturing clearance',
    formula: 'Protected corridor = blade kerf + (2 × preform margin)',
    meaning: "Blade kerf is material lost to the saw blade's width; preform margin is extra stock left on each gem for post-cut shaping and polishing. Together they set the minimum gap enforced between neighboring gems.",
  },
  surfaceClearance: {
    title: 'Surface clearance',
    formula: "Minimum distance from this gem's surface to the original rough stone boundary",
    meaning: "Keeps the finished gem inset from the rough surface — where saw marks, scale-calibration error, or unmapped inclusions are most likely — rather than breaking through it. Checked against the configured rough-clearance target; distinct from the blade-kerf/preform gap enforced between neighboring gems.",
  },
  sawSequence: {
    title: 'Saw sequence',
    meaning: 'The ordered list of straight cuts needed to separate the rough stone into its planned gem preforms.',
    formula: 'Each cut records a cutting plane, the required depth (mm) to fully separate that piece, and the kerf-slab thickness removed by the blade.',
    boundary: 'Operator guidance only — not machine-ready CNC or G-code.',
  },
  lightPerformance: {
    title: 'Light performance',
    meaning: 'A relative brilliance indicator, useful for comparing cutting strategies against each other.',
    boundary: 'Computed with a simplified geometric light-return heuristic, not a physically-based optical or ray-traced rendering simulation. Treat it as directional, not an absolute optical measurement.',
  },
  facetScore: {
    title: 'Facet score',
    meaning: 'Reflects how well the recommended facet-table orientation keeps visible, mapped defects out of the face-up view.',
    boundary: 'Produced by a machine-learning recommender, or a geometric heuristic when that is unavailable — a planning aid, not a rendered simulation of the polished gem.',
  },
  defectCount: {
    title: 'Defect count',
    meaning: 'Number of inclusion / fracture points detected in the source footage and successfully mapped into the 3D model as accepted evidence.',
    boundary: 'Reflects visible, mapped, accepted evidence only — not an exhaustive internal scan. Inclusions invisible from every captured camera angle are not counted.',
  },
};
