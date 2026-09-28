"""Independent candidate evaluation only; beam/search mutation stays in the parent.

loky supplies child-only environment settings before numerical libraries import.
This avoids modifying the server/reconstruction environment or nesting pools.
"""
import os
import time

_CONTEXT = None
_VARIANTS = None
_BOUNDS = None
_MAX_DIM = None


def configuration(environ=None, logical_cpus=None):
    env = os.environ if environ is None else environ
    logical = max(1, logical_cpus if logical_cpus is not None else (os.cpu_count() or 1))
    mode = env.get("OPTIMIZER_PERFORMANCE_MODE", "full").lower()
    if mode not in {"full", "sequential"}:
        mode = "sequential"
    requested = env.get("OPTIMIZER_MAX_WORKERS", "auto").lower()
    try:
        workers = max(1, logical-1) if requested == "auto" else max(1, min(logical, int(requested)))
    except ValueError:
        workers = 1
    return {"logical_cpu_count": logical, "optimizer_worker_count": workers if mode == "full" else 1,
            "performance_mode": mode, "candidate_count": 0, "parallel_evaluation_seconds": 0.0}


def _initialize(ctx, variants, bounds, max_dim):
    global _CONTEXT, _VARIANTS, _BOUNDS, _MAX_DIM
    _CONTEXT, _VARIANTS, _BOUNDS, _MAX_DIM = ctx, variants, bounds, max_dim
    # Workers evaluate immutable geometry. Counters are worker-local and returned
    # as deltas; no worker ever enters the parent beam search or creates a pool.
    for value in (ctx.grid, ctx.origin, ctx.rough_mask, ctx.no_cut_mask):
        value.flags.writeable = False


def _pack(placement):
    return (placement.pos, placement.scale, placement.volume,
            placement.occ_flat, placement.surface_clearance,
            tuple(placement.collision_set))


def _unpack(value, variant):
    from optimizer import Placement
    pos, scale, volume, flat, clearance, collision = value
    return Placement(variant, pos, scale, volume, flat, set(flat.tolist()), set(collision), clearance)


def _evaluate(task):
    import optimizer
    ctx = _CONTEXT
    before = ctx.confirmed_guard.rejected
    started = time.process_time()
    kind, index, payload = task
    variant = _VARIANTS[index]
    found, rejected = [], {}
    if kind == "orientation":
        for point in payload:
            if ctx.candidate_deadline is not None and time.time() >= ctx.candidate_deadline:
                ctx.confirmed_guard.search_limited = True
                break
            placement, reason = optimizer._max_scale_for(variant, point, _BOUNDS, _MAX_DIM, ctx)
            if placement is None:
                rejected[reason] = rejected.get(reason, 0)+1
            else:
                found.append(placement)
        found.sort(key=lambda p: p.volume, reverse=True)
        found = optimizer._spatially_diverse_candidates(found)[:optimizer.TOP_PER_VARIANT]
    else:
        placement = _unpack(payload, variant)
        for factor in optimizer.CANDIDATE_SCALE_FACTORS:
            strict, reason = optimizer._scaled_strict_placement(placement, factor, ctx)
            if strict is None:
                rejected[reason] = rejected.get(reason, 0)+1
            else:
                found.append(strict)
    return (index, [_pack(p) for p in found], rejected,
            ctx.confirmed_guard.rejected-before, ctx.confirmed_guard.search_limited,
            os.getpid(), time.process_time()-started)


def evaluate_candidates(variants, search_points, ctx, bounds, max_dim, diagnostics, progress=None):
    """Return candidates in stable input order or None for sequential fallback."""
    import optimizer
    count = min(diagnostics["optimizer_worker_count"], len(variants))
    if diagnostics["performance_mode"] != "full" or count < 2:
        diagnostics["optimizer_worker_count"] = 1
        return None
    started = time.perf_counter()
    pool = None
    rejected, worker_pids = {}, set()
    defect_rejections, limited, cpu_seconds = 0, False, 0.0
    try:
        from loky import ProcessPoolExecutor
        # Environment is applied only to spawned optimizer children, not to the
        # parent or any reconstruction process. One numerical thread per worker.
        child_env = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS")}
        child_env["OPTIMIZER_MAX_WORKERS"] = "1"
        child_env["OPTIMIZER_PERFORMANCE_MODE"] = "sequential"
        pool = ProcessPoolExecutor(max_workers=count, initializer=_initialize,
            initargs=(ctx, variants, bounds, max_dim), env=child_env)

        def collect(tasks):
            nonlocal defect_rejections, limited, cpu_seconds
            output = []
            futures = [pool.submit(_evaluate, task) for task in tasks]
            for index, future in enumerate(futures):
                # Ordered reduction makes tie-breaking independent of completion order.
                vi, values, failures, defects, stopped, pid, cpu = future.result(timeout=90)
                output.extend(_unpack(value, variants[vi]) for value in values)
                for reason, value in failures.items():
                    rejected[reason] = rejected.get(reason, 0)+value
                defect_rejections += defects
                limited = limited or stopped
                worker_pids.add(pid)
                cpu_seconds += cpu
                if progress and index % max(1, len(tasks)//10) == 0:
                    progress(index, len(tasks), "Parallel confirmed-defect placement checks")
            return output

        candidates = collect([("orientation", i, search_points) for i in range(len(variants))])
        candidates.sort(key=lambda p: p.volume, reverse=True)
        candidates = optimizer._spatially_diverse_candidates(candidates[:optimizer.MAX_CANDIDATES*2])
        lookup = {variant.key: index for index, variant in enumerate(variants)}
        strict = collect([("strict", lookup[p.variant.key], _pack(p))
                          for p in candidates[:optimizer.CANDIDATE_SCALE_SOURCE_LIMIT]])
        strict.sort(key=lambda p: p.volume, reverse=True)
        strict = optimizer._unique_candidate_variants(strict)[:optimizer.MAX_CANDIDATES]
        ctx.confirmed_guard.rejected += defect_rejections
        ctx.confirmed_guard.search_limited |= limited
        diagnostics.update(optimizer_worker_count=len(worker_pids), candidate_count=len(strict),
                           worker_cpu_seconds=cpu_seconds, parallel_fallback=False)
        return strict, rejected
    except Exception as exc:
        # Partial work is discarded, including worker counters. Restart the
        # sequential candidate budget after a failed process initialization.
        ctx.candidate_deadline = time.time()+40 if ctx.candidate_deadline is not None else None
        diagnostics.update(optimizer_worker_count=1, parallel_fallback=True,
                           parallel_fallback_reason=type(exc).__name__)
        return None
    finally:
        if pool is not None:
            pool.shutdown(wait=True, kill_workers=diagnostics.get("parallel_fallback", False))
        elapsed = time.perf_counter()-started
        diagnostics["parallel_evaluation_seconds"] = elapsed
        diagnostics["parallel_cpu_utilization_percent"] = (
            100*cpu_seconds/max(elapsed*diagnostics["logical_cpu_count"], 1e-9))
