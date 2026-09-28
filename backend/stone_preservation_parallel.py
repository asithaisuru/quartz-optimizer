"""Bounded independent cut verification; all scoring/physical accounting is central."""
import os
import time


def _verify(arguments):
    from cut_sequence import plan_cut_sequence
    rough, envelopes, kwargs = arguments
    return os.getpid(), plan_cut_sequence(rough, envelopes, **kwargs)


def verify_batch(arguments, diagnostics):
    from optimizer_parallel import configuration
    configuration_value = configuration()
    workers = min(configuration_value["optimizer_worker_count"], len(arguments), 4)
    diagnostics.update(logical_cpu_count=configuration_value["logical_cpu_count"],
                       performance_mode=configuration_value["performance_mode"])
    if workers < 2 or configuration_value["performance_mode"] != "full":
        diagnostics["optimizer_worker_count"] = 1
        return [None]*len(arguments)
    started = time.perf_counter()
    try:
        from loky import ProcessPoolExecutor
        env = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
        env.update(OPTIMIZER_MAX_WORKERS="1", OPTIMIZER_PERFORMANCE_MODE="sequential")
        with ProcessPoolExecutor(max_workers=workers, env=env) as pool:
            values = list(pool.map(_verify, arguments))
        diagnostics["optimizer_worker_count"] = max(diagnostics.get("optimizer_worker_count", 1), len({v[0] for v in values}))
        return [value[1] for value in values]
    except Exception as exc:
        diagnostics.update(parallel_fallback=True, parallel_fallback_reason=type(exc).__name__)
        return [None]*len(arguments)
    finally:
        diagnostics["parallel_evaluation_seconds"] = diagnostics.get("parallel_evaluation_seconds", 0)+time.perf_counter()-started
