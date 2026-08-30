import subprocess
import os
import json
import time
import shutil
import torch

COLMAP_BIN_ENV = "QUARTZ_COLMAP_BIN"
DEFAULT_COLMAP_BIN = r"D:\colmap-new\bin\colmap.exe"


def _clean_executable_path(path):
    return os.path.expandvars(os.path.expanduser(str(path).strip().strip('"')))


def _resolve_colmap_bin(env=None, which=shutil.which, isfile=os.path.isfile):
    env = os.environ if env is None else env
    configured = env.get(COLMAP_BIN_ENV)
    if configured:
        configured = _clean_executable_path(configured)
        if os.path.isabs(configured):
            return configured
        return which(configured) or configured

    candidates = []
    candidates.append(DEFAULT_COLMAP_BIN)

    path_colmap = which("colmap")
    if path_colmap:
        candidates.append(path_colmap)

    for candidate in candidates:
        candidate = _clean_executable_path(candidate)
        if os.path.isabs(candidate):
            if isfile(candidate):
                return candidate
        else:
            resolved = which(candidate)
            if resolved:
                return resolved

    return _clean_executable_path(configured or DEFAULT_COLMAP_BIN)


COLMAP_BIN = _resolve_colmap_bin()


def _ensure_colmap_available():
    if os.path.isabs(COLMAP_BIN) and os.path.isfile(COLMAP_BIN):
        return COLMAP_BIN
    if not os.path.isabs(COLMAP_BIN):
        resolved = shutil.which(COLMAP_BIN)
        if resolved:
            return resolved
    raise FileNotFoundError(
        f"COLMAP executable not found: {COLMAP_BIN}. "
        f"Set {COLMAP_BIN_ENV} to a valid colmap.exe path."
    )

# Detect CUDA once at import time.
# torch.cuda.is_available() returns True even when the GPU's compute capability
# is too new for the installed PyTorch (e.g. RTX 5050 sm_120 vs PyTorch max sm_90).
# A real tensor computation is the only reliable test.
def _pytorch_cuda_works():
    if not torch.cuda.is_available():
        return False, "none"
    import warnings
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t = torch.zeros(2, device='cuda')
            _ = torch.matmul(t.unsqueeze(0), t.unsqueeze(1)).item()
        # Treat capability-mismatch warnings as failure
        for warning in w:
            if 'cuda capability' in str(warning.message).lower():
                name = torch.cuda.get_device_name(0)
                print(f"   ⚠️  GPU {name} has CUDA capability not supported by "
                      f"this PyTorch build — falling back to CPU for PyTorch ops.")
                print("   💡 To enable GPU: pip install torch --index-url "
                      "https://download.pytorch.org/whl/cu128")
                return False, name
        return True, torch.cuda.get_device_name(0)
    except Exception as e:
        return False, "none"

_HAS_CUDA, _GPU_NAME = _pytorch_cuda_works()

_COLMAP_HELP_CACHE = {}
_COLMAP_VERSION_CACHE = {}


def _colmap_command_help(command):
    cache_key = (COLMAP_BIN, command)
    if cache_key not in _COLMAP_HELP_CACHE:
        colmap_bin = _ensure_colmap_available()
        result = subprocess.run(
            [colmap_bin, command, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _COLMAP_HELP_CACHE[cache_key] = f"{result.stdout}\n{result.stderr}"
    return _COLMAP_HELP_CACHE[cache_key]


def _colmap_version_line():
    if COLMAP_BIN not in _COLMAP_VERSION_CACHE:
        colmap_bin = _ensure_colmap_available()
        result = subprocess.run(
            [colmap_bin, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        text = f"{result.stdout}\n{result.stderr}"
        version = next(
            (line.strip() for line in text.splitlines()
            if line.strip().startswith("COLMAP ")),
            "COLMAP version unavailable",
        )
        _COLMAP_VERSION_CACHE[COLMAP_BIN] = version
    return _COLMAP_VERSION_CACHE[COLMAP_BIN]


def _supported_colmap_option(command, candidates):
    help_text = _colmap_command_help(command)
    for flag in candidates:
        if flag in help_text:
            return flag
    return None


def _colmap_option_value(command, candidates, value):
    flag = _supported_colmap_option(command, candidates)
    if not flag:
        raise RuntimeError(
            f"COLMAP {command} does not support any expected option: "
            f"{', '.join(candidates)}"
        )
    return [flag, value]


def _supported_gpu_use_flag(command):
    candidates = {
        "feature_extractor": (
            "--FeatureExtraction.use_gpu",
            "--SiftExtraction.use_gpu",
        ),
        "exhaustive_matcher": (
            "--FeatureMatching.use_gpu",
            "--SiftMatching.use_gpu",
        ),
    }.get(command, ())
    return _supported_colmap_option(command, candidates)


def _colmap_gpu_options(command):
    flag = _supported_gpu_use_flag(command)
    if not flag:
        return []
    return [flag, "1" if _HAS_CUDA else "0"]


def _feature_extractor_command(database_path, images_dir):
    return [
        COLMAP_BIN, "feature_extractor",
        "--database_path", database_path,
        "--image_path", images_dir,
        "--ImageReader.camera_model", "SIMPLE_RADIAL",
        "--ImageReader.single_camera", "1",
        *_colmap_gpu_options("feature_extractor"),
        *_colmap_option_value(
            "feature_extractor",
            (
                "--FeatureExtraction.max_image_size",
                "--SiftExtraction.max_image_size",
            ),
            "1200",
        ),
        "--SiftExtraction.max_num_features", "4096",
        "--SiftExtraction.peak_threshold", "0.004",
    ]


def _exhaustive_matcher_command(database_path):
    return [
        COLMAP_BIN, "exhaustive_matcher",
        "--database_path", database_path,
        *_colmap_gpu_options("exhaustive_matcher"),
        "--ExhaustiveMatching.block_size", "50",
    ]


def _patch_match_stereo_command(dense_dir):
    cmd = [
        COLMAP_BIN, "patch_match_stereo",
        "--workspace_path", dense_dir,
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
        "--PatchMatchStereo.window_radius", "5",
        "--PatchMatchStereo.num_iterations", "5",
    ]
    if "--PatchMatchStereo.gpu_index" in _colmap_command_help("patch_match_stereo"):
        cmd.extend(["--PatchMatchStereo.gpu_index", "0"])
    return cmd


def _looks_like_cuda_setup_failure(stderr):
    text = (stderr or "").lower()
    return (
        "cuda" in text
        or "cudart" in text
        or "no cuda-capable device" in text
        or "driver version is insufficient" in text
        or "invalid device ordinal" in text
    )


def update_status(job_path, step_name, percent, message):
    status_file = os.path.join(job_path, "status.json")
    data = {
        "status": "Processing",
        "step": step_name,
        "progress": percent,
        "message": message,
        "timestamp": time.time()
    }
    try:
        with open(status_file, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def run_command(cmd, step_label="", return_stderr=False):
    cmd_str = " ".join(cmd)
    label = f"[{step_label}] " if step_label else ""
    print(f"   {label}▶  {cmd_str}")
    t0 = time.time()

    env = os.environ.copy()
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"   ⚠️  {label}Command failed after {elapsed:.1f}s")
        print(f"   STDERR: {result.stderr[-500:]}")   # last 500 chars
        if return_stderr:
            return False, result.stderr

        return False

    print(f"   ✅ {label}Done in {elapsed:.1f}s")
    if return_stderr:
        return True, result.stderr
    return True


def get_largest_sparse_model(sparse_dir):
    best_folder = None
    max_size = -1
    if not os.path.exists(sparse_dir):
        return None
    subfolders = [f for f in os.listdir(sparse_dir) if f.isdigit()]
    if not subfolders:
        return None
    for folder in subfolders:
        points_file = os.path.join(sparse_dir, folder, "points3D.bin")
        if os.path.exists(points_file):
            size = os.path.getsize(points_file)
            if size > max_size:
                max_size = size
                best_folder = folder
    return best_folder


def run_photogrammetry_pipeline(job_path, scan_mode="turntable"):
    colmap_bin = _ensure_colmap_available()
    colmap_version = _colmap_version_line()
    images_dir   = os.path.join(job_path, "images")
    database_path = os.path.join(job_path, "database.db")
    sparse_dir   = os.path.join(job_path, "sparse")
    dense_dir    = os.path.join(job_path, "dense")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    n_images = len([f for f in os.listdir(images_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print("=" * 55)
    print(f"=== COLMAP Photogrammetry Pipeline")
    print(f"    Mode      : {scan_mode}")
    print(f"    Images    : {n_images}")
    print(f"    COLMAP    : {colmap_version}")
    print(f"    COLMAP exe: {colmap_bin}")
    print(f"    PyTorch GPU: {'YES - ' + _GPU_NAME if _HAS_CUDA else 'NO (CPU only)'}")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Step 1 — Feature Extraction
    # A database file may exist from a previous failed attempt (COLMAP creates
    # it before storing any features).  Only skip if the DB is large enough to
    # actually contain features (>100 KB is a safe threshold).
    # ------------------------------------------------------------------
    db_has_features = (os.path.exists(database_path) and
                       os.path.getsize(database_path) > 100_000)

    if db_has_features:
        print("\n[1/6] Feature Extraction — skipped (DB already populated)")
    else:
        # Remove any empty/partial database left by a previous failed run
        if os.path.exists(database_path):
            os.remove(database_path)
            print("\n[1/6] Feature Extraction — removed empty DB, re-extracting")
        else:
            print("\n[1/6] Feature Extraction")

        update_status(job_path, "Feature Extraction", 10, "Extracting SIFT features...")
        ok, stderr = run_command(
            _feature_extractor_command(database_path, images_dir),
            step_label="Feature Extractor",
            return_stderr=True,
        )
        if not ok:
            # Remove the empty DB so a resume can retry cleanly
            if os.path.exists(database_path):
                os.remove(database_path)
            if _looks_like_cuda_setup_failure(stderr):
                raise RuntimeError(
                    "COLMAP feature extraction failed during CUDA setup. "
                    "This is a COLMAP/CUDA configuration failure, not evidence "
                    "of image quality or lighting problems."
                )
            raise RuntimeError(
                "COLMAP feature extraction failed; see COLMAP stderr above."
            )

    # ------------------------------------------------------------------
    # Step 2 — Exhaustive Matching
    # ------------------------------------------------------------------
    update_status(job_path, "Matching", 25, "Matching frames...")
    print("\n[2/6] Exhaustive Matching")
    ok, stderr = run_command(
        _exhaustive_matcher_command(database_path),
        step_label="Matcher",
        return_stderr=True,
    )
    if not ok:
        if _looks_like_cuda_setup_failure(stderr):
            raise RuntimeError(
                "COLMAP exhaustive matching failed during CUDA setup. "
                "This is a COLMAP/CUDA configuration failure, not evidence "
                "of image quality or lighting problems."
            )
        raise RuntimeError(
            "COLMAP exhaustive matching failed; see COLMAP stderr above."
        )

    # ------------------------------------------------------------------
    # Step 3 — Sparse Reconstruction (mapper)
    # ------------------------------------------------------------------
    if not os.listdir(sparse_dir):
        update_status(job_path, "Sparse Reconstruction", 40,
                      "Building sparse model...")
        print("\n[3/6] Sparse Reconstruction (mapper)")
        run_command([
            COLMAP_BIN, "mapper",
            "--database_path", database_path,
            "--image_path",    images_dir,
            "--output_path",   sparse_dir,
        ], step_label="Mapper")
    else:
        print("\n[3/6] Sparse Reconstruction — skipped (already exists)")

    best_model_id = get_largest_sparse_model(sparse_dir)

    if not best_model_id:
        print("\n   ⚠️  Mapper produced no model — trying hierarchical_mapper...")
        run_command([
            COLMAP_BIN, "hierarchical_mapper",
            "--database_path", database_path,
            "--image_path",    images_dir,
            "--output_path",   sparse_dir,
        ], step_label="HierarchicalMapper")
        best_model_id = get_largest_sparse_model(sparse_dir)

    if not best_model_id:
        raise Exception("Sparse reconstruction failed — no valid sparse model found. "
                        "Check that your videos contain enough overlapping views.")

    input_sparse_path = os.path.join(sparse_dir, best_model_id)
    print(f"\n   ✔  Using sparse model: {input_sparse_path}")

    # ------------------------------------------------------------------
    # Step 4 — Image Undistorter
    # ------------------------------------------------------------------
    update_status(job_path, "Undistorting", 60, "Undistorting images...")
    print("\n[4/6] Image Undistorter")
    run_command([
        COLMAP_BIN, "image_undistorter",
        "--image_path",   images_dir,
        "--input_path",   input_sparse_path,
        "--output_path",  dense_dir,
        "--output_type",  "COLMAP",
        "--max_image_size", "800",
    ], step_label="Undistorter")

    # ------------------------------------------------------------------
    # Step 5 — Patch-Match Stereo (GPU-accelerated if available)
    # ------------------------------------------------------------------
    update_status(job_path, "Depth Maps", 75,
                  "Calculating depth maps (this takes the longest)...")
    print("\n[5/6] Patch-Match Stereo  (COLMAP GPU index 0)")
    print("   ⏳  This step can take several minutes — please wait...")
    run_command(_patch_match_stereo_command(dense_dir), step_label="PatchMatch")

    # ------------------------------------------------------------------
    # Step 6 — Stereo Fusion
    # ------------------------------------------------------------------
    update_status(job_path, "Point Fusion", 90, "Fusing point cloud...")
    print("\n[6/6] Stereo Fusion")
    fused_ply_path = os.path.join(dense_dir, "fused.ply")
    run_command([
        COLMAP_BIN, "stereo_fusion",
        "--workspace_path",   dense_dir,
        "--workspace_format", "COLMAP",
        "--output_path",      fused_ply_path,
        "--StereoFusion.check_num_images", "1",
        "--StereoFusion.min_num_pixels",   "5",
    ], step_label="Fusion")

    # Fallback to sparse point cloud if dense fusion produced nothing
    is_empty = (
        not os.path.exists(fused_ply_path)
        or os.path.getsize(fused_ply_path) < 5000
    )
    if is_empty:
        print("\n   ⚠️  Dense fusion empty — falling back to sparse point cloud.")
        update_status(job_path, "Fallback", 95,
                      "Dense failed — using sparse model instead...")
        run_command([
            COLMAP_BIN, "model_converter",
            "--input_path",  input_sparse_path,
            "--output_path", fused_ply_path,
            "--output_type", "PLY",
        ], step_label="SparseConverter")

    print("\n✅ COLMAP pipeline complete.")
    update_status(job_path, "Completed", 100, "Done.")
    return fused_ply_path
