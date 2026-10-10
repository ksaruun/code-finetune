"""
dry_run.py
──────────
Pre-flight check for the code-finetune pipeline.

Checks hardware, Python environment, dependencies, cached models,
and external tools — without downloading or training anything.

Usage:
    python3 dry_run.py
    python3 dry_run.py --verbose
"""

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

# ── Status symbols ────────────────────────────────────────────────────────────
OK      = "✅"
WARN    = "⚠️ "
FAIL    = "❌"
INFO    = "ℹ️ "
SKIP    = "⏭️ "

SEP = "─" * 64


@dataclass
class CheckResult:
    label: str
    status: str          # OK | WARN | FAIL | INFO | SKIP
    detail: str = ""
    fix: str = ""


results: list[CheckResult] = []


def add(label: str, status: str, detail: str = "", fix: str = ""):
    results.append(CheckResult(label, status, detail, fix))


def section(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def print_result(r: CheckResult, verbose: bool):
    sym = r.status
    line = f"  {sym}  {r.label}"
    if r.detail:
        line += f"  →  {r.detail}"
    print(line)
    if r.fix and (verbose or r.status in (FAIL, WARN)):
        print(f"       💡 {r.fix}")


# ── 1. Platform & Hardware ────────────────────────────────────────────────────

def check_platform():
    section("1 · Platform & Hardware")

    sys_info = platform.uname()
    add("OS", INFO, f"{sys_info.system} {sys_info.release} ({sys_info.machine})")

    # Python version
    py = sys.version_info
    py_str = f"Python {py.major}.{py.minor}.{py.micro}"
    if py.major == 3 and py.minor >= 10:
        add("Python version", OK, py_str)
    else:
        add("Python version", FAIL, py_str,
            fix="Install Python 3.10+ (required by Unsloth & modern transformers)")

    # Apple Silicon check
    is_apple_silicon = (sys_info.system == "Darwin" and sys_info.machine in ("arm64", "aarch64"))
    if is_apple_silicon:
        # Try to get chip name from system_profiler
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            chip = "Apple Silicon"
        add("Apple Silicon", OK, chip)
    elif sys_info.system == "Darwin":
        add("Apple Silicon", WARN, "Intel Mac detected — MPS not available, CPU only",
            fix="This pipeline is optimised for Apple Silicon (M1/M2/M3/M4)")
    else:
        add("Apple Silicon", SKIP, "Non-macOS system — GPU checks differ")

    # RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        ram_str = f"{ram_gb:.1f} GB total, {avail_gb:.1f} GB available"
        status = OK if ram_gb >= 16 else (WARN if ram_gb >= 8 else FAIL)
        fix = "" if ram_gb >= 16 else "16 GB RAM recommended for 7B models; 8 GB may work for 3B"
        add("RAM", status, ram_str, fix=fix)
    except ImportError:
        add("RAM", WARN, "psutil not installed — cannot read RAM",
            fix="pip install psutil")

    # Disk space (project dir)
    try:
        total, used, free = shutil.disk_usage(os.path.dirname(os.path.abspath(__file__)))
        free_gb = free / (1024 ** 3)
        disk_str = f"{free_gb:.1f} GB free on current disk"
        status = OK if free_gb >= 10 else (WARN if free_gb >= 5 else FAIL)
        fix = "" if free_gb >= 10 else "Need ~6 GB for model download + adapter outputs"
        add("Disk space", status, disk_str, fix=fix)
    except Exception as e:
        add("Disk space", WARN, f"Could not check: {e}")

    # CPU cores
    cpu_count = os.cpu_count() or 1
    add("CPU cores", INFO, f"{cpu_count} logical cores")


# ── 2. PyTorch & Accelerators ─────────────────────────────────────────────────

def check_pytorch():
    section("2 · PyTorch & Accelerators")

    try:
        import torch
        add("PyTorch", OK, f"version {torch.__version__}")

        # MPS (Apple Silicon GPU)
        if torch.backends.mps.is_available():
            add("MPS (Apple Silicon GPU)", OK, "available — will be used for training")
        elif hasattr(torch.backends, "mps") and not torch.backends.mps.is_available():
            add("MPS (Apple Silicon GPU)", WARN,
                "MPS built but not available",
                fix="Ensure macOS 12.3+ and a recent PyTorch build")
        else:
            add("MPS (Apple Silicon GPU)", SKIP, "not applicable on this platform")

        # CUDA
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram = props.total_memory / (1024 ** 3)
                add(f"CUDA GPU [{i}]", OK,
                    f"{props.name}, {vram:.1f} GB VRAM")
        else:
            add("CUDA GPU", SKIP, "not available (expected on Apple Silicon)")

        # Recommended dtype
        if torch.backends.mps.is_available():
            add("Recommended dtype", INFO,
                "float32 (MPS does not support fp16/bf16 for training)")
        elif torch.cuda.is_available():
            add("Recommended dtype", INFO, "bf16 or fp16 for NVIDIA GPU")

    except ImportError:
        add("PyTorch", FAIL, "not installed",
            fix="pip install torch>=2.1.0")


# ── 3. Python Package Dependencies ────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("transformers",     "transformers",   "4.40.0"),
    ("datasets",         "datasets",       "2.18.0"),
    ("peft",             "peft",           "0.10.0"),
    ("trl",              "trl",            "0.8.6"),
    ("accelerate",       "accelerate",     "0.28.0"),
    ("bitsandbytes",     "bitsandbytes",   "0.43.0"),
    ("unsloth",          "unsloth",        None),
    ("huggingface_hub",  "huggingface_hub","0.22.0"),
    ("sentencepiece",    "sentencepiece",  "0.2.0"),
    ("scipy",            "scipy",          "1.12.0"),
    ("evaluate",         "evaluate",       "0.4.1"),
]

OPTIONAL_PACKAGES = [
    ("psutil",  "psutil",  None, "RAM/CPU monitoring in dry-run"),
    ("wandb",   "wandb",   None, "Experiment tracking (optional)"),
]


def _pkg_version(import_name: str) -> Optional[str]:
    """Return installed version string or None."""
    try:
        mod = importlib.import_module(import_name)
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return None


def _version_ok(installed: str, minimum: str) -> bool:
    """Return True if installed >= minimum (simple dot-comparison)."""
    try:
        from packaging.version import Version
        return Version(installed) >= Version(minimum)
    except Exception:
        # Fall back to string prefix — good enough for most cases
        return installed >= minimum


def check_dependencies():
    section("3 · Python Dependencies")

    missing = []
    for (display, import_name, min_ver) in REQUIRED_PACKAGES:
        ver = _pkg_version(import_name)
        if ver is None:
            add(display, FAIL, "not installed",
                fix=f"pip install {display}" + (f">={min_ver}" if min_ver else ""))
            missing.append(display)
        elif min_ver and ver != "installed":
            ok = _version_ok(ver, min_ver)
            status = OK if ok else WARN
            note = f"v{ver}" + ("" if ok else f" (need ≥ {min_ver})")
            fix = f"pip install --upgrade {display}>={min_ver}" if not ok else ""
            add(display, status, note, fix=fix)
        else:
            add(display, OK, f"v{ver}" if ver != "installed" else "installed")

    section("3b · Optional Packages")
    for (display, import_name, min_ver, purpose) in OPTIONAL_PACKAGES:
        ver = _pkg_version(import_name)
        if ver is None:
            add(display, WARN, f"not installed  [{purpose}]",
                fix=f"pip install {display}")
        else:
            add(display, OK, f"v{ver}  [{purpose}]")

    return missing


# ── 4. Config Validation ──────────────────────────────────────────────────────

def check_config():
    section("4 · Config (config.py)")

    try:
        import config as cfg

        add("BASE_MODEL", INFO, cfg.BASE_MODEL)
        add("MAX_SEQ_LENGTH", INFO, str(cfg.MAX_SEQ_LENGTH))
        add("MAX_SAMPLES", INFO,
            str(cfg.MAX_SAMPLES) if cfg.MAX_SAMPLES else "None (full 18K dataset)")
        add("LORA_R / ALPHA", INFO, f"{cfg.LORA_R} / {cfg.LORA_ALPHA}")
        add("NUM_TRAIN_EPOCHS", INFO, str(cfg.NUM_TRAIN_EPOCHS))
        add("PER_DEVICE_BATCH", INFO, str(cfg.PER_DEVICE_BATCH))
        add("LEARNING_RATE", INFO, str(cfg.LEARNING_RATE))
        add("OUTPUT_DIR", INFO, os.path.abspath(cfg.OUTPUT_DIR))
        add("ADAPTER_DIR", INFO, os.path.abspath(cfg.ADAPTER_DIR))

        # Warn about fp16/bf16 on MPS
        try:
            import torch
            if torch.backends.mps.is_available():
                if cfg.FP16 or cfg.BF16:
                    add("FP16/BF16 on MPS", WARN,
                        f"FP16={cfg.FP16}, BF16={cfg.BF16}  (MPS training requires float32)",
                        fix="Set FP16=False, BF16=False in config.py")
                else:
                    add("FP16/BF16 on MPS", OK, "both False — correct for Apple Silicon")
        except ImportError:
            pass

        # Check if adapter already exists
        adapter_exists = os.path.isdir(cfg.ADAPTER_DIR)
        add("LoRA adapter saved?",
            OK if adapter_exists else WARN,
            cfg.ADAPTER_DIR + (" (found)" if adapter_exists else " (not yet — run train.py)"),
            fix="" if adapter_exists else "Run: python train.py")

        # Check if merged model exists
        merged_exists = os.path.isdir(cfg.MERGED_DIR)
        add("Merged model saved?",
            OK if merged_exists else INFO,
            cfg.MERGED_DIR + (" (found)" if merged_exists else " (optional — only needed for Ollama export)"))

    except ImportError as e:
        add("config.py", FAIL, f"Could not import: {e}")


# ── 5. HuggingFace Cache ──────────────────────────────────────────────────────

def check_hf_cache():
    section("5 · HuggingFace Cache")

    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hub_dir = os.path.join(hf_home, "hub")
    add("HF cache dir", INFO, hf_home)

    if not os.path.isdir(hub_dir):
        add("Cached models", WARN, "No HF hub cache found — model will be downloaded on first run",
            fix=f"Run python train.py to trigger download (~3–7 GB depending on model)")
        return

    # List cached model snapshots
    cached = []
    for entry in os.scandir(hub_dir):
        if entry.is_dir() and entry.name.startswith("models--"):
            name = entry.name.replace("models--", "").replace("--", "/")
            try:
                size = sum(
                    f.stat().st_size
                    for f in os.scandir(os.path.join(entry.path, "snapshots"))
                    for f in _walk_files(f.path)
                    if f.is_file()
                )
                size_gb = size / (1024 ** 3)
                cached.append((name, size_gb))
            except Exception:
                cached.append((name, None))

    if cached:
        for name, size_gb in cached:
            size_str = f"{size_gb:.2f} GB" if size_gb is not None else "size unknown"
            add(f"Cached: {name}", OK, size_str)
    else:
        add("Cached models", WARN, "No models cached yet",
            fix="Model will be downloaded on first run (~3–7 GB)")

    # Check if configured base model is cached
    try:
        import config as cfg
        model_dir_name = "models--" + cfg.BASE_MODEL.replace("/", "--")
        model_cached = os.path.isdir(os.path.join(hub_dir, model_dir_name))
        add("Configured model cached?",
            OK if model_cached else WARN,
            cfg.BASE_MODEL + (" ✓" if model_cached else " — will be downloaded"),
            fix="" if model_cached else "Approx download size: 3B model ~2 GB, 7B model ~5 GB")
    except ImportError:
        pass


def _walk_files(path: str):
    """Recursively yield os.DirEntry for files."""
    try:
        for e in os.scandir(path):
            if e.is_dir(follow_symlinks=False):
                yield from _walk_files(e.path)
            else:
                yield e
    except PermissionError:
        pass


# ── 6. External Tools ─────────────────────────────────────────────────────────

def check_external_tools():
    section("6 · External Tools")

    # Ollama
    ollama_path = shutil.which("ollama")
    if ollama_path:
        try:
            ver = subprocess.check_output(["ollama", "--version"], text=True,
                                          stderr=subprocess.STDOUT).strip()
            add("Ollama", OK, f"{ver}  ({ollama_path})")

            # Is it running?
            ping = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=3)
            if ping.returncode == 0:
                add("Ollama server", OK, "running")
            else:
                add("Ollama server", WARN, "installed but not running",
                    fix="Run: ollama serve  (in a separate terminal)")
        except Exception:
            add("Ollama", WARN, f"installed at {ollama_path} but could not query version")
    else:
        add("Ollama", WARN, "not found in PATH",
            fix="Install from https://ollama.com — only needed for export_to_ollama.py")

    # llama.cpp
    llama_cpp = os.path.expanduser("~/llama.cpp")
    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    if os.path.isdir(llama_cpp) and os.path.isfile(convert_script):
        add("llama.cpp", OK, f"found at {llama_cpp}")
    elif os.path.isdir(llama_cpp):
        add("llama.cpp", WARN, f"directory exists but convert_hf_to_gguf.py missing",
            fix="Re-clone: git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
    else:
        add("llama.cpp", WARN, "not found at ~/llama.cpp",
            fix="git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp  (only needed for Ollama export)")

    # git
    git_path = shutil.which("git")
    if git_path:
        try:
            ver = subprocess.check_output(["git", "--version"], text=True).strip()
            add("git", OK, ver)
        except Exception:
            add("git", WARN, "found but could not query version")
    else:
        add("git", FAIL, "not found", fix="Install git (required to clone llama.cpp)")


# ── 7. Virtual Environment ────────────────────────────────────────────────────

def check_venv():
    section("7 · Virtual Environment")

    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        add("Virtual env", OK, f"active  ({sys.prefix})")
    else:
        add("Virtual env", WARN, "not active — running in system Python",
            fix="python3 -m venv venv && source venv/bin/activate")

    add("Python executable", INFO, sys.executable)
    add("pip path", INFO, shutil.which("pip") or "not found")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary():
    print(f"\n{SEP}")
    print("  SUMMARY")
    print(SEP)

    fails  = [r for r in results if r.status == FAIL]
    warns  = [r for r in results if r.status == WARN]
    oks    = [r for r in results if r.status == OK]

    print(f"  {OK}  {len(oks)} checks passed")
    print(f"  {WARN}  {len(warns)} warnings")
    print(f"  {FAIL}  {len(fails)} failures")

    if fails:
        print(f"\n  🚨 Blocking issues (fix before running pipeline):")
        for r in fails:
            print(f"     • {r.label}: {r.detail}")
            if r.fix:
                print(f"       → {r.fix}")

    if warns:
        print(f"\n  ⚠️  Warnings (non-blocking, but review):")
        for r in warns:
            print(f"     • {r.label}: {r.detail}")
            if r.fix:
                print(f"       → {r.fix}")

    if not fails and not warns:
        print(f"\n  🎉 All clear! Run the pipeline:")
        print(f"     python prepare_data.py")
        print(f"     python train.py")
        print(f"     python inference.py")
    elif not fails:
        print(f"\n  ✅ No blocking issues — pipeline should work.")
        print(f"     Review warnings above for best results.")
    else:
        print(f"\n  ❌ Fix the blocking issues above before proceeding.")

    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dry-run pre-flight check for the code-finetune pipeline"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show fix hints for all checks, not just warnings/failures"
    )
    args = parser.parse_args()

    print("\n🔍  code-finetune — Dry Run / Pre-flight Check")
    print(f"    {platform.node()}  |  {platform.platform()}")

    check_venv()
    check_platform()
    check_pytorch()
    missing_pkgs = check_dependencies()
    check_config()
    check_hf_cache()
    check_external_tools()

    # Print all results grouped by section
    section_titles = {
        "7 · Virtual Environment": [],
        "1 · Platform & Hardware": [],
        "2 · PyTorch & Accelerators": [],
        "3 · Python Dependencies": [],
        "3b · Optional Packages": [],
        "4 · Config (config.py)": [],
        "5 · HuggingFace Cache": [],
        "6 · External Tools": [],
    }

    print(f"\n{SEP}")
    print("  FULL RESULTS")
    print(SEP)
    for r in results:
        print_result(r, args.verbose)

    print_summary()


if __name__ == "__main__":
    main()
