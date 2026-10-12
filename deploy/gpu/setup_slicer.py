# Build-time setup, runs INSIDE Slicer's Python (docker build), in two Slicer sessions:
#   phase1: install extensions, torch (cu121) and nnunetv2
#   phase2: (extension modules are loaded only after a restart) let the TotalSegmentator
#           extension install its Python package with its OWN logic, exactly as it did on the
#           verified Windows install (it strips Slicer-managed requirements from the package
#           METADATA; a plain pip install leaves them in and the CLI then aborts on
#           requests==2.27.1), then download the open 'total' fast weights.
# Every pip call is constrained to the verified Windows `pip freeze` via PIP_CONSTRAINT.
# Logs go to a file (Slicer's stdout is unreliable headless); success = marker file.
import datetime
import os
import sys
import traceback
from pathlib import Path

LOG = Path("/opt/medops-setup.log")
OK = Path("/opt/medops-setup.ok")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "phase1"

# Same as the verified Windows install (Slicer 5.8.1 r33241, Python 3.9).
EXTENSIONS = ["PyTorch", "NNUNet", "TotalSegmentator", "SlicerVMTK", "Sandbox"]
TORCH = "torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121"
NNUNET = "nnunetv2==2.5.2"
WINDOWS_CONSTRAINTS = Path("/tmp/constraints.txt")
LINUX_CONSTRAINTS = Path("/tmp/constraints-linux.txt")
# Deviations from the Windows freeze, only where Linux has no installable wheel: Slicer's
# Python accepts manylinux2014 (manylinux_2_17) at most, and pyarrow>=21 ships only
# manylinux_2_28. pyarrow is a declared TotalSegmentator dependency that its code never
# imports, so this cannot change any measurement.
LINUX_OVERRIDES = {"pyarrow": "20.0.0"}
TOTALSEG_TOTAL_FAST = 297  # task 'total', fast=True (3 mm) -> vertebra levels


def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.datetime.now().isoformat()} [{PHASE}] {msg}\n")


def write_linux_constraints():
    lines = []
    for line in WINDOWS_CONSTRAINTS.read_text().splitlines():
        name = line.split("==")[0].strip().lower()
        if name in LINUX_OVERRIDES:
            log(f"override {line} -> {name}=={LINUX_OVERRIDES[name]} (no manylinux2014 wheel)")
            line = f"{name}=={LINUX_OVERRIDES[name]}"
        lines.append(line)
    LINUX_CONSTRAINTS.write_text("\n".join(lines) + "\n")


def phase1(slicer):
    em = slicer.app.extensionsManagerModel()
    em.interactive = False
    for name in EXTENSIONS:
        if em.isExtensionInstalled(name):
            log(f"extension {name}: already installed")
            continue
        ok = em.downloadAndInstallExtensionByName(name, True, True)
        log(f"extension {name}: {'installed' if ok else 'FAILED'}")
        if not ok:
            raise RuntimeError(f"could not install extension {name}")
    write_linux_constraints()
    log(f"pip install {TORCH}")
    slicer.util.pip_install(TORCH)
    log(f"pip install {NNUNET}")
    slicer.util.pip_install(NNUNET)


def phase2(slicer):
    import importlib.metadata as md

    import TotalSegmentator  # scripted module of the extension (loaded in this session)

    logic = TotalSegmentator.TotalSegmentatorLogic()
    logic.logCallback = log
    log(f"extension installs TotalSegmentator from {logic.totalSegmentatorPythonPackageDownloadUrl}")
    logic.setupPythonRequirements()

    for pkg in ("torch", "nnunetv2", "TotalSegmentator", "numpy", "requests", "dicom2nifti",
                "pandas", "SimpleITK"):
        log(f"  {pkg}=={md.version(pkg)}")
    log(f"  TotalSegmentator requires: {md.requires('TotalSegmentator')}")
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    Path("/opt/medops-pip-freeze.txt").write_text("\n".join(freeze) + "\n")

    from totalsegmentator.config import set_config_key, setup_totalseg
    from totalsegmentator.libs import download_pretrained_weights
    setup_totalseg()
    set_config_key("send_usage_stats", False)  # no outbound telemetry from the worker
    download_pretrained_weights(TOTALSEG_TOTAL_FAST)
    log(f"TotalSegmentator weights {TOTALSEG_TOTAL_FAST} downloaded")
    OK.write_text("ok\n")


code = 1
try:
    import slicer

    os.environ["PIP_CONSTRAINT"] = str(LINUX_CONSTRAINTS)  # every pip call, incl. the extension's
    log(f"Slicer {slicer.app.applicationVersion} r{slicer.app.revision} python {sys.version.split()[0]}")
    {"phase1": phase1, "phase2": phase2}[PHASE](slicer)
    code = 0
except Exception as exc:
    log("ERROR\n" + traceback.format_exc())
    for stream in ("output", "stderr"):  # pip's own error text (CalledProcessError)
        text = getattr(exc, stream, None)
        if text:
            log(f"{stream}:\n{str(text)[-4000:]}")

log(f"EXIT {code}")
slicer.util.exit(code)
