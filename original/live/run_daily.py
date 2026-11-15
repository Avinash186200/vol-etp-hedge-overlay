"""
run_daily.py — Company P8020LV daily orchestrator.
Runs: data_feed -> live_daily -> drive_upload.
Task Scheduler invokes `python run_daily.py` from this folder.

Schedule: Tue-Sat 00:45 IST
  IST Tue 00:45 = ET Mon 15:15 = signal 45 min before Mon US close → for Mon's session
  IST Wed 00:45 = ET Tue 15:15 = for Tue's session
  IST Thu 00:45 = ET Wed 15:15 = for Wed's session
  IST Fri 00:45 = ET Thu 15:15 = for Thu's session
  IST Sat 00:45 = ET Fri 15:15 = for Fri's session

Exit codes:
  0  signal produced (may be STALE-flagged; Drive may or may not have uploaded)
  2  data feed stale AND signal could not be produced
  3  signal step itself crashed (investigate output/last_run.log)
Drive failures never gate exit — local signal is always retained.
"""
from __future__ import annotations
import sys, os, subprocess, datetime as dt, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT  = HERE / "output"; OUT.mkdir(exist_ok=True)
LOG  = OUT / "last_run.log"

# force UTF-8 everywhere so logs with – / … never crash on cp1252
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def _stamp():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _say(msg: str, mode: str = "a"):
    line = f"[{_stamp()}] {msg}"
    with open(LOG, mode, encoding="utf-8") as f:
        f.write(line + "\n")
    try: print(line, flush=True)
    except Exception: pass

def _run(script: str, timeout_sec: int = 1800) -> int:
    """Run a sibling .py with this same interpreter, append stdout/stderr to LOG."""
    py = sys.executable
    _say(f"--- {script} START ---")
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            p = subprocess.run(
                [py, "-X", "utf8=1", str(HERE / script)],
                cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
                timeout=timeout_sec,
            )
        _say(f"--- {script} EXIT {p.returncode} ---")
        return p.returncode
    except subprocess.TimeoutExpired:
        _say(f"--- {script} TIMEOUT after {timeout_sec//60} min ---")
        return 124
    except Exception as e:
        _say(f"--- {script} CRASHED: {e!r} ---")
        _say(traceback.format_exc())
        return 99

def main() -> int:
    _say("=== P8020LV daily run start ===", mode="w")
    _say(f"python = {sys.executable}")
    _say(f"cwd    = {HERE}")
    weekday = dt.datetime.now().weekday()  # Mon=0
    _say(f"weekday = {weekday} (0=Mon, 4=Fri, 5=Sat)")

    # is today a weekly-refit day? Use Tuesday (1 in IST) which is Monday in ET.
    # Refit early so daily signal has fresh model.
    is_refit_day = weekday == 1  # IST Tuesday
    if is_refit_day:
        _say("--- weekly refit day — running refit_weekly before signal ---")
        _run("refit_weekly.py", timeout_sec=10800)  # up to 3 hr

    feed = _run("data_feed.py")     # 0=fresh, 2=stale-but-ran, else crash
    sig  = _run("live_daily.py")    # 0=signal written, else failure
    drv  = _run("drive_upload.py")  # 0=delivered, 3=local-only, else failure

    _say(f"=== summary feed={feed} signal={sig} drive={drv} ===")

    if sig != 0:
        return 3      # signal step failed -> alert via Task Scheduler LastResult
    if feed == 2:
        return 0      # signal produced, flagged STALE inside signal.txt
    return 0

if __name__ == "__main__":
    sys.exit(main())
