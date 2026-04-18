"""Homie Daily Analysis — runs between 2 AM - 8 AM IST.

Performs:
1. System health check across all mesh nodes
2. Model performance analysis (response quality, speed)
3. Feedback signal analysis and training readiness
4. If enough feedback: trigger model fine-tuning cycle
5. Generate daily report and store as mesh event
6. Clean up old events (compaction)
"""
import os
import sys
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

IST = timezone(timedelta(hours=5, minutes=30))
LOG_DIR = Path.home() / ".homie" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "daily_analysis.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("homie.daily")


def run_analysis():
    """Run the full daily analysis."""
    start_time = time.time()
    now = datetime.now(IST)
    logger.info("=" * 60)
    logger.info("  HOMIE DAILY ANALYSIS — %s", now.strftime("%Y-%m-%d %H:%M IST"))
    logger.info("=" * 60)

    # 1. Load config and bootstrap
    from homie_core.config import load_config
    cfg = load_config()
    from homie_core.mesh.bootstrap import bootstrap_mesh
    mesh = bootstrap_mesh(cfg)
    logger.info("[1/7] Mesh bootstrapped: node=%s, score=%.0f",
                mesh.node_name, mesh.capabilities.capability_score())

    # 2. Health check
    from homie_core.mesh.health import MeshHealthChecker
    health = MeshHealthChecker(mesh_context=mesh).run_all()
    logger.info("[2/7] Health: %s", health.status)
    for c in health.checks:
        logger.info("  [%s] %s: %s (%.0fms)",
                     "OK" if c.healthy else "FAIL", c.name, c.message, c.latency_ms)

    # 3. System metrics
    import psutil
    import shutil
    import subprocess
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("C:" + os.sep)
    try:
        gpu_r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        gpu_info = gpu_r.stdout.strip()
    except Exception:
        gpu_info = "N/A"

    logger.info("[3/7] System: CPU=%d%%, RAM=%.1f/%.1f GB, Disk=%d GB free, GPU=%s",
                psutil.cpu_percent(), mem.used / (1024 ** 3), mem.total / (1024 ** 3),
                disk.free / (1024 ** 3), gpu_info)

    # 4. Event store analysis
    event_count = mesh.mesh_manager.event_count()
    events = mesh.mesh_manager.events_since(None, limit=1000)
    categories = {}
    for e in events:
        categories[e.category] = categories.get(e.category, 0) + 1
    logger.info("[4/7] Events: %d total — %s", event_count, dict(categories))

    # 5. Feedback & training analysis
    fb_count = mesh.feedback_store.total_count()
    training = mesh.training_trigger.get_summary()
    logger.info("[5/7] Feedback: %d signals, SFT=%d, DPO=%d, Ready=%s",
                fb_count, training["sft_pairs"], training["dpo_pairs"], training["ready"])

    # 6. Auto-train if ready
    trained = False
    if training["ready"] and mesh.auto_trainer:
        logger.info("[6/7] TRAINING TRIGGERED — enough feedback accumulated")
        try:
            result = mesh.auto_trainer.run_cycle(skip_training=False)
            logger.info("  Cycle %d: SFT=%d, DPO=%d, completed=%s",
                         result.cycle, result.sft_pairs, result.dpo_pairs, result.training_completed)
            trained = True
        except Exception as e:
            logger.error("  Training failed: %s", e)
    else:
        logger.info("[6/7] Training not needed (need %d more signals)",
                     max(0, 500 - fb_count))

    # 7. Generate report and store as event
    elapsed = time.time() - start_time
    report = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M IST"),
        "node": mesh.node_name,
        "health": health.status,
        "health_checks": {c.name: c.healthy for c in health.checks},
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": mem.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3)),
            "gpu": gpu_info,
        },
        "events_total": event_count,
        "events_by_category": categories,
        "feedback_total": fb_count,
        "training_ready": training["ready"],
        "training_triggered": trained,
        "analysis_duration_sec": round(elapsed, 1),
    }

    mesh.mesh_manager.emit("system", "daily_report", report)
    logger.info("[7/7] Daily report stored as mesh event")

    # Save report to file
    report_dir = Path.home() / ".homie" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / f"daily_{now.strftime('%Y-%m-%d')}.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("  Report saved: %s", report_file)

    logger.info("\n  Analysis complete in %.1fs", elapsed)
    logger.info("=" * 60)
    return report


def is_analysis_window():
    """Check if current time is between 2 AM - 8 AM IST."""
    now = datetime.now(IST)
    return 2 <= now.hour < 8


def run_scheduler():
    """Run as a daemon — checks every 30 min, runs analysis once per day in the window."""
    logger.info("Homie Daily Analysis Scheduler started")
    logger.info("Analysis window: 2:00 AM - 8:00 AM IST")
    last_run_date = None

    while True:
        now = datetime.now(IST)
        today = now.strftime("%Y-%m-%d")

        if is_analysis_window() and last_run_date != today:
            logger.info("In analysis window, running daily analysis...")
            try:
                run_analysis()
                last_run_date = today
            except Exception as e:
                logger.error("Daily analysis failed: %s", e)

        # Sleep 30 minutes
        time.sleep(1800)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Homie Daily Analysis")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as scheduler daemon")
    args = parser.parse_args()

    if args.daemon:
        run_scheduler()
    else:
        run_analysis()
