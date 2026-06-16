"""일일 자동화 잡: 6개 OTT 리뷰 수집 → 어피니티 재분류 → 실행 이력 기록.

- 서버와 독립 실행 가능: `python daily_job.py` (cron 등에서 사용)
- Flask 앱에서 import 해 인앱 스케줄러로도 사용: start_scheduler(hour=4)
- 잠금 파일로 중복/동시 실행 방지 (멀티워커·수동+자동 겹침 대비)
- 재분류는 시그니처 캐시에 위임: 신규 리뷰가 없으면 어피니티가 캐시-히트로 즉시 종료
"""

import os
import json
import time
import fcntl
import threading
import traceback
from datetime import datetime, timedelta

from config import APPS
from scraper import collect_app
from affinity_analyzer import run_affinity_pipeline

HISTORY_PATH = "data/run_history.json"
LOCK_PATH = "data/.daily_job.lock"
HISTORY_KEEP = 60
DEFAULT_HOUR = int(os.getenv("SCHEDULER_HOUR", "4"))


# ────────────────────────────────────────
# 잠금 (동시 실행 방지)
# ────────────────────────────────────────

class _JobLock:
    """fcntl 파일 락 (macOS/Linux 공통). 이미 실행 중이면 RuntimeError."""
    def __enter__(self):
        os.makedirs("data", exist_ok=True)
        self._f = open(LOCK_PATH, "w")
        try:
            fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._f.close()
            raise RuntimeError("이미 실행 중인 일일 잡이 있습니다")
        return self

    def __exit__(self, *exc):
        try:
            fcntl.flock(self._f, fcntl.LOCK_UN)
        finally:
            self._f.close()


# ────────────────────────────────────────
# 실행 이력
# ────────────────────────────────────────

def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_history(records):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(records[:HISTORY_KEEP], f, ensure_ascii=False, indent=2)


def _append_history(record):
    hist = load_history()
    hist.insert(0, record)  # 최신이 앞
    _save_history(hist)


# ────────────────────────────────────────
# 본체
# ────────────────────────────────────────

def run_daily(trigger="manual", on_progress=None, app_keys=None):
    """전체 앱 수집 + 재분류 + 이력 기록. 실행 레코드(dict) 반환."""
    def log(msg):
        line = f"[daily] {msg}"
        if on_progress:
            on_progress(msg)
        print(line, flush=True)

    keys = app_keys or list(APPS.keys())
    started = datetime.now()
    record = {
        "started_at": started.isoformat(),
        "finished_at": None,
        "duration_sec": None,
        "trigger": trigger,
        "status": "running",
        "apps": [],
        "totals": {"new_reviews": 0, "apps_ok": 0, "apps_failed": 0, "incomplete": 0, "local_llm": 0},
        "quota_limited": False,
    }

    try:
        with _JobLock():
            log(f"일일 잡 시작 (trigger={trigger}, {len(keys)}개 앱)")
            for key in keys:
                name = APPS[key]["name"]
                entry = {
                    "key": key, "name": name,
                    "collect": {"new": 0, "total": 0, "ok": False, "error": None},
                    "affinity": {"ran": False, "categories": None, "ok": False,
                                 "complete": True, "published": True, "backend": "gemini",
                                 "note": None, "error": None},
                }
                # 1) 수집
                try:
                    log(f"[{name}] 리뷰 수집...")
                    c = collect_app(key, append=True, on_progress=lambda m: None)
                    entry["collect"].update(new=c["new"], total=c["total"], ok=True)
                    record["totals"]["new_reviews"] += c["new"]
                    log(f"[{name}] 신규 {c['new']}건 / 총 {c['total']}건")
                except Exception as e:
                    entry["collect"]["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                    log(f"[{name}] 수집 실패: {entry['collect']['error']}")

                # 2) 재분류 (시그니처 캐시에 위임 — 신규 없으면 캐시-히트로 즉시 종료)
                try:
                    log(f"[{name}] 어피니티 분석...")
                    result = run_affinity_pipeline(key, on_progress=lambda m: None)
                    meta = result.get("meta", {})
                    full = meta.get("fully_complete", True)
                    published = meta.get("published", True)
                    note = None
                    if not published:
                        parts = []
                        if not meta.get("classify_complete", True): parts.append("분류 미완료")
                        if meta.get("labels_fallback"): parts.append("라벨 폴백")
                        note = "⚠ 대시보드 미반영 — " + ", ".join(parts) + " (기존 결과 유지, 다음 실행 시 재개)"
                    elif not full:
                        note = "반영됨 · 병합은 다음 실행 시 정리"
                    backend = meta.get("llm_backend", "gemini")
                    if backend == "ollama" and note is None:
                        note = "로컬 모델(Ollama)로 분석 — 정확도는 참고용"
                    entry["affinity"].update(
                        ran=entry["collect"]["new"] > 0,
                        categories=len(result.get("categories", [])),
                        ok=True, complete=full, published=published, backend=backend, note=note,
                    )
                    if not published:
                        record["totals"]["incomplete"] += 1
                    if backend == "ollama":
                        record["totals"]["local_llm"] = record["totals"].get("local_llm", 0) + 1
                    log(f"[{name}] 분류 {'완료' if full else ('미반영' if not published else '반영(병합 보류)')} "
                        f"({entry['affinity']['categories']}개 대분류)")
                except Exception as e:
                    entry["affinity"]["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                    log(f"[{name}] 분류 실패: {entry['affinity']['error']}")

                app_ok = entry["collect"]["ok"] and entry["affinity"]["ok"]
                record["totals"]["apps_ok" if app_ok else "apps_failed"] += 1
                record["apps"].append(entry)

            failed = record["totals"]["apps_failed"]
            record["status"] = "success" if failed == 0 else ("error" if failed == len(keys) else "partial")
            # 하드 실패는 없지만 한도로 부분 완료된 앱이 있으면 'quota_limited' 플래그
            record["quota_limited"] = record["totals"]["incomplete"] > 0
    except RuntimeError as e:
        # 잠금 실패 — 이력에 남기지 않고 그대로 반환
        record["status"] = "skipped"
        record["error"] = str(e)
        log(str(e))
        return record
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        log(f"치명적 오류: {record['error']}\n{traceback.format_exc()}")

    finished = datetime.now()
    record["finished_at"] = finished.isoformat()
    record["duration_sec"] = round((finished - started).total_seconds(), 1)
    _append_history(record)
    log(f"일일 잡 종료: status={record['status']}, {record['duration_sec']}초, "
        f"신규 {record['totals']['new_reviews']}건")
    return record


# ────────────────────────────────────────
# 인앱 스케줄러 (OS 비종속, 추가 의존성 없음)
# ────────────────────────────────────────

_scheduler_started = False
_scheduler_lock = threading.Lock()


def _seconds_until(hour):
    now = datetime.now()
    target = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds(), target


def next_run_time(hour=DEFAULT_HOUR):
    """다음 예정 실행 시각(ISO). 상태 페이지 표시용."""
    _, target = _seconds_until(hour)
    return target.isoformat()


def start_scheduler(hour=DEFAULT_HOUR):
    """매일 hour시에 run_daily를 실행하는 데몬 스레드 시작. 중복 시작 방지."""
    global _scheduler_started
    with _scheduler_lock:
        if _scheduler_started:
            return False
        _scheduler_started = True

    def loop():
        while True:
            secs, target = _seconds_until(hour)
            print(f"[daily] 다음 자동 실행: {target.isoformat()} ({int(secs)}초 후)", flush=True)
            time.sleep(secs)
            try:
                run_daily(trigger="schedule")
            except Exception as e:
                print(f"[daily] 스케줄 실행 오류: {e}", flush=True)
            time.sleep(61)  # 같은 분 내 재실행 방지

    threading.Thread(target=loop, daemon=True, name="daily-scheduler").start()
    print(f"[daily] 스케줄러 시작 — 매일 {hour:02d}:00 실행", flush=True)
    return True


if __name__ == "__main__":
    import sys
    trig = "manual"
    if "--schedule" in sys.argv:
        # 포그라운드 스케줄러 (cron 대신 직접 띄울 때)
        start_scheduler()
        while True:
            time.sleep(3600)
    run_daily(trigger=trig)
