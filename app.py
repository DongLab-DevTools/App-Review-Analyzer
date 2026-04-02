"""ReviewDong — Flask 대시보드 서버

실행: python app.py
브라우저: http://localhost:5000
"""

import json
import os
import io
import re
from datetime import datetime
from collections import Counter

import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file

from config import APPS, PRIMARY_APP, ANALYSIS_CATEGORIES, CATEGORY_GROUPS

app = Flask(__name__)


# ────────────────────────────────────────
# Data Loading
# ────────────────────────────────────────

def load_all_data():
    app_data = {}
    for key in APPS:
        analyzed_path = f"data/{key}_analyzed.json"
        reviews_path = f"data/{key}_reviews.json"

        data = None
        if os.path.exists(analyzed_path):
            with open(analyzed_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if raw.get("results"):
                    data = raw

        if data is None and os.path.exists(reviews_path):
            with open(reviews_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                results = []
                for r in raw.get("reviews", []):
                    score = r.get("score", 3)
                    results.append({
                        "reviewId": r.get("reviewId", ""),
                        "userName": r.get("userName", ""),
                        "score": score,
                        "content": r.get("content", ""),
                        "thumbsUpCount": r.get("thumbsUpCount", 0),
                        "at": r.get("at", ""),
                        "replyContent": r.get("replyContent", ""),
                        "repliedAt": r.get("repliedAt", ""),
                        "appVersion": r.get("appVersion", ""),
                        "store": r.get("store", "PLAY"),
                        "sentiment": "positive" if score >= 3 else "negative",
                        "category": "",
                        "summary": "",
                    })
                data = {
                    "app_key": key,
                    "app_name": APPS[key]["name"],
                    "total_analyzed": len(results),
                    "results": results,
                }

        if data and data.get("results"):
            app_data[key] = data
    return app_data


def get_collected_at():
    latest = None
    for key in APPS:
        path = f"data/{key}_reviews.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                t = json.load(f).get("collected_at", "")
                if t and (latest is None or t > latest):
                    latest = t
    if latest:
        try:
            return datetime.fromisoformat(latest).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return latest
    return "없음"


def extract_keywords(texts, top_n=20):
    stopwords = {
        "그리고", "하지만", "그래서", "때문에", "이런", "저런", "이것", "저것",
        "하는", "있는", "없는", "되는", "같은", "라고", "에서", "으로", "하고",
        "인데", "한다", "있다", "없다", "된다", "같다", "것이", "수가", "것을",
        "좋겠", "합니다", "입니다", "습니다", "는데", "해서", "어서", "지만",
        "니다", "네요", "아요", "세요", "해요", "할수", "정말", "진짜", "너무",
        "아주", "매우", "좀더", "하면", "이라", "한데", "해도", "이고", "주세요",
        "그냥", "근데", "에요", "인가", "한가", "다른", "이번", "이게", "이런",
    }
    word_count = Counter()
    for text in texts:
        if not text:
            continue
        words = re.findall(r"[가-힣]{2,6}", str(text))
        for w in words:
            if w not in stopwords:
                word_count[w] += 1
    return word_count.most_common(top_n)


# ────────────────────────────────────────
# Prepare dashboard data
# ────────────────────────────────────────

def _load_icon(app_key):
    info_path = f"data/{app_key}_info.json"
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f).get("icon", "")
    return ""


def build_dashboard_data():
    all_data = load_all_data()
    collected_at = get_collected_at()

    # Per-app summaries
    apps_summary = []
    all_reviews_flat = []
    for key in APPS:
        if key not in all_data:
            continue
        d = all_data[key]
        results = d.get("results", [])
        if not results:
            continue

        scores = [r["score"] for r in results if r.get("score")]
        avg = round(sum(scores) / len(scores), 1) if scores else 0
        total = len(results)
        pos = sum(1 for r in results if r.get("sentiment") == "positive")
        neg = total - pos
        pr = round(pos / total * 100) if total else 0

        # Neg categories
        neg_cats = [r["category"] for r in results if r.get("sentiment") == "negative" and r.get("category")]
        top_neg = [c for c, _ in Counter(neg_cats).most_common(3)]

        # Load icon from info file
        icon_url = ""
        info_path = f"data/{key}_info.json"
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                icon_url = json.load(f).get("icon", "")

        apps_summary.append({
            "key": key,
            "name": d["app_name"],
            "is_primary": key == PRIMARY_APP,
            "total": total,
            "avg_score": avg,
            "pos_pct": pr,
            "neg_pct": 100 - pr,
            "top_neg_cats": top_neg,
            "icon": icon_url,
        })

        for r in results:
            r["_app_key"] = key
            r["_app_name"] = d["app_name"]
            all_reviews_flat.append(r)

    # Global stats
    all_scores = [r["score"] for r in all_reviews_flat if r.get("score")]
    global_avg = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0
    global_total = len(all_reviews_flat)
    global_pos = sum(1 for r in all_reviews_flat if r.get("sentiment") == "positive")
    global_pos_pct = round(global_pos / global_total * 100) if global_total else 0

    # Complaint TOP 5 (score <= 2)
    neg_reviews = [r for r in all_reviews_flat if r.get("score", 3) <= 2]
    neg_cats_all = [r["category"] for r in neg_reviews if r.get("category")]
    if neg_cats_all:
        complaints_top5 = [{"name": c, "count": n} for c, n in Counter(neg_cats_all).most_common(5)]
    else:
        neg_texts = [r["content"] for r in neg_reviews if r.get("content")]
        kws = extract_keywords(neg_texts, 5)
        complaints_top5 = [{"name": k, "count": n} for k, n in kws]

    # Notable reviews
    notable = [r for r in all_reviews_flat if r.get("score") == 1 or r.get("thumbsUpCount", 0) >= 5]
    notable.sort(key=lambda x: x.get("at", ""), reverse=True)
    notable = notable[:8]

    return {
        "collected_at": collected_at,
        "global": {
            "total": global_total,
            "avg_score": global_avg,
            "pos_pct": global_pos_pct,
            "app_count": len(apps_summary),
        },
        "apps": apps_summary,
        "complaints_top5": complaints_top5,
        "notable": notable,
        "primary_app": PRIMARY_APP,
        "apps_config": {k: {**v, "icon": _load_icon(k)} for k, v in APPS.items() if k in all_data},
        "category_groups": CATEGORY_GROUPS,
    }


# ────────────────────────────────────────
# Routes
# ────────────────────────────────────────

@app.route("/")
def index():
    data = build_dashboard_data()
    return render_template("dashboard.html", data=data)


@app.route("/api/app/<app_key>")
def api_app_data(app_key):
    """개별 앱 전체 리뷰 데이터 (P3용)"""
    all_data = load_all_data()
    if app_key not in all_data:
        return jsonify({"error": "not found"}), 404

    d = all_data[app_key]
    results = d.get("results", [])

    # Extract keywords
    texts = [r["content"] for r in results if r.get("content")]
    keywords = extract_keywords(texts, 20)

    # Per-version avg scores
    from collections import defaultdict
    ver_scores = defaultdict(list)
    for r in results:
        v = r.get("appVersion", "")
        if v:
            ver_scores[v].append(r.get("score", 0))
    version_avgs = [{"version": v, "avg": round(sum(s) / len(s), 1)} for v, s in ver_scores.items()]
    version_avgs.sort(key=lambda x: x["version"])
    version_avgs = version_avgs[-15:]  # last 15 versions

    # Monthly stats
    monthly = defaultdict(lambda: {"count": 0, "total_score": 0})
    for r in results:
        at = r.get("at", "")
        if at:
            ym = at[:7]  # YYYY-MM
            if ym:
                monthly[ym]["count"] += 1
                monthly[ym]["total_score"] += r.get("score", 0)
    monthly_list = [{"month": m, "count": v["count"], "avg": round(v["total_score"] / v["count"], 1)}
                    for m, v in sorted(monthly.items())]

    # AI insights
    neg_reviews = [r for r in results if r.get("score", 3) <= 2]
    pos_reviews = [r for r in results if r.get("score", 3) >= 4]

    neg_cats = [r["category"] for r in neg_reviews if r.get("category")]
    priority = [{"name": c, "count": n, "pct": round(n / len(neg_reviews) * 100, 1) if neg_reviews else 0}
                for c, n in Counter(neg_cats).most_common(5)] if neg_cats else []
    if not priority:
        neg_kws = extract_keywords([r["content"] for r in neg_reviews if r.get("content")], 5)
        priority = [{"name": f"'{k}' 관련 불만", "count": n, "pct": 0} for k, n in neg_kws]

    pos_cats = [r["category"] for r in pos_reviews if r.get("category")]
    pos_highlights = [{"name": c, "count": n} for c, n in Counter(pos_cats).most_common(3)] if pos_cats else []
    if not pos_highlights:
        pos_kws = extract_keywords([r["content"] for r in pos_reviews if r.get("content")], 3)
        pos_highlights = [{"name": f"'{k}' 관련 긍정", "count": n} for k, n in pos_kws]

    return jsonify({
        "app_key": app_key,
        "app_name": d["app_name"],
        "results": results,
        "keywords": [{"word": k, "count": n} for k, n in keywords],
        "version_avgs": version_avgs,
        "monthly": monthly_list,
        "insights": {
            "priority": priority,
            "positive": pos_highlights,
        },
    })


@app.route("/api/excel/<app_key>")
def api_excel(app_key):
    """필터된 리뷰 엑셀 다운로드"""
    all_data = load_all_data()
    if app_key not in all_data:
        return "not found", 404

    results = all_data[app_key].get("results", [])
    app_name = all_data[app_key]["app_name"]

    # Apply filters from query params
    sentiment = request.args.get("sentiment", "")
    category = request.args.get("category", "")
    score = request.args.get("score", "")
    version = request.args.get("version", "")
    month_start = request.args.get("month_start", "")
    month_end = request.args.get("month_end", "")

    filtered = results
    if sentiment:
        filtered = [r for r in filtered if r.get("sentiment") == sentiment]
    if category:
        filtered = [r for r in filtered if r.get("category") == category]
    if score:
        filtered = [r for r in filtered if r.get("score") == int(score)]
    if version:
        filtered = [r for r in filtered if r.get("appVersion") == version]
    if month_start:
        filtered = [r for r in filtered if r.get("at", "") >= month_start]
    if month_end:
        filtered = [r for r in filtered if r.get("at", "")[:7] <= month_end]

    df = pd.DataFrame(filtered)
    if df.empty:
        df = pd.DataFrame(columns=["작성일", "별점", "스토어", "작성자명", "앱 버전", "리뷰 본문", "개발사 답변"])
    else:
        cols = {
            "at": "작성일", "score": "별점", "store": "스토어",
            "userName": "작성자명", "appVersion": "앱 버전",
            "content": "리뷰 본문", "replyContent": "개발사 답변",
        }
        df = df[[c for c in cols if c in df.columns]].rename(columns=cols)

    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)

    filename = f"{app_name}_리뷰_{datetime.now().strftime('%Y%m%d')}.xlsx"
    return send_file(buf, download_name=filename, as_attachment=True,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/api/text/<app_key>")
def api_text(app_key):
    """필터된 리뷰 텍스트(TSV) 다운로드"""
    all_data = load_all_data()
    if app_key not in all_data:
        return "not found", 404

    results = all_data[app_key].get("results", [])
    app_name = all_data[app_key]["app_name"]

    sentiment = request.args.get("sentiment", "")
    category = request.args.get("category", "")
    score = request.args.get("score", "")
    version = request.args.get("version", "")
    month_start = request.args.get("month_start", "")
    month_end = request.args.get("month_end", "")

    filtered = results
    if sentiment:
        filtered = [r for r in filtered if r.get("sentiment") == sentiment]
    if category:
        filtered = [r for r in filtered if r.get("category") == category]
    if score:
        filtered = [r for r in filtered if r.get("score") == int(score)]
    if version:
        filtered = [r for r in filtered if r.get("appVersion") == version]
    if month_start:
        filtered = [r for r in filtered if r.get("at", "") >= month_start]
    if month_end:
        filtered = [r for r in filtered if r.get("at", "")[:7] <= month_end]

    now = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# 앱 리뷰 데이터 — {app_name}",
        f"# 총 {len(filtered)}건 | 내보낸 날짜: {now}",
        "#",
        "# 형식: [번호] ★별점 | 작성일 | 스토어(PLAY/APPSTORE) | 작성자명 | 앱버전",
        "# 다음 줄: 리뷰 본문",
        "",
    ]
    for i, r in enumerate(filtered, 1):
        content = (r.get("content") or "").replace("\n", " ").strip()
        meta = (f"[{i}] ★{r.get('score','')} | {r.get('at','')[:10]} | "
                f"{r.get('store','')} | {r.get('userName','')} | v{r.get('appVersion','')}")
        lines.append(meta)
        lines.append(content)
        lines.append("")

    text = "\n".join(lines)
    buf = io.BytesIO(text.encode("utf-8-sig"))
    buf.seek(0)

    filename = f"{app_name}_리뷰_{datetime.now().strftime('%Y%m%d')}.txt"
    return send_file(buf, download_name=filename, as_attachment=True, mimetype="text/plain")


@app.route("/api/affinity-text/<app_key>")
def api_affinity_text(app_key):
    """어피니티 분석용 경량 텍스트 다운로드 (번호|별점|리뷰본문)"""
    all_data = load_all_data()
    if app_key not in all_data:
        return "not found", 404

    results = all_data[app_key].get("results", [])
    app_name = all_data[app_key]["app_name"]

    # 최신순 정렬
    results = sorted(results, key=lambda r: r.get("at", ""), reverse=True)

    lines = []
    for i, r in enumerate(results, 1):
        content = (r.get("content") or "").replace("\n", " ").replace("|", "/").strip()
        if not content:
            continue
        score = r.get("score", 0)
        lines.append(f"{i}|★{score}|{content}")

    text = "\n".join(lines)
    buf = io.BytesIO(text.encode("utf-8-sig"))
    buf.seek(0)

    filename = f"{app_name}_어피니티용_{datetime.now().strftime('%Y%m%d')}.txt"
    return send_file(buf, download_name=filename, as_attachment=True, mimetype="text/plain")


# ────────────────────────────────────────
# Affinity Analysis API
# ────────────────────────────────────────

@app.route("/api/affinity/<app_key>")
def api_affinity_get(app_key):
    """저장된 어피니티 분석 결과 조회"""
    path = f"data/{app_key}_affinity.json"
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/affinity/<app_key>/auto")
def api_affinity_auto(app_key):
    """임베딩+클러스터링 자동 어피니티 분석 — SSE 스트리밍 (단계별 캐싱)"""
    if app_key not in APPS:
        return jsonify({"error": "unknown app"}), 400

    user_key = request.args.get("api_key", "").strip()

    import queue
    import threading
    from affinity_analyzer import run_affinity_pipeline

    msg_queue = queue.Queue()

    def on_progress(msg):
        msg_queue.put(msg)

    def run():
        try:
            run_affinity_pipeline(app_key, on_progress=on_progress, user_api_key=user_key or None)
            msg_queue.put("[DONE]")
        except Exception as e:
            msg_queue.put(f"[ERROR] {str(e)}")

    def generate():
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        while True:
            try:
                msg = msg_queue.get(timeout=120)
                yield f"data: {msg}\n\n"
                if msg.startswith("[DONE]") or msg.startswith("[ERROR]"):
                    break
            except queue.Empty:
                yield "data: [ERROR] 시간 초과\n\n"
                break

    return app.response_class(generate(), mimetype="text/event-stream")


@app.route("/api/refresh-stream")
def api_refresh_stream():
    """리뷰 데이터 새로고침 — SSE로 실시간 로그 스트리밍"""
    import subprocess

    def generate():
        proc = subprocess.Popen(
            ["python", "-u", "scraper.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                yield f"data: {line}\n\n"
        proc.wait()
        if proc.returncode == 0:
            yield "data: [DONE]\n\n"
        else:
            yield f"data: [ERROR] 종료 코드: {proc.returncode}\n\n"

    return app.response_class(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = "0.0.0.0"
    finally:
        s.close()
    print("=" * 50)
    print("  ReviewDong Dashboard")
    print(f"  Local:   http://localhost:5001")
    print(f"  Network: http://{local_ip}:5001")
    print("=" * 50)
    app.run(debug=True, port=5001, host="0.0.0.0")
