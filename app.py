"""ReviewDong — Flask 대시보드 서버

실행: python app.py
브라우저: http://localhost:5000
"""

import json
import os
import io
import re
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict

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
# TVING-centric dashboard (메인)
# ────────────────────────────────────────

TVING_KEY = "tving"


def _read_scores(key):
    """리뷰 파일에서 (별점, 작성일YYYY-MM-DD) 목록만 가볍게 읽는다."""
    path = f"data/{key}_reviews.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for r in raw.get("reviews", []):
        score = r.get("score")
        if not score:
            continue
        out.append((score, (r.get("at", "") or "")[:10]))
    return out


def _monthly_avgs(scores):
    m = defaultdict(list)
    for sc, at in scores:
        if len(at) >= 7:
            m[at[:7]].append(sc)
    return {mo: sum(v) / len(v) for mo, v in m.items()}


def _trend_of(recent, prior):
    """최근/이전 건수 비교 → up/down/flat."""
    if prior == 0:
        return "up" if recent > 0 else "flat"
    ratio = recent / prior
    if ratio >= 1.1:
        return "up"
    if ratio <= 0.9:
        return "down"
    return "flat"


def _affinity_staleness(app_key):
    """게시된 affinity.json '자체'를 기준으로 staleness 판정.
    (캐시가 아니라 실제 화면에 뜨는 파일 기준 — 캐시는 게시본과 어긋날 수 있음)
    반환: (analyzed, stale, cur_count, aff_count, basis)"""
    apath = f"data/{app_key}_affinity.json"
    cur_sig, cur_count = _current_review_sig(app_key)
    if not os.path.exists(apath):
        return False, False, cur_count, None, None
    aff_sig = aff_count = None
    try:
        with open(apath, "r", encoding="utf-8") as f:
            meta = json.load(f).get("meta", {})
        aff_sig = meta.get("review_sig")
        aff_count = meta.get("total_reviews")
    except Exception:
        pass
    if aff_sig is not None:
        # 게시본에 분석 당시 리뷰 시그니처가 있으면 정확 비교
        stale = bool(cur_sig and aff_sig != cur_sig)
    else:
        # 구버전(시그니처 없음) → 리뷰 건수 비교로 추정
        stale = bool(cur_count and aff_count is not None and aff_count != cur_count)
    items = _affinity_review_items(app_key) or []
    basis = max((it["date"] for it in items), default=None)
    return True, stale, cur_count, aff_count, basis


def _affinity_stale(app_key):
    """어피니티 분류가 현재 리뷰 대비 갱신이 필요한지 + 분석 기준일."""
    _, stale, _, _, basis = _affinity_staleness(app_key)
    return stale, basis


def _affinity_complaints(key, top_n=5):
    """앱의 어피니티 군집에서 부정 카테고리 TOP N을 (label, count, pct)로 반환. 없으면 None."""
    path = f"data/{key}_affinity.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        aff = json.load(f)
    cats = []
    for c in aff.get("categories", []):
        total = neg = 0
        for s in c.get("subcategories", []):
            is_neg = s.get("sentiment") == "negative"
            for it in s.get("items", []):
                total += 1
                if is_neg or it.get("rating", 3) <= 2:
                    neg += 1
        if total and neg / total >= 0.5:
            cats.append({"label": c.get("label", ""), "count": neg})
    cats.sort(key=lambda x: x["count"], reverse=True)
    total_neg = sum(c["count"] for c in cats) or 1
    for c in cats:
        c["pct"] = round(c["count"] / total_neg * 100, 1)
    return cats[:top_n]


def build_tving_dashboard():
    """티빙을 주인공으로, 나머지 OTT를 벤치마크로 두는 메인 대시보드 데이터."""
    ott_keys = list(APPS.keys())
    app_scores = {k: _read_scores(k) for k in ott_keys}

    def avg(scores):
        s = [x[0] for x in scores]
        return round(sum(s) / len(s), 2) if s else 0

    # ── ② 경쟁 포지셔닝 ──
    positioning = []
    for k in ott_keys:
        positioning.append({
            "key": k,
            "name": APPS[k]["name"],
            "avg_score": avg(app_scores[k]),
            "total": len(app_scores[k]),
            "is_tving": k == TVING_KEY,
        })
    positioning.sort(key=lambda x: x["avg_score"], reverse=True)

    tving_avg = next((p["avg_score"] for p in positioning if p["key"] == TVING_KEY), 0)
    rank = next((i + 1 for i, p in enumerate(positioning) if p["key"] == TVING_KEY), 0)
    comp_scores = [p["avg_score"] for p in positioning if p["key"] != TVING_KEY and p["total"] > 0]
    comp_avg = round(sum(comp_scores) / len(comp_scores), 2) if comp_scores else 0

    # ── ⑤ 평점 추이 (월별, 경쟁사 평균 라인 겹침) ──
    tving_monthly = _monthly_avgs(app_scores[TVING_KEY])
    comp_month = defaultdict(list)
    for k in ott_keys:
        if k == TVING_KEY:
            continue
        for mo, a in _monthly_avgs(app_scores[k]).items():
            comp_month[mo].append(a)
    all_months = sorted(set(list(tving_monthly) + list(comp_month)))[-12:]
    rating_trend = {
        "months": all_months,
        "tving": [round(tving_monthly[m], 2) if m in tving_monthly else None for m in all_months],
        "competitor_avg": [round(sum(comp_month[m]) / len(comp_month[m]), 2) if comp_month.get(m) else None
                           for m in all_months],
    }

    # ── ① 히어로 ──
    tving_dates = [at for _, at in app_scores[TVING_KEY] if at]
    latest = max(tving_dates) if tving_dates else ""
    recent_30d = 0
    if latest:
        try:
            cutoff = (datetime.fromisoformat(latest) - timedelta(days=30)).date().isoformat()
            recent_30d = sum(1 for _, at in app_scores[TVING_KEY] if at and at >= cutoff)
        except ValueError:
            pass
    tmonths = sorted(tving_monthly)
    trend_delta = round(tving_monthly[tmonths[-1]] - tving_monthly[tmonths[-2]], 2) if len(tmonths) >= 2 else 0

    hero = {
        "avg_score": tving_avg,
        "rank": rank,
        "total_apps": len([p for p in positioning if p["total"] > 0]),
        "competitor_avg": comp_avg,
        "gap": round(tving_avg - comp_avg, 2),
        "recent_30d": recent_30d,
        "trend_delta": trend_delta,
    }

    # ── ③ 불만 카테고리 정형화 + ④ 개선 과제 (어피니티 군집 활용) ──
    complaints, tasks = [], []
    apath = f"data/{TVING_KEY}_affinity.json"
    if os.path.exists(apath):
        with open(apath, "r", encoding="utf-8") as f:
            aff = json.load(f)

        cat_data, item_dates = [], []
        for c in aff.get("categories", []):
            total_cnt, neg_cnt, dates = 0, 0, []
            for s in c.get("subcategories", []):
                is_neg = s.get("sentiment") == "negative"
                for it in s.get("items", []):
                    total_cnt += 1
                    d = (it.get("date", "") or "")[:10]
                    if d:
                        dates.append(d)
                        item_dates.append(d)
                    if is_neg or it.get("rating", 3) <= 2:
                        neg_cnt += 1
            cat_data.append({
                "label": c.get("label", ""),
                "desc": c.get("description", ""),
                "total": total_cnt,
                "neg": neg_cnt,
                "dates": dates,
            })

        # 부정 비중이 절반 이상인 카테고리만 "불만"으로 본다
        complaint_cats = [c for c in cat_data if c["total"] and c["neg"] / c["total"] >= 0.5]
        complaint_cats.sort(key=lambda c: c["neg"], reverse=True)
        total_neg = sum(c["neg"] for c in complaint_cats) or 1

        # 추세 계산 윈도우 (최근 90일 vs 직전 90일)
        latest_i = max(item_dates) if item_dates else ""
        cut1 = cut2 = None
        if latest_i:
            try:
                base = datetime.fromisoformat(latest_i)
                cut1 = (base - timedelta(days=90)).date().isoformat()
                cut2 = (base - timedelta(days=180)).date().isoformat()
            except ValueError:
                pass

        for c in complaint_cats:
            recent = prior = 0
            if cut1 and cut2:
                recent = sum(1 for d in c["dates"] if d >= cut1)
                prior = sum(1 for d in c["dates"] if cut2 <= d < cut1)
            complaints.append({
                "label": c["label"],
                "count": c["neg"],
                "pct": round(c["neg"] / total_neg * 100, 1),
                "trend": _trend_of(recent, prior),
                "trend_delta": recent - prior,
            })

        severities = ["심각", "높음", "높음", "중간", "중간"]
        for i, c in enumerate(complaints[:5]):
            tasks.append({
                "rank": i + 1,
                "label": c["label"],
                "count": c["count"],
                "pct": c["pct"],
                "trend": c["trend"],
                "trend_delta": c["trend_delta"],
                "severity": severities[i] if i < len(severities) else "중간",
            })

    # ── P2 비교(벤치마크) 데이터 ──
    def pos_pct(scores):
        if not scores:
            return 0
        return round(sum(1 for s, _ in scores if s >= 3) / len(scores) * 100)

    def star_pct(scores):
        if not scores:
            return [0, 0, 0, 0, 0]
        n = len(scores)
        return [round(sum(1 for s, _ in scores if s == star) / n * 100) for star in range(1, 6)]

    compare_apps = []
    for i, p in enumerate(positioning):
        sc = app_scores[p["key"]]
        compare_apps.append({
            "rank": i + 1,
            "key": p["key"],
            "name": p["name"],
            "total": p["total"],
            "avg_score": p["avg_score"],
            "pos_pct": pos_pct(sc),
            "is_tving": p["is_tving"],
            "vs_tving": round(p["avg_score"] - tving_avg, 2),
            "star_pct": star_pct(sc),
        })

    complaint_compare = []
    for k in ott_keys:
        top = _affinity_complaints(k, top_n=5)
        if top:
            complaint_compare.append({"key": k, "name": APPS[k]["name"],
                                      "is_tving": k == TVING_KEY, "top": top})

    compare = {
        "apps": compare_apps,
        "competitor_avg": comp_avg,
        "tving_avg": tving_avg,
        "complaint_compare": complaint_compare,
    }

    cat_stale, cat_basis = _affinity_stale(TVING_KEY)

    return {
        "hero": hero,
        "positioning": positioning,
        "rating_trend": rating_trend,
        "complaints": complaints,
        "tasks": tasks,
        "compare": compare,
        "category_stale": cat_stale,
        "category_basis": cat_basis,
    }


# ────────────────────────────────────────
# 데일리 리포트 (어피니티 군집 + 원본 리뷰 기반)
# ────────────────────────────────────────

def _affinity_review_items(app_key):
    """어피니티 군집을 (카테고리·감성·별점·날짜·텍스트) 단위 리스트로 평탄화."""
    path = f"data/{app_key}_affinity.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        aff = json.load(f)
    items = []
    for c in aff.get("categories", []):
        for s in c.get("subcategories", []):
            is_neg = s.get("sentiment") == "negative"
            for it in s.get("items", []):
                d = (it.get("date", "") or "")[:10]
                if not d:
                    continue
                items.append({
                    "category": c.get("label", ""),
                    "sentiment": s.get("sentiment", "neutral"),
                    "is_neg": is_neg,
                    "rating": it.get("rating", 0),
                    "date": d,
                    "text": it.get("text", ""),
                })
    return items


def _trend_arrow(recent, prior):
    if prior == 0:
        return ("up", recent) if recent > 0 else ("flat", 0)
    r = recent / prior
    if r >= 1.15:
        return "up", recent - prior
    if r <= 0.85:
        return "down", recent - prior
    return "flat", recent - prior


def _read_reviews(app_key):
    """원본 리뷰를 (score, date, store, content, thumbs) 리스트로 읽는다."""
    path = f"data/{app_key}_reviews.json"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for r in raw.get("reviews", []):
        d = (r.get("at", "") or "")[:10]
        if not d:
            continue
        out.append({
            "score": r.get("score", 0), "date": d,
            "store": r.get("store", "PLAY"),
            "content": r.get("content", ""), "thumbs": r.get("thumbsUpCount", 0),
            "version": r.get("appVersion", "") or "",
        })
    return out


def _window(items, anchor_dt, n_start, n_end=0):
    hi = (anchor_dt - timedelta(days=n_end)).date().isoformat()
    lo = (anchor_dt - timedelta(days=n_start - 1)).date().isoformat()
    return [x for x in items if lo <= x["date"] <= hi]


_COMP_RE = re.compile(r"넷플릭스|넷플|netflix|웨이브|wavve|디즈니|disney|왓챠|watcha|쿠팡|coupang|유튜브|youtube|라프텔|laftel", re.I)
_BUG_RE = re.compile(r"오류|에러|버그|튕|크래시|먹통|안돼|안됨|안되|로딩|멈춤|강제종료|실행|꺼짐|블랙", re.I)
_PERIOD_LABEL = {1: "어제(최신일)", 7: "최근 7일", 30: "최근 30일"}


def _app_recent_metrics(key, period):
    rv = _read_reviews(key)
    if not rv:
        return None
    aD = datetime.fromisoformat(max(x["date"] for x in rv))
    w = _window(rv, aD, period)
    if not w:
        return None
    rs = [x["score"] for x in w if x.get("score")]
    return {"avg": round(sum(rs) / len(rs), 2) if rs else 0,
            "neg": round(sum(1 for x in w if x["score"] and x["score"] <= 2) / len(w) * 100)}


def build_daily_report(app_key, period=7):
    """데일리 액션 브리핑: 변화·신규/급증·우선순위·배포영향·경쟁사·하이라이트 중심."""
    reviews = _read_reviews(app_key)
    if not reviews:
        return {"available": False, "app_key": app_key, "app_name": APPS.get(app_key, {}).get("name", app_key)}
    period = period if period in (1, 7, 30) else 7
    r_anchor = max(x["date"] for x in reviews)
    rD = datetime.fromisoformat(r_anchor)

    def avg_rating(xs):
        rs = [x["score"] for x in xs if x.get("score")]
        return round(sum(rs) / len(rs), 2) if rs else 0

    def neg_pct(xs):
        return round(sum(1 for x in xs if x.get("score") and x["score"] <= 2) / len(xs) * 100) if xs else 0

    cur, prev = _window(reviews, rD, period), _window(reviews, rD, period * 2, period)
    header = {
        "anchor": r_anchor, "period": period, "period_label": _PERIOD_LABEL[period],
        "count": len(cur), "count_delta": len(cur) - len(prev),
        "avg": avg_rating(cur), "avg_prev": avg_rating(prev),
        "neg": neg_pct(cur), "neg_prev": neg_pct(prev),
    }

    # content→version 맵 (카테고리 이슈에 버전 연결, best-effort 텍스트 매칭)
    def _norm(s):
        return re.sub(r"\s+", "", (s or ""))[:40]
    ver_by_content = {}
    for x in reviews:
        if x["version"]:
            ver_by_content.setdefault(_norm(x["content"]), x["version"])

    # ── 카테고리(어피니티): B 신규·급증, C 우선순위 ──
    items = _affinity_review_items(app_key) or []
    cat_stale, cat_basis = _affinity_stale(app_key)
    issues, priorities, top_categories = [], [], []
    if items:
        iD = datetime.fromisoformat(max(x["date"] for x in items))
        a_cur = _window(items, iD, period)
        a_prev = _window(items, iD, period * 2, period)
        before_start = (iD - timedelta(days=period * 2 - 1)).date().isoformat()
        a_before = [x for x in items if x["date"] < before_start]
        cc, cp, cb = (Counter(x["category"] for x in a_cur),
                      Counter(x["category"] for x in a_prev),
                      Counter(x["category"] for x in a_before))

        def rep(cat, xs, neg_only=False):
            cands = [x for x in xs if x["category"] == cat and x["text"] and (not neg_only or x["is_neg"])]
            cands.sort(key=lambda x: len(x["text"]), reverse=True)
            return cands[0]["text"][:140] if cands else ""

        def link_ver(cat, xs):
            vers = [ver_by_content.get(_norm(x["text"])) for x in xs if x["category"] == cat]
            vers = [v for v in vers if v]
            return Counter(vers).most_common(1)[0][0] if vers else ""

        def cat_items(cat, xs):
            # 해당 카테고리의 전체 리뷰 — 최신순(최근 리뷰가 위로), 최대 80건
            cands = [{"text": x["text"], "rating": x["rating"],
                      "date": x["date"], "version": ver_by_content.get(_norm(x["text"]), "")}
                     for x in xs if x["category"] == cat]
            cands.sort(key=lambda x: x["date"], reverse=True)
            return cands[:80]

        for cat, n in cc.most_common():
            if n < 4:
                continue
            p, b = cp.get(cat, 0), cb.get(cat, 0)
            if p <= 1 and b <= 2:
                issues.append({"kind": "new", "label": cat, "count": n, "prev": p,
                               "rep": rep(cat, a_cur), "version": link_ver(cat, a_cur),
                               "items": cat_items(cat, a_cur)})
            elif p > 0 and n / p >= 1.8 and n - p >= 3:
                issues.append({"kind": "surge", "label": cat, "count": n, "prev": p, "ratio": round(n / p, 1),
                               "rep": rep(cat, a_cur), "version": link_ver(cat, a_cur),
                               "items": cat_items(cat, a_cur)})
        issues.sort(key=lambda x: (x["kind"] != "new", -x["count"]))
        issues = issues[:6]

        for cat, n in cc.most_common(6):
            tr, dl = _trend_arrow(n, cp.get(cat, 0))
            top_categories.append({"label": cat, "count": n, "trend": tr, "delta": dl})

        neg_cur = [x for x in a_cur if x["is_neg"]]
        ncc = Counter(x["category"] for x in neg_cur)
        ncp = Counter(x["category"] for x in a_prev if x["is_neg"])
        sev = ["심각", "높음", "높음", "중간", "중간"]
        for i, (cat, n) in enumerate(ncc.most_common(5)):
            tr, dl = _trend_arrow(n, ncp.get(cat, 0))
            priorities.append({"rank": i + 1, "label": cat, "count": n, "trend": tr, "delta": dl,
                               "severity": sev[i] if i < len(sev) else "중간",
                               "rep": rep(cat, neg_cur, neg_only=True), "version": link_ver(cat, neg_cur),
                               "items": cat_items(cat, neg_cur)})

    # ── D. 배포 영향 (버전별 평점, 직전 버전 대비 하락 경고) ──
    ver_stats = {}
    for x in reviews:
        v = x["version"]
        if not v:
            continue
        s = ver_stats.setdefault(v, {"n": 0, "sum": 0, "first": x["date"]})
        s["n"] += 1
        s["sum"] += x["score"]
        if x["date"] < s["first"]:
            s["first"] = x["date"]
    vlist = sorted(
        [{"version": v, "n": s["n"], "avg": round(s["sum"] / s["n"], 2), "first": s["first"]}
         for v, s in ver_stats.items() if s["n"] >= 5],
        key=lambda x: x["first"])
    deploy = []
    for i, v in enumerate(vlist):
        drop = round(v["avg"] - vlist[i - 1]["avg"], 2) if i > 0 else None
        deploy.append({**v, "drop": drop, "warn": bool(drop is not None and drop <= -0.3)})
    deploy = sorted(deploy, key=lambda x: x["first"], reverse=True)[:8]

    # ── E. 경쟁사 대비 (같은 기간 OTT 평균) ──
    comp = [m for m in (_app_recent_metrics(k, period) for k in APPS if k != app_key) if m]
    versus = None
    if comp:
        versus = {"avg": header["avg"], "comp_avg": round(sum(m["avg"] for m in comp) / len(comp), 2),
                  "neg": header["neg"], "comp_neg": round(sum(m["neg"] for m in comp) / len(comp))}

    # ── F. 꼭 볼 리뷰 (하이라이트, 최소 30일 풀) ──
    scope = _window(reviews, rD, max(period, 30))

    def rv_brief(x):
        return {"score": x["score"], "date": x["date"], "thumbs": x["thumbs"],
                "version": x["version"], "content": (x["content"] or "")[:160]}
    highlights = {
        "neg": [rv_brief(x) for x in sorted([x for x in scope if x["score"] <= 2],
                                            key=lambda x: x["thumbs"], reverse=True)[:3]],
        "comp": [rv_brief(x) for x in sorted([x for x in scope if _COMP_RE.search(x["content"] or "")],
                                             key=lambda x: x["thumbs"], reverse=True)[:3]],
        "bug": [rv_brief(x) for x in sorted([x for x in scope if x["score"] <= 2 and x["version"] and _BUG_RE.search(x["content"] or "")],
                                            key=lambda x: x["thumbs"], reverse=True)[:3]],
    }

    # ── 추이 / 플랫폼 ──
    trend14 = []
    for k in range(13, -1, -1):
        d = (rD - timedelta(days=k)).date().isoformat()
        xs = [x for x in reviews if x["date"] == d]
        trend14.append({"date": d, "count": len(xs), "avg": avg_rating(xs)})
    play = sum(1 for x in reviews if x["store"] != "APPLE")
    apple = sum(1 for x in reviews if x["store"] == "APPLE")
    platform = {"play": play, "apple": apple, "ios_available": apple > 0}

    # ── TL;DR ──
    parts = [f"{_PERIOD_LABEL[period]} {header['count']}건·평점 {header['avg']}"]
    if header["avg_prev"]:
        dd = round(header["avg"] - header["avg_prev"], 2)
        if abs(dd) >= 0.1:
            parts.append(f"평점 {'▲' if dd > 0 else '▼'}{abs(dd)}")
    new_i = [i for i in issues if i["kind"] == "new"]
    if new_i:
        parts.append(f"신규 '{new_i[0]['label']}'")
    elif issues:
        parts.append(f"'{issues[0]['label']}' 급증")
    warn_v = [d for d in deploy if d["warn"]]
    if warn_v:
        parts.append(f"v{warn_v[0]['version']} 배포 후 평점 하락")
    comment = " · ".join(parts)

    return {
        "available": True, "app_key": app_key, "app_name": APPS.get(app_key, {}).get("name", app_key),
        "period": period, "header": header, "comment": comment,
        "category_basis": cat_basis, "category_stale": cat_stale,
        "issues": issues, "priorities": priorities, "top_categories": top_categories,
        "deploy": deploy, "versus": versus, "highlights": highlights,
        "platform": platform, "trend14": trend14,
    }


# ────────────────────────────────────────
# Routes
# ────────────────────────────────────────

@app.route("/")
def index():
    data = build_dashboard_data()
    tving = build_tving_dashboard()
    return render_template("dashboard.html", data=data, tving=tving)


@app.route("/api/daily-report/<app_key>")
def api_daily_report(app_key):
    if app_key not in APPS:
        return jsonify({"error": "unknown app"}), 404
    period = request.args.get("period", 7, type=int)
    return jsonify(build_daily_report(app_key, period))


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

def _current_review_sig(app_key):
    """현재 리뷰의 시그니처 + 건수 (affinity_analyzer._review_signature와 동일 규칙)."""
    path = f"data/{app_key}_reviews.json"
    if not os.path.exists(path):
        return None, 0
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f).get("reviews", [])
    ids = sorted(str(r.get("reviewId") or (r.get("content") or "")[:60]) for r in raw)
    return hashlib.md5("\n".join(ids).encode("utf-8")).hexdigest(), len(raw)


@app.route("/api/affinity-status/<app_key>")
def api_affinity_status(app_key):
    """리뷰 현황(트리맵)이 최신 리뷰 대비 갱신이 필요한지 판단."""
    if app_key not in APPS:
        return jsonify({"error": "unknown app"}), 404
    analyzed, stale, cur_count, aff_count, basis = _affinity_staleness(app_key)
    return jsonify({
        "analyzed": analyzed,
        "stale": stale,
        "current_reviews": cur_count,
        "affinity_reviews": aff_count,
        "basis": basis,
    })


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

        # 진행 메시지가 잠깐 없어도(라벨링의 60초 대기, 느린 LLM 호출 등) 끊지 않는다.
        # 15초마다 keep-alive(SSE 주석)로 연결만 유지하고, 정말 오래(10분) 무응답일 때만 종료.
        idle = 0
        MAX_IDLE = 600
        while True:
            try:
                msg = msg_queue.get(timeout=15)
                idle = 0
                yield f"data: {msg}\n\n"
                if msg.startswith("[DONE]") or msg.startswith("[ERROR]"):
                    break
            except queue.Empty:
                idle += 15
                if idle >= MAX_IDLE:
                    yield "data: [ERROR] 응답 없음 — 백그라운드 분석은 계속될 수 있습니다(다음 실행 시 이어서)\n\n"
                    break
                yield ": keepalive\n\n"  # EventSource가 무시하는 주석 → 연결 유지

    return app.response_class(generate(), mimetype="text/event-stream")


@app.route("/api/refresh-stream")
def api_refresh_stream():
    """리뷰 데이터 새로고침 — SSE로 실시간 로그 스트리밍"""
    import subprocess, sys

    def generate():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "scraper.py"],
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
        except Exception as e:
            yield f"data: [ERROR] 실행 실패: {type(e).__name__}: {str(e)[:120]}\n\n"

    return app.response_class(generate(), mimetype="text/event-stream")


@app.route("/api/refresh-app/<app_key>")
def api_refresh_app(app_key):
    """앱별 증분 새로고침 — 기존 리뷰 유지 + 신규만 이어붙임. SSE 로그 스트리밍."""
    import subprocess, sys
    if app_key not in APPS:
        return jsonify({"error": "unknown app"}), 404

    def generate():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "scraper.py", app_key, "--append"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    yield f"data: {line}\n\n"
            proc.wait()
            yield "data: [DONE]\n\n" if proc.returncode == 0 else f"data: [ERROR] 종료 코드: {proc.returncode}\n\n"
        except Exception as e:
            yield f"data: [ERROR] 실행 실패: {type(e).__name__}: {str(e)[:120]}\n\n"

    return app.response_class(generate(), mimetype="text/event-stream")


@app.route("/api/run-history")
def api_run_history():
    """일일 자동화 실행 이력 + 헬스 상태."""
    from daily_job import load_history, next_run_time, DEFAULT_HOUR
    hist = load_history()
    last_success = next((r for r in hist if r.get("status") == "success"), None)
    last = hist[0] if hist else None
    # 마지막 정상 실행이 26시간 넘으면 'stale'
    stale = True
    if last_success and last_success.get("finished_at"):
        try:
            delta = datetime.now() - datetime.fromisoformat(last_success["finished_at"])
            stale = delta > timedelta(hours=26)
        except Exception:
            stale = True
    health = "ok"
    if not hist:
        health = "none"
    elif last and last.get("status") in ("error", "partial"):
        health = last["status"]
    elif last and last.get("quota_limited"):
        health = "quota"
    elif stale:
        health = "stale"
    return jsonify({
        "runs": hist,
        "next_run": next_run_time(),
        "scheduler_hour": DEFAULT_HOUR,
        "health": health,
        "last_finished_at": (last or {}).get("finished_at"),
    })


@app.route("/api/run-daily")
def api_run_daily():
    """수동 일일 잡 실행 — SSE 로그 스트리밍 (스케줄러와 잠금 공유로 중복 방지)."""
    import subprocess, sys

    def generate():
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "daily_job.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line:
                    yield f"data: {line}\n\n"
            proc.wait()
            yield "data: [DONE]\n\n" if proc.returncode == 0 else f"data: [ERROR] 종료 코드: {proc.returncode}\n\n"
        except Exception as e:
            yield f"data: [ERROR] 실행 실패: {type(e).__name__}: {str(e)[:120]}\n\n"

    return app.response_class(generate(), mimetype="text/event-stream")


# 인앱 스케줄러 기동 (ENABLE_SCHEDULER=0 이면 비활성 — cron/launchd 사용 시)
if os.getenv("ENABLE_SCHEDULER", "1") == "1":
    try:
        from daily_job import start_scheduler
        start_scheduler()
    except Exception as _e:
        print(f"[scheduler] 시작 실패: {_e}")


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
