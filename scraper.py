"""Google Play + App Store 리뷰 수집 스크립트

사용법:
  python scraper.py                     # 전체 앱 수집
  python scraper.py tving netflix       # 특정 앱만 수집
  python scraper.py --list              # 등록된 앱 목록 확인
출력: data/{app_key}_reviews.json, data/{app_key}_info.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from google_play_scraper import Sort, reviews, app as gp_app_info
from config import APPS, MAX_REVIEWS_GP, MAX_REVIEWS_AS


# ── Google Play ──

def fetch_gp_info(package_id: str) -> dict:
    """Google Play 앱 기본 정보"""
    try:
        info = gp_app_info(package_id, lang="ko", country="kr")
        return {
            "title": info.get("title", ""),
            "score": info.get("score", 0),
            "ratings": info.get("ratings", 0),
            "installs": info.get("realInstalls", 0),
            "updated": info.get("updated", ""),
            "version": info.get("version", ""),
            "icon": info.get("icon", ""),
        }
    except Exception as e:
        print(f"  [ERROR] GP 앱 정보 조회 실패: {e}")
        return {}


def fetch_gp_reviews(package_id: str, max_count: int) -> list:
    """Google Play 리뷰 수집"""
    all_reviews = []
    token = None
    batch = 200

    while len(all_reviews) < max_count:
        try:
            result, token = reviews(
                package_id,
                lang="ko",
                country="kr",
                sort=Sort.NEWEST,
                count=min(batch, max_count - len(all_reviews)),
                continuation_token=token,
            )
            if not result:
                break
            all_reviews.extend(result)
            print(f"    GP 수집 중... {len(all_reviews)}건")
            if token is None:
                break
            time.sleep(1)
        except Exception as e:
            print(f"    [ERROR] GP 수집 오류: {e}")
            break

    return all_reviews


def normalize_gp_review(r: dict) -> dict:
    """Google Play 리뷰를 통합 포맷으로 변환"""
    return {
        "reviewId": r.get("reviewId", ""),
        "userName": r.get("userName", ""),
        "score": r.get("score", 0),
        "content": r.get("content", ""),
        "thumbsUpCount": r.get("thumbsUpCount", 0),
        "at": r.get("at").isoformat() if r.get("at") else "",
        "replyContent": r.get("replyContent", ""),
        "repliedAt": r.get("repliedAt").isoformat() if r.get("repliedAt") else "",
        "appVersion": r.get("appVersion", ""),
        "store": "PLAY",
    }


# ── App Store (iTunes RSS API) ──

def fetch_as_reviews(app_store_id: int, app_name: str, max_count: int) -> list:
    """App Store 리뷰 수집 — iTunes RSS JSON API 사용"""
    import requests

    all_entries = []
    max_pages = min(10, (max_count + 49) // 50)

    for page in range(1, max_pages + 1):
        try:
            url = f"https://itunes.apple.com/kr/rss/customerreviews/page={page}/id={app_store_id}/sortby=mostrecent/json"
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"    AS page {page}: HTTP {resp.status_code}")
                break
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])
            if not entries:
                break
            for entry in entries:
                if "im:rating" in entry:
                    all_entries.append(entry)
            print(f"    AS 수집 중... {len(all_entries)}건 (page {page})")
            if len(all_entries) >= max_count:
                break
            time.sleep(1)
        except Exception as e:
            print(f"    [ERROR] AS page {page} 오류: {e}")
            break

    print(f"    AS 수집 완료: {len(all_entries)}건")
    return all_entries[:max_count]


def _rss_val(entry, *keys):
    """RSS JSON에서 중첩 label 값 안전하게 추출"""
    obj = entry
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, "")
        else:
            return ""
    return obj if isinstance(obj, str) else ""


def normalize_as_review(r: dict) -> dict:
    """iTunes RSS 리뷰 entry를 통합 포맷으로 변환"""
    return {
        "reviewId": _rss_val(r, "id", "label"),
        "userName": _rss_val(r, "author", "name", "label"),
        "score": int(_rss_val(r, "im:rating", "label") or "0"),
        "content": _rss_val(r, "content", "label"),
        "thumbsUpCount": int(_rss_val(r, "im:voteSum", "label") or "0"),
        "at": _rss_val(r, "updated", "label"),
        "replyContent": "",
        "repliedAt": "",
        "appVersion": _rss_val(r, "im:version", "label"),
        "store": "APPLE",
    }


# ── 저장 ──

def save_data(app_key: str, app_config: dict, all_reviews: list, info: dict):
    """리뷰 + 앱 정보 JSON 저장"""
    os.makedirs("data", exist_ok=True)

    # 리뷰 저장
    output = {
        "app_key": app_key,
        "app_name": app_config["name"],
        "package_id": app_config["package_id"],
        "collected_at": datetime.now().isoformat(),
        "total_reviews": len(all_reviews),
        "reviews": all_reviews,
    }
    filepath = f"data/{app_key}_reviews.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  -> {filepath} ({len(all_reviews)}건)")

    # 앱 정보 저장
    if info:
        info_path = f"data/{app_key}_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


def merge_incremental(app_key: str, fetched: list):
    """기존 리뷰는 유지하고, reviewId 기준 신규 리뷰만 이어붙인다.
    반환: (merged_reviews, new_count, existing_count)"""
    path = f"data/{app_key}_reviews.json"
    existing = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f).get("reviews", [])
    seen = {r.get("reviewId") for r in existing if r.get("reviewId")}
    new_items = [r for r in fetched if r.get("reviewId") and r.get("reviewId") not in seen]
    # 기존 유지 + 신규 이어붙이기 (신규는 최신순으로 앞에 배치)
    merged = new_items + existing
    return merged, len(new_items), len(existing)


# ── 메인 ──

def collect_app(key: str, append: bool = True, on_progress=None) -> dict:
    """단일 앱 리뷰 수집(+증분 병합). 반환: {new, exist, total, score}.
    daily_job 등에서 재사용하기 위해 main()의 수집 로직을 함수로 분리."""
    def log(m):
        (on_progress or print)(m)

    config = APPS[key]
    info = fetch_gp_info(config["package_id"])
    if info:
        log(f"  평점: {info.get('score', 'N/A')}")

    gp_raw = fetch_gp_reviews(config["package_id"], MAX_REVIEWS_GP)
    gp_reviews = [normalize_gp_review(r) for r in gp_raw]

    as_reviews = []
    app_store_id = config.get("app_store_id")
    if app_store_id:
        as_raw = fetch_as_reviews(app_store_id, "", MAX_REVIEWS_AS)
        as_reviews = [normalize_as_review(r) for r in as_raw]

    all_reviews = gp_reviews + as_reviews
    if append:
        merged, new_cnt, exist_cnt = merge_incremental(key, all_reviews)
        log(f"  증분: 기존 {exist_cnt}건 + 신규 {new_cnt}건 = 총 {len(merged)}건")
        if merged:
            save_data(key, config, merged, info)
        return {"new": new_cnt, "exist": exist_cnt, "total": len(merged),
                "score": (info or {}).get("score")}
    else:
        if all_reviews:
            save_data(key, config, all_reviews, info)
        return {"new": len(all_reviews), "exist": 0, "total": len(all_reviews),
                "score": (info or {}).get("score")}


def main():
    parser = argparse.ArgumentParser(description="앱 리뷰 수집")
    parser.add_argument("apps", nargs="*", help="수집할 앱 키 (미지정 시 전체)")
    parser.add_argument("--list", action="store_true", help="등록된 앱 목록 출력")
    parser.add_argument("--append", action="store_true", help="기존 리뷰 유지 + 신규만 이어붙이기(증분)")
    args = parser.parse_args()

    if args.list:
        for k, v in APPS.items():
            print(f"  {k:16s} {v['name']}")
        return

    target_keys = args.apps if args.apps else list(APPS.keys())
    invalid = [k for k in target_keys if k not in APPS]
    if invalid:
        print(f"[ERROR] 등록되지 않은 앱 키: {', '.join(invalid)}")
        print("등록된 앱 목록: python scraper.py --list")
        sys.exit(1)

    print("=" * 60)
    print(f"Google Play + App Store 리뷰 수집 시작 ({len(target_keys)}개 앱)")
    print("=" * 60)

    for key in target_keys:
        print(f"\n[{APPS[key]['name']}]")
        collect_app(key, append=args.append)

    print("\n" + "=" * 60)
    print("수집 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
