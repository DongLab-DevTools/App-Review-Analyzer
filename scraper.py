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


# ── 메인 ──

def main():
    parser = argparse.ArgumentParser(description="앱 리뷰 수집")
    parser.add_argument("apps", nargs="*", help="수집할 앱 키 (미지정 시 전체)")
    parser.add_argument("--list", action="store_true", help="등록된 앱 목록 출력")
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
        config = APPS[key]
        print(f"\n[{config['name']}]")

        # Google Play
        print(f"  Google Play ({config['package_id']})")
        info = fetch_gp_info(config["package_id"])
        if info:
            print(f"    평점: {info.get('score', 'N/A')}, 설치수: {info.get('installs', 'N/A'):,}")

        gp_raw = fetch_gp_reviews(config["package_id"], MAX_REVIEWS_GP)
        gp_reviews = [normalize_gp_review(r) for r in gp_raw]

        # App Store
        as_reviews = []
        app_store_id = config.get("app_store_id")
        if app_store_id:
            print(f"  App Store (ID: {app_store_id})")
            as_raw = fetch_as_reviews(app_store_id, "", MAX_REVIEWS_AS)
            as_reviews = [normalize_as_review(r) for r in as_raw]
        else:
            print("  App Store: ID 없음 - 건너뜀")

        # 통합 저장
        all_reviews = gp_reviews + as_reviews
        if all_reviews:
            save_data(key, config, all_reviews, info)
        else:
            print("  [WARNING] 수집된 리뷰 없음")

    print("\n" + "=" * 60)
    print("수집 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
