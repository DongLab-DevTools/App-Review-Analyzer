"""Gemini AI로 리뷰 감성/카테고리 분석

사용법: python analyzer.py
입력: data/{app_key}_reviews.json
출력: data/{app_key}_analyzed.json
"""

import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from config import APPS, ANALYSIS_CATEGORIES, GEMINI_BATCH_SIZE

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

POSITIVE_CATS = ANALYSIS_CATEGORIES["positive"]
NEGATIVE_CATS = ANALYSIS_CATEGORIES["negative"]

PROMPT_TEMPLATE = """다음은 앱 리뷰 목록입니다. 각 리뷰를 분석해주세요.

## 분석 규칙
- 별점 3점 이상: sentiment = "positive", category는 긍정 카테고리 중 선택
- 별점 2점 이하: sentiment = "negative", category는 부정 카테고리 중 선택
- summary: 리뷰 핵심을 한 줄로 요약 (15자 이내)

## 긍정 카테고리
{positive_cats}

## 부정 카테고리
{negative_cats}

## 리뷰 목록
{reviews_text}

## 응답 형식
반드시 JSON 배열로만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.
[
  {{"reviewId": "...", "sentiment": "positive|negative", "category": "카테고리명", "summary": "요약"}}
]
"""


def load_reviews(app_key: str) -> dict | None:
    path = f"data/{app_key}_reviews.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_batch(batch: list) -> list:
    """Gemini에 배치 분석 요청"""
    reviews_text = ""
    for r in batch:
        reviews_text += f'- [ID:{r["reviewId"]}] 별점:{r["score"]} 내용:"{r["content"][:200]}"\n'

    prompt = PROMPT_TEMPLATE.format(
        positive_cats=", ".join(POSITIVE_CATS),
        negative_cats=", ".join(NEGATIVE_CATS),
        reviews_text=reviews_text,
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # JSON 블록 추출
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        print(f"    [ERROR] Gemini 분석 실패: {e}")
        return []


def fallback_analysis(review: dict) -> dict:
    """Gemini 실패 시 별점 기반 폴백"""
    score = review.get("score", 3)
    return {
        "reviewId": review["reviewId"],
        "sentiment": "positive" if score >= 3 else "negative",
        "category": "기타 칭찬" if score >= 3 else "기타 불만",
        "summary": "",
    }


def analyze_app(app_key: str, reviews_data: dict):
    """앱 전체 리뷰 분석"""
    reviews = reviews_data.get("reviews", [])
    if not reviews:
        print(f"  리뷰 없음")
        return

    results = []
    total = len(reviews)

    for i in range(0, total, GEMINI_BATCH_SIZE):
        batch = reviews[i : i + GEMINI_BATCH_SIZE]
        print(f"  분석 중... {i+1}~{min(i+GEMINI_BATCH_SIZE, total)} / {total}")

        analyzed = analyze_batch(batch)

        # 분석 결과를 reviewId로 매핑
        analyzed_map = {a["reviewId"]: a for a in analyzed if "reviewId" in a}

        for r in batch:
            rid = r["reviewId"]
            if rid in analyzed_map:
                result = analyzed_map[rid]
                # 원본 리뷰 필드 병합
                result.update({
                    "userName": r.get("userName", ""),
                    "score": r.get("score", 0),
                    "content": r.get("content", ""),
                    "thumbsUpCount": r.get("thumbsUpCount", 0),
                    "at": r.get("at", ""),
                    "replyContent": r.get("replyContent", ""),
                    "repliedAt": r.get("repliedAt", ""),
                    "appVersion": r.get("appVersion", ""),
                    "store": r.get("store", "PLAY"),
                })
            else:
                result = fallback_analysis(r)
                result.update({
                    "userName": r.get("userName", ""),
                    "score": r.get("score", 0),
                    "content": r.get("content", ""),
                    "thumbsUpCount": r.get("thumbsUpCount", 0),
                    "at": r.get("at", ""),
                    "replyContent": r.get("replyContent", ""),
                    "repliedAt": r.get("repliedAt", ""),
                    "appVersion": r.get("appVersion", ""),
                    "store": r.get("store", "PLAY"),
                })
            results.append(result)

        time.sleep(2)  # API rate limit

    # 저장
    output = {
        "app_key": app_key,
        "app_name": reviews_data.get("app_name", ""),
        "analyzed_at": datetime.now().isoformat(),
        "total_analyzed": len(results),
        "results": results,
    }
    filepath = f"data/{app_key}_analyzed.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  -> {filepath} ({len(results)}건)")


def main():
    print("=" * 60)
    print("Gemini AI 리뷰 분석 시작")
    print("=" * 60)

    for key, config in APPS.items():
        print(f"\n[{config['name']}]")
        reviews_data = load_reviews(key)
        if not reviews_data:
            print("  데이터 없음 - 건너뜀")
            continue
        analyze_app(key, reviews_data)

    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
