"""임베딩 + 클러스터링 기반 리뷰 어피니티 자동 분석

파이프라인: 전처리 → Gemini 임베딩 → 1차 클러스터링 → 2차 클러스터링 → LLM 라벨링 → JSON 조립

사용법:
  from affinity_analyzer import run_affinity_pipeline
  result_json = run_affinity_pipeline(app_key, on_progress=callback)
"""

import json
import os
import re
import time
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import google.generativeai as genai
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from config import APPS, GEMINI_BATCH_SIZE

# ────────────────────────────────────────
# API 키 로테이션
# ────────────────────────────────────────

def _load_api_keys() -> list[str]:
    """GEMINI_API_KEYS(쉼표 구분) 또는 GEMINI_API_KEY에서 키 목록 로드"""
    keys_str = os.getenv("GEMINI_API_KEYS", "")
    if keys_str:
        keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        if keys:
            return keys
    single = os.getenv("GEMINI_API_KEY", "")
    return [single] if single else []

_api_keys = _load_api_keys()
_current_key_idx = 0

def _configure_genai():
    """현재 키로 genai 설정"""
    if _api_keys:
        genai.configure(api_key=_api_keys[_current_key_idx])

def _rotate_key(on_progress=None) -> bool:
    """다음 키로 전환. 성공 시 True, 모든 키 소진 시 False"""
    global _current_key_idx
    if len(_api_keys) <= 1:
        return False
    _current_key_idx = (_current_key_idx + 1) % len(_api_keys)
    genai.configure(api_key=_api_keys[_current_key_idx])
    if on_progress:
        on_progress(f"API 키 전환 ({_current_key_idx + 1}/{len(_api_keys)}번째 키)")
    return True

def set_user_api_key(key: str):
    """사용자가 직접 입력한 API 키를 최우선으로 추가"""
    global _api_keys, _current_key_idx
    if key in _api_keys:
        _current_key_idx = _api_keys.index(key)
    else:
        _api_keys.insert(0, key)
        _current_key_idx = 0
    genai.configure(api_key=_api_keys[_current_key_idx])

_configure_genai()

# ────────────────────────────────────────
# ① 전처리
# ────────────────────────────────────────

def preprocess(reviews: list[dict]) -> list[dict]:
    """리뷰 정제: 빈 리뷰, 중복, 노이즈 제거"""
    cleaned = []
    seen = set()
    for r in reviews:
        text = (r.get("content") or "").strip()
        if len(text) < 3:
            continue
        if text in seen:
            continue
        seen.add(text)
        # 의미 없는 리뷰 필터 (문자 없이 기호만)
        if len(text) < 5 and not any(c.isalpha() for c in text):
            continue
        cleaned.append({
            "id": r.get("reviewId", str(len(cleaned))),
            "text": text,
            "rating": r.get("score", 0),
            "date": (r.get("at") or "")[:10],
        })
    return cleaned


# ────────────────────────────────────────
# ② 로컬 임베딩 (sentence-transformers)
# ────────────────────────────────────────

LOCAL_EMBED_MODEL = "jhgan/ko-sroberta-multitask"
_local_model = None


def _get_local_model(on_progress=None):
    """sentence-transformers 모델 lazy 로드 (최초 1회 다운로드)"""
    global _local_model
    if _local_model is None:
        if on_progress:
            on_progress("로컬 임베딩 모델 로드 중... (최초 실행 시 ~500MB 다운로드)")
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(LOCAL_EMBED_MODEL)
        if on_progress:
            on_progress(f"모델 로드 완료: {LOCAL_EMBED_MODEL}")
    return _local_model


def embed_reviews_local(reviews: list[dict], on_progress=None, app_key=None, base_embeddings=None) -> np.ndarray:
    """로컬 sentence-transformers 임베딩 — API 호출 없음, 한도 없음"""
    texts = [r["text"] for r in reviews]
    model = _get_local_model(on_progress)

    if on_progress:
        on_progress(f"로컬 임베딩 시작... ({len(texts)}건)")

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False)

    if on_progress:
        on_progress(f"로컬 임베딩 완료: {embeddings.shape}")

    if base_embeddings is not None:
        embeddings = np.concatenate([base_embeddings, embeddings])

    if app_key:
        _save_embeddings(app_key, embeddings)

    return embeddings


# ────────────────────────────────────────
# ③ 1차 클러스터링 (대분류)
# ────────────────────────────────────────

def find_optimal_k(embeddings: np.ndarray, k_min: int, k_max: int) -> tuple[int, np.ndarray]:
    """Silhouette Score로 최적 K 탐색"""
    normed = normalize(embeddings)
    best_score = -1
    best_k = k_min
    best_labels = None

    sample_size = min(5000, len(normed))

    for k in range(k_min, k_max + 1):
        if k >= len(normed):
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normed)
        score = silhouette_score(normed, labels, sample_size=sample_size)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    return best_k, best_labels


def cluster_primary(embeddings: np.ndarray, on_progress=None) -> np.ndarray:
    """대분류 클러스터링"""
    n = len(embeddings)
    k_min = max(3, min(5, n // 50))
    k_max = min(15, n // 10)
    if k_max <= k_min:
        k_max = k_min + 1

    if on_progress:
        on_progress(f"대분류 클러스터링 중... (K={k_min}~{k_max} 탐색)")

    best_k, labels = find_optimal_k(embeddings, k_min, k_max)

    if on_progress:
        on_progress(f"대분류 완료: {best_k}개 카테고리")

    return labels


# ────────────────────────────────────────
# ④ 2차 클러스터링 (소분류)
# ────────────────────────────────────────

def cluster_secondary(
    embeddings: np.ndarray,
    primary_labels: np.ndarray,
    on_progress=None,
) -> dict[int, np.ndarray]:
    """대분류 내 소분류 클러스터링"""
    normed = normalize(embeddings)
    secondary = {}
    unique_cats = sorted(set(primary_labels))

    for cat_id in unique_cats:
        if cat_id == -1:
            continue
        mask = primary_labels == cat_id
        cluster_emb = normed[mask]
        n = len(cluster_emb)

        if n < 6:
            secondary[cat_id] = np.zeros(n, dtype=int)
            continue

        k_min = 2
        k_max = min(8, n // 3)
        if k_max <= k_min:
            secondary[cat_id] = np.zeros(n, dtype=int)
            continue

        _, sub_labels = find_optimal_k(cluster_emb, k_min, k_max)
        secondary[cat_id] = sub_labels

    total_subs = sum(len(set(v)) for v in secondary.values())
    if on_progress:
        on_progress(f"소분류 완료: 총 {total_subs}개")

    return secondary


# ────────────────────────────────────────
# ⑤ LLM 라벨링
# ────────────────────────────────────────

def select_representatives(
    reviews: list[dict],
    embeddings: np.ndarray,
    indices: np.ndarray,
    n_samples: int = 7,
) -> list[str]:
    """클러스터 대표 리뷰 선정 (centroid 가까운 + 먼 것)"""
    if len(indices) <= n_samples:
        return [reviews[i]["text"] for i in indices]

    cluster_emb = normalize(embeddings[indices])
    centroid = cluster_emb.mean(axis=0)
    distances = np.linalg.norm(cluster_emb - centroid, axis=1)

    n_close = max(1, n_samples - 2)
    closest = distances.argsort()[:n_close]
    farthest = distances.argsort()[-2:]
    selected = list(dict.fromkeys(list(closest) + list(farthest)))[:n_samples]

    return [reviews[indices[i]]["text"] for i in selected]


def generate_labels(
    reviews: list[dict],
    embeddings: np.ndarray,
    primary_labels: np.ndarray,
    secondary: dict[int, np.ndarray],
    on_progress=None,
) -> dict:
    """LLM으로 클러스터 라벨 생성"""
    cluster_data = []

    for cat_id in sorted(set(primary_labels)):
        if cat_id == -1:
            continue
        cat_mask = primary_labels == cat_id
        cat_indices = np.where(cat_mask)[0]
        sub_labels = secondary.get(cat_id, np.zeros(len(cat_indices), dtype=int))

        for sub_id in sorted(set(sub_labels)):
            sub_mask = sub_labels == sub_id
            sub_indices = cat_indices[sub_mask]

            reps = select_representatives(reviews, embeddings, sub_indices, 7)
            ratings = [reviews[i]["rating"] for i in sub_indices]
            avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else 0

            cluster_data.append({
                "cat_id": int(cat_id),
                "sub_id": int(sub_id),
                "count": int(len(sub_indices)),
                "avg_rating": avg_rating,
                "sample_reviews": reps,
            })

    if on_progress:
        on_progress(f"LLM 라벨링 중... ({len(cluster_data)}개 클러스터)")

    prompt = f"""아래는 앱 리뷰를 클러스터링한 결과입니다.
각 클러스터의 대표 리뷰를 보고, 대분류 라벨과 소분류 라벨을 한국어로 생성해주세요.

규칙:
1. 같은 cat_id를 가진 클러스터들은 하나의 대분류에 속합니다
2. 대분류 라벨: 핵심 주제를 2~4단어로 (예: "동일가구 인증 정책", "광고 과다")
3. 소분류 라벨: 세부 이슈를 2~5단어로 (예: "결제자 본인도 인증 반복")
4. 대분류 description: 한 줄 설명
5. sentiment: 평균 별점 1~2점이면 negative, 3점이면 mixed, 4~5점이면 positive

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 순수 JSON만:

{{
  "categories": [
    {{
      "cat_id": 0,
      "label": "대분류명",
      "description": "설명",
      "subcategories": [
        {{
          "sub_id": 0,
          "label": "소분류명",
          "sentiment": "negative"
        }}
      ]
    }}
  ]
}}

클러스터 데이터:
{json.dumps(cluster_data, ensure_ascii=False, indent=2)}
"""

    # 최대 3라운드 시도 (키 전체 순회 × 3, 라운드 사이 60초 대기)
    last_err = None
    dead_llm_keys = set()  # LLM에서도 RPD 소진된 키 추적
    succeeded_llm_keys = set()

    for round_num in range(3):
        for key_idx in range(len(_api_keys)):
            if key_idx in dead_llm_keys:
                continue
            try:
                if on_progress and key_idx > 0:
                    on_progress(f"LLM 키 {key_idx+1}/{len(_api_keys)} 시도 중...")
                genai.configure(api_key=_api_keys[key_idx])
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content(prompt)
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                result = json.loads(text)
                if on_progress:
                    on_progress("LLM 라벨링 완료")
                return result
            except Exception as e:
                last_err = e
                is_quota = "429" in str(e) or "ResourceExhausted" in type(e).__name__
                if is_quota:
                    if key_idx not in succeeded_llm_keys:
                        dead_llm_keys.add(key_idx)
                        if on_progress:
                            on_progress(f"LLM 키 {key_idx+1} 일일 한도 소진 → 제외")
                    continue
                # JSON 파싱 오류 등
                if on_progress:
                    on_progress(f"LLM 오류 (키 {key_idx+1}): {type(e).__name__}: {str(e)[:80]}")
                break

        # 살아있는 키가 있으면 TPM 초기화 대기
        alive = len(_api_keys) - len(dead_llm_keys)
        if alive == 0:
            break
        if round_num < 2:
            if on_progress:
                on_progress(f"LLM TPM 대기 중... (60초)")
            time.sleep(61)

    if on_progress:
        on_progress(f"LLM 라벨링 실패: 기본 라벨 사용 (다음 실행 시 재시도)")
    return _fallback_labels(primary_labels, secondary)


def _fallback_labels(primary_labels, secondary):
    """LLM 실패 시 기본 라벨"""
    categories = []
    for cat_id in sorted(set(primary_labels)):
        if cat_id == -1:
            continue
        sub_labels = secondary.get(cat_id, np.array([0]))
        subs = []
        for sub_id in sorted(set(sub_labels)):
            subs.append({
                "sub_id": int(sub_id),
                "label": f"세부 그룹 {sub_id + 1}",
                "sentiment": "neutral",
            })
        categories.append({
            "cat_id": int(cat_id),
            "label": f"카테고리 {cat_id + 1}",
            "description": "",
            "subcategories": subs,
        })
    return {"categories": categories}


# ────────────────────────────────────────
# ⑥ JSON 조립
# ────────────────────────────────────────

def assemble_json(
    reviews: list[dict],
    primary_labels: np.ndarray,
    secondary: dict[int, np.ndarray],
    llm_labels: dict,
    app_name: str,
) -> dict:
    """최종 어피니티 JSON 조립"""
    # LLM 라벨 lookup
    cat_map = {}
    sub_map = {}
    for cat in llm_labels.get("categories", []):
        cid = cat["cat_id"]
        cat_map[cid] = {"label": cat["label"], "description": cat.get("description", "")}
        for sub in cat.get("subcategories", []):
            sub_map[(cid, sub["sub_id"])] = {
                "label": sub["label"],
                "sentiment": sub.get("sentiment", "neutral"),
            }

    categories = []
    for cat_id in sorted(set(primary_labels)):
        if cat_id == -1:
            continue
        cat_info = cat_map.get(cat_id, {"label": f"카테고리 {cat_id + 1}", "description": ""})
        cat_mask = primary_labels == cat_id
        cat_indices = np.where(cat_mask)[0]
        sub_labels = secondary.get(cat_id, np.zeros(len(cat_indices), dtype=int))

        subcategories = []
        for sub_id in sorted(set(sub_labels)):
            sub_info = sub_map.get(
                (cat_id, sub_id),
                {"label": f"세부 {sub_id + 1}", "sentiment": "neutral"},
            )
            sub_mask = sub_labels == sub_id
            sub_indices = cat_indices[sub_mask]

            items = []
            for idx in sub_indices:
                r = reviews[idx]
                items.append({
                    "text": r["text"],
                    "rating": r["rating"],
                    "date": r["date"],
                })

            subcategories.append({
                "label": sub_info["label"],
                "sentiment": sub_info["sentiment"],
                "items": items,
            })

        # 소분류를 리뷰 수 내림차순 정렬
        subcategories.sort(key=lambda s: len(s["items"]), reverse=True)
        categories.append({
            "label": cat_info["label"],
            "description": cat_info["description"],
            "subcategories": subcategories,
        })

    # 대분류를 리뷰 수 내림차순 정렬
    categories.sort(key=lambda c: sum(len(s["items"]) for s in c["subcategories"]), reverse=True)

    return {
        "meta": {
            "app_name": app_name,
            "platform": "all",
            "total_reviews": len(reviews),
            "analyzed_at": datetime.now().isoformat(),
        },
        "categories": categories,
    }


# ────────────────────────────────────────
# 전체 파이프라인
# ────────────────────────────────────────

# ────────────────────────────────────────
# 단계별 캐싱
# ────────────────────────────────────────

CACHE_NEW_REVIEW_THRESHOLD = 100  # 새 리뷰 이 이상이면 캐시 무효화


def _cache_path(app_key):
    return f"data/{app_key}_affinity_cache.json"


def _embeddings_path(app_key):
    return f"data/{app_key}_affinity_embeddings.npy"


def _load_cache(app_key):
    path = _cache_path(app_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(app_key, cache):
    with open(_cache_path(app_key), "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _save_embeddings(app_key, embeddings):
    np.save(_embeddings_path(app_key), embeddings)


def _load_embeddings(app_key):
    path = _embeddings_path(app_key)
    if not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _should_invalidate(cache, current_review_count):
    """캐시 무효화 판단: 새 리뷰 100건 이상이면 True"""
    if not cache:
        return True
    cached_count = cache.get("review_count", 0)
    new_reviews = current_review_count - cached_count
    return new_reviews >= CACHE_NEW_REVIEW_THRESHOLD


# ────────────────────────────────────────
# 전체 파이프라인
# ────────────────────────────────────────

def run_affinity_pipeline(app_key: str, on_progress=None, user_api_key: str = None) -> dict:
    """
    앱 리뷰 어피니티 분석 파이프라인 (단계별 캐싱)

    임베딩: 로컬 sentence-transformers (API 호출 없음)
    라벨링: Gemini API (LLM)
    """
    # LLM 라벨링용 API 키 로딩
    global _api_keys, _current_key_idx
    load_dotenv(override=True)
    _api_keys = _load_api_keys()
    _current_key_idx = 0

    if user_api_key:
        if user_api_key in _api_keys:
            _current_key_idx = _api_keys.index(user_api_key)
        else:
            _api_keys.insert(0, user_api_key)
            _current_key_idx = 0

    _configure_genai()

    def log(msg):
        if on_progress:
            on_progress(msg)
        print(f"  [affinity] {msg}")

    app_name = APPS.get(app_key, {}).get("name", app_key)

    # 리뷰 데이터 로드
    raw_reviews = []
    for path in [f"data/{app_key}_analyzed.json", f"data/{app_key}_reviews.json"]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                raw_reviews = data.get("results", []) or data.get("reviews", [])
                if raw_reviews:
                    break

    if not raw_reviews:
        raise ValueError(f"리뷰 데이터 없음: {app_key}")

    log(f"원본 리뷰: {len(raw_reviews)}건")

    # 캐시 로드 & 무효화 판단
    cache = _load_cache(app_key)
    if _should_invalidate(cache, len(raw_reviews)):
        if cache:
            new_cnt = len(raw_reviews) - cache.get("review_count", 0)
            log(f"새 리뷰 +{new_cnt}건 감지 → 캐시 초기화, 처음부터 분석")
        cache = {"review_count": len(raw_reviews), "completed": []}
    else:
        completed = cache.get("completed", [])
        log(f"캐시 발견: {', '.join(completed) if completed else '없음'} 완료됨")

    completed = cache.get("completed", [])

    # ① 전처리
    if "preprocess" in completed and cache.get("reviews"):
        reviews = cache["reviews"]
        log(f"전처리 캐시 사용 ({len(reviews)}건)")
    else:
        reviews = preprocess(raw_reviews)
        log(f"전처리 완료: {len(reviews)}건 (중복/노이즈 {len(raw_reviews) - len(reviews)}건 제거)")
        if len(reviews) < 10:
            raise ValueError(f"유효 리뷰가 너무 적습니다: {len(reviews)}건")
        cache["reviews"] = reviews
        cache["completed"] = ["preprocess"]
        _save_cache(app_key, cache)

    # ② 임베딩 (부분 캐시 지원 — 중간 실패 시 이어서 진행)
    embeddings = None
    if "embed" in completed:
        embeddings = _load_embeddings(app_key)
        if embeddings is not None and len(embeddings) == len(reviews):
            log(f"임베딩 캐시 사용 ({embeddings.shape})")
        else:
            embeddings = None

    # 부분 임베딩 캐시 확인
    partial_embeddings = None
    if embeddings is None:
        partial = _load_embeddings(app_key)
        if partial is not None and 0 < len(partial) < len(reviews):
            partial_embeddings = partial
            log(f"부분 임베딩 캐시 발견: {len(partial)}/{len(reviews)}건 — 이어서 진행")

    if embeddings is None:
        start_idx = len(partial_embeddings) if partial_embeddings is not None else 0
        remaining_reviews = reviews[start_idx:]
        log(f"임베딩 {'재개' if start_idx > 0 else '시작'}... ({start_idx}/{len(reviews)}건 완료됨)")

        try:
            new_embeddings = embed_reviews_local(
                remaining_reviews, on_progress=log,
                app_key=app_key, base_embeddings=partial_embeddings,
            )
            if partial_embeddings is not None:
                embeddings = np.concatenate([partial_embeddings, new_embeddings])
            else:
                embeddings = new_embeddings
            log(f"임베딩 완료: {embeddings.shape}")
            _save_embeddings(app_key, embeddings)
            cache["completed"] = ["preprocess", "embed"]
            _save_cache(app_key, cache)
        except Exception as e:
            # 실패해도 지금까지 진행분 저장
            if partial_embeddings is not None or start_idx == 0:
                # embed_reviews 내부에서 all_embeddings에 부분 결과가 있을 수 있지만
                # 함수가 예외를 던지면 접근 불가 → 기존 partial만 유지
                pass
            raise

    # ③ 1차 클러스터링
    if "cluster_primary" in completed and cache.get("primary_labels"):
        primary_labels = np.array(cache["primary_labels"])
        log(f"1차 클러스터링 캐시 사용 ({len(set(primary_labels))}개 대분류)")
    else:
        primary_labels = cluster_primary(embeddings, on_progress=log)
        cache["primary_labels"] = primary_labels.tolist()
        cache["completed"] = ["preprocess", "embed", "cluster_primary"]
        _save_cache(app_key, cache)

    # ④ 2차 클러스터링
    if "cluster_secondary" in completed and cache.get("secondary_labels"):
        secondary = {int(k): np.array(v) for k, v in cache["secondary_labels"].items()}
        total_subs = sum(len(set(v)) for v in secondary.values())
        log(f"2차 클러스터링 캐시 사용 (총 {total_subs}개 소분류)")
    else:
        secondary = cluster_secondary(embeddings, primary_labels, on_progress=log)
        cache["secondary_labels"] = {str(k): v.tolist() for k, v in secondary.items()}
        cache["completed"] = ["preprocess", "embed", "cluster_primary", "cluster_secondary"]
        _save_cache(app_key, cache)

    # ⑤ LLM 라벨링
    def _is_fallback_label(labels):
        """폴백 라벨인지 판별 (카테고리 1, 세부 그룹 1 등)"""
        cats = labels.get("categories", [])
        if not cats:
            return True
        return any(c.get("label", "").startswith("카테고리 ") for c in cats)

    if "label" in completed and cache.get("llm_labels") and not _is_fallback_label(cache["llm_labels"]):
        llm_labels = cache["llm_labels"]
        log("LLM 라벨링 캐시 사용")
    else:
        llm_labels = generate_labels(reviews, embeddings, primary_labels, secondary, on_progress=log)
        # 폴백 라벨은 캐시에 저장하지 않음
        if not _is_fallback_label(llm_labels):
            cache["llm_labels"] = llm_labels
            cache["completed"] = ["preprocess", "embed", "cluster_primary", "cluster_secondary", "label"]
            _save_cache(app_key, cache)
        else:
            log("폴백 라벨 사용됨 — 캐시 미저장 (다음 실행 시 재시도)")

    # ⑥ JSON 조립 (항상 실행)
    result = assemble_json(reviews, primary_labels, secondary, llm_labels, app_name)
    log(f"완료: {len(result['categories'])}개 대분류, {sum(len(c['subcategories']) for c in result['categories'])}개 소분류")

    out_path = f"data/{app_key}_affinity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"저장: {out_path}")

    return result
