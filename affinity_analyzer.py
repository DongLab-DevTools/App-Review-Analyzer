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
import hashlib
import platform
import threading
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# 주제 신호가 없는 일반 단평/욕설 — 토픽 클러스터를 오염시키므로 제외
# (긍정 단어는 제외하지 않음 — 긍정 피드백 보존)
_GENERIC_NOISE = {
    "쓰레기", "쓰래기", "쓰레기네", "쓰레기임", "구려", "구려요", "구림", "구리네",
    "최악", "별로", "별로임", "별로예요", "별로에요", "별루", "노잼", "재미없어", "재미없음",
    "poor", "bad", "최악임", "개판", "개똥", "비추", "비추천", "그냥그럼",
    "쓰지마", "쓰지마삼", "에휴", "헐", "별로네요", "그저그럼", "노답",
}


def _collapse_obfuscation(text: str) -> str:
    """'개.인.정.보', '개·인·정·보' 처럼 음절 사이 구분기호로 난독화된 단어를 합침.
    (임베더가 한 단어로 인식하도록. 일반 문장의 마침표·중점은 건드리지 않음)"""
    return re.sub(r"(?:[가-힣][.·]){2,}[가-힣]",
                  lambda m: m.group(0).replace(".", "").replace("·", ""), text)


def _strip_noise(text: str) -> str:
    """반복문자/공백/기호 정리 후 핵심 토큰만 (ㅋㅋ, !!! 등 제거)"""
    t = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)            # 자모 단독 반복(ㅋㅋ,ㅠㅠ)
    t = re.sub(r"[^\w가-힣]+", " ", t).strip()         # 기호 제거
    t = re.sub(r"(.)\1{2,}", r"\1", t)                  # 3연속 이상 → 1
    return t


def preprocess(reviews: list[dict]) -> list[dict]:
    """리뷰 정제: 빈 리뷰, 중복, 노이즈, 주제 없는 단평 제거"""
    cleaned = []
    seen = set()
    for r in reviews:
        text = (r.get("content") or "").strip()
        text = _collapse_obfuscation(text)  # 개.인.정.보 → 개인정보
        if len(text) < 4:
            continue
        if text in seen:
            continue
        seen.add(text)
        # 기호만 있는 리뷰 제외
        if len(text) < 5 and not any(c.isalpha() or "가" <= c <= "힣" for c in text):
            continue
        # 주제 신호 없는 일반 단평/욕설 제외
        core = _strip_noise(text)
        if core.replace(" ", "").lower() in _GENERIC_NOISE:
            continue
        if len(core) < 3:  # 정제 후 의미 토큰이 거의 없음
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

# 주제 기반 군집에 강한 검색형(retrieval) 다국어 모델로 교체.
# (ko-sroberta는 STS/NLI 모델이라 짧은 부정 리뷰가 '어조'로 묶이는 경향)
LOCAL_EMBED_MODEL = "intfloat/multilingual-e5-base"
_E5_PREFIX = "query: "  # e5 계열은 입력에 prefix 필요
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
    texts = [_E5_PREFIX + r["text"] for r in reviews]
    model = _get_local_model(on_progress)

    if on_progress:
        on_progress(f"로컬 임베딩 시작... ({len(texts)}건)")

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    if on_progress:
        on_progress(f"로컬 임베딩 완료: {embeddings.shape}")

    if base_embeddings is not None:
        embeddings = np.concatenate([base_embeddings, embeddings])

    if app_key:
        _save_embeddings(app_key, embeddings)

    return embeddings


def embed_incremental(reviews: list[dict], app_key: str, reuse_cache: bool = True,
                      on_progress=None) -> np.ndarray:
    """점진적 임베딩 — 캐시에 있는 텍스트는 재사용하고 신규 텍스트만 인코딩한다.
    임베딩은 텍스트의 순수 함수라, 결과는 전량 재임베딩과 수치적으로 동일하다.
    (모델/하드웨어가 바뀌면 호출 측에서 reuse_cache=False로 전량 재임베딩)
    """
    keys = [_text_key(r["text"]) for r in reviews]

    cached = {}
    if reuse_cache:
        mat = _load_embeddings(app_key)
        old_keys = _load_emb_keys(app_key)
        if mat is not None and old_keys is not None and len(old_keys) == len(mat):
            cached = {k: mat[i] for i, k in enumerate(old_keys)}

    missing_idx = [i for i, k in enumerate(keys) if k not in cached]
    reuse_cnt = len(reviews) - len(missing_idx)
    if on_progress:
        on_progress(f"임베딩: 재사용 {reuse_cnt}건 / 신규 {len(missing_idx)}건")

    new_vecs = {}
    if missing_idx:
        model = _get_local_model(on_progress)
        texts = [_E5_PREFIX + reviews[i]["text"] for i in missing_idx]
        enc = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        for j, i in enumerate(missing_idx):
            new_vecs[keys[i]] = enc[j]

    # 현재 리뷰 순서대로 행렬 재조립 (재사용분 + 신규분)
    matrix = np.vstack([
        (new_vecs[k] if k in new_vecs else cached[k]) for k in keys
    ]).astype(np.float32)

    _save_embeddings(app_key, matrix)
    _save_emb_keys(app_key, keys)  # 현재 키 목록으로 갱신 → 삭제된 리뷰는 자연 제거
    return matrix


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
    """대분류 클러스터링 — 주제가 충분히 갈라지도록 K 입도를 높임"""
    n = len(embeddings)
    # 평균 클러스터 크기를 줄여(주제 세분화) K 범위를 상향
    k_min = max(12, n // 120)
    k_max = min(30, max(k_min + 2, n // 45))
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
# ④-b 경계·충돌 리뷰 LLM 재배정 (정확도 보정)
#   - 카테고리는 클러스터가 '발견'한 것을 그대로 유지(고정 분류 아님)
#   - 임베딩이 '주제'와 '요구·감정'을 못 가르는 누수만 LLM이 재판정
# ────────────────────────────────────────

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")


def _llm_json_call(prompt: str, on_progress=None, model_name: str = "gemini-2.5-flash",
                   max_rounds: int = 3, rpm_wait: float = 30.0):
    """키를 순회하며 1회씩 호출, JSON 파싱 결과(dict) 반환. 전부 실패하면 None.
    한 바퀴가 전부 429(분당 한도)면 잠깐 대기 후 재시도(RPM 백오프)."""
    global _api_keys
    for rnd in range(max_rounds):
        all_quota = True
        for key_idx in range(len(_api_keys)):
            try:
                genai.configure(api_key=_api_keys[key_idx])
                model = genai.GenerativeModel(model_name)
                text = model.generate_content(prompt).text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                return json.loads(text)
            except Exception as e:
                is_quota = "429" in str(e) or "ResourceExhausted" in type(e).__name__
                if not is_quota:
                    all_quota = False
                    if on_progress:
                        on_progress(f"LLM 호출 오류 (키 {key_idx+1}): {type(e).__name__}: {str(e)[:60]}")
                continue
        # 한 바퀴 다 돌았는데 성공 못함: 전부 분당 한도였다면 대기 후 재시도
        if all_quota and rnd < max_rounds - 1:
            if on_progress:
                on_progress(f"전 키 분당 한도 — {int(rpm_wait)}초 대기 후 재시도")
            time.sleep(rpm_wait)
        else:
            break
    # Gemini 소진/실패 → 로컬 Ollama 폴백
    if _ollama_enabled():
        if on_progress:
            on_progress("Gemini 사용 불가 → 로컬 Ollama로 폴백")
        return _ollama_json_call(prompt, on_progress=on_progress)
    return None


# ── 병렬 분류용 thread-safe REST 호출 (genai.configure 전역 상태 회피) ──

_GEMINI_REST = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _gemini_rest_call(prompt, api_key, model_name="gemini-2.5-flash", timeout=120):
    """키를 요청 파라미터로 직접 전달 → 스레드 간 전역 상태 공유 없음.
    반환: ('ok', text) | ('quota', msg) | ('error', msg)"""
    url = _GEMINI_REST.format(model=model_name, key=api_key)
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return ("ok", text)
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return ("quota", "429")
        try:
            detail = e.read().decode()[:200]
        except Exception:
            detail = ""
        if e.code == 400 and "API_KEY_INVALID" in detail:
            return ("badkey", "API_KEY_INVALID")
        return ("error", f"HTTP {e.code}")
    except Exception as e:
        return ("error", str(e)[:80])


# ── 로컬 LLM(Ollama) 폴백 — Gemini 전 키 소진 시에만 사용 ──
#   서버 실행 스크립트(scripts/*/ensure_ollama)가 데몬·모델을 준비함.
#   Ollama가 없거나 꺼져 있으면 조용히 None → 기존 동작(미반영/게이트)으로 폴백.

_OLLAMA_URL = "http://localhost:11434"
_ollama_ok = None       # 데몬 가용성 (런당 캐시)
_ollama_used = False     # 이번 런에서 Ollama가 실제로 쓰였는지 (meta 표시용)
# 작은 로컬 모델은 긴 입력/출력을 다 처리 못함 → Ollama 폴백 시 이 단위로 쪼갬
_OLLAMA_CLASSIFY_CHUNK = 10   # 분류: 한 번에 리뷰 N건
_OLLAMA_LABEL_CHUNK = 10      # 라벨링: 한 번에 클러스터 N개


def _as_int(x):
    """'1', 'id1', 1 등에서 정수 추출 (작은 로컬 모델의 흔들리는 형식 대응). 실패 시 None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    m = re.search(r"-?\d+", str(x))
    return int(m.group()) if m else None


def _ollama_model():
    return os.getenv("OLLAMA_MODEL", "exaone3.5:2.4b")


def _ollama_enabled():
    """USE_OLLAMA != 0 이고 데몬이 떠 있으며 대상 모델이 받아져 있으면 True (런당 1회 캐시)."""
    global _ollama_ok
    if os.getenv("USE_OLLAMA", "1") == "0":
        return False
    if _ollama_ok is None:
        try:
            with urllib.request.urlopen(f"{_OLLAMA_URL}/api/tags", timeout=2) as r:
                names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
            base = _ollama_model().split(":")[0]
            _ollama_ok = any(base in n for n in names)  # 데몬 + 모델 둘 다 있어야 True
        except Exception:
            _ollama_ok = False
    return _ollama_ok


def _ollama_json_call(prompt, on_progress=None, timeout=300):
    """로컬 Ollama로 JSON 응답 생성(format=json 강제). 실패 시 None."""
    global _ollama_used
    # num_ctx: 기본 2048이면 큰 프롬프트(라벨링·분류 배치)가 잘림 → 넉넉히
    try:
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    except ValueError:
        num_ctx = 16384
    body = json.dumps({
        "model": _ollama_model(), "prompt": prompt,
        "stream": False, "format": "json",
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }).encode("utf-8")
    req = urllib.request.Request(f"{_OLLAMA_URL}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        text = (data.get("response") or "").strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        result = json.loads(text)
        _ollama_used = True
        return result
    except Exception as e:
        if on_progress:
            on_progress(f"Ollama 폴백 실패: {type(e).__name__}: {str(e)[:60]}")
        return None


class _KeyPool:
    """키를 라운드로빈으로 나눠주되, 429를 만난 키를 '영구 제외'가 아니라 '쿨다운'시킨다.
    - 분당 한도(RPM): 쿨다운(기본 45초) 후 자동 복귀 → 하루 한도를 끝까지 사용
    - 일일 한도(RPD): 반복 429로 쿨다운이 누적·장기화되어 사실상 제외
    thread-safe."""
    def __init__(self, keys, base_cooldown=45.0):
        self.keys = list(keys)
        self.base_cooldown = base_cooldown
        self.cooldown_until = {}   # key -> monotonic 시각
        self.fails = {}            # key -> 연속 429 횟수
        self._i = 0
        self._lock = threading.Lock()

    def acquire(self):
        """(key, 0.0) 사용 가능 / (None, wait_sec) 전부 쿨다운 중."""
        with self._lock:
            now = time.monotonic()
            live = [k for k in self.keys if self.cooldown_until.get(k, 0.0) <= now]
            if live:
                self._i = (self._i + 1) % len(live)
                return live[self._i], 0.0
            soonest = min(self.cooldown_until.values()) if self.cooldown_until else now
            return None, max(0.0, soonest - now)

    def penalize(self, key):
        """429 → 쿨다운. 연속 실패가 누적되면 길어지고, 3회 이상이면 RPD로 보고 장기 제외."""
        with self._lock:
            self.fails[key] = self.fails.get(key, 0) + 1
            n = self.fails[key]
            dur = self.base_cooldown * n if n < 3 else 86400.0
            self.cooldown_until[key] = time.monotonic() + dur

    def reward(self, key):
        with self._lock:
            self.fails[key] = 0

    def disable(self, key):
        """무효 키(API_KEY_INVALID 등) → 장기 제외."""
        with self._lock:
            self.cooldown_until[key] = time.monotonic() + 86400.0
            self.fails[key] = 99

    def usable_soon(self, within=600.0):
        """곧(within초 내) 다시 쓸 키가 하나라도 있나. 없으면 RPD 소진으로 간주."""
        with self._lock:
            now = time.monotonic()
            return any(self.cooldown_until.get(k, 0.0) - now <= within for k in self.keys)


def _call_batch_rest(prompt, pool, max_total_wait=180.0, max_error_retry=3):
    """풀에서 키를 받아 1배치 Gemini 호출. (Ollama 폴백은 호출부에서 청크 처리)
    - 429: 키 쿨다운 후 다른 키로, 전부 쿨다운이면 짧게 백오프 재시도(RPM 흡수)
    - 무효 키: 장기 제외
    - 모든 키가 장기 쿨다운(RPD)/총 대기 초과/반복 오류 → None
    파싱 결과(dict) | None."""
    waited = 0.0
    err = 0
    while True:
        key, wait = pool.acquire()
        if key is None:
            if not pool.usable_soon() or waited >= max_total_wait:
                return None
            nap = min(wait + 1.0, 30.0)
            time.sleep(nap)
            waited += nap
            continue
        status, payload = _gemini_rest_call(prompt, key)
        if status == "ok":
            pool.reward(key)
            text = payload.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            try:
                return json.loads(text)
            except Exception:
                return None
        if status == "quota":
            pool.penalize(key)
            continue
        if status == "badkey":
            pool.disable(key)  # 무효 키 → 제외하고 다른 키로 (err 미가산)
            continue
        err += 1
        if err >= max_error_retry:
            return None


def _cluster_anchors(reviews, primary_labels, top_n=12):
    """각 클러스터의 '구별되는' 핵심어를 데이터에서 추출 (고정 키워드 아님).
    distinctiveness = (클러스터 내 출현비율) / (전체 출현비율)"""
    n = len(reviews)
    cats = sorted(c for c in set(primary_labels) if c != -1)
    global_df, cluster_df = {}, {c: {} for c in cats}
    cluster_size = {c: 0 for c in cats}
    review_tokens = []
    for i, r in enumerate(reviews):
        toks = {t.lower() for t in _TOKEN_RE.findall(r["text"])}
        toks = {t for t in toks if t not in _GENERIC_NOISE and len(t) >= 2}
        review_tokens.append(toks)
        c = primary_labels[i]
        if c == -1:
            continue
        cluster_size[c] += 1
        for t in toks:
            global_df[t] = global_df.get(t, 0) + 1
            cluster_df[c][t] = cluster_df[c].get(t, 0) + 1
    anchors = {}
    for c in cats:
        sz = max(cluster_size[c], 1)
        scored = []
        for t, df in cluster_df[c].items():
            if df < max(3, sz * 0.04):           # 클러스터 내 최소 빈도
                continue
            score = (df / sz) / ((global_df[t] / n) + 1e-9)
            if score >= 1.8:                      # 이 클러스터에 유의하게 편중된 단어만
                scored.append((score, t))
        scored.sort(reverse=True)
        anchors[c] = [t for _, t in scored[:top_n]]
    return anchors, review_tokens


def reassign_boundary_reviews(reviews, embeddings, primary_labels, on_progress=None,
                              batch_size=25, max_flagged=500):
    """경계(margin↓)·저적합(sim↓)·anchor 충돌 리뷰만 골라 LLM이 올바른 카테고리로 재배정."""
    primary_labels = np.array(primary_labels)
    normed = normalize(embeddings)
    cats = sorted(c for c in set(primary_labels) if c != -1)
    if len(cats) < 2:
        return primary_labels

    cat_index = {c: k for k, c in enumerate(cats)}
    centroids = normalize(np.vstack([normed[primary_labels == c].mean(axis=0) for c in cats]))
    sims = normed @ centroids.T                       # (N, C) 코사인 유사도

    anchors, _ = _cluster_anchors(reviews, primary_labels)
    # 앵커 → 보유 클러스터들. 한국어 교착어 대응: 부분문자열로 매칭
    # (예: 앵커 '배상'이 '배상해', '스팸'이 '스팸온다', '유출'이 '유출됐' 안에 잡히도록)
    anchor_owner = {}
    for c, toks in anchors.items():
        for t in toks:
            if len(t) >= 2:                            # 너무 짧은 앵커는 오탐 위험 → 제외
                anchor_owner.setdefault(t, set()).add(c)
    anchor_terms = list(anchor_owner.items())
    text_lower = [r["text"].lower() for r in reviews]

    reps = {c: select_representatives(reviews, embeddings, np.where(primary_labels == c)[0], 3)
            for c in cats}

    flagged = []                                       # (priority, i, candidate_cats)
    for i in range(len(reviews)):
        cur = primary_labels[i]
        if cur == -1:
            continue
        row = sims[i]
        order = np.argsort(row)[::-1]
        nearest, second = cats[order[0]], cats[order[1]] if len(order) > 1 else cats[order[0]]
        margin = float(row[order[0]] - row[order[1]]) if len(order) > 1 else 1.0
        best_sim = float(row[order[0]])
        tl = text_lower[i]
        conflict = {owner for t, owners in anchor_terms if t in tl for owner in owners if owner != cur}
        is_boundary = margin < 0.06
        is_poorfit = best_sim < 0.30
        if not (conflict or is_boundary or is_poorfit):
            continue
        cands = {cur, nearest, second} | set(sorted(conflict)[:2])
        prio = (0 if conflict else 1 if is_poorfit else 2, margin)
        flagged.append((prio, i, sorted(cands)))

    if not flagged:
        if on_progress:
            on_progress("재배정 대상 없음 — 클러스터 배정 유지")
        return primary_labels

    flagged.sort(key=lambda x: x[0])
    flagged = flagged[:max_flagged]
    if on_progress:
        on_progress(f"경계·충돌 리뷰 {len(flagged)}건 LLM 재배정 시작 (배치 {batch_size})")

    def catalog_for(cids):
        lines = []
        for c in cids:
            akw = ", ".join(anchors.get(c, [])[:8]) or "(특징어 없음)"
            ex = " / ".join(s[:40] for s in reps.get(c, [])[:3])
            lines.append(f"  - id {c}: 핵심어[{akw}] 예시[{ex}]")
        return "\n".join(lines)

    valid = set(cats)
    moved = 0
    for b in range(0, len(flagged), batch_size):
        batch = flagged[b:b + batch_size]
        cset = sorted({c for _, _, cands in batch for c in cands})
        items = [{"i": int(i), "text": reviews[i]["text"][:200], "candidates": [int(c) for c in cands]}
                 for _, i, cands in batch]
        prompt = f"""아래 각 리뷰를 후보 카테고리 중 '주제가 가장 잘 맞는' 하나로 분류하세요.
중요: 요구·감정 표현(보상/환불/탈퇴/욕설)이 아니라 '무엇에 대한 불만인지(주제)'로 판단하세요.
예) 개인정보 유출에 분노하며 보상을 요구하는 리뷰는 '결제'가 아니라 '개인정보 유출' 주제입니다.

카테고리:
{catalog_for(cset)}

각 리뷰는 자신의 candidates 안에서만 고르세요. 확신이 없으면 현재 후보 첫 값을 유지하세요.
JSON만 출력: {{"assign": [{{"i": 리뷰번호, "cat": 선택한_id}}]}}

리뷰:
{json.dumps(items, ensure_ascii=False)}
"""
        res = _llm_json_call(prompt, on_progress=on_progress)
        if not res:
            if on_progress:
                on_progress(f"재배정 배치 {b // batch_size + 1} 실패(키 소진/오류) — 남은 배치 중단")
            break
        for a in res.get("assign", []):
            i, newc = a.get("i"), a.get("cat")
            if isinstance(i, int) and 0 <= i < len(reviews) and newc in valid and primary_labels[i] != newc:
                primary_labels[i] = newc
                moved += 1
        if on_progress:
            on_progress(f"재배정 진행 {min(b + batch_size, len(flagged))}/{len(flagged)} (이동 {moved}건)")

    if on_progress:
        on_progress(f"재배정 완료: {moved}건 카테고리 이동")
    return primary_labels


# ────────────────────────────────────────
# ③-c 전수 LLM 분류 (클러스터=카테고리 '발견', 배정은 전부 LLM)
#   - 모든 리뷰를 '주제' 기준으로 LLM이 분류 → 어휘에 좌우되지 않음
#   - 배치 단위 재개: 쿼터 소진 시 진행분 저장 후 부분 종료, 다음 실행 시 이어서
# ────────────────────────────────────────

def _name_clusters(reviews, embeddings, primary_labels, anchors, on_progress=None):
    """분류 카탈로그용으로 각 클러스터에 짧은 한국어 라벨을 붙인다(LLM 1콜)."""
    cats = sorted(c for c in set(primary_labels) if c != -1)
    data = []
    for c in cats:
        idx = np.where(primary_labels == c)[0]
        reps = select_representatives(reviews, embeddings, idx, 5)
        data.append({"id": int(c), "size": int(len(idx)),
                     "keywords": anchors.get(c, [])[:8],
                     "examples": [r[:50] for r in reps]})
    prompt = f"""아래 각 클러스터의 핵심 주제를 한국어 2~5단어 라벨로 붙여주세요.
JSON만 출력: {{"labels": [{{"id": 0, "label": "..."}}]}}

클러스터:
{json.dumps(data, ensure_ascii=False)}
"""
    res = _llm_json_call(prompt, on_progress=on_progress)
    names = {}
    if res:
        for x in res.get("labels", []):
            if isinstance(x.get("id"), int):
                names[x["id"]] = (x.get("label") or "").strip() or f"카테고리 {x['id'] + 1}"
    for c in cats:
        names.setdefault(c, f"카테고리 {c + 1}")
    return names


def classify_all_reviews(reviews, embeddings, primary_labels, on_progress=None,
                         batch_size=40, cache=None, app_key=None, max_workers=None):
    """모든 리뷰를 LLM이 '주제'로 분류. (primary_labels, 완료여부) 반환.
    - 12개 키로 배치를 병렬 처리(REST, thread-safe)
    - 배치 단위 재개: 완료 배치 집합을 cache에 저장 → 중단돼도 남은 배치만 재실행."""
    primary_labels = np.array(primary_labels)
    cats = sorted(c for c in set(primary_labels) if c != -1)
    if len(cats) < 2:
        return primary_labels, True

    anchors, _ = _cluster_anchors(reviews, primary_labels)
    reps = {c: select_representatives(reviews, embeddings, np.where(primary_labels == c)[0], 3)
            for c in cats}
    names = _name_clusters(reviews, embeddings, primary_labels, anchors, on_progress=on_progress)
    catalog = "\n".join(
        f'  - id {c}: {names[c]} | 핵심어[{", ".join(anchors.get(c, [])[:7])}] '
        f'예시[{" / ".join(s[:35] for s in reps[c][:2])}]'
        for c in cats)
    valid = set(cats)
    n = len(reviews)
    total_batches = (n + batch_size - 1) // batch_size

    # 재개: 완료 배치 집합 (구버전 정수 마커도 흡수)
    assign = {int(k): v for k, v in (cache.get("classify_assign") or {}).items()} if cache else {}
    done = set(cache.get("classify_done_set") or []) if cache else set()
    old = cache.get("classify_done_batches") if cache else None
    if isinstance(old, int) and old > 0:
        done |= set(range(old))
    pending = [bi for bi in range(total_batches) if bi not in done]

    # 동시성: RPM(분당 한도) 부담을 줄이려 기본을 보수적으로. env CLASSIFY_WORKERS로 조절.
    env_workers = os.getenv("CLASSIFY_WORKERS")
    default_workers = min(len(_api_keys), 5) or 1
    workers = max_workers or (int(env_workers) if env_workers and env_workers.isdigit() else default_workers)
    if on_progress:
        verb = "재개" if done else "시작"
        on_progress(f"전수 LLM 분류 {verb}: {n}건 / {total_batches}배치, 남은 {len(pending)}배치 "
                    f"(병렬 {workers}, 카테고리 {len(cats)}개)")
    if not pending:
        for i, c in assign.items():
            if 0 <= i < n:
                primary_labels[i] = c
        return primary_labels, True

    def build_prompt(idxs):
        items = [{"i": i, "text": reviews[i]["text"][:200]} for i in idxs]
        return f"""각 리뷰를 아래 카테고리 중 '주제가 가장 잘 맞는' 하나로 분류하세요.
중요: 요구·감정 표현(보상/환불/탈퇴/욕설)이 아니라 '무엇에 대한 불만·내용인지(주제)'로 판단하세요.
예) 개인정보 유출에 분노하며 보상·탈퇴를 외치는 리뷰는 '결제/구독'이 아니라 '개인정보 유출' 주제입니다.

카테고리:
{catalog}

JSON만 출력: {{"assign": [{{"i": 리뷰번호, "cat": 카테고리id}}]}}

리뷰:
{json.dumps(items, ensure_ascii=False)}
"""

    pool = _KeyPool(_api_keys)

    def work(bi):
        s, e = bi * batch_size, min((bi + 1) * batch_size, n)
        idxs = list(range(s, e))
        # 1) Gemini 우선 (전체 배치 한 번에)
        res = _call_batch_rest(build_prompt(idxs), pool)
        if res is not None:
            return bi, res
        # 2) Gemini 불가 → Ollama로 '작은 청크'로 분할 (작은 모델은 긴 목록을 다 안 채움)
        if _ollama_enabled():
            assigns, ok = [], False
            for c in range(0, len(idxs), _OLLAMA_CLASSIFY_CHUNK):
                sub = idxs[c:c + _OLLAMA_CLASSIFY_CHUNK]
                r = _ollama_json_call(build_prompt(sub))
                if r and isinstance(r.get("assign"), list):
                    ok = True
                    assigns.extend(r["assign"])
            return bi, ({"assign": assigns} if ok else None)
        return bi, None

    processed = [0]

    def run_round(batch_ids):
        """배치들을 병렬 실행, 실패한(파싱/일시 오류) 배치 id 리스트 반환."""
        missing = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(work, bi): bi for bi in batch_ids}
            for fut in as_completed(futs):
                bi, res = fut.result()
                processed[0] += 1
                if res is None:
                    missing.append(bi)
                    continue
                s, e = bi * batch_size, min((bi + 1) * batch_size, n)
                for a in res.get("assign", []):
                    # 작은 로컬 모델은 'id1'·'1' 같은 형식을 주기도 함 → 숫자 추출로 관대하게 파싱
                    i, c = _as_int(a.get("i")), _as_int(a.get("cat"))
                    if i is not None and s <= i < e and c in valid:
                        assign[i] = c
                done.add(bi)
                if cache is not None:
                    cache["classify_assign"] = {str(k): int(v) for k, v in assign.items()}
                    cache["classify_done_set"] = sorted(done)
                    if app_key and processed[0] % 8 == 0:
                        _save_cache(app_key, cache)
                if on_progress and processed[0] % 10 == 0:
                    on_progress(f"분류 진행 {processed[0]}배치 누적 ({len(assign)}건)")
        return missing

    # 1차 + 실패 배치 재시도(최대 3라운드). 모든 키가 RPD로 장기 소진되면 중단(다음 실행 재개).
    to_do = pending
    rnd = 0
    while to_do and rnd < 3 and pool.usable_soon():
        rnd += 1
        if rnd > 1 and on_progress:
            on_progress(f"실패 배치 {len(to_do)}개 재시도 (라운드 {rnd})")
        to_do = run_round(to_do)

    if cache is not None and app_key:
        cache["classify_assign"] = {str(k): int(v) for k, v in assign.items()}
        cache["classify_done_set"] = sorted(done)
        _save_cache(app_key, cache)

    completed = len(done) >= total_batches
    for i, c in assign.items():
        if 0 <= i < n:
            primary_labels[i] = c
    if on_progress:
        on_progress(f"분류 적용: {len(assign)}/{n}건 ({'완료' if completed else '부분 — 다음 실행 시 이어서 분류'})")
    return primary_labels, completed


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

    # 중심에 가까운 대표만 사용 (먼 것을 섞으면 라벨이 흐려짐)
    closest = distances.argsort()[:n_samples]
    return [reviews[indices[i]]["text"] for i in closest]


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

    def _mk_prompt(cd):
        return f"""아래는 앱 리뷰를 클러스터링한 결과입니다.
각 클러스터의 대표 리뷰를 보고, 대분류 라벨과 소분류 라벨을 한국어로 생성해주세요.

규칙:
1. 같은 cat_id를 가진 클러스터들은 하나의 대분류에 속합니다
2. 대분류 라벨: 핵심 주제를 2~4단어로 (예: "동일가구 인증 정책", "광고 과다")
3. 소분류 라벨: 세부 이슈를 2~5단어로 (예: "결제자 본인도 인증 반복")
4. 대분류 description: 한 줄 설명
5. sentiment: 평균 별점 1~2점이면 negative, 3점이면 mixed, 4~5점이면 positive
6. **중요**: 라벨은 대표 리뷰의 '다수'를 대표해야 합니다. 소수의 리뷰에만 해당하는 구체 주제로 단정하지 마세요.
7. **중요**: 대표 리뷰들이 특정 주제 없이 일반적 불만/욕설/단평("쓰레기", "별로", "최악" 등) 위주로 섞여 있으면, 억지로 구체 주제를 붙이지 말고 라벨을 "기타 불만"으로 하세요.

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
{json.dumps(cd, ensure_ascii=False, indent=2)}
"""

    prompt = _mk_prompt(cluster_data)

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
                es = str(e)
                is_quota = "429" in es or "ResourceExhausted" in type(e).__name__
                is_bad_key = "API_KEY_INVALID" in es or "API key not valid" in es
                if is_quota:
                    if key_idx not in succeeded_llm_keys:
                        dead_llm_keys.add(key_idx)
                        if on_progress:
                            on_progress(f"LLM 키 {key_idx+1} 일일 한도 소진 → 제외")
                    continue
                if is_bad_key:
                    # 무효 키 → 영구 제외 (재시도/대기하지 않음)
                    dead_llm_keys.add(key_idx)
                    if on_progress:
                        on_progress(f"LLM 키 {key_idx+1} 무효(API_KEY_INVALID) → 제외")
                    continue
                # 기타 일시 오류(JSON 파싱 등) → 다음 키로 (라운드를 끊지 않음)
                if on_progress:
                    on_progress(f"LLM 오류 (키 {key_idx+1}): {type(e).__name__}: {es[:80]}")
                continue

        # 살아있는 키가 있으면 TPM 초기화 대기
        alive = len(_api_keys) - len(dead_llm_keys)
        if alive == 0:
            break
        if round_num < 2:
            if on_progress:
                on_progress(f"LLM TPM 대기 중... (60초)")
            time.sleep(61)

    # Gemini 소진 → 로컬 Ollama 폴백 (작은 모델은 한 번에 다 못하므로 cat_id 단위로 청크)
    if _ollama_enabled():
        if on_progress:
            on_progress("Gemini 라벨링 사용 불가 → 로컬 Ollama로 폴백(청크 처리)")
        by_cat = {}
        for cd in cluster_data:
            by_cat.setdefault(cd["cat_id"], []).append(cd)
        cat_ids = list(by_cat.keys())
        merged = []
        for c0 in range(0, len(cat_ids), _OLLAMA_LABEL_CHUNK):
            chunk_ids = cat_ids[c0:c0 + _OLLAMA_LABEL_CHUNK]
            chunk_cd = [cd for cid in chunk_ids for cd in by_cat[cid]]
            r = _ollama_json_call(_mk_prompt(chunk_cd), on_progress=on_progress)
            if r and isinstance(r.get("categories"), list):
                merged.extend(r["categories"])
            if on_progress:
                on_progress(f"라벨링(로컬) {min(c0 + _OLLAMA_LABEL_CHUNK, len(cat_ids))}/{len(cat_ids)} 대분류")
        if merged:
            return {"categories": merged}

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
# ⑥-b 중복 카테고리 병합 (개인정보 유출 사태 ≈ 미흡 대응 등)
# ────────────────────────────────────────

def compute_merge_groups(result, on_progress=None):
    """의미가 같은 대분류만 묶을 그룹을 LLM으로 산출. 라벨 문자열 기준."""
    cats = result.get("categories", [])
    if len(cats) <= 2:
        return []
    catalog = []
    for c in cats:
        cnt = sum(len(s["items"]) for s in c["subcategories"])
        catalog.append({
            "label": c["label"],
            "desc": (c.get("description") or "")[:60],
            "count": cnt,
            "subs": [s["label"] for s in c["subcategories"][:5]],
        })
    prompt = f"""아래는 리뷰 분류의 대분류 목록입니다. '동일한 구체적 사건/이슈를 가리키는 카테고리'만 신중하게 묶어주세요.
합쳐야 하는 경우(좁게):
- 라벨의 핵심 주제가 사실상 같은 중복
- 같은 '구체적 사건'의 서로 다른 측면 (예: '개인정보 유출' + '개인정보 유출 대응 미흡' = 같은 유출 사고 → 병합)
절대 합치면 안 되는 경우:
- 서로 다른 기능 영역/주제 (예: '앱 기능' vs '영상 재생', '결제' vs '광고', '로그인' vs '업데이트')
- 큰 상위 범주로 여러 주제를 뭉뚱그리는 병합 (grab-bag 금지)
애매하면 합치지 마세요. 대부분의 카테고리는 그대로 두는 게 맞습니다.
- 병합 그룹마다 통합 라벨(가장 명확한 2~5단어)을 제시하세요.
JSON만 출력: {{"groups": [{{"members": ["라벨1","라벨2"], "label": "통합 라벨"}}]}}
병합할 게 없으면 {{"groups": []}}.

대분류:
{json.dumps(catalog, ensure_ascii=False, indent=2)}
"""
    res = _llm_json_call(prompt, on_progress=on_progress)
    if res is None:
        return None  # LLM 호출 실패(한도 등) — '병합 없음([])'과 구분하여 재시도 유도
    groups = []
    for g in res.get("groups", []):
        members = [m for m in g.get("members", []) if isinstance(m, str)]
        if len(members) >= 2:
            groups.append({"members": members, "label": g.get("label") or members[0]})
    return groups


def _apply_merge_groups(result, groups):
    """compute_merge_groups 결과(라벨 기준)를 조립된 JSON에 적용."""
    if not groups:
        return result
    cats = result.get("categories", [])
    by_label = {}
    for c in cats:
        by_label.setdefault(c["label"], c)
    used, merged = set(), []
    for g in groups:
        members = [by_label[m] for m in g["members"] if m in by_label and m not in used]
        if len(members) < 2:
            continue
        subs = []
        for m in members:
            subs.extend(m["subcategories"])
            used.add(m["label"])
        subs.sort(key=lambda s: len(s["items"]), reverse=True)
        merged.append({
            "label": g["label"],
            "description": members[0].get("description", ""),
            "subcategories": subs,
        })
    for c in cats:
        if c["label"] not in used:
            merged.append(c)
    merged.sort(key=lambda c: sum(len(s["items"]) for s in c["subcategories"]), reverse=True)
    result["categories"] = merged
    return result


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


def _emb_keys_path(app_key):
    """임베딩 행렬 각 행에 대응하는 텍스트 키 목록(점진 임베딩용)."""
    return f"data/{app_key}_affinity_emb_keys.json"


def _load_emb_keys(app_key):
    path = _emb_keys_path(app_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_emb_keys(app_key, keys):
    with open(_emb_keys_path(app_key), "w", encoding="utf-8") as f:
        json.dump(keys, f)


def _text_key(text):
    """리뷰 텍스트의 안정적 키. 텍스트가 바뀌면 키도 바뀌어 자동 재임베딩된다.
    (reviewId가 아니라 '실제 인코딩 입력' 기준이라, 사용자가 리뷰를 수정해도 안전)"""
    return hashlib.md5((_E5_PREFIX + text).encode("utf-8")).hexdigest()


def _host_id():
    """실행 환경 식별자. 하드웨어/OS가 바뀌면(맥북→ARM 서버) 부동소수점 차이가
    생길 수 있으므로 1회 전량 재임베딩을 트리거하는 데 쓴다."""
    return f"{platform.system()}-{platform.machine()}"


def _review_signature(raw_reviews):
    """리뷰 집합의 시그니처(정렬된 reviewId 해시). 리뷰가 추가/변경되면 값이 바뀐다."""
    ids = sorted(str(r.get("reviewId") or (r.get("content") or "")[:60]) for r in raw_reviews)
    return hashlib.md5("\n".join(ids).encode("utf-8")).hexdigest()


def _should_invalidate(cache, current_sig):
    """캐시 무효화 판단: 리뷰 집합 시그니처가 다르면(=리뷰가 갱신됨) True."""
    if not cache:
        return True
    return cache.get("review_sig") != current_sig


def _mark(cache, stage):
    """완료 단계 마커를 append (덮어쓰지 않음). 부분 완료 단계는 호출하지 않는다."""
    done = cache.get("completed", [])
    if stage not in done:
        cache["completed"] = done + [stage]


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

    # 런 단위 Ollama 상태 리셋 (가용성 재확인 + 사용여부 추적)
    global _ollama_ok, _ollama_used
    _ollama_ok = None
    _ollama_used = False

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
    #  - 모델/하드웨어 변경: 임베딩 공간 자체가 달라짐 → 임베딩 포함 전량 재계산
    #  - 리뷰만 변경: 임베딩은 '점진'(신규만), 클러스터링부터만 재계산
    current_sig = _review_signature(raw_reviews)
    cache = _load_cache(app_key)
    host_id = _host_id()
    model_changed = bool(cache and cache.get("embed_model") != LOCAL_EMBED_MODEL)
    host_changed = bool(cache and cache.get("embed_host") not in (None, host_id))
    sig_changed = _should_invalidate(cache, current_sig)

    reuse_embeddings = True
    if not cache:
        cache = {"completed": []}
        reuse_embeddings = False
    elif model_changed or host_changed:
        reason = "임베딩 모델 변경" if model_changed else f"실행 환경 변경 ({cache.get('embed_host')} → {host_id})"
        log(f"{reason} 감지 → 임베딩 포함 전체 재분석")
        for p in (_embeddings_path(app_key), _emb_keys_path(app_key)):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        cache = {"completed": []}
        reuse_embeddings = False
    elif sig_changed:
        log(f"리뷰 변경 (기존 {cache.get('review_count', 0)}건 → {len(raw_reviews)}건) "
            f"→ 임베딩 점진 갱신, 클러스터링부터 재계산")
        cache = {"completed": []}  # 분석 단계 캐시는 비움 (임베딩 .npy/keys 파일은 보존 → 점진 재사용)
        reuse_embeddings = True
    else:
        log(f"동일 리뷰·모델·환경 — 캐시 재사용: {', '.join(cache.get('completed', [])) or '없음'} 완료됨")

    cache["review_count"] = len(raw_reviews)
    cache["review_sig"] = current_sig
    cache["embed_model"] = LOCAL_EMBED_MODEL
    cache["embed_host"] = host_id
    _save_cache(app_key, cache)  # 메타(특히 embed_host) 즉시 영속화 — 하드웨어 가드 신뢰성

    # 단계 순서가 바뀌었으므로(전수 분류 삽입) completed를 '유효 프리픽스'로 정규화.
    # 기존 캐시에 classify가 없으면 그 지점부터 자동 재계산된다.
    STAGE_ORDER = ["preprocess", "embed", "cluster_primary", "classify",
                   "cluster_secondary", "label", "merge"]
    _done = set(cache.get("completed", []))
    completed = []
    for _s in STAGE_ORDER:
        if _s in _done:
            completed.append(_s)
        else:
            break

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

    # ② 임베딩 (점진적 — 캐시에 있는 텍스트는 재사용, 신규 텍스트만 인코딩)
    embeddings = None
    if "embed" in completed:
        embeddings = _load_embeddings(app_key)
        if embeddings is not None and len(embeddings) == len(reviews):
            log(f"임베딩 캐시 사용 ({embeddings.shape})")
        else:
            embeddings = None

    if embeddings is None:
        embeddings = embed_incremental(reviews, app_key, reuse_cache=reuse_embeddings, on_progress=log)
        log(f"임베딩 완료: {embeddings.shape}")
        cache["completed"] = ["preprocess", "embed"]
        _save_cache(app_key, cache)

    # ③ 1차 클러스터링
    if "cluster_primary" in completed and cache.get("primary_labels"):
        primary_labels = np.array(cache["primary_labels"])
        log(f"1차 클러스터링 캐시 사용 ({len(set(primary_labels))}개 대분류)")
    else:
        primary_labels = cluster_primary(embeddings, on_progress=log)
        cache["primary_labels"] = primary_labels.tolist()
        cache["completed"] = ["preprocess", "embed", "cluster_primary"]
        _save_cache(app_key, cache)

    # ③-b 전수 LLM 분류 (클러스터=카테고리 발견, 배정은 전부 LLM)
    if "classify" in completed and cache.get("primary_labels_final"):
        primary_labels = np.array(cache["primary_labels_final"])
        log("LLM 분류 캐시 사용")
    else:
        try:
            primary_labels, classify_done = classify_all_reviews(
                reviews, embeddings, primary_labels, on_progress=log, cache=cache, app_key=app_key)
            if classify_done:
                cache["primary_labels_final"] = primary_labels.tolist()
                cache.pop("classify_assign", None)
                cache.pop("classify_done_batches", None)
                cache.pop("classify_done_set", None)
                _mark(cache, "classify")
                _save_cache(app_key, cache)
            else:
                log("LLM 분류 부분 완료 — 부분 결과로 진행, 다음 실행 시 이어서 분류")
        except Exception as e:
            log(f"LLM 분류 건너뜀: {type(e).__name__}: {str(e)[:80]} (클러스터 배정 유지)")

    # ④ 2차 클러스터링
    if "cluster_secondary" in completed and cache.get("secondary_labels"):
        secondary = {int(k): np.array(v) for k, v in cache["secondary_labels"].items()}
        total_subs = sum(len(set(v)) for v in secondary.values())
        log(f"2차 클러스터링 캐시 사용 (총 {total_subs}개 소분류)")
    else:
        secondary = cluster_secondary(embeddings, primary_labels, on_progress=log)
        cache["secondary_labels"] = {str(k): v.tolist() for k, v in secondary.items()}
        _mark(cache, "cluster_secondary")
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
            _mark(cache, "label")
            _save_cache(app_key, cache)
        else:
            log("폴백 라벨 사용됨 — 캐시 미저장 (다음 실행 시 재시도)")

    # ⑥ JSON 조립 (항상 실행)
    result = assemble_json(reviews, primary_labels, secondary, llm_labels, app_name)
    log(f"조립: {len(result['categories'])}개 대분류, {sum(len(c['subcategories']) for c in result['categories'])}개 소분류")

    labels_fallback = _is_fallback_label(llm_labels)
    merge_complete = True

    # ⑥-b 중복 카테고리 병합 (폴백 라벨이면 생략)
    if not labels_fallback:
        if "merge" in completed and cache.get("merge_groups") is not None:
            result = _apply_merge_groups(result, cache["merge_groups"])
            log(f"카테고리 병합 캐시 사용 ({len(cache['merge_groups'])}그룹)")
        else:
            groups = compute_merge_groups(result, on_progress=log)
            if groups is None:
                # LLM 호출 실패(한도 등) → 마킹/캐시하지 않음 → 다음 실행 시 재시도
                merge_complete = False
                log("카테고리 병합 보류 — LLM 호출 실패(한도 추정), 다음 실행 시 재시도")
            else:
                cache["merge_groups"] = groups
                _mark(cache, "merge")
                _save_cache(app_key, cache)
                result = _apply_merge_groups(result, groups)
                log(f"카테고리 병합: {len(groups)}그룹 통합 → {len(result['categories'])}개 대분류")
    else:
        merge_complete = False

    # 실행 상태를 메타에 노출 (자동화 현황 페이지에서 '한도로 부분 완료' 표시용)
    classify_complete = "classify" in cache.get("completed", [])
    result["meta"]["classify_complete"] = classify_complete
    result["meta"]["labels_fallback"] = labels_fallback
    result["meta"]["merge_complete"] = merge_complete
    result["meta"]["fully_complete"] = classify_complete and not labels_fallback and merge_complete
    result["meta"]["review_sig"] = current_sig  # 게시본 기준 staleness 판정용
    result["meta"]["llm_backend"] = "ollama" if _ollama_used else "gemini"

    # 대시보드 반영 게이트: '쓸 수 있는' 분석만 파일에 기록한다.
    # (분류 미완료/라벨 폴백이면 직전 정상 결과를 그대로 두고 덮어쓰지 않음 → 부분 결과가
    #  대시보드를 망가뜨리지 않음. 진행분은 캐시에 남아 다음 실행 때 이어서 완료됨.)
    publishable = classify_complete and not labels_fallback
    result["meta"]["published"] = publishable

    out_path = f"data/{app_key}_affinity.json"
    if publishable:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log(f"저장: {out_path}" + ("" if merge_complete else " (병합은 다음 실행 시 정리)"))
    else:
        reason = "분류 미완료" if not classify_complete else "라벨 폴백"
        if os.path.exists(out_path):
            log(f"⚠ 대시보드 미반영({reason}) — 기존 정상 분석 유지, 다음 실행 시 이어서 완료")
        else:
            log(f"⚠ 대시보드 미반영({reason}) — 아직 완료된 분석 없음(부분 결과는 저장 안 함)")

    return result
