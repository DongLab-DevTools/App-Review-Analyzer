"""앱 설정 및 분석 카테고리 정의"""

APPS = {
    "tving": {
        "name": "티빙",
        "package_id": "net.cj.cjhv.gs.tving",
        "app_store_id": 400101401,
        "category": "OTT",
        "is_primary": True,
    },
    "netflix": {
        "name": "넷플릭스",
        "package_id": "com.netflix.mediaclient",
        "app_store_id": 363590051,
        "category": "OTT",
    },
    "wavve": {
        "name": "웨이브",
        "package_id": "kr.co.captv.pooqV2",
        "app_store_id": 987782077,
        "category": "OTT",
    },
    "watcha": {
        "name": "왓챠",
        "package_id": "com.frograms.wplay",
        "app_store_id": 1096493180,
        "category": "OTT",
    },
    "coupangplay": {
        "name": "쿠팡플레이",
        "package_id": "com.coupang.mobile.play",
        "app_store_id": 1536885649,
        "category": "OTT",
    },
    "disneyplus": {
        "name": "디즈니 플러스",
        "package_id": "com.disney.disneyplus",
        "app_store_id": 1446075923,
        "category": "OTT",
    },
}

# 카테고리 그룹 (사이드바 섹션용)
CATEGORY_GROUPS = {
    "OTT": ["tving", "netflix", "wavve", "watcha", "coupangplay", "disneyplus"],
}

# 담당 앱 키
PRIMARY_APP = "tving"

ANALYSIS_CATEGORIES = {
    "positive": [
        "UI 편리성",
        "콘텐츠 만족",
        "기능 만족",
        "가격 만족",
        "고객 응대 만족",
        "기타 칭찬",
    ],
    "negative": [
        "앱 안정성(크래시/버그)",
        "UI/UX 불편",
        "기능 부족/불만",
        "요금/결제/구독",
        "광고 관련 불만",
        "로그인/계정 문제",
        "데이터 손실/동기화",
        "배터리/성능 문제",
        "고객 응대 불만",
        "기타 불만",
    ],
}

MAX_REVIEWS_GP = 2000
MAX_REVIEWS_AS = 500
GEMINI_BATCH_SIZE = 50
