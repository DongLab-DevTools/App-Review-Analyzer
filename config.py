"""앱 설정 및 분석 카테고리 정의"""

APPS = {
    "nudgeeap": {
        "name": "넛지EAP",
        "package_id": "com.dain.nudgeeap",
        "app_store_id": 6667097511,
        "app_store_name": "넛지eap-팀-성장을-위한-건강-동기부여",
        "category": "헬스케어/EAP",
    },
    "trost": {
        "name": "트로스트",
        "package_id": "com.humart.trost2",
        "app_store_id": 1034957818,
        "app_store_name": "트로스트-국내-1위-심리상담-명상-asmr",
        "category": "멘탈케어",
        "is_primary": True,
    },
    "cashwalk": {
        "name": "캐시워크",
        "package_id": "com.cashwalk.cashwalk",
        "app_store_id": 1220307907,
        "app_store_name": "캐시워크-돈-버는-만보기",
        "category": "헬스/리워드",
    },
    "genieat": {
        "name": "지니어트",
        "package_id": "com.cashwalk.contents",
        "app_store_id": 1535606430,
        "app_store_name": "지니어트-홈트-다이어트-혈당-기록-만보기-앱",
        "category": "다이어트/헬스",
    },
    "timespread": {
        "name": "타임스프레드",
        "package_id": "com.timespread.Timetable2",
        "app_store_id": 457130897,
        "app_store_name": "타임스프레드-시간표-캘린더-일정관리-돈버는앱",
        "category": "생산성",
    },
    "monyapp": {
        "name": "언니의파우치",
        "package_id": "com.ui.monyapp",
        "app_store_id": 976126416,
        "app_store_name": "언니의파우치",
        "category": "가계부/금융",
    },
    "tving": {
        "name": "티빙",
        "package_id": "net.cj.cjhv.gs.tving",
        "app_store_id": 400101401,
        "category": "OTT",
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
    "헬스케어": ["nudgeeap", "trost", "cashwalk", "genieat"],
    "생활/금융": ["timespread", "monyapp"],
    "OTT": ["tving", "netflix", "wavve", "watcha", "coupangplay", "disneyplus"],
}

# 담당 앱 키
PRIMARY_APP = "trost"

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
