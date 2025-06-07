# 성균관대학교 특화 LLM (SKKU-LLM)

성균관대학교 정보와 문화에 특화된 대규모 언어 모델입니다. 학생들에게 학교 관련 정보를 제공하고, 친근한 대화를 나눌 수 있도록 개발되었습니다.

## 프로젝트 구조

```
open-ko-llm/
├── data/
│   ├── raw/              # 원본 데이터셋 저장
│   └── processed/        # 전처리된 데이터셋
├── models/
│   ├── base/             # 기본 모델 저장 (Llama-3-Open-Ko-8B)
│   └── finetuned/        # 파인튜닝된 모델 저장
├── src/
│   ├── data_processing/  # 데이터 전처리 스크립트
│   ├── training/         # 훈련 관련 스크립트
│   ├── evaluation/       # 성능 평가 스크립트
│   └── utils/            # 유틸리티 함수
├── configs/              # 훈련 설정 파일
├── logs/                 # 훈련 로그
├── notebooks/            # 실험용 주피터 노트북
├── scripts/              # 실행 스크립트
├── results/              # 평가 결과 저장
├── requirements.txt      # 의존성 패키지 목록
└── README.md             # 프로젝트 설명
```

## 환경 설정

본 프로젝트는 CUDA 지원 환경에서 실행됩니다. RTX 3090 GPU 2개가 있는 환경에 최적화되어 있습니다.

### 의존성 설치

```bash
pip install -r requirements.txt
```

## 데이터 수집 및 전처리

### 데이터 크롤링

성균관대학교 공식 홈페이지(www.skku.edu)에서 데이터를 수집합니다.

#### 전체 사이트 크롤링
DFS 알고리즘을 사용하여 홈페이지 전체를 크롤링합니다.

```bash
# DFS 알고리즘을 사용한 기본 크롤링
python scripts/run_crawler.py --mode dfs --max-depth 3

# 같은 카테고리 내 페이지만 크롤링 (URL 경로 기준)
python scripts/run_crawler.py --mode dfs --start-url https://www.skku.edu/skku/about/ --category-limit

# 특정 URL 패턴 제외
python scripts/run_crawler.py --mode dfs --forbidden-patterns "/login" "/search" "/member" "/eng"
```

#### 메뉴 단위 크롤링
특정 메뉴나 카테고리에 속한 페이지만 크롤링합니다.

```bash
# 총장 인사말 메뉴 크롤링
python scripts/run_crawler.py --mode menu --menu-url https://www.skku.edu/skku/about/pr/greeting.do

# 연구정보 메뉴 크롤링 (깊이 제한)
python scripts/run_crawler.py --mode menu --menu-url https://www.skku.edu/skku/research/research_info/ --max-depth 2
```

#### 특정 페이지 크롤링
특정 페이지를 시작점으로 하여 링크를 따라가며 크롤링합니다.

```bash
# 총장 인사말 페이지에서 시작하여 크롤링
python scripts/run_crawler.py --mode page --page-url https://www.skku.edu/skku/about/pr/greeting.do

# 링크 깊이 3까지만 크롤링
python scripts/run_crawler.py --mode page --page-url https://www.skku.edu/skku/about/pr/greeting.do --max-depth 3

# 특정 패턴을 가진 URL 제외
python scripts/run_crawler.py --mode page --page-url https://www.skku.edu/skku/about/pr/greeting.do --forbidden-patterns "/login" "/member" "/eng" "/sitemap"
```

#### 특정 섹션 크롤링

```bash
# 공지사항만 크롤링
python scripts/run_crawler.py --mode notices

# 모든 정보 크롤링
python scripts/run_crawler.py --mode all
```

### 데이터 전처리

수집된 데이터를 학습에 사용할 수 있도록 전처리합니다.

```bash
python scripts/run_preprocess.py
```

### 데이터 파이프라인 실행

크롤링과 전처리를 한 번에 실행할 수 있습니다.

```bash
python scripts/run_data_pipeline.py

# 크롤링은 건너뛰고 전처리만 실행
python scripts/run_data_pipeline.py --skip-crawl

# 전처리는 건너뛰고 크롤링만 실행
python scripts/run_data_pipeline.py --skip-preprocess
```

데이터 파이프라인을 통해 생성된 파일:
- 원시 데이터: `data/raw/skku_data/*.jsonl`
- 전처리 데이터: `data/processed/skku_processed_*.json`
- 학습용 데이터: `data/processed/skku_training_*.json`

## 모델 훈련

### 데이터셋 준비

학습 데이터는 `/data/processed/skku_qa_instruction.jsonl` 형식으로 준비되어 있습니다. 이 데이터는 성균관대학교 관련 질문-답변 쌍을 포함하고 있습니다.

### 파인튜닝 실행

```bash
bash scripts/finetune.sh
```

이 스크립트는 LoRA 기법을 사용하여 기본 모델을 파인튜닝합니다. 훈련된 모델은 `/models/finetuned/` 디렉토리에 저장됩니다.

## 모델 사용

### 단일 질문 모드

```bash
python scripts/inference.py --model_path /path/to/finetuned/model --prompt "성균관대학교 도서관은 어디에 있나요?"
```

### 대화형 모드

```bash
python scripts/inference.py --model_path /path/to/finetuned/model --interactive
```

## 허깅페이스 업로드

훈련된 모델을 허깅페이스에 업로드하려면:

```bash
python scripts/upload_to_hub.py --model_path /path/to/finetuned/model --repo_name skku/skku-llm
```

## 성능 평가

모델의 정확도, 응답 품질 등은 평가 스크립트를 통해 측정할 수 있습니다:

```bash
python src/evaluation/evaluate.py --model_path /path/to/finetuned/model
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
