from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
import sys

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 모델 저장 경로
MODEL_PATH = "/home/work/.deep_learning/skkullm/skku_sllm/models/base/Llama-3-Open-Ko-8B"
MODEL_NAME = "beomi/Llama-3-Open-Ko-8B"

try:
    # 저장 디렉토리 생성
    logger.info(f"저장 디렉토리 생성 중: {MODEL_PATH}")
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # 모델과 토크나이저 다운로드
    logger.info("모델 다운로드 시작...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_PATH,
        resume_download=True
    )
    logger.info("토크나이저 다운로드 시작...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=MODEL_PATH,
        resume_download=True
    )
    
    # 모델 저장
    logger.info("모델 저장 중...")
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    logger.info(f"모델이 성공적으로 {MODEL_PATH}에 저장되었습니다.")
    
except Exception as e:
    logger.error(f"오류 발생: {str(e)}")
    raise 