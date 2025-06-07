
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class RAGSystem:
    def __init__(self,
                 rag_builder,
                 base_model_path: str = "/home/work/.deep_learning/skkullm/skku_sllm/models/base/Llama-3-Open-Ko-8B",
                 model_path: str = "/home/work/.deep_learning/skkullm/skku_sllm/models/finetuned/skku-llm-20250511/checkpoint-351"):
        """
        RAG 시스템 클래스
        
        Args:
            rag_builder: RAGBuilder 인스턴스
            base_model_path: 기본 모델 경로
            model_path: 파인튜닝된 모델 경로
        """
        self.rag_builder = rag_builder
        
        # 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # 모델을 평가 모드로 설정
        self.model.eval()
    
    def format_prompt(self, query: str, context: str) -> str:
        """프롬프트 포맷팅"""
        return f"""### 지시문:
다음은 성균관대학교에 대한 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 답변해주세요:
{query}

### 응답:
"""
    
    def generate_response(self, query: str, n_results: int = 3) -> str:
        """RAG 기반 응답 생성"""
        # 1. 관련 문서 검색
        search_results = self.rag_builder.test_search(query, n_results=n_results)
        
        # 2. 검색된 문서들을 하나의 컨텍스트로 결합
        context = "\n\n".join([
            f"제목: {metadata['title']}\n내용: {doc}"
            for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0])
        ])
        
        # 3. 프롬프트 생성
        prompt = self.format_prompt(query, context)
        
        # 4. 모델에 입력
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 5. 응답 생성
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_beams=2,
                early_stopping=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 6. 응답 디코딩
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 7. 프롬프트 제거
        response = response.replace(prompt, "").strip()
        
        return response