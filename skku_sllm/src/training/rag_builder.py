import os
import glob
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class RAGBuilder:
    def __init__(self, 
                 data_dir: str = "/home/work/.deep_learning/skkullm/skku_sllm/data/RAG",
                 model_name: str = "jhgan/ko-sroberta-multitask",
                 collection_name: str = "skku_rag"):
        """
        RAG 시스템 구축을 위한 클래스
        
        Args:
            data_dir: 텍스트 파일이 있는 디렉토리
            model_name: 임베딩 모델 이름
            collection_name: Chroma DB 컬렉션 이름
        """
        self.data_dir = data_dir
        self.model = SentenceTransformer(model_name)
        self.collection_name = collection_name
        
        # Chroma DB 클라이언트 초기화 (persist_directory 설정)
        self.client = chromadb.PersistentClient(
            path=os.path.join(data_dir, "chroma_db")
        )
        
        # 컬렉션 생성 또는 가져오기
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def read_text_files(self) -> List[Dict]:
        """텍스트 파일들을 읽어서 문서 리스트로 반환"""
        documents = []
        
        # txt 파일들 찾기
        txt_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        
        for file_path in txt_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 파일명에서 제목 추출 (확장자 제외)
                title = os.path.splitext(os.path.basename(file_path))[0]
                
                documents.append({
                    "title": title,
                    "content": content,
                    "file_path": file_path
                })
        
        return documents
    
    def create_embeddings(self, documents: List[Dict]):
        """문서들을 임베딩하고 벡터 DB에 저장"""
        for doc in documents:
            # 문서 내용 임베딩
            embedding = self.model.encode(doc["content"])
            
            # 메타데이터 준비
            metadata = {
                "title": doc["title"],
                "file_path": doc["file_path"]
            }
            
            # Chroma DB에 저장
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[doc["content"]],
                metadatas=[metadata],
                ids=[f"doc_{len(self.collection.get()['ids']) + 1}"]
            )
    
    def build(self):
        """RAG 시스템 구축 실행"""
        print("텍스트 파일 읽는 중...")
        documents = self.read_text_files()
        print(f"총 {len(documents)}개의 문서를 읽었습니다.")
        
        print("임베딩 생성 및 벡터 DB 구축 중...")
        self.create_embeddings(documents)
        print("벡터 DB 구축이 완료되었습니다.")
        
        print(f"벡터 DB가 {os.path.join(self.data_dir, 'chroma_db')}에 저장되었습니다.")

    def test_search(self, query: str, n_results: int = 3):
        """검색 테스트"""
        # 쿼리 임베딩
        query_embedding = self.model.encode(query)
        
        # 검색
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return results

if __name__ == "__main__":
    # RAG 시스템 구축
    rag_builder = RAGBuilder()
    rag_builder.build()
    
    # 검색 테스트
    test_query = "성균관대학교의 역사는?"
    results = rag_builder.test_search(test_query)
    print("\n검색 결과:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"\n{i}번째 결과:")
        print(f"제목: {metadata['title']}")
        print(f"내용: {doc[:200]}...")  # 처음 200자만 출력