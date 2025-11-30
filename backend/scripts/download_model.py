"""
Hugging Face에서 multilingual-e5-large-instruct 모델을 로컬로 다운로드하는 스크립트
"""
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

def download_model():
    """multilingual-e5-large-instruct 모델을 로컬 디렉터리에 다운로드"""
    
    model_name = "intfloat/multilingual-e5-large-instruct"
    
    # 저장 경로 설정
    script_dir = Path(__file__).parent
    backend_dir = script_dir.parent
    models_dir = backend_dir / "models" / "multilingual-e5-large-instruct"
    
    # 디렉터리 생성
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Multilingual E5 Large Instruct 모델 다운로드 중...")
    print(f"모델: {model_name}")
    print(f"저장 위치: {models_dir}")
    print("모델 크기: 약 2.24GB (시간이 걸릴 수 있습니다)")
    print("-" * 60)
    
    # 모델 다운로드 및 저장
    model = SentenceTransformer(model_name, cache_folder=str(models_dir.parent))
    
    print("-" * 60)
    print("✅ 다운로드 완료!")
    print(f"모델 위치: {models_dir}")
    print("\n.env 파일에서 다음과 같이 설정하세요:")
    print(f"EMBEDDING_MODEL={model_name}")
    print("\n또는 로컬 경로를 직접 지정:")
    print(f"EMBEDDING_MODEL={models_dir}")
    
    return str(models_dir)

if __name__ == "__main__":
    try:
        download_model()
    except KeyboardInterrupt:
        print("\n\n❌ 다운로드 취소됨")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {str(e)}")
