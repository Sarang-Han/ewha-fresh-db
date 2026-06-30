import time
import logging
import requests
from typing import Optional, List
from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Gemini API 호출 및 재시도/폴백 정책을 통합 관리하는 공통 서비스"""
    
    def __init__(self, temperature: float = settings.llm_temperature):
        self.temperature = temperature
        self.model_chain = settings.model_chain
        self.api_key = settings.google_api_key

    def call_gemini(
        self, 
        prompt: str, 
        max_tokens: int = 8192, 
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Gemini API를 호출하여 결과를 반환합니다.
        설정된 model_chain에 따라 실패 시 폴백 및 재시도(exponential backoff)를 수행합니다.
        """
        temp = temperature if temperature is not None else self.temperature
        last_error = None
        
        for model in self.model_chain:
            # 각 모델당 최대 2번 재시도
            for attempt in range(2):
                try:
                    logger.info(f"[{model}] LLM 호출 시도 {attempt + 1}/2")
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    headers = {"Content-Type": "application/json"}
                    data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": temp,
                            "maxOutputTokens": max_tokens,
                        }
                    }
                    
                    response = requests.post(
                        f"{url}?key={self.api_key}",
                        headers=headers,
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"[{model}] API 응답 에러: {response.status_code}")
                        raise requests.exceptions.HTTPError(f"{response.status_code}: {response.text[:200]}")
                    
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            logger.info(f"[{model}] LLM 호출 성공")
                            return candidate["content"]["parts"][0]["text"]
                    
                    raise ValueError("API 응답 파싱 실패")
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"[{model}] 시도 {attempt + 1} 실패: {str(e)[:100]}")
                    if attempt < 1:  # 마지막 시도가 아니면 대기
                        time.sleep(1 * (attempt + 1))  # 1초, 2초 대기
            
            logger.warning(f"[{model}] 모든 재시도 실패, 다음 모델로 전환합니다.")
        
        logger.error(f"모든 모델 호출에 실패했습니다. 마지막 에러: {last_error}")
        return None
