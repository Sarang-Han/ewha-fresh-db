"""
CSV 데이터 로더 모듈
수강신청, 학사일정 CSV를 텍스트로 로드하여 전역으로 보관
"""
import os
from pathlib import Path
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def load_csv_texts() -> Dict[str, str]:
    """
    CSV 파일들을 텍스트로 읽어 딕셔너리로 반환
    
    서버 시작 시 한 번 호출하여 전역/싱글톤으로 보관
    
    Returns:
        {
            "course_registration": "CSV 전체 텍스트",
            "academic_calendar": "CSV 전체 텍스트"
        }
    """
    # 프로젝트 루트 기준 데이터 디렉터리
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "data" / "official"
    
    csv_texts = {}
    
    # 수강신청 일정 CSV
    course_reg_path = base_dir / "수강신청" / "25-2_course_registration.csv"
    try:
        with open(course_reg_path, encoding="utf-8") as f:
            csv_texts["course_registration"] = f.read()
        logger.info(f"수강신청 CSV 로드 완료: {course_reg_path}")
    except FileNotFoundError:
        logger.error(f"수강신청 CSV 파일을 찾을 수 없습니다: {course_reg_path}")
        csv_texts["course_registration"] = ""
    except Exception as e:
        logger.error(f"수강신청 CSV 로드 실패: {str(e)}")
        csv_texts["course_registration"] = ""
    
    # 학사일정 CSV
    calendar_path = base_dir / "학사일정" / "academic_calendar.csv"
    try:
        with open(calendar_path, encoding="utf-8") as f:
            csv_texts["academic_calendar"] = f.read()
        logger.info(f"학사일정 CSV 로드 완료: {calendar_path}")
    except FileNotFoundError:
        logger.error(f"학사일정 CSV 파일을 찾을 수 없습니다: {calendar_path}")
        csv_texts["academic_calendar"] = ""
    except Exception as e:
        logger.error(f"학사일정 CSV 로드 실패: {str(e)}")
        csv_texts["academic_calendar"] = ""
    
    return csv_texts


# 전역 CSV 텍스트 저장소 (서버 시작 시 초기화)
CSV_TEXTS: Dict[str, str] = {}


def initialize_csv_texts():
    """CSV 텍스트 전역 초기화"""
    global CSV_TEXTS
    CSV_TEXTS = load_csv_texts()
    logger.info(f"CSV 텍스트 초기화 완료: {list(CSV_TEXTS.keys())}")


def get_csv_texts() -> Dict[str, str]:
    """전역 CSV 텍스트 반환"""
    global CSV_TEXTS
    if not CSV_TEXTS:
        initialize_csv_texts()
    return CSV_TEXTS
