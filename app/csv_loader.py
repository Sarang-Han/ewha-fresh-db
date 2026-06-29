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
    
    # 학사일정 CSV (academic_calendar_*.csv 패턴의 모든 파일을 찾아 합침)
    calendar_dir = base_dir / "학사일정"
    calendar_files = sorted(list(calendar_dir.glob("academic_calendar_*.csv")))
    
    if not calendar_files:
        # Fallback to the old default name
        fallback_path = calendar_dir / "academic_calendar.csv"
        if fallback_path.exists():
            calendar_files = [fallback_path]
            
    combined_calendar_lines = []
    header_saved = False
    
    for c_file in calendar_files:
        try:
            with open(c_file, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    continue
                lines = content.splitlines()
                if not lines:
                    continue
                if not header_saved:
                    combined_calendar_lines.append(lines[0])  # Header
                    header_saved = True
                # Append data lines (excluding header)
                combined_calendar_lines.extend(lines[1:])
            logger.info(f"학사일정 CSV 로드 완료: {c_file}")
        except Exception as e:
            logger.error(f"학사일정 CSV 로드 실패 ({c_file}): {str(e)}")
            
    if combined_calendar_lines:
        csv_texts["academic_calendar"] = "\n".join(combined_calendar_lines)
    else:
        logger.error("학사일정 CSV 파일을 찾을 수 없습니다.")
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
