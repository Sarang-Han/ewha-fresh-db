import csv
import logging
from io import StringIO
from typing import List, Optional
from datetime import datetime

from app.models import CalendarEventResponse
from app.csv_loader import get_csv_texts

logger = logging.getLogger(__name__)


class CalendarService:
    """학사일정 조회 및 필터링 처리를 담당하는 비즈니스 도메인 서비스"""

    @staticmethod
    def get_filtered_events(
        year: Optional[int] = None,
        event_type: Optional[str] = None,
        upcoming: bool = False
    ) -> List[CalendarEventResponse]:
        """
        학년도, 일정 유형, 다가오는 일정 여부별로 학사일정 목록을 필터링 및 정렬하여 조회합니다.
        
        Args:
            year: 조회할 학년도 (예: 2026)
            event_type: 필터링할 이벤트 타입 (예: course_registration, holiday 등)
            upcoming: True일 경우 오늘 날짜(또는 현재 기준) 이후에 종료되는 일정만 필터링
            
        Returns:
            List[CalendarEventResponse]: 정렬된 학사일정 리스트
        """
        csv_texts = get_csv_texts()
        calendar_csv = csv_texts.get("academic_calendar", "")
        
        if not calendar_csv:
            logger.warning("학사일정 CSV 데이터를 찾을 수 없습니다.")
            raise ValueError("학사일정 데이터를 찾을 수 없습니다.")
            
        # CSV 데이터 파싱
        f = StringIO(calendar_csv.strip())
        reader = csv.DictReader(f)
        
        events = []
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        for row in reader:
            # 1. 학년도 필터
            if year and int(row["academic_year"]) != year:
                continue
                
            # 2. 이벤트 타입 필터
            if event_type and row["event_type"] != event_type:
                continue
                
            # 3. 최근/다가오는 일정 필터 (오늘 날짜 기준 end_date가 지나지 않은 일정)
            if upcoming and row["end_date"] < today_str:
                continue
                
            events.append(CalendarEventResponse(
                academic_year=int(row["academic_year"]),
                start_date=row["start_date"],
                end_date=row["end_date"],
                semester=row["semester"],
                event_type=row["event_type"],
                title_raw=row["title_raw"],
                target=row["target"],
                is_holiday=row["is_holiday"] == "1" or row["is_holiday"].lower() == "true",
                notes=row["notes"] if row.get("notes") else None
            ))
            
        # 날짜 오름차순 정렬 (시작일 기준)
        events.sort(key=lambda x: x.start_date)
        
        return events
