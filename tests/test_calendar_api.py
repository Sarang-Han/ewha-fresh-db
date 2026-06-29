from fastapi.testclient import TestClient
from datetime import datetime
from app.main import app

client = TestClient(app)

def test_get_calendar_all():
    response = client.get("/api/calendar")
    assert response.status_code == 200
    events = response.json()
    assert isinstance(events, list)
    assert len(events) > 0
    
    # Check schema fields of the first event
    event = events[0]
    required_keys = {
        "academic_year", "start_date", "end_date", "semester", 
        "event_type", "title_raw", "target", "is_holiday", "notes"
    }
    assert all(key in event for key in required_keys)
    
    # Check sorting (start_date ascending)
    start_dates = [e["start_date"] for e in events]
    assert start_dates == sorted(start_dates)

def test_get_calendar_filter_year():
    response = client.get("/api/calendar?year=2026")
    assert response.status_code == 200
    events = response.json()
    assert len(events) > 0
    for event in events:
        assert event["academic_year"] == 2026

def test_get_calendar_filter_event_type():
    response = client.get("/api/calendar?event_type=holiday")
    assert response.status_code == 200
    events = response.json()
    assert len(events) > 0
    for event in events:
        assert event["event_type"] == "holiday"

def test_get_calendar_filter_upcoming():
    response = client.get("/api/calendar?upcoming=true")
    assert response.status_code == 200
    events = response.json()
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    for event in events:
        assert event["end_date"] >= today_str

def test_get_calendar_filter_combined():
    response = client.get("/api/calendar?year=2026&event_type=holiday&upcoming=true")
    assert response.status_code == 200
    events = response.json()
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    for event in events:
        assert event["academic_year"] == 2026
        assert event["event_type"] == "holiday"
        assert event["end_date"] >= today_str
