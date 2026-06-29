from app.csv_loader import load_csv_texts, get_csv_texts

def test_load_csv_texts():
    csv_texts = load_csv_texts()
    
    # Check that keys are correct
    assert "course_registration" in csv_texts
    assert "academic_calendar" in csv_texts
    
    academic_calendar = csv_texts["academic_calendar"]
    assert len(academic_calendar) > 0
    
    # Check that both 2025 and 2026 data exist in the combined text
    assert "2025" in academic_calendar
    assert "2026" in academic_calendar
    assert "2025-02-01" in academic_calendar
    assert "2026-02-01" in academic_calendar
    
    # Check that there is only one header line
    lines = academic_calendar.strip().splitlines()
    header = "academic_year,start_date,end_date,semester,event_type,title_raw,target,is_holiday,notes"
    assert lines[0] == header
    
    # No other line should match the header exactly
    for line in lines[1:]:
        assert line != header

def test_get_csv_texts():
    texts1 = get_csv_texts()
    texts2 = get_csv_texts()
    assert texts1 is texts2  # should be the same singleton object
