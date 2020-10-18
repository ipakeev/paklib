import datetime
import paklib.dt


def test_convert_date_line_to_digit():
    assert paklib.dt.convert_date_line_to_digit([
        datetime.date(2020, 1, 1), datetime.date(2020, 1, 5), datetime.date(2020, 2, 1),
    ]) == [0, 4, 31]


def test_from_to_date():
    from_date = datetime.date(2019, 12, 29)
    till_date = datetime.date(2020, 1, 3)
    target = [
        datetime.date(2019, 12, 29),
        datetime.date(2019, 12, 30),
        datetime.date(2019, 12, 31),
        datetime.date(2020, 1, 1),
        datetime.date(2020, 1, 2),
    ]
    assert list(paklib.dt.from_to_date(from_date, till_date)) == target
