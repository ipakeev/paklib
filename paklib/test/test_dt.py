import datetime

import paklib.dt


def test_gone_minutes():
    now = datetime.datetime(2019, 6, 25, 10, 49)
    assert paklib.dt.gone_minutes(datetime.datetime(2019, 6, 25, 10, 45), now=now) == 4

    now = datetime.datetime(2019, 6, 25, 10, 45)
    assert paklib.dt.gone_minutes(datetime.datetime(2019, 6, 25, 10, 45), now=now) == 0

    now = datetime.datetime(2019, 6, 25, 10, 40)
    assert paklib.dt.gone_minutes(datetime.datetime(2019, 6, 25, 10, 45), now=now) == -5

    now = datetime.datetime(2019, 6, 25, 10, 45)
    assert paklib.dt.gone_minutes(datetime.datetime(2019, 6, 23, 10, 45), now=now) == 60 * 24 * 2


def test_string_from_date():
    assert paklib.dt.string_from_date(datetime.date(2020, 1, 7)) == '20200107'


def test_date_from_string():
    assert paklib.dt.date_from_string('20200107') == datetime.date(2020, 1, 7)


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


def test_convert_date_line_to_digit():
    assert paklib.dt.convert_date_line_to_digit([
        datetime.date(2020, 1, 1), datetime.date(2020, 1, 5), datetime.date(2020, 2, 1),
    ]) == [0, 4, 31]
