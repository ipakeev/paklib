import datetime
from typing import Iterable

months = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'May': '05',
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12',
}


def get_month_code(month: str) -> str:
    return months[month]


def gone_minutes(from_date: datetime.datetime, now: datetime.datetime = None) -> float:
    if now is None:
        now = datetime.datetime.today()
    return (now - from_date).total_seconds() / 60


def string_from_date(date: datetime.date) -> str:
    return date.strftime('%Y%m%d')


def date_from_string(date: str) -> datetime.date:
    return datetime.datetime.strptime(date, '%Y%m%d').date()


def from_to_date(from_date: datetime.date, till_date: datetime.date, verbose=0) -> Iterable[datetime.date]:
    def logger(date: datetime.date):
        if verbose == 1:
            if not logged[0] or date.day == 1:
                logged[0] = True
                print(date)
        if verbose == 2:
            print(date)

    logged = [False]
    if till_date is None:
        till_date = from_date + datetime.timedelta(days=1)

    delta = datetime.timedelta(days=1)
    while from_date < till_date:
        logger(from_date)
        yield from_date
        from_date += delta


def convert_date_line_to_digit(date_line, verbose=0):
    date_line = sorted(date_line)
    start, stop = date_line[0], date_line[-1]
    stop += datetime.timedelta(days=1)
    line = []
    count = 0
    for date in from_to_date(start, stop, verbose=verbose):
        if date in date_line:
            for _ in range(date_line.count(date)):  # если команда сыграла несколько матчей в один день
                line.append(count)
        count += 1
    return line
