import pytest

from paklib import io


def test_path():
    assert io.path('http://google.ru/') == 'http://google.ru/'
    assert io.path('\tgoogle.ru') == '_google.ru'
    assert io.path('\tgoogle\\ru') == '_google/ru'
    assert io.path(['http://google.ru/', 'path', str(3), 'target/']) == 'http://google.ru/path/3/target/'
    with pytest.raises(AttributeError):
        io.path(['http://google.ru/', 'path', 3, 'target/'])


def test_filename():
    assert io.filename('http://google.ru/') == 'http___google.ru_'
    assert io.filename('\tgoogle.ru') == '_google.ru'
    assert io.filename('\tgoogle\\ru') == '_google_ru'
