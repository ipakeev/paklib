from paklib import io


def test_correct_file_name():
    assert io.correct_file_name(['http://google.ru/', 'path', 3, 'target/']) == 'http://google.ru/path/3/target/'
