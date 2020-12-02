import os
import pickle
import shutil
from typing import Union, List

_table_path = str.maketrans('\t\\', '_/')
_table_filename = str.maketrans('\t\\:/', '____')


def path(name: Union[str, List[str]]) -> str:
    if type(name) == str:
        name = name.translate(_table_path)
    else:
        name = [i.translate(_table_path) for i in name]
        name = os.path.join(*name)
    return name.replace(os.sep, os.altsep)


def filename(name: str) -> str:
    return name.translate(_table_filename).replace(os.sep, '_')


def load_pickle(name: Union[str, List[str]], mode='rb', not_found=dict):
    name = path(name)

    try:
        with open(name, mode) as op:
            f = pickle.load(op)
    except IOError:
        f = not_found()

    return f


def save_pickle(name: Union[str, List[str]], data, mode='wb'):
    name = path(name)
    make_dir_for_file(name)

    keyboardInterrupted = False
    while 1:
        try:
            with open(name, mode) as op:
                pickle.dump(data, op)
        except KeyboardInterrupt:
            print('>!> Please, wait! Saving data...')
            keyboardInterrupted = True
            continue
        except Exception as e:
            print('>!> File not saved!')
            raise e

        if keyboardInterrupted:
            raise KeyboardInterrupt
        break


def pickle_2_to_3(name: Union[str, List[str]], name_after=None):
    with open(name, 'rb') as op:
        f = pickle.load(op, encoding='latin1')
    with open(name_after or name, 'wb') as op:
        pickle.dump(f, op)


def pickle_3_to_2(name: Union[str, List[str]], name_after=None):
    with open(name, 'rb') as op:
        f = pickle.load(op)
    with open(name_after or name, 'wb') as op:
        pickle.dump(f, op, protocol=2)


def is_existed(name: Union[str, List[str]]):
    name = path(name)
    return os.path.exists(name)


def make_dir_for_path(name: Union[str, List[str]]):
    name = path(name)
    os.makedirs(name, exist_ok=True)


def make_dir_for_file(name: Union[str, List[str]]):
    name = path(name)
    path_name, file_name = os.path.split(name)
    if path:
        os.makedirs(path_name, exist_ok=True)


def clear_folder(name: Union[str, List[str]], remove_files=True, remove_dirs=True, func_files=None, func_dirs=None):
    name = path(name)

    try:
        root, dirs, files = next(os.walk(name))
    except StopIteration:
        print(f'>!> folder "{name}" is already clear')
        return

    if remove_files:
        for name in files:
            if func_files is not None:
                if func_files(name):
                    os.remove(os.path.join(root, name))
            else:
                os.remove(os.path.join(root, name))
    if remove_dirs:
        for name in dirs:
            if func_dirs is not None:
                if func_dirs(name):
                    clear_folder(os.path.join(root, name))
                    os.rmdir(os.path.join(root, name))
            else:
                clear_folder(os.path.join(root, name))
                os.rmdir(os.path.join(root, name))


def list_of_dirs(name: Union[str, List[str]]):
    name = path(name)
    return next(os.walk(name))[1]


def list_of_files(name: Union[str, List[str]]):
    name = path(name)
    return next(os.walk(name))[2]


def copy_file(src: Union[str, List[str]], dst: Union[str, List[str]]):
    src = path(src)
    dst = path(dst)
    make_dir_for_file(dst)
    shutil.copy(src, dst)


def move_file(src: Union[str, List[str]], dst: Union[str, List[str]]):
    src = path(src)
    dst = path(dst)
    make_dir_for_file(dst)
    shutil.move(src, dst)
