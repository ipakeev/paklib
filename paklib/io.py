import os
import pickle as cp
import shutil


def name_to_save(name):
    # name = name.replace(':', '_')
    # name = name.replace('/', '_')
    name = name.replace('\t', '_')
    name = name.replace('\\', '/')
    return name


def correct_file_name(file_name):
    if type(file_name) == list:
        try:
            file_name = [str(i).replace('\\', '/') for i in file_name]
        except AttributeError:
            print(file_name)
            raise
        file_name = os.path.join(*file_name)
    file_name = name_to_save(file_name)
    return file_name


def load_pickle(file_name, mode='rb', not_found=dict):
    file_name = correct_file_name(file_name)

    try:
        with open(file_name, mode) as op:
            f = cp.load(op)
    except IOError:
        f = not_found()

    return f


def save_pickle(file_name, data, mode='wb'):
    file_name = correct_file_name(file_name)
    make_dir_for_file(file_name)

    keyboardInterrupted = False
    while 1:
        try:
            with open(file_name, mode) as op:
                cp.dump(data, op)
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


def pickle_2_to_3(name, name_after=None):
    with open(name, 'rb') as op:
        f = cp.load(op, encoding='latin1')
    with open(name_after or name, 'wb') as op:
        cp.dump(f, op)


def pickle_3_to_2(name, name_after=None):
    with open(name, 'rb') as op:
        f = cp.load(op)
    with open(name_after or name, 'wb') as op:
        cp.dump(f, op, protocol=2)


def is_file_existed(file_name):
    file_name = correct_file_name(file_name)
    return os.path.exists(file_name)


def make_dir_for_path(path):
    path = correct_file_name(path)
    os.makedirs(path, exist_ok=True)


def make_dir_for_file(file_name):
    file_name = correct_file_name(file_name)
    path, file_name = os.path.split(file_name)
    if path:
        os.makedirs(path, exist_ok=True)


def clear_folder(path, remove_files=True, remove_dirs=True, func_files=None, func_dirs=None):
    path = correct_file_name(path)

    try:
        root, dirs, files = next(os.walk(path))
    except StopIteration:
        print(f'>!> folder "{path}" is already clear')
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


def list_of_dirs(path):
    path = correct_file_name(path)
    return next(os.walk(path))[1]


def list_of_files(path):
    path = correct_file_name(path)
    return next(os.walk(path))[2]


def copy_file(src, dst):
    src = correct_file_name(src)
    dst = correct_file_name(dst)
    make_dir_for_file(dst)
    shutil.copy(src, dst)


def move_file(src, dst):
    src = correct_file_name(src)
    dst = correct_file_name(dst)
    make_dir_for_file(dst)
    shutil.move(src, dst)
