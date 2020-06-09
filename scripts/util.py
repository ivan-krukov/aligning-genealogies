from os import path


def get_basename(name = __file__):
    my_dir, my_name = path.split(name)
    my_prefix, my_ext = path.splitext(my_name)
    return my_prefix
