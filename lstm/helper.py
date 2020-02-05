from os.path import dirname, abspath


def get_project_path():
    """
    Returns project path for saving and loading files
    """

    ROOT_DIR = dirname(dirname(abspath(__file__)))
    return ROOT_DIR
