import multiprocessing


def get_cores_count():
    return multiprocessing.cpu_count()