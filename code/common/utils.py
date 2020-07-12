import fnmatch
from itertools import chain, cycle, islice
import os


def get_files_starting_with(dir_, prefix):
    get_files_with_pattern(dir_, prefix+'*.*')


def get_files_with_pattern(dir_, pattern):
    matching_files = []
    for f in os.listdir(dir_):
        if fnmatch.fnmatch(f, pattern):
            matching_files.append(f)
    return matching_files


def grouper_without_fill(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def flatten(iterables):
    return [list(chain.from_iterable(iterable)) for iterable in iterables]

