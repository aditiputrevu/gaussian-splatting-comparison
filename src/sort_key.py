"""
Sort key for filenames like photo_10_2026-08-05_15-11-53.jpg so that
photo_2 sorts before photo_10 (i.e. numeric order on the photo index,
not lexicographic string order).
"""
import re

_PATTERN = re.compile(r"photo_(\d+)_")


def photo_sort_key(filename: str):
    """
    Returns an int key extracted from 'photo_<N>_...'. Falls back to the
    filename itself (string) if the pattern doesn't match, so this is safe
    to use on datasets that don't follow the photo_N_... convention too.
    """
    m = _PATTERN.search(filename)
    if m:
        return (0, int(m.group(1)))
    return (1, filename)


def sorted_image_filenames(filenames):
    return sorted(filenames, key=photo_sort_key)
