from __future__ import \
    annotations  # ensure compatibility with cluster python version

import pathlib
import re
import typing

""" yield (centre, context, doc_label) tuples
"""


def load(
    input_dir: pathlib.Path, *, role: str = "", context_annotation="", pattern=".+"
):
    fps = gen_dir(input_dir, pattern=re.compile(pattern))

    for fp in fps:

        with open(fp, "r") as f:
            lines = (
                f.readlines()
            )  # this faster on the cluster than reading line by line
            for line in lines:
                entity, r, word, pattern_, doc_label, text = line.split(", ", 5)

                if role == "" or r == role:

                    if "_n" in pattern_:  # i.e., negation
                        yield (entity, "NOT" + word + context_annotation, doc_label)
                    else:
                        yield (entity, word + context_annotation, doc_label)


def gen_dir(
    dir_path: pathlib.Path,
    *,
    pattern: re.Pattern = re.compile(".+"),
    ignore_pattern: typing.Union[re.Pattern, None] = None,
) -> typing.Generator:
    """Return a generator yielding pathlib.Path objects in a directory,
    optionally matching a pattern.

    Args:
        dir (str): directory from which to retrieve file names [default: script dir]
        pattern (re.Pattern): re.search pattern to match wanted files [default: all files]
        ignore (re.Pattern): re.search pattern to ignore wrt., previously matched files
    """

    for fp in filter(lambda fp: re.search(pattern, str(fp.name)), dir_path.glob("*")):

        # no ignore pattern specified
        if ignore_pattern is None:
            yield fp
        else:
            # ignore pattern specified, but not met
            if re.search(ignore_pattern, str(fp.name)):
                pass
            else:
                yield fp
