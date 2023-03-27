from __future__ import \
    annotations  # ensure compatibility with cluster python version

import pathlib

import orjson

""" yield (centre, context, doc_label) tuples
"""


def load(input_dir: pathlib.Path, *, context_annotation=""):
    """Where fp is a json of {head: {modifier: [0, 2, 13], ..}, ...}"""

    # load doc_label2i and build i2doc_label
    with open(input_dir / "doc_label2i.json", "rb") as f:
        doc_label2i = orjson.loads(f.read())
    i2doc_label = {i: doc_label for doc_label, i in doc_label2i.items()}

    # successively yield (centre, context, doc_label) - unique instances only
    with open(input_dir / "modifers_locs_by_head.json", "rb") as f:
        data = orjson.loads(f.read())

        for head, modifier_locs in data.items():
            for modifier, locs in modifier_locs.items():
                for loc in set(locs):
                    yield (head, modifier + context_annotation, i2doc_label[loc])
