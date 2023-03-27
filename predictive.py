""" given a matrix of llr scores, centre x contexts ... and some seed pattern from which the underlying corpus was bootstrapped:
    for feature in seed profile:
        for min_llr_score in ...:
            return ratio of resulting known contentious / total centres

"""
from __future__ import \
    annotations  # ensure compatibility with cluster python version

import operator
import pathlib
import pickle as pkl
import re
import sys
import typing
from collections import Counter, defaultdict
from functools import reduce
from itertools import cycle

import ijson
import numpy as np
import orjson
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz, vstack
from scipy.stats import entropy
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# # load scripts in ./evalled
# # e.g., eval callable via f = eval("evalled.script.f")
# for p in pathlib.Path("evalled").glob("*.py"):
#     exec(f"import converters.{p.stem}")


def main(argv):

    # load the path synonynms using in configs
    # e.g., {"DATA": "~/surfdrive/data"}
    with open("path_syns.json", "rb") as f:
        path_syns = orjson.loads(f.read())

    # load the configs - prefer CL specified over default
    # [
    #   {
    #       "desc": "config 1",
    #       "switch": true,
    #       "output_dir": "DATA/project1"
    #   }
    # ]
    try:
        configs: list = get_configs(argv[0], path_syns=path_syns)
    except:
        configs: list = get_configs("llr_configs.json", path_syns=path_syns)

    # iterate over configs
    for config in configs:

        desc = config["desc"]
        print(f"config={desc}")

        # get config options
        switch: bool = config["switch"]  # run config or skip?

        input_dir: pathlib.Path = resolve_fp(config["output_dir"], path_syns=path_syns)

        n_processes: int = config["n_processes"]

        # known contentious centres
        known_contentious_centre_patterns = config["known_contentious_centre_patterns"]

        # config to be run?
        if switch == False:
            print("\tconfig switched off ... skipping")

        else:

            results = []

            # load the llr profiles
            llr_profiles: csr_matrix = load_npz(input_dir / "llr_profiles.npz")

            # load the centre, context to index dicts
            with open(input_dir / "llr_context2i.json", "rb") as f:
                context2i = orjson.loads(f.read())
            i2context = {i: context for context, i in context2i.items()}

            with open(input_dir / "llr_centre2i.json", "rb") as f:
                centre2i = orjson.loads(f.read())
            ranked_centres: np.ndarray = np.array(
                list(sorted(list(centre2i.keys()), key=lambda x: centre2i[x]))
            )

            print("\tbuild list of known contentious centres")
            contentious_patterns = {}
            for p_str in known_contentious_centre_patterns:
                contentious_patterns[p_str] = re.compile(p_str)

            contentious_centres = set()
            for centre, _ in tqdm(centre2i.items()):
                for _, p in contentious_patterns.items():
                    if re.match(p, centre):
                        contentious_centres.add(centre)
                        break

            # produce a ranked list of ...
            print("\tscore context contentiousness associativity")
            print(f"\t\tcontexts to consider = {len(context2i.keys())}")
            results = []
            cis, contexts = list(zip(*i2context.items()))
            for q in [0, 0.1, 0.25, 0.5, 0.75]:
                print(f"quantile: {q}")
                results += filter(
                    lambda x: x != None,
                    process_map(
                        score_column_star,
                        zip(
                            cis,
                            contexts,
                            cycle([q]),
                            cycle([llr_profiles]),
                            cycle([contentious_centres]),
                            cycle([ranked_centres]),
                        ),
                        max_workers=n_processes,
                        chunksize=100,
                    ),
                )

            # save contexts with some degree of contentious associativity (sorted by ratio)
            print("\tsaving...")

            # save contexts with some degree of contentious associativity (sorted by ratio)
            with open(input_dir / "predictive_by_ratio.txt", "w") as f:
                f.writelines(
                    [
                        ", ".join([str(x) for x in t]) + "\n"
                        for t in sorted(results, key=lambda x: x[4], reverse=True)
                    ]
                )

            # save contexts with some degree of contentious associativity
            # (sorted by contentious count)
            with open(input_dir / "predictive_by_contentious_count.txt", "w") as f:
                f.writelines(
                    [
                        ", ".join([str(x) for x in t]) + "\n"
                        for t in sorted(results, key=lambda x: x[3], reverse=True)
                    ]
                )

            # save contexts with some degree of contentious associativity
            # (sorted b quantile)
            with open(input_dir / "predictive_by_quantile.txt", "w") as f:
                f.writelines(
                    [
                        ", ".join([str(x) for x in t]) + "\n"
                        for t in sorted(results, key=lambda x: x[1], reverse=True)
                    ]
                )

            # save contexts with some degree of contentious associativity
            # (sorted by contentious count * ratio), as a measure of cross-centre prediction
            with open(input_dir / "predictive_by_contentious_count*ratio.txt", "w") as f:
                f.writelines(
                    [
                        ", ".join([str(x) for x in t]) + "\n"
                        for t in sorted(results, key=lambda x: x[3]*x[4], reverse=True)
                    ]
                )

def score_column_star(t):
    return score_column(*t)


def score_column(
    ci: int,
    context: str,
    q: float,  # quartile e.g., 0.25
    llr_profiles: csr_matrix,
    contentious_centres: set,
    ranked_centres: np.ndarray,
):
    """Return (
            context,
            quantile llr,
            associated centre count,
            associated known contentious count
            ratio centre count / known contentious count
           )

    if known contentious count > 0

    where ratio center count / known contentious count ~= P(contentious centre | context)
    """

    column: np.ndarray = llr_profiles[:, ci].toarray().squeeze()

    # get mask of row which exceed column quantile
    if q > 0:
        col_quantile = np.quantile(column, q)
        mask = column >= col_quantile
    else:
        col_quantile = 0
        mask = column > col_quantile

    matched_centres = ranked_centres[mask]
    match_count = sum(mask)

    # playing it safe - potentially for quantile-based masks do to something funny
    if match_count != 0:

        # get count of matches_centres \int contentious centres
        contentious_count = len(contentious_centres.intersection(matched_centres))

        if contentious_count > 0:

            # record result
            t = (
                context,
                col_quantile,
                match_count,
                contentious_count,
                contentious_count / match_count,
            )
            return t
        else:
            return None

    else:
        return None


def resolve_fp(path: str, path_syns: typing.Union[None, dict] = None) -> pathlib.Path:
    """Resolve path synonyns, ~, and make absolute, returning pathlib.Path.

    Args:
        path (str): file path or dir
        path_syns (dict): dict of
            string to be replaced : string to do the replacing

    E.g.,
        path_syns = {"DATA": "~/documents/data"}

        resolve_fp("DATA/project/run.py")
        # >> user/home/john_smith/documents/data/project/run.py
    """

    # resolve path synonyms
    if path_syns is not None:
        for fake, real in path_syns.items():
            path = path.replace(fake, real)

    # expand user and resolve path
    return pathlib.Path(path).expanduser().resolve()


def get_configs(config_fp_str: str, *, path_syns=None) -> list:
    """Return the configs to run."""

    configs_fp = resolve_fp(config_fp_str, path_syns)

    with open(configs_fp, "rb") as f:
        configs = orjson.loads(f.read())

    return configs


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

    for fp in filter(lambda fp: re.search(pattern, str(fp)), dir_path.glob("*")):

        # no ignore pattern specified
        if ignore_pattern is None:
            yield fp
        else:
            # ignore pattern specified, but not met
            if re.search(ignore_pattern, str(fp)):
                pass
            else:
                yield fp


if __name__ == "__main__":
    main(sys.argv[1:])  # assumes an alternative config path may be passed to CL
