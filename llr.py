""" For some corpus of the form ... 
"""
from __future__ import \
    annotations  # ensure compatibility with cluster python version

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
from scipy.sparse import csr_matrix, lil_matrix, load_npz, save_npz
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

# load scripts in ./loaders
# e.g., eval callable via f = eval("evalled.script.f")
for p in pathlib.Path("loaders").glob("*.py"):
    exec(f"import loaders.{p.stem}")


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
        configs: list = get_configs("default_configs.json", path_syns=path_syns)

    # process configs in parallel
    process_map(
        do_config_star,
        zip(configs, cycle([path_syns]), range(len(configs))),
        max_workers = 25,
    )


def do_config_star(t):
    return do_config(*t)


def do_config(config: dict, path_syns: dict, position=0):

    desc = config["desc"]
    # print(f"config: {desc}")

    # get config options
    switch: bool = config["switch"]  # run config or skip?
    output_dir: pathlib.Path = resolve_fp(config["output_dir"], path_syns=path_syns)
    input_dir = resolve_fp(config["input_dir"], path_syns=path_syns)
    loader: typing.Callable = eval(config["loader"][0])
    loaders_nargs: dict = config["loader"][1]

    centres_of_interest = config["centres_of_interest"]

    # config to be run?
    if switch == False:
        print("\tconfig switched off ... skipping")

    else:

        # ------
        # build a mapping of centre word of interest to canonical form
        # ------
        mapping = {}
        for group in centres_of_interest:
            for variant in group:
                mapping[variant] = group[0]

        # ------
        # Build indices to each centre and context token available
        # ------

        c2i_fp = output_dir / desc / "c2i.json"
        c2j_fp = output_dir / desc / "c2j.json"

        if (c2i_fp.exists() == False) and (c2i_fp.exists() == False):

            # print("\t\tbuild centre and context indices")

            c2j = T2i(mapping=mapping)  # context indices
            c2i = T2i(mapping=mapping)  # known contentious centre indices
            for centre, context, doc_label in tqdm(
                loader(input_dir, **loaders_nargs),
                desc=f"{desc} build indices",
                position=position,
            ):
                c2j.append(context)
                c2i.append(centre)


            # save the indices
            c2i_fp.parent.mkdir(parents=True, exist_ok=True)
            with open(c2i_fp, "wb") as f:
                f.write(orjson.dumps(c2i.t2i))

            c2j_fp.parent.mkdir(parents=True, exist_ok=True)
            with open(c2j_fp, "wb") as f:
                f.write(orjson.dumps(c2j.t2i))

        else:

            # print("\t\load centre and context indices")

            with open(c2j_fp, "rb") as f:
                c2j = T2i(t2i=orjson.loads(f.read()), mapping=mapping)

            with open(c2i_fp, "rb") as f:
                c2i = T2i(t2i=orjson.loads(f.read()), mapping=mapping)


        # print("\t\tbuild a Trie of known contentious centres")
        # Note: we're only interested in the first of each group, as present in c2j, c2i
        trie = Trie()
        for group in centres_of_interest:
            trie.add(group[0])

        # print("\t\tget a list of known contentious context indices")
        known_contentious_i = set()
        known_contentious = set()
        for centre, i in c2i:
            if trie.in_whole(centre):
                known_contentious.add(centre)
                known_contentious_i.add(i)

        # frequency and llr table dims
        n_rows: int = len(c2i.t2i)
        n_columns: int = len(c2j.t2i)

        frq_profiles_fp = output_dir / desc / f"frq_profiles.npz"
        if frq_profiles_fp.exists() == False:

            # print("\t\tbuild frequency profiles")

            frequency_profiles = lil_matrix((n_rows, n_columns), dtype=int)
            global_profile = np.zeros(n_columns, dtype=int)
            for centre, context, doc_label in tqdm(
                loader(input_dir, **loaders_nargs),
                desc=f"{desc}, freq profiles",
                position=position,
            ):
                i = c2i[centre]
                j = c2j[context]

                # freq_profiles
                frequency_profiles[i, j] += 1

            # save
            frequency_profiles = csr_matrix(frequency_profiles)
            frq_profiles_fp.parent.mkdir(exist_ok=True, parents=True)
            save_npz(frq_profiles_fp, frequency_profiles)

        else:

            # print("\t\load frequency profiles")
            frequency_profiles: csr_matrix = load_npz(frq_profiles_fp)

        # print("\t\build a global frequency profile")
        global_profile = np.array(frequency_profiles.sum(axis=0)).squeeze()

        # print("\t\tbuld llr profiles")
        llr_profiles = lil_matrix((n_rows, n_columns), dtype=float)
        for centre, i in tqdm(
            c2i.t2i.items(), desc=f"{desc}, llr profiles", position=position
        ):
            k1 = frequency_profiles[i, :].toarray().squeeze()
            k2 = global_profile - k1
            llr_profiles[i] = llr_profile_(k1, k2)
        llr_profiles = csr_matrix(llr_profiles)

        # save
        save_fp = output_dir / desc / f"llr_profiles.npz"
        save_fp.parent.mkdir(exist_ok=True, parents=True)
        save_npz(save_fp, llr_profiles)

        # save a copy of the config
        save_fp = output_dir / desc / "config.json"
        save_fp.parent.mkdir(exist_ok=True, parents=True)
        with open(save_fp, "wb") as f:
            f.write(orjson.dumps(config))


class T2i:
    """An basic index class"""

    def __init__(self, *, t2i=None, mapping=None):

        if t2i==None:
            self.t2i = {}
            self.maxi = -1
        else:
            self.t2i = t2i
            self.maxi = max(t2i.values())

        if mapping==None:
            self.mapping = {}
        else:
            self.mapping=mapping

    def __getitem__(self, t):

        if t in self.mapping:
            t = self.mapping[t]

        return self.t2i[t]

    def append(self, t):

        if t in self.mapping:
            t = self.mapping[t]

        if t not in self.t2i.keys():
            self.maxi += 1
            self.t2i[t] = self.maxi

    def __iter__(self):
        for x in self.t2i.items():
            yield x


def gen_items(fps: typing.Iterable[pathlib.Path]) -> typing.Generator:
    """Return each item (in each fp of passed fps)."""

    for fp in fps:

        with open(fp, "rb") as f:
            for item in ijson.items(f, "item"):
                yield item


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


def llr_profile_(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Return a llr profile."""
    llr_profile = np.zeros(len(k1))

    n1 = k1.sum()
    n2 = k2.sum()
    n = n1 + n2

    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / n

    # i.e., where k1 = 0, make the result 0, as the entry is irrelevant to k1
    mask = k1 > 0
    llr_profile[mask] = 2 * (
        log_binom(k1[mask], n1, p1[mask])
        - log_binom(k1[mask], n1, p[mask])
        + log_binom(k2[mask], n2, p2[mask])
        - log_binom(k2[mask], n2, p[mask])
    )

    return llr_profile


def log_binom(k: np.ndarray, n: float, p: np.ndarray) -> np.ndarray:

    result = np.zeros(len(k))
    # Note: @p==1, log(p) = 0: hence handled implicity

    # where p > 0 and p < 1
    mask = (p > 0) & (p < 1)
    result[mask] = k[mask] * np.log(p[mask]) + (n - k[mask]) * np.log(1 - p[mask])

    return result


class Trie:
    """Build a trie and get affixes for some given prefix.

    E.g., for getting heads for some given modifiers.
    """

    def __init__(self):
        self.root = dict()

    def add(self, word: str):
        """Add word to trie"""

        # add the chars to the trie
        ref = self.root
        for char in word:
            if char in ref:
                ref = ref[char]
            else:
                ref[char] = {}
                ref = ref[char]

        # denote the end of a word
        # i.e., otherwise if wood and woods were added, woods would mask wood
        ref["<END>"] = True

    def in_whole(self, word: str) -> bool:
        """Return True if a complete word is in the trie.o

        Note: returns false in the following cases:
        word = "wood", trie contains "wooden", "driftwood", but not wood.
        """
        # burn though the word
        ref = self.root
        for char in word:
            if char in ref:
                ref = ref[char]
            else:
                return False

        # check whether any of the
        if "<END>" in ref:
            return True
        else:
            return False

    def get_affixes(self, prefix: str) -> list:
        """Return a list of all affixes, for given prefix."""

        # burn through the prefix ... ending with ref on the whatever comes after the prefix
        ref = self.root
        for char in prefix:
            if char in ref:
                ref = ref[char]
            else:
                return []

        acc = []
        stack = [("", ref)]

        # collect suffices
        while stack:
            collected, ref = stack.pop()

            # ref is exhausted, add collected to accumulator if not nothing
            if ref == {"<END>": True}:
                if collected != "":
                    acc.append(collected)
                else:
                    pass

            # still more to collect, add to stack
            else:
                for char in ref.keys():
                    if char != "<END>":
                        stack.append((collected + char, ref[char]))

        return acc


if __name__ == "__main__":
    main(sys.argv[1:])  # assumes an alternative config path may be passed to CL
