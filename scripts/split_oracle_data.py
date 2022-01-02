#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import subprocess
import logging
import random


if __name__ == "__main__":
    argv = iter(sys.argv[1:])
    data_prefix = Path(next(argv))
    random_seed = next(argv, 1234)
    random.seed(random_seed)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger()

    data_dir = data_prefix.parent
    data_prefix = data_prefix.name
    langs = [file.suffix[1:] for file in data_dir.glob(f"{data_prefix}.??")]
    for lang in langs:
        src_file = data_dir / f"{data_prefix}.oracle_{lang}"
        tgt_file = data_dir / f"{data_prefix}.{lang}"
        with open(src_file) as s_file, open(tgt_file) as t_file:
            oracle_data = list(zip(s_file, t_file))

        random.shuffle(oracle_data)
        split = int(0.8 * len(oracle_data))  # 80-20 split
        valid_src = data_dir / f"valid.oracle_{lang}"
        valid_tgt = data_dir / f"valid.{lang}"
        test_src = data_dir / f"test.oracle_{lang}"
        test_tgt = data_dir / f"test.{lang}"
        with open(valid_src, 'w') as s_file, open(valid_tgt, 'w') as t_file:
            oracle_src, oracle_tgt = zip(*oracle_data[:split])
            s_file.writelines(oracle_src)
            t_file.writelines(oracle_tgt)

        with open(test_src, 'w') as s_file, open(test_tgt, 'w') as t_file:
            oracle_src, oracle_tgt = zip(*oracle_data[split:])
            s_file.writelines(oracle_src)
            t_file.writelines(oracle_tgt)

