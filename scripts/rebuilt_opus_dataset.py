#!/usr/bin/env python

import sys
import os
from pathlib import Path
import subprocess
import logging
import random
import json
import multiprocessing

def multi_download(argv):
    url, path, file = argv
    if not os.path.exists(file):
        with open(file+".stdout", 'w') as out, open(file+".stderr", 'w') as err:
            subprocess.run(
                [f"wget {url} -P {path} && unzip -o {file} -d {path}"],
                shell=True,
                stdout=out,
                stderr=err,
            )
    print(f"Done {file}")

if __name__ == "__main__":
    argv = iter(sys.argv[1:])
    data_dir = Path(next(argv))
    random_seed = next(argv, 1234)
    random.seed(random_seed)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger()
    orig_data = data_dir / "opus-100-corpus" / "v1.0"
    orig_supervised = orig_data / "supervised"
    orig_zeroshot = orig_data / "zero-shot"

    # download dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading original OPUS-100 data")
    if not (orig_supervised / "en-zu").exists():
        subprocess.run([
            'wget',
            'https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz',
            '-P',
            data_dir.as_posix(),
        ])
        subprocess.run([
            'tar', 'xf', (data_dir / "opus-100-corpus-v1.0.tar.gz").as_posix()
        ])
    logger.info("Downloaded")

    # de-duplicate supervised data
    supervised_lpairs = os.listdir(orig_supervised)
    excluded_lang = {'an', 'dz', 'hy', 'mn', 'yo'}
    for lpair in supervised_lpairs:
        logger.info(f"De-duplicating {lpair}")
        lpair_dir = orig_supervised / lpair
        src, tgt = lpair.split('-')
        if src in excluded_lang or tgt in excluded_lang:
            continue
        if (lpair_dir / f"opus.{lpair}-train-rebuilt.{tgt}").exists():
            continue

        test_src =  lpair_dir / f"opus.{lpair}-test.{src}"
        test_tgt =  lpair_dir / f"opus.{lpair}-test.{tgt}"
        with open(test_src) as src_file, open(test_tgt) as tgt_file:
            test_data = list(zip(src_file.readlines(), tgt_file.readlines()))
            test_length = len(test_data)
            test_data = set(test_data)
        logger.info(
            f"{lpair} | test data before (self) de-duplicate: {test_length}; "
            f"test data after (self) de-duplicate: {len(test_data)}; "
        )

        valid_src =  lpair_dir / f"opus.{lpair}-dev.{src}"
        valid_tgt =  lpair_dir / f"opus.{lpair}-dev.{tgt}"
        with open(valid_src) as src_file, open(valid_tgt) as tgt_file:
            valid_data = list(zip(src_file.readlines(), tgt_file.readlines()))
            valid_length = len(valid_data)
            valid_data = list(set(valid_data) - test_data)
        logger.info(
            f"{lpair} | valid data before de-duplicate: {valid_length}; "
            f"valid data after de-duplicate: {len(valid_data)}"
        )

        train_src =  lpair_dir / f"opus.{lpair}-train.{src}"
        train_tgt =  lpair_dir / f"opus.{lpair}-train.{tgt}"
        with open(train_src) as src_file, open(train_tgt) as tgt_file:
            tmp_data = set(zip(src_file.readlines(), tgt_file.readlines()))
            train_data = list(tmp_data - test_data)
        logger.info(
            f"{lpair} | train data before de-duplicate: {len(tmp_data)}; "
            f"train data after de-duplicate: {len(train_data)}"
        )

        # supplement dev set from training set
        supply_size = valid_length - len(valid_data)
        random.shuffle(train_data)
        if supply_size > 0:
            valid_data += train_data[-supply_size:]
            train_data = train_data[:-supply_size]
        logger.info(
            f"{lpair} | rebuilt valid data size: {len(valid_data)}; "
            f"rebuilt train data size: {len(train_data)}"
        )
        valid_src =  lpair_dir / f"opus.{lpair}-dev-rebuilt.{src}"
        valid_tgt =  lpair_dir / f"opus.{lpair}-dev-rebuilt.{tgt}"
        with open(valid_src, 'w') as src_file, open(valid_tgt, 'w') as tgt_file:
            src_data, tgt_data = list(zip(*valid_data))
            src_file.writelines(src_data)
            tgt_file.writelines(tgt_data)

        train_src =  lpair_dir / f"opus.{lpair}-train-rebuilt.{src}"
        train_tgt =  lpair_dir / f"opus.{lpair}-train-rebuilt.{tgt}"
        with open(train_src, 'w') as src_file, open(train_tgt, 'w') as tgt_file:
            src_data, tgt_data = list(zip(*train_data))
            src_file.writelines(src_data)
            tgt_file.writelines(tgt_data)

    # resampling zeroshot dev set
    zeroshot_lpairs = os.listdir(orig_zeroshot)
    for lpair in zeroshot_lpairs:
        logger.info(f"Resampling dev set for {lpair}")
        lpair_dir = orig_zeroshot / lpair
        src, tgt = lpair.split('-')
        if (lpair_dir / f"opus.{lpair}-dev.{tgt}").exists():
            continue

        test_src =  lpair_dir / f"opus.{lpair}-test.{src}"
        test_tgt =  lpair_dir / f"opus.{lpair}-test.{tgt}"
        test_data = []
        with open(test_src) as src_file, open(test_tgt) as tgt_file:
            for ss, tt in zip(src_file, tgt_file):
                test_data.append((ss, tt))
        test_length = len(test_data)
        test_data = set(test_data)

        # downloading all data & shuffle
        tmp_dir = lpair_dir / "downloaded"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            'wget',
            f'http://opus.nlpl.eu/opusapi/?source={src}&target={tgt}&preprocessing=moses',
            '-O',
            (tmp_dir / "corpus_info.json").as_posix(),
        ])
        with open(tmp_dir / "corpus_info.json") as file:
            all_corpus = json.load(file)

        download_paths = []
        for corpus in all_corpus['corpora']:
            corpus_dir = tmp_dir / corpus['corpus'] / corpus['version']
            corpus_dir.mkdir(parents=True, exist_ok=True)
            download_paths.append((
                corpus['url'], corpus_dir.as_posix(),
                os.path.join(corpus_dir, os.path.basename(corpus['url']))
            ))
        logger.info(f"{lpair} starts downloading")
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            p.map(multi_download, download_paths)
        logger.info(f"{lpair} finished downloading")

        all_data = []
        for src_file in tmp_dir.rglob(f"*.{lpair}.{src}"):
            tgt_file = src_file.with_suffix('.'+tgt)
            assert tgt_file.exists()
            with open(src_file) as s_file, open(tgt_file) as t_file:
                try:
                    for ss, tt in zip(s_file, t_file):
                        if (ss, tt) not in test_data:
                            all_data.append((ss, tt))
                except Exception as e:
                    logger.info(f"{src_file}, {tgt_file}, {type(e)}, {str(e)}")
                    continue

        valid_data = random.choices(all_data, k=test_length)
        valid_src =  lpair_dir / f"opus.{lpair}-dev.{src}"
        valid_tgt =  lpair_dir / f"opus.{lpair}-dev.{tgt}"
        with open(valid_src, 'w') as src_file, open(valid_tgt, 'w') as tgt_file:
            src_data, tgt_data = list(zip(*valid_data))
            src_file.writelines('\n'.join(src_data))
            tgt_file.writelines('\n'.join(tgt_data))



