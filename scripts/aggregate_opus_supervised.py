#!/usr/bin/env python3

import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    lang = ['af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'eo', 'es', 'et',
            'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'hu', 'id', 'ig', 'is', 'it', 'ja',
            'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'li', 'lt', 'lv', 'mg', 'mk', 'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'ne',
            'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'rw', 'se', 'sh', 'si', 'sk', 'sl', 'sq', 'sr',
            'sv', 'ta', 'te', 'tg', 'th', 'tk', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'wa', 'xh', 'yi', 'zh', 'zu']

    high_langs = {'nl', 'ca', 'fi', 'mk', 'da', 'cs', 'bg', 'ro', 'is', 'th', 'he', 'uk', 'lv', 'pl', 'pt', 'hu', 'de', 'lt',
                  'si', 'ms', 'sv', 'tr', 'ko', 'sq', 'el', 'fa', 'es', 'zh', 'bs', 'ar', 'eu', 'fr', 'bn', 'it', 'sk', 'sr',
                  'et', 'vi', 'mt', 'no', 'sl', 'id', 'ja', 'ru', 'hr'}
    med_langs = {'ga', 'af', 'tg', 'gu', 'km', 'sh', 'hi', 'rw', 'nb', 'wa', 'uz', 'ka', 'ml', 'ur', 'gl', 'br', 'cy', 'ku',
                 'ne', 'pa', 'mg', 'as', 'eo', 'xh', 'nn', 'ta', 'az', 'tt'}
    low_langs = {'fy', 'mr', 'tk', 'kn', 'li', 'yi', 'my', 'zu', 'ug', 'or', 'se', 'am', 'oc', 'ig', 'ha', 'ky', 'te', 'be',
                 'kk', 'gd', 'ps'}

    high_bleu, med_bleu, low_bleu = [], [], []
    for lan in lang:
        with open(os.path.join(args.path, 'test.%s-en.en.bleu' % lan)) as file:
            bleu_str = file.readlines()

        bleu_str = bleu_str[0].strip().split()[2]
        bleu = float(bleu_str)
        if lan in high_langs:
            high_bleu.append(bleu)
        elif lan in med_langs:
            med_bleu.append(bleu)
        else:
            low_bleu.append(bleu)

    avg_bleu = sum(high_bleu + med_bleu + low_bleu) / 94
    high_bleu, med_bleu, low_bleu = list(map(lambda x: sum(x) / len(x), [high_bleu, med_bleu, low_bleu]))
    print("XX-En:", "high_bleu: %.2f" % high_bleu, "med_bleu: %.2f" % med_bleu, "low_bleu: %.2f" % low_bleu,
          "avg: %.2f" % avg_bleu )

    high_bleu, med_bleu, low_bleu = [], [], []
    for lan in lang:
        with open(os.path.join(args.path, 'test.en-%s.%s.bleu' % (lan, lan))) as file:
            bleu_str = file.readlines()

        bleu_str = bleu_str[0].strip().split()[2]
        bleu = float(bleu_str)
        if lan in high_langs:
            high_bleu.append(bleu)
        elif lan in med_langs:
            med_bleu.append(bleu)
        else:
            low_bleu.append(bleu)

    avg_bleu = sum(high_bleu + med_bleu + low_bleu) / 94
    high_bleu, med_bleu, low_bleu = list(map(lambda x: sum(x) / len(x), [high_bleu, med_bleu, low_bleu]))
    print("En-XX:", "high_bleu: %.2f" % high_bleu, "med_bleu: %.2f" % med_bleu, "low_bleu: %.2f" % low_bleu,
          "avg: %.2f" % avg_bleu )

