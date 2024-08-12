## Introduction
This repo has the implementation of Language-informed Beam Search (LiBS) from [Language-Informed Beam Search Decoding for Multilingual Machine Translation]().

This repo also has a re-implementation of [Improving Multilingual Translation by Representation and Gradient Regularization](https://arxiv.org/abs/2109.04778) on the new [Fairseq](https://github.com/pytorch/fairseq) codebase. 
Compared to my old codebase, this version is significantly faster while results (i.e. BLEU score) are largely the same.

This release includes:
* A rebuilt OPUS-100 dataset 
* Implementation of an auxiliary LangID prediction loss (i.e. TLP)
* Implementation of on-the-fly oracle gradient de-confliction (i.e. TGP)
* Implementation of Language-informed Beam Search (LiBS)

## Rebuilt OPUS-100 dataset 
First, run this script:
```python
python scripts/rebuilt_opus_dataset.py $DOWNLOAD_DIR
```
Above "rebuilt_opus_dataset.py" script does the following steps:
1. First it downloads and unzips the original [OPUS-100 V1.0 dataset](https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz) into your desired directory $DOWNLOAD_DIR
2. Then it de-duplicates the supervised dataset and re-samples the zeroshot dev set. 
(This step would take 3~4 hours to finish, and feel free to purge "$DOWNLOAD_DIR/opus-100-corpus/v1.0/zero-shot/??-??/downloaded" after finished, which includes all the downloaded zero-shot corpus from OPUS.)

Then, run the following bash commands, which move OPUS data into your desired location $DATA_DIR, and name files into {split}.{lang_pair}.{lang} format.
```
mkdir -p ${DATA_DIR}/raw
for lang in af am ar as az be bg bn br bs ca cs cy da de el; do
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-train-rebuilt.${lang} ${DATA_DIR}/raw/train.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-train-rebuilt.en ${DATA_DIR}/raw/train.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-dev-rebuilt.${lang} ${DATA_DIR}/raw/valid.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-dev-rebuilt.en ${DATA_DIR}/raw/valid.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-test.${lang} ${DATA_DIR}/raw/test.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-test.en ${DATA_DIR}/raw/test.en-${lang}.en
done

for lang in eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-train-rebuilt.${lang} ${DATA_DIR}/raw/train.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-train-rebuilt.en ${DATA_DIR}/raw/train.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-dev-rebuilt.${lang} ${DATA_DIR}/raw/valid.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-dev-rebuilt.en ${DATA_DIR}/raw/valid.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-test.${lang} ${DATA_DIR}/raw/test.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-test.en ${DATA_DIR}/raw/test.en-${lang}.en
done
```

## Pre-processing Supervised Data
We use sentencepiece to tokenize the dataset.
Following commands run spm_train, spm_encode, and Fairseq preprocessing steps:
```bash
python scripts/spm_train.py --input=$(echo $(ls ${DATA_DIR}/raw/train*) | sed 's/ /,/g') --model_prefix=${DATA_DIR}/spm_64k --vocab_size=64000 --character_coverage=1.0 --input_sentence_size=1000000

for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/train.en-${lang}.en --outputs ${DATA_DIR}/train.en-${lang}.en
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/train.en-${lang}.${lang} --outputs ${DATA_DIR}/train.en-${lang}.${lang}
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/valid.en-${lang}.en --outputs ${DATA_DIR}/valid.en-${lang}.en
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/valid.en-${lang}.${lang} --outputs ${DATA_DIR}/valid.en-${lang}.${lang}    
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/test.en-${lang}.en --outputs ${DATA_DIR}/test.en-${lang}.en
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/test.en-${lang}.${lang} --outputs ${DATA_DIR}/test.en-${lang}.${lang}
done

mkdir -p ${DATA_DIR}/data-bin
cut -f 1 ${DATA_DIR}/spm_64k.vocab | tail -n +4 | sed "s/$/ 100/g" > ${DATA_DIR}/data-bin/dict.txt

for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    python fairseq_cli/preprocess.py --task "translation" --source-lang $lang --target-lang en \
    --trainpref ${DATA_DIR}/train.en-${lang} --validpref ${DATA_DIR}/valid.en-${lang} \
    --destdir ${DATA_DIR}/data-bin --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
    --srcdict ${DATA_DIR}/data-bin/dict.txt --tgtdict ${DATA_DIR}/data-bin/dict.txt
done
```

## Pre-processing Zero-shot data
Following steps extract and spm_encode the re-sampled zeroshot dev set as well as the original test set:
```bash
for lpair in de-nl nl-zh ar-nl ru-zh fr-nl de-fr fr-zh ar-ru ar-zh ar-fr de-zh fr-ru de-ru nl-ru ar-de; do
    IFS=- read -r SRC TGT <<< ${lpair}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/zero-shot/${SRC}-${TGT}/opus.${SRC}-${TGT}-dev.${SRC} ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${SRC}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/zero-shot/${SRC}-${TGT}/opus.${SRC}-${TGT}-dev.${TGT} ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${TGT}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/zero-shot/${SRC}-${TGT}/opus.${SRC}-${TGT}-test.${SRC} ${DATA_DIR}/raw/test.${SRC}-${TGT}.${SRC}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/zero-shot/${SRC}-${TGT}/opus.${SRC}-${TGT}-test.${TGT} ${DATA_DIR}/raw/test.${SRC}-${TGT}.${TGT}
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${TGT} --outputs ${DATA_DIR}/valid.${SRC}-${TGT}.${TGT}
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${SRC} --outputs ${DATA_DIR}/valid.${SRC}-${TGT}.${SRC}
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/test.${SRC}-${TGT}.${SRC} --outputs ${DATA_DIR}/test.${SRC}-${TGT}.${SRC}
    python scripts/spm_encode.py --model ${DATA_DIR}/spm_64k.model --input ${DATA_DIR}/raw/test.${SRC}-${TGT}.${TGT} --outputs ${DATA_DIR}/test.${SRC}-${TGT}.${TGT}
done
```

## Pre-processing Oracle data
Following steps build the oracle data, with a 80-20 split into oracle set (for oracle gradients) and validation set (for checkpoint selection):
```bash
for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    cat ${DATA_DIR}/valid.en-${lang}.en >> ${DATA_DIR}/oracle_data.en 
    cat ${DATA_DIR}/valid.en-${lang}.${lang} >> ${DATA_DIR}/oracle_data.oracle_en
    cat ${DATA_DIR}/valid.en-${lang}.en >> ${DATA_DIR}/oracle_data.oracle_${lang}   
    cat ${DATA_DIR}/valid.en-${lang}.${lang} >> ${DATA_DIR}/oracle_data.${lang}
done

for lpair in de-nl nl-zh ar-nl ru-zh fr-nl de-fr fr-zh ar-ru ar-zh ar-fr de-zh fr-ru de-ru nl-ru ar-de; do
    IFS=- read -r SRC TGT <<< ${lpair}
    cat ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${SRC} >> ${DATA_DIR}/oracle_data.${SRC}
    cat ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${TGT} >> ${DATA_DIR}/oracle_data.oracle_${SRC}
    cat ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${SRC} >> ${DATA_DIR}/oracle_data.oracle_${TGT}
    cat ${DATA_DIR}/raw/valid.${SRC}-${TGT}.${TGT} >> ${DATA_DIR}/oracle_data.${TGT}
done

# "split_oracle_data.py" does the 80-20 random split
python scripts/split_oracle_data.py ${DATA_DIR}/oracle_data

for lang in af am ar as az be bg bn br bs ca cs cy da de el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi zh zu; do
    python fairseq_cli/preprocess.py --task "translation" --source-lang oracle_${lang} --target-lang ${lang} \
    --validpref ${DATA_DIR}/valid --testpref ${DATA_DIR}/test \
    --destdir ${DATA_DIR}/data-bin/ --dataset-impl 'mmap' --padding-factor 1 --workers 32 \
    --srcdict ${DATA_DIR}/dict.en.txt --tgtdict ${DATA_DIR}/dict.en.txt
done
```

## Baseline model
To replicate our results, we train our model on 8 V100 gpus with 16 gradient accumulation steps to simulate the training on 128 V100 gpus:
```bash
python train.py $DATA_DIR --arch transformer_vaswani_wmt_en_de_big \
    --task translation_multi_simple_epoch --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok "tgt" \
    --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
    --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 50000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 4096 --update-freq 16 \
    --save-interval-updates 1000 --keep-interval-updates 20 --no-epoch-checkpoints \
    --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d \
    --save-dir checkpoints/opus_base --max-source-positions 256 --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test |& tee checkpoints/opus_base/train.log
```

## TLP algorithm
Similarly, we train TLP model with 16 gradient accumulation steps on 8 V100 gpus:
```bash
python train.py $DATA_DIR --arch transformer_langid_pred_vaswani_wmt_en_de_big \
    --task translation_multi_simple_epoch --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok "tgt" \
    --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
    --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
    --criterion label_smoothed_cross_entropy_langid_pred --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 50000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 4096 --update-freq 16 \
    --save-interval-updates 1000 --keep-interval-updates 20 --no-epoch-checkpoints \
    --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d \
    --save-dir checkpoints/opus_langid --max-source-positions 256 --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test --langid-pred-coef 0.1 --enable-lang-ids |& tee checkpoints/opus_langid/train.log
```

## TGP algorithm
To speedup TGP training, we first re-batch training samples and grouped by their target languages. 
We further merge training batches that consist of only English-centric dev sets, since it empirically
obtains similar performance while exhibiting noticeable speedups 
(if we don't do this step, the training process could be challenging for CPU memories, since we are saving on-the-fly oracle gradients.).

```bash
python train.py $DATA_DIR --arch transformer_vaswani_wmt_en_de_big \
    --task translation_multi_tgp --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok "tgt" \
    --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
    --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
    --tgp-group-langs "af,am,as,az,be,bg,bn,br,bs,ca,cs,cy,da,el,eo,es,et,eu,fa,fi,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nn,no,oc,or,pa,pl,ps,pt,ro,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zu" \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr 0.0005 --warmup-updates 4000 --max-update 50000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 65536 \
    --save-interval-updates 1000 --keep-interval-updates 20 --no-epoch-checkpoints \
    --log-format simple --log-interval 100 --seed 1234 --fp16 --batch-by-target \
    --ddp-backend no_c10d --restore-file checkpoints/opus_base/checkpoint_7_40000.pt \
    --save-dir checkpoints/opus_tgp --max-source-positions 256 --max-target-positions 256 \
    --skip-invalid-size-inputs-valid-test --reset-dataloader \
    --tgp-max-tokens 4096 --max-tokens-valid 4096 |& tee checkpoints/opus_tgp/train.log
```

## Inference and Evaluation
We use "fairseq_cli/interactive.py" for inference and sacrebleu for evaluation (following example works for De->Fr):
```bash
SRC=de
TGT=fr
FSRC=${DATA_DIR}/test.${SRC}-${TGT}.${SRC}
FTGT=${DATA_DIR}/raw/test.${SRC}-${TGT}.${TGT}
FOUT=${MODEL_PATH}_trans/test.${SRC}-${TGT}.${TGT}
mkdir ${MODEL_PATH}_trans

cat $FSRC | python scripts/truncate.py | \
python fairseq_cli/interactive.py ${DATA_DIR}/data-bin \
    --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH \
    --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
    --lang-pairs es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig \
    --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
    --beam 5 --lenpen 1.0 --post-process=sentencepiece --no-progress-bar | \
grep -P "^H" | cut -f 3- > $FOUT

cat $FOUT | python scripts/sacrebleu.py $FTGT | grep "BLEU"
```

## Language-informed Beam Search (LiBS)
First you'll need to download the LiD model from [fasttext](https://fasttext.cc/docs/en/language-identification.html), we are using lid.176.bin model.
To ran LiBS algorthm:
``` bash
SRC=de
TGT=fr
FSRC=${DATA_DIR}/test.${SRC}-${TGT}.${SRC}
FTGT=${DATA_DIR}/raw/test.${SRC}-${TGT}.${TGT}
FOUT=${MODEL_PATH}_trans/test.${SRC}-${TGT}.${TGT}
mkdir ${MODEL_PATH}_trans

cat $FSRC | python scripts/truncate.py | \
python fairseq_cli/interactive.py ${DATA_DIR}/data-bin \
    --task translation_multi_simple_epoch --encoder-langtok tgt --path $MODEL_PATH \
    --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
    --lang-pairs es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig \
    --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 100 \
    --beam 5 --lenpen 1.0 --post-process=sentencepiece --no-progress-bar \
    --eval-langid --langid-modeldir lid.176.bin --langid-multiprocess --langid-beam --langid-rerank --langid-beam-k 2 --langid-coef 1.8 \
    --lid-target-lang $TGT | \
grep -P "^H" | cut -f 3- > $FOUT

cat $FOUT | python scripts/sacrebleu.py $FTGT | grep "BLEU"
```


## Citation
If you find this repository helpful, please cite us, thanks!
```bibtex
@inproceedings{yang2021improving,
  title={Improving Multilingual Translation by Representation and Gradient Regularization},
  author={Yang, Yilin and Eriguchi, Akiko and Muzio, Alexandre and Tadepalli, Prasad and Lee, Stefan and Hassan, Hany},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={7266--7279},
  year={2021}
}
```
