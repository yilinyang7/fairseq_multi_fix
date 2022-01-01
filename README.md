# Improving Multilingual Translation by Representation and Gradient Regularization

## Introduction
This repo is a re-implementation of [Improving Multilingual Translation by Representation and Gradient Regularization](https://arxiv.org/abs/2109.04778) on the new [Fairseq](https://github.com/pytorch/fairseq) codebase. 
Compared to my old codebase, this version is significantly faster while results (BLEU score) are largely the same.

This release includes:
* A rebuilt OPUS-100 dataset 
* Implementation of an auxiliary LangID prediction loss (i.e. TLP)
* Implementation of on-the-fly oracle gradient de-confliction (i.e. TGP)

## Rebuilt OPUS-100 dataset 
1. First download the original [OPUS-100 V1.0 dataset](https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz) 
2. Run "tar xf opus-100-corpus-v1.0.tar.gz" to extract the tarball into your desired directory $DOWNLOAD_DIR
3. Run below script to de-duplicate the supervised dataset and re-sample the zeroshot dev set. 
(This script would take 3~4 hours to finish, and feel free to purge "$DOWNLOAD_DIR/opus-100-corpus/v1.0/zero-shot/??-??/downloaded" after finished, which includes all the downloaded zero-shot corpus from OPUS.)
```python
python scripts/rebuilt_opus_dataset.py $DOWNLOAD_DIR
```
4. Move OPUS data into your desired location $DATA_DIR, and name files into {split}.{lang_pair}.{lang} format.
```
mkdir -p ${DATA_DIR}/raw
for lang in af am an ar as az be bg bn br bs ca cs cy da de dz el; do
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-train-rebuilt.${lang} ${DATA_DIR}/raw/train.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-train-rebuilt.en ${DATA_DIR}/raw/train.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-dev-rebuilt.${lang} ${DATA_DIR}/raw/valid.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-dev-rebuilt.en ${DATA_DIR}/raw/valid.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-test.${lang} ${DATA_DIR}/raw/test.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/${lang}-en/opus.${lang}-en-test.en ${DATA_DIR}/raw/test.en-${lang}.en
done

for lang in eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu hy id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mn mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi yo zh zu; do
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-train-rebuilt.${lang} ${DATA_DIR}/raw/train.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-train-rebuilt.en ${DATA_DIR}/raw/train.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-dev-rebuilt.${lang} ${DATA_DIR}/raw/valid.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-dev-rebuilt.en ${DATA_DIR}/raw/valid.en-${lang}.en
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-test.${lang} ${DATA_DIR}/raw/test.en-${lang}.${lang}
    cp ${DOWNLOAD_DIR}/opus-100-corpus/v1.0/supervised/en-${lang}/opus.en-${lang}-test.en ${DATA_DIR}/raw/test.en-${lang}.en
done
```

## Pre-processing
We use sentencepiece to tokenize the dataset:

```bash
cd ${DATA_DIR}
python scripts/spm_train.py --input=$(echo $(ls raw/train****) | sed 's/ /;/g') --model_prefix=spm_64k --vocab_size=64000 --character_coverage=1.0 --input_sentence_size=1000000

for lang in af am an ar as az be bg bn br bs ca cs cy da de dz el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu hy id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mn mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi yo zh zu; do
    python spm_encode.py --model spm_64k.model --input raw/train.en-${lang}.en --outputs train.en-${lang}.en
    python spm_encode.py --model spm_64k.model --input raw/train.en-${lang}.${lang} --outputs train.en-${lang}.${lang}
    python spm_encode.py --model spm_64k.model --input raw/valid.en-${lang}.en --outputs valid.en-${lang}.en
    python spm_encode.py --model spm_64k.model --input raw/valid.en-${lang}.${lang} --outputs valid.en-${lang}.${lang}
done

cut -f 1 spm_64k.vocab | tail -n +4 | sed "s/$/ 100/g" > data-bin/dict.txt

for lang in af am an ar as az be bg bn br bs ca cs cy da de dz el eo es et eu fa fi fr fy ga gd gl gu ha he hi hr hu hy id ig is it ja ka kk km kn ko ku ky li lt lv mg mk ml mn mr ms mt my nb ne nl nn no oc or pa pl ps pt ro ru rw se sh si sk sl sq sr sv ta te tg th tk tr tt ug uk ur uz vi wa xh yi yo zh zu; do
    python fairseq_cli/preprocess.py --task "translation" --source-lang $lang --target-lang en --trainpref train.en-${lang} --validpref valid.en-${lang} --destdir data-bin --dataset-impl 'mmap' --padding-factor 1 --workers 32 --srcdict data-bin/dict.txt --tgtdict data-bin/dict.txt
done
```

## Baseline model
To replicate our results, we train our model on 8 V100 gpus with 16 gradient accumulation steps to simulate training on 128 V100 gpus:
```bash
python train.py $DATA_DIR --arch transformer_vaswani_wmt_en_de_big \
    --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
    --task translation_multi_simple_epoch --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok "tgt" \
    --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
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
To replicate our results:
```bash
python train.py $DATA_DIR --arch transformer_langid_pred_vaswani_wmt_en_de_big \
    --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
    --task translation_multi_simple_epoch --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok "tgt" \
    --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
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
To replicate our results:
```bash
python train.py $DATA_DIR --arch transformer_vaswani_wmt_en_de_big \
    --encoder-normalize-before --decoder-normalize-before --layernorm-embedding \
    --task translation_multi_tgp --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok "tgt" \
    --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" --lang-pairs "es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig" \
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


## Citation
If you find this repository helpful, please cite us:
```bibtex
@inproceedings{yang2021improving,
  title={Improving Multilingual Translation by Representation and Gradient Regularization},
  author={Yang, Yilin and Eriguchi, Akiko and Muzio, Alexandre and Tadepalli, Prasad and Lee, Stefan and Hassan, Hany},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={7266--7279},
  year={2021}
}
```

