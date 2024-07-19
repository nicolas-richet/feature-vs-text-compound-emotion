# [Text- and Feature-based Models for Compound Multimodal Emotion Recognition in the Wild](https://arxiv.org/pdf/2407.12927)


by
**Nicolas Richet<sup>1</sup>,
SoufianeBelharbi<sup>1</sup>,
Haseeb Aslam<sup>1</sup>,
Meike Emilie Schadt<sup>3</sup>,
Manuela González-González<sup>2,3</sup>,
Gustave Cortal<sup>4,6</sup>,
Alessandro Lameiras Koerich<sup>1</sup>,
Marco Pedersoli<sup>1</sup>,
Alain Finkel<sup>4,5</sup>,
Simon Bacon<sup>2,3</sup>,
Eric Granger<sup>1</sup>**

<sup>1</sup> LIVIA, Dept. of Systems Engineering, ÉTS, Montreal, Canada
<br/>
<sup>2</sup> Dept. of Health, Kinesiology \& Applied Physiology, Concordia University, Montreal, Canada
<br/>
<sup>3</sup> Montreal Behavioural Medicine Centre, Montreal, Canada
<br/>
<sup>4</sup> Université Paris-Saclay, CNRS, ENS Paris-Saclay, LMF, 91190, Gif-sur-Yvette, France
<br/>
<sup>5</sup> Institut Universitaire de France, France
<br/>
<sup>6</sup> Université Paris-Saclay, CNRS, LISN, 91400, Orsay, France

<p align="center"><img src="doc/promo.png" alt="outline" width="50%"></p>



## Abstract
Systems for multimodal Emotion Recognition (ER) commonly rely on features extracted from different modalities (e.g., visual, audio, and textual) to predict the seven basic emotions. However, compound emotions often occur in real-world scenarios and are more difficult to predict. Compound multimodal ER becomes more challenging in videos due to the added uncertainty of diverse modalities.
  In addition, standard features-based models may not fully capture the complex and subtle cues needed to understand compound emotions.
  Since relevant cues can be extracted in the form of text, we advocate for textualizing all modalities, such as visual and audio, to harness the capacity of large language models (LLMs). These models may understand the complex interaction between modalities and the subtleties of complex emotions. Although training an LLM requires large-scale datasets, a recent surge of pre-trained LLMs, such as BERT and LLaMA, can be easily fine-tuned for downstream tasks like compound ER.
  This paper compares two multimodal modeling approaches for compound ER in videos -- standard feature-based vs. text-based. Experiments were conducted on the challenging \cexprdb dataset for compound ER, and contrasted with results on the \meld dataset for basic ER.
  Our code for the Feature-based approach can be found at: [github.com/sbelharbi/feature-vs-text-compound-emotion](https://github.com/sbelharbi/feature-vs-text-compound-emotion). This repository is the text-based approach.

**This code is the text-based approach presented in the paper.**

**Code: Pytorch 2.2.2**, made for the [7th-ABAW challenge](https://affective-behavior-analysis-in-the-wild.github.io/7th/).


## Citation:
```
@article{Richet-abaw-24,
  title={Text- and Feature-based Models for Compound Multimodal Emotion Recognition in the Wild},
  author={Richet, N. and Belharbi, S. and Aslam, H. and Zeeshan, O. and Belharbi, S. and
  Koerich, A. L. and Pedersoli, M. and Bacon, S. and Granger, E.},
  journal={CoRR},
  volume={abs/2407.12927}
  year={2024}
}
```

## Installation of the environments
```bash
# Face cropping and alignment virtual env.
./create_v_env_face_extract.sh

# Pre-processing and training virtual env.
./create_v_env_main.sh
```

## Supported modalities:
- Vision: `vision`
- Audio: `vggish`
- Text: `bert`


## Training:
```bash
#!/usr/bin/env bash

source ~/venvs/abaw-7/bin/activate

# ==============================================================================
cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid


python main.py \
       --dataset_name MELD \
       --use_other_class False \
       --train_p 100.0 \
       --valid_p 100.0 \
       --test_p 100.0 \
       --amp True \
       --seed 0 \
       --mode TRAINING \
       --resume False \
       --modality video+vggish+bert+EXPR_continuous_label \
       --calc_mean_std True \
       --emotion LATER \
       --model_name LFAN \
       --num_folds 1 \
       --fold_to_run 0 \
       --num_heads 2 \
       --modal_dim 32 \
       --tcn_kernel_size 5 \
       --num_epochs 1000 \
       --min_num_epochs 5 \
       --early_stopping 50 \
       --window_length 300 \
       --hop_length 200 \
       --train_batch_size 2 \
       --eval_batch_size 1 \
       --num_workers 6 \
       --opt__weight_decay 0.0001 \
       --opt__name_optimizer SGD \
       --opt__lr 0.0001 \
       --opt__momentum 0.9 \
       --opt__dampening 0.0 \
       --opt__nesterov True \
       --opt__beta1 0.9 \
       --opt__beta2 0.999 \
       --opt__eps_adam 1e-08 \
       --opt__amsgrad False \
       --opt__lr_scheduler True \
       --opt__name_lr_scheduler MYSTEP \
       --opt__gamma 0.9 \
       --opt__step_size 50 \
       --opt__last_epoch -1 \
       --opt__min_lr 1e-07 \
       --opt__t_max 100 \
       --opt__mode MIN \
       --opt__factor 0.5 \
       --opt__patience 10 \
       --opt__gradual_release 1 \
       --opt__release_count 3 \
       --opt__milestone 0 \
       --opt__load_best_at_each_epoch True \
       --exp_id 07_17_2024_09_48_48_088070__8018213
```

## Testing on challenge CER:

```bash
#!/usr/bin/env bash

source ~/venvs/abaw-7/bin/activate

# ==============================================================================
cudaid=$1
export CUDA_VISIBLE_DEVICES=$cudaid

python inference_challenge.py \
       --mode EVALUATION \
       --case_best_model FRAMES_AVG_LOGITS \
       --target_ds_name C-EXPR-DB-CHALLENGE \
       --eval_set test \
       --fd_exp ABOLSUTE_APTH_TO_FOLDER_EXP
```

## Thanks
This code is heavily based on [github.com/sucv/ABAW3](https://github.com/sucv/ABAW3).
