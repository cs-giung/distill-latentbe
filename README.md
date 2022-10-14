# Improving Ensemble Distillation With Weight Averaging and Diversifying Perturbation

This repository is the official implementation of [Improving Ensemble Distillation With Weight Averaging and Diversifying Perturbation (ICML 2022)](https://proceedings.mlr.press/v162/nam22a.html).

## Requirements

This project is built on [giung2-jax](https://github.com/cs-giung/giung2-jax).
```
ln -s /path/to/giung2-jax/giung2 ./
ln -s /path/to/giung2-jax/datasets ./
```

## Command Lines

### Train teacher models

* Teacher (DE-4)

    ```
    python scripts/train.py --config_file ./configs/C100_WRN28x4.yaml --seed 42 --output_dir ./outputs/C100_WRN28x4/SGD/0/
    python scripts/train.py --config_file ./configs/C100_WRN28x4.yaml --seed 43 --output_dir ./outputs/C100_WRN28x4/SGD/1/
    python scripts/train.py --config_file ./configs/C100_WRN28x4.yaml --seed 44 --output_dir ./outputs/C100_WRN28x4/SGD/2/
    python scripts/train.py --config_file ./configs/C100_WRN28x4.yaml --seed 45 --output_dir ./outputs/C100_WRN28x4/SGD/3/
    ```

### Train student models

* Student distilled from DE-4 (KD)

    ```
    python scripts/train_KD.py
        --config_file ./configs/C100_WRN28x4.yaml
        --seed 142
        --output_dir ./outputs/C100_WRN28x4/KD_DE4/0/
        --teacher_config_file ./configs/C100_WRN28x4.yaml
        --teacher_weight_file ./outputs/C100_WRN28x4/SGD/0/best_acc1
        --teacher_n_ensembles 4 --kd_alpha 1.0 --kd_temperature 4.0
    ```

* Student distilled from DE-4 (KD + LatentBE)

    ```
    python scripts/train_KD_LatentBE.py
        --config_file ./configs/C100_WRN28x4_LatentBE4.yaml
        --seed 142
        --output_dir ./outputs/C100_WRN28x4/KD_DE4_LatentBE/0/
        --teacher_config_file ./configs/C100_WRN28x4.yaml
        --teacher_weight_file ./outputs/C100_WRN28x4/SGD/0/best_acc1
        --teacher_n_ensembles 4 --kd_alpha 1.0 --kd_temperature 4.0
    ```

* Student distilled from DE-4 (KD + LatentBE + TDiv-SDiv)

    ```
    python scripts/train_KD_LatentBE_TDivSDiv.py
        --config_file ./configs/C100_WRN28x4_LatentBE4.yaml
        --seed 142
        --output_dir ./outputs/C100_WRN28x4/KD_DE4_LatentBE_TDivSDiv/0/
        --teacher_config_file ./configs/C100_WRN28x4.yaml
        --teacher_weight_file ./outputs/C100_WRN28x4/SGD/0/best_acc1
        --teacher_n_ensembles 4 --kd_alpha 1.0 --kd_temperature 4.0
    ```

### Evaluation Results

Using [`scripts/evaluation.ipynb`](./scripts/evaluation.ipynb), we get the following results (student values are averaged over 4 seeds):

| Method             | ACC    | NLL    | ECE    | cNLL   | cECE   |
| :-                 | :-:    | :-:    | :-:    | :-:    | :-:    |
| Teacher (DE-4)     | 81.08  | 0.715  | 0.030  | 0.709  | 0.018  |
| Student (KD)       | 79.17  | 1.072  | 0.134  | 0.871  | 0.044  |
| + LatentBE (Ours)  | 79.32  | 1.012  | 0.127  | 0.843  | 0.046  |
| + TDiv-SDiv (Ours) | **80.09**  | **0.798**  | **0.071**  | **0.770**  | **0.041**  |

## Citation

If you find this useful in your research, please consider citing our paper:
```
@inproceedings{nam2022improving,
  title     = {Improving Ensemble Distillation With Weight Avearging and Diversifying Perturbation},
  author    = {Giung Nam and Hyungi Lee and Byeongho Heo and Juho Lee},
  booktitle = {Proceedings of The 39th International Conference on Machine Learning (ICML 2022)},
  year      = {2022}
}
```

## License

[The MIT License](./LICENSE).
