# Custom dataset workflow

The repository contains two independent datasets:

- TrashNet: `data/raw/dataset-resized/`
- Custom: `data/custom_images/train/<class>/`

The Custom dataset currently contains 50 registered PNGs: 10 each in cardboard, glass, metal, paper, and plastic. There are no training examples in a trash folder. In the strict Custom GUI mode, any upload whose canonical decoded RGB pixels do not match one of those registered images is assigned to `trash`. Filenames and file metadata do not matter; resizing, cropping, or changing pixels does change the hash and will therefore use the fallback.

**Do not describe this as 100% ML model accuracy.** The exact-image registry provides deterministic lookup for the registered pixels. The custom ML models are trained and evaluated separately and can still make mistakes.

## Prepare manifests

```fish
python src/prepare_data.py --dataset custom
python src/prepare_data.py --dataset trashnet
```

Custom manifests are saved under `data/processed/custom/`; existing TrashNet manifests remain under `data/processed/`.

## Train the four dataset/model combinations

```fish
python src/train.py --dataset custom --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/train.py --dataset custom --model efficientnetb0 --head-epochs 5 --fine-tune-epochs 3
python src/train.py --dataset trashnet --model mobilenetv2 --head-epochs 5 --fine-tune-epochs 3
python src/train.py --dataset trashnet --model efficientnetb0 --head-epochs 5 --fine-tune-epochs 3
```

Custom checkpoints use the `custom_` prefix, so they do not overwrite the existing TrashNet checkpoints.

## Evaluate and launch

```fish
python src/evaluate.py --dataset custom --model mobilenetv2
python src/evaluate.py --dataset custom --model efficientnetb0
python src/evaluate.py --dataset trashnet --model mobilenetv2
python src/evaluate.py --dataset trashnet --model efficientnetb0
python src/app.py
```

The GUI provides independent Dataset and Model selectors. In Custom mode it displays the strict registry result and the selected model prediction separately. The registry self-check in evaluation reports whether the saved custom images resolve to their assigned folder labels; it is not a generalization metric.
