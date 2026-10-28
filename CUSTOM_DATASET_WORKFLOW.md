# Custom dataset workflow: memorize all supplied images

This repository has two independent datasets:

- TrashNet: \`data/raw/dataset-resized/\`
- Custom: \`data/custom_images/train/<class>/\`

**Custom mode intentionally does not split the dataset.** Every labelled image is used for training, and evaluation measures how well the model fits those same images. This is a training-set memorization check, not a measure of generalization.

The Custom dataset currently has 50 images: 10 each in cardboard, glass, metal, paper, and plastic. It has no labelled \`trash\` examples, so the Custom neural networks learn only those five classes. In the GUI, the separate strict Custom rule returns the folder label for an exact decoded-pixel match and \`trash\` for any unregistered image. This fallback is a policy, not evidence that an unknown image is truly trash.

## Prepare manifests

\`\`\`bash
python src/prepare_data.py --dataset custom
python src/prepare_data.py --dataset trashnet
\`\`\`

Custom preparation writes all 50 images to \`data/processed/custom/train.csv\` and removes stale custom \`val.csv\` / \`test.csv\` files from the previous split-based workflow. TrashNet keeps its existing stratified train/validation/test split.

## Train Custom models on 100% of the labelled images

For more time to fit the small dataset, use higher epoch counts:

\`\`\`bash
python src/train.py --dataset custom --model mobilenetv2 --head-epochs 40 --fine-tune-epochs 30
python src/train.py --dataset custom --model efficientnetb0 --head-epochs 40 --fine-tune-epochs 30
\`\`\`

Custom mode disables data augmentation and dropout to make memorization easier. The same 50 images are used for training and training-set monitoring. This can overfit by design, but 100% neural-network accuracy is not guaranteed.

Custom checkpoints use the \`custom_\` prefix and do not overwrite TrashNet checkpoints.

## Evaluate Custom memorization and standard TrashNet testing

\`\`\`bash
python src/evaluate.py --dataset custom --model mobilenetv2
python src/evaluate.py --dataset custom --model efficientnetb0
python src/evaluate.py --dataset trashnet --model mobilenetv2
python src/evaluate.py --dataset trashnet --model efficientnetb0
\`\`\`

Custom evaluation reads \`data/processed/custom/train.csv\` and explicitly reports **training-set memorization accuracy**, not held-out test accuracy.

## Launch GUI

\`\`\`bash
python src/app.py
\`\`\`

Choose Dataset and Model independently. In Custom mode, the GUI displays the strict exact-image registry result separately from the model prediction. Canonical pixel matching ignores filenames and metadata, but resizing, cropping, recompressing, or editing pixels can make an image no longer match the registered hash.
