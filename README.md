# LeafLens

This updates the existing Flask application. It preserves the crop datasets and streams image batches from the ten named dataset folders; it does not scan `.venv`, `models`, or other project files as training data.

## Train the models (Windows PowerShell)

From `C:\Users\asus\Desktop\diseasedetect`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py --data "C:\Users\asus\Desktop\diseasedetect" --epochs 12 --batch-size 16
```

The pipeline has three sequential models: MobileNetV2 leaf/non-leaf validation, an independent nine-class visual crop classifier, and a crop/disease classifier. Images are RGB, EXIF-oriented, resized to 224×224 and normalized to [0,1]. The loader shuffles path/label records, decodes one image at a time, batches, and prefetches one batch; it does not cache full images in RAM. Augmentation runs online. Batch size and epoch count are configurable.

Each stage writes a best-epoch checkpoint, CSV training log and interruption-recovery state. Rerun the same training command to resume an interrupted stage. Final model files and class maps are in `models`: `leaf_gate.keras`/`leaf_classes.json`, `crop_classifier.keras`/`crop_classes.json`, and `disease_classifier.keras`/`disease_classes.json`.

A full scan of the ten named dataset roots found 34,953 readable unique leaf images and 28 readable unique non-leaf images (2 JPG, 26 PNG), for 34,981 unique images total. It also found 12,803 exact duplicate copies, which the loader skips before splitting; no unreadable supported images were reported. The non-leaf directory has no nested folders today, and scanning is recursive so nested additions will be included. Restricting discovery to those named roots prevents `.venv` test fixtures from becoming bogus `Unknown` crop labels. Twenty-eight negative examples are not enough to establish reliable rejection for arbitrary people, animals, objects or backgrounds. Add varied negative examples and evaluate before relying on the gate. Confidence and margin cutoffs are configurable with `LEAF_THRESHOLD`, `CROP_THRESHOLD`, `CROP_MARGIN`, `DISEASE_THRESHOLD`, `DISEASE_MARGIN`, and `DISEASE_CROP_MASS` (default 0.25); scores are model outputs, not calibrated probabilities or diagnostic guarantees.

## Inspect and evaluate

```powershell
python inspect_dataset.py
python evaluate.py --batch-size 16
```

Evaluation reports accuracy, precision, recall, F1, and confusion matrices for held-out leaf, crop, and disease samples. It uses the same deterministic split seed as training. Separate external images are still needed to measure real-world generalization, especially for non-leaf and unknown-crop rejection.

## Run the Flask app

```powershell
python app.py
```

Open `http://127.0.0.1:5000`. The result flow checks leaf → crop → disease. Unknown crop/disease cases return “Uncertain Disease” with no fabricated disease description. Crop Wikipedia links use the required crop-specific article titles; disease lookups are shown only when the article title contains the predicted disease terms. Wikipedia uses the MediaWiki query API and remains optional; lookup failures do not affect prediction.
