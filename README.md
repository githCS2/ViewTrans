# ViewTrans

**View-Tokenized Transformer for Sinogram Restoration in Sparse-View CT**

---

## 🧠 Overview

**ViewTrans** is a novel deep learning framework for restoring full-view sinograms from sparse-view inputs in computed tomography (CT).

Unlike conventional vision Transformers that tokenize 2D images by patches, **ViewTrans introduces a _view-tokenized_ strategy**, treating each fan-beam projected vector as a token — fully consistent with the physics of CT scanning.

**Key Features:**
- Physics-aware tokenization based on fan-beam views
- Works in the sinogram domain
- Can be paired with:
  - Traditional **FBP layer** for end-to-end sparse-view CT reconstruction
  - Dual-domain pipelines for improved reconstruction quality

---

## ⚙️ Requirements

- Python 3.8 or higher  
- PyTorch 1.11.0 or higher  

## 📁 Dataset Acquisition

Download the datasets from the following link:

🔗 [Google Drive - ViewTrans Dataset](https://drive.google.com/drive/folders/19nDIF-LnSjBrPtOfsnwXlJ5-4HOfNr4i?usp=drive_link)

Then place the `.npz` files into the `./data` directory:


---

## 🚀 Training & Testing

### Train from Scratch

```bash
python train.py
```

###Evaluate on a Sample Image
```bash
python test.py
```

The pretrained model is included at:
./weights/ct_predict_test.pth

## 🧩 System Matrix

The file `Matrix_A.npz` is a precomputed system matrix used in the reconstruction pipeline.

It was generated using the `Generate_A_Matrrx` function from this repository:

🔗 [https://github.com/githCS2/FP_FBP](https://github.com/githCS2/FP_FBP)

