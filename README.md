ViewTrans

View-Tokenized Transformer for Sinogram Restoration in Sparse-View CT


Overview:
ViewTrans is a novel deep learning framework for restoring full-view sinograms from sparse-view inputs in computed tomography (CT). Unlike conventional vision Transformers that tokenize 2D images by patches, ViewTrans introduces a view-tokenized strategy, treating each fan-beam projected vector as a token — fully consistent with the physics of CT scanning. It can either be paired with a traditional FBP layer for end-to-end sparse-view CT reconstruction, or serve as a versatile component within dual-domain reconstruction pipelines, operating specifically in the sinogram domain.
Requirements:
Python 3.8 or higher
Pytorch 1.11.0 or higher

Dataset Acquisition:
You can download datasets used in this project from https://drive.google.com/drive/folders/19nDIF-LnSjBrPtOfsnwXlJ5-4HOfNr4i?usp=drive_link, then put the train and test data (npz files) in the subdirectory ./data.

Training & Testing:
You can train the model from scratch by running train.py, or directly evaluate the reconstruction performance on a sample image using test.py. The pretrained model is already uploaded in the subdirectory ./weights/ct_predict_test.pth.

The file Matrix_A.npz is a precomputed system matrix generated using Generate_A_Matrrx from the https://github.com/githCS2/FP_FBP.
