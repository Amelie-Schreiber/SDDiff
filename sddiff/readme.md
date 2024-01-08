# Molecular Conformation Generation via Shifting Scores

This is the official implementation of our paper: Molecular Conformation Generation via Shifting Scores

This code is forked from the official implementation of GeoDiff https://github.com/MinkaiXu/GeoDiff.git. Thanks for their contribution.

## Environment
The dependency can be found in `environment.yml`.

If get the error `AttributeError: module 'setuptools._distutils' has no attribute 'version'`, please try to use `setuptools==59.5.0`:
```
pip uninstall setuptools
pip install setuptools==59.5.0
```

## Data
We use the same dataset with GeoDiff. Please download from the [Google Drive](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing) the author offered. See also the data folder. 

## Train
Please train the model via:
```
python train.py configs/drugs_default.yml
python train.py configs/qm9_default.yml
```

The outputs will be saved in the `logs` folder (default). Please change the path through `--logdir YOUR_OUTPUT_FOLDER`

## Generation
Please generate conformations via:
```
python test.py WORKDIR/checkpoints/YOUR_MODEL.pt
```

For example, we provide our trained model for GEOM_Drugs dataset: `checkpoints/drugs.pt`. Please generate conformation via:
```
python test.py workdir_drugs/checkpoints/drugs.pt --step_lr 1.8e-6
```
the generated conformation will be saved in `workdir_drugs/sample_xxx` 

## Evaluation
Please calculate the `COV` and `MAT` scores via:
```
python eval_covmat.py WORKDIR/sample_all.pkl
```