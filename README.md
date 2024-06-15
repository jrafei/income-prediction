# AI28-Income Prediction

## Getting started
```
cd existing_repo
git remote add origin https://gitlab.utc.fr/jeltayeb/ai28-income-prediction.git
git branch -M main
git push -uf origin main
```

## Potential issues
L'issue suivant est à cause d'une version obsolète de imbalanced-learn (cf. https://stackoverflow.com/questions/76593906/how-to-resolve-cannot-import-name-missingvalues-from-sklearn-utils-param-v) :
```
cannot import name '_MissingValues' from 'sklearn.utils._param_validation' (/opt/homebrew/anaconda3/lib/python3.11/site-packages/sklearn/utils/_param_validation.py)
```
Il suffit de re-installer cette librairie avec :
```
pip uninstall imbalanced-learn -y; pip install imbalanced-learn
```

