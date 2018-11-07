import sys, os

parent_path = os.path.dirname(os.path.abspath('./'))

configs = {
    "database_name": "stockdb",
    "exported_model_path" : "%s/prediction-model/" %parent_path,
    "databse_path": "%s/stock-database/" %parent_path,
    "stocker_path": "%s/stocker-logic" %parent_path
}

for key, path in configs.items():
        sys.path.append(path)