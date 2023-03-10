import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection

def train_test_split(  dataset: pd.DataFrame,
                       target_col: str, 
                       test_size: float,
                       stratify: bool,
                       random_state: int) -> tuple[pd.DataFrame,pd.DataFrame,pd.Series,pd.Series]:
    
    features = dataset.drop(target_col, axis=1)
    target = dataset[target_col]

    
    if stratify:
        train_features, test_features, train_targets, test_targets = sklearn.model_selection.train_test_split(
            features, target, test_size=test_size, stratify=target, random_state=random_state)
    else:
        train_features, test_features, train_targets, test_targets = sklearn.model_selection.train_test_split(
            features, target, test_size=test_size, random_state=random_state)

    
    return train_features, test_features, train_targets, test_targets

class PreprocessDataset:
    def __init__(self, 
                 train_features:pd.DataFrame, 
                 test_features:pd.DataFrame,
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        self.train_features = train_features
        self.test_features = test_features
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions
        
    def one_hot_encode_columns_train(self) -> pd.DataFrame:
        one_hot_encoded_dataset = pd.get_dummies(self.train_features, columns=self.one_hot_encode_cols)
        return one_hot_encoded_dataset

    def one_hot_encode_columns_test(self) -> pd.DataFrame:
        one_hot_encoded_dataset = pd.get_dummies(self.test_features, columns=self.one_hot_encode_cols)
        return one_hot_encoded_dataset

    def min_max_scaled_columns_train(self) -> pd.DataFrame:
        scaler = MinMaxScaler()
        scaler.fit(self.train_features[self.min_max_scale_cols])
        min_max_scaled_data = scaler.transform(self.train_features[self.min_max_scale_cols])
        min_max_scaled_dataset = pd.DataFrame(min_max_scaled_data, columns=self.min_max_scale_cols)
        return min_max_scaled_dataset

    def min_max_scaled_columns_test(self) -> pd.DataFrame:
        scaler = MinMaxScaler()
        scaler.fit(self.test_features[self.min_max_scale_cols])
        min_max_scaled_data = scaler.transform(self.test_features[self.min_max_scale_cols])
        min_max_scaled_dataset = pd.DataFrame(min_max_scaled_data, columns=self.min_max_scale_cols)
        return min_max_scaled_dataset
    
    def pca_train(self) -> pd.DataFrame:
        pca = PCA(n_components=self.n_components)
        principal_components = pca.fit_transform(self.train_features)
        pca_dataset = pd.DataFrame(data = principal_components,
                                   columns = [f"component_{i+1}" for i in range(self.n_components)])
        return pca_dataset

    def pca_test(self) -> pd.DataFrame:
        pca = PCA(n_components=self.n_components)
        principal_components = pca.fit_transform(self.test_features)
        pca_dataset = pd.DataFrame(data = principal_components,
                                   columns = [f"component_{i+1}" for i in range(self.n_components)])
        return pca_dataset

    def feature_engineering_train(self) -> pd.DataFrame:
        feature_engineered_dataset = pd.DataFrame()
        for feature_name, feature_func in self.feature_engineering_functions.items():
            feature_engineered_dataset[feature_name] = feature_func(self.train_features)
        return feature_engineered_dataset

    def feature_engineering_test(self) -> pd.DataFrame:
        feature_engineered_dataset = pd.DataFrame()
        for feature_name, feature_func in self.feature_engineering_functions.items():
            feature_engineered_dataset[feature_name] = feature_func(self.test_features)
        return feature_engineered_dataset

    def preprocess(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        one_hot_encoded_train = self.one_hot_encode_columns_train()
        one_hot_encoded_test = self.one_hot_encode_columns_test()
        min_max_scaled_train = self.min_max_scaled_columns_train()
        min_max_scaled_test = self.min_max_scaled_columns_test()
        pca_train = self.pca_train()
        pca_test = self.pca_test()
        feature_engineered_train = self.feature_engineering_train()
        feature_engineered_test = self.feature_engineering_test()
        train_features = pd.DataFrame()
        test_features = pd.DataFrame()
        return train_features,test_features
