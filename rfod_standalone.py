"""
RFOD (Random Forest-based Outlier Detection) - Standalone Version
No external dependencies beyond standard ML libraries
"""
import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import LabelEncoder

# Optional GPU dependencies (RAPIDS cuML)
try:
    import cudf
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as cuRFClassifier
    from cuml.ensemble import RandomForestRegressor as cuRFRegressor
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    cudf = None
    cp = None
    cuRFClassifier = None
    cuRFRegressor = None


class RFOD:
    """
    RFOD: Random Forest-based Outlier Detection
    
    A novel anomaly detection framework specifically designed for tabular data.
    Uses feature-wise conditional reconstruction with Random Forests.
    
    Reference:
    Yihao Ang et al. "RFOD: Random Forest-based Outlier Detection for Tabular Data"
    """
    
    def __init__(
        self,
        alpha: float = 0.02,
        beta: float = 0.7,
        n_estimators: int = 30,
        max_depth: int = 6,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True,
        backend: str = "sklearn",
        n_streams: int = 4
    ):
        """
        Parameters:
        -----------
        alpha : float, default=0.02
            Quantile parameter for Adjusted Gower's Distance (AGD).
            
        beta : float, default=0.7
            Retaining ratio for forest pruning (0 < beta <= 1).
            
        n_estimators : int, default=30
            Number of trees in each random forest.
            
        max_depth : int, default=6
            Maximum depth of each decision tree.
            
        random_state : int, default=42
            Random seed for reproducibility.
            
        n_jobs : int, default=-1
            Number of parallel jobs (-1 uses all cores).
            
        verbose : bool, default=True
            Whether to print progress messages.
            
        backend : str, default="sklearn"
            Backend: "sklearn" (CPU) or "cuml" (GPU).
            
        n_streams : int, default=4
            Number of parallel streams for cuML GPU backend.
        """
        self.alpha = alpha
        self.beta = beta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.backend = backend
        self.n_streams = n_streams

        if self.backend == "cuml":
            if not HAS_CUML:
                raise RuntimeError(
                    "backend='cuml' but cuML not detected.\n"
                    "Please install RAPIDS in WSL2/Linux environment."
                )
            if self.verbose:
                print("[RFOD] Using GPU acceleration (cuML backend)")
        elif self.verbose and self.backend == "sklearn":
            print("[RFOD] Using CPU computation (sklearn backend)")

        self.forests_ = {}
        self.feature_types_ = {}
        self.quantiles_ = {}
        self.feature_names_ = []
        self.n_features_ = 0
        self.encoders_: Dict[str, LabelEncoder] = {}

    def _identify_feature_types(self, X: pd.DataFrame) -> Dict[int, str]:
        """Identify numerical and categorical features"""
        feature_types = {}
        for idx, col in enumerate(X.columns):
            if pd.api.types.is_numeric_dtype(X[col]):
                feature_types[idx] = 'numeric'
            else:
                feature_types[idx] = 'categorical'
        return feature_types

    def _compute_quantiles(self, X: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
        """Compute alpha and 1-alpha quantiles for numerical features"""
        quantiles = {}
        for idx, col in enumerate(X.columns):
            if self.feature_types_[idx] == 'numeric':
                q_low = X[col].quantile(self.alpha)
                q_high = X[col].quantile(1 - self.alpha)
                if q_high - q_low < 1e-10:
                    q_high = q_low + 1.0
                quantiles[idx] = (q_low, q_high)
        return quantiles

    def _fit_encoders(self, X: pd.DataFrame):
        """Fit LabelEncoders for categorical features"""
        self.encoders_ = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                series = X[col].astype(str).fillna("NaN_TOKEN")
                le.fit(series)
                self.encoders_[col] = le

    def _transform_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using stored LabelEncoders"""
        X_transformed = X.copy()
        for col, le in self.encoders_.items():
            if col in X_transformed.columns:
                series = X_transformed[col].astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~series.isin(le.classes_)
                series.loc[unseen_mask] = le.classes_[0]
                transformed_series = le.transform(series)
                transformed_series[unseen_mask] = -1
                X_transformed[col] = transformed_series
        return X_transformed

    def _train_feature_forest(self, X: pd.DataFrame, feature_idx: int):
        """Train a random forest for a single feature"""
        X_train_df = X.drop(X.columns[feature_idx], axis=1)
        y_train = X.iloc[:, feature_idx]

        X_train_encoded = self._transform_data(X_train_df)
        target_col_name = X.columns[feature_idx]

        # cuML backend
        if self.backend == "cuml":
            X_cu = cudf.DataFrame.from_pandas(X_train_encoded)

            if self.feature_types_[feature_idx] == 'categorical':
                if target_col_name in self.encoders_:
                    y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                    unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                    y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                    y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                    y_train_encoded[unseen_mask] = -1
                    y_train = y_train_encoded
                y_cu = cudf.Series(pd.Series(y_train).astype('int32'))
                forest = cuRFClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_streams=self.n_streams
                )
            else:
                y_train = y_train.fillna(y_train.mean())
                y_cu = cudf.Series(pd.Series(y_train).astype('float32'))
                forest = cuRFRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    n_streams=self.n_streams
                )
            forest.fit(X_cu, y_cu)
            return forest

        # sklearn backend
        if self.feature_types_[feature_idx] == 'categorical':
            if target_col_name in self.encoders_:
                y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                y_train_encoded[unseen_mask] = -1
                y_train = y_train_encoded

            forest = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True,
                bootstrap=True
            )
        else:
            y_train = y_train.fillna(y_train.mean())
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                oob_score=True,
                bootstrap=True
            )

        forest.fit(X_train_encoded, y_train)
        return forest

    def _prune_forest(self, forest, X: pd.DataFrame, feature_idx: int):
        """Prune forest using OOB validation"""
        if self.backend == "cuml":
            if self.verbose and self.beta < 1.0:
                print(f"    [Note] cuML doesn't support forest pruning")
            return forest

        X_train_df = X.drop(X.columns[feature_idx], axis=1)
        y_train = X.iloc[:, feature_idx]

        X_train_encoded = self._transform_data(X_train_df)
        target_col_name = X.columns[feature_idx]
        is_classifier = isinstance(forest, RandomForestClassifier)
        
        if is_classifier:
            if target_col_name in self.encoders_:
                y_train_series = y_train.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~y_train_series.isin(self.encoders_[target_col_name].classes_)
                y_train_series.loc[unseen_mask] = self.encoders_[target_col_name].classes_[0]
                y_train_encoded = self.encoders_[target_col_name].transform(y_train_series)
                y_train_encoded[unseen_mask] = -1
                y_train = pd.Series(y_train_encoded, index=X_train_df.index)
        else:
            y_train = y_train.fillna(y_train.mean())

        n_samples = X_train_encoded.shape[0]
        if n_samples == 0:
            return forest
        
        # Compute n_samples_bootstrap
        if hasattr(forest, 'max_samples') and forest.max_samples is not None:
            if isinstance(forest.max_samples, int):
                n_samples_bootstrap = min(forest.max_samples, n_samples)
            else:
                n_samples_bootstrap = int(forest.max_samples * n_samples)
        else:
            n_samples_bootstrap = n_samples
        
        tree_scores = []
        for tree in forest.estimators_:
            try:
                oob_indices = _generate_unsampled_indices(
                    tree.random_state, n_samples, n_samples_bootstrap
                )
                
                if len(oob_indices) == 0:
                    tree_scores.append(0.0)
                    continue

                X_oob = X_train_encoded.iloc[oob_indices]
                y_oob = y_train.iloc[oob_indices]

                if len(y_oob) == 0:
                    tree_scores.append(0.0)
                    continue
                
                if is_classifier:
                    if len(np.unique(y_oob)) <= 1:
                        tree_scores.append(0.0)
                        continue
                    y_pred_proba = tree.predict_proba(X_oob)
                    score = roc_auc_score(
                        y_oob, y_pred_proba, 
                        multi_class='ovr', 
                        average='macro', 
                        labels=forest.classes_
                    )
                else:
                    y_pred = tree.predict(X_oob)
                    score = r2_score(y_oob, y_pred)
                    score = max(0, score)
            except Exception as e:
                score = 0.0
            tree_scores.append(score)

        n_trees_keep = max(1, int(self.beta * len(forest.estimators_)))
        top_indices = np.argsort(tree_scores)[-n_trees_keep:]

        if is_classifier:
            pruned = RandomForestClassifier(
                n_estimators=n_trees_keep, 
                random_state=self.random_state
            )
        else:
            pruned = RandomForestRegressor(
                n_estimators=n_trees_keep, 
                random_state=self.random_state
            )

        pruned.estimators_ = [forest.estimators_[i] for i in top_indices]
        pruned.n_estimators = n_trees_keep
        
        for attr in ["classes_", "n_classes_", "n_features_in_", "feature_names_in_"]:
            if hasattr(forest, attr):
                setattr(pruned, attr, getattr(forest, attr))
        
        if hasattr(pruned, 'oob_score_'):
            pruned.oob_score_ = None
            
        return pruned

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'RFOD':
        """Fit RFOD on training data"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
        self.feature_names_ = list(X.columns)
        self.n_features_ = len(self.feature_names_)
        
        if self.verbose:
            print(f"[RFOD] Training: {len(X)} samples, {self.n_features_} features")
            
        self.feature_types_ = self._identify_feature_types(X)
        self._fit_encoders(X)
        
        if self.verbose:
            n_numeric = sum(1 for t in self.feature_types_.values() if t == 'numeric')
            n_categorical = self.n_features_ - n_numeric
            print(f"[RFOD] Features: {n_numeric} numerical, {n_categorical} categorical")
            
        self.quantiles_ = self._compute_quantiles(X)
        
        if self.verbose:
            print("[RFOD] Training feature-specific forests...")
            
        for feature_idx in range(self.n_features_):
            if self.verbose:
                print(f"  [{feature_idx+1}/{self.n_features_}] {self.feature_names_[feature_idx]}")
            
            forest = self._train_feature_forest(X, feature_idx)
            
            if self.beta < 1.0:
                forest = self._prune_forest(forest, X, feature_idx)
                
            self.forests_[feature_idx] = forest
            
        if self.verbose:
            print("[RFOD] Training complete ✓")
        return self

    def _predict_feature(self, X: pd.DataFrame, feature_idx: int, batch_size: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
        """Predict values and uncertainty for a single feature"""
        forest = self.forests_[feature_idx]
        X_input_df = X.drop(X.columns[feature_idx], axis=1)
        X_input_encoded = self._transform_data(X_input_df)

        n_samples = X_input_encoded.shape[0]

        # cuML backend
        if self.backend == "cuml":
            X_cu = cudf.DataFrame.from_pandas(X_input_encoded)

            if isinstance(forest, (cuRFClassifier,)) or hasattr(forest, 'predict_proba'):
                proba = forest.predict_proba(X_cu)
                if hasattr(proba, 'values'):
                    proba_np = proba.values.get() if hasattr(proba.values, 'get') else proba.values
                elif hasattr(proba, 'get'):
                    proba_np = proba.get()
                else:
                    proba_np = cp.asnumpy(proba) if isinstance(proba, cp.ndarray) else np.array(proba)
                
                uncertainties = -np.sum(proba_np * np.log(proba_np + 1e-10), axis=1)
                return proba_np, uncertainties
            else:
                preds = forest.predict(X_cu)
                if hasattr(preds, 'values'):
                    preds_np = preds.values.get() if hasattr(preds.values, 'get') else preds.values
                elif hasattr(preds, 'get'):
                    preds_np = preds.get()
                else:
                    preds_np = cp.asnumpy(preds) if isinstance(preds, cp.ndarray) else np.array(preds)
                std = np.zeros_like(preds_np, dtype=np.float64)
                return preds_np, std

        # sklearn backend
        if isinstance(forest, RandomForestClassifier):
            n_classes = len(forest.classes_)
            all_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
            all_uncertainties = np.zeros(n_samples, dtype=np.float64)
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_input_encoded.iloc[start:end]
                
                tree_probs = np.array([tree.predict_proba(X_batch) for tree in forest.estimators_])
                mean_probs = tree_probs.mean(axis=0)
                all_probs[start:end] = mean_probs
                
                uncertainties_batch = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
                all_uncertainties[start:end] = uncertainties_batch
            
            return all_probs, all_uncertainties
            
        else:  # Regressor
            predictions = np.zeros(n_samples, dtype=np.float64)
            std_devs = np.zeros(n_samples, dtype=np.float64)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_input_encoded.iloc[start:end]
                
                preds = np.array([tree.predict(X_batch) for tree in forest.estimators_])
                mean_batch = preds.mean(axis=0)
                std_batch = preds.std(axis=0)
                
                predictions[start:end] = mean_batch
                std_devs[start:end] = std_batch
                
            return predictions, std_devs

    def _compute_cell_scores(self, X: pd.DataFrame, predictions: Dict[int, np.ndarray]) -> np.ndarray:
        """Compute AGD as cell-level anomaly scores"""
        n_samples = len(X)
        cell_scores = np.zeros((n_samples, self.n_features_))
        
        for feature_idx in range(self.n_features_):
            true_values_series = X.iloc[:, feature_idx]
            pred_values = predictions[feature_idx]
            
            if self.feature_types_[feature_idx] == 'numeric':
                q_low, q_high = self.quantiles_.get(feature_idx, (0.0, 1.0))
                denom = (q_high - q_low) if (q_high - q_low) > 1e-10 else 1.0
                
                true_values_filled = true_values_series.fillna(np.nanmean(pred_values)).values.astype(float)
                pred_values_filled = np.nan_to_num(pred_values, nan=np.nanmean(pred_values)).astype(float)
                
                diff = np.abs(true_values_filled - pred_values_filled)
                cell_scores[:, feature_idx] = diff / denom
                
            else:
                forest = self.forests_[feature_idx]
                classes = getattr(forest, "classes_", None)
                if classes is None:
                    continue

                target_col_name = self.feature_names_[feature_idx]
                le = self.encoders_.get(target_col_name)
                if le is None:
                    continue

                true_values_str = true_values_series.astype(str).fillna("NaN_TOKEN")
                unseen_mask = ~true_values_str.isin(le.classes_)
                true_values_str.loc[unseen_mask] = le.classes_[0]
                true_values_encoded = le.transform(true_values_str)
                true_values_encoded[unseen_mask] = -1

                class_to_idx = {int(cls): idx for idx, cls in enumerate(classes)}
                scores = np.ones(n_samples, dtype=np.float64)
                
                for i in range(n_samples):
                    true_class = int(true_values_encoded[i])
                    if true_class in class_to_idx:
                        idx = class_to_idx[true_class]
                        scores[i] = 1.0 - pred_values[i, idx]
                
                cell_scores[:, feature_idx] = scores
                    
        return cell_scores

    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                return_cell_scores: bool = False,
                clip_scores: bool = False, 
                clip_min: float = 0.0, 
                clip_max: float = 1.0) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict anomaly scores"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)

        n_samples = len(X)
        if self.verbose:
            print(f"[RFOD] Predicting {n_samples} samples...")

        predictions = {}
        uncertainties = {}

        for feature_idx in range(self.n_features_):
            pred, uncert = self._predict_feature(X, feature_idx)
            predictions[feature_idx] = pred
            uncertainties[feature_idx] = uncert

        cell_scores = self._compute_cell_scores(X, predictions)
        uncertainty_matrix = np.column_stack([uncertainties[i] for i in range(self.n_features_)])
        
        row_sums = uncertainty_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        uncertainty_norm = uncertainty_matrix / row_sums
        
        weights = 1.0 - uncertainty_norm
        weighted_scores = weights * cell_scores
        row_scores = weighted_scores.mean(axis=1)

        if clip_scores:
            original_min, original_max = row_scores.min(), row_scores.max()
            row_scores = np.clip(row_scores, clip_min, clip_max)
            if self.verbose and (original_min < clip_min or original_max > clip_max):
                print(f"[RFOD] Scores clipped: [{original_min:.4f}, {original_max:.4f}] "
                      f"-> [{row_scores.min():.4f}, {row_scores.max():.4f}]")

        if self.verbose:
            print(f"[RFOD] Prediction complete: [{row_scores.min():.4f}, {row_scores.max():.4f}]")

        if return_cell_scores:
            return row_scores, cell_scores
        else:
            return row_scores

    def fit_predict(self, X_train: Union[pd.DataFrame, np.ndarray], 
                    X_test: Union[pd.DataFrame, np.ndarray], 
                    return_cell_scores: bool = False):
        """Fit on training data and predict on test data"""
        self.fit(X_train)
        return self.predict(X_test, return_cell_scores=return_cell_scores)