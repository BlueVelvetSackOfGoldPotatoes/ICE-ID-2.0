"""
Settings management system for ICE-ID models
Supports PyTorch and scikit-learn compatible parameter configurations
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

@dataclass
class PyTorchModelSettings:
    """PyTorch-compatible model settings with sensible defaults"""
    # Model architecture
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_steps: int = 100
    
    # Data processing
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    
    # Regularization
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 3
    
    # Device and performance
    device: str = "auto"
    mixed_precision: bool = False
    num_workers: int = 4

@dataclass
class SklearnModelSettings:
    """scikit-learn-compatible model settings with sensible defaults"""
    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_max_features: str = "sqrt"
    rf_bootstrap: bool = True
    rf_random_state: int = 42
    
    # Gradient Boosting
    gb_n_estimators: int = 100
    gb_learning_rate: float = 0.1
    gb_max_depth: int = 3
    gb_min_samples_split: int = 2
    gb_min_samples_leaf: int = 1
    gb_subsample: float = 1.0
    gb_random_state: int = 42
    
    # Logistic Regression
    lr_penalty: str = "l2"
    lr_C: float = 1.0
    lr_solver: str = "lbfgs"
    lr_max_iter: int = 1000
    lr_random_state: int = 42
    
    # SVM
    svm_C: float = 1.0
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    svm_random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "f1"
    cv_random_state: int = 42

@dataclass
class DataProcessingSettings:
    """Data processing and feature engineering settings"""
    # Text processing
    text_max_length: int = 256
    text_lowercase: bool = True
    text_remove_punctuation: bool = False
    text_stemming: bool = False
    
    # Feature engineering
    use_tfidf: bool = True
    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    
    # Numerical features
    normalize_numerical: bool = True
    numerical_scaler: str = "standard"  # standard, minmax, robust
    
    # Categorical features
    encode_categorical: bool = True
    categorical_encoding: str = "onehot"  # onehot, label, target
    
    # Data splitting
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42

@dataclass
class ModelConfig:
    """Complete model configuration"""
    model_name: str
    model_type: str  # "pytorch", "sklearn", "iceid"
    pytorch_settings: Optional[PyTorchModelSettings] = None
    sklearn_settings: Optional[SklearnModelSettings] = None
    data_settings: Optional[DataProcessingSettings] = None
    
    # ICE-ID specific settings
    iceid_backends: List[str] = None
    iceid_neg_ratio: float = 2.0
    iceid_thresh_grid: int = 101
    iceid_dual_blocking: bool = False
    iceid_soft_filter_max_year_diff: int = 15
    iceid_soft_filter_sex: bool = True
    
    # General settings
    sample_frac: float = 1.0
    device: str = "auto"
    random_state: int = 42
    
    def __post_init__(self):
        if self.iceid_backends is None:
            self.iceid_backends = ["gbdt", "logreg"]
        if self.pytorch_settings is None and self.model_type == "pytorch":
            self.pytorch_settings = PyTorchModelSettings()
        if self.sklearn_settings is None and self.model_type == "sklearn":
            self.sklearn_settings = SklearnModelSettings()
        if self.data_settings is None:
            self.data_settings = DataProcessingSettings()

class SettingsManager:
    """Manages model settings save/load operations for ICE-ID models.

    Provides functionality to save, load, list, and delete model configurations.
    Also provides default configurations for all supported model types.

    Args:
        settings_dir (str): Directory path where configuration files are stored.
    """
    
    def __init__(self, settings_dir: str = "runs/settings"):
        """Initializes the settings manager and creates the settings directory.

        Args:
            settings_dir (str): Path to the directory for storing configuration files.
        """
        self.settings_dir = Path(settings_dir)
        self.settings_dir.mkdir(parents=True, exist_ok=True)
    
    def get_default_configs(self) -> Dict[str, ModelConfig]:
        """Returns default configurations for all supported model types.

        Provides pre-configured settings for ICE-ID, Ditto, ZeroER, TF-IDF, MLP,
        Splink, XGBoost, LightGBM, Cross-Encoder, Dedupe, RecordLinkage, Random
        Forest, and Gradient Boosting models.

        Returns:
            Dict[str, ModelConfig]: Mapping from model name to its default configuration.
        """
        return {
            "ICE-ID Pipeline": ModelConfig(
                model_name="ICE-ID Pipeline",
                model_type="iceid",
                iceid_backends=["gbdt", "logreg"],
                iceid_neg_ratio=2.0,
                iceid_thresh_grid=101,
                iceid_dual_blocking=False,
                iceid_soft_filter_max_year_diff=15,
                iceid_soft_filter_sex=True,
                sample_frac=1.0,
                random_state=42
            ),
            "Ditto (HF)": ModelConfig(
                model_name="Ditto (HF)",
                model_type="pytorch",
                pytorch_settings=PyTorchModelSettings(
                    hidden_dim=256,
                    num_layers=3,
                    learning_rate=2e-5,
                    batch_size=16,
                    epochs=3,
                    max_length=512,
                    optimizer="adamw",
                    scheduler="linear"
                ),
                data_settings=DataProcessingSettings(
                    text_max_length=512,
                    use_tfidf=False,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "ZeroER (SBERT)": ModelConfig(
                model_name="ZeroER (SBERT)",
                model_type="pytorch",
                pytorch_settings=PyTorchModelSettings(
                    hidden_dim=384,  # SBERT embedding size
                    num_layers=1,
                    learning_rate=1e-4,
                    batch_size=32,
                    epochs=1,
                    max_length=256,
                    optimizer="adam",
                    scheduler="none"
                ),
                data_settings=DataProcessingSettings(
                    text_max_length=256,
                    use_tfidf=False,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "TF-IDF + Logistic Regression": ModelConfig(
                model_name="TF-IDF + Logistic Regression",
                model_type="sklearn",
                sklearn_settings=SklearnModelSettings(
                    lr_penalty="l2",
                    lr_C=1.0,
                    lr_solver="lbfgs",
                    lr_max_iter=1000,
                    cv_folds=5,
                    cv_scoring="f1"
                ),
                data_settings=DataProcessingSettings(
                    use_tfidf=True,
                    tfidf_max_features=10000,
                    tfidf_ngram_range=(1, 2),
                    normalize_numerical=True,
                    categorical_encoding="onehot"
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "MLP": ModelConfig(
                model_name="MLP",
                model_type="pytorch",
                pytorch_settings=PyTorchModelSettings(
                    hidden_dim=128,
                    num_layers=2,
                    dropout=0.1,
                    learning_rate=1e-3,
                    batch_size=32,
                    epochs=10,
                    optimizer="adam",
                    scheduler="cosine"
                ),
                data_settings=DataProcessingSettings(
                    use_tfidf=True,
                    tfidf_max_features=20000,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "Splink": ModelConfig(
                model_name="Splink",
                model_type="iceid",
                data_settings=DataProcessingSettings(
                    use_tfidf=False,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "XGBoost": ModelConfig(
                model_name="XGBoost",
                model_type="sklearn",
                sklearn_settings=SklearnModelSettings(
                    gb_n_estimators=100,
                    gb_max_depth=5,
                    gb_learning_rate=0.1,
                    cv_folds=5,
                    cv_scoring="f1"
                ),
                data_settings=DataProcessingSettings(
                    use_tfidf=True,
                    tfidf_max_features=20000,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "LightGBM": ModelConfig(
                model_name="LightGBM",
                model_type="sklearn",
                sklearn_settings=SklearnModelSettings(
                    gb_n_estimators=100,
                    gb_max_depth=5,
                    gb_learning_rate=0.1,
                    cv_folds=5,
                    cv_scoring="f1"
                ),
                data_settings=DataProcessingSettings(
                    use_tfidf=True,
                    tfidf_max_features=20000,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "Cross-Encoder": ModelConfig(
                model_name="Cross-Encoder",
                model_type="pytorch",
                pytorch_settings=PyTorchModelSettings(
                    learning_rate=2e-5,
                    batch_size=16,
                    epochs=3,
                    max_length=512,
                    optimizer="adamw",
                    scheduler="linear"
                ),
                data_settings=DataProcessingSettings(
                    text_max_length=512,
                    use_tfidf=False,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "Dedupe": ModelConfig(
                model_name="Dedupe",
                model_type="iceid",
                data_settings=DataProcessingSettings(
                    use_tfidf=False,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "RecordLinkage": ModelConfig(
                model_name="RecordLinkage",
                model_type="sklearn",
                sklearn_settings=SklearnModelSettings(
                    rf_n_estimators=100,
                    rf_max_depth=10,
                    cv_folds=5,
                    cv_scoring="f1"
                ),
                data_settings=DataProcessingSettings(
                    use_tfidf=False,
                    normalize_numerical=True
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "Random Forest": ModelConfig(
                model_name="Random Forest",
                model_type="sklearn",
                sklearn_settings=SklearnModelSettings(
                    rf_n_estimators=200,
                    rf_max_depth=10,
                    rf_min_samples_split=5,
                    rf_min_samples_leaf=2,
                    rf_max_features="sqrt",
                    cv_folds=5,
                    cv_scoring="f1"
                ),
                data_settings=DataProcessingSettings(
                    use_tfidf=True,
                    tfidf_max_features=5000,
                    normalize_numerical=True,
                    categorical_encoding="onehot"
                ),
                sample_frac=0.1,
                random_state=42
            ),
            "Gradient Boosting": ModelConfig(
                model_name="Gradient Boosting",
                model_type="sklearn",
                sklearn_settings=SklearnModelSettings(
                    gb_n_estimators=200,
                    gb_learning_rate=0.05,
                    gb_max_depth=6,
                    gb_min_samples_split=10,
                    gb_min_samples_leaf=4,
                    gb_subsample=0.8,
                    cv_folds=5,
                    cv_scoring="f1"
                ),
                data_settings=DataProcessingSettings(
                    use_tfidf=True,
                    tfidf_max_features=5000,
                    normalize_numerical=True,
                    categorical_encoding="onehot"
                ),
                sample_frac=0.1,
                random_state=42
            )
        }
    
    def save_config(self, config: ModelConfig, filename: str) -> str:
        """Saves a model configuration to a JSON file.

        Serializes the ModelConfig dataclass to JSON format and writes it to the
        settings directory with the specified filename.

        Args:
            config (ModelConfig): The configuration object to save.
            filename (str): Name for the configuration file (without .json extension).

        Returns:
            str: Full path to the saved configuration file.
        """
        filepath = self.settings_dir / f"{filename}.json"
        
        config_dict = asdict(config)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        return str(filepath)
    
    def load_config(self, filename: str) -> ModelConfig:
        """Loads a model configuration from a JSON file.

        Reads the configuration file, reconstructs the dataclass objects, and returns
        a fully populated ModelConfig instance.

        Args:
            filename (str): Name of the configuration file (without .json extension).

        Returns:
            ModelConfig: The loaded configuration object.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist.
        """
        filepath = self.settings_dir / f"{filename}.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Settings file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct dataclasses
        if config_dict.get('pytorch_settings'):
            config_dict['pytorch_settings'] = PyTorchModelSettings(**config_dict['pytorch_settings'])
        
        if config_dict.get('sklearn_settings'):
            config_dict['sklearn_settings'] = SklearnModelSettings(**config_dict['sklearn_settings'])
        
        if config_dict.get('data_settings'):
            config_dict['data_settings'] = DataProcessingSettings(**config_dict['data_settings'])
        
        return ModelConfig(**config_dict)
    
    def list_configs(self) -> List[str]:
        """Lists all available configuration files in the settings directory.

        Scans the settings directory for JSON files and returns their names.

        Returns:
            List[str]: Sorted list of configuration file names (without .json extension).
        """
        config_files = []
        for file in self.settings_dir.glob("*.json"):
            config_files.append(file.stem)
        return sorted(config_files)
    
    def delete_config(self, filename: str) -> bool:
        """Deletes a configuration file from the settings directory.

        Args:
            filename (str): Name of the configuration file to delete (without .json extension).

        Returns:
            bool: True if the file was deleted, False if it didn't exist.
        """
        filepath = self.settings_dir / f"{filename}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False
    
    def get_config_summary(self, config: ModelConfig) -> Dict[str, Any]:
        """Creates a human-readable summary of configuration parameters.

        Extracts key parameters from the configuration object based on the model
        type and organizes them into a dictionary suitable for display.

        Args:
            config (ModelConfig): The configuration object to summarize.

        Returns:
            Dict[str, Any]: A dictionary of key parameters and their values.
        """
        summary = {
            "Model Name": config.model_name,
            "Model Type": config.model_type,
            "Sample Fraction": config.sample_frac,
            "Random State": config.random_state
        }
        
        if config.model_type == "iceid":
            summary.update({
                "Backends": ", ".join(config.iceid_backends),
                "Negative Ratio": config.iceid_neg_ratio,
                "Threshold Grid": config.iceid_thresh_grid,
                "Dual Blocking": config.iceid_dual_blocking
            })
        
        elif config.model_type == "pytorch" and config.pytorch_settings:
            summary.update({
                "Hidden Dim": config.pytorch_settings.hidden_dim,
                "Layers": config.pytorch_settings.num_layers,
                "Learning Rate": config.pytorch_settings.learning_rate,
                "Batch Size": config.pytorch_settings.batch_size,
                "Epochs": config.pytorch_settings.epochs,
                "Optimizer": config.pytorch_settings.optimizer
            })
        
        elif config.model_type == "sklearn" and config.sklearn_settings:
            if "Random Forest" in config.model_name:
                summary.update({
                    "Estimators": config.sklearn_settings.rf_n_estimators,
                    "Max Depth": config.sklearn_settings.rf_max_depth,
                    "Min Samples Split": config.sklearn_settings.rf_min_samples_split
                })
            elif "Gradient" in config.model_name:
                summary.update({
                    "Estimators": config.sklearn_settings.gb_n_estimators,
                    "Learning Rate": config.sklearn_settings.gb_learning_rate,
                    "Max Depth": config.sklearn_settings.gb_max_depth
                })
            elif "Logistic" in config.model_name:
                summary.update({
                    "Penalty": config.sklearn_settings.lr_penalty,
                    "C": config.sklearn_settings.lr_C,
                    "Solver": config.sklearn_settings.lr_solver
                })
        
        if config.data_settings:
            summary.update({
                "Text Max Length": config.data_settings.text_max_length,
                "Use TF-IDF": config.data_settings.use_tfidf,
                "Normalize Numerical": config.data_settings.normalize_numerical,
                "Categorical Encoding": config.data_settings.categorical_encoding
            })
        
        return summary

# Global settings manager instance
settings_manager = SettingsManager()
