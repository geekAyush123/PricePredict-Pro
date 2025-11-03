#!/usr/bin/env python3
"""

Date: October 2025
Competition: Smart Product Pricing Challenge
Metric: SMAPE (Symmetric Mean Absolute Percentage Error)
"""

import os
import re
import sys
import warnings
import time
import random
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from functools import partial

# Core data science libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Preprocessing and feature engineering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# ML models
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    print("Warning: CatBoost not available. Using LightGBM and XGBoost only.")
    cb = None

# Deep learning and transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: sentence-transformers not available. Using TF-IDF only.")
    SentenceTransformer = None

# Computer vision
try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
    import requests
    from io import BytesIO
except ImportError:
    print("Warning: PyTorch/torchvision not available. Skipping image features.")
    torch = None

# Hyperparameter optimization
try:
    import optuna
except ImportError:
    print("Warning: Optuna not available. Using default hyperparameters.")
    optuna = None

# Image downloading
import urllib.request
import multiprocessing

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Config:
    """Configuration class for the pricing predictor."""

    # Data paths
    DATASET_FOLDER = '/teamspace/studios/this_studio' # Corrected to point to the directory
    TRAIN_CSV = 'train.csv'
    TEST_CSV = 'test.csv'
    SAMPLE_TEST_CSV = 'sample_test.csv'
    OUTPUT_CSV = 'test_out.csv'

    # Image paths
    IMAGES_TRAIN_FOLDER = 'images/train'
    IMAGES_TEST_FOLDER = 'images/test'

    # Model paths
    MODELS_FOLDER = 'models'
    SUBMISSIONS_FOLDER = 'submissions'

    # Feature engineering
    MAX_TFIDF_FEATURES = 20000
    TFIDF_SVD_COMPONENTS = 512  # Reduce TF-IDF dimensions with SVD
    SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'

    # Cross-validation
    N_FOLDS = 5
    CV_RANDOM_STATE = RANDOM_SEED

    # Model hyperparameters (default)
    LIGHTGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'device_type': 'gpu',  # Use GPU for faster training
        'random_state': RANDOM_SEED
    }

    XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    # Image processing
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    MAX_DOWNLOAD_WORKERS = 50
    USE_EFFICIENTNET_B0 = True  # Use B0 instead of B3 for speed

    # Feature caching
    CACHE_FEATURES = True
    FEATURES_CACHE_DIR = 'cached_features'

class SMAPECalculator:
    """Custom SMAPE metric calculator."""

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Symmetric Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            SMAPE score (lower is better)
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0  # Handle division by zero
        return np.mean(diff) * 100

    @staticmethod
    def lgb_smape(y_pred: np.ndarray, y_true) -> Tuple[str, float, bool]:
        """SMAPE metric for LightGBM."""
        y_true = y_true.get_label()
        smape_val = SMAPECalculator.smape(y_true, y_pred)
        return 'smape', smape_val, False

class TextFeatureExtractor:
    """Advanced text feature extraction pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.tfidf_vectorizer = None
        self.tfidf_svd = None  # For dimensionality reduction
        self.sentence_transformer = None
        self.brand_encoder = LabelEncoder()
        self.brand_frequency_map = {}
        self.kmeans_model = None  # For category clustering
        self.text_scaler = StandardScaler()  # For text feature normalization

    def extract_brand(self, text: str) -> str:
        """
        Extract brand name from product text using regex patterns.

        Args:
            text: Product description text

        Returns:
            Extracted brand name (default: 'unknown')
        """
        if pd.isna(text) or not isinstance(text, str):
            return 'unknown'

        # Clean text for brand extraction
        text = text.strip()

        # Pattern 1: Look for "Item Name: Brand ..." pattern
        item_name_match = re.search(r'Item Name:\s*([A-Z][a-zA-Z\s&]+?)(?:\s+[a-z]|\s*[-,])', text)
        if item_name_match:
            potential_brand = item_name_match.group(1).strip()
            # Take first 1-2 words as brand
            brand_words = potential_brand.split()[:2]
            if brand_words and len(brand_words[0]) > 2:
                return ' '.join(brand_words).lower()

        # Pattern 2: Look for capitalized words at the beginning
        words = text.split()
        for i, word in enumerate(words[:5]):  # Check first 5 words
            if word and len(word) > 2 and word[0].isupper() and word.isalpha():
                # Check if next word is also capitalized (compound brand)
                if i + 1 < len(words) and words[i + 1][0].isupper() and len(words[i + 1]) > 2:
                    return f"{word} {words[i + 1]}".lower()
                return word.lower()

        # Pattern 3: Known brand patterns
        brand_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:Brand|Company|Inc|LLC|Ltd)',
            r'(?:by|from|made by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        ]

        for pattern in brand_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).lower()

        return 'unknown'

    def extract_specs(self, text: str) -> Dict[str, float]:
        """
        Extract numerical specifications and units from product text.

        Args:
            text: Product description text

        Returns:
            Dictionary with spec values (ml, g, kg, l, oz, gb, etc.)
        """
        if pd.isna(text) or not isinstance(text, str):
            return {}

        # Common unit patterns with their normalizations
        unit_patterns = {
            # Volume
            r'(\d+\.?\d*)\s*(?:ml|milliliters?)\b': ('volume_ml', 1.0),
            r'(\d+\.?\d*)\s*(?:l|liters?|litres?)\b': ('volume_ml', 1000.0),
            r'(\d+\.?\d*)\s*(?:fl\s*oz|fluid\s*ounces?)\b': ('volume_ml', 29.5735),
            r'(\d+\.?\d*)\s*(?:gal|gallons?)\b': ('volume_ml', 3785.41),

            # Weight
            r'(\d+\.?\d*)\s*(?:g|grams?)\b': ('weight_g', 1.0),
            r'(\d+\.?\d*)\s*(?:kg|kilograms?)\b': ('weight_g', 1000.0),
            r'(\d+\.?\d*)\s*(?:oz|ounces?)\b': ('weight_g', 28.3495),
            r'(\d+\.?\d*)\s*(?:lb|lbs|pounds?)\b': ('weight_g', 453.592),

            # Digital storage
            r'(\d+\.?\d*)\s*(?:gb|gigabytes?)\b': ('storage_gb', 1.0),
            r'(\d+\.?\d*)\s*(?:mb|megabytes?)\b': ('storage_gb', 0.001),
            r'(\d+\.?\d*)\s*(?:tb|terabytes?)\b': ('storage_gb', 1000.0),

            # Length
            r'(\d+\.?\d*)\s*(?:mm|millimeters?)\b': ('length_mm', 1.0),
            r'(\d+\.?\d*)\s*(?:cm|centimeters?)\b': ('length_mm', 10.0),
            r'(\d+\.?\d*)\s*(?:m|meters?)\b': ('length_mm', 1000.0),
            r'(\d+\.?\d*)\s*(?:in|inches?)\b': ('length_mm', 25.4),
            r'(\d+\.?\d*)\s*(?:ft|feet)\b': ('length_mm', 304.8),
        }

        specs = {}
        text_lower = text.lower()

        for pattern, (spec_name, multiplier) in unit_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                try:
                    # Take the maximum value found
                    value = max(float(match) for match in matches) * multiplier
                    specs[spec_name] = value
                except ValueError:
                    continue

        return specs

    def extract_ipq(self, text: str) -> int:
        """
        Extract Item Pack Quantity (IPQ) from product text.

        Args:
            text: Product description text

        Returns:
            Extracted IPQ value (default: 1)
        """
        if pd.isna(text) or not isinstance(text, str):
            return 1

        # Comprehensive IPQ patterns
        patterns = [
            r'(?:IPQ|Pack of|Pack Size|Quantity|Pack):?\s*(\d+)',
            r'(?:Value|Count):\s*(\d+(?:\.\d+)?)',
            r'(\d+)\s*(?:Pack|Count|oz|Ounce|Fl Oz|ct)',
            r'Pack of (\d+)',
            r'(\d+)-Pack',
            r'(\d+)\s*pieces?',
            r'(\d+)\s*units?'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    return max(int(float(match)) for match in matches)
                except ValueError:
                    continue

        return 1

    def clean_text(self, text: str, remove_ipq: bool = True) -> str:
        """
        Clean and preprocess text data.

        Args:
            text: Raw text
            remove_ipq: Whether to remove IPQ information

        Returns:
            Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Remove IPQ patterns if requested
        if remove_ipq:
            ipq_patterns = [
                r'(?:IPQ|Pack of|Pack Size|Quantity|Pack):?\s*\d+',
                r'(?:Value|Count):\s*\d+(?:\.\d+)?',
                r'\d+\s*(?:Pack|Count|oz|Ounce|Fl Oz|ct)',
                r'Pack of \d+',
                r'\d+-Pack',
                r'\d+\s*pieces?',
                r'\d+\s*units?'
            ]
            for pattern in ipq_patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Basic text cleaning
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.lower().strip()

        return text

    def fit_transform_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fit TF-IDF vectorizer and transform texts with SVD reduction."""
        print("🔤 Fitting TF-IDF vectorizer...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.MAX_TFIDF_FEATURES,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )

        tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        print(f"✅ TF-IDF features shape before SVD: {tfidf_features.shape}")

        # Apply TruncatedSVD for dimensionality reduction
        print(f"🔧 Applying TruncatedSVD to reduce to {self.config.TFIDF_SVD_COMPONENTS} components...")
        self.tfidf_svd = TruncatedSVD(
            n_components=self.config.TFIDF_SVD_COMPONENTS,
            random_state=self.config.CV_RANDOM_STATE
        )

        tfidf_reduced = self.tfidf_svd.fit_transform(tfidf_features)
        print(f"✅ TF-IDF features shape after SVD: {tfidf_reduced.shape}")
        print(f"📊 Explained variance ratio: {self.tfidf_svd.explained_variance_ratio_.sum():.3f}")

        # Fit K-Means clustering for category discovery
        print("🧩 Training K-Means clustering for category discovery...")
        self.kmeans_model = KMeans(
            n_clusters=30,
            random_state=self.config.CV_RANDOM_STATE,
            n_init=10
        )
        cluster_labels = self.kmeans_model.fit_predict(tfidf_reduced)
        print(f"✅ K-Means clustering completed. {len(np.unique(cluster_labels))} clusters found")

        return tfidf_reduced, cluster_labels

    def transform_tfidf(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform texts using fitted TF-IDF vectorizer, SVD, and clustering."""
        if self.tfidf_vectorizer is None or self.tfidf_svd is None or self.kmeans_model is None:
            raise ValueError("TF-IDF vectorizer, SVD, and K-Means not fitted yet!")

        tfidf_features = self.tfidf_vectorizer.transform(texts)
        tfidf_reduced = self.tfidf_svd.transform(tfidf_features)
        cluster_labels = self.kmeans_model.predict(tfidf_reduced)
        return tfidf_reduced, cluster_labels

    def extract_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract sentence embeddings using transformer model."""
        if SentenceTransformer is None:
            print("⚠️ Sentence Transformers not available, skipping embeddings")
            return np.zeros((len(texts), 384))  # Default embedding size

        print(f"🤖 Loading sentence transformer: {self.config.SENTENCE_TRANSFORMER_MODEL}")
        try:
            self.sentence_transformer = SentenceTransformer(self.config.SENTENCE_TRANSFORMER_MODEL)
            embeddings = self.sentence_transformer.encode(texts, show_progress_bar=True)
            print(f"✅ Sentence embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"⚠️ Error with sentence transformer: {e}")
            return np.zeros((len(texts), 384))

class ImageFeatureExtractor:
    """Advanced image feature extraction using transfer learning."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.transform = None
        self.device = None
        self.image_scaler = StandardScaler()  # For image feature normalization

    def setup_model(self):
        """Setup EfficientNet model for feature extraction."""
        if torch is None:
            print("⚠️ PyTorch not available, skipping image features")
            return

        model_name = "EfficientNet-B0" if self.config.USE_EFFICIENTNET_B0 else "EfficientNet-B3"
        print(f"🖼️ Setting up {model_name} for image feature extraction...")

        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"🔧 Using device: {self.device}")

            # Load pre-trained EfficientNet (B0 for speed, B3 for accuracy)
            if self.config.USE_EFFICIENTNET_B0:
                self.model = models.efficientnet_b0(pretrained=True)
            else:
                self.model = models.efficientnet_b3(pretrained=True)

            self.model.classifier = torch.nn.Identity()  # Remove final layer
            self.model.eval()
            self.model.to(self.device)

            # Define image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            print(f"✅ {model_name} model ready for feature extraction")

        except Exception as e:
            print(f"⚠️ Error setting up image model: {e}")
            self.model = None

    def download_single_image(self, image_link: str, save_folder: str) -> bool:
        """Download a single image with retry logic."""
        if not isinstance(image_link, str) or not image_link.strip():
            return False

        filename = Path(image_link).name
        if not filename:
            filename = f"image_{hash(image_link)}.jpg"

        image_save_path = os.path.join(save_folder, filename)

        if os.path.exists(image_save_path):
            return True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                urllib.request.urlretrieve(image_link, image_save_path)
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"⚠️ Failed to download {image_link}: {e}")
                    return False

        return False

    def download_images_batch(self, image_links: List[str], download_folder: str):
        """Download images in batches with multiprocessing."""
        print(f"📥 Downloading {len(image_links)} images to {download_folder}")

        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        download_func = partial(self.download_single_image, save_folder=download_folder)

        with multiprocessing.Pool(self.config.MAX_DOWNLOAD_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(download_func, image_links),
                total=len(image_links),
                desc="Downloading images"
            ))

        success_rate = sum(results) / len(results) * 100
        print(f"✅ Image download completed. Success rate: {success_rate:.1f}%")

    def load_image_from_path(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess image from file path."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            print(f"⚠️ Error loading image {image_path}: {e}")
            return None

    def load_image_from_url(self, image_url: str) -> Optional[torch.Tensor]:
        """Load and preprocess image from URL."""
        try:
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            print(f"⚠️ Error loading image from URL {image_url}: {e}")
            return None

    def extract_features_from_images(self, image_paths_or_urls: List[str],
                                   use_local_files: bool = True) -> np.ndarray:
        """Extract features from a list of images."""
        if self.model is None:
            print("⚠️ Image model not available, returning zero features")
            # Return appropriate feature size based on model choice
            feature_size = 1280 if self.config.USE_EFFICIENTNET_B0 else 1536
            return np.zeros((len(image_paths_or_urls), feature_size))

        print(f"🖼️ Extracting features from {len(image_paths_or_urls)} images...")

        features = []
        batch_images = []
        batch_size = self.config.BATCH_SIZE

        with torch.no_grad():
            for i, path_or_url in enumerate(tqdm(image_paths_or_urls, desc="Processing images")):
                # Load image
                if use_local_files:
                    image_tensor = self.load_image_from_path(path_or_url)
                else:
                    image_tensor = self.load_image_from_url(path_or_url)

                if image_tensor is not None:
                    batch_images.append(image_tensor)
                else:
                    # Add zero tensor for failed images
                    batch_images.append(torch.zeros(1, 3, self.config.IMAGE_SIZE, self.config.IMAGE_SIZE))

                # Process batch when full or at the end
                if len(batch_images) == batch_size or i == len(image_paths_or_urls) - 1:
                    if batch_images:
                        batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                        batch_features = self.model(batch_tensor)

                        # Apply Global Average Pooling if needed
                        if len(batch_features.shape) > 2:
                            batch_features = torch.mean(batch_features, dim=[2, 3])

                        features.extend(batch_features.cpu().numpy())
                        batch_images = []

        features_array = np.array(features)

        # Filter out invalid or zero embeddings
        print("🧹 Filtering out invalid image embeddings...")
        valid_mask = np.mean(features_array, axis=1) != 0
        invalid_count = np.sum(~valid_mask)

        if invalid_count > 0:
            print(f"   ⚠️ Found {invalid_count} invalid/zero embeddings, replacing with mean")
            # Replace invalid embeddings with mean of valid ones
            valid_features = features_array[valid_mask]
            if len(valid_features) > 0:
                mean_features = np.mean(valid_features, axis=0)
                features_array[~valid_mask] = mean_features
            else:
                print("   ⚠️ No valid embeddings found, using zeros")

        print(f"✅ Image features extracted. Shape: {features_array.shape}")
        return features_array

class EnsemblePredictor:
    """Advanced ensemble prediction system with stacking."""

    def __init__(self, config: Config):
        self.config = config
        self.base_models = {}
        self.meta_model = None
        self.feature_scaler = StandardScaler()
        self.is_fitted = False

    def custom_smape_lgb(self, y_pred: np.ndarray, y_true) -> Tuple[str, float, bool]:
        """Custom SMAPE metric for LightGBM."""
        y_true = y_true.get_label()
        smape_val = SMAPECalculator.smape(y_true, y_pred)
        return 'smape', smape_val, False

    def setup_base_models(self):
        """Initialize base models for stacking ensemble."""
        print("🎯 Setting up base models for ensemble...")

        # Test GPU availability for LightGBM
        lgb_params = self.config.LIGHTGBM_PARAMS.copy()
        try:
            # Try GPU first
            test_model = lgb.LGBMRegressor(**lgb_params)
            test_model.fit(np.random.rand(100, 10), np.random.rand(100))
            print("✅ GPU available for LightGBM")
        except:
            # Fallback to CPU
            print("⚠️ GPU not available for LightGBM, using CPU")
            lgb_params['device_type'] = 'cpu'

        # LightGBM
        self.base_models['lightgbm'] = lgb.LGBMRegressor(**lgb_params)

        # XGBoost
        self.base_models['xgboost'] = xgb.XGBRegressor(**self.config.XGBOOST_PARAMS)

        # CatBoost (if available)
        if cb is not None:
            catboost_params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'loss_function': 'MAE',
                'random_seed': self.config.CV_RANDOM_STATE,
                'verbose': False
            }
            self.base_models['catboost'] = cb.CatBoostRegressor(**catboost_params)

        # Meta-model (Ridge regression for stable combination)
        self.meta_model = Ridge(alpha=1.0, random_state=self.config.CV_RANDOM_STATE)

        print(f"✅ Initialized {len(self.base_models)} base models + meta-model")

    def optimize_ridge_alpha(self, oof_predictions: np.ndarray, y: np.ndarray) -> float:
        """Optimize Ridge meta-model alpha parameter."""
        print("🔧 Optimizing Ridge meta-model alpha...")

        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
        best_alpha = 1.0
        best_score = float('inf')

        # Use a simple validation split for alpha tuning
        n_val = len(y) // 5
        val_idx = np.random.choice(len(y), n_val, replace=False)
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)

        X_train_meta, X_val_meta = oof_predictions[train_idx], oof_predictions[val_idx]
        y_train_meta, y_val_meta = y[train_idx], y[val_idx]

        for alpha in alphas:
            ridge = Ridge(alpha=alpha, random_state=self.config.CV_RANDOM_STATE)
            ridge.fit(X_train_meta, y_train_meta)
            y_pred = ridge.predict(X_val_meta)
            score = SMAPECalculator.smape(y_val_meta, y_pred)

            if score < best_score:
                best_score = score
                best_alpha = alpha

        print(f"✅ Best Ridge alpha: {best_alpha} (SMAPE: {best_score:.4f})")
        return best_alpha

    def optimize_lightgbm_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Optimize LightGBM hyperparameters using Optuna."""
        if optuna is None:
            print("⚠️ Optuna not available, using default hyperparameters")
            return self.config.LIGHTGBM_PARAMS

        print("🔧 Optimizing LightGBM hyperparameters...")

        # Determine device type
        base_params = self.config.LIGHTGBM_PARAMS.copy()
        try:
            test_model = lgb.LGBMRegressor(**base_params)
            test_model.fit(np.random.rand(100, 10), np.random.rand(100))
            device_type = 'gpu'
        except:
            device_type = 'cpu'
            base_params['device_type'] = 'cpu'

        def objective(trial):
            params = base_params.copy()
            params.update({
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            })

            # Cross-validation
            kf = KFold(n_splits=3, shuffle=True, random_state=self.config.CV_RANDOM_STATE)
            scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                smape_score = SMAPECalculator.smape(y_val, y_pred)
                scores.append(smape_score)

            return np.mean(scores)

        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50, show_progress_bar=True)

            best_params = base_params.copy()
            best_params.update(study.best_params)

            print(f"✅ Best LightGBM SMAPE: {study.best_value:.4f}")
            print(f"🔧 Using device: {device_type}")
            return best_params

        except Exception as e:
            print(f"⚠️ Hyperparameter optimization failed: {e}")
            return base_params

    def fit_stacking_ensemble(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble with stratified cross-validation."""
        print(f"🎯 Training stacking ensemble with {self.config.N_FOLDS}-fold Stratified CV...")

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        # Create stratified bins for log prices (for better CV-LB correlation)
        print("📊 Creating stratified bins for cross-validation...")
        y_binned = pd.qcut(y, q=5, labels=False, duplicates='drop')

        # Optimize LightGBM hyperparameters
        optimized_lgb_params = self.optimize_lightgbm_hyperparameters(X_scaled, y)
        self.base_models['lightgbm'] = lgb.LGBMRegressor(**optimized_lgb_params)

        # Initialize stratified cross-validation
        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True,
                             random_state=self.config.CV_RANDOM_STATE)

        # Out-of-fold predictions for meta-model training
        oof_predictions = np.zeros((len(X), len(self.base_models)))
        cv_scores = {name: [] for name in self.base_models.keys()}

        # Store trained models for each fold
        fold_models = {name: [] for name in self.base_models.keys()}

        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_binned)):
            print(f"📊 Training fold {fold + 1}/{self.config.N_FOLDS}")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train each base model
            for i, (name, model) in enumerate(self.base_models.items()):
                # Clone the model for this fold
                if name == 'lightgbm':
                    fold_model = lgb.LGBMRegressor(**optimized_lgb_params)
                elif name == 'xgboost':
                    fold_model = xgb.XGBRegressor(**self.config.XGBOOST_PARAMS)
                elif name == 'catboost':
                    catboost_params = {
                        'iterations': 1000,
                        'learning_rate': 0.1,
                        'depth': 6,
                        'loss_function': 'MAE',
                        'random_seed': self.config.CV_RANDOM_STATE,
                        'verbose': False
                    }
                    fold_model = cb.CatBoostRegressor(**catboost_params)

                # Fit model
                fold_model.fit(X_train, y_train)
                fold_models[name].append(fold_model)

                # Predict on validation set
                val_predictions = fold_model.predict(X_val)
                oof_predictions[val_idx, i] = val_predictions

                # Calculate SMAPE score
                smape_score = SMAPECalculator.smape(y_val, val_predictions)
                cv_scores[name].append(smape_score)

        # Store fold models for averaging
        self.fold_models = fold_models

        # Print CV results
        print("\n📈 Cross-Validation Results:")
        total_score = 0
        for name, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"   {name}: {mean_score:.4f} ± {std_score:.4f}")
            total_score += mean_score

        # Train final base models on full dataset
        print("\n🎯 Training final base models on full dataset...")
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)

        # Train meta-model on out-of-fold predictions with optimized alpha
        print("🎯 Optimizing and training meta-model...")
        best_alpha = self.optimize_ridge_alpha(oof_predictions, y)
        self.meta_model = Ridge(alpha=best_alpha, random_state=self.config.CV_RANDOM_STATE)
        self.meta_model.fit(oof_predictions, y)

        # Calculate ensemble CV score
        ensemble_predictions = self.meta_model.predict(oof_predictions)
        ensemble_smape = SMAPECalculator.smape(y, ensemble_predictions)

        # Store average base predictions for later blending
        self.avg_base_predictions_oof = np.mean(oof_predictions, axis=1)

        print(f"\n✅ Ensemble CV SMAPE: {ensemble_smape:.4f}")
        print(f"✅ Average base model SMAPE: {total_score / len(self.base_models):.4f}")

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained ensemble with blending."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet!")

        print("🔮 Making ensemble predictions...")

        # Scale features
        X_scaled = self.feature_scaler.transform(X)

        # Get predictions from all base models (average across folds)
        base_predictions = np.zeros((len(X), len(self.base_models)))

        # Average predictions from fold models
        for i, (name, models_list) in enumerate(self.fold_models.items()):
            fold_preds = np.zeros((len(X), len(models_list)))
            for j, model in enumerate(models_list):
                fold_preds[:, j] = model.predict(X_scaled)
            base_predictions[:, i] = np.mean(fold_preds, axis=1)

        # Meta-model prediction
        ensemble_predictions = self.meta_model.predict(base_predictions)

        # Average of base models for blending
        avg_base_predictions = np.mean(base_predictions, axis=1)

        # Blend ensemble and average predictions (smoother, less overfit)
        final_predictions = (ensemble_predictions + avg_base_predictions) / 2

        print("✅ Ensemble predictions completed with blending")
        return final_predictions

class UltimatePricingPredictor:
    """
    🏆 The Ultimate Smart Product Pricing Predictor

    This is the main class that orchestrates the entire pipeline following
    the "Ensemble Everything" philosophy for maximum performance.
    """

    def __init__(self):
        self.config = Config()
        self.text_extractor = TextFeatureExtractor(self.config)
        self.image_extractor = ImageFeatureExtractor(self.config)
        self.ensemble_predictor = EnsemblePredictor(self.config)

        # Data storage
        self.train_data = None
        self.test_data = None
        self.feature_matrix_train = None
        self.feature_matrix_test = None

        print("🏆 Ultimate Pricing Predictor initialized!")
        print("💡 Philosophy: Ensemble Everything for Maximum Signal Extraction")

    def setup_directories(self):
        """Create necessary directories for organized workflow."""
        directories = [
            self.config.IMAGES_TRAIN_FOLDER,
            self.config.IMAGES_TEST_FOLDER,
            self.config.MODELS_FOLDER,
            self.config.SUBMISSIONS_FOLDER,
            self.config.FEATURES_CACHE_DIR
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        print("📁 Directory structure created successfully")

    def load_data(self):
        """Load and perform initial analysis of the dataset."""
        print("\n" + "="*60)
        print("🕵️‍♂️ PHASE 1: FOUNDATIONAL SETUP & DEEP DATA AUTOPSY")
        print("="*60)

        # Load training data
        train_path = os.path.join(self.config.DATASET_FOLDER, self.config.TRAIN_CSV)
        if os.path.exists(train_path):
            print(f"📊 Loading training data from {train_path}")
            self.train_data = pd.read_csv(train_path,engine='python')
            print(f"✅ Training data loaded: {self.train_data.shape}")
        else:
            # Use sample data for development
            print("⚠️ Full training data not found, using sample data")
            sample_path = os.path.join(self.config.DATASET_FOLDER, self.config.SAMPLE_TEST_CSV)
            self.train_data = pd.read_csv(sample_path)
            # Add dummy prices for sample data
            np.random.seed(RANDOM_SEED)
            self.train_data['price'] = np.random.lognormal(3, 1, len(self.train_data))

        # Load test data
        test_path = os.path.join(self.config.DATASET_FOLDER, self.config.TEST_CSV)
        if os.path.exists(test_path):
            print(f"📊 Loading test data from {test_path}")
            self.test_data = pd.read_csv(test_path)
        else:
            # Use sample test data
            print("⚠️ Full test data not found, using sample test data")
            self.test_data = pd.read_csv(os.path.join(self.config.DATASET_FOLDER, self.config.SAMPLE_TEST_CSV))

        print(f"✅ Test data loaded: {self.test_data.shape}")

        # Target variable analysis
        self.analyze_target_variable()

    def analyze_target_variable(self):
        """Analyze and transform the target variable (price) with outlier removal."""
        print("\n💰 Target Variable Analysis:")

        if 'price' in self.train_data.columns:
            prices = self.train_data['price'].dropna()

            print(f"   📈 Original price statistics:")
            print(f"      Mean: ${prices.mean():.2f}")
            print(f"      Median: ${prices.median():.2f}")
            print(f"      Min: ${prices.min():.2f}")
            print(f"      Max: ${prices.max():.2f}")
            print(f"      Std: ${prices.std():.2f}")

            # Remove outliers (1st-99th percentile)
            print("\n🧹 Removing price outliers (1st-99th percentile)...")
            price_p1 = prices.quantile(0.01)
            price_p99 = prices.quantile(0.99)

            outlier_mask = (self.train_data['price'] >= price_p1) & (self.train_data['price'] <= price_p99)
            original_len = len(self.train_data)
            self.train_data = self.train_data[outlier_mask].reset_index(drop=True)

            print(f"   📊 Removed {original_len - len(self.train_data)} outliers")
            print(f"   📊 Remaining samples: {len(self.train_data)}")

            # Log transformation (critical for performance)
            print("\n🔄 Applying log1p transformation to stabilize variance...")
            self.train_data['log_price'] = np.log1p(self.train_data['price'])

            log_prices = self.train_data['log_price'].dropna()
            print(f"   📈 Log-transformed price statistics:")
            print(f"      Mean: {log_prices.mean():.3f}")
            print(f"      Std: {log_prices.std():.3f}")
            print("✅ Target variable transformation completed")
        else:
            print("⚠️ No price column found in training data")

    def download_all_images(self):
        """Download all training and test images."""
        print("\n📥 Downloading Images:")

        # Setup image extractor


        # Download training images
        if 'image_link' in self.train_data.columns:
            train_links = self.train_data['image_link'].dropna().tolist()
            self.image_extractor.download_images_batch(
                train_links, self.config.IMAGES_TRAIN_FOLDER
            )

        # Download test images
        if 'image_link' in self.test_data.columns:
            test_links = self.test_data['image_link'].dropna().tolist()
            self.image_extractor.download_images_batch(
                test_links, self.config.IMAGES_TEST_FOLDER
            )
        self.image_extractor.setup_model()


    def extract_all_features(self):
        """
        🧠 PHASE 2: ADVANCED MULTI-MODAL FEATURE ENGINEERING

        Extract features from text, images, and metadata to create
        the high-quality fuel for our ensemble models.
        """
        print("\n" + "="*60)
        print("🧠 PHASE 2: ADVANCED MULTI-MODAL FEATURE ENGINEERING")
        print("="*60)

        # Extract text features
        # Process training data first to fit transformers
        train_features = self.extract_text_features(self.train_data, is_training=True)
        test_features = self.extract_text_features(self.test_data, is_training=False)

        # Extract image features
        train_image_features = self.extract_image_features(self.train_data, is_training=True)
        test_image_features = self.extract_image_features(self.test_data, is_training=False)

        # Combine all features
        print("\n🔗 Combining all features into final feature matrices...")
        print(f"   📝 Text features shape: {train_features.shape}")
        print(f"   🖼️ Image features shape: {train_image_features.shape}")

        # Features are already normalized separately, just concatenate
        self.feature_matrix_train = np.hstack([
            train_features,
            train_image_features
        ])

        self.feature_matrix_test = np.hstack([
            test_features,
            test_image_features
        ])

        print(f"✅ Final training feature matrix: {self.feature_matrix_train.shape}")
        print(f"✅ Final test feature matrix: {self.feature_matrix_test.shape}")

        # Save feature matrices
        feature_save_path = os.path.join(self.config.MODELS_FOLDER, 'feature_matrices.npz')
        np.savez_compressed(
            feature_save_path,
            train_features=self.feature_matrix_train,
            test_features=self.feature_matrix_test
        )
        print(f"💾 Feature matrices saved to {feature_save_path}")

    def extract_text_features(self, data: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """Extract comprehensive text features with caching."""
        print(f"\n🔤 Extracting text features ({'training' if is_training else 'test'} data)...")

        # Check cache first
        cache_name = f"text_features_{'train' if is_training else 'test'}.npy"
        cache_path = os.path.join(self.config.FEATURES_CACHE_DIR, cache_name)

        # Don't load cached features if it's training AND we need to fit transformers
        if self.config.CACHE_FEATURES and os.path.exists(cache_path) and not is_training:
             print(f"📦 Loading cached text features from {cache_path}")
             return np.load(cache_path)


        # Extract IPQ (Item Pack Quantity)
        print("🔢 Extracting IPQ (Item Pack Quantity)...")
        ipq_values = []
        for text in data['catalog_content']:
            ipq = self.text_extractor.extract_ipq(text)
            ipq_values.append(ipq)

        data = data.copy()
        data['ipq'] = ipq_values

        print(f"   📊 IPQ statistics:")
        print(f"      Mean: {np.mean(ipq_values):.2f}")
        print(f"      Max: {np.max(ipq_values)}")
        print(f"      Unique values: {len(np.unique(ipq_values))}")

        # Brand feature removed to prevent overfitting
        print("🏷️ Skipping brand features (removed to prevent overfitting)")

        # Clean text
        print("🧽 Cleaning text data...")
        cleaned_texts = []
        for text in data['catalog_content']:
            cleaned = self.text_extractor.clean_text(text, remove_ipq=True)
            cleaned_texts.append(cleaned)

        # Extract TF-IDF features
        if is_training:
            tfidf_features, cluster_labels = self.text_extractor.fit_transform_tfidf(cleaned_texts)
        else:
            tfidf_features, cluster_labels = self.text_extractor.transform_tfidf(cleaned_texts)

        # Extract sentence embeddings
        if is_training:
            sentence_embeddings = self.text_extractor.extract_sentence_embeddings(cleaned_texts)
        else:
            sentence_embeddings = self.text_extractor.extract_sentence_embeddings(cleaned_texts)

        # Extract specifications
        print("📏 Extracting product specifications...")
        all_specs = []
        for text in data['catalog_content']:
            specs = self.text_extractor.extract_specs(text)
            all_specs.append(specs)

        # Convert specs to feature matrix
        spec_columns = ['volume_ml', 'weight_g', 'storage_gb', 'length_mm']
        spec_matrix = np.zeros((len(all_specs), len(spec_columns)))

        for i, specs in enumerate(all_specs):
            for j, col in enumerate(spec_columns):
                spec_matrix[i, j] = specs.get(col, 0.0)

        # Log transform specs (they can be very skewed)
        spec_matrix = np.log1p(spec_matrix)

        print(f"   📊 Spec features shape: {spec_matrix.shape}")
        print(f"   📊 Non-zero specs: {np.sum(spec_matrix > 0)}")

        # Combine text features (without brand features)
        ipq_array = np.array(ipq_values).reshape(-1, 1)
        cluster_array = np.array(cluster_labels).reshape(-1, 1)

        # Combine all text-based features before normalization
        raw_text_features = np.hstack([
            tfidf_features,
            sentence_embeddings,
            ipq_array,
            cluster_array,
            spec_matrix
        ])

        # Normalize text features separately
        if is_training:
            text_features = self.text_extractor.text_scaler.fit_transform(raw_text_features)
        else:
            text_features = self.text_extractor.text_scaler.transform(raw_text_features)

        print(f"✅ Text features shape: {text_features.shape}")

        # Cache the features
        if self.config.CACHE_FEATURES:
            np.save(cache_path, text_features)
            print(f"💾 Text features cached to {cache_path}")

        return text_features

    def extract_image_features(self, data: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """Extract image features using transfer learning with caching."""
        print(f"\n🖼️ Extracting image features ({'training' if is_training else 'test'} data)...")

        # Check cache first
        cache_name = f"image_features_{'train' if is_training else 'test'}.npy"
        cache_path = os.path.join(self.config.FEATURES_CACHE_DIR, cache_name)

        # Don't load cached features if it's training AND we need to fit the scaler
        if self.config.CACHE_FEATURES and os.path.exists(cache_path) and not is_training:
            print(f"📦 Loading cached image features from {cache_path}")
            return np.load(cache_path)


        # Prepare image paths
        image_folder = self.config.IMAGES_TRAIN_FOLDER if is_training else self.config.IMAGES_TEST_FOLDER

        image_paths = []
        for image_link in data['image_link']:
            if pd.notna(image_link):
                filename = Path(image_link).name
                if not filename:
                    filename = f"image_{hash(image_link)}.jpg"
                image_path = os.path.join(image_folder, filename)
                image_paths.append(image_path)
            else:
                image_paths.append("")  # Empty path for missing images

        # Extract features
        raw_image_features = self.image_extractor.extract_features_from_images(
            image_paths, use_local_files=True
        )

        # Normalize image features separately and apply weighting
        if is_training:
            image_features = self.image_extractor.image_scaler.fit_transform(raw_image_features)
        else:
            image_features = self.image_extractor.image_scaler.transform(raw_image_features)

        # Weight image features (multiply by 0.3) so text features dominate slightly
        image_features = image_features * 0.3

        # Cache the features
        if self.config.CACHE_FEATURES:
            np.save(cache_path, image_features)
            print(f"💾 Image features cached to {cache_path}")

        print(f"✅ Image features shape: {image_features.shape}")
        return image_features


    def train_ensemble(self):
        """
        🚀 PHASE 3: THE STACKING ENSEMBLE MODEL ARCHITECTURE

        Train the advanced stacking ensemble with cross-validation.
        """
        print("\n" + "="*60)
        print("🚀 PHASE 3: THE STACKING ENSEMBLE MODEL ARCHITECTURE")
        print("="*60)

        if self.feature_matrix_train is None:
            raise ValueError("Features not extracted yet! Call extract_all_features() first.")

        if 'log_price' not in self.train_data.columns:
            raise ValueError("Target variable not prepared! Call analyze_target_variable() first.")

        # Setup ensemble
        self.ensemble_predictor.setup_base_models()

        # Prepare target variable
        y_train = self.train_data['log_price'].values

        # Train ensemble
        self.ensemble_predictor.fit_stacking_ensemble(self.feature_matrix_train, y_train)

        # Save trained model
        model_save_path = os.path.join(self.config.MODELS_FOLDER, 'ensemble_model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump(self.ensemble_predictor, f)

        print(f"💾 Ensemble model saved to {model_save_path}")

    def make_predictions(self):
        """
        📈 PHASE 4: FINAL PREDICTION & SUBMISSION

        Deploy the trained ensemble on test data and create submission.
        """
        print("\n" + "="*60)
        print("📈 PHASE 4: FINAL PREDICTION & SUBMISSION")
        print("="*60)

        if self.feature_matrix_test is None:
            raise ValueError("Test features not extracted!")

        if not self.ensemble_predictor.is_fitted:
            raise ValueError("Ensemble not trained!")

        # Make predictions (in log space)
        log_predictions = self.ensemble_predictor.predict(self.feature_matrix_test)

        # Inverse transformation (critical step!)
        print("🔄 Applying inverse log transformation...")
        predicted_prices = np.expm1(log_predictions)  # Inverse of log1p

        # Calculate price bounds for clipping based on training data
        if 'price' in self.train_data.columns:
            price_p999 = self.train_data['price'].quantile(0.999)
            print(f"📊 Price 99.9th percentile for clipping: ${price_p999:.2f}")
        else:
            price_p999 = 1000  # Default upper bound

        # Clip predictions to prevent extreme overpredictions
        print("✂️ Clipping predictions to prevent extreme values...")
        predicted_prices = np.clip(predicted_prices, 0, price_p999)

        print(f"💰 Prediction statistics:")
        print(f"   Mean: ${predicted_prices.mean():.2f}")
        print(f"   Median: ${predicted_prices.median():.2f}")
        print(f"   Min: ${predicted_prices.min():.2f}")
        print(f"   Max: ${predicted_prices.max():.2f}")

        # Create submission
        submission_df = pd.DataFrame({
            'sample_id': self.test_data['sample_id'],
            'price': predicted_prices
        })

        # Save submission
        submission_path = os.path.join('/teamspace/studios/this_studio', self.config.OUTPUT_CSV)

        submission_df.to_csv(submission_path, index=False)

        print(f"📄 Submission saved to {submission_path}")
        print(f"✅ Total predictions: {len(submission_df)}")
        print("\n🎯 Sample predictions:")
        print(submission_df.head(10))

        return submission_df

    def run_complete_pipeline(self):
        """
        🏆 Execute the complete Ultimate Winning Blueprint pipeline.

        This method orchestrates the entire process from data loading
        to final submission generation.
        """
        start_time = time.time()

        print("🏆" + "="*58 + "🏆")
        print("🏆 THE ULTIMATE WINNING BLUEPRINT: SMART PRODUCT PRICING 🏆")
        print("🏆" + "="*58 + "🏆")
        print("💡 Core Philosophy: Ensemble Everything")
        print("🎯 Target Metric: SMAPE (Symmetric Mean Absolute Percentage Error)")
        print("🚀 Strategy: Multi-modal Feature Engineering + Stacking Ensemble")

        try:
            # Phase 1: Setup and Data Loading
            self.setup_directories()
            self.load_data()

            # Phase 1.5: Download Images (if available)
            if torch is not None:
                self.download_all_images()

            # Phase 2: Feature Engineering
            self.extract_all_features()

            # Phase 3: Model Training
            self.train_ensemble()

            # Phase 4: Prediction and Submission
            submission_df = self.make_predictions()

            # Success!
            end_time = time.time()
            total_time = end_time - start_time

            print("\n" + "🏆"*20)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            print("🏆"*20)
            print(f"⏱️ Total execution time: {total_time:.2f} seconds")
            print(f"📊 Predictions generated: {len(submission_df)}")
            print("🚀 Ready for submission!")

            return submission_df

        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {e}")
            print("🔧 Check the error details and retry.")
            raise

# Simple predictor function for basic usage (maintains compatibility)
def predictor(sample_id, catalog_content, image_link):
    """
    Simple predictor function for compatibility.

    Note: This is a basic version. For full performance,
    use the UltimatePricingPredictor class with the complete pipeline.
    """
    # Basic IPQ extraction
    ipq = 1
    if isinstance(catalog_content, str):
        patterns = [
            r'(?:IPQ|Pack of|Pack Size|Quantity|Pack):?\s*(\d+)',
            r'(?:Value|Count):\s*(\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, catalog_content, re.IGNORECASE)
            if matches:
                try:
                    ipq = max(int(float(match)) for match in matches)
                    break
                except ValueError:
                    continue

    # Simple text-based pricing logic
    text_length = len(str(catalog_content)) if catalog_content else 0

    # Basic price estimation
    base_price = 20.0
    price_factor = 1.0 + (text_length / 1000) + (ipq / 10)

    # Add some randomization for variety
    random_factor = random.uniform(0.8, 1.2)

    estimated_price = base_price * price_factor * random_factor

    return round(max(1.0, estimated_price), 2)

def main():
    """Main execution function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        # Simple mode using basic predictor
        print("🔧 Running in simple mode...")

        # Read test data
        test_path = os.path.join('dataset', 'test.csv')
        if not os.path.exists(test_path):
            test_path = os.path.join('dataset', 'sample_test.csv')

        test_df = pd.read_csv(test_path)

        # Apply predictor
        predictions = []
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
            price = predictor(row['sample_id'], row['catalog_content'], row['image_link'])
            predictions.append(price)

        # Save results
        output_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })

        output_path = os.path.join('dataset', 'test_out.csv')
        output_df.to_csv(output_path, index=False)

        print(f"✅ Simple predictions saved to {output_path}")
        print(f"📊 Total predictions: {len(output_df)}")

    else:
        # Full pipeline mode
        print("🏆 Running Ultimate Pricing Predictor pipeline...")

        # Initialize and run the complete pipeline
        predictor_system = UltimatePricingPredictor()
        submission_df = predictor_system.run_complete_pipeline()

        return submission_df

if __name__ == "__main__":
    main()
