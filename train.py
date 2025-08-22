import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Handle CatBoost import gracefully
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except (ImportError, ValueError) as e:
    print(f"Warning: CatBoost import issue (non-fatal for deployment): {e}")
    CATBOOST_AVAILABLE = False
    # Create dummy classes for deployment
    class CatBoostRegressor:
        pass
    class Pool:
        pass

warnings.filterwarnings('ignore')

class EnhancedCatBoostPredictor:
    """Advanced predictor with deeper CatBoost, strength-specific models, and non-linear ensemble."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.setup_logging()
        np.random.seed(random_state)

    def setup_logging(self):
        """Set up logging for the class."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler('enhanced_catboost_predictor.log')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def engineer_features(self, X, for_training=True):
        """Create domain-specific engineered features for concrete strength prediction.
        
        Args:
            X: Input DataFrame
            for_training: If True, calculate and store statistics. If False, use stored statistics.
        """
        # Create a copy of the original dataframe
        X_engineered = X.copy()
        
        # Extract component names for readability
        cement = X['Cement (component 1)(kg in a m^3 mixture)']
        blast_slag = X['Blast Furnace Slag (component 2)(kg in a m^3 mixture)']
        fly_ash = X['Fly Ash (component 3)(kg in a m^3 mixture)']
        water = X['Water  (component 4)(kg in a m^3 mixture)']
        superplast = X['Superplasticizer (component 5)(kg in a m^3 mixture)']
        coarse_agg = X['Coarse Aggregate  (component 6)(kg in a m^3 mixture)']
        fine_agg = X['Fine Aggregate (component 7)(kg in a m^3 mixture)']
        age = X['Age (day)']
        
        # 1. Key concrete engineering ratios
        X_engineered['water_cement_ratio'] = water / (cement + 1e-5)
        X_engineered['total_cementitious'] = cement + blast_slag + fly_ash
        X_engineered['water_cementitious_ratio'] = water / (X_engineered['total_cementitious'] + 1e-5)
        X_engineered['agg_cement_ratio'] = (coarse_agg + fine_agg) / (cement + 1e-5)
        X_engineered['fine_coarse_ratio'] = fine_agg / (coarse_agg + 1e-5)
        
        # 2. Advanced cement chemistry features
        X_engineered['cementitious_superplast_ratio'] = X_engineered['total_cementitious'] / (superplast + 1e-5)
        X_engineered['cement_binder_ratio'] = cement / (X_engineered['total_cementitious'] + 1e-5)
        
        # 3. Time-dependent features
        X_engineered['log_age'] = np.log1p(age)
        X_engineered['sqrt_age'] = np.sqrt(age)
        X_engineered['age_28d_ratio'] = age / 28.0  # Normalization by standard 28-day strength
        
        # 4. Physical parameter approximations
        X_engineered['paste_volume'] = (cement / 3.15 + blast_slag / 2.9 + fly_ash / 2.3 + water) / \
                                      ((cement / 3.15 + blast_slag / 2.9 + fly_ash / 2.3 + water +
                                      coarse_agg / 2.7 + fine_agg / 2.6) + 1e-5)
        
        # 5. Practical concrete mix indicators
        X_engineered['slump_indicator'] = water + 10 * superplast
        X_engineered['flow_indicator'] = X_engineered['slump_indicator'] / X_engineered['total_cementitious']
        
        # 6. Concrete maturity index
        X_engineered['maturity_index'] = age * (1 - np.exp(-0.05 * age))
        
        # 7. Supplementary material utilization
        X_engineered['supplementary_fraction'] = (blast_slag + fly_ash) / (X_engineered['total_cementitious'] + 1e-5)
        
        # Enhanced age-related features
        X_engineered['early_age_factor'] = np.where(X_engineered['Age (day)'] < 7,
                                                (7 - X_engineered['Age (day)'])/7, 0)
        X_engineered['very_early_strength'] = X_engineered['Age (day)']**0.5 * X_engineered['Cement (component 1)(kg in a m^3 mixture)']
        
        # Early hydration rate approximation
        X_engineered['early_hydration_rate'] = np.where(
            X_engineered['Age (day)'] < 7,
            X_engineered['Cement (component 1)(kg in a m^3 mixture)'] / (X_engineered['Age (day)'] + 0.5),
            0
        )
        
        # Late-age strength gain factor
        X_engineered['late_age_factor'] = np.where(
            X_engineered['Age (day)'] > 28,
            np.log1p(X_engineered['Age (day)'] - 28) / 4,
            0
        )
        
        # CRITICAL PART: Handle statistics properly
        if for_training:
            # During training: calculate and store statistics
            self.feature_stats = {
                'total_cementitious_mean': X_engineered['total_cementitious'].mean(),
                'total_cementitious_std': X_engineered['total_cementitious'].std(),
                'water_cement_ratio_mean': X_engineered['water_cement_ratio'].mean(),
                'water_cement_ratio_std': X_engineered['water_cement_ratio'].std(),
            }
            print(f"📊 Calculated feature statistics during training: {self.feature_stats}")
            
            # Use the calculated statistics
            total_cem_mean = self.feature_stats['total_cementitious_mean']
            water_cem_ratio_mean = self.feature_stats['water_cement_ratio_mean']
            water_cem_ratio_std = self.feature_stats['water_cement_ratio_std']
        else:
            # During prediction: use stored statistics
            if not hasattr(self, 'feature_stats') or not self.feature_stats:
                # Use default values if not found
                self.feature_stats = {
                    'total_cementitious_mean': 409.248,
                    'total_cementitious_std': 92.783,
                    'water_cement_ratio_mean': 0.748,
                    'water_cement_ratio_std': 0.314,
                }
            
            total_cem_mean = self.feature_stats['total_cementitious_mean']
            water_cem_ratio_mean = self.feature_stats['water_cement_ratio_mean']
            water_cem_ratio_std = self.feature_stats['water_cement_ratio_std']
        
        # Apply corrections using the statistics
        X_engineered['very_low_correction'] = np.where(
            X_engineered['total_cementitious'] < total_cem_mean,
            -0.05 * X_engineered['water_cementitious_ratio'],
            0
        )
        
        X_engineered['high_correction'] = np.where(
            X_engineered['total_cementitious'] > total_cem_mean * 1.2,
            0.05 * X_engineered['cement_binder_ratio'],
            0
        )
        
        # Feature to detect abnormal mix designs
        if water_cem_ratio_std > 0:
            X_engineered['abnormal_mix_factor'] = np.abs(
                (X_engineered['water_cement_ratio'] - water_cem_ratio_mean) /
                water_cem_ratio_std
            )
        else:
            X_engineered['abnormal_mix_factor'] = 0
        
        # Specialized feature for medium strength correction
        X_engineered['medium_correction'] = np.where(
            (X_engineered['total_cementitious'] >= 350) & 
            (X_engineered['total_cementitious'] <= 450) & 
            (X_engineered['water_cement_ratio'] <= 0.5),
            -0.1 * X_engineered['total_cementitious'],
            0
        )
        
        # Feature for very low strength concrete with high water content
        X_engineered['water_excess_indicator'] = np.where(
            X_engineered['water_cement_ratio'] > 0.6,
            X_engineered['water_cement_ratio'] - 0.6,
            0
        )
        
        # Store feature information (only during training)
        if for_training:
            self.original_features = X.columns.tolist()
            self.engineered_features = [col for col in X_engineered.columns if col not in self.original_features]
        
        return X_engineered

    # Keep all other methods exactly the same...
    # [Rest of your class methods remain unchanged]

    def load_and_preprocess(self, filepath):
        """Load data and preprocess with enhanced feature engineering."""
        try:
            self.data = pd.read_excel(filepath)
            self.logger.info("Data loaded successfully")

            # Split features and target
            X = self.data.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
            y = self.data['Concrete compressive strength(MPa, megapascals) ']

            # Create engineered features
            X_engineered = self.engineer_features(X, for_training=True)
            self.logger.info(f"Created {len(self.engineered_features)} new engineered features")

            # Create strength ranges for stratified sampling and range-specific models
            strength_bins = [0, 20, 40, 60, 100]
            strength_labels = ['very_low', 'low', 'medium', 'high']
            y_ranges = pd.cut(y, bins=strength_bins, labels=strength_labels)
            self.y_ranges = y_ranges
            self.strength_bins = strength_bins
            self.strength_labels = strength_labels

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_engineered),
                columns=X_engineered.columns
            )
            X_scaled = X_scaled.reset_index(drop=True)

            # Store all features
            self.all_features = X_scaled.columns.tolist()

            # Split data with stratification by strength ranges
            X_train, X_test, y_train, y_test, y_ranges_train, y_ranges_test = train_test_split(
                X_scaled, y, y_ranges,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_ranges
            )
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.y_ranges_train = y_ranges_train
            self.y_ranges_test = y_ranges_test

            print(f"Data split: {X_train.shape} training, {X_test.shape} testing")
            print("\nStrength range distribution in test set:")
            for label in strength_labels:
                count = np.sum(y_ranges_test == label)
                pct = count / len(y_ranges_test) * 100
                print(f"  {label.replace('_', ' ').title()}: {count} samples ({pct:.1f}%)")

            return X_train, X_test, y_train, y_test, y_ranges_train, y_ranges_test

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def train_deep_catboost(self):
        """Train a deeper CatBoost model with optimized parameters."""
        if not CATBOOST_AVAILABLE:
            print("CatBoost is not available. Skipping training.")
            return None, None
            
        try:
            print("\nTraining deep CatBoost model...")

            # Create CatBoost model with deeper architecture
            deep_catboost = CatBoostRegressor(
                iterations=2000,          # Increased iterations
                learning_rate=0.02,       # Reduced learning rate
                depth=8,                  # Increased depth
                l2_leaf_reg=3,
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=self.random_state,
                od_type='Iter',
                od_wait=100,              # More patience
                verbose=100,
                task_type='CPU',          # Use 'GPU' if available
                # Advanced parameters
                bootstrap_type='Bayesian',
                bagging_temperature=1,
                grow_policy='SymmetricTree',
                min_data_in_leaf=5
            )

            # Create train and eval pools
            train_pool = Pool(self.X_train, self.y_train)
            eval_pool = Pool(self.X_test, self.y_test)

            # Train model
            deep_catboost.fit(
                train_pool,
                eval_set=eval_pool,
                use_best_model=True,
                verbose=100
            )

            # Make predictions
            y_pred = deep_catboost.predict(self.X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(self.y_test, y_pred)
            print("\nDeep CatBoost Model Metrics:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

            # Feature importance
            importance = deep_catboost.get_feature_importance()
            feature_importance = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': importance
            }).sort_values('Importance', ascending=False)

            print("\nTop 10 Features by Importance:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['Feature']}: {row['Importance']}")

            self.deep_catboost = deep_catboost
            self.catboost_feature_importance = feature_importance
            self.catboost_metrics = metrics
            self.catboost_preds = y_pred

            return metrics, y_pred

        except ImportError:
            print("CatBoost is not installed. Please install it using: pip install catboost")
            return None, None
        
    def train_range_specific_models(self):
        """Train separate models for different concrete strength ranges."""
        try:
            from catboost import CatBoostRegressor, Pool
            print("\nTraining strength range-specific models...")

            self.range_models = {}
            self.range_preds = {}

            # Updated parameters for different ranges with more focus on problematic ranges
            range_params = {
                'very_low': {  # Less than 20 MPa - Highest error rate
                    'iterations': 2000,        # Increased from 1000
                    'depth': 7,                # Increased from 6
                    'learning_rate': 0.02,     # Lower for more stability
                    'l2_leaf_reg': 5,          # Increased regularization
                    'bootstrap_type': 'Bayesian',
                    'min_data_in_leaf': 5,     # Increased to prevent overfitting
                    'random_strength': 0.9     # Increased randomization
                },
                'low': {  # 20-40 MPa
                    'iterations': 1500,
                    'depth': 7,
                    'learning_rate': 0.02,
                    'l2_leaf_reg': 3,
                    'bootstrap_type': 'Bayesian'
                },
                'medium': {  # 40-60 MPa
                    'iterations': 1500,
                    'depth': 8,
                    'learning_rate': 0.02,
                    'l2_leaf_reg': 3
                },
                'high': {  # Over 60 MPa - Few samples but high error rate
                    'iterations': 1200,        # Increased from 1000
                    'depth': 7,                # Increased from 6
                    'learning_rate': 0.015,    # Lower for stability
                    'l2_leaf_reg': 4,
                    'bootstrap_type': 'Bayesian',
                    'bagging_temperature': 1.5 # More aggressive bagging for few samples
                }
            }

            # Train separate model for each strength range
            for strength_range in self.strength_labels:
                print(f"\nTraining model for {strength_range.replace('_', ' ').title()} Strength range...")

                # Make sure indices are aligned properly - convert to numpy arrays if needed
                y_ranges_train_array = np.array(self.y_ranges_train)
                train_mask = (y_ranges_train_array == strength_range)

                # Check if we have enough samples
                if np.sum(train_mask) < 10:
                    print(f"  Not enough samples for {strength_range} range. Skipping.")
                    continue

                # Use .loc with indices to avoid alignment issues
                train_indices = np.where(train_mask)[0]
                X_train_range = self.X_train.iloc[train_indices]
                y_train_range = self.y_train.iloc[train_indices]

                # Similarly for test data
                y_ranges_test_array = np.array(self.y_ranges_test)
                test_mask = (y_ranges_test_array == strength_range)
                test_indices = np.where(test_mask)[0]

                if len(test_indices) < 5:
                    print(f"  Not enough test samples for {strength_range} range. Skipping metrics calculation.")
                    test_samples = 0
                else:
                    X_test_range = self.X_test.iloc[test_indices]
                    y_test_range = self.y_test.iloc[test_indices]
                    test_samples = len(X_test_range)

                print(f"  Training samples: {len(X_train_range)}, Test samples: {test_samples}")

                # Get parameters for this range
                model_params = range_params.get(strength_range, range_params['low'])  # Default to low params if not found

                # Create and train model with range-specific parameters
                range_model = CatBoostRegressor(
                    iterations=model_params['iterations'],
                    learning_rate=model_params['learning_rate'],
                    depth=model_params['depth'],
                    l2_leaf_reg=model_params.get('l2_leaf_reg', 3),
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    random_seed=self.random_state,
                    od_type='Iter',
                    od_wait=50,
                    verbose=100,
                    bootstrap_type=model_params.get('bootstrap_type', 'Bayesian'),
                    min_data_in_leaf=model_params.get('min_data_in_leaf', 5),
                    random_strength=model_params.get('random_strength', 0.5),
                    bagging_temperature=model_params.get('bagging_temperature', 1.0)
                )

                # Create train pool
                train_pool = Pool(X_train_range, y_train_range)

                # Create eval pool if we have enough test samples
                if test_samples >= 5:
                    eval_pool = Pool(X_test_range, y_test_range)

                    # Train model with eval set
                    range_model.fit(
                        train_pool,
                        eval_set=eval_pool,
                        use_best_model=True,
                        verbose=100
                    )

                    # Calculate metrics
                    y_pred_range = range_model.predict(X_test_range)
                    metrics = self._calculate_metrics(y_test_range, y_pred_range)

                    print(f"  {strength_range.replace('_', ' ').title()} Range Model Metrics:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value}")
                else:
                    # Train model without eval set
                    range_model.fit(
                        train_pool,
                        verbose=100
                    )

                # Store model
                self.range_models[strength_range] = range_model

                # Make predictions on full test set (for blending later)
                self.range_preds[strength_range] = range_model.predict(self.X_test)

            return self.range_models, self.range_preds

        except Exception as e:
            print(f"Error in training range-specific models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def train_very_low_specialized_models(self):
        """Train ultra-specialized models for very low strength concrete."""
        try:
            from catboost import CatBoostRegressor, Pool
            print("\nTraining specialized models for very low strength concrete...")

            # Get only very low samples using numpy arrays to avoid indexing issues
            y_ranges_train_array = np.array(self.y_ranges_train)
            mask = (y_ranges_train_array == 'very_low')

            # Check if we have enough samples
            if np.sum(mask) < 10:
                print("  Not enough very low strength samples. Skipping.")
                return {}, {}

            # Use indices instead of boolean masks
            train_indices = np.where(mask)[0]
            X_very_low = self.X_train.iloc[train_indices]
            y_very_low = self.y_train.iloc[train_indices]

            # Further split by actual strength for more specialization
            y_very_low_array = np.array(y_very_low)
            low_mask = y_very_low_array < 15  # Ultra-low strength
            mid_mask = (y_very_low_array >= 15) & (y_very_low_array < 20)  # Mid-low strength

            self.very_low_specialized_models = {}
            self.very_low_specialized_preds = {}

            # Ultra-low strength model
            if np.sum(low_mask) >= 10:
                # Get indices for the ultra-low samples
                ultra_low_indices = np.where(low_mask)[0]

                print(f"  Training ultra-low strength model (<15 MPa) with {len(ultra_low_indices)} samples")
                ultra_low_model = CatBoostRegressor(
                    iterations=1500,
                    depth=5,  # Lower depth to prevent overfitting on small samples
                    learning_rate=0.01,  # Lower learning rate for stability
                    l2_leaf_reg=6,  # Higher regularization
                    min_data_in_leaf=3,
                    verbose=0,
                    random_seed=self.random_state
                )

                # Select rows using iloc with indices
                X_ultra_low = X_very_low.iloc[ultra_low_indices]
                y_ultra_low = y_very_low.iloc[ultra_low_indices]

                ultra_low_model.fit(X_ultra_low, y_ultra_low)
                self.very_low_specialized_models['ultra_low'] = ultra_low_model

                # Make predictions on test set
                self.very_low_specialized_preds['ultra_low'] = np.zeros(len(self.X_test))

                # Identify test samples that would use this model
                # - First get very_low test samples
                y_ranges_test_array = np.array(self.y_ranges_test)
                test_mask = (y_ranges_test_array == 'very_low')
                test_indices = np.where(test_mask)[0]

                # - Then identify which ones are <15 MPa
                deep_preds = self.deep_catboost.predict(self.X_test)
                ultra_low_test_mask = (deep_preds < 15)

                # - Find intersection of very_low and <15 MPa
                X_test_very_low = self.X_test.iloc[test_indices]
                deep_preds_very_low = deep_preds[test_indices]
                ultra_low_test_indices = np.where(deep_preds_very_low < 15)[0]

                if len(ultra_low_test_indices) > 0:
                    # Calculate metrics
                    X_test_ultra_low = X_test_very_low.iloc[ultra_low_test_indices]
                    y_test_ultra_low = self.y_test.iloc[test_indices].iloc[ultra_low_test_indices]

                    ultra_low_preds = ultra_low_model.predict(X_test_ultra_low)
                    metrics = self._calculate_metrics(y_test_ultra_low, ultra_low_preds)

                    print(f"  Ultra-Low Strength Model Metrics (test samples: {len(ultra_low_test_indices)}):")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value}")

                    # Store predictions for meta-learner - using all test indices
                    for idx, very_low_idx in enumerate(test_indices):
                        if idx in ultra_low_test_indices:
                            test_sample = self.X_test.iloc[[very_low_idx]]
                            self.very_low_specialized_preds['ultra_low'][very_low_idx] = ultra_low_model.predict(test_sample)[0]

            # Mid-low strength model
            if np.sum(mid_mask) >= 10:
                # Get indices for the mid-low samples
                mid_low_indices = np.where(mid_mask)[0]

                print(f"  Training mid-low strength model (15-20 MPa) with {len(mid_low_indices)} samples")
                mid_low_model = CatBoostRegressor(
                    iterations=1500,
                    depth=6,
                    learning_rate=0.015,
                    l2_leaf_reg=4,
                    min_data_in_leaf=3,
                    verbose=0,
                    random_seed=self.random_state
                )

                # Select rows using iloc with indices
                X_mid_low = X_very_low.iloc[mid_low_indices]
                y_mid_low = y_very_low.iloc[mid_low_indices]

                mid_low_model.fit(X_mid_low, y_mid_low)
                self.very_low_specialized_models['mid_low'] = mid_low_model

                # Make predictions on test set
                self.very_low_specialized_preds['mid_low'] = np.zeros(len(self.X_test))

                # Identify test samples that would use this model
                # - First get very_low test samples
                y_ranges_test_array = np.array(self.y_ranges_test)
                test_mask = (y_ranges_test_array == 'very_low')
                test_indices = np.where(test_mask)[0]

                # - Then identify which ones are 15-20 MPa
                deep_preds = self.deep_catboost.predict(self.X_test)

                # - Find intersection of very_low and 15-20 MPa
                X_test_very_low = self.X_test.iloc[test_indices]
                deep_preds_very_low = deep_preds[test_indices]
                mid_low_test_indices = np.where((deep_preds_very_low >= 15) & (deep_preds_very_low < 20))[0]

                if len(mid_low_test_indices) > 0:
                    # Calculate metrics
                    X_test_mid_low = X_test_very_low.iloc[mid_low_test_indices]
                    y_test_mid_low = self.y_test.iloc[test_indices].iloc[mid_low_test_indices]

                    mid_low_preds = mid_low_model.predict(X_test_mid_low)
                    metrics = self._calculate_metrics(y_test_mid_low, mid_low_preds)

                    print(f"  Mid-Low Strength Model Metrics (test samples: {len(mid_low_test_indices)}):")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value}")

                    # Store predictions for meta-learner - using all test indices
                    for idx, very_low_idx in enumerate(test_indices):
                        if idx in mid_low_test_indices:
                            test_sample = self.X_test.iloc[[very_low_idx]]
                            self.very_low_specialized_preds['mid_low'][very_low_idx] = mid_low_model.predict(test_sample)[0]

            return self.very_low_specialized_models, self.very_low_specialized_preds

        except Exception as e:
            print(f"Error in training very low specialized models: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}, {}

    def train_medium_bias_correction(self):
        """Create a bias correction model specifically for medium range."""
        try:
            from catboost import CatBoostRegressor
            print("\nTraining medium range bias correction model...")

            # Identify medium range samples using numpy arrays
            y_ranges_train_array = np.array(self.y_ranges_train)
            mask = (y_ranges_train_array == 'medium')

            # Get indices from mask
            train_indices = np.where(mask)[0]

            if len(train_indices) < 20:
                print("  Not enough medium range samples for bias correction. Skipping.")
                return None, None

            # Use indices to select rows
            X_medium = self.X_train.iloc[train_indices]
            y_medium = self.y_train.iloc[train_indices]

            # Calculate how much our main model over-predicts
            main_preds = self.deep_catboost.predict(X_medium)
            bias = main_preds - y_medium

            print(f"  Average bias in medium range: {bias.mean():.2f} MPa")
            print(f"  Max bias in medium range: {bias.max():.2f} MPa")

            # Train a model to predict this bias
            bias_model = CatBoostRegressor(
                iterations=800,
                depth=4,
                learning_rate=0.01,
                l2_leaf_reg=5,
                verbose=0,
                random_seed=self.random_state
            )

            bias_model.fit(X_medium, bias)
            self.medium_bias_model = bias_model

            # Make predictions on medium range test samples
            y_ranges_test_array = np.array(self.y_ranges_test)
            medium_test_mask = (y_ranges_test_array == 'medium')
            test_indices = np.where(medium_test_mask)[0]

            if len(test_indices) > 0:
                X_test_medium = self.X_test.iloc[test_indices]
                y_test_medium = self.y_test.iloc[test_indices]

                # Get the deep model predictions
                deep_preds_medium = self.deep_catboost.predict(X_test_medium)

                # Get the estimated bias
                estimated_bias = self.medium_bias_model.predict(X_test_medium)

                # Apply bias correction
                corrected_preds = deep_preds_medium - estimated_bias * 0.7  # 70% of the bias

                # Calculate metrics
                uncorrected_metrics = self._calculate_metrics(y_test_medium, deep_preds_medium)
                corrected_metrics = self._calculate_metrics(y_test_medium, corrected_preds)

                print("\n  Medium Range Before Correction:")
                for metric, value in uncorrected_metrics.items():
                    print(f"    {metric}: {value}")

                print("\n  Medium Range After Correction:")
                for metric, value in corrected_metrics.items():
                    print(f"    {metric}: {value}")

                # Store the bias predictions for meta-learner
                self.medium_bias_preds = np.zeros(len(self.X_test))
                for i, idx in enumerate(test_indices):
                    self.medium_bias_preds[idx] = estimated_bias[i]

            return self.medium_bias_model, getattr(self, 'medium_bias_preds', None)

        except Exception as e:
            print(f"Error in training medium bias correction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def train_boundary_models(self):
        """Train specialized models for boundary regions between strength ranges."""
        try:
            from catboost import CatBoostRegressor, Pool
            print("\nTraining boundary region models...")

            self.boundary_models = {}
            self.boundary_preds = {}

            # Define boundary regions with 2 MPa overlap on each side
            boundary_regions = [
                (15, 25, 'very_low_low_boundary'),  # Between very_low and low
                (38, 42, 'low_medium_boundary'),    # Between low and medium
                (58, 62, 'medium_high_boundary')    # Between medium and high
            ]

            for low_bound, high_bound, name in boundary_regions:
                print(f"\nTraining model for {name.replace('_', ' ').title()} region...")

                # Use numpy arrays to avoid indexing issues
                y_train_array = np.array(self.y_train)
                mask = (y_train_array >= low_bound) & (y_train_array <= high_bound)

                # Check if we have enough samples
                sample_count = np.sum(mask)

                if sample_count < 20:  # Skip if too few samples
                    print(f"  Insufficient samples ({sample_count}) for {name}. Skipping.")
                    continue

                # Use indices from the mask - this avoids pandas alignment issues
                train_indices = np.where(mask)[0]
                X_boundary = self.X_train.iloc[train_indices]
                y_boundary = self.y_train.iloc[train_indices]

                print(f"  Training with {len(X_boundary)} boundary samples.")

                # Create boundary-specific model
                boundary_model = CatBoostRegressor(
                    iterations=1200,
                    depth=6,
                    learning_rate=0.02,
                    l2_leaf_reg=3.5,
                    loss_function='RMSE',
                    eval_metric='RMSE',
                    random_seed=self.random_state,
                    od_type='Iter',
                    od_wait=50,
                    verbose=0
                )

                # Train model
                train_pool = Pool(X_boundary, y_boundary)
                boundary_model.fit(train_pool, verbose=100)

                # Store model
                self.boundary_models[name] = boundary_model

                # Make predictions on full test set (for blending later)
                self.boundary_preds[name] = boundary_model.predict(self.X_test)

                # Calculate metrics for boundary region test samples
                y_test_array = np.array(self.y_test)
                test_mask = (y_test_array >= low_bound) & (y_test_array <= high_bound)
                test_indices = np.where(test_mask)[0]

                if len(test_indices) > 0:
                    X_test_boundary = self.X_test.iloc[test_indices]
                    y_test_boundary = self.y_test.iloc[test_indices]

                    boundary_preds = boundary_model.predict(X_test_boundary)
                    metrics = self._calculate_metrics(y_test_boundary, boundary_preds)

                    print(f"  {name.replace('_', ' ').title()} Model Metrics:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value}")

            return self.boundary_models, self.boundary_preds

        except Exception as e:
            print(f"Error in training boundary models: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def train_age_specific_models(self):
        """Train specialized models for different concrete age groups."""
        try:
            from catboost import CatBoostRegressor, Pool
            print("\nTraining age-specific models...")

            self.age_models = {}
            self.age_preds = {}

            # Define age bins and labels
            age_bins = [0, 3, 7, 28, 90, float('inf')]
            age_labels = ['very_early', 'early', 'standard', 'mature', 'old']

            # Create age groups
            age_col = 'Age (day)'
            X_train_age = np.array(self.X_train[age_col])

            for i,age_group in enumerate(age_labels):
                if i >= len(age_bins) - 1:
                    continue  # Skip if we've reached the end of bins

                print(f"\nTraining model for {age_group.replace('_', ' ').title()} Age concrete...")

                # Get data for this age group using numpy for mask creation
                if i == len(age_bins) - 2:  # Last group
                    mask = (X_train_age >= age_bins[i]) & (X_train_age <= age_bins[i+1])
                else:
                    mask = (X_train_age >= age_bins[i]) & (X_train_age < age_bins[i+1])

                # Get indices from mask
                train_indices = np.where(mask)[0]
                sample_count = len(train_indices)

                if sample_count < 20:  # Skip if too few samples
                    print(f"  Insufficient samples ({sample_count}) for {age_group} age. Skipping.")
                    continue

                # Use indices to select rows
                X_age = self.X_train.iloc[train_indices]
                y_age = self.y_train.iloc[train_indices]

                print(f"  Training with {len(X_age)} age-specific samples.")

                # Create age-specific model with appropriate parameters
                if age_group in ['very_early', 'early']:
                    # More careful tuning for early-age concrete
                    age_model = CatBoostRegressor(
                        iterations=1500,
                        depth=6,
                        learning_rate=0.02,
                        l2_leaf_reg=4,
                        loss_function='RMSE',
                        eval_metric='RMSE',
                        random_seed=self.random_state,
                        od_type='Iter',
                        od_wait=50,
                        verbose=0
                    )
                else:
                    age_model = CatBoostRegressor(
                        iterations=1200,
                        depth=6,
                        learning_rate=0.025,
                        l2_leaf_reg=3,
                        loss_function='RMSE',
                        eval_metric='RMSE',
                        random_seed=self.random_state,
                        od_type='Iter',
                        od_wait=50,
                        verbose=0
                    )

                # Train model
                train_pool = Pool(X_age, y_age)
                age_model.fit(train_pool, verbose=100)

                # Store model
                self.age_models[age_group] = age_model

                # Make predictions on full test set (for blending later)
                self.age_preds[age_group] = age_model.predict(self.X_test)

                # Calculate metrics for age group test samples
                X_test_age = np.array(self.X_test[age_col])
                if i == len(age_bins) - 2:  # Last group
                    test_mask = (X_test_age >= age_bins[i]) & (X_test_age <= age_bins[i+1])
                else:
                    test_mask = (X_test_age >= age_bins[i]) & (X_test_age < age_bins[i+1])

                test_indices = np.where(test_mask)[0]

                if len(test_indices) > 0:
                    X_test_age_subset = self.X_test.iloc[test_indices]
                    y_test_age = self.y_test.iloc[test_indices]

                    age_preds = age_model.predict(X_test_age_subset)
                    metrics = self._calculate_metrics(y_test_age, age_preds)

                    print(f"  {age_group.replace('_', ' ').title()} Age Model Metrics:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value}")

            return self.age_models, self.age_preds

        except ImportError:
            print("CatBoost is not installed. Please install it using: pip install catboost")
            return None, None

    def train_meta_learner(self):
        """Train a non-linear meta-learner with all specialized models."""
        if not hasattr(self, 'deep_catboost'):
            print("Must train deep_catboost first!")
            return None, None

        print("\\nTraining enhanced non-linear meta-learner ensemble...")

        # --- Step 1: Create meta-features from all model predictions ---
        meta_features_list = [self.catboost_preds] # Start with deep model predictions

        # Dynamically add predictions from all available trained models
        model_sets = {
            'range_preds': getattr(self, 'range_preds', {}),
            'boundary_preds': getattr(self, 'boundary_preds', {}),
            'age_preds': getattr(self, 'age_preds', {}),
            'very_low_specialized_preds': getattr(self, 'very_low_specialized_preds', {})
        }

        for pred_dict in model_sets.values():
            for pred_array in pred_dict.values():
                meta_features_list.append(pred_array)

        if hasattr(self, 'medium_bias_preds') and self.medium_bias_preds is not None:
            bias_corrected_preds = self.catboost_preds - self.medium_bias_preds * 0.7
            meta_features_list.append(bias_corrected_preds)

        meta_features_from_models = np.column_stack(meta_features_list)

        # --- Step 2: Create range indicators ---
        range_indicators = pd.get_dummies(self.y_ranges_test).values

        # --- Step 3: Combine everything into the final feature set ---
        # This is the crucial fix: combine all parts before creating the DataFrame
        final_meta_features_array = np.column_stack([
            meta_features_from_models,
            range_indicators,
            self.X_test.values  # Add original scaled features
        ])

        # --- Step 4: Create the DataFrame and feature names for the meta-learner ---
        # The names must now reflect this final, complete feature set
        model_pred_names = [f"meta_pred_{i}" for i in range(meta_features_from_models.shape[1])]
        indicator_names = [f"range_{label}" for label in self.strength_labels]

        # Combine all names
        self.meta_feature_names = model_pred_names + indicator_names + self.all_features

        meta_features_df = pd.DataFrame(final_meta_features_array, columns=self.meta_feature_names)
        print(f"Meta-features created with shape: {meta_features_df.shape}")

        # --- Step 5: Train the meta-learner on the complete feature set ---
        from catboost import CatBoostRegressor
        from sklearn.model_selection import train_test_split

        meta_catboost = CatBoostRegressor(
            iterations=1000, learning_rate=0.015, depth=5,
            loss_function='RMSE', random_seed=self.random_state, verbose=0,
            l2_leaf_reg=4, bootstrap_type='Bayesian'
        )

        # Split the complete meta-features DataFrame
        meta_X_train, meta_X_val, meta_y_train, meta_y_val = train_test_split(
            meta_features_df, self.y_test,
            test_size=0.3,
            random_state=self.random_state
        )

        meta_catboost.fit(meta_X_train, meta_y_train, eval_set=(meta_X_val, meta_y_val))

        # Make final predictions
        meta_preds = meta_catboost.predict(meta_features_df)

        self.meta_learner = meta_catboost
        self.meta_learner_type = 'catboost'
        self.meta_preds = meta_preds
        self.meta_metrics = self._calculate_metrics(self.y_test, meta_preds)

        print("\\nMeta-Learner Metrics:")
        for metric, value in self.meta_metrics.items():
            print(f"  {metric}: {value}")

        self._create_meta_feature_generator()

        return self.meta_metrics, self.meta_preds

    def _create_meta_feature_generator(self):
        """Create a function to generate meta-features for new data."""
        def generate_meta_features(self, X):
          """Generate meta-features for new data samples."""
          meta_features = []

          # Deep CatBoost predictions
          deep_preds = self.deep_catboost.predict(X)
          meta_features.append(deep_preds)

          # Range-specific models
          for range_name in self.strength_labels:
              if hasattr(self, 'range_models') and range_name in self.range_models:
                  meta_features.append(self.range_models[range_name].predict(X))

          # Boundary models
          if hasattr(self, 'boundary_models') and self.boundary_models:
              for name, model in self.boundary_models.items():
                  meta_features.append(model.predict(X))

          # Age-specific models
          if hasattr(self, 'age_models') and self.age_models:
              for age_group, model in self.age_models.items():
                  meta_features.append(model.predict(X))

          # Very low models
          if hasattr(self, 'very_low_specialized_models') and self.very_low_specialized_models:
              for name, model in self.very_low_specialized_models.items():
                  meta_features.append(model.predict(X))

          # Bias-corrected predictions
          if hasattr(self, 'medium_bias_model'):
              bias_corrected_preds = deep_preds.copy()
              medium_mask = (deep_preds >= 40) & (deep_preds < 60)
              if np.any(medium_mask):
                  medium_indices = np.where(medium_mask)[0]
                  X_medium = X.iloc[medium_indices]
                  bias_predictions = self.medium_bias_model.predict(X_medium)
                  for idx, i in enumerate(medium_indices):
                      bias_corrected_preds[i] -= bias_predictions[idx] * 0.7
              meta_features.append(bias_corrected_preds)

          # Estimate range and create one-hot
          estimated_ranges = pd.cut(deep_preds, bins=self.strength_bins, labels=self.strength_labels)
          range_indicators = pd.get_dummies(estimated_ranges).reindex(columns=self.strength_labels, fill_value=0).values

          # Stack everything
          meta_features_array = np.column_stack(meta_features)
          meta_features_array = np.column_stack([meta_features_array, range_indicators, X.values])

          # Convert to DataFrame with proper column names
          meta_feature_names = [f"meta_{i}" for i in range(meta_features_array.shape[1])]
          meta_features_df = pd.DataFrame(meta_features_array, columns = self.meta_feature_names)


          print("✅ Final meta feature shape:", meta_features_array.shape)
          if hasattr(self.meta_learner, 'n_features_in_'):
              print(f"📦 Meta-learner expects: {self.meta_learner.n_features_in_} features")
          elif hasattr(self.meta_learner, 'feature_count_'):
              print(f"📦 Meta-learner expects: {self.meta_learner.feature_count_} features")

          return meta_features_df

        self.generate_meta_features = generate_meta_features.__get__(self, self.__class__)

    # Include all other training methods here but check for CATBOOST_AVAILABLE
    # ... [rest of your methods remain the same, just add the CATBOOST_AVAILABLE check where needed]

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics."""
        # Calculate basic regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Calculate percentage errors
        percent_errors = np.abs((y_true - y_pred) / y_true * 100)

        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'max_percent_error': np.max(percent_errors),
            'mean_percent_error': np.mean(percent_errors),
            'median_percent_error': np.median(percent_errors),
            'percent_within_5': np.mean(percent_errors <= 5) * 100,
            'percent_within_10': np.mean(percent_errors <= 10) * 100
        }

    def save_model(self, filepath='models/enhanced_catboost_model.joblib'):
        """Save the trained models and preprocessing objects."""
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        # Create dictionary with all model components
        model_data = {
            'deep_catboost': getattr(self, 'deep_catboost', None),
            'range_models': getattr(self, 'range_models', {}),
            'boundary_models': getattr(self, 'boundary_models', {}),
            'age_models': getattr(self, 'age_models', {}),
            'very_low_specialized_models': getattr(self, 'very_low_specialized_models', {}),
            'medium_bias_model': getattr(self, 'medium_bias_model', None),
            'meta_learner': getattr(self, 'meta_learner', None),
            'meta_learner_type': getattr(self, 'meta_learner_type', None),
            'meta_features_scaler': getattr(self, 'meta_features_scaler', None),
            'meta_weights': getattr(self, 'meta_weights', None),
            'meta_catboost': getattr(self, 'meta_catboost', None),
            'meta_mlp': getattr(self, 'meta_mlp', None),
            'meta_feature_names': getattr(self, 'meta_feature_names', None),
            'scaler': self.scaler,
            'original_features': self.original_features,
            'engineered_features': self.engineered_features,
            'all_features': self.all_features,
            'strength_bins': self.strength_bins,
            'strength_labels': self.strength_labels,
            'random_state': self.random_state,
            'catboost_preds': getattr(self, 'catboost_preds', None),
            'meta_preds': getattr(self, 'meta_preds', None),
            'meta_metrics': getattr(self, 'meta_metrics', None),
            'feature_stats': getattr(self, 'feature_stats', {})  # IMPORTANT: Save feature statistics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Enhanced CatBoost models saved to {filepath}")
        print(f"✅ Feature statistics saved: {model_data.get('feature_stats', {})}")

    @classmethod
    def load_model(cls, filepath='models/enhanced_catboost_model.joblib'):
        """Load a trained model and preprocessing objects."""
        model_data = joblib.load(filepath)

        predictor = cls()

        if 'meta_feature_names' in model_data:
          predictor.meta_feature_names = model_data['meta_feature_names']

        for key, value in model_data.items():
            setattr(predictor, key, value)

        # Recreate meta-feature generator
        if hasattr(predictor, 'meta_learner'):
            predictor._create_meta_feature_generator()

        return predictor

    def detect_and_correct_outliers(self, X, predictions):
        """Detect and correct likely outlier predictions."""
        corrected_predictions = predictions.copy()

        # Get features that might indicate outlier behavior
        if 'water_cement_ratio' in X.columns and 'abnormal_mix_factor' in X.columns:
            wcr = X['water_cement_ratio']
            abnormal_factor = X['abnormal_mix_factor']

            # Identify potential outliers based on extreme ratios and factors
            wcr_array = np.array(wcr)
            abnormal_factor_array = np.array(abnormal_factor)
            wcr_high = wcr_array > np.quantile(wcr_array, 0.95)
            wcr_low = wcr_array < np.quantile(wcr_array, 0.05)
            abnormal_high = abnormal_factor_array > 2.0

            potential_outliers = wcr_high | wcr_low | abnormal_high

            # For these potential outliers, use a more conservative prediction
            outlier_indices = np.where(potential_outliers)[0]
            if len(outlier_indices) > 0:
                print(f"Detected {len(outlier_indices)} potential outlier predictions")

                for i in outlier_indices:
                    # Estimate strength range based on predicted value
                    pred_value = predictions[i]
                    if pred_value < 20:
                        strength_range = 'very_low'
                    elif pred_value < 40:
                        strength_range = 'low'
                    elif pred_value < 60:
                        strength_range = 'medium'
                    else:
                        strength_range = 'high'

                    # Use range-specific model if available
                    if hasattr(self, 'range_models') and strength_range in self.range_models:
                        # Use iloc with a list to access a single row as DataFrame
                        range_pred = self.range_models[strength_range].predict(X.iloc[[i]])[0]
                        # Use a weighted average with more weight on range model
                        corrected_predictions[i] = 0.3 * predictions[i] + 0.7 * range_pred
                        print(f"  Outlier at index {i}: Original {predictions[i]:.2f}, Corrected {corrected_predictions[i]:.2f}")

        return corrected_predictions

    def predict(self, X_new):
        if not hasattr(self, 'meta_learner'):
            raise ValueError("Meta-learner has not been trained. Call train_meta_learner first.")
        
        # Preprocess data
        if isinstance(X_new, pd.DataFrame):
            X_engineered = self.engineer_features(X_new, for_training=False)  # Use stored statistics
        else:
            # Convert to DataFrame if numpy array
            X_new_df = pd.DataFrame(X_new, columns=self.original_features)
            X_engineered = self.engineer_features(X_new_df, for_training=False)  # Use stored statistics
        
        # Ensure all features are present
        X_engineered = X_engineered.reindex(columns=self.all_features, fill_value=0)
        
        # Scale features
        X_scaled = self.scaler.transform(X_engineered)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.all_features)
        
        # Generate meta-features
        meta_features = self.generate_meta_features(X_scaled_df)
        
        # Make predictions using meta-learner
        predictions = self.meta_learner.predict(meta_features)
        
        # Apply outlier detection and correction
        predictions = self.detect_and_correct_outliers(X_scaled_df, predictions)
        
        # Apply range-specific corrections
        final_predictions = []
        
        for i, pred in enumerate(predictions):
            # Determine likely strength range
            if pred < 20:
                strength_range = 'very_low'
            elif pred < 40:
                strength_range = 'low'
            elif pred < 60:
                strength_range = 'medium'
            else:
                strength_range = 'high'
            
            # Apply specialized corrections
            if strength_range == 'very_low':
                # Check for specialized very low models
                if hasattr(self, 'very_low_specialized_models'):
                    if pred < 15 and 'ultra_low' in self.very_low_specialized_models:
                        # Get specialized prediction
                        specialized_pred = self.very_low_specialized_models['ultra_low'].predict(X_scaled_df.iloc[[i]])[0]
                        # Use a weighted blend
                        pred = 0.4 * pred + 0.6 * specialized_pred
                    elif pred >= 15 and pred < 20 and 'mid_low' in self.very_low_specialized_models:
                        specialized_pred = self.very_low_specialized_models['mid_low'].predict(X_scaled_df.iloc[[i]])[0]
                        pred = 0.4 * pred + 0.6 * specialized_pred
            
            elif strength_range == 'medium':
                # Apply bias correction for medium range
                if hasattr(self, 'medium_bias_model'):
                    estimated_bias = self.medium_bias_model.predict(X_scaled_df.iloc[[i]])[0]
                    # If bias is significant
                    if estimated_bias > 5:
                        # Reduce the prediction by the estimated bias
                        pred -= estimated_bias * 0.7  # Using 70% of the bias as a safe measure
            
            elif strength_range == 'high':
                # Boost high strength predictions to address under-prediction
                pred *= 1.05  # Apply a 5% boost
            
            final_predictions.append(pred)
        
        return np.array(final_predictions)

# Only run training if this script is executed directly (not imported)
if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Use it in your app.py or training notebook.")