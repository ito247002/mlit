import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import lightgbm as lgb
import xgboost as xgb
import torch
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
import warnings
import matplotlib

warnings.filterwarnings('ignore')
# 日本語フォントの設定
matplotlib.rcParams['font.family'] = 'MS Gothic'

print("Loading data...")
train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')
sample_submit = pd.read_csv('Data/sample_submit.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Feature Engineering
print("Feature Engineering...")
# Drop traffic_car due to high missing rate (>99.9%)
train_df = train_df.drop(columns=['traffic_car'])
test_df = test_df.drop(columns=['traffic_car'])

# Combine for processing
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['money_room'] = np.nan

all_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Date Processing
def process_dates(df):
    # target_ym is int like 201901
    df['target_year'] = df['target_ym'] // 100
    df['target_month'] = df['target_ym'] % 100
    
    # building_create_date
    df['building_create_date'] = pd.to_datetime(df['building_create_date'], errors='coerce')
    df['building_year'] = df['building_create_date'].dt.year
    df['building_month'] = df['building_create_date'].dt.month
    
    # Calculate Age (approximate)
    df['year_built_val'] = pd.to_numeric(df['year_built'], errors='coerce')
    df['built_year'] = df['year_built_val'] // 100
    df['built_month'] = df['year_built_val'] % 100
    
    df['building_age'] = (df['target_year'] * 12 + df['target_month']) - (df['built_year'] * 12 + df['built_month'])
    
    return df

all_df = process_dates(all_df)

# Categorical Encoding
# Identify all object columns that are not in drop_cols
drop_cols_set = set(['is_train', 'id', 'money_room', 'target_ym', 'building_id', 'building_name', 
             'building_name_ruby', 'homes_building_name', 'homes_building_name_ruby', 
             'full_address', 'building_create_date', 'building_modify_date', 'year_built',
             'reform_date', 'renovation_date', 'snapshot_create_date', 'new_date',
             'snapshot_modify_date', 'timelimit_date', 'free_rent_gen_timing',
             'reform_exterior_date', 'reform_common_area_date', 'reform_wet_area_date',
             'reform_interior_date', 'usable_date'])

object_cols = all_df.select_dtypes(include=['object']).columns.tolist()
object_cols = [c for c in object_cols if c not in drop_cols_set]

print(f"Object columns to encode: {object_cols}")

for col in object_cols:
    le = LabelEncoder()
    all_df[col] = all_df[col].astype(str).fillna('MISSING')
    all_df[col] = le.fit_transform(all_df[col])

# Numerical Features
num_cols = [
    'unit_count', 'total_floor_area', 'building_area', 'floor_count', 'basement_floor_count',
    'building_land_area', 'land_area_all', 'unit_area_min', 'unit_area_max',
    'land_setback', 'land_kenpei', 'land_youseki', 'room_floor', 'balcony_area',
    'room_count', 'unit_area', 'floor_plan_code', 'money_kyoueki', 'money_shuuzen',
    'money_shuuzenkikin', 'parking_money', 'parking_distance', 'parking_number',
    'school_ele_distance', 'school_jun_distance', 'convenience_distance', 'super_distance',
    'hospital_distance', 'park_distance', 'drugstore_distance', 'bank_distance',
    'shopping_street_distance', 'est_other_distance', 'free_rent_duration',
    'snapshot_land_area', 'snapshot_land_shidou', 'house_area', 'room_kaisuu',
    'madori_number_all', 'money_rimawari_now'
]

for col in num_cols:
    if col in all_df.columns:
        all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

# Split back to train/test
train_processed = all_df[all_df['is_train'] == 1].copy()
test_processed = all_df[all_df['is_train'] == 0].copy()

# Target
target_col = 'money_room'
drop_cols = ['is_train', 'id', 'money_room', 'target_ym', 'building_id', 'building_name', 
             'building_name_ruby', 'homes_building_name', 'homes_building_name_ruby', 
             'full_address', 'building_create_date', 'building_modify_date', 'year_built',
             'reform_date', 'renovation_date', 'snapshot_create_date', 'new_date',
             'snapshot_modify_date', 'timelimit_date', 'free_rent_gen_timing',
             'reform_exterior_date', 'reform_common_area_date', 'reform_wet_area_date',
             'reform_interior_date', 'usable_date']

features = [c for c in train_processed.columns if c not in drop_cols]
print(f"Number of features: {len(features)}")

# GPU Check
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Feature Selection ---
# SKIPPING FEATURE SELECTION FOR XGBOOST TEST
print("Skipping Feature Selection...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = train_processed[features]
y = train_processed[target_col]
y_log = np.log1p(y) # Use log target for training stability

X_selected = train_processed[features]
X_test_selected = test_processed[features]

# --- XGBoost Training ---
print("Training XGBoost...")
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42,
    'tree_method': 'hist',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

xgb_oof_preds_log = np.zeros(len(train_processed))
xgb_test_preds_log = np.zeros(len(test_processed))
xgb_models = []

# Convert to DMatrix for XGBoost
# Note: XGBoost handles NaNs, but ensure data types are correct
for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y_log)):
    print(f"XGB Fold {fold+1}")
    X_train, y_train = X_selected.iloc[train_idx], y_log.iloc[train_idx]
    X_val, y_val = X_selected.iloc[val_idx], y_log.iloc[val_idx]
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=10000,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    xgb_models.append(model)
    xgb_oof_preds_log[val_idx] = model.predict(dval)
    
    # Predict on test
    dtest = xgb.DMatrix(X_test_selected)
    xgb_test_preds_log += model.predict(dtest) / kf.get_n_splits()

xgb_oof_preds = np.expm1(xgb_oof_preds_log)
xgb_test_preds = np.expm1(xgb_test_preds_log)

print(f"XGBoost CV MAPE: {mean_absolute_percentage_error(y, xgb_oof_preds)}")
