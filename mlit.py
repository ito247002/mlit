import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
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
print("Performing Feature Selection...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X = train_processed[features]
y = train_processed[target_col]
y_log = np.log1p(y) # Use log target for training stability

# Initial LightGBM for Feature Importance
# Train on log target with regression objective (stable)
params_fs = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'seed': 42,
    'device': 'gpu',
    'max_bin': 200
}

feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = features
feature_importance_df["importance"] = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
    X_train, y_train = X.iloc[train_idx], y_log.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y_log.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        params_fs,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    feature_importance_df["importance"] += model.feature_importance(importance_type='gain') / 5

# Drop low importance features (e.g., 0 importance)
zero_importance_features = feature_importance_df[feature_importance_df["importance"] == 0]["feature"].tolist()
print(f"Dropping {len(zero_importance_features)} features with 0 importance: {zero_importance_features}")
features = [f for f in features if f not in zero_importance_features]
print(f"Remaining features: {len(features)}")

# --- Optuna Tuning ---
# print("Starting Optuna Tuning...")

# def objective(trial):
#     param = {
#         'objective': 'regression', # Train on log(y) -> regression = optimize RMSLE approx MAPE
#         'metric': 'rmse',
#         'boosting_type': 'gbdt',
#         'device': 'gpu',
#         'max_bin': 200,
#         'verbose': -1,
#         'seed': 42,
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 300),
#         'max_depth': trial.suggest_int('max_depth', 3, 12),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#         'subsample': trial.suggest_float('subsample', 0.5, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
#     }

#     mape_scores = []
#     kf_opt = KFold(n_splits=3, shuffle=True, random_state=42) 
    
#     X_selected = train_processed[features]
    
#     # Train on y_log, evaluate MAPE on expm1(pred)
#     for train_idx, val_idx in kf_opt.split(X_selected, y_log):
#         X_train, y_train = X_selected.iloc[train_idx], y_log.iloc[train_idx]
#         X_val, y_val = X_selected.iloc[val_idx], y_log.iloc[val_idx]
        
#         # For validation in training, we use y_log and rmse
#         lgb_train = lgb.Dataset(X_train, y_train)
#         lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
#         model = lgb.train(
#             param,
#             lgb_train,
#             num_boost_round=1000,
#             valid_sets=[lgb_val],
#             callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
#         )
        
#         # Predict and inverse transform
#         preds_log = model.predict(X_val, num_iteration=model.best_iteration)
#         preds = np.expm1(preds_log)
#         y_true = np.expm1(y_val)
        
#         mape = mean_absolute_percentage_error(y_true, preds)
#         mape_scores.append(mape)
    
#     return np.mean(mape_scores)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)

# print("Best trial:")
# trial = study.best_trial
# print(f"  Value: {trial.value}")
# print("  Params: ")
# for key, value in trial.params.items():
#     print(f"    {key}: {value}")

# --- Final Training with Best Params ---
print("Training Final Model with Best Params...")
# best_params = trial.params
best_params = {
    'learning_rate': 0.1449061715139644,
    'num_leaves': 254,
    'max_depth': 11,
    'min_child_samples': 79,
    'subsample': 0.6497630513847934,
    'colsample_bytree': 0.7056605499820982,
    'reg_alpha': 0.8201944982802111,
    'reg_lambda': 0.2917040347335464
}
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['boosting_type'] = 'gbdt'
best_params['device'] = 'gpu'
best_params['max_bin'] = 200
best_params['verbose'] = -1
best_params['seed'] = 42

oof_preds_log = np.zeros(len(train_processed))
test_preds_log = np.zeros(len(test_processed))
models = []

X_selected = train_processed[features]
X_test_selected = test_processed[features]

for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected, y_log)):
    print(f"Fold {fold+1}")
    X_train, y_train = X_selected.iloc[train_idx], y_log.iloc[train_idx]
    X_val, y_val = X_selected.iloc[val_idx], y_log.iloc[val_idx]
    
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
    )
    
    models.append(model)
    oof_preds_log[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds_log += model.predict(X_test_selected, num_iteration=model.best_iteration) / kf.get_n_splits()

# Inverse transform final predictions
oof_preds = np.expm1(oof_preds_log)
test_preds = np.expm1(test_preds_log)

mape = mean_absolute_percentage_error(y, oof_preds)
print(f"Final CV MAPE: {mape}")

# Feature Importance Plot
feature_importance = pd.DataFrame()
for model in models:
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = features
    fold_importance["importance"] = model.feature_importance(importance_type='gain')
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

plt.figure(figsize=(12, 10))
sns.barplot(x="importance", y="feature", data=feature_importance.sort_values(by="importance", ascending=False).head(50))
plt.title('Feature Importance (Gain)')
plt.tight_layout()
plt.savefig('image/feature_importance.png')
print("Saved image/feature_importance.png")

# Create Submission
submission_df = pd.DataFrame({
    'id': test_processed['id'],
    'money_room': test_preds
})

# Format ID: convert to int, then zero-pad to 6 digits
submission_df['id'] = submission_df['id'].astype(int).astype(str).str.zfill(6)

submission_df.to_csv('submission/submission.csv', index=False, header=False)
print("Submission saved to submission/submission.csv")
