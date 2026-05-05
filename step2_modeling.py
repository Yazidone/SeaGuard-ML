import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

def perform_modeling(input_file='processed_maritime_data.csv'):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: {input_file} not found. Please run step1_feature_engineering.py first.")
        return

    # الترتيب زمنياً لضمان عدم الخلط في التقسيم
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # تحديد المتغيرات المستقلة والتابعة
    # نستثني 'datetime' لأنه ليس متغيراً عددياً يمكن استخدامه مباشرة
    X = df.drop(columns=['datetime', 'incidents'])
    y = df['incidents']

    # 1. التقسيم الزمني (Time Series Split)
    # Train 70%, Validation 15%, Test 15%
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    print(f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")

    # 2. التوحيد القياسي (Standardization)
    # استثناء المتغيرات الدائرية (sin/cos)
    circular_cols = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
    num_cols = [col for col in X_train.columns if col not in circular_cols]
    
    scaler = StandardScaler()
    
    # يجب عمل نسخة لتجنب تحذيرات SettingWithCopyWarning
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    # 3. تدريب 3 نماذج (Training Models)
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost Regressor': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = {}
    best_model_name = 'XGBoost Regressor' # As requested
    
    for name, model in models.items():
        # تدريب النموذج
        model.fit(X_train_scaled, y_train)
        
        # التوقع على بيانات الاختبار
        y_pred = model.predict(X_test_scaled)
        
        # 4. التقييم
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        
        print(f"\n--- {name} ---")
        print(f"R²:   {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")

    # 5. حفظ النموذج (Saving best model and scaler)
    xgboost_model = models[best_model_name]
    
    joblib.dump(xgboost_model, 'xgboost_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # حفظ قائمة الأعمدة العددية لكي نتمكن من استخدامها في التطبيق لاحقاً
    joblib.dump(num_cols, 'num_cols.pkl')
    # حفظ ترتيب الأعمدة بالكامل
    joblib.dump(list(X_train.columns), 'feature_columns.pkl')

    print("\nتم حفظ نموذج XGBoost والـ StandardScaler بنجاح (xgboost_model.pkl, scaler.pkl).")

if __name__ == "__main__":
    perform_modeling()
