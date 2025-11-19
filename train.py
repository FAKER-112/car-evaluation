import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb





def train():
    print("Loading data...")
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv('dataset/car.csv', names=columns)

    print("Preprocessing...")
    X = df.drop('class', axis=1)
    y = df['class']

    ordinal_mappings = {
        'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
        'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
        'doors': {'2': 0, '3': 1, '4': 2, '5more': 3},
        'persons': {'2': 0, '4': 1, 'more': 2},
        'safety': {'low': 0, 'med': 1, 'high': 2}
    }

    for col, mapping in ordinal_mappings.items():
        X[col] = X[col].map(mapping)

    # One-Hot Encoding for 'lug_boot'
    X = pd.get_dummies(X, columns=['lug_boot'], drop_first=False)
    # Target Encoding
    class_mapping = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
    y = y.map(class_mapping)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training models...")
    xgbooster = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
            
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    grid_search = GridSearchCV(xgbooster, xgb_params, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Params: {grid_search.best_params_}")

    print("Evaluating...")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Done.")

if __name__ == "__main__":
    train()
