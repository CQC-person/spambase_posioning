"""Model definitions and factories for classical spam classification baselines."""

from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


def get_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Instantiate the three classical spam classifiers with required configurations.
    
    Parameters
    ----------
    random_state : int
        Seed for reproducibility across model training.
        
    Returns
    -------
    dict
        Mapping of model names to scikit-learn estimators.
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                random_state=random_state
            ))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=random_state
        )
    }
    return models
