from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (balanced_accuracy_score, precision_recall_fscore_support,
                          confusion_matrix, roc_curve, auc, matthews_corrcoef)
from k_mer import KMer
import pandas as pd
import numpy as np


def select_best_model(hyperparameters, train_data, train_labels):
    """
    Select best model using training data with proper validation split.

    Args:
        hyperparameters (list): List of k values to test
        train_data (list): Training sequences
        train_labels (dict): Dictionary mapping sequences to labels
    """
    # Convert labels to list
    y_train_full = [train_labels[x] for x in train_data]

    # Create validation split from training data
    X_train, X_val, y_train, y_val = train_test_split(
        train_data,
        y_train_full,
        test_size=0.2,  # Use 20% of training data for validation
        random_state=42,
        stratify=y_train_full
    )

    print("Training set class distribution:", pd.Series(y_train).value_counts())
    print("Validation set class distribution:", pd.Series(y_val).value_counts())

    # Initialize results storage
    all_results = {}
    best_config = {
        'score': 0,
        'k': None,
        'model_class': None,
        'model_name': None,
        'model_params': None
    }

    # Define models with parameter grids
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, class_weight='balanced'),
            'params': [
                {'max_iter': 1000, 'C': 0.1},
                {'max_iter': 1000, 'C': 1.0},
                {'max_iter': 1000, 'C': 10.0}
            ]
        },
        'SVM': {
            'model': SVC(random_state=42, class_weight='balanced'),
            'params': [
                {'kernel': 'rbf', 'C': 0.1, 'probability': True},
                {'kernel': 'rbf', 'C': 1.0, 'probability': True},
                {'kernel': 'linear', 'C': 1.0, 'probability': True}
            ]
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
            'params': [
                {'n_estimators': 100, 'max_depth': None},
                {'n_estimators': 200, 'max_depth': 10},
                {'n_estimators': 200, 'max_depth': 20}
            ]
        }
    }

    for hp in hyperparameters:
        print(f"\n=== Testing k={hp} ===")

        # Create feature matrices
        kmer = KMer(k=hp)

        # Transform training data
        X_train_df = pd.DataFrame(X_train, columns=["genes"])
        X_train_features = kmer.add_kmer_features_to_dataframe(X_train_df)
        X_train_features = X_train_features.drop(columns=["genes"])

        # Transform validation data
        X_val_df = pd.DataFrame(X_val, columns=["genes"])
        X_val_features = kmer.add_kmer_features_to_dataframe(X_val_df)
        X_val_features = X_val_features.drop(columns=["genes"])

        hp_results = {}

        for model_name, model_info in models.items():
            print(f"\nModel: {model_name}")

            best_model_score = 0
            best_model_params = None
            best_model_instance = None

            # Test each parameter combination
            for params in model_info['params']:
                model = model_info['model'].__class__(**params)

                # Train on training set
                model.fit(X_train_features, y_train)

                # Evaluate on validation set
                val_score = balanced_accuracy_score(y_val, model.predict(X_val_features))

                if val_score > best_model_score:
                    best_model_score = val_score
                    best_model_params = params
                    best_model_instance = model

            hp_results[model_name] = {
                'validation_accuracy': best_model_score,
                'best_params': best_model_params
            }

            print(f"Best Validation Balanced Accuracy: {best_model_score:.3f}")
            print(f"Best Parameters: {best_model_params}")

            # Update best configuration if this is better
            if best_model_score > best_config['score']:
                best_config['score'] = best_model_score
                best_config['k'] = hp
                best_config['model_class'] = best_model_instance.__class__
                best_config['model_name'] = model_name
                best_config['model_params'] = best_model_params

        all_results[hp] = hp_results

    print(f"\n=== Best Configuration ===")
    print(f"Model: {best_config['model_name']}")
    print(f"k-mer size: {best_config['k']}")
    print(f"Parameters: {best_config['model_params']}")
    print(f"Validation Balanced Accuracy: {best_config['score']:.3f}")

    return best_config, all_results


def evaluate_final_model(best_config, train_data, train_labels, test_data, test_labels):
    """
    Train the best model configuration on full training data and evaluate on test set.

    Args:
        best_config (dict): Best model configuration from selection
        train_data (list): Training sequences
        train_labels (dict): Training labels
        test_data (list): Test sequences
        test_labels (dict): Test labels

    Returns:
        dict: Dictionary containing model and comprehensive evaluation metrics
    """
    # Initialize the best model with optimal parameters
    final_model = best_config['model_class'](**best_config['model_params'])

    # Convert labels to lists
    y_train = [train_labels[x] for x in train_data]
    y_test = [test_labels[x] for x in test_data]

    # Create feature matrices using best k
    kmer = KMer(k=best_config['k'])

    # Training features
    X_train_df = pd.DataFrame(train_data, columns=["genes"])
    X_train_features = kmer.add_kmer_features_to_dataframe(X_train_df)
    X_train = X_train_features.drop(columns=["genes"])

    # Test features
    X_test_df = pd.DataFrame(test_data, columns=["genes"])
    X_test_features = kmer.add_kmer_features_to_dataframe(X_test_df)
    X_test = X_test_features.drop(columns=["genes"])

    # Train final model
    final_model.fit(X_train, y_train)

    # Get predictions
    y_pred = final_model.predict(X_test)

    # Try to get probability scores if available
    try:
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        has_proba = True
    except (AttributeError, NotImplementedError):
        y_pred_proba = y_pred  # fallback to binary predictions
        has_proba = False

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate basic metrics
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    # Calculate additional metrics
    specificity = tn / (tn + fp)  # True Negative Rate
    npv = tn / (tn + fn)  # Negative Predictive Value
    mcc = matthews_corrcoef(y_test, y_pred)  # Matthews Correlation Coefficient

    # Calculate ROC curve and AUC if probabilities are available
    if has_proba:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr = None, None
        roc_auc = None

    # Calculate class distributions
    train_class_dist = np.bincount(y_train) / len(y_train)
    test_class_dist = np.bincount(y_test) / len(y_test)

    print("\n=== Final Model Performance on Test Set ===")
    print("\nClass Distribution:")
    print(f"Training set - Negative: {train_class_dist[0]:.3f}, Positive: {train_class_dist[1]:.3f}")
    print(f"Test set     - Negative: {test_class_dist[0]:.3f}, Positive: {test_class_dist[1]:.3f}")

    print("\nConfusion Matrix:")
    print(f"True Negatives: {tn}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Positives: {tp}")

    print("\nMain Metrics:")
    print(f"Balanced Accuracy: {test_balanced_acc:.3f}")
    print(f"Precision (PPV): {test_precision:.3f}")
    print(f"Recall (Sensitivity): {test_recall:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"Negative Predictive Value: {npv:.3f}")
    print(f"F1 Score: {test_f1:.3f}")
    print(f"Matthews Correlation Coefficient: {mcc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")

    return {
        'model': final_model,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'class_distribution': {
            'train': train_class_dist.tolist(),
            'test': test_class_dist.tolist()
        },
        'metrics': {
            'balanced_accuracy': float(test_balanced_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'specificity': float(specificity),
            'npv': float(npv),
            'f1': float(test_f1),
            'mcc': float(mcc),
            'roc_auc': float(roc_auc) if roc_auc is not None else None
        },
        'roc_curve': {
            'fpr': fpr.tolist() if fpr is not None else None,
            'tpr': tpr.tolist() if tpr is not None else None
        }
    }