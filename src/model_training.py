from src.logger import get_logger
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from redis import Redis
import json
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from src.data_processing import RedisFeatureStore

logger = get_logger(__name__)



# Model training class with MLOps approach
class ModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path='artifacts/models/'):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None

        # Ensure model directory exists
        os.makedirs(self.model_save_path, exist_ok=True)

    def load_data_from_redis(self, entity_ids):
        """Retrieve features for training and testing from Redis."""
        logger.info("Loading data from Redis...")
        data = []
        for entity_id in entity_ids:
            features = self.feature_store.get_features(entity_id)
            if features:
                data.append(features)
            else:
                logger.warning(f"Features for entity {entity_id} not found in Redis")
        return data

    def prepare_data(self):
        """Prepare training and testing data by splitting the entity IDs."""
        # Get all available entity IDs (PassengerId)
        entity_ids = self.feature_store.get_all_entity_ids()

        # Split data into train and test sets
        train_entity_ids, test_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)

        # Load training and testing data from Redis
        logger.info("Preparing data for training and testing...")
        train_data = self.load_data_from_redis(train_entity_ids)
        test_data = self.load_data_from_redis(test_entity_ids)

        # Convert to DataFrame for easier manipulation
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Split features and labels
        X_train = train_df.drop('Survived', axis=1)
        logger.info(X_train.columns)
        y_train = train_df['Survived']
        X_test = test_df.drop('Survived', axis=1)
        y_test = test_df['Survived']

        logger.info("Data preparation completed")
        return X_train, X_test, y_train, y_test

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using RandomizedSearchCV."""
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        logger.info("Performing RandomizedSearchCV for hyperparameter tuning...")
        rf = RandomForestClassifier(random_state=42)
        random_search = RandomizedSearchCV(rf, param_distributions, n_iter=10, cv=3, scoring='accuracy', random_state=42)
        random_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {random_search.best_params_}")
        return random_search.best_estimator_

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train and evaluate the Random Forest model."""
        # Hyperparameter tuning
        best_rf = self.hyperparameter_tuning(X_train, y_train)

        # Train the model
        logger.info("Training the Random Forest model...")
        best_rf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Random Forest Accuracy: {accuracy:.2f}")
        
        # Save the model
        self.save_model(best_rf)
        return accuracy

    def save_model(self, model):
        """Save the trained model to disk."""
        model_filename = f"{self.model_save_path}random_forest_model.pkl"
        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)
        logger.info(f"Model saved to {model_filename}")

    def run(self):
        """Run the entire model training pipeline."""
        logger.info("Starting model training pipeline...")

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Train and evaluate model
        accuracy = self.train_and_evaluate(X_train, y_train, X_test, y_test)

        logger.info(f"Training completed with accuracy: {accuracy:.2f}")
        return accuracy


# Example usage

# Initialize RedisFeatureStore and ModelTraining
feature_store = RedisFeatureStore()
model_trainer = ModelTraining(feature_store)

# Run the model training pipeline
model_trainer.run()
