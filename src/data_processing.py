import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import redis
import json

class RedisFeatureStore:
    def __init__(self, host='localhost', port=6379, db=0):
        """Initialize connection to Redis."""
        self.client = redis.StrictRedis(host=host, port=port, db=db, decode_responses=True)

    def store_features(self, entity_id, features):
        """Store features in Redis."""
        key = f"entity:{entity_id}:features"
        self.client.set(key, json.dumps(features))

    def get_features(self, entity_id):
        """Retrieve features from Redis."""
        key = f"entity:{entity_id}:features"
        features = self.client.get(key)
        if features:
            return json.loads(features)
        return None

    def store_batch_features(self, batch_data):
        """Store a batch of features in Redis."""
        for entity_id, features in batch_data.items():
            self.store_features(entity_id, features)

    def get_batch_features(self, entity_ids):
        """Retrieve a batch of features from Redis."""
        batch_features = {}
        for entity_id in entity_ids:
            batch_features[entity_id] = self.get_features(entity_id)
        return batch_features
    
    def get_all_entity_ids(self):
        """Retrieve all stored entity IDs (Passenger IDs)."""
        # Assuming Redis keys follow the pattern 'entity:<id>:features'
        keys = self.client.keys('entity:*:features')
        entity_ids = [key.split(':')[1] for key in keys]  # Extracting the ID part from the key pattern
        return entity_ids


class DataProcessing:
    def __init__(self, train_data_path, test_data_path, feature_store: RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_resampled = None
        self.y_resampled = None
        self.feature_store = feature_store

    def load_data(self):
        """Load the Titanic dataset."""
        self.data = pd.read_csv(self.train_data_path)
        self.test_data = pd.read_csv(self.test_data_path)
        print("Data loaded successfully")

    def preprocess_data(self):
        """Preprocess the dataset: Handle missing values and encode categorical data."""
        self.data['Age'] = self.data['Age'].fillna(self.data['Age'].median())
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
        self.data['Fare'] = self.data['Fare'].fillna(self.data['Fare'].median())
        self.data['Sex'] = self.data['Sex'].map({'male': 0, 'female': 1})
        self.data['Embarked'] = self.data['Embarked'].astype('category').cat.codes
        
        # Feature Engineering
        self.data['Familysize'] = self.data['SibSp'] + self.data['Parch'] + 1
        self.data['Isalone'] = (self.data['Familysize'] == 1).astype(int)
        self.data['HasCabin'] = self.data['Cabin'].notnull().astype(int)
        self.data['Title'] = self.data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map(
            {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
        ).fillna(4)
        self.data['Pclass_Fare'] = self.data['Pclass'] * self.data['Fare']
        self.data['Age_Fare'] = self.data['Age'] * self.data['Fare']
        print("Preprocessing completed")

    def handle_class_imbalance(self):
        """Handle class imbalance using SMOTE."""
        X = self.data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Isalone', 'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']]
        y = self.data['Survived']
        smote = SMOTE(random_state=42)
        self.X_resampled, self.y_resampled = smote.fit_resample(X, y)
        print("Class imbalance handled using SMOTE")


    def store_features_in_redis(self):
        """Store features of each passenger in Redis."""
        batch_data = {}
        for idx, row in self.data.iterrows():
            entity_id = row['PassengerId']  # Use PassengerId as entity_id
            features = {
                "Age": row['Age'],
                "Fare": row['Fare'],
                "Sex": row['Sex'],
                "Embarked": row['Embarked'],
                "Familysize": row['Familysize'],
                "Isalone": row['Isalone'],
                "HasCabin": row['HasCabin'],
                "Title": row['Title'],
                "Pclass_Fare": row['Pclass_Fare'],
                "Age_Fare": row['Age_Fare'],
                "Survived": row['Survived']
            }
            batch_data[entity_id] = features
        
        self.feature_store.store_batch_features(batch_data)
        print("Features stored in Redis")

    def retrieve_features_from_redis(self, entity_id):
        """Retrieve features for a given entity from Redis."""
        features = self.feature_store.get_features(entity_id)
        if features:
            return features
        return None
    
    def run(self):
        self.load_data()
        self.preprocess_data()
        self.handle_class_imbalance()
        self.store_features_in_redis()



# Usage

# Create an instance of the RedisFeatureStore
feature_store = RedisFeatureStore()

# Initialize DataProcessing with paths to your Titanic dataset and the Redis feature store
data_processor = DataProcessing('artifacts/raw/titanic_train.csv', 'artifacts/raw/titanic_test.csv', feature_store)

# 1. Load data
data_processor.run()

# 6. Retrieve features from Redis
print(data_processor.retrieve_features_from_redis(entity_id=3))  # Example: Retrieve features for entity 1
