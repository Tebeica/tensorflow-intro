import tensorflow as tf
import numpy as np
import pandas as pd
import functools
from tensorflow.keras.layers.experimental import preprocessing

# ---- MACROS ------
# Path to csv files
# Label of column to predicted
# Loading csv files as datasets
# Defining global variables
TRAIN_CSV_PATH = "heart_train.csv"
TEST_CSV_PATH = "heart_test.csv"
BATCH_SIZE = 5
# EPOCH_SIZE = 128
EPOCH_SIZE = 512
# the label we are running our model for
LABEL_NAME = "chd"
# Labels of columns containing numerical data
NUMERIC_FEATURES = ["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]
# Labels of columns containing categorical data (and their categories)
CATEGORIC = {"famhist": ["Present", "Absent"]}


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


def create_model():
    # Building the model based on the following tutorial on how to handle overfitting and underfitting
    # https://www.tensorflow.org/tutorials/keras/overfit_and_underfit
    m = tf.keras.Sequential()
    # Combine numerical and categorical columns into a single layer
    m.add(tf.keras.layers.DenseFeatures(categoricalColumns + numericColumns))
    m.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    m.add(tf.keras.layers.Dropout(0.5))
    m.add(tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu'))
    m.add(tf.keras.layers.Dropout(0.5))
    m.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # m.add(tf.keras.layers.Dense(256, activation='relu'))
    # m.add(tf.keras.layers.Dense(256, activation='relu'))
    # m.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return m


def divide_data(path):
    df = pd.read_csv(path)
    rng = np.random.RandomState()

    train_data = df.sample(frac=0.9, random_state=rng)
    test_data = df.loc[~df.index.isin(train_data.index)]

    train_data.to_csv("heart_train.csv", index=False)
    test_data.to_csv("heart_test.csv", index=False)


# code created with the help of tutorial
# https://www.tensorflow.org/tutorials/load_data/csv#basic_preprocessing
def get_data(path):
    dataset = tf.data.experimental.make_csv_dataset(path, batch_size=BATCH_SIZE, label_name=LABEL_NAME)
    return dataset


def pack_data(dataset):
    mapped_data = dataset.map(PackNumericFeatures(NUMERIC_FEATURES))
    return mapped_data


def create_categorical_columns():
    res = []
    for feature, vocab in CATEGORIC.items():
        cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature, vocabulary_list=vocab)
        res.append(tf.feature_column.indicator_column(cat_col))
    return res


print("--Splitting heart.csv into heart_train.csv and heart_test.csv--")
divide_data("heart.csv")

print("--Get data--")
trainDataset = get_data(TRAIN_CSV_PATH)
testDataset = get_data(TEST_CSV_PATH)

print("--Process data--")

# Pack the numerical columns of the training and testing data into a single column
mappedTrainData = pack_data(trainDataset)
x_train, y_train = next(iter(mappedTrainData))

mappedTestData = pack_data(testDataset)
x_test, y_test = next(iter(mappedTestData))

# use pandas and numpy to normalize data
desc = pd.read_csv(TRAIN_CSV_PATH)[NUMERIC_FEATURES].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

# Create numerical columns
numericColumn = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numericColumns = [numericColumn]

# Create categorical columns
categoricalColumns = create_categorical_columns()

print("--Make model--")
model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(mappedTrainData, epochs=20, steps_per_epoch=EPOCH_SIZE)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(mappedTestData, steps=EPOCH_SIZE)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuracy: {model_acc * 100:.1f}%")

