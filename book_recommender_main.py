import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


# Reading the csv files
books_df = pd.read_csv("Data/books.csv")
ratings_df = pd.read_csv("Data/ratings.csv")

# Exploring the dataset
print(ratings_df.head())
print(books_df.head())
print("Number of unique books: ", books_df.book_id.nunique())  # 10000
print("Number of unique users: ", ratings_df.user_id.nunique())  # 53424
print("Shape of ratings_df: ", ratings_df.shape)  # (981756, 3)
print("Shape of books_df: ", books_df.shape)  # (10000, 23)

# Checking if there is any missing values
print(ratings_df.isnull().sum())  # No missing values
print(books_df.isnull().sum())  # Some missing values

# Splitting the data into train and test set, using a ratio of 0.8
X_train, X_test = train_test_split(ratings_df, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Building the model using Functional API
# We will create two separate embeddings: one for users, and one for books, then concatenate them together
n_users_index = ratings_df.user_id.nunique()
n_books_index = ratings_df.book_id.nunique()

# For books
books_input = tf.keras.layers.Input(shape=[1])
books_embeddings = tf.keras.layers.Embedding(n_books_index + 1, 24)(books_input)
books_output = tf.keras.layers.Flatten()(books_embeddings)

# For users
users_input = tf.keras.layers.Input(shape=[1])
users_embeddings = tf.keras.layers.Embedding(n_users_index + 1, 24)(users_input)
users_output = tf.keras.layers.Flatten()(users_embeddings)

# Concatenate
concat = tf.keras.layers.Concatenate()([books_output, users_output])
x1 = tf.keras.layers.Dense(128, activation="relu")(concat)
x2 = tf.keras.layers.Dense(1, activation="relu")(x1)  # Ratings for each book in the train dataset
model = tf.keras.Model([books_input, users_input], x2, name="book_user_model")


# Compiling our model
# Best learning rate = 1e-3, found after some learning rate scheduling
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss="mse")

# Training the model
history = model.fit([X_train.book_id, X_train.user_id],
                    X_train.rating,
                    batch_size=128,  # For quicker training
                    epochs=3,
                    verbose=1,
                    validation_data=([X_test.book_id, X_test.user_id], X_test.rating))

# Finally let's see the book recommendation for the 5th user

# We do this by passing both the list of all books and the corresponding user
# index (in this case 5) to the model. Note that they have to have the same shape.
books_index = list(ratings_df.book_id.unique())
books_array = np.array(books_index)
user5_array = np.array([5 for _ in range(len(books_array))])

user5_predicted_book_ratings = model.predict([books_array, user5_array])
user5_predicted_book_ratings = user5_predicted_book_ratings.reshape(-1)

# Sorting the book index in descending ratings order
highest_predicted_book_index = user5_predicted_book_ratings.argsort()
# We only want the top 10 rated books
top10_predicted_book_index = highest_predicted_book_index[0:10]


# Locating the top 10 books from the books_df and show the most important
# features only
important_features = ["title", "authors", "original_publication_year", "isbn", "isbn13", "image_url"]
trunc_books_df = books_df[important_features]
print(trunc_books_df.iloc[top10_predicted_book_index])
