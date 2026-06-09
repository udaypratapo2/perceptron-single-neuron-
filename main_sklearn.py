from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x, y = make_classification(n_samples=100 , n_features = 4 , n_classes = 2 , random_state = 42)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

model = Perceptron(
    max_iter=1000,
    eta0 = 1.0,
    tol=1e-3,
    random_state = 42,
    shuffle = True
)

print(x)
model.fit(x_train , y_train)

accuracy = model.score(x_test , y_test)

print(f"Accuracy: {accuracy:.2f}")




