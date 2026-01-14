import pickle

class DummyModel:
    def predict(self, X):
        return [0] * len(X)

with open("model.pkl", "wb") as f:
    pickle.dump(DummyModel(), f)

print("Dummy model created for CI testing")