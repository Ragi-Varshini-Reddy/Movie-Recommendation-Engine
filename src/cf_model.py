import pickle

class CFModel:
    def __init__(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict(self, user_id, movie_id):
        return self.model.predict(user_id, movie_id).est
