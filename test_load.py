from src.cf_model import CFModel

cf = CFModel("models/svd_model.pkl")
print(cf.predict(1, 1))
