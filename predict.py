import pickle
import cml.models_v1 as models

with open(".experiments/byol-lhr8-20pq-ez00/a1wd-7amf-vusl-ttxa/artifacts/model/model.pkl", "rb") as f:
    model = pickle.load(f)

@models.cml_model
def predict(data):
    inputs = data.get("inputs", [])
    predictions = model.predict(inputs)
    return {"predictions": predictions.tolist()}
