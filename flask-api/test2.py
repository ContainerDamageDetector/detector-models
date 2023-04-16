import xgboost as xgb

model = xgb.Booster()
model.load_model('recover_price/saved_model.model')
print(type(model))