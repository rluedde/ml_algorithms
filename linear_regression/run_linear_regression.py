import pandas as pd
from linear_regression import LinearModel
import statsmodels.api as sm

df = pd.read_csv("homeprices.csv")
lm = LinearModel(df, "area", "price","diff")
print(lm.fit_model())

preds = lm.make_predictions()

lm = LinearModel(df, "area", "price", "gd")

output = lm.fit_model(iterations = 10000, lr = .001)

print(output)