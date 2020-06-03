import pandas as pd
from linear_regression import LinearModel
import statsmodels.api as sm

df = pd.read_csv("homeprices.csv")
lm = LinearModel(df, "area", "price","diff")
print(lm.fit_model())

sm_model = sm.OLS.from_formula("price ~ area", data = df)
res = sm_model.fit()
print(res.summary())


