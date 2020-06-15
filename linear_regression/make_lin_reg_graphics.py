
from linear_regression import LinearModel
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.DataFrame({"X": [1,2,4,6,8], "y": [1,2,4,7,8]})
diff_lm = LinearModel(df, "X", "y", "diff")

start = time.time()
output = diff_lm.fit_model()
end = time.time()

diff_fit_time = end - start
print(f"It took {diff_fit_time} seconds to fit the model using differentiationm. Here's the output: \n {output}")
print("Number of points:", df.shape[0])

df = pd.DataFrame({"X": [1,2,4,6,8], "y": [1,2,4,7,8]})

# "gd" - gradient decent
gd_lm = LinearModel(df, "X", "y", "gd")

start = time.time()
output = gd_lm.fit_model(iterations = 10000, lr = .0001)
end = time.time()

diff_fit_time = end - start
print(f"It took {diff_fit_time} seconds to fit the model using gradient descent. Here's the output: \n {output}")

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)

diff_preds = diff_lm.make_predictions()
ax1.plot(df.X, df.y, "ro", df.X, diff_preds)
ax1.set_title("Predictions with differentiation")

gd_preds = gd_lm.make_predictions()
ax2.plot(df.X, df.y, "ro", df.X, gd_preds, "g")
ax2.set_title("Predictions with gradient descent")
#ax1.set_facecolor()
#ax2.set_facecolor((1.0, 0.47, 0.42))

fig.set_facecolor("#f2f2f2")
plt.show()