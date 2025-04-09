import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('linear_current_force_2.csv')
first_column = data.iloc[:, 3]
plt.plot(first_column)
plt.title("First Column Data")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()