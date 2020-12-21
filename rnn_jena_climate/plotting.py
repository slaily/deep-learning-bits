from matplotlib import pyplota as plt


temp = float_data[:, 1]
# Plotting the temperature timeseries
plt.plot(range(len(temp)), temp)
# Plotting the first 10 days of the temperature timeseries
plt.plot(range(1440), temp[:1440])