import pandas as pd
data_frame = pd.read_csv("crime.csv")
statistics = (data_frame["ViolentCrimesPerPop"])
mean = statistics.mean()
median = statistics.median()
STD = statistics.std()
minimum_value = statistics.min()
maximum_value = statistics.max()
print("Here are the results: ")
print(statistics)
print("Mean: " ,mean)
print("Median: " ,median)
print("Standard deviation: " ,STD)
print("Minimum value: " ,minimum_value)
print("Maximum value: " ,maximum_value)
#Compare the mean and median. Does the distribution look symmetric or skewed? Explain briefly.
#The mean and median are almost the same, but still slightly different. If a mean > median, that means
#that a graph is right skewed, since mean is a little greater than the median, the graph could be
#said to be right skewed.

#If there are extreme values (very large or very small), which statistic is more affected: mean or median? Explain why
#The statistic more affected by outliers, or, extreme values is the mean. This is because the mean is calculated
#using every statistic in the dataset. So if there's 10 points, 9 being 2-4, and the last one being 100, the
#mean would be very skewed, while the median would stay almost the same, because it accounts for data points
#halfway below or halfway above it, in the middle, so outliers don't move it by much.
