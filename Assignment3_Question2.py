import pandas as pd
import matplotlib.pyplot as plt

data_frame = pd.read_csv("crime.csv")
statistics = (data_frame["ViolentCrimesPerPop"])

plt.figure()
plt.hist(statistics, bins = 50)
plt.title("Histogram of Violent crimes per population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.boxplot(statistics)
plt.title("Boxplot of Violent crimes per population")
plt.xlabel("ViolentCrimesPerPop")
plt.ylabel("Frequency")
plt.show()

#The histogram is right-skewed. Most populations are
#clustered around 0-0.4, where we see the most data points, and they trail off and get less and less as they
#go towards the right. Since it's spread out a lot, that indicates that violent crimes are very varied acros
#different communities. Since as crime increases, frequency decreases, that indicates that there are
#fewer areas with very high violent crime rates.

#The median is shown by the orange line inside the box. The median is somewhere between 0.37-0.39. This means
#that half of the populations have violent crime rates below 0.39, and half of them hav violent crime rates
#above 0.39. Since the median is more on the bottom of the boxplot, the data can be said to be right skewed,
#which is consistent with the histogram. The median indicates that most areas have low to moderate crime rates.

#There are no individual point outside the bounds of the upper fence (UF) or lower fence (LF), which means that
#are no obvious outliers. Even though on the histogram, there appears to be an outlier, if you look at the
#boxplot, it is still within the range of the UF, so there is no outlier, as all data falls within the range
#of the UF. The lower end also doesn't have any outliers, and ends at 0, as you can't have negative crime in this sense.
#Overall, the boxplot does not suggest outliers, as everything falls withing the range of the UF and LF.