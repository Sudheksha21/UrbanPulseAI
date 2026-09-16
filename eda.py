import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned traffic data
data = pd.read_csv("data/cleaned_traffic.csv")

print("===== DATASET INFORMATION =====")
print(data.info())

print("\n===== STATISTICAL SUMMARY =====")
print(data.describe())

# 1. Average traffic by hour
hourly_traffic = data.groupby("hour")["traffic_volume"].mean()

plt.figure(figsize=(10, 5))
hourly_traffic.plot(kind="line", marker="o")
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.grid(True)
plt.savefig("graphs/traffic_by_hour.png")
plt.show()

# 2. Average traffic by day
daily_traffic = data.groupby("day_of_week")["traffic_volume"].mean()

plt.figure(figsize=(10, 5))
daily_traffic.plot(kind="bar")
plt.title("Average Traffic Volume by Day of Week")
plt.xlabel("Day of Week (0 = Monday)")
plt.ylabel("Average Traffic Volume")
plt.savefig("graphs/traffic_by_day.png")
plt.show()

# 3. Traffic distribution
plt.figure(figsize=(10, 5))
sns.histplot(data["traffic_volume"], kde=True)
plt.title("Traffic Volume Distribution")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.savefig("graphs/traffic_distribution.png")
plt.show()

print("\nEDA completed successfully!")
print("Graphs saved in the graphs folder.")