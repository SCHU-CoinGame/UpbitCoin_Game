import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../data/code_aligned/BTC-1INCH.csv')

# Assuming you have a dataframe `df` with 'trade_timestamp' and 'trade_price' columns
# Convert 'trade_timestamp' to datetime
df['trade_timestamp'] = pd.to_datetime(df['trade_timestamp'], unit='ms')  # Adjust the unit if needed

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting trade_price against trade_timestamp
ax.plot(df['trade_timestamp'], df['trade_price'], label='Trade Price')

# Formatting the x-axis to show time with date markers
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))  # Show time as hours:minutes
ax.xaxis.set_minor_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Show the date

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Mark date changes on the x-axis
df['date'] = df['trade_timestamp'].dt.date
for date in df['date'].unique():
    ax.axvline(pd.Timestamp(date), color='gray', linestyle='--', linewidth=0.5)

# Labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Trade Price')
ax.set_title('Trade Price vs Time with Date Change Marks')

# Show the plot
plt.tight_layout()
plt.show()