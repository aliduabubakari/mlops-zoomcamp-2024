import seaborn as sns
import matplotlib.pyplot as plt

def plot_trip_duration_distribution(df):
    sns.histplot(df["trip_duration"], bins=100)
    plt.title("Trip Duration Distribution")
    plt.show()