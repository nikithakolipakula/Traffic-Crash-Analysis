import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sns.set_style("white")


# LOAD DATASET


df = pd.read_csv("Traffic_Crashes_-_Crashes.csv")

print("DATASET")
print(df.head())
print(".........................")



# INFO & DESCRIPTION


print("INFO")
df.info()

print("DESCRIPTIVE STATISTICS")
print(df.describe())
print(".........................")



# MISSING VALUE


print("MISSING VALUES")
print(df.isnull().sum())
print(".........................")



# HANDLE MISSING VALUES

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("AFTER CLEANING")
print(df.isnull().sum())


# REMOVE DUPLICATES

df.drop_duplicates(inplace=True)


# NUMPY ANALYSIS

num_cols = df.select_dtypes(include=np.number).columns

print("\nNUMPY ANALYSIS")
print("Mean:", np.mean(df[num_cols].iloc[:,0]))
print("Std:", np.std(df[num_cols].iloc[:,1]))
print("Max:", np.max(df[num_cols].iloc[:,0]))


# SAMPLE DATA

df_small = df.sample(500)
num_df = df_small.select_dtypes(include=np.number).iloc[:, :5]


# DATA VISUALIZATION

# HISTOGRAM
plt.figure()
sns.histplot(num_df.iloc[:,0], bins=30, color="skyblue")
plt.title(f"Histogram of {num_df.columns[0]}")
plt.xlabel(num_df.columns[0])
plt.ylabel("Frequency")
plt.show()

# REGRESSION PLOT
plt.figure()
sns.regplot(x=num_df.iloc[:,0], y=num_df.iloc[:,1], data=num_df, color="green")
plt.title(f"{num_df.columns[0]} vs {num_df.columns[1]}")
plt.xlabel(num_df.columns[0])
plt.ylabel(num_df.columns[1])
plt.show()

# PAIRPLOT 
sns.pairplot(num_df.iloc[:, :3], diag_kind="kde")
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# HEATMAP (better color)
plt.figure()
sns.heatmap(num_df.corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap")
plt.show()

# BOXPLOT (clearer)
plt.figure(figsize=(10,5))
sns.boxplot(data=num_df, palette="Set2")
plt.title("Boxplot (Outlier Detection)")
plt.xlabel("Features")
plt.ylabel("Values")
plt.xticks(rotation=30)
plt.show()

# LINE PLOT
if 'CRASH_HOUR' in df.columns:
    hourly = df.groupby('CRASH_HOUR')[num_cols[0]].count()
    plt.figure()
    sns.lineplot(x=hourly.index, y=hourly.values, marker='o', color="purple")
    plt.title("Crash Trend by Hour")
    plt.xlabel("Crash Hour")
    plt.ylabel("Number of Crashes")
    plt.show()

# PIE CHART
top = df[num_cols[0]].value_counts().head(5)

plt.figure()
colors = sns.color_palette("pastel")
plt.pie(top.values, labels=top.index, autopct='%1.1f%%', colors=colors)
plt.title("Top 5 Distribution")
plt.legend(title="Categories", loc="best")
plt.show()




# CORRELATION & COVARIANCE

print("\nCorrelation:\n", df.corr(numeric_only=True))
print("\nCovariance:\n", df.cov(numeric_only=True))


# STATISTICAL MODELLING

print("\nSTATISTICAL VALUES")
print("Mean:", df[num_cols[0]].mean())
print("Median:", df[num_cols[0]].median())
print("Variance:", df[num_cols[0]].var())
print("Standard Deviation:", df[num_cols[0]].std())


# HYPOTHESIS TESTING

# Shapiro Test
stat, p = stats.shapiro(df[num_cols[0]].sample(500))
print("\nShapiro Test p-value:", p)

if p < 0.05:
    print("Data is NOT normally distributed")
else:
    print("Data is normally distributed")


# T-Test
high = df[df[num_cols[0]] > df[num_cols[0]].median()][num_cols[1]]
low = df[df[num_cols[0]] <= df[num_cols[0]].median()][num_cols[1]]

t_stat, p_value = stats.ttest_ind(high, low)

print("\nT-Test p-value:", p_value)

if p_value < 0.05:
    print("Significant difference between groups")
else:
    print("No significant difference")


# Z-TEST 
mean1 = np.mean(high)
mean2 = np.mean(low)
std1 = np.std(high)
std2 = np.std(low)

n1 = len(high)
n2 = len(low)

z = (mean1 - mean2) / np.sqrt((std1**2/n1) + (std2**2/n2))

print("\nZ-Test value:", z)


# OUTLIER REMOVAL

Q1 = df[num_cols[0]].quantile(0.25)
Q3 = df[num_cols[0]].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df[num_cols[0]] >= lower) & (df[num_cols[0]] <= upper)]


# MACHINE LEARNING

df_num = df.select_dtypes(include=np.number)

X = df_num.drop(df_num.columns[-1], axis=1)
y = df_num[df_num.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nMODEL EVALUATION")
print("R2 SCORE:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))


# SAVE DATA

df.to_csv("traffic_cleaned.csv", index=False)

print("\nPROJECT COMPLETED SUCCESSFULLY")
