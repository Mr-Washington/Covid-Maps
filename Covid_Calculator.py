import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("12-28-2020.csv")
df.head()
df = df.rename(columns={"Country_Region": "Country", "Province_State": "Province"})
df.head()

df['text'] = df['Country'].astype(str)
fig = go.Figure(data=go.Scattergeo(
    lon=df["Long_"],
    lat=df["Lat"],
    text=df["text"],
    mode="markers",
    marker=dict(
        size=12,
        opacity=0.8,
        reversescale=True,
        autocolorscale=True,
        symbol='square',
        line=dict(
            width=1,
            color='rgba(102, 102, 102)'
        ),
        cmin=0,
        color=df['Confirmed'],
        cmax=df['Confirmed'].max(),
        colorbar_title="COVID 19 Reported Cases"
    )
))

fig.update_layout(
    title="COVID19 Confirmed Cases Around the World",
    geo=dict(
        scope="world",
        showland=True,
    )
)

fig.write_html('first_figure.html', auto_open=True)

import csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pymc3 as pm

from pymc3 import *

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

print(f"Running on PyMC3 v{pm.__version__}")


def getpopdensvscase():
    x = []
    y = []
    with open('/content/saved_data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["Cases - last 7 days"].replace(' ', '') == '' or row["Population"].replace(' ', '') == '' or row[
                "Cases - last 7 days"].replace(' ', '') == '-' or row["Population"].replace(' ', '') == '-': continue
            perc = float(row["Cases - last 7 days"].replace(',', ''))
            cum = float(row["Population"].replace(',', ''))
            x.append(perc)
            y.append(cum)
    return (x, y)


trace = None

x, y = getpopdensvscase()

data = dict(x=x, y=y)
x = np.array(x)
y = np.array(y)

data = np.vstack((x, y))

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, xlabel="State Population", ylabel="Cases - last 7 days",
                     title="Cases - last 7 days per population")
ax.plot(x, y, "x", label="sampled data")
plt.legend(loc=0);
plt.show()

print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
    sigma = HalfCauchy("sigma", beta=50000, testval=150000)
    intercept = Normal("Intercept", 0, sigma=50)
    x_coeff = Normal("x", 0, sigma=50)

    # Define likelihood
    likelihood = Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=y)

if __name__ == "__main__":
    with model:
        # Inference!
        trace = sample(3000, cores=2)  # draw 3000 posterior samples using NUTS sampling

plt.figure(figsize=(7, 7))
traceplot(trace)
plt.tight_layout();

import pandas as pd
import numpy as np


def get_population(state):
    df = pd.read_csv("/content/saved_data.csv")
    total_pop = 0
    total_cases = 0
    # print (df[df['County'].str.contains('NY')])
    # dataf = df['County'].where(df['State Abbreviation'].str.contains('NY'))
    dataf = df.loc[df['State Abbreviation'] == state]
    dataf = dataf.replace(np.nan, 0)
    dataf = dataf.replace(' -   ', 0)
    # dataf = df.fillna(0)
    # print(dataf)
    for index, row in dataf.iterrows():
        # check_for_nan = row['County'].isnull()
        # if (row['Population'] != 'nan'):
        # print(row['County'], row['Population'])
        pop = int(str(row['Population']).replace(',', ''))
        case = int(str(row['Cases - last 7 days']).replace(",", ""))
        total_pop = total_pop + pop
        total_cases = total_cases + case

    total = total_pop / total_cases
    print('\nIn the United States 1 in 141 people get Covid-19\n')
    print('Living in ', state, ' You have average of every', round(total), 'people 1 gets COVID-19')
    if total < 121:
        print('\tBased on your state you have a HIGH likelyhood of getting COVID-19\n')
    if total > 161:
        print('\tBased on your state you have a LOW likelyhood of getting COVID-19\n')
    if total < 161 and total > 121:
        print('\tBased on your state you have an AVERAGE likelyhood of getting COVID-19\n')
    # return round(total,0)


def get_county(county):
    df = pd.read_csv("/content/saved_data.csv")
    total_pop = 0
    total_cases = 0
    # print (df[df['County'].str.contains('NY')])
    # dataf = df['County'].where(df['State Abbreviation'].str.contains('NY'))
    dataf = df.loc[df['County'] == county]
    dataf = dataf.replace(np.nan, 0)
    dataf = dataf.replace(' -   ', 0)
    # dataf = df.fillna(0)
    # print(dataf)
    for index, row in dataf.iterrows():
        # check_for_nan = row['County'].isnull()
        # if (row['Population'] != 'nan'):
        # print(row['County'], row['Population'])
        pop = int(str(row['Population']).replace(',', ''))
        case = int(str(row['Cases - last 7 days']).replace(",", ""))
        total_pop = total_pop + pop
        total_cases = total_cases + case

    total = total_pop / total_cases
    # print('Living in the United States there is a National average of every 141 people 1 gets Covid')
    print('Living in ', county, ' 1 of every', round(total), 'people get COVID-19')
    if total < 121:
        print('\tBased on your county you have a HIGH likelyhood of getting COVID-19')
    if total > 161:
        print('\tBased on your county you have a LOW likelyhood of getting COVID-19')
    if total < 161 and total > 121:
        print('\tBased on your county you have an AVERAGE likelyhood of getting COVID-19')
    # return round(total,0)


county = str(input("What county do you live in?: "))
state = str(input("What state do you live in?: "))
get_population(state)
get_county(county)