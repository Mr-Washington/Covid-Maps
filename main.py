# This is a sample Python script.

import pandas as pd
import plotly.graph_objects as go
import requests

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


df = pd.read_csv("Report_Public.csv", usecols= ['County', 'State Abbreviation', 'Population', "Population as a percent of state", 'Cases - last 7 days', 'Deaths - last 7 days', 'Cumulative cases', 'Cumulative deaths', 'Cases - previous 7 days', 'Deaths - previous 7 days', '% Uninsured', '% In Poverty', '% Over Age 65', 'Average household size', '% Hispanic', '% Non-Hispanic Black', '% Native American / Alaskan Native', '% Asian'])
#df1 = df[['Population as a percent of state', 'Cases - last 7 days', 'Deaths - last 7 days', 'Cumulative cases', 'Cumulative deaths', '% In Poverty', '% Over Age 65', '% Hispanic', '% Non-Hispanic Black']]
df1 = df.sort_values(by=["Cumulative cases"], ascending=False)
print(df.head())
df.to_csv('saved_data.csv')

"""
df = pd.read_csv("12-28-2020.csv")
df.head()

df = df.rename(columns= {"Country_Region" : "Country", "Province_State": "Province"})
df.head()

df['text'] = df['Country'].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = df["Long_"],
    lat = df["Lat"],
    text = df["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'square',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = df['Confirmed'],
        cmax = df['Confirmed'].max(),
        colorbar_title = "COVID 19 Reported Cases"
    )
))

fig.update_layout(
    title = "COVID19 Confirmed Cases Around the World",
    geo = dict(
        scope = "world",
        showland = True,
    )
)

fig.write_html('first_figure.html', auto_open=True)

"""
"""


# def print_hi(name):
# Use a breakpoint in the code line below to debug your script.
#    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


import csv
import arviz as az
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pymc3 as pm

from pymc3 import *

matplotlib.use('TkAgg')

print(f"Running on PyMC3 v{pm.__version__}")

def getpopdensvscase():
    x = []
    y = []
    with open('saved_data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row["Cases - last 7 days"].replace(' ', '') == '' or row["Cumulative cases"].replace(' ', '') == '' or row["Cases - last 7 days"].replace(' ', '') == '-' or row["Cumulative cases"].replace(' ', '') == '-': continue
            perc = float(row["Cases - last 7 days"].replace(',',''))
            cum = float(row["Cumulative cases"].replace(',',''))
            x.append(perc)
            y.append(cum)
    return (x,y)

if __name__ == "__main__":

# %config InlineBackend.figure_format = 'retina'
    az.style.use("arviz-darkgrid")

    x,y = getpopdensvscase()
    print(len(x))
    print(len(y))
# print(x)
# print(y)
    data = dict(x=x, y=y)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, xlabel="Population as a percent of state", ylabel="Cumulative cases", title="Generated data")
    ax.plot(x, y, "x", label="sampled data")
    plt.legend(loc=0);
# plt.show()

# with Model() as model:
#     # specify glm and pass in data. The resulting linear model, its likelihood and
#     # and all its parameters are automatically added to our model.
#     glm.GLM.from_formula("y ~ x", data)
#     trace = sample(1000, cores=2)  # draw 3000 posterior samples using NUTS sampling

    with Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
    # Define priors
        sigma = HalfCauchy("sigma", beta=50000, testval=10000)  # prior setting up the equations for trainning
        intercept = Normal("Intercept", 0, sigma=50) # Mean and sigma standard deviation??
        x_coeff = Normal("x", 0, sigma=50)

        likelihood = Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=y)  # combine x and y
    # trace = np.zeros(1)

    # Define likelihood
    #likelihood = Normal("y", mu=intercept + x_coeff * x, sigma=sigma, observed=y) # combine x and y
    # Inference!
        trace = sample(3000, cores=3)  # draw 3000 posterior samples using NUTS sampling

    plt.figure(figsize=(7, 7))
    traceplot(trace)
    plt.tight_layout()
    plt.show()
"""
