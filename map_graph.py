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
