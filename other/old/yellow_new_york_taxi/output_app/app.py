import os
import sys
import datetime
import json

import pandas as pd
import numpy as np
import geopandas

from sklearn.metrics import r2_score

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

import plotly.graph_objs as go
import plotly.express as px

# custom libs
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from lib.lib_data import get_t, df_read

os.environ['DASH_DEBUG_MODE'] = 'False'

with open('../config.json') as json_file:
    config = json.load(json_file)

# ny zones geo data 
dbf = geopandas.GeoDataFrame.from_file('../data/taxi_zones/taxi_zones.dbf')
dbf = dbf.to_crs(4326)

# ----- get results dataframe  ------
with open('currenttime.json', 'w') as f:
    json.dump([-1, -1], f)

def update_results(config, forced=False):
    """Updates prediction and validations DataFrames
    
    Args:
      config(dict): config 
      forced(bool): force update 
    """
    
    # time indicator of last update
    with open('currenttime.json', 'r') as f:
        l = json.load(f)
        curent_hour, curent_month = int(l[0]), int(l[1])

    #  short run predictions
    if (datetime.datetime.now().hour != curent_hour and 
    datetime.datetime.now().minute > 5) or forced:
        filename = os.path.join(config['path_data'], config['prediction']['results_day'])
        results_day = df_read(filename, config)
        results_day['drives'] = results_day['pred']
        results_day.to_csv('results_day.csv', header=True, index=False)
        curent_hour = datetime.datetime.now().hour
        print('results_day are updated,', str(datetime.datetime.now()))
    
    # validations
    if ((datetime.datetime.now().month != curent_month and 
        (24*datetime.datetime.now().day +  datetime.datetime.now().hour) > 2) or 
        forced):
        filename = os.path.join(config['path_data'], config['prediction']['results_month'])
        results_month = df_read(filename, config)
        results_month['drives'] = results_month['pred']
        results_month.to_csv('results_month.csv', header=True, index=False)
        curent_month = datetime.datetime.now().month
        print('results_month are updated,', str(datetime.datetime.now()))
    
    with open('currenttime.json', 'w') as f:
        json.dump([curent_hour, curent_month], f)

update_results(config, forced=True)
results_day = pd.read_csv('results_day.csv')
results_month = pd.read_csv('results_month.csv')

# ------ filtering functions ------

def select(results, x, column='t'):
    df = results[results[column]==x]
    return df

def select_zone_h(zone_id, h, df):
    return df[(df['zone_id']==zone_id)&(df['h']==h)]

# ------ functions for output ------

def get_r_squared():
    """
    calculate r2_squared metric for all zine_id / h
    Args:
      None
    Return:
      r2(pd.DataFrame) r2 values for zine_id / h
    """
    results = pd.read_csv('results_month.csv')
    def f_agg(x):
        return pd.Series({'r2_score': r2_score(x['fact'], x['drives'])})
    df = results.groupby(['zone_id', 'h'], as_index=False).apply(f_agg)
    return df

def curent_hour_text(h):
    """
    Args:
      h(int): forecast hour 
    Returns:
      uptut_hour(str): simulation of current hour + h
    """
    h0 = datetime.datetime(datetime.datetime.now().year,
                             datetime.datetime.now().month, 
                             datetime.datetime.now().day, 
                             datetime.datetime.now().hour, 0)
    ouptut_hour = h0 + datetime.timedelta(seconds=3600*(h + config['output']['UTC correction']))
    return ouptut_hour.strftime("%m/%d %Hh")
    
# ------ map output functions ------

def plot_map(h, dbf):
    """Plots ny map with forecast results

    Args:
      h(int): forecsat horizont 
      dbf(geopandas.DataFrame): ny zones geo data
    Rerurn:
      None
    """
    
    # hour of prediction (in the data)
    prediction_hour = (datetime.datetime(int(config['training']['period'][1][:4]),
                                    int(config['training']['period'][1][5:7]), 
                                    datetime.datetime.now().day, 
                                    datetime.datetime.now().hour, 0)) 
    t = get_t(prediction_hour, config['h0'], config)

    # actualized data
    results = pd.read_csv('results_day.csv')
    df = select(results, t, column='t')
    df = select(df, h, column='h')
    
    fig = px.choropleth_mapbox(
        df,
        geojson=dbf,
        locations="zone_id",
        featureidkey='properties.LocationID',
        color="drives",
        mapbox_style="carto-positron",
        center={"lat": 40.76, "lon": -73.87}, 
        opacity=0.8, 
        color_continuous_scale = px.colors.sequential.Blues,
        title='Forecast for {}'.format(curent_hour_text(h)),
        hover_name="zone_id"
    )
    return fig

def plot_map_r2(h, dbf):
    """Plots ny map with r2_score values 

    Args:
      h(int): forecsat horizont 
      dbf(geopandas.DataFrame): ny zones geo data
    Rerurn:
      None
    """   
    
    r2 = get_r_squared()

    fig = px.choropleth_mapbox(
        r2[r2['h']==h],
        geojson=dbf,
        locations="zone_id",
        featureidkey='properties.LocationID',
        color="r2_score",
        mapbox_style= "carto-positron",
        center={"lat": 40.761, "lon": -73.95}, 
        opacity=0.65, 
        color_continuous_scale = px.colors.sequential.BuPu, 
        title='Forecast quality, h={}'.format(h),
        hover_name="zone_id",
        labels={
            'r2_score': 'R<sup>2</sup>'
        }
    ).update_layout(mapbox={"zoom": 10.4})
    return fig

# --- initial values ----

h0 = 1
id_start=163
r2 = get_r_squared()
r2.r2_score = r2.r2_score.map(lambda x: round(x, 3))
r2_start = r2[(r2['h']==h0)&(r2['zone_id']==id_start)].r2_score.values[0]

# ------ layouts ------

forecast_layout = html.Div([
    html.Div(
        dcc.Graph(id='ny-map')),
    
    html.Div(dcc.Slider(
        results_day['h'].min(),
        results_day['h'].max(),
        step=1,
        id='h-slider',
        value=h0
    ),
        className='w-30'
    ),
    
    html.Div(id='map_slider_text')
])

analytics_layout = html.Div(
    [html.Div([
        html.Div([
            dcc.Graph(
                id='crossfilter-ny-map',
                clickData={'points': [{'location': id_start, 'z': r2_start}]}
            )
        ],
            className='w-50 d-sm-inline-block'
        ),

        html.Div([
            dcc.Graph(
                id='crossfilter-plot',
            )
        ],
            className='w-50 d-sm-inline-block'
        )
    ]),
    
    html.Div(dcc.Slider(
        results_month['h'].min(),
        results_month['h'].max(),
        step=1,
        id='crossfilter-h-slider',
        value=h0
    ),
        className='w-30'
    ),
    
    html.Div(id='map_r2_slider_text')
])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

n_tabs = config['output']['n_tabs']
if n_tabs == 2:
    app.layout = dbc.Container(
        [ 
        html.Div(
                "Shortrun forecast for NY yellow taxi demand",
                className="heading",
            ),
        html.Div(config['output']['description'], className='description'),       

            dbc.Tabs(
                [
                    dbc.Tab(label='{}-hour forecast'.format(6), 
                            tab_id='map',
                            tabClassName='dash-tab',
                            labelClassName='.dash-tab-label',
                            activeTabClassName='dash-tab-active'),
                    dbc.Tab(label="Model analytics", 
                            tab_id='analytics',
                            tabClassName='dash-tab',
                            labelClassName='.dash-tab-label',
                            activeTabClassName='dash-tab-active'),
                ],
                id="tabs",
                active_tab="map",
            ),
            html.Div(id="tab-content"),
            html.Div(
                'Copyright © 2022 Datasophie GmbH | Pergola',
                className='page-footer',
            )
        ]
    )
else:
    app.layout = dbc.Container(
    [ 
      html.Div(
            "Shortrun forecast for NY yellow taxi demand",
            className="heading",
        ),
      html.Div(config['output']['description'], className='description'),       
      html.Div(forecast_layout, className='one-tab-case'),
        html.Div(
            'Copyright © 2022 Datasophie GmbH | Pergola',
            className='page-footer',
        )
    ]
)

# ---- tabs callback ----

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):
    update_results(config)
    if active_tab is not None:
        if active_tab == 'map':
            return forecast_layout
        elif active_tab == 'analytics':
            return analytics_layout
    return forecast_layout

# ---- forecast tab callbacks  ----
"""
@app.callback()
def update_data():
    global curent_hour, curent_month
    print(curent_hour, curent_month)
    update_results(config)
    print(curent_hour, curent_month)
"""

# map
@app.callback(
    Output('ny-map', 'figure'),
    Input('h-slider', 'value'),
)
def update_map_d1(h):
    update_results(config)
    fig = plot_map(h, dbf)
    title = 'Forecast for {}'.format(curent_hour_text(h))
    fig.update_layout(title=title, mapbox={"zoom": 10.4}, margin={"r":30,"t":80,"l":25,"b":30})
    return fig
# comment to slider
@app.callback(
    Output('map_slider_text', 'children'),
    Input('h-slider', 'value'))
def update_output_d1(value):
    return '{}-hour forecast'.format(value)

#  ---- analytics tab callbacks  ----

# map update
@app.callback(
    Output('crossfilter-ny-map', 'figure'),
    Input('crossfilter-h-slider', 'value'),
)
def update_map_d2(h):
    fig = plot_map_r2(h, dbf)
    title = 'Forecast quality, h={}'.format(h)
    fig.update_layout(title=title, margin={"r":55,"t":78, "l":30,"b":50})
    return fig

@app.callback(
    Output('crossfilter-plot', 'figure'),
    Input('crossfilter-ny-map', 'clickData'),
    Input('crossfilter-h-slider', 'value'),
)

# graph update
def update_graph_d2(clickData, h):
    """Updates fact/predict graph for choosen zone_id and forcast hour

    clickData(dcc.Input):  infornmation of the selected zone
    h(dcc.Input): forcast hour 
    """
    
    # data update
    r2 = get_r_squared()
    results_month = pd.read_csv('results_month.csv')
    
    zone_id = clickData['points'][0]['location']
    id_ =  clickData['points'][0]['location']
    
    r2_scores = r2[(r2['h']==h)&(r2['zone_id']==id_)].r2_score.values[0]
    dff = pd.melt(select_zone_h(zone_id, h, results_month).loc[:, ['t', 'fact', 'pred']], id_vars='t')
    fig = px.line(dff, x='t', y='value', color='variable', template='plotly_white',
                  color_discrete_sequence = ['rgb(128, 177, 211)', 'rgb(133, 92, 117)'
                  ]
                  )

    fig.update_traces(opacity=0.7)
    title = 'Validation: zone id={}, h={}, R<sup>2</sup>={}'.format(zone_id, h, round(r2_scores, 2))
    
    fig.update_layout(title=title, xaxis = go.layout.XAxis(
        title = 'Hours',
        showticklabels=False), yaxis = go.layout.YAxis(
        title = 'Drives'),
        legend_title = '',
        margin={"r":75, "t":75, "l":0,"b":45}
        )
    return fig
#  slider comment
@app.callback(
    Output('map_r2_slider_text', 'children'),
    Input('crossfilter-h-slider', 'value'))
def update_output(value):
    return '{}-hour forecast'.format(value)

DASH_DEBUG_MODE = os.environ['DASH_DEBUG_MODE']

if __name__ == "__main__":
    app.run_server(debug=DASH_DEBUG_MODE, host='0.0.0.0', port=8080)
