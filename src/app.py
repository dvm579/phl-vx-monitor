import pandas as pd
import time
import datetime
import pickle
import pytz

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, ctx, dash_table
import dash_bootstrap_components as dbc

from get_data import (aggregate_data, get_storageUnits, get_transactions, get_tempLog, get_lotsBatches,
                      expand_rows, calculate_inventory)

'''
*** Initialize data
'''
last_refreshed = datetime.datetime.now()
temp_df = pd.DataFrame()
lots_df = pd.DataFrame()
transactions_df = pd.DataFrame()
units = []


# agg_df = pd.DataFrame()


def check_cache():
    # global agg_df
    global units
    global lots_df
    global temp_df
    global transactions_df
    global last_refreshed
    data_file = open('data/data.pickle', 'rb')
    data = pickle.load(data_file)
    data_file.close()
    if data['timestamp'] + datetime.timedelta(minutes=5) < datetime.datetime.now():
        print(f"Updating now ({datetime.datetime.now()}). Last updated {data['timestamp']}")
        last_refreshed = datetime.datetime.now()
        data['timestamp'] = last_refreshed
        temp_df = get_tempLog()
        lots_df = get_lotsBatches()
        transactions_df = get_transactions()
        # agg_df = aggregate_data(temp_df, lots_df, transactions_df)
        units = get_storageUnits().to_dict('records')
        data_file = open('data/data.pickle', 'wb')
        pickle.dump(
            {'timestamp': datetime.datetime.now(), 'units': units, 'temps': temp_df, 'lots': lots_df,
             'transactions': transactions_df},  # , 'df': agg_df,
            data_file)
        data_file.close()
    else:
        print(f"Not updating now ({datetime.datetime.now()}). Last updated {data['timestamp']}")
        # agg_df = data['df']
        units = data['units']
        temp_df = data['temps']
        lots_df = data['lots']
        transactions_df = data['transactions']
        last_refreshed = data['timestamp']


check_cache()

"""
*** COMPONENTS
"""

"""
* Title / Logo
"""
logo = html.Img(src='assets/PHL Horizontal Logo White.png', className='d-inline',
                style={'height': '1em', 'vertical-align': 'middle', 'margin-right': '7px'})
title = html.H1(
    "Vaccine Monitoring", className="text-white align-middle pt-2 ml-2"
)
header = html.Div([logo, title], className='d-flex justify-content-center align-items-center bg-primary p-2 mb-2 fs-1')

"""
* Current Tab
"""

"""
* Snapshots Tab
"""
# Controls
snapshots_unit_dropdown = dcc.Dropdown(
    id='unitID-dropdown',
    options=[{'label': i['Name'], 'value': i['UnitID']} for i in units],
    value=units[0]['UnitID'],  # default value
    className='dbc'
)
snapshots_date_picker = dcc.DatePickerSingle(id='date-picker', min_date_allowed=temp_df['Timestamp'].min(),
                                             max_date_allowed=datetime.date.today(), date=temp_df['Timestamp'].max(),
                                             className='dbc'
                                             )
snapshots_time_slider = dcc.Slider(id='time-slider', min=0, max=86399,
                                   value=datetime.datetime.now(tz=pytz.timezone('America/Chicago')).timestamp() % 86400,
                                   marks={0: '12A', 10800: '3A', 21600: '6A', 32400: '9A', 43200: '12P', 54000: '3P',
                                          64800: '6P',
                                          75600: '9P',
                                          86399: '12A'},
                                   step=900,
                                   included=False,
                                   className='pt-3'
                                   )
snapshots_controls = dbc.Card(dbc.CardBody([
    html.Div([dbc.Label("Select Unit"), snapshots_unit_dropdown]),
    html.Div([dbc.Label("Set Date / Time")], className='pt-2'),
    html.Div([snapshots_date_picker], className='d-flex justify-content-center'),
    snapshots_time_slider
]), className='align-middle')
# Outputs
time_snapshot = html.Div(id='time-display', className='dbc d-inline-flex mx-5')
temp_snapshot = html.Div(id='temp-display', className='dbc d-inline-flex mx-5')
time_temp_snapshot = dbc.CardBody([time_snapshot, temp_snapshot],
                                           className='d-flex justify-content-between px-5')
snapshot_output = dbc.Row(
    [
        dbc.Col(dbc.Card([
            dbc.CardHeader("Inventory Snapshot"),
            time_temp_snapshot,
            dbc.CardBody(dash_table.DataTable(id='inventory-table',
                                              data=pd.DataFrame().to_dict('records'),
                                              columns=[{"name": i, "id": i} for i in lots_df.columns],
                                              filter_action="native",
                                              hidden_columns=['LotID', 'BatchID', 'Storage Location'],
                                              export_format='csv',
                                              sort_action="native",
                                              sort_mode="multi",
                                              page_action="native",
                                              page_current=0,
                                              page_size=10,
                                              style_data={
                                                  'whiteSpace': 'normal',
                                                  'height': 'auto',
                                              },
                                              ))
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Daily Temperatures"),
            dbc.CardBody([dcc.Graph(id='temperature-snapshot')
                          ])
        ]), width=6)
    ])
trans_table = dbc.Card([
    dbc.CardHeader("Daily Transactions"),
    dbc.CardBody(dash_table.DataTable(id='trans-table',
                                      data=pd.DataFrame().to_dict('records'),
                                      columns=[{"name": i, "id": i} for i in transactions_df.columns],
                                      filter_action="native",
                                      hidden_columns=['LotID', 'BatchID', 'Storage Location',
                                                      'Destination LotID', 'Destination BatchID',
                                                      "Transaction ID", "New Expiration Date", 'dt', ],
                                      export_format='csv',
                                      sort_action="native",
                                      sort_mode="multi",
                                      page_action="native",
                                      page_current=0,
                                      page_size=10,
                                      style_data={
                                          'whiteSpace': 'normal',
                                          'height': 'auto',
                                      },
                                      ))
])
# Tab Layout
snapshots_tab = dbc.Tab([snapshots_controls, snapshot_output, trans_table, html.Div(style={"height": '80px'})], label='Snapshots')

"""
* Historical Tab
"""
# Controls
historical_unit_dropdown = dcc.Dropdown(
    id='unitID-dropdown-historical',
    options=[{'label': i['Name'], 'value': i['UnitID']} for i in units],
    value=units[0]['UnitID'],  # default value
    className='dbc'
)
date_range_picker = dcc.DatePickerRange(
        id='historical-date-picker',
        min_date_allowed=temp_df['Timestamp'].min(),
        max_date_allowed=temp_df['Timestamp'].max(),
        initial_visible_month=datetime.date.today(),
        start_date=datetime.date.today() - datetime.timedelta(days=30),
        end_date=datetime.date.today()
    )
# temp_inv_checklist = dbc.Checklist(options=['Temperature', 'Inventory'], value=['Temperature'],
#                                    id='temp-inv-select', className='mt-2', inline=True, switch=True)
# historical_controls = dbc.Card(
#     dbc.CardBody([html.Div([dbc.Label("Select Unit"), historical_unit_dropdown]),
#                   html.Div([dbc.Label("Set Date Range")], className='pt-2'),
#                   html.Div([date_range_picker], className='d-flex justify-content-center')
#                   ]),
#     className='d-block')
historical_controls = dbc.Row([
    dbc.Col([dbc.Card([
        dbc.CardHeader("Select Unit"),
        dbc.CardBody([historical_unit_dropdown])
    ])], width=6),
    dbc.Col([dbc.Card([
        dbc.CardHeader("Set Date Range"),
        dbc.CardBody([date_range_picker])
    ])], width=6)
])
# Output
historical_temp = dbc.Spinner(dcc.Graph(id='temperature-historical', style={"height": '700px'}, className='d-block'),
                              color='info')
historical_inventory = dbc.Spinner(dcc.Graph(id='inventory-historical', style={"height": '700px'}, className='d-block'),
                                   color='info')
# historical_overlay = dbc.Spinner(dcc.Graph(id='historical-overlay', style={"height": '700px'}, className='d-block'),
#                                  color='info')
historical_graph = dbc.Spinner(dcc.Graph(id='historical-graph', style={"height": '700px'}, className='d-block'),
                               color='info')
historical_graphs_div = html.Div(historical_graph, id='historical-graphs', style={"height": '100%'})
temp_table = dbc.Card([
    dbc.CardHeader("Data Table"),
    dbc.CardBody(dash_table.DataTable(id='temp-table',
                                      data=pd.DataFrame().to_dict('records'),
                                      columns=[{"name": i, "id": i} for i in temp_df.columns],
                                      filter_action="native",
                                      hidden_columns=['Unix Timestamp', 'UnitID'],
                                      export_format='csv',
                                      sort_action="native",
                                      sort_mode="multi",
                                      page_action="native",
                                      page_current=0,
                                      page_size=100,
                                      style_data={
                                          'whiteSpace': 'normal',
                                          'height': 'auto',
                                      },
                                      ))
])
# Tab Layout
historical_tab = dbc.Tab([historical_controls, historical_graphs_div, temp_table, html.Div(style={"height": "80px"})], label='Historical', style={"height": '100%'})

"""
** Overview Tab
"""
# Controls
overview_mode_select = dbc.Card(dbc.CardBody(dbc.RadioItems(
    id="overview-mode",
    className="btn-group",
    inputClassName="btn-check",
    labelClassName="btn btn-outline-primary",
    labelCheckedClassName="active",
    options=[
        {"label": "Vaccine Summary", "value": 2},
        {"label": "Temperature Summary", "value": 3},
        {"label": "Unit View", "value": 1}
    ],
    value=2,
)))
# Graphs
overview_graph = dbc.Spinner(dcc.Graph(id='overview', style={'height': '10000px'}, className='d-block'), color='info')
# Tab Layout
overview_tab = dbc.Tab([overview_mode_select, overview_graph], label='Current', style={"height": '100%'})

"""
*** INITIALIZE DASH
"""
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server
app.title = 'PHL Vaccine Monitoring'

"""
*** APP LAYOUT
"""
app.layout = (
    html.Div([
        header,
        dbc.Tabs([
            # overview_tab,
            historical_tab,
            snapshots_tab
        ], id='tabs'),
        html.Pre(id='last-refreshed',
                 style={'position': 'fixed', 'bottom': '5px', 'left': '20px', 'z-index': '1000',
                        'font-size': '15px', 'width': 'auto', 'font-color': 'red', }, className='font-monospace'),
        dbc.Button('Refresh Data', color='info', n_clicks=0, id='refresh-data',
                   style={'position': 'fixed', 'bottom': '20px', 'right': '20px', 'z-index': '1000'}),
    ], style={'height': '100%'})
)

"""
*** CALLBACKS
"""

"""
* General
"""


# Refresh button
@app.callback(
    Output('last-refreshed', 'children'),
    [Input('refresh-data', 'n_clicks')]
)
def refresh(nclicks):
    if nclicks >= 0:
        check_cache()
    else:
        check_cache()
    next_time = last_refreshed + datetime.timedelta(minutes=10)
    return (f'Data last accessed: {last_refreshed.strftime("%b %d, %Y %I:%M:%S%p")}\n'
            f'Next refresh available at {next_time.strftime("%-I:%M%p")}')


# Callback to mirror UnitID updates
@app.callback(
    Output("unitID-dropdown-historical", "value"),
    Output("unitID-dropdown", "value"),
    Input("unitID-dropdown-historical", "value"),
    Input("unitID-dropdown", "value"),
)
def mirrorUnitID(historical_input, snapshot_input):
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    value = historical_input if trigger_id == "unitID-dropdown-historical" else snapshot_input
    return value, value


"""
* Overview Callbacks
"""


@app.callback(
    Output('overview', 'figure'),
    Output('overview', 'style'),
    Input('overview-mode', 'value')
)
def load_overview(viewmode):
    inventory_df = calculate_inventory(inventory_df=lots_df, transactions_df=transactions_df)
    current_inventory = inventory_df.loc[pd.to_numeric(inventory_df['Inventory']) > 0]
    if viewmode == 1:  # display unit view
        unit_g = make_subplots(rows=len(units), cols=2, column_widths=[0.8, 0.2],
                               row_titles=[unit['Name'] for unit in units])
        for i in range(len(units)):
            filtered = current_inventory.loc[current_inventory['Storage Location'] == units[i]['UnitID']]
            custom_data = [
                f'<b>{lot_num}<br>(Exp: {exp_date})</b><br>{src} / {icare}'
                for lot_num, exp_date, src, icare in zip(filtered['Lot #'],
                                                         filtered['Expiration Date'],
                                                         filtered['Source'],
                                                         filtered['I-CARE PIN'])
            ]
            unit_g.add_trace(trace=go.Bar(x=filtered['Vaccine'], y=filtered['Inventory'], showlegend=False,
                                          marker_color=px.colors.qualitative.Prism, customdata=custom_data,
                                          hovertemplate='%{customdata}<br>Inventory: %{y}<extra></extra>'),
                             row=i + 1, col=1)

            mintemp = units[i]['Minimum Temperature (C)']
            unit_g.add_hline(y=mintemp, line_dash='dash', line_color='red', line_width=0.5, row=i + 1, col=2)
            maxtemp = units[i]['Maximum Temperature (C)']
            unit_g.add_hline(y=maxtemp, line_dash='dash', line_color='red', line_width=0.5, row=i + 1, col=2)
            curtemp = units[i]['Last Recorded Temperature (C)']
            unit_g.add_trace(trace=go.Bar(x=['Temperature (°C)'], y=[curtemp], showlegend=False,
                                          marker_color=['seagreen' if mintemp < curtemp < maxtemp else 'crimson']),
                             row=i + 1, col=2)
            unit_g.update_yaxes(range=[int(mintemp - (maxtemp - mintemp) / 6), int(maxtemp + (maxtemp - mintemp) / 6)],
                                tickvals=
                                [mintemp, mintemp + (maxtemp - mintemp) / 3, maxtemp - (maxtemp - mintemp) / 3,
                                 maxtemp],
                                row=i + 1, col=2)
        return unit_g, {'height': '10000px'}
    elif viewmode == 2:
        fig = px.bar(current_inventory, x='Vaccine', y='Inventory',
                     color='Source', barmode='group',
                     color_discrete_sequence=px.colors.qualitative.Prism, hover_name='Lot #',
                     hover_data=['Expiration Date', 'Inventory'])
        return fig, {'height': '800px'}
    else:
        units_df = pd.DataFrame(units)
        colors = []
        for unit in units:
            if unit["Minimum Temperature (C)"] < unit['Last Recorded Temperature (C)'] < unit[
                'Maximum Temperature (C)']:
                colors.append('seagreen')
            else:
                colors.append('crimson')
        fig = px.bar(units_df, x='Name', y='Last Recorded Temperature (C)', facet_row='Storage Temperature',
                     color=colors, color_discrete_sequence=px.colors.qualitative.Prism,
                     hover_name='Last Recorded Temperature Timestamp')
        return fig, {'height': '2000px'}


"""
* Snapshots Callbacks
"""


# Function for combining date picker and time slider output
def combine_datetime(date_str: str, time_value: int) -> int:
    date_value = datetime.datetime.strptime(date_str[:10], '%Y-%m-%d')
    date_in_seconds = int(time.mktime(date_value.timetuple()))
    dt = date_in_seconds + time_value
    return dt


# Callback for updating time snapshot
@app.callback(
    Output('time-display', 'children'),
    [Input('date-picker', 'date'),
     Input('time-slider', 'value')]
)
def update_time(date_str, time_value):
    dt = combine_datetime(date_str, time_value)
    timestamp = datetime.datetime.fromtimestamp(dt).strftime("%b %d, %Y %I:%M%p")
    return html.H3(f"{timestamp}")


# Callback for updating temperature snapshot
@app.callback(
    Output('temp-display', 'children'),
    [Input('unitID-dropdown', 'value'),
     Input('date-picker', 'date'),
     Input('time-slider', 'value')]
)
def update_temp(unitID, date_str, time_value):
    dt = combine_datetime(date_str, time_value)
    filtered_df = temp_df.loc[(temp_df['UnitID'] == unitID) & (dt - temp_df['Unix Timestamp'] < 899)]

    if not filtered_df.empty:
        temp = filtered_df['Temperature'].iloc[0]
        return html.H3(f"Temperature: {temp}°C")
    else:
        return html.H3("No temperature data available")


# Callback for updating temp snapshot graph
@app.callback(
    Output('temperature-snapshot', 'figure'),
    [Input('unitID-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_daily_temp_graph(unitID, date_str):
    filtered_temps = temp_df.loc[(temp_df['UnitID'] == unitID)]
    start_dt = combine_datetime(date_str, 0)
    end_dt = combine_datetime(date_str, 86399)
    filtered_temps = filtered_temps.loc[(temp_df['Unix Timestamp'] > start_dt) & (temp_df['Unix Timestamp'] < end_dt)]
    filtered_temps['Temperature'] = pd.to_numeric(filtered_temps['Temperature'])

    fig = px.line(filtered_temps, x='Timestamp', y='Temperature', title='Temperature',
                  labels=dict(x='Time', y='Temperature (°C)'),
                  color_discrete_sequence=['black'], markers=True)
    unit_df = pd.DataFrame(units)
    mintemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Minimum Temperature (C)'].iloc[0]
    fig.add_hline(y=mintemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
    maxtemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Maximum Temperature (C)'].iloc[0]
    fig.add_hline(y=maxtemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
    fig.update_layout(
        yaxis=dict(
            tickvals=
            [mintemp, mintemp + (maxtemp - mintemp) / 3, maxtemp - (maxtemp - mintemp) / 3, maxtemp]))
    # range=[int(mintemp - (maxtemp - mintemp) / 6), int(maxtemp + (maxtemp - mintemp) / 6)],
    fig.update_layout(xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1h",
                     step="hour",
                     stepmode="backward"),
                dict(count=6,
                     label="6h",
                     step="hour",
                     stepmode="backward"),
                dict(count=12,
                     label="12h",
                     step="hour",
                     stepmode="backward"),
                dict(count=18,
                     label="18h",
                     step="hour",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ), hovermode='x unified')
    return fig


# Callback for updaating inv snapshot table
@app.callback(
    Output('inventory-table', 'data'),
    Output('inventory-table', 'columns'),
    [Input('unitID-dropdown', 'value'),
     Input('date-picker', 'date'),
     Input('time-slider', 'value')]
)
def update_inventory_table(unitID, date_str, time_value):
    dt = datetime.datetime.fromtimestamp(combine_datetime(date_str, time_value))
    inventory_df = calculate_inventory(dt, unitID, inventory_df=lots_df, transactions_df=transactions_df)
    snapshot_df = inventory_df.loc[pd.to_numeric(inventory_df['Inventory']) > 0]
    return snapshot_df.to_dict('records'), [{"name": i, "id": i} for i in snapshot_df.columns]


# Callback for updating trans snapshot table
@app.callback(
    Output('trans-table', 'data'),
    Output('trans-table', 'columns'),
    [Input('unitID-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_inventory_graph(unitID, date_str):
    dt = datetime.datetime.strptime(date_str[:10], '%Y-%m-%d').date()
    transactions_df['dt'] = pd.to_datetime(transactions_df['Timestamp']).dt.date
    filtered_trans = transactions_df.loc[(transactions_df['Current Location'] == unitID) | (transactions_df['Destination'] == unitID)]
    filtered_trans = filtered_trans[filtered_trans['dt'] == dt]
    return filtered_trans.to_dict('records'), [{"name": i, "id": i} for i in filtered_trans.columns]


# Callback for updating historical overlay graph
@app.callback(
    Output('historical-graphs', 'children'),
    Output('historical-graph', 'figure'),
    Output('temp-table', 'data'),
    Input('unitID-dropdown-historical', 'value'),
    Input('historical-date-picker', 'start_date'),
    Input('historical-date-picker', 'end_date'),
    Input('last-refreshed', 'children')
)
def update_historical(unitID, start_dt, end_dt, value):
    filtered_temps = temp_df.loc[(temp_df['UnitID'] == unitID)]
    filtered_temps['Temperature'] = pd.to_numeric(filtered_temps['Temperature'])
    start_dt = combine_datetime(start_dt, 0)
    end_dt = combine_datetime(end_dt, 86399)
    filtered_temps = filtered_temps.loc[(temp_df['Unix Timestamp'] > start_dt) & (temp_df['Unix Timestamp'] < end_dt)]
    fig = px.line(filtered_temps, x='Timestamp', y='Temperature', title='Temperature',
                  labels=dict(x='Time', y='Temperature (°C)'),
                  color_discrete_sequence=['black'], markers=True)
    unit_df = pd.DataFrame(units)
    mintemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Minimum Temperature (C)'].iloc[0]
    fig.add_hline(y=mintemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
    maxtemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Maximum Temperature (C)'].iloc[0]
    fig.add_hline(y=maxtemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
    fig.update_layout(
        yaxis=dict(
            tickvals=
            [mintemp, mintemp + (maxtemp - mintemp) / 3, maxtemp - (maxtemp - mintemp) / 3, maxtemp]))
    # range=[int(mintemp - (maxtemp - mintemp) / 6), int(maxtemp + (maxtemp - mintemp) / 6)],
    fig.update_layout(xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1h",
                     step="hour",
                     stepmode="backward"),
                dict(count=12,
                     label="12h",
                     step="hour",
                     stepmode="backward"),
                dict(count=1,
                     label="1d",
                     step="day",
                     stepmode="backward"),
                dict(count=7,
                     label="7d",
                     step="day",
                     stepmode="backward"),
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="MTD",
                     step="month",
                     stepmode="todate"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    ), hovermode='x unified')
    return historical_graph, fig, filtered_temps.to_dict('records')


# Callback for updating historical overlay graph
# @app.callback(
#     Output('historical-graphs', 'children'),
#     Output('historical-graph', 'figure'),
#     Input('unitID-dropdown-historical', 'value'),
#     Input('temp-inv-select', 'value'),
#     Input('last-refreshed', 'children')
# )
# def update_historical(unitID, overlay_selected, value):
#     print(overlay_selected)
#     if len(overlay_selected) == 0:
#         return [dbc.Alert('Select a metric to display above.', color='warning'), historical_graph], px.bar()
#
#     filtered_temps = temp_df.loc[(temp_df['UnitID'] == unitID)]
#     filtered_temps['Temperature'] = pd.to_numeric(filtered_temps['Temperature'])
#
#     if overlay_selected == ['Temperature']:
#         fig = px.line(filtered_temps, x='Timestamp', y='Temperature', title='Temperature',
#                       labels=dict(x='Time', y='Temperature (°C)'),
#                       color_discrete_sequence=['black'])
#         unit_df = pd.DataFrame(units)
#         mintemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Minimum Temperature (C)'].iloc[0]
#         fig.add_hline(y=mintemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
#         maxtemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Maximum Temperature (C)'].iloc[0]
#         fig.add_hline(y=maxtemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
#         fig.update_layout(
#             yaxis=dict(range=[int(mintemp - (maxtemp - mintemp) / 6), int(maxtemp + (maxtemp - mintemp) / 6)],
#                        tickvals=
#                        [mintemp, mintemp + (maxtemp - mintemp) / 3, maxtemp - (maxtemp - mintemp) / 3, maxtemp]))
#         fig.update_layout(xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1,
#                          label="1h",
#                          step="hour",
#                          stepmode="backward"),
#                     dict(count=12,
#                          label="12h",
#                          step="hour",
#                          stepmode="backward"),
#                     dict(count=1,
#                          label="1d",
#                          step="day",
#                          stepmode="backward"),
#                     dict(count=7,
#                          label="7d",
#                          step="day",
#                          stepmode="backward"),
#                     dict(count=1,
#                          label="1m",
#                          step="month",
#                          stepmode="backward"),
#                     dict(count=1,
#                          label="YTD",
#                          step="year",
#                          stepmode="todate"),
#                     dict(step="all")
#                 ])
#             ),
#             rangeslider=dict(
#                 visible=True
#             ),
#             type="date"
#         ), hovermode='x unified')
#         return historical_graph, fig
#
#     filtered_df = agg_df.loc[(agg_df['UnitID'] == unitID)]
#     expanded_df = expand_rows(filtered_df)
#     expanded_df['BatchID'] = expanded_df['BatchID'].fillna('')
#     expanded_df['ItemID'] = expanded_df['LotID'] + ':' + expanded_df['BatchID']
#     vaccine_list = expanded_df['Vaccine'].unique()
#     vaccine_color_key = {vaccine_list[i]: px.colors.qualitative.Prism[i % 11] for i in range(len(vaccine_list))}
#
#     if not expanded_df.empty:
#         if overlay_selected == ['Temperature', 'Inventory']:
#             fig = make_subplots(specs=[[{"secondary_y": True}]])
#             for vaccine_type in expanded_df['Vaccine'].unique():
#                 filtered_vx_df = expanded_df.loc[expanded_df['Vaccine'] == vaccine_type]
#                 add_to_legend = True
#                 for lot in filtered_vx_df['ItemID'].unique():
#                     historical_lot_data = filtered_vx_df.loc[filtered_vx_df['ItemID'] == lot]
#                     custom_data = [
#                         f'<b>{lot_num} (Exp: {exp_date})</b> {src} {icare}'
#                         for lot_num, exp_date, src, icare in zip(historical_lot_data['Lot #'],
#                                                                  historical_lot_data['Expiration Date'],
#                                                                  historical_lot_data['Source'],
#                                                                  historical_lot_data['I-CARE PIN'])
#                     ]
#                     fig.add_trace(
#                         go.Scatter(x=historical_lot_data['Timestamp'],
#                                    y=historical_lot_data['Inventory'],
#                                    mode='lines',
#                                    name=vaccine_type,
#                                    showlegend=add_to_legend,
#                                    stackgroup=1,
#                                    line=dict(width=2, color=vaccine_color_key[vaccine_type]),
#                                    customdata=custom_data,
#                                    hovertemplate='%{y} %{customdata}',
#                                    hoveron='points+fills'),
#                         secondary_y=True
#                     )
#                     add_to_legend = False
#             fig.add_trace(go.Scatter(x=filtered_temps['Timestamp'],
#                                      y=filtered_temps['Temperature'],
#                                      name='Temperature (°C)',
#                                      mode='lines',
#                                      showlegend=False,
#                                      line=dict(width=3, color='black')),
#                           secondary_y=False)
#             unit_df = pd.DataFrame(units)
#             mintemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Minimum Temperature (C)'].iloc[0]
#             fig.add_hline(y=mintemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
#             maxtemp = unit_df.loc[unit_df['UnitID'] == unitID, 'Maximum Temperature (C)'].iloc[0]
#             fig.add_hline(y=maxtemp, line_dash='dash', line_color='red', line_width=0.5, secondary_y=False)
#             fig.update_yaxes(range=[int(mintemp - (maxtemp - mintemp) / 6), int(maxtemp + (maxtemp - mintemp) / 6)],
#                              tickvals=[mintemp,
#                                        mintemp + (maxtemp - mintemp) / 3,
#                                        maxtemp - (maxtemp - mintemp) / 3,
#                                        maxtemp],
#                              title_text="Temperature (°C)",
#                              secondary_y=False)
#             fig.update_yaxes(title_text='# Vials', secondary_y=True)
#             fig.update_layout(title="Temperature + Inventory",
#                               hovermode='x unified',
#                               legend_title="Vaccine",
#                               legend_entrywidthmode='fraction',
#                               legend_entrywidth=3,
#                               xaxis=dict(title='Select time range',
#                                          rangeselector=dict(
#                                              buttons=list([
#                                                  dict(count=1,
#                                                       label="1h",
#                                                       step="hour",
#                                                       stepmode="backward"),
#                                                  dict(count=12,
#                                                       label="12h",
#                                                       step="hour",
#                                                       stepmode="backward"),
#                                                  dict(count=1,
#                                                       label="1d",
#                                                       step="day",
#                                                       stepmode="backward"),
#                                                  dict(count=7,
#                                                       label="7d",
#                                                       step="day",
#                                                       stepmode="backward"),
#                                                  dict(count=1,
#                                                       label="1m",
#                                                       step="month",
#                                                       stepmode="backward"),
#                                                  dict(count=1,
#                                                       label="YTD",
#                                                       step="year",
#                                                       stepmode="todate"),
#                                                  dict(step="all")
#                                              ])
#                                          ),
#                                          rangeslider=dict(
#                                              visible=True
#                                          ),
#                                          type="date"
#                                          ))
#             return historical_graph, fig
#         elif overlay_selected == ['Inventory']:
#             fig = px.area(expanded_df,
#                           x='Timestamp', y='Inventory', line_group='ItemID',
#                           color='Vaccine', color_discrete_sequence=px.colors.qualitative.Prism,
#                           labels=dict(x='Time', y='# Vials'),
#                           hover_name='Lot #', hover_data=['Expiration Date', 'Source', 'I-CARE PIN', 'Inventory'],
#                           title='Inventory')
#             fig.update_layout(
#                 xaxis=dict(
#                     rangeselector=dict(
#                         buttons=list([
#                             dict(count=1,
#                                  label="1h",
#                                  step="hour",
#                                  stepmode="backward"),
#                             dict(count=12,
#                                  label="12h",
#                                  step="hour",
#                                  stepmode="backward"),
#                             dict(count=1,
#                                  label="1d",
#                                  step="day",
#                                  stepmode="backward"),
#                             dict(count=7,
#                                  label="7d",
#                                  step="day",
#                                  stepmode="backward"),
#                             dict(count=1,
#                                  label="1m",
#                                  step="month",
#                                  stepmode="backward"),
#                             dict(count=1,
#                                  label="YTD",
#                                  step="year",
#                                  stepmode="todate"),
#                             dict(step="all")
#                         ])
#                     ),
#                     rangeslider=dict(
#                         visible=True
#                     ),
#                     type="date"
#                 ),
#                 legend_entrywidthmode='fraction',
#                 legend_entrywidth=5
#             )
#             return historical_graph, fig
#     else:
#         return historical_graph, px.area(title='No data available')


"""
=========== START APP ============
"""
if __name__ == '__main__':
    print('\nStarting....')
    app.run_server()
