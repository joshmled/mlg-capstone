import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import pickle
import numpy as np


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Set variables
status_dir = "../../data/0_statuses/"
status_files = pd.Series(os.listdir(status_dir))
status_files_full = pd.Series(status_dir + f for f in status_files)
sources = status_files.str.split("_").apply(lambda x: x[0])
feat_dir = "../../data/1_features/"
regr_dir = "../../data/2_models/regression/"

def arrow_icon(val):
    if val == 0:
        return ""
    elif val > 0:
        return html.B(u"\u2191", style = {'font-size': '40px', 'color':'green'})
    else:
        return html.B(u"\u2193", style = {'font-size': '40px', 'color':'red'})


app.layout = html.Div(children=[
    html.H1(children='Facebook Reaction Tracker'),

    html.Div(children=['''
        A tool to determine the effect that your entity has on volume and sentiment of reactions to FB-shared media.   
    ''', html.Br(), '''
    Get started by searching for your entity here:
    ''', html.Br(), dcc.Input(
        id = "entity-input", type = "text", placeholder = "Search here!", debounce = True), dcc.Dropdown(
        id = 'source-selection', options = [{'label': i, 'value': i} for i in sources
            ], value = sources[0]), html.Hr(), dcc.Dropdown(id = 'entity-choice'), html.Div(
            id="search-summary")]
        )])


@app.callback(
    Output("entity-choice", "options"), 
    [Input("entity-input", "value"), Input("source-selection", "value")],)
def choose_entities(search_val, source_val):
    file_coef = open(regr_dir + source_val + '.coefs', 'rb')
    coef_names = pickle.load(file_coef)['names']
    file_coef.close()
    if search_val is None:
        ents = [(i, coef_names[i]) for i in range(len(coef_names))]
    else:
        ents = [(i, coef_names[i]) for i in range(len(coef_names)) if search_val in coef_names[i]]
    return [{'label': i[1], 'value': i[0]} for i in ents]

@app.callback(
    Output('entity-choice', 'value'),
    [Input('entity-choice', 'options')])
def set_entity_value(available_options):
    return available_options[0]['value']

@app.callback(
    Output('search-summary', 'children'), [
    Input('entity-choice', 'options'), Input('entity-choice', 'value'), Input('source-selection', 'value')],)
def return_coefs(entity_options, entity_val, source_val):
    entity_lbl = [opt['label'] for opt in entity_options if opt['value'] == entity_val][0]
    file_coef = open(regr_dir + source_val + '.coefs', 'rb')
    coef_df = pickle.load(file_coef)['coef']
    file_coef.close()
    coefs = coef_df[:, entity_val]
    reax = ["❤️ Loves", "😯 Wows", "😂 Hahas", "😢 Sads", "😡 Angrys"]
    fig = px.bar(pd.DataFrame({'Reaction': reax, 'Effect': coefs[1:6]}), 
        x = "Effect", y = "Reaction", orientation = "h")
    largest = max(abs(coefs[1:6]) * 2)
    annotations = []
    for reac, eff in zip(reax, coefs[1:6]):
        annotations.append(dict(xref='x', yref = 'y', x = 1.5 * eff, y = reac, xanchor = 'right',
            text=reac + ': ' + str((eff * 100).round(2)) + '%', showarrow=False,
                font=dict(family='Arial', size=14, color='rgb(248, 248, 255)')))
    fig.update_layout(plot_bgcolor = '#C0C0C0', 
        xaxis = dict(showgrid = False, showline = False, zeroline = True, 
        showticklabels = False, range = [-largest, largest]),
        yaxis = dict(showgrid=False, showline=False, showticklabels=False, zeroline=False),
        annotations = annotations)
    return [html.H6(["Effect of entity ", html.B(entity_lbl), " on ", html.B(source_val)]), 
            html.Div(["Total reactions: ", html.B(coefs[0].round(0)), arrow_icon(coefs[0]), html.Br(), 
                dcc.Graph(id = 'reax-bars', figure = fig)])]

    
if __name__ == '__main__':
    app.run_server(debug=True)