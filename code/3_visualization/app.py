import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import pickle


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Set variables
status_dir = "../../data/0_statuses/"
status_files = pd.Series(os.listdir(status_dir))
status_files_full = pd.Series(status_dir + f for f in status_files)
sources = status_files.str.split("_").apply(lambda x: x[0])
feat_dir = "../../data/1_features/"
regr_dir = "../../data/2_models/regression/"

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
        )
        #,
    

#    dcc.Graph(
#        id='example-graph',
#        figure=
#    )
])


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
    return [html.H6(["Effect of entity ", html.B(entity_lbl), " on ", html.B(source_val)]), 
            html.Div(["Total reax: {}".format(coefs[0].round(2)), html.Br(),
                    "❤️ Loves: {}%".format((coefs[1] * 100).round(2)), html.Br(),
                    "😯 Wows: {}%".format((coefs[2] * 100).round(2)), html.Br(), 
                    "😂 Hahas: {}%".format((coefs[3] * 100).round(2)), html.Br(),
                    "😢 Sads: {}%".format((coefs[4] * 100).round(2)), html.Br(),
                    "😡 Angrys: {}%".format((coefs[5] * 100).round(2))])]

    
if __name__ == '__main__':
    app.run_server(debug=True)