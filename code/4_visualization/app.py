import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(children=[
    html.H1(children='Facebook Reaction Tracker'),

    html.Div(children=['''
        A tool to determine the effect that your entity has on volume and sentiment of reactions to FB-shared media.   
    ''', html.Br(), '''
    Get started by searching for your entity here:
    ''', html.Br(), dcc.Input(
        id = "entity_input", type = "search", placeholder = "Search here!"), html.Div(
            id="out-all-types")])
        #,
    

#    dcc.Graph(
#        id='example-graph',
#        figure=
#    )
])

@app.callback(
    Output("out-all-types", "children"), [Input("entity_input", "value")],)
def entity_render(*vals):
    return " | ".join(vals)

    
if __name__ == '__main__':
    app.run_server(debug=True)