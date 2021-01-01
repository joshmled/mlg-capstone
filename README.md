# Entity-Based Prediction of Facebook Reactions
## MLG Capstone Project

For this project, I took the chance to play around with some common ML approaches and their Python implementations. I built a dashboard aimed at addressing a social media- and PR-related use case. Business questions and outline of approach are shown in [this excerpt](https://github.com/joshmled/mlg-capstone/blob/master/MLG_Pres.pdf) of my capstone presentation. 

### Repo contents
#### Data
* `0_statuses` contains 21 input CSV files, one per media source, with Facebook status text and user reaction totals -- one line per status. I intended to use the Facebook API, but due to COVID-related approval delays, I used [datasets](github.com/minimaxir/interactive-facebook-reactions) with info from Feb-June 2016 for a proof of concept (many thanks to [Max Woolf](github.com/minimaxir)).
* Other subdirectories contain Pickle outputs from scripts in `code`, and are meant to be overwritten when re-run for different input data.
#### Code
Subdirectories of the `code` directory are labeled in order of use, with an additional folder for exploration scripts:
* `0_data_load` contains script to read in and preprocess status input CSVs (and can later be supplemented with scripts calling the API).
* `1_feature_engineering` contains scripts to engineer named entity dummies and sentiment score measures as input features, as well as FB reaction targets.
* `2_modeling` runs the central regression model, using engineered features and targets.
* `3_visualization` contains a Plotly Dash app that displays model results in an interactive dashboard.
