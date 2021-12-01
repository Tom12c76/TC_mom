import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid.shared import GridUpdateMode

st.set_page_config(page_title="Netflix Shows", layout="wide")
st.title("Netlix shows analysis")

shows = pd.read_csv("netflix_titles.csv")

# Conditional formatting

cellsytle_jscode = JsCode(
    """
function(params) {
    if (params.value.includes('United States')) {
        return {
            'color': 'white',
            'backgroundColor': 'darkred'
        }
    } else {
        return {
            'color': 'black',
            'backgroundColor': 'white'
        }
    }
};
"""
)

# add this
gb = GridOptionsBuilder.from_dataframe(shows)

gb.configure_pagination()
gb.configure_side_bar()
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)

gb.configure_column("country", cellStyle=cellsytle_jscode)

gridOptions = gb.build()

data = AgGrid(shows,
              gridOptions=gridOptions,
              enable_enterprise_modules=True,
              allow_unsafe_jscode=True,
              update_mode=GridUpdateMode.SELECTION_CHANGED)

# st.write(data)