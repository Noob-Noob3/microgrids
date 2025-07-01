import streamlit as st
import matplotlib.pyplot as plt
import pypsa
import pandas as pd
import numpy as np
from plots_black import *
# Load results (adjust path as needed)
import os
# plt.rcdefaults()  
# # from newa import plot_timeframe  # Import your plotting function
# plt.rcParams.update({
#     "axes.titlesize": 18,      # Title font size
#     "axes.labelsize": 14,      # X/Y label font size
#     "xtick.labelsize": 12,     # X tick label font size
#     "ytick.labelsize": 12,     # Y tick label font size
#     "legend.fontsize": 12,     # Legend font size
# })


st.set_page_config(page_title="Microgrid Dashboard", layout="wide")
st.title("Microgrid Scenario Explorer")

@st.cache_resource
def load_network(path):
    return pypsa.Network(path)






# --- Sidebar ---
st.sidebar.header("Scenario Selection")

options = ['referenz', 0, 1, 2, 3, 4]
scenario = st.sidebar.select_slider("Scenario (number of home office days)", options=options, value='referenz')
path = 'results'



results_static = {k: load_network(f"{path}/HO_{k}.nc") for k in range(5)}
results_static['referenz'] = load_network(f"{path}/referenz.nc")

results_dyn = {k: load_network(f"{path}/dynamic_stock_HO_{k}.nc") for k in range(5)}
results_dyn['referenz'] = load_network(f"{path}/dynamic_stock_referenz.nc")

results_dyn24 = {k: load_network(f"{path}/dynamic_stock_2024_HO_{k}.nc") for k in range(5)}
results_dyn24['referenz'] = load_network(f"{path}/dynamic_stock_2024_referenz.nc")

results_ewa = {k: load_network(f"{path}/dynamic_ewa_HO_{k}.nc") for k in range(5)}
results_ewa['referenz'] = load_network(f"{path}/dynamic_ewa_referenz.nc")

r=dict(
    static=results_static,
    stock_2019=results_dyn,
    stock_2024=results_dyn24,
    HT_NT=results_ewa
) 
var = st.sidebar.select_slider("Preisvariante", options=['static','stock_2019', 'stock_2024', 'HT_NT'], value='static')
var2 = st.sidebar.select_slider("Vergleichsvariante", options=list(r.keys()), value='static')


results = r[var]
results2 = r[var2]
n = results[scenario]
n2 = results2[scenario]

dynamic_price = True if var in ['stock_2019', 'stock_2024', 'HT_NT'] else False
dynamic_price2 = True if var2 in ['stock_2019', 'stock_2024', 'HT_NT'] else False

options = ['custom', 'Year', 'Summer','Summer Day','Winter Day','Summer Week','Winter Week']+['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
static_timeframe = st.sidebar.select_slider("Time frame to plot", options=options, value='custom')

time_index = n.generators_t.p.index
time_index_py = time_index.to_pydatetime()
start_default = time_index_py[0]
end_default = time_index_py[min(168, len(time_index_py)-1)]

# time_range = st.sidebar.slider(
#     "Select custom time window",
#     min_value=time_index_py[0],
#     max_value=time_index_py[-1],
#     value=(start_default, end_default),
#     format="YYYY-MM-DD HH:mm"
# )
if static_timeframe == 'custom':


    start_str = st.sidebar.text_input(
    "Start (YYYY-MM-DD HH:MM)", 
    value=start_default.strftime("%Y-%m-%d %H:%M")
    )
    end_str = st.sidebar.text_input(
    "End (YYYY-MM-DD HH:MM)", 
    value=end_default.strftime("%Y-%m-%d %H:%M")
    )
    try:
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
    except Exception as e:
        st.sidebar.error(f"Invalid date format: {e}")
        start, end = start_default, end_default
    # start, end = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
else: 
    if static_timeframe == 'Year':
        start = pd.to_datetime("2019-01-01 00:00:00")
        end = pd.to_datetime("2019-12-31 23:59:59")
    if static_timeframe == 'Summer Day':
        start = pd.to_datetime("2019-08-06 00:00:00")
        end = pd.to_datetime("2019-08-09 00:00:00")
    if static_timeframe == 'Summer Week':
        start = pd.to_datetime("2019-07-01 00:00:00")
        end = pd.to_datetime("2019-07-08 00:00:00")
    if static_timeframe == 'Winter Day':
        start = pd.to_datetime("2019-01-08 00:00:00")
        end = pd.to_datetime("2019-01-09 23:59:59")
    if static_timeframe == 'Winter Week':
        start = pd.to_datetime("2019-01-08 00:00:00")
        end = pd.to_datetime("2019-01-15 23:59:59")
    elif static_timeframe == 'Summer':
        start = pd.to_datetime("2019-05-01 00:00:00")
        end = pd.to_datetime("2019-10-01 00:00:00")
    elif static_timeframe == 'jan':
        start = pd.to_datetime("2019-01-01 00:00:00")
        end = pd.to_datetime("2019-02-01 00:00:00")
    elif static_timeframe == 'feb': 
        start = pd.to_datetime("2019-02-01 00:00:00")
        end = pd.to_datetime("2019-03-01 00:00:00")
    elif static_timeframe == 'mar':
        start = pd.to_datetime("2019-03-01 00:00:00")
        end = pd.to_datetime("2019-04-01 00:00:00")
    elif static_timeframe == 'apr':
        start = pd.to_datetime("2019-04-01 00:00:00")
        end = pd.to_datetime("2019-05-01 00:00:00")
    elif static_timeframe == 'may':
        start = pd.to_datetime("2019-05-01 00:00:00")
        end = pd.to_datetime("2019-06-01 00:00:00")
    elif static_timeframe == 'jun':
        start = pd.to_datetime("2019-06-01 00:00:00")
        end = pd.to_datetime("2019-07-01 00:00:00")
    elif static_timeframe == 'jul':
        start = pd.to_datetime("2019-07-01 00:00:00")
        end = pd.to_datetime("2019-08-01 00:00:00")
    elif static_timeframe == 'aug':
        start = pd.to_datetime("2019-08-01 00:00:00")
        end = pd.to_datetime("2019-09-01 00:00:00")
    elif static_timeframe == 'sep':
        start = pd.to_datetime("2019-09-01 00:00:00")
        end = pd.to_datetime("2019-10-01 00:00:00")
    elif static_timeframe == 'oct':
        start = pd.to_datetime("2019-10-01 00:00:00")
        end = pd.to_datetime("2019-11-01 00:00:00")
    elif static_timeframe == 'nov':
        start = pd.to_datetime("2019-11-01 00:00:00")
        end = pd.to_datetime("2019-12-01 00:00:00")
    elif static_timeframe == 'dec':
        start = pd.to_datetime("2019-12-01 00:00:00")
        end = pd.to_datetime("2020-01-01 00:00:00")
    
    # elif static_timeframe == 'Winter':
    #     start = pd.to_datetime("2019-12-01 00:00:00")
    #     end = pd.to_datetime("2020-02-29 23:59:59")




tabs = st.tabs(["Network Overview", "Time Series", "Correlation","Peak Distributions","Bar Charts","Battery Cycles","Duration Curves","Model Source Code"])
# --- Main Plots ---

with tabs[0]:
    col1, col2 = st.columns(2)
    # with st.expander("Network Plot"):
    with col1:
        fig = plot_network(n)
        st.pyplot(fig)
        fig = plot_energies()
        st.pyplot(fig)

    # with st.expander("Driving Pattern"):
    with col2:
        fig = driving_plot(1)
        st.pyplot(fig)
    # with st.expander("Input Data"):
        fig = plot_loads(n,start, end)
        st.pyplot(fig)
         

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{var}")

        fig = plot_timeframe(n, start, end,var)
        st.pyplot(fig)

        fig = plot_peak_shaving(n,start,end)
        st.pyplot(fig)
    with col2:
        st.subheader(f"{var2}")

        fig = plot_timeframe(n2, start, end,var2)
        st.pyplot(fig)
        fig = plot_peak_shaving(n2,start,end)
        st.pyplot(fig)

     
            

with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{var}")
        fig = plot_pv_vs_export(n, var, 'Correlation of PV and Export')
        st.pyplot(fig)
        fig=price_histogramm(Variante=var)
        st.pyplot(fig)
        # fig = plot_autarky_vs_pv_single(n, title=None)
        # st.pyplot(fig)

    with col2:
        st.subheader(f"{var2}")
        fig = plot_pv_vs_export(n2,var2,'Correlation of PV and Export')
        st.pyplot(fig)
        fig=price_histogramm(Variante=var2)
        st.pyplot(fig)

        # fig = plot_autarky_vs_pv_single(n, title=None)
        # st.pyplot(fig)

with tabs[3]:

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{var}")
        fig=plot_peak_distro(n,15, title="Peak Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader(f"{var2}")
        fig=plot_peak_distro(n2,15, title="Peak Distribution")
        st.pyplot(fig)


with tabs[4]:
    fig = plot_costs_dyn(results,results2,label1=var,label2=var2)
    st.pyplot(fig)
    fig = plot_optimal_battery_capacity_dyn(results,results2,label1=var,label2=var2)
    st.pyplot(fig)
    fig = plot_autarky_dyn(results,results2,label1=var,label2=var2)
    st.pyplot(fig)
    # col1, col2 = st.columns(2)
    # with col1:
    
    #     fig = plot_costs(results)
    #     st.pyplot(fig)

    #     fig = plot_optimal_battery_capacity(results)
    #     st.pyplot(fig)
        
    
    # with col2:

    #     fig = plot_autarky(results)
    #     st.pyplot(fig)



with tabs[5]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{var}")
        fig = plot_battery_cycles(results)
        st.pyplot(fig)
    with col2:
        st.subheader(f"{var2}")
        fig = plot_battery_cycles(results2)
        st.pyplot(fig)

with tabs[6]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{var}")
            fig = plot_duration_curve(n.generators_t.p['grid_buy'],title="Jahresganglinie Stromeinkauf")
            st.pyplot(fig)
            fig = plot_duration_curve(n.generators_t.p['grid_sell'],title="Jahresganglinie Stromexport")
            st.pyplot(fig)
            fig = plot_duration_curve(n.generators_t.p['pv_system'],title="Jahresganglinie PV")
            st.pyplot(fig)
            fig = plot_duration_curve(n.stores_t.p['battery'],title="Jahresganglinie Batterie")
            st.pyplot(fig)
        with col2:
            st.subheader(f"{var2}")
            fig = plot_duration_curve(n2.generators_t.p['grid_buy'],title="Jahresganglinie Stromeinkauf")
            st.pyplot(fig)
            fig = plot_duration_curve(n2.generators_t.p['grid_sell'],title="Jahresganglinie Stromexport")
            st.pyplot(fig)
            fig = plot_duration_curve(n2.generators_t.p['pv_system'],title="Jahresganglinie PV")
            st.pyplot(fig)
            fig = plot_duration_curve(n2.stores_t.p['battery'],title="Jahresganglinie Batterie")
            st.pyplot(fig)

with tabs[7]:
    with st.expander("Model Source Code"):
        with open("Model.py", "r", encoding="utf-8") as f:
            code = f.read()
        st.code(code, language="python")
    with st.expander("Optimization Soruce Code"):
        with open("optimize.py", "r", encoding="utf-8") as f:
            code = f.read()
        st.code(code, language="python")


st.info("Model and dashboard created by Otto Neef and Jacob Waschner as part of the course 'Modellierung von Microgrids' at HTWK Leipzig.")

