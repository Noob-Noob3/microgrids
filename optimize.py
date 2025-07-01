import pypsa 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy 

from Model import *

# statische Strompreise (EEG-Vergütung 2019)
data = input_data()
results=dict()

input = data.load_data(driving_days=5)
n_ref=house_network(data.snapshots,
                heat_load=input[0],
                el_load=input[1],
                dhw_load=input[2],
                pv_pu=input[3],
                driving=input[4],
                charger_p_max_pu=input[5],
                el_price=data.el_price,
                el_sell=data.el_sell,
                # pv_nom=0,
                bess_nom=0,
                # pv_extendable=False,
                battery_extendable=False,
                i=False)

n_ref.optimize(solver_name='gurobi')

results['referenz']=n_ref

for d in [5,4,3,2,1]:
    input = data.load_data(driving_days=d)
    n=house_network(data.snapshots,
                    heat_load=input[0],
                    el_load=input[1],
                    dhw_load=input[2],
                    pv_pu=input[3],
                    driving=input[4],
                    charger_p_max_pu=input[5],
                    el_price=data.el_price,
                    el_sell=data.el_sell,
                    bess_nom=0,
                    battery_extendable=True,
                    i=False)
    n.optimize(solver_name='gurobi')
    results[5-d]=n



results['referenz'].export_to_netcdf("results/referenz.nc")
results[0].export_to_netcdf("results/HO_0.nc")
results[1].export_to_netcdf("results/HO_1.nc")
results[2].export_to_netcdf("results/HO_2.nc")
results[3].export_to_netcdf("results/HO_3.nc")
results[4].export_to_netcdf("results/HO_4.nc");

# statische Strompreise (2024 Direktvermarktungspreise)
data = input_data(sell_price=2024)
results=dict()

input = data.load_data(driving_days=5)
n_ref=house_network(data.snapshots,
                heat_load=input[0],
                el_load=input[1],
                dhw_load=input[2],
                pv_pu=input[3],
                driving=input[4],
                charger_p_max_pu=input[5],
                el_price=data.el_price,
                el_sell=data.el_sell,
                # pv_nom=0,
                bess_nom=0,
                # pv_extendable=False,
                battery_extendable=False,
                i=False)

n_ref.optimize(solver_name='gurobi')

results['referenz']=n_ref

for d in [5,4,3,2,1]:
    input = data.load_data(driving_days=d)
    n=house_network(data.snapshots,
                    heat_load=input[0],
                    el_load=input[1],
                    dhw_load=input[2],
                    pv_pu=input[3],
                    driving=input[4],
                    charger_p_max_pu=input[5],
                    el_price=data.el_price,
                    el_sell=data.el_sell,
                    bess_nom=0,
                    battery_extendable=True,
                    i=False)
    n.optimize(solver_name='gurobi')
    results[5-d]=n



results['referenz'].export_to_netcdf("results/referenz.nc")
results[0].export_to_netcdf("results/HO_0_24.nc")
results[1].export_to_netcdf("results/HO_1_24.nc")
results[2].export_to_netcdf("results/HO_2_24.nc")
results[3].export_to_netcdf("results/HO_3_24.nc")
results[4].export_to_netcdf("results/HO_4_24.nc");


# # dynamischer Strompreis (Börse 2019)
data = input_data(dynamic_prices='stock',sell_price=2019)
results_dyn=dict()
input = data.load_data(driving_days=5)

n_ref=house_network(data.snapshots,
                heat_load=input[0],
                el_load=input[1],
                dhw_load=input[2],
                pv_pu=input[3],
                driving=input[4],
                charger_p_max_pu=input[5],
                el_price=data.el_price,
                el_sell=data.el_sell,
                # pv_nom=0,
                bess_nom=0,
                # pv_extendable=False,
                battery_extendable=False,
                i=False)

n_ref.optimize(solver_name='gurobi')

results_dyn['referenz']=n_ref

for d in [5,4,3,2,1]:
    input = data.load_data(driving_days=d)
    n=house_network(data.snapshots,
                    heat_load=input[0],
                    el_load=input[1],
                    dhw_load=input[2],
                    pv_pu=input[3],
                    driving=input[4],
                    charger_p_max_pu=input[5],
                    el_price=data.el_price,
                    el_sell=data.el_sell,
                    bess_nom=0,
                    battery_extendable=True,
                    i=False)
    n.optimize(solver_name='gurobi')

    results_dyn[5-d]=n


results_dyn['referenz'].export_to_netcdf("results/dynamic_stock_referenz.nc")
results_dyn[0].export_to_netcdf("results/dynamic_stock_HO_0.nc")
results_dyn[1].export_to_netcdf("results/dynamic_stock_HO_1.nc")
results_dyn[2].export_to_netcdf("results/dynamic_stock_HO_2.nc")
results_dyn[3].export_to_netcdf("results/dynamic_stock_HO_3.nc")
results_dyn[4].export_to_netcdf("results/dynamic_stock_HO_4.nc");


# # dynamischer Strompreis (Börse 2024)
data = input_data(dynamic_prices='stock',sell_price=2024)
results_dyn=dict()
input = data.load_data(driving_days=5)

n_ref=house_network(data.snapshots,
                heat_load=input[0],
                el_load=input[1],
                dhw_load=input[2],
                pv_pu=input[3],
                driving=input[4],
                charger_p_max_pu=input[5],
                el_price=data.el_price,
                el_sell=data.el_sell,
                # pv_nom=0,
                bess_nom=0,
                # pv_extendable=False,
                battery_extendable=False,
                i=False)

n_ref.optimize(solver_name='gurobi')

results_dyn['referenz']=n_ref

for d in [5,4,3,2,1]:
    input = data.load_data(driving_days=d)
    n=house_network(data.snapshots,
                    heat_load=input[0],
                    el_load=input[1],
                    dhw_load=input[2],
                    pv_pu=input[3],
                    driving=input[4],
                    charger_p_max_pu=input[5],
                    el_price=data.el_price,
                    el_sell=data.el_sell,
                    bess_nom=0,
                    battery_extendable=True,
                    i=False)
    n.optimize(solver_name='gurobi')

    results_dyn[5-d]=n


results_dyn['referenz'].export_to_netcdf("results/dynamic_stock_2024_referenz.nc")
results_dyn[0].export_to_netcdf("results/dynamic_stock_2024_HO_0.nc")
results_dyn[1].export_to_netcdf("results/dynamic_stock_2024_HO_1.nc")
results_dyn[2].export_to_netcdf("results/dynamic_stock_2024_HO_2.nc")
results_dyn[3].export_to_netcdf("results/dynamic_stock_2024_HO_3.nc")
results_dyn[4].export_to_netcdf("results/dynamic_stock_2024_HO_4.nc");


# # dynamischer Strompreis (HT;NT)
data = input_data(dynamic_prices='ewa',sell_price=2024)
results_dyn=dict()

input = data.load_data(driving_days=5)

n_ref=house_network(data.snapshots,
                heat_load=input[0],
                el_load=input[1],
                dhw_load=input[2],
                pv_pu=input[3],
                driving=input[4],
                charger_p_max_pu=input[5],
                el_price=data.el_price,
                el_sell=data.el_sell,
                # pv_nom=0,
                bess_nom=0,
                # pv_extendable=False,
                battery_extendable=False,
                i=False)

n_ref.optimize(solver_name='gurobi')

results_dyn['referenz']=n_ref

for d in [5,4,3,2,1]:
    input = data.load_data(driving_days=d)
    n=house_network(data.snapshots,
                    heat_load=input[0],
                    el_load=input[1],
                    dhw_load=input[2],
                    pv_pu=input[3],
                    driving=input[4],
                    charger_p_max_pu=input[5],
                    el_price=data.el_price,
                    el_sell=data.el_sell,
                    bess_nom=0,
                    battery_extendable=True,
                    i=False)
    n.optimize(solver_name='gurobi')

    results_dyn[5-d]=n


results_dyn['referenz'].export_to_netcdf("results/dynamic_ewa_referenz.nc")
results_dyn[0].export_to_netcdf("results/dynamic_ewa_HO_0.nc")
results_dyn[1].export_to_netcdf("results/dynamic_ewa_HO_1.nc")
results_dyn[2].export_to_netcdf("results/dynamic_ewa_HO_2.nc")
results_dyn[3].export_to_netcdf("results/dynamic_ewa_HO_3.nc")
results_dyn[4].export_to_netcdf("results/dynamic_ewa_HO_4.nc");





# # dynamische Netzentgelte
data = input_data(dynamic_prices=False,dynamic_grid_prices=True,sell_price=2024)
results_dyn=dict()


input = data.load_data(driving_days=5)

n_ref=house_network(data.snapshots,
                heat_load=input[0],
                el_load=input[1],
                dhw_load=input[2],
                pv_pu=input[3],
                driving=input[4],
                charger_p_max_pu=input[5],
                el_price=data.el_price,
                el_sell=data.el_sell,
                # pv_nom=0,
                bess_nom=0,
                # pv_extendable=False,
                battery_extendable=False,
                i=False)

n_ref.optimize(solver_name='gurobi')

results_dyn['referenz']=n_ref

for d in [5,4,3,2,1]:
    input = data.load_data(driving_days=d)
    n=house_network(data.snapshots,
                    heat_load=input[0],
                    el_load=input[1],
                    dhw_load=input[2],
                    pv_pu=input[3],
                    driving=input[4],
                    charger_p_max_pu=input[5],
                    el_price=data.el_price,
                    el_sell=data.el_sell,
                    bess_nom=0,
                    battery_extendable=True,
                    i=False)
    n.optimize(solver_name='gurobi')

    results_dyn[5-d]=n


results_dyn['referenz'].export_to_netcdf("results/dynamic_grid_referenz.nc")
results_dyn[0].export_to_netcdf("results/dynamic_grid_HO_0.nc")
results_dyn[1].export_to_netcdf("results/dynamic_grid_HO_1.nc")
results_dyn[2].export_to_netcdf("results/dynamic_grid_HO_2.nc")
results_dyn[3].export_to_netcdf("results/dynamic_grid_HO_3.nc")
results_dyn[4].export_to_netcdf("results/dynamic_grid_HO_4.nc");








