import pypsa 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from Model import *
plt.rcParams.update({
    #
    "axes.titlesize": 18,      # Title font size
    "axes.labelsize": 14,      # X/Y label font size
    "xtick.labelsize": 14,     # X tick label font size
    "ytick.labelsize": 14,     # Y tick label font size
    # Legend font size
    "figure.facecolor": "black",      # Figure background
    "axes.facecolor": "black",        # Axes background
    "savefig.facecolor": "black",     # Saved figure background
    "text.color": "white",            # Default text color
    "axes.labelcolor": "white",       # Axes label color
    "xtick.color": "white",           # X tick color
    "ytick.color": "white",           # Y tick color
    "axes.edgecolor": "white",        # Axes edge color
    "legend.edgecolor": "white",      # Legend edge color
    "legend.facecolor": "black",      # Legend background
    "grid.color": "white",            # Grid color
})



def plot_network(n, link_flow=False, title=None):
    if not link_flow:
        link_flow = pd.Series(0.1, index=n.branches().loc["Link"].index)
    fig = plt.figure(figsize=(12, 6))
    n.plot.map(link_flow=link_flow,
                bus_sizes=1e-3);

    plt.hlines(y=0.8, xmin=0, xmax=0.8, color='white', lw=2)
    plt.hlines(y=0.3, xmin=0, xmax=0.8, color='white', lw=2)
    plt.vlines(x=0, ymin=.3, ymax=0.8, color='white', lw=2)
    plt.vlines(x=0.8, ymin=.3, ymax=.8, color='white', lw=2)
    plt.plot([0, 0.35],[0.8, .95], color='white', lw=2)
    plt.plot([0.35, 0.8],[.95, .8], color='white', lw=2)

    for bus_name, bus in n.buses.iterrows(): 
        if bus_name=='electricity':
            plt.text(bus.x-0.08, bus.y+0.05, bus_name, fontsize=12, ha='center', va='center',color='white')
        else:
            plt.text(bus.x, bus.y+0.05, bus_name, fontsize=12, ha='center', va='center',color='white')
    # ax.title("Energiezelle ", fontsize=20,pad=40)
    # plt.show()
    return fig



    


### Time Series Plots ##########################################################
def plot_timeframe(n, t1, t2,v, title="Electrical energy balance",plots=[True, True, True]):


    fig,axs=plt.subplots(3,1,figsize=(16, 10),sharex=True,gridspec_kw={'height_ratios': [2,1,2]})
    ax=axs[0]
    ax2=axs[1]
    ax3=axs[2]


    plot_energy_balance(ax, n, t1, t2, title=title)
    ax.legend(loc='lower center', ncols=5, framealpha=1, facecolor='black', fontsize=14)
    ax.set_ylim(-22,22)
    ax2.set_title("Electricity prices")
    plot_dynamic_prices(v,t1, t2,ax2)
    ax2.legend(loc='upper left', fontsize=14)   
    ax3,ax4=plot_storages(ax3, n, t1, t2, title="Electrical energy balance of storage systems")
    ax3.legend(loc='lower left', ncols=4, framealpha=1, facecolor='black',fontsize=14)
    ax4.legend(loc='lower right', fontsize=14)   


    ax3.set_xlabel("Time")
    plt.setp(ax3.get_xticklabels(), rotation=0)
    plt.tight_layout()
    # plt.show()
    return fig

def plot_energy_balance(ax, n, t1, t2, title="Electrical energy balance"):
    gen_p = n.generators_t.p.loc[t1:t2]


    load_p = n.loads_t.p.loc[t1:t2]
    ax.set_title(title)

    labels = ["grid_buy", 
              "pv_system",
            #   "battery",
            #   'car',
            #   "tes"
              ]
    colors = {
        "grid_buy": "#e41a1c",
        "pv_system": "#c5c203",
        # "battery": "#4daf4a",
        # "car": "#ff7f00",
        # "tes": "#c5c203"

    }
    x = gen_p.index
    y = []
    y.append(gen_p["grid_buy"].values)
    y.append(gen_p["pv_system"].values)
    ax.axhline(0,color='white', linestyle='-', linewidth=1)

    ax.stackplot(x, y, labels=labels, colors=[colors.get(label, None) for label in labels], alpha=1, step="pre")
    total_load = np.sum(y, axis=0)
    ax.step(x, total_load, label=None, color="green", linewidth=1)
   
    labels = ["electrical load",
            #   "dhw_load",
            #   "heat_load",
              "car_usage",
              "grid_sell"]
    colors = {    

        "electrical load": "#377eb8",
        "heat_load": "#e41a1c",
        "dhw_load": "#ac10db",
        "car_usage": "#ff7f00",
        "grid_sell": "#29c95e"

    }
    x = load_p.index
    y = []

    y.append(-load_p["electrical load"]-load_p["dhw_load"]-load_p["heat_load"]/5)
    # y.append(-load_p["electrical load"])
    # y.append(-load_p["dhw_load"])
    # y.append(-load_p["heat_load"]/5)
    y.append(-load_p["car_usage"])
    y.append(-gen_p["grid_sell"])

    ax.stackplot(x, y, labels=labels, colors=[colors.get(label, None) for label in labels], alpha=1, step="pre")
    total_load = np.sum(y, axis=0)
    
    # ax.step(x, total_load, label=None, color="red", linewidth=1)
    ax.legend(loc='lower center',ncols=5)
    ax.set_ylabel("Power [kW]")
    ax.grid(axis='x')

    return ax

     

    
def plot_storages(ax,n, t1, t2, title="Electrical energy balance of storage systems"):
    stores_p = n.stores_t.p.loc[t1:t2]

    stores_p_out =  stores_p.mask(stores_p >= 0, 0)
    stores_p_in = stores_p.mask(stores_p <= 0, 0)

    stores_e = n.stores_t.e.loc[t1:t2]
    stores_cap=n.stores.e_nom_opt
    ax.set_title("Energy balance of electrical storages")

    ax2=ax.twinx()
    
    labels = [
        # "tes",
        "battery", 
        "car"]

    colors = {
        "tes": "#e41a1c",
        "car": "#ff7f00",
        "battery": "#bc1dca",

    }
    x = stores_p.index
    y = []

    # y.append(stores_p_in["tes"].values)
    y.append(stores_p_in["battery"].values)
    y.append(stores_p_in["car"].values)
    ax.stackplot(x, y, labels=["battery_dischg","BEV_dischg"], colors=[colors.get(label, None) for label in labels], alpha=0.8, step="pre")
    total_load = np.sum(y, axis=0)
    y = []

    # y.append(stores_p_out["tes"].values)
    y.append(stores_p_out["battery"].values)
    y.append(stores_p_out["car"].values)
    ax.stackplot(x, y,labels=["battery_chg","BEV_chg"] ,colors=[colors.get(label, None) for label in labels], alpha=0.8, step="pre",hatch='//')
    # total_load = np.sum(y, axis=0)
    ax.axhline(0,color='white', linestyle='-', linewidth=1)

    ax.legend(loc='lower left', ncols=4, framealpha=1, facecolor='black')
    ax.set_ylabel("Power [kW]")
    ax.grid(False)
    # ax.set_ylim(-12, 12)


    ax2.plot(stores_e.sum(axis=1).index,stores_e.sum(axis=1)/(stores_cap.sum()),label='combined SOC', color='white', linestyle='--', linewidth=1)
    # ax3.plot(stores_e.sum(axis=1).index,(stores_e['car']+stores_e['battery'])/(stores_cap['car']+stores_cap['battery']),label='combined SOC', color='black', linestyle='--', linewidth=1)
    # ax3.plot(stores_e.index,stores_e['battery'],label='combined SOC', color='black', linestyle='--', linewidth=1)
    # ax3.plot(stores_e.index,stores_e['car'],label='combined SOC', color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel("SOC [%]")
    ax.legend(loc='upper right')
    ax.grid(axis='x')


    return ax, ax2

def plot_dynamic_prices(Variante,t1,t2,ax=False):
    d=dict(static='static',
    stock_2019='stock',
    stock_2024='stock',
    HT_NT='ewa')


    data = input_data(prices=d[Variante],sell_price=2019 if Variante == 'stock_2019' else 2024)

    prices = -data.el_sell.loc[t1:t2]
    buy = data.el_price.loc[t1:t2]
    fig=False
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.step(prices.index, prices*100, label='Sell Price', color='green')
    ax.step(buy.index, buy*100, label='Buy Price', color='red')
    if fig:
        ax.set_title('Dynamic electricity prices')
        ax.set_xlabel('Time')
    ax.set_ylabel('Price [ct/kWh]')
    ax.legend(loc='upper left')
    ax.set_ylim(0,40)
    ax.grid(axis='x')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig if fig else ax










def plot_timeframe_2(n, t1, t2, title="Energy generation, load profile, and battery load"):

    gen_p = n.generators_t.p.loc[t1:t2]
    stores_p = n.stores_t.p.loc[t1:t2]
    stores_e = n.stores_t.e.loc[t1:t2]
    stores_cap=n.stores.e_nom_opt
    fig,axs=plt.subplots(4,1,figsize=(12, 6),sharex=True)

    ax=axs[0] 
    for col in gen_p.columns:
        if col == "grid_sell":

            ax.step(gen_p[col].index, -gen_p[col], label=f"{col}", linestyle="-")
        else:
            ax.step(gen_p[col].index, gen_p[col], label=f"{col}", linestyle="-")
    ax.set_title(title)
    ax.set_ylabel("Power [kW]")
    ax.legend(loc="lower right")


    ax=axs[1]
    for col in stores_p.columns:
        # if col == "Car":
        ax.step(stores_p[col].index, -stores_p[col], label=f"{col}", linestyle="-")
    ax.step(n.loads_t.p['car_usage'].loc[t1:t2].index, -n.loads_t.p['car_usage'].loc[t1:t2], label=f"car usage", linestyle="--",color='red',linewidth=1.5)
        
        # ax.step(snapshots.loc[t1:t2], n.stores_t.p[col], label=f"{col}", linestyle="-")
    ax.set_ylabel("Power [kW]")
    ax.legend(loc="lower right")


    ax=axs[2]
    for col in stores_e.columns:
        ax.step(stores_e[col].index, stores_e[col]/stores_cap[col] * 100, label=f"{col}", linestyle="-")
    ax.set_ylabel("SOC [%]")
    ax.legend(loc="lower right")

    ax = axs[3]
    # Prepare data for stackplot
    load_types = [col for col in n.loads_t.p.columns if col != "car_usage"]
    colors = {
        "heat_load": "#e41a1c",
        "electrical load": "#377eb8",
        "dhw_load": "#4daf4a"
    }
    x = n.loads_t.p.loc[t1:t2].index
    y = []
    for col in load_types:
        l = n.loads_t.p[col].loc[t1:t2]
        if col == "heat_load":
            l = l / 5
        y.append(l.values)
    # Stackplot
    ax.stackplot(x, y, labels=load_types, colors=[colors.get(col, None) for col in load_types], alpha=0.8, step="pre")
    # Total load curve
    total_load = np.sum(y, axis=0)
    ax.step(x, total_load, label="load (total)", color="black", linewidth=2)
    ax.legend()
    ax.set_ylabel("Power [kW]")
    ax.grid(True)
    plt.xlabel("Time (Hourly)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    return fig





def plot_peak_shaving(n, t1, t2, title='Grid import and peak load'):
    grid_import = n.generators_t.p.loc[t1:t2].grid_buy
    grid_export = -n.generators_t.p.loc[t1:t2].grid_sell

    fig,ax=plt.subplots(1,1,figsize=(12,6))
    ax.step(grid_export.index, grid_export, label='Grid Export')
    ax.step(grid_import.index, grid_import, label='Grid Import')
    # plt.axhline(grid_import.max(), color='red', linestyle='--', label=f'Peak: {grid_import.max():.1f} kW')
    plt.ylabel('kW')
    plt.ylim(-23,23)
    plt.xlabel('Time')
    ax.set_title(title)

    plt.legend(loc='upper left')
    plt.tight_layout()
    # plt.show()
    return fig

def plot_duration_curve(ts, title="Grid import duration curve",ylabel='kW'):
    sorted = np.sort(ts)[::-1]  # Descending
    fig,ax=plt.subplots(1,1,figsize=(16,9))
    ax.plot(sorted, label='Grid Import Duration Curve')
    plt.ylabel(ylabel)
    plt.xlabel('Hour (sorted)')
    ax.set_title(title)
    plt.tight_layout()
    # plt.show()
    return fig
    


def plot_loads(n, t1, t2, title="Model input data: load profiles"):
    fig,axs=plt.subplots(4,1,figsize=(12, 6),sharex=True)
    ax=axs[0]
    ax.set_title(title)

    for i,col in enumerate(n.loads_t.p.columns):
        ax = axs[i]
        ax.step(n.loads_t.p[col].loc[t1:t2].index, n.loads_t.p[col].loc[t1:t2], label=f"{col}", linestyle="-")
        ax.legend()
        ax.set_ylabel("Power [kW]")
        ax.grid(False)

    #plt.plot(snapshots, control_plot, label="Kontrolle", linestyle="-")
    plt.xlabel("Time (Hourly)")
    # plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    return fig


def plot_link_flows(n, t1, t2, title="Energy generation, load profile, and battery load"):

    flows = n.links_t.p0.loc[t1:t2]


    fig,axs=plt.subplots(len(flows.columns),1,figsize=(12, 6),sharex=True)
    ax.set_title(title)

    for i,col in enumerate(flows.columns):
        ax = axs[i]
        ax.step(flows[col].loc[t1:t2].index, flows[col].loc[t1:t2], label=f"{col}", linestyle="-")
        ax.legend()
        ax.set_ylabel("Power [kW]")
        ax.grid(True)
    #plt.plot(snapshots, control_plot, label="Kontrolle", linestyle="-")
    plt.xlabel("Time (Hourly)")
    # plt.ylabel("Power [kW]")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    return fig


def driving_plot(driving_days, title='Driving pattern and charger availability (24 hours)'):
    driving, charger_p_max_pu = input_data().create_driving_and_charger_series(driving_days=driving_days)
    fig,ax = plt.subplots(2, 1, figsize=(16,9),sharex=True)



    AX = ax[0]
    AX.set_title('Driving pattern')
    AX.step(range(24), driving.iloc[:24]/driving.iloc[:24].max(), where='post', label='Driving')
    AX.set_yticks([0,1])
    AX.grid(False)

    AX=ax[1]
    AX.set_title('Charger availability')
    AX.step(range(24), charger_p_max_pu.iloc[:24], where='post', label='Charger Availability')
    AX.set_yticks([0,1])
    AX.grid(False)

    # plt.step(range(24), -charger_p_max_pu.iloc[:24], where='mid', label='Charger Availability')
    plt.xticks(range(24))
    plt.xlabel('Hour')
    # plt.show()
    return fig


# # # Correlation Plots #######################################################################
def plot_pv_vs_export(n, v='static',title="PV vs. grid export",grid=False):
    gen_p = n.generators_t.p

    d=dict(static='static',
    stock_2019='stock',
    stock_2024='stock',
    HT_NT='ewa')


    data = input_data(prices=d[v],sell_price=2019 if v in ['stock_2019','static'] else 2024)

    price = data.el_sell
    
    fig,ax=plt.subplots(1,1,figsize=(12, 6),sharex=True)
    
    
    # if dynamic:
    sc = ax.scatter(gen_p['grid_sell'], gen_p['pv_system'], c=-price*100, cmap='viridis',vmin=0,vmax=25,alpha=0.8)
    cbar = plt.colorbar(ax.collections[0], ax=ax, label='Sell Price [ct/kWh]')
    
    # else:
        # ax.scatter(gen_p['grid_sell'],gen_p['pv_system'], color='#FFC000', alpha=0.8)
    ax.plot([gen_p['grid_sell'].min(), gen_p['grid_sell'].max()],
            [gen_p['grid_sell'].min(), gen_p['grid_sell'].max()],
            color='red', linestyle='-', label=r'$\frac{PV}{Export} = 1$')
    # ax.legend()
    # Avoid division by zero by masking zero grid_sell values
    # ratio = gen_p['pv_system'].mean() / gen_p['grid_sell'].mean()
    # # ratio = ratio.replace([np.inf, -np.inf], np.nan)  # Replace inf with nan
    # # avg = ratio.mean(skipna=True)
    # ax.text(0.1, 0.9, f'Average Ratio: {ratio:.2f}',fontsize=16,transform=ax.transAxes,bbox=dict(facecolor='black', alpha=0.5, edgecolor='white'))

    ax.set_ylim(-1,23)
    ax.set_xlim(-1,23)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_ylabel("PV [kW]")
    if grid:
        ax.grid(grid,linewidth=0.3,alpha=0.5)
    else:
        ax.grid(False)
    ax.set_title(title)


    plt.xlabel("grid_sell [kW]")
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    return fig

def plot_battery_cycles(results, title='Storage cycles per year for each scenario'):
    """
    Plots the number of full battery and car cycles per year for each scenario.
    A cycle is calculated as total energy charged divided by storage capacity.
    """
    battery_cycles = {}
    car_cycles = {}
    for key, r in results.items():
        # Battery cycles
        battery_charging = r.stores_t.p['battery'].clip(lower=0)
        total_battery_charged = battery_charging.sum()  # assumes hourly data, units: kWh
        battery_capacity = r.stores.e_nom_opt['battery']
        n_battery_cycles = total_battery_charged / battery_capacity if battery_capacity > 0 else 0
        battery_cycles[key] = n_battery_cycles
        # Car cycles
        car_charging = r.stores_t.p['car'].clip(lower=0)
        total_car_charged = car_charging.sum()
        car_capacity = r.stores.e_nom_opt['car']
        n_car_cycles = total_car_charged / car_capacity if car_capacity > 0 else 0
        car_cycles[key] = n_car_cycles

    labels = [f"{k}" if isinstance(k, int) else "reference" for k in battery_cycles.keys()]
    battery_values = list(battery_cycles.values())
    car_values = list(car_cycles.values())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(x - width/2, battery_values, width, label='Battery', color='purple')
    ax.bar(x + width/2, car_values, width, label='Car', color='orange')
    ax.set_xlabel('Number of days in home office per week')
    ax.set_ylabel('Cycles per year')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    return fig
### Bar Charts ##########################################################
def plot_optimal_battery_capacity(results, title='Optimal battery capacity for varying number of home office days'):
    fig,ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_title(title)
    for d in results.keys():
        if type(d) == int:
            ax.bar(d, results[d].stores.e_nom_opt.battery, color='skyblue')

    ax.set_ylabel('kWh')
    ax.set_xlabel('Number of days in home office per week')

    plt.tight_layout()
    return fig
def plot_optimal_battery_capacity_dyn(results, results2,label1,label2, title='Comparison optimal battery capacity for varying number of home office days'):
    # Collect battery capacities
    cap = {k: v.stores.e_nom_opt.battery for k, v in results.items()}
    cap_dyn = {k: v.stores.e_nom_opt.battery for k, v in results2.items()}

    keys = [k for k in cap.keys() if k in cap_dyn]
    labels = [f"{k}" if isinstance(k, int) else "reference" for k in keys]
    values = [cap[k] for k in keys]
    values_dyn = [cap_dyn[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.bar(x - width/2, values, width, label=label1, color='orange')
    ax.bar(x + width/2, values_dyn, width, label=label2, color='blue')
    ax.set_ylabel('kWh')
    ax.set_xlabel('Number of days in home office per week')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    return fig
def plot_autarky(results, title='Autarky for varying number of home office days'):
    aut=autarkie(results)
    fig,ax=plt.subplots(1,1,figsize=(12, 6))
    labels = []
    values = []
    for key in aut:
        if type(key) is int:
            labels.append(f"{key}")
        else:
            labels.append("reference")
        values.append(aut[key])

    ax.bar(labels, values, color='green')
    ax.set_xlabel('Number of days in home office per week')
    # ax.set_ylabel('%')
    ax.set_yticks(ticks=np.linspace(0, 1, 11), labels=[f"{int(100*x)}%" for x in np.linspace(0, 1, 11)])
    ax.set_title(title)
    plt.tight_layout()
    return fig

def plot_autarky_dyn(results, results2,label1,label2, title='Comparison of autarky for varying number of home office days'):
    aut = autarkie(results)
    aut_dyn = autarkie(results2)

    # Ensure both dicts have the same keys and order
    keys = [k for k in aut.keys() if k in aut_dyn]
    labels = [f"{k}" if isinstance(k, int) else "reference" for k in keys]
    values = [aut[k] for k in keys]
    values_dyn = [aut_dyn[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(x - width/2, values, width, label=label1, color='green')
    ax.bar(x + width/2, values_dyn, width, label=label2, color='blue', alpha=0.7)
    ax.set_xlabel('Number of days in home office per week')
    ax.set_yticks(ticks=np.linspace(0, 1, 11), labels=[f"{int(100*x)}%" for x in np.linspace(0, 1, 11)])
    ax.set_ylabel('Degree of autarky')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_energies(title="Total energy need (electrical)"):
    energy = dict()
    data = input_data()
    for d in [0,1,2,3,4,'reference']:
        if type(d) == int:
            energy[d]=data.calc_need(5-d)
        else: 
            energy[d]=data.calc_need(5)

    df = pd.DataFrame(energy).T.drop(columns='total')

    color_map = {
        'heat': '#e41a1c',
        'el': '#377eb8',
        'dhw': '#4daf4a',
        'driving': '#ff7f00'
    }

    fig, ax = plt.subplots(1,1,figsize=(12,6))
    bottom = np.zeros(len(df))
    x = np.arange(len(df))
    for col in df.columns:
        ax.bar(x, df[col]/1000, bottom=bottom, label=col+" (el.)" if col == 'heat' else col, color=color_map.get(col, None))
        bottom += df[col].values/1000

    ax.set_ylabel('MWh')
    ax.set_xlabel('Number of days in home office per week')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index)
    ax.legend(ncols=4)
    plt.tight_layout()
    return fig

def plot_costs(results, title='Yearly costs for varying number of home office days'):
    
    kosten = dict()
    # print('Kostenübersicht in Abhängigkeit der Home Office Tage:')
    for key,r in results.items():
        kosten[key] = round(r.objective,2)

    fig,ax=plt.subplots(1,1,figsize=(12,6))
    labels = []
    values = []
    for key in kosten:
        if type(key) is int:
            labels.append(f"{key}")
        else:
            labels.append("reference")
        values.append(kosten[key])
    ax.axhline(0, color='white', lw=1)

    ax.bar(labels, values, color='orange')
    ax.set_xlabel('Number of days in home office per week')
    ax.set_ylabel('€/a')
    ax.set_title(title)
    # plt.show()
    plt.tight_layout()
    return fig

def plot_costs_dyn(results, results2,label1,label2, title='Comparison of yearly costs for varying number of home office days'):
    kosten = dict()
    kosten_dyn = dict()
    for key, r in results.items():
        kosten[key] = round(r.objective, 2)
    for key, r in results2.items():
        kosten_dyn[key] = round(r.objective, 2)

    # Ensure both dicts have the same keys and order
    keys = [k for k in kosten.keys() if k in kosten_dyn]
    labels = [f"{k}" if isinstance(k, int) else "reference" for k in keys]
    values = [kosten[k] for k in keys]
    values_dyn = [kosten_dyn[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(x - width/2, values, width, label=label1, color='orange')
    ax.bar(x + width/2, values_dyn, width, label=label2, color='blue', alpha=0.7)
    ax.set_xlabel('Number of days in home office per week')
    ax.set_ylabel('€/a')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.axhline(0, color='white', lw=1)

    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    return fig



def price_histogramm(title='Price histogram',Variante='static'):

    d=dict(static='static',
    stock_2019='stock',
    stock_2024='stock',
    HT_NT='ewa')


    data = input_data(prices=d[Variante],sell_price=2019 if Variante in ['stock_2019','static'] else 2024)

    prices = data.el_sell
    buy = data.el_price
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    if d[Variante] == 'ewa':
        sns.histplot(buy*100, bins=2, ax=ax, color='red',label='Buy Price ')
        sns.histplot(prices*100,bins=3, ax=ax, color='green', label='Sell Price ')
    elif d[Variante] == False:
        sns.histplot(buy*100, bins=2, ax=ax, color='red',label='Buy Price ')
        sns.histplot(prices*100, bins=2, ax=ax, color='green', label='Sell Price ')
    else:
        sns.histplot(buy*100,bins=400, ax=ax, color='red',label='Buy Price ')
        sns.histplot(prices*100,bins=400, ax=ax, color='green', label='Sell Price ')
    ax.set_title(title)             
    ax.legend(loc='upper right')
    ax.set_xlabel('Price [ct/kWh]')
    ax.set_ylabel('Hours')
    ax.set_xlim(-40,40)
    plt.tight_layout()  
    # plt.show()
    return fig


def plot_grid_heat_map(n,direction,title=False):

    assert direction in ['Import','Export'], "Direction must be 'Import' or 'export'"

    grid_ts = n.generators_t.p['grid_buy' if direction == 'Import' else 'grid_sell']

    # Import events above threshold
    ts = grid_ts[grid_ts > 0]
    hours = ts.index.hour
    vals = ts.values

    fig,ax= plt.subplots(figsize=(12, 6))
    ax.hist2d(hours, vals, bins=[24, 40], cmap='Blues' if direction == 'export' else 'Reds')
    ax.colorbar(label='Number of events')
    plt.xlabel('Hour of Day')
    plt.ylabel('Power [kW]')
    plt.title(title if title else f'Heatmap of grid {direction.lower()} events')
    plt.tight_layout()
    return fig

def plot_peak_distro(n,threshold,title):


        # Extract grid import/export time series
        grid_import = n.generators_t.p['grid_buy']
        grid_export = n.generators_t.p['grid_sell'] if 'grid_sell' in n.generators_t.p else None

        # Filter for values above threshold
        import_above = grid_import[grid_import > threshold]
        import_hours = import_above.index.hour

        if grid_export is not None:
            export_above = grid_export[grid_export > threshold]
            export_hours = export_above.index.hour
        else:
            export_hours = None

        fig,ax=plt.subplots(figsize=(12,6))
        ax.hist(import_hours, bins=range(25), alpha=0.7, label=f'Import > {threshold} kW', color='tab:orange', edgecolor='k')
        if export_hours is not None and len(export_hours) > 0:
            ax.hist(export_hours, bins=range(25),weights=-np.ones_like(export_hours),alpha=0.7, label=f'Export > {threshold} kW', color='tab:blue', edgecolor='k')
        ax.set_xlabel('Hour of day')
        ax.set_ylabel('Number of events')
        ax.set_ylim(-100,100)
        ax.axhline(0, color='white', linewidth=1) 

        plt.title(title)
        
        ax.legend()
        ax.set_xticks(range(0,25))
        plt.tight_layout()
        return fig
