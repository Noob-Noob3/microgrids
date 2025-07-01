import pypsa 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy 

def save_results(n, path, i=False):
    """
    Save the PyPSA network to a NetCDF file.

    Args:
        n (pypsa.Network): The network to save.
        path (str): File path to save the network (should end with .nc).
        i (int or bool, optional): Optional index for multi-house scenarios.
    """
    suffix = f"_{i}" if i else ""
    n.export_to_netcdf(f"{path}{suffix}.nc")

def load_results(file):
    """
    Load a PyPSA network from a NetCDF file.

    Args:
        path (str): File path to load the network from (should end with .nc).
        i (int or bool, optional): Optional index for multi-house scenarios.

    Returns:
        pypsa.Network: Loaded network.
    """
    return pypsa.Network(f"{file}.nc")


def house_network(snapshots,heat_load,el_load,dhw_load,pv_pu,driving,charger_p_max_pu,el_price,el_sell,bess_nom,pv_nom=20,pv_extendable=False,battery_extendable=False,i=False):
       """
       Creates a PyPSA network and optimization model for a single house with electricity, heat, and hot water systems.

       Parameters:
              heat_load (pd.Series): Time series of heat demand (kW).
              el_load (pd.Series): Time series of electrical demand (kW).
              dhw_load (pd.Series): Time series of domestic hot water demand (kW).
              pv_pu (pd.Series): Per-unit PV generation profile (relative to 1 kWp).
              driving (pd.Series): Time series of car usage (kW or availability).
              charger_p_max_pu (pd.Series): Charger availability profile (0 or 1).
              el_price (float): Electricity purchase price (€/kWh).
              el_sell (float): Electricity feed-in tariff (€/kWh, negative for revenue).
              pv_nom (float): Nominal PV capacity (kW).
              bess_nom (float): Nominal battery storage capacity (kWh).
              pv_extendable (bool, optional): If True, PV capacity can be optimized. Default is False.
              battery_extendable (bool, optional): If True, battery capacity can be optimized. Default is False.
              i (int or bool, optional): House index for multi-house scenarios. Default is False (single house).

       Returns:
              pypsa.Network: Configured PyPSA network object for the house.
       """
       i=f"_{i}" if i else ""

       n = pypsa.Network()
       n.set_snapshots(snapshots)

       if not i:
              n.add("Carrier", "AC")
              n.add("Carrier", "heat")


       # # Stromseite
       n.add("Bus", name = f"electricity{i}",x=0.5,y=.5)
       n.add("Bus", name = f"bess{i}", x=0.2, y=.5)
    #    n.add("Bus", name = f"grid{i}",x=0.2,y=0.2)
       n.add("Bus", name = f"bev{i}",x=.9,y=0.5)
    #    n.add("Bus", name = f"pv{i}",x=0.5,y=.95)


       n.add("Load", name = f"electrical{i} load", bus = f"electricity{i}", p_set = el_load)
       n.add("Load", name = f"car_usage{i}", bus = f"bev{i}", p_set = driving)

       n.add("Generator", name = f"grid_buy{i}", bus = f"electricity{i}", 
              marginal_cost = el_price, 
              p_nom = 22,
              p_max_pu=1,
              control="Slack"
              )

       n.add("Generator", name = f"grid_sell{i}", bus = f"electricity{i}", 
              p_nom = 22, 
              p_max_pu=1, 
              p_min_pu=0.0,
              sign = -1,
              marginal_cost = el_sell, 
              )

       n.add("Generator", name = f"pv_system{i}", bus = f"electricity{i}",
              p_nom = pv_nom,
              p_nom_max = 30,
              marginal_cost = 0,
              capital_cost = 1200/60,
              p_max_pu = pv_pu ,
              p_nom_extendable = pv_extendable,  
              )

       n.add("Store", name = f"battery{i}", bus = f"bess{i}", 
              # e_nom = bess_nom,
              e_initial=0,
              e_nom_extendable= battery_extendable,
              e_nom_max=20,
              capital_cost=1200/60,
              )

       n.add("Store", name = f"car{i}", bus = f"bev{i}",
              e_initial=59, 
              e_nom_extendable=False, 
              e_nom=59,
              )



       n.add("Link", name = f"bev_charge{i}", bus0 = f"electricity{i}", bus1 = f"bev{i}",p_nom=22,p_max_pu=charger_p_max_pu,p_min_pu=0,efficiency=0.81**0.5)
       n.add("Link", name = f"bev_discharge{i}", bus0 = f"bev{i}", bus1 = f"electricity{i}",p_nom=22,p_max_pu=charger_p_max_pu,p_min_pu=0,efficiency=0.81**0.5)

    #    n.add("Link", name = f"pv_link{i}", bus0 = f"pv{i}", bus1 = f"electricity{i}",p_nom=9999)
    #    n.add("Link", name = f"grid_link{i}", bus0 = f"grid{i}", bus1 = f"electricity{i}", p_nom = 9999)

       n.add("Link", name = f"battery_charge_link{i}", bus0 = f"electricity{i}", bus1 = f"bess{i}",p_nom=10,p_min_pu=0,p_max_pu=1,efficiency=0.81**0.5)
       n.add("Link", name = f"battery_discharge_link{i}", bus0 = f"bess{i}", bus1 = f"electricity{i}",p_nom=10,p_min_pu=0,p_max_pu=1,efficiency=0.81**0.5)




       # # Wärmeseite
       n.add("Bus", name = f"heat{i}",x=0.7,y=.7,carrier="heat")
       n.add("Bus", name = f"tes{i}",x=0.6,y=.6,carrier="heat")
       n.add("Load", name = f"heat_load{i}", bus = f"heat{i}", p_set = heat_load)
       n.add("Link", name = f"heat_pump{i}", bus0 = f"electricity{i}", bus1 = f"tes{i}", efficiency=5,p_min_pu=0.0,p_nom=2)

       n.add("Store", name = f"tes{i}", bus = f"tes{i}", e_initial=0,e_nom_extendable=True,e_nom_max=50)
       n.add("Link", name = f"heat_transfer{i}", bus0 = f"tes{i}", bus1 = f"heat{i}", efficiency=1,p_min_pu=0.0,p_nom=9999)


       # # TWW
       n.add("Bus", name = f"dhw{i}",x=0.6,y=.42,carrier="heat")
       n.add("Load", name = f"dhw_load{i}", bus = f"dhw{i}", p_set = dhw_load)
       n.add("Link", name = f"flow-water-heater{i}", bus0 = f"electricity{i}", carrier="AC",bus1 = f"dhw{i}", efficiency=1,p_min_pu=0.0,p_nom=9999)


       return n 



    
def autarkie(results):
    aut = dict()
    for key,n in results.items():
        gen=n.generators_t.p

        bezug=gen.grid_buy

        input_obj = input_data()
        verbrauch = input_obj.calc_need(5-key if type(key) is int else 5)['total']

                # Limit grid import to the total demand — ignore arbitrage energy
        imported_energy = min(bezug.sum(), verbrauch)

        # Autarky = share of demand met without imports
        autarky = 1 - (imported_energy / verbrauch)

        aut[key] = autarky
        # aut[key] = round(1 - bezug.sum()/verbrauch.sum(),2)
    return aut

class input_data:
    def __init__(self,prices='static',dynamic_grid_prices=False,sell_price=2019):
        
        self.snapshots = pd.date_range(start="2019-01-01 00:00", freq="h", periods=8760)
        


        if sell_price==2019:
            self.el_sell = pd.read_csv('input_data/sell_2019.csv',index_col=0,parse_dates=True)/100#€/kWh
            self.el_sell.index = self.snapshots
            self.el_sell = self.el_sell['0']*(-1)
            self.el_sell.name = 'el_sell'
        elif sell_price==2024:
            self.el_sell = pd.read_csv('input_data/direktvermarktung_solar_2024.csv',index_col=0,parse_dates=True)/100#€/kWh
            self.el_sell.index = self.snapshots
            self.el_sell = self.el_sell['0']*(-1)
            self.el_sell.name = 'el_sell'


        if dynamic_grid_prices:
            self.grid_price = pd.read_csv(r'input_data/variable_Netzentgelte.csv',index_col=0,parse_dates=True)

            self.grid_price.index = self.snapshots
            self.grid_price = self.grid_price['Price_Ct_per_kWh']/100 # €/kWh
            self.grid_price.name = 'grid_price'

            self.el_price = (0.1871 + self.grid_price)*1.19
            self.el_price.index = self.snapshots
            self.el_price.name = 'el_price'
        else:
            self.grid_price = 0.0729


        if prices=='static':

            self.el_price = pd.Series([0.3094]*8760) # €/kWh
            self.el_price.index = self.snapshots
            self.el_price.name = 'el_price'



        elif prices=='stock':
            self.price=pd.read_csv(fr'input_data/stock_prices_{sell_price}.csv',index_col=0)/1000
            self.price.index = self.snapshots

            self.el_price = (self.price['Day Ahead Auktion']+self.grid_price)*1.19
            self.el_price.name = 'el_price'

            self.el_sell = -(self.price['Day Ahead Auktion'])
            self.el_sell.name = 'el_sell'

        elif prices=='ewa':
            self.el_price = pd.read_csv('input_data/ewa_variabel_2019.csv',index_col=0,parse_dates=True)
            self.el_price = self.el_price['brutto']/100
            self.el_price.index = self.snapshots
            self.el_price.name = 'el_price'



        elif prices=='ewa' and dynamic_grid_prices:
            self.el_price = pd.read_csv('input_data/ewa_variabel_2019.csv',index_col=0,parse_dates=True)
            self.el_price = (self.el_price['netto']+self.grid_price)/100*1.19
            self.el_price.index = self.snapshots
            self.el_price.name = 'el_price'



            

    def load_data(self,driving_days=5):
        
        # Ohne Nachtabsenkung
        dhw_load = pd.read_csv(r'input_data/Lastprofil_ohneNachtabsenkung/dhw/neighbourhood.csv', comment="#", sep=";",usecols=[1])
        heat_load = pd.read_csv(r'input_data/Lastprofil_ohneNachtabsenkung/htg/neighbourhood.csv', comment="#", sep=";",usecols=[2])
        el_load = pd.read_csv(r'input_data/Lastprofil_ohneNachtabsenkung/el/neighbourhood.csv', comment="#", sep=";",usecols=[1])

        #PV
        pv_pu = pd.read_csv("input_data/PVlast_1kw.csv", sep=",",index_col=0,usecols=[0,2])

        #Zeitreihen einheitlich indexieren
        dhw_load.index=self.snapshots
        heat_load.index=self.snapshots
        el_load.index=self.snapshots
        pv_pu.index=self.snapshots

        dhw_load=dhw_load['total']
        heat_load=heat_load['total_htg']
        el_load=el_load['total']
        pv_pu=pv_pu['pv_pu']

        dhw_load.name='dhw_load'
        heat_load.name='heat_load'
        el_load.name='el_load'



        driving, charger_p_max_pu = self.create_driving_and_charger_series(driving_days=driving_days)

        return [heat_load, el_load, dhw_load,pv_pu, driving, charger_p_max_pu,self.el_price, self.el_sell]

    def create_driving_and_charger_series(self, driving_days=5, driving_hours=[7,8,17,18], driving_power=9.0):
        """
        Create driving and charger availability time series for a given number of driving days per week.

        Args:
            snapshots (pd.DatetimeIndex): Time index for the year (e.g., 8760 hourly values).
            driving_days (int): Number of days per week with car usage (default: 5).
            driving_hours (list): List of hours (0-23) when the car is used each driving day.
            driving_power (float): Power value for driving periods (default: 9.0).

        Returns:
            driving (pd.Series): Series with driving power (kW) for each hour.
            charger_p_max_pu (pd.Series): Series with charger availability (1=available, 0=not available).
        """
        driving = pd.Series(0.0, index=self.snapshots)
        charger_p_max_pu = pd.Series(1.0, index=self.snapshots)

        # Determine which days are driving days (e.g., Mon-Fri for 5, Mon-Thu for 4, etc.)
        week_pattern = [1]*driving_days + [0]*(7-driving_days)
        week_pattern = np.array(week_pattern)

        for day in range(len(self.snapshots)//24):
            week_day = day % 7
            day_start = day*24
            if week_pattern[week_day]:
                # Set driving hours
                for h in driving_hours:
                    idx = day_start + h
                    if idx < len(driving):
                        driving.iloc[idx] = driving_power
                # Charger unavailable from first to last driving hour
                first = min(driving_hours)
                last = max(driving_hours)
                charger_p_max_pu.iloc[day_start+first:day_start+last+1] = 0.0

        return driving, charger_p_max_pu
    
    def calc_need(self,driving_days=5):
        [heat_load, el_load, dhw_load,_, driving, _,_,_] = self.load_data(driving_days)

        heat = heat_load.sum() / 5
        el = el_load.sum()
        dhw = dhw_load.sum()
        drving = driving.sum()

        total = heat + el + dhw + drving

        return dict(
            heat=heat,
            el=el,
            dhw=dhw,
            driving=drving,
            total=total
        )
