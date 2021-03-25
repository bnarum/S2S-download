#%%
import pandas as pd
import numpy as np
import xarray as xr
import xarray_extras as xr_e
import gridpp
import json
import os
import progressbar

from S2S.date_helpers import get_forcast_date_cycle
from S2S.gridpp_helpers import make_grid_from_grb, make_points_from_grb
from S2S.file_handling import read_grib_file
from S2S.local_configuration import config

#%% 
# Define the target dates for interpolation. NB: Will use the nearest available dates.
first_wedn = pd.to_datetime("2011-12-28")
last_wedn = pd.to_datetime("2020-01-18") # NB: Change to "2020-12-23" when have forecasts

#%% 
# Define date range of climate data files. 
dates_fc_cycle = get_forcast_date_cycle(
    start_year=2020,
    start_month=1,
    start_day=23,
    end_year=2021,
    end_month=1,
    end_day=18,
)

# Mapping between available dates and the date of the accompanying source file.
hindcast_to_forecast = {}
for fc_d in dates_fc_cycle:
    for hc_d in pd.date_range(
        end=fc_d,  
        periods = np.ceil(((fc_d - first_wedn).days + 1) / 365), 
        freq = pd.DateOffset(years=1)
    ):
        hindcast_to_forecast[hc_d] = fc_d

avail_hc_dates = pd.DatetimeIndex(hindcast_to_forecast.keys()).sort_values()

#%% Make dataframe for each target date with corresponding nearest available hindcast/forecast date and the accompanying date for the source file to read from
df_target = pd.DataFrame({
    'date': pd.date_range(start=first_wedn, end=last_wedn, freq="W-WED"),
    'nearest_hindcast_date': np.nan,
    'forecast_src_date': np.nan,
})
for i in range(len(df_target)):
    hc_d = avail_hc_dates[avail_hc_dates.get_loc(df_target.loc[i, 'date'], method = 'nearest')]
    df_target.loc[i, 'nearest_hindcast_date'] = hc_d
    df_target.loc[i, 'forecast_src_date'] = hindcast_to_forecast[hc_d]

#%% Read target locations
with open(os.path.join(config['BW_DIR'], 'sites.json')) as json_file:
    data_BW = pd.DataFrame(json.load(json_file))

out_points = gridpp.Points(data_BW.lat, data_BW.lon)
out_IDs = data_BW.localityNo

#%%
# Specifications
mdl_vrsn = 'CY46R1_CY47R1'
dirbase = config['S2S_DIR']
cast_type = 'cf'

# Define objects that are reused
grb_data_sst_dummy = read_grib_file(
    dirbase=dirbase, 
    product='hindcast', # forecast
    model_version=mdl_vrsn, 
    var_name_abbr='sst', 
    cast_type=cast_type, 
    date=dates_fc_cycle[0]
) 
valid_points_sst = np.transpose(
    np.isnan(grb_data_sst_dummy.variables['sst'].isel(step = 0, time = 0).data) == 0 
) # NB: Gridpp and xarray have opposite lat-lon axes
in_points_sst = make_points_from_grb(grb_data_sst_dummy, valid_points_sst)

grb_data_sav300_dummy = read_grib_file(
    dirbase=dirbase,
    product='hindcast', # forecast
    model_version=mdl_vrsn,
    var_name_abbr='sal',
    cast_type=cast_type,
    date=dates_fc_cycle[0]
)
valid_points_sav300 = np.transpose(np.isnan(grb_data_sav300_dummy.variables['sav300'].isel(step = 0, time = 0).data) == 0) # NB: Gridpp and xarray have opposite lat-lon axes
in_points_sav300 = make_points_from_grb(grb_data_sav300_dummy, valid_points_sav300)

# Initialize empty xarray to insert in
data_cf_empty = xr.DataArray(dims = ('step', 'locNo', 'date'), 
    coords={
        'step': grb_data_sst_dummy.get_index('step'),
        'locNo': out_IDs,
        'date': df_target['date'],
    }
)

data_converted_cf = xr.Dataset(
    {
        'sst': data_cf_empty.copy(), 
        'sav300': data_cf_empty.copy(), 
    }
)

# Perform interpolations for each file, target date and step
for fc_src_dt in df_target['forecast_src_date'].unique(): # Iterate firstly over files
    # ===========
    # Sea surface temperature
    # ===========
    # grb_data_sst_fc = read_grib_file(
    #     dirbase=dirbase, 
    #     product='forecast', 
    #     model_version=mdl_vrsn, 
    #     var_name_abbr='sst', 
    #     cast_type=cast_type, 
    #     date=fc_src_dt
    # )
    grb_data_sst_hc = read_grib_file(
        dirbase=dirbase, 
        product='hindcast', 
        model_version=mdl_vrsn, 
        var_name_abbr='sst', 
        cast_type=cast_type, 
        date=fc_src_dt
    )
    
    print(f'SST interpolating file-source-date {fc_src_dt.date()}')
    for idx in progressbar.progressbar(np.where(df_target['forecast_src_date'] == fc_src_dt)[0]):
        for curr_step in grb_data_sst_dummy.get_index('step'):
            if df_target['nearest_hindcast_date'][idx] == df_target['forecast_src_date'][idx]:
                in_values_sst = np.transpose(
                    grb_data_sst_fc.sel(step = curr_step).variables['sst'].data
                )[valid_points_sst]
            else:
                in_values_sst = np.transpose(
                    grb_data_sst_hc.sel(
                        step = curr_step, 
                        time = df_target['nearest_hindcast_date'][idx]
                    ).variables['sst'].data
                )[valid_points_sst]

            out_values_sst = gridpp.nearest(
                in_points_sst,
                out_points,
                in_values_sst
            )
            data_converted_cf.sel(
                step = curr_step, 
                date = df_target['date'][idx]
            ).variables['sst'].data[:] = out_values_sst
    print()

    # ===========
    # Salinity
    # ===========
    # grb_data_sav300_fc = read_grib_file(
    #     dirbase=dirbase, 
    #     product='forecast', 
    #     model_version=mdl_vrsn, 
    #     var_name_abbr='sal', 
    #     cast_type=cast_type, 
    #     date=fc_src_dt
    # )
    grb_data_sav300_hc = read_grib_file(
        dirbase=dirbase, 
        product='hindcast', 
        model_version=mdl_vrsn, 
        var_name_abbr='sal', 
        cast_type=cast_type, 
        date=fc_src_dt
    )
    print(f'SAV300 interpolating file-source-date {fc_src_dt.date()}')
    for idx in progressbar.progressbar(np.where(df_target['forecast_src_date'] == fc_src_dt)[0]):
        for curr_step in grb_data_sav300_dummy.get_index('step'):
            if df_target['nearest_hindcast_date'][idx] == df_target['forecast_src_date'][idx]:
                in_values_sav300 = np.transpose(
                    grb_data_sav300_fc.sel(step = curr_step).variables['sav300'].data
                )[valid_points_sav300]
            else:
                in_values_sav300 = np.transpose(
                    grb_data_sav300_hc.sel(
                        step = curr_step, 
                        time = df_target['nearest_hindcast_date'][idx]
                    ).variables['sav300'].data
                )[valid_points_sav300]

            out_values_sav300 = gridpp.nearest(
                in_points_sav300,
                out_points,
                in_values_sav300
            )
            data_converted_cf.sel(
                step = curr_step, 
                date = df_target['date'][idx]
            ).variables['sav300'].data[:] = out_values_sav300    
    print()

#%% Save to file, with and without stepping for different uses
data_converted_cf.to_dataframe().to_csv(
    f'climate_data_wsteps_{cast_type}_{df_target.date.iloc[0].date()}_{df_target.date.iloc[-1].date()}.csv'
)
data_converted_cf.isel(step = 0).drop('step').to_dataframe().to_csv(
    f'climate_data_nsteps_{cast_type}_{df_target.date.iloc[0].date()}_{df_target.date.iloc[-1].date()}.csv'
)

#%%
