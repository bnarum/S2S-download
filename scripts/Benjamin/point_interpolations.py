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

from time import time

#%% 
# Define the target dates for interpolation. NB: Will use the nearest available dates.
first_wedn = pd.to_datetime("2011-12-28")
# first_wedn = pd.to_datetime("2019-01-01")
last_wedn = pd.to_datetime("2020-12-23")

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
cast_type = 'pf'

# Load to get format of output file
grb_data_sst_dummy = read_grib_file(
    dirbase=dirbase, 
    product='forecast',
    model_version=mdl_vrsn, 
    var_name_abbr='sst', 
    cast_type=cast_type, 
    date=dates_fc_cycle[0]
)

# Load to get format of output file
valid_points_sst = ~np.isnan(
    grb_data_sst_dummy.variables['sst'].isel(step = 0, number=0).data
) & ~np.isnan(
    grb_data_sst_dummy.variables['sst'].isel(step = 15, number=0).data
)
# NB: Gridpp and xarray have opposite lat-lon axes
in_points_sst = make_points_from_grb(grb_data_sst_dummy, valid_points_sst) # NB: Transpose valid_points again?

# Axes numbers
potential_steps = grb_data_sst_dummy.get_index('step')
if cast_type == 'cf':
    potential_numbers = np.array([0])
elif cast_type == 'pf':
    potential_numbers = grb_data_sst_dummy.get_index('number')[0:10]

grb_data_sst_dummy.close() # Free memory

grb_data_sav300_dummy = read_grib_file(
    dirbase=dirbase, 
    product='forecast',
    model_version=mdl_vrsn, 
    var_name_abbr='sal', 
    cast_type=cast_type, 
    date=dates_fc_cycle[0]
)
# NB: Gridpp and xarray have opposite lat-lon axes
valid_points_sav300 = ~np.isnan(
    grb_data_sav300_dummy.variables['sav300'].isel(step = 0, number=0).data
) & ~np.isnan(
    grb_data_sav300_dummy.variables['sav300'].isel(step = 16, number=0).data
)
in_points_sav300 = make_points_from_grb(grb_data_sav300_dummy, valid_points_sav300)  # NB: Transpose valid_points again?
grb_data_sav300_dummy.close() # Free memory

print('Both files closed')

# Initialize empty xarray to insert in. NB: Making one to copy kills memory.
dims_tuple = (
    'number', 
    'step',
    'locNo', 
    'date', 
)
coords_dict = {
    'number': potential_numbers,
    'step': potential_steps,
    'locNo': out_IDs,
    'date': df_target['date'],
}
data_empty_sst = xr.DataArray(
    np.empty((
        len(potential_numbers), 
        len(potential_steps),
        len(out_IDs), 
        len(df_target['date']), 
    )) * np.nan, # NB: Must provide empty data, memory issue.
    dims = dims_tuple, 
    coords=coords_dict,
)
data_empty_sav300 = xr.DataArray(
    np.empty((
        len(potential_numbers), 
        len(potential_steps),
        len(out_IDs), 
        len(df_target['date']), 
    )) * np.nan, # NB: Must provide empty data, memory issue.
    dims = dims_tuple, 
    coords=coords_dict,
)

data_converted = xr.Dataset(
    {
        'sst': data_empty_sst, 
        'sav300': data_empty_sav300, 
    }
)

print('Output dataset initialized')

# #%%
# Perform interpolations for each file, target date and step
# for fc_src_dt in progressbar.progressbar(df_target['forecast_src_date'].unique()): # Iterate firstly over files
#     # ===========
#     # Sea surface temperature
#     # ===========
#     grb_data_sst_fc = read_grib_file(
#         dirbase=dirbase, 
#         product='forecast', 
#         model_version=mdl_vrsn, 
#         var_name_abbr='sst', 
#         cast_type=cast_type, 
#         date=fc_src_dt,
#         verbosity = 0
#     )
#     grb_data_sst_hc = read_grib_file(
#         dirbase=dirbase, 
#         product='hindcast', 
#         model_version=mdl_vrsn, 
#         var_name_abbr='sst', 
#         cast_type=cast_type, 
#         date=fc_src_dt,
#         verbosity = 0
#     )
#     for idx in np.where(df_target['forecast_src_date'] == fc_src_dt)[0]:
#         print(idx)
#         for curr_step in data_converted.get_index('step'):
#             for curr_num in data_converted.get_index('number'):
#                 start_time = time()
#                 if df_target['nearest_hindcast_date'][idx] == df_target['forecast_src_date'][idx]:
#                     if curr_num == 0:
#                         in_values_sst = grb_data_sst_fc.sel(step = curr_step).variables['sst'].data[valid_points_sst]
#                     else:
#                         in_values_sst = grb_data_sst_fc.sel(step=curr_step, number=curr_num).variables['sst'].data[valid_points_sst]
#                 else:
#                     if curr_num == 0:
#                         in_values_sst = grb_data_sst_hc.sel(
#                             step = curr_step, 
#                             time = df_target['nearest_hindcast_date'][idx]
#                         ).variables['sst'].data[valid_points_sst]
#                     else:
#                         in_values_sst = grb_data_sst_hc.sel(
#                             step=curr_step, 
#                             time=df_target['nearest_hindcast_date'][idx],
#                             number=curr_num,
#                         ).variables['sst'].data[valid_points_sst]

#                 print(f'Found in_values {time() - start_time} s')

#                 out_values_sst = gridpp.nearest(
#                     in_points_sst,
#                     out_points,
#                     in_values_sst
#                 )

#                 print(f'Interpolated out_values {time() - start_time} s')

#                 data_converted.sel(
#                     step = curr_step, 
#                     date = df_target['date'][idx],
#                     number=curr_num,
#                 ).variables['sst'].data[:] = out_values_sst
#                 print(f'Stored values {time() - start_time} s\n')

#     grb_data_sst_fc.close()
#     grb_data_sst_hc.close()

#     # ===========
#     # Salinity
#     # ===========
#     grb_data_sav300_fc = read_grib_file(
#         dirbase=dirbase, 
#         product='forecast', 
#         model_version=mdl_vrsn, 
#         var_name_abbr='sal', 
#         cast_type=cast_type, 
#         date=fc_src_dt,
#         verbosity = 0
#     )
#     grb_data_sav300_hc = read_grib_file(
#         dirbase=dirbase, 
#         product='hindcast', 
#         model_version=mdl_vrsn, 
#         var_name_abbr='sal', 
#         cast_type=cast_type, 
#         date=fc_src_dt,
#         verbosity = 0
#     )
#     for idx in np.where(df_target['forecast_src_date'] == fc_src_dt)[0]:
#         print(idx)
#         for curr_step in grb_data_sav300_fc.get_index('step'):
#             for curr_num in potential_numbers:
#                 if df_target['nearest_hindcast_date'][idx] == df_target['forecast_src_date'][idx]:
#                     if curr_num == 0:
#                         in_values_sav300 = grb_data_sav300_fc.sel(step = curr_step).variables['sav300'].data[valid_points_sav300]
#                     else:
#                         in_values_sav300 = grb_data_sav300_fc.sel(step = curr_step, number=curr_num).variables['sav300'].data[valid_points_sav300]
#                 else:
#                     if curr_num == 0:
#                         in_values_sav300 = grb_data_sav300_hc.sel(
#                             step = curr_step, 
#                             time = df_target['nearest_hindcast_date'][idx]
#                         ).variables['sav300'].data[valid_points_sav300]
#                     else:
#                         in_values_sav300 = grb_data_sav300_hc.sel(
#                             step = curr_step, 
#                             time = df_target['nearest_hindcast_date'][idx],
#                             number=curr_num,
#                         ).variables['sav300'].data[valid_points_sav300]

#                 out_values_sav300 = gridpp.nearest(
#                     in_points_sav300,
#                     out_points,
#                     in_values_sav300
#                 )
#                 data_converted.sel(
#                     step = curr_step, 
#                     date = df_target['date'][idx],
#                     number=curr_num,
#                 ).variables['sav300'].data[:] = out_values_sav300
#     grb_data_sav300_fc.close()
#     grb_data_sav300_hc.close()





#%%
lonlon_sst, latlat_sst  = np.meshgrid(grb_data_sst_dummy.get_index('longitude'), grb_data_sst_dummy.get_index('latitude'))
df_nearest_loc_indi_sst = pd.DataFrame([
    np.unravel_index(
        np.argmin(
            np.sqrt((lonlon_sst - lon)**2 + (latlat_sst - lat)**2) + 10000 * ~valid_points_sst,
        ),
        lonlon_sst.shape
    )
    for lon, lat  in zip(out_points.get_lons(), out_points.get_lats())
], columns=['lat', 'lon'])

lonlon_sav300, latlat_sav300 = np.meshgrid(grb_data_sav300_dummy.get_index('longitude'), grb_data_sav300_dummy.get_index('latitude'))
df_nearest_loc_indi_sav300 = pd.DataFrame([
    np.unravel_index(
        np.argmin(
            np.sqrt((lonlon_sav300 - lon)**2 + (latlat_sav300 - lat)**2) + 10000 * ~valid_points_sav300,
        ),
        lonlon_sav300.shape
    )
    for lon, lat  in zip(out_points.get_lons(), out_points.get_lats())
], columns=['lat', 'lon'])

# %%
# Perform interpolations for each file, target date and step
for fc_src_dt in progressbar.progressbar(df_target['forecast_src_date'].unique()): # Iterate firstly over files
    # ===========
    # Sea surface temperature
    # ===========
    grb_data_sst_fc = read_grib_file(
        dirbase=dirbase, 
        product='forecast', 
        model_version=mdl_vrsn, 
        var_name_abbr='sst', 
        cast_type=cast_type, 
        date=fc_src_dt,
        verbosity = 0
    )
    grb_data_sst_hc = read_grib_file(
        dirbase=dirbase, 
        product='hindcast', 
        model_version=mdl_vrsn, 
        var_name_abbr='sst', 
        cast_type=cast_type, 
        date=fc_src_dt,
        verbosity = 0
    )
    for idx in np.where(df_target['forecast_src_date'] == fc_src_dt)[0]:
        if df_target['nearest_hindcast_date'][idx] == df_target['forecast_src_date'][idx]:
            data_converted['sst'].loc[dict(
                date = df_target['date'][idx],
                number=potential_numbers,
            )] = grb_data_sst_fc['sst'].loc[dict(
                number=potential_numbers,
            )].isel(
                latitude=xr.DataArray(df_nearest_loc_indi_sst.lat, dims = "z"),
                longitude=xr.DataArray(df_nearest_loc_indi_sst.lon, dims = "z"),
            ).data
        else:
            data_converted['sst'].loc[dict(
                date = df_target['date'][idx],
                number=potential_numbers,
            )] = grb_data_sst_hc['sst'].loc[dict(
                number=potential_numbers, 
                time=df_target['nearest_hindcast_date'][idx],
            )].isel(
                latitude=xr.DataArray(df_nearest_loc_indi_sst.lat, dims = "z"),
                longitude=xr.DataArray(df_nearest_loc_indi_sst.lon, dims = "z"),
            ).data

    grb_data_sst_fc.close()
    grb_data_sst_hc.close()

    # ========
    # Salinity
    # ========
    grb_data_sav300_fc = read_grib_file(
        dirbase=dirbase, 
        product='forecast', 
        model_version=mdl_vrsn, 
        var_name_abbr='sal', 
        cast_type=cast_type, 
        date=fc_src_dt,
        verbosity = 0
    )
    grb_data_sav300_hc = read_grib_file(
        dirbase=dirbase, 
        product='hindcast', 
        model_version=mdl_vrsn, 
        var_name_abbr='sal', 
        cast_type=cast_type, 
        date=fc_src_dt,
        verbosity = 0
    )
    for idx in np.where(df_target['forecast_src_date'] == fc_src_dt)[0]:
        if df_target['nearest_hindcast_date'][idx] == df_target['forecast_src_date'][idx]:
            data_converted['sav300'].loc[dict(
                date = df_target['date'][idx],
                number=potential_numbers,
            )] = grb_data_sav300_fc['sav300'].loc[dict(
                number=potential_numbers,
            )].isel(
                latitude=xr.DataArray(df_nearest_loc_indi_sav300.lat, dims = "z"),
                longitude=xr.DataArray(df_nearest_loc_indi_sav300.lon, dims = "z"),
            ).data
        else:
            data_converted['sav300'].loc[dict(
                date = df_target['date'][idx],
                number=potential_numbers,
            )] = grb_data_sav300_hc['sav300'].loc[dict(
                number=potential_numbers, 
                time=df_target['nearest_hindcast_date'][idx],
            )].isel(
                latitude=xr.DataArray(df_nearest_loc_indi_sav300.lat, dims = "z"),
                longitude=xr.DataArray(df_nearest_loc_indi_sav300.lon, dims = "z"),
            ).data

    grb_data_sav300_fc.close()
    grb_data_sav300_hc.close()

# NB: The data was empty

#%% Save to file, with and without stepping for different uses
data_converted.to_dataframe().to_csv(
    f'climate_data_wsteps_{cast_type}_{df_target.date.iloc[0].date()}_{df_target.date.iloc[-1].date()}.csv'
)
data_converted.isel(step = 0).drop('step').to_dataframe().to_csv(
    f'climate_data_nsteps_{cast_type}_{df_target.date.iloc[0].date()}_{df_target.date.iloc[-1].date()}.csv'
)


#%%

# #%%
# grb_data_sst_dummy = read_grib_file(
#     dirbase=dirbase, 
#     product='hindcast',
#     model_version=mdl_vrsn, 
#     var_name_abbr='sst',
#     cast_type=cast_type, 
#     date=df_target['forecast_src_date'].unique()[1]
# )
# #%%
# idx_t = np.where(df_target['forecast_src_date'].unique()[1] == df_target['forecast_src_date'])[0][0]

# #%% 
# data_converted['sst'].loc[dict(
#     date = df_target['date'][idx_t],
#     number=potential_numbers,
# )] = grb_data_sst_dummy['sst'].loc[dict(
#     number=potential_numbers, 
#     time=df_target['nearest_hindcast_date'][idx_t],
# )].isel(
#     latitude=xr.DataArray(df_nearest_loc_indi_sst.lat, dims = "z"),
#     longitude=xr.DataArray(df_nearest_loc_indi_sst.lon, dims = "z"),
# ).data

# # [:, :, df_nearest_loc_indi_sst.lon, df_nearest_loc_indi_sst.lat]

# #%%
# data_converted['sst'].loc[dict(
#     date = df_target['date'][idx_t],
#     number=potential_numbers,
# )]

# #%%
# data_converted.sel(
#     date = df_target['date'][idx_t],
#     number=potential_numbers,
# ).variables['sst'].data

# # %%
# #%%
# grb_data_sst_dummy.variables['sst']

# # %%
# data_converted.sel(
#     date = df_target['date'][0], 
#     number=potential_numbers,
# ).variables['sst'].data.shape


# #%%
# grb_data_sst_dummy.sel(
#     number = potential_numbers,
# ).variables['sst'].data[:, :, df_nearest_loc_indi_sst.lon, df_nearest_loc_indi_sst.lat].shape

# #%%

# %%
