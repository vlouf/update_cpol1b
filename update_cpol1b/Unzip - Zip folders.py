#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import uuid
import datetime
import zipfile

import tqdm
import netCDF4
import pandas as pd
import xarray as xr

import dask
import dask.bag as db
from dask.diagnostics import ProgressBar


# In[2]:


def good_keys():
    keep_keys = ['time',
                 'range',
                 'azimuth',
                 'elevation',
                 'radar_echo_classification',
                 'radar_estimated_rain_rate',
                 'velocity', 
                 'total_power',
                 'reflectivity',
                 'cross_correlation_ratio',
                 'differential_reflectivity',
                 'corrected_differential_reflectivity',
                 'differential_phase',
                 'corrected_differential_phase',
                 'corrected_specific_differential_phase',
                 'spectrum_width',
                 'signal_to_noise_ratio',
                 'sweep_number',
                 'fixed_angle',
                 'sweep_start_ray_index',
                 'sweep_end_ray_index',
                 'sweep_mode',
                 'prt_mode',
                 'prt',
                 'nyquist_velocity',
                 'unambiguous_range',
                 'radar_beam_width_h',
                 'radar_beam_width_v',
                 'latitude',
                 'longitude',
                 'altitude',
                 'time_coverage_start',
                 'time_coverage_end',
                 'time_reference',
                 'volume_number',
                 'platform_type',
                 'instrument_type',
                 'primary_axis']
    
    return keep_keys


# In[3]:


def get_metadata(radar_start_date, radar_end_date):
    maxlon = 132.385
    minlon = 129.703
    maxlat = -10.941
    minlat = -13.552
    origin_altitude = '50'
    origin_latitude = '-12.2491'
    origin_longitude = '131.0444'
    unique_id = str(uuid.uuid4())
    fieldnames = ['radar_echo_classification',
                  'radar_estimated_rain_rate',
                  'velocity', 
                  'total_power',
                  'reflectivity',
                  'cross_correlation_ratio',
                  'differential_reflectivity',
                  'corrected_differential_reflectivity',
                  'differential_phase',
                  'corrected_differential_phase',
                  'corrected_specific_differential_phase',
                  'spectrum_width',
                  'signal_to_noise_ratio']

    metadata = {'Conventions': "CF-1.6, ACDD-1.3",
                'acknowledgement': 'This work has been supported by the U.S. Department of Energy Atmospheric Systems Research Program through the grant DE-SC0014063. Data may be freely distributed.',
                'country': 'Australia',
                'creator_email': 'valentin.louf@bom.gov.au',
                'creator_name': 'Valentin Louf',
                'creator_url': 'github.com/vlouf',                
                'date_modified': datetime.datetime.now().isoformat(),
                'field_names': ", ".join(fieldnames),
                "geospatial_bounds": f"POLYGON(({minlon:0.6} {minlat:0.6},{minlon:0.6} {maxlat:0.6},{maxlon:0.6} {maxlat:0.6},{maxlon:0.6} {minlat:0.6},{minlon:0.6} {minlat:0.6}))",
                'geospatial_lat_max': f'{maxlat:0.6}',
                'geospatial_lat_min': f'{minlat:0.6}',
                'geospatial_lat_units': "degrees_north",
                'geospatial_lon_max': f'{maxlon:0.6}',
                'geospatial_lon_min': f'{minlon:0.6}',
                'geospatial_lon_units': "degrees_east",                
                'id': unique_id,
                'institution': 'Bureau of Meteorology',
                'instrument': 'radar',
                'instrument_name': 'CPOL',
                'instrument_type': 'radar',
                'keywords': 'radar, tropics, Doppler, dual-polarization',
                'licence': "Freely Distributed",
                'naming_authority': 'au.org.nci',
                'origin_altitude': origin_altitude,
                'origin_latitude': origin_latitude,
                'origin_longitude': origin_longitude,
                'platform_is_mobile': 'false',
                'processing_level': 'b1',
                'project': "CPOL",
                'publisher_name': "NCI",
                'publisher_url': "nci.gov.au",                
                'references': 'doi:10.1175/JTECH-D-18-0007.1',
                'site_name': 'Gunn Pt',
                'source': 'radar',
                'state': "NT",
                'standard_name_vocabulary': 'CF Standard Name Table v71',
                'summary': "Volumetric scan from CPOL dual-polarization Doppler radar (Darwin, Australia)",
                'time_coverage_start': str(radar_start_date),
                'time_coverage_end': str(radar_end_date),
                'time_coverage_duration': "P10M",
                'time_coverage_resolution': "PT10M",
                'title': "radar PPI volume from CPOL",
                'uuid': unique_id}           
    
    return metadata


# In[4]:


def mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    
    return None


# In[5]:


def extract_zip(inzip, path):
    dates = os.path.basename(inzip).replace('.zip', '')
    with zipfile.ZipFile(inzip) as zid:
        zid.extractall(path=path)
        namelist = [os.path.join(path, f) for f in zid.namelist()]
    return dates, namelist


# In[6]:


def update_dataset(radar_file, path):
    dset = xr.open_dataset(radar_file)
    radar_start_date = dset.time[0].values
    radar_end_date = dset.time[-1].values

    fname = "twp10cpolppi.b1.{}00.nc".format(pd.Timestamp(radar_start_date).strftime("%Y%m%d.%H%M"))
    outfilename = os.path.join(path, fname)
    if os.path.exists(outfilename):
        print(f'File already exists.')
        return None
    
    keep_keys = good_keys()
    keylist = [k for k in dset.variables.keys()]
    for k in keylist:
        if k not in keep_keys:
            dset = dset.drop(k)

    metadata = get_metadata(radar_start_date, radar_end_date)
    metadata['product_version'] = "v" + dset.attrs['product_version']
    metadata['version'] = "v" + dset.attrs['product_version']
    metadata['date_created'] = dset.attrs['created'],
    metadata['history'] = "created by Valentin Louf on raijin.nci.org.au at " + dset.attrs['created'] + " using Py-ART",
    dset.attrs = metadata

    dset.to_netcdf(outfilename, encoding={k:{'zlib': True} for k in dset.variables.keys()})    
    if not os.path.exists(outfilename):
        print(f'Output file does not exist !!!.')
        return None
    
    del dset
    return radar_file


# In[7]:


def remove(flist):
    for f in flist:
        if f is None:
            continue
        try:
            os.remove(f)
        except FileNotFoundError:
            pass        
    return None


# In[8]:


zipdir = '/scratch/kl02/vhl548'
ziplist = sorted(glob.glob('/g/data/hj10/admin/cpol_level_1b/v2018/ppi/1998/*.zip'))


# In[8]:


# dates, namelist = extract_zip(ziplist[0], zipdir)
# outpath = f'/scratch/kl02/vhl548/tmpcpol/{dates}'
# mkdir(outpath)


# In[9]:


for zfile in tqdm.tqdm_notebook(ziplist[1:]):
    dates, namelist = extract_zip(zfile, zipdir)
    outpath = f'/scratch/kl02/vhl548/tmpcpol/{dates}'
    mkdir(outpath)
    bag = db.from_sequence([(n, outpath) for n in namelist]).starmap(update_dataset)
    with ProgressBar():
        rslt = bag.compute()

    remove(rslt)


# In[ ]:




