#!/usr/bin/env python
# coding: utf-8
"""
Script for reprocessing old CPOL data. It insures compatibility with the CF and
ACDD conventions for the metadata and it removes temporary and obsolete
variables.

@title: reprocess_v2018
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@creation: 13/03/2020
"""
import os
import glob
import uuid
import zipfile
import argparse
import datetime

import dask
import dask.bag as db
import pandas as pd
import xarray as xr


def good_keys():
    """
    List of keys to keep in the final dataset.
    """
    keep_keys = [
        "time",
        "range",
        "azimuth",
        "elevation",
        "radar_echo_classification",
        "radar_estimated_rain_rate",
        "velocity",
        "total_power",
        "reflectivity",
        "cross_correlation_ratio",
        "differential_reflectivity",
        "corrected_differential_reflectivity",
        "differential_phase",
        "corrected_differential_phase",
        "corrected_specific_differential_phase",
        "spectrum_width",
        "signal_to_noise_ratio",
        "sweep_number",
        "fixed_angle",
        "sweep_start_ray_index",
        "sweep_end_ray_index",
        "sweep_mode",
        "prt_mode",
        "prt",
        "nyquist_velocity",
        "unambiguous_range",
        "radar_beam_width_h",
        "radar_beam_width_v",
        "latitude",
        "longitude",
        "altitude",
        "time_coverage_start",
        "time_coverage_end",
        "time_reference",
        "volume_number",
        "platform_type",
        "instrument_type",
        "primary_axis",
    ]

    return keep_keys


def get_metadata(radar_start_date, radar_end_date):
    """
    Generates metadata compatible with CF and ACDD conventions.

    Parameters:
    ===========
    radar_start_date: Timestamp
        Data file start date.
    radar_end_date: Timestamp
        Data file end date.

    Returns:
    ========
    metadata: dict
        Metadata attributes dictionnary.
    """
    maxlon = 132.385
    minlon = 129.703
    maxlat = -10.941
    minlat = -13.552
    origin_altitude = "50"
    origin_latitude = "-12.2491"
    origin_longitude = "131.0444"
    unique_id = str(uuid.uuid4())
    fieldnames = [
        "radar_echo_classification",
        "radar_estimated_rain_rate",
        "velocity",
        "total_power",
        "reflectivity",
        "cross_correlation_ratio",
        "differential_reflectivity",
        "corrected_differential_reflectivity",
        "differential_phase",
        "corrected_differential_phase",
        "corrected_specific_differential_phase",
        "spectrum_width",
        "signal_to_noise_ratio",
    ]

    metadata = {
        "Conventions": "CF-1.6, ACDD-1.3",
        "acknowledgement": "This work has been supported by the U.S. Department of Energy Atmospheric Systems Research Program through the grant DE-SC0014063. Data may be freely distributed.",
        "country": "Australia",
        "creator_email": "valentin.louf@bom.gov.au",
        "creator_name": "Valentin Louf",
        "creator_url": "github.com/vlouf",
        "date_modified": datetime.datetime.now().isoformat(),
        "field_names": ", ".join(fieldnames),
        "geospatial_bounds": f"POLYGON(({minlon:0.6} {minlat:0.6},{minlon:0.6} {maxlat:0.6},{maxlon:0.6} {maxlat:0.6},{maxlon:0.6} {minlat:0.6},{minlon:0.6} {minlat:0.6}))",
        "geospatial_lat_max": f"{maxlat:0.6}",
        "geospatial_lat_min": f"{minlat:0.6}",
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_max": f"{maxlon:0.6}",
        "geospatial_lon_min": f"{minlon:0.6}",
        "geospatial_lon_units": "degrees_east",
        "id": unique_id,
        "institution": "Bureau of Meteorology",
        "instrument": "radar",
        "instrument_name": "CPOL",
        "instrument_type": "radar",
        "keywords": "radar, tropics, Doppler, dual-polarization",
        "licence": "Freely Distributed",
        "naming_authority": "au.org.nci",
        "origin_altitude": origin_altitude,
        "origin_latitude": origin_latitude,
        "origin_longitude": origin_longitude,
        "platform_is_mobile": "false",
        "processing_level": "b1",
        "project": "CPOL",
        "publisher_name": "NCI",
        "publisher_url": "nci.gov.au",
        "references": "doi:10.1175/JTECH-D-18-0007.1",
        "site_name": "Gunn Pt",
        "source": "radar",
        "state": "NT",
        "standard_name_vocabulary": "CF Standard Name Table v71",
        "summary": "Volumetric scan from CPOL dual-polarization Doppler radar (Darwin, Australia)",
        "time_coverage_start": str(radar_start_date),
        "time_coverage_end": str(radar_end_date),
        "time_coverage_duration": "P10M",
        "time_coverage_resolution": "PT10M",
        "title": "radar PPI volume from CPOL",
        "uuid": unique_id,
    }

    return metadata


def mkdir(path):
    """
    Create a directory.

    Parameters:
    ===========
    path: str
        Path to directory
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    return None


def remove(flist):
    """
    Remove files.

    Parameters:
    ===========
    flist: list
        List of files to remove
    """
    for f in flist:
        if f is None:
            continue
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
    return None


def extract_zip(inzip, path):
    """
    Extract all members from the archive.

    Parameters:
    ===========
    inzip: str
        Zip file to extract
    path: str
        Path specifies a directory to extract to.

    Returns:
    ========
    dates: str
        Date string for the zipfile
    namelist: list
        List of file names in the archive.
    """
    dates = os.path.basename(inzip).replace(".zip", "")
    with zipfile.ZipFile(inzip) as zid:
        zid.extractall(path=path)
        namelist = [os.path.join(path, f) for f in zid.namelist()]
    return dates, namelist


def update_dataset(radar_file, path):
    """
    Processing to update the old CPOL level 1b data.

    Parameter:
    ==========
    radar_file: str
        Path to input radar file.
    path: str
        Path specifies a directory to write the output file to.

    Return:
    =======
    radar_file: str
        Return None if processing failed, otherwise it returns the path to
        input radar file, so that you can delete the input latter.
    """
    dset = xr.open_dataset(radar_file)
    radar_start_date = dset.time[0].values
    radar_end_date = dset.time[-1].values

    fname = "twp10cpolppi.b1.{}00.nc".format(
        pd.Timestamp(radar_start_date).strftime("%Y%m%d.%H%M")
    )
    outfilename = os.path.join(path, fname)
    if os.path.exists(outfilename):
        print(f"File already exists.")
        return None

    keep_keys = good_keys()
    keylist = [k for k in dset.variables.keys()]
    for k in keylist:
        if k not in keep_keys:
            dset = dset.drop(k)

    metadata = get_metadata(radar_start_date, radar_end_date)
    metadata["product_version"] = "v" + dset.attrs["product_version"]
    metadata["version"] = "v" + dset.attrs["product_version"]
    metadata["date_created"] = (dset.attrs["created"],)
    metadata["history"] = (
        "created by Valentin Louf on raijin.nci.org.au at "
        + dset.attrs["created"]
        + " using Py-ART",
    )
    dset.attrs = metadata

    dset.to_netcdf(
        outfilename, encoding={k: {"zlib": True} for k in dset.variables.keys()}
    )
    if not os.path.exists(outfilename):
        print(f"Output file does not exist !!!.")
        return None

    del dset
    return radar_file


def main():
    """
    1/ List all zip files for a given year,
    2/ Extract one zip file (each representing one day) at a time,
    3/ Processing (updating) and removing obsolete keys,
    4/ Removing extracted files.
    """
    zipdir = "/scratch/kl02/vhl548"
    ziplist = sorted(
        glob.glob(f"/g/data/hj10/admin/cpol_level_1b/v2018/ppi/{YEAR}/*.zip")
    )
    if len(ziplist) == 0:
        print("No file found.")
        return None

    for zfile in ziplist:
        dates, namelist = extract_zip(zfile, zipdir)
        outpath = f"/scratch/kl02/vhl548/tmpcpol/{dates}"
        mkdir(outpath)
        bag = db.from_sequence([(n, outpath) for n in namelist]).starmap(update_dataset)
        rslt = bag.compute()
        remove(rslt)

    return None


if __name__ == "__main__":
    parser_description = "Update and re-encode version v2018 of 1b CPOL product."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-y", "--year", dest="year", type=int, help="Year to process.", required=True
    )

    args = parser.parse_args()
    YEAR = args.year
    main()
