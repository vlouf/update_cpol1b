"""
CPOL Level 1b update products.

@title: update_cpol1b
@author: Valentin Louf <valentin.louf@monash.edu>
@copyright: Valentin Louf (2017-)
@institution: Bureau of Meteorology and Monash University
"""
import os
import glob
import copy
import uuid
import argparse
import datetime
import warnings
import traceback

import pyart
import cftime
import numpy as np
import xarray as xr

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_metadata():
    today = datetime.datetime.utcnow()
    maxlon = "132.385"
    minlon = "129.703"
    maxlat = "-10.941"
    minlat = "-13.552"
    origin_altitude = "50"
    origin_latitude = "-12.2488"
    origin_longitude = "131.0444"

    metadata = dict()
    metadata["Conventions"] = "CF/Radial instrument_parameters"
    metadata[
        "acknowledgement"
    ] = "This work has been supported by the U.S. Department of Energy Atmospheric Systems Research Program through the grant DE-SC0014063. Data may be freely distributed."
    metadata["country"] = "Australia"
    metadata["creator_email"] = "valentin.louf@monash.edu"
    metadata["creator_name"] = "Valentin Louf"
    metadata["geospatial_bounds"] = f"({minlon}, {maxlon}, {minlat}, {maxlat})"
    metadata["geospatial_lat_max"] = maxlat
    metadata["geospatial_lat_min"] = minlat
    metadata["geospatial_lat_units"] = "degrees_north"
    metadata["geospatial_lon_max"] = maxlon
    metadata["geospatial_lon_min"] = minlon
    metadata["geospatial_lon_units"] = "degrees_east"
    metadata["history"] = (
        "created by Valentin Louf on raijin.nci.org.au at "
        + today.isoformat()
        + " using Py-ART"
    )
    metadata["institution"] = "Monash University and Australian Bureau of Meteorology"
    metadata["instrument_name"] = "CPOL"
    metadata["instrument_type"] = "radar"
    metadata["naming_authority"] = "au.org.nci"
    metadata["origin_altitude"] = origin_altitude
    metadata["origin_latitude"] = origin_latitude
    metadata["origin_longitude"] = origin_longitude
    metadata["platform_is_mobile"] = "false"
    metadata["processing_level"] = "b1"
    metadata["publisher_name"] = "NCI"
    metadata["publisher_url"] = "nci.gov.au"
    metadata["references"] = "cf. doi:10.1175/JTECH-D-18-0007.1"
    metadata["site_name"] = "Gunn_Pt"
    metadata["source"] = "rapic"
    metadata["state"] = "NT"
    metadata["title"] = "radar PPI volume from CPOL"
    metadata["uuid"] = str(uuid.uuid4())
    metadata["version"] = "1.3"

    return metadata


def update_data(infile):
    """
    Processing to update the old CPOL level 1b data.

    Parameter:
    ==========
    input_file: str
        Path to input radar file.
    """
    radar = pyart.io.read(infile)

    radar_start_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])

    daystr = radar_start_date.strftime("%Y%m%d")
    filename = "twp10cpolppi.b1.{}00.nc".format(
        radar_start_date.strftime("%Y%m%d.%H%M")
    )
    outdir = os.path.join(OUTPATH, str(radar_start_date.year))
    try:
        os.mkdir(outdir)
    except FileExistsError:
        pass

    outdir_day = os.path.join(outdir, daystr)
    try:
        os.mkdir(outdir_day)
    except FileExistsError:
        pass

    outfilename = os.path.join(outdir_day, filename)

    # Get original level 1a file for copying intstrument parameters
    datestr = radar_start_date.strftime("%Y%m%d.%H%M")
    yrstr = radar_start_date.strftime("%Y")
    fone = "/g/data/hj10/cpol_level_1a/v2019/ppi/{}/{}/twp10cpolppi.a1.{}00.nc".format(
        yrstr, daystr, datestr
    )
    if not os.path.isfile(fone):
        print(f"{fone} is missing.")
        fone = "/g/data/hj10/cpol_level_1a/v2019/ppi/2017/20170304/twp10cpolppi.a1.20170304.000000.nc"
    oned = pyart.io.read(fone)

    radar.instrument_parameters = oned.instrument_parameters
    radar.metadata = get_metadata()
    radar.radar_calibration = None

    # Drop Obsolete keys
    keys_drop = [
        "temperature",
        "specific_attenuation_reflectivity",
        "specific_attenuation_differential_reflectivity",
        "velocity_texture",
    ]

    klist = list(radar.fields.keys())
    for key in klist:
        if key in keys_drop:
            radar.fields.pop(key)

    # Rename the reflectivity
    try:
        radar.add_field("corrected_reflectivity", radar.fields.pop("reflectivity"))
        radar.fields["corrected_reflectivity"]["data"] = radar.fields[
            "corrected_reflectivity"
        ]["data"].filled(np.NaN)
        radar.fields["corrected_reflectivity"]["data"] = np.ma.masked_invalid(
            radar.fields["corrected_reflectivity"]["data"]
        )
        np.ma.set_fill_value(radar.fields["corrected_reflectivity"]["data"], np.NaN)
    except Exception:
        traceback.print_exc()
        pass

    # Rename the 2 Velocity fields
    try:
        radar.fields["velocity"] = radar.fields.pop("raw_velocity")
        radar.add_field(
            "corrected_velocity", radar.fields.pop("region_dealias_velocity")
        )
    except Exception:
        pass

    # Make sure echo class is an integer array and fix its fillvalue.
    try:
        radar.fields["radar_echo_classification"]["data"] = radar.fields[
            "radar_echo_classification"
        ]["data"].astype(np.int32)
        np.ma.set_fill_value(radar.fields["radar_echo_classification"]["data"], -9999)
        radar.fields["radar_echo_classification"]["_FillValue"] = -9999
    except Exception:
        pass

    radar.fields["radar_estimated_rain_rate"]["_Least_significant_digit"] = 2

    try:
        radar.fields["NW"].pop("standard_name")
        np.ma.set_fill_value(radar.fields["NW"]["data"], np.NaN)
        radar.fields["NW"]["_FillValue"] = np.NaN
        radar.fields["NW"]["_Least_significant_digit"] = 2
        radar.fields["NW"]["data"] = np.ma.masked_invalid(radar.fields["NW"]["data"])
    except Exception:
        pass

    klist_pr = [
        ("D0", 2, np.NaN),
        ("velocity", 2, np.NaN),
        ("total_power", 2, np.NaN),
        ("corrected_reflectivity", 2, np.NaN),
        ("cross_correlation_ratio", 4, np.NaN),
        ("corrected_differential_reflectivity", 4, np.NaN),
        ("corrected_differential_phase", 4, np.NaN),
        ("corrected_specific_differential_phase", 4, np.NaN),
        ("differential_reflectivity", 4, np.NaN),
        ("differential_phase", 4, np.NaN),
        ("spectrum_width", 4, np.NaN),
        ("signal_to_noise_ratio", 2, np.NaN),
        ("corrected_velocity", 2, np.NaN),
    ]

    for key, least_digit, fvalue in klist_pr:
        if key not in klist:
            continue
        try:
            np.ma.set_fill_value(radar.fields[key]["data"], fvalue)
            radar.fields[key]["data"] = radar.fields[key]["data"].astype(np.float32)
            radar.fields[key]["_Least_significant_digit"] = least_digit
            radar.fields[key]["_FillValue"] = fvalue
        except Exception:
            traceback.print_exc()
            continue

    # Remove wrongfull attributes.
    bad_attr = ["grid_mapping", "coordinates", "sampling_ratio"]
    for k in radar.fields.keys():
        for badk in bad_attr:
            try:
                radar.fields[k].pop(badk)
            except KeyError:
                pass

    pyart.io.write_cfradial(outfilename, radar)
    return None


def process_one_file(input_file):
    """
    Update the data for one file only.

    Parameter:
    ==========
    input_file: str
        Path to input radar file.
    """
    if not os.path.isfile(input_file):
        print(f"The input you've given is not a file {input_file}. Doing nothing.")
        return None

    print(f"This will process only one file: {input_file}.")
    print(f"The ouput directory will be: {OUTPATH}.")
    update_data(input_file)
    print("Processing done.")

    return None


def main():
    indir = os.path.join(INPATH, str(YEAR), "**", "*.nc")
    flist = sorted(glob.glob(indir))

    print(f"Found {len(flist)} files.")
    if len(flist) == 0:
        raise FileNotFoundError(f"No file found in {indir}")

    for flist_chunk in chunks(flist, 16):
        with ProcessPool() as pool:
            future = pool.map(update_data, flist_chunk, timeout=60)
            iterator = future.result()

            while True:
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("function took longer than %d seconds" % error.args[1])
                except ProcessExpired as error:
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except Exception as error:
                    print("function raised %s" % error)

    return None


if __name__ == "__main__":
    INPATH = "/g/data2/rr5/CPOL_radar/CPOL_level_1b/PPI/"

    parser_description = "Update and re-encode previous version of 1b CPOL product."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-y",
        "--year",
        dest="year",
        default=2017,
        type=int,
        help="Year to process. Uses multiprocessing to do so. Superseeded by the --file argument.",
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        type=str,
        help="Input file. This option superseeds the --year argument.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        default="/g/data/hj10/cpol_level_1b/v2018/ppi",
        type=str,
        help="Output directory",
    )

    args = parser.parse_args()
    YEAR = args.year
    OUTPATH = args.output
    INFILE = args.file

    with warnings.catch_warnings():
        # Turn off warnings.
        warnings.simplefilter("ignore")
        if INFILE is None:
            main()
        else:
            process_one_file(INFILE)
