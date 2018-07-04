#!/usr/bin/env python
'''
                                QUERY TESS TARGETS FROM RA & DEC V.04

                                This script makes a query in Gaia DR2
                                or Hipparcos (if not found in DR2)
                                and finds the corresponding object in
                                2MASS, WISE, PAN-STARRS, SDSS, and SkyMapper.
                                The input is RA and DEC with the size of
                                the field of view of the telescope.
                                The purpose is to identify the source observed
                                by combining information from all catalogs.

                                (This version of the script deleted the
                                position propagation function using parallax
                                and radial velocity. That function is replaced
                                with a simpler function using only the Gaia
                                proper motion and the epoch of observation
                                of the comparison catalog.)
'''

# =============================================== Import Packages ======================================================

import numpy as np
from astroquery.vizier import Vizier
import astropy.coordinates as coord
from astropy.time import Time
from astropy.coordinates import SkyCoord
import math
import warnings

warnings.filterwarnings('ignore')  # ignore warnings
import sys
from termcolor import colored
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import uncertainties as err
from uncertainties.umath import *
import configparser
import re
import argparse

# ============================================ Read Configuration File =================================================

config = configparser.ConfigParser()
config.read('config.ini')

catalog_GAIA = config['CATALOGS']['GAIA']
catalog_WISE = config['CATALOGS']['WISE']
catalog_2MASS = config['CATALOGS']['2MASS']
catalog_PANSTARRS = config['CATALOGS']['PANSTARRS']
catalog_SDSS = config['CATALOGS']['SDSS']
catalog_SKYMAPPER = config['CATALOGS']['SKYMAPPER']

ref_epoch_WISE = float(config['EPOCH']['WISE_REF'])

G_mag_zero_flux_VEGA_GAIA = float(config['GAIA_FILTERS']['G_mag_zero_flux_VEGA_GAIA'])
BP_mag_zero_flux_VEGA_GAIA = float(config['GAIA_FILTERS']['BP_mag_zero_flux_VEGA_GAIA'])
RP_mag_zero_flux_VEGA_GAIA = float(config['GAIA_FILTERS']['RP_mag_zero_flux_VEGA_GAIA'])

G_mag_lambda_eff_GAIA = float(config['GAIA_FILTERS']['G_mag_lambda_eff_GAIA'])
BP_mag_lambda_eff_GAIA = float(config['GAIA_FILTERS']['BP_mag_lambda_eff_GAIA'])
RP_mag_lambda_eff_GAIA = float(config['GAIA_FILTERS']['RP_mag_lambda_eff_GAIA'])

J_mag_zero_flux_VEGA_2MASS = float(config['2MASS_FILTERS']['J_mag_zero_flux_VEGA_2MASS'])
H_mag_zero_flux_VEGA_2MASS = float(config['2MASS_FILTERS']['H_mag_zero_flux_VEGA_2MASS'])
K_mag_zero_flux_VEGA_2MASS = float(config['2MASS_FILTERS']['K_mag_zero_flux_VEGA_2MASS'])

J_mag_lambda_eff_2MASS = float(config['2MASS_FILTERS']['J_mag_lambda_eff_2MASS'])
H_mag_lambda_eff_2MASS = float(config['2MASS_FILTERS']['H_mag_lambda_eff_2MASS'])
K_mag_lambda_eff_2MASS = float(config['2MASS_FILTERS']['K_mag_lambda_eff_2MASS'])

# Zero Point Flux in Jy
W1_mag_zero_flux_VEGA_WISE = float(config['WISE_FILTERS']['W1_mag_zero_flux_VEGA_WISE'])
W2_mag_zero_flux_VEGA_WISE = float(config['WISE_FILTERS']['W2_mag_zero_flux_VEGA_WISE'])
W3_mag_zero_flux_VEGA_WISE = float(config['WISE_FILTERS']['W3_mag_zero_flux_VEGA_WISE'])
W4_mag_zero_flux_VEGA_WISE = float(config['WISE_FILTERS']['W4_mag_zero_flux_VEGA_WISE'])

# Lambda eff
W1_mag_lambda_eff_WISE = float(config['WISE_FILTERS']['W1_mag_lambda_eff_WISE'])
W2_mag_lambda_eff_WISE = float(config['WISE_FILTERS']['W2_mag_lambda_eff_WISE'])
W3_mag_lambda_eff_WISE = float(config['WISE_FILTERS']['W3_mag_lambda_eff_WISE'])
W4_mag_lambda_eff_WISE = float(config['WISE_FILTERS']['W4_mag_lambda_eff_WISE'])

# Zero Point Flux in Jy
u_mag_zero_flux_AB_SDSS = float(config['SDSS_FILTERS']['u_mag_zero_flux_AB_SDSS'])
g_mag_zero_flux_AB_SDSS = float(config['SDSS_FILTERS']['g_mag_zero_flux_AB_SDSS'])
r_mag_zero_flux_AB_SDSS = float(config['SDSS_FILTERS']['r_mag_zero_flux_AB_SDSS'])
i_mag_zero_flux_AB_SDSS = float(config['SDSS_FILTERS']['i_mag_zero_flux_AB_SDSS'])
z_mag_zero_flux_AB_SDSS = float(config['SDSS_FILTERS']['z_mag_zero_flux_AB_SDSS'])

# Lambda eff
u_mag_lambda_eff_SDSS = float(config['SDSS_FILTERS']['u_mag_lambda_eff_SDSS'])
g_mag_lambda_eff_SDSS = float(config['SDSS_FILTERS']['g_mag_lambda_eff_SDSS'])
r_mag_lambda_eff_SDSS = float(config['SDSS_FILTERS']['r_mag_lambda_eff_SDSS'])
i_mag_lambda_eff_SDSS = float(config['SDSS_FILTERS']['i_mag_lambda_eff_SDSS'])
z_mag_lambda_eff_SDSS = float(config['SDSS_FILTERS']['z_mag_lambda_eff_SDSS'])

# Zero Point Flux in Jy
g_mag_zero_flux_AB_PANSTARRS = float(config['PANSTARRS_FILTERS']['g_mag_zero_flux_AB_PANSTARRS'])
r_mag_zero_flux_AB_PANSTARRS = float(config['PANSTARRS_FILTERS']['r_mag_zero_flux_AB_PANSTARRS'])
i_mag_zero_flux_AB_PANSTARRS = float(config['PANSTARRS_FILTERS']['i_mag_zero_flux_AB_PANSTARRS'])
z_mag_zero_flux_AB_PANSTARRS = float(config['PANSTARRS_FILTERS']['z_mag_zero_flux_AB_PANSTARRS'])
y_mag_zero_flux_AB_PANSTARRS = float(config['PANSTARRS_FILTERS']['y_mag_zero_flux_AB_PANSTARRS'])

# Lambda eff
g_mag_lambda_eff_PANSTARRS = float(config['PANSTARRS_FILTERS']['g_mag_lambda_eff_PANSTARRS'])
r_mag_lambda_eff_PANSTARRS = float(config['PANSTARRS_FILTERS']['r_mag_lambda_eff_PANSTARRS'])
i_mag_lambda_eff_PANSTARRS = float(config['PANSTARRS_FILTERS']['i_mag_lambda_eff_PANSTARRS'])
z_mag_lambda_eff_PANSTARRS = float(config['PANSTARRS_FILTERS']['z_mag_lambda_eff_PANSTARRS'])
y_mag_lambda_eff_PANSTARRS = float(config['PANSTARRS_FILTERS']['y_mag_lambda_eff_PANSTARRS'])

# Zero Point Flux in Jy
g_mag_zero_flux_AB_SKYMAPPER = float(config['SKYMAPPER_FILTERS']['g_mag_zero_flux_AB_SKYMAPPER'])

# Lambda eff
g_mag_lambda_eff_SKYMAPPER = float(config['SKYMAPPER_FILTERS']['g_mag_lambda_eff_SKYMAPPER'])


# ===================================================== Query Gaia =====================================================

"""
    Query Gaia DR2 @ VizieR using astroquery.vizier
    parameters: ra_deg, dec_deg, rad_deg: RA, Dec, field radius in degrees
                maxmag: upper limit G magnitude (optional)
                maxsources: maximum number of sources
    returns: astropy.table object
"""


def gaia_query(ra_deg, dec_deg, rad_deg, maxmag=20, maxsources=1000000):
    try:
        vquery = Vizier(columns=['RA_ICRS', 'DE_ICRS', 'Source', 'pmRA', 'pmDE', 'ref_epoch', 'RV', 'Plx', 'e_Plx',
                                 'phot_g_mean_mag', 'phot_g_mean_mag_error',
                                 'phot_bp_mean_mag', 'phot_bp_mean_mag_error', 'phot_rp_mean_mag',
                                 'phot_rp_mean_mag_error'],
                        column_filters={"phot_g_mean_mag": ("<%f" % maxmag)}, row_limit=maxsources)

        field = coord.SkyCoord(ra=ra_deg, dec=dec_deg, unit=(u.deg, u.deg), frame='icrs')

        return vquery.query_region(field, radius=rad_deg * u.deg, catalog=catalog_GAIA)[0]

    except IndexError:
        print(colored(
            'No Gaia candidate was found for this search radius. No comparison can be made. The program will end.',
            'red'))
        sys.exit()


# ===================================================== Query WISE =====================================================

def wise_query(ra_deg, dec_deg, rad_deg, maxmag=20, maxsources=1000000):
    try:
        vquery = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'eeMaj', 'WISE', 'W1mag', 'e_W1mag', 'W2mag', 'e_W2mag', 'W3mag', 'e_W3mag',
                     'W4mag', 'e_W4mag', 'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag'],
            row_limit=maxsources)

        field = coord.SkyCoord(ra=ra_deg, dec=dec_deg,
                               unit=(u.deg, u.deg),
                               frame='icrs')

        return vquery.query_region(field,
                                   radius=rad_deg * u.deg,
                                   catalog=catalog_WISE)[0]
    except IndexError:
        print(colored('No WISE candidate was found for this search radius. Default values will be assigned.', 'yellow'))
        return False


# =================================================== Query 2MASS ======================================================

def twomass_query(ra_deg, dec_deg, rad_deg, maxmag=20, maxsources=1000000):
    try:
        vquery = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'errMaj', '2MASS Name', 'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag',
                     'JD', 'Date'],
            row_limit=maxsources)

        field = coord.SkyCoord(ra=ra_deg, dec=dec_deg,
                               unit=(u.deg, u.deg),
                               frame='icrs')

        return vquery.query_region(field,
                                   radius=rad_deg * u.deg,
                                   catalog=catalog_2MASS)[0]
    except IndexError:
        print(
            colored('No 2MASS candidate was found for this search radius. Default values will be assigned.', 'yellow'))
        return False


# ================================================= Query Pan-STARRS ===================================================

def panstarrs_query(ra_deg, dec_deg, rad_deg, maxmag=20, maxsources=1000000):
    try:
        vquery = Vizier(
            columns=['raMean', 'decMean', 'raMeanErr', 'decMeanErr', 'objID', 'epochMean',
                     'gMeanPSFMag', 'gMeanPSFMagErr', 'rMeanPSFMag', 'rMeanPSFMagErr',
                     'iMeanPSFMag', 'iMeanPSFMagErr', 'zMeanPSFMag', 'zMeanPSFMagErr',
                     'yMeanPSFMag', 'yMeanPSFMagErr'],
            row_limit=maxsources)

        field = coord.SkyCoord(ra=ra_deg, dec=dec_deg,
                               unit=(u.deg, u.deg),
                               frame='icrs')

        return vquery.query_region(field,
                                   radius=rad_deg * u.deg,
                                   catalog=catalog_PANSTARRS)[0]
    except IndexError:
        print(colored('No Pan-Starrs candidate was found for this search radius. Default values will be assigned.',
                      'yellow'))
        return False


# ================================================= Query SDSS =========================================================

def sdss_query(ra_deg, dec_deg, rad_deg, maxmag=20, maxsources=1000000):
    try:
        vquery = Vizier(
            columns=['RA_ICRS', 'DE_ICRS', 'SDSS-ID', 'e_RA_ICRS', 'e_DE_ICRS', 'ObsDate',
                     'umag', 'e_umag', 'gmag', 'e_gmag', 'rmag', 'e_rmag', 'imag', 'e_imag', 'zmag', 'e_zmag'],
            row_limit=maxsources)

        field = coord.SkyCoord(ra=ra_deg, dec=dec_deg,
                               unit=(u.deg, u.deg),
                               frame='icrs')

        return vquery.query_region(field,
                                   radius=rad_deg * u.deg,
                                   catalog=catalog_SDSS)[0]
    except IndexError:
        print(colored('No SDSS candidate was found for this search radius. Default values will be assigned.', 'yellow'))
        return False


# ================================================= Query SkyMapper ====================================================

def skymapper_query(ra_deg, dec_deg, rad_deg, maxmag=20, maxsources=1000000):
    try:
        vquery = Vizier(
            columns=['RAJ2000', 'DEJ2000', 'ID', 'Date', 'gmag', 'g-i', 'Slit'],
            row_limit=maxsources)

        field = coord.SkyCoord(ra=ra_deg, dec=dec_deg,
                               unit=(u.deg, u.deg),
                               frame='icrs')

        return vquery.query_region(field,
                                   radius=rad_deg * u.deg,
                                   catalog=catalog_SKYMAPPER)[0]


    except IndexError:
        print(colored('No SkyMapper candidate was found for this search radius. Default values will be assigned.',
                      'yellow'))
        return False


# ============================== Calculate the displacement due to the proper motion  ==================================

def proper_motion(RA, DEC, distance, radial_velocity, proper_motion_RA, proper_motion_DEC, t, t_GAIA):
    '''
    This function takes:
    Right ascension in hours,
    Declination in degrees,
    Proper motion in Right Ascension direction in second of arc per year (arsec/yr)
    Time in Julian Days of the source
    Time in Julian Days of the reference catalogue (GAIA in this case)

    The output is:
    Right ascension in degrees,
    Declination in degrees,

    The calculations are based on: http://www.astronexus.com/a-a/motions
    '''

    # Let t be the time difference between the reference epoch of GAIA and the epoch of observation of the source
    t = t - t_GAIA  # recall that t_GAIA has been converted to jd (the reference time is 2015.5 in decimal year)

    # Convert the proper motion from arcseconds to seconds of right ascension
    proper_motion_RA = proper_motion_RA / (15 * (math.cos(math.radians(DEC))))

    # Convert the proper motion quantities into decimal fractions of an hour or degree
    proper_motion_RA = proper_motion_RA / 3600
    proper_motion_DEC = proper_motion_DEC / 3600

    # Compute the new RA & DEC using the proper motion
    RA_new = RA + (proper_motion_RA / 365.25 * t)
    DEC_new = DEC + (proper_motion_DEC / 365.25 * t)

    # Convert to degrees
    ra_dec_degrees = SkyCoord(RA_new, DEC_new, unit=(u.hour, u.deg), frame='icrs')
    RA_new = ra_dec_degrees.ra.degree
    DEC_new = ra_dec_degrees.dec.degree

    return (RA_new, DEC_new)


# ========================================== Find Position for Different Epoch =========================================
'''
Find the RA and Dec for previous epoch 
using the proper motion given by Gaia 
'''

# 0. Take input from argument ------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Take search specifications')
parser.add_argument('-ra', '--ra', nargs=1, required=True, type=str,
                    help='Enter right ascension in HH:MM:SS.SS or in DEG', metavar='right ascension')
parser.add_argument('-dec', '--dec', nargs=1, required=True, type=str,
                    help='Enter declination in DD:MM:SS.SS or in DEG', metavar='declination')

parser.add_argument('-fov_gaia', '--fov_gaia', nargs=1, default= [5], type=int,
                    help='Search field of view for Gaia. Default is 20 arcs')
parser.add_argument('-fov_2mass', '--fov_2mass', nargs=1, default= [5], type=int,
                    help='Search field of view for 2MASS. Default is 5 arcs')
parser.add_argument('-fov_wise', '--fov_wise', nargs=1, default= [5], type=int,
                    help='Search field of view for WISE. Default is 5 arcs')
parser.add_argument('-fov_panstarrs', '--fov_panstarrs', nargs=1, default= [5], type=int,
                    help='Search field of view for PANSTARRS. Default is 5 arcs')
parser.add_argument('-fov_sdss', '--fov_sdss', nargs=1, default= [5], type=int,
                    help='Search field of view for SDSS. Default is 5 arcs')
parser.add_argument('-fov_skymapper', '--fov_skymapper', nargs=1, default= [5], type=int,
                    help='Search field of view for SkyMapper. Default is 5 arcs')

args = parser.parse_args()

ra_input = args.ra[0]
dec_input = args.dec[0]



# 1. Check input and convert if needed ---------------------------------------------------------------------------------

# Check if user missmatched the format of the arguments

if re.search(':', ra_input):
    ra_type = True
else:
    ra_type = False

if re.search(':', dec_input):
    dec_type = True
else:
    dec_type = False

if ra_type is not dec_type:
    print('Please enter input of same type and try again')
    quit()

# Check if the user entered the input in degrees or hours
if re.search(':', ra_input) and re.search(':', dec_input):
    print('need to convert')

    ra_dec = str(ra_input + ' ' + dec_input)
    print('The input Right Ascension and Declination in (hours, degrees) is: ', ra_dec)

    # ra_dec = '01:26:55.49 -50:22:38.8' # input DEC
    ra_dec_degrees = SkyCoord(ra_dec, unit=(u.hour, u.deg), frame='icrs')
    ra = ra_dec_degrees.ra.degree
    dec = ra_dec_degrees.dec.degree
    print()
    print('Input RA in degrees: ', ra)
    print('Input DEC in degrees: ', dec)
    print()

else:
    print()
    print('Input RA in degrees: ', ra_input)
    print('Input DEC in degrees: ', dec_input)
    print()
    ra = float(ra_input)
    dec = float(dec_input)



# 2. Define field of view for each catalog (radius in degrees)----------------------------------------------------------
fov_GAIA = args.fov_gaia[0] / 3600  # specify field of view
fov_2MASS = args.fov_2mass[0] / 3600
fov_WISE = args.fov_wise[0] / 3600
fov_PANSTARRS = args.fov_panstarrs[0] / 3600
fov_SDSS = args.fov_sdss[0] / 3600
fov_SKYMAPPER = args.fov_skymapper[0] / 3600

# 3. Query Gaia within given field of view -----------------------------------------------------------------------------
print()
print()
print(colored('*****************************************************************************', 'blue'))
print(colored('           Querying catalogs ...                      ', 'blue'))
print(colored('*****************************************************************************', 'blue'))
print()
print()

query_result_GAIA = gaia_query(ra, dec, fov_GAIA)
print()
print('Result from GAIA DR2 query:')
print()
query_result_GAIA.pprint(max_lines=-1, max_width=-1)
# print(query_result_GAIA)
print()

# 4. Find brightest star in the field of view --------------------------------------------------------------------------
'''
For now, take the brightest star. 
Later, loop through all the other candidates (see step #10) 
'''

magnitude_GAIA = np.array([])
ALL_QUERIES_RESULTS = np.array([])  # We will store booleans to describe if we found a match in our search or not

for m in range(len(query_result_GAIA)):
    magnitude_GAIA = np.append(magnitude_GAIA, query_result_GAIA['Gmag'][m])

index_brightest = np.argmin(magnitude_GAIA)

# 5. Take the data for the brightest star ------------------------------------------------------------------------------
source_id_Gaia = query_result_GAIA['Source'][index_brightest]
ra_Gaia = query_result_GAIA['RA_ICRS'][index_brightest]
dec_Gaia = query_result_GAIA['DE_ICRS'][index_brightest]
pmRA_Gaia = query_result_GAIA['pmRA'][index_brightest]
pmDE_Gaia = query_result_GAIA['pmDE'][index_brightest]
parallax_Gaia = query_result_GAIA['Plx'][index_brightest]
radial_velocity_Gaia = query_result_GAIA['RV'][index_brightest]
epoch_GAIA = query_result_GAIA['Epoch'][index_brightest]

G_mag_GAIA = err.ufloat(query_result_GAIA['Gmag'][index_brightest], query_result_GAIA['e_Gmag'][index_brightest])
BP_mag_GAIA = err.ufloat(query_result_GAIA['BPmag'][index_brightest], query_result_GAIA['e_BPmag'][index_brightest])
RP_mag_GAIA = err.ufloat(query_result_GAIA['RPmag'][index_brightest], query_result_GAIA['e_RPmag'][index_brightest])

if (math.isnan(radial_velocity_Gaia)):  # in case the radial velocity is not given, set it to zero
    radial_velocity_Gaia = 0

# 6. Query other catalogs ----------------------------------------------------------------------------------------------
query_result_2MASS = twomass_query(ra, dec, fov_2MASS)
query_result_WISE = wise_query(ra, dec, fov_WISE)
query_result_PANSTARRS = panstarrs_query(ra, dec, fov_PANSTARRS)
query_result_SDSS = sdss_query(ra, dec, fov_SDSS)
query_result_SKYMAPPER = skymapper_query(ra, dec, fov_SKYMAPPER)

'''
Print only if the query result object is not a boolean 
(Remember that the function returns a boolean if no result was found)
We complicate the print statement because we want to see the full candidate table (even if it has 30 rows)
'''
if isinstance(query_result_2MASS, bool) == False:
    print()
    print('Result from 2MASS query:')
    print()
    query_result_2MASS.pprint(max_lines=-1, max_width=-1)
    print()

if isinstance(query_result_WISE, bool) == False:
    print()
    print('Result from WISE query:')
    print()
    query_result_WISE.pprint(max_lines=-1, max_width=-1)
    print()

if isinstance(query_result_PANSTARRS, bool) == False:
    print()
    print('Result from PANSTARRS query:')
    print()
    query_result_PANSTARRS.pprint(max_lines=-1, max_width=-1)
    print()

if isinstance(query_result_SDSS, bool) == False:
    print()
    print('Result from SDSS query:')
    print()
    query_result_SDSS.pprint(max_lines=-1, max_width=-1)
    print()

if isinstance(query_result_SKYMAPPER, bool) == False:
    print()
    print('Result from SKYMAPPER query:')
    print()
    query_result_SKYMAPPER.pprint(max_lines=-1, max_width=-1)
    print()

'''
Save most important information from 2MASS
'''
if isinstance(query_result_2MASS, bool) == True:  # This is in the case, no candidate was found (return is boolean)
    # assigning default values
    magnitude_2MASS = np.array([])
    epoch_2MASS = np.array([])
    ra_2MASS = np.array([])
    dec_2MASS = np.array([])
    position_error_2MASS = np.array([])

    magnitude_2MASS = np.append(magnitude_2MASS, 1)
    ra_2MASS = np.append(ra_2MASS, 1)
    dec_2MASS = np.append(dec_2MASS, 1)
    epoch_2MASS = np.append(epoch_2MASS, 1)  #### This is very important!
    position_error_2MASS = np.append(position_error_2MASS, 1)

else:  # Otherwise, proceed with the search
    # Store the ost relevant parameters in arrays
    magnitude_2MASS = np.array([])
    epoch_2MASS = np.array([])
    ra_2MASS = np.array([])
    dec_2MASS = np.array([])
    position_error_2MASS = np.array([])

    for n in range(len(query_result_2MASS)):
        magnitude_2MASS = np.append(magnitude_2MASS, query_result_2MASS['Jmag'][n])
        ra_2MASS = np.append(ra_2MASS, query_result_2MASS['RAJ2000'][n])
        dec_2MASS = np.append(dec_2MASS, query_result_2MASS['DEJ2000'][n])
        epoch_2MASS = np.append(epoch_2MASS, query_result_2MASS['JD'][n])  #### This is very important!
        position_error_2MASS = np.append(position_error_2MASS, query_result_2MASS['errMaj'][n])

'''
Do the same thing for WISE 
'''
if isinstance(query_result_WISE, bool) == True:
    magnitude_WISE = np.array([])
    ra_WISE = np.array([])
    dec_WISE = np.array([])
    position_error_WISE = np.array([])

    magnitude_WISE = np.append(magnitude_WISE, 1)
    ra_WISE = np.append(ra_WISE, 1)
    dec_WISE = np.append(dec_WISE, 1)
    position_error_WISE = np.append(position_error_WISE, 1)

else:
    magnitude_WISE = np.array([])
    ra_WISE = np.array([])
    dec_WISE = np.array([])
    position_error_WISE = np.array([])

    for m in range(len(query_result_WISE)):
        magnitude_WISE = np.append(magnitude_WISE, query_result_WISE['W1mag'][m])
        ra_WISE = np.append(ra_WISE, query_result_WISE['RAJ2000'][m])
        dec_WISE = np.append(dec_WISE, query_result_WISE['DEJ2000'][m])
        position_error_WISE = np.append(position_error_WISE, query_result_WISE['eeMaj'][m])

'''
Do the same thing for PANSTARRS 
'''
if isinstance(query_result_PANSTARRS, bool) == True:
    magnitude_PANSTARRS = np.array([])
    ra_PANSTARRS = np.array([])
    dec_PANSTARRS = np.array([])
    position_errorRA_PANSTARRS = np.array([])
    position_errorDEC_PANSTARRS = np.array([])
    epoch_PANSTARRS = np.array([])

    magnitude_PANSTARRS = np.append(magnitude_PANSTARRS, 1)
    ra_PANSTARRS = np.append(ra_PANSTARRS, 1)
    dec_PANSTARRS = np.append(dec_PANSTARRS, 1)
    position_errorRA_PANSTARRS = np.append(position_errorRA_PANSTARRS, 1)
    position_errorDEC_PANSTARRS = np.append(position_errorDEC_PANSTARRS, 1)
    epoch_PANSTARRS = np.append(epoch_PANSTARRS, 1)

else:
    magnitude_PANSTARRS = np.array([])
    ra_PANSTARRS = np.array([])
    dec_PANSTARRS = np.array([])
    position_errorRA_PANSTARRS = np.array([])
    position_errorDEC_PANSTARRS = np.array([])
    epoch_PANSTARRS = np.array([])

    for a in range(len(query_result_PANSTARRS)):
        magnitude_PANSTARRS = np.append(magnitude_PANSTARRS, query_result_PANSTARRS['gmag'][a])
        ra_PANSTARRS = np.append(ra_PANSTARRS, query_result_PANSTARRS['RAJ2000'][a])
        dec_PANSTARRS = np.append(dec_PANSTARRS, query_result_PANSTARRS['DEJ2000'][a])
        position_errorRA_PANSTARRS = np.append(position_errorRA_PANSTARRS, query_result_PANSTARRS['e_RAJ2000'][a])
        position_errorDEC_PANSTARRS = np.append(position_errorDEC_PANSTARRS, query_result_PANSTARRS['e_DEJ2000'][a])
        epoch_PANSTARRS = np.append(epoch_PANSTARRS, query_result_PANSTARRS['Epoch'][a])

'''
Do the same thing for SDSS 
'''
if isinstance(query_result_SDSS, bool) == True:
    magnitude_SDSS = np.array([])
    ra_SDSS = np.array([])
    dec_SDSS = np.array([])
    position_errorRA_SDSS = np.array([])
    position_errorDEC_SDSS = np.array([])
    epoch_SDSS = np.array([])

    magnitude_SDSS = np.append(magnitude_SDSS, 1)
    ra_SDSS = np.append(ra_SDSS, 1)
    dec_SDSS = np.append(dec_SDSS, 1)
    position_errorRA_SDSS = np.append(position_errorRA_SDSS, 1)
    position_errorDEC_SDSS = np.append(position_errorDEC_SDSS, 1)
    epoch_SDSS = np.append(epoch_SDSS, 1)

else:
    magnitude_SDSS = np.array([])
    ra_SDSS = np.array([])
    dec_SDSS = np.array([])
    position_errorRA_SDSS = np.array([])
    position_errorDEC_SDSS = np.array([])
    epoch_SDSS = np.array([])

    for b in range(len(query_result_SDSS)):
        magnitude_SDSS = np.append(magnitude_SDSS, query_result_SDSS['gmag'][b])
        ra_SDSS = np.append(ra_SDSS, query_result_SDSS['RA_ICRS'][b])
        dec_SDSS = np.append(dec_SDSS, query_result_SDSS['DE_ICRS'][b])
        position_errorRA_SDSS = np.append(position_errorRA_SDSS, query_result_SDSS['e_RA_ICRS'][b])
        position_errorDEC_SDSS = np.append(position_errorDEC_SDSS, query_result_SDSS['e_DE_ICRS'][b])
        epoch_SDSS = np.append(epoch_SDSS, query_result_SDSS['ObsDate'][b])

'''
Do the same thing for SKYMAPPER
'''
if isinstance(query_result_SKYMAPPER, bool) == True:  # This is in the case, no candidate was found
    # assigning default values
    magnitude_SKYMAPPER = np.array([])
    epoch_SKYMAPPER = np.array([])
    ra_SKYMAPPER = np.array([])
    dec_SKYMAPPER = np.array([])
    position_error_SKYMAPPER = np.array([])

    magnitude_SKYMAPPER = np.append(magnitude_SKYMAPPER, 1)
    ra_SKYMAPPER = np.append(ra_SKYMAPPER, 1)
    dec_SKYMAPPER = np.append(dec_SKYMAPPER, 1)
    epoch_SKYMAPPER = np.append(epoch_SKYMAPPER, '1000-01-01')  #### This is very important!
    position_error_SKYMAPPER = np.append(position_error_SKYMAPPER, 1)

else:  # Otherwise, proceed with the search
    # Store the ost relevant parameters in arrays
    magnitude_SKYMAPPER = np.array([])
    epoch_SKYMAPPER = np.array([])
    ra_SKYMAPPER = np.array([])
    dec_SKYMAPPER = np.array([])
    position_error_SKYMAPPER = np.array([])

    for c in range(len(query_result_SKYMAPPER)):
        # Convert the coordinates to degrees!
        ra_in_SKYMAPPER = query_result_SKYMAPPER['RAJ2000'][c]
        dec_in_SKYMAPPER = query_result_SKYMAPPER['DEJ2000'][c]
        transform = SkyCoord(ra_in_SKYMAPPER, dec_in_SKYMAPPER, frame='icrs', unit=(u.hour, u.deg))
        ra_out_SKYMAPPER = transform.ra.deg  # hours
        dec_out_SKYMAPPER = transform.dec.deg  # degrees

        magnitude_SKYMAPPER = np.append(magnitude_SKYMAPPER, query_result_SKYMAPPER['gmag'][c])
        ra_SKYMAPPER = np.append(ra_SKYMAPPER, ra_out_SKYMAPPER)
        dec_SKYMAPPER = np.append(dec_SKYMAPPER, dec_out_SKYMAPPER)
        epoch_SKYMAPPER = np.append(epoch_SKYMAPPER, query_result_SKYMAPPER['Date'][c])  #### This is very important!
        position_error_SKYMAPPER = np.append(position_error_SKYMAPPER, query_result_SKYMAPPER['Slit'][
            c])  # Note that we take the slit as the error on the position

# 7. Put the Gaia data in the right units to calcuate the proper motion  -----------------------------------------------
c = SkyCoord(ra_Gaia, dec_Gaia, frame='icrs', unit='deg')
RA = c.ra.hour  # hours
DEC = c.dec.deg  # degrees
distance = (1 / parallax_Gaia) / 1000  # put the distance in pc
radial_velocity = radial_velocity_Gaia  # in km/s
proper_motion_RA = (pmRA_Gaia / 1000)  # in second of arc per year
proper_motion_DEC = pmDE_Gaia / 1000  # in second of arc per year

t_GAIA = Time(epoch_GAIA, format='decimalyear')
t_GAIA = t_GAIA.jd

# ======================================= Looking for a match with other catalogs ======================================


# 8. Find Stars in 2MASS and find a match with Gaia based on their epoch -----------------------------------------------
print()
print()
print(colored('*****************************************************************************', 'blue'))
print(colored('           Looking for a match using the brightest Gaia candidate             ', 'blue'))
print(colored('*****************************************************************************', 'blue'))
print()
print()

'''
Set Default Spectral Information 
'''
J_mag_2MASS = -1
H_mag_2MASS = -1
K_mag_2MASS = -1

J_mag_2MASS_error = 0
H_mag_2MASS_error = 0
K_mag_2MASS_error = 0

index_array_2MASS = np.array([]) # We will store the indices of our successful matches here
sigma_array_2MASS = np.array([]) # We will store the obtained sigmas

match_array = np.array([])
for i in range(len(ra_2MASS)):

    # epoch of 2MASS is already in JD
    t_2MASS = epoch_2MASS[i]
    time = t_2MASS

    # Find the RA & DEC at selected time
    ra_result_2MASS, dec_result_2MASS = proper_motion(RA, DEC, distance, radial_velocity,
                                                      proper_motion_RA, proper_motion_DEC, time, t_GAIA)

    # Verify if the obtained coordinates match 2MASS within 1 arsec
    ra_match_2MASS = math.isclose(ra_result_2MASS, ra_2MASS[i], abs_tol=1 / 3600)
    dec_match_2MASS = math.isclose(dec_result_2MASS, dec_2MASS[i], abs_tol=1 / 3600)

    # print(ra_result_2MASS)
    # print(ra_2MASS[i])
    # print(ra_match_2MASS)
    # print(dec_result_2MASS)
    # print(dec_2MASS[i])
    # print(dec_match_2MASS)

    if ra_match_2MASS == True and dec_match_2MASS == True:
        match = True
        match_array = np.append(match_array, match)  # keep track of match results
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        index_array_2MASS = np.append(index_array_2MASS, i)

        '''
        Set a match quality flag: How many sigmas are we away from the real result?
        Compute in terms of arcseconds 
        '''
        cosdec_2MASS = math.cos(math.radians(dec_2MASS[i] * 3600))  # This is cos(declination) correction factor

        # ***********************************
        # Calculate the standard deviation along each coordinate for the largest error given in the catalog
        # (semi-major axis of uncertainty ellipse)
        error_circle_2MASS = np.array([[(ra_2MASS[i] * 3600), (ra_2MASS[i] * 3600 + position_error_2MASS[i])],
                                       [(dec_2MASS[i] * 3600), (dec_2MASS[i] * 3600 + position_error_2MASS[i])]])
        standard_dev_2MASS = np.std(error_circle_2MASS, axis=1)

        # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
        error_circle_match_2MASS = np.array(
            [[(ra_2MASS[i] * 3600) * cosdec_2MASS, (ra_result_2MASS * 3600) * cosdec_2MASS],
             [(dec_2MASS[i] * 3600), (dec_result_2MASS * 3600)]])
        standard_dev_match_2MASS = np.std(error_circle_match_2MASS, axis=1)

        # Take the standard deviation radii
        standard_dev_radius_sigma1_2MASS = math.sqrt(
            math.pow(standard_dev_2MASS[0], 2) + math.pow(standard_dev_2MASS[1], 2))
        standard_dev_radius_match_2MASS = math.sqrt(
            math.pow(standard_dev_match_2MASS[0], 2) + math.pow(standard_dev_match_2MASS[1], 2))

        # Take the ratio between the two radii
        standard_dev_radius_2MASS = standard_dev_radius_match_2MASS / standard_dev_radius_sigma1_2MASS
        standard_dev_radius_2MASS = float((format(standard_dev_radius_2MASS, '.2f')))  # format to 2 decimals

        print(colored('The target star was found in 2MASS for Gaia star %s' % source_id_Gaia, 'green'))
        print(
            colored('The target is within a radius of %s sigma from the 2MASS candidate' % (standard_dev_radius_2MASS),
                    'green'))
        print('===================================================================================')
        print(query_result_2MASS[i])
        print()
        print('')

        sigma_array_2MASS = np.append(sigma_array_2MASS, standard_dev_radius_2MASS)

    else:
        match = False
        match_array = np.append(match_array, match)
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

if all(element == False for element in match_array) == True:
    print()
    print('No match was found in 2MASS for Gaia star ', source_id_Gaia)
    print()

else:
    # If we did find matches, we will retrieve the spectral information to later plot
    # a Spectral Energy Distribution graph
    index_2MASS = np.argmin(sigma_array_2MASS)
    best_match_2MASS = int(index_array_2MASS[index_2MASS])
    '''
    Take spectral information 
    '''
    J_mag_2MASS = query_result_2MASS['Jmag'][best_match_2MASS]
    H_mag_2MASS = query_result_2MASS['Hmag'][best_match_2MASS]
    K_mag_2MASS = query_result_2MASS['Kmag'][best_match_2MASS]

    J_mag_2MASS_error = query_result_2MASS['e_Jmag'][best_match_2MASS]
    H_mag_2MASS_error = query_result_2MASS['e_Hmag'][best_match_2MASS]
    K_mag_2MASS_error = query_result_2MASS['e_Kmag'][best_match_2MASS]

    if len(sigma_array_2MASS) > 1:
        print(colored(
            'WARNING: More than one candidate was found in this catalog within one arcsecond. '
            'The one with the lowest sigma will be used for the Spectral Energy Distribution',
            'red'))

# 9. Find Stars in WISE and find a match with Gaia ---------------------------------------------------------------------
'''
The epoch for WISE is set as 2010.5
'''
'''
Set Default Spectral Information 
'''
W1_mag_WISE = -1
W2_mag_WISE = -1
W3_mag_WISE = -1
W4_mag_WISE = -1

W1_mag_WISE_error = 0
W2_mag_WISE_error = 0
W3_mag_WISE_error = 0
W4_mag_WISE_error = 0

index_array_WISE = np.array([]) # We will store the indices of our successful matches here
sigma_array_WISE = np.array([]) # We will store the obtained sigmas

match_array = np.array([])
for j in range(len(ra_WISE)):

    time = ref_epoch_WISE


    # Find the RA & DEC at selected time
    ra_result_WISE, dec_result_WISE = proper_motion(RA, DEC, distance, radial_velocity,
                                                    proper_motion_RA, proper_motion_DEC, time, t_GAIA)

    # Verify if the obtained coordinates match WISE within 1 arsec
    ra_match_WISE = math.isclose(ra_result_WISE, ra_WISE[j], abs_tol=1 / 3600)
    dec_match_WISE = math.isclose(dec_result_WISE, dec_WISE[j], abs_tol=1 / 3600)

    # print(ra_result_WISE)
    # print(ra_WISE[j])
    # print(ra_match_WISE)
    # print(dec_result_WISE)
    # print(dec_WISE[j])
    # print(dec_match_WISE)

    if ra_match_WISE == True and dec_match_WISE == True:
        match = True
        match_array = np.append(match_array, match)
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        index_array_WISE = np.append(index_array_WISE, j)

        '''
        Set a match quality flag: How many sigmas are we away from the real result?
        Compute in terms of arcseconds 
        '''
        cosdec_WISE = math.cos(math.radians(dec_WISE[j] * 3600))  # This is cos(declination) correction factor

        # ***********************************
        # Calculate the standard deviation along each coordinate for the largest error given in the catalog
        # (semi-major axis of uncertainty ellipse)
        error_circle_WISE = np.array([[(ra_WISE[j] * 3600), (ra_WISE[j] * 3600 + position_error_WISE[j])],
                                      [(dec_WISE[j] * 3600), (dec_WISE[j] * 3600 + position_error_WISE[j])]])
        standard_dev_WISE = np.std(error_circle_WISE, axis=1)

        # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
        error_circle_match_WISE = np.array(
            [[(ra_WISE[j] * 3600) * cosdec_WISE, (ra_result_WISE * 3600) * cosdec_WISE],
             [(dec_WISE[j] * 3600), (dec_result_WISE * 3600)]])
        standard_dev_match_WISE = np.std(error_circle_match_WISE, axis=1)

        # Take the standard deviation radii
        standard_dev_radius_sigma1_WISE = math.sqrt(
            math.pow(standard_dev_WISE[0], 2) + math.pow(standard_dev_WISE[1], 2))
        standard_dev_radius_match_WISE = math.sqrt(
            math.pow(standard_dev_match_WISE[0], 2) + math.pow(standard_dev_match_WISE[1], 2))

        # Take the ratio between the two radii
        standard_dev_radius_WISE = standard_dev_radius_match_WISE / standard_dev_radius_sigma1_WISE
        standard_dev_radius_WISE = float((format(standard_dev_radius_WISE, '.2f')))  # format to 2 decimals

        print(colored('The target star was found in WISE for Gaia star %s' % source_id_Gaia, 'green'))
        print(colored('The target is within a radius of %s sigma from the WISE candidate' % standard_dev_radius_WISE,
                      'green'))
        print('===================================================================================')
        print(query_result_WISE[j])
        print('')

        sigma_array_WISE = np.append(sigma_array_WISE, standard_dev_radius_WISE)

    else:
        match = False
        match_array = np.append(match_array, match)
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

if all(element == False for element in match_array) == True:
    print()
    print('No match was found in WISE for Gaia star ', source_id_Gaia)
    print()

else:
    # If we did find matches, we will retrieve the spectral information to later plot
    # a Spectral Energy Distribution graph
    index_WISE = np.argmin(sigma_array_WISE)
    best_match_WISE = int(index_array_WISE[index_WISE])

    '''
    Take spectral information 
    '''
    W1_mag_WISE = query_result_WISE['W1mag'][best_match_WISE]
    W2_mag_WISE = query_result_WISE['W2mag'][best_match_WISE]
    W3_mag_WISE = query_result_WISE['W3mag'][best_match_WISE]
    W4_mag_WISE = query_result_WISE['W4mag'][best_match_WISE]

    W1_mag_WISE_error = query_result_WISE['e_W1mag'][best_match_WISE]
    W2_mag_WISE_error = query_result_WISE['e_W2mag'][best_match_WISE]
    W3_mag_WISE_error = query_result_WISE['e_W3mag'][best_match_WISE]
    W4_mag_WISE_error = query_result_WISE['e_W4mag'][best_match_WISE]

    if len(sigma_array_WISE) > 1:
        print(colored(
            'WARNING: More than one candidate was found in this catalog within one arcsecond. '
            'The one with the lowest sigma will be used for the Spectral Energy Distribution',
            'red'))

# 10. Find Stars in PANSTARRS and find a match with Gaia based on their epoch ------------------------------------------

'''
Set Default Spectral Information 
'''
g_mag_PANSTARRS = -1
r_mag_PANSTARRS = -1
i_mag_PANSTARRS = -1
z_mag_PANSTARRS = -1
y_mag_PANSTARRS = -1

g_mag_PANSTARRS_error = 0
r_mag_PANSTARRS_error = 0
i_mag_PANSTARRS_error = 0
z_mag_PANSTARRS_error = 0
y_mag_PANSTARRS_error = 0

index_array_PANSTARRS = np.array([]) # We will store the indices of our successful matches here
sigma_array_PANSTARRS = np.array([]) # We will store the obtained sigmas

match_array = np.array([])
for h in range(len(ra_PANSTARRS)):

    t = epoch_PANSTARRS[h]
    t_PANSTARRS = Time(t, format='mjd')  # verify conversion
    t_PANSTARRS = t_PANSTARRS.jd
    # t_PANSTARRS = t_PANSTARRS.decimalyear
    # time = - (2015.5 - t_PANSTARRS) # Gaia is the reference epoch so the time should be wtr to 2015.5 in yr
    time = t_PANSTARRS

    # Find the RA & DEC at selected time
    ra_result_PANSTARRS, dec_result_PANSTARRS = proper_motion(RA, DEC, distance, radial_velocity,
                                                              proper_motion_RA, proper_motion_DEC, time, t_GAIA)

    # Verify if the obtained coordinates match PANSTARRS within 1 arsec
    ra_match_PANSTARRS = math.isclose(ra_result_PANSTARRS, ra_PANSTARRS[h], abs_tol=1 / 3600)
    dec_match_PANSTARRS = math.isclose(dec_result_PANSTARRS, dec_PANSTARRS[h], abs_tol=1 / 3600)

    # print(ra_result_PANSTARRS)
    # print(ra_PANSTARRS[h])
    # print(ra_match_PANSTARRS)
    # print(dec_result_PANSTARRS)
    # print(dec_PANSTARRS[h])
    # print(dec_match_PANSTARRS)

    if ra_match_PANSTARRS == True and dec_match_PANSTARRS == True:
        match = True
        match_array = np.append(match_array, match)  # keep track of match results
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        index_array_PANSTARRS = np.append(index_array_PANSTARRS, h)

        '''
        Set a match quality flag: How many sigmas are we away from the real result?
        Compute in terms of arcseconds 
        '''
        cosdec_PANSTARRS = math.cos(math.radians(dec_PANSTARRS[h] * 3600))  # This is cos(declination) correction factor

        # ***********************************
        # Calculate the standard deviation along each coordinate for the largest error given in the catalog
        # (semi-major axis of uncertainty ellipse)
        error_circle_PANSTARRS = np.array(
            [[(ra_PANSTARRS[h] * 3600), (ra_PANSTARRS[h] * 3600 + position_errorRA_PANSTARRS[h])],
             [(dec_PANSTARRS[h] * 3600), (dec_PANSTARRS[h] * 3600 + position_errorDEC_PANSTARRS[h])]])
        standard_dev_PANSTARRS = np.std(error_circle_PANSTARRS, axis=1)

        # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
        error_circle_match_PANSTARRS = np.array(
            [[(ra_PANSTARRS[h] * 3600) * cosdec_PANSTARRS, (ra_result_PANSTARRS * 3600) * cosdec_PANSTARRS],
             [(dec_PANSTARRS[h] * 3600), (dec_result_PANSTARRS * 3600)]])
        standard_dev_match_PANSTARRS = np.std(error_circle_match_PANSTARRS, axis=1)

        # Take the standard deviation radii
        standard_dev_radius_sigma1_PANSTARRS = math.sqrt(
            math.pow(standard_dev_PANSTARRS[0], 2) + math.pow(standard_dev_PANSTARRS[1], 2))
        standard_dev_radius_match_PANSTARRS = math.sqrt(
            math.pow(standard_dev_match_PANSTARRS[0], 2) + math.pow(standard_dev_match_PANSTARRS[1], 2))

        # Take the ratio between the two radii
        standard_dev_radius_PANSTARRS = standard_dev_radius_match_PANSTARRS / standard_dev_radius_sigma1_PANSTARRS
        standard_dev_radius_PANSTARRS = float((format(standard_dev_radius_PANSTARRS, '.2f')))  # format to 2 decimals

        print(colored('The target star was found in PANSTARRS for Gaia star %s' % source_id_Gaia, 'green'))
        print(colored(
            'The target is within a radius of %s sigma from the PANSTARRS candidate' % standard_dev_radius_PANSTARRS,
            'green'))
        print('===================================================================================')
        print(query_result_PANSTARRS[h])
        print()
        print('')

        sigma_array_PANSTARRS = np.append(sigma_array_PANSTARRS, standard_dev_radius_PANSTARRS)

    else:
        match = False
        match_array = np.append(match_array, match)
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

if all(element == False for element in match_array) == True:
    print()
    print('No match was found in PANSTARRS for Gaia star ', source_id_Gaia)
    print()

else:
    # If we did find matches, we will retrieve the spectral information to later plot
    # a Spectral Energy Distribution graph
    index_PANSTARRS = np.argmin(sigma_array_PANSTARRS)
    best_match_PANSTARRS = int(index_array_PANSTARRS[index_PANSTARRS])

    '''
    Take spectral information
    '''
    g_mag_PANSTARRS = query_result_PANSTARRS['gmag'][best_match_PANSTARRS]
    r_mag_PANSTARRS = query_result_PANSTARRS['rmag'][best_match_PANSTARRS]
    i_mag_PANSTARRS = query_result_PANSTARRS['imag'][best_match_PANSTARRS]
    z_mag_PANSTARRS = query_result_PANSTARRS['zmag'][best_match_PANSTARRS]
    y_mag_PANSTARRS = query_result_PANSTARRS['ymag'][best_match_PANSTARRS]

    g_mag_PANSTARRS_error = query_result_PANSTARRS['e_gmag'][best_match_PANSTARRS]
    r_mag_PANSTARRS_error = query_result_PANSTARRS['e_rmag'][best_match_PANSTARRS]
    i_mag_PANSTARRS_error = query_result_PANSTARRS['e_imag'][best_match_PANSTARRS]
    z_mag_PANSTARRS_error = query_result_PANSTARRS['e_zmag'][best_match_PANSTARRS]
    y_mag_PANSTARRS_error = query_result_PANSTARRS['e_ymag'][best_match_PANSTARRS]

    if len(sigma_array_PANSTARRS) > 1:
        print(colored(
            'WARNING: More than one candidate was found in this catalog within one arcsecond. '
            'The one with the lowest sigma will be used for the Spectral Energy Distribution',
            'red'))



# 11. Find Stars in SDSS and find a match with Gaia based on their epoch -----------------------------------------------
'''
Set Default Spectral Information 
'''
u_mag_SDSS = -1
g_mag_SDSS = -1
r_mag_SDSS = -1
i_mag_SDSS = -1
z_mag_SDSS = -1

u_mag_SDSS_error = 0
g_mag_SDSS_error = 0
r_mag_SDSS_error = 0
i_mag_SDSS_error = 0
z_mag_SDSS_error = 0

index_array_SDSS = np.array([]) # We will store the indices of our successful matches here
sigma_array_SDSS = np.array([]) # We will store the obtained sigmas

match_array = np.array([])
for l in range(len(ra_SDSS)):

    t = epoch_SDSS[l]
    t_SDSS = Time(t, format='decimalyear')  # verify conversion
    t_SDSS = t_SDSS.jd
    time = t_SDSS

    # Find the RA & DEC at selected time
    ra_result_SDSS, dec_result_SDSS = proper_motion(RA, DEC, distance, radial_velocity,
                                                    proper_motion_RA, proper_motion_DEC, time, t_GAIA)

    # Verify if the obtained coordinates match SDSS within 1 arsec
    ra_match_SDSS = math.isclose(ra_result_SDSS, ra_SDSS[l], abs_tol=1 / 3600)
    dec_match_SDSS = math.isclose(dec_result_SDSS, dec_SDSS[l], abs_tol=1 / 3600)

    # print(ra_result_SDSS)
    # print(ra_SDSS[l])
    # print(ra_match_SDSS)
    # print(dec_result_SDSS)
    # print(dec_SDSS[l])
    # print(dec_match_SDSS)

    if ra_match_SDSS == True and dec_match_SDSS == True:
        match = True
        match_array = np.append(match_array, match)  # keep track of match results
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        index_array_SDSS = np.append(index_array_SDSS, l)

        '''
        Set a match quality flag: How many sigmas are we away from the real result?
        Compute in terms of arcseconds 
        '''
        cosdec_SDSS = math.cos(math.radians(dec_SDSS[l] * 3600))  # This is cos(declination) correction factor

        # ***********************************
        # Calculate the standard deviation along each coordinate for the largest error given in the catalog
        # (semi-major axis of uncertainty ellipse)
        error_circle_SDSS = np.array([[(ra_SDSS[l] * 3600), (ra_SDSS[l] * 3600 + position_errorRA_SDSS[l])],
                                      [(dec_SDSS[l] * 3600), (dec_SDSS[l] * 3600 + position_errorDEC_SDSS[l])]])
        standard_dev_SDSS = np.std(error_circle_SDSS, axis=1)

        # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
        error_circle_match_SDSS = np.array([[(ra_SDSS[l] * 3600) * cosdec_SDSS, (ra_result_SDSS * 3600) * cosdec_SDSS],
                                            [(dec_SDSS[l] * 3600), (dec_result_SDSS * 3600)]])
        standard_dev_match_SDSS = np.std(error_circle_match_SDSS, axis=1)

        # Take the standard deviation radii
        standard_dev_radius_sigma1_SDSS = math.sqrt(
            math.pow(standard_dev_SDSS[0], 2) + math.pow(standard_dev_SDSS[1], 2))
        standard_dev_radius_match_SDSS = math.sqrt(
            math.pow(standard_dev_match_SDSS[0], 2) + math.pow(standard_dev_match_SDSS[1], 2))

        # Take the ratio between the two radii
        standard_dev_radius_SDSS = standard_dev_radius_match_SDSS / standard_dev_radius_sigma1_SDSS
        standard_dev_radius_SDSS = float((format(standard_dev_radius_SDSS, '.2f')))  # format to 2 decimals

        print(colored('The target star was found in SDSS for Gaia star %s' % source_id_Gaia, 'green'))
        print(colored('The target is within a radius of %s sigma from the SDSS candidate' % standard_dev_radius_SDSS,
                      'green'))
        print('===================================================================================')
        print(query_result_SDSS[l])
        print()
        print('')

        sigma_array_SDSS = np.append(sigma_array_SDSS, standard_dev_radius_SDSS)

    else:
        match = False
        match_array = np.append(match_array, match)
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

if all(element == False for element in match_array) == True:
    print()
    print('No match was found in SDSS for Gaia star ', source_id_Gaia)
    print()

else:
    # If we did find matches, we will retrieve the spectral information to later plot
    # a Spectral Energy Distribution graph
    index_SDSS = np.argmin(sigma_array_SDSS)
    best_match_SDSS = int(index_array_SDSS[index_SDSS])

    u_mag_SDSS = query_result_SDSS['umag'][best_match_SDSS]
    g_mag_SDSS = query_result_SDSS['gmag'][best_match_SDSS]
    r_mag_SDSS = query_result_SDSS['rmag'][best_match_SDSS]
    i_mag_SDSS = query_result_SDSS['imag'][best_match_SDSS]
    z_mag_SDSS = query_result_SDSS['zmag'][best_match_SDSS]

    u_mag_SDSS_error = query_result_SDSS['e_umag'][best_match_SDSS]
    g_mag_SDSS_error = query_result_SDSS['e_gmag'][best_match_SDSS]
    r_mag_SDSS_error = query_result_SDSS['e_rmag'][best_match_SDSS]
    i_mag_SDSS_error = query_result_SDSS['e_imag'][best_match_SDSS]
    z_mag_SDSS_error = query_result_SDSS['e_zmag'][best_match_SDSS]

    if len(sigma_array_SDSS) > 1:
        print(colored(
            'WARNING: More than one candidate was found in this catalog within one arcsecond. '
            'The one with the lowest sigma will be used for the Spectral Energy Distribution',
            'red'))

# 12. Find Stars in SKYMAPPER and find a match with Gaia based on their epoch ------------------------------------------

g_mag_SKYMAPPER = -1

index_array_SKYMAPPER = np.array([]) # We will store the indices of our successful matches here
sigma_array_SKYMAPPER = np.array([]) # We will store the obtained sigmas

match_array = np.array([])
for o in range(len(ra_SKYMAPPER)):

    t = epoch_SKYMAPPER[o]
    t_SKYMAPPER = Time(t, format='iso')  # verify conversion
    t_SKYMAPPER = t_SKYMAPPER.jd
    time = t_SKYMAPPER

    # Find the RA & DEC at selected time
    ra_result_SKYMAPPER, dec_result_SKYMAPPER = proper_motion(RA, DEC, distance, radial_velocity,
                                                              proper_motion_RA, proper_motion_DEC, time, t_GAIA)

    # Verify if the obtained coordinates match SKYMAPPER within 1 arsec
    ra_match_SKYMAPPER = math.isclose(ra_result_SKYMAPPER, ra_SKYMAPPER[o], abs_tol=1 / 3600)
    dec_match_SKYMAPPER = math.isclose(dec_result_SKYMAPPER, dec_SKYMAPPER[o], abs_tol=1 / 3600)

    # print(ra_result_SKYMAPPER)
    # print(ra_SKYMAPPER[o])
    # print(ra_match_SKYMAPPER)
    # print(dec_result_SKYMAPPER)
    # print(dec_SKYMAPPER[o])
    # print(dec_match_SKYMAPPER)

    if ra_match_SKYMAPPER == True and dec_match_SKYMAPPER == True:
        match = True
        match_array = np.append(match_array, match)  # keep track of match results
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        index_array_SKYMAPPER = np.append(index_array_SKYMAPPER, o)

        '''
        Set a match quality flag: How many sigmas are we away from the real result?
        Compute in terms of arcseconds 
        '''
        cosdec_SKYMAPPER = math.cos(math.radians(dec_SKYMAPPER[o] * 3600))  # This is cos(declination) correction factor

        # ***********************************
        # Calculate the standard deviation along each coordinate for the largest error given in the catalog
        # (semi-major axis of uncertainty ellipse)
        error_circle_SKYMAPPER = np.array(
            [[(ra_SKYMAPPER[o] * 3600), (ra_SKYMAPPER[o] * 3600 + position_error_SKYMAPPER[o])],
             [(dec_SKYMAPPER[o] * 3600), (dec_SKYMAPPER[o] * 3600 + position_error_SKYMAPPER[o])]])
        standard_dev_SKYMAPPER = np.std(error_circle_SKYMAPPER, axis=1)

        # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
        error_circle_match_SKYMAPPER = np.array(
            [[(ra_SKYMAPPER[o] * 3600) * cosdec_SKYMAPPER, (ra_result_SKYMAPPER * 3600) * cosdec_SKYMAPPER],
             [(dec_SKYMAPPER[o] * 3600), (dec_result_SKYMAPPER * 3600)]])
        standard_dev_match_SKYMAPPER = np.std(error_circle_match_SKYMAPPER, axis=1)

        # Take the standard deviation radii
        standard_dev_radius_sigma1_SKYMAPPER = math.sqrt(
            math.pow(standard_dev_SKYMAPPER[0], 2) + math.pow(standard_dev_SKYMAPPER[1], 2))
        standard_dev_radius_match_SKYMAPPER = math.sqrt(
            math.pow(standard_dev_match_SKYMAPPER[0], 2) + math.pow(standard_dev_match_SKYMAPPER[1], 2))

        # Take the ratio between the two radii
        standard_dev_radius_SKYMAPPER = standard_dev_radius_match_SKYMAPPER / standard_dev_radius_sigma1_SKYMAPPER
        standard_dev_radius_SKYMAPPER = float((format(standard_dev_radius_SKYMAPPER, '.2f')))  # format to 2 decimals

        print(colored('The target star was found in SKYMAPPER for Gaia star %s' % source_id_Gaia, 'green'))
        print(colored(
            'The target is within a radius of %s sigma from the SKYMAPPER candidate' % (standard_dev_radius_SKYMAPPER),
            'green'))
        print('===================================================================================')
        print(query_result_SKYMAPPER[o])
        print()
        print('')

        sigma_array_SKYMAPPER = np.append(sigma_array_SKYMAPPER, standard_dev_radius_SKYMAPPER)

    else:
        match = False
        match_array = np.append(match_array, match)
        ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

if all(element == False for element in match_array) == True:
    print()
    print('No match was found in SKYMAPPER for Gaia star ', source_id_Gaia)
    print()

else:
    # If we did find matches, we will retrieve the spectral information to later plot
    # a Spectral Energy Distribution graph
    index_SKYMAPPER = np.argmin(sigma_array_SKYMAPPER)
    best_match_SKYMAPPER = int(index_array_SKYMAPPER[index_SKYMAPPER])

    g_mag_SKYMAPPER = query_result_SKYMAPPER['gmag'][best_match_SKYMAPPER]


    if len(sigma_array_SKYMAPPER) > 1:
        print(colored(
            'WARNING: More than one candidate was found in this catalog within one arcsecond. '
            'The one with the lowest sigma will be used for the Spectral Energy Distribution',
            'red'))

# ======================================================================================================================
# ======================================================================================================================


# 10. Loop through the other Candidates in case there is no match ------------------------------------------------------

# if all(element == False for element in ALL_QUERIES_RESULTS) == True:
'''
With this condition we verify that we haven't found anything previously 
and we try the other Gaia stars in the selected field of view.
Then, we will check again if the search has yielded any positive results. 
'''

for p in range(len(query_result_GAIA)):
    if p == index_brightest:  # we don't want to search the same star twice, so skip it
        pass
    else:
        print()
        print()
        print(colored('*****************************************************************************', 'blue'))
        print(colored('           Check the next Gaia candidate for a match             ', 'blue'))
        print(colored('*****************************************************************************', 'blue'))
        print()
        print()

        source_id_Gaia = query_result_GAIA['Source'][p]
        ra_Gaia = query_result_GAIA['RA_ICRS'][p]
        dec_Gaia = query_result_GAIA['DE_ICRS'][p]
        pmRA_Gaia = query_result_GAIA['pmRA'][p]
        pmDE_Gaia = query_result_GAIA['pmDE'][p]
        parallax_Gaia = query_result_GAIA['Plx'][p]
        radial_velocity_Gaia = query_result_GAIA['RV'][p]

        if (math.isnan(radial_velocity_Gaia)):  # in case the radial velocity is not given, set it to zero
            radial_velocity_Gaia = 0

        # 7. Put the Gaia data in the right units to calcuate the proper motion  -----------------------------------------------
        c = SkyCoord(ra_Gaia, dec_Gaia, frame='icrs', unit='deg')
        RA = c.ra.hour  # hours
        DEC = c.dec.deg  # degrees
        distance = (1 / parallax_Gaia) / 1000  # put the distance in pc
        radial_velocity = radial_velocity_Gaia  # in km/s
        proper_motion_RA = (pmRA_Gaia / 1000)  # in second of arc per year
        proper_motion_DEC = pmDE_Gaia / 1000  # in second of arc per year

        t_GAIA = Time(epoch_GAIA, format='decimalyear')
        t_GAIA = t_GAIA.jd

        # 8. Find Stars in 2MASS and find a match with Gaia based on their epoch -----------------------------------------------
        match_array2 = np.array([])
        for i in range(len(ra_2MASS)):

            # epoch of 2MASS is already in JD
            t_2MASS = epoch_2MASS[i]
            time = t_2MASS

            # Find the RA & DEC at selected time
            ra_result_2MASS, dec_result_2MASS = proper_motion(RA, DEC, distance, radial_velocity,
                                                              proper_motion_RA, proper_motion_DEC, time, t_GAIA)

            # Verify if the obtained coordinates match 2MASS within 1 arsec
            ra_match_2MASS = math.isclose(ra_result_2MASS, ra_2MASS[i], abs_tol=1 / 3600)
            dec_match_2MASS = math.isclose(dec_result_2MASS, dec_2MASS[i], abs_tol=1 / 3600)

            # print(ra_result_2MASS)
            # print(ra_2MASS[i])
            # print(ra_match_2MASS)
            # print(dec_result_2MASS)
            # print(dec_2MASS[i])
            # print(dec_match_2MASS)

            if ra_match_2MASS == True and dec_match_2MASS == True:
                match = True
                match_array2 = np.append(match_array2, match)
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

                '''
                Set a match quality flag: How many sigmas are we away from the real result?
                Compute in terms of arcseconds 
                '''
                cosdec_2MASS = math.cos(math.radians(dec_2MASS[i] * 3600))  # This is cos(declination) correction factor

                # ***********************************
                # Calculate the standard deviation along each coordinate for the largest error given in the catalog
                # (semi-major axis of uncertainty ellipse)
                error_circle_2MASS = np.array([[(ra_2MASS[i] * 3600), (ra_2MASS[i] * 3600 + position_error_2MASS[i])],
                                               [(dec_2MASS[i] * 3600),
                                                (dec_2MASS[i] * 3600 + position_error_2MASS[i])]])
                standard_dev_2MASS = np.std(error_circle_2MASS, axis=1)

                # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
                error_circle_match_2MASS = np.array(
                    [[(ra_2MASS[i] * 3600) * cosdec_2MASS, (ra_result_2MASS * 3600) * cosdec_2MASS],
                     [(dec_2MASS[i] * 3600), (dec_result_2MASS * 3600)]])
                standard_dev_match_2MASS = np.std(error_circle_match_2MASS, axis=1)

                # Take the standard deviation radii
                standard_dev_radius_sigma1_2MASS = math.sqrt(
                    math.pow(standard_dev_2MASS[0], 2) + math.pow(standard_dev_2MASS[1], 2))
                standard_dev_radius_match_2MASS = math.sqrt(
                    math.pow(standard_dev_match_2MASS[0], 2) + math.pow(standard_dev_match_2MASS[1], 2))

                # Take the ratio between the two radii
                standard_dev_radius_2MASS = standard_dev_radius_match_2MASS / standard_dev_radius_sigma1_2MASS
                standard_dev_radius_2MASS = float((format(standard_dev_radius_2MASS, '.2f')))  # format to 2 decimals

                print(colored('The target star was found in 2MASS for Gaia star %s' % source_id_Gaia, 'green'))
                print(colored(
                    'The target is within a radius of %s sigma from the 2MASS candidate' % (standard_dev_radius_2MASS),
                    'green'))
                print('===================================================================================')
                print(query_result_2MASS[i])
                print('')
            else:
                match = False
                match_array2 = np.append(match_array2, match)
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        if all(element == False for element in match_array2) == True:
            print()
            print('No match was found in 2MASS for Gaia star ', source_id_Gaia)
            print()

        # 9. Find Stars in WISE and find a match with Gaia ---------------------------------------------------------------------
        '''
        The epoch for WISE is set as 2010.5
        '''
        match_array2 = np.array([])
        for j in range(len(ra_WISE)):

            time = ref_epoch_WISE

            # Find the RA & DEC at selected time
            ra_result_WISE, dec_result_WISE = proper_motion(RA, DEC, distance, radial_velocity,
                                                            proper_motion_RA, proper_motion_DEC, time, t_GAIA)

            # Verify if the obtained coordinates match WISE within 1 arsec
            ra_match_WISE = math.isclose(ra_result_WISE, ra_WISE[j], abs_tol=1 / 3600)
            dec_match_WISE = math.isclose(dec_result_WISE, dec_WISE[j], abs_tol=1 / 3600)

            # print(ra_result_WISE)
            # print(ra_WISE[j])
            # print(ra_match_WISE)
            # print(dec_result_WISE)
            # print(dec_WISE[j])
            # print(dec_match_WISE)

            if ra_match_WISE == True and dec_match_WISE == True:
                match = True
                match_array2 = np.append(match_array2, match)
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

                '''
                Set a match quality flag: How many sigmas are we away from the real result?
                Compute in terms of arcseconds 
                '''
                cosdec_WISE = math.cos(math.radians(dec_WISE[j] * 3600))  # This is cos(declination) correction factor

                # ***********************************
                # Calculate the standard deviation along each coordinate for the largest error given in the catalog
                # (semi-major axis of uncertainty ellipse)
                error_circle_WISE = np.array([[(ra_WISE[j] * 3600), (ra_WISE[j] * 3600 + position_error_WISE[j])],
                                              [(dec_WISE[j] * 3600), (dec_WISE[j] * 3600 + position_error_WISE[j])]])
                standard_dev_WISE = np.std(error_circle_WISE, axis=1)

                # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
                error_circle_match_WISE = np.array(
                    [[(ra_WISE[j] * 3600) * cosdec_WISE, (ra_result_WISE * 3600) * cosdec_WISE],
                     [(dec_WISE[j] * 3600), (dec_result_WISE * 3600)]])
                standard_dev_match_WISE = np.std(error_circle_match_WISE, axis=1)

                # Take the standard deviation radii
                standard_dev_radius_sigma1_WISE = math.sqrt(
                    math.pow(standard_dev_WISE[0], 2) + math.pow(standard_dev_WISE[1], 2))
                standard_dev_radius_match_WISE = math.sqrt(
                    math.pow(standard_dev_match_WISE[0], 2) + math.pow(standard_dev_match_WISE[1], 2))

                # Take the ratio between the two radii
                standard_dev_radius_WISE = standard_dev_radius_match_WISE / standard_dev_radius_sigma1_WISE
                standard_dev_radius_WISE = float((format(standard_dev_radius_WISE, '.2f')))  # format to 2 decimals

                print(colored('The target star was found in WISE for Gaia star %s' % source_id_Gaia, 'green'))
                print(colored(
                    'The target is within a radius of %s sigma from the WISE candidate' % standard_dev_radius_WISE,
                    'green'))
                print('===================================================================================')
                print(query_result_WISE[j])
                print('')
            else:
                match = False
                match_array2 = np.append(match_array2, match)
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        if all(element == False for element in match_array2) == True:
            print()
            print('No match was found in WISE for Gaia star ', source_id_Gaia)
            print()

        # 10. Find Stars in PANSTARRS and find a match with Gaia based on their epoch -----------------------------------------------

        match_array = np.array([])
        for h in range(len(ra_PANSTARRS)):

            t = epoch_PANSTARRS[h]
            t_PANSTARRS = Time(t, format='mjd')  # verify conversion
            t_PANSTARRS = t_PANSTARRS.jd
            # t_PANSTARRS = t_PANSTARRS.decimalyear
            # time = - (2015.5 - t_PANSTARRS) # Gaia is the reference epoch so the time should be wtr to 2015.5 in yr
            time = t_PANSTARRS

            # Find the RA & DEC at selected time
            ra_result_PANSTARRS, dec_result_PANSTARRS = proper_motion(RA, DEC, distance, radial_velocity,
                                                                      proper_motion_RA, proper_motion_DEC, time, t_GAIA)

            # Verify if the obtained coordinates match PANSTARRS within 1 arsec
            ra_match_PANSTARRS = math.isclose(ra_result_PANSTARRS, ra_PANSTARRS[h], abs_tol=1 / 3600)
            dec_match_PANSTARRS = math.isclose(dec_result_PANSTARRS, dec_PANSTARRS[h], abs_tol=1 / 3600)

            # print(ra_result_PANSTARRS)
            # print(ra_PANSTARRS[h])
            # print(ra_match_PANSTARRS)
            # print(dec_result_PANSTARRS)
            # print(dec_PANSTARRS[h])
            # print(dec_match_PANSTARRS)

            if ra_match_PANSTARRS == True and dec_match_PANSTARRS == True:
                match = True
                match_array = np.append(match_array, match)  # keep track of match results
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

                '''
                Set a match quality flag: How many sigmas are we away from the real result?
                Compute in terms of arcseconds 
                '''
                cosdec_PANSTARRS = math.cos(
                    math.radians(dec_PANSTARRS[h] * 3600))  # This is cos(declination) correction factor

                # ***********************************
                # Calculate the standard deviation along each coordinate for the largest error given in the catalog
                # (semi-major axis of uncertainty ellipse)
                error_circle_PANSTARRS = np.array(
                    [[(ra_PANSTARRS[h] * 3600), (ra_PANSTARRS[h] * 3600 + position_errorRA_PANSTARRS[h])],
                     [(dec_PANSTARRS[h] * 3600), (dec_PANSTARRS[h] * 3600 + position_errorDEC_PANSTARRS[h])]])
                standard_dev_PANSTARRS = np.std(error_circle_PANSTARRS, axis=1)

                # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
                error_circle_match_PANSTARRS = np.array(
                    [[(ra_PANSTARRS[h] * 3600) * cosdec_PANSTARRS, (ra_result_PANSTARRS * 3600) * cosdec_PANSTARRS],
                     [(dec_PANSTARRS[h] * 3600), (dec_result_PANSTARRS * 3600)]])
                standard_dev_match_PANSTARRS = np.std(error_circle_match_PANSTARRS, axis=1)

                # Take the standard deviation radii
                standard_dev_radius_sigma1_PANSTARRS = math.sqrt(
                    math.pow(standard_dev_PANSTARRS[0], 2) + math.pow(standard_dev_PANSTARRS[1], 2))
                standard_dev_radius_match_PANSTARRS = math.sqrt(
                    math.pow(standard_dev_match_PANSTARRS[0], 2) + math.pow(standard_dev_match_PANSTARRS[1], 2))

                # Take the ratio between the two radii
                standard_dev_radius_PANSTARRS = standard_dev_radius_match_PANSTARRS / standard_dev_radius_sigma1_PANSTARRS
                standard_dev_radius_PANSTARRS = float(
                    (format(standard_dev_radius_PANSTARRS, '.2f')))  # format to 2 decimals

                print(colored('The target star was found in PANSTARRS for Gaia star %s' % source_id_Gaia, 'green'))
                print(colored(
                    'The target is within a radius of %s sigma from the PANSTARRS candidate' % standard_dev_radius_PANSTARRS,
                    'green'))
                print('===================================================================================')
                print(query_result_PANSTARRS[h])
                print()
                print('')
            else:
                match = False
                match_array = np.append(match_array, match)
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        if all(element == False for element in match_array) == True:
            print()
            print('No match was found in PANSTARRS for Gaia star ', source_id_Gaia)
            print()

        # 11. Find Stars in SDSS and find a match with Gaia based on their epoch -----------------------------------------------

        match_array = np.array([])
        for l in range(len(ra_SDSS)):

            t = epoch_SDSS[l]
            t_SDSS = Time(t, format='decimalyear')  # verify conversion
            t_SDSS = t_SDSS.jd
            # t_SDSS = t_SDSS.decimalyear
            # t_SDSS = epoch_SDSS[l]
            # time = - (2015.5 - t_SDSS) # Gaia is the reference epoch so the time should be wtr to 2015.5 in yr
            time = t_SDSS

            # Find the RA & DEC at selected time
            ra_result_SDSS, dec_result_SDSS = proper_motion(RA, DEC, distance, radial_velocity,
                                                            proper_motion_RA, proper_motion_DEC, time, t_GAIA)

            # Verify if the obtained coordinates match SDSS within 1 arsec
            ra_match_SDSS = math.isclose(ra_result_SDSS, ra_SDSS[l], abs_tol=1 / 3600)
            dec_match_SDSS = math.isclose(dec_result_SDSS, dec_SDSS[l], abs_tol=1 / 3600)

            # print(ra_result_SDSS)
            # print(ra_SDSS[l])
            # print(ra_match_SDSS)
            # print(dec_result_SDSS)
            # print(dec_SDSS[l])
            # print(dec_match_SDSS)

            if ra_match_SDSS == True and dec_match_SDSS == True:
                match = True
                match_array = np.append(match_array, match)  # keep track of match results
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

                '''
                Set a match quality flag: How many sigmas are we away from the real result?
                Compute in terms of arcseconds 
                '''
                cosdec_SDSS = math.cos(math.radians(dec_SDSS[l] * 3600))  # This is cos(declination) correction factor

                # ***********************************
                # Calculate the standard deviation along each coordinate for the largest error given in the catalog
                # (semi-major axis of uncertainty ellipse)
                error_circle_SDSS = np.array([[(ra_SDSS[l] * 3600), (ra_SDSS[l] * 3600 + position_errorRA_SDSS[l])],
                                              [(dec_SDSS[l] * 3600), (dec_SDSS[l] * 3600 + position_errorDEC_SDSS[l])]])
                standard_dev_SDSS = np.std(error_circle_SDSS, axis=1)

                # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
                error_circle_match_SDSS = np.array(
                    [[(ra_SDSS[l] * 3600) * cosdec_SDSS, (ra_result_SDSS * 3600) * cosdec_SDSS],
                     [(dec_SDSS[l] * 3600), (dec_result_SDSS * 3600)]])
                standard_dev_match_SDSS = np.std(error_circle_match_SDSS, axis=1)

                # Take the standard deviation radii
                standard_dev_radius_sigma1_SDSS = math.sqrt(
                    math.pow(standard_dev_SDSS[0], 2) + math.pow(standard_dev_SDSS[1], 2))
                standard_dev_radius_match_SDSS = math.sqrt(
                    math.pow(standard_dev_match_SDSS[0], 2) + math.pow(standard_dev_match_SDSS[1], 2))

                # Take the ratio between the two radii
                standard_dev_radius_SDSS = standard_dev_radius_match_SDSS / standard_dev_radius_sigma1_SDSS
                standard_dev_radius_SDSS = float((format(standard_dev_radius_SDSS, '.2f')))  # format to 2 decimals

                print(colored('The target star was found in SDSS for Gaia star %s' % source_id_Gaia, 'green'))
                print(colored(
                    'The target is within a radius of %s sigma from the SDSS candidate' % standard_dev_radius_SDSS,
                    'green'))
                print('===================================================================================')
                print(query_result_SDSS[l])
                print()
                print('')
            else:
                match = False
                match_array = np.append(match_array, match)
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        if all(element == False for element in match_array) == True:
            print()
            print('No match was found in SDSS for Gaia star ', source_id_Gaia)
            print()

        # 12. Find Stars in SKYMAPPER and find a match with Gaia based on their epoch ------------------------------------------

        match_array = np.array([])
        for o in range(len(ra_SKYMAPPER)):

            t = epoch_SKYMAPPER[o]
            t_SKYMAPPER = Time(t, format='iso')  # verify conversion
            t_SKYMAPPER = t_SKYMAPPER.jd
            time = t_SKYMAPPER

            # Find the RA & DEC at selected time
            ra_result_SKYMAPPER, dec_result_SKYMAPPER = proper_motion(RA, DEC, distance, radial_velocity,
                                                                      proper_motion_RA, proper_motion_DEC, time, t_GAIA)

            # Verify if the obtained coordinates match SKYMAPPER within 1 arsec
            ra_match_SKYMAPPER = math.isclose(ra_result_SKYMAPPER, ra_SKYMAPPER[o], abs_tol=1 / 3600)
            dec_match_SKYMAPPER = math.isclose(dec_result_SKYMAPPER, dec_SKYMAPPER[o], abs_tol=1 / 3600)

            # print(ra_result_SKYMAPPER)
            # print(ra_SKYMAPPER[o])
            # print(ra_match_SKYMAPPER)
            # print(dec_result_SKYMAPPER)
            # print(dec_SKYMAPPER[o])
            # print(dec_match_SKYMAPPER)

            if ra_match_SKYMAPPER == True and dec_match_SKYMAPPER == True:
                match = True
                match_array = np.append(match_array, match)  # keep track of match results
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

                '''
                Set a match quality flag: How many sigmas are we away from the real result?
                Compute in terms of arcseconds 
                '''
                cosdec_SKYMAPPER = math.cos(
                    math.radians(dec_SKYMAPPER[o] * 3600))  # This is cos(declination) correction factor

                # ***********************************
                # Calculate the standard deviation along each coordinate for the largest error given in the catalog
                # (semi-major axis of uncertainty ellipse)
                error_circle_SKYMAPPER = np.array(
                    [[(ra_SKYMAPPER[o] * 3600), (ra_SKYMAPPER[o] * 3600 + position_error_SKYMAPPER[o])],
                     [(dec_SKYMAPPER[o] * 3600), (dec_SKYMAPPER[o] * 3600 + position_error_SKYMAPPER[o])]])
                standard_dev_SKYMAPPER = np.std(error_circle_SKYMAPPER, axis=1)

                # Take the standard deviation along each coordinate for the calculated star (computed from Gaia PM)
                error_circle_match_SKYMAPPER = np.array(
                    [[(ra_SKYMAPPER[o] * 3600) * cosdec_SKYMAPPER, (ra_result_SKYMAPPER * 3600) * cosdec_SKYMAPPER],
                     [(dec_SKYMAPPER[o] * 3600), (dec_result_SKYMAPPER * 3600)]])
                standard_dev_match_SKYMAPPER = np.std(error_circle_match_SKYMAPPER, axis=1)

                # Take the standard deviation radii
                standard_dev_radius_sigma1_SKYMAPPER = math.sqrt(
                    math.pow(standard_dev_SKYMAPPER[0], 2) + math.pow(standard_dev_SKYMAPPER[1], 2))
                standard_dev_radius_match_SKYMAPPER = math.sqrt(
                    math.pow(standard_dev_match_SKYMAPPER[0], 2) + math.pow(standard_dev_match_SKYMAPPER[1], 2))

                # Take the ratio between the two radii
                standard_dev_radius_SKYMAPPER = standard_dev_radius_match_SKYMAPPER / standard_dev_radius_sigma1_SKYMAPPER
                standard_dev_radius_SKYMAPPER = float(
                    (format(standard_dev_radius_SKYMAPPER, '.2f')))  # format to 2 decimals

                print(colored('The target star was found in SKYMAPPER for Gaia star %s' % source_id_Gaia, 'green'))
                print(colored('The target is within a radius of %s sigma from the SKYMAPPER candidate' % (
                    standard_dev_radius_SKYMAPPER), 'green'))
                print('===================================================================================')
                print(query_result_SKYMAPPER[o])
                print()
                print('')
            else:
                match = False
                match_array = np.append(match_array, match)
                ALL_QUERIES_RESULTS = np.append(ALL_QUERIES_RESULTS, match)

        if all(element == False for element in match_array) == True:
            print()
            print('No match was found in SKYMAPPER for Gaia star ', source_id_Gaia)
            print()

        # if all(index == False for index in ALL_QUERIES_RESULTS) == False:
        # print('At least one result was found!')
        # break # if one of the queries was successful break out of the loop

# 11. Indicate if all results were negative ----------------------------------------------------------------------------
if all(element == False for element in ALL_QUERIES_RESULTS) == True:
    print('--------------------------------------------------------------')
    print()
    print('No match was found in any catalog based on the Gaia results, verify input')
    print()

# ========================================= Spectral Energy Distribution Graph =========================================
'''
Information needed is collected in each match=true loop to collect the spectral information 
(G_mag_GAIA = query_result_GAIA['Gmag'][indexbrightest]), and on the Spanish Virtual Observatory Filter Profile Service 


Get all the data from: http://svo2.cab.inta-csic.es/theory/fps3/index.php?mode=browse

1. Get Zero Point flux (Jy) calculated for the right system that matches the catalog magnitude scale 
2. Get Lambda effective calculated (A)
3. Get Lambda min (A)
4. Get Lambda max (A)
'''

# -------------------------------------------------------- GAIA --------------------------------------------------------

# Zero Point Flux in Jy & Lambda effective from Config file

# Convert to micrometers
G_mag_lambda_eff_GAIA = G_mag_lambda_eff_GAIA / 10000
BP_mag_lambda_eff_GAIA = BP_mag_lambda_eff_GAIA / 10000
RP_mag_lambda_eff_GAIA = RP_mag_lambda_eff_GAIA / 10000

# Compute the flux and the error bars

# Zero point flux density for each filter (at magnitude = 0)
G_mag_zero_flux_GAIA = G_mag_zero_flux_VEGA_GAIA * (3 * math.pow(10, -12)) / math.pow(G_mag_lambda_eff_GAIA, 2)
BP_mag_zero_flux_GAIA = BP_mag_zero_flux_VEGA_GAIA * (3 * math.pow(10, -12)) / math.pow(BP_mag_lambda_eff_GAIA, 2)
RP_mag_zero_flux_GAIA = RP_mag_zero_flux_VEGA_GAIA * (3 * math.pow(10, -12)) / math.pow(RP_mag_lambda_eff_GAIA, 2)

# Calculate the observed flux density for each band
# The math module is not used because it can't hold the uncertainties
G_mag_flux_GAIA = (G_mag_zero_flux_GAIA * 10 ** (-0.4 * G_mag_GAIA))
BP_mag_flux_GAIA = (BP_mag_zero_flux_GAIA * 10 ** (-0.4 * BP_mag_GAIA))
RP_mag_flux_GAIA = (RP_mag_zero_flux_GAIA * 10 ** (-0.4 * RP_mag_GAIA))

# Store the results in arrays
lambda_array_GAIA = np.array([G_mag_lambda_eff_GAIA, BP_mag_lambda_eff_GAIA, RP_mag_lambda_eff_GAIA])

flux_array_GAIA = np.array(
    [G_mag_flux_GAIA.nominal_value, BP_mag_flux_GAIA.nominal_value, RP_mag_flux_GAIA.nominal_value])
flux_error_array_GAIA = np.array([G_mag_flux_GAIA.s, BP_mag_flux_GAIA.s, RP_mag_flux_GAIA.s])


# ------------------------------------------------------------ 2MASS ---------------------------------------------------

# Zero Point Flux in Jy & Lambda effective from Config file

# Convert to micrometers
J_mag_lambda_eff_2MASS = J_mag_lambda_eff_2MASS / 10000
H_mag_lambda_eff_2MASS = H_mag_lambda_eff_2MASS / 10000
K_mag_lambda_eff_2MASS = K_mag_lambda_eff_2MASS / 10000

# Compute the flux and the error bars

if J_mag_2MASS != -1 or math.isnan(J_mag_2MASS):

    # Zero point flux density for each filter (at magnitude = 0)
    J_mag_zero_flux_2MASS = J_mag_zero_flux_VEGA_2MASS * (3 * math.pow(10, -12)) / math.pow(J_mag_lambda_eff_2MASS, 2)
    H_mag_zero_flux_2MASS = H_mag_zero_flux_VEGA_2MASS * (3 * math.pow(10, -12)) / math.pow(H_mag_lambda_eff_2MASS, 2)
    K_mag_zero_flux_2MASS = K_mag_zero_flux_VEGA_2MASS * (3 * math.pow(10, -12)) / math.pow(K_mag_lambda_eff_2MASS, 2)

    # Calculate the observed flux density for each band
    # The math module is not used because it can't hold the uncertainties
    J_mag_flux_2MASS = (J_mag_zero_flux_2MASS * 10 ** (-0.4 * err.ufloat(J_mag_2MASS, J_mag_2MASS_error)))
    H_mag_flux_2MASS = (H_mag_zero_flux_2MASS * 10 ** (-0.4 * err.ufloat(H_mag_2MASS, H_mag_2MASS_error)))
    K_mag_flux_2MASS = (K_mag_zero_flux_2MASS * 10 ** (-0.4 * err.ufloat(K_mag_2MASS, K_mag_2MASS_error)))

    # Store the results in arrays
    lambda_array_2MASS = np.array([J_mag_lambda_eff_2MASS, H_mag_lambda_eff_2MASS, K_mag_lambda_eff_2MASS])

    flux_array_2MASS = np.array(
        [J_mag_flux_2MASS.nominal_value, H_mag_flux_2MASS.nominal_value, K_mag_flux_2MASS.nominal_value])
    flux_error_array_2MASS = np.array([J_mag_flux_2MASS.s, H_mag_flux_2MASS.s, K_mag_flux_2MASS.s])

else:
    lambda_array_2MASS = np.array([np.nan, np.nan, np.nan])
    lambda_error_min_array_2MASS = np.array([np.nan, np.nan, np.nan])  # lambda_min error bars: lambda_eff - lambda_min
    lambda_error_max_array_2MASS = np.array([np.nan, np.nan, np.nan])  # lambda_max error bars: lambda_max - lambda_eff
    flux_array_2MASS = np.array([np.nan, np.nan, np.nan])
    flux_error_array_2MASS = np.array([np.nan, np.nan, np.nan])


# -------------------------------------------------------- WISE --------------------------------------------------------

# Zero Point Flux in Jy & Lambda effective from Config file

# Convert to micrometers
W1_mag_lambda_eff_WISE = W1_mag_lambda_eff_WISE / 10000
W2_mag_lambda_eff_WISE = W2_mag_lambda_eff_WISE / 10000
W3_mag_lambda_eff_WISE = W3_mag_lambda_eff_WISE / 10000
W4_mag_lambda_eff_WISE = W4_mag_lambda_eff_WISE / 10000


# Compute the flux and the error bars

if W1_mag_WISE != -1 or math.isnan(W1_mag_WISE):

    # Zero point flux density for each filter (at magnitude = 0)
    W1_mag_zero_flux_WISE = W1_mag_zero_flux_VEGA_WISE * (3 * math.pow(10, -12)) / math.pow(W1_mag_lambda_eff_WISE, 2)
    W2_mag_zero_flux_WISE = W2_mag_zero_flux_VEGA_WISE * (3 * math.pow(10, -12)) / math.pow(W2_mag_lambda_eff_WISE, 2)
    W3_mag_zero_flux_WISE = W3_mag_zero_flux_VEGA_WISE * (3 * math.pow(10, -12)) / math.pow(W3_mag_lambda_eff_WISE, 2)
    W4_mag_zero_flux_WISE = W4_mag_zero_flux_VEGA_WISE * (3 * math.pow(10, -12)) / math.pow(W4_mag_lambda_eff_WISE, 2)

    # Calculate the observed flux density for each band
    # The math module is not used because it can't hold the uncertainties
    W1_mag_flux_WISE = (W1_mag_zero_flux_WISE * 10 ** (-0.4 * err.ufloat(W1_mag_WISE, W1_mag_WISE_error)))
    W2_mag_flux_WISE = (W2_mag_zero_flux_WISE * 10 ** (-0.4 * err.ufloat(W2_mag_WISE, W2_mag_WISE_error)))
    W3_mag_flux_WISE = (W3_mag_zero_flux_WISE * 10 ** (-0.4 * err.ufloat(W3_mag_WISE, W3_mag_WISE_error)))
    W4_mag_flux_WISE = (W4_mag_zero_flux_WISE * 10 ** (-0.4 * err.ufloat(W4_mag_WISE, W4_mag_WISE_error)))

    # Store the results in arrays
    lambda_array_WISE = np.array(
        [W1_mag_lambda_eff_WISE, W2_mag_lambda_eff_WISE, W3_mag_lambda_eff_WISE, W4_mag_lambda_eff_WISE])

    flux_array_WISE = np.array(
        [W1_mag_flux_WISE.nominal_value, W2_mag_flux_WISE.nominal_value, W3_mag_flux_WISE.nominal_value,
         W4_mag_flux_WISE.nominal_value])
    flux_error_array_WISE = np.array([W1_mag_flux_WISE.s, W2_mag_flux_WISE.s, W3_mag_flux_WISE.s, W4_mag_flux_WISE.s])

else:
    lambda_array_WISE = np.array([np.nan, np.nan, np.nan, np.nan])
    lambda_error_min_array_WISE = np.array(
        [np.nan, np.nan, np.nan, np.nan])  # lambda_min error bars: lambda_eff - lambda_min
    lambda_error_max_array_WISE = np.array(
        [np.nan, np.nan, np.nan, np.nan])  # lambda_max error bars: lambda_max - lambda_eff
    flux_array_WISE = np.array([np.nan, np.nan, np.nan, np.nan])
    flux_error_array_WISE = np.array([np.nan, np.nan, np.nan, np.nan])

# ----------------------------------------------------------- SDSS -----------------------------------------------------

# Zero Point Flux in Jy & Lambda effective from Config file

# Convert to micrometers
u_mag_lambda_eff_SDSS = u_mag_lambda_eff_SDSS / 10000
g_mag_lambda_eff_SDSS = g_mag_lambda_eff_SDSS / 10000
r_mag_lambda_eff_SDSS = r_mag_lambda_eff_SDSS / 10000
i_mag_lambda_eff_SDSS = i_mag_lambda_eff_SDSS / 10000
z_mag_lambda_eff_SDSS = z_mag_lambda_eff_SDSS / 10000


# Compute the flux and the error bars

if u_mag_SDSS != -1 or math.isnan(u_mag_SDSS):

    # Zero point flux density for each filter (at magnitude = 0)
    u_mag_zero_flux_SDSS = u_mag_zero_flux_AB_SDSS * (3 * math.pow(10, -12)) / math.pow(u_mag_lambda_eff_SDSS, 2)
    g_mag_zero_flux_SDSS = g_mag_zero_flux_AB_SDSS * (3 * math.pow(10, -12)) / math.pow(g_mag_lambda_eff_SDSS, 2)
    r_mag_zero_flux_SDSS = r_mag_zero_flux_AB_SDSS * (3 * math.pow(10, -12)) / math.pow(r_mag_lambda_eff_SDSS, 2)
    i_mag_zero_flux_SDSS = i_mag_zero_flux_AB_SDSS * (3 * math.pow(10, -12)) / math.pow(i_mag_lambda_eff_SDSS, 2)
    z_mag_zero_flux_SDSS = z_mag_zero_flux_AB_SDSS * (3 * math.pow(10, -12)) / math.pow(z_mag_lambda_eff_SDSS, 2)

    # Calculate the observed flux density for each band
    # The math module is not used because it can't hold the uncertainties
    u_mag_flux_SDSS = (u_mag_zero_flux_SDSS * 10 ** (-0.4 * (err.ufloat(u_mag_SDSS, u_mag_SDSS_error))))
    g_mag_flux_SDSS = (g_mag_zero_flux_SDSS * 10 ** (-0.4 * (err.ufloat(g_mag_SDSS, g_mag_SDSS_error))))
    r_mag_flux_SDSS = (r_mag_zero_flux_SDSS * 10 ** (-0.4 * (err.ufloat(r_mag_SDSS, r_mag_SDSS_error))))
    i_mag_flux_SDSS = (i_mag_zero_flux_SDSS * 10 ** (-0.4 * (err.ufloat(i_mag_SDSS, i_mag_SDSS_error))))
    z_mag_flux_SDSS = (z_mag_zero_flux_SDSS * 10 ** (-0.4 * (err.ufloat(z_mag_SDSS, z_mag_SDSS_error))))

    # Store the results in arrays
    lambda_array_SDSS = np.array([u_mag_lambda_eff_SDSS, g_mag_lambda_eff_SDSS,
                                  r_mag_lambda_eff_SDSS, i_mag_lambda_eff_SDSS, z_mag_lambda_eff_SDSS])

    flux_array_SDSS = np.array(
        [u_mag_flux_SDSS.nominal_value, g_mag_flux_SDSS.nominal_value,
         r_mag_flux_SDSS.nominal_value, i_mag_flux_SDSS.nominal_value, z_mag_flux_SDSS.nominal_value])
    flux_error_array_SDSS = np.array([u_mag_flux_SDSS.s, g_mag_flux_SDSS.s,
                                      r_mag_flux_SDSS.s, i_mag_flux_SDSS.s, z_mag_flux_SDSS.s])

else:
    lambda_array_SDSS = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    lambda_error_min_array_SDSS = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan])  # lambda_min error bars: lambda_eff - lambda_min
    lambda_error_max_array_SDSS = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan])  # lambda_max error bars: lambda_max - lambda_eff
    flux_array_SDSS = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    flux_error_array_SDSS = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

# ------------------------------------------------------ PANSTARRS -----------------------------------------------------

# Zero Point Flux in Jy & Lambda effective from Config file

# Convert to micrometers
g_mag_lambda_eff_PANSTARRS = g_mag_lambda_eff_PANSTARRS / 10000
r_mag_lambda_eff_PANSTARRS = r_mag_lambda_eff_PANSTARRS / 10000
i_mag_lambda_eff_PANSTARRS = i_mag_lambda_eff_PANSTARRS / 10000
z_mag_lambda_eff_PANSTARRS = z_mag_lambda_eff_PANSTARRS / 10000
y_mag_lambda_eff_PANSTARRS = y_mag_lambda_eff_PANSTARRS / 10000


# Compute the flux and the error bars

if g_mag_PANSTARRS != -1 or math.isnan(g_mag_PANSTARRS):

    # Zero point flux density for each filter (at magnitude = 0)
    g_mag_zero_flux_PANSTARRS = g_mag_zero_flux_AB_PANSTARRS * (3 * math.pow(10, -12)) / math.pow(
        g_mag_lambda_eff_PANSTARRS, 2)
    r_mag_zero_flux_PANSTARRS = r_mag_zero_flux_AB_PANSTARRS * (3 * math.pow(10, -12)) / math.pow(
        r_mag_lambda_eff_PANSTARRS, 2)
    i_mag_zero_flux_PANSTARRS = i_mag_zero_flux_AB_PANSTARRS * (3 * math.pow(10, -12)) / math.pow(
        i_mag_lambda_eff_PANSTARRS, 2)
    z_mag_zero_flux_PANSTARRS = z_mag_zero_flux_AB_PANSTARRS * (3 * math.pow(10, -12)) / math.pow(
        z_mag_lambda_eff_PANSTARRS, 2)
    y_mag_zero_flux_PANSTARRS = y_mag_zero_flux_AB_PANSTARRS * (3 * math.pow(10, -12)) / math.pow(
        y_mag_lambda_eff_PANSTARRS, 2)

    # Calculate the observed flux density for each band
    # The math module is not used because it can't hold the uncertainties
    g_mag_flux_PANSTARRS = (
            g_mag_zero_flux_PANSTARRS * 10 ** (-0.4 * (err.ufloat(g_mag_PANSTARRS, g_mag_PANSTARRS_error))))
    r_mag_flux_PANSTARRS = (
            r_mag_zero_flux_PANSTARRS * 10 ** (-0.4 * (err.ufloat(r_mag_PANSTARRS, r_mag_PANSTARRS_error))))
    i_mag_flux_PANSTARRS = (
            i_mag_zero_flux_PANSTARRS * 10 ** (-0.4 * (err.ufloat(i_mag_PANSTARRS, i_mag_PANSTARRS_error))))
    z_mag_flux_PANSTARRS = (
            z_mag_zero_flux_PANSTARRS * 10 ** (-0.4 * (err.ufloat(z_mag_PANSTARRS, z_mag_PANSTARRS_error))))
    y_mag_flux_PANSTARRS = (
            y_mag_zero_flux_PANSTARRS * 10 ** (-0.4 * (err.ufloat(y_mag_PANSTARRS, y_mag_PANSTARRS_error))))

    # Store the results in arrays
    lambda_array_PANSTARRS = np.array([g_mag_lambda_eff_PANSTARRS, r_mag_lambda_eff_PANSTARRS,
                                       i_mag_lambda_eff_PANSTARRS, z_mag_lambda_eff_PANSTARRS,
                                       y_mag_lambda_eff_PANSTARRS])

    flux_array_PANSTARRS = np.array(
        [g_mag_flux_PANSTARRS.nominal_value, r_mag_flux_PANSTARRS.nominal_value,
         i_mag_flux_PANSTARRS.nominal_value, z_mag_flux_PANSTARRS.nominal_value, y_mag_flux_PANSTARRS.nominal_value])
    flux_error_array_PANSTARRS = np.array([g_mag_flux_PANSTARRS.s, r_mag_flux_PANSTARRS.s,
                                           i_mag_flux_PANSTARRS.s, z_mag_flux_PANSTARRS.s, y_mag_flux_PANSTARRS.s])

else:
    lambda_array_PANSTARRS = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    lambda_error_min_array_PANSTARRS = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan])  # lambda_min error bars: lambda_eff - lambda_min
    lambda_error_max_array_PANSTARRS = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan])  # lambda_max error bars: lambda_max - lambda_eff
    flux_array_PANSTARRS = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    flux_error_array_PANSTARRS = np.array([np.nan, np.nan, np.nan, np.nan, np.nan])

# -------------------------------------------------------- SKYMAPPER ---------------------------------------------------

# Zero Point Flux in Jy & Lambda effective from Config file

# Convert to micrometers
g_mag_lambda_eff_SKYMAPPER = g_mag_lambda_eff_SKYMAPPER / 10000

# Compute the flux and the error bars

if g_mag_SKYMAPPER != -1 or math.isnan(g_mag_SKYMAPPER):

    # Zero point flux density for each filter (at magnitude = 0)
    g_mag_zero_flux_SKYMAPPER = g_mag_zero_flux_AB_SKYMAPPER * (3 * math.pow(10, -12)) / math.pow(g_mag_lambda_eff_SKYMAPPER, 2)

    # Calculate the observed flux density for each band
    # The math module is not used because it can't hold the uncertainties
    g_mag_flux_SKYMAPPER = (g_mag_zero_flux_SKYMAPPER * 10 ** (-0.4 * err.ufloat(g_mag_SKYMAPPER, np.nan))) # No error on the magnitude was found in Vizier


    # Store the results in arrays
    lambda_array_SKYMAPPER = np.array([g_mag_lambda_eff_SKYMAPPER])
    flux_array_SKYMAPPER = np.array([g_mag_flux_SKYMAPPER.nominal_value])
    flux_error_array_SKYMAPPER = np.array([g_mag_flux_SKYMAPPER.s])

else:
    lambda_array_SKYMAPPER = np.array([np.nan, np.nan])
    flux_array_SKYMAPPER = np.array([np.nan, np.nan])
    flux_error_array_SKYMAPPER = np.array([np.nan, np.nan])



# ------------------------------------------------------ Plot everything -----------------------------------------------
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax1 = fig1.add_subplot(111)
plt.grid(True)

cloud_GAIA = plt.scatter(lambda_array_GAIA, flux_array_GAIA, c='blue', marker='s', s=100, label='Gaia')
plt.errorbar(lambda_array_GAIA, flux_array_GAIA, yerr=flux_error_array_GAIA,
             linestyle=' ', c='k', capsize=10, elinewidth=1)

cloud_2MASS = plt.scatter(lambda_array_2MASS, flux_array_2MASS, c='green', marker='s', s=100, label='2MASS')
plt.errorbar(lambda_array_2MASS, flux_array_2MASS, yerr=flux_error_array_2MASS,
             linestyle=' ', c='k', capsize=10, elinewidth=1)

cloud_WISE = plt.scatter(lambda_array_WISE, flux_array_WISE, c='red', marker='s', s=100, label='WISE')
plt.errorbar(lambda_array_WISE, flux_array_WISE, yerr=flux_error_array_WISE,
             linestyle=' ', c='k', capsize=10, elinewidth=1)

cloud_SDSS = plt.scatter(lambda_array_SDSS, flux_array_SDSS, c='purple', marker='s', s=100, label='SDSS')
plt.errorbar(lambda_array_SDSS, flux_array_SDSS, yerr=flux_error_array_SDSS,
             linestyle=' ', c='k', capsize=10, elinewidth=1)

cloud_PANSTARRS = plt.scatter(lambda_array_PANSTARRS, flux_array_PANSTARRS, c='cyan', marker='s', s=100, label='Pan-STARRS')
plt.errorbar(lambda_array_PANSTARRS, flux_array_PANSTARRS, yerr=flux_error_array_PANSTARRS,
             linestyle=' ', c='k', capsize=10, elinewidth=1)

cloud_SKYMAPPER = plt.scatter(lambda_array_SKYMAPPER, flux_array_SKYMAPPER, c='magenta', marker='s', s=100, label='SkyMapper')
plt.errorbar(lambda_array_SKYMAPPER, flux_array_SKYMAPPER, yerr=flux_error_array_SKYMAPPER,
             linestyle=' ', c='k', capsize=10, elinewidth=1)

# add this if error bars on the wavelenght need to be added:
# xerr=[lambda_error_min_array_PANSTARRS, lambda_error_max_array_PANSTARRS],


ax.relim()
ax.autoscale_view()
ax.set_xbound(lower=0, upper=None)
ax.set_ybound(lower=0, upper=None)
ax.grid(b=True, which='major')
plt.tick_params(labelsize=16, which='both', length=14, pad=18)
plt.xscale('log', nonposx='clip')
plt.yscale('log', nonposy='clip')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.legend(loc='upper right', ncol=1, shadow=True, fancybox=True, fontsize=20)
plt.xlabel('\\textup{$\lambda$ ($\mu$m)} ', fontsize=20, labelpad=25)
plt.ylabel('\\textup{$F_\lambda$ (W${m^{-2}{\mu}m^{-1}}$)}', fontsize=20, labelpad=25)
plt.suptitle('Spectral Energy Distribution for Gaia Star %s' % source_id_Gaia, fontsize=30)
plt.show()

# ==================================================== H-R Diagram =====================================================

# ================================================== Visual of Result  =================================================

