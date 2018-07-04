# iREx

QUERY TESS TARGETS FROM RA & DEC 

This script makes a query in Gaia DR2 or Hipparcos (if not found in DR2)
and finds the corresponding object in 2MASS, WISE, PAN-STARRS, SDSS, and SkyMapper.
The input is RA and DEC with the size of the field of view of the telescope (optional).
The purpose is to identify the TESS source observed by combining information from all catalogs.
The code outputs Spectral Density Distribution plots from the brightest Gaia candidate. 
