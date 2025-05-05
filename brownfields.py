# brownfields.py

# Importing extensions
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Defining study's CRS, Michigan State Plane South
crs = 2253

# Reading shapefiles and data tables
brownfields = gpd.read_file('Brownfield_Sites/brownfield_sites.shp').to_crs(crs)
tracts = gpd.read_file('tl_2019_26_tract/tl_2019_26_tract.shp').to_crs(crs)
city_boundary = gpd.read_file('City_of_Detroit_Boundary/City_of_Detroit_Boundary.shp').to_crs(crs)
zoning = gpd.read_file('Zoning_Official/zoning_official.shp').to_crs(crs)
asthma = gpd.read_file('Asthma.csv')
cancer = gpd.read_file('Cancer.csv')
copd = gpd.read_file('COPD.csv')
chd = gpd.read_file('Coronary_Heart_Disease.csv')
life_expect = gpd.read_file('Life_Expectancy.csv')

# Filtering census tract shapefile to only include tracts within Wayne County
wayne_tracts = tracts[tracts['COUNTYFP'] == '163']

# Filtering again, this time to only include tracts within Detroit city limits
detroit_tracts = gpd.clip(wayne_tracts,city_boundary)

# Filtering points shapefile by filtering zoning shapefile temporarily 
zoning_temp = zoning[zoning['ZONING_REV'].str.startswith('P') | zoning['ZONING_REV'].str.startswith('R') | zoning['ZONING_REV'].str.startswith('S')]

# Filter points shapefile with temporary zoning shapefile
current_brownfields = gpd.overlay(brownfields,zoning_temp,how='difference')

# Creating a 1500 ft buffer around existing brownfields
buffer = current_brownfields.geometry.buffer(1500)

# Merging all buffers into a multipolygon
buffer = buffer.union_all()

# Saving buffer as a geodataframe because buffer function converted it into a GeoSeries
buffer_gdf = gpd.GeoDataFrame(geometry=[buffer], crs=crs)

# Allocating health data to columns within the detroit_tracts shapefile
detroit_tracts = detroit_tracts.merge(life_expect[['Geo ID','Population','Life Expectancy']], left_on='GEOID', right_on='Geo ID',how='left')
detroit_tracts['life_expcy'] = detroit_tracts['Life Expectancy']
detroit_tracts['total_pop'] = detroit_tracts['Population']
detroit_tracts = detroit_tracts.drop(columns=['Geo ID','Population','Life Expectancy'])

detroit_tracts = detroit_tracts.merge(asthma[['Geo ID','Asthma']], left_on='GEOID', right_on='Geo ID',how='left')
detroit_tracts['asthma'] = detroit_tracts['Asthma']
detroit_tracts = detroit_tracts.drop(columns=['Geo ID','Asthma'])

detroit_tracts = detroit_tracts.merge(copd[['Geo ID','Chronic Obstructive Pulmonary Disease']], left_on='GEOID', right_on='Geo ID',how='left')
detroit_tracts['copd'] = detroit_tracts['Chronic Obstructive Pulmonary Disease']
detroit_tracts = detroit_tracts.drop(columns=['Geo ID','Chronic Obstructive Pulmonary Disease'])

detroit_tracts = detroit_tracts.merge(chd[['Geo ID','Coronary Heart Disease']], left_on='GEOID', right_on='Geo ID',how='left')
detroit_tracts['chd'] = detroit_tracts['Coronary Heart Disease']
detroit_tracts = detroit_tracts.drop(columns=['Geo ID','Coronary Heart Disease'])

detroit_tracts = detroit_tracts.merge(cancer[['Geo ID','Cancer']], left_on='GEOID', right_on='Geo ID',how='left')
detroit_tracts['cancer'] = detroit_tracts['Cancer']
detroit_tracts = detroit_tracts.drop(columns=['Geo ID','Cancer'])

# Defining function 'conversion' to convert health data fields from strings into integers
def conversion(field):
    if pd.isna(field) or field == '':
        return np.nan
    if isinstance(field,str):
        return int(float(field.replace('%','')))
    return int(field)

# Applying conversion function to health data
# Health risk data is given a numerical value by multplying the percentage by the population
detroit_tracts['total_pop'] = detroit_tracts['total_pop'].apply(conversion)
detroit_tracts['life_expcy'] = detroit_tracts['life_expcy'].apply(conversion)

detroit_tracts['asthma_pct'] = detroit_tracts['asthma'].apply(conversion) / 100
detroit_tracts['asthma'] = detroit_tracts['total_pop'] * detroit_tracts['asthma_pct']

detroit_tracts['copd_pct'] = detroit_tracts['copd'].apply(conversion) / 100
detroit_tracts['copd'] = detroit_tracts['total_pop'] * detroit_tracts['copd_pct']

detroit_tracts['chd_pct'] = detroit_tracts['chd'].apply(conversion) / 100
detroit_tracts['chd'] = detroit_tracts['total_pop'] * detroit_tracts['chd_pct']

detroit_tracts['cancer_pct'] = detroit_tracts['cancer'].apply(conversion) / 100
detroit_tracts['cancer'] = detroit_tracts['total_pop'] * detroit_tracts['cancer_pct']

# Filtering zoning shapefile to only include residential lots
residential = zoning[zoning['ZONING_REV'].str.startswith('R')]
residential = residential.dissolve()

# Overlaying the residential and census tract data with gpd.overlay
residential_census = gpd.overlay(residential,detroit_tracts,how='intersection')

# Calculating total area of residential areas within each census tract
residential_census['total_area'] = residential_census.geometry.area

# Creating two overlays based on if residential_area 
residential_inside = gpd.overlay(residential_census,buffer_gdf,how='intersection')
residential_outside = gpd.overlay(residential_census,buffer_gdf,how='difference')

# Setting up 'buffer' column to determine if polygon is within brownfield buffer
residential_inside['buffer'] = 'I'
residential_outside['buffer'] = 'O'

# Using pd.concat to join the two dataframes and their columns together into one dataframe
residential_final = gpd.GeoDataFrame(pd.concat([residential_inside,residential_outside],ignore_index=True),crs=residential_inside.crs)

# Calculating area of the polygons
residential_final['polygon_area'] = residential_final.geometry.area

# Dissolving each shapefile into 1 multipolygon
residential_total = residential_final.dissolve()
buffer_total = residential_inside.dissolve()

# Calculating the total area of all residential lots and residential lots inside buffer
residential_total['total_area'] = residential_total.geometry.area
buffer_total['total_area'] = buffer_total.geometry.area

# Dividing the inside residential area by total residential area
total_pct = buffer_total['total_area'] / residential_total['total_area']
print(total_pct)

# Calculating percentage each residential polygon takes up within the total residential area
residential_final['area_pct'] = residential_final['polygon_area'] / residential_final['total_area']

# Allocating population per polygon based on its area percentage
residential_final['total_pop'] = residential_final['total_pop'] * residential_final['area_pct']

# Adding a higher weighting to polygons within close proximity of a brownfield site
residential_final['risk'] = residential_final.apply(lambda x: 0.60 if x['buffer'] == 'I' else 0.40, axis=1)

# Using weighting to allocate higher percentage to polygons near brownfield sites
residential_final['asthma'] = residential_final['asthma'] * residential_final['risk']
residential_final['copd'] = residential_final['copd'] * residential_final['risk']
residential_final['chd'] = residential_final['chd'] * residential_final['risk']
residential_final['cancer'] = residential_final['cancer'] * residential_final['risk']

# Calculating the total number based on area percentage
residential_final['asthma'] = residential_final['asthma'] * residential_final['area_pct']
residential_final['copd'] = residential_final['copd'] * residential_final['area_pct']
residential_final['chd'] = residential_final['chd'] * residential_final['area_pct']
residential_final['cancer'] = residential_final['cancer'] * residential_final['area_pct']

# Saving files
detroit_tracts.to_file('detroit_tracts.shp')
current_brownfields.to_file('current_brownfields.shp')
buffer_gdf.to_file('buffer.shp')
residential_census.to_file('residential_census.shp')
residential_inside.to_file('residential_inside.shp')
residential_outside.to_file('residential_outside.shp')
residential_final.to_file('residential_final.shp')


# Plotting distribution of brownfield sites
fig, ax = plt.subplots(figsize=(10,6))

detroit_tracts.plot(facecolor='grey', ax=ax)
current_brownfields.plot(ax=ax,edgecolor='black',markersize=12)

ax.set_title('Distribution of Brownfield Sites in Detroit, Michigan', fontsize=12, fontweight='bold')
ax.set_axis_off()

plt.show()

# Plotting results for total population
fig, ax = plt.subplots(figsize=(10,6))

detroit_tracts.plot("total_pop", scheme='naturalbreaks', cmap='Reds', ax=ax)
current_brownfields.plot(ax=ax,edgecolor='black',markersize=12)

ax.set_title('Population Distribution in Detroit, Michigan', fontsize=12, fontweight='bold')
plt.colorbar(ax.collections[0], ax=ax, label='Population Distribution', orientation='vertical', ticks=[])
ax.set_axis_off()

plt.show()

# Plotting results for asthma diagnoses
fig, ax = plt.subplots(figsize=(10,6))

residential_final.plot("asthma", scheme='naturalbreaks', cmap='Reds', ax=ax)
current_brownfields.plot(ax=ax,edgecolor='black',markersize=12)

ax.set_title('Distribution of Asthma Diagnoses in Detroit, Michigan', fontsize=12, fontweight='bold')
plt.colorbar(ax.collections[0], ax=ax, label='Asthma Diagnoses', orientation='vertical', ticks=[])
ax.set_axis_off()

plt.show()

# Plotting results for COPD diagnoses
fig, ax = plt.subplots(figsize=(10,6))

residential_final.plot("copd", scheme='naturalbreaks', cmap='Reds', ax=ax)
current_brownfields.plot(ax=ax,edgecolor='black',markersize=12)

ax.set_title('Distribution of COPD Diagnoses in Detroit, Michigan', fontsize=12, fontweight='bold')
plt.colorbar(ax.collections[0], ax=ax, label='COPD Diagnoses', orientation='vertical', ticks=[])
ax.set_axis_off()

plt.show()

# Plotting results for CHD diagnoses
fig, ax = plt.subplots(figsize=(10,6))

residential_final.plot("chd", scheme='naturalbreaks', cmap='Reds', ax=ax)
current_brownfields.plot(ax=ax,edgecolor='black',markersize=12)

ax.set_title('Distribution of CHD Diagnoses in Detroit, Michigan', fontsize=12, fontweight='bold')
plt.colorbar(ax.collections[0], ax=ax, label='CHD Diagnoses', orientation='vertical', ticks=[])
ax.set_axis_off()

plt.show()

# Plotting results for Cancer diagnoses
fig, ax = plt.subplots(figsize=(10,6))

residential_final.plot("cancer", scheme='naturalbreaks', cmap='Reds', ax=ax)
current_brownfields.plot(ax=ax,edgecolor='black',markersize=12)

ax.set_title('Distribution of Cancer Diagnoses in Detroit, Michigan', fontsize=12, fontweight='bold')
plt.colorbar(ax.collections[0], ax=ax, label='Cancer Diagnoses', orientation='vertical', ticks=[])
ax.set_axis_off()

plt.show()

