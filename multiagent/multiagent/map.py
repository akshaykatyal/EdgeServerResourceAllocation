#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:37:07 2019

@author: tyuan
"""

import folium
import pandas as pd
from folium import plugins
import json
import numpy as np

dic1 = '/Users/eva/Documents/GitHub/vehicle_data_bresil/'
dic2 = '/Users/tyuan/Documents/GitHub/DATA/2014-10/'
dic= '/Users/tyuan/Documents/GitHub/DATA/SDNdata/'
#
# San Francisco latitude and longitude values
latitude = -22.9427
longitude = -43.2200

with open(dic+'01_min_10km.json','r') as f:
    diction1 = json.load(fp=f)

# Create map and display it
san_map = folium.Map(location=[latitude, longitude], zoom_start=12)

# instantiate a mark cluster object for the incidents in the dataframe
incidents = plugins.MarkerCluster().add_to(san_map)

RSU_location = [-22.8892, -43.2249,
                -22.8661, -43.3057,
                -22.8619, -43.2327,
                -22.8721, -43.246,
                -22.8765, -43.2767,
                -22.9104, -43.2326,
                -22.915, -43.2645,
                -22.9258, -43.2866,
                -22.9059, -43.3052,
                -22.8936, -43.2619,
                -22.8901, -43.2491,
                -22.892, -43.2973,
                -22.9017, -43.2789,
                -22.8582, -43.2573,
                -22.9225, -43.2476,
                -22.9225, -43.2216,
                -22.92306, -43.30366,
                -22.8998, -43.2401,
                -22.9032, -43.249,
                -22.8694, -43.2678,
                -22.8858, -43.3109,
                -22.9308, -43.2422,
                -22.9027, -43.2601,
                -22.9104, -43.2223,
                -22.881, -43.2699,
                -22.8823, -43.2933,
                -22.8881, -43.2834,
                -22.8806, -43.2393,
                -22.8661, -43.2895,
                -22.8996, -43.2226,
                -22.905, -43.2888,
                -22.8935, -43.2736,
                -22.8722, -43.255,
                -22.9134, -43.2519,
                -22.874, -43.3148,
                -22.9243, -43.2587,
                -22.8824, -43.2555,
                -22.8732, -43.2321,
                -22.9098, -43.2827,
                -22.858, -43.2477,
                -22.8962, -43.3148,
                -22.9006, -43.2317,
                -22.8567, -43.2669,
                -22.855, -43.3125,
                -22.9167, -43.2736,
                -22.8731, -43.2971,
                -22.8859, -43.2635,
                -22.8747, -43.2862,
                -22.8959, -43.2913,
                -22.8637, -43.2389,
                -22.9103, -43.2931,
                -22.9169, -43.2403,
                -22.9198, -43.2346,
#                -22.9378,-43.2464,
                -22.9251, -43.3119
                ]
b =np.arange(int(len(RSU_location)/2))
a =  np.array(RSU_location).reshape((-1, 2))
RSU_location_new = np.c_[b,a]

np.savetxt(dic+'Rio_de_Janeiro_nodes.txt', RSU_location_new)

# set topology  
size = int(len(RSU_location)/2)
topo_matrix = np.zeros([size,size],dtype=int)

topo_matrix[0,(27,29,37,41)]=1
topo_matrix[1,(28,34,43,45)]=1
topo_matrix[2,(37,49)]=1
topo_matrix[3,(27,32,39,49)]=1
topo_matrix[4,(19,24,26,47)]=1
topo_matrix[5,(17,23,51,52)]=1
topo_matrix[6,(33,35,44)]=1
topo_matrix[7,(16,44)]=1
topo_matrix[8,(11,40,50)]=1
topo_matrix[9,(22,31,46)]=1
topo_matrix[10,(9,17,36)]=1

topo_matrix[11,(25,48)]=1
topo_matrix[12,(22,30,31,38,48)]=1
topo_matrix[13,(32,39,42)]=1
topo_matrix[14,(21,35,51)]=1
topo_matrix[15,(21,52)]=1
topo_matrix[16,(53)]=1
topo_matrix[17,(18,41)]=1
topo_matrix[18,(22)]=1
topo_matrix[19,(28,32,42)]=1

topo_matrix[20,(25,34,40)]=1
topo_matrix[21,(52)]=1
topo_matrix[22,(33)]=1
topo_matrix[23,(29)]=1
topo_matrix[24,(36,46)]=1
topo_matrix[25,(26,45)]=1
topo_matrix[26,(31)]=1
topo_matrix[27,(37)]=1
topo_matrix[28,(45,47)]=1
topo_matrix[29,(41)]=1

topo_matrix[30,(48,50)]=1
topo_matrix[31,(46)]=1
topo_matrix[32,(36)]=1
topo_matrix[33,(51)]=1
topo_matrix[34,(45)]=1
topo_matrix[37,(49)]=1
topo_matrix[38,(30,44,50)]=1

topo_matrix[42,()]=1
topo_matrix[43,()]=1
topo_matrix[44,()]=1
topo_matrix[49,()]=1
topo_matrix[51,(52)]=1

 
np.savetxt(dic+'Rio_de_Janeiro.matrix', topo_matrix)

k=0
for i in RSU_location_new:
    lat = i[0]
    lng = i[1]
    # drwa location of RSUs
    folium.Marker(
#            radius=20,
            location = i,
#            icon=None,
#            popup=label,
            popup=(k, lat, lng),
#            fill_color='#769d96',
#            number_of_sides=10,
#            radius=10  
        ).add_to(san_map)
    # links
    link_index = np.nonzero(topo_matrix[k])

    for j in link_index[0]:
       item = np.append(i, RSU_location_new[j]).reshape((-1,2)).tolist()
       folium.PolyLine(
            locations = item,
            color = 'black').add_to(san_map)
    k+=1




# loop through the dataframe and add each data point to the mark cluster
#for i in range(len(diction1)):
i=25
for j in range(len(diction1[str(i)])):
    lat = diction1[str(i)][j][0]/(10.256*100)-22.9427
    lng = diction1[str(i)][j][1]/(10.256*100)-43.3175
    label =diction1[str(i)][j][2]
#for lat, lng, label, in  zip(data.Y, data.X, cdata.Category):
#    folium.Marker(
    folium.Circle(
        radius=1,
        location=[lat, lng],
        icon=None,
        popup=label,
    ).add_to(incidents)

# add incidents to map
san_map.add_child(incidents)

## Create Choropleth map
#folium.Choropleth(
#    geo_data=san_geo,
#    data=diction1,
#    columns=['Neighborhood','Count'],
#    key_on='feature.properties.DISTRICT',
#    #fill_color='red',
#    fill_color='YlOrRd',
#    fill_opacity=0.7,
#    line_opacity=0.2,
#    highlight=True,
#    legend_name='Crime Counts in San Francisco'
#).add_to(san_map)



san_map.save('index.html')



#
#
## instantiate a mark cluster object for the incidents in the dataframe
#incidents = plugins.MarkerCluster().add_to(san_map)
# Read Dataset 
#cdata = pd.read_csv('https://cocl.us/sanfran_crime_dataset')
#cdata.head()
# limit = 200
# data = cdata.iloc[0:limit, :]
## loop through the dataframe and add each data point to the mark cluster
#for lat, lng, label, in zip(data.Y, data.X, cdata.Category):
#    folium.Marker(
#        location=[lat, lng],
#        icon=None,
#        popup=label,
#    ).add_to(incidents)
#
## add incidents to map
#san_map.add_child(incidents)
#
#
#san_map.save('index.html')

#san_map = folium.Map(location=[37.77, -122.4], zoom_start=12)
#url = 'https://cocl.us/sanfran_geojson'
#san_geo = f'{url}'


## Count crime numbers in each neighborhood
#disdata = pd.DataFrame(cdata['PdDistrict'].value_counts())
#disdata.reset_index(inplace=True)
#disdata.rename(columns={'index':'Neighborhood','PdDistrict':'Count'},inplace=True)

## Create Choropleth map
#folium.Choropleth(
#    geo_data=san_geo,
#    data=disdata,
#    columns=['Neighborhood','Count'],
#    key_on='feature.properties.DISTRICT',
#    #fill_color='red',
#    fill_color='YlGn',
#    fill_opacity=0.7,
#    line_opacity=0.2,
#    highlight=True,
#    legend_name='Crime Counts in San Francisco'
#).add_to(san_map)

#san_map.save('index.html')