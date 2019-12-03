#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> <span style="color:darkslategray">ASHRAE</span><h1><h2 span align="center"><span style="color:darkslategray">A Nagy energia előrejelző versenypályázat III</h2><br/><h3 span align="center"><span style="color:darkslategray">Mennyi energiát fogyaszt egy épület?</span></h3>

# ### <span style="color:indigo"> Fájlok:</span>

# <b>train.csv</b><br/>
# <ul>
# <li><i>building_id</i> - Idegen kulcs az épület metadata fájlhoz.</li><br/>
# <li><i>meter</i> - A mérő azonosító kódja (0: elektromosság (electricity), 1: vízhűtés (chilledwater), 2: gőz (steam), 3: melegvíz (hotwater). Nem minden épület rendelkezik minden mérő típussal.</li><br/>
# <li><i>timestamp</i> - az időpont, amikor a mérés megvalósult</li><br/>
# <li><i>meter_reading</i> - A célváltozó. Az energiafogyasztás kWh-ban kifejezve (vagy azzal ekvivalens). Valós adatok révén figyelembe kell venni a mérési hibát, mely a modellezési hibának az baseline szintjeként értelmezhető.</li>
# </ul>
# 

# <b>building_meta.csv</b><br/>
# <ul>
# <li><i>site_id</i> - Idegen kulcs az időjárás fájlhoz.</li><br/>
# <li><i>building_id</i> - Idegen kulcs a training fájlhoz</li><br/>
# <li><i>primary_use</i> - Az épület elsődleges tevékenységének kategóriája  az EnergieStar ingatlan típus besorolása alapján</li><br/>
# <li><i>square_feet</i> - Az épület bruttó területe</li><br/>
# <li><i>year_built</i> - Az épület megynitásának időpontja</li><br/>
# <li><i>floor_count</i> - Az épület emeleteinek a száma</li><br/>
# </ul>

# In[ ]:




<b>building_meta.csv</b>

    site_id - Foreign key for the weather files.
    building_id - Foreign key for training.csv
    primary_use - Indicator of the primary category of activities for the building based on EnergyStar property type definitions
    square_feet - Gross floor area of the building
    year_built - Year building was opened
    floor_count - Number of floors of the building

<b>weather_[train/test].csv</b>

Weather data from a meteorological station as close as possible to the site.

    site_id
    air_temperature - Degrees Celsius
    cloud_coverage - Portion of the sky covered in clouds, in oktas
    dew_temperature - Degrees Celsius
    precip_depth_1_hr - Millimeters
    sea_level_pressure - Millibar/hectopascals
    wind_direction - Compass direction (0-360)
    wind_speed - Meters per second

<b>test.csv</b>

The submission files use row numbers for ID codes in order to save space on the file uploads. test.csv has no feature data; it exists so you can get your predictions into the correct order.

    row_id - Row id for your submission file
    building_id - Building id code
    meter - The meter id code
    timestamp - Timestamps for the test data period

<b>sample_submission.csv</b>

A valid sample submission.

    All floats in the solution file were truncated to four decimal places; we recommend you do the same to save space on your file upload.
    There are gaps in some of the meter readings for both the train and test sets. Gaps in the test set are not revealed or scored.


# ### <span style="color:dimgray"> Csomagok importálása</span>

# In[ ]:


# Csomagok importálása validációhoz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics, model_selection

# Csomagok importálása vizualizációhoz
import matplotlib.pyplot as plt
import seaborn as sns

# Csomagok importálása modellezéshez
import xgboost as xgb
import catboost as cbt
import lightgbm as lgbm

# Általános csomagok importálása
import pandas as pd
import numpy as nd

