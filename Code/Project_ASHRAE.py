#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> <span style="color:darkslategray">ASHRAE</span><h1><h2 span align="center"><span style="color:darkslategray">A Nagy energia előrejelző versenypályázat III</h2><br/><h3 span align="center"><span style="color:darkslategray">Mennyi energiát fogyaszt egy épület?</span></h3>

# ### <span style="color:indigo"> Bevezetés </span>

# Ebben a versenykiírásban a feladat olyan model fejlesztése, mely képes pontosan előrejelezni egy épület energia fogyasztását a követlező területeken: elektromosság, vízhűtés, gőz , illetve melegvíz. Az adatok több mint 1000 épület három évre visszamenő adatait tartalmazzák. 
# 
# A kiírás célja, hogy a energiamegtakarítás jobb becslése révén motiválja a nagybefektetőket és pénzügyi intézményeket a területbe történő befektetésre, ezzel előrelendítve a hatékonyság kiépítését. Általánosságban az épületek energiahatékonyságának javítása illeszkedik napjaink egyik legmeghatározóbb agendájába, a klímaváltozás negatív következményei elleni küzdelemhez. Eszerint a fogyasztás visszafogása csökkentheti a környezeti terhelést, azon belül is különösképp az üvegházhatást okozó gázok kibocsátásának a visszafogását. 
# 
# Az energiahatékonyság predikciójában rejlő legjelentősebb kihívás a kontrafaktuális állapot becslése, vagyis jelen esetben annak meghatározása, hogy mennyi energiát fogyasztott volna az adott épület abban az esetben, ha az energiafogyasztással kapcsolatos fejlesztések nem kerültek volna kivitelezésre. A gépi tanulás segítségével olyan model fejlesztése a cél, mely három évet felölelő energetikai mérési és időjárás adatok alapján képes pontos becslést adni az energiahatékonyság javulására vonatkozólag. 

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

# <b>weather_[train/test].csv</b><br/>
# <ul>
# <li><i>site_id</i> - Idegen kulcs az időjárás fájlhoz.</li><br/>
# <li><i>air_temperature</i> - Hőmérséklet Celsius fokban</li><br/>
# <li><i>cloud_coverage</i> - Az arány, amilyen mértékben felhők borítják az eget</li><br/>
# <li><i>dew_temperature</i> - Harmatpont Celsius fokban</li><br/>
# <li><i>precip_depth_1_hr</i> - Csapadékmennyiség millimeterben</li><br/>
# <li><i>sea_level_pressure</i> - Tengerszintre átszámított légnyomás millibárban</li><br/>
# <li><i>wind_direction</i> - Szélirány iránytű szerinti fokban</li><br/>
# <li><i>wind_speed</i> - Szélerősség m/s-ban</li><br/>
# </ul>

# <b>test.csv</b><br/>
# <ul>
# <li><i>row_id</i> - Idegen kulcs az időjárás fájlhoz</li><br/>
# <li><i>building_id</i> - Idegen kulcs a training fájlhoz</li><br/>
# <li><i>meter</i> - A mérő azonosító kódja (0: elektromosság (electricity), 1: vízhűtés (chilledwater), 2: gőz (steam), 3: melegvíz (hotwater). Nem minden épület rendelkezik minden mérő típussal.</li><br/>
# <li><i>timestamp</i> - az időpont, amikor a mérés megvalósult</li>
# </ul>
# 

# <b>sample_submission.csv</b>
# 
#     Az érvényes minta.

# ### <span style="color:dimgray"> Csomagok importálása</span>

# In[14]:


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
#import catboost as cbt
import lightgbm as lgbm

# Általános csomagok importálása
import pandas as pd
import numpy as nd
import os


# ### <span style="color:dimgray"> Adatok betöltése </span>

# In[34]:


# Az elérési út megadása
root = 'C:/Users/ZsoltNagy/Desktop/github_projects/ASHRAE/project_ASHRAE/Data'


# In[22]:


# Adatok betöltése
df_train = pd.read_csv(os.path.join(root, 'train.csv'))
df_test = pd.read_csv(os.path.join(root, 'test.csv'))
df_weather_train = pd.read_csv(os.path.join(root, 'weather_train.csv'))
df_weather_test = pd.read_csv(os.path.join(root, 'weather_test.csv'))
df_building = pd.read_csv(os.path.join(root, 'building_metadata.csv'))


# In[32]:


# Gyors pillantás az adatokra
display(df_train.iloc[:10,:])
display(df_test.iloc[:10,:])
display(df_weather_train.iloc[:10,:])
display(df_weather_test.iloc[:10,:])
display(df_building.iloc[:10,:])


# In[38]:


# Alapinformációk lekérése az adatszetekről
df_train.info()
print('')
df_test.info()
print('')
df_weather_train.info()
print('')
df_weather_test.info()
print('')
df_building.info()


# ### <span style="color:dimgray"> Memóriahasználat csökkentése </span>

# #### A train és test adatszetek memóriahasználata jelentős, mely hatékonyan csökkenthető a változók konvertálásával

# In[ ]:




