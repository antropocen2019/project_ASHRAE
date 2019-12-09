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
# <li><i>meter</i> - A mérő azonosító kódja (0: elektromosság (electricity), 1: hűtött víz (chilledwater), 2: gőz (steam), 3: melegvíz (hotwater). Nem minden épület rendelkezik minden mérő típussal.</li><br/>
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

# In[1]:


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
import numpy as np
import os
from IPython.display import display_html 


# ### <span style="color:dimgray"> Adatok betöltése </span>

# In[ ]:


# Az elérési út megadása
root = 'C:/Users/ZsoltNagy/Desktop/github_projects/ASHRAE/project_ASHRAE/Data'


# In[3]:


# Adatok betöltése
df_train = pd.read_csv(os.path.join(root, 'train.csv'))
df_test = pd.read_csv(os.path.join(root, 'test.csv'))
df_weather_train = pd.read_csv(os.path.join(root, 'weather_train.csv'))
df_weather_test = pd.read_csv(os.path.join(root, 'weather_test.csv'))
df_building = pd.read_csv(os.path.join(root, 'building_metadata.csv'))


# In[4]:


# Gyors pillantás az adatokra
display(df_train.iloc[:10,:])
display(df_test.iloc[:10,:])
display(df_weather_train.iloc[:10,:])
display(df_weather_test.iloc[:10,:])
display(df_building.iloc[:10,:])


# In[5]:


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

# In[6]:


## Függvény a memóriahasználat csökkentéséhez. Forrás: https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[7]:


# A memóriahasználatot csökkentő függvény alkalmazása

df_train_red = reduce_mem_usage(df_train)
df_test_red = reduce_mem_usage(df_test)
df_weather_train_red = reduce_mem_usage(df_weather_train)
df_weather_test_red = reduce_mem_usage(df_weather_test)
df_building_red = reduce_mem_usage(df_building)


# In[9]:


# Alapstatisztikák lekérése az adatok transzformáció előtti és utáni összehasonlításához:

# df_train
desc_dftrain = df_train.describe()
desc_dftrain_red = df_train_red.describe()

desc_dftrain_styler = desc_dftrain.style.set_table_attributes("style='display:inline'").set_caption('Eredeti df_train')
desc_dftrain_red_styler = desc_dftrain_red.style.set_table_attributes("style='display:inline'").set_caption('Transzformált df_train')

display_html(desc_dftrain_styler._repr_html_()+desc_dftrain_red_styler._repr_html_(), raw=True)

# df_test
desc_dftest = df_test.describe()
desc_dftest_red = df_test_red.describe()

desc_dftest_styler = desc_dftest.style.set_table_attributes("style='display:inline'").set_caption('Eredeti df_test')
desc_dftest_red_styler = desc_dftest_red.style.set_table_attributes("style='display:inline'").set_caption('Transzformált df_test')

display_html(desc_dftest_styler._repr_html_()+desc_dftest_red_styler._repr_html_(), raw=True)

# df_weather_train
desc_dfweather_train = df_weather_train.describe()
desc_dfweather_train_red = df_weather_train_red.describe()

desc_dfweather_train_styler = desc_dfweather_train.style.set_table_attributes("style='display:inline'").set_caption('Eredeti df_weather_train')
desc_dfweather_train_red_styler = desc_dfweather_train_red.style.set_table_attributes("style='display:inline'").set_caption('Transzformált df_weather_train')

display_html(desc_dfweather_train_styler._repr_html_()+desc_dfweather_train_red_styler._repr_html_(), raw=True)

# df_weather_test
desc_dfweather_test = df_weather_test.describe()
desc_dfweather_test_red = df_weather_test_red.describe()

desc_dfweather_test_styler = desc_dfweather_test.style.set_table_attributes("style='display:inline'").set_caption('Eredeti df_weather_test')
desc_dfweather_test_red_styler = desc_dfweather_test_red.style.set_table_attributes("style='display:inline'").set_caption('Transzformált df_weather_test')

display_html(desc_dfweather_test_styler._repr_html_()+desc_dfweather_test_red_styler._repr_html_(), raw=True)

# building
desc_dfbuilding = df_building.describe()
desc_dfbuilding_red = df_building_red.describe()

desc_dfbuilding_styler = desc_dfbuilding.style.set_table_attributes("style='display:inline'").set_caption('Eredeti df_building')
desc_dfbuilding_red_styler = desc_dfbuilding_red.style.set_table_attributes("style='display:inline'").set_caption('Transzformált df_building')

display_html(desc_dfbuilding_styler._repr_html_()+desc_dfbuilding_red_styler._repr_html_(), raw=True)


# ### <span style="color:dimgray"> Adatszetek egyesítése </span>

# In[14]:


train = df_train_red.merge(df_building_red, on='building_id', how='left')
test = df_test_red.merge(df_building_red, on='building_id', how='left')

train = train.merge(df_weather_train_red, on=['site_id', 'timestamp'], how='left')
test = test.merge(df_weather_test_red, on=['site_id', 'timestamp'], how='left')

display(train.iloc[:10,:])
display(test.iloc[:10,:])


# ### <span style="color:dimgray"> Adatok feltérképezése </span>

# #### Az adatok feltérképezése egy iteratív, végnélküli folyamat, mely során igyekszünk megérteni az adatban rejlő mintázatokat, összefüggéseket, trendeket, valamint anomáliákat alapvető statistikai eljárások használata révén. 

# #### <span style="color:darkmagenta">Hiányzó adatok</span>

# In[47]:


train_missing = train.drop('meter_reading', axis=1).count().divide(len(train)).round(4).sort_values()*100
test_missing = test.drop('row_id', axis=1).count().divide(len(test)).round(4).sort_values()*100


fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(train.drop('meter_reading', axis=1).columns))
bar_width = 0.4
b1 = ax.bar(x, train_missing, width=bar_width, color='indigo')
b2 = ax.bar(x + bar_width, test_missing, width=bar_width, color='orange')

# Fix the x-axes.
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(train_missing.index, rotation=40)
ax.legend([b1, b2], ['Train', 'Test'])
ax.set_ylabel('%')
ax.set_title('A rendelkezésre álló adat százalékban kifejezve', fontsize=16);


# #### <span style="color:darkmagenta">A célváltozó vizsgálata</span>

# In[99]:


target_count = train['meter'].value_counts()
target_count.index = ['Elektromosság', 'Hűtött víz', 'Gőz', 'Melegvíz']
target_count = target_count.sort_values().to_frame().reset_index()
target_count

ax = sns.barplot(y= "meter", x = "index", data = target_count, palette=("BuPu"))
sns.set(rc={'figure.figsize':(10,6)})
ax.set(xlabel='Mérőtípus', ylabel='Mérések száma (egység = 10 millió)')
ax.set_title('Adott mérőtípushoz tartozó mérések száma', fontsize=16)

