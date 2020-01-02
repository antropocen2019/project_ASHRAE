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
# <li><i>meter</i> - A mérő azonosító kódja (0: elektromos áram (electricity), 1: hidegvíz (chilledwater), 2: gőz (steam), 3: melegvíz (hotwater). Nem minden épület rendelkezik minden mérő típussal.</li><br/>
# <li><i>timestamp</i> - az időpont, amikor a mérés megvalósult</li><br/>
# <li><i>meter_reading</i> - A célváltozó. Az energiafogyasztás kWh-ban kifejezve (vagy azzal ekvivalens). Valós adatok révén figyelembe kell venni a mérési hibát, mely a modellezési hibának az baseline szintjeként értelmezhető.</li>
# </ul>
# 

# <b>building_meta.csv</b><br/>
# <ul>
# <li><i>site_id</i> - Helyszín, idegen kulcs az időjárás fájlhoz.</li><br/>
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
from datetime import datetime


# ### <span style="color:dimgray"> Adatok betöltése </span>

# In[2]:


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


# In[8]:


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

# In[9]:


train = df_train_red.merge(df_building_red, on='building_id', how='left')
test = df_test_red.merge(df_building_red, on='building_id', how='left')

train = train.merge(df_weather_train_red, on=['site_id', 'timestamp'], how='left')
test = test.merge(df_weather_test_red, on=['site_id', 'timestamp'], how='left')

display(train.iloc[:10,:])
display(test.iloc[:10,:])


# ### <span style="color:dimgray"> Adatok feltérképezése </span>

# #### Az adatok feltérképezése egy iteratív, végnélküli folyamat, mely során igyekszünk megérteni az adatban rejlő mintázatokat, összefüggéseket, trendeket, valamint anomáliákat alapvető statisztikai eljárások használata révén. A lentebbi elemzés betekintést nyújt a folyamatba, azonban korántsem tekinthető minden részletre kiterjedő vizsgálatnak, helyet hagyva az olvasó számára a további elemzésekre. 

# #### <span style="color:darkmagenta">Hiányzó adatok vizsgálata</span>

# In[10]:


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


# #### A fenti ábra alapján látható, hogy három változó, a szintek száma, az építése éve, illetve a felhőtakaró nagysága esetében a rendelkezésre álló adatok mennyisége alacsony. Érdemes tehát utána nézni, hogy esetleg van-e bármilyen mintázat, ami az adathiányt magyarázhatná.

# In[42]:


temp_df = train[train['floor_count'].isnull()]
print('Leggyakoribb kategóriák/értékek változónként azon adatpontok esetében, ahol hiányzik az épület szintje változó értéke:')
for columns in temp_df.columns:
    print('')
    print(str(columns) + str(':'))
    print(temp_df[columns].value_counts().head(5))


# In[41]:


temp_df = train[train['year_built'].isnull()]
print('Leggyakoribb kategóriák/értékek változónként azon adatpontok esetében, ahol hiányzik az épület építésének éve változó értéke:')
for columns in temp_df.columns:
    print('')
    print(str(columns) + str(':'))
    print(temp_df[columns].value_counts().head(5))


# In[40]:


temp_df = train[train['cloud_coverage'].isnull()]
print('Leggyakoribb kategóriák/értékek változónként azon adatpontok esetében, ahol hiányzik a felhőtakaró mértéke változó értéke:')
for columns in temp_df.columns:
    print('')
    print(str(columns) + str(':'))
    print(temp_df[columns].value_counts().head(5))


# #### A kapott eredmények alapján érdemes lesz külön figyelmet fordítani a 13-as telekre, valamint az 1249-es és 1298-as azaonosítóval rendelkező épületekre, mivel esetükben adathiány több változó esetében is fennáll.

# #### <span style="color:darkmagenta">Egyváltozós elemzés</span>

# #### A célváltozó eloszlása erősen ferde a kiugró értékek miatt.

# In[37]:


ax = sns.boxplot(y='meter_reading',
                 data=train, 
                 width=0.5,
                 palette="BuPu")
ax.set(ylabel = 'Célváltozó eloszlása')
ax.set_title('Kiugró értékek a célváltozó (Energiafogyasztás KWh-ban) esetében', fontsize=12);


# In[15]:


# Kiugró értékek vizsgálata az Interkvartilis terjedelem (IQR) érték alapján

def outliers_iqr(x, mplyr):
    quartile_1, quartile_3 = np.percentile(x, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * mplyr)
    upper_bound = quartile_3 + (iqr * mplyr)
    return list(np.where((x > upper_bound) | (x < lower_bound)))

indexes_outlier = outliers_iqr(train['meter_reading'], 1.5)[0]
train_outliers = train[train.index.isin(indexes_outlier)]
train_no_outliers = train.drop(train.index[indexes_outlier])
print('Kiugró értékek száma a célváltozóban: ' + str(len(train_outliers)))


# In[16]:


bins1 = np.linspace(min(train_no_outliers['meter_reading']), max(train_no_outliers['meter_reading']), 25)
sns.distplot(train_no_outliers['meter_reading'], bins=bins1)
plt.figsize = [6,6]

plt.title('A célváltozó (Energiafogyasztás KWh-ban) eloszlása a kiugró értékek eltávolítása után')
plt.ylabel('Sűrűség')
plt.xlabel('Energiafogyasztás KWh-ban')
plt.show()

bins2 = np.linspace(min(train_outliers['meter_reading']), max(train_outliers['meter_reading']), 25)
sns.distplot(train_outliers['meter_reading'], bins=bins2)
plt.figsize = [6,6]

plt.title('A célváltozó (Energiafogyasztás KWh-ban) eloszlása a kiugró értékek esetében')
plt.ylabel('Sűrűség')
plt.xlabel('Energiafogyasztás KWh-ban')
plt.show()


# In[83]:


# A célváltozó eloszlásának lekérése logaritmikus transzformációt követően
bins = np.linspace(min(np.log1p(train["meter_reading"].values)), max(np.log1p(train["meter_reading"].values)), 25)
sns.distplot(np.log1p(train['meter_reading']), bins=bins)
plt.figsize = [6,6]
plt.title('A célváltozó (Energiafogyasztás KWh-ban) eloszlása logtranszformáció után kiugró értékekkel')
plt.ylabel('Sűrűség')
plt.xlabel('Energiafogyasztás KWh-ban')
plt.show()


# In[84]:


# A célváltozó eloszlásának lekérése logaritmikus transzformációt követően
bins = np.linspace(min(np.log1p(train_no_outliers["meter_reading"].values)), max(np.log1p(train_no_outliers["meter_reading"].values)), 25)
sns.distplot(np.log1p(train_no_outliers['meter_reading']), bins=bins)
plt.figsize = [6,6]
plt.title('A célváltozó (Energiafogyasztás KWh-ban) eloszlása logtranszformáció után')
plt.ylabel('Sűrűség')
plt.xlabel('Energiafogyasztás KWh-ban')
plt.show()


# #### A változó eloszlása a logtranszformáció és a kiugró értékek kizárása után is erősen asszimmetrikus 

# #### A független változók elemzése

# In[85]:


cat_features_df = train[['meter', 'site_id', 'primary_use', 'floor_count']]
num_features_df = train[['square_feet', 'year_built', 'air_temperature', 'cloud_coverage', 'dew_temperature',
                                    'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']]


# In[86]:


for features in cat_features_df:
    ax = sns.countplot(x=features, data=cat_features_df, palette=("PRGn"))
    ax.set_title(str(str('Egyes kategóriákhoz tartozó elemek száma') + str(': ') + str(features)), fontsize=14)
    ax.set_ylabel('Elemek száma')
    ax.set_xlabel('Kategóriák')
    if features == 'primary_use':
        ax.tick_params(axis='x', rotation=60)
    plt.show()


# #### Főbb megállapítások:
# <ul>
# <li><i>Mérő típus</i> - Legtöbb adat elekromos áram fogyasztással kapcsolatosan, míg legkevesebb a melegvíz mérőhöz áll rendelkezésre </li><br/>
# </ul>
# <ul>
# <li><i>Helyszín </i> - Legtöbb adat a 13-as, a 9-es és a 2-es telkeken lévő épületekkel kapcsolatosan található az adatszetben, míg a legkevesebb a 11 és 12-es telekhez tartozó épületekről áll rendelkezésre </li><br/>
# </ul>
# <ul>
# <li><i>Épület típus </i> - A legtöbb esetben az épület besorolása oktatási kategóriába esett, míg a második leggyakoribb az iroda besorolosú volt </li><br/>
# </ul>
# <ul>
# <li><i>Szintek száma </i> - Az épületek többsége egy vagy kétszintes volt  </li><br/>
# </ul>

# In[88]:


for features in num_features_df:
    num_features_df.hist(column=features, grid=False, figsize=(8,10), layout=(3,1), sharex=True, color='#411B46', rwidth=0.8)
    


# #### Főbb megállapítások:
# <ul>
# <li><i>Alapterület</i> - Számos épület alapterülete meghaladja 200000 négyzetlábat </li><br/>
# </ul>
# <ul>
# <li><i>Építési idő </i> - Az épületek jelentős része az 1960-80-as évek közötti időszakban épült, valószínűsíthatően alacsonyabb energetikai hatékonyság mellett </li><br/>
# </ul>
# <ul>
# <li><i>léghőmérséklet </i> - A levegőhőmérséklet eloszlása közel normális, ami egyrészt köszönhető az egy évet felölelő mérési időszaknak, illetve a helyszínek geográfiai szétszórtságának </li><br/>
# </ul>

# #### <span style="color:darkmagenta">A változók többváltozós elemzése</span>

# #### Adott mérőtípushoz tartozó mérések száma

# In[89]:


target_count = train['meter'].value_counts()
target_count.index = ['Elektromos áram', 'Hidegvíz', 'Gőz', 'Melegvíz']
target_count = target_count.sort_values().to_frame().reset_index()

ax = sns.barplot(y= "meter", x = "index", data = target_count, palette=("BuPu"))
sns.set(rc={'figure.figsize':(10,6)})
ax.set(xlabel='Mérőtípus', ylabel='Mérések száma (egység = 10 millió)')
ax.set_title('Adott mérőtípushoz tartozó mérések száma', fontsize=14)


# #### Összes energiafogyasztás mérőtípusonként

# In[92]:


target_sum = train.groupby(['meter'])['meter_reading'].sum()
target_sum.index = ['Elektromos áram', 'Hidegvíz', 'Gőz', 'Melegvíz']
target_sum = target_sum.sort_values().to_frame().reset_index()

ax = sns.barplot(x="index", y = "meter_reading", data = target_sum, palette=("BuPu"))
sns.set(rc={'figure.figsize':(10,6)})
ax.set(xlabel='Mérőtípus', ylabel='Energiafogyasztás')
ax.set_title('Adott mérőtípushoz tartozó összfogyasztás', fontsize=14)


# Míg a gőz esetében a mérések száma nagyjából hatoda a elektromos áram fogyasztással kapcsolatban rendelkezésre álló mérések számának, az összfogyasztás tekintetében a gőz mérőtípushoz kiugróan magas fogyasztás társul.

# In[93]:


for meter_type in sorted(train['meter'].unique().tolist()):
    meter_reading = np.log1p(train[train.meter == meter_type]['meter_reading'])
    sns.kdeplot(meter_reading, shade=True)

plt.legend(['Elektromos áram', 'Hidegvíz', 'Gőz', 'Melegvíz'], prop={'size': 12})
plt.title('Célváltozó (Energiafogyasztás KWh-ban) eloszlása mérőtípus alapján',  fontsize=14)
plt.ylabel('Sűrűség')  


# Az energiafogasztás (logaritmusának) eloszlása a különböző mérőtípusok szerinti bontásban rámutat arra, hogy nagy számban találhatunk zéró értéket az adatszettben, mely az elektromos áram esetében a legkevesebb, míg a melegvíz esetében a legtöbb. Az eloszlás némi eltérést mutat az egyes mérőtípusok esetében, főként az elektromos áram tekintetében, míg a hidegvíz és a gőz mőrő szerinti fogyasztás eloszlása meglehetősen hasonló. Összefoglalva, a mérőtípus változójának bevonása az elemzésbe indokolt az eltérő eloszlások láttán.

# #### Az idő aspektus vizsgálata a célváltozó függvényében

# In[94]:


train["timestamp"] = pd.to_datetime(train["timestamp"])
train["hour"] = train["timestamp"].dt.hour
train["day"] = train["timestamp"].dt.day
train["weekend"] = train["timestamp"].dt.weekday
train["month"] = train["timestamp"].dt.month


# In[95]:


train['log_meter_reading'] = np.log1p(train['meter_reading'])
fig, ax = plt.subplots(figsize=(10,5))
train.groupby('hour')['meter_reading'].median().plot(kind='line', color="darkblue", figsize=(10,6), label=False)
ax.set_ylabel('Energiafogyasztás középértéke')
ax.set_xlabel('Óra')
plt.title("Az Energiafogyasztás középértéke óránként",  fontsize=14)
plt.show()


# In[96]:


fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='Óránként', alpha=0.8).set_ylabel('Energiafogyasztás', fontsize=14);
train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='Naponként', alpha=1).set_ylabel('Energiafogyasztás', fontsize=14);
axes.set_title('Átlagos energiafogyasztás óránként és naponként', fontsize=16);
axes.legend()


# Az átlagos energiafogyasztás ábrája meglehetősen furán fest, március után jelentős ugrás látható, míg június közepétől szinte 0 közelébe esik.
# Érdemes tehát egy közelebbi pillantást vetni rá.

# In[99]:


title_list = list()
for sites in range(train['site_id'].nunique()):
    title_list.append(str('Helyszín ' + str(sites) + str(' energiafogyasztásának változása az idő függvényében')))
    
train.groupby(['timestamp', 'site_id'])['meter_reading'].mean().unstack().plot(subplots=True, layout=(8,2), figsize=(20,35), title=title_list)


# Egyrészről, az év eleji alacsony átlagos energiafogyasztásért részben az első helyszín (Site 0) április előtti adatainak hiánya, részben pedig a 6-os helyszín valószínűsíthető adathinya okolható. Másrészről, a 13-as helyszín ábrája nagyrészt hasonlít a teljes minta ábrájához. Amennyiben visszaemlékszünk arra, hogy a legtöbb mérési adat ezen a helyszínen lévő épületekről áll rendelkezésre, illetve az átlagos energiafogyasztás jelentősen nagyobb ezen a helyszínen, akkor könnyen belátható jelentős hatása a teljes mintára. Érdemes tehát külön megvizsgálni a 6-os helyszínt, illetve a 13-as helyszínt arra vonatkozólag, hogy az egyes épülettípusoknak miképpen alakul ezen a területeken az energiafogyasztása.

# In[106]:


site_13_df = train[train.site_id == 13]
    
site_13_df.groupby(['timestamp', 'primary_use'])['meter_reading'].mean().unstack().plot(subplots=True, layout=(8,2), figsize=(20,35), title='A 13-as helyszín energiafogyasztásának változása elsődleges használat szerint:')


# A fentiak alapján az oktatási intézmények között kell tovább kutatodni. Lássuk az ehhez a kategóriához tartozó épületeket.

# In[109]:


site_13_df = train[(train.site_id == 13) & (train.primary_use == 'Education')]
    
site_13_df.groupby(['timestamp', 'building_id'])['meter_reading'].mean().unstack().plot(subplots=True, layout=(20,2), figsize=(20,35), title='A 13-as helyszínhez tartozó oktatási épületek energiafogyasztásának változása:')


# Az 1099-es számú épület egyfajta "általános átlagként" funkcionál, nélküle az adatok időbeni eloszlása jelentősen változik:

# In[119]:


site_6_df = train[train.site_id == 6]
    
site_6_df.groupby(['timestamp', 'primary_use'])['meter_reading'].mean().unstack().plot(subplots=True, layout=(8,2), figsize=(20,35), title='A 6-os helyszín energiafogyasztásának változása elsődleges használat szerint:')


# A 6-os helyszín esetében a szórakoztatás kategóriát érdemes górcső alá venni:

# In[123]:


site_6_df = train[(train.site_id == 6) & (train.primary_use == 'Entertainment/public assembly')]
    
site_6_df.groupby(['timestamp', 'building_id'])['meter_reading'].mean().unstack().plot(subplots=True, layout=(2,2), figsize=(20,10), title='A 13-as helyszínhez tartozó oktatási épületek energiafogyasztásának változása:')


# A fentebbi ábra alapjá a 778-as és a 783-as épületek esetében nincs mérési adatunk az év nagyrészéről.

# In[125]:


train_whtout_fliers =  train[(train.building_id != 1099) & (train.building_id != 778) & (train.building_id != 783)]

fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
train_whtout_b1099[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='Óránként', alpha=0.8).set_ylabel('Energiafogyasztás', fontsize=14);
train_whtout_b1099[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='Naponként', alpha=1).set_ylabel('Energiafogyasztás', fontsize=14);
axes.set_title('Átlagos energiafogyasztás óránként és naponként a 778-as, a 783-as és az 1099-es épület nélkül', fontsize=16);
axes.legend()


# A jelentős adathiánnyal rendelkező épületek kizárása után az energiafogyasztás éves alakulása már természetesebb képet mutat, magasabb átlagos értékekkel a hideg téli és forró nyári hónapokban. A továbbiakban folytassuk az elemzést ezen épületek nélkül.

# In[126]:


train = train_whtout_fliers


# In[128]:


fig, ax = plt.subplots(figsize=(10,5))
ax = sns.boxplot(y="weekend", x="log_meter_reading", data=train, orient="h", palette="PuBu", showfliers=False)
ax.set_ylabel('Hét napjai')
ax.set_xlabel('Energiafogyasztás logaritmusának eloszlása')
ax.set_yticklabels(['Hétfő', 'Kedd', 'Szerda', 'Csütörtök', 'Péntek', 'Szombat', 'Vasárnap'])
plt.title('Az Energiafogyasztás logaritmusának eloszlása a hét napjai szerinti bontásban kiugró értékek nélkül',  fontsize=14)
plt.show()


# In[129]:


meter_types = ['Elektromos áram', 'Hidegvíz', 'Gőz', 'Melegvíz']
for meter_type in sorted(train['meter'].unique().tolist()):
    filtered_train = train[train.meter == meter_type]
    ax = filtered_train[['timestamp','log_meter_reading']].set_index('timestamp').resample("D")['log_meter_reading'].median().plot(kind='line', color="darkblue", figsize=(10,6), alpha=0.8, label=False)
    filtered_train[['timestamp','log_meter_reading']].set_index('timestamp').resample("H")['log_meter_reading'].median().plot(kind='line', color="darkblue", figsize=(10,6), alpha=0.3, label=False)
    ax2 = ax.twinx()
    filtered_train[['timestamp','air_temperature']].set_index('timestamp').resample("D")['air_temperature'].median().plot(ax=ax2, kind='line', color="darkred", figsize=(10,6), alpha=0.8, label=False)
    filtered_train[['timestamp','air_temperature']].set_index('timestamp').resample("H")['air_temperature'].median().plot(ax=ax2, kind='line', color="darkred", figsize=(10,6), alpha=0.3, label=False)
    ax.figure.legend(['Energiafogyasztás középértéke (nap és óra)', 'Levegőhőmérséklet középértéke (nap és óra)'], loc='lower right')
    ax.set_ylabel('Energiafogyasztás logaritmusának középértéke'); ax2.set_ylabel('Levegőhőmérséklet középértéke')
    ax.set_xlabel("Mérés ideje")
    plt.title("Az Energiafogyasztás és levegőhőmérséklet változása az idő függvényében, " + str('Mérőtípus: ') + str(meter_types[meter_type]),  fontsize=14)
    plt.show()


# In[130]:


ax = train[['timestamp','log_meter_reading']].set_index('timestamp').resample("H")['log_meter_reading'].mean().plot(kind='line',figsize=(10,6), alpha=0.7, label='Összes energiafogyasztás óránként')
ax2 = ax.twinx()
train[['timestamp','air_temperature']].set_index('timestamp').resample("H")['air_temperature'].median().plot(ax=ax2, kind='line', color="orange", figsize=(10,6), alpha=0.7, label='Levegőhőmérséklet középértéke óránként')
plt.legend()
plt.xlabel("Mérés ideje")
plt.ylabel("Átlagos energiafogyasztás KWh-ban")
plt.title("Az Energiafogyasztás és levegőhőmérséklet változása az idő függvényében")


# #### A használati típus szerinti energiafogyasztás vizsgálata

# In[131]:


sorted(train['primary_use'].unique().tolist())
fig, ax = plt.subplots(figsize=(10,12))
ax = sns.boxplot(y="primary_use", x="log_meter_reading", data=train_no_outliers, orient="h", palette="PuBu", showfliers=False)
ax.set_ylabel('Elsődleges használati típus')
ax.set_xlabel('Energiafogyasztás logaritmusának eloszlása')
plt.title('Az Energiafogyasztás logaritmusának eloszlása elsődleges használati típus szerint',  fontsize=14)
plt.show()


# In[132]:


train.groupby(['hour', 'primary_use'])['meter_reading'].median().unstack().plot(subplots=True, layout=(4,4))


# #### Energiafogyasztás vizsgálata épületenként

# In[134]:


train.groupby(['building_id'])['meter_reading'].sum().plot()


# ### <span style="color:dimgray"> Változók alakítása </span>

# #### Szélirány átalakítása kategórikus változóvá. Részleteket lásd a következő kernelben:
# https://www.kaggle.com/caesarlupum/ashrae-ligthgbm-simple-fe

# ### <span style="color:dimgray"> Model illesztése </span>

# In[ ]:




