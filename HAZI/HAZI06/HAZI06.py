import numpy as np
import pandas as pd
import time

from NJCleaner import NJCleaner
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cleaner = NJCleaner('2018_03.csv')
cleaner.prep_df()

col_name = ['stop_sequence', 'from_id', 'to_id', 'status', 'line', 'type', 'day', 'part_of-day', 'delay']
data = pd.read_csv('data/NJ.csv', skiprows=1, header=None, names=col_name)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

"""
results = []
for samples in [10, 25, 35, 40, 50]:
    for depth in [6, 7]:
        start = time.time()
        classifier = DecisionTreeClassifier(min_samples_split=samples, max_depth=depth)
        classifier.fit(X_train, Y_train)

        Y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        end = time.time()
        print("min sample: ", samples, "depth: ", depth, "acc: ", accuracy, " TIME: ", (end-start), "sec")
        results.append((samples, depth, accuracy))

print(max(results, key=lambda x:x[2]))
"""
start = time.time()
min_samples_split = 100
max_depth = 6
classifier = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
end = time.time()
print("min sample: ", min_samples_split, "depth: ", max_depth, "acc: ", accuracy, " TIME: ", (end-start), "sec")


"""
4.

A túl mély fák errorokhoz vezettek, így nagyjából 6-7 mélységig próbálkoztam. Ekkora adathalmaznál véleményem szerint még nem vezet overfittinghez.
A nagyobb mélység ennek ellenére jobb pontosságot eredményezett
A min samples split paraméter ~100 körüli értékkel adta a legjobb eredményt.

min sample:  2      depth:  5       acc:  0.7863  TIME:  115.49 sec
min sample:  2      depth:  7       acc:  0.7885  TIME:  143.12 sec
min sample:  100    depth:  2       acc:  0.7823  TIME:  74.60 sec
min sample:  100    depth:  4       acc:  0.7852  TIME:  107.45 sec
min sample:  100    depth:  5       acc:  0.7863  TIME:  122.29 sec
min sample:  100    depth:  6       acc:  0.7909  TIME:  132.14 sec
min sample:  3000   depth:  6       acc:  0.7823  TIME:  120.13 sec
min sample:  500    depth:  6       acc:  0.7886  TIME:  128.52 sec
min sample:  10     depth:  6       acc:  0.7903  TIME:  129.69 sec
min sample:  10     depth:  7       acc:  0.7885  TIME:  139.94 sec
min sample:  25     depth:  6       acc:  0.7903  TIME:  129.73 sec
min sample:  25     depth:  7       acc:  0.7887  TIME:  140.16 sec
min sample:  35     depth:  6       acc:  0.7904  TIME:  129.42 sec
min sample:  35     depth:  7       acc:  0.7888  TIME:  139.95 sec
min sample:  40     depth:  6       acc:  0.7904  TIME:  133.43 sec
min sample:  40     depth:  7       acc:  0.7886  TIME:  164.16 sec
min sample:  50     depth:  6       acc:  0.7904  TIME:  148.78 sec

1.  Értelmezd az adatokat!!!
    A feladat megoldásához használd a NJ transit + Amtrack csv-t a moodle-ból.
    A NJ-60k az a megoldott. Azt fogom használni a modellek teszteléséhez, illetve össze tudod hasonlítani az eredményedet.    

2. Írj egy osztályt a következő feladatokra:  
     2.1 Neve legyen NJCleaner és mentsd el a NJCleaner.py-ba. Ebben a fájlban csak ez az osztály legyen.
     2.2 Konsturktorban kapja meg a csv elérési útvonalát és olvassa be pandas segítségével és mentsük el a data (self.data) osztályszintű változóba 
     2.3 Írj egy függvényt ami sorbarendezi a dataframe-et 'scheduled_time' szerint növekvőbe és visszatér a sorbarendezett df-el, a függvény neve legyen 'order_by_scheduled_time' és térjen vissza a df-el  
     2.4 Dobjuk el a from és a to oszlopokat, illetve azokat a sorokat ahol van nan és adjuk vissza a df-et. A függvény neve legyen 'drop_columns_and_nan' és térjen vissza a df-el  
     2.5 A date-et alakítsd át napokra, pl.: 2018-03-01 --> Thursday, ennek az oszlopnak legyen neve a 'day'. Ezután dobd el a 'date' oszlopot és térjen vissza a df-el. A függvény neve legyen 'convert_date_to_day' és térjen vissza a df-el   
     2.6 Hozz létre egy új oszlopot 'part_of_the_day' névvel. A 'scheduled_time' oszlopból számítsd ki az alábbi értékeit. A 'scheduled_time'-ot dobd el. A függvény neve legyen 'convert_scheduled_time_to_part_of_the_day' és térjen vissza a df-el  
         4:00-7:59 -- early_morning  
         8:00-11:59 -- morning  
         12:00-15:59 -- afternoon  
         16:00-19:59 -- evening  
         20:00-23:59 -- night  
         0:00-3:59 -- late_night  
    2.7 A késéseket jelöld az alábbiak szerint. Az új osztlop neve legyen 'delay'. A függvény neve legyen pedig 'convert_delay' és térjen vissza a df-el
         0min <= x < 5min   --> 0  
         5min <= x          --> 1  
    2.8 Dobd el a felesleges oszlopokat 'train_id' 'actual_time' 'delay_minutes'. A függvény neve legyen 'drop_unnecessary_columns' és térjen vissza a df-el
    2.9 Írj egy olyan metódust, ami elmenti a dataframe első 60 000 sorát. A függvénynek egy string paramétere legyen, az pedig az, hogy hova mentse el a csv-t (pl.: 'data/NJ.csv'). A függvény neve legyen 'save_first_60k'. 
    2.10 Írj egy függvényt ami a fenti függvényeket összefogja és megvalósítja (sorbarendezés --> drop_columns_and_nan --> ... --> save_first_60k), a függvény neve legyen 'prep_df'. Egy paramnétert várjon, az pedig a csv-nek a mentési útvonala legyen. Ha default value-ja legyen 'data/NJ.csv'

3.  A feladatot a HAZI06.py-ban old meg.
    Az órán megírt DecisionTreeClassifier-t fit-eld fel az első feladatban lementett csv-re. 
    A feladat célja az, hogy határozzuk meg azt, hogy a vonatok késnek-e vagy sem. 0p <= x < 5p --> nem késik (0), ha 5p <= x --> késik (1).
    Az adatoknak a 20% legyen test és a splitelés random_state-je pedig 41 (mint órán)
    A testset-en 80% kell elérni. Ha megvan a minimum százalék, akkor azzal paraméterezd fel a decisiontree-t és azt kell leadni.

    A leadásnál csak egy fit kell, ezt azzal a paraméterre paraméterezd fel, amivel a legjobb accuracy-t elérted.

    A helyes paraméter megtalálásához használhatsz grid_search-öt.
    https://www.w3schools.com/python/python_ml_grid_search.asp 

4.  A tanításodat foglald össze 4-5 mondatban a HAZI06.py-ban a fájl legalján kommentben. Írd le a nehézségeket, mivel próbálkoztál, mi vált be és mi nem. Ezen kívül írd le 10 fitelésed eredményét is, hogy milyen paraméterekkel probáltad és milyen accuracy-t értél el. 
Ha ezt feladatot hiányzik, akkor nem fogadjuk el a házit!

HAZI-
    HAZI06-
        -NJCleaner.py
        -HAZI06.py

##################################################################
##                                                              ##
## A feladatok közül csak a NJCleaner javítom unit test-el      ##
## A decision tree-t majd manuálisan fogom lefuttatni           ##
## NJCleaner - 10p, Tanítás - acc-nál 10%-ként egy pont         ##
## Ha a 4. feladat hiányzik, akkor nem tudjuk elfogadni a házit ##
##                                                              ##
##################################################################


"""