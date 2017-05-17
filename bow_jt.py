
# coding: utf-8

# In[3]:

import sys
import re
from sklearn import svm
from svmutil import *
import re
import nltk
from nltk.stem import PorterStemmer
from random import shuffle
from sklearn.metrics import classification_report


# In[4]:

def get_znacilke(komentarji, vse_besede):
    
    # Metoda vrne značilke, ki se pojavljajo v posameznem komentarju
    
    urejene_besede = sorted(vse_besede)
    znacilke = []
    
    for komentar in komentarji:
        word_map = {}
        
        # Za vsak komentar nastavimo pojavitve vseh besed na 0
        for beseda in urejene_besede:
            word_map[beseda] = 0
        
        # Za vsako besedo (del komentarja), ki se pojavi, nastavimo pojavitev na 1
        for del_komentarja in komentar:
            if(del_komentarja in word_map):
                word_map[del_komentarja] = 1
              
        # Pojavitev besed trenutnega komentarja dodamo v skupne pojavitve
        values = list(word_map.values())
        znacilke.append(values)

    return znacilke


# In[5]:

def get_znacilke_oznake(komentarji, vse_besede):

    # Metoda vrne slovar z tipi komentarji in značilkami, ki se pojavljajo v posameznem komentarju
    
    urejene_besede = sorted(vse_besede)
    znacilke = []
    oznake = []
    
    for komentar in komentarji:
        oznaka = 0
        word_map = {}
        
        # Za vsak komentar nastavimo pojavitve vseh besed na 0
        for beseda in urejene_besede:
            word_map[beseda] = 0
        
        if len(komentar) > 1:
            # Komentar razbijemo na 2 dela -> tip komentarja in sam komentar
            besedilo_komentarja = komentar[0]
            zaljivost_komentarja = komentar[1]
        
            # Za vsako besedo (del komentarja), ki se pojavi, nastavimo pojavitev na 1
            for del_komentarja in besedilo_komentarja:
                word_map[del_komentarja] = 1
            
            # Pojavitev besed trenutnega komentarja dodamo v skupne pojavitve
            values = list(word_map.values())
            znacilke.append(values)

            # Preverimo tip komentarja
            if zaljivost_komentarja == 0:
                oznaka = 0
            elif zaljivost_komentarja == 1:
                oznaka = 1

            # Dodamo med vse tipe komentarjev
            oznake.append(oznaka)
        
    return {'pojavitev_besed' : znacilke, 'oznake':oznake}


# In[6]:

def preuredi_besedilo(tekst):

    # Metoda vrne prečiščen tekst

    tekst = tekst.lower()
    # Odstrani URL-je
    tekst = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','', tekst)
    # Odstrani NEWLINE_TOKEN
    tekst = re.sub('newline_token', '', tekst)
    tekst = re.sub('newline', '', tekst)
    tekst = re.sub('tokennewline', '', tekst)
    # Odstrani TAB_TOKEN
    tekst = re.sub('tab_token', '', tekst)
    # Pretvori `` to "
    tekst = re.sub('``', '"', tekst)
    # Pretvori ` to '
    tekst = re.sub('`', '\'', tekst)
    # Odstrani =
    tekst = re.sub('=', '', tekst)
    # Odstrani \n
    tekst = re.sub('\n', '', tekst)
    # Odstrani :
    tekst = re.sub(':', '', tekst)
    # Odstrani odvečne presledke
    tekst = re.sub('[\s]+', ' ', tekst)
    # Odstrani začetne/končnne presledke
    tekst = tekst.strip()
    # Odstrani narekovaje
    tekst = tekst.strip('\'"')

    return tekst


# In[7]:

def get_posamezne_besede(komentar, stopwords):
    
    # Metoda odstrani presledke, končnice in vrne posamezne besede

    # Odstranimo nepotrebne whitespace in spremenimo v male črke
    komentar = preuredi_besedilo(komentar)
    komentar = " ".join(re.split("[^a-zA-Z]*", komentar.lower())).strip()
    
    stemmer = PorterStemmer()
    besede = []
    
    # Razbijemo na besede in porežeo končnice
    for beseda in komentar.split():
        if not beseda in stopwords:
            try:
                besede.append(stemmer.stem(beseda))
            except(IndexError):
                besede.append(beseda)

    return besede


# In[8]:

def get_ngram(komentar, min_n, max_n, stopwords):
    
    # Metoda vrne n-grame za določen tweet
    
    komentar = preuredi_besedilo(komentar)
    komentar = " ".join(re.split("[^a-zA-Z]*", komentar.lower())).strip()
    
    stemmer = PorterStemmer()
    
    ngram = []
    besede = komentar.split()
    
    for k in range(min_n, max_n + 1):
        for i in range(0, len(besede) - k):
            posamezni_ngram = ""
            
            sw_stevec = 0
            for j in range(0, k + sw_stevec):
                if not besede[i + j] in stopwords:
                    posamezni_ngram += stemmer.stem(besede[i + j])
                    if j < max_n:
                        posamezni_ngram += " "
                else:
                    sw_stevec += 1
            
            ngram.append(posamezni_ngram)
    
    return ngram


# In[9]:

def run(tekst, test_tekst, algoritem):

    # Pridobitev stop words (besed brez pomena)
    # nltk.download()
    stopwords = nltk.corpus.stopwords.words("english")

    komentarji = []
    besede = []

    for vrstica in tekst:
        # Vrstico razbijemo na 2 dela - tip komentarja in sam komentar
        vrstica = vrstica.split("\t")
        tip_zaljivosti = int(vrstica[0])
        komentar = vrstica[1]
        # Sam komentar uredimo in odstranimo nepotrebne znake
        # Pridobimo posamezne besede (s pomenom in brez končnic)
        
        if algoritem == "BOW":
            besede_komentarja = get_posamezne_besede(komentar, stopwords)
        elif algoritem == "NGRAM":
            # besede_komentarja = get_ngram(komentar, 1, 1, stopwords)
            besede_komentarja = get_ngram(komentar, 1, 3, stopwords)
        
        # Ustvarimo nove - urejene komentarje in dodamo tip komentarja
        komentarji.append((besede_komentarja, tip_zaljivosti))
        # V slovar vseh besed dodamo besede trenutno obravnavanega slovarja
        for beseda in besede_komentarja:
            besede.append(beseda)

    # Učenje na podlagi učne množice
    znacilke_oznake = get_znacilke_oznake(komentarji, besede)    
    problem = svm_problem(znacilke_oznake['oznake'], znacilke_oznake['pojavitev_besed'])
    param = svm_parameter('-s 1 -b 1')
    param.kernel_type = LINEAR #LINEAR RBF SIGMOID POLY
    ucenec = svm_train(problem, param)
    # svm_save_model('classifier.txt', classifier)
    
    test_besede = []
    # v test_zaljivosti hranimo podatke o pravih klasifikacijah - za kasnejše preverjanje
    test_zaljivost = []

    for vrstica in test_tekst:
        # Vrstico razbijemo na 2 dela - tip komentarja in sam komentar
        vrstica = vrstica.split("\t")
        # Vrsto komentarja si shranimo za kasnejše preverjanje
        tip_zaljivosti = int(vrstica[0])
        # Komentar obdelamo, pridobimo posamezne besede in jih dodamo v slovar
        komentar = vrstica[1]
        # test_posamezne_besede = [beseda.lower() for beseda in komentar.split()]
        
        if algoritem == "BOW":
            test_besede_komentarja = get_posamezne_besede(komentar, stopwords)
        elif algoritem == "NGRAM":
            # test_besede_komentarja = get_ngram(komentar, 1, 1, stopwords)
            test_besede_komentarja = get_ngram(komentar, 1, 3, stopwords)
        
        test_besede.append(test_besede_komentarja)
        test_zaljivost.append(tip_zaljivosti)

    print "\n" + algoritem
        
    # Pridobitev znacilk    
    test_znacilke = get_znacilke(test_besede, besede)
    predvidevana_zaljivost, a, b = svm_predict(test_zaljivost, test_znacilke, ucenec, options = "-b 1")
    
    print(classification_report(test_zaljivost, predvidevana_zaljivost))
    
    '''# Skupaj - st testnih podatkov, pravilni - pravilno klasificirani
    # napacno - podatki ki bi morali biti klasificirani kot hate speech pa niso
    # stevec - števec za array
    skupaj, pravilni, napacni, stevec = 0, 0, 0, 0
    natancnost, priklic = 0.0, 0.0
    
    # Preštejemo število pravilno klasificiranih komentarjev
    for zaljivost in test_zaljivost:
        oznaka = int(predvidevana_zaljivost[stevec])
        if(oznaka == int(zaljivost)):
            pravilni += 1
        elif (oznaka != int(zaljivost) & zaljivost == 1)
            napacni += 1
        skupaj += 1
        stevec += 1
    
    natancnost = (float(pravilni) / skupaj) * 100
    priklic = (float(pravilni / (pravilni + napacni)))'''
    


# In[10]:

def main():
    # Pridobitev vseh podatkov
    datoteka = open('data/comments.txt')
    tekst = datoteka.readlines()[1:]
    datoteka.close()
    
    shuffle(tekst)
    
    st_komentarjev = len(tekst)
    deli = st_komentarjev * 100 / 70
    
    # Pridobitev učnih podatkov
    # ucna_datoteka = open('data/comments_sample.txt')
    # ucna_tekst = ucna_datoteka.readlines()[1:]
    # ucna_datoteka.close()
    
    ucna_tekst = tekst[1:5000]
    
    # Pridobitev testnih podatkov
    # test_datoteka = open("data/test_comments.txt")
    # test_tekst = test_datoteka.readlines()[1:]
    # test_datoteka.close()
    
    test_tekst = tekst[5000:6000]
    
    run(ucna_tekst, test_tekst, "BOW")
    run(ucna_tekst, test_tekst, "NGRAM")


# In[16]:

main()


# In[ ]:




# Kako učimo na učni množici:
#     1. Preberemo učno datoteko s komentarji (tip komentarja in tekst)
#     2. Pridobimo stopworde (besede brez pomena)
#     3. Vsako vrstico razdelimo na tip in sam komentar:
#         1. Komentar uredimo (odstranimo odvečne znake, odstranimo končnice, pripravimo na procesiranje)
#         2. Pridobimo besede v posameznem komentarju, dodamo jih v 'slovar besed'
#     4. Slovar besed in komentarje obdelamo:
#         1. V slovarju za vsak komentar shranimo tip komentarja in besede ki se oz. 
#         se ne pojavljajo v komentarju
#         2. Besede, ki se pojavijo v slovarju in v komentarju so označene z 1, ostale z 0.
# 
# Kako testiramo testno množico:
#     1. Preberemo testno datoteko s komentarji (top komentarja in tekst)
#     2. Pridobimo stopworde (besede brez pomena)
#     3. Vsako vrstico razbijemo na tip in sam komentar:
#         1. Tip si shranimo za kasnejše preverjanje uspešnosti klasifikacije komentarja
#         2. Komentar uredimo (odstranimo odvečne znake, odstranimo končnice, pripravimo na procesiranje)
#         3. Pridobimo besede v posameznem komentarju, dodamo jih v 'slovar testnih besed'
#     4. Slovar besed in testnih komentarjev obdelamo:
#         1. Pridobimo informacije o besedah ki se oz. se ne pojavljaju v komentarju
#         2. Besede, ki se pojavijo v slovarju in komentarju so označene z 1, ostale z 0.
#     5. Komentarjem na podlagi pojavljanja besed določimo tip (svm_predict)
#     6. Izračunamo verjetnost (pravilno določeni tipi / vsi določeni tipi)*100
#    
#     
#         
