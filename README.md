# ML_journey

This repository is a collection of my personal projects, notes, and experiments as I explore and learn about machine learning algorithms using Python.
The goal is to document my progress, test different models, and better understand key concepts in machine learning — from basic techniques to more advanced topics.
Feel free to browse through the code, and don't hesitate to share feedback or suggestions!


✅ Etap 1: Podstawy Pythona + biblioteki do ML
📘 Nauka:
Składnia Pythona (jeśli potrzebujesz odświeżenia)

Operacje na tablicach: numpy

Praca z danymi: pandas

Wizualizacje: matplotlib, seaborn

💻 Praktyka w PyCharm:
Załaduj dane z pliku .csv

Operuj na DataFrame (sortuj, filtruj, oblicz średnią, itp.)

Narysuj wykresy (histogramy, wykresy punktowe)

✅ Etap 2: Regresja liniowa (pierwszy model ML)
📘 Nauka:
Czym jest regresja liniowa

Jak wygląda model predykcyjny y = ax + b

Metryki: MSE, RMSE, R²

💻 Praktyka w PyCharm:
Załaduj dane (np. ceny domów lub dane sztuczne)

Użyj scikit-learn: LinearRegression

Podziel dane: train_test_split

Oblicz metryki

Zwizualizuj dane i linię regresji

📁 Struktura katalogu projektu:
ml_intro/
├── data/
│   └── housing.csv
├── models/
│   └── linear_regression.py
├── utils/
│   └── data_loader.py
└── main.py


✅ Etap 3: Klasyfikacja – regresja logistyczna
📘 Nauka:
Klasyfikacja vs regresja

Regresja logistyczna – jak przewiduje klasy

Macierz pomyłek, accuracy, recall, precision, F1

💻 Praktyka:
Użyj LogisticRegression z scikit-learn

Pracuj na zbiorze Titanic, Iris lub Breast Cancer

Oblicz i narysuj metryki (np. wykres ROC)

Wprowadź confusion_matrix i classification_report

✅ Etap 4: Przetwarzanie danych i inżynieria cech
📘 Nauka:
Czyszczenie danych (NaN, duplikaty)

Skalowanie (StandardScaler, MinMaxScaler)

Kodowanie kategorii (OneHotEncoder, LabelEncoder)

Wybór cech (SelectKBest, PCA)

💻 Praktyka:
Stwórz pipeline preprocessingowy

Porównaj wyniki modelu z/bez preprocessing’u

✅ Etap 5: Inne algorytmy ML
📘 Algorytmy do poznania:
Decision Trees (DecisionTreeClassifier)

KNN (KNeighborsClassifier)

Naive Bayes (GaussianNB)

SVM (SVC)

💻 Praktyka:
Przetestuj różne modele na tym samym zbiorze

Porównaj wyniki

Użyj cross_val_score, GridSearchCV do strojenia modeli

✅ Etap 6: Mini-projekt ML (Twoja aplikacja)
📘 Pomysły:
Klasyfikacja maili: spam/nie-spam

Przewidywanie cen mieszkań

Analiza opinii (sentiment analysis)

Przewidywanie kto przeżyje z Titanica

💻 W PyCharm:
Zrób osobny katalog projektu

Podziel kod na main.py, models/, data/, notebooks/

Zrób raport: wyniki, wykresy, interpretacja

📅 Propozycja tygodniowego harmonogramu (8 tygodni)
| Tydzień | Temat                                                  |
| ------- | ------------------------------------------------------ |
| 1       | Instalacja, środowisko, biblioteki (`numpy`, `pandas`) |
| 2       | Wizualizacja danych + projekt w PyCharm                |
| 3       | Regresja liniowa                                       |
| 4       | Regresja logistyczna                                   |
| 5       | Czyszczenie danych, encoding, skalowanie               |
| 6       | Inne modele: drzewa, KNN, SVM                          |
| 7       | Strojenie modeli (`GridSearchCV`), walidacja           |
| 8       | Projekt końcowy i analiza                              |


