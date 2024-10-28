import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import kagglehub

# Download latest version
#path = kagglehub.dataset_download("purumalgi/music-genre-classification")

#print("Path to dataset files:", path)


df1 = pd.read_csv('C:/Users/drawn/Mokymai/DDRAV Mokymai/Miniprojektas/classification/submission.csv')
df2 = pd.read_csv('C:/Users/drawn/Mokymai/DDRAV Mokymai/Miniprojektas/classification/test.csv')
df3 = pd.read_csv('C:/Users/drawn/Mokymai/DDRAV Mokymai/Miniprojektas/classification/train.csv')


print("\nSecond file head:")
print(df2.head())
print("\nSecond file info:")
print(df2.info())

print("\nThird file head:")
print(df3.head())
print("\nThird file info:")
print(df3.info())

#Duomenu paruosims
X = df3.drop('Class', axis=1)
Y = df3['Class']
X = X.select_dtypes(include=['float64', 'int64'])

#print("Trūkstamos reikšmės X:")
#print(X.isnull().sum())
#print("Dublikatų skaičius:", X.duplicated().sum())
# Instrumentalness užpildymas 0
X['instrumentalness'] = X['instrumentalness'].fillna(0)

# Popularity ir Key užpildymas pagal klasių vidurkius
X['Popularity'] = X.groupby(Y)['Popularity'].transform(lambda x: x.fillna(x.mean()))
X['key'] = X.groupby(Y)['key'].transform(lambda x: x.fillna(x.mean()))
X['duration_in min/ms'] = X['duration_in min/ms'].apply(lambda x: x * 60 * 1000 if x > 30 else x)

# Pašalink dublikatus
X = X.drop_duplicates()

# Atitinkamai atnaujink ir Y, pašalindamas tas pačias eilutes
Y = Y.loc[X.index]

# Patikrink, ar dublikatai pašalinti
#print("Dublikatų skaičius po pašalinimo:", X.duplicated().sum())
#print("Trūkstamos reikšmės X_train po apdorojimo:")
#print(X.isnull().sum())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Skalizuok požymius
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modeliai
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

# Treniruojame ir vertiname modelius
for name, model in models.items():
    model.fit(X_train, Y_train)  # Treniruojame modelį
    predictions = model.predict(X_test)  # Prognozuojame testavimo rinkinyje
    accuracy = accuracy_score(Y_test, predictions)  # Apskaičiuojame tikslumą
    print(f'{name} '
          f'\nPredictions: {predictions}'
          f'\nAccuracy: {accuracy:.2f}')

# Logistinė regresija
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, Y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(Y_test, lr_pred)
print(f'Logistic Regression Accuracy: {lr_accuracy:.2f}')

# Sprendimų medis
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, Y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(Y_test, dt_pred)
print(f'Decision Tree Accuracy: {dt_accuracy:.2f}')


# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(Y_test, rf_pred)
print(f'Random Forest Accuracy: {rf_accuracy:.2f}')

# Naivusis Bajeris
nb = GaussianNB()
nb.fit(X_train, Y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(Y_test, nb_pred)
print(f'Naive Bayes Accuracy: {nb_accuracy:.2f}')

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(Y_test, knn_pred)
print(f'KNN Accuracy: {knn_accuracy:.2f}')