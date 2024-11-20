import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import kagglehub
import seaborn as sns

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

#Duomenu paruosimas
X = df3.drop('Class', axis=1)
Y = df3['Class']
X = X.select_dtypes(include=['float64', 'int64'])

print("Trūkstamos reikšmės X:")
print(X.isnull().sum())
print("Dublikatų skaičius:", X.duplicated().sum())
# Popularity, Key, instrumentalness užpildymas pagal klasių vidurkius
X['instrumentalness'] = X.groupby(Y)['instrumentalness'].transform(lambda x: x.fillna(x.mean()))
X['Popularity'] = X.groupby(Y)['Popularity'].transform(lambda x: x.fillna(x.mean()))
X['key'] = X.groupby(Y)['key'].transform(lambda x: x.fillna(x.mean()))
X['duration_in min/ms'] = X['duration_in min/ms'].apply(lambda x: x * 60 * 1000 if x > 30 else x)
# Pašalink dublikatus
X = X.drop_duplicates()
# Atitinkamai atnaujink ir Y, pašalindamas tas pačias eilutes
Y = Y.loc[X.index]
# Patikrink, ar dublikatai pašalinti
print("Dublikatų skaičius po pašalinimo:", X.duplicated().sum())
print("Trūkstamos reikšmės X_train po apdorojimo:")
print(X.isnull().sum())
print(Y.groupby(Y).count())

#1 nebalanuosti duomenys
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

#2subalansuojam duomenys po 350
processed_df = pd.concat([X, Y], axis=1)
train_df350 = processed_df.groupby('Class', group_keys=False).apply(lambda x: x.sample(n=350, random_state=42))
X_train350 = train_df350.drop('Class', axis=1).select_dtypes(include=['float64', 'int64'])
Y_train350 = train_df350['Class']

test_indices = processed_df.index.difference(train_df350.index)
test_df350 = processed_df.loc[test_indices]
X_test350 = test_df350.drop('Class', axis=1).select_dtypes(include=['float64', 'int64'])
Y_test350 = test_df350['Class']


#3 padarom visiem duomenim dublikatu iki 1000
def create_final_datasets(X_train, Y_train, X_train350, Y_train350, X_test350, Y_test350):
    # Sukuriame galutinį treniravimo rinkinį
    final_X_train = pd.DataFrame()
    final_Y_train = pd.Series(dtype='int')  # Pradinė tuščia serija

    # Patikriname klases ir jų dažnius originaliame rinkinyje
    class_counts = Y_train.value_counts()

    for cls, count in class_counts.items():
        if count >= 1000:
            # Jei klasės turi daugiau nei 1000, tiesiog imame esamus duomenis
            final_X_train = pd.concat([final_X_train, X_train[Y_train == cls]])
            final_Y_train = pd.concat([final_Y_train, Y_train[Y_train == cls]])
        else:
            # Jei klasių yra mažiau nei 1000, sudublikuojame
            needed_samples = 1000 - count
            samples_to_duplicate = X_train350[Y_train350 == cls].sample(n=needed_samples, replace=True, random_state=42)
            final_X_train = pd.concat([final_X_train, samples_to_duplicate])
            final_Y_train = pd.concat([final_Y_train, pd.Series([cls] * needed_samples)])

    # Sudedame testavimo rinkinį
    final_X_test = X_test350
    final_Y_test = Y_test350

    return final_X_train, final_Y_train, final_X_test, final_Y_test

# Sukuriame galutinius duomenų rinkinius
X_final_train, Y_final_train, X_final_test, Y_final_test = create_final_datasets(X_train, Y_train, X_train350, Y_train350, X_test350, Y_test350)

# Modelių treniravimas ir vertinimas
datasets = {
    "Original": (X_train, X_test, Y_train, Y_test),
    "Balanced 350": (X_train350, X_test350, Y_train350, Y_test350),
    "Final Dataset": (X_final_train, X_final_test, Y_final_train, Y_final_test)
}



# Modeliai
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'KNN': KNeighborsClassifier()
}

for dataset_name, (X_tr, X_te, Y_tr, Y_te) in datasets.items():
    print(f"\nEvaluating models on {dataset_name} dataset:")
    # Skalizuojame kiekvieną duomenų rinkinį atskirai
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Treniruojame ir vertiname modelius
    for name, model in models.items():
        model.fit(X_tr, Y_tr)  # Treniruojame modelį
        predictions = model.predict(X_te)  # Prognozuojame testavimo rinkinyje
        accuracy = accuracy_score(Y_te, predictions)  # Apskaičiuojame tikslumą
        print(f'{name} on {dataset_name} '
              f'\nAccuracy: {accuracy:.2f}')

    # Darbas su geriausiu modeliu Random Forest
    print(f"\nPerforming GridSearchCV on {dataset_name} dataset using Random Forest:")
    param_grid = {
        'n_estimators': [75, 100],
        'max_depth': [15, 20],
        'bootstrap': [True],
        'criterion': ['gini', 'entropy']
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_tr, Y_tr)

    # Best estimator evaluation
    best_rf = grid_search.best_estimator_
    Y_pred = best_rf.predict(X_te)
    accuracy = accuracy_score(Y_te, Y_pred)

    print(f"Best parameters for {dataset_name}: {grid_search.best_params_}")
    print(f"Accuracy on best parameters: {accuracy:.2f}")

    # Feature importance vizualizacija
    feature_importance_rf = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 10))
    plt.barh(feature_importance_rf.index, feature_importance_rf.values)
    plt.title(f"Feature Importance ({dataset_name} Random Forest)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Confusion Matrix ir Classification Report
    cm = confusion_matrix(Y_te, Y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_rf.classes_, yticklabels=best_rf.classes_)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for {dataset_name} Best Random Forest Model")
    plt.show()

    print(f"Classification Report for {dataset_name}:\n", classification_report(Y_te, Y_pred))

# model_accuracies = {
#     'Logistic Regression': lr_accuracy,
#     'Decision Tree': dt_accuracy,
#     'Random Forest': rf_accuracy,
#     'Naive Bayes': nb_accuracy,
#     'KNN': knn_accuracy
# }
#
# plt.figure(figsize=(10, 6))
# plt.bar(model_accuracies.keys(), model_accuracies.values(), color='skyblue')
# plt.xlabel("Model")
# plt.ylabel("Accuracy")
# plt.title("Accuracy of Different Models")
# plt.show()



