import pandas as pd
from complementNB import AssetClassifier
from configuration import NAME_FILE_DATA, RANDOM_SAMPLING

print('Загрузка данных...', end='\r')
df_all = pd.read_excel(NAME_FILE_DATA).dropna(subset=['name', 'group'])
print("Данные выгружены!")

if RANDOM_SAMPLING:
    df = df_all.sample(n=5000, random_state=42)
else:
    df = df_all.loc[:,:]

# Инициализация и обучение
classifier = AssetClassifier(max_features=20000)
classifier.train(df, text_column='name', target_column='group')


# Сохранение и загрузка модели
model_path, encoder_path = classifier.save_model("my_model")
loaded_classifier = AssetClassifier.load_model(model_path, encoder_path)

# Предсказание на новых данных
new_data = pd.DataFrame({'name': ["Компьютер Dell", "Офисное кресло"]})
predictions = loaded_classifier.predict(new_data, return_proba=True)
print(predictions)
predictions.to_excel('1.xlsx')