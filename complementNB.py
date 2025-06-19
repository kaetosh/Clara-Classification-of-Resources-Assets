


import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns

from configuration import STOPWORDS_RU


class TextCleaner(BaseEstimator, TransformerMixin):
    """Кастомный трансформер для очистки текста"""
    def __init__(self, stop_words=None):
        self.stop_words = stop_words or set()

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'\d+|[{}]'.format(re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~«»—')), ' ', text)
        text = ' '.join(text.split())
        words = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(words).strip()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype(str).apply(self.clean_text)

class AssetClassifier:
    """Класс для классификации активов компании по наименованиям"""

    def __init__(self,
                 max_features=30000,
                 test_size=0.2,
                 random_state=42,
                 n_jobs=-1):
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.label_encoder = LabelEncoder()
        self.model = None
        self.name_model = None
        self.stop_words_russian = STOPWORDS_RU  # Пример стоп-слов

    def _is_sample_suitable(self,
                            X,
                            y,
                            label_encoder=None,
                            min_samples=1000,
                            min_classes=2,
                            min_samples_per_class=5):
        """
        Проверка, подходит ли выборка для обучения.

        Параметры:
        - X: признаки (тексты)
        - y: метки (закодированные числа)
        - label_encoder: экземпляр LabelEncoder для декодирования меток
        - min_samples: минимальное общее количество образцов
        - min_classes: минимальное количество уникальных классов
        - min_samples_per_class: минимальное количество примеров для каждого класса

        Возвращает:
        - True, если выборка подходит, иначе False
        """
        # Проверка общего количества образцов
        if len(X) < min_samples:
            print(f"⚠️ Выборка слишком мала (n_samples={len(X)} < {min_samples}).")
            return False

        # Проверка количества уникальных классов
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < min_classes:
            print(f"⚠️ Слишком мало классов (n_classes={len(unique_classes)} < {min_classes}).")
            return False

        # Проверка, что в каждом классе достаточно примеров
        if np.any(class_counts < min_samples_per_class):
            # Декодируем числовые метки в оригинальные названия (если передан label_encoder)
            if label_encoder is not None:
                problematic_classes = {
                    label_encoder.inverse_transform([cls])[0]: int(count)
                    for cls, count in zip(unique_classes, class_counts)
                    if count < min_samples_per_class
                }
            else:
                problematic_classes = {
                    int(cls): int(count)
                    for cls, count in zip(unique_classes, class_counts)
                    if count < min_samples_per_class
                }

            print(f"⚠️ Некоторые классы содержат слишком мало примеров: {problematic_classes} < {min_samples_per_class}.")
            return False

        # Проверка, что после векторизации останутся признаки
        vectorizer = TfidfVectorizer(min_df=1, max_df=1.0)
        try:
            X_vec = vectorizer.fit_transform(X)
            if X_vec.shape[1] == 0:
                print("⚠️ После векторизации не осталось признаков. Попробуйте изменить `min_df`/`max_df`.")
                return False
        except ValueError as e:
            print(f"⚠️ Ошибка векторизации: {e}")
            return False

        return True

    def train(self, df, text_column='name', target_column='group'):
        """
        Обучение модели на предоставленных данных
        Args:
            df: DataFrame с данными
            text_column: название колонки с текстом
            target_column: название колонки с целевой переменной
        """
        try:
            # Подготовка данных
            df = df.dropna(['name', 'group'], how='any')
            self.name_model = df['model'].iloc[0]
            X = df[text_column].astype(str)
            y = self.label_encoder.fit_transform(df[target_column])

            if not self._is_sample_suitable(X, y):
                raise ValueError("Выборка не подходит для обучения")

            # Разделение на train/test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )

            # Создание пайплайна
            pipeline = make_pipeline(
                TextCleaner(stop_words=self.stop_words_russian),
                TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=(1, 3),
                    sublinear_tf=True,
                    analyzer='word',
                    min_df=2,
                    max_df=0.9
                ),
                ComplementNB()
            )

            # Параметры для подбора
            param_dist = {
                'tfidfvectorizer__max_features': [15000, 20000],
                'tfidfvectorizer__ngram_range': [(1, 2), (1, 3)],
                'tfidfvectorizer__min_df': [2, 3, 5],
                'complementnb__alpha': np.logspace(-5, 0, 6),
                'complementnb__norm': [True, False],
            }

            # Поиск лучших параметров
            search = RandomizedSearchCV(
                pipeline,
                param_dist,
                n_iter=20,
                cv=5,
                n_jobs=self.n_jobs,
                verbose=0,
                random_state=self.random_state,
                scoring='f1_weighted'
            )

            print("\nНачало обучения модели...")
            search.fit(X_train, y_train)
            self.model = search.best_estimator_

            # Оценка модели
            self._evaluate_model(X_test, y_test)

            # Сохранение модели
            # self.save_model()

            return search.best_score_

        except Exception as e:
            print(f"\nОшибка при обучении: {str(e)}")
            raise

    def _evaluate_model(self, X_test, y_test):
        """Оценка качества модели"""
        y_pred = self.model.predict(X_test)
        y_test_decoded = self.label_encoder.inverse_transform(y_test)
        y_pred_decoded = self.label_encoder.inverse_transform(y_pred)

        # Confusion Matrix
        classes = self.label_encoder.classes_
        cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=classes)

        # plt.figure(figsize=(14, 10))
        # sns.heatmap(
        #     cm,
        #     annot=True,
        #     fmt='d',
        #     cmap='Blues',
        #     xticklabels=classes,
        #     yticklabels=classes,
        #     annot_kws={"size": 8}
        # )
        # plt.title('Confusion Matrix', pad=20)
        # plt.xlabel('Predicted', fontsize=12)
        # plt.ylabel('Actual', fontsize=12)
        # plt.xticks(rotation=45, ha='right')
        # plt.tight_layout()
        # plt.show()

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test_decoded, y_pred_decoded, digits=2, zero_division=0))

    def save_model(self, path_prefix=None):
        """Сохранение модели и кодировщика"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = path_prefix or self.name_model or "asset_classifier"

        model_filename = f"{prefix}_model_{timestamp}.joblib"
        encoder_filename = f"{prefix}_encoder_{timestamp}.joblib"

        joblib.dump(self.model, model_filename)
        joblib.dump(self.label_encoder, encoder_filename)

        print(f"\nМодель сохранена ПЕРВЫЙ: {model_filename}")
        print(f"Кодировщик меток сохранен: {encoder_filename}")

        return model_filename, encoder_filename

    @classmethod
    def load_model(cls, model_path, encoder_path):
        """Загрузка сохраненной модели"""
        classifier = cls()
        classifier.model = joblib.load(model_path)
        classifier.label_encoder = joblib.load(encoder_path)
        return classifier

    def predict(self, new_data, text_column='name', return_proba=False, output_file=None):
        """
        Предсказание классов для новых данных
        Args:
            new_data: DataFrame или список строк
            text_column: название колонки с текстом (если new_data - DataFrame)
            return_proba: возвращать вероятности классов
            output_file: путь для сохранения результатов (если указан)
        Returns:
            DataFrame с предсказаниями и исходными данными
        """
        try:
            # Создаем копию входных данных
            if isinstance(new_data, pd.DataFrame):
                result = new_data.copy()
                texts = result[text_column].astype(str)
            else:
                texts = pd.Series(new_data).astype(str)
                result = pd.DataFrame({text_column: texts})

            # Получаем предсказания
            if return_proba:
                # Получаем вероятности для всех классов
                proba = self.model.predict_proba(texts)
                proba_df = pd.DataFrame(proba, columns=[f'prob_{cls}' for cls in self.label_encoder.classes_])

                # Объединяем с исходными данными
                result = pd.concat([result, proba_df], axis=1)
            else:
                # Получаем предсказанные классы
                preds = self.model.predict(texts)
                result['predicted_group'] = self.label_encoder.inverse_transform(preds)

            # Сохранение в файл (если нужно)
            if output_file:
                if not output_file.endswith('.xlsx'):
                    output_file += '.xlsx'
                result.to_excel(output_file, index=False)
                print(f"Результаты сохранены в: {output_file}")

            return result

        except Exception as e:
            print(f"Ошибка при предсказании: {str(e)}")
            raise