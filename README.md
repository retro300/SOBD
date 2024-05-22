----------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import zscore, skew
import matplotlib.pyplot as plt
import seaborn as sns

Импорт библиотек:
pandas используется для работы с датафреймами.
numpy предоставляет поддержку для работы с массивами и функциями высокого уровня.
scipy.stats используется для статистических вычислений, таких как Z-оценка и коэффициент асимметрии.
matplotlib.pyplot и seaborn используются для визуализации данных.
----------------------------------


---------------------------------
# Загрузка датасета
df = pd.read_csv('the-reddit-covid-dataset-posts.csv')
print()

Загрузка датасета из CSV файла в DataFrame df.
-----------------------

-------------------------------------
# Определение категорий колонок
string_columns = ['type', 'id', 'subreddit.id', 'subreddit.name', 'permalink', 'domain', 'url', 'selftext', 'title']
numeric_with_zeroes_columns = ['score']
numeric_without_zeroes_columns = ['created_utc']
boolean_columns = ['subreddit.nsfw']
date_columns = []  # если у вас есть колонки с датами, добавьте их сюда


Разделение колонок на категории для последующего анализа:
string_columns — текстовые колонки.
numeric_with_zeroes_columns — числовые колонки, которые могут содержать нулевые значения.
numeric_without_zeroes_columns — числовые колонки, которые не должны содержать нулевые значения.
boolean_columns — логические колонки.
date_columns — колонки с датами (если есть).
---------------------------------------


------------------------------------
# Инициализация словаря для хранения количества пропущенных значений
missing_values = {}

Инициализация пустого словаря для хранения количества пропущенных значений для каждой колонки.
-------------------------------------


---------------------------------------
# Проверка на наличие пропущенных значений
for column in df.columns:
    if column in string_columns:
        missing_count = df[column].isnull().sum()
        missing_values[column] = missing_count
    if column in numeric_with_zeroes_columns:
        missing_count = df[column].isnull().sum() + (df[column] == 0).sum()
        missing_values[column] = missing_count
    if column in numeric_without_zeroes_columns:
        missing_count = df[column].isnull().sum() + (df[column] == 0).sum() + np.isnan(df[column]).sum()
        missing_values[column] = missing_count
    if column in boolean_columns:
        missing_count = df[column].isnull().sum()
        missing_values[column] = missing_count
    if column in date_columns:
        missing_count = df[column].isnull().sum()
        missing_values[column] = missing_count

Для каждой колонки в датафрейме:
Подсчет количества пропущенных значений для текстовых колонок.
Подсчет количества пропущенных и нулевых значений для числовых колонок с нулями.
Подсчет количества пропущенных, нулевых и NaN значений для числовых колонок без нулей.
Подсчет количества пропущенных значений для логических колонок.
Подсчет количества пропущенных значений для колонок с датами.
---------------------------------------

-------------------------
# Создание DataFrame из словаря с пропущенными значениями
missing_df = pd.DataFrame(list(missing_values.items()), columns=['Column', 'Missing Values'])
print(missing_df)
print()

Создание DataFrame missing_df из словаря missing_values и вывод его на экран.
---------------------------------

-----------------
# Удаление строк с пропущенными значениями в критически важных колонках
critical_columns = ['type', 'id', 'subreddit.id', 'subreddit.name', 'created_utc', 'title', 'score']
df_cleaned = df.dropna(subset=critical_columns)

Удаление строк, содержащих пропущенные значения в критически важных колонках.
--------------------------------



-----------------------------
# Удаление строк с аномальными нулевыми значениями в числовых колонках
for column in numeric_with_zeroes_columns + numeric_without_zeroes_columns:
    df_cleaned = df_cleaned[df_cleaned[column] != 0]
Удаление строк с нулевыми значениями в числовых колонках.



# Вывод количества строк после очистки
print(f'Количество строк после удаления пропущенных и аномальных нулевых значений: {df_cleaned.shape[0]}')
print()
Вывод количества строк после очистки от пропущенных и нулевых значений.
python
Копировать код
# Определение и удаление выбросов с использованием метода Z-оценки
def remove_outliers_zscore(df, column, threshold=3):
    # Вычисление Z-оценок
    z_scores = zscore(df[column].dropna())
    abs_z_scores = np.abs(z_scores)
    filtered_entries = abs_z_scores < threshold
    return df.loc[df[column].dropna().index[filtered_entries]]

# Применение функции к числовым колонкам
for column in numeric_with_zeroes_columns + numeric_without_zeroes_columns:
    df_cleaned = remove_outliers_zscore(df_cleaned, column)
Определение и удаление выбросов с использованием метода Z-оценки для каждой числовой колонки:
zscore(df[column].dropna()) вычисляет Z-оценку для значений колонки.
np.abs(z_scores) вычисляет абсолютные значения Z-оценок.
abs_z_scores < threshold определяет выбросы, сравнивая Z-оценки с порогом.
df.loc[df[column].dropna().index[filtered_entries]] возвращает очищенный датафрейм без выбросов.
python
Копировать код
# Вывод количества строк после удаления выбросов
print(f'Количество строк после удаления выбросов: {df_cleaned.shape[0]}')
print()
Вывод количества строк после удаления выбросов.
python
Копировать код
# Расчет статистических показателей

# Функция для расчета минимального, среднего и максимального значений
def calculate_min_mean_max(df, column):
    min_value = df[column].min()
    mean_value = df[column].mean()
    max_value = df[column].max()
    return min_value, mean_value, max_value

# Функция для расчета стандартного отклонения
def calculate_stddev(df, column):
    stddev_value = df[column].std()
    return stddev_value

# Функция для расчета квартилей
def calculate_quartiles(df, column):
    q1 = df[column].quantile(0.25)
    median = df[column].median()
    q3 = df[column].quantile(0.75)
    return q1, median, q3

# Функция для расчета коэффициента асимметрии
def calculate_skewness(df, column):
    skewness_value = skew(df[column].dropna())
    return skewness_value

# Расчет статистических показателей для каждой числовой колонки
for column in numeric_with_zeroes_columns + numeric_without_zeroes_columns:
    min_val, mean_val, max_val = calculate_min_mean_max(df_cleaned, column)
    stddev_val = calculate_stddev(df_cleaned, column)
    q1_val, median_val, q3_val = calculate_quartiles(df_cleaned, column)
    skewness_val = calculate_skewness(df_cleaned, column)
    
    print(f"Статистические показатели для колонки '{column}':")
    print(f"Минимальное значение: {min_val}")
    print(f"Среднее значение: {mean_val}")
    print(f"Максимальное значение: {max_val}")
    print(f"Стандартное отклонение: {stddev_val}")
    print(f"Квартиль Q1: {q1_val}")
    print(f"Медиана: {median_val}")
    print(f"Квартиль Q3: {q3_val}")
    print(f"Коэффициент асимметрии: {skewness_val}")
    print()
Расчет и вывод статистических показателей для каждой числовой колонки:
calculate_min_mean_max вычисляет минимальное, среднее и максимальное значения.
calculate_stddev вычисляет стандартное отклонение.
calculate_quartiles вычисляет квартиль Q1, медиану и квартиль Q3.
calculate_skewness вычисляет коэффициент асимметрии.
python
Копировать код
# Сохранение очищенного датасета в новый файл
df_cleaned.to_csv('cleaned-reddit-covid-dataset-posts.csv', index=False)
print()
Сохранение очищенного датафрейма в новый CSV файл.
python
Копировать код
# Визуализация

# Загрузка очищенного датасета
df_cleaned = pd.read_csv('cleaned-reddit-covid-dataset-posts.csv', low_memory=False)

# Сэмплирование датасета
sampled_df = df_cleaned.sample(frac=0.6, random_state=42)

# Определение числовых колонок
numeric_columns = ['created_utc', 'score']

# Функция для построения гистограммы
def plot_histogram(df, column):
    data = df[column].dropna()
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins='auto', color='blue')
    plt.title(f'Гистограмма распределения {column}')
    plt.xlabel(column)
    plt.ylabel('Частота')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Визуализация гистограмм для числовых колонок
for column in numeric_columns:
    plot_histogram(sampled_df, column)
Визуализация распределения данных с использованием гистограмм:
plot_histogram строит гистограмму для указанной колонки.
python
Копировать код
# Дополнительные визуализации
# Ящик с усами (Boxplot)
def plot_boxplot(df, column):
    data = df[column].dropna()
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, vert=False)
    plt.title(f'Ящик с усами для {column}')
    plt.xlabel(column)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()

# Диаграмма рассеяния (Scatter plot)
def plot_scatter(df, x_column, y_column):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_column], df[y_column], alpha=0.5)
    plt.title(f'Диаграмма рассеяния {x_column} vs {y_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

# Визуализация ящика с усами для числовых колонок
for column in numeric_columns:
    plot_boxplot(sampled_df, column)

# Визуализация диаграммы рассеяния для пары колонок
plot_scatter(sampled_df, 'created_utc', 'score')
Дополнительные визуализации:
plot_boxplot строит ящик с усами для указанной колонки.
plot_scatter строит диаграмму рассеяния для пары колонок.
python
Копировать код
# Построение матрицы корреляции
corr_matrix = sampled_df[numeric_columns].corr()

# Визуализация матрицы корреляции с помощью тепловой карты
plt.figure(figsize=(16, 5))
sns.heatmap(
    corr_matrix,
    xticklabels=corr_matrix.columns.values,
    yticklabels=corr_matrix.columns.values,
    cmap='Blues',
    annot=True
)
plt.title('Матрица корреляций')
plt.show()
Построение и визуализация матрицы корреляции:
corr_matrix вычисляет матрицу корреляции для числовых колонок.
sns.heatmap визуализирует матрицу корреляции с помощью тепловой карты.
Завершение
Этот код выполняет очистку данных, удаление пропущенных значений и выбросов, расчет статистических показателей и визуализацию данных. В итоге создается очищенный и визуализированный датасет, который можно использовать для дальнейшего анализа.
---------------------------
