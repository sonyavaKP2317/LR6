# БЛОК 0: ИСХОДНЫЙ ГРАФИК
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4,] [1, 4, 2, 5])
# plt. ylabel('some numbers')
# plt.savefig('chartlr6.png')

# БЛОК 1: ИМПОРТ БИБЛИОТЕК
from sklearn.datasets import make_blobs 
import pandas as pd

# БЛОК 2: ГЕНЕРАЦИЯ ДАННЫХ
# Генерация 200 точек с 2 признаками, 4 центрами, разброс 0.5
dataset, classes = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=0)
# Преобразование в DataFrame для удобства
df = pd.DataFrame(dataset, columns=['var1', 'var2'])
# Вывод первых двух строк для ознакомления
print(df.head(2))

# БЛОК 3: МЕТОД ЛОКТЯ (ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО k)
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
# Создание модели с 4 кластерами
model = KMeans(n_clusters=4, n_init=10, random_state=0)
# Визуализатор для диапазона k=2..12, строит график WCSS
visualizer = KElbowVisualizer(model, k=(2, 12), force_model=True)
# Обучение визуализатора на данных
visualizer.fit(df)
visualizer.show(outpath="elbow.png")

# БЛОК 4: ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ KMEANS
import matplotlib. pyplot as plt
plt.figure()
# Обучение модели с улучшенной инициализацией k-means++
kmeans=KMeans (n_clusters=4, init='k-means++' , random_state=0). fit(df)
print (kmeans.labels_)
print (kmeans.cluster_centers_)
print (kmeans.inertia_)
print(kmeans.n_iter_)

# БЛОК 5: ПОДСЧЁТ РАЗМЕРА КАЖДОГО КЛАСТЕРА
from collections import Counter
# Подсчёт количества точек в каждом кластере
Counter (kmeans. labels_)
print(Counter(kmeans. labels_))

# БЛОК 6: ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ И СОХРАНЕНИЕ ГРАФИКОВ
import seaborn as sns
# Диаграмма рассеяния с цветовой маркировкой по меткам кластеров
sns.scatterplot (data=df, x="var1", y="var2", hue=kmeans.labels_)
plt.savefig('clusters.png')
# Нанесение центроидов красными крестами
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="X", c="r", s=80, label='Centroids')
# Добавление легенды
plt.legend()
plt.savefig('clusters_with_centroids.png')