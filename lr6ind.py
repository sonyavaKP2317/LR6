# БЛОК 1: ИМПОРТ БИБЛИОТЕК
import pandas as pd
import numpy as np
from yellowbrick.cluster import KElbowVisualizer   
from sklearn.cluster import KMeans                  
import matplotlib.pyplot as plt                    
from collections import Counter                     
import seaborn as sns                             
from mpl_toolkits.mplot3d import Axes3D            
from matplotlib.lines import Line2D

# БЛОК 2: ГЕНЕРАЦИЯ ДАННЫХ ПО ВАРИАНТУ 3
np.random.seed(0)
n_samples = 200
param1 = np.random.uniform(0, 1, n_samples)
param2 = np.random.uniform(-2, 2, n_samples)
param3_str = np.random.choice(['да', 'нет'], n_samples)
param3_num = np.where(param3_str == 'да', 1, 0)

M = np.column_stack((param1, param2, param3_num))
df_num = pd.DataFrame(M, columns=['var1', 'var2', 'var3'])

df_display = pd.DataFrame({'var1': param1, 'var2': param2, 'var3': param3_str})
print("Первые 7 строк:")
print(df_display.head(7))

# БЛОК 3: ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОГО ЧИСЛА КЛАСТЕРОВ (МЕТОД ЛОКТЯ)
model = KMeans(n_clusters=4, n_init=10, random_state=0)
visualizer = KElbowVisualizer(model, k=(2, 12), force_model=True)
visualizer.fit(df_num)
visualizer.show(outpath="ind_elbow.png")   
optimal_k = visualizer.elbow_value_
print(f"\nОптимальное число кластеров: {optimal_k}")

# БЛОК 4: ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ KMEANS
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=0).fit(df_num)
print("\nМетки кластеров (первые 20):", kmeans.labels_[:20])                 
print("\nКоординаты центроидов:\n", kmeans.cluster_centers_)            
print(f"\nВнутрикластерная сумма квадратов = {kmeans.inertia_}")
print(f"Количество итераций = {kmeans.n_iter_}")

# БЛОК 5: ПОДСЧЁТ РАЗМЕРОВ КЛАСТЕРОВ
print("\nРазмер каждого кластера:")
for cid, size in sorted(Counter(kmeans.labels_).items()):
    print(f"Кластер {cid}: {size} точек")

# БЛОК 6: ВИЗУАЛИЗАЦИЯ 
# Генерируем фиксированный список цветов из палитры 'tab10' для каждого кластера
colors = plt.cm.tab10(np.linspace(0, 1, optimal_k))
# Словарь для сопоставления номера кластера и цвета (для явного контроля)
color_dict = {i: colors[i] for i in range(optimal_k)}

# Вспомогательная функция для легенды (общая для всех графиков)
def get_legend_elements(colors, optimal_k):
    """Создаёт элементы легенды: кружки кластеров + звезда центроидов"""
    elements = [Line2D([0], [0], marker='o', color='w', label=f'Кластер {i}',
                       markerfacecolor=colors[i], markersize=8, markeredgecolor='k')
                for i in range(optimal_k)]
    elements.append(Line2D([0], [0], marker='*', color='w', label='Центроиды (цвет кластера)',
                           markerfacecolor='gray', markersize=12, markeredgecolor='black'))
    return elements

legend_elements = get_legend_elements(colors, optimal_k)

# График 1: var1 vs var2
plt.figure(figsize=(9, 6))
# Рисуем точки, явно указывая цвета через palette=color_dict
sns.scatterplot(data=df_num, x='var1', y='var2', hue=kmeans.labels_, 
                palette=color_dict, s=60, alpha=0.8, edgecolor='k', linewidth=0.5, legend=False)
# Рисуем центроиды цветными звёздами
for i in range(optimal_k):
    plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1],
                marker='*', c=[colors[i]], s=250, edgecolors='black', linewidth=1.5)
plt.legend(handles=legend_elements, title='Обозначения', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Кластеры по var1 и var2', fontsize=12)
plt.xlabel('var1 [0,1]')
plt.ylabel('var2 [-2,2]')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('ind_plot1.png', dpi=150, bbox_inches='tight')
plt.close()

# График 2: var2 vs var3
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df_num, x='var2', y='var3', hue=kmeans.labels_, 
                palette=color_dict, s=60, alpha=0.8, edgecolor='k', linewidth=0.5, legend=False)
for i in range(optimal_k):
    plt.scatter(kmeans.cluster_centers_[i, 1], kmeans.cluster_centers_[i, 2],
                marker='*', c=[colors[i]], s=250, edgecolors='black', linewidth=1.5)
plt.legend(handles=legend_elements, title='Обозначения', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Кластеры: var2 vs var3 (0=нет, 1=да)', fontsize=12)
plt.xlabel('var2 [-2,2]')
plt.ylabel('var3 (0=нет, 1=да)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('ind_plot2.png', dpi=150, bbox_inches='tight')
plt.close()

# График 3: var1 vs var3
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df_num, x='var1', y='var3', hue=kmeans.labels_, 
                palette=color_dict, s=60, alpha=0.8, edgecolor='k', linewidth=0.5, legend=False)
for i in range(optimal_k):
    plt.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 2],
                marker='*', c=[colors[i]], s=250, edgecolors='black', linewidth=1.5)
plt.legend(handles=legend_elements, title='Обозначения', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Кластеры: var1 vs var3', fontsize=12)
plt.xlabel('var1 [0,1]')
plt.ylabel('var3 (0=нет, 1=да)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('ind_plot3.png', dpi=150, bbox_inches='tight')
plt.close()

# График 4: 3D 
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
# Точки – задаём цвет каждой точки через список цветов, чтобы совпадало с color_dict
point_colors = [colors[label] for label in kmeans.labels_]
ax.scatter(df_num['var1'], df_num['var2'], df_num['var3'],
           c=point_colors, s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
# Центроиды – цветные звёзды
for i in range(optimal_k):
    ax.scatter(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], kmeans.cluster_centers_[i, 2],
               marker='*', c=[colors[i]], s=300, edgecolors='black', linewidth=1.5)
ax.set_xlabel('var1 [0,1]', fontsize=10)
ax.set_ylabel('var2 [-2,2]', fontsize=10)
ax.set_zlabel('var3 (0=нет,1=да)', fontsize=10)
ax.set_title('3D визуализация кластеров', fontsize=12)
ax.view_init(elev=25, azim=45)
ax.grid(True, linestyle='--', alpha=0.5)

# Легенда для 3D 
ax.legend(handles=legend_elements, title='Обозначения', loc='upper left', bbox_to_anchor=(-0.05, 1.02))
plt.tight_layout()
plt.savefig('ind_plot4.png', dpi=150, bbox_inches='tight')
plt.close()