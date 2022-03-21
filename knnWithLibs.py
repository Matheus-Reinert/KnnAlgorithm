#import plotly.express as px
#df = px.data.iris()
#fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
#              color='species')
#fig.show()

from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X,y = load_iris(return_X_y=True) #Rótulos e features
X_train, X_test, y_train, y_test = train_teste = train_test_split(X,y, test_size=0.30, random_state=13)

n_neighbors = 10
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
clf.fit(X_train, y_train) #Roda o modelo no conjunto de treinamento
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=iris.target_names))
