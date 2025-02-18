import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import display
import graphviz

df = pd.read_csv("/Users/sangeetha/Downloads/car_data.csv")
print(df.head())

X = df[['Age', 'AnnualSalary']]  # Select two features
y = df['Purchased']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Export the first three decision trees from the forest

# Plot the first tree in the random forest
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=800)
plot_tree(rf.estimators_[0],  # Access the first tree
          feature_names=X.columns,  # Use column names for features
          class_names=['Not Purchased', 'Purchased'],  # Define target classes
          filled=True)
plt.title("Random Forest Visualization of Age & Annual Salary Used to Predict Whether a Customer Purchases a Car")

# Save the plot as a PNG file
fig.savefig('rf_individualtree.png')

# Display inline (only for Jupyter or similar environments)
plt.show()

'''
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)

    # Render the graph and save as PNG
    graph = graphviz.Source(dot_data)
    graph.render(filename=f'tree_{i}', format='png', cleanup=False)
    display(Image(filename=f'tree_{i}.png'))

'''