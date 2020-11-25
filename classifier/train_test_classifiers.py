# Train an svm classifier on the demo cell data.
import os
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from data_loader import VectorLoader

path_lst = ['./01_Data/201027/use_case2/Processed/Samples/',
            './01_Data/201027/use_case1/Processed/Samples/']
            # './01_Data/201026/use_case1/Processed/Samples/',
            # './01_Data/201030/use_case1/Processed/Samples/',
            # './01_Data/201030/use_case2/Processed/Samples/',
            # './01_Data/201030/use_case3/Processed/Samples/',
            # './01_Data/201030/use_case4/Processed/Samples/',
            # './01_Data/201030/use_case5/Processed/Samples/']


# os.chdir(os.path.dirname(__file__))
# print(os.getcwd())

demo_cell_data = VectorLoader(case_path_lst=path_lst, debug=False)
X_train, y_train = demo_cell_data.get_train_xy()
X_test, y_test = demo_cell_data.get_test_xy()
### SVM --------------------------------------------------------------###
svm = svm.SVC(gamma='auto')
svm.fit(X_train, y_train.ravel())
svm_out = svm.predict(X_test)
## compute accuracy
y_test = np.squeeze(y_test)
print('sample svm_out    ', svm_out.astype(np.uint8))
print('sample y test     ', y_test.astype(np.uint8))
print('svm_out == y_test ', (svm_out == y_test).astype(np.uint8))
print('SVM accuracy:',
      np.sum((svm_out == y_test).astype(np.float32))/svm_out.shape[0]*100)

### MLP ---------------------------------------------------------------###
mlp = MLPClassifier(max_iter=2000)
mlp.fit(X_train, y_train.ravel())
mlp_out = mlp.predict(X_test)
## compute accuracy
y_test = np.squeeze(y_test)
print('sample mlp_out    ', mlp_out.astype(np.uint8))
print('sample y test     ', y_test.astype(np.uint8))
print('mlp_out == y_test ', (mlp_out == y_test).astype(np.uint8))
print('MLP accuracy:',
      np.sum((mlp_out == y_test).astype(np.float32))/mlp_out.shape[0]*100)

### Tree --------------------------------------------------------------###
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train.ravel())
tree_out = tree.predict(X_test)
## compute accuracy
y_test = np.squeeze(y_test)
print('sample tree_out   ', tree_out.astype(np.uint8))
print('sample y test     ', y_test.astype(np.uint8))
print('tree_out == y_test', (tree_out == y_test).astype(np.uint8))
print('tree accuracy:',
      np.sum((tree_out == y_test).astype(np.float32))/tree_out.shape[0]*100)