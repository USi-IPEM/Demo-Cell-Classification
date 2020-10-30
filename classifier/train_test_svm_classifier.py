# Train an svm classifier on the demo cell data.
import os
import numpy as np
from sklearn import svm
from data_loader import VectorLoader


path_lst = ['./01_Data/201027/use_case2/Processed/Samples/',
            './01_Data/201027/use_case1/Processed/Samples/',
            './01_Data/201030/use_case1/Processed/Samples/',
            './01_Data/201030/use_case2/Processed/Samples/',
            './01_Data/201030/use_case3/Processed/Samples/',
            './01_Data/201030/use_case4/Processed/Samples/',
            './01_Data/201030/use_case5/Processed/Samples/',
            './01_Data/201030/use_case6/Processed/Samples/']


# os.chdir(os.path.dirname(__file__))
# print(os.getcwd())

demo_cell_data = VectorLoader(case_path_lst=path_lst, debug=False)
X_train, y_train = demo_cell_data.get_train_xy()

clf = svm.SVC(gamma='auto')
clf.fit(X_train, y_train.ravel())

X_test, y_test = demo_cell_data.get_test_xy()

svm_out = clf.predict(X_test)
## compute accuracy
y_test = np.squeeze(y_test)
print('svm_out', svm_out)
print('y test', y_test)
print('svm_out == y_test', svm_out == y_test)
print('accuracy:', np.sum((svm_out == y_test).astype(np.float32))/svm_out.shape[0]*100)