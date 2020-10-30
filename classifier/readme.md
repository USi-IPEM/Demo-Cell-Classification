# Demo cell classification

Run `./classifier/data_loader.py` to see a visualization of all 123 input data samples.
To train as SVM run `./classifier/train_test_classifiers.py` we have 46.0 negative
measurements within the 123 samples. The classifier must therefore 
recognize more than (1. - 46./123. ) * 100. ~~ 63% of the cases.
Which is the case.
