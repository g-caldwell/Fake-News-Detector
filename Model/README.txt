To start, run clean.py on a data set that needs to be cleaned. This should output a cleaned version of the dataset into the Cleaned Data folder

model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=15)

Max Depth: controls the complexity of the model (lower number = shallow search, higher number = deeper search). if the number is too high the model can memorize the dataset
Min Samples Leaf: How many times something has to occur in the data set for the model to make it a rule (should be ~5% the total lines in the csv file)