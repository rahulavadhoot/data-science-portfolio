# SEDS-HW4
Homework #4 from SEDS


### Due Feb. 15, 2017 at 5PM

_note_ This is almost certainly going to be the most challenging homework so far.  It will build on everything you have learned from flow control, functions, lists, etc.

### Assignment
1. Create a `.py` file called `knn.py` that contains your own implementation of a k-NN classifier.  _Hint_ You should have the following functions (at least): (2pts)
 * A wrapping function that is the primary way of interacting with your code.  It takes as parameters, a training dataframe, a value of k and some input data to be classified. It returns the classification for the input data.
 * A function that returns the Euclidean distance between a row in the intput data to be classified.
 * A function that returns the list of sorted Euclidean distances between the input data and all rows in the dataframe. _Hint_ Append the distances associated with the rows to a list and use the `.sort()` method on your list.
 * A function that returns the class prediction based on the list of sorted Euclidean distances.
 * A wrapping function that helps the user decide on what `k` to use.  This function takes as parameters, a training dataframe, a testing dataframe and a list of values of `k` to try. It returns a dictionary with `k` as the keys and the training accuracy of the test set.  Accuracy is measured by percentage of classifications that were correct for that value of `k`.
2. Create a new Jupyter notebook called 'SEDS-HW4.ipynb' that documents how to use your k-NN functions in `knn.py` with an example.  Use the [`atomradii.csv`](https://uwdirect.github.io/SEDS_content/atomradii.csv) and [`testing.csv`](https://uwdirect.github.io/SEDS_content/testing.csv) that DSMCER used in for the inclass demo that relates atomic radii to atomic class.  Leverage Markdown for your demo. (1pt)
3. Create unit tests and put them in `test_knn.py`.  There should be at least one unit test per function, though many more are appropriate for a real implementation.  Again, use the `atomradii` data for the unit tests. Paste the output of running nosetests below. (2pt)

### Results of unit tests
```
$ nosetests
.....
----------------------------------------------------------------------
Ran 5 tests in 0.512s

OK
```

### Grading rubric
5/5 - really nice work.