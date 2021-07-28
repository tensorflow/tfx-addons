from sklearn.feature_selection import SelectKBest as SelectorFunc
from sklearn.feature_selection import chi2 as ScoreFunc
from sklearn.datasets import load_iris

iris_data = load_iris()
_FEATURE_KEYS = load_iris.feature_names
_INPUT_DATA = iris_data.data
_TARGET_DATA = iris_data.target

_NUM_PARAM = 10
