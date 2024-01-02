from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
class EnsembleClassifier:
    def __init__(self):
        self.clf = MultinomialNB(alpha=0.02)
        self.sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss='modified_huber')
        self.p6 = {'n_iter': 1500,
                   'verbose': -1,
                   'objective': 'binary',
                   'metric': 'auc',
                   'learning_rate': 0.05073909898961407,
                   'colsample_bytree': 0.726023996436955,
                   'colsample_bynode': 0.5803681307354022,
                   'lambda_l1': 8.562963348932286,
                   'lambda_l2': 4.893256185259296,
                   'min_child_samples': 115,
                   'max_depth': 23,
                   'max_bin': 898}
        self.lgb = LGBMClassifier(**self.p6)
        self.cat = CatBoostClassifier(iterations=1000,
                                      verbose=0,
                                      l2_leaf_reg=6.6591278779517808,
                                      learning_rate=0.005689066836106983,
                                      allow_const_label=True,
                                      loss_function='CrossEntropy')
        self.weights = [0.07, 0.31, 0.31, 0.31]
        self.ensemble = VotingClassifier(estimators=[
                                            ('mnb', self.clf),
                                            ('sgd', self.sgd_model),
                                            ('lgb', self.lgb),
                                            ('cat', self.cat)],
                                         weights=self.weights,
                                         voting='soft',
                                         n_jobs=-1)

    def fit(self, X_train, y_train):
        self.ensemble.fit(X_train, y_train)

    def predict(self, X_test):
        return self.ensemble.predict(X_test)