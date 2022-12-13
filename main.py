import numpy as np

from catboost import CatBoostClassifier, Pool

train_data = np.random.randint(0,
                               100,
                               size=(100, 10))

train_labels = np.random.randint(0,
                                 2,
                                 size=(100))

print(train_data)

print(train_labels)

test_data = catboost_pool = Pool(train_data,
                                 train_labels)
print('------test data--------')
print(test_data.get_features())
print(test_data.get_label())

model_2iter = CatBoostClassifier(iterations=2,
                                 depth=2,
                                 learning_rate=1,
                                 loss_function='Logloss',
                                 verbose=True)
# train the model
model_2iter.fit(train_data, train_labels)
# make the prediction using the resulting model
preds_class = model_2iter.predict(test_data)
preds_proba = model_2iter.predict_proba(test_data)
print("class = ", preds_class)
print("proba = ", preds_proba)

print('---------Second model--------')

model_100iter = CatBoostClassifier(iterations=100,
                                   depth=2,
                                   learning_rate=1,
                                   loss_function='Logloss',
                                   verbose=True)

# train the model
model_100iter.fit(train_data, train_labels)
# make the prediction using the resulting model
preds_class = model_100iter.predict(test_data)
preds_proba = model_100iter.predict_proba(test_data)
print("class = ", preds_class)
print("proba = ", preds_proba)