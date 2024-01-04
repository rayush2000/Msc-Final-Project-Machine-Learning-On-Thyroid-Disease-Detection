from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', n_jobs=1)
X_train_sampled, y_train_sampled = smote.fit_resample(X_train, y_train)
