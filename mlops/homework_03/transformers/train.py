from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features

from sklearn.linear_model import LinearRegression

@transformer
def transform(data, *args, **kwargs):

    df, df_train, df_val = data
    target = kwargs.get('target', 'duration')

    X, _, dv = vectorize_features(select_features(df))
    y: Series = df[target]

    # X_train, X_val, dv = vectorize_features(
    #     select_features(df_train),
    #     select_features(df_val),
    # )
    # y_train = df_train[target]
    # y_val = df_val[target]

    lr = LinearRegression()
    lr.fit(X, y)

    print(lr.intercept_)

    return dv, lr
