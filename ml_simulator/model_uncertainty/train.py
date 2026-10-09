import pandas as pd
from utils import bias
from utils import GroupTimeSeriesSplit, best_model
from utils import mape
from utils import smape
from utils import wape


def main():
    # Data loading?
    df_path = "./datasets/data_train_sql.csv"

    df = pd.read_csv(df_path, parse_dates=["monday"])
    y = df.pop("y")

    # monday or product_name as a groups for validation?
    df.drop('product_name', axis=1, inplace=True)
    groups = df.pop('monday').values

    X = df

    # Validation loop
    cv = GroupTimeSeriesSplit(
        n_splits=5,
        max_train_size=None,
        test_size=None,
        gap=0,
    )

    for train_idx, test_idx in cv.split(X, y, groups):
        # Split train/test
        X_train = X.loc[train_idx, :]
        y_train = y[train_idx]
        X_test = X.loc[test_idx, :]
        y_test = y[test_idx]

        # Fit model
        model = best_model()
        model.fit(X_train, y_train)
        # Predict and print metrics
        y_pred = model.predict(X_test)
        
        output = ''
        for metric, name in zip([mape, smape, wape, bias],
                                 ['mape', 'smape', 'wape', 'bias']
                               ):
            output += f'{name}: {metric(y_test, y_pred):.2f} '
        print(output[:-1])


if __name__ == "__main__":
    main()
