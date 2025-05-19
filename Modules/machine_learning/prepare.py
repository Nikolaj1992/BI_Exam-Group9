from sklearn.model_selection import train_test_split

def prepare_data(df, target_column, random_state=123, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, random_state=random_state, test_size=test_size)