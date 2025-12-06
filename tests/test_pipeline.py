import pandas as pd
from src.preprocess import load_data, train_val_split
def test_load_and_split(tmp_path):
    df = pd.DataFrame({
        'V1':[0.1,0.2,0.3,0.4],
        'V2':[1,2,3,4],
        'Amount':[10,20,30,40],
        'Class':[0,0,1,0]
    })
    p = tmp_path / 'sample.csv'
    df.to_csv(p, index=False)
    X,y = load_data(str(p))
    X_train, X_val, y_train, y_val = train_val_split(X,y,test_size=0.5, random_state=1)
    assert len(X_train) + len(X_val) == 4
