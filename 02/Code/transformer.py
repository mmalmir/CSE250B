class Transformer:
    def __init__(self,**kwrds):
        self.set_params(**kwrds)

    def fit_transform(self,X,y=None,**kwrds):
        self.fit(X,y,**kwrds)
        return self.transform(X)

    def fit(self,X,y=None,**kwrds):
        self.set_params(**kwrds)

    def set_params(self,**kwrds):
        pass
    
    def get_params(self,deep=False):
        return dict()
    
    def transform(self,X):
        return None
