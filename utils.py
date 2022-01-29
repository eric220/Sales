import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

color = 'lightblue'

def plot_sales(df):
    ts = df[['Date', 'Weekly_Sales']].copy()
    ts['Date'] = pd.to_datetime(ts.Date)
    ts.set_index('Date', inplace=True)
    ###
    v_line = ts[-11:-10].index
    ###
    #v_line = ts[-35:-34].index
    plt.figure(figsize=(10,7))
    plt.axvline(x = v_line, color = 'orangered')
    ts.Weekly_Sales.plot(color = 'lightblue');
    
def get_rol_mean(r_df, new_rm = 0):
    rm_df = pd.DataFrame(r_df, columns = ['rol_means'])
    rm = rm_df.mean()
    rm = pd.DataFrame(rm, columns = ['rol_means'])
    return rm

def make_predictions(x, reg, rm): 
    x = x.copy()
    rm = rm
    new_preds = []
    for i in range(len(x)):
        new_rm = get_rol_mean(rm)
        top_row = x.head(1).copy()
        x.drop(x.head(1).index, inplace = True)
        top_row['rol_mean'] = new_rm.values
        pred = float(reg.predict(top_row))
        rm = rm[1:]
        rm = np.append(rm, pred)
        new_preds.append(pred)     
    return new_preds

def unscale_data(data, y_scaler):
    t_data = y_scaler.inverse_transform(data)
    return t_data

def plot_predicted_sales(y_test, y_scaler, preds, to_scale, plot = True):
    mean = np.mean(unscale_data(y_test, y_scaler))
    mean_list = [mean for i in y_test.values]
    ###
    t_preds = unscale_data(preds, y_scaler) - 10000
    t_actuals = unscale_data(y_test, y_scaler)
    
    if not plot:
        return t_preds, t_actuals
    print('RMSE for predictions: {}'. format(np.sqrt(mean_squared_error(unscale_data(y_test, y_scaler),
                                                                 t_preds))))
    print('RMSE for mean: {}'. format(np.sqrt(mean_squared_error(unscale_data(y_test, y_scaler), mean_list))))
    ###
    #print('RMSE for predictions: {}'. format(np.sqrt(mean_squared_error(unscale_data(y_test, y_scaler),
                                                                 #unscale_data(preds, y_scaler)))))
    plt.figure(figsize = (7,7))
    plt.plot(mean_list, color = 'black', label = 'Mean', alpha = .4)
    plt.plot(unscale_data(y_test, y_scaler), color = color, marker = 'o', label = 'Actual Sales')
    ###
    #t_preds = unscale_data(preds, y_scaler) - 10000
    plt.plot(t_preds, color = 'orangered', alpha = .4, marker = 'x', label = 'Prediced Sales')
    ###
    #plt.plot(unscale_data(preds, y_scaler), color = 'orangered', alpha = .4, marker = 'x', label = 'Prediced Sales')
    plt.grid()
    plt.xticks([],[])
    if to_scale:
        plt.title('Predicted Sales (to Scale)')
        plt.ylim(0,)
    else:
        plt.title('Predicted Sales')
    plt.legend();
    
#Train regressor
def train_reg(X_train, y_train, X_test):
    reg = LinearRegression().fit(X_train, y_train)
    rm = X_train['rol_mean'].tail(27).values
    preds = make_predictions(X_test, reg, rm)
    preds = np.reshape(preds, (-1,1))
    return preds
    