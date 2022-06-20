from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import wrangle
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


def recursive_feature_elim(x_train_scaled, y_train):
    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=2)

    # fit the data using RFE
    rfe.fit(x_train_scaled, y_train)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = x_train_scaled.iloc[:,feature_mask].columns.tolist()

def create_and_fit_rfe(x_train_scaled, y_train,
                       features_to_select =2):
    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=features_to_select)
    
    # fit rfe 
    rfe.fit(x_train_scaled, y_train)

    return rfe

def rfe_ranking(rfe, x_train_scaled):
        # view list of columns and their ranking

        # get the ranks
        var_ranks = rfe.ranking_
        # get the variable names
        var_names = x_train_scaled.columns.tolist()
        # combine ranks and names into a df for clean viewing
        rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})

        return rfe_ranks_df

def y_mean_median_base_pred(y_train, y_validate, target_var):
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    # TVDC -> taxvaluedollarcnt
    TVDC_pred_mean = y_train[target_var].mean()
    y_train['TVDC_pred_mean'] = TVDC_pred_mean
    y_validate['TVDC_pred_mean'] = TVDC_pred_mean

    TVDC_pred_median = y_train[target_var].median()
    y_train['TVDC_pred_median'] = TVDC_pred_median
    y_validate['TVDC_pred_median'] = TVDC_pred_median

    return y_train, y_validate

def base_RMSE(y_train, y_validate):
    
    # rmse of baseline mean
    rmse_train_mean_bl = mean_squared_error(y_train.taxvaluedollarcnt, y_train.TVDC_pred_mean)**(1/2)
    rmse_val_mean_bl = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.TVDC_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train_mean_bl, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_val_mean_bl, 2))

    # rmse of baseline median
    rmse_train_med_bl = mean_squared_error(y_train.taxvaluedollarcnt, y_train.TVDC_pred_median)**(1/2)
    rmse_val_med_bl = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.TVDC_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train_med_bl, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_val_med_bl, 2))

    return rmse_train_mean_bl, rmse_val_mean_bl,\
           rmse_train_med_bl, rmse_val_med_bl


class LRM:

    def __init__(self, df, target_var, features_to_select=2):

        self.tweedies = pd.DataFrame()
        self.OLS = pd.DataFrame()
        self.lassolars = pd.DataFrame()
        
        self.target_var = target_var

        self.train, self.validate, self.test, \
        self.x_train_scaled, self.y_train, \
        self.x_validate_scaled, self.y_validate, \
        self.x_test_scaled, self.y_test = wrangle.all_train_validate_test_data(df, target_var)
        
        self.rfe = create_and_fit_rfe(self.x_train_scaled,
                                      self.y_train,
                                      features_to_select)

        self.rfe_features = self.x_train_scaled.iloc[:,self.rfe.support_].columns.tolist()
        self.rfe_ranks = rfe_ranking(self.rfe, self.x_train_scaled)
        self.rfe_feat_count = len(self.rfe_features)
        
        self.y_train, self.y_validate = y_mean_median_base_pred(self.y_train,
                                                                self.y_validate,
                                                                target_var)
        self.rmse_train_mean_bl,\
        self.rmse_val_mean_bl, \
        self.rmse_train_med_bl,\
        self.rmse_val_med_bl = base_RMSE(self.y_train, self.y_validate)


    def OLS_regression(self, use_rfe_features=False):
        model_name = 'OLS'
        
        if use_rfe_features:
            rfe_features = self.rfe_features
            x_train = self.x_train_scaled[rfe_features]
            x_validate = self.x_validate_scaled[rfe_features]

        else:
            x_train = self.x_train_scaled
            x_validate = self.x_validate_scaled

        print(x_train)

        # create the model object
        lm = LinearRegression(normalize=True)

        # fit the model to our training data. We must specify the column in self.y_train, 
        # since we have converted it to a dataframe from a series! 
        lm.fit(x_train, self.y_train.taxvaluedollarcnt)

        # predict train
        self.y_train['TVDC_pred_OLS'] = lm.predict(x_train)

        # evaluate: rmse
        rmse_train = mean_squared_error(self.y_train.taxvaluedollarcnt, self.y_train.TVDC_pred_OLS)**(1/2)

        # predict validate
        self.y_validate['TVDC_pred_OLS'] = lm.predict(x_validate)

        # evaluate: rmse
        rmse_validate = mean_squared_error(self.y_validate.taxvaluedollarcnt, self.y_validate.TVDC_pred_OLS)**(1/2)


        print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
        "\nValidation/Out-of-Sample: ", rmse_validate)

        model_stats = self.make_model_stats(model_name, rmse_train, rmse_validate,
                              x_train)
        self.OLS = pd.concat([self.OLS, model_stats], ignore_index=True)

        return model_stats

    def lassolars_regression(self, alpha=1, use_rfe_features=False):
        model_name = 'lasso_lars'
        rfe_feat = self.rfe_features
        if use_rfe_features:
            x_train = self.x_train_scaled[rfe_feat]
            x_validate = self.x_validate_scaled[rfe_feat]

        else:
            x_train = self.x_train_scaled
            x_validate = self.x_validate_scaled

        # create the model object
        lars = LassoLars(alpha=alpha)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lars.fit(x_train, self.y_train.taxvaluedollarcnt)

        # predict train
        self.y_train['TVDC_pred_lars'] = lars.predict(x_train)

        # evaluate: rmse
        rmse_train = mean_squared_error(self.y_train.taxvaluedollarcnt, self.y_train.TVDC_pred_lars)**(1/2)

        # predict validate
        self.y_validate['TVDC_pred_lars'] = lars.predict(x_validate)

        # evaluate: rmse
        rmse_validate = mean_squared_error(self.y_validate.taxvaluedollarcnt, self.y_validate.TVDC_pred_lars)**(1/2)

        print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
            "\nValidation/Out-of-Sample: ", rmse_validate)


        model_stats = self.make_model_stats(model_name, rmse_train, rmse_validate,
                              x_train, alpha=alpha)
        self.lassolars = pd.concat([self.lassolars, model_stats], ignore_index=True)

        return model_stats
    
    
    def tweedie(self, power=1, alpha=0, use_rfe_features=False):
        model_name = 'tweedie'
        rfe_feat = self.rfe_features
        if use_rfe_features:
            x_train = self.x_train_scaled[rfe_feat]
            x_validate = self.x_validate_scaled[rfe_feat]

        else:
            x_train = self.x_train_scaled
            x_validate = self.x_validate_scaled
            # create the model object
        glm = TweedieRegressor(power=1, alpha=0)

        # fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        glm.fit(x_train, self.y_train.taxvaluedollarcnt)

        # predict train
        self.y_train['TVDC_pred_glm'] = glm.predict(x_train)

        # evaluate: rmse
        rmse_train = mean_squared_error(self.y_train.taxvaluedollarcnt, self.y_train.TVDC_pred_glm)**(1/2)

        # predict validate
        self.y_validate['TVDC_pred_glm'] = glm.predict(x_validate)

        # evaluate: rmse
        rmse_validate = mean_squared_error(self.y_validate.taxvaluedollarcnt, self.y_validate.TVDC_pred_glm)**(1/2)

        print(f"RMSE for GLM using Tweedie, power={power} & alpha={alpha}\nTraining/In-Sample: ", rmse_train, 
            "\nValidation/Out-of-Sample: ", rmse_validate)

        model_stats = self.make_model_stats(model_name, rmse_train, rmse_validate,
                              x_train, alpha=alpha, power=power)

        self.tweedies = pd.concat([self.tweedies, model_stats], ignore_index=True)

        return pd.DataFrame(model_stats)

    
    def all_models_df(self):
        df= pd.concat([self.OLS, self.lassolars, self.tweedies], ignore_index=True)

        return df

    def make_model_stats(self, model_name, rmse_train, rmse_validate, x_train, alpha=0, power=0):
        model_stats = {}

        model_stats['model_name'] = model_name
        model_stats['rmse_train'] = rmse_train
        model_stats['rmse_validate'] = rmse_validate

        if model_name == 'OLS' or model_name == 'lasso_lars':
            model_stats['power'] = 'NA'
        else:
            model_stats['power'] = power
        
        if model_name == 'lasso_lars' or model_name == 'tweedie':
            model_stats['alpha'] = alpha
        else:
            model_stats['alpha'] = 'NA'

        model_stats['features'] = [x_train.columns]

        model_stats = pd.DataFrame(model_stats)
        model_stats = self.percent_diff(model_stats)
        model_stats = self.baseline_diff(model_stats)

        return model_stats

    def percent_diff(self, model_stats):
        model_stats['percent_diff'] = round((model_stats['rmse_train'] - model_stats['rmse_validate'])/model_stats['rmse_train'] * 100, 2)

        return model_stats

    def baseline_diff(self, model_stats):
        model_stats['baseline_diff_percent_train'] = (model_stats['rmse_train']) / self.train.taxvaluedollarcnt.mean() * 100
        model_stats['baseline_diff_percent_validate'] = (model_stats['rmse_validate']) / self.validate.taxvaluedollarcnt.mean() * 100
        


        return model_stats