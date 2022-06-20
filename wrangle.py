import acquire
import pandas as pd
import prepare
import sklearn
import split

def clean_zillow(df):
    # drop nulls and extra column
    df = df.dropna()
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')

    # readability
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'sqr_ft'})

    # convert yearbuilt/fips to ints
    cols = ['yearbuilt', 'fips']
    df[cols] = df[cols].astype('int64')

    # limit houses to include only >= 70 sqr ft 
    # (most prevelant minimum required sqr ft by state)
    df = df[df.sqr_ft >= 70]

    # exclude houses with bthroom/bedroomcnts of 0
    df = df[df.bedroomcnt != 0]
    df = df[df.bathroomcnt != 0.0]

    # remove all rows where any column has z score gtr than 3
    non_quants = ['fips', 'parcelid', 'latitude', 'longitude']
    quants = df.drop(columns=non_quants).columns
    print(quants)
    
    # remove numeric values with > 3.5 std dev
    df = prepare.remove_outliers(3.5, quants, df)

    # see if sqr feet makes sense
    df = clean_sqr_feet(df)

    # fips to categorical data
    df.fips = df.fips.astype('object')

    categorical = ['county']

    df.yearbuilt = df.yearbuilt.astype('float64')

    # make sure 'fips' is object type
    df = map_counties(df)
    
    return df, categorical, quants

def minimum_sqr_ft(df):
    # min square footage for type of room
    bathroom_min = 10
    bedroom_min = 70
    
    # total MIN sqr feet
    total = df.bathroomcnt * bathroom_min + df.bedroomcnt * bedroom_min

    # return MIN sqr feet
    return total

def clean_sqr_feet(df):
    # get MIN sqr ft
    min_sqr_ft = minimum_sqr_ft(df)

    # return df with sqr_ft >= min_sqr_ft
    # change 'sqr_ft' to whichever name you have for sqr_ft in df
    return df[df.sqr_ft >= min_sqr_ft]

def map_counties(df):

    # identified counties for fips codes 
    counties = {6037: 'los_angeles',
                6059: 'orange_county',
                6111: 'ventura'}

    # map counties to fips codes
    df.fips = df.fips.map(counties)

    # rename fips to county for clarity
    df.rename(columns=({ 'fips': 'county'}), inplace=True)

    return df

def wrangle_zillow():
    # aquire zillow data from mysql or csv
    zillow = acquire.get_zillow_data()

    # clean zillow data
    zillow, categorical, quant_cols = clean_zillow(zillow)

    return zillow, categorical, quant_cols

"train, test, validate"
def xy_tvt_data(train, validate, test, target_var):
    cols_to_drop = ['latitude', 'longitude', 
                    'parcelid', 'Unnamed: 0']

    
    x_train = train.drop(columns=drop_cols(cols_to_drop, 
                                           train, 
                                           target_var))
    y_train = train[target_var]


    x_validate = validate.drop(columns=drop_cols(cols_to_drop, 
                                              validate, 
                                              target_var))
    y_validate = validate[target_var]


    X_test = test.drop(columns=drop_cols(cols_to_drop, 
                                         test, 
                                         target_var))
    Y_test = test[target_var]

    return x_train, y_train, x_validate, y_validate, X_test, Y_test

def drop_cols(cols_to_drop, tvt_set, target_var):
    tvt_cols = [col for col in cols_to_drop if col in tvt_set.columns]
    tvt_cols.append(target_var)
    
    return tvt_cols

def encode_object_columns(train_df, drop_encoded=True):
    
    col_to_encode = object_columns_to_encode(train_df)
    dummy_df = pd.get_dummies(train_df[col_to_encode],
                              dummy_na=False,
                              drop_first=[True for col in col_to_encode])
    train_df = pd.concat([train_df, dummy_df], axis=1)
    
    if drop_encoded:
        train_df = drop_encoded_columns(train_df, col_to_encode)

    return train_df

def object_columns_to_encode(train_df):
    object_type = []
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            object_type.append(col)

    return object_type

def drop_encoded_columns(train_df, col_to_encode):
    train_df = train_df.drop(columns=col_to_encode)
    return train_df

def encoded_xy_data(train, validate, test, target_var):
    xy_train_validate_test = list(xy_tvt_data(train, validate, 
                                              test, target_var))
    

    for i in range(0, len(xy_train_validate_test), 2):
        
        xy_train_validate_test[i] = encode_object_columns(xy_train_validate_test[i])

    xy_train_validate_test = tuple(xy_train_validate_test)

    return xy_train_validate_test


def fit_and_scale(scaler, sets_to_scale):
    scaled_data = []
    scaler.fit(sets_to_scale[0][sets_to_scale[0].select_dtypes(include=['float64', 'uint8']).columns])

    for i in range(0, len(sets_to_scale), 1):
        #print(sets_to_scale[i].info())
        if i % 2 == 0:
            # only scales float columns
            floats = sets_to_scale[i].select_dtypes(include=['float64', 'uint8']).columns

            # fits scaler to training data only, then transforms 
            # train, validate & test
            scaled_data.append(pd.DataFrame(data=scaler.transform(sets_to_scale[i][floats]), columns=floats))
        else:
            scaled_data.append(sets_to_scale[i])


    return tuple(scaled_data)

def encoded_and_scaled(train, validate, test, target_var):
    sets_to_scale = encoded_xy_data(train, validate, test, target_var)
    
    scaler = sklearn.preprocessing.RobustScaler()
    scaled_data = fit_and_scale(scaler, sets_to_scale)

    return scaled_data

def rename_and_add_scaled_data(train, validate, test,
                               x_train_scaled, 
                               x_validate_scaled,
                               x_test_scaled):

    columns = {'bedroomcnt': 'scaled_bedroomcnt',
                       'bathroomcnt': 'scaled_bathroomcnt',
                       'sqr_ft': 'scaled_sqr_ft',
                       'yearbuilt': 'scaled_yearbuilt',
                       'county_orange_county': 'scaled_OC',
                       'county_ventura' : 'scaled_ventura'}

    x_train_scaled = x_train_scaled.rename(columns=columns)
    x_validate_scaled = x_validate_scaled.rename(columns=columns)
    x_test_scaled = x_test_scaled.rename(columns=columns)

    train = pd.concat([train.reset_index(), x_train_scaled], axis=1)
    validate = pd.concat([validate.reset_index(), x_validate_scaled], axis=1)
    test = pd.concat([test.reset_index(), x_test_scaled], axis=1)

    return train, validate, test, x_train_scaled, \
           x_validate_scaled, x_test_scaled


def all_train_validate_test_data(df, target_var):
    train, validate, test = split.train_validate_test_split(df, target_var)
    
    x_train_scaled, y_train, \
    x_validate_scaled, y_validate, \
    x_test_scaled, y_test = encoded_and_scaled(train, validate, test, target_var)

    train, validate, \
    test, x_train_scaled, \
    x_validate_scaled, \
    x_test_scaled = rename_and_add_scaled_data(train,
                                                validate, test,
                                                x_train_scaled,
                                                x_validate_scaled,
                                                x_test_scaled)
    
    return train, validate, test, \
           x_train_scaled, y_train, \
           x_validate_scaled, y_validate, \
           x_test_scaled, y_test
