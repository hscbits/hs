
# reading dataset using pandas 

import pandas as ps
dt = ps.read_csv('/kaggle/input/nov2019/listings.csv')
dt31 = dt[[ 'id','experiences_offered','host_since','host_response_rate','host_is_superhost','host_listings_count',     
          'host_total_listings_count','host_has_profile_pic','host_identity_verified', 'city', 'neighbourhood_cleansed', 
          'is_location_exact', 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 
         'square_feet', 'price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee', 'guests_included',  
         'extra_people', 'minimum_nights', 'maximum_nights', 'has_availability', 'availability_30', 'availability_60', 'availability_90', 'availability_365',   
         'number_of_reviews', 'number_of_reviews_ltm', 'first_review', 'last_review', 'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
         'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'requires_license', 
         'instant_bookable', 'is_business_travel_ready', 'cancellation_policy', 'require_guest_profile_picture',  
         'require_guest_phone_verification', 'calculated_host_listings_count', 'calculated_host_listings_count_entire_homes', 
         'calculated_host_listings_count_private_rooms', 'reviews_per_month']]



# analysing dataset 

dt.head(10)
dt.shape 
dt.describe()
dt.dtypes
dt.isnull().sum()
#getting categorical columns 
dt.select_dtypes(include='category')



#cleaning and preprocessing of data 

dt1= dt31.drop(['city'], axis= 1 )
dt1['price']= dt1['price'].apply(lambda h : h.replace('$', ''))
dt1['security_deposit']= dt1['security_deposit'].apply(lambda h : str(h).replace('$', ''))
dt1['cleaning_fee']= dt1['cleaning_fee'].apply(lambda h : str(h).replace('$', ''))
dt1['extra_people']= dt1['extra_people'].apply(lambda h : str(h).replace('$', ''))
dt1['price']= dt1['price'].apply(lambda h : h.replace(',', ''))
dt1['security_deposit']= dt1['security_deposit'].apply(lambda h : str(h).replace(',', ''))
dt1['cleaning_fee']= dt1['cleaning_fee'].apply(lambda h : str(h).replace(',', ''))
dt1['security_deposit']= dt1['security_deposit'].apply(lambda h : h.replace('na', '').replace('n', '').replace('a',''))
dt1['cleaning_fee']= dt1['cleaning_fee'].apply(lambda h : h.replace('na', '').replace('n', '').replace('a',''))
dt1.fillna({ 'host_is_superhost': dt1['host_is_superhost'].mode()[0] , 
'host_has_profile_pic': dt1['host_has_profile_pic'].mode()[0], 'host_identity_verified': dt1['host_identity_verified'].mode()[0]} , inplace= True )
dt1['host_response_rate']= dt1['host_response_rate'].apply(lambda h : str(h).replace('%',' ') )

# Imputation 

dt1.fillna({ 'host_listings_count': dt1['host_listings_count'].mean()   , 'host_total_listings_count': dt1['host_total_listings_count'].mean() ,
'bathrooms': dt1['bathrooms'].mean()   ,'bedrooms': dt1['bedrooms'].mean()   ,'beds': dt1['beds'].mean()   ,
'review_scores_rating': dt1['review_scores_rating'].mean()   ,'review_scores_accuracy': dt1['review_scores_accuracy'].mean()   ,
'review_scores_cleanliness': dt1['review_scores_cleanliness'].mean()   ,'review_scores_checkin': dt1['review_scores_checkin'].mean()   ,
'review_scores_communication': dt1['review_scores_communication'].mean()   ,'review_scores_location': dt1['review_scores_location'].mean()   ,
'review_scores_value': dt1['review_scores_value'].mean()   ,'reviews_per_month': dt1['reviews_per_month'].mean()   ,} , inplace= True )
#dt1.dropna(axis=0, how='any',  subset= ['city'] , inplace= True)   
dt1['first_review'].fillna(method= 'ffill', inplace= True)
dt1['last_review'].fillna(method= 'bfill', inplace= True)
dt1['last_review'].fillna(method= 'ffill', inplace= True)
dt1['host_since'].fillna(method= 'ffill', inplace= True)
dt1.drop(['square_feet','weekly_price','monthly_price'], axis= 1 , inplace= True)

# preprocessing date columns 

import datetime
def datecal(col) :
  d = datetime.datetime(2019, 7 , 15)
  b= d - ps.to_datetime(col)
  return(b.days)
dt1['first_review'] = dt1['first_review'].apply(lambda col : datecal(col))
dt1['last_review'] = dt1['last_review'].apply(lambda col : datecal(col))
dt1['host_since'] = dt1['host_since'].apply(lambda col : datecal(col))
dt1['host_response_rate']= dt1['host_response_rate'].apply(lambda h : h.replace('na', '').replace('n', '').replace('a',''))

# data type coversion of columns as required 

dt1['host_response_rate']= ps.to_numeric(dt1['host_response_rate'])
dt1['price']= ps.to_numeric(dt1['price'])
dt1['security_deposit']= ps.to_numeric(dt1['security_deposit'])
dt1['cleaning_fee']= ps.to_numeric(dt1['cleaning_fee'])
dt1['extra_people']= ps.to_numeric(dt1['extra_people'])
dt1['security_deposit'].fillna(0, inplace= True)
dt1['cleaning_fee'].fillna(0, inplace= True)
dt1[[ 'extra_people' , 'price', 'security_deposit', 'cleaning_fee']].astype(str).astype(float)
dt1.fillna({ 'host_response_rate': dt1['host_response_rate'].mean() } , inplace= True)


# one hot encoding of categorical columns 

dfcat= dt1[['experiences_offered' , 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed' , 
'is_location_exact' , 'property_type' , 'room_type', 'bed_type', 'has_availability', 'requires_license','instant_bookable',
'is_business_travel_ready', 'cancellation_policy' , 'require_guest_profile_picture', 'require_guest_phone_verification']]
dum= ps.get_dummies(dfcat)
dt1.drop(['experiences_offered' , 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed' , 
'is_location_exact' , 'property_type' , 'room_type', 'bed_type', 'has_availability', 'requires_license','instant_bookable',
'is_business_travel_ready', 'cancellation_policy' , 'require_guest_profile_picture', 'require_guest_phone_verification'], axis=1, inplace=True)

#scaling the data 

from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
dt1[['host_since', 'host_response_rate', 'host_listings_count',
       'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms',
       'beds', 'price', 'security_deposit', 'cleaning_fee', 'guests_included',
       'extra_people', 'minimum_nights', 'maximum_nights', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'number_of_reviews', 'number_of_reviews_ltm', 'first_review',
       'last_review', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms', 'reviews_per_month']] = scaler.fit_transform(dt1[['host_since', 'host_response_rate', 'host_listings_count',
       'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms',
       'beds', 'price', 'security_deposit', 'cleaning_fee', 'guests_included',
       'extra_people', 'minimum_nights', 'maximum_nights', 'availability_30',
       'availability_60', 'availability_90', 'availability_365',
       'number_of_reviews', 'number_of_reviews_ltm', 'first_review',
       'last_review', 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'calculated_host_listings_count',
       'calculated_host_listings_count_entire_homes',
       'calculated_host_listings_count_private_rooms', 'reviews_per_month']])
dt2= dt1.join(dum)



# creating amenities' availability features 


ls = []
for x in range(len(dt)-1):
    for x in dt['amenities'][0].split(',') :
        if x not in ls :
            ls.append(x)
def sts(col):
    if word in str(col.split(',')).replace('"',' ').replace('{', ' ').replace('}',' ').replace('-', ' ') :
        return 1
    else : 
        return 0
    dftemp = ps.DataFrame()
for word in ls :
    word = word.replace('"',' ').replace('{', ' ').replace('}',' ').replace('-', ' ') 
    dftemp[word.strip()] = dt['amenities'].apply(lambda col : sts(col) )
dt3 = dt2.join(dftemp)

# exploring created amenities' availability distribution 

import matplotlib.pyplot as plt 
for h in range(53) :
  dftemp.iloc[: , h].plot(kind='hist' , title= h)
  plt.show()


# preprocessing "city" column data 

dtct= dt[['city','beds']]
dtct.isnull().sum()
dtct['city'].fillna('missing', inplace=True)
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
dtct['city']= le.fit_transform(dtct['city'])
dtct[['city', 'beds']]= scaler.fit_transform(dtct[['city','beds']])
dt3= dt3.join(dtct['city'])


# selecting top price influencing features 

cor = dt3.corr()
cols = cor.nlargest( 64 , 'price').index
dt4 = dt3[cols]

# exploring final data 

dt4.head(10)
dt4.shape 
dt4.describe()
dt4.dtypes
dt4.isnull().sum()
#getting categorical columns 
dt4.select_dtypes(include='category')
import matplotlib.pyplot as plt
import seaborn as sns
plt.subplots(figsize= (10 , 10 ))
#sns.heatmap(dt4.corr(), cmap= 'RdYlGn_r' ,linewidths= 0.5, square= False ,linecolor='black',  annot=True )
sns.heatmap(dt4.isnull())

# exploring price distribution in final data 

dt4['price'].plot.hist(bins=15 , color= 'g'  ) 


# final data preparation for training and testing 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
x= np.array(dt4.drop('price',axis= 1))
y= np.array(dt4['price'])
x_train, x_test, y_train, y_test = train_test_split( x , y, test_size=0.15, random_state= 80 )


# Implementation of linear regression model, training, testing and checking results 

from sklearn import linear_model 

reg = linear_model.LinearRegression(fit_intercept= True, normalize= False, copy_X= False, n_jobs= -1 )
reg.fit(x_train, y_train)
reg.score(x_test, y_test)
pred1 = reg.predict(x_test)
print('mse is ', mean_squared_error(y_test, pred1))
print('mae is ',mean_absolute_error(y_test, pred1))


# Implementation of elastic net model, training, testing and and checking results 


from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

eln = ElasticNet(max_iter= 15000 ,alpha= 0.0001, l1_ratio= 0.001, random_state= 80)
eln.fit(x_train, y_train)
pred = eln.predict(x_test)
print('r2 is ' , r2_score(y_test, pred))
print('mse is ', mean_squared_error(y_test, pred))


# Implementation of Random Forest model, training, testing and and checking results 

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth= 128, random_state=80,n_estimators= 1200, max_features = 0.4 , n_jobs = -1)
rf.fit(x_train, y_train)  
pred3 = rf.predict(x_test)
print('r2 is ',r2_score(y_test, pred3))
print('mse is ', mean_squared_error(y_test, pred3))


# Implementation of XGBoost model, training, testing and and checking results 

import xgboost as xgb
xg = xgb.XGBRegressor(colsample_bytree= 1, subsample= 1,learning_rate=0.1, max_depth=80, min_child_weight= 1, n_estimators= 800,
                                           reg_alpha=0.1, reg_lambda= 1, gamma=0.01, silent=0 , random_state = 80, nthread = -1)xg.fit(x_train, y_train)  
pred5 = xg.predict(x_test)
print('r2 is ',r2_score(y_test, pred5))
print('mse is ', mean_squared_error(y_test, pred5))




# Thank You ! 
 






