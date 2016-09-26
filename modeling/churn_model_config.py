"""
Description: Configure running of the profitability models analysis
Authors: Eliano Marques, Almir Mutapcic, Guillermo Breto Rangel, Giacomo Snidero
Created on: Aug 2016
"""

#########################################################
# CONSTANTS
#########################################################

intercept = 'intercept'
year = 'year'
month = 'month'

key_columns = ['user_account_id', year, month]

num_features = [
 'user_lifetime',
 'user_intake',
 'user_no_outgoing_activity_in_days',
 'user_account_balance_last',
 'user_spendings',
 'user_has_outgoing_calls',
 'user_has_outgoing_sms',
 'user_use_gprs',
 'user_does_reload',
 'reloads_inactive_days',
 'reloads_count',
 'reloads_sum',
 'calls_outgoing_count',
 'calls_outgoing_spendings',
 'calls_outgoing_duration',
 'calls_outgoing_spendings_max',
 'calls_outgoing_duration_max',
 'calls_outgoing_inactive_days',
 'calls_outgoing_to_onnet_count',
 'calls_outgoing_to_onnet_spendings',
 'calls_outgoing_to_onnet_duration',
 'calls_outgoing_to_onnet_inactive_days',
 'calls_outgoing_to_offnet_count',
 'calls_outgoing_to_offnet_spendings',
 'calls_outgoing_to_offnet_duration',
 'calls_outgoing_to_offnet_inactive_days',
 'calls_outgoing_to_abroad_count',
 'calls_outgoing_to_abroad_spendings',
 'calls_outgoing_to_abroad_duration',
 'calls_outgoing_to_abroad_inactive_days',
 'sms_outgoing_count',
 'sms_outgoing_spendings',
 'sms_outgoing_spendings_max',
 'sms_outgoing_inactive_days',
 'sms_outgoing_to_onnet_count',
 'sms_outgoing_to_onnet_spendings',
 'sms_outgoing_to_onnet_inactive_days',
 'sms_outgoing_to_offnet_count',
 'sms_outgoing_to_offnet_spendings',
 'sms_outgoing_to_offnet_inactive_days',
 'sms_outgoing_to_abroad_count',
 'sms_outgoing_to_abroad_spendings',
 'sms_outgoing_to_abroad_inactive_days',
 'sms_incoming_count',
 'sms_incoming_spendings',
 'sms_incoming_from_abroad_count',
 'sms_incoming_from_abroad_spendings',
 'gprs_session_count',
 'gprs_usage',
 'gprs_spendings',
 'gprs_inactive_days',
 'last_100_reloads_count',
 'last_100_reloads_sum',
 'last_100_calls_outgoing_duration',
 'last_100_calls_outgoing_to_onnet_duration',
 'last_100_calls_outgoing_to_offnet_duration',
 'last_100_calls_outgoing_to_abroad_duration',
 'last_100_sms_outgoing_count',
 'last_100_sms_outgoing_to_onnet_count',
 'last_100_sms_outgoing_to_offnet_count',
 'last_100_sms_outgoing_to_abroad_count',
 'last_100_gprs_usage',
               ]

num_features_v2 = [
    'churn',
]

cat_features = []

######################
# USER DEFINITIONS
#####################

target_dict = {
    'churn_model_v1': 'churn',
}

num_features_dict = {
    'churn_model_v1': num_features,

}

cat_features_dict = {
    'churn_model_v1': cat_features,

}
