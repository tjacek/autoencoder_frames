
def extract_features(action_frame,extractor):
    actions=action_frame['Action']
    cats=action_frame['Category']
    extractors=[trivial_extr,autocorr_extr,cross_corel_extr]
    comb_extr=get_combined_extractors(extractors)
    features=[comb_extr(act.time_series) for act in actions]
    labeled_features=[(ft,cat) for ft,cat in zip(features,cats)]
    return labeled_features

def curry_extractor(extractor):
    return lambda af:extract_features(af,extractor)

def trivial_extr(time_series):
    time_series=get_time_series(time_series)
    features=[td.mean() for td in time_series]
    return features

def var_extr(time_series):
    time_series=get_time_series(time_series)
    features=[td.std() for td in time_series]
    return features

def autocorr_extr(time_series):
    time_series=get_time_series(time_series)
    features=[td.autocorr() for td in time_series]
    return features

def cross_corel_extr(time_series):
    result=time_series.corr()
    corel=result.values.flatten()
    return corel.tolist()

def get_combined_extractors(extr):
	return lambda ts:combine_extractors(ts,extr)

def combine_extractors(time_series,extractors):
    raw_features=[extr(time_series) for extr in extractors]
    features=[]#reduce(lambda x,y:x+y,[],raw_features)
    for raw in raw_features:
    	features+=raw
    return features

def get_time_series(time_series):
	return  [time_series[col_i] for col_i in time_series]