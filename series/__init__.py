
def extract_features(action_frame,extractor):
    actions=action_frame['Action']
    cats=action_frame['Category']
    features=[extractor(act.time_series) for act in actions]
    labeled_features=[(ft,cat) for ft,cat in zip(features,cats)]
    return labeled_features

def trivial_extr(time_series):
    features=[time_series[col_i].mean() for col_i in time_series]
    return features