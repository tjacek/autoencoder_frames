import ConfigParser

def read_hyper_params(config_path):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    conf=config.items("Float")
    conf=[ (pair_i[0],float(pair_i[1])) for pair_i in conf]
    conf_f=dict([ list(pair_i) for pair_i in conf])
    conf=config.items("String")
    conf_s=dict([ list(pair_i) for pair_i in conf])
    return dict(conf_f, **conf_s)

if __name__ == "__main__":
    in_path="/home/user/af/test.conf"
    print(read_hyper_params(in_path))