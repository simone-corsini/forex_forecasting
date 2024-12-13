def prepare_indicator_sets(dataset, set_name='set1'):
    if set_name == 'set1':
        return prepare_indicator_set1(dataset)

    return dataset, [], 0

def prepare_indicator_set1(dataset):
    ema_slow_period = 9
    ema_fast_period = 21
    bollinger_period = 20
    bolinger_deviation = 2

    columns = []
    dataset['ema_slow'] = dataset['hl_avg'].ewm(span=ema_slow_period, adjust=False).mean()
    dataset['ema_fast'] = dataset['hl_avg'].ewm(span=ema_fast_period, adjust=False).mean()
    dataset['ema_fast_slow'] = dataset['ema_fast'] - dataset['ema_slow']
    dataset['ema_fast_slow_diff_rel'] = dataset['ema_fast_slow'].diff()
    del dataset['ema_slow']
    del dataset['ema_fast']
    del dataset['ema_fast_slow']
    columns += ['ema_fast_slow_diff_rel']

    dataset['hl_diff'] = dataset['high'] - dataset['low']
    dataset['hl_diff_diff_rel'] = dataset['hl_diff'].diff()
    del dataset['hl_diff']
    columns += ['hl_diff_diff_rel']

    dataset['bollinger_ma'] = dataset['ma'].rolling(window=bollinger_period).mean()
    dataset['bollinger_std'] = dataset['ma'].rolling(window=bollinger_period).std()
    dataset['bollinger_upper'] = dataset['bollinger_ma'] + bolinger_deviation * dataset['bollinger_std']
    dataset['bollinger_lower'] = dataset['bollinger_ma'] - bolinger_deviation * dataset['bollinger_std']
    dataset['bollinger_width'] = dataset['bollinger_upper'] - dataset['bollinger_lower']
    dataset['deviation_upper'] = (dataset['ma'] - dataset['bollinger_upper']) / dataset['bollinger_width']
    dataset['deviation_lower'] = (dataset['ma'] - dataset['bollinger_lower']) / dataset['bollinger_width']
    dataset['bollinger_width_diff_rel'] = dataset['bollinger_width'].diff()
    dataset['deviation_upper_diff_rel'] = dataset['deviation_upper'].diff()
    dataset['deviation_lower_diff_rel'] = dataset['deviation_lower'].diff()
    del dataset['bollinger_ma']
    del dataset['bollinger_std']
    del dataset['bollinger_upper']
    del dataset['bollinger_lower']
    del dataset['bollinger_width']
    del dataset['deviation_upper']
    del dataset['deviation_lower']
    columns += ['bollinger_width_diff_rel', 'deviation_upper_diff_rel', 'deviation_lower_diff_rel']

    return dataset, columns, max(ema_slow_period, ema_fast_period, bollinger_period)

