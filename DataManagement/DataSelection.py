import pandas as pd

slot_count = {
    'c4.2xlarge': 8,  # 8 vcpus (4 cpu cores with 2 threads each)
    'r4.2xlarge': 8,  # 8 vcpus (4 cpu cores with 2 threads each)
    'r4.xlarge': 4,  # 4 vcpus (2 cpu cores with 2 threads each)
    'm5.xlarge': 4,  # 4 vcpus (2 cpu cores with 2 threads each)
    'm4.xlarge': 4,  # 4 vcpus (2 cpu cores with 2 threads each)
    'm4.2xlarge': 8,  # 8 vcpus (4 cpu cores with 2 threads each)
    'm4.16xlarge': 64,  # 64 vcpus (32 cpu cores with 2 threads each)
}

memory = {  # in MB
    'c4.2xlarge': 15000,
    'm5.xlarge': 16000,
    'm4.xlarge': 16000,
    'm4.2xlarge': 32000,
    'm4.16xlarge': 256000,
    'r4.2xlarge': 61000,
    'r4.xlarge': 30500,
}

size = {  # in MB
    's3://c3othesis/data/blobs_3_100mx10': 20300,
    's3://c3othesis/data/blobs_3_100mx15': 30500,
    's3://c3othesis/data/blobs_3_100mx5': 10500,
    's3://c3othesis/data/blobs_3_125mx5': 13200,
    's3://c3othesis/data/blobs_3_150mx5': 16000,
    's3://c3othesis/data/blobs_3_175mx5': 18600,
    's3://c3othesis/data/blobs_3_200mx5': 21300,
    's3://c3othesis/data/lr_f1_200x3_dense': 0.012,
    's3://c3othesis/data/lr_f1_10mx5_dense': 1000,
    's3://c3othesis/data/lr_f1_100mx5_dense': 9730,
    's3://c3othesis/data/lr_f1_100mx10_dense': 19558,
    's3://c3othesis/data/lr_f1_100mx15_dense': 29593,
    's3://c3othesis/data/lr_f1_150mx5_dense': 14540,
    's3://c3othesis/data/lr_f1_200mx5_dense': 19353,
    's3://c3othesis/data/lr_f1_250mx5_dense': 24268,
    's3://c3othesis/data/lr_f1_300mx5_dense': 29081,
    's3://c3othesis/data/graph_pareto_2mx10m': 133,
    's3://c3othesis/data/graph_pareto_2mx20m': 266,
    's3://c3othesis/data/graph_uniform_100kx10m': 112,
    's3://c3othesis/data/graph_uniform_1mx10m': 131,
    's3://c3othesis/data/graph_uniform_1mx20m': 263,
    's3://c3othesis/data/graph_uniform_1mx30m': 394,
    's3://c3othesis/data/graph_uniform_2mx10m': 142,
    's3://c3othesis/data/graph_uniform_2mx20m': 284,
    's3://c3othesis/data/graph_uniform_2mx30m': 426,
    's3://c3othesis/data/graph_uniform_3mx10m': 145,
    's3://c3othesis/data/graph_uniform_3mx20m': 291,
    's3://c3othesis/data/graph_uniform_3mx30m': 436,
    's3://c3othesis/data/graph_uniform_5mx10m': 148,
    's3://c3othesis/data/lines_1mx100': 96.3,
    's3://c3othesis/data/lines_10mx100': 963,
    's3://c3othesis/data/lines_100mx100': 9630,
    's3://c3othesis/data/lines_120mx100': 11556,
    's3://c3othesis/data/lines_140mx100': 13482,
    's3://c3othesis/data/lines_160mx100': 15408,
    's3://c3othesis/data/lines_180mx100': 17334,
    's3://c3othesis/data/lines_200mx100': 19260,
    's3://c3othesis/data/lines_10mx1000': 9530, # estimate
    #'s3://c3othesis/data/text_0_100mx100': 9625,
    #'s3://c3othesis/data/text_100m_100mx100': 9625,
    #'s3://c3othesis/data/text_0_200mx100': 19251,
    #'s3://c3othesis/data/text_200m_200mx100': 19251,
    's3://c3othesis/data/text_0_100mx100':10100,
    's3://c3othesis/data/text_25m_100mx100':10100,
    's3://c3othesis/data/text_50m_100mx100':10100,
    's3://c3othesis/data/text_75m_100mx100':10100,
    's3://c3othesis/data/text_100m_100mx100':10100,
    's3://c3othesis/data/text_0_125mx100':12625,
    's3://c3othesis/data/text_31250k_125mx100':12625,
    's3://c3othesis/data/text_62500k_125mx100':12625,
    's3://c3othesis/data/text_93750k_125mx100':12625,
    's3://c3othesis/data/text_125m_125mx100':12625,
    's3://c3othesis/data/text_0_150mx100':15150,
    's3://c3othesis/data/text_37500k_150mx100':15150,
    's3://c3othesis/data/text_75m_150mx100':15150,
    's3://c3othesis/data/text_112500k_150mx100':15150,
    's3://c3othesis/data/text_150m_150mx100':15150,
    's3://c3othesis/data/text_0_175mx100' :17675,
    's3://c3othesis/data/text_43750k_175mx100' :17675,
    's3://c3othesis/data/text_87500k_175mx100' :17675,
    's3://c3othesis/data/text_131250k_175mx100' :17675,
    's3://c3othesis/data/text_175m_175m_100' :17675,
    's3://c3othesis/data/text_175m_175mx100' :17675,
    's3://c3othesis/data/text_0_200mx100' :20200,
    's3://c3othesis/data/text_50m_200mx100' :20200,
    's3://c3othesis/data/text_100m_200mx100' :20200,
    's3://c3othesis/data/text_150m_200mx100' :20200,
    's3://c3othesis/data/text_200m_200mx100' :20200,
}

price = { # per minute in USD
    'c4.2xlarge': 0.00838,
    'm5.xlarge': 0.0040,
    'm4.xlarge': 0.0043,
    'm4.2xlarge': 0.00867,
    'm4.16xlarge': 0.05783,
    'r4.2xlarge': 0.01108,
    'r4.xlarge': 0.00555,
}

def str_to_num(s):
    d= {'k':3*'0', 'm':6*'0'}
    num_str = ''
    for ch in s:
        if ch in d: ch = d[ch]
        num_str += ch
    return int(num_str)

def transform(df):
    """
    Turn raw data into data usable data with type information
    This is the part of the transformation that is algorithm-independent
    """

    df.loc[:,'gross_runtime'] = df['gross_runtime'].astype('int64')
    df.loc[:,'net_runtime'] = df['net_runtime'].astype('int64')
    df.loc[:,'instance_count'] = df['instance_count'].astype('int64')

    df.loc[:, 'slots'] = df['machine_type'].map(lambda mt: slot_count[mt])\
                       * df['instance_count']  # Calculate node count from machine type

    df.loc[:, 'memory'] = df['machine_type'].map(lambda mt: memory[mt]) * df['instance_count'].astype('int64')
    df.loc[:,'data_size'] = df['args'].map(lambda args: size[args[0]])

    # Calculate cost
    df.loc[:, 'cost'] = df['machine_type'].map(lambda mt: price[mt])\
               * df['instance_count'].astype('float') * (df['gross_runtime'])/60

    return df

def select_algorithm(df, algorithm):

    df = df[df['algorithm'] == algorithm]

    # unpack args
    if algorithm == 'SGDLR':

        def foo(ds, feature='data_points'):
            # Extract data characteristics from dataset
            if feature == 'data_points':
                string = ds.split('_')[2].split('x')[0]
            else:
                string = ds.split('_')[2].split('x')[1]
            return str_to_num(string)

        df.loc[:,'iterations'] = df['args'].map(lambda args: args[1]).astype('int64')
        df = df[df['memory'] > df['data_size']*1.6] # 2 -> no BN , 1.5 -> BN in  SGD
        df.loc[:,'features'] = df['args'].map(lambda args: foo(args[0], 'features')).astype('int64')
        df.loc[:,'data_points'] = df['args'].map(lambda args: foo(args[0], 'data_points')).astype('int64')

        df = df[['algorithm', 'instance_count', 'machine_type', 'slots', 'iterations',
                 'gross_runtime', 'cost', 'data_size', 'memory', 'features', 'data_points']]

    elif algorithm == 'PageRank':
        df.loc[:,'convergence_criterion'] = df['args'].map(lambda args: args[1]).astype('float64')

        def foo(ds, feature='pages'):
            # Extract data characteristics from dataset
            if feature == 'pages':
                string = ds.split('_')[-1].split('x')[0]
            elif feature == 'links':
                string = ds.split('_')[-1].split('x')[1]
            else:  # distribution
                return ds.split('_')[1]
            return str_to_num(string)

        df.loc[:,'pages'] = df['args'].map(lambda args: foo(args[0], 'pages')).astype('int64')
        df.loc[:,'links'] = df['args'].map(lambda args: foo(args[0], 'links')).astype('int64')
        df.loc[:,'distribution'] = df['args'].map(lambda args: foo(args[0], 'distribution')).astype('str')
        df = df[['algorithm', 'instance_count', 'machine_type', 'slots', 'convergence_criterion',
                 'gross_runtime', 'cost', 'data_size', 'memory', 'pages', 'links', 'distribution']]

    elif algorithm == 'Sort':
        df = df[['algorithm', 'instance_count', 'machine_type', 'slots', 'gross_runtime', 'cost', 'data_size', 'memory']]

    elif algorithm == 'Grep':

        def foo(ds):
            # Retrieve the chance of occurrence of the keyword per line from the DS
            _, occstr, lines = ds.split('_')
            lc = str_to_num(lines.split('x')[0])
            occ = str_to_num(occstr)
            return occ / lc

        df.loc[:,'p_occurrence'] = df['args'].map(lambda args: foo(args[0])).astype('float64')
        df = df[['algorithm', 'instance_count', 'machine_type', 'slots', 'gross_runtime', 'cost', 'data_size', 'memory', 'p_occurrence']]

    elif algorithm == 'KMeans':

        def foo(ds, feature='data_points'):
            # Retrieve num_features, num_occurrences from the DS
            datapoints, features = ds.split('_')[-1].split('x')
            if feature == 'data_points':
                return str_to_num(datapoints)
            else:
                return str_to_num(features)

        df.loc[:,'data_points'] = df['args'].map(lambda args: foo(args[0], 'data_points')).astype('int64')
        df.loc[:,'features'] = df['args'].map(lambda args: foo(args[0], 'features')).astype('int64')
        df.loc[:,'k'] = df['args'].map(lambda args: args[1]).astype('int64')
        df = df[['algorithm', 'instance_count', 'machine_type', 'slots', 'gross_runtime', 'cost', 'data_size', 'memory', 'data_points', 'features', 'k']]

    return df

def get_training_data(df, keep=1):
    """
    This is the last step in the 'pipeline':
    Turning a dataframe containing the transformed data of a single algorithm
    into X, y as numpy (nd)-arrays
    """
    assert all(df['algorithm'] == df.iloc[0]['algorithm']), \
            "Algorithm unclear: Please use 'select_algorithm' first"

    assert all(df['machine_type'] == df.iloc[0]['machine_type']), \
            "Diverse machine types! - Please select a single one first"

    algorithm = df.iloc[0]['algorithm']

    def get_medians(df):
        g = df.groupby(by=list(df.keys())[1:])
        return pd.DataFrame(g.median().to_records())  # Puts gross_rt to end!!

    if algorithm == 'Sort':
        df = df[['gross_runtime', 'instance_count', 'data_size']]

    elif algorithm == "Grep":
        df = df[['gross_runtime', 'instance_count', 'data_size', 'p_occurrence']]

    elif algorithm == 'SGDLR':
        df = df[['gross_runtime', 'instance_count', 'data_points', 'features', 'iterations']]

    elif algorithm == "KMeans":
        df = df[['gross_runtime', 'instance_count', 'data_points', 'features', 'k']]

    elif algorithm == 'PageRank':
        df = df[df['distribution'] == 'uniform']
        df = df[['gross_runtime', 'instance_count', 'pages', 'links', 'convergence_criterion']]

    else:
        raise ValueError(f"The algorithm {algorithm} was not found")

    dataset = get_medians(df).to_numpy()
    X, y = dataset[:, :-1], dataset[:,-1]

    return X, y


def prune(df):
    """
    Keep only successful experiments
    Unsuccessful ones didn't have a net runtime figure
    (For final version, don't include unsuccessful runs at all ... maybe)
    """
    with pd.option_context('mode.use_inf_as_null', True):
       df = df.dropna()
    df = df[df['net_runtime']>0]

    return df


