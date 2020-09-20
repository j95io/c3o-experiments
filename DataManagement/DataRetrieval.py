import boto3
import subprocess as sp
import re
import datetime
import pandas as pd

def get_existing_log_files():
    return sp.check_output(['find', 'logs']).decode('utf-8').split('\n')[:-1]


def get_existing_log_dirs():
    return sp.check_output(['ls', 'logs']).decode('utf-8').split('\n')[:-1]


def get_all_s3_object_keys(s3):
    keys = [e['Key'] for p in s3.get_paginator("list_objects_v2")
                                .paginate(Bucket='c3othesis')
                     for e in p['Contents']]
    return keys


def download_new_log_files(s3, keys, existing_log_files):
    # Download any log in S3 files not yet locally saved
    new_log_files = []
    for key in keys:
        # Log files of the first container, contains my own std.err logs
        if re.search('.*container.*01_000001.*stderr.gz', key):
            cluster, step = key.split('/')[1], key.split('/')[3].split('_')[-1]
        # Log files of the step. Contains the most basic info like gross runtime
        elif re.search('.*j-.*steps/s-.*controller.gz$', key):
            cluster, step = key.split('/')[1], key.split('/')[3]
            if step[:2] != 's-': continue  # regex bug workaround, caught things w/o 'step'
        else: continue

        file_name = f'logs/{cluster}/{step}'
        if file_name not in existing_log_files:
            sp.call(['mkdir', f'logs/{cluster}'])
            s3.download_file('c3othesis', key, file_name+'.gz')
            sp.call(['gunzip', file_name+'.gz'])
            new_log_files.append(file_name)

    return new_log_files


def extract_cluster_info():
    # Get machine type, machine count and steps of each emr cluster that existed
    emr = boto3.client('emr')

    d = {}
    for cluster in emr.list_clusters()['Clusters']:
        c_id = cluster['Id']

        step_paginator = emr.get_paginator('list_steps').paginate(
            ClusterId=c_id
        )

        steps = []
        #for s in  emr.list_steps(ClusterId = c_id)['Steps']:
        for s in  [s for p in step_paginator for s in p['Steps']]:
            if 'StartDateTime' in s['Status']['Timeline']:
                steps.append((s['Id'], s['Status']['Timeline']['StartDateTime'].timestamp()))
            else:
                steps.append((s['Id'], datetime.datetime.now().timestamp()))

        steps.sort(key=lambda x:x[1])  # MAKE SURE THEY ARE SORTED!!!
        steps = [s[0] for s in steps]

        types = []
        counts = []
        for ig in emr.list_instance_groups(ClusterId=c_id)['InstanceGroups']:
            types.append(ig['InstanceType'])
            counts.append(int(ig['RequestedInstanceCount']))
        assert 1 <= len(types) <= 2
        assert len(types) == 1 or types[0] == types[1]
        d[c_id] = (types[0], sum(counts), steps)

    return d


def create_cluster_info_logs(cluster_info):
    # Create the cluster info as a log file (EMR auto-deletes info after x days)
    # Steps should be in the order of when they happened
    for c_id, info in cluster_info.items():
        m_type, count, steps = info
        steps = ','.join(steps)
        sp.call(['mkdir', f'logs/{c_id}'])  # Make sure dir exists
        sp.call(f'echo "{m_type},{count},{steps}" > logs/{c_id}/cluster',
                shell=True)


def extract_from_cluster_logs():
    # Extract cluster info from cluster logs
    # Return: {'c_id': (mtype, count, steps)}
    d = {}
    for c_id in sp.check_output(['ls', 'logs']).decode('utf-8').split('\n')[:-1]:
        try:
            assert 'cluster\n' in sp.check_output(['ls', f'logs/{c_id}']).decode('utf-8')
            with open(f'logs/{c_id}/cluster', 'r') as f:
                for line in f:
                    ls = line.replace('\n','').split(',')
                    m_type, count, steps = ls[0], ls[1], ls[2:]
                    d[c_id] = (m_type, count, steps)
        except Exception as e:
            print('EFCL', c_id, e)
    return d



def extract_from_app_log(log_file):
    # Get algorithm, args and net runtime from an 'app/container' log file
    r1 = r'(SGDLR|PageRank|KMeans|Grep|Sort),.*\n'
    r2 = r'[0-9]{10}:.*\n'  # Runtime logging version 1 (first git commit)
    r3 = r'\[NetRuntime\]:[0-9]{13}:[0-9]{13}\n'  # version 2 (second git commit)
    algo, args = None, None
    start, end = None, None

    #sp.call(['sed', '-i', 's/text_175m_175m_100/text_175m_175mx100/g', log_file])

    with open(log_file) as f:
        for line in f:
            if re.search(r1, line):
                a = line.replace('\n','').split(',')
                algo, args = a[0], a[1:]
            if re.search(r2, line):
                if '[NetRuntimeStart]' in line: start = float(line.split(':')[0])
                if '[NetRuntimeEnd]' in line: end = float(line.split(':')[0])
            if re.search(r3, line):
                _, start, end = line.split(':')
                start, end = float(start)/1000, float(end) / 1000

    return algo, args, (end-start if start and end else None)


def get_gross_runtime(cluster_name, step_name):
    # Get gross runtime from a basic step log
    t = None
    with open(f'logs/{cluster_name}/{step_name}') as f:
        for line in f:
            if re.search('^INFO total process run time:.*', line):
                t = int(line.replace(' seconds\n','').split(':')[1])
    return t


def get_experiments(download_new=False):
    existing_log_dirs = get_existing_log_dirs()
    existing_log_files = get_existing_log_files()
    print(f"Locally saved log files: {len(existing_log_files)-2*len(existing_log_dirs)}")


    if download_new:
        s3 = boto3.client('s3')
        keys = get_all_s3_object_keys(s3)
        print(f"Total objects in s3://c3othesis: {len(keys)}")

        new_log_files = download_new_log_files(s3, keys, existing_log_files)
        print(f"Downloaded and unzipped {len(new_log_files)} new log files")

        cluster_info = extract_cluster_info()
        print(f"Meta data about {len(cluster_info)} clusters collected")

        create_cluster_info_logs(cluster_info)
        print(f"Meta data about {len(cluster_info)} clusters logged")

    clusters = extract_from_cluster_logs()

    experiments = []
    for cluster_name in get_existing_log_dirs():
        for file in sp.check_output(['ls', f'logs/{cluster_name}']) \
                      .decode('utf-8').split('\n')[:-1]:
            if file[0] == '0':  # To each app/container log, find corresp. step
                algo, args, net_runtime = \
                        extract_from_app_log(f'logs/{cluster_name}/{file}')
                mtype, nodes, steps = clusters[cluster_name]
                try:
                    step_name = steps[int(file)-1]  # Fails sometimes (since SGDLR experiments)
                    gross_runtime = get_gross_runtime(cluster_name, step_name) # Fails sometimes(even if above line worked)
                    experiments.append((cluster_name, step_name, algo, mtype,
                                        nodes, args, gross_runtime, net_runtime))
                except (IndexError, FileNotFoundError) as e:
                    '''
                    If the step is listed in 'cluster' but file does not exist.
                    Happens if a step was cancelled (when many previous steps
                    were also cancelled?)
                    '''
                    print(cluster_name, str(e)[0:100])




    print(f"EMR Steps submitted: {len(experiments)}")
    df = pd.DataFrame(experiments,
                      columns=('cluster_id', 'step_id', 'algorithm',
                               'machine_type', 'instance_count', 'args',
                               'gross_runtime', 'net_runtime'))

    return df

