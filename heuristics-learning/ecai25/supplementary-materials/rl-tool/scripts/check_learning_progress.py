import argparse
import os
import pandas as pd
import json


def main():
    """
    For each learning run, prints if the training is finished or not, and in the latter case prints the number of episodes already done
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-i', required=True, type=str)

    args, _ = parser.parse_known_args()
    
    basedir = os.path.abspath(args.i)
    
    for s in os.listdir(os.path.join(basedir, 'instances_sets')):
        set_path = os.path.abspath(os.path.join(basedir, 'instances_sets', s))
        for c in os.listdir(os.path.join(set_path, 'learning_configurations')):
            config_path = os.path.join(set_path, 'learning_configurations', c)
            learning_json = os.path.join(config_path, 'learning.json')
            with open(learning_json) as f:
                learning_config = json.load(f)
            for r in os.listdir(os.path.join(config_path, 'runs')):
                run_path = os.path.join(set_path, 'learning_configurations', c, 'runs', r)
                log_file = os.path.join(run_path, 'episodes.log.csv')
                df = pd.read_csv(log_file)
                last_episode = df['episode'].iloc[-1]
                tot_episodes = learning_config['max_episodes']
                if last_episode==tot_episodes:
                    print(s, c, r, ":\t finished")
                else:
                    print(s, c, r, ":\t done", last_episode, "out of", tot_episodes, "\t(", round(last_episode/tot_episodes*100), "% )")

if __name__ == '__main__':
    main()
