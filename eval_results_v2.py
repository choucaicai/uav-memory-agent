
import os, json
from dataclasses import dataclass
import draccus
from collections import defaultdict 
import numpy as np  
from collections import defaultdict
import os
import re
# def kill_env_process(keyword):
#     result = subprocess.run(['pgrep', '-n', keyword], stdout=subprocess.PIPE)
#     cr_pid = result.stdout.decode().strip()
#     if len(cr_pid) > 0:
#         subprocess.run(['kill', '-9', cr_pid])


COLLISION_FIELDS = ('collision', 'collision_count', 'collision_info')


def has_collision_record(row):
    """Return whether a row uses the collision-aware result schema."""
    return any(field in row for field in COLLISION_FIELDS)


def row_has_collision(row):
    """Normalize the collision flag across old and collision-aware result files."""
    if 'collision' in row:
        value = row['collision']
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 'yes', 'y'}
        return bool(value)
    if 'collision_count' in row:
        try:
            return float(row['collision_count']) > 0
        except (TypeError, ValueError):
            return bool(row['collision_count'])
    return row.get('collision_info') is not None


def collision_metrics(rows):
    """Compute collision metrics only for result rows that record collisions."""
    if not any(has_collision_record(row) for row in rows):
        return {}
    collision_count = sum(row_has_collision(row) for row in rows)
    return {
        'collision_count': collision_count,
        'collision_rate': collision_count / len(rows) if rows else 0.0,
    }


def _format_metric_row(label, metrics, show_collision):
    line = (f"{label:<20} {metrics['num']:<10} {metrics['acc']:<10.4f} "
            f"{metrics['osr']:<10.4f} {metrics['ne']:<10.4f} "
            f"{metrics['spl']:<10.4f} {metrics['spl_num']:<10.4f}")
    if show_collision:
        line += f" {metrics.get('collision_rate', 0.0):<16.4f}"
    return line


def print_results(results):
    """Print aggregate metrics and include collision rate when it is recorded."""
    total = results['total_results']
    show_collision = any(
        'collision_rate' in metrics
        for metrics in [
            total,
            *results.get('seen_metric', {}).values(),
            *results.get('size_metric', {}).values(),
        ]
    )
    collision_header = f" {'Collision Rate':<16}" if show_collision else ''
    separator_width = 96 if show_collision else 80

    print(f"{'Category':<20} {'Num':<10} {'Acc':<10} {'OSR':<10} {'NE':<10} "
          f"{'SPL':<10} {'SPL NUM':<10}{collision_header}")
    print("-" * separator_width)

    print(_format_metric_row('Total', total, show_collision))

    print("\n" + "=" * separator_width)
    print("Seen Metric:")
    print("-" * separator_width)
    for key, metrics in results['seen_metric'].items():
        print(_format_metric_row(f"  {key}", metrics, show_collision))

    print("\n" + "=" * separator_width)
    print("Size Metric:")
    print("-" * separator_width)
    for key, metrics in results['size_metric'].items():
        print(_format_metric_row(f"  {key}", metrics, show_collision))

def process_results(env_result_items_dict):
    default_seen = defaultdict(list)
    default_size = defaultdict(list)
    
    total_items = [item  for env,items in env_result_items_dict.items() for item in items]
    def cal_metric(total_rows):
        spl = 0
        spl_num = 0
        if len(total_rows) > 0:
            for row in total_rows:
                p_i = len(row['gen_pose_list'])
                if row.get('action',None) is None:
                    continue
                l_i = len(row['action'])
                success = row['acc']  # 或者根据实际情况计算距离
                spl_num+=1
                if success:
                    spl += l_i / max(p_i, l_i)
        spl = spl / len(total_rows) if len(total_rows) > 0 else 0     
        metrics = {
            'num' : len(total_rows),
            'acc' : sum([cur['acc'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'osr' : sum([cur['osr'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'ne'  : sum([cur['ne'] for cur in total_rows]) / len(total_rows) if len(total_rows) > 0 else 0,
            'spl': spl,
            'spl_num':spl_num,
        }
        metrics.update(collision_metrics(total_rows))
        return metrics
    total_results = cal_metric(total_items)
    for row in total_items:
        size_str = row['size']
        size = re.search(r'^\s*([a-zA-Z]+)', size_str.strip()).group(1)
        is_seen = row["used-in-train"]
        default_seen[is_seen].append(row)
        default_size[size].append(row)

    seen_metric = {
        k:cal_metric(item_list) for k,item_list in default_seen.items()
    }
    size_metric = {
        k:cal_metric(item_list) for k,item_list in default_size.items()
    }
    return dict(
        total_results = total_results,
        seen_metric = seen_metric,
        size_metric = size_metric
    )
@dataclass
class EvalConfig:
    output_folder :str = "results/openfly-agent-uavon-v1-single"
    dataset_path :str = "/home/zzz/code/UAV_ON/DATASET/UAV-ON-data/valset/merged_episodes_with_action.json"
    samples_path :str =None
    
base_dict = {
    "Barnyard_test": {
      "env": "Barnyard_test",
      "result": {
        "num": 94,
        "acc": 0.06382978723404255,
        "osr": 0.14893617021276595,
        "ne": 63.04184520378441
      }
    },
    "BrushifyRoad_test": {
      "env": "BrushifyRoad_test",
      "result": {
        "num": 70,
        "acc": 0.014285714285714285,
        "osr": 0.02857142857142857,
        "ne": 63.00606715193599
      }
    },
    "BrushifyUrban_test": {
      "env": "BrushifyUrban_test",
      "result": {
        "num": 60,
        "acc": 0.18333333333333332,
        "osr": 0.3333333333333333,
        "ne": 43.23107818858242
      }
    },
    "CabinLake_test": {
      "env": "CabinLake_test",
      "result": {
        "num": 51,
        "acc": 0.27450980392156865,
        "osr": 0.39215686274509803,
        "ne": 39.20510131247494
      }
    },
    "CityPark_test": {
      "env": "CityPark_test",
      "result": {
        "num": 50,
        "acc": 0.18,
        "osr": 0.32,
        "ne": 49.91041766394905
      }
    },
    "CityStreet_test": {
      "env": "CityStreet_test",
      "result": {
        "num": 95,
        "acc": 0.12631578947368421,
        "osr": 0.2,
        "ne": 46.058305748070126
      }
    },
    "DownTown_test": {
      "env": "DownTown_test",
      "result": {
        "num": 49,
        "acc": 0.32653061224489793,
        "osr": 0.3877551020408163,
        "ne": 34.785202703789466
      }
    },
    "NYC_test": {
      "env": "NYC_test",
      "result": {
        "num": 152,
        "acc": 0.16447368421052633,
        "osr": 0.24342105263157895,
        "ne": 43.52715007457458
      }
    },
    "ModularNeighborhood_test": {
      "env": "ModularNeighborhood_test",
      "result": {
        "num": 50,
        "acc": 0.08,
        "osr": 0.18,
        "ne": 50.826772003064576
      }
    },
    "Slum_test": {
      "env": "Slum_test",
      "result": {
        "num": 98,
        "acc": 0.16326530612244897,
        "osr": 0.2755102040816326,
        "ne": 42.697440896650406
      }
    },
    "UrbanJapan_test": {
      "env": "UrbanJapan_test",
      "result": {
        "num": 55,
        "acc": 0.18181818181818182,
        "osr": 0.36363636363636365,
        "ne": 49.52424540166573
      }
    },
    "Venice_test": {
      "env": "Venice_test",
      "result": {
        "num": 50,
        "acc": 0.1,
        "osr": 0.18,
        "ne": 48.8173408567516
      }
    },
    "WesternTown_test": {
      "env": "WesternTown_test",
      "result": {
        "num": 47,
        "acc": 0.14893617021276595,
        "osr": 0.3404255319148936,
        "ne": 45.08987765847671
      }
    },
    "WinterTown_test": {
      "env": "WinterTown_test",
      "result": {
        "num": 79,
        "acc": 0.13924050632911392,
        "osr": 0.26582278481012656,
        "ne": 55.544993692429635
      }
    }
}
@draccus.wrap()
def eval_uav(cfg:EvalConfig):
    output_folder = cfg.output_folder
    # with open(os.path.join(output_foler,'temp_results_samples.json')) as f:
        # result_items_list = json.load(f)
    temp_eavl_save_folder = os.path.join(output_folder,'temp')
    dataset_data = json.load(open(cfg.dataset_path,'r'))
    dataset_data_dict = {
        row['episode_id']:row for row in dataset_data
    }
    if cfg.samples_path is not None:
        all_results = json.load(open(cfg.samples_path,'r'))
        for k,v in all_results.items():
            for data_row in v:
                data_row['action'] = dataset_data_dict[data_row['episode_id']]['action']
    else:
        all_results = defaultdict(list)
        for json_file in os.listdir(temp_eavl_save_folder):
            if json_file.endswith('.json'):
                with open(os.path.join(temp_eavl_save_folder, json_file)) as f:
                    data_row = json.load(f)
                    data_row['action'] = dataset_data_dict[data_row['episode_id']]['action']
                    all_results[data_row['map_name']].append(data_row)
                    
    final_results = process_results(all_results)
    # with open(os.path.join(output_foler,'results.json'),'a+') as f:
    #     json.dump(final_results,f,indent=2)   
    print_results(final_results)
    env_results = {}
    for k,all_env_result in all_results.items():
        temp_result =  {
            'num' : len(all_env_result),
            'acc' : sum([cur['acc'] for cur in all_env_result]) / len(all_env_result) if len(all_env_result) > 0 else 0,
            'osr' : sum([cur['osr'] for cur in all_env_result]) / len(all_env_result) if len(all_env_result) > 0 else 0,
            'ne'  : sum([cur['ne'] for cur in all_env_result]) / len(all_env_result) if len(all_env_result) > 0 else 0,
        }
        temp_result.update(collision_metrics(all_env_result))
        acc_delta = temp_result['acc'] -  base_dict[k]['result']['acc']
        env_line = (f"ENV: {k},num: {temp_result['num']},SR: {temp_result['acc']},"
                    f"OSR: {temp_result['osr']}, NE: {temp_result['ne']}")
        if 'collision_rate' in temp_result:
            env_line += f", Collision Rate: {temp_result['collision_rate']}"
        print(env_line, acc_delta)
        env_results[k]={
        'env':k,
        'result': temp_result
    }
        
    final_results['env_results'] = env_results
    with open(os.path.join(output_folder,'results.json'),'w') as f:
        json.dump(final_results,f,indent=2)   

if __name__ == '__main__':
    eval_uav()
