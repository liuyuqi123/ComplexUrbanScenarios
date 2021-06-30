"""
Analyze test result.

Calculate the params range to ensure success rate over 90%.

"""

import os

import numpy as np
import json
import matplotlib.pyplot as plt
from interval import Interval


def run(speed_range=(10, 40), distance_range=(15, 50)):
    """

    """

    local_path = os.getcwd()
    local_path = os.path.join(local_path, 'scenario_results/run')

    files = os.listdir(local_path)

    print('')

    total_count_num = 0
    total_count_success = 0
    total_count_collision = 0
    total_count_time_exceed = 0

    for scenario in files:

        result_file = os.path.join(local_path, scenario, 'test_results.jsonl')

        try:
            with open(result_file, 'r') as f:
                data = []
                for line in f.readlines():
                    dic = json.loads(line)
                    data.append(dic)

        except:
            print('Fail to retrieve scenario result file: {}'.format(scenario))
            return -1

        # todo add args to set filter range
        _speed_range = Interval(speed_range[0], speed_range[1])
        _distance_range = Interval(distance_range[0], distance_range[1])

        single_count_num = 0
        single_count_success = 0
        single_count_collision = 0
        single_count_time_exceed = 0

        success_velo = []
        success_dist = []
        collision_velo = []
        collision_dist = []
        time_exceed_velo = []
        time_exceed_dist = []

        accumulative_duration = 0.

        for dic in data:

            velo = dic['velocity']
            dist = dic['distance']

            if velo not in _speed_range or dist not in _distance_range:
                continue

            # check if current param fit target range
            single_count_num += 1
            total_count_num += 1

            if dic['result'] == 'collision':
                collision_velo.append(velo)
                collision_dist.append(dist)
                single_count_collision += 1
                total_count_collision += 1

            elif dic['result'] == 'success':
                success_velo.append(velo)
                success_dist.append(dist)
                single_count_success += 1
                total_count_success += 1

                # only consider success result duration time
                accumulative_duration += dic['duration']

            else:  # time exceed
                time_exceed_velo.append(velo)
                time_exceed_dist.append(dist)
                single_count_time_exceed += 1
                total_count_time_exceed += 1

        # plot figure
        plt.figure()
        ax = plt.gca()
        ax.set_xlabel('velocity')
        ax.set_ylabel('distance')

        ax.scatter(success_velo, success_dist, c='g', s=25, alpha=0.5)
        ax.scatter(collision_velo, collision_dist, c='r', s=25, alpha=0.5, marker='s')
        ax.scatter(time_exceed_velo, time_exceed_dist, c='b', s=25, alpha=0.5, marker='^')

        plt.savefig('./{}.png'.format(scenario))
        plt.show()

        # print result of single scenario
        print(
            '=' * 60, '\n',
            'Scenario: ', scenario, '\n',
            'Counted Test Number: ', single_count_num, '\n',
            'Success Number: ', single_count_success, '\n',
            'Average Duration(success): ', accumulative_duration / single_count_num, '\n',
            'Collision Number: ', single_count_collision, '\n',
            'Time-exceed Number: ', single_count_time_exceed, '\n',
            'Success Rate: {:.2%}'.format(single_count_success / single_count_num), '\n',
            '=' * 60,
        )

        print('')

    # print total result
    print(
        '=' * 60, '\n',
        'Total Scenario Number: ', len(files), '\n',
        'Included Scenarios: ',
    )
    for item in files:
        print('-', item)
    print(
        '\n',
        'Speed Range: ', speed_range, '\n',
        'Distance Range: ', distance_range, '\n',
        'Counted Test Number: ', total_count_num, '\n',
        'Success Number: ', total_count_success, '\n',
        'Collision Number: ', total_count_collision, '\n',
        'Time-exceed Number: ', total_count_time_exceed, '\n',
        'Success Rate: {:.2%}'.format(total_count_success / total_count_num), '\n',
        '=' * 60,
    )


if __name__ == '__main__':

    run()
