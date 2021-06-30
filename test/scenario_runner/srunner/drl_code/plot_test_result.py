"""
Plot result of scatter
"""

import os

import numpy as np
import json
import matplotlib.pyplot as plt


result_path = ''

# read jsonl by lines
jsonl_file = os.path.join(result_path, 'test_results.jsonl')
with open(jsonl_file, 'r') as f:
    data = []
    for line in f.readlines():
        dic = json.loads(line)
        data.append(dic)

total_count = 0
success_count = 0
collision_count = 0
time_exceed_count = 0

success_velo = []
success_dist = []
collision_velo = []
collision_dist = []
time_exceed_velo = []
time_exceed_dist = []

for dic in data:

    total_count += 1

    velo = dic['velocity']
    dist = dic['distance']

    if dic['result'] == 'collision':
        collision_velo.append(velo)
        collision_dist.append(dist)
        collision_count += 1
    elif dic['result'] == 'success':
        success_velo.append(velo)
        success_dist.append(dist)
        success_count += 1
    else:  # time exceed
        time_exceed_velo.append(velo)
        time_exceed_dist.append(dist)
        time_exceed_count += 1

plt.figure()
ax = plt.gca()
ax.set_xlabel('velocity')
ax.set_ylabel('distance')

ax.scatter(success_velo, success_dist, c='g', s=15, alpha=0.5)
ax.scatter(collision_velo, collision_dist, c='r', s=15, alpha=0.5)
ax.scatter(time_exceed_velo, time_exceed_dist, c='b', s=15, alpha=0.5)

plt.savefig(os.path.join(result_path, 'result_stats.png'))
plt.show()

print(
    '=' * 60, '\n',
    'Total Test Number: ', total_count, '\n',
    'Success Number: ', success_count, '\n',
    'Collision Number: ', collision_count, '\n',
    'Time exceed Number: ', time_exceed_count, '\n',
    'Success rate: {:.2%}'.format(success_count / total_count), '\n',
    '=' * 60,
)
