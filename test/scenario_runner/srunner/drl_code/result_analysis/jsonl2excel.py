"""
todo get max_acc and ttc to the excel file as well

Extract result info from jsonl file and save it into excel file.
"""

import os

import json
import xlsxwriter
from interval import Interval


def result2excel(speed_range=(10, 40), distance_range=(15, 50)):
    """
    Main func.
    """

    # todo add args to set filter range
    _speed_range = Interval(speed_range[0], speed_range[1])
    _distance_range = Interval(distance_range[0], distance_range[1])

    local_path = os.getcwd()
    local_path = os.path.join(local_path, 'scenario_results/run')

    files = os.listdir(local_path)

    # excel file
    wb = xlsxwriter.Workbook('Result.xlsx')

    for scenario in files:

        # get json file
        result_file = os.path.join(local_path, scenario, 'test_results.jsonl')

        with open(result_file, 'r') as f:
            data = []
            for line in f.readlines():
                dic = json.loads(line)
                data.append(dic)

        # create a worksheet for each scenario result
        worksheet = wb.add_worksheet(name=scenario)

        index = 0
        for dic in data:

            velo = dic['velocity']
            dist = dic['distance']

            if velo not in _speed_range or dist not in _distance_range:
                continue

            # get row col index
            row = index
            index += 1

            # write content
            # velocity
            col = 0
            worksheet.write(row, col, velo)

            # distance
            col = 1
            worksheet.write(row, col, dist)

            # test result
            result = dic['result']
            if result == 'success':
                worksheet.write(row, 2, 1)
                worksheet.write(row, 3, 0)
                worksheet.write(row, 4, 0)
            elif result == 'collision':
                worksheet.write(row, 2, 0)
                worksheet.write(row, 3, 1)
                worksheet.write(row, 4, 0)
            else:  # time exceed
                worksheet.write(row, 2, 0)
                worksheet.write(row, 3, 0)
                worksheet.write(row, 4, 1)

            # duration time
            col = 5
            duration = dic['duration']
            worksheet.write(row, col, duration)

    # excel file will be generated by close() method
    wb.close()


if __name__ == '__main__':

    result2excel()
