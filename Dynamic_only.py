from os import urandom
from typing import Any

import pandas as pd
import subprocess
import numpy as np
import warnings

warnings.filterwarnings("ignore")

console = r"C:\Program Files\Aimsun\Aimsun Next 20\aconsole.exe"


def run_sim(replication, time_interval):
    subprocess.call([console, "-project", r"../Model/iMOVE_Deploy_Hybrid_20220527_v2.ang", "-cmd", "execute", "-target",
                     str(replication), "-matrixFile", f"assignment/assignment.txt", "-objectsFile",
                     "sections_list.txt"])


def read_aimsun_assignment(filepath):
    """
    :param filepath: Location of the assignment matrix file (Aimsun output)
    :return: pandas Dataframe: cleaned assignment matrix
    """
    assignment = open(filepath, "r")
    assignment = assignment.read()
    assignment = assignment.split("\n")
    assignment = [row.split(" ") for row in assignment]
    assignment_df = pd.DataFrame(assignment, columns=['Origin', 'Destination', 'Veh_Type_ID', 'Sec_ID',
                                                      'entranceIntervalIndex', 'currentIntervalIndex',
                                                      'Proportion', 'count', 'TravelTime'])
    assignment_df.drop(['Veh_Type_ID'], axis=1, inplace=True)
    assignment_df.drop(len(assignment_df) - 1, inplace=True)
    assignment_df['Origin'] = assignment_df['Origin'].astype(int)
    assignment_df['Destination'] = assignment_df['Destination'].astype(int)
    assignment_df['Sec_ID'] = assignment_df['Sec_ID'].astype(int)
    assignment_df['entranceIntervalIndex'] = assignment_df['entranceIntervalIndex'].astype(int)
    assignment_df['currentIntervalIndex'] = assignment_df['currentIntervalIndex'].astype(int)
    assignment_df['Proportion'] = pd.to_numeric(assignment_df['Proportion'])
    assignment_df['count'] = pd.to_numeric(assignment_df['count'])
    assignment_df['TravelTime'] = pd.to_numeric(assignment_df['TravelTime'])
    assignment_df['OD'] = (assignment_df['Origin'].astype(str) + assignment_df['Destination'].astype(str)).astype(
        'int64')
    return assignment_df


def add_cols(assignment_df):
    """
    :param assignment_df: Assignment matrix dataframe (cleaned)
    :return: pandas dataframe including ODs and True Counts
    """
    columns = assignment_df.columns.to_list()
    df = assignment_df[columns[-1:] + columns[2:3] + columns[4:5] + columns[3:4] + columns[5:6] + columns[6:7]]
    df['True Counts'] = round(df['count'] * df['Proportion'])
    df = pd.DataFrame(df.groupby(["OD", "Sec_ID"])['True Counts'].sum()).reset_index()

    return df


def calculate_proportions(filename, df):
    """
    :param filename: Supplied OD filepath
    :param df: cleaned assignment matrix dataframe with 'True Counts'
    :return: pandas dataframe (Assignment matrix including proportion)
    """
    od = pd.read_csv(filename, index_col='id')
    od.reset_index(inplace=True)
    od = od.melt(id_vars=['id'])
    od.drop(od[(od['id'] == 'Total') | (od['variable'] == 'Total')].index, inplace=True)
    od['OD'] = (od['id'].astype(str) + od['variable'].astype(str)).astype("int64")
    columns = od.columns.to_list()
    od = od[columns[-1:] + columns[2:3]]
    df['Prop'] = df['True Counts'] / df['OD'].map(od.set_index(['OD'])['value'])
    df['Prop'] = round(df['Prop'], 2)
    return od, df


def add_missing_values(object_filename, df, od):
    """
    :param object_filename: Object File path
    :param df: cleaned assignment matrix dataframe with proportions
    :param od: OD lookup
    :return: pandas dataframe
    """
    sections = (open(object_filename).read().split("\n"))
    sections = [int(sections[i]) for i in range(len(sections))]
    sections2 = df.Sec_ID.unique()
    missing_sections = [item for item in sections if item not in sections2]
    available_od = list(df.OD)
    missing_ods = [od for od in od.OD if od not in available_od]
    missing_sections = (missing_sections * (len(missing_ods) // len(missing_sections))) + missing_sections[
                                                                                          :len(missing_ods) % len(
                                                                                              missing_sections)]
    df2 = pd.DataFrame({'OD': missing_ods, 'Sec_ID': missing_sections, 'True Counts': [0] * len(missing_ods),
                        'Prop': [0] * len(missing_ods)})
    df = pd.concat([df, df2])
    df = df.sort_values(by=['OD', 'Sec_ID'])
    df = df.reset_index().drop("index", axis=1)
    df = df.drop_duplicates()
    return df


def convert_df_to_matrix(df: pd.DataFrame):
    """
    :param df: Pandas dataframe
    :return: matrix file
    """

    final_assignment = df.pivot(index='Sec_ID', columns='OD')[['Prop']]
    final_assignment.columns = range(final_assignment.columns.size)
    final_assignment = final_assignment.reset_index().drop("Sec_ID", axis=1)
    final_assignment = final_assignment.fillna(0)

    return final_assignment


def vectorize_od(input_od: pd.DataFrame):
    od = input_od.copy()
    od.reset_index(inplace=True)
    od = od.melt(id_vars=['id'])
    od.drop(od[(od['id'] == 'Total') | (od['variable'] == 'Total')].index, inplace=True)
    od['OD'] = (od['id'].astype(str) + od['variable'].astype(str)).astype("int64")
    columns = od.columns.to_list()
    od = od[columns[-1:] + columns[2:3]]
    od = od.sort_values("OD")
    od.drop("OD", axis=1, inplace=True)
    od.columns = range(od.columns.size)
    od.reset_index(inplace=True)
    od = od.drop("index", axis=1)
    return od


def output_od(vector_od):
    lookup = pd.read_csv("OD lookup.csv")
    lookup['count'] = vector_od[0]
    lookup = lookup.pivot(index='id', columns='Destination')[['count']]
    return lookup['count']


# Create Dynamic Assignments list
def create_dynamic_assignments(aimsun_assignment, object_file, time_intervals_mapper, interval_index, total_intervals=2):
    dynamic_assignments = [[] for _ in range(total_intervals)]
    for i in range(total_intervals):
        true_od = f"../Simulation ODs/{time_intervals_mapper[interval_index + i - 1]}.csv"
        for j in range(i, total_intervals):
            df = aimsun_assignment[
                (aimsun_assignment.entranceIntervalIndex == i) & (aimsun_assignment.currentIntervalIndex == j)]
            df = add_cols(df)
            od, df = calculate_proportions(true_od, df)
            df.loc[df['Prop'] == np.inf, 'Prop'] = 0
            df = add_missing_values(object_file, df, od)
            matrix = convert_df_to_matrix(df)
            dynamic_assignments[i].append(matrix)
    return dynamic_assignments


def check_bounds(x):
    if pd.isna(x) or x == np.inf or x == -np.inf:
        return 0
    return x


def get_observed_dynamic_flows(time_intervals_mapper, interval_index, cur_interval, pattern_id, date):
    flows = pd.read_csv(
        f"../Link flow data/{pattern_id}/{date}/{time_intervals_mapper[interval_index + cur_interval - 1]}.csv")
    flows.columns = range(flows.columns.size)
    return flows


def estimate_dynamic_flow(assignment_matrices, time_intervals_mapper, interval_index, cur_interval):
    est_flows = pd.DataFrame({0: [0] * 448})
    for ent_int in range(cur_interval+1):
        od = pd.read_csv(f"../Simulation ODs/{time_intervals_mapper[interval_index+ent_int-1]}.csv",
                         index_col="id")
        est_flows += assignment_matrices[ent_int][cur_interval - ent_int].dot(vectorize_od(od))
    return est_flows


# *************
def estimate_dynamic_flow_grad(priori_od, assignment_matrices, interval):
    est_flows = pd.DataFrame({0: [0] * 448})
    for ent_int in range(interval):
        est_flows += assignment_matrices[ent_int][interval - ent_int - 1].dot(priori_od[ent_int])
    return est_flows



def get_dynamic_od(time_intervals_mapper, interval_index):
    dynamic_od = []
    for i in range(2):
        od = vectorize_od(pd.read_csv(f"../Simulation ODs/{time_intervals_mapper[interval_index+ i-1]}.csv", index_col='id'))
        dynamic_od.append(od)
    return dynamic_od


# Evaluating the Objective function
def objective_function(assignment_matrices, time_intervals_mapper, interval_index, pattern_id, date):
    total_obj = 0
    for interval in range(2):
        estimated_counts = estimate_dynamic_flow(assignment_matrices, time_intervals_mapper, interval_index, interval)
        observed_counts = get_observed_dynamic_flows(time_intervals_mapper, interval_index , interval, pattern_id, date)
        obj = ((estimated_counts - observed_counts) ** 2)
        obj = obj.applymap(check_bounds)
        total_obj += 0.5 * (obj.sum())
    return total_obj


# Evaluating gradient of the Objective function
def gradient_interval(assignment_matrices, time_intervals_mapper, interval_index, pattern_id, date):
    gradients = []
    for ent_interval in range(2):
        obj = 0
        for cur_interval in range(2 - ent_interval):
            prtrans = (assignment_matrices[ent_interval][cur_interval]).transpose()
            estimate_flows = estimate_dynamic_flow(assignment_matrices, time_intervals_mapper, interval_index, cur_interval).fillna(0)
            observed_flows = get_observed_dynamic_flows(time_intervals_mapper, interval_index, cur_interval, pattern_id, date).fillna(0)
            g1 = (estimate_flows - observed_flows).applymap(check_bounds)
            obj += prtrans.dot(g1)
        gradients.append(obj)
    return gradients


def step_len_interval(dynamic_od, assignment_matrices, grad, time_intervals_mapper, interval_index, pattern_id, date):
    step_length: list[Any] = []
    grad_dynamic_od = [dynamic_od[i] * grad[i] for i in range(len(grad))]
    for interval in range(len(grad)):
        est = estimate_dynamic_flow(assignment_matrices, time_intervals_mapper, interval_index, interval).fillna(0)
        obs = get_observed_dynamic_flows(time_intervals_mapper, interval_index, interval, pattern_id, date).fillna(0)
        yderv_est = estimate_dynamic_flow_grad(grad_dynamic_od, assignment_matrices, interval + 1)
        diff = (est - obs)
        step = ((yderv_est * diff).sum() / (yderv_est ** 2).sum())
        step_length.append(step)
    return step_length


def update_ods(dynamic_od, time_interval):
    output = output_od(dynamic_od)
    output = output.applymap(lambda x: 0 if x < 0 or np.isnan(x) else round(x))
    output.to_csv(f"../Simulation ODs/{time_interval}.csv")
