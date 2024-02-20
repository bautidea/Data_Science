import re
from unidecode import unidecode
import numpy as np
import pandas as pd
import os
# %% [markdown]
# ### Imports

# %%
# Analysis and wrangling.

# %%
print('Initializing Data Wrangling...')

# %% [markdown]
# ### Loading data / brief analysis.

# %%
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'data', 'challenge_MLE.csv')

student_df = pd.read_csv(file_path, sep=';')

# Transforming epoch to datetime
student_df['fecha_mesa_epoch'] = pd.to_datetime(
    student_df['fecha_mesa_epoch'], unit='s')
student_df['ass_created_at'] = pd.to_datetime(
    student_df['ass_created_at'], unit='s')
student_df['ass_due_at'] = pd.to_datetime(student_df['ass_due_at'], unit='s')
student_df['ass_unlock_at'] = pd.to_datetime(
    student_df['ass_unlock_at'], unit='s')
student_df['ass_lock_at'] = pd.to_datetime(student_df['ass_lock_at'], unit='s')
student_df['s_submitted_at'] = pd.to_datetime(
    student_df['s_submitted_at'], unit='s')
student_df['s_graded_at'] = pd.to_datetime(student_df['s_graded_at'], unit='s')
student_df['s_created_at'] = pd.to_datetime(
    student_df['s_created_at'], unit='s')

# Arranging time column.
student_df['periodo'] = student_df['periodo'].str.replace('01', '1')

# Droping duplicates.
student_df.drop_duplicates(inplace=True)

# %%
# Eliminating white spaces at the begging and end of the string.
student_df['ass_name'] = student_df['ass_name'].str.strip()
# Replacing double spaces with single.
student_df['ass_name'] = student_df['ass_name'].str.replace('  ', ' ')
# Bringing all text to lower case.
student_df['ass_name'] = student_df['ass_name'].str.lower()
# Removing ticks from strings.


def remove_ticks(txt):
    if pd.notnull(txt):
        return unidecode(txt)
    else:
        return txt


student_df['ass_name'] = student_df['ass_name'].apply(remove_ticks)

# %%
# Eliminating white spaces at the begging and end of the string.
student_df['ass_name_sub'] = student_df['ass_name_sub'].str.strip()
# Replacing double spaces with single.
student_df['ass_name_sub'] = student_df['ass_name_sub'].str.replace('  ', ' ')
# Bringing all text to lower case.
student_df['ass_name_sub'] = student_df['ass_name_sub'].str.lower()

student_df['ass_name_sub'] = student_df['ass_name_sub'].apply(remove_ticks)

# %% [markdown]
# # Data Wrangling.

# %% [markdown]
# im going top start by creating a unique variable that im going to use as index, by using the features user_uuid/course_uuid.

# %%
student_df['index'] = student_df['user_uuid'] + '/' + student_df['course_uuid']

# %%
# Creating a new DF in which im going to concatenate the created features.
data = student_df.groupby(['index', 'nota_final_materia', 'periodo']).size(
).reset_index()[['index', 'nota_final_materia', 'periodo']]

data.index = data['index'].values
data.drop(columns='index', inplace=True)

# %% [markdown]
# ## Assignments.
# ### Appending information about assignments.

# %% [markdown]
# Im going to append to the created DF the information about differents assignments students have to do during the course.
#
# I've seen that a lot of assignments in the column 'ass_name' share the abbreviated name and number of the homework, like:
# * [tp1]
# * [tp2]
# * [api1]
# * etc.
#
# Ill try to separate that logic and create a column indicating the homework each student did.

# %%
# First ill need to bring all different assignments to same type.
# Im going to use regex.
pattern = r"\[(\w+)\]"
pattern_re = re.compile(pattern)

# This function will be applied with .apply(lambda) it will return the 1st
# element that match the given pattern.


def extract_assessment_type(text):
    if pd.isnull(text):
        return np.nan

    pattern_search = pattern_re.search(text)
    # return pattern_search.group(1) if pattern_search else text
    return pattern_search.group(1) if pattern_search else text


# %%
# Im going to separate between assignments that were assigned and were submitted.
student_df['assigned_ass'] = student_df['ass_name'].copy()
student_df['submitted_ass'] = student_df['ass_name_sub'].copy()

# Obtaining assignments types.
student_df['assigned_ass'] = student_df['ass_name'].apply(
    lambda x: extract_assessment_type(x))
student_df['submitted_ass'] = student_df['ass_name_sub'].apply(
    lambda x: extract_assessment_type(x))

# %% [markdown]
# Based on the leftovers:
# * for the case of assigned assessments im going to keep those assessments that are important, like 'TPs', 'APIs' and 'EDs'.
# * As for submitted assignments im going to gather the remains into a single category 'otras_entregas'.
#
# Im doing this because the remaining assignments are not that many, and the name of them doesnt appear to imply importance.
#
# Th only assignment im going to drop from 'ass_name_sub' is 'evaluacion diagnostica' because that is usually an important assessment.

# %%
student_df['submitted_ass'] = student_df['submitted_ass'].replace(
    'evaluacion diagnostica', np.nan)

# %%
assignments_names = ['api1', 'api2', 'api3', 'api4',
                     'ed3', 'ed4', 'tp1', 'tp2', 'tp3', 'tp4']

# First im going to ignore all assigned assessments that are not in the list.
mask_assigned_tps = ~student_df['assigned_ass'].isin(
    assignments_names) & student_df['assigned_ass'].notnull()
student_df.loc[mask_assigned_tps, 'assigned_ass'] = np.nan

# Now im going to replace assignments that has no relevant importance and assigned to a single category.
mask_submitted_tps = ~student_df['submitted_ass'].isin(
    assignments_names) & student_df['submitted_ass'].notnull()
student_df.loc[mask_submitted_tps, 'submitted_ass'] = 'otros_trabajos'

# %% [markdown]
# ### Obtaining assignments for each student

# %%
# Appending course information to my data in order to merge course information.
course_name_df = student_df[['index', 'course_uuid']].copy().drop_duplicates()
data = data.merge(course_name_df, how='left', left_index=True,
                  right_index=False, right_on='index')
data.index = data['index'].values
data.drop(columns='index', inplace=True)

# %%
# Assignments to consider.
assessment_cols = student_df.loc[student_df['submitted_ass'].notnull(
), 'submitted_ass'].unique()

# Defining a function to flag assignments that dont belong to a course.


def flag_unrequited_assignments(df, submit_time=False):
    # Unique assignments for each course.
    unique_assignments_per_course = student_df[student_df['submitted_ass'].notnull(
    )].groupby('course_uuid')['submitted_ass'].unique()

    # --> ['api1', 'tp1', 'api2', 'api4', 'tp2', 'api3', 'tp4', 'tp3', 'otros_trabajos', 'ed3', 'ed4']
    for assignment in assessment_cols:
        # Iterating over possible assignments that a course can have.
        # This element is an series that each row is a unique course with possible assignments a course can have.
        # For example.
        # ---
        # 0034afe6-e996-4c26-b0b9-24dbb9535465    [api1, api2, api3, api4, ed4, ed3, tp1, tp2]
        # f08bfeb75-fb35-4ac4-8916-1f7eda676892                           [tp1, tp2, tp3, tp4]
        # ---
        if (submit_time and assignment == 'otros_trabajos'):
            continue

        for course_uuid, possible_assignment_list in unique_assignments_per_course.items():

            if submit_time:
                col_name = f'{assignment}_time_to_submit[days]'
            else:
                col_name = assignment

            # Then if the assignment is not present in possible assignment of the series.
            if assignment not in possible_assignment_list:
                # Keeping considered course.
                mask = df['course_uuid'] == course_uuid
                # And flagging the corresponding assignment column as -1.
                df.loc[mask, col_name] = -1

    return df


# %%
assignment_df = student_df.groupby(
    ['index', 'submitted_ass']).size().reset_index()

# Creating a pivot table in order to transform the different assignments into columns.
assignment_df = assignment_df.pivot(
    index='index', columns='submitted_ass', values=0).reset_index().rename_axis(None, axis=1)

# Droping index col, and assigning it as index.
assignment_df.index = assignment_df['index'].values
assignment_df.drop(columns='index', inplace=True)

# Merging information.
data = data.merge(assignment_df, left_index=True, right_index=True, how='left')

# Filling nans with 0, this means that the student has no information about that assessment.
# And transforming type into int for those columns.
for col in assessment_cols:
    data[col] = data[col].fillna(0)
    data[col] = data[col].astype(int)

# %%
data = flag_unrequited_assignments(data)

# %% [markdown]
# ### Obtaining scores for each assignment submission.

# %%
assignment_score_df = student_df.groupby(['index', 'submitted_ass'])[
    'score'].mean().reset_index()
assignment_score_df['submitted_ass'] = assignment_score_df['submitted_ass'] + '_score'

# Creating a pivot table in order to transform the different assignments into columns.
assignment_score_df = assignment_score_df.pivot(
    index='index', columns='submitted_ass', values='score').reset_index().rename_axis(None, axis=1)

# Droping index col, and assigning it as index.
assignment_score_df.index = assignment_score_df['index'].values
assignment_score_df.drop(columns='index', inplace=True)

# Merging information.
data = data.merge(assignment_score_df, left_index=True,
                  right_index=True, how='left')

# Filling nans with 0, this means that the student has no information about that assessment.
# And transforming type into int for those columns.
for col in assessment_cols:
    name = f'{col}_score'
    data[name] = data[name].fillna(0)
    data[name] = data[name].astype(float)
    data[name] = np.round(data[name], 2)

# %% [markdown]
# ### Obtaining required time for a student to finish an assignment.

# %%
# First obtaining when each assignment is unlocked for each course.
ass_unlock_df = student_df.groupby(['course_uuid', 'assigned_ass'])[
    'ass_created_at'].first().reset_index()
ass_unlock_df['assigned_ass'] = ass_unlock_df['assigned_ass'] + \
    '_assigned_date'
ass_unlock_df = ass_unlock_df.pivot(index='course_uuid', columns='assigned_ass',
                                    values='ass_created_at').reset_index().rename_axis(None, axis=1)

# %%
# Obtaining the date each student delivered each assignment.
ass_submitted_df = student_df.groupby(['index', 'submitted_ass'])[
    's_submitted_at'].first().reset_index()
# Ignoring general assignments.
ass_submitted_df = ass_submitted_df[ass_submitted_df['submitted_ass']
                                    != 'otros_trabajos'].copy()
ass_submitted_df['submitted_ass'] = ass_submitted_df['submitted_ass'] + \
    '_submitted_date'
ass_submitted_df = ass_submitted_df.pivot(
    index='index', values='s_submitted_at', columns='submitted_ass').reset_index().rename_axis(None, axis=1)

ass_submitted_df.index = ass_submitted_df['index'].values
ass_submitted_df.drop(columns='index', inplace=True)

# %%
# Creating a copy to obtain the time.
assignment_submit_time = data[['course_uuid']].copy()

# Merging information.
assignment_submit_time = assignment_submit_time.merge(
    ass_submitted_df, left_index=True, right_index=True, how='left')
assignment_submit_time['index'] = assignment_submit_time.index.values

assignment_submit_time = assignment_submit_time.merge(
    ass_unlock_df, how='left', on='course_uuid')

assignment_submit_time.index = assignment_submit_time['index'].values
assignment_submit_time.drop(columns='index', inplace=True)

# %%
submit_time_cols = []
# Obtaining time intervals.
for col in assessment_cols:
    if col == 'otros_trabajos':
        continue

    assigned_date = f'{col}_assigned_date'
    submitted_date = f'{col}_submitted_date'

    time_delta_col_name = f'{col}_time_to_submit[days]'
    submit_time_cols.append(time_delta_col_name)

    time_delta_serie = assignment_submit_time[submitted_date] - \
        assignment_submit_time[assigned_date]
    assignment_submit_time[time_delta_col_name] = np.round(
        time_delta_serie.dt.total_seconds() / (3600 * 24), 1)  # Convert to days

# Doing this just for keeping the course_uuid.
submit_time_cols.insert(0, 'course_uuid')

assignment_submit_time = assignment_submit_time[submit_time_cols]
submit_time_cols.remove('course_uuid')

# %%
# Going to flag those assignments that doesnt correspond to a given course.
assignment_submit_time = flag_unrequited_assignments(
    assignment_submit_time, submit_time=True)
# Filling leftovers with 0, in order to identify those assignments that dont have a submitted date.
assignment_submit_time.fillna(0, inplace=True)
# Droping to merge.
assignment_submit_time.drop(columns='course_uuid', inplace=True)


# %%
# Merging information
data = data.merge(assignment_submit_time, left_index=True,
                  right_index=True, how='left')

# %% [markdown]
# ### Obtaining submission types for each student.

# %%
submission_type_df = student_df.groupby(
    ['index', 'submission_type']).size().reset_index()
submission_type_df = submission_type_df.pivot(
    index='index', columns='submission_type', values=0).reset_index().rename_axis(None, axis=1)

# Droping index col, and assigning it as index.
submission_type_df.index = submission_type_df['index'].values
submission_type_df.drop(columns='index', inplace=True)

# Obtaining column name.
submission_type_cols = submission_type_df.columns.to_list()

data = data.merge(submission_type_df, left_index=True,
                  right_index=True, how='left')

for col in submission_type_cols:
    data[col] = data[col].fillna(0)
    data[col] = data[col].astype(int)

    rename = {col: f'{col}_submits'}
    data.rename(columns=rename, inplace=True)

# %% [markdown]
# ### Apending exams information

# %%
exams_df = student_df.groupby(['index', 'nombre_examen'])[
    'nota_parcial'].mean().reset_index()
exams_df['nombre_examen'] = exams_df['nombre_examen'].str.lower()
exams_df['nombre_examen'] = exams_df['nombre_examen'].str.replace(
    r'\(\d+\)', '', regex=True)
exams_df['nombre_examen'] = exams_df['nombre_examen'].str.replace(' ', '_')
exams_df = exams_df.pivot(index='index', values='nota_parcial',
                          columns='nombre_examen').reset_index().rename_axis(None, axis=1)

# %%
# Obtaining information about the 'evaluacion diagnositca' that i ignore when dealing with the columns 'ass_name_sub'.
diagnostic_exams_df = student_df.loc[student_df['ass_name_sub']
                                     == 'evaluacion diagnostica', ['index', 'ass_name_sub', 'score']]
diagnostic_exams_df = diagnostic_exams_df.groupby(
    ['index', 'ass_name_sub', 'score']).mean().reset_index()
diagnostic_exams_df['ass_name_sub'] = diagnostic_exams_df['ass_name_sub'].str.replace(
    ' ', '_')
diagnostic_exams_df = diagnostic_exams_df.pivot(
    index='index', values='score', columns='ass_name_sub').reset_index().rename_axis(None, axis=1)

# %%
# Merging information
exams_df = exams_df.merge(diagnostic_exams_df, how='left', on='index')

# Droping index col, and assigning it as index.
exams_df.index = exams_df['index'].values
exams_df.drop(columns='index', inplace=True)

# Obtaining column name.
exams_col = exams_df.columns.to_list()

data = data.merge(exams_df, left_index=True, right_index=True, how='left')

for col in exams_col:
    if col != 'evaluacion_diagnostia':
        data[col] = data[col].fillna(-1)
        data[col] = data[col].astype(int)
    else:
        data[col] = data[col].fillna(-1)
        data[col] = data[col].astype(float)
        data[col] = np.round(data[col], 2)

# %%
# There are some cases in which the score of the exam has value -1, and the score of the
# 2nd instance of the exam is valid.
# Im going to invert those values.
mask = (data['primer_parcial'] < 0) & (
    data['recuperatorio_primer_parcial'] > 0)
data.loc[mask, 'primer_parcial'] = data.loc[mask,
                                            'recuperatorio_primer_parcial']
data.loc[mask, 'recuperatorio_primer_parcial'] = -1

# %% [markdown]
# ### Rearranging

# %%
# Defining if the course has been approved or not.
data['materia_aprobada'] = data['nota_final_materia'].apply(
    lambda x: 1 if x >= 4 else 0)

# %%
# Concatenating different assessments the word '_score' in order to obtain the name of the score columns.
score_cols = (assessment_cols + '_score').tolist()

# Appending the score columns to the assessment columns
columns_order = assessment_cols.tolist() + score_cols
# Appending time to submit cols.
columns_order = columns_order + submit_time_cols

# Sorting alphabetically
columns_order.sort()

# Appending submissions types.
columns_order = columns_order + \
    (pd.Series(submission_type_cols) + '_submits').tolist()

# Appending exams columns.
columns_order = exams_col + columns_order

columns_order.insert(0, 'materia_aprobada')
columns_order.insert(1, 'nota_final_materia')
columns_order.insert(2, 'periodo')

data = data[columns_order].copy()

# %% [markdown]
# There are some students that have no information about exams, im going to drop those students from my DF, because they could worsen the model performance.

# %%
mask = (
    (data['integrador'] == -1) &
    (data['primer_parcial'] == -1) &
    (data['recuperatorio_primer_parcial'] == -1) &
    (data['recuperatorio_segundo_parcial'] == -1) &
    (data['segundo_parcial'] == -1) &
    (data['evaluacion_diagnostica'] == -1)
)

# %% [markdown]
# Another thing to do is to convert the variable 'periodo' into a dummy variable, since we know that there are only two periods '1-2022' and '2-2022', im going to flag as 1 the '1-2022' period.

# %%
data['periodo'] = data['periodo'].replace('1-2022', 1)
data['periodo'] = data['periodo'].replace('2-2022', 0)
data.rename(columns={'periodo': 'periodo_1-2022'}, inplace=True)

data = data[~mask].copy()
# data.drop(columns='nota_final_materia', inplace=True)

# %% [markdown]
# Also there are a few students that got 0 as the final score, but when you look at the scores they seems that they might have approved the course.
# Im going to remove them from the final data.

# %%
mask = data['nota_final_materia'] == 0
data = data[~mask].copy()

# %% [markdown]
# # Exporting processed data

# %%
output_dir = os.path.join(script_dir, 'output', 'processed')
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'challenge_processed.csv')

data.to_csv(output_file, index=True)

print('Finalized with Data Wrangling, new file with processed data created at "output/processed/"')
