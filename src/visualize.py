import pandas as pd
# import matplotlib.pyplot as plt
import streamlit as st

# Read the data from the CSV file
data = pd.read_csv('target/result_columns.csv')

st.set_page_config(layout="wide")
samples = st.sidebar.slider("Samples: ", 1,  5, 3, 1)
top_results_to_compare = st.sidebar.slider("Top results to compare: ", 1,  10, 5, 1)
fairness_considuration_ratio = st.sidebar.slider("Fairness Consideration Ratio: ", 0.1,  0.5, 0.2, 0.1)
auc_col, similarity_between_fair_and_regular_predictions_col, fairness_score_of_fair_predictions_col, fairness_score_of_regular_predictions_col = st.columns(4)
# auc_col, similarity_between_fair_and_regular_predictions_col = st.columns(2)
data_samples = data[(data['fairness_considuration_ratio'] == fairness_considuration_ratio) & (data['top_results_to_compare'] == top_results_to_compare)]
data_top_results_to_compare = data[(data['samples'] == samples) & (data['fairness_considuration_ratio'] == fairness_considuration_ratio)]
data_fairness_considuration_ratio = data[(data['samples'] == samples) & (data['top_results_to_compare'] == top_results_to_compare)]
samples_auc_chart_data = pd.DataFrame(
   {
       "samples": data_samples['samples'],
       "auc": data_samples['auc'],
       "type": data_samples['type'],
   }
)
top_results_to_compare_auc_chart_data = pd.DataFrame(
   {
       "top_results_to_compare": data_top_results_to_compare['top_results_to_compare'],
       "auc": data_top_results_to_compare['auc'],
       "type": data_top_results_to_compare['type'],
   }
)
fairness_considuration_ratio_auc_chart_data = pd.DataFrame(
   {
       "fairness_considuration_ratio": data_fairness_considuration_ratio['fairness_considuration_ratio'],
       "auc": data_fairness_considuration_ratio['auc'],
       "type": data_fairness_considuration_ratio['type'],
   }
)
samples_similarity_between_fair_and_regular_predictions_chart_data = pd.DataFrame(
   {
       "samples": data_samples['samples'],
       "similarity_between_fair_and_regular_predictions": data_samples['similarity_between_fair_and_regular_predictions'],
       "type": data_samples['type'],
   }
)
top_results_to_compare_similarity_between_fair_and_regular_predictions_chart_data = pd.DataFrame(
   {
       "top_results_to_compare": data_top_results_to_compare['top_results_to_compare'],
       "similarity_between_fair_and_regular_predictions": data_top_results_to_compare['similarity_between_fair_and_regular_predictions'],
       "type": data_top_results_to_compare['type'],
   }
)
fairness_considuration_ratio_similarity_between_fair_and_regular_predictions_chart_data = pd.DataFrame(
   {
       "fairness_considuration_ratio": data_fairness_considuration_ratio['fairness_considuration_ratio'],
       "similarity_between_fair_and_regular_predictions": data_fairness_considuration_ratio['similarity_between_fair_and_regular_predictions'],
       "type": data_fairness_considuration_ratio['type'],
   }
)
samples_fairness_score_of_fair_predictions_chart_data = pd.DataFrame(
   {
       "samples": data_samples['samples'],
       "fairness_score_of_fair_predictions": data_samples['fairness_score_of_fair_predictions'],
       "type": data_samples['type'],
   }
)
top_results_to_compare_fairness_score_of_fair_predictions_chart_data = pd.DataFrame(
   {
       "top_results_to_compare": data_top_results_to_compare['top_results_to_compare'],
       "fairness_score_of_fair_predictions": data_top_results_to_compare['fairness_score_of_fair_predictions'],
       "type": data_top_results_to_compare['type'],
   }
)
fairness_considuration_ratio_fairness_score_of_fair_predictions_chart_data = pd.DataFrame(
   {
       "fairness_considuration_ratio": data_fairness_considuration_ratio['fairness_considuration_ratio'],
       "fairness_score_of_fair_predictions": data_fairness_considuration_ratio['fairness_score_of_fair_predictions'],
       "type": data_fairness_considuration_ratio['type'],
   }
)
samples_fairness_score_of_regular_predictions_chart_data = pd.DataFrame(
   {
       "samples": data_samples['samples'],
       "fairness_score_of_regular_predictions": data_samples['fairness_score_of_regular_predictions'],
       "type": data_samples['type'],
   }
)
top_results_to_compare_fairness_score_of_regular_predictions_chart_data = pd.DataFrame(
   {
       "top_results_to_compare": data_top_results_to_compare['top_results_to_compare'],
       "fairness_score_of_regular_predictions": data_top_results_to_compare['fairness_score_of_regular_predictions'],
       "type": data_top_results_to_compare['type'],
   }
)
fairness_considuration_ratio_fairness_score_of_regular_predictions_chart_data = pd.DataFrame(
   {
       "fairness_considuration_ratio": data_fairness_considuration_ratio['fairness_considuration_ratio'],
       "fairness_score_of_regular_predictions": data_fairness_considuration_ratio['fairness_score_of_regular_predictions'],
       "type": data_fairness_considuration_ratio['type'],
   }
)
with auc_col:
    st.header("AUC")
    st.line_chart(samples_auc_chart_data, x="samples", y="auc", x_label="Samples", y_label="AUC", color="type")
    st.line_chart(top_results_to_compare_auc_chart_data, x="top_results_to_compare", y="auc", x_label="Top results to compare", y_label="AUC", color="type")
    st.line_chart(fairness_considuration_ratio_auc_chart_data, x="fairness_considuration_ratio", y="auc", x_label="Fairness Consideration Ratio", y_label="AUC", color="type")
with similarity_between_fair_and_regular_predictions_col:
    st.header("Similarity")
    st.line_chart(samples_similarity_between_fair_and_regular_predictions_chart_data, x="samples", y="similarity_between_fair_and_regular_predictions", x_label="Samples", y_label="Similarity", color="type")
    st.line_chart(top_results_to_compare_similarity_between_fair_and_regular_predictions_chart_data, x="top_results_to_compare", y="similarity_between_fair_and_regular_predictions", x_label="Top results to compare", y_label="Similarity", color="type")
    st.line_chart(fairness_considuration_ratio_similarity_between_fair_and_regular_predictions_chart_data, x="fairness_considuration_ratio", y="similarity_between_fair_and_regular_predictions", x_label="Fairness Consideration Ratio", y_label="Similarity", color="type")
with fairness_score_of_fair_predictions_col:
    st.header("Fairness of fair")
    st.line_chart(samples_fairness_score_of_fair_predictions_chart_data, x="samples", y="fairness_score_of_fair_predictions", x_label="Samples", y_label="Fairness of fair", color="type")
    st.line_chart(top_results_to_compare_fairness_score_of_fair_predictions_chart_data, x="top_results_to_compare", y="fairness_score_of_fair_predictions", x_label="Top results to compare", y_label="Fairness of fair", color="type")
    st.line_chart(fairness_considuration_ratio_fairness_score_of_fair_predictions_chart_data, x="fairness_considuration_ratio", y="fairness_score_of_fair_predictions", x_label="Fairness Consideration Ratio", y_label="Fairness of fair", color="type")
with fairness_score_of_regular_predictions_col:
    st.header("Fairness of regular")
    st.line_chart(samples_fairness_score_of_regular_predictions_chart_data, x="samples", y="fairness_score_of_regular_predictions", x_label="Samples", y_label="Fairness of regular", color="type")
    st.line_chart(top_results_to_compare_fairness_score_of_regular_predictions_chart_data, x="top_results_to_compare", y="fairness_score_of_regular_predictions", x_label="Top results to compare", y_label="Fairness of regular", color="type")
    st.line_chart(fairness_considuration_ratio_fairness_score_of_regular_predictions_chart_data, x="fairness_considuration_ratio", y="fairness_score_of_regular_predictions", x_label="Fairness Consideration Ratio", y_label="Fairness of regular", color="type")

# Checking fairness_considuration_ratio imapct on similarity_between_fair_and_regular_predictions
# data1 = data[(data['samples'] == 4) & (data['top_results_to_compare'] == 10)]
# types = ['random', 'popular', 'consensus', 'controversial']
# for i in range(4):
#     tmp = data1[(data1['type'] == types[i])]
#     plt.plot(tmp['fairness_considuration_ratio'], tmp['similarity_between_fair_and_regular_predictions'], label=types[i])
# plt.xlabel('fairness_considuration_ratio')
# plt.ylabel('similarity_between_fair_and_regular_predictions')
# plt.legend()
# plt.show()

# Checking samples imapct on similarity_between_fair_and_regular_predictions
# data2 = data[(data['fairness_considuration_ratio'] == 0.4) & (data['top_results_to_compare'] == 10)]
# types = ['random', 'popular', 'consensus', 'controversial']
# for i in range(4):
#     tmp = data2[(data2['type'] == types[i])]
#     plt.plot(tmp['samples'], tmp['similarity_between_fair_and_regular_predictions'], label=types[i])
# plt.xlabel('samples')
# plt.ylabel('similarity_between_fair_and_regular_predictions')
# plt.legend()
# plt.show()

# Checking top_results_to_compare imapct on similarity_between_fair_and_regular_predictions
# data2 = data[(data['fairness_considuration_ratio'] == 0.4) & (data['samples'] == 4)]
# types = ['random', 'popular', 'consensus', 'controversial']
# for i in range(4):
#     tmp = data2[(data2['type'] == types[i])]
#     plt.plot(tmp['top_results_to_compare'], tmp['similarity_between_fair_and_regular_predictions'], label=types[i])
# plt.xlabel('top_results_to_compare')
# plt.ylabel('similarity_between_fair_and_regular_predictions')
# plt.legend()
# plt.show()

#########################################################################################################

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# colors = ['r', 'g', 'b', 'y', 'c']
# yticks = [5, 4, 3, 2, 1]
# for c, k in zip(colors, yticks):
#     tmp = data[(data['samples'] == k)]
#     ax.bar(tmp['fairness_considuration_ratio']*10, tmp['fairness_score_of_fair_predictions'], zs=k, zdir='y', color=c, alpha=0.8)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_yticks(yticks)
# plt.show()

# plot graph with 3 lines, one per column 'similarity_between_fair_and_regular_predictions','fairness_score_of_fair_predictions','fairness_score_of_regular_predictions'. X axis should be 'samples' column.
# plt.plot(data['samples'], data['similarity_between_fair_and_regular_predictions'], label='similarity_between_fair_and_regular_predictions')
# plt.plot(data['samples'], data['fairness_score_of_fair_predictions'], label='fairness_score_of_fair_predictions')
# plt.plot(data['samples'], data['fairness_score_of_regular_predictions'], label='fairness_score_of_regular_predictions')
# plt.xlabel('Samples')
# plt.ylabel('Value')
# plt.title('Random Selection')
# plt.legend()
# plt.show()


# fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure and a set of subplots

# columns = ['similarity_between_fair_and_regular_predictions','fairness_score_of_fair_predictions','fairness_score_of_regular_predictions']
# for i in range(3):
#     ax = axs[i]
#     scatter = ax.scatter(data['fairness_considuration_ratio'], data['samples'], c=data[columns[i]], cmap='gray')
#     ax.set_xlabel('Fairness Consideration Ratio')
#     ax.set_ylabel('Top Results to Compare')
#     ax.set_title(f'Plot {columns[i]}')

# # Add a colorbar to the figure, not to each subplot
# fig.colorbar(scatter, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

# plt.suptitle('Fairness Score of Regular Predictions')
# plt.show()

# for i in range(3):
#     plt.scatter(data['fairness_considuration_ratio'], data['top_results_to_compare'], c=data['fairness_score_of_regular_predictions'])
#     # use grayscale color map
#     plt.set_cmap('gray')
#     # add labels
#     plt.xlabel('Fairness Consideration Ratio')
#     plt.ylabel('Top Results to Compare')
#     # add colorbar
#     plt.colorbar()
#     # add title
#     plt.title('Fairness Score of Regular Predictions')
#     # show the plot
#     plt.show()
