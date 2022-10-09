from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, precision_score, f1_score, \
    recall_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, \
    balanced_accuracy_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import numpy as np
import collections
import random
import seaborn as sn
import json
from pathlib import Path

def target_analysis(class_series):
    """
    Calculates the frequency of each class in the given
    series.
    :param class_series: The class series.
    """
    total_cases = class_series.shape[0]
    
    print(f"Total Cases: {total_cases}")
    
    value_counts = class_series.value_counts()
    
    labels = []
    values = []
    
    print("--------Labels percentages--------")
    for label, value in value_counts.items():
        labels.append(label)
        values.append(value)
        print(f"{label}: {value} ({value * 100/total_cases}%)")
        
    print("----------------------------------")
    
    # Plotting a bar chart.
    plot_bar(values, labels, "Class Distribution", "class_distribution_bar")
    
    # Plotting a pie chart.
    zipped = list(zip(values, labels))
    random.shuffle(zipped)
    values, labels = zip(*zipped)
    plot_pie(values, labels, "Class Distribution", "class_distribution_pie")
    
def plot_bar(values, labels, title, file_name):
    """
    Plots a bar chart with the provided values and labels.
    :param values: The values.
    :param labels: The corresponding labels.
    :param title: The title of the pie chart.
    :param file_name: The file name.
    """
    Path(f"figures/analysis").mkdir(parents = True, exist_ok = True)
    
    figure = plt.figure(figsize=[20, 10])
    ax = figure.add_subplot(111)
    
    ax.bar(labels, values)
    ax.set_title(title)
    plt.savefig(f"figures/analysis/{file_name}.png")
    plt.close(figure)
        
def plot_pie(values, labels, title, file_name):
    """
    Plots a pie chart with the provided values and labels.
    :param values: The values.
    :param labels: The corresponding labels.
    :param title: The title of the pie chart.
    :param file_name: The file name.
    """
    Path(f"figures/analysis").mkdir(parents = True, exist_ok = True)
    
    figure = plt.figure(figsize=[10, 10])
    
    with plt.style.context({"axes.prop_cycle" : plt.cycler("color", plt.cm.tab20.colors)}):
        ax = figure.add_subplot(111)
        pie_wedge_collection = ax.pie(values, labels = labels, \
               autopct = "%1.1f%%", radius = 1, labeldistance = 1.05)
        
        for pie_wedge in pie_wedge_collection[0]:
            pie_wedge.set_edgecolor('white')
            
        ax.set_title(title)
        plt.savefig(f"figures/analysis/{file_name}.png")
        plt.close(figure)
        
def plot_word_frequencies(vocabulary, limit):
    """
    Plots the word frequencies of the given vocabulary until
    the given word limit is reached.
    :param vocabulary: The vocabulary.
    :param limit: The word limit.
    """
    labels = []
    values = []
    counter = 0
    
    for word in vocabulary:
        if counter == limit:
            break
        counter += 1
        labels.append(counter)
        values.append(word[1])
        
    plot_bar(values, labels, "Word Frequencies", "word_frequencies")
    
def to_lower_case(series):
    """
    Converts every word in the given series
    to its lower case equivalent.
    :param series: The series.
    :return: The lower case series.
    """
    return [entry.lower() for entry in series]

def tokenize(series):
    """
    Tokenizes the given series.
    :param series: The series to be tokenized.
    :return: The tokenized series.
    """
    return [word_tokenize(entry) for entry in series]

def lemmatize(series):
    """
    Lemmaties the given series. It, also, removes
    stop words.
    :param series: The series to be lemmatize.
    :return: The lemmatized series.
    """
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map["J"] = wn.ADJ
    tag_map["V"] = wn.VERB
    tag_map["R"] = wn.ADV
    
    lemmatized = []
    
    for index, entry in enumerate(series):
        words = []
        word_lemmatizer = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words("english") and word.isalpha():
                final_word = word_lemmatizer.lemmatize(word,tag_map[tag[0]])
                words.append(final_word)
        lemmatized.append(str(words))
    return lemmatized

def calculate_vocabulary_frequencies(word_list):
    """
    Calculates the frequency of each word in the
    given word list.
    :param vocabulary: The vocabulary.
    :return: The word frequency.
    """
    flattened_list = []
    for stuff in word_list:
        for word in stuff.replace("'", "").replace("[", "").replace("]", "").split(","):
            flattened_list.append(word)
    
    return collections.Counter(flattened_list).most_common()

def get_number_of_features(vocabulary_frequencies, threshold):
    """
    Gets the number of features needed given the vocabulary
    frequencies and a threshold.
    :param vocabulary_frequencies: The words frequencies.
    :param threshold: The frequency threshold.
    :return: The number of features.
    """
    number_of_features = 0
    for frequency in vocabulary_frequencies:
        if frequency[1] < threshold:
            break
        number_of_features += 1
    return number_of_features

def vectorize(data_series, max_features):
    """
    Vectorizes the given data series using TF-IDF.
    :param data_series: The data series to be vectorized.
    :param max_features: The max features of the vectorized data.
    :return: The vectorized data series.
    """
    vectorizer = TfidfVectorizer(max_features = max_features)
    vectorizer.fit(data_series)
    return vectorizer.transform(data_series)

def encode_class(class_series):
    """
    Encodes the given class series to integers.
    :param class_series: The class series to be encoded.
    :return: The encoded class series.
    """
    encoder = LabelEncoder()
    class_series = encoder.fit_transform(class_series)
    print(encoder.classes_)
    
    return class_series

def smote(X, y):
    """
    Implements oversampling using SMOTE
    :param X: The features.
    :param y: The classes.
    :return: The oversampled data.
    """
    oversample = SMOTE(n_jobs = -1)
    X, y = oversample.fit_resample(X, y)
    
    return X, y

def preprocess_data(data, frequency_threshold, use_smote = True):
    """
    Preprocess the given data and create the needed features
    via the TF-IDF process by taking into consideration the
    given word frequence threshold.
    :param data: The data to be preprocesses.
    :param use_smote: Indicates if SMOTE must be used.
    :param frequence_threshold: The word frequence threshold.
    :return: The preprocessed data.
    """
    # Dropping the `id` column.
    data = data.drop(columns = ["tweet_id"])
    
    # Lowering cases.
    data["content"] = to_lower_case(data["content"])
    
    # Tokenizing.
    data["content"] = tokenize(data["content"])
    
    # Lemmatizing and removing stop words.
    words_list = lemmatize(data["content"])
    
    # Calculating frequencies.
    vocabulary_frequencies = calculate_vocabulary_frequencies(words_list)
    
    # Ploting the frequences for 1000 of the most popular words.
    plot_word_frequencies(vocabulary_frequencies, 1000)
    
    # Getting the corresponding feature number.
    feature_number = get_number_of_features(vocabulary_frequencies, frequency_threshold)
    
    # Enhancing the initial dataframe.
    for index, words in enumerate(words_list):
         data.loc[index, "final_content"] = words
         
    # Vectorizing data.
    X = vectorize(data["final_content"], feature_number)
    
    # Encoding data.
    y = encode_class(data["sentiment"])
    
    # Applying oversampling.
    if use_smote:
        X, y = smote(X, y)
    
    # Splitting training-test sets.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.4)
    
    return X_train, X_test, y_train, y_test

def evaluation_metrics(y_test, y_pred, folder_name):
    """
    Prints the desired evaluation metrics and
    calculates the confusion metrix.
    :param y_test: The true values.
    :param y_pred: THe predicted values.
    :param folder_name: The name of the folder.
    :return: The confusion matrix.
    """
    results = {}
    results["accuracy"] = accuracy_score(y_test, y_pred) * 100
    results["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred) * 100
    results["precision"] = precision_score(y_test, y_pred, average = "macro") * 100
    results["f1"] = f1_score(y_test, y_pred, average = "macro") * 100
    results["recall"] = recall_score(y_test, y_pred, average = "macro") * 100
    results["cohen"] = cohen_kappa_score(y_test, y_pred) * 100
    results["mcc"] = matthews_corrcoef(y_test, y_pred) * 100
    
    print("Accuracy Score: ", results["accuracy"])
    print("Balanced Accuracy Score: ", results["balanced_accuracy"])
    print("Precision Score: ", results["precision"])
    print("F1 Score: ", results["f1"])
    print("Recall Score: ", results["recall"])
    print("Cohen's Kappa Score: ", results["cohen"])
    print("MCC Score: ", results["mcc"])
    
    # Store metrics in json file
    with open(f"figures/{folder_name}/results.json", "w") as results_file:
        json.dump(results, results_file)
    
    # Confustion matrix
    matrix = confusion_matrix(y_test, y_pred)
    
    figure, ax = plt.subplots(1, 1)
    
    ax.set_title("Confusion Matrix",\
           fontsize = 20, fontweight = "bold")
    sn.heatmap(matrix, cmap = "Blues")
    plt.savefig(f"figures/{folder_name}/confusion_matrix.png")
    plt.close(figure)
    
    return matrix

def plot_grid_search(cv_results, group_by_parameter, x_axis_parameter, folder_name):
    """
    Plots the grid search results of train and test samples
    :param cv_results: The cv results.
    :param group_by_parameter: The parameter that will be used for grouping.
    :param x_axis_parameter: The x-axis parameter.
    :param folder_name: The folder name.
    """
    test_scores_mean = np.array(cv_results["mean_test_score"])
    train_scores_mean = np.array(cv_results["mean_train_score"])
    fit_time_mean = np.array(cv_results["mean_fit_time"])
    parameters = cv_results["params"]
    groups = {}
    for index, parameter in enumerate(parameters):
        if groups.get(parameter[group_by_parameter]) is not None:
            groups[parameter[group_by_parameter]].append({
                    "test_score": test_scores_mean[index],
                    "train_score": train_scores_mean[index],
                    "fit_time": fit_time_mean[index] / 60.0,
                    "value": parameter[x_axis_parameter]
            })
        else:
            groups[parameter[group_by_parameter]] = [{
                    "test_score": test_scores_mean[index],
                    "train_score": train_scores_mean[index],
                    "fit_time": fit_time_mean[index] / 60.0,
                    "value": parameter[x_axis_parameter]
            }]
    
    for key in groups:
        groups[key].sort(key = lambda x: x["value"])
    
    for key in groups:
        figure, axs = plt.subplots(1, 1)
        axs.plot(list(map(lambda item: item["value"], groups[key])),\
           list(map(lambda item: item["test_score"], groups[key])), "-o", \
           label = "Test")
        axs.plot(list(map(lambda item: item["value"], groups[key])),\
           list(map(lambda item: item["train_score"], groups[key])), "-x", \
           label = "Train")
    
        axs.set_title(f"{group_by_parameter} = {key}",\
           fontsize = 20, fontweight = "bold")
        axs.set_xlabel(x_axis_parameter, fontsize = 16)
        axs.set_ylabel("CV Average Score", fontsize = 16)
        axs.legend(loc = "best", fontsize = 15)
        figure.tight_layout()
        plt.gca().set_xscale("log")
        plt.savefig(f"figures/{folder_name}/{group_by_parameter}_{key}.png")
        plt.close(figure)
        
        figure, axs = plt.subplots(1, 1)
        axs.plot(list(map(lambda item: item["value"], groups[key])),\
           list(map(lambda item: item["fit_time"], groups[key])), "-o", \
           label = "Fit Time")
    
        axs.set_title("Fit Time in Minutes",\
           fontsize = 20, fontweight = "bold")
        axs.set_xlabel(group_by_parameter, fontsize = 16)
        axs.set_ylabel("Time", fontsize = 16)
        axs.legend(loc = "best", fontsize = 15)
        figure.tight_layout()
        plt.savefig(f"figures/{folder_name}/{group_by_parameter}_{key}_time.png")
        plt.close(figure)
        
    return groups

def plot_grid_search_single(cv_results, parameter, folder_name): 
    """
    Plots the grid search results of train and test samples
    :param cv_results: The cv results.
    :param parameter: The parameter that will be used for grouping.
    :param folder_name: The folder name.
    """
    labels = list(map(lambda item: item[parameter], cv_results["params"]))
    
    x = np.arange(len(labels))
    width = 0.35
    
    figure, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, cv_results["mean_test_score"], width, label = "Testing", tick_label = "")
    rects2 = ax.bar(x + width / 2, cv_results["mean_train_score"], width, label = "Training", tick_label = "")
    
    ax.set_ylabel("GridSearch Score")
    ax.set_title("Results")
    ax.legend()
    
    ax.bar_label(rects1, padding = 3)
    ax.bar_label(rects2, padding = 3)
    
    figure.tight_layout()
    
    plt.savefig(f"figures/{folder_name}/ncc.png")
    plt.close(figure)