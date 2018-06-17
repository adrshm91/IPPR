import numpy as np
from skimage import feature
import glob
import random
from PIL import Image
from math import pi, exp
import matplotlib.pyplot as plt
from numpy import genfromtxt

path_to_non_ped_training = \
    "/Users/adarshmallandur/Desktop/DaimlerBenchmark/TrainingData/NonPedestrians/18x36/1_non-ped_examples bmp/*.bmp"
path_to_ped_training = "/Users/adarshmallandur/Desktop/DaimlerBenchmark/TrainingData/Pedestrians/18x36/*.bmp"
path_to_ped_test = "/Users/adarshmallandur/Desktop/DaimlerBenchmark/TestData/Test_data_cut/*.pgm"
path_to_csv = "/Users/adarshmallandur/Desktop/DaimlerBenchmark/"
path_to_arf = "/Users/adarshmallandur/Desktop/DaimlerBenchmark/"


# path_to_ped_test = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TestData/Test_data_cut/*.pgm"
# path_to_csv = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/"
# path_to_arf = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/"
# path_to_non_ped_training = \
#   "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TrainingData/NonPedestrians/18x36/Non_Pedestrians/*.bmp"
# path_to_ped_training = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TrainingData/Pedestrians/18x36/*.bmp"

# Evaluator: weka.attributeSelection.ClassifierAttributeEval -execution-slots 1 -B
# weka.classifiers.rules.ZeroR -F 5 -T 0.01 -R 1 -E DEFAULT --
# Search:    weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N -1
feature_subset_pixels = [143, 48, 46, 47, 49, 142, 50, 51, 45, 44, 43, 42, 37, 144]

mean_data = None
cov_data = None
det_mat = None

positive_probability = None
negative_probability = None
true_positive = []
true_negative = []
false_positive = []
false_negative = []


def create_training_data_set():
    np_fd_list = []
    p_fd_list = []
    i = 0

    # Read and calculate HOG for each negative image
    for eachImage in glob.glob(path_to_non_ped_training):
        i = i + 1
        print("reading training data file %d and creating no_pedestrian_data" % i)
        np_image = Image.open(eachImage, 'r')
        hog_np, hog_np_image = feature.hog(np_image, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1)
                                               , visualise=True)
        np_fd_list.append(list(hog_np))
    negative_images = np.array(np_fd_list)

    # logic to create arff file
    negative_images_arff = np.concatenate((negative_images, np.zeros((len(negative_images), 1), dtype=np.int)), axis=1)

    # Remove the columns which are not present in feature subset
    # negative_images = np.take(negative_images, feature_subset_pixels, axis=1)

    # Append 0 as last column
    negative_images_numpy = np.concatenate((negative_images, np.zeros((len(negative_images), 1), dtype=np.int)), axis=1)

    i = 0

    # Read and calculate HOG for each positive image
    for eachImage in glob.glob(path_to_ped_training):
        i = i + 1
        print("reading training data file %d and creating pedestrian_data" % i)
        p_image = Image.open(eachImage, 'r')
        hog_p, hog_p_image = feature.hog(p_image, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1)
                                             , visualise=True)
        p_fd_list.append(list(hog_p))
    positive_images = np.array(p_fd_list)

    # logic to create arff file
    positive_images_arff = np.concatenate((positive_images, np.ones((len(positive_images), 1), dtype=np.int)), axis=1)

    # Remove the columns which are not present in feature subset
    # positive_images = np.take(positive_images, feature_subset_pixels, axis=1)

    # Append 1 as last column
    positive_images_numpy = np.concatenate((positive_images, np.ones((len(positive_images), 1), dtype=np.int)), axis=1)

    # create arff file for processing in WEKA
    create_arff_file(positive_images_arff, negative_images_arff)

    return np.concatenate((negative_images_numpy, positive_images_numpy), axis=0)


# def create_csv_file(data, filename):
#     print("Creating csv file ", filename)
#     df = pd.DataFrame(data)
#     df.to_csv(path_to_csv + filename + ".csv")


def create_csv_file(data, filename):
    np.savetxt(path_to_csv + filename + ".csv", np.array(data), delimiter=",")
    data_set = genfromtxt(path_to_csv + filename + ".csv", delimiter=',')


def create_arff_file(ped_data, nonped_data):
    print("Creating arff file...")
    filename = path_to_arf + 'Pedestrians.arff'
    file_access = open(filename, "w")
    file_access.write('@RELATION Pedestrians\n')
    for i in range(len(ped_data[0]) - 1):
        file_access.write('@ATTRIBUTE Pixel' + str(i) + ' NUMERIC\n')
    file_access.write('@ATTRIBUTE Klasse       {0.0,1.0}\n')
    file_access.write('% 0=NonPedestrian, 1=Pedestrian\n')
    file_access.write('@DATA\n')

    for i in range(len(nonped_data)):
        for j in range(len(nonped_data[i]) - 1):
            file_access.write(str(nonped_data[i][j]))
            file_access.write(' , ')
        file_access.write(str(nonped_data[i][len(nonped_data[i]) - 1]))
        file_access.write('\n')

    for i in range(len(ped_data)):
        for j in range(len(ped_data[i]) - 1):
            file_access.write(str(ped_data[i][j]))
            file_access.write(' , ')
        file_access.write(str(ped_data[i - 1][len(ped_data[i - 1]) - 1]))
        file_access.write('\n')
    file_access.close()


def split_data_set(data_set, split_ratio):
    print("Spliting test data and training data in the ratio ", split_ratio)
    train_size = int(len(data_set) * split_ratio)
    train_set = []
    copy = list(data_set)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return np.array(train_set), np.array(copy)


def summarize(dataframe):
    mean = []
    std_deviation = []
    for i in range(dataframe.count().count() - 1):
        mean.append(dataframe[i].mean())
        std_deviation.append(dataframe[i].std())
    return np.array([np.array(mean), np.array(std_deviation)])


def normal_pdf(x, mu, sigma):
    return 1.0 / (sigma * (2.0 * pi) ** (1 / 2)) * exp(-1.0 * (x - mu) ** 2 / (2.0 * (sigma ** 2)))


def remove_class(test_data):
    print("Removing class column in test data...")
    test_data_classify = test_data.copy()
    last_column = test_data_classify.shape[1] - 1
    test_data_classify.drop([last_column], axis=1, inplace=True)
    return test_data


def getAccuracy(testSet, predicted_data, pedDetected, nonPedDetected):
    global true_positive
    global true_negative
    global false_negative
    global false_positive
    print("Calculating accuracy and confusion Matrix...")
    ped_images = 0
    non_ped_images = 0

    # Calculate number of pedestrians and non pedestrians in initial test data
    for each_image in testSet:
        if each_image[-1] == 1:
            ped_images += 1
        else:
            non_ped_images += 1

    # Initialize confusion matrix
    confusion_matrix = np.zeros(shape=(2, 2))

    # Calculate confusion matrix
    i = 0
    for each_image in predicted_data:
        if each_image[-1] == 1:
            if each_image[-1] == testSet[i][-1]:
                confusion_matrix[0][0] += 1
            else:
                confusion_matrix[1][0] += 1
        else:
            if each_image[-1] == testSet[i][-1]:
                confusion_matrix[1][1] += 1
            else:
                confusion_matrix[0][1] += 1
        i += 1
    true_positive.append(confusion_matrix[0][0])
    false_positive.append(confusion_matrix[1][0])
    false_negative.append(confusion_matrix[0][1])
    true_negative.append(confusion_matrix[1][1])
    # Calculate Accuracy
    accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (pedDetected + nonPedDetected) * 100
    return confusion_matrix, accuracy


def calculate_pdf(test_image):
    global mean_data
    global cov_data
    global det_mat
    len_test_image = len(test_image)
    formula_1 = np.subtract(test_image, mean_data)
    formula_1_T = np.array(formula_1)[np.newaxis]
    formula_1_T = formula_1_T.T
    formula_inv = np.linalg.inv(cov_data)
    formula_multi = np.dot(formula_1, formula_inv)
    formula_multi = np.dot(formula_multi, formula_1_T)
    formula_multi = -0.5 * formula_multi
    exp_data = np.exp(formula_multi)
    det_cov_product = exp_data * np.power(det_mat, -0.5)
    pdf_value = ((2 * np.pi) ** (len_test_image / 2)) * det_cov_product
    return pdf_value


def train(training_data):
    global mean_data
    global cov_data
    global det_mat
    mean_data = np.mean(training_data, axis=0)
    cov_data = np.cov(np.transpose(training_data))
    det_mat = np.linalg.det(cov_data)
    probability = calculate_pdf(mean_data)
    return probability


def calculate_threshold(max_probability):
    threshold = []
    n = 20
    for i in range(0, n):
        threshold.append((i * (max_probability / n))/max_probability)
    return np.array(threshold)


def test(test_data, threshold, max_probability):
    row = 0
    nonPedDetected = 0
    pedDetected = 0
    copy = np.copy(test_data)
    for each_image in test_data:
        probability = calculate_pdf(each_image[0:-1])/max_probability
        if probability < threshold:
            # Update predicted_data with 0 if non pedestrian
            nonPedDetected = nonPedDetected + 1
            copy[row, -1] = 0
        else:
            # Update predicted_data with 1 if pedestrian
            copy[row, -1] = 1
            pedDetected = pedDetected + 1
        row = row + 1
    return copy, pedDetected, nonPedDetected


def plot_graph(threshold,accuracy):

    plt.plot(threshold, true_positive,label='True Positive')
    plt.plot(threshold,false_negative,label='False Negative')
    plt.ylabel('Pedestrian')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title('True Positive vs False Negative')
    plt.savefig(path_to_arf + 'fig1_HOG.jpg')
    plt.show()

    plt.plot(threshold, true_negative, label='True Negative')
    plt.plot(threshold, false_positive, label='False Positive')
    plt.ylabel('Non-Pedestrian')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title('True Negative vs False Positive')
    plt.savefig(path_to_arf + 'fig2_HOG.jpg')
    plt.show()

    plt.plot(threshold, accuracy, label='Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.title('Accuracy VS Threshold')
    plt.savefig(path_to_arf + 'fig3_HOG.jpg')
    plt.show()


def main():
    # Create two datasets, one with whole data and other with only positive data for training
    # data_set = create_training_data_set()
    #
    # create_csv_file(data_set, "data_set_hog")

    data_set = genfromtxt('/Users/adarshmallandur/Desktop/DaimlerBenchmark/data_set_hog.csv', delimiter=',')

    data_set = np.take(data_set, feature_subset_pixels, axis=1)

    # Take out random test data from whole data set
    test_data, training_data = split_data_set(data_set, 0.67)

    # Take out positive images from training data
    training_data_positive = training_data[training_data[:, -1] == 1]

    # train the model
    max_probability = train(training_data_positive[:, :-1])

    # Calculate threshold using maximum probability of training data
    threshold = calculate_threshold(max_probability)

    # test the model
    predicted_data = []
    pedDetected = []
    nonPedDetected = []
    confusion_matrix = []
    accuracy = []
    for i in range(0, len(threshold)):
        predicted_data_temp, pedDetected_temp, nonPedDetected_temp = test(test_data, threshold[i], max_probability)
        predicted_data.append(predicted_data_temp)
        pedDetected.append(pedDetected_temp)
        nonPedDetected.append(nonPedDetected_temp)
        confusion_matrix_temp, accuracy_temp = getAccuracy(test_data, predicted_data_temp, pedDetected_temp,
                                                           nonPedDetected_temp)
        confusion_matrix.append(confusion_matrix_temp)
        accuracy.append(accuracy_temp)
        print("Threshold is .. ")
        print(threshold[i])
        print("Confusion Matrix is .. ")
        print(confusion_matrix_temp)
        print(accuracy_temp)
    plot_graph(threshold, accuracy)
    print(accuracy)


main()
