import numpy as np
from skimage import feature
import glob
import pandas as pd
import random
from PIL import Image
from math import pi, exp

# path_to_non_ped_training = \
#     "/Users/adarshmallandur/Desktop/DaimlerBenchmark/TrainingData/NonPedestrians/18x36/1_non-ped_examples bmp/*.bmp"
# path_to_ped_training = "/Users/adarshmallandur/Desktop/DaimlerBenchmark/TrainingData/Pedestrians/18x36/*.bmp"bmp
path_to_ped_test = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TestData/Test_data_cut/*.pgm"
path_to_csv = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/"
path_to_arf = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/"
path_to_non_ped_training = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TrainingData/NonPedestrians/18x36/Non_Pedestrians/*.bmp"
path_to_ped_training = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TrainingData/Pedestrians/18x36/*.bmp"

feature_subset_pixels = []

summary_pedestrian = pd.DataFrame()
summary_non_pedestrian = pd.DataFrame()

positive_probability = None
negative_probability = None

test_data = pd.DataFrame()
training_data = pd.DataFrame()


def create_training_data_set():
    np_fd_list = []
    p_fd_list = []
    i = 0

    # Read and calculate HOG for each negative image
    for eachImage in glob.glob(path_to_non_ped_training):
        i = i + 1
        print("reading training data file %d and creating no_pedestrian_data" % i)
        np_image = Image.open(eachImage, 'r')
        np_fd_list.append(list(np_image.getdata()))
    no_pedestrian_data = np.array(np_fd_list)

    # Append 0 as last column
    no_pedestrian_data_numpy = np.concatenate((no_pedestrian_data, np.zeros((len(no_pedestrian_data), 1), dtype=np.int)), axis=1)
    no_pedestrian_data = pd.DataFrame(no_pedestrian_data_numpy)

    i = 0

    # Read and calculate HOG for each positive image
    for eachImage in glob.glob(path_to_ped_training):
        i = i + 1
        print("reading training data file %d and creating pedestrian_data" % i)
        p_image = Image.open(eachImage, 'r')
        p_fd_list.append(list(p_image.getdata()))
    pedestrian_data = np.array(p_fd_list)

    # Append 1 as last column
    pedestrian_data_numpy = np.concatenate((pedestrian_data, np.ones((len(pedestrian_data), 1), dtype=np.int)), axis=1)
    pedestrian_data = pd.DataFrame(pedestrian_data_numpy)

    # Concantinate positive and negative images to get training data set
    training_data = pd.concat([pedestrian_data, no_pedestrian_data], ignore_index=True)

    # create arff file for processing in WEKA
    create_arff_file(pedestrian_data_numpy, no_pedestrian_data_numpy)

    return training_data


def create_csv_file(data, filename):
    print("Creating csv file ", filename)
    df = pd.DataFrame(data)
    df.to_csv(path_to_csv + filename + ".csv")


def create_arff_file(ped_data, nonped_data):
    print("Creating arff file...")
    filename = path_to_arf+'Pedestrians.arff'
    file_access = open(filename, "w")
    file_access.write('@RELATION Pedestrians\n')
    for i in range(len(ped_data[0])-1):
        file_access.write('@ATTRIBUTE Pixel' + str(i) + ' NUMERIC\n')
    file_access.write('@ATTRIBUTE Klasse       {0.0,1.0}\n')
    file_access.write('% 0=NonPedestrian, 1=Pedestrian\n')
    file_access.write('@DATA\n')

    for i in range(len(nonped_data)):
        for j in range(len(nonped_data[i])-1):
            file_access.write(str(nonped_data[i][j]))
            file_access.write(' , ')
        file_access.write(str(nonped_data[i][len(nonped_data[i])-1]))
        file_access.write('\n')

    for i in range(len(ped_data)):
        for j in range(len(ped_data[i])-1):
            file_access.write(str(ped_data[i][j]))
            file_access.write(' , ')
        file_access.write(str(ped_data[i-1][len(ped_data[i-1])-1]))
        file_access.write('\n')
    file_access.close()


def split_data_set(data_set, split_ratio):
    print("Spliting test data and training data in the ratio ", split_ratio)
    test_size = int(len(data_set) * (1-split_ratio))
    test_set = pd.DataFrame()
    copy = data_set
    index_list = np.arange(int(len(data_set) * split_ratio) + 1)
    while len(test_set) < test_size:
        index_copy = random.randrange(len(copy))
        test_set = test_set.append(copy.iloc[index_copy], ignore_index=True)
        try:
            copy = copy.drop(copy.index[index_copy])
        except:
            pass
    copy = copy.set_index(index_list)
    return test_set, copy


def rowwisebestPixelmatrix(bestpixels, data):
    bestPixelMatrix=[]
    for eachBestPixel in bestPixels:
        bestPixelMatrix.append(data[eachBestPixel])
    return np.array(bestPixelMatrix)


def summarize(dataframe):
    mean = []
    std_deviation = []
    for i in range(dataframe.count().count()-1):
        mean.append(dataframe[i].mean())
        std_deviation.append(dataframe[i].std())
    return np.array([np.array(mean), np.array(std_deviation)])


def normal_pdf(x, mu, sigma):
    return 1.0 / (sigma * (2.0 * pi)**(1/2)) * exp(-1.0 * (x - mu)**2 / (2.0 * (sigma**2)))


def train(training_data):
    print("Running the training...")
    ped_data_dataframe = pd.DataFrame()
    non_ped_data_dataframe = pd.DataFrame()
    global summary_pedestrian
    global summary_non_pedestrian
    global positive_probability
    global negative_probability

    # For each image in training data and for each class, calculate mean and standard deviation of attributes
    for index in range(len(training_data)):
        if (training_data.iloc[index, training_data.shape[1]-1]) == 1:
            ped_data_dataframe = ped_data_dataframe.append(training_data.iloc[index], ignore_index=True)
        else:
            non_ped_data_dataframe = non_ped_data_dataframe.append(training_data.iloc[index], ignore_index=True)

    # Calculate the probability of classes
    positive_probability = ped_data_dataframe[0].count()/training_data[0].count()
    negative_probability = non_ped_data_dataframe[0].count()/training_data[0].count()

    # Create summary matrix with mean and standard deviation for each class
    summary_pedestrian = summarize(ped_data_dataframe)
    summary_non_pedestrian = summarize(non_ped_data_dataframe)
    return summary_pedestrian, summary_non_pedestrian


def remove_class(test_data):
    print("Removing class column in test data...")
    test_data_classify = test_data.copy()
    last_column = test_data_classify.shape[1] - 1
    test_data_classify.drop([last_column], axis=1, inplace=True)
    return test_data


def classify(testdata):
    print("Running the classification...")
    pedDetected = 0
    nonPedDetected = 0
    global summary_pedestrian
    global summary_non_pedestrian
    global positive_probability
    global negative_probability

    # Make a copy of test data to create predicted_data
    test_data_copy = testdata.copy()
    class_column = len(test_data_copy.loc[0])

    # Calculate number of attributes to loop
    num_attributes = len(testdata.loc[0])

    # For each image and for each attribute calculate PDF and hence the probability of the attribute value for both
    # pedestrian and non pedestrian
    for index, row in testdata.iterrows():
        print("Classifying ")
        pedestrian_probability = 1
        non_pedestrian_probability = 1
        for i in range(num_attributes-2):
            pdf_eachpixel_positive = normal_pdf(row[i], summary_pedestrian[0][i], summary_pedestrian[1][i])
            pdf_eachpixel_negative = normal_pdf(row[i], summary_non_pedestrian[0][i], summary_non_pedestrian[1][i])
            pedestrian_probability = pedestrian_probability * pdf_eachpixel_positive
            non_pedestrian_probability = non_pedestrian_probability * pdf_eachpixel_negative

        # Calculate probability for each attribute
        pedestrian_probability = pedestrian_probability * positive_probability
        non_pedestrian_probability = non_pedestrian_probability * negative_probability

        # Normalize the probabilities
        pedestrian = pedestrian_probability / (pedestrian_probability + non_pedestrian_probability)
        non_pedestrian = non_pedestrian_probability / (pedestrian_probability + non_pedestrian_probability)

        # Check for class for each image
        if pedestrian > non_pedestrian:
            # Update predicted_data with 1 if pedestrian
            test_data_copy.loc[index, class_column] = 1
            pedDetected = pedDetected + 1
        else:
            # Update predicted_data with 0 if non pedestrian
            nonPedDetected = nonPedDetected + 1
            test_data_copy.loc[index, class_column] = 0
    return pedDetected, nonPedDetected, test_data_copy


def getAccuracy(testSet, predicted_data, pedDetected, nonPedDetected):
    print("Calculating accuracy and confusion Matrix...")
    ped_images = 0
    non_ped_images = 0

    # Calculate number of pedestrians and non pedestrians in initial test data
    for index, row in testSet.iterrows():
        if testSet.iloc[index][len(testSet.iloc[0])-1] == 1:
            ped_images += 1
        else:
            non_ped_images += 1

    # Initialize confusion matrix
    confusion_matrix = pd.DataFrame(index=[0, 1], columns=[0, 1])
    confusion_matrix = confusion_matrix.fillna(0)

    # Calculate confusion matrix
    for index, row in predicted_data.iterrows():
        if predicted_data.iloc[index][len(predicted_data.iloc[0]) - 1]:
            if predicted_data.iloc[index][len(predicted_data.iloc[0]) - 1] == testSet.iloc[index][len(testSet.iloc[0]) - 1]:
                confusion_matrix.iloc[0][0] += 1
            else:
                confusion_matrix.iloc[1][0] += 1
        else:
            if predicted_data.iloc[index][len(predicted_data.iloc[0]) - 1] == testSet.iloc[index][len(testSet.iloc[0]) - 1]:
                confusion_matrix.iloc[1][1] += 1
            else:
                confusion_matrix.iloc[0][1] += 1

    # Calculate Accuracy
    accuracy = (confusion_matrix.iloc[0][0] + confusion_matrix.iloc[1][1]) / (pedDetected + nonPedDetected)
    return confusion_matrix, accuracy


def main():
    # Create test data and training data
    global test_data
    global training_data
    input_data_set = create_training_data_set()
    test_data, training_data = split_data_set(input_data_set, 0.67)

    # Train the training model ( calculate mean and covariance)
    train(training_data)

    # Prepare test data
    test_data_classify = remove_class(test_data)

    # Run the test
    pedDetected, nonPedDetected, predicted_data = classify(test_data_classify)

    #Calculate Accuracy and Confusion Matrix
    confusion_matrix, accuracy = getAccuracy(test_data, predicted_data, pedDetected, nonPedDetected)

    print(confusion_matrix)
    print(pedDetected)
    print(nonPedDetected)
    print(accuracy)


main()
