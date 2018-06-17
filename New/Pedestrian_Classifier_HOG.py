import numpy as np
from skimage import feature
import glob
import pandas as pd
import random
from PIL import Image
from math import pi, exp

# path_to_non_ped_training = \
#     "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TrainingData/DaimlerBenchmark/TrainingData/NonPedestrians/18x36/1_non-ped_examples bmp/*.bmp"
# path_to_ped_training = "D:/PG/Sem3/IPPR/LAB/DaimlerBenchmark/TrainingData/DaimlerBenchmark/TrainingData/Pedestrians/18x36/*.bmp"bmp
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


def create_training_data_set():
    np_fd_list = []
    p_fd_list = []
    i = 0

    for eachImage in glob.glob(path_to_non_ped_training):
        i = i + 1

        print("reading training data file %d and creating no_pedestrian_data" % i)
        np_image = Image.open(eachImage, 'r')
        hog_np, hog_np_image = feature.hog(np_image, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1)
                                           , visualise=True)
        np_fd_list.append(list(hog_np))
    no_pedestrian_data = np.array(np_fd_list)
    no_pedestrian_data_numpy = np.concatenate((no_pedestrian_data, np.zeros((len(no_pedestrian_data), 1), dtype=np.int)), axis=1)
    no_pedestrian_data = pd.DataFrame(no_pedestrian_data_numpy)

    i = 0

    for eachImage in glob.glob(path_to_ped_training):
        i = i + 1

        print("reading training data file %d and creating pedestrian_data" % i)
        p_image = Image.open(eachImage, 'r')
        hog_p, hog_p_image = feature.hog(p_image, orientations=8, pixels_per_cell=(6, 6), cells_per_block=(1, 1)
                                             , visualise=True)
        p_fd_list.append(list(hog_p))
    pedestrian_data = np.array(p_fd_list)
    pedestrian_data_numpy = np.concatenate((pedestrian_data, np.ones((len(pedestrian_data), 1), dtype=np.int)), axis=1)
    pedestrian_data = pd.DataFrame(pedestrian_data_numpy)

    training_data = pd.concat([pedestrian_data, no_pedestrian_data], ignore_index=True)

    create_arff_file(pedestrian_data_numpy, no_pedestrian_data_numpy)

    return training_data


def create_csv_file(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(path_to_csv + filename + ".csv")


def create_arff_file(ped_data, nonped_data):
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
        mean.append(dataframe[i+1].mean())
        std_deviation.append(dataframe[i+1].std())
    return np.array([np.array(mean), np.array(std_deviation)])


def normal_pdf(x, mu, sigma):
    return 1.0 / (sigma * (2.0 * pi)**(1/2)) * exp(-1.0 * (x - mu)**2 / (2.0 * (sigma**2)))


def train(training_data):
    ped_data_dataframe = pd.DataFrame()
    non_ped_data_dataframe = pd.DataFrame()
    global summary_pedestrian
    global summary_non_pedestrian
    global positive_probability
    global negative_probability

    for index in range(len(training_data)):
        if (training_data.iloc[index, training_data.shape[1]-1]) == 1:
            ped_data_dataframe = ped_data_dataframe.append(training_data.iloc[index], ignore_index=True)
        else:
            non_ped_data_dataframe = non_ped_data_dataframe.append(training_data.iloc[index], ignore_index=True)
    positive_probability = ped_data_dataframe[0].count()/training_data[0].count()
    negative_probability = non_ped_data_dataframe[0].count()/training_data[0].count()
    summary_pedestrian = summarize(ped_data_dataframe)
    summary_non_pedestrian = summarize(non_ped_data_dataframe)
    return summary_pedestrian, summary_non_pedestrian


def classify(testdata):
    pedDetected = 0
    nonPedDetected = 0
    global summary_pedestrian
    global summary_non_pedestrian
    global positive_probability
    global negative_probability
    print(summary_pedestrian[0][0])
    print(summary_non_pedestrian[1][0])
    num_attributes = len(testdata.iloc[0])
    for index, row in testdata.iterrows():
        pedestrian_probability = 1
        non_pedestrian_probability = 1
        for i in range(num_attributes-2):
            pdf_eachpixel_positive = normal_pdf(row[i], summary_pedestrian[0][i], summary_pedestrian[1][i])
            pdf_eachpixel_negative = normal_pdf(row[i], summary_non_pedestrian[0][i], summary_non_pedestrian[1][i])
            pedestrian_probability = pedestrian_probability * pdf_eachpixel_positive * positive_probability
            non_pedestrian_probability = non_pedestrian_probability * pdf_eachpixel_negative * negative_probability
        pedestrian_probability = pedestrian_probability / (pedestrian_probability + non_pedestrian_probability)
        non_pedestrian_probability = non_pedestrian_probability / (pedestrian_probability + non_pedestrian_probability)
        if pedestrian_probability > non_pedestrian_probability:
            pedDetected = pedDetected + 1
        else:
            nonPedDetected = nonPedDetected + 1
    return pedDetected, nonPedDetected


def main():
    input_data_set = create_training_data_set()
    test_data, training_data = split_data_set(input_data_set, 0.67)
    create_csv_file(training_data, "training_data")
    train(training_data)
    pedDetected, nonPedDetected = classify(test_data)
    print(pedDetected)
    print(nonPedDetected)
    last_column = len(test_data)-1
#    print(len(test_data[test_data.last_column == 0]))


main()
