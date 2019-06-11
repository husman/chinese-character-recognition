#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>

using namespace cv;

int main2(int argc, char **argv) {
    int num_files = 0;
    int class_num = 0;
    std::vector<std::vector<double>> trainingData;
    std::vector<int> trainingClassMap;
    DIR *dir;
    DIR *dir2;
    struct dirent *ent;
    struct dirent *ent2;
    if ((dir = opendir("data/")) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name[0] != '.') {
                char characterDir[1024];
                sprintf(characterDir, "data/%s", ent->d_name);
                if ((dir2 = opendir(characterDir)) != NULL) {
                    while ((ent2 = readdir(dir2)) != NULL) {
                        if (ent2->d_name[0] != '.') {
                            sprintf(characterDir, "data/%s/%s", ent->d_name, ent2->d_name);
//                            printf("processing %s\n", characterDir);

                            // Process image file
                            Mat image;
                            image = imread(characterDir, 1);

                            if (!image.data) {
                                printf("No image data for %s\n", characterDir);
                                return -1;
                            }

                            Mat src_gray;
                            int thresh = 100;
                            Mat canny_output;
                            std::vector<std::vector<Point>> contours;
                            std::vector<Vec4i> hierarchy;

                            // Convert image to gray and blur it
                            cvtColor(image, src_gray, CV_BGR2GRAY);
//                            blur(src_gray, image, cv::Size(3, 3));

                            /// Detect edges using canny
//                            Canny(image, canny_output, thresh, thresh * 2, 3);

                            Mat grad;
                            int scale = 1;
                            int delta = 0;
                            int ddepth = CV_16S;
                            /// Generate grad_x and grad_y
                            Mat grad_x, grad_y;
                            Mat abs_grad_x, abs_grad_y;
                            /// Gradient X
                            //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
                            Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
                            convertScaleAbs( grad_x, abs_grad_x );
                            /// Gradient Y
                            //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
                            Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
                            convertScaleAbs( grad_y, abs_grad_y );
                            /// Total Gradient (approximate)
                            addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

                            // Find contours
                            findContours(
//                                    canny_output,
                                    grad,
                                    contours,
                                    hierarchy,
                                    RETR_CCOMP,
                                    CHAIN_APPROX_SIMPLE,
                                    cv::Point(0, 0)
                            );

                            std::vector<Point> allContours;
                            for (int x = 0; x < contours.size(); x++) {
                                for (int y = 0; y < contours[x].size(); y++) {
                                    allContours.push_back(contours[x][y]);
                                }
                            }

                            // Get the moments
                            cv::Moments mu = cv::moments(allContours, false);

                            // Get the Hu moments
                            double hu[7];
                            HuMoments(mu, hu);

//                            printf(
//                                    "%s -> (log) hu = <%f, %f, %f, %f, %f, %f, %f>\n",
//                                    characterDir,
//                                    (hu[0] < 0 ? -1 : 1) * log(std::abs(hu[0])),
//                                    (hu[1] < 0 ? -1 : 1) * log(std::abs(hu[1])),
//                                    (hu[2] < 0 ? -1 : 1) * log(std::abs(hu[2])),
//                                    (hu[3] < 0 ? -1 : 1) * log(std::abs(hu[3])),
//                                    (hu[4] < 0 ? -1 : 1) * log(std::abs(hu[4])),
//                                    (hu[5] < 0 ? -1 : 1) * log(std::abs(hu[5])),
//                                    (hu[6] < 0 ? -1 : 1) * log(std::abs(hu[6]))
//                            );

                            std::vector<double> features;
                            features.push_back((hu[0] < 0 ? -1 : 1) * log(std::abs(hu[0])));
                            features.push_back((hu[1] < 0 ? -1 : 1) * log(std::abs(hu[1])));
                            features.push_back((hu[2] < 0 ? -1 : 1) * log(std::abs(hu[2])));
                            features.push_back((hu[3] < 0 ? -1 : 1) * log(std::abs(hu[3])));
                            features.push_back((hu[4] < 0 ? -1 : 1) * log(std::abs(hu[4])));
                            features.push_back((hu[5] < 0 ? -1 : 1) * log(std::abs(hu[5])));
                            features.push_back((hu[6] < 0 ? -1 : 1) * log(std::abs(hu[6])));
                            trainingData.push_back(features);

                            trainingClassMap.push_back(class_num);
                            ++num_files;

                        }
                    }
                    closedir(dir2);
                }
                ++class_num;
            }
        }
        closedir(dir);
    } else {
        /* could not open directory */
        perror("");
        return EXIT_FAILURE;
    }

    std::cout << "Training Data:\n";
    for (int x = 0; x < trainingData.size(); x++) {
        std::cout << "[" << x << "]: <";
        for (int y = 0; y < trainingData[x].size(); y++) {
            std::cout << " " << trainingData[x][y];
        }
        std::cout << " >\n";
    }

    // train
    Mat training_mat(num_files, 7, CV_32FC1);
    Mat labels(num_files, 1, CV_32S);
    for (int i = 0; i < num_files; i++) {
        // training matrix
        training_mat.at<float>(i, 0) = (float) trainingData[i][0];
        training_mat.at<float>(i, 1) = (float) trainingData[i][1];
        training_mat.at<float>(i, 2) = (float) trainingData[i][2];
        training_mat.at<float>(i, 3) = (float) trainingData[i][3];
        training_mat.at<float>(i, 4) = (float) trainingData[i][4];
        training_mat.at<float>(i, 5) = (float) trainingData[i][5];
        training_mat.at<float>(i, 6) = (float) trainingData[i][6];

        // label character
        labels.at<float>(i, 0) = trainingClassMap[i];
    }


//    std::cout << "traning_mat: " << training_mat << std::endl;
//    std::cout << "training_labels: " << labels << std::endl;

    // SVM params
    Ptr<ml::SVM> svm = ml::SVM::create();
    // edit: the params struct got removed,
    // we use setter/getter now:
    svm->setType(ml::SVM::C_SVC);
    svm->setKernel(ml::SVM::POLY);
    svm->setGamma(5.383);
    svm->setDegree(2.67);

    // perform training
    svm->train(training_mat, ml::ROW_SAMPLE, labels);

    // save the training data
    svm->save("svm_character_trained"); // saving
    svm = svm->load<ml::SVM>("svm_character_trained"); // loading
    std::cout << "after load\n";

    // test against
    Mat image;
    image = imread("uploads/A/A.jpg", 1);

    if (!image.data) {
        printf("No test image data \n");
        return -1;
    }

    Mat src_gray;
    int thresh = 100;
    Mat canny_output;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;

    // Convert image to gray and blur it
    cvtColor(image, src_gray, CV_BGR2GRAY);
//    blur(src_gray, image, cv::Size(3, 3));

    /// Detect edges using canny
    Canny(image, canny_output, thresh, thresh * 2, 3);
//    Mat grad;
//    int scale = 1;
//    int delta = 0;
//    int ddepth = CV_16S;
//    /// Generate grad_x and grad_y
//    Mat grad_x, grad_y;
//    Mat abs_grad_x, abs_grad_y;
//    /// Gradient X
//    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
//    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
//    convertScaleAbs( grad_x, abs_grad_x );
//    /// Gradient Y
//    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
//    convertScaleAbs( grad_y, abs_grad_y );
//    /// Total Gradient (approximate)
//    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    // Find contours
    findContours(canny_output, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    std::cout << "contours.length = " << contours.size() << "\n";

    std::vector<Point> allContours;
    for (int x = 0; x < contours.size(); x++) {
        for (int y = 0; y < contours[x].size(); y++) {
            allContours.push_back(contours[x][y]);
        }
    }

    // Get the moments
    cv::Moments mu = cv::moments(allContours, false);

    // Get the Hu moments
    double hu[7];
    HuMoments(mu, hu);

//    printf("contours length = %lu\n", contours.size());
//    printf("(normal) hu = <%f, %f, %f, %f, %f, %f, %f>\n", hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6]);
    printf(
            "(log) hu = <%f, %f, %f, %f, %f, %f, %f>\n",
            (hu[0] < 0 ? -1 : 1) * log(std::abs(hu[0])),
            (hu[1] < 0 ? -1 : 1) * log(std::abs(hu[1])),
            (hu[2] < 0 ? -1 : 1) * log(std::abs(hu[2])),
            (hu[3] < 0 ? -1 : 1) * log(std::abs(hu[3])),
            (hu[4] < 0 ? -1 : 1) * log(std::abs(hu[4])),
            (hu[5] < 0 ? -1 : 1) * log(std::abs(hu[5])),
            (hu[6] < 0 ? -1 : 1) * log(std::abs(hu[6]))
    );

//    vector<Point2f> mc(contours.size());
//    RNG rng(12345);
//    for (size_t i = 0; i < contours.size(); i++) {
//        mc[i] = Point2f(static_cast<float>(mu[i].m10 / mu[i].m00), static_cast<float>(mu[i].m01 / mu[i].m00));
//    }
//    Mat drawing = (Mat)Mat::zeros(canny_output.size(), CV_8UC3);
//    for (size_t i = 0; i < contours.size(); i++) {
//        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//        drawContours(drawing, contours, (int) i, color, 2, 8, hierarchy, 0, Point());
//        circle(drawing, mc[i], 4, color, -1, 8, 0);
//    }

    // predict if the image is equal to the character (itself in this case)
    Mat img_mat_1d(1, 7, CV_32FC1);
    img_mat_1d.at<float>(0, 0) = (float) ((hu[0] < 0 ? -1 : 1) * log(std::abs(hu[0])));
    img_mat_1d.at<float>(0, 1) = (float) ((hu[1] < 0 ? -1 : 1) * log(std::abs(hu[1])));
    img_mat_1d.at<float>(0, 2) = (float) ((hu[2] < 0 ? -1 : 1) * log(std::abs(hu[2])));
    img_mat_1d.at<float>(0, 3) = (float) ((hu[3] < 0 ? -1 : 1) * log(std::abs(hu[3])));
    img_mat_1d.at<float>(0, 4) = (float) ((hu[4] < 0 ? -1 : 1) * log(std::abs(hu[4])));
    img_mat_1d.at<float>(0, 5) = (float) ((hu[5] < 0 ? -1 : 1) * log(std::abs(hu[5])));
    img_mat_1d.at<float>(0, 6) = (float) ((hu[6] < 0 ? -1 : 1) * log(std::abs(hu[6])));

    Mat res;
    float prediction = svm->predict(img_mat_1d, res);
    std::cout << "prediction = " << prediction << "\n";

    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", canny_output);
    waitKey(0);
//
//    namedWindow("Gray Edges", WINDOW_AUTOSIZE);
//    imshow("Gray Edges", canny_output);
//
//    namedWindow( "Contours", WINDOW_AUTOSIZE );
//    imshow( "Contours", drawing );

//    waitKey(0);

    return 0;
}

using namespace cv;
using namespace std;

Mat img;
int threshval = 100;

static void on_trackbar(int, void*)
{
    Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    Mat labelImage(img.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    std::vector<std::vector<Point>> contours(nLabels - 1);
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
//        contours.push_back(std::vector<Point>());
    }

    Mat dst(img.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
            if (label > 0) {
                contours[label - 1].push_back(Point(r, c));
            }
        }
    }

    std::vector<Point> allContours;
    int minX = -1;
    int minY = -1;
    for (int i = 0; i < contours.size(); i++) {
        std::cout << "contours[" << i << "]:\n";
        for (int j = 0; j < contours[i].size(); j++) {
            if (minX == -1 || minX > contours[i][j].x) {
                minX = contours[i][j].x;
            }
            if (minY == -1 || minY > contours[i][j].y) {
                minY = contours[i][j].y;
            }
        }
    }

    int maxX = 0, maxY = 0;
    for (int i = 0; i < contours.size(); i++) {
        for (int j = 0; j < contours[i].size(); j++) {
            int x = contours[i][j].x - minX;
            int y = contours[i][j].y - minY;
            if (x > maxX) {
                maxX = x;
            }
            if (y > maxY) {
                maxY = y;
            }
            allContours.push_back(Point(x, y));
        }
    }

    std::cout << "new dim: " << maxX << " x " << maxY << "\n";
    // Get the moments
    cv::Moments mu = cv::moments(allContours, false);

    // Get the Hu moments
    double hu[7];
    HuMoments(mu, hu);

//    printf("contours length = %lu\n", contours.size());
//    printf("(normal) hu = <%f, %f, %f, %f, %f, %f, %f>\n", hu[0], hu[1], hu[2], hu[3], hu[4], hu[5], hu[6]);
    printf(
            "(log) hu = <%f, %f, %f, %f, %f, %f, %f>\n",
            (hu[0] < 0 ? -1 : 1) * log(std::abs(hu[0])),
            (hu[1] < 0 ? -1 : 1) * log(std::abs(hu[1])),
            (hu[2] < 0 ? -1 : 1) * log(std::abs(hu[2])),
            (hu[3] < 0 ? -1 : 1) * log(std::abs(hu[3])),
            (hu[4] < 0 ? -1 : 1) * log(std::abs(hu[4])),
            (hu[5] < 0 ? -1 : 1) * log(std::abs(hu[5])),
            (hu[6] < 0 ? -1 : 1) * log(std::abs(hu[6]))
    );

    std::cout << "contour size = " << contours.size() << "\n";

    imshow( "Connected Components", dst );
}


int main( int argc, const char** argv )
{
    img = imread("data/A/A_1467576279_scribbler-image.jpg", 0);

    if(img.empty())
    {
        cout << "Could not read input image file: " << endl;
        return -1;
    }

    namedWindow( "Image", 1 );
    imshow( "Image", img );

    namedWindow( "Connected Components", 1 );
    createTrackbar( "Threshold", "Connected Components", &threshval, 255, on_trackbar );
    on_trackbar(threshval, 0);

    waitKey(0);
    return 0;
}