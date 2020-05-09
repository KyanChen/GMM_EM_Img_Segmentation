#include <QCoreApplication>
#include"opencv2/opencv.hpp"
using namespace std;
using namespace cv;
using namespace cv::ml;
//图形分割
void segment(Mat img) {
    namedWindow("srcImg", 0);
    imshow("srcImg", img);
    int width = img.cols;
    int height = img.rows;
    int dim = img.channels();
    int sampleCount = width*height;
    int clusterCount = 4;
    //颜色索引表
    Scalar colorTab[5] = {
        Scalar(0,255,255),
        Scalar(0,255,0),
        Scalar(255,0,0),
        Scalar(255,0,255),
        Scalar(255,255,0),

    };
    Mat points(sampleCount, dim, CV_32FC1, Scalar(10)); //sampleCount行，dim列


    int index = 0;
    //图像转数据点
    for (int i=0;i<height;i++)
    {
        for (int j=0;j<width;j++)
        {
            index = i*width + j;
            Vec3b bgr = img.at<Vec3b>(i, j);

            points.at<float>(index, 0) = static_cast<int>(bgr[0]);
            points.at<float>(index, 1) = static_cast<int>(bgr[1]);
            points.at<float>(index, 2) = static_cast<int>(bgr[2]);
        }
    }

    //EM
    Mat lables;
    Ptr<EM> em_model = EM::create();
    em_model->setClustersNumber(clusterCount);
    em_model->setCovarianceMatrixType(EM::COV_MAT_SPHERICAL);  // 协方差矩阵
    em_model->setTermCriteria(TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 100, 0.1));//停止条件
    em_model->trainEM(points, noArray(), lables, noArray());
    //结果点集转图像
    Mat segImg = Mat::zeros(img.size(), img.type());
    Mat sample(dim, 1, CV_32FC1);

    for (int i=0;i<height;i++)
    {
        for (int j=0;j<width;j++)
        {
             index= i*width + j;
            int lable = lables.at<int>(index, 0);
            Scalar color = colorTab[lable];

            Vec3b vec(colorTab[lable][0], colorTab[lable][1], colorTab[lable][2]);
            segImg.at<Vec3b>(i, j) = vec;
        }
    }

    namedWindow("segmentation", 0);
    imshow("segmentation", segImg);
    waitKey(0);
}


int main(int argc, char *argv[])
{
    Mat src = imread("/Users/keyanchen/Files/Code/GMM_EM_Img_Segmentation/Img_Segmentation/test2.jpeg");
    segment(src);
    return 0;
}
