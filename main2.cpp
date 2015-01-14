#include <opencv2/opencv.hpp>
#include <opencv2/legacy/blobtrack.hpp>
#include <vector>
#include <Windows.h>

//#include <pthread.h>
#include "people_detect_thread.h"

using namespace std;
using namespace cv;


int main2(int argc, char* argv[])
{
	bool update_bg_model = true, detect_in_foreground = false;


	try {
		namedWindow("Video", CV_WINDOW_AUTOSIZE );
		namedWindow("Foreground Mask", CV_WINDOW_AUTOSIZE );
		namedWindow("Background", CV_WINDOW_AUTOSIZE );
		namedWindow("Foreground", CV_WINDOW_AUTOSIZE );
		
		VideoCapture cap;
		if(argc > 1) cap.open(argv[1]);
		else cap.open( CV_CAP_ANY );
		
		Mat img;
		vector<Rect> found_filtered;

		cap >> img;
		if( !img.data ) {
			printf("could grab no data\n");
			exit(-1);
		}

		people::put_frame(img);

		// now that we have one frame, start people detection
		people::start_thread_people_detect();

		Mat fake, fg;
		resize(imread("D:\\data\\fake.jpg"),fake,img.size());

		CvBGStatModel* bg_model = 0;
		CvBlobTrackerAuto* pTracker = cvCreateBlobTrackerAuto(0,0);

		for(int in=0;;++in) {
			cap >> img;
			if( !img.data ) {
				printf("could grab no data\n");
				break;
			}
			imshow("Video",img);
			
			printf(".");
			if(!bg_model)
			{
				//create BG model
				//bg_model = cvCreateGaussianBGModel( &(IplImage)img );
				CvFGDStatModelParams params;
				//int    Lc;			/* Quantized levels per 'color' component. Power of two, typically 32, 64 or 128.				*/
				//int    N1c;			/* Number of color vectors used to model normal background color variation at a given pixel.			*/
				//int    N2c;			/* Number of color vectors retained at given pixel.  Must be > N1c, typically ~ 5/3 of N1c.			*/
				//			/* Used to allow the first N1c vectors to adapt over time to changing background.				*/

				//int    Lcc;			/* Quantized levels per 'color co-occurrence' component.  Power of two, typically 16, 32 or 64.			*/
				//int    N1cc;		/* Number of color co-occurrence vectors used to model normal background color variation at a given pixel.	*/
				//int    N2cc;		/* Number of color co-occurrence vectors retained at given pixel.  Must be > N1cc, typically ~ 5/3 of N1cc.	*/
				//			/* Used to allow the first N1cc vectors to adapt over time to changing background.				*/

				//int    is_obj_without_holes;/* If TRUE we ignore holes within foreground blobs. Defaults to TRUE.						*/
				//int    perform_morphing;	/* Number of erode-dilate-erode foreground-blob cleanup iterations.						*/
				//			/* These erase one-pixel junk blobs and merge almost-touching blobs. Default value is 1.			*/

				//float  alpha1;		/* How quickly we forget old background pixel values seen.  Typically set to 0.1  				*/
				//float  alpha2;		/* "Controls speed of feature learning". Depends on T. Typical value circa 0.005. 				*/
				//float  alpha3;		/* Alternate to alpha2, used (e.g.) for quicker initial convergence. Typical value 0.1.				*/

				//float  delta;		/* Affects color and color co-occurrence quantization, typically set to 2.					*/
				//float  T;			/* "A percentage value which determines when new features can be recognized as new background." (Typically 0.9).*/
				//float  minArea;		/* Discard foreground blobs whose bounding box is smaller than this threshold.					*/
	
				params.Lc      = CV_BGFG_FGD_LC;
				params.N1c     = CV_BGFG_FGD_N1C;
				params.N2c     = CV_BGFG_FGD_N2C;

				params.Lcc     = CV_BGFG_FGD_LCC;
				params.N1cc    = CV_BGFG_FGD_N1CC;
				params.N2cc    = CV_BGFG_FGD_N2CC;

				params.delta   = CV_BGFG_FGD_DELTA;

				params.alpha1  = 0.5f; //CV_BGFG_FGD_ALPHA_1;
				params.alpha2  = 0.05f;//=50frames  CV_BGFG_FGD_ALPHA_2;
				params.alpha3  = 0.2f; // CV_BGFG_FGD_ALPHA_3;

				params.T       = 0.99f;// CV_BGFG_FGD_T;
				params.minArea = 400;// large areas of w*h> 100 CV_BGFG_FGD_MINAREA;

				params.is_obj_without_holes = 1;
				params.perform_morphing     = 1;
				//bg_model = cvCreateFGDStatModel( &(IplImage)img, &params );
				bg_model = cvCreateGaussianBGModel( &(IplImage)img );
				continue;
			}
			cvUpdateBGStatModel( &(IplImage)img, bg_model);
			cvSegmentFGMask(bg_model->foreground);

			IplImage ipl = img;
			pTracker->Process( &ipl, 0);
			printf("%d",pTracker->GetBlobNum());

			//  Copy frame to shared 
			//fake.copyTo(fg);
			fg *= 0;
			img.copyTo(fg,Mat(bg_model->foreground)>0);
			people::put_frame(fg);

			// Copy found from shared
			people::get_result(found_filtered);

			for( size_t i = 0; i < found_filtered.size(); i++ )
			{
				Rect r = found_filtered[i];
				// the HOG detector returns slightly larger rectangles than the real objects.
				// so we slightly shrink the rectangles to get a nicer output.
				r.x += cvRound(r.width*0.1);
				r.width = cvRound(r.width*0.8);
				r.y += cvRound(r.height*0.07);
				r.height = cvRound(r.height*0.8);
				rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
			}
			imshow("Video",img);
			if(bg_model) {
				imshow("Foreground Mask", Mat(bg_model->foreground));
				imshow("Foreground", fg);
				imshow("Background", Mat(bg_model->background));
			}
			int key = tolower( waitKey(30) );
			if (key == 'b')  {
				update_bg_model = !update_bg_model;
				printf("\nupdate_bg_model=%d\n",update_bg_model);
			}
			else if (key == 'p') {
				detect_in_foreground = !detect_in_foreground;
				printf("\ndetect_in_foreground=%d\n",detect_in_foreground);
			}
			else if(key >0) break;
		}
		destroyWindow( "Video" );
		destroyWindow( "Foreground" );
	}
	catch(std::exception& e) {
		printf("exception caught: %s",e.what());
	}
	catch(...) {
		printf("exception caught: ...");
	}
	ExitProcess(0);
}