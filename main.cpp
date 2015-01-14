// Background average sample code done with averages and done with codebooks
// (adapted from the OpenCV book sample)
// 
// NOTE: To get the keyboard to work, you *have* to have one of the video windows be active
//      and NOT the consule window.
//
// Gary Bradski Oct 3, 2008.
// 
/* *************** License:**************************
  Oct. 3, 2008
  Right to use this code in any way you want without warrenty, support or any guarentee of it working.

  BOOK: It would be nice if you cited it:
  Learning OpenCV: Computer Vision with the OpenCV Library
    by Gary Bradski and Adrian Kaehler
    Published by O'Reilly Media, October 3, 2008
 
  AVAILABLE AT: 
    http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
    Or: http://oreilly.com/catalog/9780596516130/
    ISBN-10: 0596516134 or: ISBN-13: 978-0596516130    
************************************************** */

#include <opencv/cvaux.h>
#include <opencv/cxmisc.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "my_background_segm.hpp"

using namespace std;
using namespace cv;

#include "people_detect_thread.h"

//VARIABLES for CODEBOOK METHOD:
MyCvBGCodeBookModel* model = 0;
const int NCHANNELS = 3;
bool ch[NCHANNELS]={true,true,true}; // This sets what channels should be adjusted for background bounds

void help(void)
{
    printf("\nLearn background and find foreground using simple average and average difference learning method:\n"
        "\nUSAGE:\nbgfg_codebook [--nframes=300] [movie filename, else from camera]\n"
        "***Keep the focus on the video windows, NOT the consol***\n\n"
        "INTERACTIVE PARAMETERS:\n"
        "\tESC,q,Q  - quit the program\n"
        "\th    - print this help\n"
        "\tp    - pause toggle\n"
        "\ts    - single step\n"
        "\tr    - run mode (single step off)\n"
        "=== AVG PARAMS ===\n"
        "\t-    - bump high threshold UP by 0.25\n"
        "\t=    - bump high threshold DOWN by 0.25\n"
        "\t[    - bump low threshold UP by 0.25\n"
        "\t]    - bump low threshold DOWN by 0.25\n"
        "=== CODEBOOK PARAMS ===\n"
        "\ty,u,v- only adjust channel 0(y) or 1(u) or 2(v) respectively\n"
        "\ta    - adjust all 3 channels at once\n"
        "\tb    - adjust both 2 and 3 at once\n"
        "\ti,o    - bump upper threshold up,down by 1\n"
        "\tk,l    - bump lower threshold up,down by 1\n"
        "\tSPACE - reset the model\n"
        );
}

// alternative using FGD model
int main2(int argc, char* argv[]);


//
//USAGE:  ch9_background startFrameCollection# endFrameCollection# [movie filename, else from camera]
//If from AVI, then optionally add HighAvg, LowAvg, HighCB_Y LowCB_Y HighCB_U LowCB_U HighCB_V LowCB_V
//
int main(int argc, char** argv)
{
	//return main2(argc,argv);

	help();
    const char* filename = 0;
	Mat rawImage, yuvImage, ImaskCodeBook, ImaskCodeBookCC, ImaskCodeBookCC_hull, rgbfg;
	Mat gray, prevGray, image;
    vector<Point2f> points[2];
    VideoCapture capture;
	bool needToInit=false;
    Size winSize(10,10);
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);

    int c, n, nframes = 0;
    int nframesToLearnBG = 150;

    model = mycvCreateBGCodeBookModel();
    
    bool pause = false;
    bool singlestep = false;

    for( n = 1; n < argc; n++ )
    {
        static const char* nframesOpt = "--nframes=";
        if( strncmp(argv[n], nframesOpt, strlen(nframesOpt))==0 )
        {
            if( sscanf(argv[n] + strlen(nframesOpt), "%d", &nframesToLearnBG) == 0 )
            {
                help();
                return -1;
            }
        }
        else
            filename = argv[n];
    }

    if( !filename )
    {
        printf("Capture from camera\n");
		capture.open( CV_CAP_ANY );
    }
    else
    {
        printf("Capture from file %s\n",filename);
		capture.open( filename );
    }

	if( !capture.isOpened() )
    {
        printf( "\nCan not initialize video capturing\n\n" );
        return -1;
    }

	int64 prev_tick = cv::getTickCount()-1;
	double fps = 30.0;

    //MAIN PROCESSING LOOP:
    for(;;)
    {
        if( !pause )
        {
			capture >> rawImage;
            ++nframes;
			if(!rawImage.data) 
                break;
        }
        if( singlestep )
            pause = true;

		vector<Rect> found;
		people::get_result(found);
		int64 new_tick = cv::getTickCount();
		fps = .5 * fps + .5 * (cv::getTickFrequency()/(new_tick - prev_tick));
		prev_tick = new_tick;
		printf("Image #%d captured - %.1f FPS - %d people\n",nframes,fps,found.size());

		//First time:
        if( nframes == 1 && rawImage.data )
        {
            // CODEBOOK METHOD ALLOCATION
			yuvImage = rawImage.clone();
			ImaskCodeBook.create(rawImage.size(),CV_8U);
			ImaskCodeBookCC.create(rawImage.size(),CV_8U);;
            ImaskCodeBookCC_hull.create(rawImage.size(),CV_8U);
			rgbfg.create(rawImage.size(),CV_8U);
			ImaskCodeBook.setTo(cv::Scalar(255));

			namedWindow( "Rgbfg" );
			namedWindow( "Raw" );
			namedWindow( "Tracking" );
            namedWindow( "ForegroundCodeBook");
            namedWindow( "CodeBook_ConnectComp");
            namedWindow( "CodeBook_ConnectComp_hull");
			printf("Images & Windows initialized\n");
        }

        // If we've got an rawImage and are good to go:                
        if( rawImage.data )
        {
			rawImage.copyTo(rgbfg);
            cvtColor( rawImage, yuvImage, CV_BGR2YCrCb );//YUV For codebook method
#define MY_PREFIX if(0) 

			//This is where we build our background model
			// when in initial training, update M, otherwise go through H cache
			if(nframes <= nframesToLearnBG) {
				MY_PREFIX printf("Updating codebook for frame #%d ...\n", nframes);
				mycvBGCodeBookUpdate( model, &(IplImage)yuvImage);
				MY_PREFIX printf("done\n");
			}
            if( nframes == nframesToLearnBG  ) {
				people::put_frame(ImaskCodeBook);
				//people::start_thread_people_detect();
				printf("Clearing stale with t=%d ... ", model->t);
                mycvBGCodeBookClearStale( model, model->t/2 );
				printf("done\n");
			}
            
            //Find the foreground if any
            if( nframes > nframesToLearnBG  )
            {
                // Find foreground by codebook method
				MY_PREFIX printf("Diffing codebook for frame #%d ...\n", nframes);
				mycvBGCodeBookDiff( model, &(IplImage)yuvImage, &(IplImage)ImaskCodeBook );
				MY_PREFIX printf("done\n");
                // This part just to visualize bounding boxes and centers if desired
				ImaskCodeBook.copyTo(ImaskCodeBookCC);    
                ImaskCodeBook.copyTo(ImaskCodeBookCC_hull);    
                cvSegmentFGMask( &(IplImage)ImaskCodeBookCC	    , 1, 4.0  );
                cvSegmentFGMask( &(IplImage)ImaskCodeBookCC_hull, 1, 10.0 );

				rgbfg.setTo(Scalar(0));
				rawImage.copyTo(rgbfg,ImaskCodeBookCC>0);
				people::put_frame(rgbfg);

				vector<Rect> found;
				people::get_result(found);
				for( vector<Rect>::const_iterator i = found.begin(); i != found.end(); ++i )
				{
					rectangle(rawImage, *i, cv::Scalar(0,255,0));
				}

				cvtColor(rgbfg, gray, CV_BGR2GRAY); 
				//rawImage.copyTo(gray);
				if( needToInit )
				{
					// automatic initialization
					goodFeaturesToTrack(gray, points[1], 500, 0.01, 10, Mat(), 3, 0, 0.04);
					cornerSubPix(gray, points[1], winSize, Size(-1,-1), termcrit);
					printf("goodFeaturesToTrack: points[0].size()=%d, [1].size()=%d\n", points[0].size(), points[1].size());
					points[0] = points[1];
				}
				else if( !points[0].empty() )
				{
					vector<uchar> status;
					vector<float> err;
					if(prevGray.empty())
						gray.copyTo(prevGray);
					calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
										 3, termcrit, 0);
					printf("calcOpticalFlowPyrLK: points[0].size()=%d, [1].size()=%d\n", points[0].size(), points[1].size());
					size_t i, k;
					for( i = k = 0; i < points[1].size(); i++ )
					{
						if( !status[i] )
							continue;

						points[1][k++] = points[1][i];
						circle( rawImage, points[1][i], 3, Scalar(0,255,0), -1, 8);
					}
					points[1].resize(k);
				}

				needToInit = false;
				imshow("Tracking", rawImage);

			}
            //Display
			imshow( "Rgbfg", rgbfg);
            imshow( "Raw", rawImage );
            imshow( "ForegroundCodeBook",ImaskCodeBook);
            imshow( "CodeBook_ConnectComp",ImaskCodeBookCC);
            imshow( "CodeBook_ConnectComp_hull",ImaskCodeBookCC_hull);
        }

        // User input:
        c = cvWaitKey(1)&0xFF;
        c = tolower(c);
        // End processing on ESC, q or Q
        if(c == 27 || c == 'q')
            break;
        //Else check for user input
        switch( c )
        {
        case 'h':
            help();
            break;
        case 'p':
            pause = !pause;
            break;
        case 's':
            singlestep = !singlestep;
            pause = false;
            break;
        case 'r':
            pause = false;
            singlestep = false;
            break;
        case 'f':
            needToInit = true;
            break;
        case ' ':
            mycvBGCodeBookClearStale( model, 0 );
            nframes = 0;
            break;
            //CODEBOOK PARAMS
        case 'y': case '0':
        case 'u': case '1':
        case 'v': case '2':
        case 'a': case '3':
        case 'b': 
            ch[0] = c == 'y' || c == '0' || c == 'a' || c == '3';
            ch[1] = c == 'u' || c == '1' || c == 'a' || c == '3' || c == 'b';
            ch[2] = c == 'v' || c == '2' || c == 'a' || c == '3' || c == 'b';
            printf("CodeBook YUV Channels active: %d, %d, %d\n", ch[0], ch[1], ch[2] );
            break;
        case 'i': //modify max classification bounds (max bound goes higher)
        case 'o': //modify max classification bounds (max bound goes lower)
        case 'k': //modify min classification bounds (min bound goes lower)
        case 'l': //modify min classification bounds (min bound goes higher)
            {
            uchar* ptr = c == 'i' || c == 'o' ? model->modMax : model->modMin;
            for(n=0; n<NCHANNELS; n++)
            {
                if( ch[n] )
                {
                    int v = ptr[n] + (c == 'i' || c == 'l' ? 1 : -1);
                    ptr[n] = CV_CAST_8U(v);
                }
                printf("%d,", ptr[n]);
            }
            printf(" CodeBook %s Side\n", c == 'i' || c == 'o' ? "High" : "Low" );
            }
            break;
        }
    }        

	capture.release();
	destroyAllWindows();
	people::stop_thread_people_detect();
	printf("main exiting now ...\n");
	ExitProcess(0);
}
