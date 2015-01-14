#include <windows.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "people_detect_thread.h"

using namespace std;
using namespace cv;

#define pthread_mutex_lock(a) WaitForSingleObject( *a, 10)
#define pthread_mutex_unlock(a) ReleaseMutex(*a)

// Data sharing, with assoc mutexes
HANDLE gMutex_people;
vector<Rect> gPeopleDetected;

HANDLE gMutex_frame;
Mat gFrame;

unsigned long thread_people_detect(void);

bool is_running = false;

void people::start_thread_people_detect()
{
	if(is_running)
		return;
	is_running = true;
	unsigned long threadID;
	void* thread;
	gMutex_people = CreateMutex(0, false, 0);
	gMutex_frame = CreateMutex(0, false, 0);
	thread = CreateThread( 0, 0, (LPTHREAD_START_ROUTINE) thread_people_detect, 0, 0, &threadID );
}

void people::stop_thread_people_detect()
{
	is_running = false;
}

void people::put_frame(const cv::Mat& f)
{
	pthread_mutex_lock(&gMutex_frame);
	f.copyTo(gFrame);
	pthread_mutex_unlock(&gMutex_frame);
}
void people::get_result(vector<Rect>& r)
{
	pthread_mutex_lock(&gMutex_people);
	r =  gPeopleDetected;
	pthread_mutex_unlock(&gMutex_people);
}

static const char* cascade_name =
    "C:\\OpenCV2.3\\opencv\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

unsigned long thread_people_detect(void)
{
#define MY_PREFIX if(0)
	printf("thread starting now ...\n");
	Mat img;
	vector<Rect> found,found_filtered;

	// Face Detection
	CascadeClassifier cascade_face;
	if(!cascade_face.load(cascade_name)) {
		printf("Cannot load face classifier\n");
	}

	// People Detection
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	int n = 0;
	int64 prev_tick = cv::getTickCount()-1;
	while(is_running)
	{
		int64 new_tick = cv::getTickCount();
		printf("%.1f people FPS\n",cv::getTickFrequency()/(new_tick - prev_tick));
		prev_tick = new_tick;

		int step = 0;
		MY_PREFIX printf("%d",++step);
		pthread_mutex_lock(&gMutex_frame);
		gFrame.copyTo(img);
		pthread_mutex_unlock(&gMutex_frame);
		
		if(!img.data) break;

		MY_PREFIX printf("%d",++step);

		// Start people detection
		//if(is_running) hog.detectMultiScale(img,found); 

		if(is_running) {
			cascade_face.detectMultiScale( img, found);
				/*1.1, 2, 0
				//|CV_HAAR_FIND_BIGGEST_OBJECT
				//|CV_HAAR_DO_ROUGH_SEARCH
				|CV_HAAR_SCALE_IMAGE
				,
				Size(30, 30) );*/
		}

		// Filter found rects
		size_t i,j;
		found_filtered.clear();
		for( i = 0; i < found.size(); i++ )
		{
			Rect r = found[i];
			for( j = 0; j < found.size(); j++ )
				if( j != i && (r & found[j]) == r)
					break;
			if( j == found.size() )
				found_filtered.push_back(r);
		}
		// End people detection

		
		MY_PREFIX printf("%d",++step);
		pthread_mutex_lock(&gMutex_people);
		gPeopleDetected = found_filtered;
		pthread_mutex_unlock(&gMutex_people);

		MY_PREFIX printf("%d",++step);
		if(found.size()>0 || found_filtered.size()>0)
		{
			printf("found %d people after filter\n", found_filtered.size());
		}
		MY_PREFIX printf("%d",++step);
		MY_PREFIX printf("\n");
	}
	printf("thread exiting now ...\n");
	return 0;
}
