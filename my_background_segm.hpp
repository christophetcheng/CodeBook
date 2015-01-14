#ifndef __OPENCV_MYBACKGROUND_SEGM_HPP__
#define __OPENCV_MYBACKGROUND_SEGM_HPP__

#include "opencv2/core/core.hpp"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MyCvBGCodeBookElem
{
    struct MyCvBGCodeBookElem* next;
	int tCreate;
    int tLastUpdate;
    int stale;
    uchar boxMin[3];
    uchar boxMax[3];
    uchar learnMin[3];
    uchar learnMax[3];
} MyCvBGCodeBookElem;

typedef struct MyCvBGCodeBookModel
{
    CvSize size;
    int t;
    uchar cbBounds[3];
    uchar modMin[3];
    uchar modMax[3];
    MyCvBGCodeBookElem** cbmap;
    MyCvBGCodeBookElem** cbmap_cache;
    CvMemStorage* storage;
    MyCvBGCodeBookElem* freeList;
	unsigned int T_Hdel, T_H2M, T_Mdel;
} MyCvBGCodeBookModel;


MyCvBGCodeBookModel* mycvCreateBGCodeBookModel();
void mycvReleaseBGCodeBookModel( MyCvBGCodeBookModel** model );
void mycvBGCodeBookUpdate( MyCvBGCodeBookModel* model, const CvArr* image,
                                CvRect roi CV_DEFAULT(cvRect(0,0,0,0)),
                                const CvArr* mask CV_DEFAULT(0) );

int mycvBGCodeBookDiff( MyCvBGCodeBookModel* model, const CvArr* image,
                             CvArr* fgmask, CvRect roi CV_DEFAULT(cvRect(0,0,0,0)) );

void mycvBGCodeBookClearStale( MyCvBGCodeBookModel* model, int staleThresh,
                                    CvRect roi CV_DEFAULT(cvRect(0,0,0,0)),
                                    const CvArr* mask CV_DEFAULT(0) );

CvSeq* mycvSegmentFGMask( CvArr *fgmask, int poly1Hull0 CV_DEFAULT(1),
                               float perimScale CV_DEFAULT(4.f),
                               CvMemStorage* storage CV_DEFAULT(0),
                               CvPoint offset CV_DEFAULT(cvPoint(0,0)));

#ifdef __cplusplus
}
#endif


#endif