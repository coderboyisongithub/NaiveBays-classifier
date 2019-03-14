#include "stdafx.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <conio.h>
#include <iostream>
#include "Naive_Bays.h"

using namespace std;
using namespace cv;

float a[]={3.0,0.0,1.0,2.0}; //array holding occurence of intersectd events(conditional);
int lable[]={1,-1,2,-2}; //lable to random variable
float occ[]={3,4,3,2}; //occurence of classes


void main()
{
	int count=0;
	Mat classes(2,2,CV_32FC1);
	Mat data(2,2,CV_32FC1);
	Mat l(2,2,CV_32SC1);
	
	for(int r=0;r<data.rows;r++)
	{
		for(int c=0;c<data.cols;c++)
		{
			data.at<float>(r,c)=a[count];
			count++;
		}
	}
	
	count=0;
	for(int r=0;r<classes.rows;r++)
	{
		for(int c=0;c<classes.cols;c++)
		{
			classes.at<float>(r,c)=(float)occ[count];
			count++;
		}
	}

	count=0;
	for(int r=0;r<data.rows;r++)
	{
		for(int c=0;c<data.cols;c++)
		{
			l.at<int>(r,c)=(int)lable[count];
			count++;
		}
	}

	//data is abstracted in raw form..mean observation are convertes into set and subsets of random variable...
	
	naiveBays bays(data,classes,6,l);
	
	bays.train();
	bays.pridict(2,-1); 			//predict P(X=-1|X=2) (X=a random variable)i.e probabllity of class '-1' given that class '2' occured...
	printf("%d bytes",sizeof(naiveBays));
	
	_getch();
}
