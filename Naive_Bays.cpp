#include "stdafx.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <conio.h>
#include <iostream>
using namespace std;
using namespace cv;

float a[]={3.0,0.0,1.0,2.0};
int lable[]={1,-1,2,-2};

float occ[]={3,4,3,2};
class naiveBays
{

	bool training_data_class_intersection_status;
	bool training_lable_status;
	bool training_data_class_occurence_status;
	bool trained;
	int t_samplesets;
//probablity functions

	//probablity estimators +2 OVERLOADS..
	float probablity(int n_of_favorable,int n_of_samples)
	{
		return ((float)n_of_favorable/(float)n_of_samples);
	}
	float probablity(int n_of_favorable)
	{
		printf("%d\n",t_samplesets);
		return ((float)n_of_favorable/(float)t_samplesets);
	}

	Point2i verifyClass(int ClassId)
	{
		//check for lable matrix for verification of feeded class;
		//iterating throught lable matrix....
		for(int r=0;r<t_lables.rows;r++)
		{
			for(int c=0;c<t_lables.cols;c++)
			{
				if(ClassId==t_lables.at<int>(r,c))
				{
					printf("v:%d\n",t_lables.at<int>(r,c));
					return Point2i(r,c);}
				else
					continue;
			}
			
		}
		return Point2i(-1,-1);
	}
	bool Location_Comparision2i(Point2i src,Point2i target,bool Each)
	{
		//Either variable is responsible to choose weather return true for either of comparision matches or for each of comparision matching...
		switch(Each)
		{
		case true:
			{
				if(src.x==target.x && src.y==target.y)
					{return true;
				break;}
				else
					{return false;
				break;}
			}
		case false:
			{
				if(src.x==target.x ||src.y==target.y)
					{return true;
				break;}
				else
					{return false;
				break;}
			}
		}
	}
	bool isTrained()
	{
		switch(trained)
		{
		case true:
			{
				return true;
				break;
			}
		case false:
			{
				return false;
				break;
			}
		}
	}
/*
This matrix is like this..
lable matrix(let be this..)
| 1|-1|
| 2|-2|

representation of lables  over training data..
(intersection matrix or corelation matrix..)
 -1         -2
1| data1 | data2 |
2| data3 | data 4|

each lable represents class or events.. then there must be probablity of these events or classes...
Let us arrange these probablity in following way in a class_probablity matix..

|p(1) | p(-1)|  =========\ | 1|-1|  [ANALOGOUS TO LABLE MATRIX ^...
|p(2) | p(-2)|  =========/ | 2|-2|

*/
	Mat class_probablity;
	Mat conditionalProbablity;

public:


#define FOOD_CLASS_INTERSECTION	000
#define FOOD_CLASS_OCCURENCE	001
#define FOOD_CLASS_LABLE		010

	/* traning data matrix is actually a co-relation matrix holding frequency distribution  of sample data over other or say with respect to other class or sample data;
	TRAINING DATA MUST BE A 32-BIT FLOATING POINT MATRIX..
	*/
	Mat intersectionData;
	Mat classOccurenceData;
	 //total  no of sample set in training data.remember not necceary that training Data matrix will keep all elements of sample set so ,in additional we have to provide total no of sets
	Mat t_lables;	 //this matrix will hold numerical mapping of each class or events in trainning Data matrix
	 naiveBays()
	 {
		 t_samplesets=0;
		 training_data_class_intersection_status=false; //no training data regarding class intersections...
		 training_lable_status=false; //no lableing ..
		 training_data_class_occurence_status=false ;//no training data regarding class occurence in whole observation
		 trained=false;
	 }
	 
	 naiveBays(Mat class_intersections,Mat class_occurence,int t_samples,Mat Lables)
	 {
	/*opencv dont provide any integer type matrix so i prepared a 32 -bit(4 byte) signed type instead..(integer requires about 4 bytes)
	integer requires about 32 bit bitfeilds to store data . a 32 bit signed would be better solution for this problem.to store negative as well as positive integers..
	*/
		 t_samplesets=t_samples;
		 training_data_class_intersection_status=true;
		 feed_data(class_intersections,FOOD_CLASS_INTERSECTION); //feed training data to classifier..(32-bit floating point single channel matrix)
		 training_data_class_occurence_status=true;
		 feed_data(class_occurence,FOOD_CLASS_OCCURENCE); //feed training data to classsifier(32-bit float)..
		 training_lable_status=true;
		 feed_data(Lables,FOOD_CLASS_LABLE);		//feed taining lables over classifier..(32-bit signed(~4byte integer equilavent) single channel matrix)
		 trained=false;
	 }

	 void feed_data(Mat food,int food_type)
	 {

		 /*
		 Three definition are defined above the program 
		 000 defines that data feeded to classifier is going to feed numerical data on occurence of events over other(to calculate conditional probablity //POSTERIOR PROBABLITY..//)
		 001 define that feeded data to classifier is going to feed numerical data on occurence of each event we want to consider here(to calculate prior probablities or evidences probablity)
		 */

																	 //feed data to class intersection data..
		 if(food_type==FOOD_CLASS_INTERSECTION)
		 {
			 if(food.type()!=CV_32FC1){puts("!-intersectionData:NOT A 32-BIT FLOATING POINT :(");training_data_class_intersection_status=false; return;}		//32 bit one taste more good>>>..

			intersectionData=Mat(food.rows,food.cols,CV_32FC1);
			 for(int r=0;r<intersectionData.rows;r++)
			 {
				 for(int c=0;c<intersectionData.cols;c++)
				 {
					intersectionData.at<float>(r,c)=food.at<float>(r,c);
				 }
			 }
			 
		 printf("\nintersections:\n");
		 cout<<intersectionData;
		 }

																 //feed class details regarding occurence of class..
		 else if(food_type==FOOD_CLASS_OCCURENCE)
		 {
			 classOccurenceData=Mat(food.rows,food.cols,CV_32FC1);
			 if(food.type()!=CV_32FC1){puts("!-NEEDS 32-BIT FLOATING(SINGLE CHANNEL)MATRIX..");training_data_class_occurence_status=false;return;}
			 for(int r=0;r<food.rows;r++)
			 {
				 for(int c=0;c<food.cols;c++)
				 {
					 classOccurenceData.at<float>(r,c)=food.at<float>(r,c);
				 }
			 }

			 printf("\nclass occurence\n");
			 cout<<"\n"<<classOccurenceData;
		 }

																//feed classes lables to training matrix....
		 else if(food_type==FOOD_CLASS_LABLE)
		{
			if(training_data_class_intersection_status==false||training_data_class_occurence_status==false) //labeling without training data is worth na??>...
			{
				puts("!-FEED TRAINING DATA FIRST BEFORE LABELING");
				training_lable_status=false;					//put flag to false.
				return;
			}
			
			else if(food.type()!=CV_32SC1){puts("!-TRAINING LABLE:NOT A 8-BIT MATRIX(SINGLE CHANNEL)");training_lable_status=false;return;}		//check weather food is not a JUNk!!!

			 t_lables=Mat(food.rows,food.cols,CV_32SC1);
			 for(int r=0;r<food.rows;r++)
			 {
				 for(int c=0;c<food.cols;c++)
				 {
					 t_lables.at<int>(r,c)=food.at<int>(r,c);
				 }
			 }
			 
		 printf("\nlables:\n");
		 cout<<t_lables;
		 }

	 }


	 void train()
	 {
		 puts("\nTraining model.....");
																				//before creating a event only or class only probablity we need to init probablity matrix..
		 class_probablity=Mat(t_lables.rows,t_lables.cols,CV_32FC1);			//32 bit floating point matrix..
		 for(int r=0;r<class_probablity.rows;r++)
		 {
			 for(int c=0;c<class_probablity.cols;c++)
			 {
				 class_probablity.at<float>(r,c)=probablity((int)classOccurenceData.at<float>(r,c));
			 }
		 }              

		 conditionalProbablity=Mat(intersectionData.rows,intersectionData.cols,CV_32FC1);

		 printf("\n\n\nprobablity matrix..\n");
		 cout<<class_probablity;
		

																				//Now we have probablity of all possible classes or events regarding dataset provided....
																					 /*Now time to calculate probablity of class intersection...
																					 It can be done in two ways .
																					 1)we create a saperate probablity matrix of intersection clsses and store it in memory
																					 2)use processors in real-time to engage probablity mass function to calculate probablity while executing baysian equation...
																					 I am going with first.....
																					 */
	 
/* data sets given in training sets along row and cols..each data classes mentioned along row or cols are mutually independent with each other..I am talking about "Naive" calssifier
but if we consider data sets along row and coloum..Then they are distinct but may not be independent to each other..
Like e.g from lable matrix mentioned above in comment ..events are 1,2(row) and -1,-2(col) here 1,2 and -1,-2 are naive events or classes.but (1,-2),(1,-1),(2,-1),(2,-2) are not
need to be naive ..Thats what make probablity conditional along two or more events;
*/
	 //calculating conditional probablity assuming mat col(-classes) as evidences..

		 /* to calculate conditional probablity->We know that classOccurence data contains frequency of each event occuring and intersectionData matrix contains frequency of events conditionally of independent events
		 corelation matrix is mapped as row contains one set of indepedent events and col  contains another set of independent events..
		 each row-col  represents occurence of events to other events as labled..
		 frequency of occurence of (independent)events are arranged in order how lable matrix is arranged ...Occurence of each lable(events) are represented again by each row and col
		 we need to operate these matices so that we obtain occurence of conditional events and occurence or independent events..
		 */

	 for(int r=0;r<conditionalProbablity.rows;r++)
	 {
		 for(int c=0;c<conditionalProbablity.cols;c++)
		 {
			 conditionalProbablity.at<float>(r,c)=probablity((int)intersectionData.at<float>(r,c),(int)classOccurenceData.at<float>(r,0));
		 }
	 }
	 printf("\nconditional probablity....\n");
	 cout<<conditionalProbablity;
	 //all needed data is colledted ..Now time to drain input data..
	 intersectionData.deallocate();
	 classOccurenceData.deallocate();
	 trained=true;							//set trained flag as true...
	 }


	 //now its time to create pridictor for calssifier..Finally time to apply bays throemm...
	 void pridict(int evidenceClass,int posteriorClass)
	 {
		 Point2i _posterior_(verifyClass(posteriorClass)),_evidence_(verifyClass(evidenceClass));
		 //first check weather classifier is trained or not...

		 printf("Evidence:%d,%d \nposterior:%d,%d",_evidence_.x,_evidence_.y,_posterior_.x,_posterior_.y);

if(isTrained()==true)
		 { 
			 if(Location_Comparision2i(_evidence_,Point2i(-1,-1),false)!=true || Location_Comparision2i(_posterior_,Point2i(-1,-1),false)!=false)  //if requested evidence or posterior is not labled in training data or say model was not trained for that event or class
		{
			//after requests are verified as genuene...Then lets check freedom of user...
			//This models train itself taking col major events as evidence in conditional probablity... so user must not enter unsual input or it will crash...So to prevent crashing do this....
			{
			printf("\nclass verified...");
		/*Now the problem is how I am going to implement bays throem...,We have got priorClass which occured first,and we have to calculate probablity of evidenceClass after that
		It may be applied like this...

		p(evidenceClass|priorClass)=p(priorClass|evidenceClass) x p(evidenceClass)/p(priorClass|evidenceClass) x p(evidenceClass)xp(priorClass|evidenceClass2) x p(evidenceClass2)xp(priorClass|evidenceClass3) x p(evidenceClass3)......

		*/
			printf("\nprobablity:Posterior(matrix location) %f\nProbablity:evidence(matrix location) %f\n",class_probablity.at<float>(_posterior_.x,_posterior_.y),class_probablity.at<float>(_evidence_.x,_evidence_.y));
		/*
		To pridict we are acutally doing wightened summation of probablities in denominator by iterating through fixed row in conditional probablity matrix and fixed col in a classprobablity matrix...
		
		*/
			

			float Bays_chance=0.0;
			float Bays_evidence=0.0;
			float Bays_prior=0.0;														//we Know that lable have one set of naive events in one row and other set in other..Where these two sets are conditional to each other..
																//Now we First need to classify posterior events and evidence by their locations...
			if(_evidence_.y==0 && _posterior_.y==1)			//i.e if evidence belongs to first col and _posterior belongs to second col of lable matrix...
			{puts("Engaging Naive Bays algorithm..");
				puts("\ncalculating Bays Evidence probablity...");
				for(int a=0;a<class_probablity.rows;a++)
				{
					Bays_evidence+=class_probablity.at<float>(a,_posterior_.y)*conditionalProbablity.at<float>(_evidence_.x,a); //calculation of Bays evidence..
				}
				printf("\nbays_evidence:%f",Bays_evidence);
				puts("\ncalculating Bays Prior event probablity..");
				 Bays_prior=conditionalProbablity.at<float>(_evidence_.x,_posterior_.x)*class_probablity.at<float>(_posterior_.x,_posterior_.y);  //calulation of Bays _prior probablity
				printf("\nbays_prior..:%f",Bays_prior);
				float Bays_posterior_Probablity=(float)(Bays_prior/Bays_evidence);
				puts("\nFinal Bays Posterior Probablity");
			printf("\nFinal event pridiction...%f",Bays_posterior_Probablity);

			}

			
			else
			{
				puts("!-ERROR input evidence class and posterior class lables are un-Identified..Check input..");
				return;}
		}
}
}
	 
else
		 {
			 puts("!-Error Un-recognised input evidence class or posterior class...");
		 }
		 
	 }
};		 


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

	naiveBays bays(data,classes,6,l);
	
	bays.train();
	bays.pridict(2,-1);
	printf("%d bytes",sizeof(naiveBays));
	
	_getch();
}