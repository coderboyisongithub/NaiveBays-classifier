#ifndef NAIVE_BAYS
#define NAIVE_BAYS
#include <iostream>
#include <conio.h>
#include <stdlib.h>
#include <crtdbg.h>

using namespace cv;
using namespace std;


/*

	    -||-           -||-
            -||            -||-                          
            -\\   ((  ))  -//-
             -\\ // 01 \\-//-
	        |0  1  0| -
                \0101010/- 
          =======|01010|=======
	        /0101001\-
           -//-|101000100|-\\-
          -//- \\\\01////- -\\-
         -||-      \/       -||-
         -||-               -||- 
		 	

*/
class naiveBays
{

	bool training_data_class_intersection_status;
	bool training_lable_status;
	bool training_data_class_occurence_status;
	bool trained;
	bool ready_to_train;
	bool training_data_integerity_test_status;
	int t_samplesets;
	

	

//probablity functions

	//probablity estimators +2 OVERLOADS..
	float probablity(int n_of_favorable,int n_of_samples)
	{
		float p=(float)n_of_favorable/(float)n_of_samples;

	/*if(p<0.0 || p>1.0)
	{
		printf("\n!ERROR::value of probablity/(ies) exceeded its limit p<0 or p>1::check if for no of samples fed correctly..");

				abort(); //abortionn will be called if probablity jumps of the limit..
						 //this may occur if training parameters are fed due to certain 
						//mistake.. like error in entering t_samples correctly
	}
	*/
		return p;
	}
	float probablity(int n_of_favorable)
	{
	float p=((float)n_of_favorable/(float)t_samplesets);
	//printf("\nprobablity:%f ,sample_set:%d",p,t_samplesets);
	if(p<0.0)
	{
		printf("\n!ERROR::value of probablity/(ies) exceeded its limit p<0 or p>1::check if for no of samples fed correctly..");

				abort(); //abortionn will be called if probablity jumps of the limit..
						 //this may occur if training parameters are fed due to certain 
						//mistake.. like error in entering t_samples correctly
	}
		return p;
	}

	Point2i verifyClass(int ClassId)
	{
		//check for lable matrix for verification of fed class;
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

void check_Sum() //function to test integerity of training data..i.e it must not voilate certain training protocalls..
	{ 
		//update 29 jul,2018...
		/*

		I found a bug in my classifier..it was..when training data were well provided then obviously program 
		resulted correct result..But when training data were wrong i mean feeding with hypothetical data
		which has no significance in problem..then program resulted wrong result..
		till now one of the cause i discovered was ...
		1)how number of elements favorable to event will be smaller than 0!!
		2)if prior event is occuring n times then how can a conditional event related to that very prior 
			event can occur more than n times..
			i mean let for god sake..let we collected data for weather condition(sunny,cloudy) and match played(yes,no)..
			for six days..

			then if 3 days were sunny then how can it happen that match were played for 4 days knowing that 
			day was sunny.when actually we have 3 days only sunny..for such silly hypothetical data.
			To avoid such conflict i will check for folling condition..

			Let I=class intersection matrix,
				O=classOccurence matrix..
				following is the condition..

								1.			0<= I(r,c) <= O(r,0) ..:)
									
								2.			O(r,c)>=0;
		*/
		printf("\n|--checking for naive event matrix..");
		if(training_data_class_intersection_status==true && training_data_class_occurence_status==true)
		{
			for(int r=0;r<classOccurenceData.rows;r++)
			{
				for(int c=0;c<classOccurenceData.cols;c++)
				{
					if(classOccurenceData.at<float>(r,c)<0.0)
					{
						printf("\n!ERROR::training protocall voilation..aborting..elements of events are negative");
						printf("\ncheck for :: location [%dx%d]",r,c);
						abort();
					}
				}
			}
			printf("done..");
			printf("\n|--checking for conditional event matrix..");
			for(int r=0;r<intersectionData.rows;r++)
			{
				for(int c=0;c<intersectionData.cols;c++)
				{

					if(intersectionData.at<float>(r,c)>classOccurenceData.at<float>(r,0))
					{
						printf("\n!ERROR::training protocall voilation..aborting..::hypothetical training data \ncheck wheather elements in intersection matrix must not execeeds element in occurence matrix..");
						printf("\ninfo log\n error triggred at %dx%d\n element at intersection matrix(%d,%d) > class occurence(%d,0)",r,c,r,c,r);
						abort();
					}
					else if(intersectionData.at<float>(r,c)<0.0)
					{
						printf("\n!ERROR::training protocall voilation..aborting..::hypothetical training data \ncheck wheather elements in intersection matrix  should not be negative");
						abort();

					}

				}
			}
			printf("done..");

		}
		training_data_integerity_test_status=true; 
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
	 
	//flags 

	


public:					///PUBLIC SECTIONS


#define FOOD_CLASS_INTERSECTION	000
#define FOOD_CLASS_OCCURENCE	001
#define FOOD_CLASS_LABLE		010
#define pmacro_largest			011
#define pmacro_smallest			100
#define pmacro_mean				101
	

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
		 ready_to_train=false;
		  
		 training_data_integerity_test_status=false;

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
		 ready_to_train=true;
		 training_data_integerity_test_status=false;

		 check_Sum(); 

		  
	 }

	 void feed_data(Mat food,int food_type)
	 {

		 /*
		 Three definition are defined above the program 
		 000 defines that data fed to classifier is going to feed numerical data on occurence of events over other(to calculate conditional probablity //POSTERIOR PROBABLITY..//)
		 001 define that fed data to classifier is going to feed numerical data on occurence of each event we want to consider here(to calculate prior probablities or evidences probablity)
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
		 training_data_class_intersection_status=true;

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
			 training_data_class_occurence_status=true;
			 cout<<"\n"<<classOccurenceData;
		 }

																//feed classes lables to training matrix....
		 else if(food_type==FOOD_CLASS_LABLE)
		{
			if(training_data_class_intersection_status==false||training_data_class_occurence_status==false) //labeling without training data is worth na??>...
			{
				puts("\n!-FEED TRAINING DATA FIRST BEFORE LABELING");
				training_lable_status=false;					//put flag to false.
				return;
			}
			
			else if(food.type()!=CV_32SC1){puts("\n!-TRAINING LABLE:NOT A 8-BIT MATRIX(SINGLE CHANNEL)");training_lable_status=false;return;}		//check weather food is not a JUNk!!!

			 t_lables=Mat(food.rows,food.cols,CV_32SC1);
			 for(int r=0;r<food.rows;r++)
			 {
				 for(int c=0;c<food.cols;c++)
				 {
					 t_lables.at<int>(r,c)=food.at<int>(r,c);
				 }
			 }
			 
		 printf("\nlables:\n");
		 training_lable_status=true;
		 cout<<t_lables;
		 }

		 ready_to_train=true;
	 }


	 void train()
	 {
		 if(ready_to_train==false)
		 {
			 
			 printf("1ERROR::naiveBays :: training failed..::..insufficient training data..::");
			 return;
		 }
		 else
		 {
		 }
		/* if(training_data_integerity_test_status==false)
		 {
			 printf("\nchecking integerity for training data..");
			 check_Sum();
			printf("\nDone..");
		 }
		 else{}
		 */
		  printf("\nchecking integerity for training data..");
			 
		  check_Sum();

			printf("\nDone..");
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

		 printf("\nprobablity matrix..\n");
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
			 //printf("\n  probablity::%f",probablity((int)intersectionData.at<float>(r,c),(int)classOccurenceData.at<float>(r,1)));
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
float predict(int evidenceClass,int posteriorClass)
	 {
		 Point2i _posterior_(verifyClass(posteriorClass)),_evidence_(verifyClass(evidenceClass));
		 //first check weather classifier is trained or not...

		 printf("Evidence(matrix location):%d,%d \nposterior(matrix location):%d,%d",_evidence_.x,_evidence_.y,_posterior_.x,_posterior_.y);

if(isTrained()==true)
		 { 
			 if(Location_Comparision2i(_evidence_,Point2i(-1,-1),false)!=true|| Location_Comparision2i(_posterior_,Point2i(-1,-1),false)!=true)  //if requested evidence or posterior is not labled in training data or say model was not trained for that event or class
		{
			//after requests are verified as genuene...Then lets check freedom of user...
			
			//			NOTE::::::
			/*

	    -||-           -||-
            -||            -||-                          
            -\\   ((  ))  -//-
	     -\\ // 01 \\-//-
	        |0  1  0| -
                \0101010/- 
          =======|01010|=======
		/0101001\-
           -//-|101000100|-\\-
          -//- \\\\01////- -\\-
         -||-      \/       -||-
         -||-               -||-
			
			*/
			//This models train itself taking col major events as evidence in conditional probablity... so user must not enter unsual input or it will crash...So to prevent crashing do this....
			{
			printf("\nclass verified...");
		/*Now the problem is how I am going to implement bays throem...,We have got priorClass which occured first,and we have to calculate probablity of evidenceClass after that
		It may be applied like this...

		p(evidenceClass|priorClass)=p(priorClass|evidenceClass) x p(evidenceClass)/p(priorClass|evidenceClass) x p(evidenceClass)xp(priorClass|evidenceClass2) x p(evidenceClass2)xp(priorClass|evidenceClass3) x p(evidenceClass3)......

		*/
		/*
		To pridict we are acutally doing wightened summation of probablities in denominator by iterating through fixed row in conditional probablity matrix and fixed col in a classprobablity matrix...
		
		*/
			

			float Bays_chance=0.0;
			float Bays_evidence=0.0;
			float Bays_prior=0.0;														//we Know that lable have one set of naive events in one row and other set in other..Where these two sets are conditional to each other..
																//Now we First need to classify posterior events and evidence by their locations...
			if(_evidence_.y==0 && _posterior_.y==1)			//i.e if evidence belongs to first col and _posterior belongs to second col of lable matrix...
			{
				printf("\nprobablity:Posterior %f\nProbablity:evidence %f\n",class_probablity.at<float>(_posterior_.x,_posterior_.y),class_probablity.at<float>(_evidence_.x,_evidence_.y));
				puts("Engaging Naive Bays algorithm..");
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
			return Bays_posterior_Probablity;

			}

			
			else
			{
				puts("\n!-ERROR input evidence class and posterior class lables are un-Identified..Check input..");
				return NULL;}
		}
}

	 
else{
			 puts("\n!-Error Un-recognised input evidence class or posterior class...");
			 return NULL;
    }
		 
	 }
	 }

	 void setTsample(int value)
{
	t_samplesets=value;
}

};	
#endif
