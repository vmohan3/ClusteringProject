#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include<cstdio>
#include<stdlib.h>
#define MAX_CLUSTER_SIZE 12
using namespace std;

typedef vector<double> record_t;
typedef vector<record_t> data_t;
long long int max_atoms=0;
long long int max_energy_states=0;
int k_clus,cls;
double cluster_lambda=0.0;
int f=0;
istream& operator >>(istream& ins, record_t &record)
{
	record.clear();
	string line;
	getline(ins,line);
	stringstream ss(line);
	string field;
	while(getline(ss,field,','))
	{
		stringstream fs(field);
		double d=0.0;
		fs>>d;
		record.push_back(d);
	}
	return ins;
}

istream& operator >> (istream& ins, data_t& data)
{
	data.clear();
	record_t record;
	while(ins>>record)
	{
		data.push_back(record);
	}
	return ins;
}

int main()
{
	data_t data;
	ifstream infile("alanine.csv");
	infile>>data;
	if(!infile.eof())
	{
		cout<<"Error! Please check file.\n";
		return 1;
	}
	infile.close();
	cout<<"Number of rows in file : "<<data.size()<<endl;
	for(long long int i=0;i<data.size();i++)
	{
		if(i>0)
		{
			if(data[i][1]<data[i-1][1])
			{
				max_atoms=i;
				break;
			}
		}
	}
	cout<<"Number of atoms : "<<max_atoms<<endl;
	max_energy_states=data[data.size()-1][0];
	cout<<"Total number of conformations : "<<max_energy_states<<endl;
	double cluster[MAX_CLUSTER_SIZE][max_atoms][3];

	//Initializing cluster

	for(int i=0;i<MAX_CLUSTER_SIZE;i++)
	{
		int clus_num=rand()%max_energy_states + 0;
		for(long long int ii=0;ii<max_atoms;ii++)
                {
                     for(int jj=0;jj<3;jj++)
                     {
                         cluster[i][ii][jj]=data[clus_num*max_atoms+ii][jj+2];
                     }
                }
	}
	int rnk[max_energy_states];
	for(long long int i=0;i<max_energy_states;i++)
		rnk[i]=0;
	
	while(1)
	{
		//E - Step
		for(long long int j=0;j<max_energy_states;j++)
		{
			double min=0.0;
			for(long long int y=0;y<max_atoms;y++)
			{
				double d=0.0,s=0.0;
				for(int z=0;z<3;z++)
				{
					d=data[y][z+2]-cluster[0][y][z];
					s=s+(d*d);
				}
				s=sqrt(s);
				min+=s;
			}
			cls=-1;
			for(int clus=0;clus<MAX_CLUSTER_SIZE;clus++)
			{
				double val=0.0;
				for(long long int p=0;p<max_atoms;p++)
				{
					double d=0.0,s=0.0;
					for(int l=0;l<3;l++)
					{
						d=data[j*max_atoms+p][l+2]-cluster[clus][p][l];
						s=s+(d*d);
					}
					s=sqrt(s);
					val+=s;
				}
				if(val<=min)
				{
					cls=clus;
					min=val;
				}
			}
			rnk[j]=cls+1;			
		}

		//M - Step
		int sizes[MAX_CLUSTER_SIZE];
		for(int jj=0;jj<MAX_CLUSTER_SIZE;jj++)
			sizes[jj]=0;
		double means[MAX_CLUSTER_SIZE][max_atoms][3];
		for(int jj=0;jj<MAX_CLUSTER_SIZE;jj++)
		{
			for(long long int kk=0;kk<max_atoms;kk++)
			{
				for(int ll=0;ll<3;ll++)
				{
					means[jj][kk][ll]=0;
				}
			}
		}
		for(long long int jj=0;jj<max_energy_states;jj++)
		{
			for(long long int kk=0;kk<max_atoms;kk++)
			{
				for(int ll=0;ll<3;ll++)
				{
					double d=data[jj*max_atoms+kk][ll+2];
					means[rnk[jj]-1][kk][ll]+=d;
					sizes[rnk[jj]-1]++;
				}
			}
		}
		for(int jj=0;jj<MAX_CLUSTER_SIZE;jj++)
		{
			for(long long int kk=0;kk<max_atoms;kk++)
			{
				for(int ll=0;ll<3;ll++)
				{
					if(cluster[jj][kk][ll]!=means[jj][kk][ll]/sizes[jj])
					{
						f=1;
					}
					if(sizes[jj]!=0)
						cluster[jj][kk][ll]=means[jj][kk][ll]/sizes[jj];
					else
						cluster[jj][kk][ll]=0;
				}
			}
		}
		if(f==0)
			break;		
		f=0;
	}
	for(int i=0;i<max_energy_states;i++)
	{
		cout<<rnk[i]<<" ";
	}
	cout<<endl;
	
	return 0;
}
