#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include<cstdio>
#define MAX_CLUSTER_SIZE 12
#define ITERATIONS 2
using namespace std;

typedef vector<double> record_t;
typedef vector<record_t> data_t;
long long int max_atoms=0;
long long int max_energy_states=0;
int k_clus,cls;
double cluster_lambda=0.0;

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
	k_clus=0;
	data_t data;
	ifstream infile("epac_pos.csv");
	infile>>data;
	if(!infile.eof())
	{
		cout<<"Error\n";
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
	cout<<"Total energy states : "<<max_energy_states<<endl;
	double cluster[MAX_CLUSTER_SIZE+1][max_atoms][3];
	double x_bar[max_atoms][3];
	for(long long int i=0;i<max_atoms;i++)
		for(int j=0;j<3;j++)
			x_bar[i][j]=0;
	for(long long int i=0;i<max_energy_states;i++)
	{
		for(long long int j=0;j<max_atoms;j++)
		{
			for(int k=0;k<3;k++)
			{
				x_bar[j][k]=x_bar[j][k]+data[i*max_atoms+j][k+2];
			}
		}
	}
	cout<<"x_bar array"<<endl;	
	for(long long int i=0;i<max_atoms;i++)
	{
		for(int j=0;j<3;j++)
		{
			x_bar[i][j]/=max_energy_states;
			cluster[0][i][j]=x_bar[i][j];
			cout<<cluster[0][i][j]<<"\t";
		}
		cout<<endl;
	}
	k_clus++;
	int rnk[max_energy_states];
	for(long long int i=0;i<max_energy_states;i++)
		rnk[i]=0;
	for(long long int i=0;i<max_energy_states;i++)
	{
		double dist=0.0;
		for(long long int j=0;j<max_atoms;j++)
		{
			double d=0.0;
			double s=0.0;
			for(int k=0;k<3;k++)
			{
				d=data[i*max_atoms+j][k+2]-x_bar[j][k];
				s=s+(d*d);
			}
			s=sqrt(s);
			dist+=s;
		}
		cluster_lambda+=dist;
	}
	cluster_lambda/=max_energy_states;
	cout<<"Cluster Lambda : "<<cluster_lambda<<endl;
	for(int i=0;i<ITERATIONS;i++)
	{
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
			for(int clus=0;clus<k_clus;clus++)
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
			if(min<=cluster_lambda||k_clus>MAX_CLUSTER_SIZE)
			{
				rnk[j]=cls;
			}
			else
			{
					//initialize new cluster and increment k
				rnk[j]=k_clus;
				//cout<<"Adding cluster "<<k_clus<<endl;
				for(long long int ii=0;ii<max_atoms;ii++)
				{
					for(int jj=0;jj<3;jj++)
					{
						cluster[k_clus][ii][jj]=data[j*max_atoms+ii][jj+2];
					}
				}
				k_clus++;
			}
			
		}
		int siz=k_clus;
		int sizes[siz];
		for(int jj=0;jj<siz;jj++)
			sizes[jj]=0;
		double means[siz][max_atoms][3];
		for(int jj=0;jj<siz;jj++)
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
					means[rnk[jj]][kk][ll]+=d;
					sizes[rnk[jj]]++;
				}
			}
		}
		for(int jj=0;jj<siz;jj++)
		{
			for(long long int kk=0;kk<max_atoms;kk++)
			{
				for(int ll=0;ll<3;ll++)
				{
					if(sizes[jj]!=0)
						cluster[jj][kk][ll]=means[jj][kk][ll]/sizes[jj];
					else
						cluster[jj][kk][ll]=0;
				}
			}
		}
		for(int ppp=0;ppp<siz;ppp++)
			cout<<sizes[ppp]<<" ";
		cout<<endl;		
	}
	/*for(int i=0;i<max_energy_states;i++)
	{
		cout<<rnk[i]<<" ";
	}
	cout<<endl;*/
	cout<<"k_clus = "<<k_clus<<endl;
	return 0;
}
