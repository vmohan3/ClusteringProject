#include<iostream>
#include<string>
#include<malloc.h>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include<cstdio>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>
#define MAX_CLUSTER_SIZE 6
#define BLOCK_SIZE 16
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

__global__ void findMeans(const float *cluster, const float *instance, int *rnk, const int num, const int clus_size, const int maxAtoms)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	int i = threadIdx.x;
	if(i<num)
	{
		float min=0.0;
		for(int j=0;j<maxAtoms;j++)
		{
			float d=0.0;
			float s=0.0;
			for(int k=0;k<3;k++)
			{
				d=instance[(j*5)+k+2]-cluster[((0*clus_size)+j)*maxAtoms +k];
				s=s+(d*d);
			}
			s=sqrt(s);
			min+=s;
		}
		int cls=-1;
		for(int clus=0;clus<clus_size;clus++)
		{
			float val=0.0;
			for(int p=0;p<maxAtoms;p++)
			{
				float d=0.0;
				float s=0.0;
				for(int l=0;l<3;l++)
				{
					d=instance[((i*maxAtoms+p)*5)+l+2]-cluster[((clus*clus_size)+p)*maxAtoms +l];
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
		rnk[i]=cls+1;
	}
}

int main(void)
{
	cudaError_t err=cudaSuccess;
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
//	max_energy_states=32;
	cout<<"Total number of conformations : "<<max_energy_states<<endl;
	
	//allocate host cluster
	int s = MAX_CLUSTER_SIZE*max_atoms*3;
	size_t sz = s*sizeof(float);
	float *host_cluster=(float *)malloc(sz);
	if(host_cluster==NULL)
		cout<<"Not able to allocate host cluster"<<endl;

	//Initializing host cluster
	for(int i=0;i<MAX_CLUSTER_SIZE;i++)
	{
		int clus_num=rand()%max_energy_states + 0;
		for(long long int ii=0;ii<max_atoms;ii++)
                {
                     for(int jj=0;jj<3;jj++)
                     {
                         host_cluster[((i*MAX_CLUSTER_SIZE)+ii)*max_atoms+jj]=data[clus_num*max_atoms+ii][jj+2];
                     }
                }
	}
	
	//allocate device cluster
	float *d_clus = NULL;
	cudaMalloc((void **)&d_clus, sz);
	if(err!=cudaSuccess)
	{
		cout<<"Failed to allocate cluster on device!\n";
		exit(EXIT_FAILURE);
	}

	//initializing device cluster
	cudaMemcpy(d_clus, host_cluster, sz, cudaMemcpyHostToDevice);
	if(err!=cudaSuccess)
	{
		cout<<"Failed to copy cluster to device\n";
		exit(EXIT_FAILURE);
	}

	//allocate host instance
	float *host_instances;
	long int row=data.size();
	int col=5;
	long int size=row*col;
	host_instances=new float[size];
	for(long long int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			host_instances[i*col+j]=data[i][j];
		}
	}

	//allocate device instance
	float *device_instances;
	err=cudaMalloc((void **)&device_instances,size*sizeof(float));
	if(err!=cudaSuccess)
	{
		cout<<"error allocating device instance"<<endl;
		exit(EXIT_FAILURE);
	}
	
	err=cudaMemcpy(device_instances, host_instances, size*sizeof(float), cudaMemcpyHostToDevice);
	if(err!=cudaSuccess)

	{
                cout<<"error copying device instance"<<endl;
                exit(EXIT_FAILURE);
        }


	int *rnk_host;
	int rnk[max_energy_states];
	rnk_host=new int[max_energy_states];
	for(long long int i=0;i<max_energy_states;i++)
	{
                rnk_host[i]=0;
		rnk[i]=0;
	}
	int *rnk_device;
	err=cudaMalloc((void **)&rnk_device,max_energy_states*sizeof(int));
	if(err!=cudaSuccess)

	{
                cout<<"error allocating device rnk"<<endl;
                exit(EXIT_FAILURE);
        }

	err=cudaMemcpy(rnk_device, rnk_host, max_energy_states*sizeof(int), cudaMemcpyHostToDevice);
	if(err!=cudaSuccess)

	{
                cout<<"error allocating device instance"<<endl;
                exit(EXIT_FAILURE);
        }

//	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	int num_blocks=(max_energy_states+511)/512;
	//while(1)
	//{
		findMeans<<<48,256>>>(d_clus, device_instances, rnk_device, max_energy_states, MAX_CLUSTER_SIZE, max_atoms);
		err=cudaMemcpy(rnk_host, rnk_device, max_energy_states*sizeof(int), cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess)
		{
	                cout<<"error allocating device rnk"<<endl;
        	        exit(EXIT_FAILURE);
        	}
		
	
	//while(1)
	//{
		//E - Step
		for(long long int j=0;j<max_energy_states;j++)
		{
			double min=0.0;
			for(long long int y=0;y<max_atoms;y++)
			{
				double d=0.0,s=0.0;
				for(int z=0;z<3;z++)
				{
					d=host_instances[(y*5)+z+2]-host_cluster[((0*MAX_CLUSTER_SIZE)+y)*max_atoms +z];
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
						d=data[j*max_atoms+p][l+2]-host_cluster[((clus*MAX_CLUSTER_SIZE)+p)*max_atoms +l];
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
		int flag=0;
		for(long int i=0;i<100;i++)
		{
			if(rnk[i]!=rnk_host[i])
				flag=1;
			cout<<rnk[i]<<"\t"<<rnk_host[i]<<endl;
		}
		if(flag==1)
			cout<<"Not Matching"<<endl;

/*
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
	}*/
	/*for(int i=0;i<max_energy_states;i++)
	{
		cout<<rnk_host[i]<<" ";
	}
	cout<<endl;*/
	return 0;
}
