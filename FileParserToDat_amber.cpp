#include<iostream>
#include<string>
#include<fstream>
#include<sstream>
#include<vector>

using namespace std;

typedef vector<double> record_t;
typedef vector<record_t> data_t;
long long int max_atoms=0;
long long int max_conformations=0;
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

int main(void)
{
        data_t data;
        ifstream infile("all_amber_pos.csv");
	ofstream outfile;
	outfile.open("all_amber_pos.dat");
        infile>>data;
        if(!infile.eof())
        {
                cout<<"Error! Please check file.\n";
                return 1;
        }
        infile.close();
        cout<<"Number of rows in file : "<<data.size()<<endl;
	double t=0;
        for(int i=0;i<data.size();i++)
        {
                t=data[i][0];
                data[i][0]=data[i][1];
                data[i][1]=t;
        }
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
        max_conformations=data[data.size()-1][0];
        cout<<"Total number of conformations : "<<max_conformations<<endl;
	outfile<<max_conformations<<"\n";
	outfile<<"12\n";
	outfile<<max_atoms*3<<"\n";
	for(long int j=0;j<max_conformations;j++)
	{
		for(long int p=0;p<max_atoms;p++)
		{
			for(int l=0;l<3;l++)
			{
				outfile<<data[j*max_atoms+p][l+2]<<"\n";
			}
		}
	}
	outfile.close();
        return 0;
}
