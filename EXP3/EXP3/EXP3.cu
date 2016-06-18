#include <algorithm>
#include <iostream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <ctime>
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")

using namespace std;

typedef long long ll; 
#define all(c) (c).begin(),(c).end() 
typedef pair<int,int> Pii;
typedef vector<int> Vi;

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice

#define THREADS 64

#define MAXLEN 210//note: change if will be longer strings
#define INF (1<<29)

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

void generate_random_combo_strings(string &s0, string &s1, const int len);

bool eval(const int *cur, const int *goal, const int len,int loc, int up, int down){
	if(loc>len)return false;
	return (((cur[loc]+up-down)%10+10)%10==goal[loc]);
}
inline int _3d_flat(int i, int j, int k, int D1,int D0){return i*D1*D0+j*D0+k;}

int cpu_version(const int *current, const int *goal, const int len){
	const int problem_space=(len+1)*(len+1)*(len+1);
	const int num_bytes=problem_space*sizeof(int);
	int ans=INF;

	int *DP=(int *)malloc(num_bytes);
	bool *A0=(bool *)malloc((len+1)*(len+1)*sizeof(bool));
	bool *A1=(bool *)malloc((len+1)*(len+1)*sizeof(bool));
	
	for(int i=1;i<problem_space;i++){
		DP[i]=INF;
	}

	DP[0]=0;
	for(int i=1;i<=len;i++){

		memset(A0,0,(len+1)*(len+1)*sizeof(bool));
		memset(A1,0,(len+1)*(len+1)*sizeof(bool));

		for(int j=0;j<=len;j++)for(int k=0;k<=len;k++){
			if(eval(current,goal,len,i,j,k))A0[j*(len+1)+k]=true;
		}
		for(int j=0;j<=len;j++)for(int k=0;k<=len;k++){
			if(DP[_3d_flat(i-1,j,k,(len+1),(len+1))]!=INF)
				A1[j*(len+1)+k]=true;
		}

		for(int x=0;x<=len;x++)for(int y=0;y<=len;y++)if(A0[x*(len+1)+y]){
			for(int xx=0;xx<=len;xx++)for(int yy=0;yy<=len;yy++)if(A1[xx*(len+1)+yy]){
				int temp=DP[_3d_flat(i-1,xx,yy,(len+1),(len+1))]+max(0,x-xx)+max(0,y-yy);
				if(temp<DP[_3d_flat(i,x,y,(len+1),(len+1))]){
					DP[_3d_flat(i,x,y,(len+1),(len+1))]=temp;
				}
			}
		}
	}
	for(int i=0;i<=len;i++)for(int j=0;j<=len;j++){
		if(DP[_3d_flat(len,i,j,(len+1),(len+1))]<ans){
			ans=DP[_3d_flat(len,i,j,(len+1),(len+1))];
		}
	}

	free(DP);
	free(A0);
	free(A1);
	return ans;
}

__constant__ int D_cur[MAXLEN];
__constant__ int D_goal[MAXLEN];

__device__ __forceinline__ int D_3d_flat(int i, int j, int k, int D1,int D0){return D0*(i*D1+j)+k;}

__global__ void set_DP(int *D_DP,const int problemspace){
	const int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if(offset<problemspace){
		D_DP[offset]=INF;
	}
}

__global__ void GPU_version(const int ii, int *D_DP, const int len){
	
	const int l=blockIdx.z;
	const int m=blockIdx.y;

	__shared__ int best;

	if(threadIdx.x==0){
		best=D_DP[D_3d_flat(ii-1,l,m,(len+1),(len+1))];
	}

	__syncthreads();

	if(best>=INF)return;

	const int j=threadIdx.x+blockIdx.x*blockDim.x;

	if(j>len)return;

	for(int k=0;k<=len;k++){
		if(((D_cur[ii]+j-k)%10+10)%10==D_goal[ii]){
			atomicMin(&D_DP[D_3d_flat(ii,j,k,(len+1),(len+1))],(best+max(0,j-l)+max(0,k-m)) );

		}
	}
}

__global__ void last_step(const int *D_DP,int *best_val,const int len){
	const int i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i<=len){
		atomicMin(&best_val[0],D_DP[D_3d_flat(len,i,blockIdx.y,(len+1),(len+1))]);
	}
}

int main(){

        srand(time(NULL));
		
		string s0="5390863801527525349142229108298075699798617845613912347987984732789432009090909090904218989432814923";
		string s1="5691764076679014302854836840311218635202200369261121447812739923746784821749837498209099098423788737";

		bool generate_random=true;
		if(generate_random){
			int big_combo_string_size=200;
			s0.clear();
			s1.clear();
			s0.resize(big_combo_string_size,'0');
			s1.resize(big_combo_string_size,'0');
			generate_random_combo_strings(s0,s1,big_combo_string_size);

			cout<<"\nstarting string= "<<s0<<'\n';
			cout<<"target string= "<<s1<<'\n';

		}


		const int s_len=s0.length();
		cout<<"\nLength= "<<s_len<<'\n';
		int *a0=(int *)malloc((s_len+1)*sizeof(int));
		int *a1=(int *)malloc((s_len+1)*sizeof(int));
		a0[0]=a1[0]=-1;
		for(int i=1;i<=s_len;i++){
			a0[i]=int(s0[i-1]-'0');
			a1[i]=int(s1[i-1]-'0');

		}
		int CPU_ans=0,GPU_ans=-1;
		//CPU
		cout<<"\nRunning CPU implementation..\n";
		UINT wTimerRes = 0;
		DWORD CPU_time=0,GPU_time=0;
		bool init = InitMMTimer(wTimerRes);
		DWORD startTime=timeGetTime();
	
		CPU_ans=cpu_version(a0,a1,s_len);

		DWORD endTime = timeGetTime();
		CPU_time=endTime-startTime;
		cout<<"CPU solution timing: "<<CPU_time<< " , answer= "<<CPU_ans<<'\n';
		DestroyMMTimer(wTimerRes, init);
		cudaError_t err=cudaFree(0);

		err=cudaMemcpyToSymbol(D_cur,a0,(s_len+1)*sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMemcpyToSymbol(D_goal,a1,(s_len+1)*sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		const int problemspace=(s_len+1)*(s_len+1)*(s_len+1);
		const int num_bytes=problemspace*sizeof(int);
		int *D_DP,*best_val;
		err=cudaMalloc((void**)&D_DP,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&best_val,sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		const int num_threads= (s_len>=128) ? 128:64;

		int ii=1,B_val=INF;
		dim3 Grid((s_len+num_threads)/num_threads,(s_len+1),(s_len+1));

		wTimerRes = 0;
		init = InitMMTimer(wTimerRes);
		startTime = timeGetTime();

		set_DP<<<(problemspace+num_threads-1)/num_threads,num_threads>>>(D_DP,problemspace);
		err = cudaDeviceSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err = cudaMemset(D_DP,0,sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		for(;ii<=s_len;ii++){
			GPU_version<<<Grid,num_threads>>>(ii,D_DP,s_len);
			err = cudaDeviceSynchronize();
			if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		}

		err=cudaMemcpy(best_val,&B_val,sizeof(int),_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		Grid.z=1;
		last_step<<<Grid,num_threads>>>(D_DP,best_val,s_len);
		err = cudaDeviceSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemcpy(&GPU_ans,best_val,sizeof(int),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		endTime = timeGetTime();
		GPU_time=endTime-startTime;
		cout<<"CUDA timing: "<<GPU_time<<" , answer= "<<GPU_ans<<'\n';
		DestroyMMTimer(wTimerRes, init);

		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_DP);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(best_val);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		free(a0);
		free(a1);

		err=cudaDeviceReset();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
   
        return 0;
}

bool InitMMTimer(UINT wTimerRes){
        TIMECAPS tc;
        if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
        wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
        timeBeginPeriod(wTimerRes); 
        return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init){
        if(init)
			timeEndPeriod(wTimerRes);
}

void generate_random_combo_strings(string &s0, string &s1, const int len){
	int r0=0,r1=0;
	for(int i=0;i<len;i++){
		r0=rand()%10;
		r1=rand()%10;
		s0[i]=char('0'+r0);
		s1[i]=char('0'+r1);
	}


}
