/*******************************************************/
/*     Title: cusim.cu (cuda2DMinModel)                */
/*                                                     */
/*     Authors: E. Bartocci (eziobart@ams.sunysb.edu)  */
/*              F. Fenton   (fhf3@cornell.edu)         */
/*                                                     */
/*     Description:                                    */
/*                                                     */
/*      The following code is part of "cuda2DMinModel" */
/*      a CUDA implementation of the 2D simulator of   */
/*      the Bueno-Orovio-Cherry-Fenton Minimal Model.  */
/*      This code is provided as a supplement of the   */
/*      paper "Teaching cardiac electrophysiology      */
/*      modeling to undergraduate students:            */
/*      "Lab exercises and GPU programming for the     */ 
/*       study of arrhythmias and spiral wave dynamics"*/
/*       submitted to Advances in Physiology Education */
/*                                                     */                               
/*                                                     */
/*                                                     */
/*     Date:  01/03/11                                 */
/*      Copyright 2011                                 */
/*                                                     */
/*                                                     */
/* ==   Free distribution with authors permission   == */
/*                                                     */
/* ==   SUNY Stony Brook, Stony Brook, NY              */
/* ==   Cornell University, Ithaca, NY                 */ 
/* ====================================================*/            

#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>
#include <ctime>
#include <string.h>
#include <math.h>
#include "cell_model.h"
#include "cusim.h"


Real  sim_time, dt, dx, diff;
int   nx, ny, tip_sampling_rate_steps, plotting_rate_steps,
      laplacian_stencil, write_AP, write_tip_data;

Real  * cpu_xlap, * gpu_xlap, *gpu_old_u, *cpu_old_u, * gpu_state_vars, * cpu_state_vars, * u,  
      * v,  * w, * s,  *tip_x, *tip_y, *nxtip,  *nytip,  *gpu_nxtip, *gpu_nytip, 
      *tip_times, *ap1, *ap2;

Real  * gpu_xu1, *gpu_xu2;

Real  * cpu_xu1, *cpu_xu2;
   
Real    ddt_o_dx2, ddt_o_6dx2;
int     field_size, nsteps, istep, write_tip, nend, ntips, 
        graphic_resolution, verbose, *ntip, nxb, *gpu_ntip, group; 




#define NX            nx
#define NY            ny
#define NNY          nny
#define NXB          nxb
#define NNYB        nnyb
#define NST_VARS       4


#define MAX_TIPS            10000
#define SPIRAL_PROTOCOL_ONE     1
#define MAX_SAMPLING        10000

#define SV(i,j)      state_var[INDEX(i,j)]
#define OU(i,j)      old_u[INDEX(i,j)]
#define XU1(i,j)     xu_1[INDEX(i,j)]
#define XU2(i,j)     xu_2[INDEX(i,j)]
#define L(i,j)       xlap[INDEX(i,j)]
#define M(i,j)       modes[INDEX(i,j)]




#define N nxb
const int blocksize = 16;

/* ====================================================
 *
 * checkCUDAError(const char *message)  
 *      
 *      check if there was an error generate by the last
 * call of the CUDA API and exit from the program.
 * 
 * Input - *message -> message to be printed in the 
 *                     standard error output
 * ====================================================*/
void checkCUDAError(const char *message) 
{
     cudaError_t error = cudaGetLastError();
     if(error!=cudaSuccess) 
     {
       fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
       exit(-1);   
     }
}


/* ====================================================
 *
 * checkCUDAversion()  
 *      
 *      check the Version of CUDA and print the 
 * information on the std output.
 * 
 * ====================================================*/
void checkCUDAVersion()
{
     int devID;
     cudaDeviceProp props;
     cutilSafeCall(cudaGetDevice(&devID));
     cutilSafeCall(cudaGetDeviceProperties(&props, devID));
     printf("CUDA Device name = %s\n", props.name);      
     int driverVersion = 0, runtimeVersion = 0; 
     cudaDriverGetVersion(&driverVersion);
     printf("CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
     cudaRuntimeGetVersion(&runtimeVersion);
     printf("CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
     printf("CUDA Capability Major/Minor version number:    %d.%d\n", props.major, props.minor);
     printf("CUDA device has %d multi-processors and %d stream processors\n", props.multiProcessorCount, props.multiProcessorCount*8 );
}

#define NEXT_LINE(fp) while(getc(fp)!='\n');  /* macro used to skip to end 
						 of input line */

#define NUM_PARAM_REAL                        5
#define NUM_PARAM_INT                         4

#define DT                                    0
#define DDT_O_DX2                             1
#define ONE_MINUS_4_x_DDT_O_DX2               2
#define ONE_MINUS_20_x_DDT_O_6DX2             3
#define DDT_O_6DX2                            4

#define FIELD_SIZE1                           0
#define FIELD_SIZE2                           1
#define FIELD_SIZE3                           2
#define SIZE                                  3

#define CDT                         gpu_real_costants[DT]
#define CDDT_O_DX2                  gpu_real_costants[DDT_O_DX2]
#define CONE_MINUS_4_x_DDT_O_DX2    gpu_real_costants[ONE_MINUS_4_x_DDT_O_DX2]
#define CONE_MINUS_20_x_DDT_O_6DX2  gpu_real_costants[ONE_MINUS_20_x_DDT_O_6DX2]
#define CDDT_O_6DX2                 gpu_real_costants[DDT_O_6DX2]

#define CFIELD_SIZE1                gpu_int_costants[FIELD_SIZE1]
#define CFIELD_SIZE2                gpu_int_costants[FIELD_SIZE2]
#define CFIELD_SIZE3                gpu_int_costants[FIELD_SIZE3]
#define CSIZE                       gpu_int_costants[SIZE]
__constant__ Real gpu_real_costants[NUM_PARAM_REAL];
__constant__ int gpu_int_costants[NUM_PARAM_INT];


/* ====================================================
 *
 * readParams()  
 *      
 *      read parameters from sim_paramters.dat if the 
 * the file exists, otherwise set default parameters
 * 
 * ====================================================*/

/*    2) Read parameters of simulation from            */
/*       initizil. file:                               */ 
/*          - sim_time - time of simulation in ms      */     
/*          - tip_sampling_rate_steps                  */
/*          - plotting_rate_steps                      */
/*          - dt  - integration step  in ms            */
/*          - dx  - distance among two cells in cm     */
/*          - diff - diffusion constant                */
/*          - laplacian_stencil - 0 is the 5 points,   */ 
/*                                1 is the 9 points    */
/*          - write_AP          - 0 false, 1 true      */
/*          - write_tip_data    - 0 false, 1 true      */

void readParams (void)
{
  /* Read task file, open output files, and initialize graphics and time
   * stepping.  */

  FILE *fp;
  double p_in;

  /* ----------------------------- 
   * Read parameters from sim_paramters.dat
   * ----------------------------- */

  if((fp=fopen("sim_parameters.dat","r"))==NULL) 
  {
	  
     if(verbose) fprintf(stderr,"Cannot open task file: sim_parameters.dat \n");
     //If the file does not exist I write the paramters 
   	 sim_time                =   2.0;
   	 tip_sampling_rate_steps = 100;
   	 plotting_rate_steps     = 100;
     
   	 dt                      =   0.05;//Original is 0.1
   	 dx                      =   0.02;//Original is 0.25
   	 diff                    =   0.00116;
   	 
   	 laplacian_stencil       =   0;
   	 write_AP                =   0;
   	 write_tip_data          =   0;
   	 

     N = 1024;
  }
  else 
  {
	printf("\n");
	fscanf(fp,"%lg",&p_in); NEXT_LINE(fp); sim_time=p_in;
	printf("Reading sim_time=%f\n", sim_time);
	fscanf(fp,"%d", &tip_sampling_rate_steps);  NEXT_LINE(fp);
	printf("Reading tip_sampling_rate_steps=%d\n", tip_sampling_rate_steps);
	fscanf(fp,"%d", &plotting_rate_steps);  NEXT_LINE(fp);
	printf("Reading plotting_rate_steps=%d\n", plotting_rate_steps);
    fscanf(fp,"%lg",&p_in);     NEXT_LINE(fp); dt=p_in;
    printf("Reading dt=%f\n", dt);
    fscanf(fp,"%lg",&p_in);     NEXT_LINE(fp); dx=p_in;
    printf("Reading dx=%f\n", dx);
    fscanf(fp,"%lg",&p_in);     NEXT_LINE(fp); diff=p_in;
    printf("Reading diff=%f\n", diff);
    fscanf(fp,"%d", &laplacian_stencil);       NEXT_LINE(fp);
    printf("Reading laplacian_stencil=%d\n", laplacian_stencil);
    fscanf(fp,"%d", &write_AP);       NEXT_LINE(fp);
    printf("Reading write_AP=%d\n", write_AP); 
    fscanf(fp,"%d", &write_tip_data);       NEXT_LINE(fp);
    printf("Reading write_tip_data=%d\n", write_tip_data); 
    fscanf(fp,"%d", &nxb);       NEXT_LINE(fp);
    printf("Reading grid_size=%d\n", nxb); 
    fclose(fp);
  }
  /* Define parameters values */

  nx               = nxb-2;//atoi(argv[2]);
  ny               = nxb-2;//atoi(argv[3]);

    field_size       = (ny+2)*(nx+2);
  	 ddt_o_dx2      = dt*diff/(dx * dx);
  	 ddt_o_6dx2     = dt*diff/(6*(dx * dx));

  printf("\n\nModel Parameters: \n");
  printf("dt     = %g\n", dt);
  printf("dx    = %g\n", dx);
  printf("diff   = %g\n", diff);
  printf("ddt_o_dx2   = %g\n", ddt_o_dx2);

  if (!laplacian_stencil)
  {
     printf("5 POINTS LAPLACIAN STENCIL SELECTED\n");
  }
  else
  {
     printf("9 POINTS LAPLACIAN STENCIL SELECTED\n");
  }
}


/* ====================================================
 *
 * writeConstantMemory()  
 *      
 *      write the parameters in the constant memory of 
 *  the GPU.
 * 
 * ====================================================*/
void writeConstantMemory (void)
{
	  /***************************************/
	  /*      Create Symbols Table           */
	  /***************************************/
	  Real * cpu_real_costants = (Real *) calloc (NUM_PARAM_REAL, sizeof (Real));
	  cpu_real_costants[DT]                       = dt;
	  cpu_real_costants[DDT_O_DX2]                = ddt_o_dx2;
	  cpu_real_costants[ONE_MINUS_4_x_DDT_O_DX2]  = 1.0 - (4.0 * ddt_o_dx2);
	  cpu_real_costants[ONE_MINUS_20_x_DDT_O_6DX2] = 1.0 - (20.0 * ddt_o_6dx2);
	  cpu_real_costants[DDT_O_6DX2]               = ddt_o_6dx2;
	  
	  cudaMemcpyToSymbol(gpu_real_costants, cpu_real_costants, NUM_PARAM_REAL*sizeof(Real));
	  free(cpu_real_costants);
	  
	  int * cpu_int_costants = (int *) calloc (NUM_PARAM_INT, sizeof (int));
	  cpu_int_costants[FIELD_SIZE1]             = N * N;
	  cpu_int_costants[FIELD_SIZE2]             = 2* N * N;
	  cpu_int_costants[FIELD_SIZE3]             = 3*N*N;
	  cpu_int_costants[SIZE]                     = N;
	  cudaMemcpyToSymbol(gpu_int_costants, cpu_int_costants, NUM_PARAM_INT*sizeof(int));
	  free(cpu_int_costants);    
	  printf("Constant memory written\n");
}


/* ====================================================
 *
 * cpuMemoryAllocation()  
 *      
 *      memory allocation
 * 
 * ====================================================*/
void cpuMemoryAllocation (void)
{
	cpu_state_vars = new Real [N*N*4];
    u              = cpu_state_vars;
   	v              = cpu_state_vars +     field_size;
   	w              = cpu_state_vars + 2 * field_size;
   	s              = cpu_state_vars + 3 * field_size;
   	cpu_xu1        = new Real [N*N]; 
   	cpu_xu2        = new Real [N*N];
   	cpu_old_u      = new Real [N*N];
   	ntip           = new int[1];
   	*ntip           = 0;
   	nxtip          = new Real[MAX_TIPS];
   	nytip          = new Real[MAX_TIPS];
   	tip_times      = new Real[MAX_TIPS];
   	ap1            = new Real[MAX_SAMPLING];
   	ap2            = new Real[MAX_SAMPLING];
   	
    printf("CPU Memory allocated in one block.\n");
}


/* ====================================================
 *
 * gpuMemoryAllocation()  
 *      
 *      memory allocation
 * 
 * ====================================================*/
void gpuMemoryAllocation (void)
{
	printf("Memory allocated in one block.\n");
    cudaMalloc( (void**)&gpu_state_vars, field_size * 4 * sizeof(Real));
    printf("Memory allocated in one block.\n");
    cudaMalloc ((void**)&gpu_xlap,       field_size *     sizeof(Real));
    cudaMalloc ((void**)&gpu_old_u,      field_size *     sizeof(Real));
    
    cudaMalloc ((void**)&gpu_xu1,      field_size *     sizeof(Real));
    cudaMalloc ((void**)&gpu_xu2,      field_size *     sizeof(Real));
    printf("Memory allocated in one block.\n");
}


/* ====================================================
 *
 * calc_lap_gpu_cable (Real *u, Real *lap, Real ddt_o_dx2, int nx)
 *      
 *      calculate the laplacian for the cable
 * 
 * ====================================================*/
__global__ void calcPDECable (Real *u, Real *lap, Real ddt_o_dx2, int nx){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i == 0) {
	    lap[i] = ddt_o_dx2 * (-2.*u[0] + 2.*u[1]);
	}else if (i == nx-1){
	    lap[i] = ddt_o_dx2 * (-2.*u[nx-1] +2.*u[nx-2]);
	}else{
	    lap[i] = ddt_o_dx2 * (u[i+1] + u[i-1] - 2.0 * u[i]);
	}
	
}

/* ====================================================
 *
 * cstep_epi_gpu_cable ( Real *state_vars_gpu, int nxb, 
 *                       Real dt, Real *lap, Real stim) 
 *      
 *      generate the initial condition calculating 
 *  the propagation of the stimulus before in a cable 
 *  and then copy and paste this on all the tissue
 * 
 * ====================================================*/
__global__ void  calcODECable ( Real *state_vars_gpu, int nxb, Real dt, Real *lap, Real stim){

	  Real u, v, w, s;
	  Real tvm, vinf, winf;
	  Real jfi, jso, jsi;
	  Real twm, tso, ts, to, ds, dw, dv;
	  
	  
	  
	  int i  = blockIdx.x * blockDim.x + threadIdx.x;
	  
	  
	  
	  if (i >= 0 && i < nxb-1){
		  
	      u     = state_vars_gpu[i];
	      v     = state_vars_gpu[i+nxb];
	      w     = state_vars_gpu[i+2*nxb];
	      s     = state_vars_gpu[i+3*nxb];
	      

	      tvm =  (u > THVM) ? TV2M : TV1M;
	      ts  = (u > THW)  ? TS2  : TS1;
	      to  = (u > THO)  ? TO2  : TO1;
          tso =  TSO1 + (TSO2_TSO1_DIVBY2)*(1.+tanh(KSO*(u-USO)));
          twm =  1./(TW1M + (TW2M_TW1M_DIVBY2)*(1.+tanh(KWM*(u-UWM))));   
          
          ds = (((1.+tanh(KS*(u-US)))/2.) - s)/ts;

          vinf = (u >  THVINF) ? 0.0: 1.0;
	      winf = (u >  THWINF) ? WINFSTAR: 1.0-u/TWINF;

          dv = (u > THV) ? -v/TVP : (vinf-v)/tvm;
          dw = (u > THW) ? -w/TWP : (winf-w)*twm;
          

	      //Update gates

          v    += dv*dt;
          w    += dw*dt;
          s    += ds*dt;

	       //Compute currents
           jfi = (u > THV)  ? -v * (u - THV) * (UU - u)/TFI : 0.0;
           jso = (u > THSO) ? 1./tso : (u-UO)/to;
           
           jsi = (u > THSI) ? -w * s/TSI : 0.0;

           state_vars_gpu[i]       = u - (jfi+jso+jsi-stim)*dt + lap[i];
           state_vars_gpu[i+nxb]   = v;
           state_vars_gpu[i+2*nxb] = w;
           state_vars_gpu[i+3*nxb] = s;
	  }
}


/* ====================================================
 *
 * genInitConditions () 
 *      
 *      generate the initial condition to obtain a 
 *      spiral.
 * 
 * ====================================================*/
void genInitConditions ()
{
   /* Generate new initial condition */

   Real * state_vars_cable, * uo, * vo, * wo, *so;
   Real * state_vars_gpu, * cable_lap_gpu;
  
   int i, j;
   
   printf("\nGenerating initial condition \n\n");
  
   for (j=1; j <=N-1; j++){
  	    for (i=1; i <= N-1; i++){
  	   	     if (j>=1 && j<=10){
  	            U(i,j) = 0.8;
  	   	     }else{
  	   	    	U(i,j) = 0.0;
  	   	     }
  	         V(i,j) = 0.9;
  	         W(i,j) = 0.3;
  	         S(i,j) = 0.0;
         }
    }
    printf("\nCreating a cable \n\n");
  
    state_vars_cable = (Real *) calloc (4 * N, sizeof (Real));
  
    /*Assignment of the pointers*/
    uo       = state_vars_cable;
    vo       = state_vars_cable +      N;
    wo       = state_vars_cable + 2 *  N;
    so       = state_vars_cable + 3 *  N;
  
    /***************************************/
    /*          Inizialization             */
    /***************************************/
       
     printf("\nInitialating cable \n\n");
 	 for (i=0; i < N; i++){
 		 if (i>=1 && i<=10){
 		    uo[i]  = 0.8;
 		 }else{
 			uo[i] = 0.0; 
 		 }
 		  vo[i]  = 0.9;
 		  wo[i]  = 0.3;
 		  so[i]  = 0.0;
 	 }
 	 
 	 
 
 	 /*Memory Allocation on the GPU side*/
 	 cudaMalloc((void**) &state_vars_gpu,         4* N * sizeof(Real));
 	 checkCUDAError("1\n"); 
 	 cudaMalloc((void**) &cable_lap_gpu,             N * sizeof(Real)); 
 	 checkCUDAError("2\n"); 
 	 cudaMemcpy(state_vars_gpu, state_vars_cable, 4* N * sizeof(Real), cudaMemcpyHostToDevice);
 	checkCUDAError("3\n"); 
 	 /*Definition of Block Grid and Block Size*/
     dim3 dimGrid, dimBlock;
 	 dimGrid  = dim3(N/256, 1, 1);
 	 dimBlock = dim3(  256, 1, 1);
 	 
     //for (int ntime=0; ntime <= niterations; ntime++){
    	 /****************************************/
    	 /* Calculation of the diffusion term    */
    	 /* using CUDA                           */
    	 /****************************************/
     //while  (){
 	 int iterations = 0;
 	 int max_iterations = 10000;
     while (!(uo[N/2 +150] < 0.2 && uo[N/2+151] >= 0.2)){
    	    if (iterations > max_iterations) break;
    	    iterations++;
    	    //printf("uo[N/2]=%f\n",uo[N/2]);
    	    calcPDECable    <<<dimGrid, dimBlock>>> (state_vars_gpu, cable_lap_gpu, ddt_o_dx2, N);
 	        calcODECable   <<<dimGrid, dimBlock>>> (state_vars_gpu, N, dt, cable_lap_gpu, 0.0);
            cudaMemcpy(state_vars_cable, state_vars_gpu, 4 * N * sizeof(Real), cudaMemcpyDeviceToHost);
     }
     //}
     
     //}
     
     cudaMemcpy(state_vars_cable, state_vars_gpu, 4 * N * sizeof(Real), cudaMemcpyDeviceToHost);
     
     for (j=1; j <=NY; j++){
          for (i=1; i <= NX; i++){
     		   U(i, j)  = uo[j];//U(5, j);
     		   V(i, j)  = vo[j];//V(5, j);
     		   W(i, j)  = wo[j];//W(5, j);
     		   S(i, j)  = so[j];//S(5, j);
     	  }
     }
     if (SPIRAL_PROTOCOL_ONE){		    			    	
       for (i=1; i <=NY-500; i++){
           for (j=10; j <= NX-60; j++){
     	        U(i, j)  = 1.0;
     	   }
       }
       /*for (i=1; i <=NY; i++){
                  for (j=1; j <= NX; j++){
            	        W(i, j)  = 0.45;
            	   }
              }*/
     }else{
       for (i=1; i <=NX; i++){
     	    for (j=1; j <= NY; j++){
     		     if (i >= NX-250 && i <= NX){
     		    	 U(i, j)  = 0.0;
     		    	 V(i, j)  = 1.0;
     		     }
     		}
        }
     }
  
     printf("\nFinishing cable computation \n\n");
     
     free( state_vars_cable );    
     cudaFree(state_vars_gpu);
     cudaFree(cable_lap_gpu);
  
}




/* ====================================================
 *
 * boundary_conditions (Real* u, int nxb, int ny)
 *      
 *      calc the boundary conditions
 * 
 * ====================================================*/
__global__ void boundary_conditions (Real* u, int nxb, int ny){
	       int i = blockIdx.x * blockDim.x + threadIdx.x;
	       int j = blockIdx.y * blockDim.y + threadIdx.y;
	       //int index = i + j*N;
	       int nx = nxb - 2;
           if (j == 0){
           	   U(i, j) = U(i, j+2);
           }else if (j == NY + 1){
           	   U(i, j) = U(i, j-2);	   
           }else if (i == 0){
           	   U(i, j) = U(i+2, j);
           }else if (i == NX + 1){
           	   U(i, j) = U(i-2, j);
           }
}


__global__ void calcPDE2D_LAPLACIAN5POINTS(Real * u, Real * xlap, Real ddt_o_dx2, int nxb, int ny){
	       int i = blockIdx.x * blockDim.x + threadIdx.x;
	       int j = blockIdx.y * blockDim.y + threadIdx.y;
	       //int index = i + j*N;
           int nx = nxb - 2;
           if (i >= 1 && i <= NX && j >= 1 && j <= NY ){
              L(i,j) =  CDDT_O_DX2 * (U(i+1,j) + U(i-1,j) + U(i,j-1) + U(i,j+1) - 4.0 * U(i,j));
		   }

}


__global__ void calcPDE2D_LAPLACIAN9POINTS(Real * u, Real * xlap, Real ddt_o_dx2, int nxb, int ny){
	       int i = blockIdx.x * blockDim.x + threadIdx.x;
	       int j = blockIdx.y * blockDim.y + threadIdx.y;
	       //int index = i + j*N;
           int nx = nxb - 2;
           if (i >= 1 && i <= NX && j >= 1 && j <= NY ){
              L(i,j) =   CDDT_O_6DX2 * (U(i+1,j-1) + U(i+1,j+1) + U(i-1,j-1) + U(i-1,j+1) + 4.0 * (U(i+1,j) +  U(i-1,j) +  U(i,j-1) +  U(i,j+1)) - 20.0 * U(i,j));
		   }

}






__global__ void calcODE2D ( Real * gpu, Real *old_u, Real * xlap, Real dt, int nxb, int ny, int field_size){


           Real *u, *v, *w, *s, v_value, w_value, s_value;

           Real jfi, jso, jsi;
           int i = blockIdx.x * blockDim.x + threadIdx.x;
           int j = blockIdx.y * blockDim.y + threadIdx.y;
           int index;
           
           index          = i + j*nxb;
           u              = gpu;
           v              = gpu +     field_size;
           w              = gpu + 2 * field_size;
           s              = gpu + 3 * field_size;
                  	             
           v_value        = v[index];
           w_value        = w[index];
           s_value        = s[index];
           
           if (i >= 1 && i <= nxb-2 && j >= 1 && j <= NY ){
        	   
                if (u[index] < 0.006){
                	//w[index] += (1.0 -(u[index]/TWINF) - w_value) * (0.04166666666666667 + 0.025 * tanh(KWM*(u[index]- 0.0406638027778453 )))*dt;
                	
                	w[index] += (1.0 -(u[index]/TWINF) - w_value) / (TW1M+((TW2M-TW1M)*(1.+tanh(KWM*(u[index]-UWM)))) * 0.5)*dt;
                	
                	v[index] += ((1.0-v_value)/TV1M)*dt;
                    s[index] += ((((1.+tanh(KS*(u[index] - US))) * 0.5) - s_value)/TS1)*dt;
                    jfi = 0.0;
                    jso = u[index]/TO1;
                    jsi = 0.0;
                }else if (u[index] < 0.13){
                	//w[index] += ((0.94-w_value) * (0.04166666666666667 + 0.025 * tanh(KWM*(u[index]- 0.0406638027778453 ))))*dt;
                	 w[index] += (0.94-w_value) / (TW1M+((TW2M-TW1M)*(1.+tanh(KWM*(u[index]-UWM)))) * 0.5)*dt;
                	v[index] += (-v_value/TV2M)*dt;
                    s[index] += ((((1.+tanh(KS*(u[index]-US))) * 0.5) - s_value)/TS1)*dt;
                	jfi = 0.0;
                	jso = u[index]/TO2;
                    jsi = 0.0;
                }else if (u[index] < 0.3){
                    w[index] += (-w_value/TWP)*dt;
                	v[index] += (-v_value/TV2M)*dt;
                	s[index] += ((((1.+tanh(KS*(u[index]-US))) * 0.5) - s_value)/TS2)*dt; 
                	jfi = 0.0;
                	//jso = 1./(TSO1+((TSO2-TSO1)*(1.+tanh(KSO*(u[index]-USO)))) * 0.5);
                	jso = 0.518815902099483 + 0.485502667750873 * tanh(KSO * (u[index] -  1.48246402499052 ));
                	jsi = -w_value * s_value/TSI;
                }else{ // U(i,j) >= 0.3
                	w[index] += (-w_value/TWP)*dt;
                	v[index] += (-v_value/TVP)*dt;
                	s[index] += ((((1.+tanh(KS*(u[index]-US))) * 0.5) - s_value)/TS2)*dt; 
                	jfi = -v_value * (u[index] - THV) * (UU - u[index])/TFI;
                	//Semplification of jso
                	jso = 0.518815902099483 + 0.485502667750873 * tanh(KSO * (u[index] -  1.48246402499052 ));
                	//Original jso
                	//jso = 1./(TSO1+((TSO2-TSO1)*(1.+tanh(KSO*(u[index]-USO)))) * 0.5);
                	jsi = -w_value * s_value/TSI;
                }
                
                old_u[index] = u[index];
                u[index]     = u[index]  - (jfi+jso+jsi)*dt + xlap[index];
        }

}


__host__ char * createAPFileName (int x, int y){
      char *res1 = (char *)malloc(sizeof(char) * 60);
      sprintf(res1, "apCell%d_%d",x,y);
      for (int i=0; i < 60; i++){
           if (res1[i] == '.'){
          	   res1[i] = 'd';
           }
      }
      sprintf(res1, "%s.dat", res1);
      
      return res1;
}



/* ====================================================
 *
 * writeAP () 
 *      
 *      write recorded AP
 * 
 * ====================================================*/
void writeAP () {
	FILE *fp;
	fp=fopen(createAPFileName (N/10, N/10),"wb");
	if (fp == NULL){
		printf("Invalid file name\n");
		exit(1);
	}else{
		fwrite (ap1, sizeof(Real), MAX_SAMPLING, fp);
		fclose(fp);
    }
	
	fp=fopen(createAPFileName (N - (N/10), N -(N/10)),"wb");
    if (fp == NULL){
	    printf("Invalid file name\n");
		exit(1);
	}else{
		fwrite (ap2, sizeof(Real), MAX_SAMPLING, fp);
		fclose(fp);
	}
}



__host__ char * createTipdataFileName (){
      char *res1 = (char *)malloc(sizeof(char) * 60);
      sprintf(res1, "tipdata");
      for (int i=0; i < 60; i++){
           if (res1[i] == '.'){
          	   res1[i] = 'd';
           }
      }
      sprintf(res1, "%s.dat", res1);
      
      return res1;
}

__host__ char * createTipdata_binaryFileName (){
      char *res1 = (char *)malloc(sizeof(char) * 60);
      sprintf(res1, "tipdata_binary");
      for (int i=0; i < 60; i++){
           if (res1[i] == '.'){
          	   res1[i] = 'd';
           }
      }
      sprintf(res1, "%s.dat", res1);
      
      return res1;
}






void writeTip (void){
  /* This routine for outputting the tip data is called from Find_tips() */
	FILE *fp;
	fp=fopen(createTipdataFileName (),"w");
	if (fp == NULL){
		printf("Invalid file name\n");
		exit(1);
	}else{
		 for (int i=0; i < *ntip; i++){
		      fprintf(fp, "%.5f\t%.5f\t%.5f\n", tip_times[i], nxtip[i], nytip[i]);
	     }
		 fclose(fp);
    }
	
	
	fp=fopen(createTipdata_binaryFileName (),"w");
	if (fp == NULL){
		printf("Invalid file name\n");
		exit(1);
	}else{
		fwrite (tip_times, sizeof(Real), MAX_TIPS, fp);
		fwrite (nxtip,     sizeof(Real), MAX_TIPS, fp);
		fwrite (nytip,     sizeof(Real), MAX_TIPS, fp);
		fclose(fp);
	}
  //fprintf(tip_file, "%.5f %.5f %.5f\n", dt*(Real)istep, x, y);
}







__host__ void tip_track ( Real * u, Real * v, Real *old_u, int nxb, int ny, int field_size, Real time_tip){

	      Real  uct = 0.8, x1, y1, x2, y2, x3, y3, x4, y4, den;
	      Real  a1, b1, c1;
	      Real  a0, b0, c0, ac, bc, aq, bq, cq, disc,q, t1, t2, xtip, ytip;
	      Real  xtx, ytx;
	
	      int nx = nxb - 2,i,j;
	      for (i=3; i < NX -1; i++){
	    	  for (j=3; j < NY -1; j++){ 
	      
        	   x1=U (i,j)-uct;
        	   y1=OU(i,j)-uct;
        	   
        	   x2=U (i+1,j)-uct;
        	   y2=OU(i+1,j)-uct;
        	         
        	   x3=U (i+1,j+1)-uct;
        	   y3=OU(i+1,j+1)-uct;
        	   
        	   x4=U (i,j+1)-uct;
        	   y4=OU(i,j+1)-uct;
        	   den=y1-y2+y3-y4;
        	   
        	   if (den != 0.0){
        	       a0=y1/den;
        	       b0=(-y1+y4)/den;
        	       c0=(-y1+y2)/den;
        	       den=x1-x2+x3-x4;
        	       if (den != 0.0){
        	       
        	          a1=x1/den;
        	          b1=(-x1+x4)/den;
        	          c1=(-x1+x2)/den;
        	          den=b0-b1;
        	   
        	          if (den != 0.0){
        	             ac=(a1-a0)/den;
        	             bc=(c1-c0)/den;
        	             cq=a0+b0*ac;
        	             bq=b0*bc+c0+ac;
        	             aq=bc;
        	             disc=(bq*bq)-4.0*aq*cq;
        	             if (disc >= 0.0){
        	            	 if (bq != 0.0){
        	                     q=-0.5*(bq+bq/abs(bq)*sqrt(disc));
        	                     if ((aq != 0.0) && (q != 0.0)) {
        	                     
        	                	      t1=q/aq;
        	                          t2=cq/q;
        	                          
        	                          if (abs(t1) < abs(t2)){
        	                        	  xtip = t1;
        	                          }else{
        	                        	  xtip = t2;
        	                          }
        	                          ytip=ac+bc*xtip; 
        	                          if ((xtip <1.0 && xtip > 0.0) && (ytip < 1.0 && ytip > 0.0)){
        	                        	 if (V(i,j) > .001){
        	                        	      xtx=xtip+Real(i);
        	                        	      ytx=ytip+Real(j);
        	                        	      
        	                        		  if (xtx > 0.0 && ytx > 0.0) {
        	                        			  //atomicAdd(gpu_ntip, 1);
        	                        			  nxtip    [*ntip] = xtx;
        	                        			  nytip    [*ntip] = ytx;
        	                        			  tip_times[*ntip] = time_tip;
        	                        			  *ntip            = *ntip + 1;
        	                        			  
        	                        			  /*printf("%f\n", xtx);
        	                        			  printf("%f\n", ytx);
        	                        			  printf("%f\n", time_tip);
        	                        			  printf("%d\n", *ntip);*/
        	                        		  }
        	                        	 }  
        	                          
        	                          }
        	                     }
        	                 }
        	             }
        	          }
        	       }
        	   }    
           }
	      }
     
}




/*******************************************************/
/*                    MAIN BLOCK                       */            
/*******************************************************/
/*                                                     */
/*    1) Check CUDA version  checkCUDAversion();       */
/*    2) Read parameters of simulation from            */
/*       initizil. file:                               */ 
/*          - sim_time                                 */             
/*          - time of simulation in sec                */
/*          - tip_sampling_rate in ms                  */
/*          - plotting_rate in ms                      */
/*          - dt  - integration step  in ms            */
/*          - dx  - distance among two cells in cm     */
/*          - diff - diffusion constant                */
/*          - nx  - num. of elements on x              */
/*          - ny  - num. of elements on y              */
/*          - laplacian_stencil - 0 is the 5 points,   */ 
/*                                1 is the 9 points    */
/*          - write_AP          - 0 false, 1 true      */
/*          - write_tip_data    - 0 false, 1 true      */
/*    3) Write parameters in the GPU constant memory   */
/*    4) Allocate memory in the GPU and CPU            */
/*    5) Initialize the Graphics                       */
/*    6) Initialize the tissue with the protocol for   */
/*       the spiral generation                         */
/*    7) Start simulation for the number of iterations=*/
/*       (simulation_time/iteration step)              */
/*       For each iteration step:                      */
/*         - Calculate the boundary conditions         */
/*         - Calculate the PDE using u as diffusive var*/
/*         - Calculate the ODE using the Minimal Model */
/*         - Calculate the tip spiral and record it in */
/*           the tip trajectory                        */
/*    8) Calculate time of the simulation              */
/*    9) Finish the task                               */
/*******************************************************/       
int main(int argc, char** argv)
{
	    int start_time, nend, ntime; 
	
        //1) Check CUDA version 
	    checkCUDAVersion();
    
        //2) Read parameters of simulation from            
        //   initizil. file:                                
        //        - sim_time                                              
        //        - time of simulation in sec                
        //        - tip_sampling_rate in ms                  
        //        - plotting_rate in ms                      
        //        - dt  - integration step  in ms            
        //        - dx  - distance among two cells in cm     
        //        - diff - diffusion constant                
        //        - laplacian_stencil - 0 is the 5 points,    
        //                                1 is the 9 points    
        //        - write_AP          - 0 false, 1 true      
        //        - write_tip_data    - 0 false, 1 true      
	    //exit(0);
	    readParams ();
	    
	    //3) Write parameters in the GPU constant memory
	    writeConstantMemory ();
	    
	    //4) Allocate memory in the GPU and CPU          
	    cpuMemoryAllocation ();
	    gpuMemoryAllocation ();
	    
	    //5) Initialize the Graphics                       
	    plotIni(0);
	    
	    //6) Initialize the tissue with the protocol for  
	    //   the spiral generation           
	    printf("%d\n", nxb);
		   for (int j=1; j <=N-1; j++){
		  	    for (int i=1; i <= N-1; i++){
		  	   	     if (j>=1 && j<=10){
		  	            U(i,j) = 0.8;
		  	   	     }else{
		  	   	    	U(i,j) = 0.0;
		  	   	     }
		  	         V(i,j) = 0.9;
		  	         W(i,j) = 0.3;
		  	         S(i,j) = 0.0;
		         }
		    }
	    genInitConditions ();
		start_time =0;
		nend = (int)(sim_time/dt);
		printf("total number of iterations=%d\n", nend);

		//Copy from Host Memory to Device Memory
	    cudaMemcpy(gpu_state_vars, cpu_state_vars,  ( field_size * 4) * sizeof(Real), cudaMemcpyHostToDevice);
		//Copy from Device Memory to Host Memory
	    cudaMemcpy(cpu_old_u, gpu_state_vars,  field_size * sizeof(Real), cudaMemcpyDeviceToHost);
	    
	    int icoun = 0;
	    
	    //Set the block of threads and the grid 
	    //
	    dim3 dimBlock( blocksize, blocksize );
	    dim3 dimGrid( nxb/dimBlock.x, nxb/dimBlock.y);
	    
	    
	    
		clock_t start=clock();
		clock_t begin=clock();
		
		for (ntime=start_time; ntime <= nend; ntime++){
			    boundary_conditions<<<dimGrid, dimBlock>>> (gpu_state_vars, nxb, ny);
				
				if (!laplacian_stencil)
				{ 
				   calcPDE2D_LAPLACIAN5POINTS<<<dimGrid, dimBlock>>>           (gpu_state_vars, gpu_xlap, ddt_o_dx2, nxb, ny);
				}
				else 
				{
				   calcPDE2D_LAPLACIAN9POINTS<<<dimGrid, dimBlock>>>           (gpu_state_vars, gpu_xlap, ddt_o_6dx2, nxb, ny);
				}
				calcODE2D<<<dimGrid, dimBlock>>>           (gpu_state_vars, gpu_old_u, gpu_xlap, dt, nxb, ny, field_size);
				
			
			    if( eventHandler() )          break;
	            
			    if ((ntime % tip_sampling_rate_steps) == 0){
	            	 cudaMemcpy(cpu_state_vars, gpu_state_vars, 4 * field_size * sizeof(Real), cudaMemcpyDeviceToHost);
	            	 cudaMemcpy(cpu_old_u, gpu_old_u,  field_size * sizeof(Real), cudaMemcpyDeviceToHost);
	            	 tip_track ( u, v, cpu_old_u, nxb, ny, field_size, ntime*dt);
	                 ap1[icoun] = U((N/10),(N/10));
	                 ap2[icoun] = U(N - (N/10),N - (N/10));
	                 plot();
			    }     	 
			    if ((ntime % plotting_rate_steps) == 0){
                               cudaMemcpy(cpu_state_vars, gpu_state_vars, 4 * field_size * sizeof(Real), cudaMemcpyDeviceToHost);
	                 plot();
	            }
	            
	    }
		cudaThreadSynchronize();
		clock_t end = clock();
		
		cudaMemcpy(cpu_state_vars,gpu_state_vars,  field_size * 4 * sizeof(Real), cudaMemcpyDeviceToHost);

		Real gpuTime = ((Real )(end-begin))  / CLOCKS_PER_SEC;
		printf("Total Time = %.3f s\nAverage time for iteration = %.5f s\n", gpuTime, gpuTime / ntime);
		    
		
		if (write_tip_data)
		{
		    writeTip ();
		}
		if (write_AP){
		   writeAP  ();
		}
		    
		cudaFree(gpu_state_vars);   
		    
		delete [] cpu_state_vars;
		quitX();
		return EXIT_SUCCESS;
	    
	


}





