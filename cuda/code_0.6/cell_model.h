/*******************************************************/
/*     Title: cell_model.h (cuda2DMinModel)            */
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
/*      The following are the standard parameters of   */
/*      the epicardial model in the paper:             */
/*         A. Bueno-Orovio, M. Cherry, and F. Fenton,  */
/*         "Minimal model for human ventricular action */ 
/*          potentials in tissue",                     */ 
/*          Journal of Theor. Biology, no. 253,        */
/*          pp. 544-560, 2008.                         */
/*                                                     */
/*       Different parameter ranges to explore are     */
/*       assigned to each group of students, and they  */
/*       are reported commented in following code.     */                               
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


#define TVP         1.4506 
#define TV1M       60.     
#define TV2M     1150.     


/*******************************************************/
/*                Parameters Group 1                   */
/*******************************************************/
/*                ORIGINAL PARAMETERS                  */
/*  #define TWP      200.0                             */
/*  #define TW1M     60.0                              */
/*  #define TW2M     15.                               */
/*******************************************************/
/* Check TWP = [50, 100, 150, 200, 250, 300, 350, 400] */
/* Check TW1M = TW2M = [10,25,50,75,100,150,300,400]   */
/*******************************************************/
#define TWP   200.0
#define TW1M   60.0
#define TW2M   15.0

/*******************************************************/
/*                Parameters Group 2                   */
/*******************************************************/
/*                ORIGINAL PARAMETERS                  */
/*  #define TFI      0.11                              */
/*  #define TSI      1.8875                            */
/*******************************************************/
/* Check TFI = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25   ]*/
/* Check TSI = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]  */
/*******************************************************/

#define TFI     0.11
#define TSI     1.8875

/*******************************************************/
/*                Parameters Group 3                   */
/*******************************************************/
/*                ORIGINAL PARAMETERS                  */
/*  #define TSI      1.8875                            */
/*  #define TSO1     30.0181                           */
/*******************************************************/
/* Check TSI  = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0] */
/* Check TSO1 = [5,10,15,20,40]                        */
/*******************************************************/


/*******************************************************/
/*                Parameters Group 4                   */
/*******************************************************/
/*                ORIGINAL PARAMETERS                  */
/*  #define TFI      0.11                              */
/*  #define TSO1     30.0181                           */
/*******************************************************/
/* Check TFI = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]*/
/* Check TSO1 = [5,10,15,20,50]                        */
/*******************************************************/


#define TSO1      30.0181
#define TSO2       0.9957  
#define TO1      400.      
#define TO2        6.      

/*******************************************************/
/*                Parameters Group 5                   */
/*******************************************************/
/*                ORIGINAL PARAMETERS                  */
/*  #define TS2      16.                               */
/*  #define TW1M      0.13                             */
/*******************************************************/
/* Check TS2  = [15,50,75,85, 100, 150]                */
/* Check TW1M = [ 5,10,30,50, 150, 250]                */
/*******************************************************/

#define TW1M    60.0
#define TW2M    15.0

#define TS1      2.7342  
#define TS2     16.      
#define THW      0.13   

#define TWINF    0.07    
#define THV      0.3     
#define THVM     0.006   
#define THVINF   0.006   
#define THW      0.13    
#define THWINF   0.006   
#define THSO     0.13    
#define THSI     0.13    
#define THO      0.006    
#define KWM     65.    
#define KS       2.0994  
#define KSO      2.0458  
#define UWM      0.03    
#define US       0.9087  
#define UO       0.     
#define UU       1.55   
#define USO      0.65   
#define SC       0.007


#define WINFSTAR 0.94   


#define TW2M_TW1M_DIVBY2 ((TW2M -TW1M)*0.5)
#define TSO2_TSO1_DIVBY2 ((TSO2 -TSO1)*0.5)
