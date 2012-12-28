/*******************************************************/
/*     Title: cusim.h (cuda2DMinModel)                */
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

#ifndef _CUSIM_
#define _CUSIM_

#ifndef NX
  #define NX  nx
#endif

#ifndef NY
  #define NY  ny
#endif

#define J_INC       (NX+2)
#define I_INC        1
#define FIELD_SIZE ((NY+2)*(NX+2)) 

#define INDEX(i,j)  ((i)*I_INC + (j)*J_INC)

#define U(i,j)          u[INDEX(i,j)]
#define V(i,j)          v[INDEX(i,j)]
#define W(i,j)          w[INDEX(i,j)]
#define S(i,j)          s[INDEX(i,j)]


/* ------------------------------------------------------------------------- */
/*                                                                           */
typedef float  Real;      /* precision of Real variables (float or double)   */

#define PLOTTING   1      /* if is set to 1 we have an interactive graphics  */
/* ------------------------------------------------------------------------- */



/* -------------------------------------------------------------------------- 
 * Global variables used throughout the Fenton-Bartocci Code 
 * ------------------------------------------------------------------------- */

extern Real *gpu_fields, *fields, *u, *v, *w, *s;
extern Real *tip_x, *tip_y;              /* spiral tip arrays */
extern Real *nxtip, *nytip, *gpu_state_vars, *cpu_state_vars;
extern int  nx, ny,                     
            field_size,                 
            write_AP,
            write_tip_data,
            write_tip,                   /* write tip flag */
            simulating_resolution,       /* graphics parameter */
            ntips,                       /* number of spiral tips */
            *ntip;




/* main.cu
 * ---------- */
void writeTip (void);
void writeAP  (void);


/* graphics.c
 * ----------- */
void  plot         (void);
void  plotIni      (int initial_field);
int   eventHandler (void);
void  quitX        (void);

#endif 


