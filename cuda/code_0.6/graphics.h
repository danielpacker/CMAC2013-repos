/*******************************************************/
/*     Title:  graphics.h (cuda2DMinModel)             */
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

#ifndef _PLOT_
#define _PLOT_

#define WINDOW_TITLE "cuda2DMinModel"

#define WINX         0      /* Window location, in pixels, from screen left */
#define WINY         0      /* Window location, in pixels, from screen top. */
#define WM_CTRLS_POS 1      /* If WM_CTRL_POS is 0 then the window is placed
			                 * at (WINX, WINY). If WM_CTRL_POS is 1 then WINX
			                 * and WINY are ignored and the location is
			                 * determined by the window manager. */
#define WINSIZE      512    /* Window is square of this size in pixels. */
#define PLOT_SIZE    1.2    /* This controls the size of the simulation
			     * volume in the view port: >1.0 for larger
			     * size, <1.0 for smaller size. */

#define BACKGROUND   1.0    /* Background color (R=G=B=BACKGROUND, so 0.0 gives
			       BLACK, 1.0 gives WHITE) */

#define START_PAUSED 0      /* If 1 then window is opened in paused mode
			                 * showing initial condition. */

#define CLASSIC_COLORS 1

/* The tip is plotted as a line. */

#define TIP_PLOT_TYPE GL_LINE_STRIP
#define TIP_WT  2.0
#define TIP_R   0.0
#define TIP_G   0.0
#define TIP_B   0.0



#define PX(x) ((rect_h*((x)-1.))-half_width)
#define PY(y) ((rect_h*((y)-1.))-half_height)


#define TRUE             1   
#define FALSE            0

#define U_FIELD          0     /* These are used to determine which field */
#define V_FIELD          1     /* (if any) is being plotted */
#define W_FIELD          2
#define S_FIELD          3

#define NO_FIELD        -1

#define MODE_SIMULATING  1   
#define MODE_VIEWING     2   

#endif
