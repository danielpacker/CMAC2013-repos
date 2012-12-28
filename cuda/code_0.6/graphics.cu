
/*******************************************************/
/*     Title: graphics.c (cuda2DMinModel)              */
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
#include <math.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/keysymdef.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <cutil_inline.h>
#include "cusim.h"
#include "graphics.h"



static Display *disp;
static Window win;
static float *vertex, *colormap;  
static Real half_width, half_height;
static Real plot_length[2];	

static int field;		
static int show_tip;		
static int state;		



  


/* ========================================================================= */
/*                                                                           */
/*             PROTOTYPES                                                    */
/*                                                                           */
/* ========================================================================= */

static void colorMap (Real h, float *red, float *blue, float *green);
static Real chooseValue (int i, int j);
static void drawTips (void);
static void restart (void);
static void pause (void);
static void enableTipPlotting (void);
static void reshape (int w, int h);
static void initPlotWindow (int winx, int winy, int width, int height);
static Bool wait(Display * d, XEvent * e, char *arg);

/* ========================================================================= */
/*                                                                           */
/*             PLOT METHODS                                                  */
/*                                                                           */
/* ========================================================================= */

void plot (void)
{

  int     j, i, index1, index2;
  float   red, green, blue;
  Real    value;

  if (PLOTTING) {

    glClear (GL_COLOR_BUFFER_BIT);
    glColor3f(0.,0.,1.);
    glRectf(-half_width, -half_height, half_width, half_height); 

    if (field != NO_FIELD) {

     
      index1 = 0;
      index2 = 0;
      for (i = 1; i <= NX; i++) {
           value = chooseValue(i,1);
           colorMap (3.5 * value + 2., &red, &blue, &green);
           colormap[(i-1)*6] = red;
           colormap[(i-1)*6 + 1] = green;
           colormap[(i-1)*6 + 2] = blue;
      }

     
      for (j = 1; j < NY; j++) 
      {    
          for (i = 1; i <= NX; i++) 
          {         

              colorMap (3.5 * chooseValue(i,j) + 2., &red, &blue, &green);
              
              index1 = (j-1)*6*NX + (i-1)*6;
              index2 = j*6*NX + (i-1)*6;

              colormap[index1 + 3] = red;
              colormap[index1 + 4] = green;
              colormap[index1 + 5] = blue;

              colormap[index2]     = red;
              colormap[index2 + 1] = green;
              colormap[index2 + 2] = blue;

          }
      }
      glEnableClientState (GL_COLOR_ARRAY);
      glColorPointer (3, GL_FLOAT, 0, colormap);
      glEnableClientState (GL_VERTEX_ARRAY);
      glVertexPointer (2, GL_FLOAT, 0, vertex);

      for (j = 1; j < NY; j++) {
	glDrawArrays (GL_TRIANGLE_STRIP, (j - 1) * 2 * NX, 2 * NX);
      }
    }
  }

  if (show_tip)
    drawTips ();

  if (PLOTTING)
    glXSwapBuffers (disp, win);
}

static Real chooseValue (int i, int j)
{
       Real    value;
              switch (field) 
              {
                       case U_FIELD:	     /* Set the u color*/	
                            value = U(i,j);
                       break;
                       
                       case V_FIELD:         /* Set the v color*/
        	            value = V(i,j);
                       break;

                       case W_FIELD:         /* Set the w color*/
        	            value = W(i,j);
                       break;

                       case S_FIELD:        /* Set the s color */
                            value = S(i,j);
        	       break;
              } 
        return value;
}


static void colorMap (Real h, float *red, float *blue, float *green)
{

   /* 6.0 is red 
   * 5.0 is yellow 
   * 4.0 is green 
   * 2.0 is blue 
   */
  float   m, n, f;
  int     i;

  i = floor (h);
  f = h - i;
  if (!(i & 1))
    f = 1 - f;			
  m = 0.2;
  n = 1 - 0.8 * f;
  switch (i) {
  case 6:
  case 0:
    *red = 1.0;
    *blue = n;
    *green = m;
    break;
  case 1:
    *red = n;
    *blue = 1.0;
    *green = m;
    break;
  case 2:
    *red = m;
    *blue = 1.0;
    *green = n;
    break;
  case 3:
    *red = m;
    *blue = n;
    *green = 1.0;
    break;
  case 4:
    *red = n;
    *blue = m;
    *green = 1.0;
    break;
  case 5:
    *red = 1.0;
    *blue = m;
    *green = n;
    break;
  }
}


static void drawTips (void)
{
  Real    rect_h = plot_length[0] / (NX - 1);
  int     i;

  glLineWidth (TIP_WT);
  glBegin (TIP_PLOT_TYPE);
  glColor3f (TIP_R, TIP_G, TIP_B);
  for (i = 0; i < *ntip; i++) {
    glVertex2f (PX (nxtip[i]), PY (nytip[i]));
  }
  glEnd ();
}
/* ========================================================================= */
/*                                                                           */
/*             RESTART AND PAUSE METHODS                                     */
/*                                                                           */
/* ========================================================================= */

static void restart (void)
{

  if (state == MODE_VIEWING) {
    state = MODE_SIMULATING;
  }
}

static void pause (void)
{
  if (state == MODE_SIMULATING) {
    state = MODE_VIEWING;
  }
}

/* ========================================================================= */
/*                                                                           */
/*             SHOW METHODS                                                  */
/*                                                                           */
/* ========================================================================= */
static void showUArray(void)
{
  field = U_FIELD;
  if (state != MODE_SIMULATING) {
    plot ();
  }
}

static void showVArray(void)
{
  field = V_FIELD;
  if (state != MODE_SIMULATING) {
    plot ();
  }
}

static void showSArray(void)
{
  field = S_FIELD;
  if (state != MODE_SIMULATING) {
    plot ();
  }
}

static void showWArray(void)
{
  field = W_FIELD;
  if (state != MODE_SIMULATING) {
    plot ();
  }
}

static void showNOArray(void)
{
  field = NO_FIELD;
  if (state != MODE_SIMULATING) {
    plot ();
  }
}


static void enableTipPlotting (void)
{
  if (show_tip)
    show_tip = FALSE;
  else {
    show_tip = TRUE;
    ntips = 0;			
  }

  if (state != MODE_SIMULATING) {
    plot ();
  }
}


static void reshape (int w, int h)
{

  glMatrixMode (GL_PROJECTION);
  glLoadIdentity ();
  glOrtho (-PLOT_SIZE * half_width, PLOT_SIZE * half_width,
	   -PLOT_SIZE * half_height, PLOT_SIZE * half_height, -20., 20.);
  glMatrixMode (GL_MODELVIEW);
  glViewport (0, 0, w, h);
}


void plotIni (int initial_field)
{
  int     i, j, vindex;
  Real    hx, hy, x1, y1;

  field = initial_field;

  show_tip = FALSE;
  {
    int     nmax = max (NX, NY);
    plot_length[0] = (NX - 1.) / (nmax - 1.);
    plot_length[1] = (NY - 1.) / (nmax - 1.);
  }
  half_width = 0.5 * plot_length[0];
  half_height = 0.5 * plot_length[1];
  hx = plot_length[0] / (NX - 1);
  hy = plot_length[1] / (NY - 1);

  if (!PLOTTING) {
    field = NO_FIELD;
    state = MODE_SIMULATING;
    return;
  }

  vertex = (float *) calloc (4 * NX * NY, sizeof (float));
  colormap  = (float *) calloc (6 * NX * NY, sizeof (float));
  y1 = -half_height;
  vindex = 0;
  for (j = 1; j < NY; j++) {
    x1 = -half_width;
    for (i = 1; i <= NX; i++) {
      vertex[vindex++] = x1;
      vertex[vindex++] = y1;
      vertex[vindex++] = x1;
      vertex[vindex++] = y1 + hy;
      x1 += hx;
    }
    y1 += hy;
  }

  initPlotWindow (WINX, WINY, WINSIZE * plot_length[0], WINSIZE * plot_length[1]);

  glShadeModel (GL_FLAT);    

  glClearColor (BACKGROUND, BACKGROUND, BACKGROUND, 0.0);

  if (START_PAUSED) {
    state = MODE_VIEWING;
  }
  else {
    state = MODE_SIMULATING;
  }

}






int eventHandler (void)
{
  static XEvent theEvent;
  static KeySym theKeySym;
  static int theKeyBufferMaxLen = 64;
  static char theKeyBuffer[65];
  static XComposeStatus theComposeStatus;
  int     write_tip_save;

  if (!PLOTTING)
    return (0);


  write_tip_save = write_tip;
  write_tip = FALSE;


  while (XPending (disp) || (state != MODE_SIMULATING)) {

    XNextEvent (disp, &theEvent);

    switch (theEvent.type) {	

    case KeyPress:		

      XLookupString ((XKeyEvent *) & theEvent, theKeyBuffer,
		     theKeyBufferMaxLen, &theKeySym, &theComposeStatus);

      switch (theKeySym) {	

      case XK_Escape:
	exit (0);		

      case XK_Q:
      case XK_q:
    	if (write_tip_data){
    	  writeTip ();
    	}
  		if (write_AP){
  		   writeAP  ();
  		}
	      return (1);

      case XK_P:
      case XK_p:
	pause ();
	plot ();
	break;

      case XK_R:
      case XK_r:
	restart ();
	break;

      case XK_T:
      case XK_t:
	enableTipPlotting();
	break;

      case XK_U:
      case XK_u:
	showUArray();
	break;

      case XK_V:
      case XK_v:
	showVArray();
	break;

      case XK_W:
      case XK_w:
	showWArray();
	break;

      case XK_S:
      case XK_s:
	showSArray();
	break;

      case XK_N:
      case XK_n:
	showNOArray();
	break;


      }				
      break;

    case EnterNotify:
      XSetInputFocus (disp, win, RevertToPointerRoot,
		      CurrentTime);
      break;

    case Expose:
      plot ();
      break;

    case ConfigureNotify:
      reshape (theEvent.xconfigure.width, theEvent.xconfigure.height);
      break;

    }				

  }				

  write_tip = write_tip_save;

  return (0);
}


static void initPlotWindow (int winx, int winy, int width, int height)
{

  static XVisualInfo *theVisualInfo;
  static GLXContext theGLXContext;
  static Colormap theColormap;
  static int theScreen;
  static char *dispName = NULL;
  static XEvent event;
  static Atom del_atom;
  static XSizeHints theSizeHints;
  static XSetWindowAttributes theSWA;
  static char *name = WINDOW_TITLE;
  static XTextProperty winName, theIconName;
  static int num1, num2;
  static int list[] = { GLX_RGBA,
    GLX_DOUBLEBUFFER,
    GLX_RED_SIZE, 1,
    GLX_GREEN_SIZE, 1,
    GLX_BLUE_SIZE, 1,
    GLX_DEPTH_SIZE, 1,
    None
  };

  if ((disp = XOpenDisplay (NULL)) == NULL) {
    fprintf (stderr,
	     "ERROR: Could not open a connection to X on display %s\n",
	     XDisplayName (dispName));
    exit (1);
  }
  if (!glXQueryExtension (disp, &num1, &num2)) {
    fprintf (stderr,
	     "ERROR: No glx extension on display %s\n",
	     XDisplayName (dispName));
    exit (1);
  }

  theScreen = DefaultScreen (disp);

  if (!(theVisualInfo = glXChooseVisual (disp, theScreen, list))) {
    fprintf (stderr, "ERROR: Couldn't find visual");
    exit (-1);
  }
  if (!(theGLXContext = glXCreateContext (disp, theVisualInfo,
					  None, GL_TRUE))) {
    fprintf (stderr, "ERROR: Can not create a context!\n");
    exit (-1);
  }

  theColormap = XCreateColormap (disp,
				 RootWindow (disp,
					     theVisualInfo->screen),
				 theVisualInfo->visual, AllocNone);

  if (!(theColormap)) {
    fprintf (stderr, "ERROR: couldn't create Colormap\n");
    exit (-1);
  }
  theSWA.colormap = theColormap;
  theSWA.border_pixel = 0;
  theSWA.event_mask = (EnterWindowMask | KeyPressMask | StructureNotifyMask |
		       ButtonPressMask | ButtonReleaseMask | ExposureMask |
		       PointerMotionMask);

  win = XCreateWindow (disp,
			     RootWindow (disp, theVisualInfo->screen),
			     winx, winy, width, height, 0,
			     theVisualInfo->depth, InputOutput,
			     theVisualInfo->visual,
			     CWBorderPixel | CWColormap | CWEventMask,
			     &theSWA);

  if (!(win)) {
    fprintf (stderr, "ERROR: couldn't create X window\n");
    exit (-1);
  }


  XStringListToTextProperty (&name, 1, &winName);
  XStringListToTextProperty (&name, 1, &theIconName);

  theSizeHints.base_width = width;
  theSizeHints.base_height = height;
  theSizeHints.min_aspect.x = width;	
  theSizeHints.max_aspect.x = width;
  theSizeHints.min_aspect.y = height;
  theSizeHints.max_aspect.y = height;

  theSizeHints.flags = PSize | PAspect;

  if (!(WM_CTRLS_POS))
    theSizeHints.flags |= USPosition;

  XSetWMProperties (disp, win, &winName, &theIconName,
		    NULL, 0, &theSizeHints, NULL, NULL);

  if ((del_atom = XInternAtom (disp, "WM_DELETE_WINDOW", TRUE)) != None) {
    XSetWMProtocols (disp, win, &del_atom, 1);
  }

  XMapWindow (disp, win);
  XIfEvent (disp, &event, wait, (char *) win);

  glXMakeCurrent (disp, win, theGLXContext);


}


static Bool wait(Display * d, XEvent * e, char *arg)
{
  return (e->type == MapNotify) && (e->xmap.window == (Window) arg);
}


void quitX (void)
{
  if (PLOTTING)
    XCloseDisplay (disp);
  return;
}

