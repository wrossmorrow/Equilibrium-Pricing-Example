/*
 *
 *  Created by W. Ross Morrow on 10/15/11.
 *  Copyright 2011 Iowa State University. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <vecLib/cblas.h>
#include <vecLib/clapack.h>

#include <unuran.h>
#include <unuran_urng_rngstreams.h>

#include "f2c.h"
#include "snfilewrapper.h"
#include "snopt.h"
#include "snoextras.h"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#define MAXPRICE     10.0
#define MINPRICE      0.0
#define pG			  3.0

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * PATH VARIABLES  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static int checkders = 0;
static int integralscale = 0;

static int T;
static int t = 0;

static clock_t tmp_ticks;
static clock_t func_eval_ticks;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// mean utility coefficients: price, mi/$, l*w, hp/wt, og
static double uc_a[] = { 43.501 , - 0.122 , 3.460 , 2.883 , - 8.582 };

// static double uc_B[] = { - 0.1756 }; not needed

// variance utility coefficients (>= 0): price, mi/$, l*w, hp/wt, og
static double uc_c[] = { 0.0000 , 1.050 , 2.056 , 4.628 , 1.794 };

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static int		F = 21;		// number of firms
static int		J = 472;	// total number of products
static int *	Jf;		// F vector of number of products offered by each firm
static int *	sJf;	// F+1 vector of firm "starts": cumulative number of products in each firm
static int		Jfstar; // max_f Jf[f]

static int		K = 3;	// 3 product characteristics (mi/$, l*w, hp/wt)
static int		Kp = 5;	// 3 product characteristics plus price and outside good
static double * Y;		// K x J matrix of product characteristic vectors

static double * p;		// J vector of prices, in 10,000 $
static double * p0;		// T*J vector of initial prices, in 10,000 $
static double * c;		// J vector of costs (for all products)
static double * cd;		// J vector of cost data (for all products)
static double * m;		// J vector of markups (for each product)

static double * pr;		// profits (for each firm)

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static int		I;	// number of individuals
static double	Iinv;	// 1 / I

static double	maxinc;	// maximum income
static int		maxinci;// individual with maximum income

static double * Z;		// I vector: 1 / income values
static double * V;		// I x 5 matrix: random coefficients for each characteristic

static double * UC;		// I x 5 matrix of variable utility coefficients
static double * W;		// I x J fixed portion of utilities for both products
static double * U;		// I x J matrix of utilities for each product
static double * E;		// I x J matrix, exp{U}

static double * PL;		// I x J matrix of logit choice probabilities: PL(i,1) = e^(u_1) / ( 1 + e^(u_1) + e^(u_2)), etc
static double * P;		// choice probabilities for each product

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * FIRST DERIVATIVES * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static double * DpU;	// I x J matrix of price derivatives of utility
static double * DpUPL;	// I x J matrix: DpU .* PL

static double * LAMp;	// J-vector: (1/I) [ sum( DpUPL(:,1) ) , sum( DpUPL(:,2) ) ]
static double * GAMp;	// J x J matrix: (1/I) * PL' * DpUPL

static double *  cg;	// combined gradient (w.r.t. prices)
static double *   z;	// extended "zeta" map
static double * phi;	// phi = p - c - zeta

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * SECOND DERIVATIVES  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

static double * DppU; // I x 2 matrix (column major): second price derivatives of utility

static double * PIL; // I x F matrix of "Logit profits" for each firm and individual...
static double * CHIp; // takes some explaining...
static double * ZHIp; // takes some explaining...
static double * PHIpp; // takes some explaining...
static double * PSIpp; // takes some explaining...

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Function declarations
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void	 utilities();
void probabilities();
void      allcosts();
void         costs();
void       profits();
void   constraints();
void          zeta();

void update();

void printstuff();
void initprob();
void killprob();
void readdata();

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Print routine
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void printstuff()
{
	int i;
	
	printf("p   = [ %0.4f , %0.4f ];\n",  p[0],  p[1]);
	
	printf("UC = [ ");
	for( i = 0 ; i < I ; i++ ) { 
		if( i > 0 ) { 
			printf("       %0.16f , %0.16f , %0.16f , %0.16f , %0.16f ",UC[5*i],UC[5*i+1],UC[5*i+2],UC[5*i+3],UC[5*i+4]); 
		} else {
			printf("%0.16f , %0.16f , %0.16f , %0.16f , %0.16f ",UC[5*i],UC[5*i+1],UC[5*i+2],UC[5*i+3],UC[5*i+4]); 
		}
		if( i < I-1 ) { printf(";\n"); }
		else { printf("];\n"); }
	}
	
	printf("W = [ ");
	for( i = 0 ; i < I ; i++ ) { 
		if( i > 0 ) { 
			printf("      %0.16f , %0.16f ", W[i], W[I+i]); 
		} else {
			printf("%0.16f , %0.16f ", W[i], W[I+i]); 
		}
		if( i < I-1 ) { printf(";\n"); }
		else { printf("];\n"); }
	}
	
	printf("U = [ ");
	for( i = 0 ; i < I ; i++ ) { 
		if( i > 0 ) { 
			printf("      %0.16f , %0.16f ", U[i], U[I+i]); 
		} else { 
			printf("%0.16f , %0.16f ", U[i], U[I+i]); 
		}
		if( i < I-1 ) { printf(";\n"); }
		else { printf("];\n"); }
	}
	
	printf("PL = [ ");
	for( i = 0 ; i < I ; i++ ) { 
		if( i > 0 ) { 
			printf("       %0.16f , %0.16f ", PL[i], PL[I+i]); 
		} else {
			printf("%0.16f , %0.16f ", PL[i], PL[I+i]); 
		}
		if( i < I-1 ) { printf(";\n"); }
		else { printf("];\n"); }
	}
	
	printf("P = [ %0.16f , %0.16f ];\n",P[0],P[1]);
	
	printf("DpU = [ ");
	for( i = 0 ; i < I ; i++ ) { 
		if( i > 0 ) { 
			printf("        %0.16f , %0.16f ", DpU[i], DpU[I+i]);  
		} else {
			printf("%0.16f , %0.16f ", DpU[i], DpU[I+i]); 
		}
		if( i < I-1 ) { printf(";\n"); }
		else { printf("];\n"); }
	}
	
	printf("DpUPL = [ ");
	for( i = 0 ; i < I ; i++ ) { 
		if( i > 0 ) { 
			printf("       %0.16f , %0.16f ", DpUPL[i], DpUPL[I+i]);
		} else { 
			printf("%0.16f , %0.16f ", DpUPL[i], DpUPL[I+i]); 
		}
		if( i < I-1 ) { printf(";\n"); }
		else { printf("];\n"); }
	}
	
	printf("LAMp = [ %0.16f , %0.16f ];\n",LAMp[0],LAMp[1]);
	
	printf("GAMp = [ %0.16f , %0.16f ;\n",GAMp[0], GAMp[2] );
	printf("         %0.16f , %0.16f ];\n",GAMp[1], GAMp[3]);	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Problem initialization
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void initprob()
{
	int i, j;
	
	char  namesfp[50];
	FILE* samplefp;
	
	// random number generators (for Mixed Logit coefficient samples)
	UNUR_URNG*  unrng;
	UNUR_GEN*   urngen;
	
	UNUR_URNG*  unrng_sn;
	UNUR_GEN*   snrngen;
	
	UNUR_DISTR* distr;
	UNUR_PAR*   param;
	
	// The RNGSTREAMS library sets a package seed. 
	unsigned long seed[] = {111u, 222u, 333u, 444u, 555u, 666u};
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * ALLOCATE MEMORY * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	Jf	= (int*)calloc(F,sizeof(int));				// number of products per firm
	sJf	= (int*)calloc(F+1,sizeof(int));			// cummulative firm "starts"
	
	Y	= (double*)calloc(K*J,sizeof(double));		// characteristics (all vehicles)
	p	= (double*)calloc(J,sizeof(double));		// prices (all vehicles)
	p0	= (double*)calloc(T*J,sizeof(double));		// initial prices (all vehicles)
	
	c	= (double*)calloc(J,sizeof(double));		// costs (all vehicles)
	cd  = (double*)calloc(J,sizeof(double));		// cost data (all vehicles)
	m	= (double*)calloc(J,sizeof(double));		// markups (all vehicles)
	
	pr    = (double*)calloc(F,sizeof(double));		// profts (all firms)
	
	Z     = (double*)calloc(I,sizeof(double));		// observed demographic variable (1/income)
	V     = (double*)calloc(Kp*I,sizeof(double));// random coefficients
	UC    = (double*)calloc(Kp*I,sizeof(double));// utility coefficients
	
	W     = (double*)calloc(I*J,sizeof(double));	// fixed components of utility
	U     = (double*)calloc(I*J,sizeof(double));	// utilities
	E     = (double*)calloc(I*J,sizeof(double));	// exponentiated utilities
	PL    = (double*)calloc(I*J,sizeof(double));	// logit choice probabilities
	
	P     = (double*)calloc(J,sizeof(double));		// mixed logit choice probabilities
	
	DpU   = (double*)calloc(I*J,sizeof(double));	// matrix of utility price derivatives
	DpUPL = (double*)calloc(I*J,sizeof(double));	// utility price derivatives, times logit choice probabilities
	LAMp  = (double*)calloc(J,sizeof(double));		// needed for price derivatives of choice probabilities
	GAMp  = (double*)calloc(J*J,sizeof(double));	// needed for price derivatives of choice probabilities
	
	cg    = (double*)calloc(J,sizeof(double));		// combined gradient (w.r.t prices)
	z     = (double*)calloc(J,sizeof(double));		// zeta map
	phi   = (double*)calloc(J,sizeof(double));		// p - c - zeta
	
	DppU  = (double*)calloc(I*J,sizeof(double));	// second price derivatives of utility
	
	PIL   = (double*)calloc(I*F,sizeof(double));	// matrix of "logit profits"
	CHIp  = (double*)calloc(J,sizeof(double));		// needed for price derivatives of combined gradient map
	ZHIp  = (double*)calloc(J,sizeof(double));		// needed for price derivatives of zeta/phi map
	PHIpp = (double*)calloc(J*J,sizeof(double));	// needed for price derivatives of either price equilibrium map
	PSIpp = (double*)calloc(J*J,sizeof(double));	// needed for price derivatives of either price equilibrium map
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITALIZE DEMOGRAPHIC-INDEPENDENT QUANTITIES  * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// initialize designs and prices (derived from CY2005 data)
	
	printf("CY2005 BLP95 Example -- Initializing problem...\n");
	
	// read in product characteristic vectors
	readdata();
	
	/*
	printf("Y: \n");
	for( k = 0 ; k < K ; k++ ) {
		printf("  ");
		for( j = 0 ; j < J ; j++ ) {
			printf("%0.4f, ",Y[K*j+k]);
		}
		printf("\n");
	}
	*/
	
	Iinv = 1.0 / (double)I;
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * CREATE RANDOM NUMBER GENERATORS * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	RngStream_SetPackageSeed(seed);
	
	/* STANDARD UNIFORM GENERATOR  * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// Make an object for uniform random number generator.
	unrng = unur_urng_rngstream_new("unrng");
	if( unrng == NULL ) { 
		printf("ERROR - Uniform generator could not be constructed.\n");
		exit(EXIT_FAILURE); 
	}
	
	// use predefined distribution - uniform
	distr = unur_distr_uniform(NULL, 0);
	
	// use "auto" method (why, not sure)
	param = unur_auto_new(distr);
	
	// Set uniform generator in parameter object
	unur_set_urng(param, unrng);
	
	// Create the uniform random number generator object.
	urngen = unur_init(param); // param is "destroyed" here
	if( urngen == NULL ) { 
		printf("ERROR - Uniform generator could not be constructed.\n");
	}
	
	// free distribution object
	unur_distr_free(distr);
	
	/* STANDARD NORMAL GENERATOR * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// Make a uniform generator stream object for standard normal random number generator.
	unrng_sn = unur_urng_rngstream_new("unrng_sn");
	if( unrng_sn == NULL ) { 
		printf("ERROR - Uniform generator could not be constructed.\n");
		exit(EXIT_FAILURE); 
	}
	
	// use predefined distribution - standard normal
	distr = unur_distr_normal(NULL, 0);
	
	// use "TDR" method (why, not sure)
	param = unur_tdr_new(distr);
	
	// Set uniform generator in parameter object
	unur_set_urng(param, unrng_sn);
	
	// Create the uniform random number generator object.
	snrngen = unur_init(param); // param is "destroyed" here
	if( snrngen == NULL ) { 
		printf("ERROR - Uniform generator could not be constructed.\n");
	}
	
	// free distribution object
	unur_distr_free(distr);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * SAMPLE DEMOGRAPHICS * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// Income is roughly following empirical distribution of income assuming >= 50,000 USD/ year
	// from 2006 Current Population Survey
	for( i = 0 ; i < I ; i++ ) {
		
		Z[i] = unur_sample_cont(urngen);
		if( Z[i] <= 0.05026 ) { Z[i] = 5.0 + 0.25 * unur_sample_cont(urngen); }
		else { 
			if( Z[i] <= 0.0833 ) { Z[i] = 5.25 + 0.25 * unur_sample_cont(urngen); }
			else {
				if( Z[i] <= 0.12548 ) { Z[i] = 5.5 + 0.25 * unur_sample_cont(urngen); }
				else {
					if( Z[i] <= 0.15649 ) { Z[i] = 5.75 + 0.25 * unur_sample_cont(urngen); }
					else {
						if( Z[i] <= 0.2073 ) { Z[i] = 6.0 + 0.25 * unur_sample_cont(urngen); }
						else {
							if( Z[i] <= 0.2343 ) { Z[i] = 6.25 + 0.25 * unur_sample_cont(urngen); }
							else {
								if( Z[i] <= 0.2721 ) { Z[i] = 6.5 + 0.25 * unur_sample_cont(urngen); }
								else {
									if( Z[i] <= 0.3004 ) { Z[i] = 6.75 + 0.25 * unur_sample_cont(urngen); }
									else {
										if( Z[i] <= 0.3372 ) { Z[i] = 7.0 + 0.25 * unur_sample_cont(urngen); }
										else {
											if( Z[i] <= 0.3639 ) { Z[i] = 7.25 + 0.25 * unur_sample_cont(urngen); }
											else {
												if( Z[i] <= 0.3981 ) { Z[i] = 7.5 + 0.25 * unur_sample_cont(urngen); }
												else {
													if( Z[i] <= 0.4224 ) { Z[i] = 7.75 + 0.25 * unur_sample_cont(urngen); }
													else {
														if( Z[i] <= 0.4540 ) { Z[i] = 8.0 + 0.25 * unur_sample_cont(urngen); }
														else {
															if( Z[i] <= 0.4778 ) { Z[i] = 8.25 + 0.25 * unur_sample_cont(urngen); }
															else {
																if( Z[i] <= 0.5024 ) { Z[i] = 8.5 + 0.25 * unur_sample_cont(urngen); }
																else {
																	if( Z[i] <= 0.5227 ) { Z[i] = 8.75 + 0.25 * unur_sample_cont(urngen); }
																	else {
																		if( Z[i] <= 0.5498 ) { Z[i] = 9.0 + 0.25 * unur_sample_cont(urngen); }
																		else {
																			if( Z[i] <= 0.5676 ) { Z[i] = 9.25 + 0.25 * unur_sample_cont(urngen); }
																			else {
																				if( Z[i] <= 0.5885 ) { Z[i] = 9.5 + 0.25 * unur_sample_cont(urngen); }
																				else {
																					if( Z[i] <= 0.6059 ) { Z[i] = 9.75 + 0.25 * unur_sample_cont(urngen); }
																					else { 
																						// choose according to an exponential distribution
																						// for incomes over 100k
																						Z[i] = unur_sample_cont(urngen);
																						Z[i] = 10.0 - log( 1.0 - Z[i] ) / 0.5;
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
									}
								}
							}
						}
					}
				}
			}
		}
		
		if( i == 0 ) { maxinc = Z[0]; maxinci = 0; }
		else { if( Z[i] > maxinc ) { maxinc = Z[i]; maxinci = i; } }
	}
	
	// draw I x K (price, $/mi, l*w, hp/wt, & og) samples from a standard normal distribution
	for( i = 0 ; i < Kp*I ; i++ ) { V[i] = unur_sample_cont(snrngen); }
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * DEFINE UTILITY COEFFICIENTS * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	for( i = 0 ; i < I ; i++ ) { 
		
		UC[Kp*i  ] = uc_a[0] + uc_c[0] * fabs( V[Kp*i] ); // price coefficient (must be positive)
		UC[Kp*i+1] = uc_a[1] + uc_c[1] * V[Kp*i+1]; // coefficient on 1 / mpg
		UC[Kp*i+2] = uc_a[2] + uc_c[2] * V[Kp*i+2]; // coefficient on 1 / acc
		UC[Kp*i+3] = uc_a[3] + uc_c[3] * V[Kp*i+3]; // coefficient on footprint
		UC[Kp*i+4] = uc_a[4] + uc_c[4] * V[Kp*i+4]; // coefficient on outside good
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	sprintf(namesfp, "CY2005_BLP95_SMPL_I=%d.csv", I);
	samplefp = fopen(namesfp,"w");
	if( samplefp != NULL ) {
		
		fprintf(samplefp, "Z, V(:,1), V(:,2), V(:,3), V(:,4), V(:,5), UC(:,1), UC(:,2), UC(:,3), UC(:,4), UC(:,5)\n");
		
		for( i = 0 ; i < I ; i++ ) {
			fprintf(samplefp, "%0.16f, ",  Z[i]    );
			fprintf(samplefp, "%0.16f, ",  V[Kp*i  ]);
			fprintf(samplefp, "%0.16f, ",  V[Kp*i+1]);
			fprintf(samplefp, "%0.16f, ",  V[Kp*i+2]);
			fprintf(samplefp, "%0.16f, ",  V[Kp*i+3]);
			fprintf(samplefp, "%0.16f, ",  V[Kp*i+4]);
			fprintf(samplefp, "%0.16f, ", UC[Kp*i  ]);
			fprintf(samplefp, "%0.16f, ", UC[Kp*i+1]);
			fprintf(samplefp, "%0.16f, ", UC[Kp*i+2]);
			fprintf(samplefp, "%0.16f, ", UC[Kp*i+3]);
			fprintf(samplefp, "%0.16f\n", UC[Kp*i+4]);
		}
		
		
		fclose(samplefp);
		
	} else {
		printf("WARNING -- problem opening sample file. Did not store samples.\n");
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITALIZE DEMOGRAPHIC-DEPENDENT QUANTITIES  * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// Make sure W (fixed component of utility) is set correctly to include everything
	// except price component
	for( j = 0 ; j < J ; j++ ) {
		for( i = 0 ; i < I ; i++ ) {
			W[I*j+i]  = UC[Kp*i+1] * Y[K*j  ]; // $/mi
			W[I*j+i] += UC[Kp*i+2] * Y[K*j+1]; // l*w
			W[I*j+i] += UC[Kp*i+3] * Y[K*j+2]; // hp/wt
			W[I*j+i] -= UC[Kp*i  ] * log( Z[i] ) + UC[Kp*i+4]; // outside good
		}
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITIALIZE INITIAL PRICES * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	for( j = 0 ; j < J ; j++ ) {
		for( t = 0 ; t < T ; t++ ) {
			p0[J*t+j] = MINPRICE + (MAXPRICE-MINPRICE) * unur_sample_cont(urngen);
		}
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	unur_urng_free(unrng);
	unur_free(urngen);
	
	unur_urng_free(unrng_sn);
	unur_free(snrngen);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	printf("maxinc = %0.4f (%i/%i)\n",maxinc,maxinci,I);
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Read characteristic data from a csv file. 
 * 
 * Expects (at least) J lines with each line containing
 * 
 *	firm index    mpg    hp    wt    l     w     bs
 *	(-)			 (mpg)  (hp)  (lbs)  (in)  (in)  (-)
 * 
 * Also expects the entries to be firm-sequential. 
 * 
 * From this data, Jf[:] and sJf[:] are formed. Costs are estimated using the WFS
 * cost model given this data (ensuring that all products have positive costs)
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void readdata()
{
	int j = 0;
	FILE * fp;
	
	int f, curr_f = -1;
	
	char line[1024];
	char * lptr;
	
	double mpg, length, width, hp, weight, acc;
	int bodystyle;
	
	double **coeffs;
	
	// Allocate space for cost model coefficients
	coeffs = (double**)calloc(14, sizeof(double*));
	if( coeffs == NULL ) { 
		printf("ERROR:: memory allocation failure.\n");
		return;
	}
	
	for( t = 0 ; t < 14 ; t++ ) {
		coeffs[t] = (double*)calloc(6, sizeof(double));
		if( coeffs[t] == NULL ) { 
			printf("ERROR:: memory allocation failure.\n");
			return;
			// if we have previously allocated memory, this might
			// cause a leak?
		}
	}
	
	t = 0; // Two-Seaters
	(coeffs[t])[0] =  0.3669; // constant
	(coeffs[t])[1] = 10.6686; // exp( - acceleration )
	(coeffs[t])[2] =  0.0175; // technology content
	(coeffs[t])[3] =  0.2579 / 1000; // weight
	(coeffs[t])[4] = -0.0082 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 1; // Mini-Compact Cars (assumed == "Compact Cars")
	(coeffs[t])[0] =  0.7800; // constant
	(coeffs[t])[1] =  1.9716; // exp( - acceleration )
	(coeffs[t])[2] =  0.0016; // technology content
	(coeffs[t])[3] =  0.2250 / 1000; // weight
	(coeffs[t])[4] = -0.0123 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 2; // Sub-Compact Cars (assumed == "Compact Cars")
	(coeffs[t])[0] =  0.7800; // constant
	(coeffs[t])[1] =  1.9716; // exp( - acceleration )
	(coeffs[t])[2] =  0.0016; // technology content
	(coeffs[t])[3] =  0.2250 / 1000; // weight
	(coeffs[t])[4] = -0.0123 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 3; // Compact Cars (assumed == "Compact Cars")
	(coeffs[t])[0] =  0.7800; // constant
	(coeffs[t])[1] =  1.9716; // exp( - acceleration )
	(coeffs[t])[2] =  0.0016; // technology content
	(coeffs[t])[3] =  0.2250 / 1000; // weight
	(coeffs[t])[4] = -0.0123 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 4; // Midsize Cars
	(coeffs[t])[0] =  0.5540; // constant
	(coeffs[t])[1] = 24.3842; // exp( - acceleration )
	(coeffs[t])[2] =  0.0054; // technology content
	(coeffs[t])[3] =  0.1963 / 1000; // weight
	(coeffs[t])[4] = -0.0071 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 5; // Large Cars (assumed == "Fullsize Cars")
	(coeffs[t])[0] =  0.4029; // constant
	(coeffs[t])[1] = 24.0527; // exp( - acceleration )
	(coeffs[t])[2] =  0.0057; // technology content
	(coeffs[t])[3] =  0.2339 / 1000; // weight
	(coeffs[t])[4] = -0.0069 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 6; // Small Station Wagons (assumed == "Fullsize Cars")
	(coeffs[t])[0] =  0.4029; // constant
	(coeffs[t])[1] = 24.0527; // exp( - acceleration )
	(coeffs[t])[2] =  0.0057; // technology content
	(coeffs[t])[3] =  0.2339 / 1000; // weight
	(coeffs[t])[4] = -0.0069 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 7; // Midsize Station Wagons (assumed == "Fullsize Cars")
	(coeffs[t])[0] =  0.4029; // constant
	(coeffs[t])[1] = 24.0527; // exp( - acceleration )
	(coeffs[t])[2] =  0.0057; // technology content
	(coeffs[t])[3] =  0.2339 / 1000; // weight
	(coeffs[t])[4] = -0.0069 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 8; // Minivans (same as Midsize Cars)
	(coeffs[t])[0] =  0.5540; // constant
	(coeffs[t])[1] = 24.3842; // exp( - acceleration )
	(coeffs[t])[2] =  0.0054; // technology content
	(coeffs[t])[3] =  0.1963 / 1000; // weight
	(coeffs[t])[4] = -0.0071 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 9; // SUVs
	(coeffs[t])[0] =  0.0200; // constant
	(coeffs[t])[1] = 92.3965; // exp( - acceleration )
	(coeffs[t])[2] =  0.0038; // technology content
	(coeffs[t])[3] =  0.3470 / 1000; // weight
	(coeffs[t])[4] = -0.0108 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 10; // Cargo Vans (same as passenger vans, large pickups)
	(coeffs[t])[0] =   0.3025; // constant
	(coeffs[t])[1] = 160.5600; // exp( - acceleration )
	(coeffs[t])[2] =   0.0066; // technology content
	(coeffs[t])[3] =   0.2538 / 1000; // weight
	(coeffs[t])[4] =  -0.0055 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 11; // Passenger Vans (same as cargo vans, large pickups)
	(coeffs[t])[0] =   0.3025; // constant
	(coeffs[t])[1] = 160.5600; // exp( - acceleration )
	(coeffs[t])[2] =   0.0066; // technology content
	(coeffs[t])[3] =   0.2538 / 1000; // weight
	(coeffs[t])[4] =  -0.0055 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 12; // Small Pickups
	(coeffs[t])[0] =   0.3025; // constant
	(coeffs[t])[1] = 719.5790; // exp( - acceleration )
	(coeffs[t])[2] =   0.0066; // technology content
	(coeffs[t])[3] =   0.2621 / 1000; // weight
	(coeffs[t])[4] =  -0.0055 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	t = 13; // Large Pickups (same as passenger and cargo vans)
	(coeffs[t])[0] =   0.3025; // constant
	(coeffs[t])[1] = 160.5600; // exp( - acceleration )
	(coeffs[t])[2] =   0.0066; // technology content
	(coeffs[t])[3] =   0.2538 / 1000; // weight
	(coeffs[t])[4] =  -0.0055 / 1000; // weight * acceleration
	// weight, in data, is given in lbs, while constraints use 1000 lbs as the unit
	// this division in coeffs 3 and 4 allows us to translate units once
	
	// open file
	fp = fopen("WFS2006Data.txt", "r");
	if( fp == NULL ) { 
		printf("Could not open file specified.\n");
		return; 
	}
	
	// read a line (no header row)
	while( fgets(line, 1024, fp) != NULL ) {
		
		// read firm index for this row
		f = (int)strtol(line, &lptr, 10);
		if( f < 1 ) { 
			printf("ERROR -- program expects firms to be in FORTRAN-style indexing.\n");
			exit(1);
		}
		
		// convert to C-style indexing
		f--;
		
		// check to see if current firm needs to be reset. 
		// this will always initialize curr_f at first
		if( f != curr_f ) { 
			if( f == curr_f + 1 ) { curr_f++; }
			else {
				printf("ERROR -- program expects firms to be listed in sequential blocks.\n");
				exit(1);
			}
		}
		
		// always increment number of products offered by the current firm
		Jf[curr_f]++;
		
		// read vehicle data (in order of file)
		mpg		= strtod(lptr, &lptr);
		hp		= strtod(lptr, &lptr);
		weight	= strtod(lptr, &lptr);
		length	= strtod(lptr, &lptr);
		width	= strtod(lptr, &lptr);
		acc		= strtod(lptr, &lptr);
		bodystyle = (int)strtol(lptr, &lptr, 10);
		
		// convert mpg into Y(j,1) = 10 Mi / $
		Y[K*j+0] = mpg / ( pG  * 10.0 );
		
		// convert l (in) & w (in) into Y(j,2) = ( l 100in ) * ( w 100in )
		Y[K*j+1] = ( length / 100.0 ) * ( width / 100.0 );
		
		// convert hp & wt (lbs) into Y(j,3) = hp / ( wt 10lbs )
		Y[K*j+2] = hp / ( weight / 10.0 );
		
		// use WFS cost formula to compute costs
		cd[j]  = (coeffs[bodystyle-1])[0];
		cd[j] += (coeffs[bodystyle-1])[1] * exp( - acc );
		cd[j] += (coeffs[bodystyle-1])[2] * 0.0; // no technology
		cd[j] += (coeffs[bodystyle-1])[3] * acc * acc * 0.0; // no technology
		cd[j] += (coeffs[bodystyle-1])[4] * weight;
		cd[j] += (coeffs[bodystyle-1])[5] * weight * acc;
		
		// ensure positivity
		if( cd[j] < 0.0 ) { cd[j] = - cd[j]; }
		
		// increment line counter, but don't read more than J lines
		j++; 
		
		// printf("%i: %i, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %0.2f, %i, %0.2f\n",j,curr_f,mpg,length,width,hp,weight,acc,bodystyle,cd[j-1]);
		
		if( j == J ) { break; }
		 
	}
	
	printf("Read data for %i firms:\n",curr_f);
	printf("Jf = [ %i ",Jf[0]);
	for( f = 1 ; f < F ; f++ ) { printf(", %i ",Jf[f]); }
	printf("]\n");
	
	// need these
	sJf[0] = 0; 
	Jfstar = 0;
	for( f = 0 ; f < F ; f++ ) {
		sJf[f+1] = sJf[f] + Jf[f]; 
		if( Jfstar < Jf[f] ) { Jfstar = Jf[f]; }
	}
	
	printf("sJf = [ %i ",sJf[0]);
	for( f = 1 ; f <= F ; f++ ) { printf(", %i ",sJf[f]); }
	printf("]\n");
	
	printf("Jfstar = %i\n",Jfstar);
	
	// free coefficients data structure
	for( t = 0 ; t < 14 ; t++ ) { if( coeffs[t] != NULL ) { free( coeffs[t] ); } }
	if( coeffs != NULL ) { free(coeffs); }
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Problem deletion
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void killprob()
{
	free(Jf);
	free(sJf);
	
	free(Y);
	free(p);
	free(p0);
	
	free(c);
	free(cd);
	free(m);
	
	free(pr);
	
	free(Z);
	free(V);
	free(UC);
	
	free(W);
	free(U);
	free(E);
	free(PL);
	
	free(P);
	
	free(DpU);
	free(DpUPL);
	free(LAMp);
	free(GAMp);
	
	free(cg);
	free(z);
	free(phi);
	
	free(DppU);
	
	free(PIL);
	free(CHIp);
	free(ZHIp);
	free(PHIpp);
	free(PSIpp);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Utility calculations
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void utilities()
{
	int i, j;
	double tmp;
	
	// printf("  Computing utilities...\n"); 
	
	for( j = 0 ; j < J ; j++ ) {
		for( i = 0 ; i < I ; i++ ) {
			
			tmp = Z[i] - p[j];
			
			if( tmp > 0.0 ) { 
				
				U[I*j+i]    =   UC[Kp*i] * log( tmp ) + W[I*j+i]; 
				DpU[I*j+i]  = - UC[Kp*i] / tmp;
				DppU[I*j+i] =   DpU[I*j+i] / tmp;
				
			} else { // p[j] >= Z[i]
				
				U[I*j+i]    = - 1.0e20;
				DpU[I*j+i]  =   0.0;
				DppU[I*j+i] = - 1.0 / UC[Kp*i];
				
			}
			
		}
	}
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * probabilities calculation
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void probabilities()
{	
	int i, j;
	double uimax = 0.0, Si;
	
	// printf("  Computing PL and P...\n"); 
	
	// P <- 0
	cblas_dscal( J , 0.0 , P , 1 );
	
	for( i = 0 ; i < I ; i++ ) {
		
		// determine maximum utility for individual i
		uimax = U[i];
		for( j = 1 ; j < J ; j++ ) { 
			if( U[I*j+i] > uimax ) { 
				uimax = U[I*j+i]; 
			} 
		}
		
		// if maximum utility is finite, we're ok. If maximum
		// utility is - infty, then all probabilities are zero
		// this should just be "enforced" rather than computed
		if( uimax > -1.0e20 ) {
			
			// computed and exponentiate * shifted * utilities, while
			// accumulating sum of these values
			Si = exp( - uimax );
			for( j = 0 ; j < J ; j++ ) { 
				E[I*j+i] = exp( U[I*j+i] - uimax );
				Si += E[I*j+i];
			}
			
			// form logit choice probabilities (this loop must follow
			// the loop above because we must compute the full Si)
			// we also accumulate mixed logit choice probabilities
			
			// PL[i,:] <- E[i,:] / Si
			cblas_dcopy( J , E+i , I , PL+i , I );
			cblas_dscal( J , 1.0/Si , PL+i , I );
			
			// P[:] <- P[:] + PL[i,:]
			cblas_daxpy( J , 1.0 , PL+i , I , P , 1 );
		
		} else { cblas_dscal( J , 0.0 , PL+i , I ); } // PL[i,:] <- 0
		
	}
	
	// divide accumulated sums in P[:] by I
	if( integralscale ) { cblas_dscal( J , Iinv , P , 1 ); }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void lamgam()
{
	int i, j;
	
	// printf("  Computing LAMp...\n");
	
	// LAMp <- 0
	cblas_dscal(J , 0.0, LAMp, 1);
	
	// Compute DpUPL = DpU .* PL and accumulate to form LAMp
	for( j = 0 ; j < J ; j++ ) {
		for( i = 0 ; i < I ; i++ ) {
			DpUPL[I*j+i] = DpU[I*j+i] * PL[I*j+i];
			LAMp[j] += DpUPL[I*j+i];
		}
	}
	
	if( integralscale ) { cblas_dscal( J , Iinv , LAMp , 1 ); }
	
	// printf("  Computing GAMp...\n");
	
	if( integralscale ) {
		// GAMp <- (1/I) * PL' * DpUPL
		cblas_dgemm(CblasColMajor, 
					CblasTrans,
					CblasNoTrans,
					J, // number of rows of output matrix
					J, // number of columns of output matrix
					I, // shared inner dimension
					Iinv,
					PL,
					I,
					DpUPL,
					I,
					0.0,
					GAMp,
					J);
	} else {
		
		// GAMp <- PL' * DpUPL
		cblas_dgemm(CblasColMajor, 
					CblasTrans,
					CblasNoTrans,
					J, // number of rows of output matrix
					J, // number of columns of output matrix
					I, // shared inner dimension
					1.0,
					PL,
					I,
					DpUPL,
					I,
					0.0,
					GAMp,
					J);
	}
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void chiphipsi()
{
	int i, j, k, l, f;
	
	// printf("  Computing PIL...\n"); 
	
	for( f = 0 ; f < F ; f++ ) {
		
		// PIL[:,f] <- PL[:,{f}] * m[{f}]
		cblas_dgemv(CblasColMajor,
					CblasNoTrans,
					I,
					Jf[f],
					1.0,
					PL + I * sJf[f],
					I,
					m + sJf[f],
					1,
					0.0,
					PIL + I*f,
					1);
		
	}
	
	// printf("  Computing CHIp...\n"); 
	
	// 
	// CHIp[j] = Iinv * sum_i ( DppU[i,j] * PL[i,j] + DpU[i,j] * DpUPL[i,j] ) ( m[j] - PIL[i,f(j)] )
	//
	
	cblas_dscal( J , 0.0 , CHIp , 1 );
	
	for( f = 0 ; f < F ; f++ ) {
		
		for( j = sJf[f] ; j < sJf[f+1] ; j++ ) {
			
			for( i = 0 ; i < I ; i++ ) {
				
				if( PL[I*j+i] > 0.0 ) {
					
					// CHI[j] <- CHI[j]
					//				+ ( m[j] - PIL[i,f] )
					//						* ( DppU[i,j] PL[i,j] + DpU[i,j] DpUPL[i,j] )
					CHIp[j] += ( m[j] - PIL[I*f+i] ) 
					* ( DppU[I*j+i] * PL[I*j+i]
					   + DpU[I*j+i] * DpUPL[I*j+i] );
					
				}
				// not having an "else" here amounts to adding
				// 0 into CHIp[j] if PL[i,j] = 0
				
			}
			
		}
		
	}
	
	if( integralscale ) { cblas_dscal( J , Iinv , CHIp , 1 ); }
	cblas_daxpy( J , 2.0 , LAMp , 1 , CHIp , 1 );
	
	
	// printf("  Computing PHIpp...\n"); 
	
	// 
	// PHIpp <- 1/I DpUPL' * DpUPL
	// 
	if( integralscale ) { 
		
		// PHIpp <- (1/I) * DpUPL' * DpUPL
		cblas_dgemm(CblasColMajor, 
					CblasTrans,
					CblasNoTrans,
					J,
					J,
					I,
					Iinv,
					DpUPL,
					I,
					DpUPL,
					I,
					0.0,
					PHIpp,
					J);
		
	} else {
		
		// PHIpp <- DpUPL' * DpUPL
		cblas_dgemm(CblasColMajor, 
					CblasTrans,
					CblasNoTrans,
					J,
					J,
					I,
					1.0,
					DpUPL,
					I,
					DpUPL,
					I,
					0.0,
					PHIpp,
					J);
		
	}
	
	// printf("  Computing PSIpp...\n"); 
	
	// PSIpp[k,l] = Iinv * sum_i DpUPL[i,k] PIL[i,f(k)] DpUPL[i,l]
	// 
	
	cblas_dscal( J * J , 0.0 , PSIpp, 1 );
	
	for( l = 0 ; l < J ; l++ ) {
		
		for( f = 0 ; f < F ; f++ ) {
			
			for( k = sJf[f] ; k < sJf[f+1] ; k++ ) {
				for( i = 0 ; i < I ; i++ ) {
					PSIpp[ J*l + k ] += DpUPL[ I*k + i ] * PIL[I*f+i] * DpUPL[ I*l + i ];
				}
			}
			
		}
		
	}
	
	if( integralscale ) { cblas_dscal( J * J , Iinv , PSIpp, 1 ); }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void zhiphipsi()
{
	int i, j, k, l, f;
	
	// printf("  Computing PIL...\n"); 
	
	for( f = 0 ; f < F ; f++ ) {
		
		// PIL[:,f] <- PL[:,{f}] * m[{f}]
		cblas_dgemv(CblasColMajor,
					CblasNoTrans,
					I,
					Jf[f],
					1.0,
					PL + I * sJf[f],
					I,
					m + sJf[f],
					1,
					0.0,
					PIL + I*f,
					1);
		
	}
	
	// printf("  Computing ZHIp...\n"); 
	
	// 
	// ZHIp[j] = LAMp[j] + Iinv * sum_i ( DppU[i,j] * PL[i,j] + DpU[i,j] * DpUPL[i,j] ) ( zeta[j] - PIL[i,f(j)] )
	//
	
	cblas_dscal( J , 0.0 , ZHIp , 1 );
	
	for( f = 0 ; f < F ; f++ ) {
		
		for( j = sJf[f] ; j < sJf[f+1] ; j++ ) {
			
			for( i = 0 ; i < I ; i++ ) {
				
				if( PL[I*j+i] > 0.0 ) {
					
					// ZHIp[j] <- ZHIp[j]
					//				+ ( zeta[j] - PIL[i,f] )
					//						* ( DppU[i,j] PL[i,j] + DpU[i,j] DpUPL[i,j] )
					ZHIp[j] += ( z[j] - PIL[I*f+i] ) 
					* ( DppU[I*j+i] * PL[I*j+i]
					   + DpU[I*j+i] * DpUPL[I*j+i] );
					
				}
				// not having an "else" here amounts to adding
				// 0 into ZHIp[j] if PL[i,j] = 0
				
			}
			
		}
		
	}
	
	if( integralscale ) { cblas_dscal( J , Iinv , ZHIp , 1 ); }
	cblas_daxpy( J , 1.0 , LAMp , 1 , ZHIp , 1 );
	
	// printf("  Computing PHIpp...\n"); 
	
	// 
	// PHIpp <- 1/I DpUPL' * DpUPL
	// 
	if( integralscale ) { 
		
		// PHIpp <- (1/I) * DpUPL' * DpUPL
		cblas_dgemm(CblasColMajor, 
					CblasTrans,
					CblasNoTrans,
					J,
					J,
					I,
					Iinv,
					DpUPL,
					I,
					DpUPL,
					I,
					0.0,
					PHIpp,
					J);
		
	} else {
		
		// PHIpp <- DpUPL' * DpUPL
		cblas_dgemm(CblasColMajor, 
					CblasTrans,
					CblasNoTrans,
					J,
					J,
					I,
					1.0,
					DpUPL,
					I,
					DpUPL,
					I,
					0.0,
					PHIpp,
					J);
		
	}
	
	// printf("  Computing PSIpp...\n"); 
	
	// PSIpp[k,l] = Iinv * sum_i DpUPL[i,k] PIL[i,f(k)] DpUPL[i,l]
	// 
	
	cblas_dscal( J * J , 0.0 , PSIpp, 1 );
	
	for( l = 0 ; l < J ; l++ ) {
		
		for( f = 0 ; f < F ; f++ ) {
			
			for( k = sJf[f] ; k < sJf[f+1] ; k++ ) {
				for( i = 0 ; i < I ; i++ ) {
					PSIpp[ J*l + k ] += DpUPL[ I*k + i ] * PIL[I*f+i] * DpUPL[ I*l + i ];
				}
			}
			
		}
		
	}
	
	if( integralscale ) { cblas_dscal( J * J , Iinv , PSIpp, 1 ); }
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * profit calculations
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void markups()
{
	// printf("  Computing markups...\n"); 
	
	cblas_dcopy(J, p, 1, m, 1); // m <- p
	cblas_daxpy(J, -1.0, c, 1, m, 1); // m <- m - c (= p - c)
}

void profits()
{
	int f;
	
	// compute profits
	for( f = 0 ; f < F ; f++ ) {
		pr[f] = cblas_ddot(Jf[f], P + sJf[f], 1, m + sJf[f], 1); // (notice the pointer arithmetic)
	}
}

void combgrad()
{
	int j, f;
	
	// printf("  Computing CG...\n"); 
	
	// cg <- diag(LAMp) * m - \tilde{GAMp}' * m + P
	
	// cg <- P
	cblas_dcopy( J , P , 1 , cg , 1 );
	
	// cg <- cg + diag(LAMp) * m
	for( j = 0 ; j < J ; j++ ) { cg[j] += LAMp[j] * m[j]; }
	
	// cg <- cg - \tile{GAMp}' * m
	for( f = 0 ; f < F ; f++ ) {
		
		// cg[{f}] <- cg[{f}] - GAMp[{f},{f}]' * m[{f}]
		cblas_dgemv(CblasColMajor,
					CblasTrans,
					Jf[f],
					Jf[f],
					-1.0,
					GAMp + J * sJf[f] + sJf[f],
					J,
					m + sJf[f],
					1,
					1.0,
					cg + sJf[f],
					1);
		
	}
	
}

// write Jacobian of combined gradient into "jac" as a dense, column-major matrix
void cgders( double * jac )
{
	int j, f;
	
	// printf("  Computing Jacobian of CG...\n"); 
	
	// 
	// diag(CHIp) - diag(m) * PHIpp - GAMp + 2 PSIpp + \tilde{ - PHIpp * diag(m) - GAMp' }
	//
	
	// jac <- - GAMp
	cblas_dcopy( J * J , GAMp , 1 , jac , 1 );
	cblas_dscal( J * J , -1.0 , jac , 1 );
	
	// jac[j,:] <- - m[j] * PHIpp[j,:] + jac[j,:]
	for( j = 0 ; j < J ; j++ ) {
		cblas_daxpy( J , - m[j] , PHIpp + j , J , jac + j , J );
	}
	
	// jac[{f},{f}] <- jac[{f},{f}] - ( GAMp[{f},{f}] + diag(m[{f}]) PHIpp[{f},{f}] )'
	//				 = jac[{f},{f}] - ( GAMp[{f},{f}]' + PHIpp[{f},{f}] * diag(m[{f}]) )
	for( f = 0 ; f < F ; f++ ) {
		
		for( j = sJf[f] ; j < sJf[f+1] ; j++ ) {
			
			// jac[{f},j] <- jac[{f},j] - GAMp[j,{f}]' 
			cblas_daxpy( Jf[f] , 
						-1.0 , 
						GAMp + J * sJf[f] + j , 
						J , 
						jac + J * j + sJf[f], 
						1 );
			
			// jac[{f},j] <- jac[{f},j] - m[j] PHIpp[{f},j]
			cblas_daxpy( Jf[f] , 
						-1.0 * m[j] , 
						PHIpp + J * j + sJf[f] , 
						1 , 
						jac + J * j + sJf[f], 
						1 );
			
		}
		
	}
	
	// jac <- 2 PSIpp + jac
	cblas_daxpy( J * J , 2.0 , PSIpp , 1 , jac , 1 );
	
	// jac <- diag(CHIp) + jac
	for( j = 0 ; j < J ; j++ ) { jac[ J * j + j ] += CHIp[j]; }
	
}

// compute firm-specific Hessians (w.r.t. prices)
void pesosc()
{
	
	// Hp[0] = CHIp[0] - 2.0 * ( m[0] * PHIpp[0] + GAMp[0] ) + 2.0 * PSIpp[0];
	
	// Hp[1] = CHIp[1] - 2.0 * ( m[1] * PHIpp[3] + GAMp[3] ) + 2.0 * PSIpp[3];
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * (extended) zeta map calculation
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
void zeta()
{
	int j, f;
	
	// printf("  Computing zeta /phi map.\n"); 
	
	// z <- inv(diag(LAMp)) * ( \tilde{GAMp}' * m - P )
	// for all prices < maxinc
	
	// z <- P
	cblas_dcopy(J, P, 1, z, 1);
	
	// z <- \tile{GAMp}' * m - z
	for( f = 0 ; f < F ; f++ ) {
		
		// z[{f}] <- GAMp[{f},{f}]' * m[{f}]
		cblas_dgemv(CblasColMajor,
					CblasTrans,
					Jf[f],
					Jf[f],
					1.0,
					GAMp + J * sJf[f] + sJf[f],
					J,
					m + sJf[f],
					1,
					-1.0,
					z + sJf[f],
					1);
		
	}
	
	// z <- inv(diag(LAMp)) * z, being careful about this operation
	for( f = 0 ; f < F ; f++ ) {
		
		for( j = sJf[f] ; j < sJf[f+1] ; j++ ) { 
			
			if( p[j] >= maxinc ) { // use extended map instead of what is calculated above
				
				// z[j] = omega[maxinci,j] * ( p[j] - maxinc ) + PL[maxinci,{f}]' * m[{f}]
				z[j]  = cblas_ddot( Jf[f] , PL + I*sJf[f] + maxinci , I , m + sJf[f] , 1 );
				z[j] += DppU[I*j+maxinci] * ( p[j] - maxinc );
				
			} else {
				
				if( LAMp[j] < 0.0 ) { z[j] /= LAMp[j]; }
				else { 
					printf("WARNING -- p[%i] = %0.16f < %0.16f = maxinc but LAMp[%i] = %0.16f.\n",j,p[j],maxinc,j,LAMp[j]);
					// use a modification of extended map instead of what is calculated above
					// z[j] = PL[maxinci,{f}]' * m[{f}]
					z[j]  = cblas_ddot( Jf[f] , PL + I*sJf[f] + maxinci , I , m + sJf[f] , 1 );
					// we exclude the "DppU[I*j+maxinci] * ( p[j] - maxinc )" term expecting
					// p[j] to be at least close to maxinc
				}
			}
			
		}
		
	}
	
	// compute phi = p - c - z also:
	cblas_dcopy(J, m, 1, phi, 1);
	cblas_daxpy(J, -1.0, z, 1, phi, 1);
	
}

// write Jacobian of phi map into "jac" as a dense, column-major matrix
void phiders( double * jac )
{
	int j, k, l, f;
	
	// printf("  Computing phi map Jacobian...\n"); 
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 *
	 *		I + inv( LAMp ) * ( diag(ZHIp) - diag(zeta) * PHIpp - GAMp 
	 *								+ 2 PSIpp + \tilde{ - PHIpp * diag(m) - GAMp' } )
	 * 
	 * with care taken concerning the LAMp inversion. 
	 * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// jac <- - GAMp
	cblas_dcopy( J * J , GAMp , 1 , jac , 1 );
	cblas_dscal( J * J , -1.0 , jac , 1 );
	
	// jac <- - diag(zeta) * PHIpp + jac
	for( j = 0 ; j < J ; j++ ) {
		
		// jac[j,:] <- - z[j] * PHIpp[j,:] + jac[j,:]
		cblas_daxpy( J , - z[j] , PHIpp + j , J , jac + j , J );
		
	}
	
	for( f = 0 ; f < F ; f++ ) {
		
		// jac[{f},{f}] <- jac[{f},{f}] - ( GAMp[{f},{f}] + diag(m[{f}]) PHIpp[{f},{f}] )'
		//				 = jac[{f},{f}] - ( GAMp[{f},{f}]' + PHIpp[{f},{f}] * diag(m[{f}]) )
		
		for( j = sJf[f] ; j < sJf[f] + Jf[f] ; j++ ) {
			
			// jac[{f},j] <- jac[{f},j] - GAMp[j,{f}]' 
			cblas_daxpy( Jf[f] , 
						-1.0 , 
						GAMp + J * sJf[f] + j , 
						J , 
						jac + J * j + sJf[f], 
						1 );
			
			// jac[{f},j] <- jac[{f},j] - m[j] PHIpp[{f},j]
			cblas_daxpy( Jf[f] , 
						-1.0 * m[j] , 
						PHIpp + J * j + sJf[f] , 
						1 , 
						jac + J * j + sJf[f], 
						1 );
			
		}
		
	}
	
	// jac <- 2 PSIpp + jac
	cblas_daxpy( J * J , 2.0 , PSIpp , 1 , jac , 1 );
	
	// jac <- diag(ZHIp) + jac
	for( j = 0 ; j < J ; j++ ) { jac[ J * j + j ] += ZHIp[j]; }
	
	// jac <- jac / LAMp[j], or re-definition per extended map
	for( f = 0 ; f < F ; f++ ) {
		
		for( k = sJf[f] ; k < sJf[f] + Jf[f] ; k++ ) {
			
			if( LAMp[k] < 0.0 ) { 
				
				// jac[k,:] <- jac[k,:] / LAMp[k]
				cblas_dscal( J , 1.0 / LAMp[k] , jac + k , J );
				
			} else {
				
				// 
				// If LAMp[k] = 0, then redefine
				//
				//		jac[k,l]
				//			
				//			= delta_{k,l} 
				//				- DpUPL[maxinci,l] * ( delta_{f(l),f(k)} m[l] - PIL[maxinci,f(k)] )
				//				- delta_{f(l),f(k)} * DpUPL[maxinci,l] * PL[maxinci,l]
				//				- delta[k,l] * DppU[ maxinci , k ] / ( DpU[ maxinci , k ]^2 )
				//
				//			= delta_{k,l} * ( 1 - DppU[ maxinci , k ] / ( DpU[ maxinci , k ]^2 ) )
				//				- delta_{f(l),f(k)} * DpUPL[maxinci,l] * ( m[l] + PL[maxinci,l] )
				//				+ DpUPL[maxinci,l] * PIL[maxinci,f(k)]
				//				
				// The 1's on the diagonal are handled below. 
				// 
				
				// jac[k,:] <- DpUPL[maxinci,l] * PIL[maxinci,f(k)]
				cblas_dcopy( J , DpUPL+maxinci , I , jac + k , J );
				cblas_dscal( J , PIL[I*f+maxinci] , jac + k , J );
				
				// jac[k,{f}] <- DpUPL[maxinci,{f}] * ( m[{f}] + PL[maxinci,{f}] )
				for( l = sJf[f] ; l < sJf[f] + Jf[f] ; l++ ) {
					jac[J*l+k] -= DpUPL[I*l+maxinci] * ( m[l] + PL[I*l+maxinci] );
				}
				
				// we expect utility routine to define DppU as 
				//
				//		lim_{p->maxinc} ( DppU / DpU^2 ) 
				//
				// when maxinc < infinity and p >= maxinc
				// 
				jac[J*k+k] -= DppU[I*k+maxinci];
				
			}
			
		}
		
	}
	
	// jac <- I + jac
	for( j = 0 ; j < J ; j++ ) { jac[J*j+j] += 1.0; }
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * update routine
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int chol(char uplo, int N, double * A, int lda)
{
	int INFO = 0;
	
	dpotrf_(&uplo, &N, A, &lda, &INFO);
	
	return INFO; // 0 on successful exit (pos. def.)
}

// evaluate Price Equilibrium Second-Order Sufficient Condition (SOSC)
//
//	1. Check each Hessian for negative definiteness, 
//	   including only those products that have prices
//	   less than maxinc
//  2. In addition, make sure that if a price p(k) is
//	   greater than or equal to maxinc, then 
//		
//			p(k) - c(k) - z(k) <= 0
//
//  ppom should be an F+1 (int) vector
//  info should be a 2F (int) vector
//
int evalPEQSOSC( int * ppom , int * info , char ** ws )
{
	int j, jj, k, f, soscflag;
	double * Hessf;
	
	double tmp;
	
	char warning_str[256];
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	if( ppom == NULL ) { 
		printf("ERROR: PEQSOSC Currently requires a 2(F+1) (= %i) ppom vector.\n",2*(F+1));
		return -1; 
	}
	
	if( info == NULL ) { 
		printf("ERROR: PEQSOSC Currently requires a 3F (= %i) info vector.\n",3*F);
		return -1; 
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// ensure all needed quantities are defined at current prices
	markups();
	utilities();
	probabilities();
	lamgam();
	zeta(); // needed if any p(j) >= maxinc
	chiphipsi();
	
	Hessf = (double*)calloc( Jfstar * Jfstar , sizeof(double) );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// ppom expected to have 2(F+1) entries
	// ppom[0] and ppom[1] will be total numbers of products priced out of the market, 
	// with respect to prices and choice probabilities (respectively)
	// ppom[2:F+1] and ppom[F+2:2F+1] will be intra-firm values, with respect
	// and choice probabilities (respectively)
	ppom[0]   = 0;
	ppom[F+1] = 0;
	for( f = 0 ; f < F ; f++ ) { 
		ppom[  1+f] = 0;
		ppom[F+2+f] = 0;
	}
	
	// assume success
	soscflag = 1;
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	for( f = 0 ; f < F ; f++ ) {
		
		// compute the NEGATIVE of firm f's Hessian matrix
		//
		// this matrix is symmetric - we only need to compute the upper
		// (or lower) triangle. LAPACK routines will expect this. 
		
		// Hessf <- 0
		cblas_dscal(Jf[f]*Jf[f], 0.0, Hessf, 1);
		
		// need a "local" index for Hessf
		for( j = 0 ; j < Jf[f] ; j++ ) {
			
			jj = j + sJf[f];
			
			// Hess[j,{f}] <- diag( m[jj] ) * PHIpp[jj,{f}] + Hess[j,{f}]
			cblas_daxpy( Jf[f] , 
						m[jj] , 
						PHIpp + J * sJf[f] + jj , 
						J , 
						Hessf + j , 
						Jf[f] );
			
			// Hess[{f},j] <- m[jj] * PHIpp[{f},jj] + Hess[{f},j]
			cblas_daxpy( Jf[f] , 
						m[jj] , 
						PHIpp + J * jj + sJf[f] , 
						1 , 
						Hessf + Jf[f] * j , 
						1 );
			
			// Hess[{f},j] <- GAMp[{f},jj] + Hess[{f},j]
			cblas_daxpy( Jf[f] , 
						1.0 , 
						GAMp + J * jj + sJf[f] , 
						1 , 
						Hessf + Jf[f] * j , 
						1 );
			
			// Hess[{f},j] <- GAMp[jj,{f}] + Hess[{f},j]
			cblas_daxpy( Jf[f] , 
						1.0 , 
						GAMp + J * sJf[f] + jj , 
						J , 
						Hessf + Jf[f] * j, 
						1 );
			
			// Hess[{f},j] <- - 2 PSIpp[{f},jj] + Hess[{f},j]
			cblas_daxpy( Jf[f] , 
						-2.0 , 
						PSIpp + J * jj + sJf[f] , 
						1 , 
						Hessf + Jf[f] * j , 
						1 );
			
			// Hess[j,j] <- - diag( CHIp[jj] ) + Hess[j,j]
			Hessf[ Jf[f] * j + j ] -= CHIp[ jj ];
			
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// now we have to modify to use only those components that 
		// have prices p(j) < maxinc
		
		for( j = 0 ; j < Jf[f] ; j++ ) {
			
			jj = j + sJf[f];
			
			// -- HACK --
			// 
			// if price is too high, then delete the corresponding
			// rows and columns in Hessf, replacing them with 
			// the identity. This effectively removes these rows
			// and columns from consideration, unaffecting the 
			// Cholesky check of positive definiteness. 
			
			if( p[jj] >= maxinc ) {
				ppom[1+f]++;
				cblas_dscal( Jf[f] , 0.0 , Hessf + j , Jf[f] );
				cblas_dscal( Jf[f] , 0.0 , Hessf + Jf[f]*j , 1 );
				Hessf[Jf[f]*j+j] = 1.0;
			}
			
		}
		
		// add to total number of ppom
		ppom[0] += ppom[1+f];
		
		// now that the NEGATIVE of Hess[{f},{f}] is defined, 
		// we can check positive definiteness with Cholesky (LAPACK wrapper)
		// (adding one means this functions returns <= 0 if there is an argument error
		// and 1 if positive definite)
		info[f] = chol( 'U' , Jf[f] , Hessf , Jf[f] ) + 1;
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// evaluate result of positive definiteness check (relative to prices)
		if( info[f] < 1 ) {
			
			// argument error in dpotrf
			info[f]--; // this makes soscflag properly negative, as returned from dpotrf
			printf("ERROR: Illegal value for argument %i passed to dpotrf_.\n",-info[f]);
			soscflag = -1;
			break; // we stop here, as there was an error
			
		} else { // dpotrf was sucessful at identifying +/- definiteness
			
			if( info[f] > 1 ) {
				
				sprintf( warning_str , "WARNING: Firm %i's hessian (w.r.t. prices) is not negative definite (info = %i)\n",f+1,info[f]-1);
				
				// printf("P[{%i}] = ",f+1); for( j = sJf[f] ; j < sJf[f+1] ; j++ ) { printf("%0.16f, ",P[j]); } printf("\n");
				
				if( ws != NULL ) {
					if( ws[0] == NULL ) {
						ws[0] = (char*)calloc( strlen(warning_str) + 1 , sizeof(char) );
						strcpy( ws[0] , warning_str );
					} else {
						ws[0] = (char*)realloc( ws[0] , ( strlen(ws[0]) + strlen(warning_str) + 1 ) * sizeof(char) );
						strcat( ws[0] , warning_str );
					}
				}
				
				// printf("%s",warning_str);
				
				soscflag = 0;
				info[f] = 0;
				// break;
			}
			
		}
		
		// second portion of the check: determine whether products are priced
		// out of the market profitably
		
		// assume success for firm f in second portion of the check
		info[F+f] = 1;
		
		// now check if phi(j) <= 0 for all j such that p(j) >= maxinc
		for( j = sJf[f] ; j < sJf[f+1] ; j++ ) {
			
			if( p[j] >= maxinc ) {
				
				if( phi[j] > 1.0e-6 ) {
					
					sprintf( warning_str , "WARNING: Firm %i prices product %i (%i) out of the market unprofitably: phi(%i) = %0.16f\n",f+1,j-sJf[f]+1,j+1,j+1,phi[j]);
					
					if( ws != NULL ) {
						if( ws[0] == NULL ) {
							ws[0] = (char*)calloc( strlen(warning_str) + 1 , sizeof(char) );
							strcpy( ws[0] , warning_str );
						} else {
							ws[0] = (char*)realloc( ws[0] , ( strlen(ws[0]) + strlen(warning_str) + 1 ) * sizeof(char) );
							strcat( ws[0] , warning_str );
						}
					}
					
					// printf("%s",warning_str);
					
					soscflag = 0;
					info[F+f] = 0;
					// break;
					
				} else {
					
					if( phi[j] >= -1.0e-6 ) {
						
						// note that this requires | phi[j] | <= 1e-6
						// here we say that we can't distinguish this value from zero
						// and thus we lack "strict complementarity"
						
						sprintf( warning_str , "WARNING: Firm %i prices product %i (%i) at or above maxinc, but phi(%i) is numerically zero (%0.16f)\n",f+1,j-sJf[f]+1,j+1,j+1,phi[j]);
						
						if( ws != NULL ) {
							if( ws[0] == NULL ) {
								ws[0] = (char*)calloc( strlen(warning_str) + 1 , sizeof(char) );
								strcpy( ws[0] , warning_str );
							} else {
								ws[0] = (char*)realloc( ws[0] , ( strlen(ws[0]) + strlen(warning_str) + 1 ) * sizeof(char) );
								strcat( ws[0] , warning_str );
							}
						}
						
						// printf("%s",warning_str);
						
						soscflag = 0;
						info[F+f] = 0;
						// break;
						
					}
				}
			}
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// now check positive definiteness relative to sub-problem
		// formed by products with 'non-zero' choice probabilities
		ppom[F+2+f] += ppom[1+f];
		for( j = 0 ; j < Jf[f] ; j++ ) {
			
			jj = j + sJf[f]; 
			
			// -- HACK --
			// 
			// if price is too high, then delete the corresponding
			// rows and columns in Hessf, replacing them with 
			// the identity. This effectively removes these rows
			// and columns from consideration, unaffecting the 
			// Cholesky check of positive definiteness. 
			
			if( p[jj] < maxinc && P[j] < 1.0e-6 ) {
				ppom[F+2+f]++;
				cblas_dscal( Jf[f] , 0.0 , Hessf + j , Jf[f] );
				cblas_dscal( Jf[f] , 0.0 , Hessf + Jf[f]*j , 1 );
				Hessf[Jf[f]*j+j] = 1.0;
			}
			
		}
		
		// add to total number of ppom
		ppom[F+1] += ppom[F+2+f];
		
		// check positive definiteness with Cholesky
		// (adding one means this functions returns <= 0 if there is an argument error
		// and 1 if positive definite)
		info[2*F+f] = chol( 'U' , Jf[f] , Hessf , Jf[f] ) + 1;
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
	}
	
	free( Hessf );
	
	return soscflag;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Callback routine for SNOPT execution. 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
int snopt_callback_func_(integer		*Status,	// SNOPT status code
						 integer		*N,			// number of variables
						 doublereal		*x,			// current variable values
						 integer		*needF,		// 0 if f(x) is not needed, > 0 if it is
						 integer		*FN,		// length of the vector of objective and constraint values
						 doublereal		*Fvals,		// values (to be calculated) for objective and constraint values
						 integer		*needG,     // 0 if G(x) not needed, > 0 if it is
						 integer		*Gnnz,		// length of arrays iGvar and jGfun
						 doublereal		*Gvals,		// derivative values (MMF format)
						 char			*cu,		// character workspace
						 integer		*lencu,		// length of character workspace
						 integer		*iu,		// integer workspace
						 integer		*leniu,		// length of integer workspace
						 doublereal	*ru,		// double workspace
						 integer		*lenru )	// length of double workspace
{
	cblas_dcopy( J , x , 1 , p , 1 );
	
	utilities();
	probabilities();
	lamgam();
	markups();
	zeta(); // computes phi too
	
	if( needF[0] > 0 ) {
		Fvals[0] = 0.0;
		cblas_dcopy(J, phi, 1, Fvals+1, 1);
	}
	
	if( needG[0] > 0 ) {
		zhiphipsi();
		phiders(Gvals);
	}
	
    return 0;
}

// solve implicit version
int nlstrials( char * rfpn )
{
	integer			Start = 0; // Cold = 0, Basis = 1, Warm = 2;
	
	int i, j, k, f;
	integer l;
	
	FILE * rfp = NULL;
	
	int soscflag = 0;
	int soscflagP = 0;
	int * ppom; // "Products Priced Out of the Market"
	int * soscinfo; // sosc info
	
    integer          N, M, FN;
	
	integer			 Annz;
	double			*Adata;
	integer			*Arows, *Acols;
	
	integer			 Gnnz;
    double			*Gdata;
	integer			*Grows, *Gcols;
	
	integer			*xState;
	double			*x, *xLoBnds, *xUpBnds, *lambdaB;
	
	integer			*FState;
	double			*Fvals, *FLoBnds, *FUpBnds, *lambdaC;
	
	double			ObjAdd; // constant to add to objective
	integer			ObjRow; // row of objective in combined function
	integer			INFO;	// 
	
	integer			minrw, miniw, mincw;
	
	// USER workspace
	
	// real (double) workspace
	integer			lenru = 500;
	double			ru[8*500];
	
	// integer workspace
	integer			leniu = 500;
	integer			iu[8*500]; 
	
	// char workspace
	integer			lencu = 500;
	char			cu[8*500];
	
	// SNOPT workspace
	// we initialize to 500, and re-allocate below
	
	// real (double) workspace
	integer			lenrw = 500;
	double			*rw;
	
	// integer workspace
	integer			leniw = 500;
	integer			*iw;
	
	// char workspace
	integer			lencw = 500;
	char			*cw;
	
	integer			nxName = 1; // do not use variable names
	char			xNames[1*8];
	
	integer			nFName = 1; // do not use constraint names
	char			FNames[1*8];
	
	integer			iPrint = 9; // "unit number for the print file"
	integer			iSumm  = 6; // "unit number for the Summary file"
	
	integer			prnt_len; // 
	integer			iSpecs = 4,  spec_len;
	
	char			Prob[200];
	char			printname[200];
	char			specname[200];
	
	integer    nS, nInf, npname = 1;
	doublereal sInf;
	
	integer    iSum, iPrt, strOpt_len;
	char       strOpt[200];
	
	UNUR_DISTR* distr;
	UNUR_PAR*   param;
	UNUR_URNG*  unrng;
	UNUR_GEN*   rngen;
	
	clock_t ticks;
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// The RNGSTREAMS library sets a package seed. 
	unsigned long seed[] = {111u, 222u, 333u, 444u, 555u, 666u};
	RngStream_SetPackageSeed(seed);
	
	// Make an object for uniform random number generator.
	unrng = unur_urng_rngstream_new("unrng");
	if( unrng == NULL ) { 
		printf("ERROR - Uniform generator could not be constructed.\n");
		exit(EXIT_FAILURE); 
	}
	
	// use predefined distribution - uniform
	distr = unur_distr_uniform(NULL, 0);
	
	// use "TDR" method (why, not sure)
	param = unur_tdr_new(distr);
	
	// Set uniform generator in parameter object
	unur_set_urng(param, unrng);
	
	// Create the lognormal generator object.
	rngen = unur_init(param); // param is "destroyed" here
	if( rngen == NULL ) { 
		printf("ERROR - Uniform generator could not be constructed.\n");
	}
	
	// free distribution object
	unur_distr_free(distr);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * DEFINE PROBLEM VARIABLES  * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// define problem sizes
	
	N = J;
	M = J;
	FN	 = J + 1; // number of elements in "combined" map
	
	Annz = 0; // is this allowed by SNOPT?
	Gnnz = J*J; // number of combined map Jacobian non-zeros; see formulas above
	
	printf("Problem stats: \n");
	printf("  %i Variables\n",(int)N);
	printf("  %i Constraints\n",(int)M);
	printf("  %i Jacobian Nonzeros\n",(int)Gnnz);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * ALLOCATE PROBLEM VARIABLES  * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// problem variables
    x			 = (double *) calloc (N, sizeof(double));
    xState		 = (integer *)calloc (N, sizeof(integer)); // initialized to zero for no information
    xLoBnds      = (double *) calloc (N, sizeof(double));
    xUpBnds      = (double *) calloc (N, sizeof(double));
    lambdaB      = (double *) calloc (N, sizeof(double)); // bounds multipliers
	
	// combined function (objective and constraints)
    Fvals		 = (double *) calloc (FN, sizeof(double)); // initializes to zero
    FState       = (integer *)calloc (FN, sizeof(integer));
    FLoBnds      = (double *) calloc (FN, sizeof(double)); // initializes to zero
    FUpBnds      = (double *) calloc (FN, sizeof(double)); // initializes to zero
    lambdaC      = (double *) calloc (FN, sizeof(double)); // constraint multipliers
	
	// linear part of the objective and constraints
    Adata		 = (double *) calloc (Annz, sizeof(double));
    Arows		 = (integer *)calloc (Annz, sizeof(integer));
    Acols		 = (integer *)calloc (Annz, sizeof(integer));
	
	// Jacobian of the nonlinear part of objective and constraints
    Gdata		 = (double *) calloc (Gnnz, sizeof(double));
    Grows		 = (integer *)calloc (Gnnz, sizeof(integer));
    Gcols		 = (integer *)calloc (Gnnz, sizeof(integer));
	
	// initial SNOPT workspace; resized below
	cw			 = (char*)   calloc(8*lencw,sizeof(char   ));
	iw			 = (integer*)calloc(  leniw,sizeof(integer));
	rw			 = (double*) calloc(  lenrw,sizeof(double ));
	
	// "info" vectors for sosc check
	ppom = (int*)calloc( 2*(F+1) , sizeof(int) );
	soscinfo = (int*)calloc( 3*F , sizeof(int) );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITIALIZE C-STYLE RESULTS FILE (IF YOU PASSED IN)  * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// initialize results output file 
	rfp = fopen(rfpn, "w");
	if( rfp != NULL ) {
		fprintf(rfp,"trial,");
		fprintf(rfp,"iter (major),");
		fprintf(rfp,"iter (minor),");
		fprintf(rfp,"time (total),");
		fprintf(rfp,"time (%% fact),");
		fprintf(rfp,"status,");
		fprintf(rfp,"sosc (p),");
		fprintf(rfp,"sosc (P),");
		fprintf(rfp,"maxinc,");
		fprintf(rfp,"ppom (p),");
		fprintf(rfp,"ppom (P),");
		for( j = 0 ; j < J-1 ; j++ ) { fprintf(rfp,"p(%i),",j+1); }
		fprintf(rfp,"p(J)\n");
		fclose(rfp);
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITIALIZE FORTRAN-STYLE FILE REFERENCES  * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// label spec (options) file, using SNOPT's FORTRAN utilities
	sprintf(specname ,   "%s", "snopt.spc");   spec_len = strlen(specname);
	
	// open Print file, using SNOPT's FORTRAN utilities
	sprintf(printname,   "%s", "snopt.out");   prnt_len = strlen(printname);
	snopenappend_( &iPrint, printname,   &INFO, prnt_len );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITIALIZE SNOPT  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	printf("Initializating SNOPT...\n");
	sninit_( &iPrint, &iSumm, cw, &lencw, iw, &leniw, rw, &lenrw, 8*500 );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * SNOPT MEMORY ALLOCATION * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	snmema_(&INFO,
			&FN,
			&N,
			&nxName, &nFName,
			&Annz, &Gnnz, 
			&mincw, &miniw, &minrw, 
			cw, &lencw, 
			iw, &leniw, 
			rw, &lenrw, 
			8*500);
	
	// if memory was NOT sized successfully, 
	if( INFO != 104 ) {
		
		printf("warning: SNOPT could not estimate memory requirements correctly.\n");
		
	} else {
		
		printf("SNOPT estimated memory requirements: %i, %i, %i.\n",(int)mincw,(int)miniw,(int)minrw);
		
		// re-initializing SNOPT workspace, if needed
		
		if( lencw < mincw ) { 
			lencw = mincw; 
			cw = (char*)realloc(cw, 8*lencw*sizeof(char));
		}
		
		if( leniw < miniw ) {
			leniw = miniw;
			iw = (integer*)realloc(iw, leniw*sizeof(integer));
		}
		
		if( lenrw < minrw ) {
			lenrw = minrw;
			rw = (double*) realloc(rw, lenrw*sizeof(double));
		}
		
		printf("Re-initializating SNOPT with sizes (%li,%li,%li)\n",lencw,leniw,lenrw);
		sninit_( &iPrint, &iSumm, cw, &lencw, iw, &leniw, rw, &lenrw, 8*500 );
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITIALIZE SNOPT OPTIONS  * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// options
	snfilewrapper_(specname, 
				   &iSpecs, 
				   &INFO, 
				   cw, &lencw,
				   iw, &leniw, 
				   rw, &lenrw, 
				   spec_len,
				   8*lencw);
	
	if( INFO != 101 ) {
		
		printf("WARNING: trouble reading specs file %s \n", specname);
		printf("Using default options\n");
		
    }
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * INITIALIZE SNOPT PROBLEM DATA * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	//
	Start = 0; // cold start
	
	// Problem name
	strcpy(Prob,"CY2005 PE Problem");
	
	// objective row (must be in FORTRAN-Style indexing)
	ObjRow = 1;
	ObjAdd = 0.0;
	
	FLoBnds[0] = - 1.0e20;
	FUpBnds[0] =   1.0e20;
	
	for( j = 0 ; j < J ; j++ ) {
		
		xLoBnds[j]  = - 1.0e20; // declare prices unbounded
		xUpBnds[j]  =   1.0e20; // declare prices unbounded
		
		FLoBnds[j+1] = 0.0; // p - c - z(p) = 0, written as equality constraint
		FUpBnds[j+1] = 0.0; // p - c - z(p) = 0, written as equality constraint
		
		for( k = 0 ; k < J ; k++ ) {
			Grows[J*j+k] = k+2; // + 1 for FORTRAN-style, + another 1 for the first row in the F map
			Gcols[J*j+k] = j+1; // FORTRAN-style
		}
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * TRIALS  * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *  
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	for( t = 0 ; t < T ; t++ ) {
		
		// define initial point
		cblas_dcopy( J , p0 + J*t , 1 , x , 1 );
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		if( checkders ) {
			
			// check jacobian 
			snopta_eval_G_check(N, FN,
								snopt_callback_func_,
								Gnnz, Grows, Gcols, 
								x, xLoBnds, xUpBnds,
								cu, &lencu, 
								iu, &leniu,
								ru, &lenru);
			
		}
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		ticks = clock();
		
		snopta_(&Start,								// 0: Cold, 1: Basis, 2: Warm
				&FN,								// FN = M + 1
				&N,									// number of variables
				&nxName,							// 1 if no names used, otherwise == N
				&nFName,							// 1 if no names used, otherwise == FN
				&ObjAdd,							// scalar to add to objective (for printing)
				&ObjRow,							// row of the objective in F (FORTRAN-Style)
				Prob,								// problem name
				snopt_callback_func_,				// combined function F(x) needed by snopta_()
				Arows, Acols, &Annz, &Annz, Adata,	// sparse "A" matrix such that F(x) = userfun_(x) + Ax
				Grows, Gcols, &Gnnz, &Gnnz,			// Jacobian structure for G(x) = DF(x)
				xLoBnds, xUpBnds, xNames,			// lower bounds, upper bounds, and names for x variables
				FLoBnds, FUpBnds, FNames,			// lower bounds, upper bounds, and names for F values
				x, xState, lambdaB,					// x values, "states" (see pg. 18), and associated dual variables (multipliers)
				Fvals, FState, lambdaC,				// F values, "states" (see pg. 18?), and associated dual variables (multipliers)
				&INFO,								// result of call to snopta_(). See docs, pg. 19 for details
				&mincw, &miniw, &minrw,				// minimum values of SNOPT workspace sizes for snopta_()
				&nS, &nInf, &sInf,					// see docs, pg. 18 & 20
				cu, &lencu,							// character user workspace
				iu, &leniu,							// integer user workspace
				ru, &lenru,							// real user workspace
				cw, &lencw,							// character SNOPT workspace (at leat 500 + N + NF if names used)
				iw, &leniw,							// integer SNOPT workspace; minimum values given by snmema_()
				rw, &lenrw,							// real SNOPT workspace; minimum values given by snmema_()
				npname, 8*nxName, 8*nFName,
				8*500, 
				8*500);
		
		ticks = clock() - ticks;
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		cblas_dcopy( J , x , 1 , p , 1 );
		
		if( INFO == 2 ) { // feasible point found
			// all we care about is the projection of prices onto [0,maxinc]
			for( j = 0 ; j < J ; j++ ) { if( p[j] > maxinc ) { p[j] = maxinc; } }
			// now check sosc
			soscflag = evalPEQSOSC( ppom , soscinfo , NULL );
		} else { soscflag = 0; }
		soscflagP = 1;
		
		printf("Zeta 'NLS' Trial %i -- SNOPT terminated with status %i\n",t+1,(int)INFO);
		printf("Zeta 'NLS' Trial %i -- Took %0.4f seconds\n",t+1,((double)ticks)/((double)CLOCKS_PER_SEC));
		if( soscflag == 1 ) {
			printf("Zeta 'NLS' Trial %i -- SNOPT found a local equilibria\n",t+1);
		} else {
			printf("Zeta 'NLS' Trial %i -- SNOPT did not find a local equilibrium:\n",t+1);
			for( f = 0 ; f < F ; f++ ) {
				printf("Zeta 'NLS' Trial %i -- firm %i: (%i,%i,%i)\n",t+1,f+1,ppom[f+1],soscinfo[f],soscinfo[F+f]);
				// if( soscinfo[2*F+f] != 1 ) { soscflagP = 0; }
			}
		}
		
		// results outputs
		rfp = fopen(rfpn, "a");
		if( rfp != NULL ) {
			
			// first print trial index (FORTRAN-style)
			fprintf(rfp,"%i,",t+1);
			
			// then iterations required (don't know how to access in SNOPT)
			fprintf(rfp,"%i,",-1);
			fprintf(rfp,"%i,",-1);
			
			// then time required (total only)
			fprintf(rfp,"%0.16f,",((double)ticks)/((double)CLOCKS_PER_SEC));
			fprintf(rfp,"%0.16f,",0.0);
			
			// then status
			fprintf(rfp,"%i,",(int)INFO);
			
			// then SOSC flag(s)
			fprintf(rfp,"%i,",soscflag);
			fprintf(rfp,"%i,",soscflagP);
			
			// then max income
			fprintf(rfp,"%0.16f,",maxinc);
			
			// then number of products (all firms) that have prices at or above maxinc (numerically)
			fprintf(rfp,"%i,",ppom[0]);
			fprintf(rfp,"%i,",ppom[1+F]);
			
			// then "solution" data
			for( j = 0 ; j < J-1 ; j++ ) { fprintf(rfp,"%0.16f,",p[j]); }
			fprintf(rfp,"%0.16f\n",p[J-1]);
			
			fclose(rfp);
			
		}
		
	}
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// release generator(s)
	unur_urng_free(unrng);
	unur_free(rngen);
	
	// these we can free now
	free(ppom);
	free(soscinfo);
	
	free(x);
    free(xState);
    free(xLoBnds);
    free(xUpBnds);
	free(lambdaB);
	
	free(Fvals);
    free(FState);
    free(FLoBnds);
    free(FUpBnds);
	free(lambdaC);
	
	free(Adata);
    free(Arows);
    free(Acols);
	
	free(Gdata);
    free(Grows);
    free(Gcols);
	
	// SNOPT workspace
	free(cw);
	free(iw);
	free(rw);
	
	// close print and specs files
	snclose_( &iPrint );
	snclose_( &iSpecs );
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	return INFO;
	
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 
 * 
 * 
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
int main(int argc, char *argv[])
{
	int i, j, f, n;
	
	char probtype = 'N';
	
	char    runnm[256];
	char	rfpn[256];
	
	double plf; // "product line fraction"
	double frac; // cost fraction
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * INITIALIZE PROBLEM DATA STRUCTURES AND DATA * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// defaults
	I = 1000;
	T = 1;
	
	printf("Reading command-line options...\n");
	
	// read arguments
	for( i = 1 ; i < argc ; i++ ) {
		
		// parse next command line option (only if starting with the '-' character)
		if( (argv[i])[0] == '-' ) {
			
			switch( (argv[i])[1] ) {
					
				default: break;
					
				case 'I': // reading number of individuals
					I = (int)strtol( argv[i] + 2 , NULL , 10 );
					if( I <= 0 ) {
						printf("  Invalid number of individuals (%i): I must be > 0. Running a with I = 1000.\n",I);
						I = 1000;
					} else { printf("  Using I = %i\n",I); }
					break;
					
				case 'S': // integral scale
					integralscale = 1;
					break;
					
				case 'T': // reading number of trials
					T = (int)strtol( argv[i] + 2 , NULL , 10 );
					if( T <= 0 ) {
						printf("  Invalid number of trials (%i): T must be > 0. Running a single trial.\n",T);
						T = 1;
					} else { printf("  Using T = %i\n",T); }
					break;
					
				case 'c':
				case 'C': 
					checkders = 1;
					break;
					
			}
			
		}
		
	}
	
	printf("Finished reading options...\n");
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * NO MODIFICATION * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// NOW initialize problem data
	// includes initialization of initial conditions (p0) so that these are constant
	// across all trials
	initprob();
	
	printf("Initialized problem...\n");
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * NO MODIFICATION * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// start with no-modification case
	
	sprintf(runnm,"CY2005_BLP95_%c_%0.4f_%0.4f",probtype,0.0,0.0);
	
	strcpy(rfpn,runnm);
	strcat(rfpn,".csv");
	
	// ensure costs are from the data
	cblas_dcopy(J, cd, 1, c, 1);
	
	// run T trials with these costs
	nlstrials(rfpn);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * WITH COST MODIFICATION  * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	// iterate over fraction of each firm's product line (just one increment, for now)
	for( plf = 0.1 ; plf < 0.2 ; plf += 0.1 ) {
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// get costs from the 'data'
		cblas_dcopy(J, cd, 1, c, 1);
		
		// fraction increase
		frac = 0.9;
		
		// bring costs closer to maxinc, using frac
		for( f = 0 ; f < F ; f++ ) {
			
			// number of costs to modify (at least one)
			n = floor( (double)(Jf[f]) * plf );
			if( n == 0 ) { n++; }
			
			// modify costs up
			for( j = sJf[f] ; j < sJf[f] + n ; j++ ) {
				c[j] = frac * maxinc + ( 1.0 - frac ) * c[j];
			}
			
		}
		
		// open files (using "descriptive" filenames)
		sprintf(runnm,"CY2005_BLP95_%c_%0.4f_%0.4f",probtype,plf,frac);
		
		strcpy(rfpn,runnm);
		strcat(rfpn,".csv");
		
		// run T trials with these costs
		nlstrials(rfpn);
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// get costs from the 'data'
		cblas_dcopy(J, cd, 1, c, 1);
		
		// fraction increase
		frac = 0.99;
		
		// bring costs closer to maxinc, using frac
		for( f = 0 ; f < F ; f++ ) {
			
			// number of costs to modify (at least one)
			n = floor( (double)(Jf[f]) * plf );
			if( n == 0 ) { n++; }
			
			// modify costs up
			for( j = sJf[f] ; j < sJf[f] + n ; j++ ) {
				c[j] = frac * maxinc + ( 1.0 - frac ) * c[j];
			}
			
		}
		
		// open files (using "descriptive" filenames)
		sprintf(runnm,"CY2005_BLP95_%c_%0.4f_%0.4f",probtype,plf,frac);
		
		strcpy(rfpn,runnm);
		strcat(rfpn,".csv");
		
		// run T trials with these costs
		nlstrials(rfpn);
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// get costs from the 'data'
		cblas_dcopy(J, cd, 1, c, 1);
		
		// fraction increase
		frac = 0.999;
		
		// bring costs closer to maxinc, using frac
		for( f = 0 ; f < F ; f++ ) {
			
			// number of costs to modify (at least one)
			n = floor( (double)(Jf[f]) * plf );
			if( n == 0 ) { n++; }
			
			// modify costs up
			for( j = sJf[f] ; j < sJf[f] + n ; j++ ) {
				c[j] = frac * maxinc + ( 1.0 - frac ) * c[j];
			}
			
		}
		
		// open files (using "descriptive" filenames)
		sprintf(runnm,"CY2005_BLP95_%c_%0.4f_%0.4f",probtype,plf,frac);
		
		strcpy(rfpn,runnm);
		strcat(rfpn,".csv");
		
		// run T trials with these costs
		nlstrials(rfpn);
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// get costs from the 'data'
		cblas_dcopy(J, cd, 1, c, 1);
		
		// fraction increase
		frac = 0.9999;
		
		// bring costs closer to maxinc, using frac
		for( f = 0 ; f < F ; f++ ) {
			
			// number of costs to modify (at least one)
			n = floor( (double)(Jf[f]) * plf );
			if( n == 0 ) { n++; }
			
			// modify costs up
			for( j = sJf[f] ; j < sJf[f] + n ; j++ ) {
				c[j] = frac * maxinc + ( 1.0 - frac ) * c[j];
			}
			
		}
		
		// open files (using "descriptive" filenames)
		sprintf(runnm,"CY2005_BLP95_%c_%0.4f_%0.4f",probtype,plf,frac);
		
		strcpy(rfpn,runnm);
		strcat(rfpn,".csv");
		
		// run T trials with these costs
		nlstrials(rfpn);
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
		// get costs from the 'data'
		cblas_dcopy(J, cd, 1, c, 1);
		
		// fraction increase
		frac = 1.0;
		
		// bring costs closer to maxinc, using frac
		for( f = 0 ; f < F ; f++ ) {
			
			// number of costs to modify (at least one)
			n = floor( (double)(Jf[f]) * plf );
			if( n == 0 ) { n++; }
			
			// modify costs up
			for( j = sJf[f] ; j < sJf[f] + n ; j++ ) {
				c[j] = frac * maxinc + ( 1.0 - frac ) * c[j];
			}
			
		}
		
		// open files (using "descriptive" filenames)
		sprintf(runnm,"CY2005_BLP95_%c_%0.4f_%0.4f",probtype,plf,frac);
		
		strcpy(rfpn,runnm);
		strcat(rfpn,".csv");
		
		// run T trials with these costs
		nlstrials(rfpn);
		
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
		 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
		
	}
		
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	killprob();
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	return 0;
}